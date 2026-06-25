"""DIAGNOSTIC (scratch only): pin the MESH-CONVERGED run-on infiltration/runoff partition for the two
PIDS overland schemes on the b1 fixture (mild planar loam, where upwind converges).

THE QUESTION: under MESH refinement, do SequentialCoupledProblem(route_substeps=8) and
CoupledProblem(overland_scheme="upwind") converge to the SAME partition (-> the coarse ~32pp gap is
just under-resolution) or to DIFFERENT partitions (-> a structural operator-split bias)? And what is
the mesh-converged "resolved target"?

This is a DIAGNOSTIC, not a test/fix: it does not modify any pids_forward/ file, does not assert, does
not commit. It prints the RAW numbers and an incremental table after EACH run (so a slow tail never
loses earlier results).

The partition metric is IDENTICAL to tests/test_sequential_hardening.py::test_caseA_runon_partition_*
(the b1 test), reusing the SAME quantities both schemes expose:
  R_in   = RAIN * top_area * STORM_DUR     (top_area = int 1 ds_top, degree-8)
  routed = cum_outflow
  drained= cum_drainage
  soil_gain = int theta(psi_final) - int theta(psi_initial)   (degree-8 dx, public prob.psi)
  surf   = surface store  (seq: surface_water() = int max(psi,0) ds_top ; mono: int d ds_top)
Report routed/R_in, infil/R_in = (soil_gain+drained)/R_in, and the CLOSURE
  (routed + soil_gain + drained + surf)/R_in   (must be ~1; flagged if not).

Run (WSL pids-fem, threads pinned):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/runon_partition_investigation.py'
"""
from __future__ import annotations

import time

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.sequential_coupling import SequentialCoupledProblem

COMM = MPI.COMM_WORLD

# ---- the b1 fixture (mild planar loam; upwind converges) -- IDENTICAL to test_sequential_hardening ---
A_LX, A_LY, A_LZ = 8.0, 5.0, 1.0
A_S0 = 0.03                 # gentle planar fall to the y=0 outlet
A_PSI_I = -0.4
A_RAIN = 0.5               # m/day, > loam Ks=0.25 -> ponds modestly + runs off
A_STORM = 0.08            # d (short burst)
A_TEND = 0.45             # d (storm + recession so both surfaces drain -> stable partition)
A_SOIL = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)   # LOAM


# ---- partition helpers (lifted verbatim from tests/test_sequential_hardening.py) ---------------------
def _top_area_ds(mesh, ztop):
    """int 1 ds_top (plan area of the top facet), degree-8 -- the denominator of R_in."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    tf = np.sort(dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[mesh.geometry.dim - 1], ztop))).astype(np.int32)
    ft = dmesh.meshtags(mesh, fdim, tf, np.ones(tf.size, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=mesh, subdomain_data=ft,
                         metadata={"quadrature_degree": 8})(1)
    return mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ds_top)), op=MPI.SUM)


def _soil_water_deg8(prob, soil):
    """int theta(psi) dV, degree-8 dx via the public prob.psi (identical for both schemes)."""
    dxq = ufl.dx(metadata={"quadrature_degree": prob._quad_degree})
    return prob.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(soil.theta_ufl(prob.psi) * dxq)), op=MPI.SUM)


def _mono_surface_store(prob):
    """The monolith surface store int d ds_top (== its surface_water())."""
    return prob.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(prob.d * prob._ds_top)), op=MPI.SUM)


def _march_storm(prob, rain, *, storm_dur, storm_rain, t_end, dt0=1e-3, dt_max=0.03):
    """March a storm-then-recession with the production BAND dt-controller (it<=4 grow*1.4, it>=12
    shrink*0.7, capped dt_max). Works for BOTH schemes (same step(dt)->(conv,it)). On a non-converged
    step HALVE dt; ``collapsed`` flags a drop below the 1e-9 float floor. Identical to the b1 marcher.
    Returns (nstep, collapsed, t_reached, wall)."""
    t, nstep, dt = 0.0, 0, dt0
    collapsed = False
    t0 = time.perf_counter()
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if t < storm_dur - 1e-12 and t + h > storm_dur:
            h = storm_dur - t                                   # land exactly on the storm end
        rain.value = storm_rain if t < storm_dur - 1e-12 else 0.0
        conv, it = prob.step(h)
        if not conv:
            dt *= 0.5
            if dt < 1e-9:
                collapsed = True
                break
            continue
        t += h
        nstep += 1
        if it <= 4:
            dt = min(dt * 1.4, dt_max)
        elif it >= 12:
            dt = dt * 0.7
    return nstep, collapsed, t, time.perf_counter() - t0


# ---- run one scheme on one mesh, return the partition (R_in-normalized) ------------------------------
def _run_seq(nx, ny, nz, route_substeps):
    soil = VanGenuchten(**A_SOIL)
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [A_LX, A_LY, A_LZ]], [nx, ny, nz])
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05, route_substeps=route_substeps)
    prob.set_topography(lambda x: A_S0 * x[1])
    prob.set_initial_condition(lambda x: A_PSI_I + 0.0 * x[0])
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=A_S0)
    rain = prob.add_rain(0.0)
    th0 = _soil_water_deg8(prob, soil)
    nstep, coll, tend, wall = _march_storm(
        prob, rain, storm_dur=A_STORM, storm_rain=A_RAIN, t_end=A_TEND)
    top_area = _top_area_ds(msh, A_LZ)
    R_in = A_RAIN * top_area * A_STORM
    soil_gain = _soil_water_deg8(prob, soil) - th0
    routed = prob.cum_outflow
    drained = prob.cum_drainage
    surf = prob.surface_water()
    ncells = msh.topology.index_map(msh.topology.dim).size_local
    # the scheme's OWN conserved-ledger closure (machine-tight ~1e-12): a clean leak check that is
    # INDEPENDENT of the degree-8 soil_gain measurement artifact in the R_in closure below.
    bal_frac = abs(prob.balance()) / prob.cum_rain if prob.cum_rain > 0 else float("nan")
    return dict(
        ok=(not coll and tend >= A_TEND - 1e-9), coll=coll, tend=tend, nstep=nstep, wall=wall,
        ncells=ncells, R_in=R_in, top_area=top_area, bal_frac=bal_frac,
        routed_R=routed / R_in, infil_R=(soil_gain + drained) / R_in,
        surf_R=surf / R_in, drained_R=drained / R_in, soilgain_R=soil_gain / R_in,
        closure=(routed + soil_gain + drained + surf) / R_in,
        eff="seq")


def _run_mono(nx, ny, nz):
    soil = VanGenuchten(**A_SOIL)
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [A_LX, A_LY, A_LZ]], [nx, ny, nz])
    prob = CoupledProblem(msh, soil, overland_scheme="upwind")
    prob.set_topography(lambda x: A_S0 * x[1])
    prob.set_initial_condition(lambda x: A_PSI_I + 0.0 * x[0], d_value=0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=A_S0)
    rain = prob.add_rain(0.0)
    th0 = _soil_water_deg8(prob, soil)
    nstep, coll, tend, wall = _march_storm(
        prob, rain, storm_dur=A_STORM, storm_rain=A_RAIN, t_end=A_TEND)
    top_area = _top_area_ds(msh, A_LZ)
    R_in = A_RAIN * top_area * A_STORM
    soil_gain = _soil_water_deg8(prob, soil) - th0
    routed = prob.cum_outflow
    drained = prob.cum_drainage
    surf = _mono_surface_store(prob)
    ncells = msh.topology.index_map(msh.topology.dim).size_local
    return dict(
        # GUARD: the monolith must genuinely converge here (upwind, reached t_end, no dt-collapse).
        ok=(prob._effective_overland_scheme == "upwind" and not coll and tend >= A_TEND - 1e-9),
        coll=coll, tend=tend, nstep=nstep, wall=wall, ncells=ncells, R_in=R_in, top_area=top_area,
        bal_frac=float("nan"),   # the monolith exposes no balance()/cum_rain; use the R_in closure only
        routed_R=routed / R_in, infil_R=(soil_gain + drained) / R_in,
        surf_R=surf / R_in, drained_R=drained / R_in, soilgain_R=soil_gain / R_in,
        closure=(routed + soil_gain + drained + surf) / R_in,
        eff=prob._effective_overland_scheme)


# ---- incremental table printers ---------------------------------------------------------------------
def _fmt_part(r):
    if r is None:
        return "      (skipped)        "
    # The R_in closure carries a KNOWN ~0.6% deficit: soil_gain uses degree-8 int theta(psi) while the
    # scheme conserves a LUMPED int theta (verified: SEQ balance()/cum_rain ~5e-12 yet R_in-closure
    # ~0.994). So flag only a closure that strays MORE than the quadrature artifact band (>1.5%) -- a
    # genuine leak. bal_frac (SEQ only) is the machine-tight leak check, printed alongside.
    flag = "" if abs(r["closure"] - 1.0) < 0.015 else " <<CLOSURE-LEAK!"
    bf = "" if not np.isfinite(r.get("bal_frac", float("nan"))) else f" bal/rain={r['bal_frac']:.1e}"
    tail = "" if r["ok"] else f" <<NOT-OK(coll={r['coll']},t={r['tend']:.3f},eff={r['eff']})"
    return (f"routed/R={r['routed_R']:.4f} infil/R={r['infil_R']:.4f} "
            f"surf/R={r['surf_R']:.4f} clo={r['closure']:.5f}{flag}{bf}{tail}")


def _print_mesh_table(rows):
    print("\n================ MESH-REFINEMENT TABLE (route_substeps=8 for SEQ) ================")
    print(f"{'mesh':>12} {'cells':>7} | SEQ(rs8) / MONO(upwind)")
    for mesh_str, ncells, rseq, rmono in rows:
        print(f"{mesh_str:>12} {ncells:>7} | SEQ : {_fmt_part(rseq)}")
        print(f"{'':>12} {'':>7} | MONO: {_fmt_part(rmono)}")
    print("==================================================================================\n",
          flush=True)


def _print_substep_table(rows, mesh_str):
    print(f"\n========== SUBSTEP-REFINEMENT TABLE (SEQ, fixed mesh {mesh_str}) ==========")
    print(f"{'rs':>4} {'cells':>7} {'nstep':>6} {'wall':>7} | partition")
    for rs, r in rows:
        print(f"{rs:>4} {r['ncells']:>7} {r['nstep']:>6} {r['wall']:>6.1f}s | {_fmt_part(r)}")
    print("===========================================================================\n", flush=True)


def main():
    print(">>> b1 fixture: mild planar LOAM hillslope, RAIN=0.5 m/d (>Ks=0.25), STORM_DUR=0.08 d, "
          f"T_END=0.45 d, S0=0.03, box {A_LX}x{A_LY}x{A_LZ}", flush=True)

    # ---- SWEEP 1: MESH refinement at route_substeps=8 (MESH is the only variable) -------------------
    meshes = [(12, 8, 4), (20, 14, 6), (30, 20, 8), (40, 28, 9)]
    mesh_rows = []   # (mesh_str, ncells, rseq, rmono)
    print("\n### SWEEP 1: mesh refinement (SEQ route_substeps=8 vs MONO upwind) ###", flush=True)
    for (nx, ny, nz) in meshes:
        mesh_str = f"{nx}x{ny}x{nz}"
        t0 = time.perf_counter()

        print(f"\n--- mesh {mesh_str}: SEQ(rs8) ...", flush=True)
        rseq = _run_seq(nx, ny, nz, route_substeps=8)
        print(f"    SEQ(rs8) {mesh_str}: {_fmt_part(rseq)}  "
              f"[nstep={rseq['nstep']} wall={rseq['wall']:.1f}s cells={rseq['ncells']}]", flush=True)

        print(f"--- mesh {mesh_str}: MONO(upwind) ...", flush=True)
        rmono = _run_mono(nx, ny, nz)
        # if the monolith dt-collapsed / fell back, RECORD it (informative) but keep it in the table.
        print(f"    MONO     {mesh_str}: {_fmt_part(rmono)}  "
              f"[nstep={rmono['nstep']} wall={rmono['wall']:.1f}s cells={rmono['ncells']} "
              f"eff={rmono['eff']}]", flush=True)

        mesh_rows.append((mesh_str, rseq["ncells"], rseq, rmono))
        _print_mesh_table(mesh_rows)   # incremental: reprint the full table after EACH mesh
        print(f"    (mesh {mesh_str} total wall {time.perf_counter() - t0:.1f}s)", flush=True)

    # ---- SWEEP 2: SUBSTEP refinement at a fixed mid mesh (30,20,8) ----------------------------------
    mid = (30, 20, 8)
    mid_str = f"{mid[0]}x{mid[1]}x{mid[2]}"
    print(f"\n### SWEEP 2: substep refinement (SEQ) at fixed mid mesh {mid_str} ###", flush=True)
    sub_rows = []
    for rs in (4, 8, 16):
        print(f"\n--- mid mesh {mid_str}: SEQ rs={rs} ...", flush=True)
        r = _run_seq(*mid, route_substeps=rs)
        print(f"    SEQ rs={rs}: {_fmt_part(r)}  [nstep={r['nstep']} wall={r['wall']:.1f}s]", flush=True)
        sub_rows.append((rs, r))
        _print_substep_table(sub_rows, mid_str)   # incremental

    # ---- FINAL CONSOLIDATED TABLES + a few derived deltas -------------------------------------------
    print("\n\n################################ FINAL CONSOLIDATED ################################")
    _print_mesh_table(mesh_rows)
    _print_substep_table(sub_rows, mid_str)

    print(">>> Mesh-refinement trend (routed/R), SEQ(rs8) vs MONO, and the per-mesh gap:")
    for mesh_str, ncells, rseq, rmono in mesh_rows:
        gap = rseq["routed_R"] - rmono["routed_R"] if rmono["ok"] else float("nan")
        print(f"    {mesh_str:>10} cells={ncells:>6}: SEQ routed/R={rseq['routed_R']:.4f}  "
              f"MONO routed/R={rmono['routed_R']:.4f}  (SEQ-MONO)={gap:+.4f}  "
              f"[SEQ ok={rseq['ok']} MONO ok={rmono['ok']}]")
    print("###################################################################################\n",
          flush=True)


if __name__ == "__main__":
    main()
