"""DIAGNOSTIC (scratch only): confirm-or-refute that ``route_substeps`` (rs) is a CFL/Courant number,
not a constant, for the merged ``SequentialCoupledProblem`` run-on infiltration/runoff partition.

THE HYPOTHESIS: at FIXED rs a finer mesh makes the explicit Manning routing under-travel (water crosses
fewer physical metres per Richards step) -> water lingers and over-infiltrates. The claim under test:
at a FIXED mesh, driving rs high enough (CFL-resolved routing) makes SEQ's routed-out fraction RISE and
converge to a limit; and that limit should approach the MONOLITHIC UPWIND value AT THE SAME MESH.
  - SEQ(rs->large) converges UP to the monolith at each mesh  -> PURE CFL (fix = adaptive substepping).
  - SEQ plateaus BELOW the monolith                            -> CFL + a residual operator-split bias.

This is a DIAGNOSTIC, not a test/fix: it does NOT modify any pids_forward/ file, does NOT assert, does
NOT commit. It prints RAW numbers and an incremental table after EACH run (a slow tail never loses
earlier results).

The b1 fixture + partition metric are IDENTICAL to scratch/runon_partition_investigation.py and
tests/test_sequential_hardening.py::test_caseA_runon_partition_* :
  R_in   = RAIN * top_area * STORM_DUR        (top_area = int 1 ds_top, degree-8)
  routed = cum_outflow
  infil  = soil_gain + cum_drainage,  soil_gain = int theta(final) - int theta(initial)  (degree-8 dx)
  closure= (routed + soil_gain + drained + surf)/R_in  (~1; flagged > 1.5% off = the quadrature band).

Run (WSL pids-fem, threads pinned):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/cfl_substep_confirm.py 2>&1 | tail -60'
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
A_NMAN = 0.05


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


def _march_storm(prob, rain, *, storm_dur, storm_rain, t_end, dt0=1e-3, dt_max=0.03,
                 track_pond=False):
    """March a storm-then-recession with the production BAND dt-controller (it<=4 grow*1.4, it>=12
    shrink*0.7, capped dt_max). Identical to the b1 marcher; OPTIONALLY records the running peak pond
    depth (max over nodes of max(psi,0)) and a representative dt (the median accepted step) for the
    Courant estimate. On a non-converged step HALVE dt; ``collapsed`` flags a drop below the 1e-9 floor.
    Returns (nstep, collapsed, t_reached, wall, peak_pond, dt_repr)."""
    t, nstep, dt = 0.0, 0, dt0
    collapsed = False
    peak_pond = 0.0
    dts = []
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
        if track_pond:
            dts.append(h)
            # peak ponded depth (m): max over surface nodes of max(psi,0). SEQ carries pond in psi
            # (_rp.psi); the monolith carries it in d. Use whichever the scheme exposes.
            try:
                arr = prob._rp.psi.x.array                       # SEQ pond-in-psi
            except AttributeError:
                arr = prob.d.x.array                             # monolith surface depth
            peak_pond = max(peak_pond, float(np.max(np.maximum(arr, 0.0))))
        if it <= 4:
            dt = min(dt * 1.4, dt_max)
        elif it >= 12:
            dt = dt * 0.7
    dt_repr = float(np.median(dts)) if dts else float("nan")
    return nstep, collapsed, t, time.perf_counter() - t0, peak_pond, dt_repr


# ---- run one scheme on one mesh, return the partition (R_in-normalized) ------------------------------
def _run_seq(nx, ny, nz, route_substeps, *, track_pond=False):
    soil = VanGenuchten(**A_SOIL)
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [A_LX, A_LY, A_LZ]], [nx, ny, nz])
    prob = SequentialCoupledProblem(msh, soil, n_man=A_NMAN, route_substeps=route_substeps)
    prob.set_topography(lambda x: A_S0 * x[1])
    prob.set_initial_condition(lambda x: A_PSI_I + 0.0 * x[0])
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=A_S0)
    rain = prob.add_rain(0.0)
    th0 = _soil_water_deg8(prob, soil)
    nstep, coll, tend, wall, peak_pond, dt_repr = _march_storm(
        prob, rain, storm_dur=A_STORM, storm_rain=A_RAIN, t_end=A_TEND, track_pond=track_pond)
    top_area = _top_area_ds(msh, A_LZ)
    R_in = A_RAIN * top_area * A_STORM
    soil_gain = _soil_water_deg8(prob, soil) - th0
    routed = prob.cum_outflow
    drained = prob.cum_drainage
    surf = prob.surface_water()
    ncells = msh.topology.index_map(msh.topology.dim).size_local
    bal_frac = abs(prob.balance()) / prob.cum_rain if prob.cum_rain > 0 else float("nan")
    return dict(
        ok=(not coll and tend >= A_TEND - 1e-9), coll=coll, tend=tend, nstep=nstep, wall=wall,
        ncells=ncells, R_in=R_in, top_area=top_area, bal_frac=bal_frac,
        peak_pond=peak_pond, dt_repr=dt_repr,
        routed_R=routed / R_in, infil_R=(soil_gain + drained) / R_in,
        surf_R=surf / R_in, drained_R=drained / R_in, soilgain_R=soil_gain / R_in,
        closure=(routed + soil_gain + drained + surf) / R_in,
        eff="seq", rs=route_substeps)


def _run_mono(nx, ny, nz, *, track_pond=False):
    soil = VanGenuchten(**A_SOIL)
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [A_LX, A_LY, A_LZ]], [nx, ny, nz])
    prob = CoupledProblem(msh, soil, overland_scheme="upwind")
    prob.set_topography(lambda x: A_S0 * x[1])
    prob.set_initial_condition(lambda x: A_PSI_I + 0.0 * x[0], d_value=0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=A_S0)
    rain = prob.add_rain(0.0)
    th0 = _soil_water_deg8(prob, soil)
    nstep, coll, tend, wall, peak_pond, dt_repr = _march_storm(
        prob, rain, storm_dur=A_STORM, storm_rain=A_RAIN, t_end=A_TEND, track_pond=track_pond)
    top_area = _top_area_ds(msh, A_LZ)
    R_in = A_RAIN * top_area * A_STORM
    soil_gain = _soil_water_deg8(prob, soil) - th0
    routed = prob.cum_outflow
    drained = prob.cum_drainage
    surf = _mono_surface_store(prob)
    ncells = msh.topology.index_map(msh.topology.dim).size_local
    return dict(
        ok=(prob._effective_overland_scheme == "upwind" and not coll and tend >= A_TEND - 1e-9),
        coll=coll, tend=tend, nstep=nstep, wall=wall, ncells=ncells, R_in=R_in, top_area=top_area,
        bal_frac=float("nan"), peak_pond=peak_pond, dt_repr=dt_repr,
        routed_R=routed / R_in, infil_R=(soil_gain + drained) / R_in,
        surf_R=surf / R_in, drained_R=drained / R_in, soilgain_R=soil_gain / R_in,
        closure=(routed + soil_gain + drained + surf) / R_in,
        eff=prob._effective_overland_scheme, rs=None)


# ---- formatting -------------------------------------------------------------------------------------
def _fmt_part(r):
    if r is None:
        return "      (skipped)        "
    flag = "" if abs(r["closure"] - 1.0) < 0.015 else " <<CLOSURE-LEAK!"
    bf = "" if not np.isfinite(r.get("bal_frac", float("nan"))) else f" bal/rain={r['bal_frac']:.1e}"
    tail = "" if r["ok"] else f" <<NOT-OK(coll={r['coll']},t={r['tend']:.3f},eff={r['eff']})"
    return (f"routed/R={r['routed_R']:.4f} infil/R={r['infil_R']:.4f} "
            f"surf/R={r['surf_R']:.4f} clo={r['closure']:.5f}{flag}{bf}{tail}")


def _print_table(mesh_str, mono, seq_rows):
    """Per-mesh table: the monolith reference + the SEQ rs ladder, with the routed/R gap to the mono."""
    print(f"\n========== rs-LADDER @ FIXED mesh {mesh_str} (cells={mono['ncells']}) ==========")
    mref = mono["routed_R"]
    print(f"  MONO(upwind) ref : {_fmt_part(mono)}"
          f"  [nstep={mono['nstep']} wall={mono['wall']:.1f}s eff={mono['eff']}]")
    print(f"  {'rs':>4} {'nstep':>6} {'wall':>7} | {'routed/R':>9} {'infil/R':>8} {'clo':>8}"
          f" | {'mono-SEQ':>9}")
    for r in seq_rows:
        gap = mref - r["routed_R"]
        leak = " <<LEAK" if abs(r["closure"] - 1.0) >= 0.015 else ""
        notok = "" if r["ok"] else " <<NOT-OK"
        print(f"  {r['rs']:>4} {r['nstep']:>6} {r['wall']:>6.1f}s | {r['routed_R']:>9.4f} "
              f"{r['infil_R']:>8.4f} {r['closure']:>8.5f} | {gap:>+9.4f}{leak}{notok}")
    print("=====================================================================\n", flush=True)


def _courant(mesh_str, nx, ref, label):
    """Estimate the Courant number for the explicit Manning routing at a given mesh:
        dx ~ LX/nx ;  v = (1/n) h^{2/3} S^{1/2}  (n=0.05, S~S0=0.03, h~peak pond seen) ;
        rs_needed ~ v * dt / dx   at the representative (median accepted) dt.
    NOTE on units: RAIN/Ks are m/DAY and dt is in DAYS, so the consistent Manning velocity for this
    CFL must also be per-DAY -- v_SI [m/s] * 86400. We print BOTH so the day-consistent rs_needed is
    unambiguous (the routing kernel marches dt in days)."""
    dx = A_LX / nx
    h = max(ref["peak_pond"], 1e-9)
    S = A_S0
    v_si = (1.0 / A_NMAN) * (h ** (2.0 / 3.0)) * (S ** 0.5)        # m/s  (Manning, SI)
    v_day = v_si * 86400.0                                          # m/day (dt is in days)
    dt = ref["dt_repr"]
    rs_si = v_si * dt / dx                                          # WRONG units (dt in days, v in m/s)
    rs_day = v_day * dt / dx                                        # day-consistent Courant
    print(f"  [{mesh_str}] {label}: dx={dx:.4f} m  peak_pond h={h:.4e} m  "
          f"S={S}  dt_repr={dt:.4e} d"
          f"\n           v_Manning={v_si:.4e} m/s = {v_day:.4f} m/d  "
          f"-> rs_needed(day-consistent)=v_day*dt/dx={rs_day:.2f}"
          f"  (rs from SI-v would be {rs_si:.2e}, units-inconsistent)", flush=True)
    return rs_day


def main():
    print(">>> b1 fixture: mild planar LOAM hillslope, RAIN=0.5 m/d (>Ks=0.25), STORM_DUR=0.08 d, "
          f"T_END=0.45 d, S0=0.03, n_man={A_NMAN}, box {A_LX}x{A_LY}x{A_LZ}", flush=True)
    print(">>> HYPOTHESIS: rs is a CFL number. At FIXED mesh, SEQ(rs->large) routed/R should RISE and "
          "plateau; does the plateau reach MONO(upwind) at the same mesh?\n", flush=True)

    courant_lines = []

    # ============ COARSE mesh (12,8,4) -- mono routed/R ref ~ 0.6384 (recompute fresh) ============
    cnx, cny, cnz = 12, 8, 4
    cstr = f"{cnx}x{cny}x{cnz}"
    print(f"### FIXED mesh {cstr} (COARSE). prior mono routed/R ref ~ 0.6384 (recomputing fresh) ###",
          flush=True)
    print(f"--- {cstr}: MONO(upwind) ref ...", flush=True)
    cmono = _run_mono(cnx, cny, cnz, track_pond=True)
    print(f"    MONO {cstr}: {_fmt_part(cmono)}  [nstep={cmono['nstep']} wall={cmono['wall']:.1f}s "
          f"peak_pond={cmono['peak_pond']:.4e} dt_repr={cmono['dt_repr']:.4e}]", flush=True)

    coarse_seq = []
    for rs in (8, 16, 32, 64, 128):
        print(f"--- {cstr}: SEQ rs={rs} ...", flush=True)
        r = _run_seq(cnx, cny, cnz, route_substeps=rs, track_pond=(rs == 8))
        print(f"    SEQ rs={rs} {cstr}: {_fmt_part(r)}  [nstep={r['nstep']} wall={r['wall']:.1f}s]",
              flush=True)
        coarse_seq.append(r)
        _print_table(cstr, cmono, coarse_seq)          # incremental reprint after EACH rs

    # Courant estimate for the coarse mesh (use the rs=8 SEQ peak pond + the mono peak pond).
    cl = _courant(cstr, cnx, coarse_seq[0], "SEQ rs=8 pond")
    cl_m = _courant(cstr, cnx, cmono, "MONO pond")
    courant_lines.append((cstr, cl, cl_m))

    # ============ MID mesh (30,20,8) -- mono routed/R ref ~ 0.5465 (recompute fresh) ============
    mnx, mny, mnz = 30, 20, 8
    mstr = f"{mnx}x{mny}x{mnz}"
    print(f"\n### FIXED mesh {mstr} (MID). prior mono routed/R ref ~ 0.5465 (recomputing fresh) ###",
          flush=True)
    print(f"--- {mstr}: MONO(upwind) ref ...", flush=True)
    mmono = _run_mono(mnx, mny, mnz, track_pond=True)
    print(f"    MONO {mstr}: {_fmt_part(mmono)}  [nstep={mmono['nstep']} wall={mmono['wall']:.1f}s "
          f"peak_pond={mmono['peak_pond']:.4e} dt_repr={mmono['dt_repr']:.4e}]", flush=True)

    mid_seq = []
    mid_ladder = [16, 32, 64, 128]
    for rs in mid_ladder:
        print(f"--- {mstr}: SEQ rs={rs} ...", flush=True)
        r = _run_seq(mnx, mny, mnz, route_substeps=rs, track_pond=(rs == 16))
        print(f"    SEQ rs={rs} {mstr}: {_fmt_part(r)}  [nstep={r['nstep']} wall={r['wall']:.1f}s]",
              flush=True)
        mid_seq.append(r)
        _print_table(mstr, mmono, mid_seq)             # incremental

    # OPTIONAL rs=256 ONLY if the trend at rs=128 clearly hasn't plateaued (still climbing > 0.5pp
    # from rs=64 -> rs=128) -- a guard so we don't burn time when it has converged.
    if len(mid_seq) >= 2:
        climb = mid_seq[-1]["routed_R"] - mid_seq[-2]["routed_R"]
        if abs(climb) > 0.005:
            print(f"--- {mstr}: rs=128 still moving (drs64->128={climb:+.4f}); adding rs=256 ...",
                  flush=True)
            r = _run_seq(mnx, mny, mnz, route_substeps=256)
            print(f"    SEQ rs=256 {mstr}: {_fmt_part(r)}  [nstep={r['nstep']} wall={r['wall']:.1f}s]",
                  flush=True)
            mid_seq.append(r)
            _print_table(mstr, mmono, mid_seq)
        else:
            print(f"--- {mstr}: rs=128 plateaued (drs64->128={climb:+.4f}); skipping rs=256.",
                  flush=True)

    cl = _courant(mstr, mnx, mid_seq[0], "SEQ rs=16 pond")
    cl_m = _courant(mstr, mnx, mmono, "MONO pond")
    courant_lines.append((mstr, cl, cl_m))

    # ============================== FINAL CONSOLIDATED + VERDICT ==============================
    print("\n\n############################## FINAL CONSOLIDATED ##############################")
    _print_table(cstr, cmono, coarse_seq)
    _print_table(mstr, mmono, mid_seq)

    print(">>> VERDICT INPUTS -- per fixed mesh: SEQ routed/R across the rs ladder vs MONO routed/R")
    for mesh_str, mono, seq_rows in ((cstr, cmono, coarse_seq), (mstr, mmono, mid_seq)):
        mref = mono["routed_R"]
        seq_lo = seq_rows[0]["routed_R"]
        seq_hi = seq_rows[-1]["routed_R"]
        last_climb = (seq_rows[-1]["routed_R"] - seq_rows[-2]["routed_R"]) if len(seq_rows) >= 2 \
            else float("nan")
        residual = mref - seq_hi
        direction = "UP" if seq_hi > seq_lo else ("DOWN" if seq_hi < seq_lo else "FLAT")
        plateaued = "YES" if abs(last_climb) < 0.005 else "NO(still climbing)"
        reach = "REACHES mono (pure CFL)" if abs(residual) < 0.01 else \
            f"STAYS BELOW mono by {residual:+.4f} (CFL + residual ordering bias)"
        print(f"    {mesh_str:>10}: MONO routed/R={mref:.4f} | SEQ rs{seq_rows[0]['rs']}={seq_lo:.4f}"
              f" -> rs{seq_rows[-1]['rs']}={seq_hi:.4f}  ({direction}, last drs={last_climb:+.4f}, "
              f"plateaued={plateaued})\n               residual(mono-SEQ_hi)={residual:+.4f} -> {reach}")

    print("\n>>> COURANT CHECK -- implied rs_needed (day-consistent) vs where SEQ plateaus:")
    for mesh_str, cl, cl_m in courant_lines:
        print(f"    {mesh_str:>10}: rs_needed ~ {cl:.2f} (SEQ-pond v) / {cl_m:.2f} (MONO-pond v)")
    print("################################################################################\n",
          flush=True)


if __name__ == "__main__":
    main()
