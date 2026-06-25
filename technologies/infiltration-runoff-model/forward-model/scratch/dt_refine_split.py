"""DIAGNOSTIC (scratch only): confirm-or-refute that the merged ``SequentialCoupledProblem``'s ~24 pp
run-on partition gap vs the MONOLITHIC upwind scheme is an O(dt) OPERATOR-SPLITTING error from solving
routing-then-infiltration in sequence at the coupling timestep.

THE SIGNATURE OF AN O(dt) SPLITTING ERROR: at a FIXED mesh with CFL-RESOLVED routing (rs held high so
rs is NOT the variable), shrinking the coupling timestep dt->0 should make SEQ's routed-out fraction
RISE and converge to the monolith's dt-converged value at that mesh.
  - SEQ(dt->0) RISES and REACHES MONO            -> the gap IS the O(dt) splitting error (iterated /
                                                    Picard split CONFIRMED as the fix).
  - SEQ(dt->0) PLATEAUS BELOW MONO               -> a deeper formulation/BC difference, not just
                                                    splitting (iterated split alone may not close it).

This is a DIAGNOSTIC, not a test/fix: it does NOT modify any pids_forward/ file, does NOT assert, does
NOT commit. It marches with a FIXED dt (NOT the production band controller) so dt is the controlled
variable. ``route_substeps=32`` at the coarse mesh keeps routing CFL-resolved (the prior probe showed
the coarse case plateaus in rs by ~16). It prints a partial table after EACH leg.

The b1 fixture + partition metric are IDENTICAL to scratch/cfl_substep_confirm.py /
scratch/runon_partition_investigation.py / tests/test_sequential_hardening.py::test_caseA_runon_*:
  R_in   = RAIN * top_area * STORM_DUR        (top_area = int 1 ds_top, degree-8)
  routed = cum_outflow ; routed/R = cum_outflow / R_in
  infil/R= (soil_gain + cum_drainage)/R_in,  soil_gain = int theta(final) - int theta(initial) deg-8 dx
  closure= (routed + soil_gain + drained + surf)/R_in  (~1; the degree-8 vs lumped quadrature band is
           ~0.6%, so >1.5% off is flagged a genuine leak). SEQ ALSO reports balance()/cum_rain (the
           machine-tight conserved-ledger leak check, INDEPENDENT of the degree-8 soil_gain artifact).

Run (WSL pids-fem, threads pinned):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/dt_refine_split.py 2>&1 | tail -50'
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

# COARSE mesh ONLY (fast). prior MONO routed/R ref ~ 0.6384 (recomputed fresh below).
CNX, CNY, CNZ = 12, 8, 4
CSTR = f"{CNX}x{CNY}x{CNZ}"


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


# ---- the FIXED-dt marcher (NOT the band controller; dt is the controlled variable) ------------------
def _march_storm_fixed(prob, rain, *, storm_dur, storm_rain, t_end, dt_fixed):
    """March a storm-then-recession with a FIXED coupling timestep ``dt_fixed`` -- NO band adaptation
    (no it<=4 grow / it>=12 shrink). The fixed step is clipped ONLY to (a) land exactly on the storm
    end and (b) not overshoot t_end. On a non-converged step we HALVE dt for THAT attempt only (a
    local solver-robustness retry, restoring the nominal dt_fixed afterwards); a drop below the 1e-9
    floor sets ``collapsed``. Returns (nstep, collapsed, t_reached, wall, n_halved).

    NOTE: rs (route_substeps inside SEQ) is set on the problem, so routing is sub-stepped to
    rs-resolution regardless of dt_fixed -- exactly the intended separation of variables.
    """
    t, nstep = 0.0, 0
    collapsed = False
    n_halved = 0                 # count of accepted steps that needed a local halving (solver retry)
    t0 = time.perf_counter()
    while t < t_end - 1e-12:
        h = min(dt_fixed, t_end - t)
        if t < storm_dur - 1e-12 and t + h > storm_dur:
            h = storm_dur - t                                   # land exactly on the storm end
        rain.value = storm_rain if t < storm_dur - 1e-12 else 0.0
        # try the nominal step; on failure halve locally and retry (robustness only, NOT band control)
        h_try = h
        had_halving = False
        while True:
            conv, it = prob.step(h_try)
            if conv:
                break
            had_halving = True
            h_try *= 0.5
            if h_try < 1e-9:
                collapsed = True
                break
        if collapsed:
            break
        t += h_try
        nstep += 1
        if had_halving:
            n_halved += 1
    return nstep, collapsed, t, time.perf_counter() - t0, n_halved


# ---- run one scheme on the coarse mesh at a FIXED dt, return the R_in-normalized partition ----------
def _run_seq(dt_fixed, route_substeps):
    soil = VanGenuchten(**A_SOIL)
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [A_LX, A_LY, A_LZ]], [CNX, CNY, CNZ])
    prob = SequentialCoupledProblem(msh, soil, n_man=A_NMAN, route_substeps=route_substeps)
    prob.set_topography(lambda x: A_S0 * x[1])
    prob.set_initial_condition(lambda x: A_PSI_I + 0.0 * x[0])
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=A_S0)
    rain = prob.add_rain(0.0)
    th0 = _soil_water_deg8(prob, soil)
    nstep, coll, tend, wall, nhalf = _march_storm_fixed(
        prob, rain, storm_dur=A_STORM, storm_rain=A_RAIN, t_end=A_TEND, dt_fixed=dt_fixed)
    top_area = _top_area_ds(msh, A_LZ)
    R_in = A_RAIN * top_area * A_STORM
    soil_gain = _soil_water_deg8(prob, soil) - th0
    routed = prob.cum_outflow
    drained = prob.cum_drainage
    surf = prob.surface_water()
    bal_frac = abs(prob.balance()) / prob.cum_rain if prob.cum_rain > 0 else float("nan")
    return dict(
        ok=(not coll and tend >= A_TEND - 1e-9), coll=coll, tend=tend, nstep=nstep, wall=wall,
        nhalf=nhalf, R_in=R_in, bal_frac=bal_frac, dt=dt_fixed, rs=route_substeps,
        routed_R=routed / R_in, infil_R=(soil_gain + drained) / R_in,
        surf_R=surf / R_in, drained_R=drained / R_in, soilgain_R=soil_gain / R_in,
        closure=(routed + soil_gain + drained + surf) / R_in, eff="seq")


def _run_mono(dt_fixed):
    soil = VanGenuchten(**A_SOIL)
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [A_LX, A_LY, A_LZ]], [CNX, CNY, CNZ])
    prob = CoupledProblem(msh, soil, overland_scheme="upwind")
    prob.set_topography(lambda x: A_S0 * x[1])
    prob.set_initial_condition(lambda x: A_PSI_I + 0.0 * x[0], d_value=0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=A_S0)
    rain = prob.add_rain(0.0)
    th0 = _soil_water_deg8(prob, soil)
    nstep, coll, tend, wall, nhalf = _march_storm_fixed(
        prob, rain, storm_dur=A_STORM, storm_rain=A_RAIN, t_end=A_TEND, dt_fixed=dt_fixed)
    top_area = _top_area_ds(msh, A_LZ)
    R_in = A_RAIN * top_area * A_STORM
    soil_gain = _soil_water_deg8(prob, soil) - th0
    routed = prob.cum_outflow
    drained = prob.cum_drainage
    surf = _mono_surface_store(prob)
    return dict(
        ok=(prob._effective_overland_scheme == "upwind" and not coll and tend >= A_TEND - 1e-9),
        coll=coll, tend=tend, nstep=nstep, wall=wall, nhalf=nhalf, R_in=R_in,
        bal_frac=float("nan"), dt=dt_fixed, rs=None,
        routed_R=routed / R_in, infil_R=(soil_gain + drained) / R_in,
        surf_R=surf / R_in, drained_R=drained / R_in, soilgain_R=soil_gain / R_in,
        closure=(routed + soil_gain + drained + surf) / R_in, eff=prob._effective_overland_scheme)


# ---- formatting -------------------------------------------------------------------------------------
def _fmt(r):
    flag = "" if abs(r["closure"] - 1.0) < 0.015 else " <<CLOSURE-LEAK!"
    bf = "" if not np.isfinite(r.get("bal_frac", float("nan"))) else f" bal/rain={r['bal_frac']:.1e}"
    tail = "" if r["ok"] else f" <<NOT-OK(coll={r['coll']},t={r['tend']:.3f},eff={r['eff']})"
    nh = f" nhalf={r['nhalf']}" if r["nhalf"] else ""
    return (f"routed/R={r['routed_R']:.4f} infil/R={r['infil_R']:.4f} "
            f"surf/R={r['surf_R']:.4f} clo={r['closure']:.5f}{flag}{bf}{tail}{nh}")


def _print_seq_table(rows, mono_ref):
    print(f"\n===== SEQ(rs=32) dt-REFINEMENT @ FIXED mesh {CSTR} (MONO ref routed/R={mono_ref:.4f}) =====")
    print(f"  {'dt':>8} {'nsteps':>7} {'wall':>7} | {'routed/R':>9} {'infil/R':>8} {'closure':>8}"
          f" | {'mono-SEQ':>9}")
    for r in rows:
        gap = mono_ref - r["routed_R"]
        leak = " <<LEAK" if abs(r["closure"] - 1.0) >= 0.015 else ""
        notok = "" if r["ok"] else " <<NOT-OK"
        print(f"  {r['dt']:>8.0e} {r['nstep']:>7} {r['wall']:>6.1f}s | {r['routed_R']:>9.4f} "
              f"{r['infil_R']:>8.4f} {r['closure']:>8.5f} | {gap:>+9.4f}{leak}{notok}")
    print("==========================================================================================\n",
          flush=True)


def _print_mono_table(rows):
    print(f"\n===== MONO(upwind) dt-REFINEMENT @ FIXED mesh {CSTR} (is it dt-converged?) =====")
    print(f"  {'dt':>8} {'nsteps':>7} {'wall':>7} | {'routed/R':>9} {'infil/R':>8} {'closure':>8}")
    for r in rows:
        notok = "" if r["ok"] else " <<NOT-OK"
        print(f"  {r['dt']:>8.0e} {r['nstep']:>7} {r['wall']:>6.1f}s | {r['routed_R']:>9.4f} "
              f"{r['infil_R']:>8.4f} {r['closure']:>8.5f}{notok}")
    print("==============================================================================\n", flush=True)


def main():
    print(">>> b1 fixture: mild planar LOAM hillslope, RAIN=0.5 m/d (>Ks=0.25), STORM_DUR=0.08 d, "
          f"T_END=0.45 d, S0=0.03, n_man={A_NMAN}, box {A_LX}x{A_LY}x{A_LZ}, COARSE mesh {CSTR}",
          flush=True)
    print(">>> HYPOTHESIS: SEQ's ~24pp run-on gap vs MONO is an O(dt) operator-splitting error. At FIXED"
          " mesh + CFL-resolved routing (rs=32), does SEQ(routed/R) RISE -> MONO as dt->0?\n", flush=True)

    # ============ MONO(upwind) dt-convergence reference FIRST (so the SEQ target is a pinned number) ===
    print(f"### MONO(upwind) @ {CSTR}: dt-convergence reference (dt in 1e-2, 3e-3, 1e-3) ###", flush=True)
    mono_rows = []
    for dt in (1e-2, 3e-3, 1e-3):
        print(f"--- MONO dt={dt:.0e} ...", flush=True)
        r = _run_mono(dt)
        print(f"    MONO dt={dt:.0e}: {_fmt(r)}  [nsteps={r['nstep']} wall={r['wall']:.1f}s]", flush=True)
        mono_rows.append(r)
        _print_mono_table(mono_rows)
    # the dt-converged MONO target = the finest converged leg (1e-3), fall back to last ok leg.
    mono_ok = [r for r in mono_rows if r["ok"]]
    mono_ref = (mono_ok[-1] if mono_ok else mono_rows[-1])["routed_R"]
    print(f">>> MONO dt-converged target routed/R = {mono_ref:.4f} "
          f"(spread over dt: {min(r['routed_R'] for r in mono_rows):.4f}.."
          f"{max(r['routed_R'] for r in mono_rows):.4f})\n", flush=True)

    # ============ SEQ(rs=32) dt-refinement: 3e-4 LAST (it is the longest) ============
    print(f"### SEQ(rs=32) @ {CSTR}: dt-refinement (3e-2,1e-2,3e-3,1e-3,3e-4); 3e-4 last ###", flush=True)
    seq_rows = []
    for dt in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
        print(f"--- SEQ rs=32 dt={dt:.0e} ...", flush=True)
        r = _run_seq(dt, route_substeps=32)
        print(f"    SEQ rs=32 dt={dt:.0e}: {_fmt(r)}  [nsteps={r['nstep']} wall={r['wall']:.1f}s]",
              flush=True)
        seq_rows.append(r)
        _print_seq_table(seq_rows, mono_ref)          # incremental reprint after EACH dt leg

    # ============================== FINAL CONSOLIDATED + VERDICT ==============================
    print("\n\n############################## FINAL CONSOLIDATED ##############################")
    _print_mono_table(mono_rows)
    _print_seq_table(seq_rows, mono_ref)

    seq_ok = [r for r in seq_rows if r["ok"]]
    print(">>> VERDICT INPUTS:")
    print(f"    MONO dt-converged routed/R target = {mono_ref:.4f}")
    if seq_ok:
        seq_lo = seq_ok[0]
        seq_hi = seq_ok[-1]                       # finest dt that finished ok
        direction = ("UP" if seq_hi["routed_R"] > seq_lo["routed_R"] else
                     "DOWN" if seq_hi["routed_R"] < seq_lo["routed_R"] else "FLAT")
        residual = mono_ref - seq_hi["routed_R"]
        last_climb = (seq_ok[-1]["routed_R"] - seq_ok[-2]["routed_R"]) if len(seq_ok) >= 2 \
            else float("nan")
        # is it monotone-rising across the ok legs?
        rr = [r["routed_R"] for r in seq_ok]
        monotone = all(rr[i + 1] >= rr[i] - 1e-4 for i in range(len(rr) - 1))
        print(f"    SEQ routed/R: dt={seq_lo['dt']:.0e} -> {seq_lo['routed_R']:.4f}   "
              f"FINEST ok dt={seq_hi['dt']:.0e} -> {seq_hi['routed_R']:.4f}  "
              f"({direction}, monotone-rising={monotone}, last drs={last_climb:+.4f})")
        print(f"    SEQ limit as dt->0 (finest ok leg) = {seq_hi['routed_R']:.4f}  "
              f"-> residual (MONO - SEQ) = {residual:+.4f}")
        if abs(residual) < 0.01:
            print("    >>> VERDICT: SEQ -> MONO as dt->0  =>  the gap is the O(dt) OPERATOR-SPLITTING "
                  "error. The iterated/Picard split is CONFIRMED as the fix.")
        elif direction == "UP" and residual > 0.01:
            print(f"    >>> VERDICT: SEQ RISES toward MONO but PLATEAUS BELOW it by {residual:+.4f} as "
                  "dt->0  =>  a deeper formulation/BC difference (splitting is PART of it, but the "
                  "iterated split ALONE may not fully close the gap).")
        else:
            print(f"    >>> VERDICT: SEQ does NOT rise to MONO (direction={direction}, residual="
                  f"{residual:+.4f})  =>  NOT a simple O(dt) splitting error; deeper formulation/BC "
                  "difference dominates.")
        # ledger leak check at the small-dt legs
        worst_bal = max((r["bal_frac"] for r in seq_ok if np.isfinite(r["bal_frac"])), default=float("nan"))
        print(f"    SEQ conserved-ledger leak check: worst balance()/cum_rain over ok legs = "
              f"{worst_bal:.2e}  (machine-tight ~1e-12 => no leak)")
    else:
        print("    >>> ALL SEQ legs NOT-OK (collapsed / short) -- cannot form a verdict; see the table.")
    print("################################################################################\n",
          flush=True)


if __name__ == "__main__":
    main()
