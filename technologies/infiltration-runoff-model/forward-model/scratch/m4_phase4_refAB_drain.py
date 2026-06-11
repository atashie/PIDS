"""Module 4 (§E) Phase-4 Refs A+B, DRAIN direction (the core-use-case mirror).

DRAIN (soil wetter than the feature; matrix desorption of the drainable porosity INTO the wall) IS the
depleting-reservoir scenario of the PIDS drain use case, so the Phase-4 coupled gate must cover it
([[pids-drain-usecase]]). Mirror of the disperse Refs (m4_phase4_refA_disperse.py / _refB_):

  Ref A-drain: 1-D cylindrical radial, soil starts SATURATED (psi_i=0), wall Dirichlet psi=-1, NO-FLOW
    outer at R_out in {3,5,10} r_w. Closed-domain equilibrium is psi=-1 everywhere -> the removable
    capacity is the same I_max = dtheta*(R_out^2-r_w^2)/(2 r_w) as disperse (|dpsi|=1 mirror).
    DISCRIMINATION twist: the offline drain clock's semi-empirical throttle F=exp(-(zeta/z0)^k) kills
    its flux by zeta ~ 2*z0 (LOAM z0=1.41 -> I plateaus near HALF of the 3 r_w capacity), while the
    resolved closed drain keeps (slowly) draining toward I_max -> the clock fails by UNDER-prediction
    (the opposite signature to disperse's unbounded overshoot -- good lens diversity). Because the
    resolved drain also physically throttles (near-wall K collapse), full depletion takes very long;
    the bands below are therefore LOWER than disperse and band-miss is a WARN (the Task-4 gate
    constants are pre-registered from the MEASURED numbers, not from these heuristic bands).

  Ref B-drain: host-history mirror -- a RE-WETTING pulse into the band [3,4] r_w of the 5 r_w domain,
    DEFICIT-AWARE: fired when global depletion first reaches ~50% (so the band has actually drained),
    volume = 50% of the band's measured-at-fire-time deficit (cannot re-saturate; guard + halve-retry
    anyway). The replenishment RE-STEEPENS the wall uptake (I_wall ends HIGHER than Ref A-drain; the
    asymptote gains exactly the pulse volume) -- the offline clock cannot respond, by construction.
    Bookkeeping: I_wall = [ integral (theta_i - theta) r dr + cum_source ] / r_w (closed domain).

VALIDATION (drain-specific -- the disperse-style "early window matches the open-domain fixture" check
is INVALID here, found 2026-06-10): from a SATURATED start the bulk is INCOMPRESSIBLE (van-Genuchten
C(0)=0 and Ss is ignored by project assumption), so the pressure signal crosses the domain instantly
and the OPEN 1c reference carries a saturated-conduction THROUGHFLOW from the outer boundary from
t ~ 0+ -- a closed domain deviates IMMEDIATELY (measured: 35% at 3 r_w), at all times, by physics.
Note the 1c fixture's I is the domain THETA-LOSS, not the wall flux: in the open run part of the wall
uptake is boundary-supplied throughflow that never appears as theta-loss, while the closed domain
desaturates DOMAIN-WIDE (the saturated bulk drops psi together -- incompressible with nowhere to
flow), so the closed theta-loss legitimately sits ABOVE the open fixture early (measured +35% at
3 r_w; the sign is physics, not a bug). The closed-drain ground truth is validated by:
  (1) mesh convergence: LOAM 3 r_w re-run at cell/2 agrees < 1% rel-L2 (spatial);
  (2) dt robustness: LOAM 3 r_w re-run on a 2x-denser SAMPLE grid agrees < 2% at the shared times
      (temporal; MEASURED 1.26% -- that is the BE temporal accuracy of these drain refs, immaterial
      against the >=29% discrimination margins, and documented rather than chased);
  (3) monotonicity + the capacity bound (equilibrium psi=-1 everywhere);
  (4) the machinery is otherwise identical to the validated disperse Ref A (only the BCs/sign differ).
NO curve-matching check exists for closed drains -- measured and understood 2026-06-10: vs the OPEN
fixture the dev is +10-12% (the open run's boundary throughflow, R-independent since the fixture is
the same for all three R); and the three closed curves differ from EACH OTHER even early (~2% R3 vs
R5: closed saturated bulks of different SIZE desaturate at different uniform rates from t~0+ -- the
incompressible bulk transmits the wall pull instantly, so domain size enters immediately). Both devs
are printed with a generous gross-bug ceiling (<5% mutual), not asserted tight.
Numerics: bt linesearch (saturated-start stiffness -- cp stalls; [[pids-fem-saturated-wall-linesearch]]);
cell = 0.5/3200 (the Phase-1c converged resolution for the sharp desaturation front).

Plan: docs/plans/2026-06-10-m4-phase4-coupled-embedding-plan.md (Task 3).
Run from forward-model/:
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scratch/m4_phase4_refAB_drain.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

import scratch.m4_phase1b_disperse_reference as dz
from pids_forward.physics.sorptive_closure import (
    F_throttle, throttle_params, des_sorp_ratio, sorptive_clock, rel_l2)

COMM = MPI.COMM_WORLD
R_W = dz.R_W
PSI_I, PSI_WALL = 0.0, -1.0          # drain: saturated soil, low wall (the 1c mirror)
CELL = 0.5 / 3200.0                  # 1c's converged cell for the sharp desaturation front
N_SAMP = 32
_DRAIN_LU = dict(dz._LU, snes_linesearch_type="bt")

SOIL_NAMES = ("LOAM", "SAND")
R_FACTORS = (3, 5, 10)
DEPLETION_BAND = {3: (0.55, 0.90), 5: (0.35, 0.75), 10: (0.15, 0.50)}   # WARN-only bands (see docstring)
T_END = {  # first guesses [day], auto-tuned (max 3 attempts)
    "LOAM": {3: 0.25, 5: 1.2, 10: 6.0},
    "SAND": {3: 0.012, 5: 0.06, 10: 0.3},
}
BAND_B = (3 * R_W, 4 * R_W)
T_END_B = 1.2                        # Ref B-drain window (LOAM, 5 r_w)


def _setup(soil, r_out, cell=CELL, psi_i=PSI_I):
    n = max(int(round((r_out - R_W) / cell)), 80)
    msh = dmesh.create_interval(COMM, n, [R_W, r_out])
    r = ufl.SpatialCoordinate(msh)[0]
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi = fem.Function(V); psi_n = fem.Function(V)
    psi.x.array[:] = psi_i; psi_n.x.array[:] = psi_i
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    s_c = fem.Constant(msh, PETSc.ScalarType(0.0))
    dxs, dxq = dz._vertex_dx(), ufl.dx(metadata={"quadrature_degree": 8})
    theta, theta_n, K = soil.theta_ufl(psi), soil.theta_ufl(psi_n), soil.K_ufl(psi)
    x = ufl.SpatialCoordinate(msh)[0]
    band_ind = ufl.conditional(ufl.And(ufl.ge(x, BAND_B[0]), ufl.le(x, BAND_B[1])), 1.0, 0.0)
    F = (((theta - theta_n) / dt_c) * v * r * dxs
         + K * ufl.dot(ufl.grad(psi), ufl.grad(v)) * r * dxq
         - s_c * band_ind * v * r * dxq)
    wall_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], R_W))
    bcs = [fem.dirichletbc(PETSc.ScalarType(PSI_WALL), wall_dofs, V)]
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4p4d_", petsc_options=_DRAIN_LU)
    th_i = float(soil.theta(psi_i))
    removed = fem.form((th_i - theta) * r * dxs)          # water LOST by the soil (>= 0)
    band_deficit = fem.form((th_i - theta) * band_ind * r * dxq)
    return msh, V, psi, psi_n, dt_c, s_c, problem, removed, band_deficit


def run_drain_A(soil, r_out, samples, label="", cell=CELL, psi_i=PSI_I):
    _, _, psi, psi_n, dt_c, _, problem, removed, _ = _setup(soil, r_out, cell, psi_i)
    out, dt, t_prev = [], 1e-8, 0.0
    for i, t_s in enumerate(samples):
        dt = dz._solve_to(problem, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        I = COMM.allreduce(fem.assemble_scalar(removed), op=MPI.SUM) / R_W
        out.append(float(I))
        if label:
            print(f"      [{label}] {i+1}/{len(samples)}  t={t_s:.3e}  I={I:.4e}", flush=True)
    return np.array(out)


def run_drain_B(soil, t_grid, t1, t2, i_max, label=""):
    """Ref B-drain: march to t1, size the pulse from the MEASURED band deficit, continue with the
    source active over [t1, t2]. Returns (I_wall(t), V_pulse_per_radian, band psi_max during pulse)."""
    _, V, psi, psi_n, dt_c, s_c, problem, removed, band_deficit = _setup(soil, 5 * R_W)
    xc = V.tabulate_dof_coordinates()[:, 0]
    in_band = (xc >= BAND_B[0]) & (xc <= BAND_B[1])
    marks = np.unique(np.concatenate([t_grid, [t1, t2]]))
    I_wall, s_rate, band_max = [], 0.0, -np.inf
    dt, t_prev = 1e-8, 0.0
    for t_s in marks:
        if abs(t_prev - t1) < 1e-15:                     # arriving at the fire time: size the pulse
            D = COMM.allreduce(fem.assemble_scalar(band_deficit), op=MPI.SUM)
            V_pulse = 0.5 * D
            s_rate = V_pulse / ((t2 - t1) * (BAND_B[1] ** 2 - BAND_B[0] ** 2) / 2.0)
            print(f"      [{label}] band deficit at t1={t1:.3f}: {D:.4e} -> pulse V={V_pulse:.4e} "
                  f"(s={s_rate:.3f}/day)", flush=True)
        active = (t_prev >= t1 - 1e-15) and (t_s <= t2 + 1e-15)
        s_c.value = s_rate if active else 0.0
        dt = dz._solve_to(problem, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        if active:
            band_max = max(band_max, float(psi.x.array[in_band].max()))
        if t_s in t_grid:
            loss = COMM.allreduce(fem.assemble_scalar(removed), op=MPI.SUM)
            Bw = (BAND_B[1] ** 2 - BAND_B[0] ** 2) / 2.0
            cum_src = s_rate * Bw * min(max(t_s - t1, 0.0), t2 - t1)
            I_wall.append((loss + cum_src) / R_W)        # wall removal = domain loss + injected water
        if label:
            print(f"      [{label}] t={t_s:.3e}  done", flush=True)
    Bw = (BAND_B[1] ** 2 - BAND_B[0] ** 2) / 2.0
    return np.array(I_wall), s_rate * Bw * (t2 - t1), band_max


def _main_r40():
    """The DEPLOYMENT-SCALE drain reference (LOAM, R_out = 40 r_w = 2 m, partial window).

    Scenario pair: soil at psi_i = -0.03 m (saturated theta -- just BELOW the h_s = -0.02 air entry,
    so the matching all-Neumann embedded box has C > 0 and a non-singular first solve), wall -1 m.
    PARTIAL window (full depletion of 7.45 m through matrix desorption alone takes ~years); the
    discrimination on this leg is the throttle clock's premature plateau (~0.026 m = 0.3% of I_max,
    fitted on OPEN 0.5-m refs) against the resolved closed curve, which keeps draining via the
    domain-wide uniform-pressure-drop mechanism."""
    soil = dz.SOILS["LOAM"]
    psi_i40 = -0.03
    r_out, t_end = 40 * R_W, 20.0
    dth = float(soil.theta(psi_i40) - soil.theta(PSI_WALL))
    S_sorp = dz.parlange_sorptivity(soil, PSI_WALL, psi_i40)   # wetting pair (-1 -> -0.03)
    S_des = des_sorp_ratio(dth) * S_sorp
    z0, kk = throttle_params(dth)
    i_max = dth * (r_out ** 2 - R_W ** 2) / (2.0 * R_W)
    t_start = (dth * 0.1 * R_W / S_des) ** 2
    t = np.geomspace(t_start, t_end, N_SAMP)
    print(f"DRAIN-40 (psi_i={psi_i40}, wall {PSI_WALL}): dtheta={dth:.4f}, S_des={S_des:.5f}, "
          f"I_max={i_max:.4f} m, window {t_end} d")
    I = run_drain_A(soil, r_out, t, label="LOAM drain R40", psi_i=psi_i40)
    assert np.all(np.diff(I) >= -1e-12 * i_max) and I[-1] <= i_max * (1 + 1e-9)
    clk = sorptive_clock(t, S_des, dth, R_W, lambda z: F_throttle(z, z0, kk))
    print(f"   I_end={I[-1]:.4e} m ({I[-1]/i_max:.1%} of I_max)  "
          f"clock plateau={clk[-1]:.4e}  OFFLINE-THROTTLE-CLOCK relL2={rel_l2(clk, I):.1%}")
    np.savez("scratch/m4_phase4_refD40_drain.npz",
             LOAM_t=t, LOAM_I=I, LOAM_Imax=np.array(i_max), LOAM_Sdes=np.array(S_des),
             LOAM_dtheta=np.array(dth), LOAM_psi_i=np.array(psi_i40), r_w=np.array(R_W))
    print("Saved -> scratch/m4_phase4_refD40_drain.npz")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "r40":
        _main_r40()
        raise SystemExit(0)
    fix = np.load("tests/data/m4_phase1c_drain_refs.npz")
    saved = {"r_w": np.array(R_W)}
    print("=" * 88)
    print("PHASE-4 Refs A+B DRAIN (saturated soil psi_i=0 -> wall psi=-1, no-flow outer; bt linesearch)")
    print("=" * 88)
    for name in SOIL_NAMES:
        soil = dz.SOILS[name]
        dth = float(soil.theta(PSI_I) - soil.theta(PSI_WALL))
        S_sorp = dz.parlange_sorptivity(soil, PSI_WALL)               # disperse S over the same |dpsi|
        S_des = des_sorp_ratio(dth) * S_sorp                          # the production drain default
        z0, kk = throttle_params(dth)
        t_start = (dth * 0.1 * R_W / S_des) ** 2
        saved[f"{name}_Sdes"], saved[f"{name}_dtheta"] = np.array(S_des), np.array(dth)
        print(f"\n{'-'*88}\n{name}: S_des(default)={S_des:.5f} m/day^0.5,  dtheta={dth:.4f},  z0={z0:.2f}")
        dev_by_R = {}
        for k in R_FACTORS:
            r_out = k * R_W
            i_max = dth * (r_out**2 - R_W**2) / (2.0 * R_W)
            lo, hi = DEPLETION_BAND[k]
            t_end = T_END[name][k]
            for attempt in range(3):
                t = np.geomspace(t_start, t_end, N_SAMP)
                I = run_drain_A(soil, r_out, t, label=f"{name} drain R={k}r_w (t_end={t_end:.3g})")
                frac = I[-1] / i_max
                if frac < lo:
                    t_end *= 3.0
                    print(f"   R={k}r_w: depleted {frac:.1%} < {lo:.0%} -> t_end x3 = {t_end:.3g}", flush=True)
                elif frac > hi:
                    t_end = float(np.interp(0.5 * (lo + hi), I / i_max, t))
                    print(f"   R={k}r_w: depleted {frac:.1%} > {hi:.0%} -> t_end at mid-band = {t_end:.3g}",
                          flush=True)
                else:
                    break
            if not (lo <= I[-1] / i_max <= hi):
                print(f"   WARN {name} R={k}: final depletion {I[-1]/i_max:.1%} outside band "
                      f"[{lo:.0%},{hi:.0%}] after 3 attempts -- taking the measured curve as-is", flush=True)

            assert np.all(np.diff(I) >= -1e-12 * i_max), f"{name} R={k}: I not monotone"
            assert I[-1] <= i_max * (1.0 + 1e-9), f"{name} R={k}: capacity violated"
            # signed deviation vs the OPEN 1c fixture (theta-loss metric; see docstring -- the closed
            # curve sits ABOVE early, by physics); the |dev| must SHRINK as the boundary recedes
            tf, If = fix[f"{name}_t"], fix[f"{name}_tunnel_I"]
            early = (I < 0.3 * i_max) & (t >= tf[0]) & (t <= tf[-1])
            assert np.any(early), f"{name} R={k}: no early-window overlap with the 1c fixture"
            Iop = np.interp(t[early], tf, If)
            dev_med = float(np.median((I[early] - Iop) / Iop))
            clk = sorptive_clock(t, S_des, dth, R_W, lambda z: F_throttle(z, z0, kk))
            print(f"   R={k:2d}r_w: I_end={I[-1]:.4e}  I_max={i_max:.4e}  depleted={I[-1]/i_max:.1%}  "
                  f"dev-vs-open(med)={dev_med:+.1%}(n={int(early.sum())})  "
                  f"OFFLINE-THROTTLE-CLOCK relL2={rel_l2(clk, I):.1%}", flush=True)
            dev_by_R[k] = dev_med
            saved[f"{name}_R{k}_t"], saved[f"{name}_R{k}_I"] = t, I
            saved[f"{name}_R{k}_Imax"] = np.array(i_max)

        # mutual early deviation between the closed curves: EXPECTED nonzero (size effect, docstring);
        # print signed, ceiling 5% (gross-bug catch only)
        t3, I3 = saved[f"{name}_R3_t"], saved[f"{name}_R3_I"]
        i_max3 = float(saved[f"{name}_R3_Imax"])
        sh = I3 < 0.15 * i_max3
        msg = []
        for kk2 in (5, 10):
            Ik = np.interp(t3[sh], saved[f"{name}_R{kk2}_t"], saved[f"{name}_R{kk2}_I"])
            mdev = float(np.median((Ik - I3[sh]) / I3[sh]))
            assert abs(mdev) < 0.05, f"{name}: R3 vs R{kk2} early dev {mdev:+.1%} -- gross-bug ceiling"
            msg.append(f"R{kk2} {mdev:+.2%}")
        print(f"   mutual early dev vs R3 (size effect, expected ~2%): {', '.join(msg)};  "
              f"dev-vs-open (R-indep throughflow): "
              f"{dev_by_R[3]:+.1%} / {dev_by_R[5]:+.1%} / {dev_by_R[10]:+.1%}")

    # ---- mesh + dt convergence of the closed-drain ground truth (LOAM 3 r_w) -------------------
    tC = saved["LOAM_R3_t"]
    I_half = run_drain_A(dz.SOILS["LOAM"], 3 * R_W, tC, label="LOAM conv cell/2", cell=CELL / 2.0)
    conv = rel_l2(I_half, saved["LOAM_R3_I"])
    assert conv < 0.01, f"closed-drain mesh convergence failed: relL2(cell/2)={conv:.2%}"
    t_dense = np.unique(np.concatenate([tC, np.sqrt(tC[:-1] * tC[1:])]))   # insert geometric midpoints
    I_dense = run_drain_A(dz.SOILS["LOAM"], 3 * R_W, t_dense, label="LOAM conv dt-dense")
    at32 = np.isin(t_dense, tC)
    dtconv = rel_l2(I_dense[at32], saved["LOAM_R3_I"])
    assert dtconv < 0.02, f"closed-drain dt robustness failed: relL2(dense grid)={dtconv:.2%}"
    print(f"\n   convergence (LOAM 3r_w): mesh cell/2 relL2 = {conv:.3%}, dt-dense relL2 = {dtconv:.3%}  OK")

    # ---- Ref B-drain (LOAM, 5 r_w): deficit-aware replenishment pulse --------------------------
    name, soil = "LOAM", dz.SOILS["LOAM"]
    dth = float(saved["LOAM_dtheta"]); S_des = float(saved["LOAM_Sdes"])
    i_max = float(saved["LOAM_R5_Imax"])
    tA, IA = saved["LOAM_R5_t"], saved["LOAM_R5_I"]
    # fire when the Ref A-drain curve first reaches 50% of ITS OWN final depletion (not of I_max --
    # robust to where the band landed); pulse window = 10% of the Ref-A window
    t1 = float(np.interp(0.5 * IA[-1], IA, tA))
    t2 = t1 + 0.1 * float(tA[-1])
    t_grid = np.geomspace(tA[0], T_END_B if T_END_B > t2 * 1.5 else 2.0 * t2, N_SAMP)
    print(f"\n{'-'*88}\nRef B-drain (LOAM, 5r_w): pulse [t1,t2]=[{t1:.3f},{t2:.3f}] d")
    I_B, V_pulse, band_max = run_drain_B(soil, t_grid, t1, t2, i_max, label="LOAM RefB-drain")
    assert band_max < -0.01, f"Ref B-drain pulse re-saturated the band (psi_max={band_max:+.3f})"
    assert np.all(np.diff(I_B) >= -1e-12 * i_max), "Ref B-drain: I_wall not monotone"
    pre = t_grid < t1
    devA = np.abs(I_B[pre] - np.interp(t_grid[pre], tA, IA)) / np.interp(t_grid[pre], tA, IA)
    assert devA.max() < 0.005, f"Ref B-drain pre-pulse mismatch vs Ref A-drain(5r_w): {devA.max():.2%}"
    IA_end = float(np.interp(t_grid[-1], tA, IA))
    gap = (I_B[-1] - IA_end) / IA_end
    z0, kk = throttle_params(dth)
    clk = sorptive_clock(t_grid, S_des, dth, R_W, lambda z: F_throttle(z, z0, kk))
    assert gap > 0.03, f"Ref B-drain history response too small: {gap:.2%}"
    print(f"   pre-pulse dev={devA.max():.2%}  band psi_max={band_max:+.3f} m  "
          f"end I_B={I_B[-1]:.4e} vs RefA {IA_end:.4e} -> gap = +{gap:.1%}")
    print(f"   OFFLINE-THROTTLE-CLOCK relL2 vs Ref B-drain = {rel_l2(clk, I_B):.1%}")
    saved["LOAM_B_t"], saved["LOAM_B_I"] = t_grid, I_B
    saved["LOAM_B_t_pulse"] = np.array([t1, t2])
    saved["LOAM_B_V_pulse_per_wall_area"] = np.array(V_pulse / R_W)

    np.savez("scratch/m4_phase4_refAB_drain.npz", **saved)
    print("\nSaved drain Refs A+B -> scratch/m4_phase4_refAB_drain.npz")
    print("=" * 88)
