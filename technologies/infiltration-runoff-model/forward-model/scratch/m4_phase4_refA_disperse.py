"""Module 4 (§E) Phase-4 Ref A: DEPLETING-RESERVOIR disperse references (the discriminating gate's
evolving-far-field ground truth).

The Phase-3 coupled-embedding retraction (d42432c) showed a fixed-far-field reference CANNOT detect a
passive "offline clock + deposition" scheme — any accumulator passes. Ref A makes the far field EVOLVE
by construction: the same 1-D cylindrical radial disperse solve as Phase-1b (saturated wall psi=0 into
dry soil psi_i=-1) but with a NO-FLOW outer boundary at R_out in {3, 5, 10}*r_w. The finite reservoir
depletes; I(t) bends away from the infinite-domain curve toward the hard capacity

    I_max = dtheta * (R_out^2 - r_w^2) / (2 r_w)        [uptake per unit wall area]

while the offline clock (cyl Green-Ampt + Parlange S, the SOLID Phase-3 a-priori gate) grows unbounded
-> it FAILS this reference BY CONSTRUCTION. That failure margin is the discrimination assertion of the
Phase-4 coupled gate (tests/test_coupled_gate_refs.py); the matching embedded test is a feature in a
CLOSED host box of equal capacity (scratch/m4_phase4_embedded_harness.py).

Per-curve self-checks (asserted here, not just printed):
  * I non-decreasing and final I <= I_max (capacity bound);
  * EARLY-WINDOW cross-check: before the front feels the boundary (I < 0.3*I_max), the depleting curve
    must match the committed infinite-domain fixture tests/data/m4_phase1b_disperse_refs.npz < 1.5%
    (validates the no-flow machinery against the signed-off Phase-1b reference);
  * window design (see FULL_R/CONTROL_BAND comment -- measured lesson: a partial-depletion window lets
    the clock PASS): 3/5 r_w = full-depletion discriminators, 10 r_w = partial-depletion control.
Printed for Task-4 to pre-register: offline-clock AND capacity-clamped-clock rel-L2 per curve.

Plan: docs/plans/2026-06-10-m4-phase4-coupled-embedding-plan.md (Task 1).
Run from forward-model/:
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scratch/m4_phase4_refA_disperse.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

import scratch.m4_phase1b_disperse_reference as dz  # soils, marcher, vertex rule, cp-linesearch LU
from pids_forward.physics.sorptive_closure import F_cylindrical, sorptive_clock, rel_l2

COMM = MPI.COMM_WORLD
R_W = dz.R_W                       # 0.05 m feature radius (matches Phase-1 references)
PSI_I, PSI_WALL = -1.0, 0.0        # disperse: saturated wall into dry soil (the 1b scenario)
CELL = 0.5 / 400.0                 # 1b's converged radial cell size (n=400 on the 0.5 m span)
N_SAMP = 32

SOIL_NAMES = ("LOAM", "SAND", "SILT")          # CLAY stays flag-excluded (Phase-3 decision)
R_FACTORS = (3, 5, 10)
# WINDOW DESIGN (measured, 2026-06-10 -- the first two runs taught this): the cumulative I(t) lags the
# flux divergence, so a window stopping at ~90% depletion lets the offline clock PASS (relL2 0.5-2%!);
# the discrimination lives in the DEEP BEND + PLATEAU (windows past full depletion gave clock relL2
# 33-43%). So 3/5 r_w are FULL-DEPLETION discriminators: final >= 97% of I_max AND the 90% crossing in
# the first 3/4 of the window (the bend occupies the late window, not a long flat tail). 10 r_w is the
# PARTIAL-depletion control (clock ~right there -> catches an embedded scheme that over-corrects).
FULL_R = (3, 5)
T90_FRAC_MAX = 0.75
CONTROL_BAND = (0.30, 0.70)
# t_end [day] first guesses (auto-tuned, max 3 attempts): LOAM from the plan; SAND/SILT scaled by the
# Phase-1b per-soil window ratio (SAND 3e-3/6e-2, SILT 4e-1/6e-2)
T_END = {
    "LOAM": {3: 0.12, 5: 0.6, 10: 3.0},
    "SAND": {3: 6e-3, 5: 3e-2, 10: 0.15},
    "SILT": {3: 0.8, 5: 4.0, 10: 20.0},
}


def run_depleting(soil, r_out, samples, label=""):
    """1-D cylindrical radial disperse with wall Dirichlet psi=0 and a NO-FLOW outer boundary at r_out
    (the far Dirichlet of dz._run is simply omitted -> natural BC). Gravity-free (z-invariant tunnel).
    Returns I(t) [m, per wall area] at the sample times."""
    n = max(int(round((r_out - R_W) / CELL)), 40)
    msh = dmesh.create_interval(COMM, n, [R_W, r_out])
    r = ufl.SpatialCoordinate(msh)[0]
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi = fem.Function(V); psi_n = fem.Function(V)
    psi.x.array[:] = PSI_I; psi_n.x.array[:] = PSI_I
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    dxs = dz._vertex_dx()
    dxq = ufl.dx(metadata={"quadrature_degree": 8})   # quadrature-degree cap (memory: FFCX auto ~26)
    theta, theta_n, K = soil.theta_ufl(psi), soil.theta_ufl(psi_n), soil.K_ufl(psi)
    F = ((theta - theta_n) / dt_c) * v * r * dxs + K * ufl.dot(ufl.grad(psi), ufl.grad(v)) * r * dxq
    wall_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], R_W))
    bcs = [fem.dirichletbc(PETSc.ScalarType(PSI_WALL), wall_dofs, V)]   # NO outer BC: no-flow natural
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4p4a_", petsc_options=dz._LU)

    th_i = float(soil.theta(PSI_I))
    stored = fem.form((theta - th_i) * r * dxs)
    out, dt, t_prev = [], 1e-8, 0.0
    for i, t_s in enumerate(samples):
        dt = dz._solve_to(problem, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        I = COMM.allreduce(fem.assemble_scalar(stored), op=MPI.SUM) / R_W
        out.append(float(I))
        if label:
            print(f"      [{label}] {i+1}/{len(samples)}  t={t_s:.3e}  I={I:.4e}", flush=True)
    return np.array(out)


if __name__ == "__main__":
    fix = np.load("tests/data/m4_phase1b_disperse_refs.npz")   # the signed-off infinite-domain fixture
    saved = {"r_w": np.array(R_W)}
    print("=" * 88)
    print("PHASE-4 Ref A: DEPLETING-RESERVOIR disperse references  "
          f"(r_w={R_W} m, no-flow outer at 3/5/10 r_w)")
    print("=" * 88)
    for name in SOIL_NAMES:
        soil = dz.SOILS[name]
        S_an = dz.parlange_sorptivity(soil, PSI_I)
        dth = float(soil.theta(PSI_WALL) - soil.theta(PSI_I))
        t_start = (dth * 0.1 * R_W / S_an) ** 2          # 1b's early-start rule (front ~0.1 r_w)
        saved[f"{name}_S"], saved[f"{name}_dtheta"] = np.array(S_an), np.array(dth)
        print(f"\n{'-'*88}\n{name}: S_an={S_an:.5f} m/day^0.5,  dtheta={dth:.4f}")
        for k in R_FACTORS:
            r_out = k * R_W
            i_max = dth * (r_out**2 - R_W**2) / (2.0 * R_W)
            t_end = T_END[name][k]
            for attempt in range(3):
                t = np.geomspace(t_start, t_end, N_SAMP)
                I = run_depleting(soil, r_out, t, label=f"{name} R={k}r_w (t_end={t_end:.3g})")
                frac = I[-1] / i_max
                if k in FULL_R:
                    if frac < 0.97:
                        t_end *= 3.0
                        print(f"   R={k}r_w: depleted {frac:.1%} < 97% -> t_end x3 = {t_end:.3g}", flush=True)
                        continue
                    t90 = float(np.interp(0.90, I / i_max, t))
                    if t90 / t_end > T90_FRAC_MAX:
                        t_end = t90 / 0.5                  # put the 90% crossing at mid-window
                        print(f"   R={k}r_w: t90/t_end={t90/t_end:.2f} > {T90_FRAC_MAX} -> "
                              f"t_end = {t_end:.3g}", flush=True)
                        continue
                    break
                else:                                       # 10 r_w: partial-depletion control
                    lo, hi = CONTROL_BAND
                    if frac < lo:
                        t_end *= 3.0
                        print(f"   R={k}r_w: depleted {frac:.1%} < {lo:.0%} -> t_end x3 = {t_end:.3g}",
                              flush=True)
                    elif frac > hi:
                        t_end = float(np.interp(0.5 * (lo + hi), I / i_max, t))
                        print(f"   R={k}r_w: depleted {frac:.1%} > {hi:.0%} -> t_end at mid-band = "
                              f"{t_end:.3g}", flush=True)
                    else:
                        break
            else:
                raise SystemExit(f"{name} R={k}: window auto-tune failed after 3 attempts")

            # -- self-checks -------------------------------------------------------------------
            assert np.all(np.diff(I) >= -1e-12 * i_max), f"{name} R={k}: I not monotone"
            assert I[-1] <= i_max * (1.0 + 1e-9), \
                f"{name} R={k}: capacity violated (I_end={I[-1]:.4e} > I_max={i_max:.4e})"
            # early window vs the infinite-domain fixture (front far from the boundary)
            tf, If = fix[f"{name}_t"], fix[f"{name}_tunnel_I"]
            early = (I < 0.3 * i_max) & (t >= tf[0]) & (t <= tf[-1])
            dev = (np.abs(I[early] - np.interp(t[early], tf, If)) / np.interp(t[early], tf, If)
                   if np.any(early) else np.array([np.inf]))
            assert np.any(early) and dev.max() < 0.015, \
                f"{name} R={k}: early-window mismatch vs Phase-1b fixture (max {dev.max():.2%}, n={early.sum()})"
            # the offline clock on the same grid (the discrimination candidate, asserted in Task 4) and
            # the CAPACITY-CLAMPED clock (the cheapest passive capacity-respecting scheme an adversary
            # would try: clamp(clock, I_max) -- its plateau APPROACH is a kink, the resolved one smooth)
            clk = sorptive_clock(t, S_an, dth, R_W, F_cylindrical)
            msg = (f"   R={k:2d}r_w: I_end={I[-1]:.4e}  I_max={i_max:.4e}  depleted={frac:.1%}  "
                   f"early-dev={dev.max():.2%}(n={int(early.sum())})  OFFLINE-CLOCK relL2={rel_l2(clk, I):.1%}")
            if k in FULL_R:
                msg += f"  CLAMPED-CLOCK relL2={rel_l2(np.minimum(clk, i_max), I):.1%}"
            print(msg, flush=True)
            saved[f"{name}_R{k}_t"], saved[f"{name}_R{k}_I"] = t, I
            saved[f"{name}_R{k}_Imax"] = np.array(i_max)

    out_path = "scratch/m4_phase4_refA_disperse.npz"
    np.savez(out_path, **saved)
    print(f"\nSaved Ref-A depleting-reservoir tables -> {out_path}")
    print("=" * 88)
