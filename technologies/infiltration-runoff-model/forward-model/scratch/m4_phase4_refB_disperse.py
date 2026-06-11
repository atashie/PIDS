"""Module 4 (§E) Phase-4 Ref B: HOST-HISTORY disperse reference (mid-uptake far-field change).

Ref A (depleting reservoir) catches a passive accumulator via the capacity bend; Ref B catches it via
HISTORY: the far field changes MID-uptake, and the uptake curve must track that change -- the offline
clock (which knows only its own state I) cannot, BY CONSTRUCTION. Scenario: the Ref-A closed domain at
R_out = 5 r_w (LOAM), plus a VOLUMETRIC RE-WETTING PULSE -- a uniform source over the annular band
r in [3, 4] r_w, active over [0.05, 0.10]*t_end (while the wetting front is still inside ~3 r_w), with
total volume = 30% of the band's water capacity. The pulse wets the soil ahead of the front -> the
sorptive uptake through the wall slows -> I_wall(t) drops below the Ref-A(5 r_w) curve by a clear,
measured margin (asymptotically the wall's share of the capacity shrinks by exactly the pulse volume).

Wall uptake in a closed domain with a known source is EXACT bookkeeping:
    I_wall(t) = [ integral (theta - theta_i) r dr  -  cum_source(t) ] / r_w
(only the wall Dirichlet and the source add water). The source bookkeeping itself is verified by a
SEALED control (no wall Dirichlet at all): gain must equal cum_source to machine precision. (The plan's
wall-reaction cross-check is superseded by this control -- simpler and exact.)

Saturation guard: if the band's psi exceeds -0.01 m during the pulse (local saturation -> ponding
stiffness + a corrupted scenario), the run retries with half the pulse volume (max 2 retries, printed).

Plan: docs/plans/2026-06-10-m4-phase4-coupled-embedding-plan.md (Task 2). Requires the Task-1 output
scratch/m4_phase4_refA_disperse.npz (pre-pulse cross-check).
Run from forward-model/:
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scratch/m4_phase4_refB_disperse.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

import scratch.m4_phase1b_disperse_reference as dz
from pids_forward.physics.sorptive_closure import F_cylindrical, sorptive_clock, rel_l2

COMM = MPI.COMM_WORLD
R_W = dz.R_W
PSI_I, PSI_WALL = -1.0, 0.0
R_OUT = 5 * R_W                      # 0.25 m (the Ref-A mid case)
BAND = (3 * R_W, 4 * R_W)            # the pulse band [0.15, 0.20] m
PULSE_FRAC = (0.05, 0.10)            # active window as fractions of t_end
PULSE_FILL = 0.30                    # pulse volume = this fraction of the band's full capacity
T_END = 0.6                          # = Ref A LOAM 5 r_w window
N_SAMP = 32
CELL = 0.5 / 400.0
SOIL_NAME = "LOAM"                   # Ref B is a discrimination instrument; one soil suffices


def run_refB(soil, t_grid, t1, t2, s_rate, *, sealed=False, label=""):
    """Closed-domain radial disperse + the band source. sealed=True drops the wall Dirichlet entirely
    (the source-bookkeeping control). Returns (I_wall(t), max band psi during pulse, gain(t))."""
    n = max(int(round((R_OUT - R_W) / CELL)), 40)
    msh = dmesh.create_interval(COMM, n, [R_W, R_OUT])
    r = ufl.SpatialCoordinate(msh)[0]
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi = fem.Function(V); psi_n = fem.Function(V)
    psi.x.array[:] = PSI_I; psi_n.x.array[:] = PSI_I
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    s_c = fem.Constant(msh, PETSc.ScalarType(0.0))           # the pulse rate, toggled by segment
    dxs = dz._vertex_dx()
    dxq = ufl.dx(metadata={"quadrature_degree": 8})
    theta, theta_n, K = soil.theta_ufl(psi), soil.theta_ufl(psi_n), soil.K_ufl(psi)
    x = ufl.SpatialCoordinate(msh)[0]
    band_ind = ufl.conditional(ufl.And(ufl.ge(x, BAND[0]), ufl.le(x, BAND[1])), 1.0, 0.0)
    F = (((theta - theta_n) / dt_c) * v * r * dxs
         + K * ufl.dot(ufl.grad(psi), ufl.grad(v)) * r * dxq
         - s_c * band_ind * v * r * dxq)
    bcs = []
    if not sealed:
        wall_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], R_W))
        bcs = [fem.dirichletbc(PETSc.ScalarType(PSI_WALL), wall_dofs, V)]
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4p4b_", petsc_options=dz._LU)

    th_i = float(soil.theta(PSI_I))
    stored = fem.form((theta - th_i) * r * dxs)
    xc = V.tabulate_dof_coordinates()[:, 0]
    in_band = (xc >= BAND[0]) & (xc <= BAND[1])
    B = (BAND[1] ** 2 - BAND[0] ** 2) / 2.0               # exact band weight integral (per radian)

    # march on the union of sample times and the pulse switch points so no step crosses a toggle
    marks = np.unique(np.concatenate([t_grid, [t1, t2]]))
    I_wall, gains, psi_band_max = [], [], -np.inf
    dt, t_prev = 1e-8, 0.0
    for t_s in marks:
        active = (t_prev >= t1 - 1e-15) and (t_s <= t2 + 1e-15)   # segment lies inside the pulse window
        s_c.value = s_rate if active else 0.0
        dt = dz._solve_to(problem, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        if active:
            psi_band_max = max(psi_band_max, float(psi.x.array[in_band].max()))
        if t_s in t_grid:
            gain = COMM.allreduce(fem.assemble_scalar(stored), op=MPI.SUM)
            active_elapsed = min(max(t_s - t1, 0.0), t2 - t1)
            cum_src = s_rate * B * active_elapsed
            gains.append(float(gain))
            I_wall.append((gain - cum_src) / R_W)
        if label:
            print(f"      [{label}] t={t_s:.3e}  done", flush=True)
    return np.array(I_wall), psi_band_max, np.array(gains)


if __name__ == "__main__":
    soil = dz.SOILS[SOIL_NAME]
    refA = np.load("scratch/m4_phase4_refA_disperse.npz")
    tA, IA = refA[f"{SOIL_NAME}_R5_t"], refA[f"{SOIL_NAME}_R5_I"]
    S_an = float(refA[f"{SOIL_NAME}_S"]); dth = float(refA[f"{SOIL_NAME}_dtheta"])
    i_max = float(refA[f"{SOIL_NAME}_R5_Imax"])

    t1, t2 = PULSE_FRAC[0] * T_END, PULSE_FRAC[1] * T_END
    B = (BAND[1] ** 2 - BAND[0] ** 2) / 2.0
    t_start = (dth * 0.1 * R_W / S_an) ** 2
    t_grid = np.geomspace(t_start, T_END, N_SAMP)

    print("=" * 88)
    print(f"PHASE-4 Ref B: HOST-HISTORY disperse ({SOIL_NAME}, R_out=5r_w, re-wetting pulse "
          f"band [3,4]r_w over t=[{t1:.3f},{t2:.3f}] d)")
    print("=" * 88)

    pulse_fill = PULSE_FILL
    for attempt in range(3):
        s_rate = pulse_fill * dth / (t2 - t1)             # fills pulse_fill of band capacity over the window
        V_pulse = s_rate * B * (t2 - t1)                  # per radian
        print(f"\nattempt {attempt+1}: pulse_fill={pulse_fill:.2f}  s={s_rate:.3f}/day  "
              f"V_pulse/wall_area={V_pulse/R_W:.4e} m", flush=True)

        # -- sealed control: source bookkeeping must be EXACT (machine precision) --------------
        ctrl_grid = np.array([t1, 0.5 * (t1 + t2), t2, 1.2 * t2])
        Ic, _, gc = run_refB(soil, ctrl_grid, t1, t2, s_rate, sealed=True)
        cum_end = s_rate * B * (t2 - t1)
        ctrl_err = abs(gc[-1] - cum_end) / cum_end
        assert ctrl_err < 1e-9, f"sealed-control source bookkeeping off: rel err {ctrl_err:.2e}"
        print(f"   sealed control: gain={gc[-1]:.6e} vs cum_source={cum_end:.6e}  rel err={ctrl_err:.1e}  OK")

        I_B, band_max, _ = run_refB(soil, t_grid, t1, t2, s_rate, label=f"{SOIL_NAME} RefB")
        if band_max <= -0.01:
            break
        pulse_fill *= 0.5
        print(f"   band psi reached {band_max:+.3f} m during pulse (>-0.01) -> halving pulse volume")
    else:
        raise SystemExit("Ref B: pulse saturates the band even after 2 halvings -- redesign timing")

    # -- checks ---------------------------------------------------------------------------------
    assert np.all(np.diff(I_B) >= -1e-12 * i_max), "I_wall not monotone"
    pre = t_grid < t1
    devA = np.abs(I_B[pre] - np.interp(t_grid[pre], tA, IA)) / np.interp(t_grid[pre], tA, IA)
    assert devA.max() < 0.005, f"pre-pulse mismatch vs Ref A(5r_w): {devA.max():.2%}"
    IA_end = float(np.interp(t_grid[-1], tA, IA))
    gap = (IA_end - I_B[-1]) / IA_end
    assert gap > 0.03, f"history response too small to discriminate: end gap {gap:.2%}"
    clk = sorptive_clock(t_grid, S_an, dth, R_W, F_cylindrical)
    print(f"\n   pre-pulse dev vs RefA(5r_w) = {devA.max():.2%}  |  band psi_max during pulse = {band_max:+.3f} m")
    print(f"   end I_B={I_B[-1]:.4e} vs RefA {IA_end:.4e}  -> history gap = {gap:.1%}")
    print(f"   OFFLINE-CLOCK relL2 vs Ref B = {rel_l2(clk, I_B):.1%}")

    np.savez("scratch/m4_phase4_refB_disperse.npz",
             **{f"{SOIL_NAME}_t": t_grid, f"{SOIL_NAME}_I": I_B,
                f"{SOIL_NAME}_t_pulse": np.array([t1, t2]),
                f"{SOIL_NAME}_V_pulse_per_wall_area": np.array(s_rate * B * (t2 - t1) / R_W),
                f"{SOIL_NAME}_Imax": np.array(i_max),
                f"{SOIL_NAME}_S": np.array(S_an), f"{SOIL_NAME}_dtheta": np.array(dth),
                "r_w": np.array(R_W)})
    print("\nSaved Ref-B host-history table -> scratch/m4_phase4_refB_disperse.npz")
    print("=" * 88)
