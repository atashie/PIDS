"""P0 review follow-up (F9): the CAUSAL control for "the dt-pin is the limiter<->Newton fight".

March the standalone 2-D V (48x30) to the storm plateau, then compare, from the SAME state:
  (a) 12 normal steps (limiter active, spike controller)        -> iters per step
  (b) 12 steps with the positivity limiter BYPASSED (the converged iterate, negatives and
      all, becomes d_n)                                          -> iters per step
  (c) from the same plateau state, limiter active but dt DOUBLED -> does Newton degrade?

If (b) collapses to ~2-3 iterations while (a) sits at ~6, the limiter's global-rescale
perturbation IS the per-step re-equilibration cost (causal); if (b) stays ~6, the cost is
intrinsic front stiffness and the mechanism claim must soften to correlational.

Run (pids-fem): PYTHONPATH=. python scratch/_v2d_limiter_control.py
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland import OverlandProblem

LX, LY = 1620.0, 1000.0
XC = LX / 2.0
SX, SY = 0.05, 0.02
N_MAN = 0.015
RAIN = 0.2592
Q_EQ = RAIN * LX * LY
NX, NY = 48, 30
T_PLATEAU = 0.035


def z_b(x):
    return SY * (LY - x[1]) + SX * np.abs(x[0] - XC)


def march(prob, rain, t0, t_end, dt0, label, bypass=False, dt_cap=1e-3, record=None):
    t, dt = t0, dt0
    if bypass:
        orig = prob._enforce_positivity
        prob._enforce_positivity = lambda: 0.0  # keep the converged iterate verbatim
    try:
        while t < t_end - 1e-12:
            h = min(dt, t_end - t)
            rain.value = RAIN
            converged, it = prob.step(h)
            if not converged:
                dt = h * 0.5
                if record is not None:
                    record.append((t, h, it, prob.last_reason, prob.last_fnorm, False, np.nan))
                continue
            t += h
            dmin = float(prob.d.x.array.min())
            if record is not None:
                record.append((t, h, it, prob.last_reason, prob.last_fnorm, True, dmin))
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), dt_cap)
    finally:
        if bypass:
            prob._enforce_positivity = orig
    return t, dt


def fresh_to_plateau():
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [LX, LY]], [NX, NY])
    prob = OverlandProblem(msh, N_MAN)
    prob.set_topography(z_b)
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)
    t, dt = march(prob, rain, 0.0, T_PLATEAU, 1e-5, "spinup")
    return prob, rain, t, dt


def run_block(prob, rain, t, dt, n, label, bypass=False, dt_fixed=None):
    rec = []
    dt_use = dt_fixed if dt_fixed is not None else dt
    tt = t
    for _ in range(n):
        rain.value = RAIN
        if bypass:
            orig = prob._enforce_positivity
            prob._enforce_positivity = lambda: 0.0
        try:
            converged, it = prob.step(dt_use)
        finally:
            if bypass:
                prob._enforce_positivity = orig
        dmin = float(prob.d.x.array.min())
        rec.append((it, int(prob.last_reason), float(prob.last_fnorm), bool(converged), dmin))
        if converged:
            tt += dt_use
    its = [r[0] for r in rec if r[3]]
    print(f"[{label}] dt={dt_use:.2e}  iters={its}  "
          f"min(d) range [{min(r[4] for r in rec):+.4f}, {max(r[4] for r in rec):+.4f}] m  "
          f"rejected={sum(1 for r in rec if not r[3])}", flush=True)
    return tt


def main():
    print(f"[setup] standalone 2-D V {NX}x{NY}, spin to t={T_PLATEAU} (plateau)", flush=True)
    prob, rain, t, dt = fresh_to_plateau()
    q = prob.last_outflow
    print(f"[plateau] t={t:.4f}  q/Qeq={q/Q_EQ:.3f}  dt={dt:.2e}", flush=True)
    d_save = prob.d.x.array.copy()
    dn_save = prob.d_n.x.array.copy()

    # (a) control: limiter ON, the pinned dt
    run_block(prob, rain, t, dt, 12, "a: limiter ON ", bypass=False)

    # (b) counterfactual: limiter BYPASSED, same start state, same dt
    prob.d.x.array[:] = d_save; prob.d_n.x.array[:] = dn_save
    prob.d.x.scatter_forward(); prob.d_n.x.scatter_forward()
    run_block(prob, rain, t, dt, 12, "b: limiter OFF", bypass=True)

    # (c) limiter ON, dt doubled (does the churn scale with dt?)
    prob.d.x.array[:] = d_save; prob.d_n.x.array[:] = dn_save
    prob.d.x.scatter_forward(); prob.d_n.x.scatter_forward()
    run_block(prob, rain, t, dt, 12, "c: limiter ON 2dt", bypass=False, dt_fixed=2 * dt)

    # (d) limiter OFF, dt doubled
    prob.d.x.array[:] = d_save; prob.d_n.x.array[:] = dn_save
    prob.d.x.scatter_forward(); prob.d_n.x.scatter_forward()
    run_block(prob, rain, t, dt, 12, "d: limiter OFF 2dt", bypass=True, dt_fixed=2 * dt)


if __name__ == "__main__":
    main()
