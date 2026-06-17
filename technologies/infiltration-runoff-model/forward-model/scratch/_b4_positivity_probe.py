"""B4 probe 3: characterize the 2-D upwind positivity (run-min d) vs scenario steepness.

The 1-D steep 5% front is bit-strict (run-min ~6e-34); a first 2-D 'steep' scenario undershot
~0.86 mm. Question: is the 2-D scheme genuinely strict on a STEEP front (head-drop >> eps_H), or
does the undershoot track the front head-drop / geometry exactly as B3 documented for 1-D? Probe
several 2-D scenarios + slopes + eps_H, mirroring the 1-D steep structure (blob ON the slope,
front advancing downslope into dry ground). Report run-min d and conservation drift.
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland_upwind import UpwindOverlandProblem

N_MAN = 0.03
TRI = dmesh.CellType.triangle


def march(prob, t_end, dt0=1e-4, dt_max=5e-3, grow=1.5, cut=0.5, dt_min=1e-9,
          target_low=3, target_high=8, shrink=0.7):
    t, dt, nsteps = 0.0, dt0, 0
    min_d = float(prob.d.x.array.min())
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        conv, iters = prob.step(h)
        if conv:
            t += h; nsteps += 1
            min_d = min(min_d, float(prob.d.x.array.min()))
            if iters <= target_low: dt = min(dt * grow, dt_max)
            elif iters >= target_high: dt *= shrink
        else:
            dt = h * cut
            if dt < dt_min:
                print("  NOT CONVERGING"); break
    return t, nsteps, min_d


def scenario_xslope(slope, eps_H, t_end=0.01, nx=24, ny=12):
    """Blob on the slope, front advances downslope in x (1-D-steep analogue, extruded in y)."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (20.0, 5.0)], [nx, ny], cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN, eps_H=eps_H)
    prob.set_topography(lambda x: slope * (20.0 - x[0]))  # slopes down toward x=20 (as the 1-D test)
    prob.set_initial_condition(lambda x: 0.25 * np.exp(-((x[0] - 7.0) / 1.5) ** 2))
    w0 = prob.total_water()
    t, nsteps, min_d = march(prob, t_end)
    drift = abs(prob.total_water() - w0) / max(1.0, abs(w0))
    return nsteps, min_d, drift


def scenario_vslope(slope, eps_H, t_end=0.01):
    """Blob off-channel on a cross-slope V (the first 2-D 'steep' attempt)."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (2.0, 1.0)], [24, 12], cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN, eps_H=eps_H)
    prob.set_topography(lambda x: slope * np.abs(x[1] - 0.5) / 0.5)
    prob.set_initial_condition(lambda x: 0.2 * np.exp(-(((x[0] - 1.0) ** 2 + (x[1] - 0.2) ** 2) / 0.02)))
    w0 = prob.total_water()
    t, nsteps, min_d = march(prob, t_end)
    drift = abs(prob.total_water() - w0) / max(1.0, abs(w0))
    return nsteps, min_d, drift


def main():
    print("== x-slope front (1-D-steep analogue, extruded in y), eps_H sweep ==")
    print(f"{'slope':>6} {'eps_H':>8} {'nsteps':>7} {'min_d (m)':>14} {'drift':>10}")
    for slope in (0.05, 0.10):
        for eps_H in (1e-3, 1e-4, 1e-5):
            ns, md, dr = scenario_xslope(slope, eps_H)
            print(f"{slope:6.2f} {eps_H:8.0e} {ns:7d} {md:14.3e} {dr:10.2e}")

    print("\n== off-channel V-slope blob (first 2-D 'steep' attempt), eps_H sweep ==")
    print(f"{'slope':>6} {'eps_H':>8} {'nsteps':>7} {'min_d (m)':>14} {'drift':>10}")
    for slope in (0.05, 0.10):
        for eps_H in (1e-3, 1e-4, 1e-5):
            ns, md, dr = scenario_vslope(slope, eps_H)
            print(f"{slope:6.2f} {eps_H:8.0e} {ns:7d} {md:14.3e} {dr:10.2e}")


if __name__ == "__main__":
    main()
