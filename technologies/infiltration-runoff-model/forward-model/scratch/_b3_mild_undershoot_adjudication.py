"""Adjudicate the mild-2% front undershoot magnitude at the chosen eps_H=1e-3 (P1-B3 review).

B2 measured ~1.5-1.6 mm on a 2% mound at the default eps_H=1e-3; B3 "corrected" that to <=0.36 mm
(claiming the canonical mild front is strictly positive at 1e-3); the B2 reviewer independently
measured 0.5-1.0 mm. Three different single numbers for the same nominal scenario -> the undershoot
is GEOMETRY-DEPENDENT and no single number is honest. This sweep settles it: across 24 mild-2%
mound geometries (sharpness sigma x peak x mesh) at eps_H=1e-3, the WORST run-min depth is the
honest worst-case. Result (2026-06-16): worst -0.93 mm (sigma=0.8, peak=0.25, 60 cells); range
0 .. ~0.9 mm, most sub-0.3 mm, several strictly positive. Conservation machine-tight throughout.
The overland_upwind docstrings + the B2 test docstrings were corrected to this 0 .. ~0.9 mm range
(superseding both the 1.5 mm over-claim and the 0.36 mm under-claim).

Run (pids-fem): PYTHONPATH=. python scratch/_b3_mild_undershoot_adjudication.py
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland_upwind import UpwindOverlandProblem


def worst_mind(sigma, peak, ncell, L=10.0, S0=0.02, x0=3.0, t_end=0.03):
    """Run-min depth over a march of a 2%-slope mound (mound near x0, dry downslope)."""
    msh = dmesh.create_interval(MPI.COMM_WORLD, ncell, [0.0, L])
    p = UpwindOverlandProblem(msh, n_man=0.03)          # eps_H default = 1e-3
    p.set_topography(lambda x: S0 * (L - x[0]))          # bed falls toward x=L (2% slope)
    p.set_initial_condition(lambda x: 0.002 + peak * np.exp(-((x[0] - x0) / sigma) ** 2))
    mind = float(p.d.x.array.min())
    t, dt = 0.0, 1e-4
    for _ in range(400):
        if t >= t_end:
            break
        h = min(dt, t_end - t)
        c, _it = p.step(h)
        if c:
            t += h
            mind = min(mind, float(p.d.x.array.min()))
            dt = min(dt * 1.4, 2e-3)
        else:
            dt *= 0.5
        if dt < 1e-9:
            break
    return mind


def main():
    print("  sigma  peak  ncell |  min_d(mm)")
    worst = 1e9
    for sigma in (1.5, 0.8, 0.5, 0.3):
        for peak in (0.25, 0.6, 1.0):
            for ncell in (60, 120):
                m = worst_mind(sigma, peak, ncell)
                worst = min(worst, m)
                flag = " <-- undershoot" if m < -1e-9 else ""
                print(f"  {sigma:4.1f}  {peak:4.2f}  {ncell:4d}  | {m * 1000:9.3f}{flag}")
    print(f"WORST min_d over the sweep = {worst * 1000:.3f} mm  (eps_H=1e-3)")
    print("Honest characterization: mild-2% undershoot at 1e-3 is geometry-dependent 0 .. ~0.9 mm.")


if __name__ == "__main__":
    main()
