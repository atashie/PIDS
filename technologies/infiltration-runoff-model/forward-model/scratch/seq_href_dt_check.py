"""SCRATCH DIAGNOSTIC -- is realization B's partition dt-CONVERGENT, or dt-sensitive?

The closure study found B at b1 h_ref=2mm gives routed/R=0.441 (dt_max=0.02) vs the earlier spike's
0.546 (dt_max=0.03) -- a ~10 pp swing from the time-step config alone, plus a NON-MONOTONE h_ref sweep.
This grid pins it: routed/R(h_ref, dt_max) on b1 loam. If routed/R converges as dt_max->0 (to a value we
can then calibrate h_ref against), B is salvageable; if it swings without converging, the explicit
route/cap/infiltrate split makes the partition ill-posed (=> reconsider, e.g. the implicit NCP A).
Monolith target ~0.547.

Run (WSL pids-fem) -- guarded, live (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_href_dt_check.py'
"""
from __future__ import annotations

import time

import numpy as np
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from scratch.seq_href_cap_spike import HrefCappedPondInPsi
from scratch.seq_iterative_prototype import _top_area_ds
from scratch.seq_href_closure_study import make_box, _march

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
F = dict(Lx=8.0, Ly=5.0, Lz=1.0, S0=0.03, psi_i=-0.4, RAIN=0.5, STORM=0.08, TEND=0.45,
         n=0.05, mesh=(30, 20, 8))


def run(h_ref, dt_max, rs=4):
    soil = VanGenuchten(**LOAM)
    msh = make_box(*F["mesh"], F["Lx"], F["Ly"], F["Lz"])
    prob = HrefCappedPondInPsi(msh, soil, n_man=F["n"], route_substeps=rs, h_ref=h_ref)
    prob.set_initial_condition(lambda x: F["psi_i"] + 0.0 * x[0])
    prob.set_topography(lambda x: F["S0"] * x[1])
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=F["S0"])
    top_area = _top_area_ds(msh, F["Lz"])
    R_in = F["RAIN"] * top_area * F["STORM"]
    t0 = time.perf_counter()
    ns, coll, tend, _ = _march(prob, rain_c, storm_dur=F["STORM"], storm_rain=F["RAIN"],
                               t_end=F["TEND"], dt0=dt_max / 4.0, dt_max=dt_max, max_steps=600)
    return dict(routed=prob.cum_outflow / R_in, bal=abs(prob.balance()) / prob.cum_rain,
                ns=ns, coll=coll, tend=tend, wall=time.perf_counter() - t0)


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("#" * 88)
    print("B partition dt-CONVERGENCE on b1 loam (monolith target routed/R ~ 0.547)")
    print("#" * 88, flush=True)
    H = [0.001, 0.002, 0.004]      # mm caps
    DTM = [0.03, 0.01, 0.004]      # decreasing dt_max (finer storm resolution)
    print(f"\n{'h_ref':>7} |" + "".join(f" dt_max={d:<6}" for d in DTM), flush=True)
    grid = {}
    for h in H:
        cells = []
        for d in DTM:
            r = run(h, d)
            grid[(h, d)] = r
            flag = "" if (not r["coll"] and r["tend"] >= F["TEND"] - 1e-9) else "!"
            cells.append(f" {r['routed']:.4f}{flag:<1}[{r['ns']:>3}]")
            print(f"   h={h*1000:.0f}mm dt_max={d:.3f}: routed/R={r['routed']:.4f} bal={r['bal']:.1e} "
                  f"[{r['ns']} steps {r['wall']:.0f}s coll={r['coll']} t={r['tend']:.3f}]", flush=True)
        print(f"{h*1000:>5.0f}mm |" + "".join(cells), flush=True)

    print("\n" + "#" * 88)
    print("READ: down each column (fixed h_ref, dt_max->0) -> does routed/R CONVERGE?  Across each row")
    print("(fixed dt_max) -> is it MONOTONE in h_ref?  '!' = did not complete (coll/max_steps).")
    print("#" * 88, flush=True)


if __name__ == "__main__":
    main()
