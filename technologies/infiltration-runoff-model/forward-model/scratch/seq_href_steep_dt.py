"""SCRATCH DIAGNOSTIC -- is b1_STEEP's B over-routing an O(dt) operator-split ORDER error, or structural?

The closure study (seq_href_closure2) found B over-routes b1_steep (S=0.10) at +26pp vs the monolith
(0.81-0.85 vs 0.551), FLAT in h_ref. Hypothesis: route-first books outflow before infiltration claims
its ~Ks share; the split order error is O(dt), so finer dt -> routed/R falls toward the monolith. This
run pins it: b1_steep @ h_ref=2mm at decreasing dt_max. Known point: dt_max=0.004 -> 0.813.
O(dt) prediction (linear toward 0.551): dt_max=0.002 -> ~0.68, 0.001 -> ~0.61. Plateau near 0.81 =>
structural (=> reformulate the split).

Run (WSL pids-fem) -- guarded, live (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_href_steep_dt.py'
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
# b1_steep fixture (S=0.10); monolith target = 0.5508
Lx, Ly, Lz, S0, PSI_I, RAIN, STORM, TEND, NMAN, MESH = \
    8.0, 5.0, 1.0, 0.10, -0.4, 0.5, 0.08, 0.45, 0.05, (30, 20, 8)
MONO = 0.5508


def run(h_ref, dt_max, rs=4):
    soil = VanGenuchten(**LOAM)
    msh = make_box(*MESH, Lx, Ly, Lz)
    prob = HrefCappedPondInPsi(msh, soil, n_man=NMAN, route_substeps=rs, h_ref=h_ref)
    prob.set_initial_condition(lambda x: PSI_I + 0.0 * x[0])
    prob.set_topography(lambda x: S0 * x[1])
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=S0)
    top_area = _top_area_ds(msh, Lz)
    R_in = RAIN * top_area * STORM
    t0 = time.perf_counter()
    ns, coll, tend, _ = _march(prob, rain_c, storm_dur=STORM, storm_rain=RAIN, t_end=TEND,
                               dt0=dt_max / 4.0, dt_max=dt_max, max_steps=1500)
    ok = (not coll) and tend >= TEND - 1e-9
    return dict(routed=prob.cum_outflow / R_in, bal=abs(prob.balance()) / prob.cum_rain,
                ns=ns, ok=ok, wall=time.perf_counter() - t0)


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("#" * 86)
    print(f"b1_STEEP (S=0.10) B @ h_ref=2mm -- dt-convergence.  monolith target = {MONO}")
    print(f"  known: dt_max=0.004 -> 0.813 (+26pp).  O(dt) => falls toward {MONO}; plateau => structural.")
    print("#" * 86, flush=True)
    pts = [(0.004, 0.8125)]   # the closure2 value (re-printed for the trend)
    for dtm in (0.002, 0.001):
        r = run(0.002, dtm)
        pts.append((dtm, r["routed"]))
        print(f"  dt_max={dtm:.4f}: routed/R={r['routed']:.4f}  (gap to monolith {r['routed']-MONO:+.3f}) "
              f"bal={r['bal']:.1e} ok={r['ok']} [{r['ns']} steps {r['wall']:.0f}s t]", flush=True)
    print("\n  trend (dt_max -> routed/R):  " +
          "  ".join(f"{d:.4f}->{v:.4f}" for d, v in pts), flush=True)
    # crude linear-in-dt extrapolation to dt=0 from the two finest points.
    (d1, v1), (d2, v2) = pts[-2], pts[-1]
    if d1 != d2:
        v0 = v2 + (v2 - v1) * (0.0 - d2) / (d2 - d1)
        print(f"  linear-in-dt extrapolation to dt->0:  routed/R ~ {v0:.4f}  "
              f"(monolith {MONO}; gap {v0-MONO:+.3f})", flush=True)
    print("#" * 86, flush=True)


if __name__ == "__main__":
    main()
