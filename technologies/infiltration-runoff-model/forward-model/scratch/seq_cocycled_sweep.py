"""SCRATCH SWEEP -- broader soil/storm/slope validation of the THIN-SKIN co-cycled split (§13 win).

QUESTION: is the film weight w~0.7 (validated on b1 LOAM, base+steep) UNIVERSAL, or does the optimal w
shift with soil (Ks), storm intensity, or slope? For each config we compare:
  * MONOLITH target  -- galerkin CoupledProblem on the UNIFORM mesh (its q_pot=INT K dpsi/ell_c cap needs
                        ell_c; the thin skin BREAKS it, so the target MUST be the uniform mesh).
  * CO-CYCLED+SKIN   -- CoCycledCappedSplit on the z-graded thin-skin mesh at w (default 0.7).
GATE: co-cycled+skin routed/R within ~+-3-5 pp of the monolith target, STABLE, conserves ~1e-11.

Soils (van Genuchten): LOAM (b1, Ks=0.25), SAND (Ks=1.5, high-K near-sat fragility), SILT (Ks=0.10).
Each needs rain > Ks for Hortonian runoff (sand uses RAIN=2.5).

Run (WSL pids-fem, threads pinned) -- LIVE to a file (NO tail):
  python -u scratch/seq_cocycled_sweep.py <mode> <soil> <S0> <rain> <arg> [p] [nz] > scratch/_sw_<tag>.txt 2>&1
    mode=mono : arg=scheme (galerkin)         e.g.  mono loam 0.03 0.5 galerkin
    mode=cocy : arg=film_w                     e.g.  cocy loam 0.03 0.5 0.7
"""
from __future__ import annotations

import sys
import time

import numpy as np
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from scratch.seq_href_iterated import CoCycledCappedSplit
from scratch.seq_href_closure_study import make_box, _march
from scratch.seq_cocycled_skin import make_graded_box
from scratch.seq_iterative_prototype import _top_area_ds

COMM = MPI.COMM_WORLD

SOILS = {
    "loam": dict(theta_r=0.078, theta_s=0.43, alpha=3.6,  n=1.56, Ks=0.25),   # b1
    "sand": dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=1.5),    # coarse, high-K
    "silt": dict(theta_r=0.067, theta_s=0.45, alpha=2.0,  n=1.41, Ks=0.10),   # tighter than loam
}
# domain + storm (b1 geometry/timing; rain set per-call so excess = rain - Ks drives the partition)
GEOM = dict(Lx=8.0, Ly=5.0, Lz=1.0, PSI_I=-0.4, STORM=0.08, TEND=0.45, NMAN=0.05, nx=30, ny=20)


def run_one(mode, soil_name, S0, rain, arg, p=2.5, nz=12, dt_max=0.004, dt_max_mono=0.008):
    g = GEOM
    soil = VanGenuchten(**SOILS[soil_name])
    t0 = time.perf_counter()
    if mode == "mono":
        msh = make_box(g["nx"], g["ny"], 8, g["Lx"], g["Ly"], g["Lz"])           # UNIFORM (cap works)
        prob = CoupledProblem(msh, soil, n_man=g["NMAN"], overland_scheme=arg)
        prob.set_initial_condition(lambda x: g["PSI_I"] + 0.0 * x[0], d_value=0.0)
        dtm, c_lo, c_hi = dt_max_mono, 3, 8
    elif mode == "qpot":
        # ★ option A: soil-aware q_pot acceptance cap, UNIFORM mesh (validated ell_c), NO w / NO skin.
        msh = make_box(g["nx"], g["ny"], 8, g["Lx"], g["Ly"], g["Lz"])
        prob = CoCycledCappedSplit(msh, soil, n_man=g["NMAN"], route_substeps=4, h_ref=2e-3, K=6,
                                   film_mode="qpot", qpot_h_sat=float(arg))   # arg = h_sat [m]
        prob.set_initial_condition(lambda x: g["PSI_I"] + 0.0 * x[0])
        dtm, c_lo, c_hi = dt_max, 4, 12
    else:
        msh = make_graded_box(g["nx"], g["ny"], nz, g["Lx"], g["Ly"], g["Lz"], p=p)   # thin skin
        prob = CoCycledCappedSplit(msh, soil, n_man=g["NMAN"], route_substeps=4, h_ref=2e-3, K=6,
                                   film_w=float(arg))
        prob.set_initial_condition(lambda x: g["PSI_I"] + 0.0 * x[0])
        dtm, c_lo, c_hi = dt_max, 4, 12
    prob.set_topography(lambda x: S0 * x[1])
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=S0)
    top_area = _top_area_ds(msh, g["Lz"])
    R_in = rain * top_area * g["STORM"]
    ns, coll, tend, _ = _march(prob, rain_c, storm_dur=g["STORM"], storm_rain=rain, t_end=g["TEND"],
                               dt0=dtm / 4.0, dt_max=dtm, ctrl_low=c_lo, ctrl_high=c_hi, max_steps=900)
    ok = (not coll) and tend >= g["TEND"] - 1e-9
    routed = prob.cum_outflow / R_in
    bal = abs(prob.balance()) / prob.cum_rain if (mode in ("cocy", "qpot") and prob.cum_rain > 0) \
        else float("nan")
    mh = getattr(prob, "min_held_seen", 0.0)
    mp = getattr(prob, "max_pond_seen", 0.0)
    return dict(routed=routed, bal=bal, ok=ok, ns=ns, tend=tend, min_held=mh, max_pond=mp,
                wall=time.perf_counter() - t0)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    mode = sys.argv[1]; soil = sys.argv[2]; S0 = float(sys.argv[3]); rain = float(sys.argv[4])
    arg = sys.argv[5]
    p = float(sys.argv[6]) if len(sys.argv) > 6 else 2.5
    nz = int(sys.argv[7]) if len(sys.argv) > 7 else 12
    r = run_one(mode, soil, S0, rain, arg, p=p, nz=nz)
    Ks = SOILS[soil]["Ks"]
    print("#" * 92)
    print(f"SWEEP {mode}  soil={soil}(Ks={Ks}) S0={S0} rain={rain} (excess~{max(rain-Ks,0)/rain:.2f}) "
          f"arg={arg}")
    if mode == "mono":
        print(f"  MONOLITH({arg}): routed/R={r['routed']:.4f}  ok={r['ok']} ns={r['ns']} "
              f"t={r['tend']:.3f} wall={r['wall']:.0f}s", flush=True)
    elif mode == "qpot":
        print(f"  Q_POT-CAP (uniform, h_sat={arg}): routed/R={r['routed']:.4f}  bal/rain={r['bal']:.2e} "
              f"max_pond={r['max_pond']*1000:.3f}mm (cap-reject if >0)  min_held={r['min_held']*1000:.2f}mm")
        print(f"     ok={r['ok']} ns={r['ns']} t={r['tend']:.3f} wall={r['wall']:.0f}s  "
              f"=> {'STABLE' if r['ok'] else 'COLLAPSED'}", flush=True)
    else:
        print(f"  CO-CYCLED+SKIN w={arg}: routed/R={r['routed']:.4f}  bal/rain={r['bal']:.2e} "
              f"min_held={r['min_held']*1000:.2f}mm  ok={r['ok']} ns={r['ns']} t={r['tend']:.3f} "
              f"wall={r['wall']:.0f}s  => {'STABLE' if r['ok'] else 'COLLAPSED'}", flush=True)
    print("#" * 92, flush=True)
