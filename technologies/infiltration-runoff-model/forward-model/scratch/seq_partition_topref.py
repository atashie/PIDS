"""SCRATCH TEST -- the DECISIVE 3-D test (4th Codex review): does the real routed runoff PARTITION
(the "0.547") collapse when ONLY the top vertical resolution is refined?

The 1-D sorptivity ladder (§21) showed coarse q_pot under-captures sorptive uptake (resolution-driven),
but that is a 1-D no-routing column -- it does NOT prove the ROUTED 3-D 0.547 partition moves. Codex's
highest-value test: take the ACTUAL b1 tilted-plane case (the monolith galerkin partition = 0.547 base /
0.551 steep), FIX the lateral mesh (30x20) + forcing, and refine ONLY the top vertical resolution via a
z-grading ladder (uniform -> ~2mm top cell; nz=8 fixed, cells concentrate near the surface; ell_c
auto-locks to the local top-cell half-height). Track routed/R, infiltration fraction, peak pond, balance,
solver health.

  * If routed/R COLLAPSES materially as the top refines -> the §21 sorptivity story CARRIES to the
    partition; the coarse 0.547 is under-resolved; a mesh-objective subgrid closure is needed.
  * If routed/R STAYS ~0.547 -> vertical sorptivity is NOT the partition driver (the 1-D result does not
    carry over; the routing<->infiltration feedback dominates).

(§18 already has two endpoints: b1_steep monolith uniform 0.551 -> 2mm-skin 0.254. This fills the ladder.)

Run (WSL pids-fem) -- LIVE to a file (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_partition_topref.py <case> <p>'
    case in {b1_base (S=0.03), b1_steep (S=0.10)};  p = z-grading (1.0 uniform .. 2.5 ~2mm top)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from scratch.seq_cocycled_skin import make_graded_box
from scratch.seq_href_closure_study import _march
from scratch.seq_iterative_prototype import _top_area_ds, _soil_water_deg8

COMM = MPI.COMM_WORLD
LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
# the documented b1 partition case (PSI_I=-0.4 -- the ACTUAL setup that gave 0.547, NOT the -3m dry IC).
B1 = dict(Lx=8.0, Ly=5.0, Lz=1.0, PSI_I=-0.4, RAIN=0.5, STORM=0.08, TEND=0.45, NMAN=0.05, nx=30, ny=20)
CASES = {"b1_base": dict(S0=0.03, target=0.5470), "b1_steep": dict(S0=0.10, target=0.5508)}


def top_cell_mm(Lz, nz, p):
    return Lz * (1.0 / nz) ** p * 1000.0


def run(case_name, p, nz=8, dt_max=0.02):
    c = B1
    case = CASES[case_name]
    soil = VanGenuchten(**LOAM)
    msh = make_graded_box(c["nx"], c["ny"], nz, c["Lx"], c["Ly"], c["Lz"], p=p)
    mono = CoupledProblem(msh, soil, n_man=c["NMAN"], overland_scheme="galerkin")  # ell_c AUTO = top dz/2
    mono.set_initial_condition(lambda x: c["PSI_I"] + 0.0 * x[0], d_value=0.0)
    mono.set_topography(lambda x: case["S0"] * x[1])
    rain_c = mono.add_rain(0.0)
    mono.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=case["S0"])
    th0 = _soil_water_deg8(mono, soil)
    top_area = _top_area_ds(msh, c["Lz"])
    R_in = c["RAIN"] * top_area * c["STORM"]
    t0 = time.perf_counter()
    ns, coll, tend, peak_sheet = _march(mono, rain_c, storm_dur=c["STORM"], storm_rain=c["RAIN"],
                                        t_end=c["TEND"], dt0=dt_max / 4.0, dt_max=dt_max,
                                        ctrl_low=3, ctrl_high=8, track_sheet=True, top_area=top_area,
                                        max_steps=900)
    ok = (not coll) and tend >= c["TEND"] - 1e-9
    soil_gain = _soil_water_deg8(mono, soil) - th0
    routed = mono.cum_outflow / R_in
    infil = (soil_gain + mono.cum_drainage) / R_in
    # conservation: dtotal vs rain - outflow - drainage (+clip)
    bal = abs(getattr(mono, "clip_mass_adjust", 0.0))
    return dict(routed=routed, infil=infil, peak_sheet=peak_sheet, ell_c=mono.ell_c, ns=ns, ok=ok,
                clip=bal, wall=time.perf_counter() - t0,
                eff=getattr(mono, "_effective_overland_scheme", "galerkin"))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    case_name = sys.argv[1] if len(sys.argv) > 1 else "b1_base"
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    tc = top_cell_mm(B1["Lz"], 8, p)
    r = run(case_name, p)
    case = CASES[case_name]
    print("#" * 96)
    print(f"3-D TOP-REFINEMENT -- {case_name} (S={case['S0']}), z-grade p={p} -> top cell {tc:.2f}mm "
          f"(uniform=125mm); lateral 30x20 FIXED")
    print(f"  routed/R = {r['routed']:.4f}  (uniform target {case['target']:.4f}; "
          f"gap {(r['routed']-case['target'])*100:+.1f}pp)")
    print(f"  infil/R  = {r['infil']:.4f}   peak_sheet = {r['peak_sheet']*1000:.3f}mm   ell_c = "
          f"{r['ell_c']*1000:.2f}mm")
    print(f"  ok={r['ok']} ns={r['ns']} clip={r['clip']:.1e} wall={r['wall']:.0f}s eff={r['eff']}")
    print("#" * 96, flush=True)
