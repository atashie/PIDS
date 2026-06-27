"""SCRATCH -- in-house CoupledProblem on the EXACT ParFlow-B5 geometry, refined NZ (Task 2 apples-to-apples).

ParFlow B5 loam_overland (coupled_hillslope_3d.py, 5x1x1, 16x6xNZ, water table z=0.35, storm 0.5 m/day x
0.30 d, overland edge + lateral-GHB at x=L) does NOT collapse with NZ: infil/R 0.327(nz8)=0.326(nz16),
then 0.252(nz32, anomalous). The in-house b1 collapse (0.547->0.265) was a DIFFERENT geometry. This runs
the in-house twin on the SAME B5 setup (mirrors viz/run_coupling_3d_sanity.py loam) at NZ=8/16/32 to
disambiguate: does the IN-HOUSE collapse on B5 (=> scheme-specific vs ParFlow) or stay flat (=> the b1
collapse was geometry/IC-specific)?

Partition (cum_rain=0.5*5*0.30=0.75 m^3), matched to ParFlow's diagnostics:
  infil_storage/R = (soil_water[-1]-soil_water[0])/R   (ParFlow 'soil dW')
  overland/R, gw/R, total_out/R = (cum_outflow + cum_drainage)/R

Run (WSL pids-fem) -- one NZ per process:
  python -u scratch/seq_b5_inhouse_refine.py <NZ> [scheme]   (scheme=galerkin default)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
L, W, H = 5.0, 1.0, 1.0
NX, NY = 16, 6
S0 = 0.05
Z_WT = 0.35
H_EXT = 0.20
RATE, STORM, TEND = 0.50, 0.30, 0.50


def run(NZ, scheme="galerkin", dt_max=2e-3):
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, H]], [NX, NY, NZ])
    prob = CoupledProblem(msh, LOAM, n_man=0.05, overland_scheme=scheme)
    prob.set_initial_condition(lambda x: Z_WT - x[2], d_value=0.0)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)
    no_ghb = os.environ.get("B5_NOGHB", "").strip() == "1"   # causal test: remove the lateral-GW buffer
    if not no_ghb:
        prob.add_drainage_bc(lambda x: np.isclose(x[0], L), conductance=0.5, external_head=H_EXT)
    top_area = L * W
    R_in = RATE * top_area * STORM
    sw0 = prob.soil_water()
    t, dt, nstep, nbad = 0.0, 2e-5, 0, 0
    peak_d = 0.0
    t0 = time.perf_counter()
    while t < TEND - 1e-12 and nstep < 4000:
        h = min(dt, TEND - t)
        rain.value = RATE if t < STORM - 1e-12 else 0.0
        conv, it = prob.step(h)
        if not conv:
            nbad += 1
            dt *= 0.5
            if dt < 1e-10:
                break
            continue
        t += h; nstep += 1
        peak_d = max(peak_d, prob.surface_depth())
        dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), dt_max)
    infil = (prob.soil_water() - sw0) / R_in
    overland = prob.cum_outflow / R_in
    gw = prob.cum_drainage / R_in
    total_out = overland + gw
    eff = getattr(prob, "_effective_overland_scheme", scheme)
    print("#" * 92)
    print(f"IN-HOUSE B5 twin -- NZ={NZ} (DZ={H/NZ*1000:.1f}mm, ell_c={H/NZ/2*1000:.1f}mm) scheme={eff}")
    print(f"  infil_storage/R = {infil:.4f}   overland/R = {overland:.4f}   gw/R = {gw:.4f}   "
          f"total_out/R = {total_out:.4f}")
    print(f"  peak_d = {peak_d*1000:.2f}mm  ok={t>=TEND-1e-9} nstep={nstep} nbad={nbad} "
          f"wall={time.perf_counter()-t0:.0f}s   (ParFlow nz8/16/32 infil/R: 0.327/0.326/0.252)")
    print("#" * 92, flush=True)


if __name__ == "__main__":
    NZ = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    scheme = sys.argv[2] if len(sys.argv) > 2 else "galerkin"
    run(NZ, scheme)
