"""SCRATCH VIZ DATA -- infiltration & runoff for the three soil classes (sand/loam/clay), on the b1
tilted-plane, at a RESOLVED surface (the corrected physics, §22). Runs the real CoupledProblem, snapshots
the fields needed for three views, saves an npz per soil. The HTML is built by make_partition_viz_html.py.

Views the npz feeds:
  * TOP-DOWN: surface ponding depth d(x,y) on the tilted plane (outlet at y=0) at snapshot times.
  * PROFILE: vertical water-content theta(z) at the centre column at snapshot times (the wetting front).
  * TIME SERIES: cumulative infiltration / runoff / ponding depths vs time (the partition over the storm).

Storm: rain=2.0 m/day for STORM=0.08 day (~3.8 h, ~160 mm) on a gentle S=0.05 plane, PSI_I=-0.4; intense
enough that all three soils show BOTH infiltration AND runoff so the soil contrast is legible. Resolved
graded mesh (p=2.5, nz=16 -> ~2 mm top skin) so the partition is near the mesh-converged value (NOT the
coarse-mesh 0.547 artifact).

Run (WSL pids-fem) -- LIVE to a file (NO tail), one soil per process:
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_partition_viz.py <sand|loam|clay>'
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from scratch.seq_cocycled_skin import make_graded_box
from scratch.seq_iterative_prototype import _top_area_ds

COMM = MPI.COMM_WORLD
SOILS = {
    "sand": dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=1.5),
    "loam": dict(theta_r=0.078, theta_s=0.43, alpha=3.6,  n=1.56, Ks=0.25),
    "clay": dict(theta_r=0.068, theta_s=0.38, alpha=0.8,  n=1.09, Ks=0.048),
}
GEO = dict(Lx=8.0, Ly=5.0, Lz=1.0, nx=30, ny=20, nz=16, p=2.5, S0=0.05, PSI_I=-0.4,
           RAIN=2.0, STORM=0.08, TEND=0.30)
SNAP_T = [0.02, 0.05, 0.08, 0.15, 0.30]   # snapshot times [day] (0.08 = storm end)
OUT = os.path.join(os.path.dirname(__file__), "_viz_{}.npz")


def _theta_deg8(prob, soil):
    dxq = ufl.dx(metadata={"quadrature_degree": 8})
    return prob.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(soil.theta_ufl(prob.psi) * dxq)), op=MPI.SUM)


def run(soil_name):
    g = GEO
    soil = VanGenuchten(**SOILS[soil_name])
    msh = make_graded_box(g["nx"], g["ny"], g["nz"], g["Lx"], g["Ly"], g["Lz"], p=g["p"])
    prob = CoupledProblem(msh, soil, n_man=0.05, overland_scheme="galerkin")
    prob.set_initial_condition(lambda x: g["PSI_I"] + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: g["S0"] * x[1])
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=g["S0"])
    top_area = _top_area_ds(msh, g["Lz"])
    R_in = g["RAIN"] * top_area * g["STORM"]

    # --- fixed geometry for the field views -------------------------------------------------------
    dcoord = prob.Vd.tabulate_dof_coordinates()
    td = np.where(np.isclose(dcoord[:, 2], g["Lz"]))[0]           # top dofs of the d-space
    top_x, top_y = dcoord[td, 0], dcoord[td, 1]
    vcoord = prob.Vpsi.tabulate_dof_coordinates()
    xc, yc = g["Lx"] / 2.0, g["Ly"] / 2.0                          # centre column (a structured node line)
    col = np.where((np.abs(vcoord[:, 0] - xc) < 1e-6) & (np.abs(vcoord[:, 1] - yc) < 1e-6))[0]
    col = col[np.argsort(vcoord[col, 2])]                          # sort up the column
    col_z = vcoord[col, 2]

    sw0 = prob.soil_water()
    ts = []                       # [t, cum_rain, cum_outflow, cum_drainage, soil_water, surface_water]
    snap_d, snap_theta, snap_done = [], [], []
    targets = list(SNAP_T)
    t, dt, nstep = 0.0, g["TEND"] / 200.0, 0
    t0 = time.perf_counter()
    while t < g["TEND"] - 1e-12 and nstep < 1500:
        h = min(dt, g["TEND"] - t)
        if targets and t + h > targets[0]:
            h = targets[0] - t
        rain_c.value = g["RAIN"] if t < g["STORM"] - 1e-12 else 0.0
        conv, it = prob.step(h)
        if not conv:
            dt *= 0.5
            if dt < 1e-9:
                break
            continue
        t += h; nstep += 1
        cum_rain = g["RAIN"] * top_area * min(t, g["STORM"])      # constant rain during the storm
        ts.append([t, cum_rain, prob.cum_outflow, prob.cum_drainage,
                   prob.soil_water(), prob.surface_water()])
        if targets and t >= targets[0] - 1e-12:
            snap_d.append(prob.d.x.array[td].copy())
            snap_theta.append(soil.theta(prob.psi.x.array[col]).copy())
            snap_done.append(t)
            targets.pop(0)
        dt = min(dt * 1.3, g["TEND"] / 60.0) if it <= 4 else (dt * 0.7 if it >= 9 else dt)

    ts = np.array(ts)
    out = OUT.format(soil_name)
    np.savez(out,
             ts=ts, snap_t=np.array(snap_done),
             top_x=top_x, top_y=top_y, snap_d=np.array(snap_d),
             col_z=col_z, snap_theta=np.array(snap_theta),
             Ks=SOILS[soil_name]["Ks"], theta_r=SOILS[soil_name]["theta_r"],
             theta_s=SOILS[soil_name]["theta_s"], top_area=top_area, R_in=R_in, sw0=sw0,
             Lx=g["Lx"], Ly=g["Ly"], Lz=g["Lz"], S0=g["S0"], rain=g["RAIN"], storm=g["STORM"],
             routed_R=prob.cum_outflow / R_in,
             infil_R=(prob.soil_water() - sw0 + prob.cum_drainage) / R_in)
    routed = prob.cum_outflow / R_in
    infil = (prob.soil_water() - sw0 + prob.cum_drainage) / R_in
    print(f"  {soil_name}: routed/R={routed:.3f} infil/R={infil:.3f} "
          f"[{nstep} steps {time.perf_counter()-t0:.0f}s snaps={len(snap_done)}] -> {os.path.basename(out)}",
          flush=True)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "loam"
    print(f"VIZ DATA -- {name} (resolved graded p={GEO['p']} nz={GEO['nz']}, "
          f"rain={GEO['RAIN']} storm={GEO['STORM']})", flush=True)
    run(name)
