"""Task 3 / option-a SPLIT EXPERIMENT (Arik 2026-06-27): does a thin SURFACE SKIN make the COARSE-mesh
runoff/infiltration partition match the mesh-converged value -- and does ONE thin skin suffice, or do you
need the graded (coarsening-with-depth) transition?

On the b1 hillslope (8x5x1, 30x20 lateral, S=0.03), ONE fixed storm (rain=0.5 m/day x 0.08 d -- the
documented short storm that is mesh-sensitive on loam), for each soil (loam/sand/clay), FOUR vertical
meshes (lateral 30x20 FIXED in all):
  * coarse : UNIFORM nz=8 (125 mm top cell)            -- the baseline (the wrong, mesh-dependent answer)
  * graded : graded nz=8, warped so top ~2 mm, deep cells coarsen (make_graded_box p=2.5)  [variant 1]
  * skin   : UNIFORM nz=8 (125 mm) with ONLY the top cell SPLIT into a ~2 mm skin + ~123 mm remainder;
             everything below stays uniform 125 mm                                          [variant 2]
  * ref    : graded p=3.0 (top ~0.6 mm) = the mesh-CONVERGED truth (Task 1: p2.75=p3.0 flat)

GATE: do BOTH `graded` and `skin` match `ref` (and differ from `coarse`)? If `skin` matches `ref`, a SINGLE
thin top layer on an otherwise-standard coarse mesh fixes the partition (the cheapest production fix; no
re-grading the column). routed/R + infil/R reported; clean solves expected.

Run (WSL pids-fem) -- one (soil, variant) per process:
  python -u scratch/seq_skin_split.py <loam|sand|clay> <coarse|graded|skin|ref>
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from scratch.seq_cocycled_skin import make_graded_box
from scratch.seq_href_closure_study import _march
from scratch.seq_iterative_prototype import _top_area_ds, _soil_water_deg8

COMM = MPI.COMM_WORLD
SOILS = {
    "loam": dict(theta_r=0.078, theta_s=0.43, alpha=3.6,  n=1.56, Ks=0.25),
    "sand": dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=1.5),
    "clay": dict(theta_r=0.068, theta_s=0.38, alpha=0.8,  n=1.09, Ks=0.048),
}
B1 = dict(Lx=8.0, Ly=5.0, Lz=1.0, S0=0.03, PSI_I=-0.4, RAIN=0.5, STORM=0.08, TEND=0.45,
          NMAN=0.05, nx=30, ny=20, nz=8)
SKIN = 2e-3   # thin-skin thickness [m] (~2 mm; matches the graded p=2.5 top cell)


def make_uniform_plus_skin(nx, ny, nz, Lx, Ly, Lz, skin):
    """UNIFORM nz vertical cells, but the TOP cell SPLIT into a `skin` + remainder (variant 2). Build a
    uniform box with nz+1 vertical cells, then remap its z-levels to the target levels
    [0, Lz/nz, ..., (nz-1)Lz/nz, Lz-skin, Lz] (uniform below, one thin skin on top). Monotonic -> no
    tet inversion."""
    uni = np.linspace(0.0, Lz, nz + 1)
    targets = np.unique(np.concatenate([uni, [Lz - skin]]))   # nz+2 sorted levels -> nz+1 cells
    ncz = targets.size - 1
    msh = dmesh.create_box(COMM, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
                           [nx, ny, ncz], cell_type=dmesh.CellType.tetrahedron)
    x = msh.geometry.x
    idx = np.rint(x[:, 2] / Lz * ncz).astype(int)             # which uniform level each node sits on
    idx = np.clip(idx, 0, ncz)
    x[:, 2] = targets[idx]                                     # remap to the target (uniform+skin) levels
    return msh, float(targets[-1] - targets[-2]) * 1000.0      # mesh, top-cell thickness [mm]


def make_skin_package(nx, ny, Lx, Ly, Lz, skins, n_uniform):
    """A general SURFACE SKIN PACKAGE on a UNIFORM subsurface (Arik 2026-06-27): `skins` = thin top-cell
    thicknesses listed TOP-DOWN (e.g. [10e-3] = one 10 mm skin; [2e-3, 8e-3] = a 2 mm cell over an 8 mm
    cell), then `n_uniform` evenly-spaced cells fill the rest of the column below. Modular -- the same skin
    package can sit on ANY subsurface mesh (here uniform; the design intent is mesh-agnostic). Monotonic
    z-remap of a uniform box -> no tet inversion."""
    skins = list(skins)
    z_sb = Lz - sum(skins)                                   # skin-package bottom
    uni = np.linspace(0.0, z_sb, n_uniform + 1)              # n_uniform uniform cells below
    sk = z_sb + np.cumsum(list(reversed(skins)))             # skin levels above z_sb (bottom skin first)
    targets = np.concatenate([uni, sk])                      # ascending, unique
    ncz = targets.size - 1
    msh = dmesh.create_box(COMM, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
                           [nx, ny, ncz], cell_type=dmesh.CellType.tetrahedron)
    x = msh.geometry.x
    idx = np.clip(np.rint(x[:, 2] / Lz * ncz).astype(int), 0, ncz)
    x[:, 2] = targets[idx]
    return msh, float(targets[-1] - targets[-2]) * 1000.0    # top-cell thickness [mm]


def build_mesh(variant):
    g = B1
    if variant.startswith("skins:"):   # ARBITRARY skin package, e.g. "skins:10" or "skins:2,8" (mm, top-down)
        mm = [float(s) for s in variant.split(":", 1)[1].split(",")]
        return make_skin_package(g["nx"], g["ny"], g["Lx"], g["Ly"], g["Lz"],
                                 [v * 1e-3 for v in mm], g["nz"])
    if variant == "skin10":      # Exp A: one 10 mm skin + uniform below
        return make_skin_package(g["nx"], g["ny"], g["Lx"], g["Ly"], g["Lz"], [10e-3], g["nz"])
    if variant == "skin2x":      # Exp B: two skins (2 mm, 8 mm) + uniform below
        return make_skin_package(g["nx"], g["ny"], g["Lx"], g["Ly"], g["Lz"], [2e-3, 8e-3], g["nz"])
    if variant == "coarse":
        msh = dmesh.create_box(COMM, [np.array([0.0, 0.0, 0.0]), np.array([g["Lx"], g["Ly"], g["Lz"]])],
                               [g["nx"], g["ny"], g["nz"]], cell_type=dmesh.CellType.tetrahedron)
        return msh, g["Lz"] / g["nz"] * 1000.0
    if variant == "graded":
        msh = make_graded_box(g["nx"], g["ny"], g["nz"], g["Lx"], g["Ly"], g["Lz"], p=2.5)
        return msh, g["Lz"] * (1.0 / g["nz"]) ** 2.5 * 1000.0
    if variant == "ref":
        msh = make_graded_box(g["nx"], g["ny"], g["nz"], g["Lx"], g["Ly"], g["Lz"], p=3.0)
        return msh, g["Lz"] * (1.0 / g["nz"]) ** 3.0 * 1000.0
    if variant == "skin":
        return make_uniform_plus_skin(g["nx"], g["ny"], g["nz"], g["Lx"], g["Ly"], g["Lz"], SKIN)
    raise ValueError(variant)


def run(soil_name, variant, dt_max=0.02):
    g = B1
    soil = VanGenuchten(**SOILS[soil_name])
    msh, topmm = build_mesh(variant)
    mono = CoupledProblem(msh, soil, n_man=g["NMAN"], overland_scheme="galerkin")  # ell_c AUTO = top dz/2
    mono.set_initial_condition(lambda x: g["PSI_I"] + 0.0 * x[0], d_value=0.0)
    mono.set_topography(lambda x: g["S0"] * x[1])
    rain_c = mono.add_rain(0.0)
    mono.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=g["S0"])
    th0 = _soil_water_deg8(mono, soil)
    top_area = _top_area_ds(msh, g["Lz"])
    rain = float(os.environ.get("SKIN_RAIN", g["RAIN"]))   # storm-intensity override
    R_in = rain * top_area * g["STORM"]
    t0 = time.perf_counter()
    ns, coll, tend, peak = _march(mono, rain_c, storm_dur=g["STORM"], storm_rain=rain,
                                  t_end=g["TEND"], dt0=dt_max / 4.0, dt_max=dt_max, ctrl_low=3,
                                  ctrl_high=8, track_sheet=True, top_area=top_area, max_steps=1200)
    ok = (not coll) and tend >= g["TEND"] - 1e-9
    routed = mono.cum_outflow / R_in
    infil = (_soil_water_deg8(mono, soil) - th0 + mono.cum_drainage) / R_in
    print("#" * 96)
    print(f"SKIN-SPLIT -- {soil_name} (Ks={SOILS[soil_name]['Ks']}) / {variant.upper()} "
          f"(top cell {topmm:.2f}mm, ell_c={mono.ell_c*1000:.2f}mm)")
    print(f"  routed/R = {routed:.4f}   infil/R = {infil:.4f}   peak_mean_sheet = {peak*1000:.3f}mm")
    print(f"  ok={ok} ns={ns} clip={abs(getattr(mono,'clip_mass_adjust',0.0)):.1e} "
          f"wall={time.perf_counter()-t0:.0f}s")
    print("#" * 96, flush=True)


if __name__ == "__main__":
    soil_name = sys.argv[1] if len(sys.argv) > 1 else "loam"
    variant = sys.argv[2] if len(sys.argv) > 2 else "skin"
    run(soil_name, variant)
