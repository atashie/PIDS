#!/usr/bin/env python3
"""ParFlow benchmark case B3 -- 1-D hillslope overland storm (diffusive wave).

*** DEFERRED 2026-06-09: a clean STANDALONE overland comparison is not achievable in ParFlow
at this regime (ParFlow is a coupled Richards+overland code; it evacuates water at ~0 depth
rather than forming the Manning-equilibrium sheet at the realistic small Manning). See
../../benchmarks/README.md (§5a) for the finding. This deck is retained as CORRECT reference
(Manning bridge n_PF=n_SI/86400, saturated-IC runoff setup, offset-corrected depth extraction)
for the coupled comparison (B4) or a rougher-Manning variant. ***

Off-the-shelf cross-check for the in-house OVERLAND module
(``pids_forward.physics.overland``, diffusion-wave). Matches the in-house
``typical_storm`` hillslope: a 100 m, 2% slope, Manning n=0.05, hit by a
0.30 m/day storm for 0.06 day then a recession to 0.18 day, dry antecedent,
impermeable bed (here: a thin very-low-K slab so essentially all rain runs off).

Uses ParFlow's **OverlandDiffusive** BC (the diffusive-wave option) to match the
in-house diffusion-wave formulation as closely as possible.

Run (inside the ParFlow Docker container, from this file's directory):
    docker run --rm -v $HOME/parflow-runs/overland_hillslope:/data parflow/parflow:latest overland_hillslope.py 1 1 1

Mapping to the in-house model (overland.py / run_overland_sanity.py)
-------------------------------------------------------------------
  in-house                              ParFlow
  ------------------------------------  --------------------------------------
  1-D interval [0,L], 100 cells         NX=100, NY=1, NZ=1 (thin slab)
  bed z_b = S0*(L-x), slope S0          TopoSlopesX = -S0 (verify outlet end), TopoSlopesY=0
  Manning n (SI, seconds)               Mannings = n_SI/86400 (ParFlow time = days)
  impermeable bed (no infiltration)     Perm=1e-7, low porosity -> runoff-dominated
  rain hyetograph (m/day)               OverlandDiffusive rain cycle, value=-q (neg=into domain)
  free-drainage Manning outlet at x=L   overland BC routes water off the downslope edge
  surface depth d(x)                    ponded depth = max(top-cell pressure, 0)
  outlet discharge (m2/day per width)   derived from storage balance: rain*L - dW/dt

NOTE (deltas to track): ParFlow ponded depth is read from the top-cell pressure
(definitional, vs the in-house depth variable); the outlet discharge is a storage-balance
estimate (both models), not ParFlow's internal boundary flux.
"""
import glob
import os

import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path

try:
    from parflow.tools.io import read_pfb
except ImportError:
    from parflow import read_pfb

# --- matched case parameters (identical to in-house typical_storm) -----------
RUN_NAME = "overland_hillslope"
L = 100.0
NX = 100
DX = L / NX
S0 = 0.02
N_SI = 0.05
N_PF = N_SI / 86400.0     # SI Manning (s) -> ParFlow day-units
Q_RAIN = 0.30             # storm intensity (m/day)
STORM = 0.06              # storm duration (day)
T_STOP = 0.18             # total (day)
DT = 0.003                # day
DUMP = 0.003              # day -> 61 frames (00000..00060), matched to in-house times

r = Run(RUN_NAME, __file__)
r.FileVersion = 4
r.Process.Topology.P = 1
r.Process.Topology.Q = 1
r.Process.Topology.R = 1

r.ComputationalGrid.Lower.X = 0.0
r.ComputationalGrid.Lower.Y = 0.0
r.ComputationalGrid.Lower.Z = 0.0
r.ComputationalGrid.DX = DX
r.ComputationalGrid.DY = 1.0
r.ComputationalGrid.DZ = 0.5
r.ComputationalGrid.NX = NX
r.ComputationalGrid.NY = 1
r.ComputationalGrid.NZ = 1

r.GeomInput.Names = "domaininput"
r.GeomInput.domaininput.GeomName = "domain"
r.GeomInput.domaininput.InputType = "Box"
r.Geom.domain.Lower.X = 0.0
r.Geom.domain.Lower.Y = 0.0
r.Geom.domain.Lower.Z = 0.0
r.Geom.domain.Upper.X = L
r.Geom.domain.Upper.Y = 1.0
r.Geom.domain.Upper.Z = 0.5
r.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

r.Geom.Perm.Names = "domain"
r.Geom.domain.Perm.Type = "Constant"
r.Geom.domain.Perm.Value = 1.0e-7        # ~impermeable -> runoff-dominated
r.Perm.TensorType = "TensorByGeom"
r.Geom.Perm.TensorByGeom.Names = "domain"
r.Geom.domain.Perm.TensorValX = 1.0
r.Geom.domain.Perm.TensorValY = 1.0
r.Geom.domain.Perm.TensorValZ = 1.0

r.SpecificStorage.Type = "Constant"
r.SpecificStorage.GeomNames = "domain"
r.Geom.domain.SpecificStorage.Value = 1.0e-4

r.Phase.Names = "water"
r.Phase.water.Density.Type = "Constant"
r.Phase.water.Density.Value = 1.0
r.Phase.water.Viscosity.Type = "Constant"
r.Phase.water.Viscosity.Value = 1.0
r.Contaminants.Names = ""
r.Geom.Retardation.GeomNames = ""
r.Gravity = 1.0

r.TimingInfo.BaseUnit = 0.06
r.TimingInfo.StartCount = 0
r.TimingInfo.StartTime = 0.0
r.TimingInfo.StopTime = T_STOP
r.TimingInfo.DumpInterval = DUMP
r.TimeStep.Type = "Constant"
r.TimeStep.Value = DT

r.Geom.Porosity.GeomNames = "domain"
r.Geom.domain.Porosity.Type = "Constant"
r.Geom.domain.Porosity.Value = 0.01
r.Domain.GeomName = "domain"

r.Phase.RelPerm.Type = "VanGenuchten"
r.Phase.RelPerm.GeomNames = "domain"
r.Geom.domain.RelPerm.Alpha = 6.0
r.Geom.domain.RelPerm.N = 2.0
r.Phase.Saturation.Type = "VanGenuchten"
r.Phase.Saturation.GeomNames = "domain"
r.Geom.domain.Saturation.Alpha = 6.0
r.Geom.domain.Saturation.N = 2.0
r.Geom.domain.Saturation.SRes = 0.2
r.Geom.domain.Saturation.SSat = 1.0

r.Wells.Names = ""

# time cycles: rain for STORM (one BaseUnit), then recession for the rest
r.Cycle.Names = "constant rainrec"
r.Cycle.constant.Names = "alltime"
r.Cycle.constant.alltime.Length = 1
r.Cycle.constant.Repeat = -1
r.Cycle.rainrec.Names = "rain rec"
r.Cycle.rainrec.rain.Length = 1   # 1 BaseUnit = 0.06 day
r.Cycle.rainrec.rec.Length = 2    # 2 BaseUnits = 0.12 day
r.Cycle.rainrec.Repeat = -1

r.BCPressure.PatchNames = r.Geom.domain.Patches
for p in ("x_lower", "x_upper", "y_lower", "y_upper", "z_lower"):
    r.Patch[p].BCPressure.Type = "FluxConst"
    r.Patch[p].BCPressure.Cycle = "constant"
    r.Patch[p].BCPressure.alltime.Value = 0.0
# overland flow on the top surface, with the storm/recession hyetograph
r.Patch.z_upper.BCPressure.Type = "OverlandDiffusive"
r.Patch.z_upper.BCPressure.Cycle = "rainrec"
r.Patch.z_upper.BCPressure.rain.Value = -Q_RAIN   # negative = into the domain
r.Patch.z_upper.BCPressure.rec.Value = 0.0

r.TopoSlopesX.Type = "Constant"
r.TopoSlopesX.GeomNames = "domain"
r.TopoSlopesX.Geom.domain.Value = -S0   # routes downslope toward +x; verified from depth profile below
r.TopoSlopesY.Type = "Constant"
r.TopoSlopesY.GeomNames = "domain"
r.TopoSlopesY.Geom.domain.Value = 0.0

r.Mannings.Type = "Constant"
r.Mannings.GeomNames = "domain"
r.Mannings.Geom.domain.Value = N_PF

r.PhaseSources.water.Type = "Constant"
r.PhaseSources.water.GeomNames = "domain"
r.PhaseSources.water.Geom.domain.Value = 0.0
r.KnownSolution = "NoKnownSolution"

# IC: water table AT the land surface (pressure 0 at z_upper) so the thin subsurface is
# saturated and CANNOT absorb the storm -> rain runs off as overland, matching the in-house
# impermeable-bed hillslope. (A dry subsurface would soak up the storm and under-pond.)
r.ICPressure.Type = "HydroStaticPatch"
r.ICPressure.GeomNames = "domain"
r.Geom.domain.ICPressure.Value = 0.0
r.Geom.domain.ICPressure.RefGeom = "domain"
r.Geom.domain.ICPressure.RefPatch = "z_upper"

# solver block (from overland_slopingslab_DWE.py -- proven for diffusive overland)
r.Solver = "Richards"
r.Solver.MaxIter = 2500
r.Solver.Nonlinear.MaxIter = 50
r.Solver.Nonlinear.ResidualTol = 1e-9
r.Solver.Nonlinear.EtaChoice = "EtaConstant"
r.Solver.Nonlinear.EtaValue = 0.01
r.Solver.Nonlinear.UseJacobian = False
r.Solver.Nonlinear.DerivativeEpsilon = 1e-15
r.Solver.Nonlinear.StepTol = 1e-20
r.Solver.Nonlinear.Globalization = "LineSearch"
r.Solver.Linear.KrylovDimension = 20
r.Solver.Linear.MaxRestart = 2
r.Solver.Linear.Preconditioner = "PFMG"
r.Solver.Drop = 1e-20
r.Solver.AbsTol = 1e-10
r.Solver.OverlandDiffusive.Epsilon = 1e-5

# --- run ---------------------------------------------------------------------
out_dir = get_absolute_path("output")
mkdir(out_dir)
r.run(working_directory=out_dir)

# --- readout: ponded depth profile + storage-balance outlet hydrograph -------
def step_index(path):
    return int(os.path.basename(path).split(".")[-2])


DZ_CELL = 0.5  # top-cell thickness (= ComputationalGrid.DZ). The cell centre sits DZ/2 below the
#                land surface, so for a SATURATED top cell the surface ponding depth is
#                (cell pressure - DZ/2). Subtracting it removes the hydrostatic offset that would
#                otherwise read out as a spurious ~DZ/2 "depth" (the 250 mm artifact).
press_files = sorted(glob.glob(os.path.join(out_dir, f"{RUN_NAME}.out.press.*.pfb")))
assert press_files, "no pressure output -- the ParFlow run failed"
xc = (np.arange(NX) + 0.5) * DX            # cell-centre x (m); the outlet cell is xc[-1]

times, depth_tx = [], []
for pf in press_files:
    i = step_index(pf)
    p = np.asarray(read_pfb(pf)).reshape(-1)        # top-cell pressure along x
    d = np.maximum(p - 0.5 * DZ_CELL, 0.0)          # surface ponding depth (hydrostatic offset removed)
    times.append(i * DUMP)
    depth_tx.append(d)

times = np.array(times)
depth_tx = np.array(depth_tx)

# export for the ../../benchmarks/ side-by-side; the outlet hydrograph is computed THERE
# (Manning normal-depth at the actual outlet cell), matching the in-house outlet law.
summary_path = os.path.join(out_dir, "overland_hillslope_profiles.npz")
np.savez(summary_path, x=xc, times=times, depth=depth_tx, L=L, slope=S0, n_man=N_SI, dz=DZ_CELL)

d_eq = (Q_RAIN * L * N_SI / 86400.0 / np.sqrt(S0)) ** 0.6   # analytic Manning-equilibrium outlet depth
imax = int(np.argmax(depth_tx.max(axis=0)))
print("=== B3 overland_hillslope  physical sanity ===")
print(f"grid {NX}x1x1 over {L} m  S0={S0}  n_SI={N_SI} (n_PF={N_PF:.3e})  q={Q_RAIN} m/d storm={STORM} d")
print(f"peak surface ponding (offset-corrected) = {1000*depth_tx.max():.3f} mm  at x={xc[imax]:.1f} m")
print(f"analytic Manning-equilibrium outlet depth = {1000*d_eq:.3f} mm  (peak should approach this)")
print(f"outlet-cell (x={xc[-1]:.1f} m) peak depth = {1000*depth_tx[:, -1].max():.3f} mm")
print(f"WROTE {summary_path}")
