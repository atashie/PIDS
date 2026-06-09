#!/usr/bin/env python3
"""ParFlow subsurface SWEEP -- the in-house scenario matrix as matched ParFlow columns.

Mirrors run_subsurface_sanity.py's four scenarios (antecedent wetness x event size) on the
1 m / 60-cell Carsel & Parrish (1988) loam column. Each applies CONSTANT rain for the storm
duration T (storm-only infiltration; the rainless recession is omitted -- a ParFlow storm/recession
time cycle proved unreliable here, applying rain the full run, and the infiltration phase is the
benchmark of interest). Output at 41 matched times over [0, T]. Exports per-scenario profiles for
the ../../benchmarks/ side-by-side.

This sweep covers the NON-PONDING (sub-Ks) infiltration regime across antecedent wetness and
storm intensity. Any scenario that SATURATES the surface is EXCLUDED -- whether q > Ks, or a WET
antecedent that fills the column against the no-flow bottom: a FluxConst top BC forces the full
rain in, so once the surface saturates (no outlet, no surface store) ParFlow drives the surface
pressure to absurd values (~1e4-1e5 m) instead of ponding. Genuine ponding needs ParFlow's surface
store (the OverlandFlow BC) -- the SAME surface-water handling deferred for the standalone-overland
case (see ../../benchmarks/README.md S5a). So ponding / saturation-excess scenarios are deferred
to the coupled comparison.

Run: docker run --rm -v $HOME/parflow-runs/column_sweep:/data parflow/parflow:latest column_sweep.py 1 1 1

Scenarios (psi0 = uniform antecedent head; q = sub-Ks storm rain; T_storm = storm length):
  mesic_q010 : psi0=-1.0 m, q=0.10 m/day, T_storm=0.50 day   (mesic, typical storm)
  mesic_q002 : psi0=-1.0 m, q=0.02 m/day, T_storm=0.50 day   (mesic, small storm)
  dry_q010   : psi0=-5.0 m, q=0.10 m/day, T_storm=0.50 day   (dry antecedent, typical storm)
  dry_q002   : psi0=-5.0 m, q=0.02 m/day, T_storm=0.50 day   (dry antecedent, small storm)
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

THETA_R, THETA_S = 0.078, 0.43
ALPHA, N, KS = 3.6, 1.56, 0.2496
SRES = THETA_R / THETA_S
NZ = 60
DEPTH = 1.0
DZ = DEPTH / NZ
NFRAMES = 41

SCENARIOS = {
    "mesic_q010": dict(psi0=-1.0, q=0.10, T_storm=0.50),   # mesic, typical sub-Ks storm
    "mesic_q002": dict(psi0=-1.0, q=0.02, T_storm=0.50),   # mesic, small storm
    "dry_q010":   dict(psi0=-5.0, q=0.10, T_storm=0.50),   # dry antecedent, typical storm
    "dry_q002":   dict(psi0=-5.0, q=0.02, T_storm=0.50),   # dry antecedent, small storm
}


def configure_static(r):
    """Grid, soil, phases, solver -- identical to cases/column_1d.py."""
    r.FileVersion = 4
    r.Process.Topology.P = 1
    r.Process.Topology.Q = 1
    r.Process.Topology.R = 1

    r.ComputationalGrid.Lower.X = 0.0
    r.ComputationalGrid.Lower.Y = 0.0
    r.ComputationalGrid.Lower.Z = 0.0
    r.ComputationalGrid.DX = 1.0
    r.ComputationalGrid.DY = 1.0
    r.ComputationalGrid.DZ = DZ
    r.ComputationalGrid.NX = 1
    r.ComputationalGrid.NY = 1
    r.ComputationalGrid.NZ = NZ

    r.GeomInput.Names = "domain_input"
    r.GeomInput.domain_input.InputType = "Box"
    r.GeomInput.domain_input.GeomName = "domain"
    r.Geom.domain.Lower.X = 0.0
    r.Geom.domain.Lower.Y = 0.0
    r.Geom.domain.Lower.Z = 0.0
    r.Geom.domain.Upper.X = 1.0
    r.Geom.domain.Upper.Y = 1.0
    r.Geom.domain.Upper.Z = DEPTH
    r.Geom.domain.Patches = "left right front back bottom top"

    r.Geom.Perm.Names = "domain"
    r.Geom.domain.Perm.Type = "Constant"
    r.Geom.domain.Perm.Value = KS
    r.Perm.TensorType = "TensorByGeom"
    r.Geom.Perm.TensorByGeom.Names = "domain"
    r.Geom.domain.Perm.TensorValX = 1.0
    r.Geom.domain.Perm.TensorValY = 1.0
    r.Geom.domain.Perm.TensorValZ = 1.0

    r.SpecificStorage.Type = "Constant"
    r.SpecificStorage.GeomNames = "domain"
    r.Geom.domain.SpecificStorage.Value = 1.0e-6

    r.Phase.Names = "water"
    r.Phase.water.Density.Type = "Constant"
    r.Phase.water.Density.Value = 1.0
    r.Phase.water.Viscosity.Type = "Constant"
    r.Phase.water.Viscosity.Value = 1.0
    r.Contaminants.Names = ""
    r.Geom.Retardation.GeomNames = ""
    r.Gravity = 1.0

    r.Geom.Porosity.GeomNames = "domain"
    r.Geom.domain.Porosity.Type = "Constant"
    r.Geom.domain.Porosity.Value = THETA_S
    r.Domain.GeomName = "domain"

    r.Phase.RelPerm.Type = "VanGenuchten"
    r.Phase.RelPerm.GeomNames = "domain"
    r.Geom.domain.RelPerm.Alpha = ALPHA
    r.Geom.domain.RelPerm.N = N
    r.Phase.Saturation.Type = "VanGenuchten"
    r.Phase.Saturation.GeomNames = "domain"
    r.Geom.domain.Saturation.Alpha = ALPHA
    r.Geom.domain.Saturation.N = N
    r.Geom.domain.Saturation.SRes = SRES
    r.Geom.domain.Saturation.SSat = 1.0

    r.Wells.Names = ""

    # single steady cycle: constant rain for the whole (storm-only) run
    r.Cycle.Names = "constant"
    r.Cycle.constant.Names = "alltime"
    r.Cycle.constant.alltime.Length = 1
    r.Cycle.constant.Repeat = -1

    r.BCPressure.PatchNames = "left right front back bottom top"
    for p in ("left", "right", "front", "back", "bottom"):
        r.Patch[p].BCPressure.Type = "FluxConst"
        r.Patch[p].BCPressure.Cycle = "constant"
        r.Patch[p].BCPressure.alltime.Value = 0.0
    # top: constant rain (value set per scenario; negative = into domain)
    r.Patch.top.BCPressure.Type = "FluxConst"
    r.Patch.top.BCPressure.Cycle = "constant"

    r.TopoSlopesX.Type = "Constant"
    r.TopoSlopesX.GeomNames = "domain"
    r.TopoSlopesX.Geom.domain.Value = 0.0
    r.TopoSlopesY.Type = "Constant"
    r.TopoSlopesY.GeomNames = "domain"
    r.TopoSlopesY.Geom.domain.Value = 0.0
    r.Mannings.Type = "Constant"
    r.Mannings.GeomNames = "domain"
    r.Mannings.Geom.domain.Value = 0.0

    r.PhaseSources.water.Type = "Constant"
    r.PhaseSources.water.GeomNames = "domain"
    r.PhaseSources.water.Geom.domain.Value = 0.0
    r.KnownSolution = "NoKnownSolution"

    r.ICPressure.Type = "Constant"
    r.ICPressure.GeomNames = "domain"
    r.Geom.domain.ICPressure.RefGeom = "domain"
    r.Geom.domain.ICPressure.RefPatch = "top"

    r.Solver = "Richards"
    r.Solver.MaxIter = 25000
    r.Solver.Nonlinear.MaxIter = 200
    r.Solver.Nonlinear.ResidualTol = 1e-9
    r.Solver.Nonlinear.EtaChoice = "Walker1"
    r.Solver.Nonlinear.EtaValue = 1e-5
    r.Solver.Nonlinear.UseJacobian = True
    r.Solver.Nonlinear.DerivativeEpsilon = 1e-10
    r.Solver.Linear.KrylovDimension = 10
    r.Solver.Linear.Preconditioner = "PFMG"


def step_index(path):
    return int(os.path.basename(path).split(".")[-2])


def run_scenario(key, cfg):
    psi0, q, T_storm = cfg["psi0"], cfg["q"], cfg["T_storm"]
    dt = T_storm / 200.0
    dump = T_storm / 40.0    # -> 41 frames over [0, T_storm]; dump/dt = 5 (integer)

    r = Run(key, __file__)
    configure_static(r)
    r.Geom.domain.ICPressure.Value = psi0
    r.Patch.top.BCPressure.alltime.Value = -q
    r.TimingInfo.BaseUnit = dt
    r.TimingInfo.StartCount = 0
    r.TimingInfo.StartTime = 0.0
    r.TimingInfo.StopTime = T_storm
    r.TimingInfo.DumpInterval = dump
    r.TimeStep.Type = "Constant"
    r.TimeStep.Value = dt

    out_dir = get_absolute_path(key)
    mkdir(out_dir)
    r.run(working_directory=out_dir)

    press_files = sorted(glob.glob(os.path.join(out_dir, f"{key}.out.press.*.pfb")))
    times, head_tz, theta_tz = [], [], []
    for pf in press_files:
        i = step_index(pf)
        P = np.asarray(read_pfb(pf)).reshape(-1)
        S = np.asarray(read_pfb(os.path.join(out_dir, f"{key}.out.satur.{i:05d}.pfb"))).reshape(-1)
        times.append(i * dump)
        head_tz.append(P.copy())
        theta_tz.append(THETA_S * S)

    zc = (np.arange(NZ) + 0.5) * DZ
    summary = os.path.join(SUMMARY_DIR, f"column_sweep_{key}.npz")
    np.savez(summary, z=zc, times=np.array(times, dtype=float),
             head=np.array(head_tz, dtype=float), theta=np.array(theta_tz, dtype=float),
             porosity=THETA_S, psi0=psi0, q=q, T_storm=T_storm)
    print(f"  {key}: {len(times)} frames, top psi range [{np.min(head_tz):.3f},{np.max(head_tz):.3f}] m"
          f" -> WROTE {os.path.basename(summary)}")


SUMMARY_DIR = get_absolute_path("summaries")
mkdir(SUMMARY_DIR)
print("=== ParFlow subsurface sweep ===")
for _key, _cfg in SCENARIOS.items():
    run_scenario(_key, _cfg)
print("DONE")
