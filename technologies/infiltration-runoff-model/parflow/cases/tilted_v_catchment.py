#!/usr/bin/env python3
"""ParFlow benchmark B6 -- canonical tilted-V catchment (overland runoff + convergent channel routing).

The Di Giammarco (1996) / Kollet & Maxwell (2006) tilted-V, the standard integrated-hydrology
intercomparison benchmark (IH-MIP2, Maxwell et al. 2014 / Kollet et al. 2017): two hillslope planes
draining toward a central channel that routes down-valley to an outlet. ParFlow's CORE design domain.

This deck serves TWO B6 roles (env-parameterized):
  * PF_SCALE=1.0 (default): the CANONICAL 1.62 km x 1.0 km case -> validate ParFlow against the known
    answer (equilibrium outlet Q_eq = rain*area ~= 4.86 m^3/s) + the published reference hydrograph.
  * PF_SCALE=0.1: a field-scale (162 x 100 m) geometrically-similar V -> the in-house<->ParFlow
    head-to-head, where the in-house FEM is inside its envelope (the canonical km scale is stiff/slow
    in-house -- a characterized envelope limit; see benchmarks/README.md S5e).

Canonical spec (PF_SCALE=1.0):
  - Two hillslope planes, each 800 m (cross-slope x) x 1000 m (down-valley y); channel at x = xc = LX/2.
  - Hillslope cross-slope Sx = 0.05 (toward the channel); channel/valley slope Sy = 0.02 (toward outlet).
  - Manning n = 0.015 (hillslope), 0.15 (channel)  [PF_TWOMAN=1; PF_TWOMAN=0 -> single 0.015].
  - Rain 3e-6 m/s = 0.2592 m/day for 90 min (0.0625 day), then 90 min recession (to 0.125 day).
  - Near-impermeable bed (surface-runoff dominated) so Q_eq = rain*area is the clean analytic anchor.
  - Outlet: overland routes off the downslope y=LY boundary (the channel carries it).

ParFlow setup: FLAT computational grid + TopoSlopes (the V is built from slopes, not geometry):
  TopoSlopesX = -Sx for x<xc, +Sx for x>xc (water -> channel); TopoSlopesY = -Sy (water -> y=LY outlet).
  OverlandKinematic (PF_WAVE=Kinematic, the IH-MIP2 standard) or OverlandDiffusive (PF_WAVE=Diffusive,
  to match the in-house DIFFUSION-wave for the field-scale head-to-head). Rain via a rain/recession
  time cycle (B3 idiom -- worked for standalone overland). Outlet hydrograph by STORAGE BALANCE
  (Q_out = rain*area - d(total storage)/dt; PrintOverlandSum is unreliable on the kinematic BC -- B5).

Run:
    docker run --rm -v $HOME/parflow-runs/tilted_v:/data parflow/parflow:latest tilted_v_catchment.py 1 1 1
    docker run --rm -e PF_SCALE=0.1 -e PF_WAVE=Diffusive -e PF_TWOMAN=0 \
        -v $HOME/parflow-runs/tilted_v:/data parflow/parflow:latest tilted_v_catchment.py 1 1 1
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

SCALE = float(os.environ.get("PF_SCALE", "1.0"))      # 1.0 canonical (1.62 km), 0.1 field-scale (162 m)
WAVE = os.environ.get("PF_WAVE", "Kinematic")          # Kinematic (IH-MIP2 std) | Diffusive (match in-house)
TWOMAN = os.environ.get("PF_TWOMAN", "1") == "1"       # two-Manning hillslope/channel vs single 0.015
TAG = os.environ.get("PF_TAG", f"tiltedv_s{SCALE:g}_{WAVE[:4].lower()}{'_2n' if TWOMAN else '_1n'}")

LX, LY = 1620.0 * SCALE, 1000.0 * SCALE
XC = LX / 2.0
SX, SY = 0.05, 0.02
N_HILL = float(os.environ.get("PF_NHILL", "0.015"))    # hillslope (single-Manning) roughness; 0.15 = rough variant
N_CHAN = float(os.environ.get("PF_NCHAN", "0.15"))
RAIN = 0.2592                                           # m/day (= 3e-6 m/s), scale-independent rate
STORM, T_END = 0.0625, 0.125                            # day (90 min storm, 90 min recession)
AREA = LX * LY
Q_EQ = RAIN * AREA                                      # equilibrium outlet discharge [m^3/day]

# grid: resolve the V; thin near-impermeable slab (surface-runoff). NX even so the channel is on a face.
NX = int(os.environ.get("PF_NX", "32"))
NY = int(os.environ.get("PF_NY", "20"))
NZ = 1
DX, DY, DZ = LX / NX, LY / NY, 0.5
CHAN_HALF = max(DX, 0.02 * LX)                          # channel band half-width for the rough-Manning geom
N_STORM_FRAMES, N_REC_FRAMES = 30, 30


def configure(r):
    r.FileVersion = 4
    r.Process.Topology.P = 1
    r.Process.Topology.Q = 1
    r.Process.Topology.R = 1

    r.ComputationalGrid.Lower.X = 0.0
    r.ComputationalGrid.Lower.Y = 0.0
    r.ComputationalGrid.Lower.Z = 0.0
    r.ComputationalGrid.DX = DX
    r.ComputationalGrid.DY = DY
    r.ComputationalGrid.DZ = DZ
    r.ComputationalGrid.NX = NX
    r.ComputationalGrid.NY = NY
    r.ComputationalGrid.NZ = NZ

    # geoms: full domain + left/right halves (for the V slopes) + a channel band (rough Manning)
    r.GeomInput.Names = "domain_input left_input right_input channel_input"
    for nm, gi in (("domain", "domain_input"), ("left", "left_input"),
                   ("right", "right_input"), ("channel", "channel_input")):
        r.GeomInput[gi].InputType = "Box"
        r.GeomInput[gi].GeomName = nm
    for nm in ("domain", "left", "right", "channel"):
        r.Geom[nm].Lower.X = 0.0 if nm in ("domain", "left") else (XC if nm == "right" else XC - CHAN_HALF)
        r.Geom[nm].Upper.X = LX if nm in ("domain", "right") else (XC if nm == "left" else XC + CHAN_HALF)
        r.Geom[nm].Lower.Y = 0.0
        r.Geom[nm].Upper.Y = LY
        r.Geom[nm].Lower.Z = 0.0
        r.Geom[nm].Upper.Z = NZ * DZ
    r.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

    r.Geom.Perm.Names = "domain"
    r.Geom.domain.Perm.Type = "Constant"
    r.Geom.domain.Perm.Value = 1.0e-6        # near-impermeable -> surface-runoff dominated
    r.Perm.TensorType = "TensorByGeom"
    r.Geom.Perm.TensorByGeom.Names = "domain"
    r.Geom.domain.Perm.TensorValX = 1.0
    r.Geom.domain.Perm.TensorValY = 1.0
    r.Geom.domain.Perm.TensorValZ = 1.0

    r.SpecificStorage.Type = "Constant"
    r.SpecificStorage.GeomNames = "domain"
    r.Geom.domain.SpecificStorage.Value = 1.0e-5

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
    r.Geom.domain.Porosity.Value = 0.05
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

    # rain (1 BaseUnit) then recession (rest)
    r.Cycle.Names = "constant rainrec"
    r.Cycle.constant.Names = "alltime"
    r.Cycle.constant.alltime.Length = 1
    r.Cycle.constant.Repeat = -1
    r.Cycle.rainrec.Names = "rain rec"
    r.Cycle.rainrec.rain.Length = 1
    r.Cycle.rainrec.rec.Length = 1
    r.Cycle.rainrec.Repeat = -1

    r.BCPressure.PatchNames = "x_lower x_upper y_lower y_upper z_lower z_upper"
    for p in ("x_lower", "x_upper", "y_lower", "y_upper", "z_lower"):
        r.Patch[p].BCPressure.Type = "FluxConst"
        r.Patch[p].BCPressure.Cycle = "constant"
        r.Patch[p].BCPressure.alltime.Value = 0.0
    # top surface: overland flow with the storm/recession hyetograph
    r.Patch.z_upper.BCPressure.Type = f"Overland{WAVE}"
    r.Patch.z_upper.BCPressure.Cycle = "rainrec"
    r.Patch.z_upper.BCPressure.rain.Value = -RAIN     # negative = into the domain
    r.Patch.z_upper.BCPressure.rec.Value = 0.0

    # V topography via slopes: water -> channel (x) -> outlet at y=LY
    r.TopoSlopesX.Type = "Constant"
    r.TopoSlopesX.GeomNames = "left right"
    r.TopoSlopesX.Geom.left.Value = -SX              # x<xc: routes toward +x (the channel)
    r.TopoSlopesX.Geom.right.Value = SX              # x>xc: routes toward -x (the channel)
    r.TopoSlopesY.Type = "Constant"
    r.TopoSlopesY.GeomNames = "domain"
    r.TopoSlopesY.Geom.domain.Value = -SY            # routes toward +y (the outlet at y=LY)

    r.Mannings.Type = "Constant"
    if TWOMAN:
        r.Mannings.GeomNames = "domain channel"
        r.Mannings.Geom.domain.Value = N_HILL / 86400.0
        r.Mannings.Geom.channel.Value = N_CHAN / 86400.0   # rough channel overrides in the band
    else:
        r.Mannings.GeomNames = "domain"
        r.Mannings.Geom.domain.Value = N_HILL / 86400.0

    r.PhaseSources.water.Type = "Constant"
    r.PhaseSources.water.GeomNames = "domain"
    r.PhaseSources.water.Geom.domain.Value = 0.0
    r.KnownSolution = "NoKnownSolution"

    # IC: water table just below the surface so the thin slab can't soak the storm -> runoff
    r.ICPressure.Type = "HydroStaticPatch"
    r.ICPressure.GeomNames = "domain"
    r.Geom.domain.ICPressure.Value = -0.1
    r.Geom.domain.ICPressure.RefGeom = "domain"
    r.Geom.domain.ICPressure.RefPatch = "z_upper"

    # robust overland solver block (= B3/B4)
    r.Solver = "Richards"
    r.Solver.MaxIter = 25000
    r.Solver.Nonlinear.MaxIter = 100
    r.Solver.Nonlinear.ResidualTol = 1e-8
    r.Solver.Nonlinear.EtaChoice = "EtaConstant"
    r.Solver.Nonlinear.EtaValue = 1e-3
    r.Solver.Nonlinear.UseJacobian = False
    r.Solver.Nonlinear.DerivativeEpsilon = 1e-14
    r.Solver.Nonlinear.StepTol = 1e-18
    r.Solver.Nonlinear.Globalization = "LineSearch"
    r.Solver.Linear.KrylovDimension = 30
    r.Solver.Linear.MaxRestart = 2
    r.Solver.Linear.Preconditioner = "PFMG"
    if WAVE == "Diffusive":
        r.Solver.OverlandDiffusive.Epsilon = 1e-5


def _set_timing(r, dt, dump, stop):
    r.TimingInfo.BaseUnit = STORM          # 1 rain BaseUnit = the storm duration
    r.TimingInfo.StartCount = 0
    r.TimingInfo.StartTime = 0.0
    r.TimingInfo.StopTime = stop
    r.TimingInfo.DumpInterval = dump
    r.TimeStep.Type = "Growth"
    r.TimeStep.InitialStep = dt
    r.TimeStep.GrowthFactor = 1.3
    r.TimeStep.MaxStep = dump
    r.TimeStep.MinStep = dt * 1e-4


def main():
    out_dir = os.path.join(RUN_ROOT, TAG)
    mkdir(out_dir)
    dump = T_END / (N_STORM_FRAMES + N_REC_FRAMES)
    dt0 = dump / 20.0
    r = Run(TAG, __file__)
    configure(r)
    _set_timing(r, dt0, dump, T_END)
    print(f"=== B6 tilted-V  scale={SCALE:g}  {LX:.0f}x{LY:.0f} m  grid {NX}x{NY}x{NZ}  "
          f"wave={WAVE}  twoMan={TWOMAN}  Q_eq={Q_EQ:.0f} m^3/day (~{Q_EQ/86400:.3f} m^3/s) ===", flush=True)
    if os.environ.get("PF_SKIPRUN", "") != "1":
        r.run(working_directory=out_dir)

    # --- outlet hydrograph by STORAGE BALANCE: Q_out = rain*area - d(total storage)/dt ---
    # ParFlow uses a FREE-OUTFLOW overland boundary: water leaves the downslope edge at ~0 ponding depth
    # (verified: d_outlet ~ 0 even at rough Manning), so the outlet flux is NOT recoverable from boundary
    # depth (and PrintOverlandSum is unreliable -- B5). The storage balance IS the reliable ParFlow outflow.
    # (The in-house add_outflow_bc instead imposes a Manning NORMAL-DEPTH outlet -> its boundary ponds and
    # outflow_rate() is direct: a documented outlet-BC formulation delta. Both reach Q_eq at equilibrium.)
    # d = offset-corrected surface ponding (top-cell CENTRE DZ/2 below the surface; README S5a).
    press = sorted(glob.glob(os.path.join(out_dir, f"{TAG}.out.press.*.pfb")),
                   key=lambda p: int(os.path.basename(p).split(".")[-2]))
    cell_area, cell_vol, theta_s = DX * DY, DX * DY * DZ, 0.05
    times, store_surf, store_sub, d_max, d_outlet_max = [], [], [], [], []
    for pf in press:
        i = int(os.path.basename(pf).split(".")[-2])
        P = np.asarray(read_pfb(pf))                                  # (NZ,NY,NX)
        S = np.asarray(read_pfb(os.path.join(out_dir, f"{TAG}.out.satur.{i:05d}.pfb")))
        d = np.maximum(P[-1] - 0.5 * DZ, 0.0)                         # surface ponding (offset-corrected)
        times.append(i * dump)
        store_surf.append(float(d.sum()) * cell_area)
        store_sub.append(float((theta_s * S).sum()) * cell_vol)
        d_max.append(float(d.max()))
        d_outlet_max.append(float(d[-1, :].max()))
    times = np.array(times); store_surf = np.array(store_surf); store_sub = np.array(store_sub)
    rain_t = np.where(times <= STORM + 1e-9, RAIN, 0.0) * AREA        # rain input rate [m^3/day]
    q_out = rain_t - np.gradient(store_surf + store_sub, times)       # storage-balance outflow
    cum_rain = RAIN * AREA * np.minimum(times, STORM)

    np.savez(os.path.join(SUMMARY_DIR, f"{TAG}.npz"), times=times, q_out=q_out, store_surf=store_surf,
             store_sub=store_sub, d_max=d_max, d_outlet_max=d_outlet_max, rain_rate=rain_t, cum_rain=cum_rain,
             Q_eq=Q_EQ, LX=LX, LY=LY, NX=NX, NY=NY, scale=SCALE, wave=WAVE, twoman=TWOMAN, storm=STORM,
             t_end=T_END, n_hill=N_HILL, n_chan=N_CHAN, sx=SX, sy=SY, rain=RAIN)

    i_se = int(np.argmin(np.abs(times - STORM)))
    qpk = float(np.max(q_out))
    print(f"  frames={times.size}  peak Q_out={qpk:.0f} m^3/day (~{qpk/86400:.3f} m^3/s)  "
          f"peak/Q_eq={qpk/Q_EQ:.3f}")
    print(f"  Q at storm-end (t={times[i_se]:.4f})={q_out[i_se]:.0f} ({q_out[i_se]/Q_EQ:.3f} Q_eq)  "
          f"Q final={q_out[-1]:.0f}  peak ponding d={1000*max(d_max):.1f} mm")
    print(f"  -> WROTE summaries/{TAG}.npz", flush=True)


SUMMARY_DIR = get_absolute_path("summaries")
mkdir(SUMMARY_DIR)
RUN_ROOT = get_absolute_path("runs")
mkdir(RUN_ROOT)
main()
print("DONE")
