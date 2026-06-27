#!/usr/bin/env python3
"""ParFlow benchmark case B5 -- 3-D coupled tilted-hillslope (overland routing + lateral GW seepage).

Off-the-shelf cross-check for the in-house COUPLED solver IN 3-D
(``pids_forward.physics.coupling.CoupledProblem`` on a hillslope), matching the in-house 3-D
coupling sanity case (``forward-model/viz/run_coupling_3d_sanity.py``). Extends B4 (flat 1-D
ponding column, no lateral flux) to ParFlow's CORE design domain: a tilted 3-D box with
**lateral overland routing off the downslope edge** + a **lateral groundwater seepage outlet**
on the downslope side face.

The in-house case (`run_coupling_3d_sanity.py`)
-----------------------------------------------
  5 x 1 x 1 m box, mesh 16 x 6 x 8, bed tilted z_b = S0*(L-x), S0 = 0.05 (downslope to x=L).
  Antecedent: hydrostatic water table at z = 0.35 (psi = 0.35 - z). Manning n = 0.05.
  Storm 0.5 m/day for 0.30 day, then recession to 0.50 day. Two soils (Carsel & Parrish):
    SAND (Ks=7.13 >> rain): rain infiltrates, water table rises, lateral GW outflow dominates.
    LOAM (Ks=0.25 <  rain): infiltration-excess -> overland runoff dominates.
  TWO outlets, both at x=L: (i) a surface Manning overland edge outlet (codim-2 ridge line),
  (ii) a lateral groundwater GHB on the x=L side FACE: q_n = C*kr(psi)*(psi + z - H_ext),
       kr = K(psi)/Ks, C = 0.5 /day, H_ext = 0.20 m (kr-weight confines seepage to the
       saturated zone -> only groundwater leaves).

ParFlow mapping (the load-bearing conversions; deltas in ../README.md S4)
------------------------------------------------------------------------
  in-house                                  ParFlow
  ----------------------------------------  ----------------------------------------------------
  5x1x1 box, mesh 16x6x8, z up              NX=16 NY=6 NZ=8, DX=5/16 DY=1/6 DZ=1/8 (k=0 = base)
  bed tilt z_b=S0*(L-x), downslope to x=L   TopoSlopesX = -S0 (neg routes toward +x, B3-verified),
                                              TopoSlopesY = 0; FLAT computational grid + slope
                                              vector (NOT a geometrically tilted mesh -> delta)
  Manning n = 0.05 (SI, seconds)            Mannings = n_SI/86400 (ParFlow time = days)
  surface ponding + overland edge outlet    Patch.z_upper OverlandFlow (store = max(psi_top,0));
                                              water leaves the downslope x=L edge via the slope
  lateral GW GHB  C*kr*(psi+z-H_ext)        Patch.x_upper DirEquilRefPatch -> Dirichlet head
                                              H = H_ext = 0.20 on the x=L face (water table z=0.20).
                                              ** This is C -> inf (constant-head): it captures the
                                              external head but NOT the in-house finite kr*C
                                              conductance -> it OVER-DRAINS the saturated toe
                                              relative to the in-house GHB. A documented delta,
                                              biggest for SAND (lateral-GW-dominated). **
  antecedent water table z = 0.35           ICPressure HydroStaticPatch, RefPatch z_lower, Value 0.35
  base + cross-slope sides: no-flow         Patch.{z_lower,y_lower,y_upper,x_lower} FluxConst = 0
  rain hyetograph (storm then recession)    two-phase: storm run -> PFBFile-IC restart, rain=0 (B4)

Run (inside the ParFlow Docker container, from a copy of this file in the mounted dir):
    docker run --rm -v $HOME/parflow-runs/coupled_hillslope_3d:/data \
        parflow/parflow:latest coupled_hillslope_3d.py 1 1 1

  Restrict to one scenario (the step-1 make-or-break gate is loam_overland):
    docker run --rm -e PF_SCENARIOS=loam_overland \
        -v $HOME/parflow-runs/coupled_hillslope_3d:/data \
        parflow/parflow:latest coupled_hillslope_3d.py 1 1 1

Per-scenario export to summaries/coupled_3d_<key>.npz: cell-centre x,y,z; times; the full
pressure(t,z,y,x) and theta(t,z,y,x) fields; grid + slope + Manning + H_ext + soil/storm
metadata. The side-by-side builder (../../benchmarks/build_comparison_coupled_3d.py) derives
d(x,y), the 4-way partition, and the hydrographs from these (thin profile dump, as for B2/B4).

Documented formulation deltas (../README.md S4), amplified here:
  * ParFlow vG omits the in-house Vogel/Ippisch air-entry cap -- matters MORE here (the surface
    SATURATES under the loam storm; the cap shifts theta/K/near-sat tangent and front speed).
  * ParFlow FLAT grid + TopoSlopes vector vs the in-house GEOMETRICALLY TILTED mesh (5% slope;
    small geometric difference, but the lateral GW face is vertical here, tilted in-house).
  * Constant-head (DirEquil) lateral GW vs the in-house finite kr-weighted GHB (over-drain delta).
  * ParFlow needs SpecificStorage (1e-6); the in-house model ignores Ss (unconfined).
  * ParFlow cell-centred FV (16x6x8 centres) vs in-house P1 FEM (17x7x9 nodes) -> compare by interp.
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

# --- matched domain + grid (= run_coupling_3d_sanity.py) ---------------------
L, W, H = 5.0, 1.0, 1.0
NX, NY = 16, 6
NZ = int(os.environ.get("PF_NZ", "8"))       # env-configurable for the vertical mesh-convergence study
DX, DY, DZ = L / NX, W / NY, H / NZ          # 0.3125, 0.16667, 0.125 (DZ=H/NZ shrinks as NZ grows)
S0 = 0.05                                     # bed slope toward the x=L downslope outlet
Z_WT = 0.35                                   # antecedent hydrostatic water table elevation (m)
H_EXT = 0.20                                  # downslope external head the lateral GHB drains toward (m)
N_SI = 0.05
N_PF = N_SI / 86400.0                          # SI Manning (s) -> ParFlow day-units (~5.79e-7)

# --- soils (Carsel & Parrish 1988, SI m/day) = run_coupling_3d_sanity.py -----
SAND = dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)
LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)

# Same persistent storm for both soils; only the texture differs (the contrast Arik asked for).
STORM = dict(rate=0.50, storm_dur=0.30, t_end=0.50)
SCENARIOS = {
    "loam_overland":   dict(soil=LOAM, **STORM),   # infiltration-excess -> overland (step-1 gate)
    "sand_lateral_gw": dict(soil=SAND, **STORM),   # infiltrate + rising WT -> lateral GW seepage
}

N_STORM_FRAMES = 20    # output frames over the storm phase
N_REC_FRAMES = 20      # output frames over the recession phase
SUBSTEPS = 5           # internal Growth substeps per dump (dump = SUBSTEPS * initial dt)


def configure_static(r, soil):
    """Grid, soil, phases, slope/Manning, BCs (OverlandFlow top + lateral-GW head face), solver."""
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

    r.GeomInput.Names = "domain_input"
    r.GeomInput.domain_input.InputType = "Box"
    r.GeomInput.domain_input.GeomName = "domain"
    r.Geom.domain.Lower.X = 0.0
    r.Geom.domain.Lower.Y = 0.0
    r.Geom.domain.Lower.Z = 0.0
    r.Geom.domain.Upper.X = L
    r.Geom.domain.Upper.Y = W
    r.Geom.domain.Upper.Z = H
    r.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

    r.Geom.Perm.Names = "domain"
    r.Geom.domain.Perm.Type = "Constant"
    r.Geom.domain.Perm.Value = soil["Ks"]      # rho=mu=g=1 -> K == Perm
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
    r.Geom.domain.Porosity.Value = soil["theta_s"]
    r.Domain.GeomName = "domain"

    r.Phase.RelPerm.Type = "VanGenuchten"
    r.Phase.RelPerm.GeomNames = "domain"
    r.Geom.domain.RelPerm.Alpha = soil["alpha"]
    r.Geom.domain.RelPerm.N = soil["n"]
    r.Phase.Saturation.Type = "VanGenuchten"
    r.Phase.Saturation.GeomNames = "domain"
    r.Geom.domain.Saturation.Alpha = soil["alpha"]
    r.Geom.domain.Saturation.N = soil["n"]
    r.Geom.domain.Saturation.SRes = soil["theta_r"] / soil["theta_s"]
    r.Geom.domain.Saturation.SSat = 1.0

    r.Wells.Names = ""

    # single always-on cycle; rain VALUE is set per phase (storm: -rate, recession: 0)
    r.Cycle.Names = "constant"
    r.Cycle.constant.Names = "alltime"
    r.Cycle.constant.alltime.Length = 1
    r.Cycle.constant.Repeat = -1

    r.BCPressure.PatchNames = "x_lower x_upper y_lower y_upper z_lower z_upper"
    # base + cross-slope sides + upslope x face: no-flow
    for p in ("x_lower", "y_lower", "y_upper", "z_lower"):
        r.Patch[p].BCPressure.Type = "FluxConst"
        r.Patch[p].BCPressure.Cycle = "constant"
        r.Patch[p].BCPressure.alltime.Value = 0.0
    # downslope x=L side FACE: lateral-GW outlet as a constant-head (DirEquil) BC -> water table z=H_ext.
    # NOTE: this is C -> inf (Dirichlet), NOT the in-house finite kr-weighted conductance (over-drain delta).
    # PF_NOGW=1 turns this face into a no-flow wall -> a CONTROL run whose total outflow is OVERLAND ONLY,
    # so lateral_GW = total_out(full) - total_out(no-GW) (difference-of-runs separation; the nonlinearity
    # of removing the toe drainage is a documented caveat, negligible when GW is small).
    if os.environ.get("PF_NOGW", "").strip() == "1":
        r.Patch.x_upper.BCPressure.Type = "FluxConst"
        r.Patch.x_upper.BCPressure.Cycle = "constant"
        r.Patch.x_upper.BCPressure.alltime.Value = 0.0
    else:
        r.Patch.x_upper.BCPressure.Type = "DirEquilRefPatch"
        r.Patch.x_upper.BCPressure.Cycle = "constant"
        r.Patch.x_upper.BCPressure.RefGeom = "domain"
        r.Patch.x_upper.BCPressure.RefPatch = "z_lower"      # ref at the base (z=0)
        r.Patch.x_upper.BCPressure.alltime.Value = H_EXT     # psi at z=0 -> head = H_EXT everywhere on the face
    # top surface: OverlandFlow ponding store + downslope routing (rain value set per phase)
    r.Patch.z_upper.BCPressure.Type = "OverlandFlow"
    r.Patch.z_upper.BCPressure.Cycle = "constant"

    # tilt: surface decreases in +x (TopoSlopesX < 0 routes overland toward +x, B3-verified)
    r.TopoSlopesX.Type = "Constant"
    r.TopoSlopesX.GeomNames = "domain"
    r.TopoSlopesX.Geom.domain.Value = -S0
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

    # robust nonlinear/overland solver block (= B4; OverlandFlow Jacobian incomplete -> finite-diff)
    r.Solver = "Richards"
    r.Solver.MaxIter = 25000
    r.Solver.Nonlinear.MaxIter = 200
    r.Solver.Nonlinear.ResidualTol = 1e-8
    r.Solver.Nonlinear.EtaChoice = "EtaConstant"
    r.Solver.Nonlinear.EtaValue = 1e-3
    r.Solver.Nonlinear.UseJacobian = False
    r.Solver.Nonlinear.DerivativeEpsilon = 1e-12
    r.Solver.Nonlinear.StepTol = 1e-16
    r.Solver.Nonlinear.Globalization = "LineSearch"
    r.Solver.Linear.KrylovDimension = 20
    r.Solver.Linear.MaxRestart = 2
    r.Solver.Linear.Preconditioner = "PFMG"
    # report the integrated overland outflow per step (separates overland from lateral GW in the partition)
    r.Solver.PrintOverlandSum = True


def _set_timing(r, dt, dump, stop):
    r.TimingInfo.BaseUnit = dt
    r.TimingInfo.StartCount = 0
    r.TimingInfo.StartTime = 0.0
    r.TimingInfo.StopTime = stop
    r.TimingInfo.DumpInterval = dump
    r.TimeStep.Type = "Growth"
    r.TimeStep.InitialStep = dt
    r.TimeStep.GrowthFactor = 1.3
    r.TimeStep.MaxStep = dump
    r.TimeStep.MinStep = dt * 1e-3


def _final_press(out_dir, run_name):
    files = sorted(glob.glob(os.path.join(out_dir, f"{run_name}.out.press.*.pfb")),
                   key=lambda p: int(os.path.basename(p).split(".")[-2]))
    if not files:
        raise SystemExit(f"{run_name}: no pressure output -- the ParFlow run failed")
    return files[-1]


def _read_series(out_dir, run_name, dump, theta_s):
    """Read all (time, pressure(z,y,x), theta(z,y,x), overlandsum(y,x)) frames, sorted by step."""
    files = sorted(glob.glob(os.path.join(out_dir, f"{run_name}.out.press.*.pfb")),
                   key=lambda p: int(os.path.basename(p).split(".")[-2]))
    times, press, theta, osum = [], [], [], []
    for pf in files:
        i = int(os.path.basename(pf).split(".")[-2])
        P = np.asarray(read_pfb(pf))                                  # (NZ, NY, NX)
        S = np.asarray(read_pfb(os.path.join(out_dir, f"{run_name}.out.satur.{i:05d}.pfb")))
        times.append(i * dump)
        press.append(P.copy())
        theta.append(theta_s * S)
        of = os.path.join(out_dir, f"{run_name}.out.overlandsum.{i:05d}.pfb")
        osum.append(np.asarray(read_pfb(of)) if os.path.exists(of) else np.zeros((1, NY, NX)))
    return (np.array(times, float), np.array(press, float),
            np.array(theta, float), np.array(osum, float))


def run_scenario(key, cfg, out_root):
    soil = cfg["soil"]
    theta_s = soil["theta_s"]
    rate, storm_dur, t_end = cfg["rate"], cfg["storm_dur"], cfg["t_end"]
    rec_dur = t_end - storm_dur

    nogw = os.environ.get("PF_NOGW", "").strip() == "1"
    nzsuf = "" if NZ == 8 else f"_nz{NZ}"    # NZ=8 keeps the ORIGINAL output names (backwards-compatible)
    tag = f"{key}{'_nogw' if nogw else ''}{nzsuf}"   # control / nz suffixes -> separate outputs
    out_dir = os.path.join(out_root, tag)
    mkdir(out_dir)

    # ---- phase 1: storm (constant rain) -------------------------------------
    dump_s = storm_dur / N_STORM_FRAMES
    dt_s = dump_s / SUBSTEPS
    r = Run(f"{tag}_storm", __file__)
    configure_static(r, soil)
    r.ICPressure.Type = "HydroStaticPatch"
    r.ICPressure.GeomNames = "domain"
    r.Geom.domain.ICPressure.RefGeom = "domain"
    r.Geom.domain.ICPressure.RefPatch = "z_lower"     # water table at z = Z_WT (psi = Z_WT - z)
    r.Geom.domain.ICPressure.Value = Z_WT
    r.Patch.z_upper.BCPressure.alltime.Value = -rate  # negative = into the domain
    _set_timing(r, dt_s, dump_s, storm_dur)
    r.run(working_directory=out_dir)
    t_s, p_s, th_s, os_s = _read_series(out_dir, f"{tag}_storm", dump_s, theta_s)
    storm_final = os.path.basename(_final_press(out_dir, f"{tag}_storm"))

    # ---- phase 2: recession (rain off, restart from storm's final pressure) --
    dump_r = rec_dur / N_REC_FRAMES
    dt_r = dump_r / SUBSTEPS
    r = Run(f"{tag}_rec", __file__)
    configure_static(r, soil)
    r.ICPressure.Type = "PFBFile"
    r.ICPressure.GeomNames = "domain"
    r.Geom.domain.ICPressure.RefGeom = "domain"
    r.Geom.domain.ICPressure.RefPatch = "z_upper"
    r.Geom.domain.ICPressure.FileName = storm_final   # in out_dir (the run cwd); carries the pond
    r.Patch.z_upper.BCPressure.alltime.Value = 0.0    # recession: no rain
    _set_timing(r, dt_r, dump_r, rec_dur)
    r.run(working_directory=out_dir)
    t_r, p_r, th_r, os_r = _read_series(out_dir, f"{tag}_rec", dump_r, theta_s)

    # ---- concatenate (drop phase-2 t=0, the shared boundary frame) ----------
    times = np.concatenate([t_s, storm_dur + t_r[1:]])
    press = np.concatenate([p_s, p_r[1:]], axis=0)
    theta = np.concatenate([th_s, th_r[1:]], axis=0)
    osum = np.concatenate([os_s, os_r[1:]], axis=0)

    xc = (np.arange(NX) + 0.5) * DX
    yc = (np.arange(NY) + 0.5) * DY
    zc = (np.arange(NZ) + 0.5) * DZ
    summary = os.path.join(SUMMARY_DIR, f"coupled_3d_{tag}.npz")
    np.savez(summary, x=xc, y=yc, z=zc, times=times, press=press, theta=theta,
             overlandsum=osum, porosity=theta_s, dx=DX, dy=DY, dz=DZ, L=L, W=W, H=H,
             slope=S0, n_man=N_SI, z_wt=Z_WT, h_ext=H_EXT, Ks=soil["Ks"],
             rate=rate, storm_dur=storm_dur, t_end=t_end)

    # ---- physical-sanity readout (global partition; the step-1 make-or-break) ----
    # ParFlow's mass-consistent OverlandFlow surface store is max(psi_top, 0) (B4 / README S5c).
    cell_vol = DX * DY * DZ
    cell_area = DX * DY
    p_top = press[:, -1, :, :]                          # (t, NY, NX) top-cell pressure
    d_map = np.maximum(p_top, 0.0)                      # surface ponding depth d(x,y)
    pond = d_map.sum(axis=(1, 2)) * cell_area           # surface store volume [m^3]
    W_soil = theta.sum(axis=(1, 2, 3)) * cell_vol       # subsurface stored water [m^3]
    top_area = (NX * DX) * (NY * DY)                     # 5 m^2
    cum_rain = rate * top_area * np.minimum(times, storm_dur)
    # total outflow (overland + lateral GW) by global balance:
    total_out = cum_rain - (W_soil - W_soil[0]) - (pond - pond[0])
    # NOTE: PrintOverlandSum (osum) proved UNRELIABLE as an overland-VOLUME measure here -- on this
    # kinematic OverlandFlow BC it read ~0.03 m^3 against a true overland of ~0.48 m^3 (cross-checked
    # vs the Manning-equilibrium ponding depth AND the no-GW control). The VALIDATED overland/GW
    # separation is the difference of two runs: overland = total_out(PF_NOGW=1 control);
    # lateral_GW = total_out(full) - total_out(no-GW control). (osum is still dumped for the record.)
    ov_cum = osum.reshape(osum.shape[0], -1).sum(axis=1)
    ov_cum = ov_cum - ov_cum[0]

    i_se = int(np.argmin(np.abs(times - storm_dur)))
    # mass-balance: the global books should close (total_out is defined to, so report its monotonicity
    # + the overland/GW split as the physical readout instead)
    print(f"  {tag}: {times.size} frames  soil Ks={soil['Ks']}  rate={rate} storm={storm_dur}d t_end={t_end}d"
          f"{'  [NO-GW CONTROL: total outflow = OVERLAND only]' if nogw else ''}")
    print(f"     peak ponding d   = {1000*d_map.max():.2f} mm "
          f"(at t={times[d_map.reshape(times.size,-1).max(axis=1).argmax()]:.3f}d); "
          f"d at storm-end max = {1000*d_map[i_se].max():.2f} mm; d final max = {1000*d_map[-1].max():.2f} mm")
    # downslope ponding profile (averaged across y) at storm end -- should rise toward the x=L outlet
    prof = d_map[i_se].mean(axis=0)
    print(f"     storm-end d(x) profile [mm], x=0..L: "
          f"{np.array2string(1000*prof, precision=1, max_line_width=200)}")
    print(f"     soil dW          = {W_soil[-1]-W_soil[0]:+.4f} m^3   pond final = {pond[-1]:.4f} m^3")
    print(f"     cum_rain         = {cum_rain[-1]:.4f} m^3")
    print(f"     total outflow    = {total_out[-1]:.4f} m^3  "
          f"({'OVERLAND only (no-GW control)' if nogw else 'overland + lateral GW, by global balance'})")
    print(f"     [PrintOverlandSum = {ov_cum[-1]:.4f} m^3 -- UNRELIABLE here; use the no-GW-control difference]")
    print(f"     -> WROTE {os.path.basename(summary)}", flush=True)


SUMMARY_DIR = get_absolute_path("summaries")
mkdir(SUMMARY_DIR)
RUN_ROOT = get_absolute_path("runs")
mkdir(RUN_ROOT)

_only = os.environ.get("PF_SCENARIOS", "").strip()
_keys = [k.strip() for k in _only.split(",") if k.strip()] if _only else list(SCENARIOS)
print(f"=== ParFlow B5 coupled 3-D hillslope ({len(_keys)} scenario(s): {', '.join(_keys)}) ===")
for _key in _keys:
    if _key not in SCENARIOS:
        raise SystemExit(f"unknown scenario {_key!r}; choose from {list(SCENARIOS)}")
    run_scenario(_key, SCENARIOS[_key], RUN_ROOT)
print("DONE")
