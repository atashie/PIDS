#!/usr/bin/env python3
"""ParFlow benchmark case B4 -- coupled surface<->subsurface ponding column.

Off-the-shelf cross-check for the in-house COUPLED solver
(``pids_forward.physics.coupling.CoupledProblem``), matching the in-house coupling
sanity matrix (``forward-model/viz/run_coupling_sanity.py``): a 2 m loam column with a
CLOSED no-flux base, hit by a storm then a rainless recession, across a rain x antecedent
matrix. This is the PONDING / infiltration-excess regime the subsurface sweep deferred
(README S5a/S5b) -- run here on ParFlow's NATIVE coupled turf: an **OverlandFlow** top BC
(ParFlow's surface ponding store = surface pressure) on a FLAT column (TopoSlopes=0, so no
lateral routing -- pure ponding / re-infiltration partition, the direct analog of the
in-house surface-ponding store).

Run (inside the ParFlow Docker container, from a copy of this file in the mounted dir):
    docker run --rm -v $HOME/parflow-runs/coupled_column:/data parflow/parflow:latest coupled_column.py 1 1 1

  Restrict to one scenario for first-light debugging:
    docker run --rm -e PF_SCENARIOS=normal_on_normal -v $HOME/parflow-runs/coupled_column:/data \
        parflow/parflow:latest coupled_column.py 1 1 1

Matched parameters (= run_coupling_sanity.py SOIL + matrix)
-----------------------------------------------------------
  in-house (coupling.py / run_coupling_sanity.py)   ParFlow
  ----------------------------------------------    --------------------------------------
  2 m column, 80 cells, z up, surface at z=2        NX=NY=1, NZ=80, DZ=2/80=0.025 (k=0 = base)
  van Genuchten loam Ks=0.25 m/day                  Perm=0.25 (rho=mu=g=1 -> K==Perm)
    theta_r=0.078, theta_s=0.43                     Porosity=theta_s; SRes=theta_r/theta_s
    alpha=3.6 /m, n=1.56                            Saturation/RelPerm Alpha=3.6, N=1.56
  closed no-flux base                               Patch.bottom FluxConst=0
  surface ponding store d (land-surface exchange)   Patch.top OverlandFlow (d = surface pressure)
  rain hyetograph (storm then recession)            two-phase: storm run -> PFBFile-IC restart, rain=0
  uniform antecedent psi0                           ICPressure Constant (phase 1)

The recession is done as a TWO-PHASE run (README S5: the storm/recession time cycle was
unreliable in the sweep). Phase 1 applies constant rain over the storm; phase 2 restarts from
the storm's FINAL pressure (PFBFile IC -- which carries the pond, since the pond is positive
top-cell surface pressure) with rain=0 and runs the recession. The two output series are
concatenated (phase-2 times offset by storm_dur; shared boundary frame de-duplicated).

Per-scenario export to summaries/coupled_<key>.npz: z (cell centres), times,
head(t,z), theta(t,z), porosity, dz -- the side-by-side builder
(../../benchmarks/build_comparison_coupled.py) computes d, the infiltration partition, and
the mass balance from these (so the export stays a thin profile dump, as for the sweep).

Documented formulation deltas (see ../README.md S4), same as the column/sweep cases:
  * ParFlow vG omits the Vogel/Ippisch air-entry cap the in-house model applies (matters MORE
    here -- the column SATURATES at the surface, where the cap shifts theta/K/tangent).
  * ParFlow needs a small SpecificStorage (1e-6); the in-house model ignores Ss (unconfined).
  * ParFlow cell-centred FV (80 centres) vs in-house P1 FEM (81 nodes) -> compare by interp.
  * ParFlow OverlandFlow surface store vs the in-house land-surface exchange-flux (lambda)
    formulation -- the partition near the ponding threshold is the comparison of interest.
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

# --- matched soil + grid (= run_coupling_sanity.py) --------------------------
THETA_R, THETA_S = 0.078, 0.43
ALPHA, N, KS = 3.6, 1.56, 0.25      # SE-Piedmont loam (coupled SOIL: Ks=0.25, not the sweep's 0.2496)
SRES = THETA_R / THETA_S
NZ = 80
DEPTH = 2.0
DZ = DEPTH / NZ                     # 0.025 m
MANNING = 0.03 / 86400.0           # SI Manning -> ParFlow day-units (irrelevant: flat single column)

# antecedent (uniform psi0, m) and storms (rate m/day, durations day) -- identical to the in-house run
ANTECEDENT = {"dry": -3.0, "normal": -1.0, "wet": -0.15}
STORMS = {
    "normal":  dict(rate=0.30, storm_dur=0.30, t_end=1.0),   # moderate SE-Piedmont storm
    "extreme": dict(rate=3.00, storm_dur=0.05, t_end=0.40),  # 100-yr Atlas-14-style burst
}
SCENARIOS = {
    f"{sk}_on_{ak}": dict(psi0=psi0, rate=s["rate"], storm_dur=s["storm_dur"], t_end=s["t_end"])
    for sk, s in STORMS.items() for ak, psi0 in ANTECEDENT.items()
}

N_STORM_FRAMES = 20    # output frames over the storm phase
N_REC_FRAMES = 40      # output frames over the recession phase
SUBSTEPS = 5           # internal steps per dump (dump = SUBSTEPS * dt)


def configure_static(r):
    """Grid, soil, phases, OverlandFlow top BC, solver -- everything scenario-independent."""
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

    # single always-on cycle; the rain VALUE is set per phase (storm: -rate, recession: 0)
    r.Cycle.Names = "constant"
    r.Cycle.constant.Names = "alltime"
    r.Cycle.constant.alltime.Length = 1
    r.Cycle.constant.Repeat = -1

    r.BCPressure.PatchNames = "left right front back bottom top"
    for p in ("left", "right", "front", "back", "bottom"):
        r.Patch[p].BCPressure.Type = "FluxConst"
        r.Patch[p].BCPressure.Cycle = "constant"
        r.Patch[p].BCPressure.alltime.Value = 0.0
    # top: OverlandFlow surface store (the ponding store). rain value set per phase.
    r.Patch.top.BCPressure.Type = "OverlandFlow"
    r.Patch.top.BCPressure.Cycle = "constant"

    # flat -> no lateral routing -> pure ponding / re-infiltration partition
    r.TopoSlopesX.Type = "Constant"
    r.TopoSlopesX.GeomNames = "domain"
    r.TopoSlopesX.Geom.domain.Value = 0.0
    r.TopoSlopesY.Type = "Constant"
    r.TopoSlopesY.GeomNames = "domain"
    r.TopoSlopesY.Geom.domain.Value = 0.0
    r.Mannings.Type = "Constant"
    r.Mannings.GeomNames = "domain"
    r.Mannings.Geom.domain.Value = MANNING

    r.PhaseSources.water.Type = "Constant"
    r.PhaseSources.water.GeomNames = "domain"
    r.PhaseSources.water.Geom.domain.Value = 0.0
    r.KnownSolution = "NoKnownSolution"

    # robust nonlinear/overland solver block (OverlandFlow Jacobian is incomplete -> finite-diff)
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


def _set_timing(r, dt, dump, stop):
    r.TimingInfo.BaseUnit = dt
    r.TimingInfo.StartCount = 0
    r.TimingInfo.StartTime = 0.0
    r.TimingInfo.StopTime = stop
    r.TimingInfo.DumpInterval = dump
    # adaptive timestep (Growth) for the ponding stiffness; still lands on dump multiples
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


def _read_series(out_dir, run_name, dump):
    """Read all (time, head(z), theta(z)) frames for a run, sorted by step index."""
    files = sorted(glob.glob(os.path.join(out_dir, f"{run_name}.out.press.*.pfb")),
                   key=lambda p: int(os.path.basename(p).split(".")[-2]))
    times, head, theta = [], [], []
    for pf in files:
        i = int(os.path.basename(pf).split(".")[-2])
        P = np.asarray(read_pfb(pf)).reshape(-1)
        S = np.asarray(read_pfb(os.path.join(out_dir, f"{run_name}.out.satur.{i:05d}.pfb"))).reshape(-1)
        times.append(i * dump)
        head.append(P.copy())
        theta.append(THETA_S * S)
    return np.array(times, float), np.array(head, float), np.array(theta, float)


def run_scenario(key, cfg, out_root):
    psi0, rate = cfg["psi0"], cfg["rate"]
    storm_dur, t_end = cfg["storm_dur"], cfg["t_end"]
    rec_dur = t_end - storm_dur

    out_dir = os.path.join(out_root, key)
    mkdir(out_dir)

    # ---- phase 1: storm (constant rain) -------------------------------------
    dump_s = storm_dur / N_STORM_FRAMES
    dt_s = dump_s / SUBSTEPS
    r = Run(f"{key}_storm", __file__)
    configure_static(r)
    r.Geom.domain.ICPressure.RefGeom = "domain"
    r.Geom.domain.ICPressure.RefPatch = "top"
    r.ICPressure.Type = "Constant"
    r.ICPressure.GeomNames = "domain"
    r.Geom.domain.ICPressure.Value = psi0
    r.Patch.top.BCPressure.alltime.Value = -rate     # negative = into the domain
    _set_timing(r, dt_s, dump_s, storm_dur)
    r.run(working_directory=out_dir)
    t_s, head_s, theta_s = _read_series(out_dir, f"{key}_storm", dump_s)
    storm_final = os.path.basename(_final_press(out_dir, f"{key}_storm"))

    # ---- phase 2: recession (rain off, restart from storm's final pressure) --
    dump_r = rec_dur / N_REC_FRAMES
    dt_r = dump_r / SUBSTEPS
    r = Run(f"{key}_rec", __file__)
    configure_static(r)
    r.ICPressure.Type = "PFBFile"
    r.ICPressure.GeomNames = "domain"
    r.Geom.domain.ICPressure.FileName = storm_final  # in out_dir (the run cwd); carries the pond
    r.Patch.top.BCPressure.alltime.Value = 0.0       # recession: no rain
    _set_timing(r, dt_r, dump_r, rec_dur)
    r.run(working_directory=out_dir)
    t_r, head_r, theta_r = _read_series(out_dir, f"{key}_rec", dump_r)

    # ---- concatenate (drop phase-2 t=0, the shared boundary frame) ----------
    times = np.concatenate([t_s, storm_dur + t_r[1:]])
    head = np.concatenate([head_s, head_r[1:]], axis=0)
    theta = np.concatenate([theta_s, theta_r[1:]], axis=0)

    zc = (np.arange(NZ) + 0.5) * DZ
    summary = os.path.join(SUMMARY_DIR, f"coupled_{key}.npz")
    np.savez(summary, z=zc, times=times, head=head, theta=theta,
             porosity=THETA_S, dz=DZ, psi0=psi0, rate=rate, storm_dur=storm_dur, t_end=t_end)

    # physical-sanity readout: did it pond during the storm and re-infiltrate after?
    # ParFlow's mass-consistent OverlandFlow surface store is max(psi_top, 0) -- NOT the DZ/2
    # offset used for the standalone overland case (README S5a): that offset breaks the closed-
    # column balance by exactly DZ/2 once ponded; max(psi_top,0) closes it to ~1e-6 (verified).
    p_top = head[:, -1]
    d = np.maximum(p_top, 0.0)                        # ParFlow OverlandFlow surface store
    W_soil = theta.sum(axis=1) * DZ
    cum_rain = rate * np.minimum(times, storm_dur)
    mbe = np.abs((W_soil - W_soil[0]) + d - cum_rain) / np.where(cum_rain > 0, cum_rain, 1.0)
    i_storm_end = int(np.argmin(np.abs(times - storm_dur)))
    print(f"  {key}: {times.size} frames  psi0={psi0} rate={rate} storm={storm_dur}d t_end={t_end}d")
    print(f"     peak ponding d = {1000*d.max():.2f} mm (at t={times[np.argmax(d)]:.3f}d); "
          f"d at storm-end = {1000*d[i_storm_end]:.2f} mm; d final = {1000*d[-1]:.2f} mm")
    print(f"     soil dW = {W_soil[-1]-W_soil[0]:.4f} m; cum_rain = {cum_rain[-1]:.4f} m; "
          f"max MBE = {mbe.max():.2e}  -> WROTE {os.path.basename(summary)}")


SUMMARY_DIR = get_absolute_path("summaries")
mkdir(SUMMARY_DIR)
RUN_ROOT = get_absolute_path("runs")
mkdir(RUN_ROOT)

_only = os.environ.get("PF_SCENARIOS", "").strip()
_keys = [k.strip() for k in _only.split(",") if k.strip()] if _only else list(SCENARIOS)
print(f"=== ParFlow B4 coupled column ({len(_keys)} scenario(s): {', '.join(_keys)}) ===")
for _key in _keys:
    if _key not in SCENARIOS:
        raise SystemExit(f"unknown scenario {_key!r}; choose from {list(SCENARIOS)}")
    run_scenario(_key, SCENARIOS[_key], RUN_ROOT)
print("DONE")
