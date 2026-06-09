#!/usr/bin/env python3
"""ParFlow benchmark case B2 -- clean 1-D vertical infiltration column (loam).

Off-the-shelf cross-check for the in-house forward-model SUBSURFACE module
(``pids_forward.physics.richards``). A uniform Carsel & Parrish (1988) loam
column under a constant rain flux below Ks (so it never ponds: a pure
flux-infiltration test), with every parameter matched to the in-house
constitutive values so the two engines compare apples-to-apples.

Run (inside the ParFlow Docker container, from this file's directory):

    docker run --rm -v $HOME/parflow-runs/column_1d:/data parflow/parflow:latest column_1d.py 1 1 1

(Use an explicit absolute mount path. The -v "$(pwd)":/data form from the ParFlow docs
did NOT resolve in this PowerShell->wsl->docker setup; see ../USAGE.md.)

It runs ParFlow (writing ``output/column_1d.out.{press,satur}.*.pfb``) and then
reads the results back with pftools and prints a physical-sanity summary.

Mapping to the in-house model (richards.py / run_subsurface_sanity.py)
----------------------------------------------------------------------
  in-house                              ParFlow
  ------------------------------------  --------------------------------------
  unit interval [0,1] m, 60 cells       NX=NY=1, NZ=60, DZ=1/60 m (z up; k=0 = bottom)
  van Genuchten-Mualem loam             Phase.Saturation/RelPerm = "VanGenuchten"
    theta_r=0.078, theta_s=0.43           Porosity = theta_s = 0.43; SRes = theta_r/theta_s
    alpha=3.6 /m, n=1.56                  Alpha=3.6, N=1.56 (same defn, m = 1 - 1/n)
    Ks=0.2496 m/day                       Perm=0.2496 with rho=mu=g=1 (K == Perm trick)
  uniform IC psi0 = -1.0 m              ICPressure.Type="Constant", Value=-1.0
  top rain flux q (m/day), no pond      Patch.top FluxConst Value = -q  (NEG = INTO domain)
  no-flow bottom (default Neumann)      Patch.bottom FluxConst Value = 0.0

Documented formulation deltas (see ../README.md S4):
  * ParFlow van Genuchten omits the Vogel/Ippisch air-entry cap the in-house model applies
    (h_s=-0.02 m). That cap modifies theta(psi), K(psi), AND the near-saturation tangent
    (constitutive.py applies it in effective_saturation, K, and capacity) -- so the delta is
    broader than saturation alone: it shifts the infiltration-front speed and matters for any
    comparison approaching saturation. Small here (stays unsaturated; top only reaches S~0.98).
  * ParFlow requires a SpecificStorage (set 1e-6); the in-house model ignores Ss (unconfined).
    Tiny but NOT inactive: ParFlow storage carries a compressible term S*Ss*psi alongside
    porosity*S. Here it is O(1e-7 m) and accounts for most of the residual in the
    water-content balance below (which tracks only porosity*S, the theta-equivalent).
  * ParFlow is cell-centred finite volume (60 cell centres at (k+0.5)*DZ); the in-house
    model is P1 FEM (61 nodes). Profiles must be compared by interpolation.
"""
import glob
import os

import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path

try:
    from parflow.tools.io import read_pfb
except ImportError:  # tolerate pftools layout differences across versions
    from parflow import read_pfb

# --- parameters, matched to the in-house model -------------------------------
RUN_NAME = "column_1d"
NZ = 60
DEPTH = 1.0                       # column depth (m)
DZ = DEPTH / NZ
THETA_R, THETA_S = 0.078, 0.43    # Carsel & Parrish (1988) loam
ALPHA, N = 3.6, 1.56              # van Genuchten (1/m, -)
KS = 0.2496                       # saturated hydraulic conductivity (m/day)
PSI0 = -1.0                       # uniform antecedent pressure head (m)
Q_RAIN = 0.10                     # top rainfall flux (m/day); < Ks -> no ponding
T_STOP = 1.0                      # simulated time (day)
DT = 0.005                        # time step (day); matches in-house dt
DUMP = 0.05                       # output interval (day) -> 21 frames (00000..00020)
SRES = THETA_R / THETA_S          # ParFlow residual SATURATION (= theta_r/theta_s)

# --- build the ParFlow run ----------------------------------------------------
r = Run(RUN_NAME, __file__)
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

# Perm == saturated hydraulic conductivity (density=viscosity=gravity=1 below)
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
r.Geom.domain.SpecificStorage.Value = 1.0e-6  # in-house ignores Ss; tiny but nonzero (see readout)

r.Phase.Names = "water"
r.Phase.water.Density.Type = "Constant"
r.Phase.water.Density.Value = 1.0
r.Phase.water.Viscosity.Type = "Constant"
r.Phase.water.Viscosity.Value = 1.0
r.Contaminants.Names = ""
r.Geom.Retardation.GeomNames = ""
r.Gravity = 1.0

r.TimingInfo.BaseUnit = 1.0
r.TimingInfo.StartCount = 0
r.TimingInfo.StartTime = 0.0
r.TimingInfo.StopTime = T_STOP
r.TimingInfo.DumpInterval = DUMP
r.TimeStep.Type = "Constant"
r.TimeStep.Value = DT

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

# one always-on cycle (constant forcing)
r.Cycle.Names = "constant"
r.Cycle.constant.Names = "alltime"
r.Cycle.constant.alltime.Length = 1
r.Cycle.constant.Repeat = -1

r.BCPressure.PatchNames = "left right front back bottom top"
# sides + bottom: no-flow (zero constant flux)
r.Patch.left.BCPressure.Type = "FluxConst"
r.Patch.left.BCPressure.Cycle = "constant"
r.Patch.left.BCPressure.alltime.Value = 0.0
r.Patch.right.BCPressure.Type = "FluxConst"
r.Patch.right.BCPressure.Cycle = "constant"
r.Patch.right.BCPressure.alltime.Value = 0.0
r.Patch.front.BCPressure.Type = "FluxConst"
r.Patch.front.BCPressure.Cycle = "constant"
r.Patch.front.BCPressure.alltime.Value = 0.0
r.Patch.back.BCPressure.Type = "FluxConst"
r.Patch.back.BCPressure.Cycle = "constant"
r.Patch.back.BCPressure.alltime.Value = 0.0
r.Patch.bottom.BCPressure.Type = "FluxConst"
r.Patch.bottom.BCPressure.Cycle = "constant"
r.Patch.bottom.BCPressure.alltime.Value = 0.0
# top: rainfall flux INTO the domain (negative sign, per ParFlow convention)
r.Patch.top.BCPressure.Type = "FluxConst"
r.Patch.top.BCPressure.Cycle = "constant"
r.Patch.top.BCPressure.alltime.Value = -Q_RAIN

r.TopoSlopesX.Type = "Constant"
r.TopoSlopesX.GeomNames = "domain"
r.TopoSlopesX.Geom.domain.Value = 0.0
r.TopoSlopesY.Type = "Constant"
r.TopoSlopesY.GeomNames = "domain"
r.TopoSlopesY.Geom.domain.Value = 0.0
r.Mannings.Type = "Constant"
r.Mannings.GeomNames = "domain"
r.Mannings.Geom.domain.Value = 0.0

r.ICPressure.Type = "Constant"
r.ICPressure.GeomNames = "domain"
r.Geom.domain.ICPressure.Value = PSI0
r.Geom.domain.ICPressure.RefGeom = "domain"
r.Geom.domain.ICPressure.RefPatch = "top"

r.PhaseSources.water.Type = "Constant"
r.PhaseSources.water.GeomNames = "domain"
r.PhaseSources.water.Geom.domain.Value = 0.0
r.KnownSolution = "NoKnownSolution"

r.Solver = "Richards"
r.Solver.MaxIter = 2500
r.Solver.Nonlinear.MaxIter = 200
r.Solver.Nonlinear.ResidualTol = 1e-9
r.Solver.Nonlinear.EtaChoice = "Walker1"
r.Solver.Nonlinear.EtaValue = 1e-5
r.Solver.Nonlinear.UseJacobian = True
r.Solver.Nonlinear.DerivativeEpsilon = 1e-10
r.Solver.Linear.KrylovDimension = 10
r.Solver.Linear.Preconditioner = "PFMG"

# --- run ----------------------------------------------------------------------
out_dir = get_absolute_path("output")
mkdir(out_dir)
r.run(working_directory=out_dir)

# --- readout + physical-sanity summary ---------------------------------------
def column(path):
    """Read a .pfb and flatten to a 1-D z-profile (index 0 = bottom, -1 = top)."""
    return np.asarray(read_pfb(path)).reshape(-1)


def step_index(path):
    return int(os.path.basename(path).split(".")[-2])


def vg_saturation(psi):
    """Analytic van Genuchten saturation S(psi) for the IC check (no air-entry cap)."""
    m = 1.0 - 1.0 / N
    se = (1.0 + (ALPHA * abs(psi)) ** N) ** (-m)
    return SRES + se * (1.0 - SRES)


press_files = sorted(glob.glob(os.path.join(out_dir, f"{RUN_NAME}.out.press.*.pfb")))
assert press_files, "no pressure output written -- the ParFlow run failed"

z_centers = (np.arange(NZ) + 0.5) * DZ  # cell-centre elevations (m)
cell_vol = 1.0 * 1.0 * DZ               # DX*DY*DZ (m^3)

SS_VAL = 1.0e-6  # must match r.Geom.domain.SpecificStorage.Value set above

times, water_theta, water_full, top_S, bot_S, front = [], [], [], [], [], []
head_tz, theta_tz = [], []  # full profiles psi(t,z), theta(t,z) for the ../../benchmarks/ side-by-side
S0 = None
for pf in press_files:
    i = step_index(pf)
    S = column(os.path.join(out_dir, f"{RUN_NAME}.out.satur.{i:05d}.pfb"))
    P = column(pf)  # pressure head psi (same timestep)
    if S0 is None:
        S0 = S.copy()
    times.append(i * DUMP)
    head_tz.append(P.copy())      # psi(z) [m]
    theta_tz.append(THETA_S * S)  # theta(z) [m3/m3] = porosity * saturation
    # water-content (theta-equivalent) storage = sum_k porosity*S_k*vol -- the quantity
    # directly comparable to the in-house model (which carries no compressible storage).
    water_theta.append(float(np.sum(S) * THETA_S * cell_vol))
    # FULL ParFlow storage = sum_k (porosity + Ss*psi_k)*S_k*vol -- what ParFlow conserves;
    # the compressible term Ss*S*psi is tiny but nonzero.
    water_full.append(float(np.sum((THETA_S + SS_VAL * P) * S) * cell_vol))
    top_S.append(float(S[-1]))
    bot_S.append(float(S[0]))
    wetted = np.where(S > S0 + 1e-4)[0]
    front.append(float(z_centers[wetted.min()]) if wetted.size else DEPTH)

cum_in = Q_RAIN * times[-1]  # no-flow bottom + q<Ks: all rain enters the column and stays
dW_theta = water_theta[-1] - water_theta[0]
dW_full = water_full[-1] - water_full[0]
mbe_theta = abs(dW_theta - cum_in) / cum_in if cum_in else 0.0
mbe_full = abs(dW_full - cum_in) / cum_in if cum_in else 0.0
ic_ok = abs(np.mean(S0) - vg_saturation(PSI0)) < 0.02
wet_ok = top_S[-1] > top_S[0] and front[-1] < front[0]
mb_ok = mbe_full < 1e-3  # ParFlow conserves the FULL storage to ~solver tolerance

print("=== B2 column_1d  physical sanity ===")
print(f"grid: {NZ} cells over {DEPTH} m   q={Q_RAIN} m/day  Ks={KS} m/day  psi0={PSI0} m")
print(f"IC saturation: ParFlow mean={np.mean(S0):.4f}  analytic vG(psi0)={vg_saturation(PSI0):.4f}  -> {'OK' if ic_ok else 'MISMATCH'}")
print(f"top S:    {top_S[0]:.4f} -> {top_S[-1]:.4f}   (should rise under rain)")
print(f"bottom S: {bot_S[0]:.4f} -> {bot_S[-1]:.4f}")
print(f"wetting-front elevation: {front[0]:.3f} -> {front[-1]:.3f} m  (advances downward)")
print(f"cumulative infiltration: {cum_in:.6f} m")
print(f"  water-content balance (theta-equiv, vs in-house): dW={dW_theta:.6f} m  residual={mbe_theta:.2e}")
print(f"    (residual is mostly the compressible term Ss*S*psi this balance omits)")
print(f"  full ParFlow balance (incl. Ss*S*psi):            dW={dW_full:.6f} m  MB error={mbe_full:.2e}")
print("PHYSICAL SANITY:", "PASS" if (ic_ok and wet_ok and mb_ok) else "CHECK")

# benchmark export: profiles for the ../../benchmarks/ side-by-side comparison
summary_path = os.path.join(out_dir, "column_1d_profiles.npz")
np.savez(summary_path, z=z_centers, times=np.array(times, dtype=float),
         head=np.array(head_tz, dtype=float), theta=np.array(theta_tz, dtype=float),
         porosity=float(THETA_S))
print(f"WROTE {summary_path}")
