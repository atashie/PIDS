"""DEMO (2026-06-22 unified-feature redesign, Stage 1 = the RESOLVED truth): a coarse well-sorted
SANDY conveyance channel OPEN TO THE SURFACE on a hillslope, through a rainfall event.

This is the "show me it works on a real hillslope" step (Arik 2026-06-22). At a fine mesh a sandy
channel needs NO special feature machinery -- it is a high-K / high-drainable-porosity MATERIAL ZONE,
and all three PIDS couplings emerge from the resolved CoupledProblem physics:
  (1) SURFACE INTAKE  : rain (and overland flow) land on the sand and infiltrate fast (Ks_sand >> rain);
  (2) LATERAL CONVEYANCE: the high-K sand carries water downslope to the outlet (Darcy through the sand
                          + any ponded surface flow), discharged by a sand-zone GHB + the surface edge;
  (3) SOIL EXCHANGE   : the wet sand disperses water DOWN into the drier native loam (the resolved
                        analogue of the funnel-factor exchange the sub-grid version will model).
We assess the channel's WATER BUDGET before / during / after the storm:
    d(sand storage) = cum_rain - surface_outflow - sand_conveyance_outflow - dispersion_into_loam
with dispersion_into_loam = d(loam storage) (the loam is closed except at the sand interface).

STAGE 1 SCOPE: 2-D longitudinal section ALONG the channel (x = along-channel to the outlet at x=Lx,
z = depth). Shows intake + conveyance + exchange + the budget. The 3-D cross-slope interception
("either side / downslope of the channel") is the Stage-1b extension. This RESOLVED run is the ground
truth the SUB-GRID funnel-factor version (Stage 2) must reproduce on a coarser mesh.

Run (WSL): conda activate pids-fem && cd .../forward-model && PYTHONPATH=. \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scratch/m4_sand_channel_demo.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

COMM = MPI.COMM_WORLD

LX, LZ = 10.0, 1.5            # along-channel length [m], depth [m]
NX, NZ = 50, 15              # 0.2 m x 0.1 m cells
S0 = 0.03                     # bed fall toward the outlet at x = LX
Z_IFACE = LZ - 0.4            # coarse-sand channel = the top 0.4 m (z >= Z_IFACE); native loam below
STORM_DUR, T_END = 0.5, 2.0   # rain for 12 h, recession to 2 d
N_OUT = 40

SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)   # coarse well-sorted
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)    # native soil

# Two regimes show the SAME channel doing BOTH jobs, set by the antecedent soil state:
#  DRY   subsoil has capacity -> the channel DISPERSES (water sinks into the soil);
#  WET   subsoil is saturated -> no room below, so the channel CONVEYS laterally to the outlet.
REGIMES = {
    "DRY  (disperse)": dict(ic=lambda x: -0.5 + 0.0 * x[0], rain_rate=0.10, He=Z_IFACE),
    "WET  (convey)":   dict(ic=lambda x: (Z_IFACE) - x[1], rain_rate=0.10, He=Z_IFACE - 0.25),
}


class SandOverLoam:
    """Coarse-sand channel (top Z_IFACE..LZ) over native loam -- the UFL surface CoupledProblem consumes."""

    def __init__(self, mesh, z_iface):
        self.sand, self.loam, self.z_iface = SAND, LOAM, z_iface
        zc = ufl.SpatialCoordinate(mesh)[mesh.geometry.dim - 1]
        self._in_sand = ufl.ge(zc, z_iface)
        self.Ks, self.theta_r, self.theta_s = SAND.Ks, SAND.theta_r, SAND.theta_s

    def theta_ufl(self, psi):
        return ufl.conditional(self._in_sand, self.sand.theta_ufl(psi), self.loam.theta_ufl(psi))

    def K_ufl(self, psi):
        return ufl.conditional(self._in_sand, self.sand.K_ufl(psi), self.loam.K_ufl(psi))

    def kirchhoff_ufl(self, a, b):
        return self.sand.kirchhoff_ufl(a, b)        # the surface infiltration leg lives in the sand top

    def theta_of(self, psi, z):
        return np.where(z >= self.z_iface, self.sand.theta(psi), self.loam.theta(psi))


def run_regime(label, ic, rain_rate, He):
    msh = dmesh.create_rectangle(COMM, [[0.0, 0.0], [LX, LZ]], [NX, NZ], dmesh.CellType.triangle)
    soil = SandOverLoam(msh, Z_IFACE)
    prob = CoupledProblem(msh, soil, n_man=0.05)
    prob.set_initial_condition(ic, d_value=0.0)
    prob.set_topography(lambda x: S0 * (LX - x[0]))
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], LX), slope=S0)                 # surface edge outlet
    # the sand CHANNEL discharges its conveyed water at x=LX (the lateral outlet): a GHB on the sand
    # part of the downslope face, draining to the channel-invert head He.
    prob.add_drainage_bc(lambda x: np.isclose(x[0], LX) & (x[1] >= Z_IFACE - 1e-9),
                         conductance=2.0, external_head=He)
    RAIN = rain_rate

    # zone-storage forms (the channel budget): theta integrated over the sand vs the loam
    z = ufl.SpatialCoordinate(msh)[1]
    dxq = ufl.dx(metadata={"quadrature_degree": prob._quad_degree})
    chi_sand = ufl.conditional(ufl.ge(z, Z_IFACE), 1.0, 0.0)
    sand_form = fem.form(soil.theta_ufl(prob.psi) * chi_sand * dxq)
    loam_form = fem.form(soil.theta_ufl(prob.psi) * (1.0 - chi_sand) * dxq)
    sand_store = lambda: COMM.allreduce(fem.assemble_scalar(sand_form), op=MPI.SUM)
    loam_store = lambda: COMM.allreduce(fem.assemble_scalar(loam_form), op=MPI.SUM)
    top_len = LX

    out_t = np.linspace(0.0, T_END, N_OUT)
    rec = {k: np.zeros(N_OUT) for k in
           ("rain_rate", "cum_rain", "cum_surf_out", "cum_conv_out", "sand_store", "loam_store",
            "surf_water", "q_surf_out", "q_conv_out", "max_pond", "mbe")}
    sand0, loam0 = sand_store(), loam_store()
    w0 = prob.total_water()

    def snap(k, cum_rain):
        rec["rain_rate"][k] = float(rain.value)
        rec["cum_rain"][k] = cum_rain
        rec["cum_surf_out"][k] = prob.cum_outflow
        rec["cum_conv_out"][k] = prob.cum_drainage          # the sand-zone GHB = channel conveyance out
        rec["sand_store"][k] = sand_store()
        rec["loam_store"][k] = loam_store()
        rec["surf_water"][k] = prob.surface_water()
        rec["q_surf_out"][k] = prob.last_outflow
        rec["q_conv_out"][k] = prob.last_drainage
        rec["max_pond"][k] = float(prob.d.x.array.max())
        expected = cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust
        rec["mbe"][k] = abs((prob.total_water() - w0) - expected) / (cum_rain + 1e-30)

    snap(0, 0.0)
    dt, t, cum_rain, k_out, nsteps = 1e-3, 0.0, 0.0, 1, 0
    print(f"\n=== [{label}] {LX}x{LZ} m, sand z>={Z_IFACE} (Ks={SAND.Ks}) over loam "
          f"(Ks={LOAM.Ks}); storm {RAIN} m/d x {STORM_DUR} d -> {T_END} d ===", flush=True)
    while t < T_END - 1e-12:
        h = min(dt, T_END - t)
        if t < STORM_DUR - 1e-12 and t + h > STORM_DUR:
            h = STORM_DUR - t
        if k_out < N_OUT and t + h > out_t[k_out]:
            h = out_t[k_out] - t
        rain.value = RAIN if t < STORM_DUR - 1e-12 else 0.0
        conv, it = prob.step(h)
        if conv:
            cum_rain += float(rain.value) * top_len * h
            t += h; nsteps += 1
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 0.05)
            while k_out < N_OUT and t >= out_t[k_out] - 1e-12:
                snap(k_out, cum_rain); k_out += 1
            if nsteps % 25 == 0:
                print(f"  t={t:.4f} dt={dt:.2e} it={it} cum_rain={cum_rain:.4f} "
                      f"surf_out={prob.cum_outflow:.4f} conv_out={prob.cum_drainage:.4f} "
                      f"max_d={prob.d.x.array.max()*1e3:.1f}mm", flush=True)
        else:
            dt *= 0.5
            if dt < 1e-9:
                print(f"  !! DT COLLAPSE at t={t:.6f}", flush=True); return
    while k_out < N_OUT:
        snap(k_out, cum_rain); k_out += 1

    # ---- the channel water budget, before / during / after ----
    k_storm = int(np.argmin(np.abs(out_t - STORM_DUR)))
    bal = (prob.total_water() - w0) - (cum_rain - prob.cum_outflow - prob.cum_drainage
                                       + prob.clip_mass_adjust)
    print(f"\n=== CHANNEL WATER BUDGET (sand zone, per unit width) ===", flush=True)
    print(f"  global mass balance |resid|/cum_rain = {abs(bal)/(cum_rain+1e-30):.2e} "
          f"(max over run {rec['mbe'].max():.2e})", flush=True)
    print(f"  cum rain in            : {cum_rain:.5f} m^2", flush=True)
    print(f"  cum surface outflow    : {prob.cum_outflow:.5f} m^2", flush=True)
    print(f"  cum conveyance outflow : {prob.cum_drainage:.5f} m^2  (sand-zone GHB at x={LX})", flush=True)
    print(f"  d(sand storage)        : {rec['sand_store'][-1]-sand0:+.5f} m^2", flush=True)
    print(f"  d(loam storage) = dispersion into soil : {rec['loam_store'][-1]-loam0:+.5f} m^2", flush=True)
    print(f"  peak ponded depth      : {rec['max_pond'].max()*1e3:.2f} mm", flush=True)
    print(f"\n  {'phase':<8} {'t [d]':>6} {'rain':>6} {'q_surf':>8} {'q_conv':>8} "
          f"{'sandΔ':>8} {'loamΔ(disp)':>11}", flush=True)
    for tag, k in (("start", 0), ("end-storm", k_storm), ("end", N_OUT - 1)):
        print(f"  {tag:<8} {out_t[k]:>6.2f} {rec['rain_rate'][k]:>6.3f} {rec['q_surf_out'][k]:>8.4f} "
              f"{rec['q_conv_out'][k]:>8.4f} {rec['sand_store'][k]-sand0:>+8.4f} "
              f"{rec['loam_store'][k]-loam0:>+11.4f}", flush=True)
    split = dict(conveyed=prob.cum_drainage / (cum_rain + 1e-30),
                 dispersed=(rec["loam_store"][-1] - loam0) / (cum_rain + 1e-30),
                 surface=prob.cum_outflow / (cum_rain + 1e-30),
                 stored=(rec["sand_store"][-1] - sand0) / (cum_rain + 1e-30))
    print(f"\n  storm-input split: conveyed {split['conveyed']:.0%}, dispersed-to-soil "
          f"{split['dispersed']:.0%}, surface-shed {split['surface']:.0%}, "
          f"stored-in-channel {split['stored']:.0%}", flush=True)
    fn = "scratch/m4_sand_channel_demo__" + label.split()[0].lower() + ".npz"
    np.savez(fn, out_t=out_t, sand0=sand0, loam0=loam0, **rec)
    return dict(label=label, cum_rain=cum_rain, mbe=float(rec["mbe"].max()), **split)


if __name__ == "__main__":
    results = [run_regime(lab, **cfg) for lab, cfg in REGIMES.items()]
    print(f"\n{'='*78}\nUNIFIED CHANNEL -- one structure, two jobs (storm-input split):")
    print(f"  {'regime':<16} {'conveyed':>9} {'dispersed':>10} {'surface':>8} "
          f"{'stored':>7} {'max|mbe|':>9}")
    for r in results:
        print(f"  {r['label']:<16} {r['conveyed']:>8.0%} {r['dispersed']:>9.0%} "
              f"{r['surface']:>7.0%} {r['stored']:>6.0%} {r['mbe']:>9.1e}")
    print("="*78)
