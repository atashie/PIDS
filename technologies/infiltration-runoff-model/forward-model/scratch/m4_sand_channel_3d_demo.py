"""DEMO (2026-06-22 unified-feature redesign, Stage 1b = the 3-D RESOLVED truth): a localized coarse
SAND conveyance channel cut into a low-permeability CLAY hillslope, through an intense rainfall event.

Arik 2026-06-22: show all three PIDS couplings together on a realistic 3-D hillslope -- intense rain
PONDS on the low-K clay and RUNS OFF; a localized sand channel (a contour swale) INTERCEPTS that runoff,
CONVEYS it laterally to a side outlet, and DISPERSES some into the soil. Assess the channel's in/outflows
before / during / after the storm.

GEOMETRY (3-D box, x=downslope, y=along-channel, z=depth up):
  * main bed slope S0 toward the toe outlet at x=LX -> runoff flows +x;
  * a coarse-SAND contour channel at x=X_CH (band |x-X_CH|<W_CH, top sand z>=Z_SAND_BASE) running along y,
    sitting in a shallow SWALE (a topographic depression at X_CH whose invert falls along y toward y=0),
    so intercepted runoff is diverted to the CHANNEL OUTLET at y=0;
  * native CLAY everywhere else (low Ks -> intense rain ponds and runs off).
The channel is RESOLVED (a few cells wide) -> all three couplings emerge from CoupledProblem physics:
  (1) SURFACE INTAKE: runoff pooling in the swale infiltrates the high-K sand;
  (2) CONVEYANCE     : sand carries it subsurface (Darcy) to the y=0 GHB outlet + swale surface flow to
                       the y=0 surface outlet;
  (3) SOIL DISPERSION: the wet sand sheds into the surrounding/under-lying clay.

BUDGET (per the engine's per-outlet forms): toe surface outflow (escaped past the channel) vs channel
capture (channel-outlet surface outflow + sand-GHB conveyance + dispersion into clay).

This RESOLVED 3-D run is the ground truth the SUB-GRID funnel-factor version (Stage 2) must reproduce on
a coarse mesh. Numerics: overland_scheme="auto" (-> upwind in 3-D serial; the swale is convergent, where
galerkin sawtooths); soil.Ks set to the sand (the channel GHB needs it; clay infiltration flux is the
spatially-varying kirchhoff leg, unaffected -- only the NCP conditioning scale uses the scalar Ks).

Run (WSL): conda activate pids-fem && cd .../forward-model && PYTHONPATH=. \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scratch/m4_sand_channel_3d_demo.py
"""
from __future__ import annotations

import time
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

COMM = MPI.COMM_WORLD

LX, LY, LZ = 8.0, 5.0, 1.0          # downslope, along-channel, depth [m]
NX, NY, NZ = 20, 12, 5              # 0.40 x 0.42 x 0.20 m cells
S0 = 0.04                           # main bed slope toward the toe x=LX
X_CH, W_CH = 4.0, 0.6              # contour channel at mid-slope, half-width (band ~3 cells)
Z_SAND_BASE = LZ - 0.4              # sand fills the channel band in the top 0.4 m; clay below + around
D_CH, SY = 0.30, 0.06               # swale invert depth at y=LY, extra fall per m toward the y=0 outlet
X_BERM, W_BERM, B_H = 5.3, 0.45, 0.30  # a downslope BERM that traps the runoff in the swale (a real
#                                        diversion channel needs it -- without it sheet flow escapes)
PSI_I = -0.30                       # antecedent: unsaturated clay (room to disperse AND to convey)
RAIN, STORM_DUR, T_END = 0.15, 0.3, 1.2     # intense relative to clay Ks -> ponds + runs off
N_OUT = 36

SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)   # coarse well-sorted
CLAY = VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.048)   # low-K native


def in_band(x):                      # the channel x-band (column footprint), independent of z
    return ufl.lt(abs(x[0] - X_CH), W_CH)


class ClaySandChannel:
    """Native CLAY with a localized coarse-SAND channel (band |x-X_CH|<W_CH, top z>=Z_SAND_BASE)."""

    def __init__(self, mesh):
        self.sand, self.clay = SAND, CLAY
        xx = ufl.SpatialCoordinate(mesh)
        self._in_sand = ufl.And(ufl.lt(abs(xx[0] - X_CH), W_CH), ufl.ge(xx[2], Z_SAND_BASE))
        self._in_col = ufl.lt(abs(xx[0] - X_CH), W_CH)       # surface infiltration leg = sand in the band
        self.Ks = SAND.Ks                                    # the channel GHB needs the sand Ks
        self.theta_r, self.theta_s = CLAY.theta_r, CLAY.theta_s

    def theta_ufl(self, psi):
        return ufl.conditional(self._in_sand, self.sand.theta_ufl(psi), self.clay.theta_ufl(psi))

    def K_ufl(self, psi):
        return ufl.conditional(self._in_sand, self.sand.K_ufl(psi), self.clay.K_ufl(psi))

    def kirchhoff_ufl(self, a, b):
        return ufl.conditional(self._in_col, self.sand.kirchhoff_ufl(a, b),
                               self.clay.kirchhoff_ufl(a, b))

    def theta_of(self, psi, x):          # numpy, for zone storage post-processing
        in_sand = (np.abs(x[:, 0] - X_CH) < W_CH) & (x[:, 2] >= Z_SAND_BASE)
        return np.where(in_sand, self.sand.theta(psi), self.clay.theta(psi))


def topo(x):
    z_main = S0 * (LX - x[0])                                # main downslope grade toward the toe
    swale = (D_CH + SY * (LY - x[1])) * np.exp(-(((x[0] - X_CH) / W_CH) ** 2))   # channel, falls to y=0
    berm = B_H * np.exp(-(((x[0] - X_BERM) / W_BERM) ** 2))  # downslope berm: traps + diverts the runoff
    return z_main - swale + berm


def run(scheme):
    t0 = time.perf_counter()
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [NX, NY, NZ])
    soil = ClaySandChannel(msh)
    prob = CoupledProblem(msh, soil, overland_scheme=scheme, n_man=0.05)
    prob.set_initial_condition(lambda x: PSI_I + 0.0 * x[0], d_value=0.0)
    prob.set_topography(topo)
    rain = prob.add_rain(0.0)
    tol = 1e-6
    # outlet ORDER fixes the per-outlet split: [0]=toe (x=LX), [1]=channel surface (y=0 in the band)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], LX), slope=S0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0) & (np.abs(x[0] - X_CH) < W_CH + tol), slope=SY)
    # channel SUBSURFACE conveyance: sand-zone GHB on the y=0 face draining to the channel invert
    prob.add_drainage_bc(
        lambda x: np.isclose(x[1], 0.0) & (np.abs(x[0] - X_CH) < W_CH + tol) & (x[2] >= Z_SAND_BASE - tol),
        conductance=2.0, external_head=Z_SAND_BASE - 0.1)
    ndof = prob.Vpsi.dofmap.index_map.size_local
    print(f"[build {time.perf_counter()-t0:.1f}s scheme={prob._effective_overland_scheme}] "
          f"{NX}x{NY}x{NZ} box, {3*ndof} DOFs, ell_c={prob.ell_c:.4f}", flush=True)

    # per-zone storage forms
    xx = ufl.SpatialCoordinate(msh)
    dxq = ufl.dx(metadata={"quadrature_degree": prob._quad_degree})
    chi_sand = ufl.conditional(ufl.And(ufl.lt(abs(xx[0] - X_CH), W_CH), ufl.ge(xx[2], Z_SAND_BASE)),
                               1.0, 0.0)
    sand_form = fem.form(soil.theta_ufl(prob.psi) * chi_sand * dxq)
    clay_form = fem.form(soil.theta_ufl(prob.psi) * (1.0 - chi_sand) * dxq)
    sand_store = lambda: COMM.allreduce(fem.assemble_scalar(sand_form), op=MPI.SUM)
    clay_store = lambda: COMM.allreduce(fem.assemble_scalar(clay_form), op=MPI.SUM)
    top_area = COMM.allreduce(fem.assemble_scalar(
        fem.form(fem.Constant(msh, 1.0) * prob._ds_top)), op=MPI.SUM)
    toe_form, chan_form = prob._outflow_forms[0], prob._outflow_forms[1]
    rate = lambda f: COMM.allreduce(fem.assemble_scalar(f), op=MPI.SUM)

    out_t = np.linspace(0.0, T_END, N_OUT)
    rec = {k: np.zeros(N_OUT) for k in ("rain_rate", "cum_rain", "cum_toe", "cum_chan_surf",
           "cum_conv", "sand_store", "clay_store", "max_pond")}
    sand0, clay0, w0 = sand_store(), clay_store(), prob.total_water()
    cum_toe = cum_chan = 0.0

    def snap(k, cum_rain):
        rec["rain_rate"][k] = float(rain.value); rec["cum_rain"][k] = cum_rain
        rec["cum_toe"][k] = cum_toe; rec["cum_chan_surf"][k] = cum_chan
        rec["cum_conv"][k] = prob.cum_drainage
        rec["sand_store"][k] = sand_store(); rec["clay_store"][k] = clay_store()
        rec["max_pond"][k] = float(prob.d.x.array.max())

    snap(0, 0.0)
    dt, t, cum_rain, k_out, nsteps = 5e-4, 0.0, 0.0, 1, 0
    print(f"=== 3-D SAND CHANNEL in CLAY: storm {RAIN} m/d x {STORM_DUR} d -> {T_END} d "
          f"(clay Ks={CLAY.Ks}, sand Ks={SAND.Ks}) ===", flush=True)
    while t < T_END - 1e-12:
        h = min(dt, T_END - t)
        if t < STORM_DUR - 1e-12 and t + h > STORM_DUR:
            h = STORM_DUR - t
        if k_out < N_OUT and t + h > out_t[k_out]:
            h = out_t[k_out] - t
        rain.value = RAIN if t < STORM_DUR - 1e-12 else 0.0
        conv, it = prob.step(h)
        if conv:
            cum_rain += float(rain.value) * top_area * h
            cum_toe += rate(toe_form) * h
            cum_chan += rate(chan_form) * h
            t += h; nsteps += 1
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 0.03)
            while k_out < N_OUT and t >= out_t[k_out] - 1e-12:
                snap(k_out, cum_rain); k_out += 1
            if nsteps % 25 == 0:
                print(f"  t={t:.4f} dt={dt:.2e} it={it} rain={float(rain.value):.2f} "
                      f"toe={cum_toe:.4f} chan_surf={cum_chan:.4f} conv={prob.cum_drainage:.4f} "
                      f"max_d={prob.d.x.array.max()*1e3:.1f}mm", flush=True)
        else:
            dt *= 0.5
            if dt < 1e-9:
                print(f"  !! DT COLLAPSE at t={t:.6f} (scheme={prob._effective_overland_scheme})",
                      flush=True)
                return None
    while k_out < N_OUT:
        snap(k_out, cum_rain); k_out += 1

    bal = (prob.total_water() - w0) - (cum_rain - prob.cum_outflow - prob.cum_drainage
                                       + prob.clip_mass_adjust)
    disp = rec["clay_store"][-1] - clay0
    capture = cum_chan + prob.cum_drainage + disp
    print(f"\n=== CHANNEL IN/OUTFLOWS (per the storm) ===", flush=True)
    print(f"  wall {time.perf_counter()-t0:.0f}s, {nsteps} steps; global |bal|/cum_rain = "
          f"{abs(bal)/(cum_rain+1e-30):.2e}", flush=True)
    print(f"  cum rain in                 : {cum_rain:.5f} m^3", flush=True)
    print(f"  ESCAPED to toe (x=LX)       : {cum_toe:.5f} m^3  surface runoff past the channel", flush=True)
    print(f"  CHANNEL captured (total)    : {capture:.5f} m^3", flush=True)
    print(f"    - conveyed: surface (y=0) : {cum_chan:.5f} m^3", flush=True)
    print(f"    - conveyed: subsurface GHB: {prob.cum_drainage:.5f} m^3", flush=True)
    print(f"    - dispersed into clay     : {disp:.5f} m^3  (clay storage gain)", flush=True)
    print(f"  d(sand storage)             : {rec['sand_store'][-1]-sand0:+.5f} m^3", flush=True)
    print(f"  peak ponded depth           : {rec['max_pond'].max()*1e3:.2f} mm", flush=True)
    k_storm = int(np.argmin(np.abs(out_t - STORM_DUR)))
    print(f"\n  {'phase':<10} {'t[d]':>5} {'rain':>5} {'cum_toe':>8} {'cum_chan':>9} "
          f"{'cum_conv':>9} {'sandΔ':>8} {'clayΔ(disp)':>11}", flush=True)
    for tag, k in (("before", 0), ("end-storm", k_storm), ("after", N_OUT - 1)):
        print(f"  {tag:<10} {out_t[k]:>5.2f} {rec['rain_rate'][k]:>5.2f} {rec['cum_toe'][k]:>8.4f} "
              f"{rec['cum_chan_surf'][k]:>9.4f} {rec['cum_conv'][k]:>9.4f} "
              f"{rec['sand_store'][k]-sand0:>+8.4f} {rec['clay_store'][k]-clay0:>+11.4f}", flush=True)
    runoff = cum_toe + cum_chan + prob.cum_drainage
    if runoff > 1e-9:
        print(f"\n  of the {runoff:.4f} m^3 that became runoff/conveyance: "
              f"channel intercepted {(cum_chan+prob.cum_drainage)/runoff:.0%}, "
              f"escaped to toe {cum_toe/runoff:.0%}", flush=True)
    np.savez("scratch/m4_sand_channel_3d_demo.npz", out_t=out_t, sand0=sand0, clay0=clay0, **rec)
    print("\nSaved timeline -> scratch/m4_sand_channel_3d_demo.npz", flush=True)
    return prob


if __name__ == "__main__":
    res = run("auto")
    if res is None:
        print("\n-> auto/upwind stalled; retrying galerkin", flush=True)
        res = run("galerkin")
    if res is None:
        raise SystemExit("both overland schemes stalled")
