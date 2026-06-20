"""P3 Part B: a permanent COUPLED CONVERGENT-FLOW Tier-2 fixture with an embedded PIDS drainage element.

The convergent-flow workstream (P0-P3) fixed the coupled UPWIND overland solver for the regime where
surface water concentrates along convergence lines -- exactly where PIDS networks / inlets install
(memory: "convergent overland flow = CORE PIDS regime"). This fixture is the permanent Tier-2
regression for that fix.

FRAMING (Arik 2026-06-19): a "swale" is NOT a PIDS feature -- it is the convergent topographic SETTING
(a graded low line, standard site grading) with VARIABLE geometry. The PIDS element embedded here is the
coupled-integrated DUAL-DRAIN -- the signed-off illustration scratch/m4_hillslope_drain_dual.py (commit
63145e2), extended from a 2-D galerkin hillslope to this 3-D UPWIND convergent geometry:
  - add_surface_inlet  : a grate on the convergence line capturing the concentrated ponded run-on;
  - add_interior_drain : an embedded OUTFLOW-ONLY tile drain along the convergence line ON the clay
                         interface, tapping the perched mound the concentrated infiltration builds.
add_interior_drain is the closest coupled-integrated analogue of an embedded 1-D PIDS conveyance; the
full bidirectional WellIndexExchange feature-with-storage is NOT wired into CoupledProblem (a separate
integration task). ALL geometry is PARAMETERIZED (variable): domain, slopes, convergence floor width,
drain band, inlet footprint, conductances.

Run (pids-fem, from forward-model):  PYTHONPATH=. python scratch/_p3_convergent_fixture.py   # smoke
"""
from __future__ import annotations

import time

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from pids_forward.physics.coupling import CoupledProblem
from scratch.m4_hillslope_drain_clay import TwoLayerSoil


def convergent_zb(W, Lx, Ly, Xc, SX, SY):
    """Convergent graded topography: a flat-bottomed low line of floor width W at x=Xc running down y
    (valley slope SY), side slopes SX converging INTO it. z_b = SY*(Ly-y) + SX*max(|x-Xc|-W/2, 0).
    The SETTING where overland flow concentrates (the upwind scheme routes it without the galerkin
    sawtooth); NOT a PIDS feature. W>0 must span >=~6 cells across the floor to resolve (else the
    measure-zero kink artifact)."""
    half = 0.5 * W

    def zb(x):
        return SY * (Ly - x[1]) + SX * np.maximum(np.abs(x[0] - Xc) - half, 0.0)

    return zb


def make_convergent_dualdrain(nx, ny, nz, *, Lx=48.0, Ly=30.0, Lz=2.0, SX=0.03, SY=0.015, W=12.0,
                              z_iface=1.0, z_wt=1.05, n_man=0.05, inlet_halfwidth=2.0,
                              inlet_coeff=500.0, drain_halfwidth=2.0, drain_band=None,
                              drain_cond=30.0, drain_head=None, drain_eps=1e-3, soil=None,
                              petsc_options=None):
    """Build the coupled convergent-flow dual-drain fixture (overland_scheme='upwind'). Returns
    (prob, info). The host is a 3-D loam-over-clay box (TwoLayerSoil, interface at z=z_iface); the
    convergent topography z_b is a field on the flat top facet (realization A). The embedded PIDS
    dual-drain shares the convergence-line footprint x in [Xc +/- halfwidth]: a SURFACE grate inlet on
    the ponded depth + an INTERIOR tile drain on the clay interface (one loam cell-row). Outlet = the
    convergence floor at the downstream end (y=Ly). Everything is parameterized (variable geometry)."""
    Xc = Lx / 2.0
    tol = 1e-6
    dz = Lz / nz
    if drain_band is None:
        drain_band = dz                     # one loam cell-row sitting ON the clay interface
    if drain_head is None:
        drain_head = z_iface                # pipe invert at the interface (outflow-only above it)

    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, Ly, Lz]], [nx, ny, nz])
    if soil is None:
        soil = TwoLayerSoil(msh, z_iface=z_iface)   # loam over tight clay (perches; 3-D-capable)
    prob = CoupledProblem(msh, soil, n_man=n_man, overland_scheme="upwind",
                          petsc_options=petsc_options)
    prob.set_initial_condition(lambda x: z_wt - x[2], d_value=0.0)   # water table at z=z_wt (in clay)
    prob.set_topography(convergent_zb(W, Lx, Ly, Xc, SX, SY))
    # outlet = the convergence FLOOR at the downstream end (the low line IS the drainage path); the
    # side slopes drain into the convergence and exit through it -> q_w = Q_eq/W is the right scale.
    prob.add_outflow_bc(
        lambda x: np.isclose(x[1], Ly) & (np.abs(x[0] - Xc) <= 0.5 * W + tol), slope=SY)
    # the embedded PIDS dual-drain along the convergence line (variable geometry):
    drain = prob.add_interior_drain(
        locator=lambda x: (np.abs(x[0] - Xc) <= drain_halfwidth + tol)
        & (x[2] >= z_iface - tol) & (x[2] <= z_iface + drain_band + tol),
        conductance_density=drain_cond, drain_head=drain_head, eps_act=drain_eps)
    inlet = prob.add_surface_inlet(
        locator=lambda x: np.abs(x[0] - Xc) <= inlet_halfwidth + tol, intake_coeff=inlet_coeff)

    info = dict(Lx=Lx, Ly=Ly, Lz=Lz, Xc=Xc, SX=SX, SY=SY, W=W, nx=nx, ny=ny, nz=nz, z_iface=z_iface,
                z_wt=z_wt, n_man=n_man, inlet_halfwidth=inlet_halfwidth, inlet_coeff=inlet_coeff,
                drain_halfwidth=drain_halfwidth, drain_band=drain_band, drain_cond=drain_cond,
                drain_head=drain_head, top_area=top_area(prob))
    return prob, info, drain, inlet


def top_area(prob):
    """Plan area of the top surface (int 1 ds_top) -- rain * top_area = the catchment inflow rate."""
    one = fem.Constant(prob.mesh, 1.0)
    return prob.mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * prob._ds_top)), op=MPI.SUM)


def balance_residual(prob, w0, cum_rain):
    """|Delta_total - (cum_rain - cum_outflow - cum_drainage + clip_mass_adjust)| -- the structural
    coupled mass balance incl. ALL sinks (outlet + surface inlet + interior drain). Upwind: clip=0."""
    dW = prob.total_water() - w0
    expected = cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust
    return abs(dW - expected)


def drive_phase(prob, rain_const, rate, t_phase, top_a, accum, *, dt0=5e-4, dt_max=1e-2,
                grow_at=3, shrink_at=8, dt_min=1e-9):
    """March one constant-rain phase with adaptive backward-Euler (mirrors test_coupling_3d_tier2's
    _run_phase). Accumulates cum_rain into accum[0]; cum_outflow/cum_drainage are engine-owned. Returns
    (dt, n_acc, n_rej, max_iter)."""
    t, dt = 0.0, dt0
    n_acc = n_rej = max_iter = 0
    while t < t_phase - 1e-12:
        h = min(dt, t_phase - t)
        rain_const.value = rate
        converged, iters = prob.step(h)
        if converged:
            accum[0] += rate * top_a * h
            t += h
            n_acc += 1
            max_iter = max(max_iter, iters)
            dt = min(dt * (1.5 if iters <= grow_at else 0.7 if iters >= shrink_at else 1.0), dt_max)
        else:
            n_rej += 1
            dt *= 0.5
            if dt < dt_min:
                raise RuntimeError(f"convergent dual-drain: dt collapse at t={t:.4g} in phase")
    return dt, n_acc, n_rej, max_iter


def _smoke():
    import os
    print("=" * 78)
    print("CONVERGENT DUAL-DRAIN fixture smoke (upwind overland + surface inlet + interface tile)")
    print("=" * 78)
    nx, ny, nz = 24, 15, 4
    z_wt = float(os.environ.get("Z_WT", "1.05"))  # water table just above the drain (z_iface=1.0)
    single = os.environ.get("SINGLE_LOAM", "0") == "1"
    soil = None
    if single:                                    # isolate the loam/clay-contrast stiffness
        from pids_forward.physics.constitutive import VanGenuchten
        soil = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)
    prob, info, drain, inlet = make_convergent_dualdrain(nx, ny, nz, z_wt=z_wt, soil=soil)
    print(f"  soil={'single LOAM' if single else 'loam/clay'}  "
          f"mesh {nx}x{ny}x{nz} on {info['Lx']}x{info['Ly']}x{info['Lz']} m  W={info['W']} "
          f"(floor cells {info['W']/info['Lx']*nx:.1f})  ell_c={prob.ell_c:.3f}  z_wt={z_wt}  "
          f"top_area={info['top_area']:.1f}")
    rain = prob.add_rain(0.0)
    w0 = prob.total_water()
    print(f"  initial drain rate (groundwater at z_wt={z_wt}): {prob.last_sinks['interior_drain']} "
          f"(drainage_rate={prob.drainage_rate():.4f})")
    accum = [0.0]
    t0 = time.time()
    storm_t = float(os.environ.get("STORM_T", "0.02"))
    # typical storm (> Ks=0.25 -> infiltration-excess runoff + convergence ponding) then recession
    _, a1, r1, it1 = drive_phase(prob, rain, 0.35, storm_t, info["top_area"], accum)
    q_inlet_peak = prob.last_sinks["surface_inlet"][0]
    q_drain_peak = prob.last_sinks["interior_drain"][0]
    d_peak = prob.surface_depth()
    _, a2, r2, it2 = drive_phase(prob, rain, 0.0, 0.01, info["top_area"], accum)
    run_s = time.time() - t0

    bal = balance_residual(prob, w0, accum[0])
    cum_inlet = prob.cum_sinks["surface_inlet"][0]
    cum_drain = prob.cum_sinks["interior_drain"][0]
    print(f"  storm+recession in {run_s:.1f}s ({a1+a2} acc / {r1+r2} rej, max_it {max(it1,it2)})")
    print(f"  conservation |bal|/cum_rain = {bal/max(accum[0],1e-30):.2e}  (gate <1e-3)")
    print(f"  cum_rain={accum[0]:.4f}  cum_outflow={prob.cum_outflow:.4f}  cum_drainage={prob.cum_drainage:.4f}")
    print(f"  surface INLET : cum={cum_inlet:.5f}  peak rate={q_inlet_peak:.4f}   (storm-peak ponded {1e3*d_peak:.2f} mm)")
    print(f"  interior DRAIN: cum={cum_drain:.5f}  peak rate={q_drain_peak:.4f}")
    print(f"  d_min={1e3*prob.d.x.array.min():+.3f} mm  max_clip_seen={1e3*prob.max_clip_seen:.3f} mm "
          f"clip_mass_adjust={prob.clip_mass_adjust:.2e}")
    print(f"  finite: psi={np.all(np.isfinite(prob.psi.x.array))} d={np.all(np.isfinite(prob.d.x.array))}")


if __name__ == "__main__":
    _smoke()
