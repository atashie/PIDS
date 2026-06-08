"""Tier-2 sanity (Module 3 coupling, 3-D): synthetic-forcing runs on a 3-D HILLSLOPE.

Per governance/claude-sanity-check-routine.md Tier 2: drive the FULL coupled system (3-D Richards +
sorptive Kirchhoff infiltration NCP + lateral Manning overland routing + the codim-2 ridge edge
outlet + a subsurface drainage GHB) with synthetic hyetographs -- a typical storm and a 100-yr-style
intense burst -- and require that under every scenario the run stays STABLE (no solver collapse,
finite, d>=0, theta in [theta_r, theta_s]) and MASS-CONSERVATIVE, while showing the expected
qualitative behaviour (runoff onset, a rising limb then recession).

Global mass balance (backward Euler + conservative limiter, all rates recorded end-of-step):
    Delta_total = cum_rain - cum_outflow - cum_drainage + clip_mass_adjust
to solver precision -- a rigorous gate even with time-varying forcing, an open edge outlet, AND a
draining base. cum_outflow/cum_drainage/clip_mass_adjust are accumulated inside CoupledProblem.step;
we accumulate cum_rain here (constant within each phase).

Meshes are SMALL and phases SHORT (the diffusion-wave overland is stiff and 3-D dof count is the perf
constraint); the quadrature cap keeps each step ~0.1 s.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)  # m/day


def _top_area(prob):
    one = fem.Constant(prob.mesh, 1.0)
    return prob.mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * prob._ds_top)), op=MPI.SUM)


def _run_phase(prob, rain_const, r_value, t_phase, top_area, accum, dt0=5e-5, dt_max=1e-3):
    """March one constant-rain phase with adaptive BE, accumulating cum_rain in accum[0].

    cum_outflow/cum_drainage/clip_mass_adjust are tracked inside prob.step(); we only need cum_rain.
    """
    t, dt = 0.0, dt0
    while t < t_phase - 1e-12:
        h = min(dt, t_phase - t)
        rain_const.value = r_value   # constant within the phase (BE: same at the new level)
        converged, iters = prob.step(h)
        if converged:
            accum[0] += r_value * top_area * h
            t += h
            dt = min(dt * (1.5 if iters <= 3 else 0.7 if iters >= 8 else 1.0), dt_max)
        else:
            dt *= 0.5
            if dt < 1e-9:
                raise RuntimeError(f"dt collapse at t={t:.4g} (solver could not converge)")
    return dt


def _theta(prob):
    return prob.soil.theta(prob.psi.x.array)


def _balance_resid(prob, w0, cum_rain):
    """|Delta_total - (cum_rain - cum_outflow - cum_drainage + clip_mass_adjust)|."""
    dW = prob.total_water() - w0
    expected = cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust
    return abs(dW - expected)


def _make_hillslope(nx=10, ny=4, nz=5, L=5.0, S0=0.05, psi0=-1.0):
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, 1.0, 1.0]], [nx, ny, nz])
    prob = CoupledProblem(msh, LOAM, n_man=0.05)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: S0 * (L - x[0]))   # slope down toward the x=L outlet edge
    return prob, L, S0


def test_3d_typical_storm_full_system_conserves():
    """A TYPICAL storm on a 3-D hillslope with the FULL coupled system (infiltration NCP + lateral
    routing + ridge edge outlet + a draining base GHB): rising runoff during rain, recession after,
    and the global books close as Delta_total = cum_rain - cum_outflow - cum_drainage + clip_adjust.
    """
    prob, L, S0 = _make_hillslope(psi0=-1.0)
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)              # open downstream EDGE
    prob.add_drainage_bc(lambda x: np.isclose(x[2], 0.0),                     # deep percolation (base)
                         conductance=0.2, external_head=-1.5)
    top_area = _top_area(prob)
    accum = [0.0]
    w0 = prob.total_water()

    _run_phase(prob, rain, 0.30, 0.02, top_area, accum)   # ~0.3 m/day storm (> Ks -> runoff)
    q_peak = prob.outflow_rate()
    _run_phase(prob, rain, 0.0, 0.03, top_area, accum)    # rain off -> recession
    q_end = prob.outflow_rate()

    # global mass balance closes (the full 3-sink balance).
    assert _balance_resid(prob, w0, accum[0]) <= 1e-6 * accum[0]
    # qualitative: runoff generated, hydrograph peaks during the storm and recedes afterward.
    assert prob.cum_outflow > 0.0
    assert q_peak > 0.0 and q_end < q_peak
    assert prob.cum_drainage != 0.0   # the base GHB is active (deep percolation)
    # plausibility.
    th = _theta(prob)
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.psi.x.array)) and np.all(np.isfinite(prob.d.x.array))
    assert th.min() >= LOAM.theta_r - 1e-12 and th.max() <= LOAM.theta_s + 1e-9
    assert abs(prob.clip_mass_adjust) < 1e-9   # well-behaved limiter (no degenerate drying)


def test_3d_100yr_burst_stable_and_conservative():
    """A 100-yr-style intense burst (~8x Ks) on the 3-D hillslope stays STABLE (no solver collapse)
    and CONSERVATIVE, generates runoff, and ponds. Exercises adaptive stepping + the wet/dry front +
    the ridge outlet under extreme intensity. (No base drain here -- isolates the overland-burst
    stability; the typical-storm test covers the full 3-sink balance.)
    """
    prob, L, S0 = _make_hillslope(psi0=-0.8)
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)
    top_area = _top_area(prob)
    accum = [0.0]
    w0 = prob.total_water()

    _run_phase(prob, rain, 2.0, 0.008, top_area, accum, dt0=1e-5, dt_max=5e-4)   # ~8x Ks burst
    d_peak = prob.surface_depth()
    _run_phase(prob, rain, 0.0, 0.02, top_area, accum, dt0=1e-5, dt_max=1e-3)    # recession

    assert _balance_resid(prob, w0, accum[0]) <= 1e-6 * accum[0]   # conserves under the burst
    assert prob.cum_outflow > 0.0                                   # the burst produced runoff
    assert d_peak > 1e-3                                            # genuine ponding during the burst
    th = _theta(prob)
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.psi.x.array)) and np.all(np.isfinite(prob.d.x.array))
    assert th.min() >= LOAM.theta_r - 1e-12 and th.max() <= LOAM.theta_s + 1e-9


def test_3d_dry_hillslope_negligible_spurious_water():
    """A dry 3-D hillslope under zero rain (open edge outlet) generates only a NEGLIGIBLE, bounded,
    conservatively-sourced surface film -- no spurious runoff or mass creation.

    Unlike the STANDALONE overland (no NCP -> exactly 0), the coupled smoothed Fischer-Burmeister NCP
    leaves a tiny eps_ncp-scale film d ~ eps^2/(tau_c*g) (~sub-micron mean depth here) even on dry
    soil. It is BOUNDED and CONVERGENT (Phase-1 convergence test: d_top -> 6.6e-9 at finer mesh) and
    sign-paired with the soil (NOT created), so total water is conserved and q_out ~ d^{5/3} vanishes.
    """
    prob, L, S0 = _make_hillslope(psi0=-2.0)
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)
    top_area = _top_area(prob)
    accum = [0.0]
    w0 = prob.total_water()
    _run_phase(prob, rain, 0.0, 0.02, top_area, accum)

    assert prob.surface_water() < 1e-5, f"non-negligible spurious film ({prob.surface_water():.2e})"
    assert prob.outflow_rate() < 1e-4   # q_out ~ d^{5/3} ~ (sub-micron)^{5/3} -> vanishing runoff
    # no rain -> conserved: Delta_total == -cum_outflow + clip_mass_adjust (the film came from soil).
    dW = prob.total_water() - w0
    assert abs(dW - (-prob.cum_outflow + prob.clip_mass_adjust)) < 1e-9
    assert np.all(np.isfinite(prob.psi.x.array))
