"""Tier-2 sanity (overland): synthetic-forcing runs -- typical AND end-member extremes.

Per governance/claude-sanity-check-routine.md Tier 2: force the module with synthetic
hyetographs (typical SE-Piedmont storm; 100-yr sub-hourly Atlas-14-style burst; dry plane;
steep vs flat), checking that under every scenario the run stays mass-conservative and
physically plausible AND shows the expected qualitative behaviour (runoff onset, a rising
limb then recession, velocities that rise under intense forcing).

Global mass balance: with backward Euler + the conservative positivity limiter, summing the
weak form against v=1 gives W^{n+1}-W^n = dt*(r*L - outflow)^{n+1} exactly, so accumulating
end-of-step rain and outflow volumes closes Delta storage = cum_rain - cum_outflow to solver
precision -- a rigorous conservation gate even with time-varying forcing and an open outlet.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland import OverlandProblem

N_MAN = 0.05  # vegetated SE-Piedmont overland


def _run_phase(prob, rain_const, r_value, t_phase, L, accum, dt0=1e-4, dt_max=5e-3):
    """March one constant-rain phase with adaptive BE, accumulating rain/outflow volumes.

    accum = [cum_rain, cum_outflow] (m^2 per unit width), updated in place with end-of-step
    rates (BE), so Delta storage must equal cum_rain - cum_outflow to solver precision.
    """
    t, dt = 0.0, dt0
    while t < t_phase - 1e-12:
        h = min(dt, t_phase - t)
        rain_const.value = r_value  # constant within the phase (BE: same at the new level)
        converged, iters = prob.step(h)
        if converged:
            accum[0] += r_value * L * h          # rain volume in over [t, t+h]
            accum[1] += prob.last_outflow * h    # outflow volume out (residual-consistent, pre-limiter)
            t += h
            dt = min(dt * (1.5 if iters <= 3 else 0.7 if iters >= 8 else 1.0), dt_max)
        else:
            dt *= 0.5
            if dt < 1e-9:
                raise RuntimeError(f"dt collapse at t={t:.4g}")
    return dt


def test_typical_storm_hydrograph_open_hillslope():
    """A typical storm on an open hillslope: rising limb during rain, recession after.

    Mass balance closes (Delta storage = cum_rain - cum_outflow), runoff is generated, the
    outlet hydrograph peaks during the storm and recedes once rain stops, depth stays >= 0.
    """
    L, S0 = 100.0, 0.02
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, L])
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    prob.set_initial_condition(lambda x: 0.0 * x[0])  # dry antecedent
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    accum = [0.0, 0.0]
    w0 = prob.total_water()
    _run_phase(prob, rain, 0.30, 0.06, L, accum)   # ~0.3 m/day storm for ~1.4 h
    q_peak = prob.outflow_rate()
    _run_phase(prob, rain, 0.0, 0.10, L, accum)    # rain off -> recession
    q_end = prob.outflow_rate()

    # global mass balance closes.
    dW = prob.total_water() - w0
    assert abs(dW - (accum[0] - accum[1])) <= 1e-6 * accum[0]
    # qualitative hydrograph: runoff generated, peak during storm, recession afterward.
    assert q_peak > 0.0
    assert q_end < q_peak
    assert accum[1] > 0.0  # net runoff left the catchment
    # plausibility.
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.d.x.array))


def test_100yr_subhourly_burst_stable_and_conservative():
    """A 100-yr sub-hourly Atlas-14-style intense burst stays stable + conservative.

    Exercises adaptive stepping + the wet/dry front under extreme intensity; checks the run
    completes (no solver collapse), mass balance closes, depth >= 0, and the peak Manning
    velocity rises well above the antecedent (the erosion-threshold-relevant response).
    """
    L, S0 = 100.0, 0.03
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, L])
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    accum = [0.0, 0.0]
    w0 = prob.total_water()
    u0 = float(np.max(np.abs(prob.velocity().x.array)))
    _run_phase(prob, rain, 4.0, 0.01, L, accum, dt0=1e-5, dt_max=1e-3)  # ~4 m/day burst ~14 min
    u_peak = float(np.max(np.abs(prob.velocity().x.array)))
    _run_phase(prob, rain, 0.0, 0.04, L, accum, dt0=1e-5, dt_max=2e-3)

    dW = prob.total_water() - w0
    assert abs(dW - (accum[0] - accum[1])) <= 1e-6 * accum[0]
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.d.x.array))
    assert u_peak > 10.0 * max(u0, 1e-9)  # velocity rises sharply under the burst
    assert accum[1] > 0.0  # the burst produced runoff


def test_dry_plane_no_spurious_runoff():
    """A dry hillslope with zero rain (open outlet) generates no spurious water or outflow."""
    L, S0 = 100.0, 0.02
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, L])
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    accum = [0.0, 0.0]
    _run_phase(prob, rain, 0.0, 0.05, L, accum)
    assert prob.total_water() == pytest.approx(0.0, abs=1e-12)
    assert prob.outflow_rate() == pytest.approx(0.0, abs=1e-12)
    assert prob.d.x.array.max() <= 1e-12


@pytest.mark.parametrize("S0", [0.05, 1e-3])
def test_steep_and_flat_both_conserve_and_plausible(S0):
    """The same storm on a STEEP and a near-FLAT reach both stay conservative + plausible.

    The flat reach is the diffusion-wave stiffness stress case (design C.6, the documented
    trigger to consider local-inertial); here we require both regimes to remain mass-
    conservative, non-negative, and finite under a typical storm + recession.
    """
    L = 100.0
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, L])
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=max(S0, 1e-4))

    accum = [0.0, 0.0]
    w0 = prob.total_water()
    _run_phase(prob, rain, 0.20, 0.05, L, accum)
    _run_phase(prob, rain, 0.0, 0.08, L, accum)

    dW = prob.total_water() - w0
    assert abs(dW - (accum[0] - accum[1])) <= 1e-6 * accum[0]
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.d.x.array))
