"""Tier-1 sanity (overland diffusion-wave solver): analytical / conservation checks.

The solver is the dimension-agnostic 2-D diffusion-wave approximation (design Sec. C):

    d(d)/dt + div(q) = sources,   q = -K_s(d) grad(H_s),   H_s = z_b + d
    K_s(d) = SECONDS_PER_DAY * d^{5/3} / ( n_man |grad H_s|^{1/2} + eps_S )

primary variable = ponding depth ``d`` >= 0 (enforced by PETSc SNES-VI), backward
Euler in time. Slope enters ONLY through the known topography field ``z_b`` (grad
z_b), so the UFL is identical for 1-D / 2-D / 3-D-top meshes. Units: length m, time
day, Manning ``n_man`` in SI s.m^{-1/3} (the SECONDS_PER_DAY factor converts the SI
Manning conveyance to m^2/day). Per governance/claude-sanity-check-routine.md each
behavior is pinned test-first by a closed-form / conservation reference.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland import OverlandProblem

N_MAN = 0.03  # Manning roughness (SI s.m^{-1/3}); smooth-ish overland plane


def test_lake_at_rest_is_held_1d():
    """A still pond over a SLOPING bed must stay still (well-balanced / lake-at-rest).

    Initial water surface H_s = z_b + d is uniform, so grad(H_s) = 0 and the Manning
    flux -K_s(d) grad(H_s) is identically zero: depth must not change, no spurious
    flux down the bed slope. This is the decisive well-balancedness test -- a naive
    discretization that does not difference (z_b + d) consistently spuriously drains
    the pond downhill. With P1 d and z_b in the SAME space, grad(z_b + d) is exact, so
    the flux vanishes to machine precision. Also confirms depth stays >= 0 (VI bound).
    """
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 20)
    prob = OverlandProblem(msh, n_man=N_MAN)

    # Sloping bed z_b = 0.5 - 0.3 x; flat water surface H_s = 0.7 => d = 0.2 + 0.3 x > 0.
    prob.set_topography(lambda x: 0.5 - 0.3 * x[0])
    still = lambda x: 0.2 + 0.3 * x[0]  # d = H_s - z_b, all positive
    prob.set_initial_condition(still)
    w0 = prob.total_water()

    for _ in range(5):
        converged, iters = prob.step(dt=0.1)
        assert converged
        assert 0 <= iters <= 50

    # Depth field unchanged to machine precision (no spurious downhill drainage).
    assert prob.max_abs_error(still) < 1e-9
    # Mass conserved and depth strictly non-negative (VI bound respected).
    assert abs(prob.total_water() - w0) / w0 < 1e-9
    assert prob.d.x.array.min() >= -1e-12


def test_dry_plane_stays_dry_1d():
    """A dry tilted plane with no rain stays dry: d >= 0, no spurious wetting.

    The depth-positivity plausibility invariant in its simplest limit: with every node
    dry (d = 0) and no forcing there is nothing to move; the SNES-VI keeps the whole
    domain at the lower bound d = 0 and must converge. A solver that wrongly produced
    negative depths (post-solve clamping, or a sign error in the degenerate diffusion)
    would fail the >= 0 check here.
    """
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 20)
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: 0.5 - 0.3 * x[0])  # tilted bed
    prob.set_initial_condition(lambda x: 0.0 * x[0])  # bone dry

    for _ in range(3):
        converged, iters = prob.step(dt=0.1)
        assert converged
        assert 0 <= iters <= 50

    assert prob.d.x.array.min() >= -1e-12  # never negative (VI bound)
    assert prob.d.x.array.max() <= 1e-12   # no spurious wetting
    assert prob.total_water() == pytest.approx(0.0, abs=1e-12)


def test_closed_slump_conserves_water_1d():
    """A water mound slumping downslope in a CLOSED domain conserves total water.

    No-flux boundaries + no rain => integrating the weak form against the constant test
    function v=1 gives d/dt (integral d dx) = 0, so the lumped storage integral is
    invariant. Backward Euler + the mass-lumped storage close it to solver tolerance
    while the mound visibly redistributes (spreads and shifts downhill), so this is a
    non-trivial dynamic conservation check, not a static state. Depth stays >= 0.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 40, [0.0, 20.0])  # 20 m hillslope
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: 0.02 * (20.0 - x[0]))  # 2% slope down toward x = 20
    prob.set_initial_condition(lambda x: 0.2 * np.exp(-((x[0] - 10.0) / 2.0) ** 2))  # mound
    w0 = prob.total_water()
    d_before = prob.d.x.array.copy()

    nsteps = prob.advance(t_end=0.03, dt=1e-3, dt_max=5e-3)
    assert nsteps > 0

    # total water conserved in the closed system (no flux, no source) to the gate.
    assert abs(prob.total_water() - w0) / w0 < 1e-6
    # the mound genuinely redistributed (slumped), not a frozen state.
    assert np.max(np.abs(prob.d.x.array - d_before)) > 1e-3
    # plausibility: non-negative depth, finite everywhere.
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.d.x.array))


def test_positivity_limiter_conserves_while_clipping_1d():
    """When the slump undershoots, the conservative limiter clips d>=0 AND keeps the budget.

    On a steeper slump the smooth Newton solve drives d to small (~mm-cm) NEGATIVE depths
    at the wet/dry front (a known degenerate-diffusion artifact). The post-step limiter
    must remove those (final d >= 0) while preserving total water to machine precision --
    the whole point of a *conservative* clip vs a naive clamp. We assert the limiter
    actually engaged (max clip > 1e-4), so this is a real regression guard: deleting the
    limiter makes ``d.min()`` go negative and fails the test.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 60, [0.0, 20.0])
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: 0.05 * (20.0 - x[0]))  # 5% slope -> stronger front
    prob.set_initial_condition(lambda x: 0.25 * np.exp(-((x[0] - 7.0) / 1.5) ** 2))
    w0 = prob.total_water()

    prob.advance(t_end=0.03, dt=1e-4, dt_max=5e-3)

    assert prob.max_clip_seen > 1e-4, (
        f"limiter never engaged (max_clip={prob.max_clip_seen:.2e}); "
        "scenario did not undershoot, so it does not guard the limiter"
    )
    assert prob.d.x.array.min() >= -1e-12  # final depth non-negative (clip worked)
    assert abs(prob.total_water() - w0) / w0 < 1e-6  # budget preserved despite clipping
    assert np.all(np.isfinite(prob.d.x.array))
