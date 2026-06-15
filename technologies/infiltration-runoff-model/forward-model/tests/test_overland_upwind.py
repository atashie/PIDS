"""Tier-1 sanity for the O1 upwind-mobility edge-flux overland solver (P1 Part B).

``UpwindOverlandProblem`` (``pids_forward/physics/overland_upwind.py``) is the standalone
Module-2 spike of a monotone, well-balanced overland scheme: a two-point edge flux on the
P1-dof edge graph with upstream-weighted Manning mobility, driven by a custom PETSc SNES
(finite-difference Jacobian + LU) because the upwind selector is not UFL-expressible. It does
NOT touch the validated galerkin ``OverlandProblem`` (which stays the MMS/regression reference).

The scheme differences the surface HEAD ``H = z_b + d`` (not depth), so a uniform-head still
pond is at rest STRUCTURALLY (every edge flux ``Q_e = T_e M(d_up)(H_i - H_j) = 0``), which is
the decisive well-balancedness gate below. Telescoping edge signs (``+`` for the i-row, ``-``
for the j-row) make discrete mass conservation structural. Units: length m, time day, Manning
``n_man`` in SI s.m^{-1/3}. Per governance/claude-sanity-check-routine.md each behaviour is
pinned test-first against a closed-form / structural reference.

B1 scope: 1-D core, lake-at-rest exact, a non-flat-H-drives-flow sanity guard, SNES health.
(Positivity-without-limiter, conservation, kinematic limb = B2; selector width = B3; 2-D = B4.)
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland_upwind import UpwindOverlandProblem

N_MAN = 0.03  # Manning roughness (SI s.m^{-1/3}); smooth-ish overland plane


def test_lake_at_rest_is_held_exactly_1d():
    """A still pond over a SLOPING bed must stay at rest (well-balanced gate).

    Initial surface head H = z_b + d is uniform, so every edge head difference H_i - H_j is zero
    -- to ROUNDOFF in z_b + d (the head sums two interpolated fields, so ~1e-16, not bit-exact),
    INDEPENDENT of eps_S / eps_H (the scheme differences H, not d, so the gate cannot be tuned to
    pass; the eps_H-independence is pinned separately below). The conveyance amplifies that ~1e-16
    head-drop into a tiny residual flux, which the Newton solve drives below tolerance -- so the
    DEPTH holds to machine precision and no spurious flux runs down the bed slope.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 20, [0.0, 1.0])
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)

    # Sloping bed z_b = 0.5 - 0.3 x; flat water surface H = 0.7 => d = 0.2 + 0.3 x > 0.
    prob.set_topography(lambda x: 0.5 - 0.3 * x[0])
    still = lambda x: 0.2 + 0.3 * x[0]  # d = H - z_b, all strictly positive
    prob.set_initial_condition(still)
    d0 = prob.d.x.array.copy()
    w0 = prob.total_water()

    converged, iters = prob.step(dt=0.1)
    assert converged

    # EXACT lake-at-rest: depth unchanged to machine precision (the headline gate).
    assert float(np.max(np.abs(prob.d.x.array - d0))) < 1e-14
    # Mass holds and depth stays strictly non-negative.
    assert abs(prob.total_water() - w0) <= 1e-14 * max(1.0, abs(w0))
    assert prob.d.x.array.min() >= 0.0


def test_lake_at_rest_independent_of_eps_H_1d():
    """Well-balancedness is STRUCTURAL: lake-at-rest holds for any eps_H (cannot be tuned).

    Decision 4 / design note: because uniform H makes every H_i - H_j = 0, the flux is zero
    regardless of the smoothed-upwind width eps_H (the selector multiplies a zero head drop).
    Re-running the gate at a wildly different eps_H proves the pass is structural, not a tuned
    coincidence of the default width.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 20, [0.0, 1.0])
    prob = UpwindOverlandProblem(msh, n_man=N_MAN, eps_H=1.0)  # 1000x the default width
    assert prob.eps_H == 1.0  # recorded as an attribute (Decision 4)

    prob.set_topography(lambda x: 0.5 - 0.3 * x[0])
    still = lambda x: 0.2 + 0.3 * x[0]
    prob.set_initial_condition(still)
    d0 = prob.d.x.array.copy()

    converged, _ = prob.step(dt=0.1)
    assert converged
    assert float(np.max(np.abs(prob.d.x.array - d0))) < 1e-14


def test_nonflat_head_drives_flow_1d():
    """SANITY (anti-degenerate): a NON-uniform surface head actually moves water.

    The lake-at-rest gate alone is also passed by a do-nothing solver, so we must show the
    scheme is live: a flat bed with a non-flat depth bump (=> non-uniform head, real head
    differences) must redistribute -- the bump draws down at the peak while the flanks rise --
    in a closed domain. Guards against shipping an "at-rest-for-everything" class. Also pins the
    headline CONSERVATION claim on a genuinely DYNAMIC step (the lake test conserves trivially):
    a real telescoping flux network must leave total water invariant -- and this assertion is the
    tripwire for a future reason-4 regression that books an unbalanced stall (the deferred-to-B2
    stall_accept_fnorm gate). B2 extends this to the full positivity/kinematic suite.
    """
    L = 10.0
    msh = dmesh.create_interval(MPI.COMM_WORLD, 40, [0.0, L])
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)  # flat bed (z_b = 0)
    prob.set_initial_condition(lambda x: 0.1 + 0.05 * np.exp(-((x[0] - 5.0) / 1.0) ** 2))
    d0 = prob.d.x.array.copy()
    w0 = prob.total_water()

    coords = prob.V.tabulate_dof_coordinates()[:, 0]
    peak = int(np.argmin(np.abs(coords - 5.0)))

    converged, iters = prob.step(dt=1e-3)
    assert converged

    moved = float(np.max(np.abs(prob.d.x.array - d0)))
    assert moved > 1e-6, f"non-flat head failed to drive any flow (max move {moved:.2e})"
    # the bump must DRAW DOWN at the peak (diffusive spreading), not just jitter.
    assert prob.d.x.array[peak] < d0[peak]
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.d.x.array))
    # CLOSED-DOMAIN CONSERVATION on a dynamic step: telescoping edge fluxes leave total water
    # invariant (no rain, no outflow). Tripwire for an unbalanced reason-4 stall being booked.
    assert abs(prob.total_water() - w0) <= 1e-13 * max(1.0, abs(w0))


def test_snes_converges_cleanly_1d():
    """The custom SNES reports a genuine (positive) converged reason on a normal step."""
    L = 10.0
    msh = dmesh.create_interval(MPI.COMM_WORLD, 40, [0.0, L])
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)
    prob.set_initial_condition(lambda x: 0.1 + 0.05 * np.exp(-((x[0] - 5.0) / 1.0) ** 2))

    converged, iters = prob.step(dt=1e-3)
    assert converged
    assert prob.last_reason > 0  # SNES CONVERGED_* (not a DIVERGED_/iterating reason)
    assert iters >= 1            # a non-trivial state took real Newton steps
