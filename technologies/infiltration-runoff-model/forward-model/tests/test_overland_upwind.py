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
B2 adds the MONOTONE-SCHEME PAYOFF: positivity WITHOUT a limiter on a wet/dry-front advance
(where the galerkin ``OverlandProblem`` must clip ~mm-cm undershoots with ``_enforce_positivity``),
multi-step machine-tight conservation, and the 1-D kinematic rising-limb hydrograph through a
Manning normal-depth ``add_outflow_bc`` outlet (the analytic reference shared with the galerkin
``test_kinematic_wave_plane_hydrograph_1d``). (Selector width = B3; 2-D = B4.)
"""
import numpy as np
import pytest
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


# ============================ B2 gates ====================================
# B1's class has no advance(); B2 marches step() in a small adaptive loop (mirroring
# OverlandProblem.advance's grow/cut logic) so the front scenarios run to a fixed t_end at a
# stable dt. Returns (t_reached, nsteps, min_d_over_run, reasons), tracking the run-minimum depth
# (the positivity headline) and the SNES reason history (to audit reason-4 stalls -- see
# test_front_advance_positive_without_limiter_1d's docstring on why they are honest here).
def _march(prob, t_end, dt0=1e-4, dt_max=5e-3, grow=1.5, cut=0.5, dt_min=1e-9,
           target_low=3, target_high=8, shrink=0.7, max_steps=100000):
    t, dt, nsteps = 0.0, dt0, 0
    min_d = float(prob.d.x.array.min())
    reasons = []
    while t < t_end - 1e-12:
        assert nsteps < max_steps, f"_march exceeded max_steps at t={t:.4g}"
        h = min(dt, t_end - t)
        converged, iters = prob.step(h)
        reasons.append(prob.last_reason)
        if converged:
            t += h
            nsteps += 1
            min_d = min(min_d, float(prob.d.x.array.min()))
            if iters <= target_low:
                dt = min(dt * grow, dt_max)
            elif iters >= target_high:
                dt *= shrink
        else:
            dt = h * cut
            assert dt >= dt_min, f"_march: dt {dt:.2e} < dt_min at t={t:.4g} (not converging)"
    return t, nsteps, min_d, reasons


def test_front_advance_positive_without_limiter_1d():
    """A steep slump advances a wet/dry FRONT into dry ground keeping d >= 0 -- NO limiter.

    THE monotone-scheme headline. The IDENTICAL scenario (5% slope, sharp Gaussian, 20 m, 60
    cells, marched to t=0.03 day) makes the galerkin ``OverlandProblem`` undershoot to NEGATIVE
    depths at the wet/dry front -- its ``_enforce_positivity`` limiter engages with
    ``max_clip_seen ~= 2.2e-3`` m (measured 2026-06-15; deleting that limiter makes galerkin
    ``d.min()`` go negative). The upwind monotone scheme holds ``d >= -1e-12`` STRUCTURALLY
    (measured run-min ~6e-34 here): ``UpwindOverlandProblem`` has NO positivity limiter (assert
    below that no clip/rescale machinery exists on it), so a pass here proves monotonicity, not a
    clip. The scenario is adversarial: the front must genuinely ADVANCE downslope into the dry
    region (the regime that drives the galerkin undershoot), not sit frozen.

    CONDITIONALITY (stated here, NOT hidden -- B2 decision point 2): this strict ``d >= 0``
    without a limiter holds because the selector is SHARP relative to this STEEP front's head-drop
    (drop >> ``eps_H=1e-3``). On a MILD-slope front (~2%, head-drop comparable to ``eps_H``) at the
    DEFAULT ``eps_H`` the smoothed (tanh) selector admits a small CENTERED flux and the scheme
    shows a small front UNDERSHOOT (~1.5-1.6 mm, measured 2026-06-15 on the 2%-slope sharp mound --
    comparable to the galerkin RAW pre-clip undershoot, which galerkin only HIDES by post-clipping).
    Conservation stays machine-tight through that undershoot (~2e-16). So this is NOT a claim of
    unconditional ``d >= 0`` at the default width: it is the clean STEEP regime; the B3
    selector-width (``eps_H``) task removes the mild-front undershoot. The default ``eps_H`` is left
    UNCHANGED at B2 (tightening it pre-empts B3 and could affect kinematic accuracy).

    Reason-4 audit: this run books many reason-4 (SNORM-stagnation) steps, which B1 accepts without
    a residual-floor gate (deferred Part A). They are HONEST floor exits here, not dirty stalls: the
    reason-4 ``||F||`` stays <= ~6.2e-8 (ratio ||F||/total-water-scale ~9e-8, ~7 orders below the
    problem scale, measured 2026-06-15), and the machine-tight conservation assert below is the
    independent tripwire -- a dirty unbalanced stall would leak mass, which this does not. So no
    floor gate is needed to keep this gate honest (decision recorded in the B2 report).
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 60, [0.0, 20.0])
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: 0.05 * (20.0 - x[0]))  # 5% slope -> stiff front
    prob.set_initial_condition(lambda x: 0.25 * np.exp(-((x[0] - 7.0) / 1.5) ** 2))
    w0 = prob.total_water()

    coords = prob.V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(coords)
    xs = coords[order]
    d_init = prob.d.x.array[order].copy()
    wet = 0.02 * 0.25  # wetted-front threshold (2% of the peak)
    front_pre = xs[d_init > wet].max()

    # The class carries NO positivity limiter (the monotone-scheme point): assert the clipping
    # machinery the galerkin path uses is simply absent, so a positive min(d) cannot be a clip.
    assert not hasattr(prob, "_enforce_positivity")
    assert not hasattr(prob, "max_clip_seen")

    t, nsteps, min_d, reasons = _march(prob, t_end=0.03)
    assert t == pytest.approx(0.03, abs=1e-9)
    assert nsteps > 0

    # (1) HEADLINE: depth never went negative across the WHOLE run, with no limiter.
    assert min_d >= -1e-12, f"upwind undershot to {min_d:.3e} without a limiter"
    assert prob.d.x.array.min() >= -1e-12  # final state non-negative too
    assert np.all(np.isfinite(prob.d.x.array))
    # (2) ADVERSARIAL: the wetted front actually advanced into the dry downslope region
    # (the wet/dry-front regime that makes galerkin undershoot) -- not a frozen state.
    ds = prob.d.x.array[order]
    front_post = xs[ds > wet].max()
    assert front_post > front_pre + 1.0, f"front did not advance: {front_pre:.2f} -> {front_post:.2f}"
    # (3) closed-domain conservation stays machine-tight despite the reason-4 stalls (the
    # tripwire that these stalls are honest floor exits, not unbalanced dirty stalls).
    assert abs(prob.total_water() - w0) <= 1e-12 * max(1.0, abs(w0))


def test_closed_domain_conserves_multistep_1d():
    """A closed domain (no rain, no outflow) conserves total_water to ~1e-13 over many steps.

    Extends B1's single-step ``test_nonflat_head_drives_flow_1d`` conservation assert to a
    MULTI-STEP march: a mound slumping on a sloped closed reach must leave the lumped water
    budget invariant step after step (telescoping edge signs => structural discrete conservation
    at every residual-converged root). A genuinely dynamic redistribution (the mound spreads and
    shifts downhill) makes this a real conservation check, not a static state.

    Scenario choice (honest, tied to the documented positivity CONDITIONALITY -- module docstring
    + ``test_front_advance...``): conservation is machine-tight REGARDLESS of any front undershoot
    (the telescoping flux network balances at every residual-converged root -- verified directly:
    a 2%-slope SHARP mound undershoots to ~-1.6 mm at the default ``eps_H`` yet still drifts only
    ~2e-16). To keep THIS gate a clean POSITIVE-and-conservative check per the B2 decision (point
    4), the mound here sits on a WET BASE and is broad/gentle (1% slope, sigma 3) so no wet/dry
    front forms and ``min(d)`` stays comfortably positive (~0.027 m) at the default width -- the
    mild-front undershoot is NOT hidden, it is characterized in ``test_front_advance...``'s mild
    note and removed by B3; here we simply isolate conservation from it.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 40, [0.0, 20.0])  # closed (no-flux) reach
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: 0.01 * (20.0 - x[0]))  # gentle 1% slope toward x=20
    # broad mound on a wet base -> dynamic redistribution but no wet/dry front (stays positive).
    prob.set_initial_condition(lambda x: 0.10 + 0.10 * np.exp(-((x[0] - 10.0) / 3.0) ** 2))
    w0 = prob.total_water()
    d_before = prob.d.x.array.copy()

    t, nsteps, min_d, _ = _march(prob, t_end=0.03, dt0=1e-3)
    assert nsteps >= 5  # genuinely multi-step (not one giant step)

    # multi-step machine-tight conservation: closed system, no source/sink.
    assert abs(prob.total_water() - w0) <= 1e-13 * max(1.0, abs(w0))
    # the mound genuinely redistributed (dynamic, not frozen).
    assert np.max(np.abs(prob.d.x.array - d_before)) > 1e-3
    # CLEAN gate: stays strictly positive at the default eps_H (no wet/dry front here), finite.
    assert min_d >= -1e-12
    assert np.all(np.isfinite(prob.d.x.array))


def test_kinematic_wave_plane_hydrograph_1d():
    """Steady rain on a tilted plane + a normal-depth OUTFLOW outlet -> kinematic rising limb.

    Reuses the galerkin ``test_kinematic_wave_plane_hydrograph_1d`` reference (same L, S0, n, r):
    diffusion-wave reduces to kinematic wave on a long plane, so at steady state the outlet
    discharge equals the total rainfall input ``q_out = r*L`` (mass balance) and the outlet depth
    matches the Manning normal depth ``d_eq = (r*L*n/sqrt(S0))^{3/5}`` (SI units). This is the
    decisive test of the conveyance law AND the day<->second unit conversion, and it REQUIRES the
    B2 ``add_outflow_bc`` (a no-flux outlet would dam the reach). The upwind scheme matches the
    analytic equilibrium tightly (well within its O(h) accuracy -- measured ratios ~1.00 at nc=50).
    """
    L, S0, n, r = 100.0, 0.01, 0.10, 0.2  # m, slope[-], Manning[s/m^1/3], rain[m/day]
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, L])
    prob = UpwindOverlandProblem(msh, n_man=n)
    prob.set_topography(lambda x: S0 * (L - x[0]))    # slopes down toward the x=L outlet
    prob.set_initial_condition(lambda x: 0.0 * x[0])  # dry start
    prob.add_rain(r)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)  # normal-depth free drain

    # March, sampling the outlet hydrograph each 0.025 day to verify the RISING LIMB shape +
    # time-to-equilibrium (~5x t_c over 0.25 day), not just the final steady state.
    hydro = []
    t, dt, t_end = 0.0, 1e-4, 0.25
    next_rec = 0.025
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        converged, iters = prob.step(h)
        assert converged, f"kinematic march failed to converge at t={t:.4g} (reason {prob.last_reason})"
        t += h
        if t >= next_rec - 1e-12:
            hydro.append(prob.outflow_rate())
            next_rec += 0.025
        if iters <= 3:
            dt = min(dt * 1.5, 5e-3)
        elif iters >= 8:
            dt *= 0.7
    hydro = np.array(hydro)

    # rising limb: monotonically non-decreasing toward equilibrium (small jitter tolerance).
    assert np.all(np.diff(hydro) >= -1e-2 * (r * L)), f"hydrograph not monotone rising: {hydro}"
    # time-to-equilibrium: outflow climbs to >=95% of r*L by the end.
    assert hydro[-1] >= 0.95 * (r * L)

    # (1) mass balance at steady state: outflow per unit width ~ total rainfall r*L (O(h) tight).
    assert prob.outflow_rate() == pytest.approx(r * L, rel=0.05)
    # (2) outlet depth ~ kinematic normal depth d_eq (validates conveyance + the day<->s units).
    r_si = r / 86400.0
    d_eq = (r_si * L * n / np.sqrt(S0)) ** (3.0 / 5.0)
    coords = prob.V.tabulate_dof_coordinates()[:, 0]
    outlet = int(np.argmin(np.abs(coords - L)))
    assert prob.d.x.array[outlet] == pytest.approx(d_eq, rel=0.15)
    # (3) depth grows downslope (increasing contributing area); positive sheet flow, no negatives.
    order = np.argsort(coords)
    d_sorted = prob.d.x.array[order]
    assert d_sorted[-1] > d_sorted[len(d_sorted) // 2] > 1e-6
    assert prob.d.x.array.min() >= -1e-12


def test_outflow_discharge_absolute_magnitude_1d():
    """Pin the ABSOLUTE day<->second conversion in add_outflow_bc independently (mirrors the
    galerkin ``test_outflow_discharge_absolute_magnitude_1d``).

    At a known uniform depth on a known slope, ``outflow_rate()`` must equal the Manning
    normal-depth value in m^2/day with a HARD-CODED 86400 (not importing the module constant),
    so a wrong SECONDS_PER_DAY factor is caught directly. In 1-D the outlet is a single node, so
    the outlet discharge is the nodal q_out.
    """
    L, S0, n, d0 = 50.0, 0.01, 0.05, 0.02
    msh = dmesh.create_interval(MPI.COMM_WORLD, 20, [0.0, L])
    prob = UpwindOverlandProblem(msh, n_man=n)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    prob.set_initial_condition(lambda x: d0 + 0.0 * x[0])
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    expected = 86400.0 * (1.0 / n) * d0 ** (5.0 / 3.0) * np.sqrt(S0)  # m^2/day, hard-coded factor
    assert prob.outflow_rate() == pytest.approx(expected, rel=1e-6)


def test_outflow_bc_rejects_nonpositive_slope_1d():
    """add_outflow_bc(slope <= 0) must raise (mirrors the galerkin guard).

    slope=0 would silently turn the outlet into a no-flux wall (damming the reach); slope<0
    injects sqrt(<0)=NaN into the residual. Both are caller errors -> rejected up front.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 20, [0.0, 50.0])
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)
    with pytest.raises(ValueError):
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 50.0), slope=0.0)
    with pytest.raises(ValueError):
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 50.0), slope=-0.01)
