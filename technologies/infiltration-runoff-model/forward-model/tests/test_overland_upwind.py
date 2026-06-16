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
    without a limiter holds because the selector is SHARP relative to the front head-drop (drop >>
    ``eps_H``). The undershoot is governed by ``eps_H`` vs the front head-drop, NOT by the slope
    alone. B3 (``scratch/_upwind_selector_probe.py``, 2026-06-15) swept ``eps_H`` and FIXED it at
    1e-3 on exactly this trade: at the chosen ``eps_H=1e-3`` this steep 5% front is strictly
    positive (run-min ~6e-34), and the MILD 2% regime (the B5 valley) is GEOMETRY-DEPENDENT
    SUB-MILLIMETER -- 0 .. ~0.9 mm (controller adjudication sweep 2026-06-16, 24 mild-2%
    geometries: worst -0.93 mm on a sharp mound; the canonical mound is strictly positive), vs
    ~2.4 mm at the looser ``eps_H=1e-2``. (Two earlier single-number claims were superseded: B2's
    "~1.5-1.6 mm at default" was high, B3's "<=0.36 mm" was low -- the honest figure is the
    geometry-dependent 0 .. ~0.9 mm range; see the module docstring "Positivity (B2)" /
    "Regularization (Decision 4)".) The undershoot is SUB-MM (vs the galerkin limiter's cm-scale
    clip it replaces) and CONSERVATION stays machine-tight regardless (~2e-16). Sharpening below
    1e-3 buys NO accuracy and costs Newton robustness (B3 table), so ``eps_H=1e-3`` is KEPT.

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
    even when a LOOSE-width 2% front undershoots a few mm, the budget still drifts only ~2e-16). To
    keep THIS gate a clean POSITIVE-and-conservative check per the B2 decision (point 4), the mound
    here sits on a WET BASE and is broad/gentle (1% slope, sigma 3) so no wet/dry front forms and
    ``min(d)`` stays comfortably positive (~0.027 m) at the default width. (B3 FIXED ``eps_H=1e-3``;
    on a mild 2% front that width gives a geometry-dependent SUB-MM undershoot, 0 .. ~0.9 mm --
    module docstring "Regularization (Decision 4)". This gate isolates conservation from positivity
    either way: it is machine-tight regardless of any residual undershoot.)
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


# ============================ B4: 2-D extension ============================
# The 1-D class (B1-B3) is extended to 2-D triangle meshes (create_rectangle). The ONLY
# dimension-aware parts are the edge graph (built from mesh.topology edge<->vertex, mapped into
# the P1-dof index space), the two-point transmissibility T_e (the COTANGENT / FV dual-mesh
# weight = the negated P1 stiffness off-diagonal, the standard monotone coefficient), and the
# nodal area A_i (the lumped P1 mass int phi_i dx, sum = domain area). The node-residual
# telescoping, smoothed-upwind selector, Manning mobility, SNES (FD-Jacobian + LU) and eps_H/eps_S
# are IDENTICAL to 1-D. Monotonicity needs T_e >= 0 (the M-matrix property), which holds on the
# structured box V (right-triangle split: DIAGONAL/hypotenuse edges get cot(90 deg)=0, AXIS edges
# get the acute cotangents > 0; verified by probe) and is GUARDED loudly. Mesh restriction:
# structured box / Delaunay non-obtuse.

TRI = dmesh.CellType.triangle  # the 2-D cell type used throughout the B4 gates


def test_cotangent_T_e_equals_negated_stiffness_offdiagonal_2d():
    """T_e VERIFICATION: the cotangent edge weight == DOLFINx's P1 stiffness off-diagonal.

    The crux of B4: the two-point transmissibility used by the upwind flux is T_e =
    1/2(cot a_ij + cot b_ij) over the (one or two) triangles sharing edge ij. This is the
    cotangent-Laplacian weight and MUST equal the negated assembled P1 stiffness off-diagonal
    -int grad(phi_i).grad(phi_j) dx -- the same lateral operator the galerkin path assembles. We
    pin them bit-for-bit on a small structured mesh (probe ``scratch/_b4_cotan_probe.py`` found
    max|diff| = 0.0), so the edge-graph T_e cannot silently diverge from the FEM Laplacian.
    """
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (1.0, 1.0)], [3, 3],
                                 cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)

    V = prob.V
    u = ufl.TrialFunction(V)
    vt = ufl.TestFunction(V)
    A = assemble_matrix(fem.form(ufl.dot(ufl.grad(u), ufl.grad(vt)) * ufl.dx))
    A.assemble()
    Ad = A.convert("dense").getDenseArray()

    maxdiff = 0.0
    for e in range(prob.n_edges):
        i, j = int(prob.edges[e, 0]), int(prob.edges[e, 1])
        stiff_off = -0.5 * (Ad[i, j] + Ad[j, i])  # symmetric P1 stiffness, negated off-diagonal
        maxdiff = max(maxdiff, abs(stiff_off - prob.T_e[e]))
    assert maxdiff < 1e-13, f"cotangent T_e diverged from stiffness off-diagonal by {maxdiff:.2e}"


def test_m_matrix_guard_holds_on_structured_V_2d():
    """M-MATRIX guard: every edge weight T_e >= 0 on the structured box V (monotonicity req).

    On a ``create_rectangle`` triangle mesh the right-angle split gives non-negative cotangent
    weights: the DIAGONAL/hypotenuse edges get cot(90 deg) = 0 (opposite the right angle), the
    AXIS edges get acute-angle cotangents > 0 (T_e in [0.5,1.0]). So min(T_e) = 0 (>= -1e-14). The
    class asserts this in its constructor; here we re-pin it on the mesh B5 will actually use.
    """
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (2.0, 1.0)], [16, 8],
                                 cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)
    assert prob.T_e.min() >= -1e-14, f"M-matrix violated: min T_e = {prob.T_e.min():.3e}"
    assert prob.mesh.topology.dim == 2  # genuinely the 2-D path
    # sum A_i = domain area (the conserved-quantity consistency: total_water = int d dx).
    assert prob.A_i.sum() == pytest.approx(2.0, rel=1e-12)


def test_m_matrix_guard_raises_on_obtuse_mesh_2d():
    """M-MATRIX guard FIRES: a deliberately OBTUSE triangulation must RAISE in the constructor.

    The monotone scheme REQUIRES T_e >= 0; an obtuse opposite angle gives cot < 0 -> a negative
    transmissibility (an anti-diffusive edge that breaks the maximum principle). The guard must
    refuse such a mesh LOUDLY rather than silently produce a non-monotone scheme. Two flat
    ("sliver") triangles sharing their long edge give cot(obtuse) ~ -4.95 on that edge (probe
    ``scratch/_b4_obtuse_probe.py``); building an ``UpwindOverlandProblem`` on it must raise.
    """
    import basix.ufl
    import ufl

    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.05], [0.5, -0.05]], dtype=np.float64)
    cells = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)  # share the long edge 0-1
    ufl_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = ufl.Mesh(ufl_el)
    obtuse = dmesh.create_mesh(MPI.COMM_WORLD, cells, domain, pts)

    with pytest.raises(ValueError, match="M-matrix"):
        UpwindOverlandProblem(obtuse, n_man=N_MAN)


def test_lake_at_rest_is_held_exactly_2d():
    """2-D lake-at-rest (well-balanced gate): a still pond over a SLOPING 2-D bed stays at rest.

    The 2-D analogue of ``test_lake_at_rest_is_held_exactly_1d``: a uniform surface head H = z_b+d
    on a 2-D tilted bed makes every edge head drop H_i - H_j = 0 (to roundoff in z_b + d), so all
    edge fluxes vanish and the depth holds to machine precision after a step -- STRUCTURAL on H
    (the scheme differences H, not d), independent of eps_S/eps_H. Depths are non-uniform (the bed
    tilts in both x and y) so this is a genuine 2-D well-balancedness check, not a flat lake.
    """
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (1.0, 1.0)], [12, 12],
                                 cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)
    # Sloping bed in both directions; flat surface H = 0.8 -> d = H - z_b strictly positive.
    prob.set_topography(lambda x: 0.5 - 0.2 * x[0] - 0.1 * x[1])
    still = lambda x: 0.3 + 0.2 * x[0] + 0.1 * x[1]  # d = 0.8 - z_b > 0 everywhere
    prob.set_initial_condition(still)
    d0 = prob.d.x.array.copy()
    w0 = prob.total_water()

    converged, iters = prob.step(dt=0.1)
    assert converged

    assert float(np.max(np.abs(prob.d.x.array - d0))) < 1e-14
    assert abs(prob.total_water() - w0) <= 1e-14 * max(1.0, abs(w0))
    assert prob.d.x.array.min() >= 0.0


def test_nonflat_head_drives_flow_2d():
    """2-D SANITY (anti-degenerate): a NON-uniform surface head actually moves water in 2-D.

    The lake gate is also passed by a do-nothing solver, so prove the 2-D scheme is live: a flat
    bed with a central depth bump (=> real edge head drops in both directions) must redistribute --
    the bump draws down at the peak while the flanks rise -- in a closed domain, conserving total
    water on a genuinely dynamic step (the 2-D telescoping flux network).
    """
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (1.0, 1.0)], [16, 16],
                                 cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)  # flat bed (z_b = 0)
    prob.set_initial_condition(
        lambda x: 0.1 + 0.1 * np.exp(-(((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02))
    )
    d0 = prob.d.x.array.copy()
    w0 = prob.total_water()
    coords = prob.V.tabulate_dof_coordinates()
    peak = int(np.argmin((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2))

    converged, iters = prob.step(dt=1e-3)
    assert converged

    moved = float(np.max(np.abs(prob.d.x.array - d0)))
    assert moved > 1e-6, f"non-flat 2-D head failed to drive flow (max move {moved:.2e})"
    assert prob.d.x.array[peak] < d0[peak]  # the bump draws down (diffusive spreading)
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.d.x.array))
    assert abs(prob.total_water() - w0) <= 1e-12 * max(1.0, abs(w0))


def test_tilted_v_catchment_conserves_2d():
    """2-D closed-slump CONSERVATION + positivity-without-limiter on the tilted-V catchment.

    Mirrors the galerkin ``test_tilted_v_catchment_conserves_2d`` (``test_overland_diffusionwave``):
    a water blob on a V-shaped bed (cross-slope to a central channel that tilts to the outlet)
    slumps in a CLOSED domain (no-flux, no rain), so total water is invariant; the blob visibly
    redistributes while depth stays >= 0. Here we assert the UPWIND path conserves to machine
    precision (telescoping edge signs => structural discrete conservation at every residual-
    converged root) AND has no negative excursion -- on this geometry, WITHOUT any limiter (the
    monotone-scheme payoff; contrast the galerkin path's ``_enforce_positivity`` clip). The blob
    here sits on a WET BASE so no wet/dry front forms (the B3-documented mild-2% sub-mm undershoot
    is a SEPARATE, characterized regime); this gate is the clean conserve-and-positive check.

    The class carries NO positivity limiter -- assert the clipping machinery is absent so a
    non-negative min(d) here is the monotone construction, not a post-step clip.
    """
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)
    # tilted-V bed: V across y (channel at y=0.5) + gentle tilt along x toward x=1 (same as galerkin).
    prob.set_topography(lambda x: 0.05 * np.abs(x[1] - 0.5) + 0.02 * (1.0 - x[0]))
    # broad blob on a wet base (0.05) -> dynamic 2-D redistribution, no wet/dry front.
    prob.set_initial_condition(
        lambda x: 0.05 + 0.1 * np.exp(-(((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02))
    )
    w0 = prob.total_water()
    d_before = prob.d.x.array.copy()

    assert not hasattr(prob, "_enforce_positivity")
    assert not hasattr(prob, "max_clip_seen")

    t, nsteps, min_d, _ = _march(prob, t_end=0.02, dt0=1e-3)
    assert nsteps > 0

    assert abs(prob.total_water() - w0) <= 1e-12 * max(1.0, abs(w0))  # conserved (closed, no rain)
    assert np.max(np.abs(prob.d.x.array - d_before)) > 1e-3  # genuine 2-D redistribution
    assert min_d >= -1e-12  # positivity over the whole run, NO limiter
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.d.x.array))


def test_steep_front_positive_without_limiter_2d():
    """2-D POSITIVITY GATE on a STEEP wet/dry FRONT: d >= 0 over the run, NO limiter.

    The 2-D analogue of the 1-D headline ``test_front_advance_positive_without_limiter_1d`` -- the
    IDENTICAL steep-front scenario (5% slope, sharp Gaussian peak 0.25 at x=7, marched to t=0.01
    day) extruded into a thin 2-D ribbon, so the wet/dry front advances downslope into dry ground
    in 2-D. This is the regime where the front head-drop >> eps_H, so the smoothed-upwind selector
    is SHARP and the monotone scheme holds ``d.min() >= -1e-12`` STRICTLY without any limiter (the
    upwind payoff; the galerkin path must clip ~mm here). The probe
    ``scratch/_b4_positivity_probe.py`` measured run-min ~6e-34 on this scenario in 2-D --
    bit-identical to the 1-D steep front, and eps_H-invariant from 1e-3 down -- confirming the 2-D
    cotangent flux is monotone on a genuine steep front exactly as 1-D.

    STEEPNESS, honestly (the B3 conditionality, now MEASURED in 2-D): "steep" means the FRONT
    head-drop >> eps_H, NOT merely a large bed slope. A small off-channel blob on a 5% cross-slope V
    is a MILD front in disguise (its surface-head drop across the wet/dry interface ~ eps_H) and
    undershoots ~0.9 mm at eps_H=1e-3 (probe), shrinking to ~1e-15 as eps_H sharpens -- exactly the
    B3-documented geometry-dependent 0..~0.9 mm mild regime, governed by eps_H vs the front
    head-drop. THIS gate uses the downslope-advancing front (head-drop >> eps_H) where strictness is
    structural; B5 must MEASURE the actual undershoot on the V's mild 2% valley (it is not assumed
    d>=0 there). The class carries NO limiter -- assert the clip machinery is absent so a
    non-negative min(d) is the monotone construction, not a post-step clip.
    """
    # thin 2-D ribbon (the 1-D steep front extruded in y): create_rectangle, 60x4 triangles.
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (20.0, 1.0)], [60, 4],
                                 cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: 0.05 * (20.0 - x[0]))  # 5% slope toward x=20 -> stiff front
    prob.set_initial_condition(lambda x: 0.25 * np.exp(-((x[0] - 7.0) / 1.5) ** 2))
    w0 = prob.total_water()

    assert not hasattr(prob, "_enforce_positivity")
    assert not hasattr(prob, "max_clip_seen")

    coords = prob.V.tabulate_dof_coordinates()
    xs = coords[:, 0]
    wet = 0.02 * 0.25  # wetted-front threshold (2% of peak)
    front_pre = xs[prob.d.x.array > wet].max()

    t, nsteps, min_d, _ = _march(prob, t_end=0.01, dt0=1e-4)
    assert nsteps > 0
    # (1) HEADLINE: depth never went negative over the WHOLE run, no limiter (strict, ~6e-34).
    assert min_d >= -1e-12, f"2-D steep front undershot to {min_d:.3e} without a limiter"
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.d.x.array))
    # (2) ADVERSARIAL: the wetted front genuinely advanced downslope into the dry region.
    front_post = xs[prob.d.x.array > wet].max()
    assert front_post > front_pre + 1.0, f"front did not advance: {front_pre:.2f} -> {front_post:.2f}"
    # (3) closed-domain conservation stays machine-tight.
    assert abs(prob.total_water() - w0) <= 1e-12 * max(1.0, abs(w0))


# ============================ B5: the decisive convergent-V gate ============================
# THE payoff of the O1 spike: does the upwind scheme reach the equilibrium plateau Q -> Q_eq on the
# convergent tilted-V where the galerkin OverlandProblem fails (P0: plateau gap-corrupted 0.676-0.996
# with dt PINNED ~5e-5; field-scale 0.876 under-resolved)? Canonical tilted-V (Di Giammarco 1996 /
# Kollet-Maxwell 2006), the SAME geometry/forcing as the galerkin diagnostic scratch/_v2d_overland_diag.py
# and the runner scratch/_v2d_upwind_V.py. The full measurement (mesh convergence 48x30 vs 96x60, field
# scale, dt distribution, oscillation, books, undershoot) lives in the runner; these two tests pin the
# decisive numbers as real assertions.

# canonical V constants (== _v2d_overland_diag.py / _v2d_upwind_V.py).
_VX, _VY = 1620.0, 1000.0      # m (SCALE=1.0)
_VXC = _VX / 2.0
_VSX, _VSY = 0.05, 0.02        # cross-slope, valley slope
_VN = 0.015                    # Manning n (SI)
_VRAIN = 0.2592                # m/day
_VSTORM = 0.0625               # day


def _v_topography(x):
    return _VSY * (_VY - x[1]) + _VSX * np.abs(x[0] - _VXC)


def test_v_outlet_line_discharge_matches_analytic_2d():
    """B5 OUTLET SUBTLETY: the 2-D LINE outlet discharge == the analytic q*LX (length-weighted).

    The V outlet is a LINE of nodes along y=LY, so outflow_rate() must INTEGRATE the per-unit-width
    Manning flux over the outlet edge length -- each node carries its boundary-edge control length
    (add_outflow_bc assembles int phi_k ds; the galerkin path gets this from its ds measure). At a
    KNOWN uniform depth d0 on the LY edge the discharge is the closed-form q_out_per_width * LX. The
    B2 NAIVE nodal sum (no length weighting) was ~33x wrong on a 48-node outlet (probe
    scratch/_b5_outlet_probe.py), which would make the +-3% Q_eq plateau gate meaningless -- so pin
    the length-weighting directly here. (1-D backward-compat -- B_k=1.0 at a point facet -- is pinned
    by test_outflow_discharge_absolute_magnitude_1d, which stays green.)
    """
    nx, ny, d0 = 48, 30, 0.01
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [_VX, _VY]], [nx, ny],
                                 cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=_VN)
    prob.set_topography(_v_topography)
    prob.set_initial_condition(lambda x: d0 + 0.0 * x[0])
    prob.add_outflow_bc(lambda x: np.isclose(x[1], _VY), slope=_VSY)

    q_per_width = 86400.0 * (1.0 / _VN) * d0 ** (5.0 / 3.0) * np.sqrt(_VSY)  # m^2/day, hard-coded 86400
    expected = q_per_width * _VX  # the analytic LINE discharge over the outlet edge [m^3/day]
    assert prob.outflow_rate() == pytest.approx(expected, rel=1e-6), (
        "2-D line-outlet discharge != analytic q*LX -- outlet length-weighting is wrong, the "
        "+-3% Q_eq V gate would be meaningless"
    )


def test_tilted_v_plateau_reaches_Qeq_2d():
    """THE decisive O1 gate: the convergent tilted-V storm plateau reaches Q_eq within +-3%.

    Drives UpwindOverlandProblem over the canonical tilted-V (dry start, constant storm rain, the
    y=LY normal-depth LINE outlet) through the storm window and asserts the late-storm outlet
    discharge plateau Q/Q_eq is within +-3% of 1.0 -- the gap the galerkin path could not close
    (P0 §8.3: galerkin plateau 0.676-0.996 with dt pinned ~5e-5; the limiter<->Newton sawtooth).
    48x30 = the ParFlow B6 grid, storm window only (the rising limb saturates within the storm),
    DT_MAX=1e-3 -- the well-resolved choice (~0.4s; the runner shows dt climbs to 1e-2 if the cap is
    raised, i.e. the galerkin pin is LIFTED, not a stiffness floor). Also asserts: the plateau
    oscillation RMS <= 2% (the sawtooth O1 removes), the books close to machine precision (the
    telescoping flux network is conservative), and the V undershoot is sub-mm (the mild 2% valley --
    MEASURED, not assumed d>=0, per the B3 conditionality). The runner _v2d_upwind_V.py carries the
    full mesh-convergence + field-scale + dt-distribution measurement; this is the Tier-1 pin.
    """
    nx, ny = 48, 30
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [_VX, _VY]], [nx, ny],
                                 cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=_VN)
    prob.set_topography(_v_topography)
    prob.set_initial_condition(lambda x: 0.0 * x[0])  # dry start
    prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], _VY), slope=_VSY)

    area = _VX * _VY
    q_eq = _VRAIN * area  # all rain runs off at equilibrium [m^3/day]

    # March the storm window with the adaptive controller (mirrors _v2d_upwind_V.py).
    t, dt, dt_max, t_end = 0.0, 1e-5, 1e-3, _VSTORM
    w_prev = prob.total_water()
    cum_rain = cum_out = 0.0
    run_min_d = float(prob.d.x.array.min())
    plateau_q = []  # Q/Q_eq over the late-storm window [0.6*STORM, STORM]
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        prob.add_rain(_VRAIN if (t + h) <= t_end + 1e-12 else 0.0)
        converged, it = prob.step(h)
        assert converged, f"V march failed to converge at t={t:.4g} (reason {prob.last_reason})"
        wn = prob.total_water()
        qout = prob.outflow_rate()
        cum_rain += h * _VRAIN * area
        cum_out += h * qout
        run_min_d = min(run_min_d, float(prob.d.x.array.min()))
        t += h
        if t >= 0.6 * _VSTORM:
            plateau_q.append(qout / q_eq)
        w_prev = wn
        dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), dt_max)

    plateau_q = np.array(plateau_q)
    plateau_mean = float(np.mean(plateau_q))
    plateau_rms = float(np.sqrt(np.mean((plateau_q - plateau_mean) ** 2)))
    books_gap = cum_rain - cum_out - (prob.total_water() - 0.0)

    # (1) THE HEADLINE: the convergent-V plateau reaches Q_eq within +-3% (galerkin: gap-corrupted).
    assert plateau_mean == pytest.approx(1.0, abs=0.03), (
        f"V plateau Q/Q_eq = {plateau_mean:.4f} not within +-3% of 1.0 (galerkin gave 0.676-0.996)"
    )
    # (2) the plateau is FLAT: oscillation RMS <= 2% (the sawtooth the monotone scheme removes).
    assert plateau_rms <= 0.02, f"plateau oscillation RMS {100*plateau_rms:.2f}% exceeds the 2% bar"
    # (3) conservative by construction: the books close to machine precision.
    assert abs(books_gap) <= 1e-6 * max(cum_rain, 1.0), f"books gap {books_gap:+.3e} not machine-tight"
    # (4) the V undershoot is SUB-MM (mild 2% valley; MEASURED per the B3 conditionality, not d>=0).
    assert run_min_d >= -1e-3, f"V undershot {run_min_d:.3e} m (> 1 mm) -- worse than the B3 mild bound"
    assert np.all(np.isfinite(prob.d.x.array))
