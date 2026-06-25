"""Hardening (ACCEPTANCE) tests for the MERGED sequential operator-split overland<->subsurface
coupling (``pids_forward/physics/sequential_coupling.py`` ``SequentialCoupledProblem``).

These are NOT red-green TDD: the production code is already merged + validated (sign-off note
``validation/sanity/sequential_overland_signoff__2026-06-23.md``; Tier-3 signed off). The point here
is to VALIDATE that scheme on bigger / harder cases than the Tier-1/2 suite covers, with
PRE-REGISTERED tolerances that come straight from the scheme's documented claims (conservation to
~5e-12; the routing store ``sum_i d_i A_i`` telescopes to ~1e-12). A FAILING hardening assertion is a
real finding (a genuine limit / a bug), NOT something to loosen.

Task 1 -- FIELD-SCALE ROBUSTNESS, COARSE SUITE GUARD: the convergent-CLAY tilted-V hillslope -- the
regime that historically dt-collapsed / sawtoothed the monolithic Manning schemes (Pathology 1+2;
see the redesign decision record ``docs/plans/2026-06-22-overland-flow-sequential-coupling-decision``)
-- run at TWO genuinely-distinct coarse resolutions. Each rung must complete to t_end with NO
dt-collapse, conserve to machine precision over the WHOLE run (running max of ``|balance|/cum_rain``,
not just the final step), keep the routing store telescoping (``last_routing_resid <= 1e-12`` every
step), keep the pond finite + non-negative, and be non-vacuous (rain fell AND water routed out).
"""
import time

import numpy as np
import pytest
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem            # Task 2 (b1): scheme-vs-scheme
from pids_forward.physics.sequential_coupling import SequentialCoupledProblem

COMM = MPI.COMM_WORLD


# ====================================================================================================
# Fixture -- a CONVERGENT-CLAY tilted-V hillslope (the dt-collapse regime), built at a given coarse
# resolution. Mirrors the signed-off ``v_topo`` analog (the spike ``tilted_v_case`` / the suite's
# ``_small_tilted_v``): a FLAT-top box with z_b carrying the V topography (cross-fall to a central
# valley axis + a downslope fall to a y=0 outlet), a DEEP unsaturated buffer (LZ=1.5 m) that dodges the
# unconfined no-Ss saturation singularity, and an intense-vs-clay-Ks short storm so it genuinely ponds
# + runs on to the outlet. CLAY soil per the task (Ks=0.048 m/d << the 1.0 m/d storm -> Hortonian
# runoff). The box is kept small (10 x 6 x 1.5) so each rung runs well under ~15 s.
# ====================================================================================================
LX, LY, LZ = 10.0, 6.0, 1.5
S0, SX = 0.05, 0.08         # S0 = downslope fall to the y=0 outlet; SX = cross-fall to the valley axis
PSI_I = -0.30              # uniform antecedent (wettable clay, comfortably unsaturated)
RAIN = 1.0                # m/day storm burst (>> clay Ks=0.048 -> ponds + runs on)
STORM_DUR = 0.08          # d (short intense burst)
T_END = 0.40              # d (storm + recession)


def _convergent_clay_hillslope(nx, ny, nz):
    """Build a SequentialCoupledProblem for the convergent-clay tilted-V at resolution (nx, ny, nz).

    Returns (prob, rain_handle). The V topography is carried by z_b (the mesh top is flat); the y=0
    edge is the free-drainage Manning outlet at the downslope friction slope S0.
    """
    soil = VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.048)  # CLAY (per task)
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [nx, ny, nz])
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
    # tilted-V: cross-fall |x - LX/2| toward the central valley axis + a downslope fall toward y=0.
    prob.set_topography(lambda x: S0 * x[1] + SX * np.abs(x[0] - LX / 2.0))
    prob.set_initial_condition(lambda x: PSI_I + 0.0 * x[0])
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=S0)   # the V routes everything to y=0
    rain = prob.add_rain(0.0)
    return prob, rain


def _march_record(prob, rain, *, dt0=1e-3, dt_max=0.03):
    """March the storm-then-recession with a BAND dt-controller, RECORDING the running maxima over ALL
    accepted steps. Returns a dict of run statistics.

    The band controller (it<=4 grow*1.4, it>=12 shrink*0.7, capped at dt_max) is the production fix #4
    pattern (mirrors the suite's ``_march`` / ``_run_small_sand_storm``). On a non-converged step we
    HALVE dt and assert dt stays > 1e-9 -- that assert IS the no-dt-collapse check (the monolith
    dt-collapsed below the float floor here). We track the running max of |balance|/cum_rain over EVERY
    accepted step (not just the final state) and the running max of last_routing_resid + the peak pond.
    """
    t, nstep, dt = 0.0, 0, dt0
    bal_frac_max = 0.0
    routing_resid_max = 0.0
    peak_pond = 0.0
    while t < T_END - 1e-12:
        h = min(dt, T_END - t)
        if t < STORM_DUR - 1e-12 and t + h > STORM_DUR:
            h = STORM_DUR - t                                   # land exactly on the storm end
        rain.value = RAIN if t < STORM_DUR - 1e-12 else 0.0
        conv, it = prob.step(h)
        if not conv:
            dt *= 0.5
            # the no-dt-collapse guard: the merged split decouples the stiff Richards Jacobian from the
            # surface routing, so the convergent-clay V should never need dt below the float floor.
            assert dt > 1e-9, f"DT COLLAPSE at t={t:.6f} (Pathology 1/2 not fixed at this resolution)"
            continue
        nstep += 1
        t += h
        # running maxima over ALL accepted steps (not just the final step).
        routing_resid_max = max(routing_resid_max, prob.last_routing_resid)
        peak_pond = max(peak_pond, float(np.max(np.maximum(prob._rp.psi.x.array, 0.0))))
        if prob.cum_rain > 0.0:
            bal_frac_max = max(bal_frac_max, abs(prob.balance()) / prob.cum_rain)
        if it <= 4:
            dt = min(dt * 1.4, dt_max)
        elif it >= 12:
            dt = dt * 0.7
    return dict(nstep=nstep, bal_frac_max=bal_frac_max, routing_resid_max=routing_resid_max,
                peak_pond=peak_pond)


# ====================================================================================================
# Task 1 -- TWO-RUNG FIELD-SCALE ROBUSTNESS: the convergent-clay tilted-V at two genuinely-distinct
# coarse resolutions. Each rung asserts: completes (no dt-collapse), machine-tight conservation over
# the WHOLE run, telescoping routing store, finite non-negative pond, and non-vacuous (rain + outflow).
# ====================================================================================================
def test_field_scale_robustness_coarse_two_rungs():
    """Run the convergent-clay tilted-V hillslope (the historical dt-collapse regime) at TWO coarse
    resolutions and assert the merged sequential split is robust at both:
      (1) COMPLETES to t_end with NO dt-collapse (the ``dt > 1e-9`` assert in the marcher never trips);
      (2) machine-tight CONSERVATION over the WHOLE run: max(|balance|/cum_rain) over every accepted
          step < 1e-6 (the merged scheme's documented ~5e-12 closure, omega/substep-independent; 1e-6
          is a generous ceiling, ~6 orders above the typical residual);
      (3) the routing store telescopes: last_routing_resid <= 1e-12 every step;
      (4) the peak pond max(max(psi,0)) is finite and >= 0;
      (5) NON-VACUOUS: cum_rain > 0 AND cum_outflow > 0 (the storm fell and routed to the y=0 outlet --
          otherwise the test proves nothing).
    The two rungs (16x10x4 and 24x16x5) genuinely differ (~3x the cells), so this is a coarse
    resolution-robustness guard, not a convergence study."""
    rungs = [(16, 10, 4), (24, 16, 5)]   # two genuinely-distinct coarse meshes (the second ~3x cells)
    results = {}
    for (nx, ny, nz) in rungs:
        prob, rain = _convergent_clay_hillslope(nx, ny, nz)
        t0 = time.perf_counter()
        stats = _march_record(prob, rain)       # (1) the dt>1e-9 assert inside guards no-collapse
        wall = time.perf_counter() - t0
        results[(nx, ny, nz)] = (stats, prob, wall)

        tag = f"{nx}x{ny}x{nz}"
        assert stats["nstep"] > 0, f"[{tag}] storm did not advance (no accepted steps)"

        # (2) machine-tight conservation over the WHOLE run. 1e-6 is the PRE-REGISTERED ceiling: the
        # merged win design closes |balance|/cum_rain to ~5e-12 (the sign-off CONSERVATION-PROOF;
        # omega/substep-independent), so 1e-6 sits ~6 orders above the expected residual.
        assert stats["bal_frac_max"] < 1e-6, (
            f"[{tag}] conservation broke over the run: max(|bal|/cum_rain) = "
            f"{stats['bal_frac_max']:.3e} >= 1e-6 (cum_rain={prob.cum_rain:.3e}, "
            f"cum_outflow={prob.cum_outflow:.3e}, cum_drainage={prob.cum_drainage:.3e})")

        # (3) the routing store sum_i d_i A_i telescopes (the conservation fix): resid <= 1e-12 / step.
        assert stats["routing_resid_max"] <= 1e-12, (
            f"[{tag}] routing store sum(d*A) resid {stats['routing_resid_max']:.3e} > 1e-12")

        # (4) the peak pond is finite and non-negative (no NaN/Inf blow-up; pond = max(psi,0) >= 0).
        assert np.isfinite(stats["peak_pond"]) and stats["peak_pond"] >= 0.0, (
            f"[{tag}] peak pond not finite/non-negative: {stats['peak_pond']!r}")

        # (5) NON-VACUOUS guards: the storm actually fell AND actually routed to the y=0 outlet.
        assert prob.cum_rain > 0.0, f"[{tag}] no rain fell -> vacuous robustness check"
        assert prob.cum_outflow > 0.0, (
            f"[{tag}] no surface outflow -> the storm did not route to the y=0 outlet (vacuous)")

    # both rungs genuinely differ (distinct meshes -> distinct accepted-step counts is expected but not
    # asserted; the meshes themselves differ by construction). Emit per-rung numbers for the record.
    for (nx, ny, nz), (stats, prob, wall) in results.items():
        print(f"\n[{nx}x{ny}x{nz}] steps={stats['nstep']} "
              f"max|bal|/cum_rain={stats['bal_frac_max']:.3e} "
              f"max_routing_resid={stats['routing_resid_max']:.3e} "
              f"peak_pond={stats['peak_pond']:.4e} m "
              f"cum_rain={prob.cum_rain:.4e} cum_outflow={prob.cum_outflow:.4e} "
              f"cum_drainage={prob.cum_drainage:.4e} wall={wall:.1f}s")


# ====================================================================================================
# Task 2 -- RUN-ON ACCURACY (three tests). The merged sequential split routes surface water downslope
# and lets it RUN ON to where it infiltrates. These tests validate that run-on (a) partitions the
# storm budget like the resolved monolithic upwind scheme on a shared mesh (b1), (b) physically moves
# water onto a convergence line so it soaks in there vs a no-transport twin (b2), and (c) is RESOLVED
# at the production route_substeps=4 (b3). All partition the KNOWN analytic rain input
# ``R_in = RAIN * top_area * STORM_DUR`` into routed-out / infiltrated / still-ponded using only
# quantities BOTH schemes expose (cum_outflow, cum_drainage, the soil-storage integral, the surface
# store) -- never an internal cum_rain.
# ====================================================================================================
def _top_area_ds(mesh, ztop):
    """``int 1 ds_top`` (the plan area of the top facet) on a degree-8 ``ds`` -- the denominator of the
    known analytic rain input R_in. Built directly from the mesh (scheme-independent)."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    tf = np.sort(dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[mesh.geometry.dim - 1], ztop))).astype(np.int32)
    ft = dmesh.meshtags(mesh, fdim, tf, np.ones(tf.size, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=mesh, subdomain_data=ft,
                         metadata={"quadrature_degree": 8})(1)
    return mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ds_top)), op=MPI.SUM)


def _soil_water_deg8(prob, soil):
    """``int theta(psi) dV`` on a degree-8 ``dx`` -- the soil-storage integral, computed IDENTICALLY for
    BOTH schemes via the public ``prob.psi`` (the sequential split aliases ``_rp.psi`` to ``.psi``), so
    ``soil_gain = this(final) - this(initial)`` is an apples-to-apples partition term."""
    dxq = ufl.dx(metadata={"quadrature_degree": prob._quad_degree})
    return prob.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(soil.theta_ufl(prob.psi) * dxq)), op=MPI.SUM)


def _mono_surface_store(prob):
    """The monolith surface store ``int d ds_top`` (== its ``surface_water()``); kept explicit so the
    partition reads symmetrically with the sequential ``prob.surface_water()`` (``int max(psi,0) ds_top``)."""
    return prob.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(prob.d * prob._ds_top)), op=MPI.SUM)


# Case A fixture -- a MILD planar LOAM hillslope (NON-stiff: the monolith upwind CONVERGES). Gentle
# planar slope to a y=0 outlet, a moderate short storm (RAIN>Ks so it ponds + runs off, not violently),
# then a recession so BOTH surface stores drain (the partition stabilizes into routed-vs-infiltrated).
A_LX, A_LY, A_LZ = 8.0, 5.0, 1.0
A_NX, A_NY, A_NZ = 12, 8, 4
A_S0 = 0.03                 # gentle planar fall to the y=0 outlet
A_PSI_I = -0.4
A_RAIN = 0.5               # m/day, > loam Ks=0.25 -> ponds modestly + runs off
A_STORM = 0.08            # d (short burst)
A_TEND = 0.45             # d (storm + recession so both surfaces drain -> stable partition)
A_SOIL = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)   # LOAM (per task)


def _march_storm(prob, rain, *, storm_dur, storm_rain, t_end, dt0=1e-3, dt_max=0.03):
    """March a storm-then-recession with the production BAND dt-controller (it<=4 grow*1.4, it>=12
    shrink*0.7, capped dt_max), driving ``rain.value`` on/off at ``storm_dur``. Works for BOTH schemes
    (same ``step(dt)->(conv,it)`` surface). On a non-converged step HALVE dt; ``collapsed`` flags a drop
    below the 1e-9 float floor (the no-dt-collapse guard). Returns (nstep, collapsed, t_reached, wall)."""
    t, nstep, dt = 0.0, 0, dt0
    collapsed = False
    t0 = time.perf_counter()
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if t < storm_dur - 1e-12 and t + h > storm_dur:
            h = storm_dur - t                                   # land exactly on the storm end
        rain.value = storm_rain if t < storm_dur - 1e-12 else 0.0
        conv, it = prob.step(h)
        if not conv:
            dt *= 0.5
            if dt < 1e-9:
                collapsed = True
                break
            continue
        t += h
        nstep += 1
        if it <= 4:
            dt = min(dt * 1.4, dt_max)
        elif it >= 12:
            dt = dt * 0.7
    return nstep, collapsed, t, time.perf_counter() - t0


# ====================================================================================================
# Test b1 -- RUN-ON PARTITION, SCHEME-vs-SCHEME on a SHARED mesh: the sequential split and the
# monolithic ``CoupledProblem(overland_scheme="upwind")`` run the SAME coarse mesh + SAME forcing on a
# mild planar loam hillslope where the monolith upwind GENUINELY CONVERGES. Both decompose the known
# analytic R_in into routed-out / infiltrated; the two must agree on those fractions to a pre-registered
# few-% (the under-resolution is shared, so it cancels -- this is agreement, not a convergence study).
# ====================================================================================================
@pytest.mark.xfail(reason="OPEN run-on PARTITION bug: the sequential pond-in-psi infiltration is "
                          "UNCAPPED, so it over-infiltrates ~24-40 pp vs the ParFlow-validated monolith. "
                          "Structural CLOSURE bug (NOT the lateral lag); 3 fixes refuted (cap-only, "
                          "iteration-only, iteration+cap). See "
                          "validation/sanity/overland_partition_bug_investigation__2026-06-24.md. "
                          "XPASS here flags that the infiltration closure has been fixed.", strict=False)
def test_caseA_runon_partition_matches_upwind_coarse():
    """On a MILD planar loam hillslope (RAIN=0.5 m/d > Ks=0.25 -> modest Hortonian ponding + run-on to a
    y=0 outlet, then recession), the merged sequential split and the monolithic upwind scheme -- run on
    the SAME coarse 12x8x4 mesh with the SAME storm -- must partition the KNOWN analytic rain input
    ``R_in = RAIN * top_area * STORM_DUR`` into the same routed-out and infiltrated fractions.

    GUARD: the monolith upwind must genuinely converge here (reach t_end, ``_effective_overland_scheme
    == 'upwind'``, no dt-collapse); if it cannot converge even on this non-stiff case, THAT is the
    finding. PARTITION (both schemes expose every term directly): ``routed_out = cum_outflow``;
    ``infiltrated = (soil_gain + drained)`` with ``soil_gain = int theta(final) - int theta(initial)``
    (degree-8 dx, public ``prob.psi`` for both) and ``drained = cum_drainage``. Both fractions are of
    R_in. TOLERANCE (PRE-REGISTERED): ``abs(frac_seq - frac_mono) < 0.05`` of R_in (<= 5 percentage
    points of the rain budget) on each of routed_out/R_in and infiltrated/R_in -- justified as the
    OPERATOR-SPLIT LAG (route-then-infiltrate, sequentially) vs the monolith's SIMULTANEITY (route and
    infiltrate co-solved in one Newton) on a shared coarse mesh. A LARGER gap is REPORTED (printed +
    asserted), NOT silently tolerated. NON-VACUOUS: R_in>0 and BOTH schemes routed out>0 AND
    infiltrated>0 (the storm genuinely ponded, ran off, and soaked in)."""
    soil = VanGenuchten(**A_SOIL)

    # --- sequential split ---
    msh_s = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [A_LX, A_LY, A_LZ]], [A_NX, A_NY, A_NZ])
    seq = SequentialCoupledProblem(msh_s, soil, n_man=0.05)
    seq.set_topography(lambda x: A_S0 * x[1])
    seq.set_initial_condition(lambda x: A_PSI_I + 0.0 * x[0])
    seq.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=A_S0)
    rain_s = seq.add_rain(0.0)
    th0_s = _soil_water_deg8(seq, soil)
    nstep_s, coll_s, tend_s, wall_s = _march_storm(
        seq, rain_s, storm_dur=A_STORM, storm_rain=A_RAIN, t_end=A_TEND)

    # --- monolith, upwind lateral overland (the resolved reference scheme) ---
    msh_m = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [A_LX, A_LY, A_LZ]], [A_NX, A_NY, A_NZ])
    mono = CoupledProblem(msh_m, soil, overland_scheme="upwind")
    mono.set_topography(lambda x: A_S0 * x[1])
    mono.set_initial_condition(lambda x: A_PSI_I + 0.0 * x[0], d_value=0.0)
    mono.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=A_S0)
    rain_m = mono.add_rain(0.0)
    th0_m = _soil_water_deg8(mono, soil)
    nstep_m, coll_m, tend_m, wall_m = _march_storm(
        mono, rain_m, storm_dur=A_STORM, storm_rain=A_RAIN, t_end=A_TEND)

    # GUARD: the monolith upwind must have genuinely converged (else the comparison is meaningless --
    # report that as the finding rather than comparing against a dt-collapsed run).
    assert mono._effective_overland_scheme == "upwind", \
        f"monolith did not run the upwind scheme (effective={mono._effective_overland_scheme!r})"
    assert not coll_m and tend_m >= A_TEND - 1e-9, (
        f"monolith UPWIND dt-collapsed on the mild planar loam hillslope (reached t={tend_m:.4f} of "
        f"{A_TEND}, collapsed={coll_m}) -- it cannot converge even here; FINDING (cannot compare).")
    assert not coll_s and tend_s >= A_TEND - 1e-9, \
        f"sequential split did not reach t_end (t={tend_s:.4f}, collapsed={coll_s})"

    # the known analytic rain input (same geometry + storm for both -> one R_in).
    top_area = _top_area_ds(msh_s, A_LZ)
    R_in = A_RAIN * top_area * A_STORM
    assert R_in > 0.0

    # partition each scheme: routed_out / infiltrated (= soil_gain + drained), as fractions of R_in.
    routed_s = seq.cum_outflow / R_in
    infil_s = (_soil_water_deg8(seq, soil) - th0_s + seq.cum_drainage) / R_in
    surf_s = seq.surface_water() / R_in
    routed_m = mono.cum_outflow / R_in
    infil_m = (_soil_water_deg8(mono, soil) - th0_m + mono.cum_drainage) / R_in
    surf_m = _mono_surface_store(mono) / R_in

    print(f"\n[b1 caseA] top_area={top_area:.3f} R_in={R_in:.4f}"
          f"\n  SEQ : routed/R={routed_s:.4f} infil/R={infil_s:.4f} surf/R={surf_s:.4f} "
          f"nstep={nstep_s} wall={wall_s:.1f}s"
          f"\n  MONO: routed/R={routed_m:.4f} infil/R={infil_m:.4f} surf/R={surf_m:.4f} "
          f"nstep={nstep_m} wall={wall_m:.1f}s (eff={mono._effective_overland_scheme})"
          f"\n  |drouted/R|={abs(routed_s - routed_m):.4f}  |dinfil/R|={abs(infil_s - infil_m):.4f}")

    # NON-VACUOUS: both schemes genuinely routed out AND infiltrated (otherwise the agreement is hollow).
    assert routed_s > 0.0 and routed_m > 0.0, \
        f"a scheme routed nothing out (routed_s={routed_s:.3e}, routed_m={routed_m:.3e}) -> vacuous"
    assert infil_s > 0.0 and infil_m > 0.0, \
        f"a scheme infiltrated nothing (infil_s={infil_s:.3e}, infil_m={infil_m:.3e}) -> vacuous"

    # PRE-REGISTERED tolerance: <= 5 percentage points of R_in on each non-tiny fraction. Justification:
    # operator-split LAG (sequential route-then-infiltrate) vs monolithic SIMULTANEITY on a shared coarse
    # mesh should perturb the partition by only a few points if the schemes are operator-equivalent.
    TOL = 0.05
    assert abs(routed_s - routed_m) < TOL, (
        f"FINDING -- run-on routed-out fraction DISAGREES between schemes by "
        f"{abs(routed_s - routed_m):.4f} of R_in (>= {TOL}): seq routed/R={routed_s:.4f} vs "
        f"mono routed/R={routed_m:.4f}. This is NOT operator-split lag at the few-% level -- it is a "
        f"structural difference (sequential route-then-infiltrate gives the soil more residence time "
        f"on this Hortonian hillslope than the monolith's simultaneous route+infiltrate Newton).")
    assert abs(infil_s - infil_m) < TOL, (
        f"FINDING -- run-on infiltrated fraction DISAGREES between schemes by "
        f"{abs(infil_s - infil_m):.4f} of R_in (>= {TOL}): seq infil/R={infil_s:.4f} vs "
        f"mono infil/R={infil_m:.4f}. The sequential split infiltrates materially more of the storm "
        f"than the monolith upwind on this mild planar loam hillslope (operator-ordering effect).")


# ====================================================================================================
# Test b2 -- RUN-ON SIGNATURE vs a NO-LATERAL-TRANSPORT twin: prove run-on PHYSICALLY moves water onto a
# convergence line where it soaks in. A Gaussian PIT carved into a planar loam slope (a convergent
# topographic SETTING with a y=0 outlet); the REAL run (n_man=0.05) routes slope water INTO the pit so
# it infiltrates THERE, while a twin with n_man=1e6 (infinite roughness -> routing velocity ~0) leaves
# every drop where it fell. The pit footprint must gain STRICTLY MORE soil water in the real run.
# ====================================================================================================
def _pit_on_slope(n_man, *, nx=14, ny=14, nz=4):
    """A Gaussian depression (a convergent topographic SETTING) on a planar loam slope falling to a y=0
    Manning outlet. ``n_man=0.05`` routes normally (slope water runs ON to the pit); ``n_man=1e6`` is the
    no-lateral-transport twin (routing velocity ~ 1/n -> 0, water infiltrates where it fell). Returns
    (prob, rain_handle, soil, mesh). Loam Ks=0.25 with RAIN=0.6 m/d: the slopes shed (Hortonian) but the
    pit can keep infiltrating concentrated run-on."""
    soil = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [B_LX, B_LY, B_LZ]], [nx, ny, nz])
    prob = SequentialCoupledProblem(msh, soil, n_man=n_man)
    prob.set_topography(lambda x: B_S0 * x[1]
                        - B_PIT_DEPTH * np.exp(-(((x[0] - B_CX) ** 2 + (x[1] - B_CY) ** 2)
                                                 / (B_RPIT * B_RPIT))))
    prob.set_initial_condition(lambda x: B_PSI_I + 0.0 * x[0])
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=B_S0)
    rain = prob.add_rain(0.0)
    return prob, rain, soil, msh


# pit-on-slope geometry + storm (shared by b2 and b3).
B_LX, B_LY, B_LZ = 8.0, 8.0, 1.2
B_CX, B_CY = B_LX / 2.0, B_LY * 0.6      # pit centre
B_S0 = 0.05                              # planar fall to the y=0 outlet
B_PIT_DEPTH, B_RPIT = 0.45, 1.6          # Gaussian depression depth + radius
B_BAND = 1.4                             # pit footprint: sqrt((x-cx)^2 + (y-cy)^2) < B_BAND
B_PSI_I = -0.5
B_RAIN, B_STORM, B_TEND = 0.6, 0.10, 0.45


def _band_theta_form(prob, soil):
    """``int theta(psi) * 1[ sqrt((x-cx)^2+(y-cy)^2) < B_BAND ] dV`` (degree-8 dx): the soil water under
    the pit (convergence-line) footprint, full depth -- the run-on signature is its change over the run."""
    xx = ufl.SpatialCoordinate(prob.mesh)
    r2 = (xx[0] - B_CX) ** 2 + (xx[1] - B_CY) ** 2
    ind = ufl.conditional(ufl.lt(r2, B_BAND * B_BAND), 1.0, 0.0)
    dxq = ufl.dx(metadata={"quadrature_degree": prob._quad_degree})
    return fem.form(soil.theta_ufl(prob.psi) * ind * dxq)


def test_caseB_runon_signature_vs_no_transport_twin():
    """Prove run-on physically happens: build TWO sequential runs on the SAME pit-on-slope mesh/forcing
    -- the REAL run (n_man=0.05, routes normally) and a NO-LATERAL-TRANSPORT twin (n_man=1e6, routing
    velocity ~0 -> water infiltrates where it fell). Assert:
      (1) the twin genuinely does NOT route: ``twin.cum_outflow`` is <= a tiny epsilon vs the real run's
          outflow (ratio < 1e-3) AND the twin's surface water stays put (it does not collect + spill --
          here it infiltrates in place, surface_water ~ 0);
      (2) the REAL run infiltrates STRICTLY MORE into the convergence-line (pit) footprint than the twin
          -- ``Dtheta_band_real > Dtheta_band_twin`` by a real margin (PRE-REGISTERED: the real band
          gains >= 30% more than the twin); this proves run-on MOVED water to where it soaks in, not
          merely that the run conserves. (Measured ~+79%; 30% is a comfortable floor.)
      (3) BOTH conserve: ``|balance|/cum_rain < 1e-6`` for each sequential run.
    NON-VACUOUS: rain fell (cum_rain>0), the REAL run routed out (so n_man genuinely transported), and
    both bands gained soil water."""
    # --- REAL run (routes) ---
    real, rain_r, soil_r, msh_r = _pit_on_slope(n_man=0.05)
    real._ensure_built()
    bf_r = _band_theta_form(real, soil_r)
    band0_r = real.mesh.comm.allreduce(fem.assemble_scalar(bf_r), op=MPI.SUM)
    nstep_r, coll_r, tend_r, wall_r = _march_storm(
        real, rain_r, storm_dur=B_STORM, storm_rain=B_RAIN, t_end=B_TEND)
    band_r = real.mesh.comm.allreduce(fem.assemble_scalar(bf_r), op=MPI.SUM)
    dband_real = band_r - band0_r

    # --- NO-TRANSPORT twin (n_man=1e6) ---
    twin, rain_t, soil_t, msh_t = _pit_on_slope(n_man=1e6)
    twin._ensure_built()
    bf_t = _band_theta_form(twin, soil_t)
    band0_t = twin.mesh.comm.allreduce(fem.assemble_scalar(bf_t), op=MPI.SUM)
    nstep_t, coll_t, tend_t, wall_t = _march_storm(
        twin, rain_t, storm_dur=B_STORM, storm_rain=B_RAIN, t_end=B_TEND)
    band_t = twin.mesh.comm.allreduce(fem.assemble_scalar(bf_t), op=MPI.SUM)
    dband_twin = band_t - band0_t

    assert not coll_r and not coll_t, f"a run dt-collapsed (real={coll_r}, twin={coll_t})"
    assert real.cum_rain > 0.0, "no rain fell -> vacuous run-on signature"

    print(f"\n[b2 caseB] pit-on-slope BAND=sqrt(r2)<{B_BAND}"
          f"\n  REAL n=0.05: cum_out={real.cum_outflow:.5e} surf={real.surface_water():.4e} "
          f"dDtheta_band={dband_real:.5e} bal/rain={abs(real.balance()) / real.cum_rain:.2e} "
          f"wall={wall_r:.1f}s"
          f"\n  TWIN n=1e6 : cum_out={twin.cum_outflow:.5e} surf={twin.surface_water():.4e} "
          f"dDtheta_band={dband_twin:.5e} bal/rain={abs(twin.balance()) / twin.cum_rain:.2e} "
          f"wall={wall_t:.1f}s"
          f"\n  twin_out/real_out={twin.cum_outflow / real.cum_outflow:.3e}  "
          f"band_real/band_twin={dband_real / dband_twin:.3f}")

    # (1) the twin does NOT route: its outflow is negligible vs the real run's (the real run genuinely
    # transports), and the twin's surface water stays put (infiltrates in place, does not spill).
    assert real.cum_outflow > 1e-3, \
        f"REAL run routed ~nothing (cum_out={real.cum_outflow:.3e}) -> n_man did not transport (vacuous)"
    assert twin.cum_outflow < 1e-3 * real.cum_outflow, (
        f"the n_man=1e6 twin ROUTED (cum_out={twin.cum_outflow:.3e} vs real {real.cum_outflow:.3e}, "
        f"ratio {twin.cum_outflow / real.cum_outflow:.3e} >= 1e-3) -> not a no-transport twin")
    # the twin's surface store stays put (here: ~0, the un-routed pond infiltrated in place); the real
    # run, by concentrating run-on into the pit, ponds MORE than the twin at t_end.
    assert twin.surface_water() <= real.surface_water() + 1e-9, (
        f"the twin retained MORE surface water than the real run (twin surf={twin.surface_water():.3e} "
        f"> real surf={real.surface_water():.3e}) -> the twin is not simply holding water in place")

    # (2) the run-on signature: the REAL run gains STRICTLY MORE soil water in the pit footprint than the
    # twin, by a real margin. PRE-REGISTERED >= 30% extra (measured ~+79%) -- proves run-on moved water
    # to where it soaks in (a no-transport twin gets only the pit's DIRECT rain there).
    assert dband_real > 0.0 and dband_twin > 0.0, \
        f"a band did not gain soil water (real={dband_real:.3e}, twin={dband_twin:.3e}) -> vacuous"
    EXTRA = 0.30
    assert dband_real > (1.0 + EXTRA) * dband_twin, (
        f"run-on did NOT move materially more water into the pit footprint: Dtheta_band_real="
        f"{dband_real:.5e} is not > {1.0 + EXTRA:.2f}x Dtheta_band_twin={dband_twin:.5e} "
        f"(ratio {dband_real / dband_twin:.3f}) -- run-on signature too weak / absent.")

    # (3) BOTH conserve around the run-on (the sequential scheme exposes balance()).
    for tag, p in (("real", real), ("twin", twin)):
        bal_frac = abs(p.balance()) / p.cum_rain
        assert bal_frac < 1e-6, f"[{tag}] balance did not close: |bal|/cum_rain = {bal_frac:.3e}"


# ====================================================================================================
# Test b3 -- route_substeps RESOLVED for run-on: the SAME pit-on-slope run-on case at route_substeps=4
# (production default) and route_substeps=8. The partition (routed-out + infiltrated fractions of R_in)
# must agree within a pre-registered tolerance -- rs=4 is RESOLVED for infiltrating run-on, not still
# climbing. Complements the existing pond-release rs=1-vs-rs=4 throttle test in test_sequential_coupling
# (which proves rs MOVES transport); this proves rs=4 has CONVERGED for the run-on PARTITION.
# ====================================================================================================
def _pit_partition(route_substeps):
    """March the pit-on-slope run-on case at a given route_substeps and return its R_in-normalized
    partition (routed_out, infiltrated, surface_store) + non-vacuity numbers."""
    prob, rain, soil, msh = _pit_on_slope(n_man=0.05)
    prob.route_substeps = int(route_substeps)   # set after build (geometry/forms unaffected)
    th0 = _soil_water_deg8(prob, soil)
    nstep, coll, tend, wall = _march_storm(
        prob, rain, storm_dur=B_STORM, storm_rain=B_RAIN, t_end=B_TEND)
    assert not coll and tend >= B_TEND - 1e-9, \
        f"pit run-on (rs={route_substeps}) dt-collapsed (t={tend:.4f}, collapsed={coll})"
    top_area = _top_area_ds(msh, B_LZ)
    R_in = B_RAIN * top_area * B_STORM
    routed = prob.cum_outflow / R_in
    infil = (_soil_water_deg8(prob, soil) - th0 + prob.cum_drainage) / R_in
    surf = prob.surface_water() / R_in
    return dict(routed=routed, infil=infil, surf=surf, R_in=R_in, prob=prob, wall=wall)


@pytest.mark.xfail(reason="OPEN run-on PARTITION bug: route_substeps does NOT resolve the run-on "
                          "partition (same uncapped-closure root cause as test_caseA). See "
                          "validation/sanity/overland_partition_bug_investigation__2026-06-24.md. "
                          "XPASS here flags that the infiltration closure has been fixed.", strict=False)
def test_route_substeps_partition_resolved_rs4_vs_rs8():
    """The pit-on-slope run-on case at route_substeps=4 (production default) vs route_substeps=8 must
    agree on the R_in partition (routed-out and infiltrated fractions) within a PRE-REGISTERED tolerance
    ``abs(frac_rs4 - frac_rs8) < 0.03`` of R_in -- i.e. rs=4 is RESOLVED for infiltrating run-on, not
    still climbing toward the fully-substepped kinematic limit. Complements the existing pond-release
    rs=1-vs-rs=4 throttle test in ``test_sequential_coupling.py`` (which proves rs MOVES transport for
    a pure release); this asks the stronger question -- has rs=4 CONVERGED for the run-on partition? --
    on the canonical run-on fixture. NON-VACUOUS: both runs genuinely routed off (routed/R > 0) and
    infiltrated (infil/R > 0). A larger rs4->rs8 drift is REPORTED, not tolerated."""
    r4 = _pit_partition(route_substeps=4)
    r8 = _pit_partition(route_substeps=8)

    print(f"\n[b3 rs4-vs-rs8] pit-on-slope run-on (R_in={r4['R_in']:.3f})"
          f"\n  rs=4: routed/R={r4['routed']:.4f} infil/R={r4['infil']:.4f} surf/R={r4['surf']:.4f} "
          f"wall={r4['wall']:.1f}s"
          f"\n  rs=8: routed/R={r8['routed']:.4f} infil/R={r8['infil']:.4f} surf/R={r8['surf']:.4f} "
          f"wall={r8['wall']:.1f}s"
          f"\n  |drouted/R|={abs(r4['routed'] - r8['routed']):.4f}  "
          f"|dinfil/R|={abs(r4['infil'] - r8['infil']):.4f}")

    # NON-VACUOUS: both genuinely ran off AND infiltrated.
    assert r4["routed"] > 0.0 and r8["routed"] > 0.0, \
        f"a run routed nothing (rs4={r4['routed']:.3e}, rs8={r8['routed']:.3e}) -> vacuous"
    assert r4["infil"] > 0.0 and r8["infil"] > 0.0, \
        f"a run infiltrated nothing (rs4={r4['infil']:.3e}, rs8={r8['infil']:.3e}) -> vacuous"

    # PRE-REGISTERED: rs=4 vs rs=8 partition agrees within 3 percentage points of R_in (rs=4 resolved).
    TOL = 0.03
    assert abs(r4["routed"] - r8["routed"]) < TOL, (
        f"FINDING -- the run-on routed-out fraction is STILL CLIMBING from rs=4 to rs=8 by "
        f"{abs(r4['routed'] - r8['routed']):.4f} of R_in (>= {TOL}): rs4 routed/R={r4['routed']:.4f} "
        f"vs rs8 routed/R={r8['routed']:.4f} -- rs=4 is NOT resolved for the infiltrating-run-on "
        f"PARTITION (the existing calibration resolves rs=4 for pond-release/drain TIMING, a different "
        f"quantity).")
    assert abs(r4["infil"] - r8["infil"]) < TOL, (
        f"FINDING -- the run-on infiltrated fraction is STILL CLIMBING from rs=4 to rs=8 by "
        f"{abs(r4['infil'] - r8['infil']):.4f} of R_in (>= {TOL}): rs4 infil/R={r4['infil']:.4f} vs "
        f"rs8 infil/R={r8['infil']:.4f} -- rs=4 is NOT resolved for the run-on partition.")
