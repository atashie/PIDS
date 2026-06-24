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
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
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
