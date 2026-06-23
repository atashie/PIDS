"""Tier-1/2 tests for the sequential operator-split overland<->subsurface coupling (B4+B5).

``pids_forward/physics/sequential_coupling.py`` holds ``SequentialCoupledProblem`` -- the PRODUCTION
extraction of the validated spike reference (``scratch/overland_split_spike.py`` ``win`` mode,
``run_case_win``; conservation proven to ~5e-12 and parent-verified, sanity note
``validation/sanity/overland_split_spike__2026-06-22.md`` CONSERVATION-PROOF section). The design
(LOCKED from that spike): each step solves implicit Richards ALONE (pond carried IN psi as
``max(psi,0)``, self-limiting), with the surface water redistributed by an explicit Manning
rate-limited routing sweep injected as an under-relaxed Neumann SOURCE, iterated to a Picard fixed
point. The conserved ledger is ``total = int theta dV + int max(psi,0) ds_top`` with the surface
terms on the SAME lumped vertex quadrature as the routing store's ``sum d_i A_i`` (the load-bearing
quadrature-match fix, 18% -> 5e-12).

These tests pin the production behaviour:
  1. lake-at-rest -- flat bed, uniform pond, no rain: depth AND psi hold to machine precision.
  2. reduce-to-vertical -- a laterally-uniform case reproduces a standalone RichardsProblem +
     add_ponding_bc run (the split == the validated vertical physics when there is no lateral move).
  3. matched-quadrature -- ``int max(psi,0) ds_top == sum_i d_i A_i`` to ~1e-12 (the conservation fix).
  4. conservation (the gate) -- a small/short sand-channel-in-clay storm closes
     ``|balance|/cum_rain < 1e-3``; no dt-collapse; routing ``sum d_i A_i`` resid <= 1e-12.
  5. falsification -- a deliberate 10% outflow mis-booking breaks the same balance to ~10% of
     cum_outflow (proves test 4 is a real detector, not a tautology).
"""
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import RichardsProblem, richards_bulk_residual
from pids_forward.physics.sequential_coupling import SequentialCoupledProblem

COMM = MPI.COMM_WORLD


# ====================================================================================================
# Test 1 -- LAKE AT REST: flat bed, uniform ponded depth, no rain -> depth AND psi hold to machine
# precision over many steps (no spurious lateral motion / no routing on a flat surface).
# ====================================================================================================
def test_lake_at_rest_holds_to_machine_precision():
    """A flat bed (z_b=0) flooded to a uniform ponded head, no rain, impermeable-ish base: with a
    uniform surface head H = z_b + d the routing finds NO downslope direction, so nothing moves
    laterally; and a saturated column at hydrostatic equilibrium does not infiltrate. The surface
    pond (max(psi,0)) and the full psi field must hold to machine precision over many steps."""
    LX, LY, LZ = 2.0, 1.0, 1.0
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [6, 4, 4])
    soil = VanGenuchten(theta_r=0.067, theta_s=0.45, alpha=1.0, n=1.5, Ks=0.05)
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
    prob.set_topography(lambda x: 0.0 * x[0])                 # FLAT bed
    # hydrostatic: psi = (z_top - z) below the surface, +0.05 m pond at the top (psi = 0.05 at z=LZ).
    POND = 0.05
    prob.set_initial_condition(lambda x: (LZ - x[2]) + POND)
    prob.add_rain(0.0)                                        # no rain

    psi0 = prob._rp.psi.x.array.copy()
    surf0 = prob.surface_water()
    for _ in range(30):
        conv, _it = prob.step(2e-3)
        assert conv, "lake-at-rest step did not converge"
    dpsi = float(np.max(np.abs(prob._rp.psi.x.array - psi0)))
    assert dpsi <= 1e-9, f"lake-at-rest psi drifted by {dpsi:.3e} (spurious motion)"
    dsurf = abs(prob.surface_water() - surf0)
    assert dsurf <= 1e-10, f"lake-at-rest surface pond drifted by {dsurf:.3e}"
    # and the balance is trivially closed (nothing happened).
    assert abs(prob.balance()) <= 1e-9


# ====================================================================================================
# Test 2 -- REDUCE-TO-VERTICAL: a laterally-uniform case (uniform rain on a flat bed, nothing to
# route) reproduces a standalone RichardsProblem + add_ponding_bc run to ~1e-10 -- the split IS the
# validated vertical physics when there is no lateral transport.
# ====================================================================================================
def test_reduce_to_vertical_matches_standalone_richards():
    """Sub-infiltration-capacity rain onto a FLAT column (rain < Ks, dry start): the soil absorbs
    every drop, so the surface pond ``max(psi,0)`` is identically 0 and the routing sweep is a pure
    no-op (routing all-zeros returns zeros). The sequential split must then reproduce the standalone
    vertical Richards solve -- same capped bulk, same lumped storage, same top Neumann-rain term, same
    'basic' linesearch -- node-for-node to ~1e-10 (the split IS the validated vertical physics when
    there is no surface water to transport laterally).

    (When the soil DOES pond, the 3-D box's top triangulation makes corner/edge nodes pond at slightly
    different depths -> a real mm-scale surface-head gradient the routing legitimately acts on, so the
    split is NOT expected to match a no-routing solve there; that lateral redistribution is the point
    of the coupling and is exercised by the conservation gate. This test isolates the no-transport
    reduction by keeping the surface dry.)"""
    import ufl as _ufl
    from petsc4py import PETSc as _PETSc
    from dolfinx import mesh as _dmesh
    LX, LY, LZ = 1.0, 1.0, 1.0
    NX, NY, NZ = 3, 3, 8
    soil = VanGenuchten(theta_r=0.067, theta_s=0.45, alpha=2.0, n=1.5, Ks=0.10)
    RAIN, PSI_I = 0.05, -0.8       # rain < Ks, dry start -> the soil never ponds (d == 0 throughout)
    dt, nstep = 5e-3, 20

    # --- reference: standalone Richards with a hand-built VERTEX-quadrature pond term (= the split's
    #     internal vertical discretization) ---
    msh_ref = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [NX, NY, NZ])
    petsc_basic = {**RichardsProblem._DEFAULT_PETSC_OPTIONS, "snes_linesearch_type": "basic"}
    rp = RichardsProblem(msh_ref, soil, lumped=True, petsc_options=petsc_basic)
    rp.F = richards_bulk_residual(rp.psi, rp.psi_n, rp._v, soil, rp.dt, rp.e_g,
                                  dx_storage=rp._dx_storage, quadrature_degree=8)
    rp.set_initial_condition(lambda x: PSI_I + 0.0 * x[0])
    fdim = msh_ref.topology.dim - 1
    msh_ref.topology.create_connectivity(fdim, msh_ref.topology.dim)
    tf = np.sort(_dmesh.locate_entities_boundary(
        msh_ref, fdim, lambda x: np.isclose(x[2], LZ))).astype(np.int32)
    ft = _dmesh.meshtags(msh_ref, fdim, tf, np.ones(tf.size, dtype=np.int32))
    ds_v = _ufl.Measure("ds", domain=msh_ref, subdomain_data=ft,
                        metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})(1)
    rain_c = fem.Constant(msh_ref, _PETSc.ScalarType(RAIN))
    pond = _ufl.max_value(rp.psi, 0.0)
    pond_n = _ufl.max_value(rp.psi_n, 0.0)
    rp.F = rp.F + ((pond - pond_n) / rp.dt) * rp._v * ds_v - rain_c * rp._v * ds_v
    rp._problem = None
    for _ in range(nstep):
        conv, _ = rp.step(dt)
        assert conv, "reference standalone Richards step diverged"
    psi_ref = rp.psi.x.array.copy()

    # --- the sequential split on the same column (flat bed, uniform rain, nothing to route) ---
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [NX, NY, NZ])
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
    prob.set_topography(lambda x: 0.0 * x[0])
    prob.set_initial_condition(lambda x: PSI_I + 0.0 * x[0])
    prob.add_rain(RAIN)
    for _ in range(nstep):
        conv, _ = prob.step(dt)
        assert conv, "sequential split step diverged on the vertical-reduction case"
    psi_split = prob._rp.psi.x.array.copy()

    # guard: the case must genuinely be no-route (surface stayed dry -> the reduction is meaningful).
    assert prob.surface_water() <= 1e-14, \
        f"surface ponded ({prob.surface_water():.3e}) -> not a no-transport reduction; lower the rain"
    dmax = float(np.max(np.abs(psi_split - psi_ref)))
    assert dmax <= 1e-10, f"split deviates from standalone vertical Richards by {dmax:.3e}"


# ====================================================================================================
# Test 3 -- MATCHED QUADRATURE (the load-bearing conservation fix): the surface-pond ledger
# ``int max(psi,0) ds_top`` integrated on the lumped VERTEX measure equals the routing store's
# ``sum_i d_i A_i`` to ~1e-12. A degree-8 ds would integrate max(psi,0) differently and reintroduce
# the leak; this pins the vertex-quadrature match.
# ====================================================================================================
def test_matched_quadrature_pond_ledger_equals_sum_d_A():
    """``surface_water()`` (the UFL ledger ``int max(psi,0) ds_top_vertex``) must equal the routing
    store's ``sum_i d_i A_i`` (d = max(psi,0) at the top dofs, A_i the lumped surface control areas
    from the SAME top-facet graph the routing uses) to ~1e-12, for an arbitrary ponded field. This
    is the quadrature-match that took the spike's |balance|/rain from 1.9e-3 to 5.3e-12."""
    LX, LY, LZ = 3.0, 2.0, 1.0
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [8, 5, 4])
    soil = VanGenuchten(theta_r=0.067, theta_s=0.45, alpha=1.0, n=1.5, Ks=0.05)
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
    prob.set_topography(lambda x: 0.04 * (LX - x[0]))
    # an arbitrary spatially-varying ponded head at the top, dry below (psi<0 interior).
    prob.set_initial_condition(
        lambda x: np.where(np.isclose(x[2], LZ), 0.02 + 0.03 * np.sin(3.0 * x[0]) ** 2, -0.4))

    ledger = prob.surface_water()           # triggers the routing-graph build (A_i, top-facet areas)
    A_i = prob._A_i
    top_dofs = prob._top_dofs_arr
    d = np.maximum(prob._rp.psi.x.array, 0.0)
    sum_dA = float(np.sum(d[top_dofs] * A_i[top_dofs]))
    assert abs(ledger - sum_dA) <= 1e-12, \
        f"pond ledger {ledger:.12e} != sum_i d_i A_i {sum_dA:.12e} (quadrature mismatch)"


# ====================================================================================================
# A SMALL/SHORT sand-channel-in-clay storm (the gate case, shrunk from the spike for suite speed):
# coarse mesh, short horizon. Shared by the conservation gate (test 4) and the falsification (test 5).
# ====================================================================================================
def _small_sand_channel():
    """Coarse/short version of the spike's sand-channel-in-clay storm fixture (run_case_win gate
    case 1, _sand_channel_setup), shrunk for suite speed: smaller box, coarser mesh, short horizon.
    A sand swale embedded in clay; a storm ponds, routes to the channel, infiltrates the sand and
    drains via a sand-interface GHB. Returns the kwargs for build_* + run."""
    LX, LY, LZ = 6.0, 3.0, 1.0
    NX, NY, NZ = 12, 7, 4
    S0 = 0.04
    X_CH, W_CH = 3.0, 0.6
    Z_SAND_BASE = LZ - 0.4
    D_CH, SY = 0.30, 0.06
    X_BERM, W_BERM, B_H = 4.0, 0.45, 0.30
    PSI_I = -0.30
    RAIN, STORM_DUR = 0.15, 0.10
    SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)
    CLAY = VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.048)
    tol = 1e-6

    class ClaySandChannel:
        def __init__(self, mesh):
            self.sand, self.clay = SAND, CLAY
            xx = ufl.SpatialCoordinate(mesh)
            self._in_sand = ufl.And(ufl.lt(abs(xx[0] - X_CH), W_CH), ufl.ge(xx[2], Z_SAND_BASE))
            self.Ks = SAND.Ks
            self.theta_r, self.theta_s = CLAY.theta_r, CLAY.theta_s

        def theta_ufl(self, psi):
            return ufl.conditional(self._in_sand, self.sand.theta_ufl(psi), self.clay.theta_ufl(psi))

        def K_ufl(self, psi):
            return ufl.conditional(self._in_sand, self.sand.K_ufl(psi), self.clay.K_ufl(psi))

    def topo(x):
        z_main = S0 * (LX - x[0])
        swale = (D_CH + SY * (LY - x[1])) * np.exp(-(((x[0] - X_CH) / W_CH) ** 2))
        berm = B_H * np.exp(-(((x[0] - X_BERM) / W_BERM) ** 2))
        return z_main - swale + berm

    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [NX, NY, NZ])
    soil = ClaySandChannel(msh)
    outlets = [(lambda x: np.isclose(x[0], LX), S0),
               (lambda x: np.isclose(x[1], 0.0) & (np.abs(x[0] - X_CH) < W_CH + tol), SY)]
    drain_loc = (lambda x: np.isclose(x[1], 0.0) & (np.abs(x[0] - X_CH) < W_CH + tol)
                 & (x[2] >= Z_SAND_BASE - tol))
    return dict(msh=msh, soil=soil, topo=topo, ztop=LZ, psi_i=PSI_I, rain=RAIN,
                storm=STORM_DUR, t_end=0.30, outlets=outlets,
                drain_loc=drain_loc, drain_C=2.0, drain_head=Z_SAND_BASE - 0.1)


def _build_small_sand_problem():
    """Assemble a SequentialCoupledProblem for the small sand-channel storm (no rain yet)."""
    cfg = _small_sand_channel()
    prob = SequentialCoupledProblem(cfg["msh"], cfg["soil"], n_man=0.05)
    prob.set_topography(cfg["topo"])
    prob.set_initial_condition(lambda x: cfg["psi_i"] + 0.0 * x[0])
    for (loc, slope) in cfg["outlets"]:
        prob.add_outflow_bc(loc, slope)
    prob.add_drainage_bc(cfg["drain_loc"], cfg["drain_C"], cfg["drain_head"])
    rain = prob.add_rain(cfg["rain"])
    return prob, cfg, rain


def _run_small_sand_storm(prob, cfg, rain):
    """March the small storm to t_end with the band controller; rain off after storm_dur. Returns
    (n_accepted, routing_resid_max)."""
    t, nstep, dt = 0.0, 0, 1e-3
    routing_resid_max = 0.0
    while t < cfg["t_end"] - 1e-12:
        h = min(dt, cfg["t_end"] - t)
        if t < cfg["storm"] - 1e-12 and t + h > cfg["storm"]:
            h = cfg["storm"] - t
        rain.value = cfg["rain"] if t < cfg["storm"] - 1e-12 else 0.0
        conv, it = prob.step(h)
        if not conv:
            dt *= 0.5
            assert dt > 1e-9, f"DT COLLAPSE at t={t:.5f}"
            continue
        routing_resid_max = max(routing_resid_max, prob.last_routing_resid)
        t += h
        nstep += 1
        if it <= 4:
            dt = min(dt * 1.4, 0.03)
        elif it >= 12:
            dt = dt * 0.7
    return nstep, routing_resid_max


# ====================================================================================================
# Test 4 -- CONSERVATION (the gate): the small/short sand-channel storm closes
# |balance|/cum_rain < 1e-3; no dt-collapse (completes); routing sum(d*A) resid <= 1e-12.
# ====================================================================================================
def test_conservation_small_sand_channel_storm():
    """The shrunk sand-channel-in-clay storm conserves: ``|balance|/cum_rain < 1e-3`` where
    ``balance = d(total) - (cum_rain - cum_outflow - cum_drainage + cum_handoff_imbalance)`` with
    every term measured independently (no fudge bucket). The storm completes without dt-collapse and
    the routing sweep conserves its store ``sum_i d_i A_i`` to <= 1e-12 each step."""
    prob, cfg, rain = _build_small_sand_problem()
    nstep, routing_resid = _run_small_sand_storm(prob, cfg, rain)
    assert nstep > 0, "storm did not advance"
    assert routing_resid <= 1e-12, f"routing store sum(d*A) resid {routing_resid:.3e} > 1e-12"
    assert prob.cum_rain > 0.0, "no rain fell -> vacuous conservation check"
    assert prob.cum_outflow > 0.0, "no surface outflow -> storm did not actually route to the outlet"
    bal_frac = abs(prob.balance()) / prob.cum_rain
    assert bal_frac < 1e-3, (
        f"global balance did not close: |bal|/cum_rain = {bal_frac:.3e} "
        f"(|bal|={abs(prob.balance()):.3e}, cum_rain={prob.cum_rain:.3e}, "
        f"cum_outflow={prob.cum_outflow:.3e}, cum_drainage={prob.cum_drainage:.3e})")


# ====================================================================================================
# Test 5 -- FALSIFICATION: a deliberate 10% outflow mis-booking breaks the SAME balance to ~10% of
# cum_outflow over cum_rain -> test 4 is a genuine detector, not a tautology (mirrors PIDS_WIN_LEAK).
# ====================================================================================================
def test_falsification_misbooked_outflow_breaks_balance():
    """Inject a deliberate 10% over-booking of outflow (``outflow_leak_frac=0.1``): the same global
    balance must then FAIL to close, reporting ~10% of cum_outflow over cum_rain. This proves the
    ~1e-3 close in test 4 is a real conservation detector, not a structural artifact (the spike's
    PIDS_WIN_LEAK falsification hook)."""
    LEAK = 0.10
    prob, cfg, rain = _build_small_sand_problem()
    prob.outflow_leak_frac = LEAK                  # deliberately mis-book 10% extra outflow
    nstep, _routing_resid = _run_small_sand_storm(prob, cfg, rain)
    assert nstep > 0 and prob.cum_outflow > 0.0
    # the broken balance = LEAK * the APPLIED outflow over cum_rain (the mis-booked excess). cum_outflow
    # holds the BOOKED (1+LEAK)-scaled value, so the applied outflow is cum_outflow/(1+LEAK).
    applied_outflow = prob.cum_outflow / (1.0 + LEAK)
    expected = LEAK * applied_outflow / prob.cum_rain
    bal_frac = abs(prob.balance()) / prob.cum_rain
    assert bal_frac > 1e-2, \
        f"falsification did not trip the balance: |bal|/cum_rain = {bal_frac:.3e} (expected ~{expected:.3e})"
    assert bal_frac == pytest.approx(expected, rel=0.05), \
        f"broken balance {bal_frac:.4e} != expected LEAK*applied_outflow {expected:.4e}"


# ====================================================================================================
# Test 6 -- ADVANCE (band dt-controller + storm hyetograph): the same small storm marched by the
# built-in ``advance`` (fix #4 band controller, rain on/off at storm_dur) completes without collapse
# and closes conservation < 1e-3. Exercises the band controller + storm switching path.
# ====================================================================================================
def test_advance_band_controller_runs_storm_and_conserves():
    """``advance(t_end, dt, storm_dur=..., storm_rain=...)`` marches the small sand-channel storm with
    the BAND dt-controller (it<=4 grow, it>=12 shrink) and the rain switched off at storm_dur. It must
    complete (no dt-collapse / no max_steps) and close the global balance < 1e-3 -- the same gate as
    the manual loop, via the production driver (so the band controller + hyetograph switching are
    covered, not only step())."""
    prob, cfg, rain = _build_small_sand_problem()
    nstep = prob.advance(cfg["t_end"], 1e-3, storm_dur=cfg["storm"], storm_rain=cfg["rain"],
                         dt_max=0.03, ctrl_low=4, ctrl_high=12)
    assert nstep > 0, "advance did not take any accepted steps"
    assert prob.cum_rain > 0.0 and prob.cum_outflow > 0.0
    bal_frac = abs(prob.balance()) / prob.cum_rain
    assert bal_frac < 1e-3, f"advance balance did not close: |bal|/cum_rain = {bal_frac:.3e}"
