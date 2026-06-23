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
# Test 5 -- FALSIFICATION (the OUTFLOW-CHANNEL detector specifically): a deliberate 10% outflow
# mis-booking breaks the SAME balance to ~10% of cum_outflow over cum_rain -> test 4's close is a
# genuine detector for the OUTFLOW channel, not a tautology (mirrors PIDS_WIN_LEAK). SCOPE: this test
# falsifies ONLY the surface-outflow booking (``outflow_leak_frac`` scales just cum_outflow); the OTHER
# conserved channels are detector-guarded elsewhere -- the DRAINAGE/sink channel by the conservation
# gate (test 4) + per-sink accounting (test 10) and the pond-ledger/quadrature channel by the
# matched-quadrature pin (test 3). Together those make balance() a non-vacuous detector across channels;
# this one isolates the outflow channel.
# ====================================================================================================
def test_falsification_misbooked_outflow_breaks_balance():
    """OUTFLOW-CHANNEL falsification (not a universal leak detector -- it targets the surface-outflow
    booking specifically; the drainage channel is covered by the conservation gate test 4 + per-sink
    accounting test 10, and the matched-quadrature pond ledger by test 3). Inject a deliberate 10%
    over-booking of OUTFLOW (``outflow_leak_frac=0.1``, which scales only cum_outflow): the same global
    balance must then FAIL to close, reporting ~10% of cum_outflow over cum_rain. This proves the ~1e-3
    close in test 4 is a real conservation detector for the outflow channel, not a structural artifact
    (the spike's PIDS_WIN_LEAK falsification hook)."""
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


# ====================================================================================================
# B6 SINKS -- the extracted surface/subsurface sinks (interior tile drain + grate inlet) + per-sink
# accounting + the F2 evaluation-state contract + the one-shot ledger-baseline fix.
# ====================================================================================================
from pids_forward.physics.coupling import CoupledProblem   # noqa: E402  (monolith cross-check, B6)


# ====================================================================================================
# Test 7 -- INTERIOR DRAIN agrees with the MONOLITH where both schemes are valid. A CLOSED box (no
# rain, no outlet, dry surface so the routing sweep is a pure no-op) with a saturated band over an
# interior tile drain is purely SUBSURFACE: the sequential split (Richards-alone, pond-in-psi=0) and
# the monolithic CoupledProblem solve the SAME drainage problem, so cum_drainage must agree closely.
# ====================================================================================================
def test_interior_drain_agrees_with_monolith_closed_box():
    """A closed 3-D box, water table partway up, an interior tile drain near the base, NO rain / NO
    outlet (the surface never ponds -> the routing sweep is a pure no-op). This is a purely
    subsurface drainage problem where BOTH schemes are valid and there is NO lateral transport for
    the sequential time-lag to act on -- so SequentialCoupledProblem + add_interior_drain must
    reproduce CoupledProblem + add_interior_drain to a few percent on the drained volume. (Not
    bit-identical: the two march with independent dt sequences and the sequential path re-snapshots
    psi_n each Picard iterate, but the DISCHARGED VOLUME over a fixed horizon is a robust physical
    invariant -- they drain the same band through the same conductance against the same head.)"""
    LX, LY, LZ = 1.0, 1.0, 1.0
    NX, NY, NZ = 4, 4, 8
    soil = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
    WT, C_DENS, T_END, DT = 0.5, 4.0, 0.30, 2e-3
    drain_loc = lambda x: x[2] < 0.13
    ic = lambda x: WT - x[2]                                  # water table at z=0.5 (psi=0 there)

    # --- monolith reference ---
    mono = CoupledProblem(dmesh.create_box(COMM, [[0, 0, 0], [LX, LY, LZ]], [NX, NY, NZ]), soil)
    mono.set_initial_condition(ic, d_value=0.0)
    mono.add_interior_drain(drain_loc, conductance_density=C_DENS, drain_head=0.0)
    mono.advance(t_end=T_END, dt=DT, dt_max=2e-2)
    assert mono.cum_drainage > 1e-4, "monolith drain did not discharge -> vacuous comparison"

    # --- sequential split, same physics ---
    seq = SequentialCoupledProblem(
        dmesh.create_box(COMM, [[0, 0, 0], [LX, LY, LZ]], [NX, NY, NZ]), soil, n_man=0.05)
    seq.set_topography(lambda x: 0.0 * x[0])
    seq.set_initial_condition(ic)
    seq.add_interior_drain(drain_loc, conductance_density=C_DENS, drain_head=0.0)
    seq.advance(t_end=T_END, dt=DT, dt_max=2e-2)
    assert seq.cum_drainage > 1e-4, "sequential drain did not discharge"
    # the surface stayed dry (genuinely a no-route subsurface problem -> the comparison is meaningful).
    assert seq.surface_water() <= 1e-12, "surface ponded -> not a pure subsurface drainage comparison"

    rel = abs(seq.cum_drainage - mono.cum_drainage) / mono.cum_drainage
    assert rel < 0.05, (f"interior-drain discharge disagrees with the monolith by {rel:.3%} "
                        f"(seq={seq.cum_drainage:.5e}, mono={mono.cum_drainage:.5e})")
    # and the sequential balance still closes around the new subsurface sink.
    bal = abs(seq.balance())
    assert bal / seq.cum_drainage < 1e-6, f"sequential balance broke with interior drain: {bal:.3e}"


# ====================================================================================================
# Test 8 -- SURFACE INLET removes ponded water at its footprint, books the intake, and the global
# balance STILL closes (|bal|/cum_rain < 1e-3 with the inlet active) -- the key conservation guard.
# ====================================================================================================
def test_surface_inlet_removes_pond_books_and_conserves():
    """A grate inlet over part of the top surface removes ponded water (q_in = intake_coeff*d) during
    the surface update (F2: a SURFACE sink on the post-Richards pond). It must (a) actually lower the
    surface store vs a no-inlet twin, (b) book the removed volume into cum_drainage / the
    surface_inlet per-sink split, and (c) keep the global balance closed |bal|/cum_rain < 1e-3 (the
    inlet removal is matched-quadrature consistent with the ledger). Coarse/short for suite speed."""
    LX, LY, LZ = 2.0, 1.0, 1.0
    msh = dmesh.create_box(COMM, [[0, 0, 0], [LX, LY, LZ]], [6, 4, 4])
    # tight clay so the storm ponds (rain >> Ks); a deep unsaturated buffer (no full saturation).
    soil = VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.05)

    def _run(with_inlet):
        prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
        prob.set_topography(lambda x: 0.02 * (LX - x[0]))      # gentle slope toward x=0
        prob.set_initial_condition(lambda x: -0.30 + 0.0 * x[0])
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 0.0), 0.02)
        rain = prob.add_rain(0.0)
        inlet_C = None
        if with_inlet:
            # a grate over the down-slope quarter (x < LX/4) of the top surface.
            inlet_C = prob.add_surface_inlet(lambda x: x[0] < LX / 4.0 + 1e-9, intake_coeff=20.0)
        prob.advance(0.30, 1e-3, storm_dur=0.10, storm_rain=0.20, dt_max=0.03)
        return prob, inlet_C

    base, _ = _run(with_inlet=False)
    prob, inlet_C = _run(with_inlet=True)
    assert prob.cum_rain > 0.0, "no rain fell"

    # (a) the inlet genuinely removed surface water (less pond remains than the no-inlet twin).
    assert prob.cum_sinks["surface_inlet"][0] > 1e-5, "the inlet booked ~no intake (footprint dry?)"
    assert prob.surface_water() < base.surface_water() - 1e-6, \
        "the inlet did not lower the surface store vs the no-inlet twin"

    # (b) the booked intake is in cum_drainage (the flat per-sink sum includes it).
    assert prob.cum_drainage == pytest.approx(
        sum(prob.cum_sinks["surface_inlet"]) + sum(prob.cum_sinks.get("ghb", [])), rel=1e-9), \
        "surface-inlet intake is not reflected in cum_drainage"

    # (c) THE CONSERVATION GUARD: the balance still closes with the inlet active.
    bal_frac = abs(prob.balance()) / prob.cum_rain
    assert bal_frac < 1e-3, (f"balance did not close with the inlet active: |bal|/cum_rain = "
                             f"{bal_frac:.3e} (cum_inlet={sum(prob.cum_sinks['surface_inlet']):.3e})")


# ====================================================================================================
# Test 8b -- INLET CLAMP robustness: a HUGE intake coefficient (intake_coeff*dt >> 1) would drain a
# footprint node in well under one step. The per-node clamp keeps psi >= 0 AND the books record the
# volume ACTUALLY removed, so balance() stays closed UNCONDITIONALLY (not just in the intake*dt<1
# regime) -- pins that the inlet bookkeeping is conservative under the clamp.
# ====================================================================================================
def test_surface_inlet_clamp_conserves_with_huge_coefficient():
    """With intake_coeff*dt >> 1 the linear intake would remove more than the available pond in one
    step; the per-node clamp caps removal at the pond (psi never crosses 0) and the booked intake is
    the volume actually removed, so the global balance still closes |bal|/cum_rain < 1e-3 and the
    booked cum_inlet never exceeds the rain that fell. Pins the unconditional conservation claim."""
    LX, LY, LZ = 2.0, 1.0, 1.0
    msh = dmesh.create_box(COMM, [[0, 0, 0], [LX, LY, LZ]], [6, 4, 4])
    # the same fixture shape as Test 8 (loam-ish clay that ponds transiently, gentle slope for routing
    # relief), but a HUGE intake_coeff so the clamp bites from the first ponded step: intake_coeff*dt =
    # 2000*1e-3 = 2.0 > 1 at the dt floor (and larger as dt grows), so the unclamped linear intake would
    # remove > the available pond -> the per-node clamp caps it at the pond every ponded step.
    soil = VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.05)
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
    prob.set_topography(lambda x: 0.02 * (LX - x[0]))         # gentle slope (routing relief, as Test 8)
    prob.set_initial_condition(lambda x: -0.30 + 0.0 * x[0])
    prob.add_outflow_bc(lambda x: np.isclose(x[0], 0.0), 0.02)
    # a grate over the down-slope quarter with a huge coefficient -> intake_coeff*dt >> 1 (clamp bites).
    prob.add_surface_inlet(lambda x: x[0] < LX / 4.0 + 1e-9, intake_coeff=2000.0)
    prob.add_rain(0.0)
    prob.advance(0.30, 1e-3, storm_dur=0.10, storm_rain=0.20, dt_max=5e-3)
    assert prob.cum_rain > 0.0
    assert sum(prob.cum_sinks["surface_inlet"]) > 1e-4, "the grate booked ~no intake despite ponding"
    assert np.all(prob._rp.psi.x.array >= -1e-12), "the clamp let psi cross 0 (over-drained a node)"
    bal_frac = abs(prob.balance()) / prob.cum_rain
    assert bal_frac < 1e-3, f"balance broke under the clamping inlet: |bal|/cum_rain = {bal_frac:.3e}"
    # the inlet cannot remove more water than rained in (a sanity bound on the booked intake).
    assert sum(prob.cum_sinks["surface_inlet"]) <= prob.cum_rain + 1e-9, "inlet booked > rain (impossible)"


# ====================================================================================================
# Test 9 -- F2 EVALUATION-STATE pinned: the surface inlet is evaluated on the POST-Richards ponded
# field. A step where the Richards solve changes the pond at the inlet footprint (infiltration draws
# it down) books a DIFFERENT total than the pre-step pond would -- so the booked intake must match the
# post-solve pond, NOT the entry pond. Locks the F2 choice (would change the books if evaluated wrong).
# ====================================================================================================
def test_surface_inlet_f2_evaluated_on_post_richards_pond():
    """Pin F2: the grate inlet is a SURFACE sink evaluated on the POST-Richards pond (before its own
    removal), NOT on the pre-solve entry pond. Start with a uniform pond over a permeable soil and
    take ONE step with no rain: the Richards solve infiltrates some pond, so the post-Richards depth
    is strictly LESS than the entry depth. The booked inlet rate (last_sinks['surface_inlet']) must
    therefore be strictly LESS than intake_coeff*sum_i d_entry,i A_i (what a pre-solve evaluation would
    book) -- this is the discriminating check. We also pin it EXACTLY: the inlet then removes a
    (1 - intake_coeff*dt) fraction of that pond uniformly, so the booked rate reconstructs from the
    final pond as intake_coeff*sum d_final A / (1 - intake_coeff*dt). Evaluating the inlet pre-solve
    would book the larger entry number; this test would then fail on the strict-inequality assert."""
    LX, LY, LZ = 1.0, 1.0, 1.0
    msh = dmesh.create_box(COMM, [[0, 0, 0], [LX, LY, LZ]], [3, 3, 6])
    soil = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.40)  # permeable: pond drops
    POND, C_IN, DT = 0.08, 5.0, 5e-3
    assert C_IN * DT < 1.0                                    # unclamped uniform removal (reconstructible)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
    prob.set_topography(lambda x: 0.0 * x[0])                 # FLAT: no routing moves the pond
    # uniform pond at the top, dry below (psi<0 interior) -> a real pond the soil will draw down.
    prob.set_initial_condition(lambda x: np.where(np.isclose(x[2], LZ), POND, -0.5))
    inlet_C = prob.add_surface_inlet(lambda x: np.ones_like(x[0], dtype=bool), intake_coeff=C_IN)

    prob._ensure_built()                                     # build the routing graph (A_i, top areas)
    top = prob._top_dofs_arr
    A_i = prob._A_i
    d_entry = np.maximum(prob._rp.psi.x.array.copy()[top], 0.0)
    rate_entry = C_IN * float(np.sum(d_entry * A_i[top]))     # what a PRE-solve evaluation would book

    conv, _it = prob.step(DT)
    assert conv, "F2 inlet step did not converge"
    booked = prob.last_sinks["surface_inlet"][0]

    # the discriminating F2 check: the soil infiltrated before the inlet saw the pond, so the booked
    # rate (read on the post-Richards pond) is STRICTLY LESS than the entry-pond rate. A pre-solve
    # evaluation would have booked exactly rate_entry -> this assert would fail.
    assert booked < rate_entry - 1e-6, (
        f"inlet booked {booked:.6e} >= the entry-pond rate {rate_entry:.6e} -- evaluated pre-solve, "
        "not on the post-Richards pond (F2 violated, or the pond did not draw down: raise Ks/POND)")
    # and pin it EXACTLY: the inlet removed (1 - C_IN*dt) of the post-Richards pond uniformly, so the
    # booked rate == intake_coeff * sum d_final A / (1 - C_IN*dt) (reconstructs the pre-removal pond).
    d_final = np.maximum(prob._rp.psi.x.array.copy()[top], 0.0)
    rate_recon = C_IN * float(np.sum(d_final * A_i[top])) / (1.0 - C_IN * DT)
    assert booked == pytest.approx(rate_recon, rel=1e-6, abs=1e-12), \
        f"inlet booked {booked:.6e} != post-Richards-pond reconstruction {rate_recon:.6e} (F2 detail)"


# ====================================================================================================
# Test 10 -- PER-SINK ACCOUNTING: with a GHB drain + an interior drain + a surface inlet all active,
# sink_rates() exposes all three keys in add-order and the flat per-step booked total equals
# last_drainage; the cumulative flat sum equals cum_drainage. (The split sums to the booked total.)
# ====================================================================================================
def test_per_sink_accounting_flat_sum_equals_booked_drainage():
    """Three sinks active (GHB boundary + interior tile drain + surface inlet): sink_rates() returns
    the three keys, each a list in add-order, and the flat sum of the per-step booked rates equals
    last_drainage while the flat sum of cum_sinks equals cum_drainage. Locks that the per-sink split
    is exhaustive (no sink booked outside the split) and sums to the headline drainage figure."""
    LX, LY, LZ = 2.0, 1.0, 1.0
    msh = dmesh.create_box(COMM, [[0, 0, 0], [LX, LY, LZ]], [6, 3, 5])
    soil = VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.05)
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
    prob.set_topography(lambda x: 0.02 * (LX - x[0]))
    prob.set_initial_condition(lambda x: -0.10 + 0.0 * x[0])  # near-saturated: the GHB/drain discharge
    prob.add_drainage_bc(lambda x: np.isclose(x[2], 0.0), conductance=2.0, external_head=-0.2)
    prob.add_interior_drain(lambda x: x[2] < 0.25, conductance_density=4.0, drain_head=0.0)
    prob.add_surface_inlet(lambda x: x[0] < LX / 3.0 + 1e-9, intake_coeff=15.0)
    prob.add_rain(0.20)

    # take a few storm steps so all three sinks are genuinely active.
    for _ in range(8):
        conv, _it = prob.step(5e-3)
        assert conv, "per-sink accounting step did not converge"

    rates = prob.sink_rates()
    assert set(rates.keys()) == {"ghb", "interior_drain", "surface_inlet"}, \
        f"sink_rates keys != the three sink kinds: {sorted(rates)}"
    assert len(rates["ghb"]) == 1 and len(rates["interior_drain"]) == 1 \
        and len(rates["surface_inlet"]) == 1, "each kind should hold one sink (add-order)"
    # sink_rates() flat sum == the booked drainage total (the task's headline invariant).
    flat_rates = sum(v for lst in rates.values() for v in lst)
    assert flat_rates == pytest.approx(prob.last_drainage, rel=1e-9, abs=1e-12), \
        f"sink_rates() flat sum {flat_rates:.6e} != last_drainage {prob.last_drainage:.6e}"

    # the per-step booked split sums to last_drainage; the cumulative split sums to cum_drainage.
    flat_last = sum(v for lst in prob.last_sinks.values() for v in lst)
    flat_cum = sum(v for lst in prob.cum_sinks.values() for v in lst)
    assert flat_last == pytest.approx(prob.last_drainage, rel=1e-9, abs=1e-12), \
        f"sum(last_sinks)={flat_last:.6e} != last_drainage={prob.last_drainage:.6e}"
    assert flat_cum == pytest.approx(prob.cum_drainage, rel=1e-9, abs=1e-12), \
        f"sum(cum_sinks)={flat_cum:.6e} != cum_drainage={prob.cum_drainage:.6e}"
    # every sink genuinely discharged (not a vacuous all-zero sum).
    assert prob.cum_sinks["ghb"][0] > 0 and prob.cum_sinks["interior_drain"][0] > 0 \
        and prob.cum_sinks["surface_inlet"][0] > 0, "a sink booked nothing -> weak accounting check"


# ====================================================================================================
# Test 11 -- BASELINE FOOTGUN: the ledger baseline (_w0/_surf0) is a ONE-SHOT snapshot taken on the
# FIRST build, never re-taken on later rebuilds -- so a setup change AFTER stepping cannot silently
# corrupt balance() by re-snapshotting the baseline at the (already advanced) current state.
# ====================================================================================================
def test_ledger_baseline_is_one_shot_not_re_snapshotted():
    """The reviewer's Minor footgun: _w0/_surf0 must be captured EXACTLY ONCE (first build) and never
    re-taken on a rebuild. Build, step a storm so the state advances well away from the IC, then call
    a setup mutator (add_drainage_bc) that flips _built=False and forces a rebuild. The baseline
    (_w0, _surf0) must be UNCHANGED by that rebuild (if it re-snapshotted at the advanced state,
    d(total) would collapse toward 0 and balance() would silently report a bogus near-zero residual,
    masking a real leak). We pin the baseline values directly across the post-step rebuild."""
    prob, cfg, rain = _build_small_sand_problem()
    prob._ensure_built()
    w0_0, surf0_0 = prob._w0, prob._surf0          # the ONE-SHOT baseline, captured at the IC
    assert w0_0 is not None

    # advance a storm so the live state moves far from the IC.
    prob.advance(cfg["t_end"], 1e-3, storm_dur=cfg["storm"], storm_rain=cfg["rain"], dt_max=0.03)
    assert prob.total_water() != pytest.approx(w0_0 + surf0_0, rel=1e-3), \
        "state did not move away from the IC -> the footgun is not exercised"

    # a setup mutation AFTER stepping: flips _built=False -> next _ensure_built rebuilds the forms.
    prob.add_drainage_bc(lambda x: np.isclose(x[1], cfg["msh"].geometry.x[:, 1].max()),
                         conductance=0.0, external_head=-1.0)
    prob._ensure_built()                            # the rebuild must NOT re-snapshot the baseline

    assert prob._w0 == w0_0 and prob._surf0 == surf0_0, (
        f"ledger baseline was RE-SNAPSHOTTED on rebuild ( _w0 {w0_0:.6e}->{prob._w0:.6e}, "
        f"_surf0 {surf0_0:.6e}->{prob._surf0:.6e} ) -- a setup change after a step corrupts balance()")


# ====================================================================================================
# Test 11b -- set_initial_condition FOOTGUN raises LOUDLY: re-setting the IC AFTER the baseline is
# snapshotted (first build) or after stepping would silently keep the stale _w0/_surf0 baseline and
# corrupt balance(). The mutator must RAISE instead of silently corrupting -- and the normal
# before-first-build usage must still work.
# ====================================================================================================
def test_set_initial_condition_after_build_or_step_raises():
    """``set_initial_condition`` is legal ONLY before the first build/step (the IC defines the one-shot
    ledger baseline). Calling it after ``_ensure_built`` snapshots the baseline -- or after a step --
    must raise a clear RuntimeError (not silently corrupt balance()). The normal pre-build call works."""
    LX, LY, LZ = 2.0, 1.0, 1.0
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [4, 3, 4])
    soil = VanGenuchten(theta_r=0.067, theta_s=0.45, alpha=1.0, n=1.5, Ks=0.05)

    # (1) the normal pre-build usage works (no raise), and can be set repeatedly before the build.
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
    prob.set_topography(lambda x: 0.0 * x[0])
    prob.set_initial_condition(lambda x: (LZ - x[2]) - 0.2)
    prob.set_initial_condition(lambda x: (LZ - x[2]) + 0.05)  # still pre-build -> allowed
    prob.add_rain(0.0)

    # (2) after the first build (baseline snapshotted), re-setting the IC RAISES.
    prob._ensure_built()
    assert prob._w0 is not None, "baseline not snapshotted by the build (test precondition)"
    with pytest.raises(RuntimeError, match="set_initial_condition"):
        prob.set_initial_condition(lambda x: (LZ - x[2]) + 0.10)

    # (3) after stepping it also RAISES (the _t > 0 guard).
    conv, _ = prob.step(2e-3)
    assert conv
    with pytest.raises(RuntimeError, match="set_initial_condition"):
        prob.set_initial_condition(lambda x: (LZ - x[2]) + 0.10)


# ====================================================================================================
# Test 12 -- ROUTE_SUBSTEPS advances transport (the transport-RATE calibration knob, record
# validation/sanity/overland_transport_calibration__2026-06-23.md). A single Manning sweep per Richards
# step under-resolves the intra-step surface travel and throttles lateral transport; sub-stepping the
# sweep (route_substeps sweeps each over dt/nsub) marches the full intra-step distance and moves water
# downslope/out FASTER. This pins (a) the knob genuinely speeds transport (rs=4 exports more / leaves
# less pond than rs=1 over the same horizon) and (b) it is conservation-NEUTRAL (both close to ~1e-12).
# ====================================================================================================
def _pond_release(route_substeps):
    """A clean pond-release transport isolation (mirrors the calibration harness, shrunk for speed):
    a SATURATED hydrostatic column carrying a uniform COMFORTABLY-POSITIVE pond in psi, NO rain, on a
    down-slope bed with a downslope outlet. Infiltration is ~off (the soil is full + a deep buffer
    dodges the no-Ss singularity), so the pond just ROUTES downslope to the outlet -- a pure lateral
    transport race. Marched over a SHORT fixed-dt horizon that leaves the pond only PARTIALLY drained
    (~few % remaining), which is where the single-sweep throttle vs the sub-stepped rate differ most.
    Returns the built+marched problem (surface_water remaining + cum_outflow + balance read off it).
    FLAT-top mesh; z_b carries the slope."""
    LX, LY, LZ = 20.0, 12.0, 1.5
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [6, 5, 3])
    # saturated loam, small Ks so the (identical) leak is tiny; deep buffer below the pond.
    soil = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.05)
    SY, POND = 0.02, 0.05
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05, route_substeps=route_substeps)
    prob.set_topography(lambda x: SY * (LY - x[1]))          # down-slope toward y=LY (the outlet edge)
    # saturated hydrostatic column + a uniform +POND pond carried IN psi (psi_top = POND > 0).
    prob.set_initial_condition(lambda x: POND + (LZ - x[2]))
    prob.add_rain(0.0)                                       # no rain -> pure release/route
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)
    # march a SHORT horizon with FIXED dt (the race is over the same fixed dt sequence for both rs);
    # t_end=2e-3 leaves ~5% (rs=1) vs ~0.5% (rs=4) -> a large, unambiguous throttle gap.
    t, dt, t_end = 0.0, 5e-4, 2e-3
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        conv, _it = prob.step(h)
        assert conv, f"pond-release (rs={route_substeps}) step did not converge at t={t:.4f}"
        t += h
    return prob


def test_route_substeps_advances_transport_and_conserves():
    """route_substeps=4 must move the released pond downslope/out FASTER than route_substeps=1 over the
    same fixed-dt horizon (substantially less surface water REMAINING and more cumulative OUTFLOW), AND
    both must conserve to ~1e-12 (the knob is a transport-RATE lever, conservation-neutral). This pins
    the calibration fix: a single sweep is throttled ~40-50x; sub-stepping closes the intra-step travel
    gap (record validation/sanity/overland_transport_calibration__2026-06-23.md)."""
    p1 = _pond_release(route_substeps=1)
    p4 = _pond_release(route_substeps=4)

    surf1, surf4 = p1.surface_water(), p4.surface_water()
    out1, out4 = p1.cum_outflow, p4.cum_outflow

    # both genuinely routed some water to the outlet (not a vacuous all-dammed comparison).
    assert out1 > 0.0 and out4 > 0.0, f"no outflow routed (out1={out1:.3e}, out4={out4:.3e})"
    # (a) sub-stepping SUBSTANTIALLY speeds transport: it exports markedly MORE and leaves markedly
    # LESS pond at the same horizon (not a marginal +epsilon -- the throttle gap is large here:
    # measured out4/out1 ~3.5x, surf4 ~10x less than surf1). Require a real margin, not just a sign.
    assert out4 > 1.5 * out1, (
        f"route_substeps=4 did not export substantially more than rs=1 (out4={out4:.6e}, "
        f"out1={out1:.6e}, ratio={out4/out1:.2f} <= 1.5x) -- the sub-step transport knob is weak")
    assert surf4 < 0.5 * surf1, (
        f"route_substeps=4 did not drain the swale substantially faster than rs=1 (surf4={surf4:.6e} "
        f"!< 0.5*surf1={0.5*surf1:.6e})")
    # (b) conservation is route_substeps-INDEPENDENT: both close the global balance to ~1e-12.
    tot0 = p1._w0 + p1._surf0                                # same IC for both runs
    for tag, p in (("rs=1", p1), ("rs=4", p4)):
        bal_frac = abs(p.balance()) / tot0
        assert bal_frac < 1e-9, f"pond-release ({tag}) balance did not close: |bal|/total0 = {bal_frac:.3e}"


# ====================================================================================================
# Test 13 -- route_substeps VALIDATION: must be >= 1 (a sub-sweep count); 0 / negative raise.
# ====================================================================================================
def test_route_substeps_validation_rejects_below_one():
    """The constructor rejects route_substeps < 1 (0 and negative) with a ValueError -- it is the
    number of Manning sub-sweeps per Richards step, so it must be at least 1 (the single-sweep base)."""
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [2, 2, 2])
    soil = VanGenuchten(theta_r=0.067, theta_s=0.45, alpha=1.0, n=1.5, Ks=0.05)
    for bad in (0, -1, -4):
        with pytest.raises(ValueError, match="route_substeps"):
            SequentialCoupledProblem(msh, soil, route_substeps=bad)
    # the valid base (1) and the new default (4) construct fine.
    SequentialCoupledProblem(msh, soil, route_substeps=1)
    SequentialCoupledProblem(msh, soil)   # default route_substeps=4


# ====================================================================================================
# Test 13b -- OMEGA-HALVING ROBUSTNESS RETRY (the step() loop is a retry, NOT a Picard fixed point):
# when the FIRST inner Richards solve FAILS, step() must HALVE omega, RE-ROUTE + RE-SOLVE from the
# RESTORED entry state, and recover -- then the step still conserves. The retry/restore path was
# previously untested. We force the first inner solve to report a failure (via a thin proxy over the
# Richards NonlinearProblem that lies about the converged reason on its first solve of the step, then
# tells the truth), so the SECOND attempt (at omega/2) is the genuine production solve.
# ====================================================================================================
class _FailFirstSolveProxy:
    """Wraps a Richards ``NonlinearProblem`` and makes its FIRST ``solve()`` report a converged
    reason of -3 (DIVERGED_LINEAR_SOLVE) even though the underlying solve ran -- so ``step()`` sees a
    failed inner solve, halves omega, restores the entry state, and re-solves. From the 2nd solve on
    it reports the REAL reason (the genuine production path). Delegates everything else to the real
    problem; ``.solver`` returns a reason-lying shim of the real SNES."""
    def __init__(self, real):
        self._real = real
        self._solves = 0

    class _SnesShim:
        def __init__(self, snes, lie):
            self._snes, self._lie = snes, lie

        def getConvergedReason(self):
            return -3 if self._lie else int(self._snes.getConvergedReason())

        def getIterationNumber(self):
            return int(self._snes.getIterationNumber())

        def getFunctionNorm(self):
            return float(self._snes.getFunctionNorm())

    def solve(self):
        self._solves += 1
        return self._real.solve()

    @property
    def solver(self):
        return self._SnesShim(self._real.solver, lie=(self._solves <= 1))


def test_omega_halving_retry_recovers_and_conserves():
    """Force the first inner solve to FAIL and assert step() recovers via the omega-halving retry AND
    still conserves: a flat lake-at-rest column (the routing is a no-op, so the only thing that changes
    between attempts is omega and the restored entry state -- a clean isolation of the retry/restore
    logic). With the first solve forced to report failure, step() must (a) return converged, (b) have
    HALVED omega exactly once (last_omega == relax/2, since the 2nd attempt's real solve succeeds), and
    (c) keep the balance closed (the restore-from-entry + re-route did not corrupt the ledger)."""
    LX, LY, LZ = 2.0, 1.0, 1.0
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [4, 3, 4])
    soil = VanGenuchten(theta_r=0.067, theta_s=0.45, alpha=1.0, n=1.5, Ks=0.05)
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05, relax=1.0, picard_iters=4)
    prob.set_topography(lambda x: 0.0 * x[0])                 # FLAT -> routing is a pure no-op
    prob.set_initial_condition(lambda x: (LZ - x[2]) + 0.05)  # hydrostatic + a +5 cm pond
    prob.add_rain(0.0)
    prob._ensure_built()
    prob._rp._ensure_problem()                                # build the real NonlinearProblem + SNES

    # wrap the real problem so its FIRST solve this step reports failure (reason -3), 2nd+ tell truth.
    real_problem = prob._rp._problem
    proxy = _FailFirstSolveProxy(real_problem)
    prob._rp._problem = proxy

    conv, _it = prob.step(2e-3)

    # (a) the step recovered despite the forced first-solve failure (the omega-halving retry worked).
    assert conv, "step did not recover from the forced first-solve failure (retry path broken)"
    # the proxy genuinely intercepted >= 2 solves (first lied -> retry -> real success).
    assert proxy._solves >= 2, f"retry did not re-solve (only {proxy._solves} solve(s)) -- not exercised"
    # (b) omega was halved exactly once: attempt 1 (omega=1.0) forced-fail -> attempt 2 (omega=0.5) ok.
    assert prob.last_omega == pytest.approx(prob.relax * 0.5), \
        f"omega not halved once on the retry (last_omega={prob.last_omega}, relax={prob.relax})"

    # restore the real problem and keep stepping a few clean steps -> the ledger stays consistent.
    prob._rp._problem = real_problem
    for _ in range(5):
        c, _ = prob.step(2e-3)
        assert c, "a clean post-retry step diverged"
    # (c) conservation held across the retry + the clean steps (the restore/re-route did not leak).
    assert abs(prob.balance()) <= 1e-9, \
        f"balance broke across the omega-halving retry: |bal|={abs(prob.balance()):.3e}"


# ====================================================================================================
# B7 -- the two formerly-FAILING-case regressions (the cases that MOTIVATED the redesign): they
# dt-collapsed / sawtoothed both monolithic schemes (galerkin + upwind). The sequential split must
# now run them clean. Plus a thin selectable-wiring factory (the plan's "selectable, not rip-out").
# Geometries/soils/forcing are the spike `win` references (scratch/overland_split_spike.py
# tilted_v_case / _sand_channel_setup), SHRUNK (coarse mesh, short horizon) for suite speed.
# ====================================================================================================


# ----------------------------------------------------------------------------------------------------
# B7 fixture -- a SMALL convergent tilted-V (the spike tilted_v_case geometry, shrunk). The V is the
# original Pathology-1 (the never-settling sawtooth + dt-pin that 39.5-h'd / collapsed the monolithic
# schemes). Field-scale-similar V on a FLAT box (topography via z_b), with the DEEP unsaturated BUFFER
# (LZ=2 m, NZ=4, wettable-but-not-saturated IC) that dodges the no-Ss saturation singularity -- KEPT,
# per the spike note. Wetter IC + higher rain (vs the field-scale run, which leans on a long horizon we
# cannot afford in-suite) so the SHORT storm drives genuine surface routing to the y=LY outlet: within
# each step rain transiently lifts the convergence-line head above 0 and the Manning sweep moves that
# water down-V to the outlet (cum_outflow > 0), while the soil re-absorbs the residual film by step-end
# (so the accepted-state pond is ~0 and the deep buffer stays comfortably unsaturated, max psi ~ -0.29).
# The exercise is the CONVERGENT V geometry under a runoff storm, which is what sawtoothed/collapsed the
# monolith -- not a standing lake. Calibrated (scratch probe) to complete in ~3-4 s with strong outflow.
# ----------------------------------------------------------------------------------------------------
def _small_tilted_v(scale=0.04, rain=0.6, psi_i=-0.30):
    """Coarse/short convergent tilted-V (spike tilted_v_case, shrunk). Returns the built problem (no
    rain yet), the rain Constant, and the run cfg. FLAT-top box; z_b carries the V topography; a deep
    unsaturated buffer keeps the unconfined no-Ss Richards non-singular."""
    LX, LY, LZ = 1620.0 * scale, 1000.0 * scale, 2.0
    XC = LX / 2.0
    SX, SY = 0.05, 0.02
    n_man = 0.015
    NX, NY, NZ = 12, 8, 4
    SOIL = VanGenuchten(theta_r=0.07, theta_s=0.40, alpha=2.0, n=1.3, Ks=0.01)  # Ks << rain -> runs off

    def topo(x):
        # V: x<XC slopes toward +x (channel at XC), x>XC toward -x; the valley falls toward y=LY.
        return SX * np.abs(x[0] - XC) + SY * (LY - x[1])

    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [NX, NY, NZ])
    prob = SequentialCoupledProblem(msh, SOIL, n_man=n_man)
    prob.set_topography(topo)
    prob.set_initial_condition(lambda x: psi_i + 0.0 * x[0])
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)   # the V routes everything to y=LY
    rain_c = prob.add_rain(0.0)
    cfg = dict(t_end=0.10, storm=0.04, rain=float(rain))   # the numeric storm rate (m/day)
    return prob, rain_c, cfg


def _march(prob, rain, cfg, dt0=2e-4, dt_max=3e-3):
    """March a storm-then-recession with the band controller; track no-collapse + routing resid +
    min dt. Returns (nstep, dt_min_seen, routing_resid_max, collapsed)."""
    t, nstep, dt = 0.0, 0, dt0
    dt_min, rr_max, collapsed = dt0, 0.0, False
    while t < cfg["t_end"] - 1e-12:
        h = min(dt, cfg["t_end"] - t)
        if t < cfg["storm"] - 1e-12 and t + h > cfg["storm"]:
            h = cfg["storm"] - t
        rain.value = cfg["rain"] if t < cfg["storm"] - 1e-12 else 0.0
        conv, it = prob.step(h)
        if not conv:
            dt *= 0.5
            dt_min = min(dt_min, dt)
            if dt < 1e-9:
                collapsed = True
                break
            continue
        rr_max = max(rr_max, prob.last_routing_resid)
        t += h
        nstep += 1
        if it <= 4:
            dt = min(dt * 1.4, dt_max)
        elif it >= 12:
            dt = dt * 0.7
    return nstep, dt_min, rr_max, collapsed


# ====================================================================================================
# Test 14 -- THE HEADLINE REGRESSION: the convergent tilted-V (original Pathology 1, the sawtooth/dt-pin
# that 39.5-h'd / collapsed the monolithic Manning schemes) runs CLEAN on the sequential split: it
# completes to t_end with NO dt-collapse, conserves |bal|/cum_rain < 1e-3 (routing resid <= 1e-12),
# and ROUTES water to the outlet (cum_outflow > 0). This is the case the whole redesign exists to fix.
# ====================================================================================================
def test_convergent_tilted_v_regression_no_collapse_conserves_routes():
    """The convergent tilted-V (a small/short swale, the spike's tilted_v_case shrunk, deep unsaturated
    buffer kept) on the sequential operator-split scheme must:
      (1) COMPLETE to t_end with NO dt-collapse (the monolithic upwind dt-collapsed here / galerkin
          sawtoothed to a 39.5-h dt-pin; the sequential split decouples the stiff Richards Jacobian
          from the surface routing so neither pathology can form -- this is the headline fix);
      (2) CONSERVE: |balance|/cum_rain < 1e-3 with the routing store sum(d_i A_i) resid <= 1e-12;
      (3) ROUTE water to the outlet (cum_outflow > 0 -- the swale genuinely drains to y=LY, i.e. the
          convergent transport actually happens, not a dammed no-op).
    Pins that the redesign's motivating Pathology-1 case is fixed."""
    prob, rain, cfg = _small_tilted_v()
    nstep, dt_min, routing_resid, collapsed = _march(prob, rain, cfg)

    # (1) no dt-collapse: completed the loop without ever cutting dt below the floor.
    assert not collapsed, f"tilted-V dt-COLLAPSED (min dt seen {dt_min:.2e}) -- Pathology 1 NOT fixed"
    assert nstep > 0, "tilted-V storm did not advance"
    # (2) conservation: the routing store telescopes (resid <= 1e-12) and the global balance closes.
    assert routing_resid <= 1e-12, f"routing store sum(d*A) resid {routing_resid:.3e} > 1e-12"
    assert prob.cum_rain > 0.0, "no rain fell -> vacuous conservation check"
    bal_frac = abs(prob.balance()) / prob.cum_rain
    assert bal_frac < 1e-3, (
        f"tilted-V balance did not close: |bal|/cum_rain = {bal_frac:.3e} "
        f"(|bal|={abs(prob.balance()):.3e}, cum_rain={prob.cum_rain:.3e}, "
        f"cum_outflow={prob.cum_outflow:.3e})")
    # (3) the convergent swale actually routed water out to the outlet (the transport happened).
    assert prob.cum_outflow > 0.0, \
        "tilted-V routed NO water to the outlet -> the swale did not drain (transport stalled)"


# ----------------------------------------------------------------------------------------------------
# B7 fixture -- a SMALL sand-channel-in-clay where the CHANNEL is the only surface outlet, so the
# interception fraction is UNAMBIGUOUS. Spike _sand_channel_setup geometry/soils, shrunk: a sand swale
# embedded in clay, a storm ponds on the clay and CONCENTRATES into the swale (the convergence line),
# where it leaves via the channel-mouth surface outlet (y=0 over the channel) + the sand-interface GHB.
# With the channel-mouth as the ONLY outlet, (cum_outflow + cum_drainage)/cum_rain IS the channel's
# interception fraction (no toe outlet to confound it). This is the Pathology-2 (stiff-clay) case.
# ----------------------------------------------------------------------------------------------------
def _small_sand_channel_intercept():
    """Coarse/short sand-channel-in-clay with the channel-mouth surface outlet + sand GHB as the ONLY
    sinks (no toe outlet) -> the channel's interception fraction is unambiguous. Returns (prob, rain,
    cfg)."""
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
    prob = SequentialCoupledProblem(msh, soil, n_man=0.05)
    prob.set_topography(topo)
    prob.set_initial_condition(lambda x: PSI_I + 0.0 * x[0])
    # the CHANNEL-MOUTH surface outlet is the ONLY surface outlet (no toe outlet) -> cum_outflow is the
    # channel surface capture; the sand-interface GHB adds the channel subsurface capture.
    chan_mouth = lambda x: np.isclose(x[1], 0.0) & (np.abs(x[0] - X_CH) < W_CH + tol)
    prob.add_outflow_bc(chan_mouth, SY)
    drain_loc = (lambda x: np.isclose(x[1], 0.0) & (np.abs(x[0] - X_CH) < W_CH + tol)
                 & (x[2] >= Z_SAND_BASE - tol))
    prob.add_drainage_bc(drain_loc, 2.0, Z_SAND_BASE - 0.1)
    rain = prob.add_rain(0.0)
    cfg = dict(t_end=0.30, storm=STORM_DUR, rain=RAIN)
    return prob, rain, cfg


# ====================================================================================================
# Test 15 -- SAND-CHANNEL-IN-CLAY REGRESSION (Pathology 2, the stiff-clay case): the full explicit gate
# -- (1) no dt-collapse, (2) |bal|/cum_rain < 1e-3 (routing resid <= 1e-12), AND (3) the channel
# INTERCEPTS a meaningful fraction (> a few %) of the rain via its outlet/sand-GHB. The two-outlet
# conservation gate (Test 4) already pins completion+conservation on the toe+channel fixture; THIS test
# adds the interception dimension on a channel-ONLY-outlet fixture (so "channel capture" is unambiguous)
# -- the stiff-clay case that dt-collapsed the monolithic schemes, now with interception asserted.
# ====================================================================================================
def test_sand_channel_in_clay_regression_intercepts_and_conserves():
    """The sand-channel-in-clay storm (the stiff-clay Pathology 2 that dt-collapsed/sawtoothed both
    monolithic schemes) on the sequential split, with the channel-mouth surface outlet + sand GHB as
    the ONLY sinks. The full gate:
      (1) COMPLETES with NO dt-collapse;
      (2) CONSERVES: |bal|/cum_rain < 1e-3, routing store sum(d_i A_i) resid <= 1e-12;
      (3) INTERCEPTION FORMS: the channel captures a meaningful fraction of the rain --
          (cum_outflow + cum_drainage)/cum_rain > a few % (the storm ponds on the clay, concentrates
          into the high-K sand swale, and leaves via the channel-mouth outlet + the sand-interface GHB,
          rather than all infiltrating the clay or sitting as a static pond).
    Pins the stiff-clay regression WITH an explicit, unambiguous interception assertion."""
    prob, rain, cfg = _small_sand_channel_intercept()
    nstep, dt_min, routing_resid, collapsed = _march(prob, rain, cfg, dt0=1e-3, dt_max=0.03)

    # (1) no dt-collapse on the stiff clay.
    assert not collapsed, f"sand-channel dt-COLLAPSED (min dt {dt_min:.2e}) -- Pathology 2 NOT fixed"
    assert nstep > 0, "sand-channel storm did not advance"
    # (2) conservation.
    assert routing_resid <= 1e-12, f"routing store sum(d*A) resid {routing_resid:.3e} > 1e-12"
    assert prob.cum_rain > 0.0, "no rain fell -> vacuous conservation check"
    bal_frac = abs(prob.balance()) / prob.cum_rain
    assert bal_frac < 1e-3, (
        f"sand-channel balance did not close: |bal|/cum_rain = {bal_frac:.3e} "
        f"(cum_rain={prob.cum_rain:.3e}, cum_outflow={prob.cum_outflow:.3e}, "
        f"cum_drainage={prob.cum_drainage:.3e})")
    # (3) THE INTERCEPTION GATE: the channel captured a meaningful fraction of the rain (channel-mouth
    # surface outflow + sand GHB), not a negligible trickle. Calibrated ~27% in the scratch probe; the
    # gate requires > 5% (well above "a few %") so it is a real interception signal, robustly above any
    # roundoff/edge-effect floor while leaving generous headroom against mesh/horizon jitter.
    capture = prob.cum_outflow + prob.cum_drainage
    capture_frac = capture / prob.cum_rain
    assert capture_frac > 0.05, (
        f"the sand channel intercepted only {capture_frac:.2%} of the rain (<= 5%) -- no meaningful "
        f"interception formed (cum_outflow={prob.cum_outflow:.3e}, cum_drainage={prob.cum_drainage:.3e}, "
        f"cum_rain={prob.cum_rain:.3e})")


# ====================================================================================================
# Test 16 -- SELECTABLE WIRING (the plan's "selectable, not rip-out"): the sequential split must
# COEXIST with the monolithic schemes as a pickable option. A thin factory make_overland_coupled(...,
# scheme=...) returns the right class for each scheme. The monolithic schemes are SEPARATE classes from
# the sequential one (coexistence is STRUCTURAL); the factory is the single selection entry point.
# ====================================================================================================
def test_make_overland_coupled_factory_dispatches_by_scheme():
    """make_overland_coupled(mesh, soil, scheme=...) is the thin selection entry point: scheme=
    'sequential' -> SequentialCoupledProblem; scheme in {'galerkin','upwind','auto'} ->
    CoupledProblem(overland_scheme=scheme). Both are constructible on the SAME 3-D mesh/soil (so the
    schemes genuinely COEXIST -- the sequential split does not replace the monolith, it is selectable
    alongside it), the requested monolithic scheme is recorded on the returned object, and an unknown
    scheme raises a clear ValueError."""
    from pids_forward.physics.sequential_coupling import make_overland_coupled

    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [3, 3, 4])  # 3-D (upwind/auto ok)
    soil = VanGenuchten(theta_r=0.067, theta_s=0.45, alpha=1.0, n=1.5, Ks=0.05)

    seq = make_overland_coupled(msh, soil, scheme="sequential")
    assert isinstance(seq, SequentialCoupledProblem), "scheme='sequential' must build the split class"
    assert not isinstance(seq, CoupledProblem)

    # the three monolithic schemes -> CoupledProblem with the requested overland_scheme recorded.
    for scheme in ("galerkin", "upwind", "auto"):
        prob = make_overland_coupled(msh, soil, scheme=scheme)
        assert isinstance(prob, CoupledProblem), f"scheme={scheme!r} must build the monolith"
        assert not isinstance(prob, SequentialCoupledProblem)
        assert prob.overland_scheme == scheme, \
            f"factory did not pass overland_scheme={scheme!r} through ({prob.overland_scheme!r})"

    # extra kwargs flow through to the chosen class (e.g. n_man) -- the factory is a pass-through.
    seq2 = make_overland_coupled(msh, soil, scheme="sequential", n_man=0.123, route_substeps=2)
    assert seq2.n_man == pytest.approx(0.123) and seq2.route_substeps == 2
    mono2 = make_overland_coupled(msh, soil, scheme="galerkin", n_man=0.077)
    assert mono2.n_man == pytest.approx(0.077)

    # an unknown scheme is rejected with a clear error naming the option.
    with pytest.raises(ValueError, match="scheme"):
        make_overland_coupled(msh, soil, scheme="bogus")
