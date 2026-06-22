"""Tier-1: wiring the Module-4 embedded-feature subsystem into the coupled [psi, d, lam] engine.

Capability A (plan docs/plans/2026-06-22-embedded-feature-coupled-integration.md): the validated
prescribed-rate ``WellIndexExchange`` sorptive driver on the coupled host -- a ridge SOURCE term in
F_psi (H_f held, no new block), with pre_step/post_step hooks inside the O5 acceptance gate and a
signed ``cum_feature`` accounting term. This file is built bottom-up, increment by increment (TDD).
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from dolfinx import fem
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.feature import EmbeddedFeature
from pids_forward.physics.wi_exchange import WellIndexExchange
from pids_forward.physics.sorptive_closure import R_W_DEFAULT

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)   # = test_wi_exchange LOAM

# acceptance-impossible tolerances + a huge stol -> the SNES can ONLY return CONVERGED_SNORM_RELATIVE
# (reason 4) far from balance, which step() must REJECT (mirrors test_step_acceptance._STALL_OPTIONS).
_STALL_OPTIONS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_rtol": 1e-30, "snes_atol": 1e-30, "snes_stol": 0.5, "snes_max_it": 50,
    "ksp_type": "preonly", "pc_type": "lu",
}


def _box(nx=4, ny=2, nz=2, Lx=2.0, Ly=1.0, Lz=1.0):
    return dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, Ly, Lz]], [nx, ny, nz])


def _feature_coupled_box(R_out=2.0, n=12, nx=4, psi_i=-1.0, psi_wall=0.0, petsc_options=None):
    """A CLOSED feature-box CoupledProblem host (ridge along x at the y-z centre, sized to R_out so the
    disperse R_out/2 ring resolves) -- the coupled analogue of the standalone run_embedded box. No rain,
    no outlet, no drainage -> the surface NCP is inert, so the coupled psi evolves like the harness's
    bare-Richards psi PLUS the feature ridge source (the reduce-to-standalone basis)."""
    L = float(np.sqrt(np.pi * (R_out ** 2 - R_W_DEFAULT ** 2)))
    h = L / n
    Lx = nx * h
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, L, L]], [nx, n, n])
    prob = CoupledProblem(msh, SOIL, overland_scheme="galerkin", n_man=0.05,
                          petsc_options=petsc_options)
    prob.set_initial_condition(lambda x: psi_i + 0.0 * x[0], d_value=0.0)
    prob.add_rain(0.0)
    feat = EmbeddedFeature(msh, lambda x: np.isclose(x[1], L / 2) & np.isclose(x[2], L / 2),
                           tangent=(1.0, 0.0, 0.0), K_feat=1.0,
                           area=np.pi * R_W_DEFAULT ** 2, porosity=0.4)
    feat.configure_sorptive(SOIL, psi_i=psi_i, psi_wall=psi_wall)
    return prob, feat, L


# ---------------------------------------------------------------------------
# Increment 4: add_embedded_exchange API + the F_psi ridge SOURCE (the - sign)
# ---------------------------------------------------------------------------
def test_add_embedded_exchange_requires_catchment_and_registers():
    """``add_embedded_exchange(feat, driver, catchment_cells, ctx=...)`` sets up the driver (catchment
    threaded in), weaves the ridge SOURCE ``- rate/length * vpsi * feat.dGamma`` into F_psi, registers a
    'feature' accounting slot, and returns the per-unit-length rate Constant. ``catchment_cells`` is
    REQUIRED (no silent whole-domain default). (The ``-`` sign + the source delivery are pinned by the
    closed-box conservation test below: a wrong sign makes ΔW negative while cum_feature stays positive,
    breaking ``Δtotal == cum_feature``.)"""
    prob, feat, L = _feature_coupled_box()
    driver = WellIndexExchange()                              # disperse
    with pytest.raises(ValueError, match="catchment"):
        prob.add_embedded_exchange(feat, driver, None, ctx={"t0": 1e-4, "R_out": 2.0})

    rate_const = prob.add_embedded_exchange(feat, driver, "all", ctx={"t0": 1e-4, "R_out": 2.0})
    assert isinstance(rate_const, fem.Constant)
    assert len(prob._features) == 1
    assert prob.cum_sinks["feature"] == [0.0] and prob.last_sinks["feature"] == [0.0]


# ---------------------------------------------------------------------------
# Increment 5: step() pre/post hooks drive the feature + the host books close (cum_feature)
# ---------------------------------------------------------------------------
def test_step_hooks_drive_feature_and_conserve_closed_box():
    """step() calls driver.pre_step BEFORE the solve (-> rate_const) and post_step INSIDE the accept
    gate (-> books cum_feature), so a registered disperse feature is driven by the resolved coupled psi
    and its host-ward volume is booked. On a CLOSED box (no rain/outlet/drainage) the feature is the
    ONLY source, so the full balance is Δtotal == cum_feature (+clip), and the driver's clock advances."""
    prob, feat, L = _feature_coupled_box(n=8)
    driver = WellIndexExchange()                              # disperse (held saturated wall psi=0)
    prob.add_embedded_exchange(feat, driver, "all", ctx={"t0": 1e-4, "R_out": 2.0})
    w0 = prob.total_water()
    prob.advance(0.05, 5e-4, dt_max=5e-3)

    assert prob.cum_feature > 0.0                            # disperse injected over the run
    assert prob.cum_outflow == 0.0 and prob.cum_drainage == 0.0
    assert prob.cum_sinks["feature"][0] == pytest.approx(prob.cum_feature)
    # closed box: Δtotal == cum_feature + clip_mass_adjust (rain/outflow/drainage all 0)
    dW = prob.total_water() - w0
    assert dW > 0.0                                         # disperse ADDED soil water (pins the - weave)
    expected = prob.cum_feature + prob.clip_mass_adjust
    assert abs(dW - expected) < 1e-6 * max(abs(prob.cum_feature), 1e-12)
    assert driver.I_total() > driver.seed_I                 # the sub-grid clock advanced (real uptake)


def test_reduce_to_standalone_disperse_tracks_harness():
    """Reduce-to-standalone: the SAME disperse closure driven through CoupledProblem.step() reproduces
    the validated STANDALONE bare-Richards harness (run_embedded) I_total trajectory within tol -- the
    coupled embedding is the validated closure on a richer host, not a new scheme. (~1-2 min: runs both
    the harness and the coupled advance on a coarse n=8 box over a short window.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded
    R_out, n = 2.0, 8
    t_grid = np.array([1e-4, 0.01, 0.02, 0.03, 0.04])

    out = run_embedded(WellIndexExchange(), "LOAM", R_out, n, t_grid, direction="disperse")
    assert out is not None, "the standalone harness run did not complete"
    I_std = np.asarray(out["I"])

    prob, feat, L = _feature_coupled_box(R_out=R_out, n=n)
    drv = WellIndexExchange()
    prob.add_embedded_exchange(feat, drv, "all", ctx={"t0": float(t_grid[0]), "R_out": R_out})
    prob._t = float(t_grid[0])                               # align the absolute clock to the seed age
    I_cpl = [drv.I_total()]
    for tm in t_grid[1:]:
        prob.advance(float(tm) - prob._t, 5e-4, dt_max=5e-3)
        I_cpl.append(drv.I_total())
    I_cpl = np.asarray(I_cpl)

    e = float(np.linalg.norm(I_cpl - I_std) / np.linalg.norm(I_std))
    assert e < 0.05, f"reduce-to-standalone I_total rel-L2 = {e:.1%}\n coupled={I_cpl}\n std    ={I_std}"


# ---------------------------------------------------------------------------
# Increment 6: reject-safety + the driver's guards fire through the coupled path
# ---------------------------------------------------------------------------
def test_rejected_step_does_not_book_feature():
    """A rejected reason-4 step must NOT book the feature: cum_feature, cum_sinks['feature'] and the
    driver's own inj all stay put (post_step is accepted-only; the retry re-runs pre_step from the
    restored ψ). Mirrors test_step_acceptance with a registered feature."""
    prob, feat, L = _feature_coupled_box(n=8, petsc_options=_STALL_OPTIONS)
    driver = WellIndexExchange()
    prob.add_embedded_exchange(feat, driver, "all", ctx={"t0": 1e-4, "R_out": 2.0})
    inj_before = driver.inj

    converged, _ = prob.step(1e-3)

    assert prob.last_reason == 4 and prob.last_fnorm > prob.stall_accept_fnorm   # a dirty stall
    assert not converged                                    # step() refused to book it
    assert prob.cum_feature == 0.0                          # nothing booked
    assert prob.cum_sinks["feature"] == [0.0]
    assert driver.inj == inj_before                         # the driver's own ledger did not advance


def test_resolved_wall_fence_fires_through_coupled_path():
    """The coupled integration does NOT bypass the driver's guards: a disperse feature in the
    resolved-wall regime below the validated band (R_out < 40 r_w) is REFUSED by the honest fence
    through add_embedded_exchange."""
    prob, feat, L = _feature_coupled_box(R_out=0.5, n=12)   # 10 r_w, fine mesh -> resolved-wall, refused
    driver = WellIndexExchange()
    with pytest.raises(ValueError, match=r"resolved-wall|VALIDATED band|R_out"):
        prob.add_embedded_exchange(feat, driver, "all", ctx={"t0": 1e-4, "R_out": 0.5})


def _dry_coupled(nx=4, ny=2, nz=2, Lx=2.0):
    """A small dry 3-D coupled hillslope that takes clean no-rain steps (galerkin; the upwind default
    is irrelevant to the time/accounting plumbing under test here)."""
    prob = CoupledProblem(_box(nx, ny, nz, Lx=Lx), SOIL, overland_scheme="galerkin", n_man=0.05)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: 0.05 * (Lx - x[0]))
    prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], Lx), slope=0.05)
    return prob


# ---------------------------------------------------------------------------
# Increment 1: persistent absolute time self._t
# ---------------------------------------------------------------------------
def test_persistent_absolute_time_accumulates_across_steps():
    """CoupledProblem carries a persistent absolute clock ``self._t``: it starts at 0 and each ACCEPTED
    step advances it by dt -- the embedded-feature driver needs monotone absolute time (seed age /
    handover timestamp)."""
    prob = _dry_coupled()
    assert prob._t == 0.0
    conv, _ = prob.step(1e-3)
    assert conv
    assert prob._t == pytest.approx(1e-3)
    conv, _ = prob.step(2e-3)
    assert conv
    assert prob._t == pytest.approx(3e-3)


def test_step_explicit_t_kwarg_overrides_clock():
    """``step(dt, t=...)`` uses the explicit absolute time (default = self._t); on accept the clock
    becomes t + dt."""
    prob = _dry_coupled()
    conv, _ = prob.step(1e-3, t=5.0)
    assert conv
    assert prob._t == pytest.approx(5.0 + 1e-3)


def test_advance_accumulates_clock_across_calls():
    """``advance()`` does NOT reset the absolute clock per call: two successive advance() calls
    accumulate self._t (only the within-call progress toward t_end resets)."""
    prob = _dry_coupled()
    prob.advance(0.01, 1e-3, dt_max=5e-3)
    assert prob._t == pytest.approx(0.01)
    prob.advance(0.02, 1e-3, dt_max=5e-3)
    assert prob._t == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Increment 2: 'feature' accounting fields (a SIGNED source term, tracked separately from the
# outward-only drainage sinks; the host balance is ... - cum_drainage + cum_feature)
# ---------------------------------------------------------------------------
def test_feature_accounting_fields_exist_and_zero():
    """CoupledProblem exposes the embedded-feature accounting surface: an aggregate ``cum_feature`` /
    ``last_feature`` (signed host-ward volume / rate) and per-feature ``cum_sinks['feature']`` /
    ``last_sinks['feature']`` lists -- all empty/zero before any feature is registered. ``cum_feature``
    is tracked SEPARATELY from ``cum_drainage`` (a signed source, not an outward-only sink)."""
    prob = _dry_coupled()
    assert prob.cum_feature == 0.0
    assert prob.last_feature == 0.0
    assert prob.cum_sinks["feature"] == []
    assert prob.last_sinks["feature"] == []
