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


def _box(nx=4, ny=2, nz=2, Lx=2.0, Ly=1.0, Lz=1.0):
    return dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, Ly, Lz]], [nx, ny, nz])


def _feature_coupled_box(R_out=2.0, n=12, nx=4, psi_i=-1.0, psi_wall=0.0):
    """A CLOSED feature-box CoupledProblem host (ridge along x at the y-z centre, sized to R_out so the
    disperse R_out/2 ring resolves) -- the coupled analogue of the standalone run_embedded box. No rain,
    no outlet, no drainage -> the surface NCP is inert, so the coupled psi evolves like the harness's
    bare-Richards psi PLUS the feature ridge source (the reduce-to-standalone basis)."""
    L = float(np.sqrt(np.pi * (R_out ** 2 - R_W_DEFAULT ** 2)))
    h = L / n
    Lx = nx * h
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, L, L]], [nx, n, n])
    prob = CoupledProblem(msh, SOIL, overland_scheme="galerkin", n_man=0.05)
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
def test_add_embedded_exchange_requires_catchment_and_weaves_signed_source():
    """``add_embedded_exchange(feat, driver, catchment_cells, ctx=...)`` registers the driver, sets it
    up (catchment threaded in), and weaves the ridge SOURCE ``- rate/length * vpsi * feat.dGamma`` into
    F_psi. catchment_cells is REQUIRED (no silent whole-domain default). The - sign is verified
    directly: a manual prescribed DISPERSE rate (no driver hooks yet) ADDS soil water ~ rate*dt."""
    prob, feat, L = _feature_coupled_box()
    driver = WellIndexExchange()                              # disperse
    with pytest.raises(ValueError, match="catchment"):
        prob.add_embedded_exchange(feat, driver, None, ctx={"t0": 1e-4, "R_out": 2.0})

    rate_const = prob.add_embedded_exchange(feat, driver, "all", ctx={"t0": 1e-4, "R_out": 2.0})
    assert isinstance(rate_const, fem.Constant)
    assert len(prob._features) == 1
    assert prob.cum_sinks["feature"] == [0.0]

    # the - sign: a manual disperse rate ADDS water; a CLOSED box (no rain/outlet/drainage) conserves
    # so the soil gain == the ridge source rate*dt to solver precision.
    w0 = prob.total_water()
    rate = 5.0                                                # m^3/day host-ward (disperse, +)
    rate_const.value = rate / feat.length                    # per unit length, as the driver sets it
    conv, _ = prob.step(1e-3)
    assert conv
    dW = prob.total_water() - w0
    assert dW > 0.0                                          # disperse ADDED water (the - sign is right)
    assert abs(dW - rate * 1e-3) < 1e-3 * (rate * 1e-3)      # == rate*dt (closed-box conservation)


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
