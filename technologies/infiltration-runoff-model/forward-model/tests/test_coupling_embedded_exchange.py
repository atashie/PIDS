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

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def _box(nx=4, ny=2, nz=2, Lx=2.0, Ly=1.0, Lz=1.0):
    return dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, Ly, Lz]], [nx, ny, nz])


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
