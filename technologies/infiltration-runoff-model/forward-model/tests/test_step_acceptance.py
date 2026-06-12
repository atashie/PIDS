"""Tier-1 sanity: step-acceptance hardening (O5, convergent-flow plan P0).

A backward-Euler step may be BOOKED only when the Newton solve actually balanced the residual:
an accepted-but-unbalanced residual is a mass error injected straight into the books (plan
docs/plans/2026-06-11-overland-convergent-flow-stabilization.md, Defect B). PETSc's
CONVERGED_SNORM_RELATIVE (reason 4) is the STAGNATION verdict -- the iterate stopped moving --
which is legitimate at the residual floor (near-flat / MMS states whose assembly floor sits
above atol; measured <= ~1.2e-6 across this suite) but certifies NOTHING about balance: a
stalled line search far from the root also returns 4 (measured |F| ~ 1e-5..3e-3 on the stiff
convergent V, and 2.4e+3 in the fixture below). So

  (a) ``step()`` books reason 4 ONLY when |F| <= ``stall_accept_fnorm`` (absolute bar, 3e-6 =
      the populations' geometric mean): floor stagnation books, dirty stalls become honest
      rejections (dt cut + retry); on reason 4 the norm is RECOMPUTED at the returned iterate
      (PETSc's failed-line-search exit can cache the previous iterate's norm),
  (b) both solvers pin ``snes_stol: 1e-8`` explicitly (prompt stagnation verdicts; stol=0 would
      grind floor states through max-it into dt death spirals),
  (c) ``step()`` records ``last_reason`` / ``last_fnorm`` so any run can audit WHAT it accepted.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.overland import OverlandProblem

N_MAN = 0.05
SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)

# acceptance-impossible fnorm tolerances + a huge stol: the default PETSc test can then ONLY
# return CONVERGED_SNORM_RELATIVE (4) -- the stalled-step verdict under test.
_STALL_OPTIONS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1e-30,
    "snes_atol": 1e-30,
    "snes_stol": 0.5,
    "snes_max_it": 50,
    "ksp_type": "preonly",
    "pc_type": "lu",
}


def _overland_blob(petsc_options=None):
    msh = dmesh.create_interval(MPI.COMM_WORLD, 40, [0.0, 20.0])
    prob = OverlandProblem(msh, n_man=N_MAN, petsc_options=petsc_options)
    prob.set_topography(lambda x: 0.02 * (20.0 - x[0]))
    prob.set_initial_condition(lambda x: 0.2 * np.exp(-((x[0] - 10.0) / 2.0) ** 2))
    return prob


def _coupled_column(petsc_options=None):
    msh = dmesh.create_interval(MPI.COMM_WORLD, 30, [0.0, 1.0])
    prob = CoupledProblem(msh, SOIL, petsc_options=petsc_options)
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    prob.add_rain(0.1)
    return prob


def test_acceptance_contract_defaults():
    """The acceptance contract is pinned: explicit stol + an absolute reason-4 booking bar.

    stol stays at PETSc's 1e-8 (pinned, not implicit) so floor-stagnation exits PROMPTLY as
    reason 4 instead of grinding to max-it; the booking decision then lives in step(), gated
    on |F| <= stall_accept_fnorm. The bar must sit between the measured legitimate-floor
    population (<= ~1.2e-6) and the measured dirty-stall population (>= ~1e-5).
    """
    assert OverlandProblem._DEFAULT_PETSC_OPTIONS["snes_stol"] == 1e-8
    assert CoupledProblem._DEFAULT_PETSC_OPTIONS["snes_stol"] == 1e-8
    assert _overland_blob().stall_accept_fnorm == 3e-6
    msh = dmesh.create_interval(MPI.COMM_WORLD, 10, [0.0, 1.0])
    assert CoupledProblem(msh, SOIL).stall_accept_fnorm == 3e-6


def test_overland_step_records_reason_and_fnorm():
    """An accepted overland step exposes WHAT was accepted: reason + residual norm."""
    prob = _overland_blob()
    converged, iters = prob.step(1e-3)
    assert converged and iters >= 1
    assert prob.last_reason in (2, 3)        # FNORM_ABS / FNORM_RELATIVE: residual-tested
    assert np.isfinite(prob.last_fnorm)
    assert prob.last_fnorm < 1e-6            # actually balanced (rtol-tested), not stalled junk


def test_coupled_step_records_reason_and_fnorm():
    """Same audit trail on the coupled [psi, d, lambda] step."""
    prob = _coupled_column()
    converged, iters = prob.step(1e-3)
    assert converged and iters >= 1
    assert prob.last_reason in (2, 3)
    assert np.isfinite(prob.last_fnorm)
    assert prob.last_fnorm < 1e-8


def test_overland_snorm_stall_is_rejected_not_booked():
    """A reason-4 (stalled line search) overland solve must be REJECTED: state restored,
    nothing booked -- even when caller-supplied options allow PETSc to report it 'converged'."""
    prob = _overland_blob(petsc_options=_STALL_OPTIONS)
    d_before = prob.d_n.x.array.copy()
    w_before = prob.total_water()

    converged, _ = prob.step(1e-3)

    assert prob.last_reason == 4             # precondition: PETSc DID report the stall verdict
    assert prob.last_fnorm > prob.stall_accept_fnorm     # ... DIRTY (far from balance), so:
    assert not converged                     # step() refused to book it
    assert np.array_equal(prob.d.x.array, d_before)      # state restored exactly
    assert np.array_equal(prob.d_n.x.array, d_before)
    assert prob.total_water() == pytest.approx(w_before, rel=1e-14)


def test_coupled_snorm_stall_is_rejected_not_booked():
    """Same contract for the coupled step: reason 4 -> rejected, books untouched."""
    prob = _coupled_column(petsc_options=_STALL_OPTIONS)
    psi_before = prob.psi_n.x.array.copy()
    d_before = prob.d_n.x.array.copy()
    cum_out_before = prob.cum_outflow

    lam_before = prob.lam.x.array.copy()
    cum_drain_before = prob.cum_drainage

    converged, _ = prob.step(1e-3)

    assert prob.last_reason == 4
    assert prob.last_fnorm > prob.stall_accept_fnorm
    assert not converged
    assert np.array_equal(prob.psi.x.array, psi_before)
    assert np.array_equal(prob.d.x.array, d_before)
    # the FULL state restores -- λ too: a stalled NCP multiplier left in place would seed the
    # retry's Newton with a wrong active-set guess and leak into exchange_flux() diagnostics.
    assert np.array_equal(prob.lam.x.array, lam_before)
    assert prob.cum_outflow == cum_out_before  # nothing booked on the rejected step
    assert prob.cum_drainage == cum_drain_before


def test_floor_stagnation_books():
    """The ACCEPT side of the reason-4 gate: stagnation AT the residual floor must book.

    A near-flat surface from a uniform-depth start is already (numerically) at its solution;
    the residual floor sits above atol, rtol cannot fire from a floor-magnitude fnorm0, so the
    default test exits CONVERGED_SNORM_RELATIVE with a tiny |F| (measured ~3.6e-10). Rejecting
    it would dt-death-spiral every near-steady run -- the measured failure of blanket stol=0.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 10.0])
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: 1e-6 * x[0])
    prob.set_initial_condition(lambda x: 0.1 + 0.0 * x[0])

    converged, _ = prob.step(1e-3)

    assert prob.last_reason == 4                          # stagnation verdict...
    assert prob.last_fnorm <= prob.stall_accept_fnorm     # ...at the residual floor
    assert converged                                      # -> bookable
