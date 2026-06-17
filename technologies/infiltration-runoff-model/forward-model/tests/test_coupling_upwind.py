"""Tier-1 sanity for the OPT-IN upwind overland scheme inside CoupledProblem (Convergent-flow P2).

P2 productionizes the validated standalone monotone upwind edge-flux scheme (P1,
``UpwindOverlandProblem``) as the LATERAL-overland operator of the coupled ``[psi, d, lam]`` solver.
It is OPT-IN: ``CoupledProblem(..., overland_scheme="upwind")``; the default ``"galerkin"`` path is
unchanged and bit-identical (pinned by the existing ``tests/test_coupling_{1,2,3}d*.py``). The upwind
path replaces ONLY the UFL ``overland_flux`` term with the non-UFL edge-flux on the top-facet graph,
supplied as a custom residual + Jacobian on the d-block via a custom block-SNES; psi, lam/NCP,
outlet, drainage and the interior-pin stay UFL and untouched, so conservation stays structural.

Scope (P2): the upwind path requires a 3-D host (the top facet is a 2-D triangulation for the
cotangent edge graph) and is serial (a multi-rank guard is added in Task B2). Galerkin is unchanged
in all dimensions.
"""
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def _box(nx=6, ny=3, nz=3, Lx=2.0, Ly=1.0, Lz=1.0):
    return dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, Ly, Lz]], [nx, ny, nz])


def test_overland_scheme_default_is_galerkin():
    assert CoupledProblem(_box(), SOIL).overland_scheme == "galerkin"


def test_overland_scheme_upwind_flag_and_validation():
    assert CoupledProblem(_box(), SOIL, overland_scheme="upwind").overland_scheme == "upwind"
    with pytest.raises(ValueError, match="overland_scheme"):
        CoupledProblem(_box(), SOIL, overland_scheme="bogus")


def test_upwind_requires_3d_host():
    """P2 scope: the cotangent top-facet edge graph needs a 2-D top triangulation (3-D host).
    A 2-D host's top is a 1-D edge -- a future extension, refused loudly for now."""
    msh2d = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (2.0, 1.0)], [8, 4])
    with pytest.raises(NotImplementedError, match="3-D"):
        CoupledProblem(msh2d, SOIL, overland_scheme="upwind")


def test_upwind_reduced_Fd_omits_lateral_flux():
    """The upwind path removes the UFL lateral conveyance from F_d (it is supplied by the edge-flux
    residual instead). At a NON-FLAT surface head, the galerkin F_d carries the lateral term on the
    top rows and the upwind F_d does not -> the assembled d-residual vectors differ."""
    g = CoupledProblem(_box(), SOIL)
    u = CoupledProblem(_box(), SOIL, overland_scheme="upwind")
    for p in (g, u):
        p.set_topography(lambda x: 0.05 * x[0] + 0.02 * x[1])    # tilted bed -> non-flat H = z_b + d
        p.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.05)
        p.add_rain(0.0)
    bg = assemble_vector(fem.form(g.F_d)); bg.assemble()
    bu = assemble_vector(fem.form(u.F_d)); bu.assemble()
    assert np.abs(bg.getArray() - bu.getArray()).max() > 1e-3
    bg.destroy(); bu.destroy()


# -- Task B2: the custom block-SNES (residual + Picard d-d Jacobian) -----------

def test_upwind_step_converges_3d():
    """A coupled upwind step solves cleanly (the custom block-SNES residual + Picard Jacobian
    drive the [psi,d,lam] Newton to convergence) on a tilted ponding box."""
    prob = CoupledProblem(_box(5, 5, 3), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * (2.0 - x[0]))
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.05)
    prob.add_rain(0.3)
    converged, iters = prob.step(1e-3)
    assert converged and prob.last_reason > 0
    assert np.all(np.isfinite(prob.d.x.array))


def test_upwind_block_jacobian_no_malloc_3d():
    """Codex blocker #1 DISSOLVED: the consistent ds_top storage-mass Jacobian already preallocates
    every top-facet d-d coupling, so the Picard edge-flux Jacobian inserts with NO new allocation.
    Setting NEW_NONZERO_ALLOCATION_ERR before the first solve makes a missing slot raise loudly."""
    prob = CoupledProblem(_box(5, 5, 3), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * (2.0 - x[0]))
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.05)
    prob.add_rain(0.3)
    prob._ensure_problem()                                  # build _A + wire callbacks (no solve yet)
    prob._problem._A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    converged, _ = prob.step(1e-3)                          # first Jacobian assembly inserts edge entries
    assert converged                                       # completed -> the edge nonzeros were preallocated


def test_upwind_closed_conservation_3d():
    """The key solver gate: a CLOSED tilted box (no outlet/drainage) under ponding rain conserves
    total water EXACTLY -- Delta total == cumulative rain. Exercises infiltration + the monotone
    lateral edge flux (water ponds and routes downslope, accumulating; nothing leaves) and confirms
    the edge flux telescopes to zero (no spurious mass) without the limiter clipping."""
    Lx, Ly = 2.0, 1.0
    prob = CoupledProblem(_box(6, 6, 4, Lx=Lx, Ly=Ly), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * (Lx - x[0]))      # tilt toward x=0 -> lateral routing
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    w0 = prob.total_water()
    rate, t_end = 0.5, 0.1                                  # > Ks=0.25 -> mild ponding + routing
    prob.add_rain(rate)
    prob.advance(t_end=t_end, dt=1e-3, dt_max=0.02)
    cum_rain = rate * Lx * Ly * t_end

    assert abs((prob.total_water() - w0) - cum_rain) / cum_rain < 1e-6
    assert prob.total_water() - w0 > 0.5 * cum_rain        # rain genuinely entered
    assert prob.d.x.array.min() >= -1e-12                  # monotone: no negative depth
    assert abs(prob.clip_mass_adjust) < 1e-9              # the limiter never had to clip
    assert np.all(np.isfinite(prob.d.x.array)) and np.all(np.isfinite(prob.psi.x.array))


def test_upwind_coupled_jacobian_matches_fd_smoke_3d():
    """Codex should-fix: a COUPLED-level Jacobian check (the kernel FD-verify is necessary but NOT
    sufficient -- it cannot catch block-offset / sign / BC bugs in the edge block once inserted into
    the [psi,d,lam] block matrix). At a plausible solved state, the assembled block Jacobian action
    J*delta matches a central finite-difference of the FULL assembled coupled residual along a random
    direction, with the interior pins active."""
    prob = CoupledProblem(_box(5, 5, 3), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * (2.0 - x[0]))
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.05)
    prob.add_rain(0.3)
    prob.step(1e-3)                                         # advance to a plausible coupled state

    snes = prob._problem.solver
    A = prob._problem._A
    x = snes.getSolution().copy()
    snes.computeJacobian(x, A, A)                           # assemble J (UFL + edge block) at x

    rng = np.random.default_rng(3)
    delta = x.duplicate()
    with delta.localForm() as dl:
        dl.array[:] = rng.standard_normal(dl.array.size)
    delta.scale(1.0 / delta.norm())

    Jd = x.duplicate(); A.mult(delta, Jd)                   # J * delta
    eps = 1e-6
    Fp = x.duplicate(); Fm = x.duplicate()
    xp = x.copy(); xp.axpy(eps, delta); snes.computeFunction(xp, Fp)
    xm = x.copy(); xm.axpy(-eps, delta); snes.computeFunction(xm, Fm)
    fd = Fp - Fm; fd.scale(1.0 / (2.0 * eps))              # central FD of the full coupled residual

    rel = (Jd - fd).norm() / max(fd.norm(), 1e-30)
    assert rel < 1e-5, f"coupled J*delta vs FD mismatch: rel err {rel:.2e}"
