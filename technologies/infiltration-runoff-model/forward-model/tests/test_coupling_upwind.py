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
