"""Tier-1 sanity for the extracted upwind edge-flux KERNEL (Convergent-flow P2, Part A).

``pids_forward/physics/overland_edge_kernel.py`` holds the pure-function core of the validated
standalone ``UpwindOverlandProblem`` scheme -- the 2-D triangle-mesh edge graph
(``build_edge_graph_2d``) and the lumped backward-Euler node residual (``edge_flux_residual``) --
so the SAME math can be reused by the coupled solver (P2) without duplicating it. These tests are
the DP-2 EXTRACTION GUARD: the kernel must reproduce the standalone class's internal graph AND
its residual BIT-IDENTICALLY (so the refactor cannot perturb the P1-validated numerics). The
full standalone suite ``tests/test_overland_upwind.py`` staying green is the second half of that
guard.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh
from petsc4py import PETSc

from pids_forward.physics.overland_upwind import UpwindOverlandProblem
from pids_forward.physics.overland_edge_kernel import build_edge_graph_2d, edge_flux_residual

TRI = dmesh.CellType.triangle
N_MAN = 0.03


def test_build_edge_graph_2d_matches_standalone_internals():
    """The kernel's (edges, L_e, T_e, A_i) == the standalone class's internally-built graph,
    bit-identically (same extracted code -> same ordering + values)."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (2.0, 1.0)], [16, 8], cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)

    edges, L_e, T_e, A_i = build_edge_graph_2d(prob.V, msh)

    assert np.array_equal(edges, prob.edges)
    assert np.array_equal(L_e, prob.L_e)
    assert np.array_equal(T_e, prob.T_e)
    assert np.array_equal(A_i, prob.A_i)


def test_edge_flux_residual_equals_standalone_at_nontrivial_state():
    """Codex DP-2 guard: pin the RESIDUAL, not just the graph. Drive a random NON-FLAT positive
    state (random z_b, d, d_n, rain, dt) through both the kernel ``edge_flux_residual`` and the
    standalone ``_assemble_residual`` SNES callback; assert bit-identical."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (2.0, 1.0)], [16, 8], cell_type=TRI)
    prob = UpwindOverlandProblem(msh, n_man=N_MAN)

    rng = np.random.default_rng(0)
    n = prob.n_dofs
    prob.set_topography(lambda x: 0.05 * x[0] + 0.02 * x[1])           # tilted bed (non-flat H)
    d_n = 0.01 + 0.02 * rng.random(n)
    prob.d_n.x.array[:n] = d_n
    prob.d_n.x.scatter_forward()
    prob.add_rain(0.3)
    prob.dt = 0.02
    d_cur = 0.01 + 0.02 * rng.random(n)                                # current iterate depths

    # standalone path: drive the SNES residual callback with d_cur in x
    x = PETSc.Vec().createWithArray(d_cur.copy(), comm=msh.comm)
    b = x.duplicate()
    prob._assemble_residual(None, x, b)
    R_standalone = b.getArray().copy()

    # kernel path: the pure function on the same state
    R_kernel = edge_flux_residual(
        d_cur, prob.z_b.x.array, prob.d_n.x.array, prob.rain, prob.dt,
        prob.edges, prob.L_e, prob.T_e, prob.A_i, prob.n_man, prob.eps_S, prob.eps_H,
        prob._outflows,
    )

    assert np.array_equal(R_kernel, R_standalone)
    x.destroy(); b.destroy()
