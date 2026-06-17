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
from pids_forward.physics.overland_edge_kernel import (
    build_edge_graph_2d,
    build_top_facet_edge_graph,
    edge_flux_residual,
)

TRI = dmesh.CellType.triangle
N_MAN = 0.03


def _top_facets_dofs(msh, V):
    """Locate the top (max-z) boundary facets + their dofs + ztop (mirrors CoupledProblem)."""
    from dolfinx import fem
    zaxis = msh.topology.dim - 1
    fdim = msh.topology.dim - 1
    zc = V.tabulate_dof_coordinates()[:, zaxis]
    ztop = float(zc.max())
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    tf = np.sort(dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[zaxis], ztop))).astype(np.int32)
    top_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[zaxis], ztop))
    return tf, top_dofs, ztop


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


# -- Task A2: the 3-D top-facet edge graph (the new coupled geometry) ----------

def test_top_facet_cotangent_equals_negated_ds_top_stiffness_3d():
    """THE correctness gate: the top-facet planar cotangent T_e == the negated off-diagonal of the
    assembled ds_top TANGENTIAL-GRADIENT stiffness (the operator CoupledProblem uses). On a flat top
    gT reduces to the (x,y) gradient, so this is the 3-D analogue of the 2-D
    test_cotangent_T_e_equals_negated_stiffness_offdiagonal_2d."""
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], [6, 3, 3])
    V = fem.functionspace(msh, ("Lagrange", 1))
    tf, _top_dofs, _ztop = _top_facets_dofs(msh, V)

    edges, L_e, T_e, A_i = build_top_facet_edge_graph(V, msh, tf)

    fdim = msh.topology.dim - 1
    ft = dmesh.meshtags(msh, fdim, tf, np.ones(tf.size, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=msh, subdomain_data=ft)(1)
    n_vec = ufl.FacetNormal(msh)
    gT = lambda f: ufl.grad(f) - ufl.dot(ufl.grad(f), n_vec) * n_vec
    u = ufl.TrialFunction(V)
    vt = ufl.TestFunction(V)
    A = assemble_matrix(fem.form(ufl.dot(gT(u), gT(vt)) * ds_top))
    A.assemble()
    Ad = A.convert("dense").getDenseArray()

    maxdiff = 0.0
    for e in range(len(edges)):
        i, j = int(edges[e, 0]), int(edges[e, 1])
        stiff_off = -0.5 * (Ad[i, j] + Ad[j, i])
        maxdiff = max(maxdiff, abs(stiff_off - T_e[e]))
    assert maxdiff < 1e-10, f"top-facet cotangent diverged from ds_top stiffness off-diag by {maxdiff:.2e}"


def test_top_facet_A_i_sums_to_top_area_3d():
    """A_i = lumped surface mass on the top facet: sum over top dofs == top-facet area; interior
    dofs carry zero (so total_water = sum d_i A_i = surface integral, conservation-consistent)."""
    from dolfinx import fem

    Lx, Ly = 2.0, 1.0
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, Ly, 1.0]], [6, 3, 3])
    V = fem.functionspace(msh, ("Lagrange", 1))
    tf, top_dofs, _ztop = _top_facets_dofs(msh, V)

    edges, L_e, T_e, A_i = build_top_facet_edge_graph(V, msh, tf)

    assert A_i.sum() == pytest.approx(Lx * Ly, rel=1e-12)
    assert A_i[top_dofs].sum() == pytest.approx(Lx * Ly, rel=1e-12)
    interior = np.setdiff1d(np.arange(A_i.size), top_dofs)
    assert np.allclose(A_i[interior], 0.0, atol=1e-15)


def test_top_facet_m_matrix_guard_holds_on_structured_box_3d():
    """M-matrix property on the structured box top facet: every edge weight T_e >= -1e-14
    (monotonicity requirement; the loud raise on obtuse triangulations is the shared guard already
    pinned at the 2-D level by test_m_matrix_guard_raises_on_obtuse_mesh_2d)."""
    from dolfinx import fem

    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], [8, 4, 4])
    V = fem.functionspace(msh, ("Lagrange", 1))
    tf, _top_dofs, _ztop = _top_facets_dofs(msh, V)

    edges, L_e, T_e, A_i = build_top_facet_edge_graph(V, msh, tf)

    assert T_e.min() >= -1e-14, f"M-matrix violated on the box top: min T_e = {T_e.min():.3e}"
    assert len(edges) > 0 and msh.topology.dim == 3
