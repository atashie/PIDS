"""B4 probe: verify the 2-D edge graph + cotangent T_e + vertex<->dof map on a tiny mesh.

Checks, on a 2x2 structured create_rectangle triangle mesh:
  1. P1 vertex<->dof correspondence (the map we need to put the edge graph in dof space).
  2. mesh.topology edge->vertex connectivity (1,0) and edge->cell (1,tdim).
  3. the cotangent T_e = 1/2 (cot a + cot b) summed over the 1 or 2 triangles on each edge,
     and that it EQUALS the negated P1 stiffness off-diagonal assembled by DOLFINx/UFL.
  4. the M-matrix property T_e >= 0 on the structured box.
  5. lumped P1 mass A_i = int phi_i dx sums to the domain area.
Run via the WSL wrapper.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI


def main():
    nx, ny = 2, 2
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (1.0, 1.0)], [nx, ny],
                                 cell_type=dmesh.CellType.triangle)
    tdim = msh.topology.dim
    print("tdim =", tdim, " gdim =", msh.geometry.dim)

    V = fem.functionspace(msh, ("Lagrange", 1))
    ndofs = V.dofmap.index_map.size_local
    print("ndofs =", ndofs, " n_vertices =", msh.topology.index_map(0).size_local)

    # --- (1) vertex<->dof map. For P1, geometry_to_dofmap / the dof coords vs vertex coords. ---
    dof_coords = V.tabulate_dof_coordinates()
    vtx_coords = msh.geometry.x
    # The standard P1 vertex->dof: dolfinx gives V.dofmap via cell_dofs; the vertex-to-dof
    # relationship is exposed through mesh.geometry. Build it by matching coordinates AND via
    # the cell-local correspondence (cell vertices order == cell dofs order for P1).
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    n_cells = msh.topology.index_map(tdim).size_local
    vtx_to_dof = -np.ones(msh.topology.index_map(0).size_local, dtype=np.int64)
    for c in range(n_cells):
        verts = c2v.links(c)
        dofs = V.dofmap.cell_dofs(c)
        for k in range(len(verts)):
            vtx_to_dof[verts[k]] = dofs[k]
    print("vtx_to_dof =", vtx_to_dof)
    # sanity: dof coord at vtx_to_dof[v] == vertex coord v
    maxerr = 0.0
    for v in range(len(vtx_to_dof)):
        maxerr = max(maxerr, float(np.linalg.norm(dof_coords[vtx_to_dof[v]] - vtx_coords[v])))
    print("vertex<->dof coord match max err =", maxerr)

    # --- (2) edge connectivity ---
    msh.topology.create_connectivity(1, 0)
    msh.topology.create_connectivity(1, tdim)
    e2v = msh.topology.connectivity(1, 0)
    e2c = msh.topology.connectivity(1, tdim)
    n_edges = msh.topology.index_map(1).size_local
    print("n_edges =", n_edges)

    # --- (3) cotangent T_e from triangle geometry ---
    x = msh.geometry.x

    def cot_opposite(tri_verts, vi, vj):
        """cotangent of the angle at the third vertex (opposite edge vi-vj) in triangle tri_verts."""
        vk = [v for v in tri_verts if v != vi and v != vj]
        assert len(vk) == 1, (tri_verts, vi, vj)
        k = vk[0]
        u = x[vi] - x[k]
        w = x[vj] - x[k]
        cross = u[0] * w[1] - u[1] * w[0]
        dot = u[0] * w[0] + u[1] * w[1]
        return dot / abs(cross)  # cot = cos/sin = dot/|cross|

    T_e = np.zeros(n_edges)
    edge_dofs = np.zeros((n_edges, 2), dtype=np.int64)
    for e in range(n_edges):
        vi, vj = e2v.links(e)
        edge_dofs[e] = (vtx_to_dof[vi], vtx_to_dof[vj])
        cells = e2c.links(e)
        s = 0.0
        for c in cells:
            tri = c2v.links(c)
            s += 0.5 * cot_opposite(tri, vi, vj)
        T_e[e] = s
    print("T_e (cotangent) =", np.round(T_e, 6))
    print("min T_e =", T_e.min())

    # --- (3b) compare against the negated P1 stiffness off-diagonal ---
    u = ufl.TrialFunction(V)
    vt = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(vt)) * ufl.dx
    A = assemble_matrix(fem.form(a))
    A.assemble()
    Ad = A.convert("dense").getDenseArray()
    maxdiff = 0.0
    for e in range(n_edges):
        i, j = int(edge_dofs[e, 0]), int(edge_dofs[e, 1])
        stiff_off = -0.5 * (Ad[i, j] + Ad[j, i])  # symmetric; -off-diag
        maxdiff = max(maxdiff, abs(stiff_off - T_e[e]))
    print("max |T_e(cotan) - (-stiffness offdiag)| =", maxdiff)

    # --- (5) lumped P1 mass A_i = int phi_i dx, sum == area ---
    one = fem.Function(V)
    one.x.array[:] = 1.0
    Mlump = fem.form(vt * ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1}))
    from dolfinx.fem.petsc import assemble_vector
    bm = assemble_vector(Mlump)
    A_i = bm.getArray().copy()
    print("A_i (lumped mass) =", np.round(A_i, 6))
    print("sum A_i =", A_i.sum(), " (domain area = 1.0)")


if __name__ == "__main__":
    main()
