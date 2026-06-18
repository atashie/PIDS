"""B4 probe 2: confirm the cotangent T_e goes NEGATIVE on an obtuse triangulation.

Build a single obtuse triangle (and its mirror) via dmesh.create_mesh with explicit
coordinates so an interior/shared edge sees an OBTUSE opposite angle -> cot < 0. This is
the mesh the M-matrix guard must REJECT. We just compute T_e here (the guard lives in the
class); this proves the guard has something to fire on.

Obtuse config: a very flat ("sliver") triangle. Two triangles sharing the long edge, each
with an obtuse angle opposite a SHORT edge -> that short edge gets cot(obtuse) < 0.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI


def cot_T_e(msh):
    tdim = msh.topology.dim
    V = fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(tdim, 0)
    msh.topology.create_connectivity(1, 0)
    msh.topology.create_connectivity(1, tdim)
    c2v = msh.topology.connectivity(tdim, 0)
    e2v = msh.topology.connectivity(1, 0)
    e2c = msh.topology.connectivity(1, tdim)
    n_edges = msh.topology.index_map(1).size_local
    x = msh.geometry.x

    def cot_opposite(tri, vi, vj):
        k = [v for v in tri if v != vi and v != vj][0]
        u = x[vi] - x[k]; w = x[vj] - x[k]
        cross = u[0] * w[1] - u[1] * w[0]
        dot = u[0] * w[0] + u[1] * w[1]
        return dot / abs(cross)

    T = np.zeros(n_edges)
    for e in range(n_edges):
        vi, vj = e2v.links(e)
        s = 0.0
        for c in e2c.links(e):
            s += 0.5 * cot_opposite(c2v.links(c), vi, vj)
        T[e] = s
    return T, V


def main():
    # Four points making two obtuse triangles sharing the diagonal edge (1)-(2).
    # Points chosen so triangle (0,1,2) and (1,2,3) are very flat -> obtuse angles.
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.05],   # nearly collinear with 0-1 -> flat triangle
        [0.5, -0.05],
    ], dtype=np.float64)
    cells = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)  # share edge 0-1 (the long one)

    # Build a mesh from explicit geometry/topology (DOLFINx 0.10 create_mesh).
    import basix.ufl
    ufl_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = ufl.Mesh(ufl_el)
    msh = dmesh.create_mesh(MPI.COMM_WORLD, cells, domain, pts)

    T, V = cot_T_e(msh)
    print("obtuse-mesh T_e =", np.round(T, 4))
    print("min T_e =", T.min(), " -> guard should RAISE" if T.min() < -1e-14 else " -> (not obtuse enough)")

    # cross-check against stiffness off-diagonal too
    u = ufl.TrialFunction(V); vt = ufl.TestFunction(V)
    A = assemble_matrix(fem.form(ufl.dot(ufl.grad(u), ufl.grad(vt)) * ufl.dx)); A.assemble()
    Ad = A.convert("dense").getDenseArray()
    print("stiffness matrix off-diagonals:")
    print(np.round(Ad, 4))


if __name__ == "__main__":
    main()
