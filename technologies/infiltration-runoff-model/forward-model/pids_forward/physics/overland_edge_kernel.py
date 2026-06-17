"""Pure-function core of the O1 monotone upwind-mobility edge-flux overland scheme.

Extracted (Convergent-flow P2, Part A) from the validated standalone ``UpwindOverlandProblem``
(``overland_upwind.py``) so the SAME edge-graph + residual math can be reused by the coupled
``CoupledProblem`` (P2) WITHOUT duplication. The extraction is behaviour-preserving and bit-identical
to the standalone internals -- pinned by ``tests/test_overland_edge_kernel.py`` (graph + residual
equality) and by the full standalone suite ``tests/test_overland_upwind.py`` staying green.

Two pieces:
  * ``build_edge_graph_2d(V, mesh)`` -- the 2-D triangle-mesh edge graph: unique P1-dof edge pairs,
    edge lengths ``L_e``, the cotangent two-point transmissibility ``T_e`` (= the negated P1
    stiffness off-diagonal, the standard monotone FV dual-mesh weight), the lumped control areas
    ``A_i = int phi_i dx`` (vertex quadrature; ``sum A_i = domain area``), with the loud M-matrix
    guard (``T_e >= 0``; obtuse/bad-split triangulations raise). See the standalone module docstring
    for the full derivation; the P2 coupled path adds a top-facet variant (Part A2).
  * ``edge_flux_residual(...)`` -- the lumped backward-Euler node residual
    ``R_i = (d_i - d_n,i) A_i/dt + sum_{e in i} +/-Q_e - r A_i (+ outflow sinks)`` with the
    smoothed-upwind Manning edge flux ``Q_e = T_e M(d_up)(H_i - H_j)``, ``H = z_b + d``,
    ``d_up`` the C1 tanh-upwind depth. Telescoping edge signs (+ on i, - on j) make discrete mass
    conservation structural (the lateral flux sums to zero over the surface).

Units: length m, time day, slope dimensionless; ``n_man`` SI s.m^{-1/3}; ``SECONDS_PER_DAY``
converts the SI Manning conveyance to m^2/day.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem

SECONDS_PER_DAY = 86400.0


def build_edge_graph_2d(V, mesh):
    """Build the 2-D triangle-mesh edge graph for the P1 space ``V`` on ``mesh``.

    Returns ``(edges, L_e, T_e, A_i)`` -- (n_edges, 2) int32 dof pairs; (n_edges,) edge lengths;
    (n_edges,) cotangent transmissibility (>= 0, guarded); (n_dofs,) lumped control areas. Raises
    ``ValueError`` (M-matrix guard) if any ``T_e < -1e-14`` (obtuse opposite angle / bad split).
    """
    from dolfinx.fem.petsc import assemble_vector

    n_dofs = V.dofmap.index_map.size_local
    tdim = mesh.topology.dim
    top = mesh.topology
    top.create_connectivity(tdim, 0)   # cell -> vertices (for the dof map + cotangents)
    top.create_connectivity(1, 0)      # edge -> vertices
    top.create_connectivity(1, tdim)   # edge -> cells (1 or 2 triangles per edge)
    c2v = top.connectivity(tdim, 0)
    e2v = top.connectivity(1, 0)
    e2c = top.connectivity(1, tdim)
    n_cells = top.index_map(tdim).size_local
    n_verts = top.index_map(0).size_local
    n_edges = top.index_map(1).size_local
    x = mesh.geometry.x  # vertex coordinates (gdim columns)

    # vertex -> P1 dof, from the cell-local correspondence (robust; matches cell_dofs order).
    vtx_to_dof = np.full(n_verts, -1, dtype=np.int64)
    for c in range(n_cells):
        verts = c2v.links(c)
        dofs = V.dofmap.cell_dofs(c)
        for k in range(len(verts)):
            vtx_to_dof[verts[k]] = dofs[k]

    def _cot_opposite(tri, vi, vj):
        """cotangent of the angle at the third (opposite) vertex of ``tri`` for edge vi-vj."""
        vk = [v for v in tri if v != vi and v != vj]
        k = int(vk[0])
        u = x[vi] - x[k]
        w = x[vj] - x[k]
        cross = u[0] * w[1] - u[1] * w[0]      # 2-D cross-product z-component (= 2*area, signed)
        dot = u[0] * w[0] + u[1] * w[1]
        return dot / abs(cross)                # cot = cos/sin = (u.w)/|u x w|

    edges = np.empty((n_edges, 2), dtype=np.int32)
    L_e = np.empty(n_edges, dtype=np.float64)
    T_e = np.zeros(n_edges, dtype=np.float64)
    for e in range(n_edges):
        vi, vj = (int(v) for v in e2v.links(e))
        di, dj = int(vtx_to_dof[vi]), int(vtx_to_dof[vj])
        edges[e] = (di, dj)
        L_e[e] = float(np.linalg.norm(x[vi] - x[vj]))
        s = 0.0
        for c in e2c.links(e):
            s += 0.5 * _cot_opposite(c2v.links(c), vi, vj)
        T_e[e] = s

    # M-MATRIX GUARD: loud failure on any negative transmissibility (obtuse triangle / bad split).
    if T_e.min() < -1e-14:
        bad = int(np.argmin(T_e))
        i, j = int(edges[bad, 0]), int(edges[bad, 1])
        raise ValueError(
            "M-matrix property violated: the cotangent transmissibility is NEGATIVE on edge "
            f"(dofs {i}, {j}), T_e = {T_e[bad]:.6e} < -1e-14. The monotone upwind scheme REQUIRES "
            "T_e >= 0 for every edge (an obtuse opposite angle gives cot < 0 -> an anti-diffusive "
            "edge that breaks the discrete maximum principle). The upwind scheme is restricted "
            "to structured-box / Delaunay NON-OBTUSE triangulations; remesh (a structured "
            "create_rectangle box, or a Delaunay mesh) so all opposite angles are <= 90 degrees."
        )

    # A_i = lumped P1 mass int phi_i dx (vertex quadrature); sum_i A_i = domain area.
    v = ufl.TestFunction(V)
    mass_form = fem.form(
        v * ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
    )
    bm = assemble_vector(mass_form)
    bm.assemble()
    A_i = bm.getArray()[:n_dofs].copy()
    bm.destroy()

    return edges, L_e, T_e, A_i


def edge_flux_residual(d, z_b, d_n, rain, dt, edges, L_e, T_e, A_i, n_man, eps_S, eps_H,
                       outflows=()):
    """Lumped backward-Euler node residual ``R`` for the upwind edge-flux scheme.

    ``R_i = (d_i - d_n,i) A_i/dt - r A_i + sum_{e in i} +/-Q_e (+ outflow sinks)`` with the
    smoothed-upwind Manning edge flux ``Q_e = T_e M(d_up)(H_i - H_j)`` (``H = z_b + d``). Edge sign
    ``+Q_e`` on the i-row, ``-Q_e`` on the j-row (telescoping -> structural conservation). No-flux
    boundaries are automatic (a boundary node simply has fewer incident edges). ``outflows`` is an
    iterable of ``(dofs, slope, B)`` Manning normal-depth free-drainage sinks (``B`` = the node's
    boundary control length; 1.0 at a 1-D point outlet). ``d`` is the current depth array; the
    return is a fresh array (``d`` is never mutated).
    """
    R = (d - d_n) * A_i / dt - rain * A_i

    i = edges[:, 0]
    j = edges[:, 1]
    H_i = z_b[i] + d[i]
    H_j = z_b[j] + d[j]
    dH = H_i - H_j                                   # edge head drop (i - j)
    slope = dH / L_e                                 # signed edge slope

    # Smoothed C1 upstream weight: w -> 1 when H_i >> H_j (take d_i), -> 0 when H_j >> H_i.
    w = 0.5 * (1.0 + np.tanh(dH / eps_H))
    d_up = w * d[i] + (1.0 - w) * d[j]
    d_up_pos = np.maximum(d_up, 0.0)                 # guard the fractional power
    slope_root = (slope * slope + eps_S ** 2) ** 0.25  # Manning slope floor (inside root)
    M = SECONDS_PER_DAY * d_up_pos ** (5.0 / 3.0) / (n_man * slope_root)
    Q_e = T_e * M * dH                               # edge flux (>0: i -> j)

    # Telescoping accumulation: +Q_e on the i-row, -Q_e on the j-row (np.add.at handles the
    # repeated indices of interior nodes shared by two edges).
    np.add.at(R, i, Q_e)
    np.add.at(R, j, -Q_e)

    # Outflow sinks: water LEAVES the located node(s); +q_out*B in the residual (mirrors the
    # galerkin +q_out v ds boundary sink, ds supplying the length factor B). max(d,0) guards the power.
    for dofs, oslope, B in outflows:
        q_out = (SECONDS_PER_DAY * (1.0 / n_man)
                 * np.maximum(d[dofs], 0.0) ** (5.0 / 3.0) * np.sqrt(oslope)) * B
        np.add.at(R, dofs, q_out)

    return R
