"""Module 4 (§E) Phase-4 Task 5: MEASURE the P1-ridge discrete well index r_0(h) (Peaceman-for-FEM).

A line source on a P1 lattice produces a DISCRETE log field whose well-block value u_h(Gamma) is
finite while the analytic field diverges; Peaceman's insight: u_h(Gamma) equals the ANALYTIC solution
at some equivalent radius r_0 ~ const*h (0.208*h for the FD 5-point stencil). The Phase-4 exchange
bridges wall->cell through WI = 2*pi/ln(r_0(h)/r_w), so r_0 must be MEASURED for OUR lattices:

  3-D: the structured TET lattice of dolfinx create_box (the embedded-harness host), ridge = the
       vertex line y=z=L/2 along x (exactly how EmbeddedFeature places a feature);
  2-D: the structured TRIANGLE lattice of create_rectangle, "ridge" = the center vertex (the
       cross-section analog; literature-comparable).

Method (exactly checkable): solve steady Laplace  -div(grad u) = delta_Gamma  (unit source per unit
length: RHS = v dGamma on the ridge measure / v dP at the vertex), with the ANALYTIC log
u = -(1/2pi) ln(rho) imposed as Dirichlet data on the lateral boundary (valid for ANY boundary shape:
the analytic solution is global, so the exact continuum solution IS the log everywhere). Then
  r_0 = exp(-2*pi * u_h(Gamma)),
and two facts make r_0 load-bearing rather than a knob:
  (a) far-field fidelity: away from the ridge the discrete field must BE the analytic log (slope of
      u_h vs ln(rho) = -1/(2pi)); (b) r_0/h is an h-independent lattice constant.
Both are asserted in tests/test_well_index_p1.py; the sweep below reports r_0/h across n.

x-ends of the 3-D box get the natural (no-flux) BC -- exact for the x-invariant solution; u_h(Gamma)
is read on interior-x ridge vertices (ends excluded) and its x-spread is reported.

Plan: docs/plans/2026-06-10-m4-phase4-coupled-embedding-plan.md (Task 5).
Run from forward-model/:
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scratch/m4_phase4_well_index.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

COMM = MPI.COMM_WORLD
_LU = {"ksp_type": "preonly", "pc_type": "lu"}


def _analytic(xc, c):
    rho = np.hypot(xc[0] - c, xc[1] - c)
    return -np.log(np.maximum(rho, 1e-300)) / (2.0 * np.pi)


def _log_slope(u_arr, coords, c, h, L):
    """Fit u_h against ln(rho) over the probe band rho in [2h, min(4h, 0.3L)] -> the discrete source
    strength seen by the far field (analytic: -1/(2pi))."""
    rho = np.hypot(coords[:, 0] - c, coords[:, 1] - c)
    band = (rho >= 2.0 * h - 1e-12) & (rho <= max(4.0 * h, 0.3 * L) + 1e-12) & (rho <= 0.35 * L)
    assert band.sum() >= 3, f"probe band too thin ({band.sum()} vertices)"
    A = np.vstack([np.log(rho[band]), np.ones(band.sum())]).T
    slope, _ = np.linalg.lstsq(A, u_arr[band], rcond=None)[0]
    return float(slope)


def measure_r0_tet(n, L=1.0, nx=4):
    """r_0 of the vertex-line ridge source on the structured create_box TET lattice (n even)."""
    assert n % 2 == 0, "n must be even (the ridge line y=z=L/2 must lie on the vertex lattice)"
    c, h = L / 2.0, L / n
    Lx = nx * h
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [Lx, L, L]], [nx, n, n])
    tdim = msh.topology.dim
    for d in (0, 1, tdim - 1):
        msh.topology.create_connectivity(d, tdim)
        msh.topology.create_connectivity(tdim, d)
    edges = np.sort(dmesh.locate_entities(
        msh, 1, lambda x: np.isclose(x[1], c) & np.isclose(x[2], c))).astype(np.int32)
    assert edges.size > 0
    ft = dmesh.meshtags(msh, 1, edges, np.ones(edges.size, dtype=np.int32))
    dG = ufl.Measure("ridge", domain=msh, subdomain_data=ft)(1)

    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = v * dG                                            # unit line source per unit length
    g = fem.Function(V)
    g.interpolate(lambda x: _analytic((x[1], x[2]), c))
    lat = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], L)
        | np.isclose(x[2], 0.0) | np.isclose(x[2], L))
    bcs = [fem.dirichletbc(g, lat)]
    uh = LinearProblem(a, rhs, bcs=bcs, petsc_options_prefix="wi3_", petsc_options=_LU).solve()
    if isinstance(uh, tuple):                               # API tolerance across 0.10.x
        uh = uh[0]

    xc = V.tabulate_dof_coordinates()
    on_line = np.isclose(xc[:, 1], c) & np.isclose(xc[:, 2], c)
    interior = on_line & (xc[:, 0] > 0.25 * Lx - 1e-12) & (xc[:, 0] < 0.75 * Lx + 1e-12)
    uG = uh.x.array[interior]
    spread = float(uG.max() - uG.min())
    u0 = float(uG.mean())
    # far-field probes on the mid x-plane only (x-invariance is checked via `spread`)
    mid = np.isclose(xc[:, 0], Lx / 2.0)
    slope = _log_slope(uh.x.array[mid], xc[mid][:, 1:3], c, h, L)
    return {"r0": float(np.exp(-2.0 * np.pi * u0)), "h": h, "u0": u0,
            "x_spread": spread, "log_slope": slope, "n": n}


def measure_r0_tri(n, L=1.0):
    """r_0 of the center-VERTEX point source on the structured create_rectangle TRIANGLE lattice
    (the 2-D cross-section analog; codim-2 in 2-D = a vertex, so the source is a vertex dP term)."""
    assert n % 2 == 0
    c, h = L / 2.0, L / n
    msh = dmesh.create_rectangle(COMM, [[0.0, 0.0], [L, L]], [n, n], dmesh.CellType.triangle)
    tdim = msh.topology.dim
    for d in (0, tdim - 1):
        msh.topology.create_connectivity(d, tdim)
        msh.topology.create_connectivity(tdim, d)
    verts = np.sort(dmesh.locate_entities(
        msh, 0, lambda x: np.isclose(x[0], c) & np.isclose(x[1], c))).astype(np.int32)
    assert verts.size == 1
    vt = dmesh.meshtags(msh, 0, verts, np.ones(verts.size, dtype=np.int32))
    dP = ufl.Measure("vertex", domain=msh, subdomain_data=vt)(1)

    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = v * dP                                            # unit point source
    g = fem.Function(V)
    g.interpolate(lambda x: _analytic((x[0], x[1]), c))
    bnd = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], L)
        | np.isclose(x[1], 0.0) | np.isclose(x[1], L))
    bcs = [fem.dirichletbc(g, bnd)]
    uh = LinearProblem(a, rhs, bcs=bcs, petsc_options_prefix="wi2_", petsc_options=_LU).solve()
    if isinstance(uh, tuple):
        uh = uh[0]

    xc = V.tabulate_dof_coordinates()
    onc = np.isclose(xc[:, 0], c) & np.isclose(xc[:, 1], c)
    u0 = float(uh.x.array[onc][0])
    slope = _log_slope(uh.x.array, xc[:, 0:2], c, h, L)
    return {"r0": float(np.exp(-2.0 * np.pi * u0)), "h": h, "u0": u0,
            "x_spread": 0.0, "log_slope": slope, "n": n}


if __name__ == "__main__":
    an = -1.0 / (2.0 * np.pi)
    print("=" * 88)
    print("PHASE-4: P1-ridge discrete well index r_0(h)  (unit source, analytic-log Dirichlet truth)")
    print("=" * 88)
    for kind, measure, ns in (("TET-3D (create_box lattice, vertex-line ridge)", measure_r0_tet,
                               (8, 12, 16, 24, 32)),
                              ("TRI-2D (create_rectangle lattice, center vertex)", measure_r0_tri,
                               (8, 12, 16, 24, 32, 48, 64))):
        print(f"\n{kind}:")
        vals = []
        for n in ns:
            r = measure(n)
            vals.append(r["r0"] / r["h"])
            print(f"   n={n:3d}  h={r['h']:.4f}  u_h(G)={r['u0']:+.5f}  x-spread={r['x_spread']:.1e}  "
                  f"log-slope={r['log_slope']:+.5f} (vs {an:+.5f}, dev {abs(r['log_slope']-an)/abs(an):.2%})  "
                  f"r0={r['r0']:.5f}  r0/h={vals[-1]:.4f}", flush=True)
        vals = np.array(vals)
        print(f"   --> r0/h = {vals.mean():.4f}  (spread {vals.std():.4f} = "
              f"{vals.std()/vals.mean():.2%}; FD-Peaceman analog 0.208)")
    print("=" * 88)
