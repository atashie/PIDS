"""Investigate option B (standalone surface mesh + manual coupling) vs A (co-located d on host).

Arik's question: is B worth the extra implementation complexity to avoid A's "2x memory"?
Two measurements:

PART 1 -- what A ACTUALLY costs. Compare a psi-only Richards-stencil operator against the A
co-located monolithic [psi, d-on-host] operator on the SAME mesh: solution size, matrix nonzeros,
and a direct (LU/MUMPS) solve time. The claim to test: A's extra d-DOFs are mostly trivial
Dirichlet identity rows, so the real overhead is far below 2x on the expensive part (the matrix).

PART 2 -- how FIDDLY B is. Actually build B's coupling core: a standalone surface mesh, the
node-matching map to the host top-facet DOFs, and a hand-rolled monolithic 2-block Newton step
with lumped diagonal coupling. Confirm it reproduces the k_ex->inf continuity, and count the
real machinery (the lines A does NOT need).
"""
from __future__ import annotations

import time
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
import dolfinx.fem.petsc as fp
from mpi4py import MPI
from petsc4py import PETSc

COMM = MPI.COMM_WORLD
LU = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def _nnz(A):
    return int(A.getInfo()["nz_used"])


def _time_lu(A, ntrials=3):
    b = A.createVecRight(); b.set(1.0)
    x = A.createVecRight()
    best = 1e30
    for _ in range(ntrials):                      # fresh KSP each trial -> re-factorizes
        ksp = PETSc.KSP().create(COMM); ksp.setOperators(A)
        ksp.setType("preonly"); pc = ksp.getPC()
        pc.setType("lu"); pc.setFactorSolverType("mumps")
        t0 = time.perf_counter(); ksp.solve(b, x); best = min(best, time.perf_counter() - t0)
        ksp.destroy()
    return best


def part1(make_mesh, label):
    host = make_mesh()
    ax = host.geometry.dim - 1                     # "up" = last spatial axis (z in 3-D, y in 2-D)
    fdim = host.topology.dim - 1
    host.topology.create_connectivity(fdim, host.topology.dim)
    top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[ax], 1.0)))
    mt = dmesh.meshtags(host, fdim, top, np.ones(top.size, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=host, subdomain_data=mt)(1)
    nv = ufl.FacetNormal(host)
    bottom = lambda x: np.isclose(x[ax], 0.0)

    # --- psi-only Richards-stencil operator ---
    Vp = fem.functionspace(host, ("Lagrange", 1))
    dpsi, vpsi = ufl.TrialFunction(Vp), ufl.TestFunction(Vp)
    a_psi = (ufl.dot(ufl.grad(dpsi), ufl.grad(vpsi)) + dpsi * vpsi) * ufl.dx
    bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_geometrical(Vp, bottom), Vp)
    A_psi = fp.assemble_matrix(fem.form(a_psi), bcs=[bc_p]); A_psi.assemble()

    # --- A: co-located [psi, d-on-host] monolithic operator ---
    Vd = fem.functionspace(host, ("Lagrange", 1))
    dd, vd = ufl.TrialFunction(Vd), ufl.TestFunction(Vd)
    kc = fem.Constant(host, PETSc.ScalarType(10.0))
    gT = lambda f: ufl.grad(f) - ufl.dot(ufl.grad(f), nv) * nv
    a00 = (ufl.dot(ufl.grad(dpsi), ufl.grad(vpsi)) + dpsi * vpsi) * ufl.dx + kc * dpsi * vpsi * ds_top
    a01 = -kc * dd * vpsi * ds_top
    a10 = -kc * dpsi * vd * ds_top
    offtop = fem.locate_dofs_geometrical(Vd, lambda x: x[ax] < 1.0 - 1e-9)
    bc_d = fem.dirichletbc(PETSc.ScalarType(0.0), offtop, Vd)
    dx_lump = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})

    def build_A(eps_consistent):
        eps_dx = ufl.dx if eps_consistent else dx_lump
        a11 = 0.2 * ufl.dot(gT(dd), gT(vd)) * ds_top + kc * dd * vd * ds_top + 1e-10 * dd * vd * eps_dx
        A = fp.assemble_matrix([[fem.form(a00), fem.form(a01)], [fem.form(a10), fem.form(a11)]],
                               bcs=[bc_p, bc_d], kind="mpi"); A.assemble()
        return A

    A_cons = build_A(True)
    A_lump = build_A(False)
    n_psi = A_psi.getSize()[0]
    nnz_psi, t_psi = _nnz(A_psi), _time_lu(A_psi)
    print(f"\n[PART 1 | {label}]  host DOFs(psi)={n_psi}  top-facet nodes~{top.size}")
    print(f"  {'variant':<22}{'unknowns':>10}{'nnz':>12}{'LU s':>10}{'  nnz x / LU x'}")
    print(f"  {'psi-only':<22}{n_psi:>10}{nnz_psi:>12}{t_psi:>10.4f}    1.00 / 1.00")
    for name, A in (("A co-located (consist.)", A_cons), ("A co-located (LUMPED eps)", A_lump)):
        nco, nnz, t = A.getSize()[0], _nnz(A), _time_lu(A)
        print(f"  {name:<22}{nco:>10}{nnz:>12}{t:>10.4f}    {nnz/nnz_psi:.2f} / {t/t_psi:.2f}")
    A_psi.destroy(); A_cons.destroy(); A_lump.destroy()


def part2():
    print("\n[PART 2] B coupling core: standalone surface mesh + node match + manual block Newton")
    n = 16
    host = dmesh.create_unit_square(COMM, n, n)
    fdim = host.topology.dim - 1
    host.topology.create_connectivity(fdim, host.topology.dim)
    top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[1], 1.0)))
    surf, _e, _v, _g = dmesh.create_submesh(host, fdim, top)  # standalone 1-D surface mesh

    Vp = fem.functionspace(host, ("Lagrange", 1))
    Vd = fem.functionspace(surf, ("Lagrange", 1))   # d lives ONLY on the surface (no volume DOFs)

    # ---- B machinery #1: node-matching map (surface DOF -> host top-facet DOF), by coordinates ----
    xs = Vd.tabulate_dof_coordinates()
    xh = Vp.tabulate_dof_coordinates()
    top_dofs = fem.locate_dofs_topological(Vp, fdim, top)        # host DOFs on the top facets
    xh_top = xh[top_dofs]
    # match each surface DOF to the nearest host top DOF (exact for coincident nodes)
    match = np.empty(xs.shape[0], dtype=np.int32)
    for i, p in enumerate(xs):
        match[i] = top_dofs[int(np.argmin(np.sum((xh_top - p[:xh_top.shape[1]]) ** 2, axis=1)))]
    # lumped nodal weight w_i (top-facet "area" per surface node) from a surface mass row-sum
    one = fem.Function(Vd); one.x.array[:] = 1.0
    wv = ufl.TestFunction(Vd)
    w = fp.assemble_vector(fem.form(one * wv * ufl.dx)); w.assemble()
    wj = w.getArray().copy()                                     # lumped weights, indexed by surface DOF

    # ---- B machinery #2: per-block single-mesh operators (these REUSE M1/M2 forms unchanged) ----
    dpsi, vpsi = ufl.TrialFunction(Vp), ufl.TestFunction(Vp)
    a_psi = (ufl.dot(ufl.grad(dpsi), ufl.grad(vpsi)) + dpsi * vpsi) * ufl.dx
    L_psi = fem.Constant(host, PETSc.ScalarType(0.0)) * vpsi * ufl.dx
    dd, vd = ufl.TrialFunction(Vd), ufl.TestFunction(Vd)
    a_d = (0.2 * ufl.dot(ufl.grad(dd), ufl.grad(vd)) + 1e-10 * dd * vd) * ufl.dx
    rate = 0.3
    L_d = fem.Constant(surf, PETSc.ScalarType(rate)) * vd * ufl.dx
    n_p = Vp.dofmap.index_map.size_local
    n_d = Vd.dofmap.index_map.size_local

    def solve_kex(kex):
        # ---- B machinery #3: assemble block matrix [[A_pp, A_pd],[A_dp, A_dd]] + inject lumped
        #      DIAGONAL coupling at matched nodes; assemble block rhs + coupling; KSP solve ----
        App = fp.assemble_matrix(fem.form(a_psi)); App.assemble()
        Add = fp.assemble_matrix(fem.form(a_d)); Add.assemble()
        bp = fp.assemble_vector(fem.form(L_psi)); bp.assemble()
        bd = fp.assemble_vector(fem.form(L_d)); bd.assemble()
        # build a monolithic [n_p+n_d] system by hand
        N = n_p + n_d
        A = PETSc.Mat().createAIJ([N, N], comm=COMM)
        A.setPreallocationNNZ(60); A.setUp()
        # scatter App into [0:n_p, 0:n_p]
        ai, aj, av = App.getValuesCSR()
        for r in range(n_p):
            cols = aj[ai[r]:ai[r + 1]]; vals = av[ai[r]:ai[r + 1]]
            if cols.size:
                A.setValues([r], cols.astype(np.int32), vals, addv=True)
        di, dj, dv = Add.getValuesCSR()
        for r in range(n_d):
            cols = dj[di[r]:di[r + 1]]; vals = dv[di[r]:di[r + 1]]
            if cols.size:
                A.setValues([n_p + r], (n_p + cols).astype(np.int32), vals, addv=True)
        # lumped coupling q_i = w_i*kex*(psi_match - d_i): diagonal cross + self terms
        rhs = np.concatenate([bp.getArray(), bd.getArray()])
        for i in range(n_d):
            h = int(match[i]); wki = wj[i] * kex
            A.setValues([h], [h], [wki], addv=True)            # dF_psi/dpsi
            A.setValues([h], [n_p + i], [-wki], addv=True)     # dF_psi/dd
            A.setValues([n_p + i], [h], [-wki], addv=True)     # dF_d/dpsi
            A.setValues([n_p + i], [n_p + i], [wki], addv=True)  # dF_d/dd
        A.assemble()
        x = A.createVecRight(); rv = A.createVecLeft(); rv.setArray(rhs); rv.assemble()
        ksp = PETSc.KSP().create(COMM); ksp.setOperators(A)
        ksp.setType("preonly"); ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps"); ksp.solve(rv, x)
        sol = x.getArray()
        psi_top = sol[match]; d = sol[n_p:]
        gap = float(np.sqrt(np.average((psi_top - d) ** 2)))
        ksp.destroy(); A.destroy()
        return gap

    print("  node-match built:", match.shape[0], "surface DOFs matched to host top DOFs")
    for kex in (1e0, 1e2, 1e4, 1e6):
        print(f"    k_ex={kex:7.0e}:  L2|psi_top-d|={solve_kex(kex):.3e}")


if __name__ == "__main__":
    part1(lambda: dmesh.create_unit_square(COMM, 96, 96), "2-D 96x96")
    part1(lambda: dmesh.create_unit_cube(COMM, 20, 20, 20), "3-D 20x20x20")
    part2()
