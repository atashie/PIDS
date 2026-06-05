"""M0 cross-mesh probe 3 (THROWAWAY) -- the FALLBACK realization: single host mesh.

Probes 1-2 established that the design's preferred top-facet-SUBMESH blocked [psi, d] does NOT
assemble in this DOLFINx 0.10 / FFCX: the surface (d) block's coupling needs psi's TRACE on the
codim-1 facet submesh, and FFCX asserts codim = tdim_integration - elem_cell_dim >= 0, which is
-1 for a host VOLUME field (dim 2) on a facet domain (dim 1). Volume<->facet-submesh monolithic
coupling with an exact cross-mesh Jacobian is therefore unavailable out-of-the-box.

This probe confirms the clean fallback (design A.2 sanctions "a facet-restricted field OR a
co-dim-1 submesh"): co-locate psi AND d on the SAME host mesh. Then the land-surface coupling
q_ls = k_ex*(psi_top - d) is a pure HOST `ds_top` facet integral -- standard single-mesh FEM,
NO entity_maps, NO cross-mesh tabulation -- so the high-level blocked NonlinearProblem assembles
the EXACT auto-derived coupled Jacobian. d's non-surface DOFs are pinned to 0 by Dirichlet, so
surface storage is exactly the facet integral of d on the top.

Checks: (1) blocked Newton assembles + solves; (2) datum/continuity guard k_ex->inf => psi_top
-> d on the top (a +z_surf datum bug would plateau the gap ~ the top elevation, here 1);
(3) structural conservation flux_into_psi == -flux_out_of_d on the SAME ds_top integrand.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

COMM = MPI.COMM_WORLD
OPTS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_rtol": 1e-12, "snes_atol": 1e-13, "snes_max_it": 50,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


def solve(kex_val, n=16):
    host = dmesh.create_unit_square(COMM, n, n)
    fdim = host.topology.dim - 1
    host.topology.create_connectivity(fdim, host.topology.dim)
    top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[1], 1.0)))
    mt = dmesh.meshtags(host, fdim, top, np.ones(top.size, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=host, subdomain_data=mt)(1)
    n_vec = ufl.FacetNormal(host)

    Vpsi = fem.functionspace(host, ("Lagrange", 1))
    Vd = fem.functionspace(host, ("Lagrange", 1))   # SAME mesh
    psi = fem.Function(Vpsi, name="psi")
    d = fem.Function(Vd, name="d")
    vpsi, vd = ufl.TestFunction(Vpsi), ufl.TestFunction(Vd)

    kc = fem.Constant(host, PETSc.ScalarType(kex_val))
    psi_ref, D_surf, rate = -2.0, 0.2, 0.3

    # psi block: bulk diffusion anchored to psi_ref + Robin land-surface coupling on the top.
    F_psi = ufl.dot(ufl.grad(psi), ufl.grad(vpsi)) * ufl.dx \
        + 1.0 * (psi - psi_ref) * vpsi * ufl.dx \
        + kc * (psi - d) * vpsi * ds_top
    # d block: surface (tangential-gradient) diffusion + rain + sign-flipped coupling, ALL on ds_top.
    gd = ufl.grad(d)
    gd_T = gd - ufl.dot(gd, n_vec) * n_vec               # tangential surface gradient
    gv = ufl.grad(vd)
    gv_T = gv - ufl.dot(gv, n_vec) * n_vec
    F_d = D_surf * ufl.dot(gd_T, gv_T) * ds_top \
        + kc * (d - psi) * vd * ds_top \
        - rate * vd * ds_top

    # pin d to 0 everywhere except the free top surface (so 'd' is a top-facet-restricted field).
    offtop = fem.locate_dofs_geometrical(Vd, lambda x: x[1] < 1.0 - 1e-9)
    bc_d = fem.dirichletbc(PETSc.ScalarType(0.0), offtop, Vd)

    problem = NonlinearProblem(
        [F_psi, F_d], [psi, d], bcs=[bc_d],
        petsc_options_prefix="p3_", petsc_options=OPTS,
        kind="mpi",  # monolithic block AIJ (not MATNEST) so a single LU/MUMPS can factor it
    )
    problem.solve()
    snes = problem.solver
    conv, iters = snes.getConvergedReason(), snes.getIterationNumber()

    # continuity gap on the TOP only (L2 over ds_top).
    gap2 = host.comm.allreduce(fem.assemble_scalar(fem.form((psi - d) ** 2 * ds_top)), op=MPI.SUM)
    area = host.comm.allreduce(fem.assemble_scalar(
        fem.form(fem.Constant(host, PETSc.ScalarType(1.0)) * ds_top)), op=MPI.SUM)
    gap = (gap2 / area) ** 0.5
    qa = host.comm.allreduce(fem.assemble_scalar(fem.form(kc * (psi - d) * ds_top)), op=MPI.SUM)
    qb = host.comm.allreduce(fem.assemble_scalar(fem.form(kc * (d - psi) * ds_top)), op=MPI.SUM)
    return dict(kex=kex_val, conv=conv, iters=iters, gap=gap, qa=qa, qb=qb,
                d_top_mean=float(d.x.array[top.size and slice(None)].max()))


if __name__ == "__main__":
    print("=== PROBE 3: single-host-mesh land-surface coupling (the FALLBACK realization) ===")
    rows = [solve(k) for k in (1e0, 1e2, 1e4, 1e6)]
    ok = True
    for r in rows:
        print(f"  k_ex={r['kex']:7.0e}: reason={r['conv']:+d} iters={r['iters']}  "
              f"L2|psi_top-d|={r['gap']:.3e}  q_in={r['qa']:+.4e} q_out={r['qb']:+.4e} "
              f"sum={r['qa']+r['qb']:+.1e}")
        if r['conv'] <= 0:
            print(f"  FAIL: not converged at k_ex={r['kex']:.0e}"); ok = False
        if abs(r['qa'] + r['qb']) > 1e-10 * max(1.0, abs(r['qa'])):
            print(f"  FAIL: flux not antisymmetric at k_ex={r['kex']:.0e}"); ok = False
    gaps = [r['gap'] for r in rows]
    if not all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)):
        print(f"  FAIL: continuity gap not monotone-decreasing: {gaps}"); ok = False
    if gaps[-1] > 1e-4:
        print(f"  FAIL: gap at k_ex=1e6={gaps[-1]:.2e} did not collapse (a +z_surf datum bug plateaus ~1)")
        ok = False
    print(f"\n  PROBE 3 VERDICT: {'PASS' if ok else 'FAIL'}  (continuity gap 1e0->1e6: "
          f"{gaps[0]:.2e} -> {gaps[-1]:.2e})")
    print(f"  => single-host-mesh fallback (facet-restricted d, exact auto-Jacobian): "
          f"{'USABLE' if ok else 'NOT usable'}")
