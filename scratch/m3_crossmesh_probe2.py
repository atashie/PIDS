"""M0 cross-mesh probe 2 (THROWAWAY) -- manual blocked assembly, the REAL Module-3 recipe.

Probe 1 established: (i) the supported cross-mesh direction is integration-domain = SUBMESH
with parent (host) test/trial/coeff pulled in via entity_maps; the reverse (host `ds` integral
referencing a submesh field) fails to JIT. (ii) `fem.form` is single-integration-domain, so the
host residual BLOCK (host-bulk Richards + surface coupling on the submesh) cannot be one form,
hence the high-level NonlinearProblem([F0,F1]) wrapper can't host it.

This probe proves the supported low-level path end to end:
  - compile each single-domain block form with fem.form(..., entity_maps=[s2p]);
  - assemble the 2x2 block Jacobian + 2-block residual as TWO summed block systems
    (block-diagonal "bulk" + full "coupling"), kind=None (monolithic block matrix/vector);
  - manual Newton solve (the linear test below converges in ONE iteration iff the auto-derived
    CROSS-block Jacobian is exact);
  - verify the design-D.3 continuity/datum guard (k_ex->inf => psi_top->d) and structural
    conservation (flux into psi == -flux out of d).

If this PASSES, R1 is de-risked and CoupledProblem (Module 3) is built on exactly this recipe.

Stage-A formulation (linear, no Dirichlet -> trivial BC handling for the probe):
  host block : r0*(psi - psi_ref)*vpsi*dx           (anchors psi; nonsingular A00)   [HOST]
             + k_ex*(psi - d)*vpsi*dx_sub            (Robin land-surface coupling)    [SUBMESH]
  surf block : alpha*(d - d_target)*vd*dx_sub        (anchors d)                       [SUBMESH]
             + k_ex*(d - psi)*vd*dx_sub              (sign-flipped coupling)           [SUBMESH]
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
import dolfinx.fem.petsc as fp
from mpi4py import MPI
from petsc4py import PETSc

COMM = MPI.COMM_WORLD
assert COMM.size == 1, "probe is serial (block-vector slicing assumes one rank)"


def build():
    n = 8
    host = dmesh.create_unit_square(COMM, n, n)
    fdim = host.topology.dim - 1
    host.topology.create_connectivity(fdim, host.topology.dim)
    top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[1], 1.0)))
    submesh, s2p, _v, _g = dmesh.create_submesh(host, fdim, top)
    return host, submesh, s2p


def solve_stage_A(kex_val):
    host, submesh, s2p = build()
    Vpsi = fem.functionspace(host, ("Lagrange", 1))
    Vd = fem.functionspace(submesh, ("Lagrange", 1))
    psi = fem.Function(Vpsi, name="psi")
    d = fem.Function(Vd, name="d")
    vpsi, dpsi = ufl.TestFunction(Vpsi), ufl.TrialFunction(Vpsi)
    vd, dd = ufl.TestFunction(Vd), ufl.TrialFunction(Vd)
    dx = ufl.dx                                   # host volume
    dxs = ufl.Measure("dx", domain=submesh)       # top submesh

    r0, psi_ref, alpha, d_target = 1.0, -2.0, 1.0, 0.5
    kc = fem.Constant(submesh, PETSc.ScalarType(kex_val))

    # --- residual forms (each single integration domain) ---
    L_psi_bulk = r0 * (psi - psi_ref) * vpsi * dx                # HOST
    L_psi_cpl = kc * (psi - d) * vpsi * dxs                      # SUBMESH (host test pulled via s2p)
    L_d = (alpha * (d - d_target) + kc * (d - psi)) * vd * dxs   # SUBMESH

    # --- Jacobian blocks via ufl.derivative (exact, auto) ---
    a00_bulk = ufl.derivative(L_psi_bulk, psi, dpsi)            # HOST   Vpsi x Vpsi
    a00_cpl = ufl.derivative(L_psi_cpl, psi, dpsi)             # SUBMESH Vpsi x Vpsi (top dofs)
    a01_cpl = ufl.derivative(L_psi_cpl, d, dd)                 # SUBMESH Vpsi x Vd   (CROSS)
    a10_cpl = ufl.derivative(L_d, psi, dpsi)                   # SUBMESH Vd   x Vpsi (CROSS)
    a11 = ufl.derivative(L_d, d, dd)                          # SUBMESH Vd   x Vd

    em = [s2p]
    # bulk block system (block-diagonal): only a00_bulk; surf block has no bulk part here.
    a_bulk = [[fem.form(a00_bulk), None],
              [None, None]]
    L_bulk = [fem.form(L_psi_bulk), None]
    # coupling block system (full 2x2), all submesh-domain -> entity_maps.
    a_cpl = [[fem.form(a00_cpl, entity_maps=em), fem.form(a01_cpl, entity_maps=em)],
             [fem.form(a10_cpl, entity_maps=em), fem.form(a11, entity_maps=em)]]
    L_cpl = [fem.form(L_psi_cpl, entity_maps=em), fem.form(L_d, entity_maps=em)]

    n0 = Vpsi.dofmap.index_map.size_local * Vpsi.dofmap.index_map_bs
    n1 = Vd.dofmap.index_map.size_local * Vd.dofmap.index_map_bs

    # manual Newton (linear -> 1 iter); start at zero.
    psi.x.array[:] = 0.0
    d.x.array[:] = 0.0
    last_rnorm = None
    iters_used = 0
    for k in range(4):
        A_bulk = fp.assemble_matrix(a_bulk, kind=None); A_bulk.assemble()
        A_cpl = fp.assemble_matrix(a_cpl, kind=None); A_cpl.assemble()
        A = A_bulk.copy()
        A.axpy(1.0, A_cpl, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

        b_bulk = fp.assemble_vector(L_bulk, kind=None)
        b_cpl = fp.assemble_vector(L_cpl, kind=None)
        b = b_bulk.copy()
        b.axpy(1.0, b_cpl)
        rnorm = b.norm()
        if k == 0 or rnorm > 1e-14:
            # Newton update: A dx = -b
            dx_vec = A.createVecRight()
            ksp = PETSc.KSP().create(COMM)
            ksp.setOperators(A)
            ksp.setType("preonly"); ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            ksp.solve(-b, dx_vec)
            arr = dx_vec.getArray()
            psi.x.array[:] += arr[:n0]
            d.x.array[:] += arr[n0:n0 + n1]
            psi.x.scatter_forward(); d.x.scatter_forward()
            iters_used = k + 1
            ksp.destroy(); dx_vec.destroy()
        A_bulk.destroy(); A_cpl.destroy(); A.destroy(); b_bulk.destroy(); b_cpl.destroy(); b.destroy()
        last_rnorm = rnorm
        if rnorm < 1e-12:
            break

    # continuity / datum guard: L2<psi_top - d> over the submesh.
    gap_num = fem.assemble_scalar(fem.form((psi - d) ** 2 * dxs, entity_maps=em))
    area = fem.assemble_scalar(fem.form(fem.Constant(submesh, PETSc.ScalarType(1.0)) * dxs))
    gap = (gap_num / area) ** 0.5
    # conservation: flux into psi vs out of d (same integrand, opposite sign -> structural).
    q_into_psi = fem.assemble_scalar(fem.form(kc * (psi - d) * dxs, entity_maps=em))
    q_out_of_d = fem.assemble_scalar(fem.form(kc * (d - psi) * dxs, entity_maps=em))
    return dict(kex=kex_val, rnorm=last_rnorm, iters=iters_used, gap=gap,
                qa=q_into_psi, qb=q_out_of_d,
                psi_top_mean=float(np.mean(psi.x.array)), d_mean=float(np.mean(d.x.array)))


if __name__ == "__main__":
    print("=== PROBE 2: manual blocked cross-mesh Newton (Stage A, linear) ===")
    rows = [solve_stage_A(k) for k in (1e0, 1e2, 1e4, 1e6)]
    ok = True
    for r in rows:
        print(f"  k_ex={r['kex']:7.0e}: newton_iters={r['iters']} final_rnorm={r['rnorm']:.2e}  "
              f"L2|psi_top-d|={r['gap']:.3e}  q_in={r['qa']:+.4e} q_out={r['qb']:+.4e} "
              f"sum={r['qa']+r['qb']:+.1e}  d_mean={r['d_mean']:.4f}")
        if r['rnorm'] > 1e-9:
            print(f"  FAIL: residual did not vanish at k_ex={r['kex']:.0e} (Jacobian inexact?)"); ok = False
        if r['iters'] > 1:
            print(f"  WARN: {r['iters']} Newton iters on a LINEAR problem at k_ex={r['kex']:.0e}"); ok = False
        if abs(r['qa'] + r['qb']) > 1e-10 * max(1.0, abs(r['qa'])):
            print(f"  FAIL: flux not antisymmetric at k_ex={r['kex']:.0e}"); ok = False
    gaps = [r['gap'] for r in rows]
    if not all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)):
        print(f"  FAIL: continuity gap not monotone-decreasing in k_ex: {gaps}"); ok = False
    if gaps[-1] > 1e-4:
        print(f"  FAIL: gap at k_ex=1e6={gaps[-1]:.2e} did not collapse (a +z_surf datum bug plateaus ~1)")
        ok = False
    print(f"\n  PROBE 2 VERDICT: {'PASS' if ok else 'FAIL'}  "
          f"(continuity gap 1e0->1e6: {gaps[0]:.2e} -> {gaps[-1]:.2e})")
    print(f"  => R1 cross-mesh blocked assembly + exact auto-Jacobian via manual block path: "
          f"{'USABLE' if ok else 'NOT usable as written'}")
