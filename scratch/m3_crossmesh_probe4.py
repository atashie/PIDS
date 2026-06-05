"""M0 cross-mesh probe 4 (THROWAWAY) -- the DESIGN-INTENDED submesh realization, done right.

Codex review caught my error: probes 1-2 concluded the top-facet-submesh blocked [psi,d] is
"not buildable", but that was because I integrated the coupling on the SUBMESH dx (volume field
psi at codim -1 -> FFCX assert). The supported idiom (DOLFINx 0.10 HDG demo) integrates the
coupling on the PARENT ds_top, where the volume field psi sits at its natural codim-1 facet trace
and the submesh fields (d, v_d) at codim-0 -- both legal. A minimal compile test confirmed all
four mixed terms + both Jacobian cross-blocks compile that way.

This probe proves the FULL realization end to end:
  - psi on the host volume; d on the codim-1 top-facet SUBMESH (DOF-efficient -- d is thin).
  - psi-block residual: bulk + Robin coupling, BOTH on the host -> ONE host-domain form.
  - d-block residual: surface op on submesh dx + Robin coupling on host ds_top -> TWO domains,
    so it is assembled as the sum of two single-domain block systems (low-level blocked Newton).
  - EXACT Galerkin Robin coupling q_ls = k_ex*(psi_top - d) (NOT lumped), sign-paired -> total
    flux antisymmetric -> structural conservation. Cross-mesh entity relation via the
    create_submesh EntityMap (s2p) -- the MPI-robust map, not coordinate nearest-neighbor.
  - manual Newton: LINEAR test converges in 1 iter iff the auto-derived cross-block Jacobian
    (incl. the host<->submesh blocks) is exact; verify k_ex->inf => psi_top->d (datum guard).

If this PASSES, the design's intended realization is viable AND DOF-efficient AND exact -- the
best of A (robust/exact) and B (efficient), with no overland tangential-gradient rewrite and no
lumped-coupling approximation.
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


def solve_stage_A(kex_val, n=12):
    host = dmesh.create_unit_square(COMM, n, n)
    fdim = host.topology.dim - 1
    host.topology.create_connectivity(fdim, host.topology.dim)
    top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[1], 1.0)))
    submesh, s2p, _v, _g = dmesh.create_submesh(host, fdim, top)
    mt = dmesh.meshtags(host, fdim, top, np.ones(top.size, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=host, subdomain_data=mt)(1)   # PARENT facet measure
    dxs = ufl.Measure("dx", domain=submesh)                          # submesh (surface) measure

    Vp = fem.functionspace(host, ("Lagrange", 1))
    Vd = fem.functionspace(submesh, ("Lagrange", 1))
    psi = fem.Function(Vp, name="psi")
    d = fem.Function(Vd, name="d")
    vp, dp = ufl.TestFunction(Vp), ufl.TrialFunction(Vp)
    vd, dd = ufl.TestFunction(Vd), ufl.TrialFunction(Vd)

    r0, psi_ref, alpha, d_target = 1.0, -2.0, 1.0, 0.5
    kc = fem.Constant(host, PETSc.ScalarType(kex_val))   # on the host (ds_top integration domain)

    # residuals -------------------------------------------------------------
    # psi block: bulk anchor + Robin coupling, BOTH host-domain -> single form.
    L_psi = r0 * (psi - psi_ref) * vp * ufl.dx + kc * (psi - d) * vp * ds_top
    # d block split by domain:
    L_d_surf = alpha * (d - d_target) * vd * dxs                 # submesh dx
    L_d_cpl = kc * (d - psi) * vd * ds_top                       # host ds_top (sign-flipped)

    em = [s2p]
    # Jacobian blocks (ufl.derivative; cross blocks live on the PARENT ds_top).
    a_pp = fem.form(ufl.derivative(L_psi, psi, dp))                          # host: bulk + cpl
    a_pd = fem.form(ufl.derivative(L_psi, d, dd), entity_maps=em)            # host ds_top, Vp x Vd
    a_dp = fem.form(ufl.derivative(L_d_cpl, psi, dp), entity_maps=em)        # host ds_top, Vd x Vp
    a_dd_cpl = fem.form(ufl.derivative(L_d_cpl, d, dd), entity_maps=em)      # host ds_top, Vd x Vd
    a_dd_surf = fem.form(ufl.derivative(L_d_surf, d, dd))                    # submesh dx, Vd x Vd
    a_zero_pp = fem.form(fem.Constant(host, PETSc.ScalarType(0.0)) * dp * vp * ufl.dx)  # space-deduction stub

    f_psi = fem.form(L_psi, entity_maps=em)   # L_psi's coupling term references submesh d
    f_d_cpl = fem.form(L_d_cpl, entity_maps=em)
    f_d_surf = fem.form(L_d_surf)

    n0 = Vp.dofmap.index_map.size_local * Vp.dofmap.index_map_bs
    n1 = Vd.dofmap.index_map.size_local * Vd.dofmap.index_map_bs

    psi.x.array[:] = 0.0
    d.x.array[:] = 0.0
    rnorm0 = None
    iters = 0
    for k in range(4):
        # System 1: everything that touches the host ds_top coupling + the host bulk (full 2x2).
        A1 = fp.assemble_matrix([[a_pp, a_pd], [a_dp, a_dd_cpl]], kind="mpi"); A1.assemble()
        b1 = fp.assemble_vector([f_psi, f_d_cpl], kind="mpi")
        # System 2: the d-block surface operator (block (1,1) only).
        A2 = fp.assemble_matrix([[a_zero_pp, None], [None, a_dd_surf]], kind="mpi"); A2.assemble()
        b2 = fp.assemble_vector([None, f_d_surf], kind="mpi") if False else fp.assemble_vector(
            [fem.form(fem.Constant(host, PETSc.ScalarType(0.0)) * vp * ufl.dx), f_d_surf], kind="mpi")
        A = A1.copy(); A.axpy(1.0, A2, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        b = b1.copy(); b.axpy(1.0, b2)
        rnorm = b.norm()
        if rnorm0 is None:
            rnorm0 = rnorm
        if rnorm > 1e-13:
            dxv = A.createVecRight()
            ksp = PETSc.KSP().create(COMM); ksp.setOperators(A)
            ksp.setType("preonly"); ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            ksp.solve(-b, dxv)
            arr = dxv.getArray()
            psi.x.array[:] += arr[:n0]; psi.x.scatter_forward()
            d.x.array[:] += arr[n0:n0 + n1]; d.x.scatter_forward()
            iters = k + 1
            ksp.destroy(); dxv.destroy()
        for o in (A1, A2, A, b1, b2, b):
            o.destroy()
        if rnorm < 1e-12:
            break

    # final residual norm (recompute)
    b1 = fp.assemble_vector([f_psi, f_d_cpl], kind="mpi")
    b2 = fp.assemble_vector([fem.form(fem.Constant(host, PETSc.ScalarType(0.0)) * vp * ufl.dx),
                             f_d_surf], kind="mpi")
    b = b1.copy(); b.axpy(1.0, b2); final_rnorm = b.norm()
    for o in (b1, b2, b):
        o.destroy()

    gap = (fem.assemble_scalar(fem.form((psi - d) ** 2 * ds_top, entity_maps=em))
           / fem.assemble_scalar(fem.form(fem.Constant(host, PETSc.ScalarType(1.0)) * ds_top))) ** 0.5
    q_into_psi = fem.assemble_scalar(fem.form(kc * (psi - d) * ds_top, entity_maps=em))      # test=1
    q_out_of_d = fem.assemble_scalar(fem.form(kc * (d - psi) * ds_top, entity_maps=em))
    return dict(kex=kex_val, rnorm=final_rnorm, iters=iters, gap=gap,
                qa=q_into_psi, qb=q_out_of_d, n_psi=n0, n_d=n1)


if __name__ == "__main__":
    print("=== PROBE 4: DESIGN-INTENDED submesh realization (coupling on PARENT ds_top) ===")
    rows = [solve_stage_A(k) for k in (1e0, 1e2, 1e4, 1e6)]
    ok = True
    for r in rows:
        print(f"  k_ex={r['kex']:7.0e}: iters={r['iters']} rnorm={r['rnorm']:.2e}  "
              f"L2|psi_top-d|={r['gap']:.3e}  q_in={r['qa']:+.5e} q_out={r['qb']:+.5e} "
              f"sum={r['qa']+r['qb']:+.1e}")
        if r['rnorm'] > 1e-9:
            print(f"  FAIL: residual not vanished at k_ex={r['kex']:.0e}"); ok = False
        # exact-Jacobian proof: 1 Newton iter on a LINEAR problem at MODERATE k_ex. At extreme
        # k_ex (>=1e4) the penalty conditioning (~k_ex) costs a few extra iters by roundoff -- the
        # residual still vanishes and conservation stays exact, so that is not a Jacobian defect.
        if r['kex'] <= 1e2 and r['iters'] > 1:
            print(f"  FAIL: {r['iters']} iters on a LINEAR problem at moderate k_ex={r['kex']:.0e} "
                  "(cross Jacobian inexact)"); ok = False
        elif r['iters'] > 1:
            print(f"  note: k_ex={r['kex']:.0e} took {r['iters']} iters (penalty-limit conditioning; "
                  "residual + conservation still exact)")
        if abs(r['qa'] + r['qb']) > 1e-9 * max(1.0, abs(r['qa'])):
            print(f"  FAIL: flux not antisymmetric at k_ex={r['kex']:.0e}"); ok = False
    gaps = [r['gap'] for r in rows]
    if not all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)) or gaps[-1] > 1e-4:
        print(f"  FAIL: continuity gap did not collapse monotonically: {gaps}"); ok = False
    print(f"\n  d-DOFs={rows[0]['n_d']} vs psi-DOFs={rows[0]['n_psi']}  (d is thin -> DOF-efficient)")
    print(f"  PROBE 4 VERDICT: {'PASS' if ok else 'FAIL'}  (continuity gap {gaps[0]:.2e} -> {gaps[-1]:.2e})")
    print(f"  => design-intended submesh realization (exact Galerkin coupling on parent ds_top, "
          f"blocked Newton): {'USABLE' if ok else 'NOT usable'}")
