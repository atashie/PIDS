"""M0 cross-mesh spike (Module 3 / design R1 gate) -- THROWAWAY de-risk script.

Question (the dominant Module-3 build risk): can DOLFINx 0.10 assemble a BLOCKED
[psi(host volume), d(top-facet submesh)] residual with an EXACT auto-derived coupled
Jacobian (J=None -> ufl.derivative across the two meshes) and solve it with one Newton
(PETSc SNES), realizing the land-surface exchange q_ls = k_ex*(d - psi_top) sign-paired
into both blocks?

If YES -> freeze the coupling contract = blocked [psi, d] with d on the host top-facet
submesh (design A.2 working default, D.3 Robin-B). If NO -> fall back to host-mesh
co-dim facet restriction / immersed-for-all (design A.2 R1 ladder, F.3.2).

KEY MECHANICS finding (this spike): cross-mesh coupling must be integrated on the SUBMESH
(codim-0) domain in the natural sub->parent direction. The host test/trial functions
(vpsi, psi) are pulled onto the submesh facets via the entity map; the host residual block
receives those contributions on its top-facet DOFs. Integrating the coupling on the HOST
`ds` measure with a submesh coefficient fails to JIT-compile (no element table for the
submesh field on host facets) -- so the residual is structured "everything cross-mesh on
dx_sub".

Realization tested (the real coupling pattern, kept minimal):
  - host = 2-D unit square; gravity / elevation along the LAST axis (y); top facets y=1.
  - psi : P1 on the 2-D host (Richards primary var).  d : P1 on the 1-D top-facet submesh.
  - F_psi = host-bulk(dx)  +  coupling k_ex(psi-d) vpsi  on dx_sub.
  - F_d   = surface terms  +  coupling k_ex(d-psi) vd    on dx_sub  (sign-flipped, SAME integrand).
  The two coupling terms are the same scalar with opposite sign on the same quadrature ->
  conservation is STRUCTURAL.

Three things must hold:
  (1) the blocked Newton ASSEMBLES + SOLVES (R1 mechanism works at all);
  (2) DATUM + CONTINUITY guard: as k_ex -> inf, L2<psi_top - d> -> ~0 (design D.3). On a top
      at z_surf=1 a spurious +z_surf in the driving potential would PLATEAU the gap near 1
      instead of ->0, so monotone decay to ~machine-zero is the gravity-datum guard;
  (3) CONSERVATION: flux into psi == minus flux out of d (sign-paired -> structural).
Stage B swaps the linear psi block for the REAL van Genuchten/Mualem Richards nonlinearity
to confirm the auto-Jacobian handles the real constitutive law across the coupled blocks,
and that the coupled global water budget closes < 1e-6.

Run (PYTHONPATH=. so Stage B's VanGenuchten import resolves):
  cd <forward-model>; PYTHONPATH=. python <repo>/scratch/m3_crossmesh_spike.py
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

COMM = MPI.COMM_WORLD
SOLVER_OPTS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1e-12,
    "snes_atol": 1e-13,
    "snes_max_it": 50,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",  # block LU over the whole coupled operator
}


def _build_host_and_top_submesh(n=16):
    host = dmesh.create_unit_square(COMM, n, n)
    fdim = host.topology.dim - 1
    host.topology.create_connectivity(fdim, host.topology.dim)
    top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[1], 1.0)))
    submesh, sub_to_parent, _vmap, _gmap = dmesh.create_submesh(host, fdim, top)
    dx_sub = ufl.Measure("dx", domain=submesh)
    return host, submesh, sub_to_parent, dx_sub, fdim


def _l2_gap(expr_sq, submesh, s2p, dx_sub):
    """sqrt(<expr_sq>) averaged over the submesh (expr may reference host fields via s2p)."""
    num = submesh.comm.allreduce(
        fem.assemble_scalar(fem.form(expr_sq * dx_sub, entity_maps=[s2p])), op=MPI.SUM)
    one = fem.Constant(submesh, PETSc.ScalarType(1.0))
    area = submesh.comm.allreduce(fem.assemble_scalar(fem.form(one * dx_sub)), op=MPI.SUM)
    return (num / area) ** 0.5 if area > 0 else 0.0


def stage_A():
    print("\n=== STAGE A: linear blocked cross-mesh coupling (R1 mechanism + datum + conservation) ===")
    host, submesh, s2p, dx_sub, fdim = _build_host_and_top_submesh(16)

    Vpsi = fem.functionspace(host, ("Lagrange", 1))
    Vd = fem.functionspace(submesh, ("Lagrange", 1))
    psi = fem.Function(Vpsi, name="psi")
    d = fem.Function(Vd, name="d")
    vpsi = ufl.TestFunction(Vpsi)
    vd = ufl.TestFunction(Vd)

    psi_bot, d_target, alpha = -2.0, 0.5, 1.0
    bot_dofs = fem.locate_dofs_geometrical(Vpsi, lambda x: np.isclose(x[1], 0.0))
    bc_psi = fem.dirichletbc(PETSc.ScalarType(psi_bot), bot_dofs, Vpsi)

    results = {}
    for kex in (1e0, 1e2, 1e4, 1e6):
        psi.x.array[:] = 0.0
        d.x.array[:] = 0.0
        kc = fem.Constant(submesh, PETSc.ScalarType(kex))  # lives on the integration (submesh) domain
        # F_psi: host-bulk diffusion + Robin coupling k_ex(psi - d) on the top submesh.
        F_psi = ufl.dot(ufl.grad(psi), ufl.grad(vpsi)) * ufl.dx + kc * (psi - d) * vpsi * dx_sub
        # F_d: reaction toward d_target + sign-FLIPPED coupling k_ex(d - psi) (same integrand, -sign).
        F_d = alpha * (d - d_target) * vd * dx_sub + kc * (d - psi) * vd * dx_sub

        problem = NonlinearProblem(
            [F_psi, F_d], [psi, d], bcs=[bc_psi],
            petsc_options_prefix="spikeA_", petsc_options=SOLVER_OPTS, entity_maps=[s2p],
        )
        problem.solve()
        snes = problem.solver
        conv = snes.getConvergedReason()
        iters = snes.getIterationNumber()

        gap = _l2_gap((psi - d) ** 2, submesh, s2p, dx_sub)
        q_into_psi = submesh.comm.allreduce(
            fem.assemble_scalar(fem.form(kc * (psi - d) * dx_sub, entity_maps=[s2p])), op=MPI.SUM)
        q_out_of_d = submesh.comm.allreduce(
            fem.assemble_scalar(fem.form(kc * (d - psi) * dx_sub, entity_maps=[s2p])), op=MPI.SUM)
        results[kex] = (conv, iters, gap, q_into_psi, q_out_of_d)
        print(f"  k_ex={kex:7.0e}: reason={conv:+d} iters={iters}  "
              f"L2|psi_top-d|={gap:.3e}  q_into_psi={q_into_psi:+.6e}  "
              f"q_out_of_d={q_out_of_d:+.6e}  sum={q_into_psi+q_out_of_d:+.2e}")

    ok = True
    for kex, (conv, iters, gap, qa, qb) in results.items():
        if conv <= 0:
            print(f"  FAIL: k_ex={kex:.0e} did not converge (reason {conv})"); ok = False
        if iters > 2:
            print(f"  WARN: k_ex={kex:.0e} took {iters} Newton iters (linear -> expect 1; "
                  "Jacobian may be inexact)")
        if abs(qa + qb) > 1e-9 * max(1.0, abs(qa)):
            print(f"  FAIL: k_ex={kex:.0e} flux not antisymmetric: {qa:+.3e} + {qb:+.3e}"); ok = False
    gaps = [results[k][2] for k in (1e0, 1e2, 1e4, 1e6)]
    if not all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)):
        print(f"  FAIL: gap not monotonically decreasing in k_ex: {gaps}"); ok = False
    if gaps[-1] > 1e-4:
        print(f"  FAIL: gap at k_ex=1e6 = {gaps[-1]:.3e} did not collapse toward 0 "
              "(a +z_surf datum bug would plateau near 1)"); ok = False
    print(f"  STAGE A: {'PASS' if ok else 'FAIL'}  (gap 1e0->1e6: {gaps[0]:.2e} -> {gaps[-1]:.2e})")
    return ok


def stage_B():
    print("\n=== STAGE B: REAL van Genuchten Richards block coupled across meshes (nonlinear) ===")
    try:
        from pids_forward.physics.constitutive import VanGenuchten
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP: could not import VanGenuchten ({e}); run with PYTHONPATH=. from forward-model/")
        return None

    host, submesh, s2p, dx_sub, fdim = _build_host_and_top_submesh(16)
    soil = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=1.0)

    Vpsi = fem.functionspace(host, ("Lagrange", 1))
    Vd = fem.functionspace(submesh, ("Lagrange", 1))
    psi = fem.Function(Vpsi, name="psi"); psi_n = fem.Function(Vpsi, name="psi_n")
    d = fem.Function(Vd, name="d"); d_n = fem.Function(Vd, name="d_n")
    vpsi = ufl.TestFunction(Vpsi); vd = ufl.TestFunction(Vd)

    psi.interpolate(lambda x: -1.0 - x[1]); psi_n.interpolate(lambda x: -1.0 - x[1])
    d.x.array[:] = 0.05; d_n.x.array[:] = 0.05
    dt = fem.Constant(host, PETSc.ScalarType(1e-3))
    dt_sub = fem.Constant(submesh, PETSc.ScalarType(1e-3))

    gdim = host.geometry.dim
    e_arr = np.zeros(gdim, dtype=PETSc.ScalarType); e_arr[-1] = 1.0
    e_g = fem.Constant(host, e_arr)
    dx_lump = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
    dx_sub_lump = ufl.Measure("dx", domain=submesh,
                              metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})

    l_c = 1.0 / 16.0
    kex = soil.K_ufl(psi) / l_c  # K_rel*Ks/l_c, psi-dependent rate limiter (design D.3)

    theta = soil.theta_ufl(psi); theta_n = soil.theta_ufl(psi_n); K = soil.K_ufl(psi)
    F_psi = ((theta - theta_n) / dt) * vpsi * dx_lump \
        + K * ufl.dot(ufl.grad(psi) + e_g, ufl.grad(vpsi)) * ufl.dx \
        + kex * (psi - d) * vpsi * dx_sub
    F_d = ((d - d_n) / dt_sub) * vd * dx_sub_lump + kex * (d - psi) * vd * dx_sub

    bot_dofs = fem.locate_dofs_geometrical(Vpsi, lambda x: np.isclose(x[1], 0.0))
    bc_psi = fem.dirichletbc(PETSc.ScalarType(-1.0), bot_dofs, Vpsi)

    problem = NonlinearProblem(
        [F_psi, F_d], [psi, d], bcs=[bc_psi],
        petsc_options_prefix="spikeB_", petsc_options=SOLVER_OPTS, entity_maps=[s2p],
    )

    def store_sub():
        return host.comm.allreduce(fem.assemble_scalar(fem.form(theta * dx_lump)), op=MPI.SUM)

    def store_ovl():
        return submesh.comm.allreduce(fem.assemble_scalar(fem.form(d * dx_sub_lump)), op=MPI.SUM)

    s0 = store_sub() + store_ovl()
    ok = True
    for step in range(8):
        problem.solve()
        snes = problem.solver
        if snes.getConvergedReason() <= 0:
            print(f"  FAIL: step {step} reason {snes.getConvergedReason()}"); ok = False; break
        psi_n.x.array[:] = psi.x.array; psi_n.x.scatter_forward()
        d_n.x.array[:] = d.x.array; d_n.x.scatter_forward()
        sb, so = store_sub(), store_ovl()
        print(f"  step {step}: iters={snes.getIterationNumber()}  store_sub={sb:.6e} "
              f"store_ovl={so:.6e}  total={sb+so:.6e}  d_mean={d.x.array.mean():.4e}")
    s1 = store_sub() + store_ovl()
    rel = abs(s1 - s0) / max(abs(s0), 1e-12)
    print(f"  global mass balance closed: |dtotal|/scale = {rel:.3e}  (gate < 1e-6)")
    if rel > 1e-6:
        print("  FAIL: coupled global mass balance violated"); ok = False
    if not (np.all(np.isfinite(psi.x.array)) and np.all(np.isfinite(d.x.array))):
        print("  FAIL: NaN/Inf in coupled state"); ok = False
    if d.x.array.mean() >= 0.05:
        print(f"  WARN: surface store did not infiltrate (d_mean={d.x.array.mean():.4e} >= 0.05)")
    else:
        print(f"  infiltration observed: d_mean 0.05 -> {d.x.array.mean():.4e}  "
              f"(soil psi_top_mean rose toward 0: {psi.x.array.max():.3f})")
    print(f"  STAGE B: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    a = stage_A()
    b = stage_B()
    print("\n=== SPIKE VERDICT ===")
    print(f"  Stage A (linear cross-mesh mechanism + datum + conservation): {'PASS' if a else 'FAIL'}")
    print(f"  Stage B (real nonlinear Richards block, coupled, mass-balanced): "
          f"{'PASS' if b else ('SKIP' if b is None else 'FAIL')}")
    verdict = a and (b is not False)
    print(f"  R1 (cross-mesh ufl.derivative + blocked SNES) USABLE: {'YES' if verdict else 'NO'}")
