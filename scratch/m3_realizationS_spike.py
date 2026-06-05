"""Realization-S foundational spike (THROWAWAY): coupled NCP via a CUSTOM blocked Newton.

d,λ live on the top-facet SUBMESH (codim-1 manifold); ψ on the host volume. The explicit interface
flux λ keeps every block single-domain:
    F_ψ = richards_bulk(ψ; host dx) − λ·v_ψ·ds_top        [HOST; refs submesh λ -> entity_maps=[s2p]]
    F_d = overland_residual(d; submesh dx) + (λ−rain)·v_d·dx_sub  [SUBMESH; PURE]
    F_λ = FB(d, τ_c·(q_pot−λ))·v_λ·ds_top, q_pot=K(ψ)/ℓ_c·(d−ψ)  [HOST ds_top; refs ψ,d,λ -> entity_maps]

The high-level NonlinearProblem auto-derives all 9 Jacobian blocks in one compile context with a
UNIFORM entity_maps=[s2p], which trips an FFCX bug on the pure-submesh manifold self-Jacobian
∂F_d/∂d (UnboundLocalError, empty element-table). FIX (Codex-reviewed): a custom blocked Newton
that compiles EACH residual/Jacobian block SEPARATELY with PER-BLOCK entity_maps (None for
pure-submesh/pure-host, [s2p] for cross-mesh), compiling the pure-submesh manifold blocks FIRST,
then assembles one kind="mpi" block system and Newton-iterates. (Serial array-slicing here is a
spike convenience; the production driver must be MPI-safe.)

Stage A: flat top, rain<Ks on unsaturated soil ⇒ reduces to the 1-D column (d≈0, all infiltrates,
closed column conserves). Proves the FFCX bug is dodged + the nonlinear coupled solve converges.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
import dolfinx.fem.petsc as fp
from mpi4py import MPI
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import richards_bulk_residual
from pids_forward.physics.overland import overland_residual
from pids_forward.physics.coupling import _fischer_burmeister

COMM = MPI.COMM_WORLD
assert COMM.size == 1, "spike uses serial block-vector slicing"
SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def build(nx=10, ny=24, height=2.0, n_man=0.05, eps_ncp=1e-4):
    host = dmesh.create_rectangle(COMM, [[0.0, 0.0], [1.0, height]], [nx, ny])
    gdim = host.geometry.dim
    fdim = host.topology.dim - 1
    host.topology.create_connectivity(fdim, host.topology.dim)
    top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[1], height)))
    submesh, s2p, _v, _g = dmesh.create_submesh(host, fdim, top)
    mt = dmesh.meshtags(host, fdim, top, np.ones(top.size, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=host, subdomain_data=mt)(1)
    dx_sub = ufl.Measure("dx", domain=submesh)
    dx_sub_l = ufl.Measure("dx", domain=submesh, metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
    dx_l = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})

    Vpsi = fem.functionspace(host, ("Lagrange", 1))
    Vd = fem.functionspace(submesh, ("Lagrange", 1))
    Vlam = fem.functionspace(submesh, ("Lagrange", 1))
    psi = fem.Function(Vpsi, name="psi"); psi_n = fem.Function(Vpsi, name="psi_n")
    d = fem.Function(Vd, name="d"); d_n = fem.Function(Vd, name="d_n")
    lam = fem.Function(Vlam, name="lam")
    z_b = fem.Function(Vd, name="z_b")
    vpsi, vd, vlam = ufl.TestFunction(Vpsi), ufl.TestFunction(Vd), ufl.TestFunction(Vlam)
    dpsi, dd, dlam = ufl.TrialFunction(Vpsi), ufl.TrialFunction(Vd), ufl.TrialFunction(Vlam)

    e_arr = np.zeros(gdim, dtype=PETSc.ScalarType); e_arr[-1] = 1.0
    e_g = fem.Constant(host, e_arr)
    dt = fem.Constant(host, PETSc.ScalarType(1.0))
    rain = fem.Constant(submesh, PETSc.ScalarType(0.0))
    ell_c = 0.5 * height / ny
    tau_c = ell_c / SOIL.Ks

    q_pot = (SOIL.K_ufl(psi) / ell_c) * (d - psi)
    L_psi = richards_bulk_residual(psi, psi_n, vpsi, SOIL, dt, e_g, dx_storage=dx_l) - lam * vpsi * ds_top
    L_d = overland_residual(d, d_n, vd, z_b, n_man, dt, 1e-3, dx=dx_sub, dx_storage=dx_sub_l) \
        + (lam - rain) * vd * dx_sub
    L_lam = _fischer_burmeister(d, tau_c * (q_pot - lam), eps_ncp) * vlam * ds_top

    EM = [s2p]
    # ---- compile blocks SEPARATELY, PURE-SUBMESH manifold blocks FIRST (Codex belt-and-suspenders) ----
    f = fem.form
    J_dd = f(ufl.derivative(L_d, d, dd))                       # pure submesh (the FFCX-fragile one) -> FIRST
    J_dl = f(ufl.derivative(L_d, lam, dlam))                   # pure submesh
    Lc_d = f(L_d)                                             # pure submesh
    J_pp = f(ufl.derivative(L_psi, psi, dpsi))                 # pure host
    J_pl = f(ufl.derivative(L_psi, lam, dlam), entity_maps=EM)  # cross
    J_lp = f(ufl.derivative(L_lam, psi, dpsi), entity_maps=EM)  # cross
    J_ld = f(ufl.derivative(L_lam, d, dd), entity_maps=EM)      # cross
    J_ll = f(ufl.derivative(L_lam, lam, dlam), entity_maps=EM)  # cross
    Lc_psi = f(L_psi, entity_maps=EM)                          # cross
    Lc_lam = f(L_lam, entity_maps=EM)                          # cross

    Jac = [[J_pp, None, J_pl], [None, J_dd, J_dl], [J_lp, J_ld, J_ll]]
    Res = [Lc_psi, Lc_d, Lc_lam]
    A = fp.create_matrix(Jac, kind="mpi")
    b = fp.create_vector(Res, kind="mpi")
    n0 = Vpsi.dofmap.index_map.size_local * Vpsi.dofmap.index_map_bs
    n1 = Vd.dofmap.index_map.size_local * Vd.dofmap.index_map_bs
    n2 = Vlam.dofmap.index_map.size_local * Vlam.dofmap.index_map_bs
    return dict(host=host, submesh=submesh, Vpsi=Vpsi, Vd=Vd, psi=psi, psi_n=psi_n, d=d, d_n=d_n,
                lam=lam, z_b=z_b, dt=dt, rain=rain, Jac=Jac, Res=Res, A=A, b=b, n=(n0, n1, n2),
                dx_l=dx_l, dx_sub_l=dx_sub_l, ell_c=ell_c)


def newton_step(S, dt, rtol=1e-9, atol=1e-11, max_it=30):
    S["dt"].value = dt
    n0, n1, n2 = S["n"]
    for f in (S["psi"], S["d"], S["lam"]):
        f.x.scatter_forward()
    rnorm0 = None
    for k in range(max_it):
        S["b"].zeroEntries()
        fp.assemble_vector(S["b"], S["Res"])
        S["b"].assemble()
        rnorm = S["b"].norm()
        if rnorm0 is None:
            rnorm0 = rnorm
        if rnorm < atol or (rnorm0 > 0 and rnorm < rtol * rnorm0):
            return True, k
        S["A"].zeroEntries()
        fp.assemble_matrix(S["A"], S["Jac"])
        S["A"].assemble()
        delta = S["A"].createVecRight()
        ksp = PETSc.KSP().create(COMM); ksp.setOperators(S["A"])
        ksp.setType("preonly"); ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        nb = S["b"].copy(); nb.scale(-1.0)
        ksp.solve(nb, delta)
        a = delta.getArray()
        S["psi"].x.array[:] += a[:n0]
        S["d"].x.array[:] += a[n0:n0 + n1]
        S["lam"].x.array[:] += a[n0 + n1:n0 + n1 + n2]
        for f in (S["psi"], S["d"], S["lam"]):
            f.x.scatter_forward()
        ksp.destroy(); delta.destroy(); nb.destroy()
    return False, max_it


def soil_water(S):
    return COMM.allreduce(fem.assemble_scalar(fem.form(SOIL.theta_ufl(S["psi"]) * S["dx_l"])), op=MPI.SUM)


def surface_water(S):
    return COMM.allreduce(fem.assemble_scalar(fem.form(S["d"] * S["dx_sub_l"])), op=MPI.SUM)


def stage_A():
    print("\n=== STAGE A: flat-top 2-D host, supply-limited (rain<Ks), custom blocked Newton ===")
    S = build()
    S["psi"].interpolate(lambda x: -2.0 + 0.0 * x[0]); S["psi_n"].interpolate(lambda x: -2.0 + 0.0 * x[0])
    S["z_b"].x.array[:] = 0.0
    w0 = soil_water(S) + surface_water(S)
    rate, t_end = 0.1, 0.5
    S["rain"].value = rate
    top_len = COMM.allreduce(fem.assemble_scalar(fem.form(
        fem.Constant(S["submesh"], PETSc.ScalarType(1.0)) * S["dx_sub_l"])), op=MPI.SUM)
    t, dt, nsteps, maxit = 0.0, 1e-3, 0, 0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        ok, it = newton_step(S, h)
        if ok:
            S["psi_n"].x.array[:] = S["psi"].x.array; S["psi_n"].x.scatter_forward()
            S["d_n"].x.array[:] = S["d"].x.array; S["d_n"].x.scatter_forward()
            t += h; nsteps += 1; maxit = max(maxit, it)
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 0.05)
        else:
            S["psi"].x.array[:] = S["psi_n"].x.array; S["psi"].x.scatter_forward()
            S["d"].x.array[:] = S["d_n"].x.array; S["d"].x.scatter_forward()
            dt *= 0.5
            if dt < 1e-7:
                print("  FAIL: dt collapse"); return False
    cum_rain = rate * top_len * t_end
    dW = (soil_water(S) + surface_water(S)) - w0
    dmax = float(S["d"].x.array.max()); dmin = float(S["d"].x.array.min())
    massbal = abs(dW - cum_rain) / cum_rain
    finite = np.all(np.isfinite(S["psi"].x.array)) and np.all(np.isfinite(S["d"].x.array))
    dspread = dmax - dmin
    print(f"  steps={nsteps} max_newton_iters={maxit}  d_max={dmax:.3e} d_min={dmin:.3e}  "
          f"cum_rain={cum_rain:.5f} dTotal={dW:.5f}  massbal_err={massbal:.2e}  finite={finite}")
    print(f"  d spread along (flat) top = {dspread:.2e}  (should be ~0)")
    ok = (maxit > 0) and (massbal < 1e-6) and (dmax < 1e-3) and finite and (dmin >= -1e-9)
    print(f"  STAGE A: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    a = stage_A()
    print(f"\n=== realization-S custom blocked Newton (per-block compile) USABLE: {'YES' if a else 'NO'} ===")
