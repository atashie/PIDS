"""Extract 2-D ANNULUS theta(y,z) field snapshots for the Phase-1 Tier-3 field viz.

Runs the disperse and drain near-field annulus (SAND -- the largest gravity asymmetry) to a mid-window
time and saves the node (y,z) coordinates + theta field + the wall/far circles to an npz the field HTML
reads. Disperse (dry soil, saturated wall) should be ~radially symmetric (gravity negligible at low K);
DRAIN (saturated soil, draining wall) should be gravity-ASYMMETRIC (the saturated K=Ks gravity throughflow
skews the desaturation top vs bottom) -- the visual counterpart of the drain/tunnel ratio rising to ~1.5.

Run from forward-model/:  PYTHONPATH=. OMP_NUM_THREADS=1 ... python scratch/m4_phase1_field_snapshots.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

import scratch.m4_phase1b_disperse_reference as dz

COMM = MPI.COMM_WORLD
SAND = dz.SOILS["SAND"]
R_W, R_OUT = dz.R_W, dz.R_OUT


def snapshot(soil, psi_i, psi_wall, t_snap, *, n_r=100, n_phi=160, grade=1.5):
    """March the (y,z) annulus to t_snap (gravity ON) and return node coords + theta + psi.

    grade=1.5 (NOT the drain reference's 2.5): the visual snapshot only needs to SHOW the field, and the
    disperse `cp` solve spuriously stalls (silent no-change "convergence") on the ultra-fine ~um near-wall
    cells of grade>=2.0 against the saturated-wall gradient -- a known steep-grade x cp-linesearch fragility
    (the drain's bt tolerates it). Moderate grading resolves both directions cleanly. Asserts wetting below."""
    msh = dz._annulus_mesh(R_W, R_OUT, n_r, n_phi, grade)
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi = fem.Function(V); psi_n = fem.Function(V)
    psi.x.array[:] = psi_i; psi_n.x.array[:] = psi_i
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    dxs = dz._vertex_dx(); dxq = ufl.dx(metadata={"quadrature_degree": 8})
    theta, theta_n, K = soil.theta_ufl(psi), soil.theta_ufl(psi_n), soil.K_ufl(psi)
    eg = fem.Constant(msh, PETSc.ScalarType([0.0, 1.0]))
    F = ((theta - theta_n) / dt_c) * v * dxs + K * ufl.dot(ufl.grad(psi) + eg, ufl.grad(v)) * dxq
    wall = fem.locate_dofs_geometrical(V, lambda x: np.isclose(np.hypot(x[0], x[1]), R_W, atol=1e-7))
    far = fem.locate_dofs_geometrical(V, lambda x: np.isclose(np.hypot(x[0], x[1]), R_OUT, atol=1e-7))
    bcs = [fem.dirichletbc(PETSc.ScalarType(psi_wall), wall, V),
           fem.dirichletbc(PETSc.ScalarType(psi_i), far, V)]
    # drain (saturated start) needs bt; disperse (dry start) needs cp -- pick by direction.
    opts = dz._LU if psi_wall > psi_i else dict(dz._LU, snes_linesearch_type="bt")
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="snap_", petsc_options=opts)
    dz._solve_to(problem, psi, psi_n, dt_c, 0.0, t_snap, 1e-8)
    xc = V.tabulate_dof_coordinates()
    return xc[:, 0].copy(), xc[:, 1].copy(), soil.theta(psi.x.array), psi.x.array.copy()


if __name__ == "__main__":
    out = {}
    # disperse: dry soil psi_i=-1, saturated wall psi=0  (gravity ~negligible -> symmetric)
    y, z, th, ps = snapshot(SAND, -1.0, 0.0, 2.0e-3)
    assert th.max() - th.min() > 0.1, f"disperse snapshot did not wet (theta flat at {th.min():.3f}) -- solve stalled"
    out.update(disp_y=y, disp_z=z, disp_theta=th, disp_psi=ps)
    print(f"disperse SAND snapshot: {y.size} nodes, theta in [{th.min():.3f},{th.max():.3f}]", flush=True)
    # drain: saturated soil psi_i=0, draining wall psi=-1  (gravity significant -> asymmetric)
    y, z, th, ps = snapshot(SAND, 0.0, -1.0, 3.0e-3)
    assert th.max() - th.min() > 0.1, f"drain snapshot did not desaturate (theta flat at {th.min():.3f})"
    out.update(drain_y=y, drain_z=z, drain_theta=th, drain_psi=ps)
    print(f"drain SAND snapshot:    {y.size} nodes, theta in [{th.min():.3f},{th.max():.3f}]", flush=True)
    out.update(r_w=R_W, r_out=R_OUT, theta_s=SAND.theta_s, theta_r=SAND.theta_r)
    np.savez("scratch/m4_phase1_field_snapshots.npz", **out)
    print("WROTE scratch/m4_phase1_field_snapshots.npz", flush=True)
