"""refD40-B: the DEPLOYMENT-SCALE drain HISTORY reference (LOAM, R=40 r_w, replenishment pulse).

The drain mirror of refB40-disperse: the refD40 scenario (psi_i=-0.03 just below air entry, wall
-1, 20-d window) with a deficit-aware replenishment pulse -- march to t1, measure the band's theta
deficit, inject 50% of it over [t1, t2] into an annular band in the BULK (far from the wall), and
keep draining. The replenishment raises the bulk state mid-history, so the wall uptake re-steepens:
a scheme can land the end state ONLY by tracking the host's live state through the pulse (an
offline/fixed-drive clock integrates the unperturbed curve and misses). Band [24, 32] r_w (the
refB40-disperse band -- outer bulk, host-mediated transmission to the near-wall drive); pulse
window [7, 10.5] d (mirrors refB40; t1 ~ where refD40 has drained ~half its window-end I).

Run (WSL): conda activate pids-fem && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
    python scratch/m4_phase4_refD40B_drain.py
"""
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ufl
from mpi4py import MPI
from dolfinx import fem

import scratch.m4_phase1b_disperse_reference as dz
import scratch.m4_phase4_refAB_drain as gen

COMM = MPI.COMM_WORLD
R_W = gen.R_W
PSI_I40, PSI_WALL = -0.03, -1.0
BAND = (24 * R_W, 32 * R_W)
T1, T2 = 7.0, 10.5
PULSE_FILL = 0.5                       # inject 50% of the band's measured deficit (deficit-aware)


def main():
    # NOTE: gen._setup hardwires its pulse band to (3,4)r_w, so the variational problem is rebuilt
    # here with the [24,32]r_w band (same numerics/constants as gen._setup otherwise).
    soil = dz.SOILS["LOAM"]
    ref = np.load(pathlib.Path(__file__).parent / "m4_phase4_refD40_drain.npz")
    t = ref["LOAM_t"]
    # ---- build the 1-D radial closed problem with the [24,32]r_w pulse band -----------------
    from dolfinx import mesh as dmesh
    from dolfinx.fem.petsc import NonlinearProblem
    from petsc4py import PETSc
    r_out, cell = 40 * R_W, gen.CELL
    n = max(int(round((r_out - R_W) / cell)), 80)
    msh = dmesh.create_interval(COMM, n, [R_W, r_out])
    r = ufl.SpatialCoordinate(msh)[0]
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi, psi_n = fem.Function(V), fem.Function(V)
    psi.x.array[:] = PSI_I40
    psi_n.x.array[:] = PSI_I40
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    s_c = fem.Constant(msh, PETSc.ScalarType(0.0))
    dxs, dxq = dz._vertex_dx(), ufl.dx(metadata={"quadrature_degree": 8})
    theta, theta_n = soil.theta_ufl(psi), soil.theta_ufl(psi_n)
    K = soil.K_ufl(psi)
    band_ind = ufl.conditional(ufl.And(ufl.ge(r, BAND[0]), ufl.le(r, BAND[1])), 1.0, 0.0)
    F = (((theta - theta_n) / dt_c) * v * r * dxs
         + K * ufl.dot(ufl.grad(psi), ufl.grad(v)) * r * dxq
         - s_c * band_ind * v * r * dxq)
    wall_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], R_W))
    bcs = [fem.dirichletbc(PETSc.ScalarType(PSI_WALL), wall_dofs, V)]
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4p4db_",
                               petsc_options=gen._DRAIN_LU)
    th_i = float(soil.theta(PSI_I40))
    removed = fem.form((th_i - theta) * r * dxs)
    band_deficit = fem.form((th_i - theta) * band_ind * r * dxq)
    xc = V.tabulate_dof_coordinates()[:, 0]
    in_band = (xc >= BAND[0]) & (xc <= BAND[1])
    Bw = (BAND[1] ** 2 - BAND[0] ** 2) / 2.0

    marks = np.unique(np.concatenate([t, [T1, T2]]))
    I_wall, s_rate, band_max, V_pulse = [], 0.0, -np.inf, 0.0
    dt, t_prev = 1e-8, 0.0
    for t_s in marks:
        if abs(t_prev - T1) < 1e-15:                   # arriving at the fire time: size the pulse
            D = COMM.allreduce(fem.assemble_scalar(band_deficit), op=MPI.SUM)
            V_pulse = PULSE_FILL * D
            s_rate = V_pulse / ((T2 - T1) * Bw)
            print(f"  band deficit at t1={T1}: {D:.4e} -> pulse V={V_pulse:.4e} "
                  f"(s={s_rate:.4f}/day)", flush=True)
        active = (t_prev >= T1 - 1e-15) and (t_s <= T2 + 1e-15)
        s_c.value = s_rate if active else 0.0
        dt = dz._solve_to(problem, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        if active:
            band_max = max(band_max, float(psi.x.array[in_band].max()))
        if t_s in t:
            loss = COMM.allreduce(fem.assemble_scalar(removed), op=MPI.SUM)
            cum_src = s_rate * Bw * min(max(t_s - T1, 0.0), T2 - T1)
            I_wall.append((loss + cum_src) / R_W)      # wall removal = domain loss + injected
            print(f"  t={t_s:.3e}  I={I_wall[-1]:.4e}", flush=True)
    I_wall = np.array(I_wall)
    assert band_max <= -0.01, f"pulse over-saturated the band (max psi {band_max:.3f})"
    I_unp = ref["LOAM_I"]
    print(f"refD40-B: end I={I_wall[-1]:.4f} m vs unpulsed {I_unp[-1]:.4f} m "
          f"(gap {(I_wall[-1]-I_unp[-1])/I_unp[-1]:+.1%}); band psi_max {band_max:.3f}")
    np.savez(pathlib.Path(__file__).parent / "m4_phase4_refD40B_drain.npz",
             LOAM_t=t, LOAM_I=I_wall, LOAM_t_pulse=np.array([T1, T2]),
             LOAM_band=np.array(BAND), LOAM_V_pulse_per_radian=np.array(V_pulse),
             LOAM_V_pulse_per_wall_area=np.array(V_pulse / R_W),
             LOAM_band_psi_max=np.array(band_max), r_w=np.array(R_W))
    print("Saved -> scratch/m4_phase4_refD40B_drain.npz")


if __name__ == "__main__":
    main()
