"""Phase-1 perf diagnostic v2: localize the ~16.5s/step cost WITHIN a Newton step and test
the VOLUME (Richards Darcy) auto-quadrature-degree hypothesis on tetrahedra.

v1 showed: per-step recurring (not compile, not step-count); the Kirchhoff FACET leg is ~20ms
(not the bottleneck) but its auto-estimated degree is ~26. Suspect: the volume Darcy term
K(psi)*(grad psi + e_g).grad v * dx uses plain dx (no quadrature cap); van-Genuchten K's
fractional powers -> high auto-degree -> O(deg^3) points/tet * 1080 tets * Jacobian kernel.

This probe times each Jacobian BLOCK assembly on the real problem, then sweeps the volume
Darcy Jacobian quadrature degree (auto vs capped) checking BOTH time and matrix-norm match.
"""
import time
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.richards import richards_bulk_residual

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
COMM = MPI.COMM_WORLD

msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [6, 6, 5])
prob = CoupledProblem(msh, SOIL)
prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
prob.add_rain(0.1)
prob.step(1e-3)  # warm the FFCX cache (compile once)


def time_mat(label, form_ufl, reps=3):
    cf = fem.form(form_ufl)
    M = assemble_matrix(cf); M.assemble()                 # warm
    t = time.perf_counter()
    for _ in range(reps):
        M = assemble_matrix(cf); M.assemble()
    print(f"  {label}: {(time.perf_counter() - t) / reps * 1000:9.1f} ms   norm={M.norm():.6e}", flush=True)


def time_vec(label, form_ufl, reps=3):
    cf = fem.form(form_ufl)
    b = assemble_vector(cf); b.assemble()
    t = time.perf_counter()
    for _ in range(reps):
        b = assemble_vector(cf); b.assemble()
    print(f"  {label}: {(time.perf_counter() - t) / reps * 1000:9.1f} ms", flush=True)


print("=== per-block assembly on the REAL coupled problem (current/auto quadrature) ===", flush=True)
time_mat("J psi-psi (volume Richards)", ufl.derivative(prob.F_psi, prob.psi))
time_mat("J d-d   (surface)          ", ufl.derivative(prob.F_d, prob.d))
time_mat("J lam-lam (NCP)            ", ufl.derivative(prob.F_lam, prob.lam))
time_mat("J psi-d (cross)            ", ufl.derivative(prob.F_psi, prob.d))
time_mat("J lam-psi (cross)          ", ufl.derivative(prob.F_lam, prob.psi))
time_mat("J lam-d (cross)            ", ufl.derivative(prob.F_lam, prob.d))
time_vec("R psi residual             ", prob.F_psi)
time_vec("R d residual               ", prob.F_d)
time_vec("R lam residual             ", prob.F_lam)

print("=== HYPOTHESIS: volume Darcy Jacobian quadrature degree (standalone) ===", flush=True)
V = prob.Vpsi
psi = fem.Function(V); psi.x.array[:] = -0.5
psi_n = fem.Function(V); psi_n.x.array[:] = -0.5
v = ufl.TestFunction(V)
dx_lump = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
print("  (norm must match across degrees if a cap is safe; auto=None is the current behavior)", flush=True)
for qd in (None, 2, 4, 6, 8, 10):
    md = {} if qd is None else {"quadrature_degree": qd}
    dx = ufl.dx(metadata=md)
    F = richards_bulk_residual(psi, psi_n, v, SOIL, prob.dt, prob.e_g, dx=dx, dx_storage=dx_lump)
    J = ufl.derivative(F, psi)
    tc = time.perf_counter(); jf = fem.form(J); tcompile = time.perf_counter() - tc
    M = assemble_matrix(jf); M.assemble()                 # warm
    t = time.perf_counter()
    for _ in range(3):
        M = assemble_matrix(jf); M.assemble()
    tasm = (time.perf_counter() - t) / 3 * 1000
    print(f"  volume Darcy J  qdeg={str(qd):>4}: compile={tcompile:7.3f}s  assemble={tasm:9.1f} ms  "
          f"norm={M.norm():.8e}", flush=True)
print("PROBE2 DONE", flush=True)
