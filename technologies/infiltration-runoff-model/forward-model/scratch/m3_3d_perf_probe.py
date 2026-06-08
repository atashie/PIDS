"""Phase-1 perf diagnostic: root-cause the ~56-min 3-D coupled smoke test.

Localizes the cost across boundaries: FFCX compile (step 1) vs per-step assemble+solve
(steps 2..) vs step count; then directly tests the surface-integral QUADRATURE-DEGREE
hypothesis -- the Kirchhoff infiltration leg's UFL integrand (a 33-term composite-Simpson
sum of van-Genuchten K) gets an auto-ESTIMATED quadrature degree that may explode on 3-D
TRIANGULAR facets (O(deg^2) points) vs 2-D INTERVAL facets (O(deg)). The sweep checks both
time AND that the assembled value is unchanged by capping the degree (i.e. the cap is a
free, physics-preserving speedup).

Run from forward-model dir with PYTHONPATH=. and *_NUM_THREADS=1.
"""
import time
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
COMM = MPI.COMM_WORLD


def stamp(msg, t0):
    print(f"  [{time.perf_counter() - t0:9.3f}s] {msg}", flush=True)
    return time.perf_counter()


print("=== build ===", flush=True)
t = time.perf_counter()
msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [6, 6, 5])
ncell = msh.topology.index_map(msh.topology.dim).size_local
t = stamp(f"create_box 6x6x5 -> {ncell} cells", t)
prob = CoupledProblem(msh, SOIL)
prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
prob.add_rain(0.1)
ndof = prob.Vpsi.dofmap.index_map.size_local
t = stamp(f"CoupledProblem build (UFL forms, NOT yet compiled): {ndof} psi-dofs x3 fields; "
          f"{prob._top_facets.size} top facets", t)

print("=== real stepping (FFCX compile lands in step 1; steps 2.. are assemble+solve) ===", flush=True)
t = time.perf_counter()
conv, it = prob.step(1e-3)
t = stamp(f"step 1  compile+assemble+solve  converged={conv} iters={it}", t)
for k in range(2, 6):
    conv, it = prob.step(1e-3)
    t = stamp(f"step {k}  assemble+solve         converged={conv} iters={it}", t)

print("=== HYPOTHESIS H2: surface-integral quadrature degree (Kirchhoff q_pot facet leg) ===", flush=True)
psi = fem.Function(prob.Vpsi); psi.x.array[:] = -0.5
d = fem.Function(prob.Vd); d.x.array[:] = 0.02
vpsi = ufl.TestFunction(prob.Vpsi)
mt = dmesh.meshtags(msh, prob._fdim, prob._top_facets,
                    np.ones(prob._top_facets.size, dtype=np.int32))
print("  (qdeg=None == auto-estimated; assemble |b| must match across degrees if cap is safe)", flush=True)
for qd in (None, 2, 4, 6, 10, 20):
    md = {} if qd is None else {"quadrature_degree": qd}
    ds_top = ufl.Measure("ds", domain=msh, subdomain_data=mt, metadata=md)(1)
    integrand = (SOIL.kirchhoff_ufl(psi, d) / prob.ell_c) * vpsi * ds_top
    tc = time.perf_counter()
    f = fem.form(integrand)
    tcompile = time.perf_counter() - tc
    ta = time.perf_counter()
    b = assemble_vector(f); b.assemble()
    tasm = time.perf_counter() - ta
    print(f"  qdeg={str(qd):>4}: compile={tcompile:8.3f}s  assemble={tasm:8.4f}s  |b|={b.norm():.8e}",
          flush=True)
print("PROBE DONE", flush=True)
