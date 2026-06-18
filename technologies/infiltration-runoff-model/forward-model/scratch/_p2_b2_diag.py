"""P2-B2 diagnostic: why does the coupled upwind block-SNES not converge? Isolate wiring vs flux."""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def one_step(scheme, tilt, rate, d0=0.0, psi0=-2.0, dt=1e-3, nx=5, ny=5, nz=3):
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0., 0., 0.], [2., 1., 1.]], [nx, ny, nz])
    prob = CoupledProblem(msh, SOIL, overland_scheme=scheme)
    prob.set_topography(lambda x: tilt * (2.0 - x[0]))
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=d0)
    prob.add_rain(rate)
    conv, it = prob.step(dt)
    return prob, conv, it


print("=== GALERKIN flat sub-Ks (reference) ===", flush=True)
p, c, it = one_step("galerkin", tilt=0.0, rate=0.1)
print(f"  conv={c} reason={p.last_reason} fnorm={p.last_fnorm:.3e} iters={it}", flush=True)

print("=== UPWIND flat sub-Ks (isolates WIRING; flux ~0) ===", flush=True)
p, c, it = one_step("upwind", tilt=0.0, rate=0.1)
print(f"  conv={c} reason={p.last_reason} fnorm={p.last_fnorm:.3e} iters={it}", flush=True)
# block-offset sanity: N, block size
N = p.Vd.dofmap.index_map.size_local
print(f"  N={N}  block_b_localsize={p._problem._b.getLocalSize()}  (expect ~3N={3*N})", flush=True)

print("=== UPWIND tilted ponding (the failing case) ===", flush=True)
p, c, it = one_step("upwind", tilt=0.05, rate=0.5, d0=0.05, psi0=-1.0)
print(f"  conv={c} reason={p.last_reason} fnorm={p.last_fnorm:.3e} iters={it}", flush=True)
print(f"  d range: [{p.d.x.array.min():.3e}, {p.d.x.array.max():.3e}]", flush=True)
