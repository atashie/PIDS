"""Investigate the 3-D flat-box psi_top non-uniformity (std~0.027) seen in the smoke test.

In the supply-limited regime (rain < Ks) all rain should infiltrate: lambda == rain uniformly,
d ~ 0, and a uniform top Neumann flux on a flat box should give x,y-uniform psi_top. The smoke
test found a SYMMETRIC, alternating (checkerboard-looking) psi_top pattern instead. This probe
asks: (a) is lambda actually uniform (== rain)?  (b) is d the checkerboard?  (c) does the psi_top
non-uniformity SHRINK under mesh refinement (-> a convergent discretization artifact, benign) or
persist (-> a real coupling issue)?
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def run(nx, ny, nz):
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [nx, ny, nz])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    prob.add_rain(0.1)
    prob.advance(t_end=0.5, dt=1e-3, dt_max=0.05)
    zc = prob.Vpsi.tabulate_dof_coordinates()[:, prob._zaxis]
    td = np.isclose(zc, zc.max())
    psit = prob.psi.x.array[td]
    dtop = prob.d.x.array[prob._top_dofs(prob.Vd)]
    lamtop = prob.lam.x.array[prob._top_dofs(prob.Vlam)]
    print(f"  {nx}x{ny}x{nz}:", flush=True)
    print(f"    psi_top  mean={psit.mean():+.5f}  std={psit.std():.3e}  rel={psit.std()/abs(psit.mean()):.2%}",
          flush=True)
    print(f"    d_top    mean={dtop.mean():+.3e}  std={dtop.std():.3e}  min={dtop.min():+.3e}", flush=True)
    print(f"    lam_top  mean={lamtop.mean():+.6f}  std={lamtop.std():.3e}   (rain=0.1)", flush=True)
    print(f"    max_clip_seen={prob.max_clip_seen:.3e}  soil_water={prob.soil_water():.6f}", flush=True)


print("=== psi_top uniformity vs mesh refinement (flat box, supply-limited rain 0.1 < Ks 0.25) ===",
      flush=True)
for res in [(6, 6, 5), (12, 12, 8)]:
    run(*res)
print("UNIFORMITY PROBE DONE", flush=True)
