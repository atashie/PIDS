"""Debug: 1-D coupled trajectory to confirm the negative-d mechanism."""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 1.0])
prob = CoupledProblem(msh, SOIL)
prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
prob.add_rain(0.1)
krel = SOIL.K(np.array([-2.0]))[0] / SOIL.Ks
print(f"ell_c={prob.ell_c:.4f}  Ks={SOIL.Ks}  K_rel(-2)={krel:.3e}  k_ex(-2)={krel*SOIL.Ks/prob.ell_c:.3e}")
hdr = ("t", "d_surf", "psi_top", "soil_w", "surf_w", "total")
print("".join(f"{h:>10}" for h in hdr))
zc = prob.Vpsi.tabulate_dof_coordinates()[:, 0]
itop = int(np.argmax(zc))
t = 0.0
for i in range(12):
    prob.advance(t_end=0.04, dt=2e-3, dt_max=0.02)
    t += 0.04
    row = (t, prob.surface_depth(), prob.psi.x.array[itop],
           prob.soil_water(), prob.surface_water(), prob.total_water())
    print("".join(f"{v:>10.4f}" for v in row))
