"""Debug: 2-D flat coupled conservation (rain appears to be lost)."""
import numpy as np, ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 2.0]], [12, 16])
prob = CoupledProblem(msh, SOIL)
prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
one = fem.Constant(msh, 1.0)
top_len = msh.comm.allreduce(fem.assemble_scalar(fem.form(one * prob._ds_top)), op=MPI.SUM)
print(f"top_len={top_len:.4f}  ell_c={prob.ell_c:.4f}  tau_c={prob._tau_c:.4f}")
rate = 0.1
rain = prob.add_rain(rate)
w0 = prob.total_water()
print(f"{'t':>7} {'soil':>10} {'surf':>10} {'total':>10} {'cum_rain':>10} {'err':>10} {'d_max':>9} {'lam_max':>9} {'it':>3}")
t, dt, cum = 0.0, 1e-3, 0.0
for k in range(10):
    target = (k + 1) * 0.05
    while t < target - 1e-12:
        h = min(dt, target - t)
        ok, it = prob.step(h)
        if ok:
            cum += rate * top_len * h
            t += h
            dt = min(dt * 1.4 if it <= 4 else dt * 0.7, 0.05)
        else:
            dt *= 0.5
    topdofs = prob._top_dofs(prob.Vlam)
    lam_max = float(prob.lam.x.array[topdofs].max()) if topdofs.size else 0.0
    sw, fw = prob.soil_water(), prob.surface_water()
    err = abs((sw + fw - w0) - cum) / (cum + 1e-30)
    print(f"{t:7.3f} {sw:10.5f} {fw:10.5f} {sw+fw:10.5f} {cum:10.5f} {err:10.2e} "
          f"{prob.surface_depth():9.4e} {lam_max:9.4e} {it:3d}")
