"""Debug: 2-D lateral overland routing convergence + downslope accumulation."""
import sys
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)

rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
psi0 = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
L = 20.0
msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, 2.0]], [40, 8])
prob = CoupledProblem(msh, SOIL, n_man=0.05)
prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
prob.set_topography(lambda x: 0.05 * (L - x[0]))
rain = prob.add_rain(rate)
one = fem.Constant(msh, 1.0)
import ufl
top_len = msh.comm.allreduce(fem.assemble_scalar(fem.form(one * prob._ds_top)), op=MPI.SUM)
w0 = prob.total_water()
print(f"rate={rate} psi0={psi0} ell_c={prob.ell_c:.4f} top_len={top_len:.2f}")
t, dt, cum = 0.0, 5e-5, 0.0
fails = 0
for k in range(8):
    target = (k + 1) * 0.02
    while t < target - 1e-12:
        h = min(dt, target - t)
        ok, it = prob.step(h)
        if ok:
            cum += rate * top_len * h
            t += h
            dt = min(dt * 1.4 if it <= 4 else dt * 0.7, 5e-3)
        else:
            dt *= 0.5; fails += 1
            if dt < 1e-10:
                print(f"  DT COLLAPSE at t={t:.5g}"); sys.exit()
    dofs = prob._top_dofs(prob.Vd)
    xc = prob.Vd.tabulate_dof_coordinates()[dofs, 0]; dv = prob.d.x.array[dofs]
    o = np.argsort(xc); dv = dv[o]
    n = dv.size; up = dv[:n//2].mean(); down = dv[n//2:].mean()
    err = abs((prob.total_water() - w0) - cum) / (cum + 1e-30)
    print(f"t={t:.3f} d_max={dv.max():.4e} d_min={dv.min():+.2e} uphill={up:.4e} downhill={down:.4e} "
          f"down/up={down/(up+1e-30):.2f} massbal={err:.1e} fails={fails}")
print("DONE: lateral routing", "OK (downhill>uphill)" if down > up else "NOT shown")
