"""Shore-up #6 (Codex): 2-D coupling MPI smoke -- run a sloped lateral-routing case on N ranks and
check the coupled solve converges and CONSERVES across the partition (the limiter's global rescale
uses allreduce; this checks it stays MPI-consistent, like Module 2's n=2 test).

  OMP_NUM_THREADS=1 mpirun --allow-run-as-root -n 2 python scratch/m3_mpi_smoke_2d.py
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

COMM = MPI.COMM_WORLD
SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
L = 4.0
msh = dmesh.create_rectangle(COMM, [[0.0, 0.0], [L, 1.0]], [16, 5])  # partitioned across ranks
prob = CoupledProblem(msh, SOIL, n_man=0.05)
prob.set_initial_condition(lambda x: -0.8 + 0.0 * x[0], d_value=0.0)
prob.set_topography(lambda x: 0.05 * (L - x[0]))
prob.add_rain(0.6)
one = fem.Constant(msh, 1.0)
tl = COMM.allreduce(fem.assemble_scalar(fem.form(one * prob._ds_top)), op=MPI.SUM)
w0 = prob.total_water()
n = prob.advance(t_end=0.02, dt=5e-5, dt_max=1e-3)
cum = 0.6 * tl * 0.02
dW = prob.total_water() - w0
err = abs(dW - cum) / cum
dmin = COMM.allreduce(float(prob.d.x.array.min()) if prob.d.x.array.size else 0.0, op=MPI.MIN)
maxclip = COMM.allreduce(prob.max_clip_seen, op=MPI.MAX)
ok = err < 1e-6 and dmin >= -1e-9 and np.all(np.isfinite(prob.psi.x.array))
if COMM.rank == 0:
    print(f"ranks={COMM.size} steps={n} massbal_err={err:.2e} d_surf={prob.surface_depth():.4e} "
          f"d_min={dmin:+.1e} max_clip={maxclip:.2e}")
    print("MPI SMOKE 2-D coupling:", "PASS" if ok else "FAIL")
