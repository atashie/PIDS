"""Measure the closed-domain conservation precision of the coupled solve (Codex finding 1).

Closed flat column under uniform sub-Ks rain: smooth, d>=0 throughout -> the positivity limiter is a
no-op, so the ONLY thing that can break |ΔW - cum_rain|=0 is a spurious term in the residual. The old
whole-domain `eps_diag*d*vd*dx` leaks a tiny sink into the free top rows; the vertex-measure dP_pin
allocation touches only pinned rows -> structural. This prints the relative conservation error so we
can set a meaningful tight regression gate.

  python scratch/m3_check_conservation.py
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import ufl
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)

msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 2.0]], [12, 16])
prob = CoupledProblem(msh, SOIL)
prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
w0 = prob.total_water()
rate = 0.1
prob.add_rain(rate)
nsteps = prob.advance(t_end=0.5, dt=1e-3, dt_max=0.05)

one = fem.Constant(msh, 1.0)
top_len = msh.comm.allreduce(fem.assemble_scalar(fem.form(one * prob._ds_top)), op=MPI.SUM)
cum_rain = rate * top_len * 0.5
dW = prob.total_water() - w0
rel = abs(dW - cum_rain) / cum_rain
print(f"steps={nsteps}  top_len={top_len:.4f}")
print(f"dW={dW:.12e}  cum_rain={cum_rain:.12e}")
print(f"|dW - cum_rain| / cum_rain = {rel:.3e}")
print(f"max_clip_seen={prob.max_clip_seen:.2e}  clip_mass_adjust={prob.clip_mass_adjust:.2e}")
print("STRUCTURAL (<1e-12):" , rel < 1e-12)
