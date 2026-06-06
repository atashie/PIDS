"""Diagnose the ponding-depth SAWTOOTH (alternating exact-zeros) in the closed BC-sweep run.

Hypothesis: the coupling's SURFACE STORAGE term ((d-d_n)/dt)*v*ds_top uses the CONSISTENT (non-lumped)
facet mass matrix. A consistent P1 mass matrix has positive off-diagonals, which produces ODD-EVEN
(checkerboard) over/undershoot at an advancing wet/dry front (the classic non-monotone backward-Euler
diffusion artifact). The post-step positivity limiter then clips the negative lobes to EXACTLY 0,
leaving the alternating zero/nonzero sawtooth. (Module 2 avoids this by LUMPING its storage; the same
vertex-quadrature trick fails on the coupling's ds_top facet integral in FFCX 0.10.)

Test: run the closed scenario with the limiter DISABLED and inspect the RAW Newton solution d(x) at the
drying upslope. If alternating +/- values appear there -> the oscillation is real and pre-limiter; the
limiter's clipping of the negative lobes is what makes the zeros.

  PYTHONPATH=. python scratch/m3_diagnose_sawtooth.py   (run from forward-model/)
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
L, H = 5.0, 1.0
msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [12, 5])
prob = CoupledProblem(msh, SOIL, n_man=0.05)
prob.set_initial_condition(lambda x: -0.8 + 0.0 * x[0], d_value=0.0)
prob.set_topography(lambda x: 0.05 * (L - x[0]))
prob.add_rain(0.6)

topdofs = prob._top_dofs(prob.Vd)
xtop = prob.Vd.tabulate_dof_coordinates()[topdofs, 0]
xorder = np.argsort(xtop)
xs = xtop[xorder]


def show(tag):
    d = prob.d.x.array[topdofs][xorder]
    neg = int(np.sum(d < -1e-12))
    print(f"\n{tag}: (#neg nodes={neg})")
    print("  x:  " + "  ".join(f"{xi:5.2f}" for xi in xs))
    print("  d:  " + "  ".join(f"{di*1e3:+5.1f}" for di in d) + "   [mm]")


# --- (A) WITH limiter (normal): advance a little, show the sawtooth (exact zeros) ---
prob.advance(t_end=0.01, dt=5e-5, dt_max=5e-4)
show("WITH limiter (normal run) -- note exact-0 alternating nodes")

# --- (B) DISABLE the limiter, take ONE more raw Newton step, show pre-clip oscillation ---
prob._enforce_positivity = lambda: 0.0   # monkeypatch: no clipping, no rescale
conv, it = prob.step(5e-5)
print(f"\n(raw step converged={conv} iters={it})")
show("RAW Newton (limiter DISABLED) -- alternating +/- = consistent-mass odd-even oscillation")
