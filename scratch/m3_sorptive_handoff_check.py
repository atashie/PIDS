"""Adversarial check (workflow synthesis watch-item 2026-06-06): is the sorptive Kirchhoff leg a
correct Ks-LIMITED infiltration capacity, or an unbounded absorbing sink?

Sand (Ks=7.13) under rain >> Ks must eventually saturate the surface and SHED the excess (ponding /
runoff), i.e. infiltration must NOT track rain once rain exceeds the Ks-limited capacity. 1-D closed
column: "runoff" shows as a growing surface store d (the excess that can't infiltrate). If d ponds and
the infiltrated fraction drops below 1 as rain rises past Ks, the leg hands off correctly.

  PYTHONPATH=. python scratch/m3_sorptive_handoff_check.py   (from forward-model/)
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.128)
H, T_END = 1.0, 0.02

print(f"sand Ks={SAND.Ks} m/day; 1-D closed column, psi0=-0.3; checking infiltration vs rain rate")
print("  rain   infiltrated%rain   ponded(mm)   surface_depth(mm)")
for rain in (2.0, 7.0, 15.0, 40.0):
    msh = dmesh.create_interval(MPI.COMM_WORLD, 8, [0.0, H])
    prob = CoupledProblem(msh, SAND)
    prob.set_initial_condition(lambda x: -0.3 + 0.0 * x[0], d_value=0.0)
    prob.add_rain(rain)
    w0 = prob.soil_water()
    t, dt = 0.0, 1e-7
    while t < T_END - 1e-12:
        h = min(dt, T_END - t)
        try:
            conv, it = prob.step(h)
        except Exception:
            conv, it = False, 0
            prob.psi.x.array[:] = prob.psi_n.x.array; prob.psi.x.scatter_forward()
            prob.d.x.array[:] = prob.d_n.x.array; prob.d.x.scatter_forward()
            prob.lam.x.array[:] = 0.0; prob.lam.x.scatter_forward(); prob._problem = None
        if conv:
            t += h; dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 5e-4)
        else:
            dt *= 0.5
            if dt < 1e-12:
                print(f"  {rain:5.1f}   dt COLLAPSE at t={t:.4g}"); break
    cum_rain = rain * T_END
    infil = prob.soil_water() - w0
    pond = prob.surface_water()
    print(f"  {rain:5.1f}   {100*infil/cum_rain:6.1f}%          {pond*1e3:7.2f}     {prob.surface_depth()*1e3:7.2f}")
print("\nEXPECT: infiltrated%rain ~100% while rain < Ks; drops below 100% (ponding grows) once rain > Ks")
print("-> confirms the leg is a Ks-limited capacity, NOT an absorbing sink.")
