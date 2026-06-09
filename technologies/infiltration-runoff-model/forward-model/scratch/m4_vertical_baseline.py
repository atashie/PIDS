"""NO-PIDS vertical-profile BASELINE: a 1-D layered column under a storm (the 'before PIDS' control).

The system-scale counterpart of the Phase-1 near-field references: a 1-D CoupledProblem column (Richards
+ ponding + the sorptive Kirchhoff infiltration NCP) showing the WATERLOGGING PROBLEM a PIDS vertical
feature will address -- a loam topsoil over a low-K clay subsoil, hit by a heavy storm, so water PERCHES
above the clay and PONDS at the surface. Records the full theta/psi(z,t) profile + the ponding and
infiltration time series, the same quantities the feasibility HILLSLOPE viz shows, so the eventual
WITH-PIDS run (Phase 2: an embedded vertical tunnel/channel draining+conveying) overlays directly.

There is NO PIDS feature here (it isn't built yet) -- this is the control. Saves an npz the viz reads.
Run from forward-model/:  PYTHONPATH=. OMP_NUM_THREADS=1 ... python scratch/m4_vertical_baseline.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

COMM = MPI.COMM_WORLD
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
CLAY = VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.048)
H, Z_SPLIT = 1.2, 0.6          # column height [m]; loam above z_split, clay below
RAIN, STORM, T_END = 0.30, 0.5, 1.5   # m/day storm for 0.5 d, then recession to 1.5 d


class TwoLayer:
    """Duck-typed van-Genuchten: loam topsoil (z>=z_split) over clay subsoil (z<z_split). Same UFL
    interface CoupledProblem consumes; the surface Kirchhoff leg lives in the loam top."""

    def __init__(self, mesh, top, bot, z_split):
        self.z = ufl.SpatialCoordinate(mesh)[mesh.geometry.dim - 1]
        self.top, self.bot, self.zsplit = top, bot, float(z_split)
        self.Ks, self.theta_r, self.theta_s = top.Ks, top.theta_r, top.theta_s
        self._c = ufl.ge(self.z, z_split)

    def theta_ufl(self, psi):
        return ufl.conditional(self._c, self.top.theta_ufl(psi), self.bot.theta_ufl(psi))

    def K_ufl(self, psi):
        return ufl.conditional(self._c, self.top.K_ufl(psi), self.bot.K_ufl(psi))

    def kirchhoff_ufl(self, a, b):
        return self.top.kirchhoff_ufl(a, b)   # surface infiltration leg is in the loam top

    def theta_of(self, psi, z):               # numpy, for the viz
        return np.where(np.asarray(z) >= self.zsplit, self.top.theta(psi), self.bot.theta(psi))


def main():
    N = 60
    msh = dmesh.create_interval(COMM, N, [0.0, H])
    soil = TwoLayer(msh, LOAM, CLAY, Z_SPLIT)
    prob = CoupledProblem(msh, soil)
    prob.set_initial_condition(lambda x: -0.8 + 0.0 * x[0], d_value=0.0)
    rain = prob.add_rain(0.0)
    zc = prob.Vpsi.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(zc)
    zc_s = zc[order]
    w0 = prob.soil_water()

    times = np.linspace(0.0, T_END, 60)
    psi_zt, th_zt = [], []
    pond, infil, rain_t, soilw = [], [], [], []
    t, dt, k = 0.0, 1e-4, 0

    def snap():
        ps = prob.psi.x.array[order]
        psi_zt.append(ps.copy())
        th_zt.append(soil.theta_of(ps, zc_s))
        pond.append(prob.surface_depth())
        infil.append(prob.soil_water() - w0)
        rain_t.append(float(rain.value))
        soilw.append(prob.soil_water())

    snap()
    k = 1
    while t < T_END - 1e-12:
        h = min(dt, T_END - t)
        if t < STORM - 1e-12 and t + h > STORM:
            h = STORM - t
        if k < len(times) and t + h > times[k]:
            h = times[k] - t
        rain.value = RAIN if t < STORM - 1e-12 else 0.0
        conv, it = prob.step(h)
        if conv:
            t += h
            if k < len(times) and t >= times[k] - 1e-12:
                snap(); k += 1
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 5e-3)
        else:
            dt *= 0.5
            assert dt > 1e-10, f"dt collapse at t={t:.4f}"
    while k < len(times):
        snap(); k += 1

    np.savez("scratch/m4_vertical_baseline.npz",
             z=zc_s, t=times, psi_zt=np.array(psi_zt), theta_zt=np.array(th_zt),
             ponding=np.array(pond), infiltration=np.array(infil), rain=np.array(rain_t),
             soil_water=np.array(soilw), z_split=Z_SPLIT, storm=STORM, rain_rate=RAIN,
             theta_s=LOAM.theta_s, theta_r=CLAY.theta_r)
    print(f"WROTE scratch/m4_vertical_baseline.npz  "
          f"(peak ponding {max(pond)*100:.1f} cm, final infil {infil[-1]*100:.1f} cm-equiv)", flush=True)


if __name__ == "__main__":
    main()
