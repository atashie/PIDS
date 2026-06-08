"""FEASIBILITY PROBE (pre-Module-4): a field-scale (2 ha), heterogeneous, layered 3-D run.

Goal: see whether the WHOLE coupled process (3-D heterogeneous Richards + sorptive Kirchhoff
infiltration NCP + lateral Manning overland + the codim-2 ridge edge outlet) is tractable at field
scale on a realistic, badly-anisotropic mesh with depth-varying soil. NOT a permanent feature -- a
one-off probe; LayeredSoil is a duck-typed shim (no changes to the validated module code).

Domain: 210 x 95 m ~ 2.0 ha (30 m x-res, 5 m y-res), 2% bed slope toward the x=L outlet, 1 m deep.
GRADED z-mesh: 0.05 m layers + a 0.01 m SAND cell at z in [0.50,0.51]; impermeable base at z=0.
Soil: surface LOAM (Ks 0.25) -> exponential decay of Ks (->0.048) and theta_s (->0.38) toward clay at
depth; a high-K SAND layer (capillary barrier) at 0.5 m. Forcing: psi0=-1.0, rain 0.3 m/day for 0.3 d
then recession; surface Manning outlet. ~10.5k DOFs, ~16.8k anisotropic tets (~600:1) -- the test.

Run from forward-model/ with PYTHONPATH=. and *_NUM_THREADS=1.
"""
from __future__ import annotations

import time
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.coupling import CoupledProblem


# --------------------------------------------------------------------------- heterogeneous soil
class LayeredSoil:
    """Duck-typed van-Genuchten with z-DEPENDENT params (same UFL interface CoupledProblem uses).

    Surface loam exponentially decaying in Ks & theta_s toward clay at depth, plus a constant SAND
    layer in z in [z_sand]. The decay branch keeps loam alpha/n (constant exponents) and only scales
    Ks(z)/theta_s(z) (linear multipliers) -> no variable-exponent codegen; the sand branch is constant.
    `Ks` is exposed as the SURFACE scalar (loam) for the NCP tau_c = ell_c/Ks (the land-surface
    exchange is at the loam top). No subsurface GHB is used here, so soil.Ks-as-scalar never feeds kr.
    """
    L_exp = 0.5
    h_s = -0.02

    def __init__(self, mesh, *, z_surf=1.0, z_sand=(0.50, 0.51),
                 loam=dict(thr=0.078, ths=0.43, alpha=3.6, n=1.56, Ks=0.25),
                 clay=dict(ths=0.38, Ks=0.048),
                 sand=dict(thr=0.045, ths=0.43, alpha=14.5, n=2.68, Ks=7.13)):
        self._z = ufl.SpatialCoordinate(mesh)[mesh.geometry.dim - 1]
        self.z_surf = float(z_surf)
        self.loam, self.sand = loam, sand
        self._betaK = np.log(loam["Ks"] / clay["Ks"]) / z_surf      # Ks loam(surface)->clay(base)
        self._betaTh = np.log(loam["ths"] / clay["ths"]) / z_surf   # theta_s decay
        self._sand_cond = ufl.And(ufl.ge(self._z, z_sand[0] - 1e-9), ufl.le(self._z, z_sand[1] + 1e-9))
        # scalars CoupledProblem reads (surface representative; tau_c uses Ks)
        self.Ks = loam["Ks"]
        self.theta_r = loam["thr"]
        self.theta_s = loam["ths"]

    def _branch(self, psi, thr, ths, alpha, n, Ks):
        m = 1.0 - 1.0 / n
        u = alpha * ufl.max_value(-psi, 0.0)
        Sc = (1.0 + (alpha * (-self.h_s)) ** n) ** (-m)
        se = ufl.min_value((1.0 + u ** n) ** (-m) / Sc, 1.0)
        num = 1.0 - (1.0 - (se * Sc) ** (1.0 / m)) ** m
        den = 1.0 - (1.0 - Sc ** (1.0 / m)) ** m
        K = Ks * se ** self.L_exp * (num / den) ** 2
        theta = thr + se * (ths - thr)
        return theta, K

    def _decay(self, psi):
        depth = self.z_surf - self._z
        Ks = self.loam["Ks"] * ufl.exp(-self._betaK * depth)
        ths = self.loam["ths"] * ufl.exp(-self._betaTh * depth)
        return self._branch(psi, self.loam["thr"], ths, self.loam["alpha"], self.loam["n"], Ks)

    def _sand(self, psi):
        s = self.sand
        return self._branch(psi, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"])

    def theta_ufl(self, psi):
        return ufl.conditional(self._sand_cond, self._sand(psi)[0], self._decay(psi)[0])

    def K_ufl(self, psi):
        return ufl.conditional(self._sand_cond, self._sand(psi)[1], self._decay(psi)[1])

    def se_ufl(self, psi):  # provided for interface completeness (unused in the solve path)
        th = self.theta_ufl(psi)
        return (th - self.theta_r) / (self.theta_s - self.theta_r)

    # surface Kirchhoff leg lives in the loam top (z=z_surf -> decay == constant loam); use a
    # CONSTANT-loam K so the 33-term Simpson sum stays a small expression (no conditional/exp blow-up).
    _KN, _KP = 32, 4.0

    def _K_loam_const(self, psi):
        lo = self.loam
        return self._branch(psi, lo["thr"], lo["ths"], lo["alpha"], lo["n"], lo["Ks"])[1]

    def kirchhoff_ufl(self, a, b):
        n, p = self._KN, self._KP
        span = b - a
        total = 0.0
        for j in range(n + 1):
            s = j / n
            psi_j = b - span * s ** p
            w = 1.0 if j in (0, n) else (4.0 if j % 2 == 1 else 2.0)
            total = total + w * self._K_loam_const(psi_j) * (p * s ** (p - 1.0))
        return span * total / (3.0 * n)

    def theta(self, psi):  # numpy fallback (diagnostics only; not used for conservation)
        lo = self.loam
        u = lo["alpha"] * np.maximum(-np.asarray(psi), 0.0)
        m = 1.0 - 1.0 / lo["n"]
        Sc = (1.0 + (lo["alpha"] * (-self.h_s)) ** lo["n"]) ** (-m)
        se = np.minimum((1.0 + u ** lo["n"]) ** (-m) / Sc, 1.0)
        return lo["thr"] + se * (lo["ths"] - lo["thr"])


# --------------------------------------------------------------------------- graded box mesh
def graded_box(comm, Lx, Ly, nx, ny, z_levels):
    """A structured box whose z node-levels are remapped to the (non-uniform) z_levels (graded)."""
    z_levels = np.asarray(z_levels, dtype=float)
    nz = z_levels.size - 1
    msh = dmesh.create_box(comm, [[0.0, 0.0, 0.0], [Lx, Ly, 1.0]], [nx, ny, nz])
    x = msh.geometry.x
    idx = np.rint(x[:, 2] * nz).astype(int)   # uniform level 0..nz
    x[:, 2] = z_levels[idx]                    # remap to graded levels
    return msh


# --------------------------------------------------------------------------- the feasibility run
def main():
    COMM = MPI.COMM_WORLD
    Lx, Ly, NX, NY, S0 = 210.0, 95.0, 7, 19, 0.02
    z_levels = np.concatenate([np.arange(0.0, 0.5 + 1e-9, 0.05),       # 0..0.5 @ 0.05 (11 nodes)
                               [0.51],                                  # sand top (0.01 m cell)
                               np.linspace(0.51, 1.0, 11)[1:]])         # 0.51..1.0 (10 nodes)
    rate, storm_dur, t_end, psi0 = 0.30, 0.30, 0.50, -1.0
    WALL_BUDGET_S = 2400.0   # 40 min guard

    t = time.perf_counter()
    msh = graded_box(COMM, Lx, Ly, NX, NY, z_levels)
    ncell = msh.topology.index_map(msh.topology.dim).size_local
    soil = LayeredSoil(msh)
    prob = CoupledProblem(msh, soil, n_man=0.05)   # quadrature cap default 8
    ndof = prob.Vpsi.dofmap.index_map.size_local
    print(f"[build {time.perf_counter()-t:6.2f}s] mesh {NX}x{NY}x{z_levels.size-1} -> {ncell} tets, "
          f"{ndof} psi-dofs x3 = {3*ndof} DOFs; ell_c={prob.ell_c:.4f} m", flush=True)
    # mesh anisotropy (min edge ~ z-layer 0.01 m; max edge ~ x 30 m)
    print(f"  z-levels (m): {np.round(z_levels,3).tolist()}", flush=True)

    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: S0 * (Lx - x[0]))
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], Lx), slope=S0)
    top_area = COMM.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * prob._ds_top)), op=MPI.SUM)
    w0 = prob.total_water()
    print(f"  top_area={top_area:.0f} m^2  initial total_water={w0:.2f} m^3", flush=True)

    print("=== marching (storm 0.3 d @ 0.3 m/day, then recession to 0.5 d) ===", flush=True)
    dt, t_sim, cum_rain, nsteps = 1e-4, 0.0, 0.0, 0
    tstart = time.perf_counter()
    first_step_s = None
    while t_sim < t_end - 1e-12:
        h = min(dt, t_end - t_sim)
        r = rate if t_sim + h <= storm_dur + 1e-12 else 0.0
        rain.value = r
        ts = time.perf_counter()
        conv, it = prob.step(h)
        step_s = time.perf_counter() - ts
        if first_step_s is None:
            first_step_s = step_s
            print(f"  step 1 (compile+solve): {step_s:.1f}s converged={conv} iters={it}", flush=True)
        if conv:
            cum_rain += r * top_area * h
            t_sim += h
            nsteps += 1
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 2e-3)
            if nsteps % 10 == 0 or step_s > 8.0:
                print(f"  t={t_sim:.4f}/{t_end} dt={dt:.2e} iters={it} step={step_s:5.2f}s "
                      f"cum_rain={cum_rain:.3f} outflow={prob.cum_outflow:.3f} d_max={prob.surface_depth():.4f}",
                      flush=True)
        else:
            dt *= 0.5
            if dt < 1e-9:
                print(f"  !! DT COLLAPSE at t={t_sim:.5f} (could not converge)", flush=True)
                break
        if time.perf_counter() - tstart > WALL_BUDGET_S:
            print(f"  !! WALL BUDGET ({WALL_BUDGET_S:.0f}s) hit at t={t_sim:.4f} ({nsteps} steps)", flush=True)
            break

    wall = time.perf_counter() - tstart
    done = t_sim >= t_end - 1e-9
    bal = (prob.total_water() - w0) - (cum_rain - prob.cum_outflow)
    print("=== FEASIBILITY RESULT ===", flush=True)
    print(f"  status            : {'COMPLETED' if done else 'PARTIAL (' + f'{t_sim:.3f}/{t_end} d)'}", flush=True)
    print(f"  wall clock        : {wall:.1f}s over {nsteps} steps "
          f"({wall/max(nsteps,1):.2f}s/step; compile+first {first_step_s:.1f}s)", flush=True)
    print(f"  conservation      : |bal|/cum_rain = {abs(bal)/(cum_rain+1e-30):.2e}", flush=True)
    print(f"  cum_rain          : {cum_rain:.4f} m^3", flush=True)
    print(f"  cum_outflow       : {prob.cum_outflow:.4f} m^3 ({100*prob.cum_outflow/(cum_rain+1e-30):.1f}% of rain)", flush=True)
    print(f"  soil water (final): {prob.soil_water():.4f} m^3", flush=True)
    print(f"  peak surface depth: {prob.surface_depth():.4f} m", flush=True)
    print(f"  max_clip_seen     : {prob.max_clip_seen:.3e}", flush=True)
    print(f"  finite state      : psi {np.all(np.isfinite(prob.psi.x.array))}, "
          f"d {np.all(np.isfinite(prob.d.x.array))}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
