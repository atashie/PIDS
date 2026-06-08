"""FEASIBILITY PROBE (pre-Module-4): a field-scale (2 ha), heterogeneous, layered 3-D run.

Goal: see whether the WHOLE coupled process (3-D heterogeneous Richards + sorptive Kirchhoff
infiltration NCP + lateral Manning overland + the codim-2 ridge edge outlet) is tractable at field
scale on a realistic, badly-anisotropic mesh with depth-varying soil. NOT a permanent feature -- a
one-off probe; LayeredSoil is a duck-typed shim (no changes to the validated module code).

Domain: 210 x 95 m ~ 2.0 ha (30 m x-res, 5 m y-res), 2% bed slope toward the x=L outlet, 1 m deep.
GRADED z-mesh: 0.05 m layers + a 0.01 m SAND cell at z in [0.50,0.51]; impermeable base at z=0.
Soil: surface LOAM retention (alpha,n) with depth-DECAYING Ks (->0.048) and theta_s (->0.38) toward
clay-LIKE values (NOTE: only Ks/porosity decay; the retention SHAPE stays loam -- a clay approximation,
not a true clay van-Genuchten); a high-K SAND layer (capillary barrier) at 0.5 m. Forcing: psi0=-1.0,
rain 0.3 m/day for 0.3 d then recession; surface Manning outlet. ~10.5k DOFs, ~16.8k tets (~600:1).

Emits a Tier-3 NetCDF (surface map + psi/theta cross-section + theta z-layers + the soil Ks profile +
time series) for the separate viz subagent. Run from forward-model/ with PYTHONPATH=. and *_NUM_THREADS=1.
"""
from __future__ import annotations

import time
import numpy as np
import ufl
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.coupling import CoupledProblem

DATE = "2026-06-08"
DATA_DIR = "../validation/sanity/data"


# --------------------------------------------------------------------------- heterogeneous soil
class LayeredSoil:
    """Duck-typed van-Genuchten with z-DEPENDENT params (same UFL interface CoupledProblem uses).

    Surface loam RETENTION (alpha,n,theta_r constant) with depth-decaying Ks(z) & theta_s(z) toward
    clay-like values, plus a constant SAND layer in z in [z_sand]. Only Ks/porosity vary -- the
    retention SHAPE (Se, Sc) stays loam in the decay branch (a clay APPROXIMATION, not a clay curve).
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
        self.z_sand = z_sand
        self.loam, self.sand = loam, sand
        self._betaK = np.log(loam["Ks"] / clay["Ks"]) / z_surf      # Ks loam(surface)->clay(base)
        self._betaTh = np.log(loam["ths"] / clay["ths"]) / z_surf   # theta_s decay
        self._sand_cond = ufl.And(ufl.ge(self._z, z_sand[0] - 1e-9), ufl.le(self._z, z_sand[1] + 1e-9))
        self.Ks = loam["Ks"]            # surface scalar (tau_c); never feeds kr here (no GHB)
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

    # --- numpy closures for the VIZ (z-aware; NOT used in the solve path, which uses theta_ufl) ---
    def _theta_np(self, psi, thr, ths, alpha, n):
        m = 1.0 - 1.0 / n
        u = alpha * np.maximum(-np.asarray(psi, float), 0.0)
        Sc = (1.0 + (alpha * (-self.h_s)) ** n) ** (-m)
        se = np.minimum((1.0 + u ** n) ** (-m) / Sc, 1.0)
        return thr + se * (ths - thr)

    def theta_of(self, psi, z):
        z = np.asarray(z, float)
        in_sand = (z >= self.z_sand[0] - 1e-9) & (z <= self.z_sand[1] + 1e-9)
        depth = self.z_surf - z
        th_d = self._theta_np(psi, self.loam["thr"], self.loam["ths"] * np.exp(-self._betaTh * depth),
                              self.loam["alpha"], self.loam["n"])
        th_s = self._theta_np(psi, self.sand["thr"], self.sand["ths"], self.sand["alpha"], self.sand["n"])
        return np.where(in_sand, th_s, th_d)

    def Ks_of_z(self, z):
        z = np.asarray(z, float)
        in_sand = (z >= self.z_sand[0] - 1e-9) & (z <= self.z_sand[1] + 1e-9)
        Ks_d = self.loam["Ks"] * np.exp(-self._betaK * (self.z_surf - z))
        return np.where(in_sand, self.sand["Ks"], Ks_d)


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


def _grid_index(coords_xy):
    xr_ = np.round(coords_xy[:, 0], 6); yr_ = np.round(coords_xy[:, 1], 6)
    xu = np.unique(xr_); yu = np.unique(yr_)
    return xu, yu, np.searchsorted(xu, xr_), np.searchsorted(yu, yr_)


# --------------------------------------------------------------------------- the feasibility run
def main():
    COMM = MPI.COMM_WORLD
    Lx, Ly, NX, NY, S0 = 210.0, 95.0, 7, 19, 0.02
    z_levels = np.concatenate([np.arange(0.0, 0.5 + 1e-9, 0.05),       # 0..0.5 @ 0.05 (11 nodes)
                               [0.51],                                  # sand top (0.01 m cell)
                               np.linspace(0.51, 1.0, 11)[1:]])         # 0.51..1.0 (10 nodes)
    rate, storm_dur, t_end, psi0 = 0.30, 0.30, 0.50, -1.0
    WALL_BUDGET_S = 2400.0   # 40 min guard
    N_OUT = 45               # NetCDF snapshots

    t = time.perf_counter()
    msh = graded_box(COMM, Lx, Ly, NX, NY, z_levels)
    ncell = msh.topology.index_map(msh.topology.dim).size_local
    soil = LayeredSoil(msh)
    prob = CoupledProblem(msh, soil, n_man=0.05)   # quadrature cap default 8
    ndof = prob.Vpsi.dofmap.index_map.size_local
    print(f"[build {time.perf_counter()-t:6.2f}s] mesh {NX}x{NY}x{z_levels.size-1} -> {ncell} tets, "
          f"{ndof} psi-dofs x3 = {3*ndof} DOFs; ell_c={prob.ell_c:.4f} m", flush=True)

    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: S0 * (Lx - x[0]))
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], Lx), slope=S0)
    top_area = COMM.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * prob._ds_top)), op=MPI.SUM)
    w0 = prob.total_water()
    print(f"  top_area={top_area:.0f} m^2  initial total_water={w0:.2f} m^3", flush=True)

    # --- viz extraction maps (structured grids on the graded mesh) ---
    topd = prob._top_dofs(prob.Vd)
    xs_u, ys_u, six, siy = _grid_index(prob.Vd.tabulate_dof_coordinates()[topd])
    pcoord = prob.Vpsi.tabulate_dof_coordinates()
    y_lvl = np.unique(np.round(pcoord[:, 1], 6))
    y_sec = float(y_lvl[np.argmin(np.abs(y_lvl - 0.5 * Ly))])
    sel = np.isclose(pcoord[:, 1], y_sec)
    xc_u = np.unique(np.round(pcoord[sel, 0], 6)); zc_u = np.unique(np.round(pcoord[sel, 2], 6))
    cix = np.searchsorted(xc_u, np.round(pcoord[sel, 0], 6))
    ciz = np.searchsorted(zc_u, np.round(pcoord[sel, 2], 6))
    z_per_sel = zc_u[ciz]                                           # elevation of each cross-section dof
    Ks_profile = soil.Ks_of_z(zc_u)                                 # 1-D Ks(z) on the section (static)
    z_all = np.unique(np.round(pcoord[:, 2], 6))
    z_layers = np.array([z_all[-1], z_all[np.argmin(np.abs(z_all-0.559))],    # surface, just-above-sand
                         z_all[np.argmin(np.abs(z_all-0.45))], z_all[np.argmin(np.abs(z_all-0.05))]])  # below-sand, base
    layer_sel = [np.isclose(pcoord[:, 2], zl) for zl in z_layers]
    layer_idx = [(_grid_index(pcoord[s][:, :2])[2], _grid_index(pcoord[s][:, :2])[3]) for s in layer_sel]

    nys, nxs = ys_u.size, xs_u.size
    nzc, nxc = zc_u.size, xc_u.size
    out_times = np.linspace(0.0, t_end, N_OUT)
    d_map = np.zeros((N_OUT, nys, nxs)); psi_xs = np.zeros((N_OUT, nzc, nxc)); th_xs = np.zeros((N_OUT, nzc, nxc))
    th_xy = np.zeros((N_OUT, z_layers.size, nys, nxs))
    rainfall = np.zeros(N_OUT); soil_w = np.zeros(N_OUT); surf_w = np.zeros(N_OUT)
    outflow = np.zeros(N_OUT); cum_out = np.zeros(N_OUT); cum_rain_t = np.zeros(N_OUT)
    mbe = np.zeros(N_OUT); iters_rec = np.zeros(N_OUT)

    def snap(k, cum_rain):
        d_map[k][siy, six] = prob.d.x.array[topd]
        psi_xs[k][ciz, cix] = prob.psi.x.array[sel]
        th_xs[k][ciz, cix] = soil.theta_of(prob.psi.x.array[sel], z_per_sel)
        for li, (s, (ixl, iyl)) in enumerate(zip(layer_sel, layer_idx)):
            th_xy[k, li][iyl, ixl] = soil.theta_of(prob.psi.x.array[s], z_layers[li])
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water()
        outflow[k] = prob.last_outflow; cum_out[k] = prob.cum_outflow
        cum_rain_t[k] = cum_rain
        expected = cum_rain - prob.cum_outflow + prob.clip_mass_adjust
        mbe[k] = abs((prob.total_water() - w0) - expected) / (cum_rain + 1e-30)

    print("=== marching (storm 0.3 d @ 0.3 m/day, then recession to 0.5 d) ===", flush=True)
    snap(0, 0.0)
    dt, t_sim, cum_rain, nsteps, k_out = 1e-4, 0.0, 0.0, 0, 1
    tstart = time.perf_counter(); first_step_s = None
    while t_sim < t_end - 1e-12:
        h = min(dt, t_end - t_sim)
        # do NOT straddle the rain cutoff: clip the step to land exactly on storm_dur (Codex review)
        if t_sim < storm_dur - 1e-12 and t_sim + h > storm_dur:
            h = storm_dur - t_sim
        if k_out < N_OUT and t_sim + h > out_times[k_out]:        # also stop on the next snapshot time
            h = out_times[k_out] - t_sim
        r = rate if t_sim < storm_dur - 1e-12 else 0.0           # rain iff the step is within the storm
        rain.value = r
        ts = time.perf_counter()
        conv, it = prob.step(h)
        step_s = time.perf_counter() - ts
        if first_step_s is None:
            first_step_s = step_s
            print(f"  step 1 (compile+solve): {step_s:.1f}s converged={conv} iters={it}", flush=True)
        if conv:
            cum_rain += r * top_area * h
            t_sim += h; nsteps += 1
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 2e-3)
            while k_out < N_OUT and t_sim >= out_times[k_out] - 1e-12:
                snap(k_out, cum_rain); iters_rec[k_out] = it; k_out += 1
            if nsteps % 20 == 0 or step_s > 8.0:
                print(f"  t={t_sim:.4f}/{t_end} dt={dt:.2e} it={it} step={step_s:5.2f}s "
                      f"cum_rain={cum_rain:.1f} outflow={prob.cum_outflow:.1f} d_max={prob.surface_depth():.4f} "
                      f"clip={prob.clip_mass_adjust:.2e}", flush=True)
        else:
            dt *= 0.5
            if dt < 1e-9:
                print(f"  !! DT COLLAPSE at t={t_sim:.5f}", flush=True); break
        if time.perf_counter() - tstart > WALL_BUDGET_S:
            print(f"  !! WALL BUDGET hit at t={t_sim:.4f} ({nsteps} steps)", flush=True); break
    while k_out < N_OUT:   # pad any unfilled snapshots with the final state
        snap(k_out, cum_rain); k_out += 1
    # rainfall series = the forcing ACTIVE at each snapshot time (rate while t < storm_dur, else 0).
    # (Codex review: the per-step `r` recorded the previous step's value -> mis-sampled the hyetograph.)
    rainfall = np.where(out_times < storm_dur - 1e-12, rate, 0.0)

    wall = time.perf_counter() - tstart
    done = t_sim >= t_end - 1e-9
    bal = (prob.total_water() - w0) - (cum_rain - prob.cum_outflow + prob.clip_mass_adjust)
    print("=== FEASIBILITY RESULT ===", flush=True)
    print(f"  status            : {'COMPLETED' if done else f'PARTIAL ({t_sim:.3f}/{t_end} d)'}", flush=True)
    print(f"  wall clock        : {wall:.1f}s over {nsteps} steps ({wall/max(nsteps,1):.2f}s/step; "
          f"compile+first {first_step_s:.1f}s)", flush=True)
    print(f"  discrete lumped mass balance |bal|/cum_rain = {abs(bal)/(cum_rain+1e-30):.2e} "
          f"(clip_mass_adjust={prob.clip_mass_adjust:.2e})", flush=True)
    print(f"  cum_rain={cum_rain:.2f} m^3  cum_outflow={prob.cum_outflow:.2f} m^3 "
          f"({100*prob.cum_outflow/(cum_rain+1e-30):.1f}%)  soil_water_final={prob.soil_water():.2f} m^3", flush=True)
    print(f"  peak surface depth={prob.surface_depth():.4f} m  max_clip_seen={prob.max_clip_seen:.2e}", flush=True)
    print(f"  finite: psi {np.all(np.isfinite(prob.psi.x.array))}, d {np.all(np.isfinite(prob.d.x.array))}", flush=True)

    ds = xr.Dataset(
        data_vars=dict(
            surface_depth_map=(("time", "y", "x"), d_map, {"units": "m", "long_name": "surface ponding depth d(x,y)"}),
            head_xsec=(("time", "z", "xc"), psi_xs, {"units": "m", "long_name": f"psi on the y={y_sec:.1f} m section (water table = psi=0)"}),
            theta_xsec=(("time", "z", "xc"), th_xs, {"units": "-", "long_name": "theta on the cross-section (layered soil)"}),
            theta_xy=(("time", "zlayer", "y", "x"), th_xy, {"units": "-", "long_name": "theta(x,y) at surface / above-sand / below-sand / base"}),
            Ks_profile=(("z",), Ks_profile, {"units": "m/day", "long_name": "Ks(z) on the section (sand spike + loam->clay decay)"}),
            rainfall=(("time",), rainfall, {"units": "m/day"}),
            soil_water=(("time",), soil_w, {"units": "m^3", "long_name": "subsurface stored water (int theta)"}),
            surface_water=(("time",), surf_w, {"units": "m^3"}),
            outflow=(("time",), outflow, {"units": "m^3/day", "long_name": "surface edge outflow"}),
            cum_outflow=(("time",), cum_out, {"units": "m^3"}),
            cum_rain=(("time",), cum_rain_t, {"units": "m^3"}),
            mass_balance_error=(("time",), mbe, {"units": "-"}),
            newton_iters=(("time",), iters_rec, {"units": "-"}),
        ),
        coords=dict(time=("time", out_times, {"units": "day"}),
                    x=("x", xs_u, {"units": "m", "long_name": "downslope x (outlet at x=L)"}),
                    y=("y", ys_u, {"units": "m"}),
                    xc=("xc", xc_u, {"units": "m"}),
                    z=("z", zc_u, {"units": "m", "long_name": "elevation (0=impermeable base, 1=surface)"}),
                    zlayer=("zlayer", z_layers, {"units": "m"})),
        attrs=dict(module="feasibility_2ha_layered", date=DATE,
                   scenario=f"2-ha heterogeneous LAYERED hillslope (loam retention + decaying Ks/theta_s + 0.01 m sand barrier @0.5 m), "
                            f"rain {rate} m/day for {storm_dur} d, 2% slope + surface outlet, impermeable base",
                   domain=f"{Lx}x{Ly}x1 m ~ 2 ha", mesh=f"{NX}x{NY}x{z_levels.size-1} ({ncell} tets, {3*ndof} DOFs)",
                   y_section_m=y_sec, status="COMPLETED" if done else "PARTIAL",
                   wall_clock_s=float(wall), steps=int(nsteps), sec_per_step=float(wall/max(nsteps,1)),
                   cum_rain_m3=float(cum_rain), cum_outflow_m3=float(prob.cum_outflow),
                   runoff_fraction=float(prob.cum_outflow/(cum_rain+1e-30)),
                   soil_water_final_m3=float(prob.soil_water()),
                   mass_balance_error_max=float(mbe.max()), clip_mass_adjust=float(prob.clip_mass_adjust),
                   peak_surface_depth_m=float(prob.surface_depth())),
    )
    out_nc = f"{DATA_DIR}/feasibility_2ha__{DATE}.nc"
    ds.to_netcdf(out_nc)
    print(f"WROTE {out_nc}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
