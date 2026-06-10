"""ILLUSTRATION (not a new validated module): a 2-D vertical hillslope transect with a buried TILE
DRAIN, run on the VALIDATED CoupledProblem engine, emitting a standardized Tier-3 NetCDF.

The engine (Modules 1+2+3 + the GHB drainage BC) is already validated; this script just *shows the
physics*: a water table at 1 m, a 2% surface slope, a moderate storm that infiltrates and mounds the
table, and a localized base drain at x in [9.5,10.5] that pulls a DRAWDOWN CONE into the table while
discharging. 2-D vertical cross-section: gravity / elevation on the LAST coordinate (index 1).

Run from forward-model/ with PYTHONPATH=. and *_NUM_THREADS=1 (BLAS pinned -- the tiny coupled
systems oversubscribe multithreaded BLAS). Structure copied from scratch/feasibility_2ha_layered.py
(CoupledProblem setup, adaptive-dt march, snapshot() pattern, xarray emit, lumped mass accounting),
reduced from 3-D to a single 2-D transect and specialized to the tile-drain scenario.
"""
from __future__ import annotations

import time
import numpy as np
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

DATE = "2026-06-09"
DATA_DIR = "../validation/sanity/data"
OUT_NC = f"{DATA_DIR}/m4_hillslope_drain__{DATE}.nc"


def _grid_index_xz(coords_xz):
    """Regular-grid index from P1 vertex coords (x, z): unique axes + searchsorted (template pattern)."""
    xr_ = np.round(coords_xz[:, 0], 6)
    zr_ = np.round(coords_xz[:, 1], 6)
    xu = np.unique(xr_)
    zu = np.unique(zr_)
    return xu, zu, np.searchsorted(xu, xr_), np.searchsorted(zu, zr_)


def _water_table(psi_col, z_col):
    """Elevation z where psi crosses 0 in one column (linear interp of the lowest psi>=0 -> psi<0
    crossing going UP from the base). NaN if the column never crosses (fully sat or fully dry)."""
    # sort by elevation ascending
    order = np.argsort(z_col)
    z = z_col[order]
    p = psi_col[order]
    # water table = top of the saturated zone: highest z with psi >= 0 that has an unsat cell above.
    # scan from the top down for the first sign change p>=0 (below) <- p<0 (above).
    for k in range(z.size - 1, 0, -1):
        p_up, p_lo = p[k], p[k - 1]
        if p_lo >= 0.0 and p_up < 0.0:
            # crossing between z[k-1] (sat) and z[k] (unsat): interpolate psi=0
            frac = p_lo / (p_lo - p_up)  # in [0,1]
            return z[k - 1] + frac * (z[k] - z[k - 1])
    # no interior crossing: fully saturated (psi>=0 at top) -> table at/above surface; fully dry -> NaN
    if p[-1] >= 0.0:
        return z[-1]          # saturated to the surface
    return np.nan             # never saturated in this column


def main():
    comm = MPI.COMM_WORLD
    t_build = time.perf_counter()

    # --- domain / mesh (2-D vertical transect; gravity on index 1) ---
    Lx, Dz, NX, NZ = 20.0, 2.0, 40, 20
    S0 = 0.02                 # 2% surface slope, descending toward x=Lx
    msh = dmesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Dz]], [NX, NZ], dmesh.CellType.triangle)
    ncell = msh.topology.index_map(msh.topology.dim).size_local

    # --- soil: a single uniform LOAM (uniform for robustness; do NOT layer) ---
    soil = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)

    # --- coupled problem (quadrature cap default 8; default petsc 'bt' linesearch) ---
    prob = CoupledProblem(msh, soil, n_man=0.05)
    ndof = prob.Vpsi.dofmap.index_map.size_local
    print(f"[build {time.perf_counter()-t_build:5.2f}s] mesh {NX}x{NZ} -> {ncell} tris, "
          f"{ndof} psi-dofs x3 = {3*ndof} DOFs; ell_c={prob.ell_c:.4f} m", flush=True)

    # --- IC: water table at z=1.0 m -> psi = z_wt - z (sat below 1 m, unsat above), no ponding ---
    Z_WT = 1.0
    prob.set_initial_condition(lambda x: Z_WT - x[1], d_value=0.0)

    # --- the hillslope: 2% bed descending toward x=Lx; surface outlet at x=Lx ---
    prob.set_topography(lambda x: S0 * (Lx - x[0]))
    prob.add_outflow_bc(lambda x: np.isclose(x[0], Lx), slope=S0)

    # --- THE TILE DRAIN: buried at the base, localized x in [9.5,10.5], draining to head 0 ---
    # locate_entities_boundary marks a base facet only if ALL its vertices satisfy the predicate; with
    # cells 0.5 m wide the relevant base vertices are at 9.5/10.0/10.5, so widen the band slightly to
    # catch the two facets centered on the drain (a strict >9.5/<10.5 would drop both edge vertices).
    DRAIN_C, DRAIN_HEXT = 2.0, 0.0
    prob.add_drainage_bc(
        locator=lambda x: np.isclose(x[1], 0.0) & (x[0] > 9.4) & (x[0] < 10.6),
        conductance=DRAIN_C, external_head=DRAIN_HEXT)
    # report the actual drain facet extent that got tagged
    drain_facets = prob._drains[-1][0]
    print(f"  tile drain: C={DRAIN_C}/day, H_ext={DRAIN_HEXT} m, {drain_facets.size} base facets "
          f"tagged near x in [9.5,10.5]", flush=True)

    # --- forcing: moderate storm 0.05 m/day for 1.0 day, then recession; march to 4.0 d ---
    rain = prob.add_rain(0.0)
    RATE, STORM_DUR, T_END = 0.05, 1.0, 4.0

    # --- viz extraction: regular (z, x) grid on the P1 vertices ---
    pcoord = prob.Vpsi.tabulate_dof_coordinates()[:, :2]
    xu, zu, ix, iz = _grid_index_xz(pcoord)
    nxc, nzc = xu.size, zu.size
    top_len = Lx          # 2-D: the top edge length = Lx (rain volume = RATE * Lx * dt)

    # columns for the water-table extraction (group dof indices by x)
    col_of = {xv: np.where(ix == k)[0] for k, xv in enumerate(xu)}

    N_OUT = 40
    out_times = np.linspace(0.0, T_END, N_OUT)
    head_field = np.full((N_OUT, nzc, nxc), np.nan)
    theta_field = np.full((N_OUT, nzc, nxc), np.nan)
    water_table = np.full((N_OUT, nxc), np.nan)
    drain_disch = np.zeros(N_OUT)
    cum_drain_t = np.zeros(N_OUT)
    rainfall_t = np.zeros(N_OUT)
    soil_w = np.zeros(N_OUT)
    surf_w = np.zeros(N_OUT)
    cum_out_t = np.zeros(N_OUT)
    cum_rain_t = np.zeros(N_OUT)
    mbe = np.zeros(N_OUT)

    w0 = prob.total_water()

    def snap(k, cum_rain):
        psi = prob.psi.x.array
        head_field[k][iz, ix] = psi
        theta_field[k][iz, ix] = soil.theta(psi)
        for kx, xv in enumerate(xu):
            cd = col_of[xv]
            water_table[k, kx] = _water_table(psi[cd], pcoord[cd, 1])
        drain_disch[k] = prob.last_drainage
        cum_drain_t[k] = prob.cum_drainage
        soil_w[k] = prob.soil_water()
        surf_w[k] = prob.surface_water()
        cum_out_t[k] = prob.cum_outflow
        cum_rain_t[k] = cum_rain
        expected = cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust
        mbe[k] = abs((prob.total_water() - w0) - expected) / (cum_rain + 1e-30)

    print(f"=== marching 0 -> {T_END} d (storm {RATE} m/day for {STORM_DUR} d, then recession) ===",
          flush=True)
    snap(0, 0.0)

    dt, t_sim, cum_rain, nsteps, k_out = 1e-3, 0.0, 0.0, 0, 1
    tstart = time.perf_counter()
    first_step_s = None
    while t_sim < T_END - 1e-12:
        h = min(dt, T_END - t_sim)
        # do not straddle the storm cutoff (clip to land exactly on STORM_DUR)
        if t_sim < STORM_DUR - 1e-12 and t_sim + h > STORM_DUR:
            h = STORM_DUR - t_sim
        # stop on the next snapshot time
        if k_out < N_OUT and t_sim + h > out_times[k_out]:
            h = out_times[k_out] - t_sim
        r = RATE if t_sim < STORM_DUR - 1e-12 else 0.0
        rain.value = r
        ts = time.perf_counter()
        conv, it = prob.step(h)
        step_s = time.perf_counter() - ts
        if first_step_s is None:
            first_step_s = step_s
            print(f"  step 1 (compile+solve): {step_s:.1f}s converged={conv} iters={it}", flush=True)
        if conv:
            cum_rain += r * top_len * h
            t_sim += h
            nsteps += 1
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 0.05)
            while k_out < N_OUT and t_sim >= out_times[k_out] - 1e-12:
                snap(k_out, cum_rain)
                k_out += 1
            if nsteps % 25 == 0:
                print(f"  t={t_sim:.4f}/{T_END} dt={dt:.2e} it={it} cum_rain={cum_rain:.4f} "
                      f"cum_out={prob.cum_outflow:.4f} cum_drain={prob.cum_drainage:.4f} "
                      f"drain_rate={prob.last_drainage:.4e}", flush=True)
        else:
            dt *= 0.5
            if dt < 1e-9:
                print(f"  !! DT COLLAPSE at t={t_sim:.6f}", flush=True)
                break
    # pad any unfilled snapshots with the final state
    while k_out < N_OUT:
        snap(k_out, cum_rain)
        k_out += 1
    rainfall_t = np.where(out_times < STORM_DUR - 1e-12, RATE, 0.0)

    wall = time.perf_counter() - tstart
    done = t_sim >= T_END - 1e-9

    # --- VERIFY (the acceptance bar) ---
    psi_finite = bool(np.all(np.isfinite(prob.psi.x.array)))
    theta_all = soil.theta(prob.psi.x.array)
    theta_finite = bool(np.all(np.isfinite(theta_all)))
    bal = (prob.total_water() - w0) - (
        cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust)
    bal_rel = abs(bal) / (cum_rain + 1e-30)
    drains_out = prob.cum_drainage > 0.0
    # Drawdown: the water table far from the drain (x~2) vs the cone near the drain (x in [9,11]) at
    # the FINAL snapshot. The drain pulls the cone DOWN; the exact centerline column (x=10) can drop
    # fully below the base (psi<0 everywhere -> NaN water table), which is the DEEPEST drawdown. So
    # measure the cone by the MINIMUM (deepest) finite water table in [9,11]; a NaN centerline with a
    # finite deep cone edge is even stronger drawdown. Drawdown is visible iff the cone is well below
    # the far-field table.
    kx2 = int(np.argmin(np.abs(xu - 2.0)))
    kx10 = int(np.argmin(np.abs(xu - 10.0)))
    wt2_final = water_table[-1, kx2]
    wt10_final = water_table[-1, kx10]                       # exact drain-centerline column (may be NaN)
    near = (xu >= 9.0) & (xu <= 11.0)
    cone = water_table[-1, near]
    wt_cone_min = float(np.nanmin(cone)) if np.any(np.isfinite(cone)) else np.nan
    drawdown_visible = bool(np.isfinite(wt2_final) and np.isfinite(wt_cone_min)
                            and wt_cone_min < wt2_final - 0.1)   # cone >=0.1 m below the far field

    print("=== HILLSLOPE-DRAIN ILLUSTRATION RESULT ===", flush=True)
    print(f"  status              : {'REACHED t_end' if done else f'PARTIAL ({t_sim:.4f}/{T_END} d)'}",
          flush=True)
    print(f"  wall clock          : {wall:.1f}s over {nsteps} steps "
          f"({wall/max(nsteps,1):.3f}s/step; compile+first {first_step_s:.1f}s)", flush=True)
    print(f"  (1) finite          : psi={psi_finite}  theta={theta_finite}", flush=True)
    print(f"  (2) mass balance    : |bal|/cum_rain = {bal_rel:.3e}  (gate < 1e-3 -> "
          f"{'PASS' if bal_rel < 1e-3 else 'FAIL'})", flush=True)
    print(f"  (3) drain discharges: cum_drainage = {prob.cum_drainage:.5f} m^2 (>0 -> "
          f"{'PASS' if drains_out else 'FAIL'})", flush=True)
    print(f"      drawdown        : wt(x~2)={wt2_final:.4f} m  wt(x=10 centerline)={wt10_final:.4f} m  "
          f"wt(cone-min, x in[9,11])={wt_cone_min:.4f} m  (cone >0.1 m below far field -> "
          f"{'PASS' if drawdown_visible else 'FAIL'})", flush=True)
    print(f"  (4) reached t_end   : {done}", flush=True)
    print(f"  cum_rain={cum_rain:.5f} m^2  cum_outflow={prob.cum_outflow:.5f} m^2  "
          f"cum_drainage={prob.cum_drainage:.5f} m^2  clip={prob.clip_mass_adjust:.2e}", flush=True)
    print(f"  soil_water_final={prob.soil_water():.4f} m^2  surface_depth_max="
          f"{prob.surface_depth():.5f} m", flush=True)

    # --- emit the standardized NetCDF (the viz data contract) ---
    ds = xr.Dataset(
        data_vars=dict(
            head_field=(("time", "z", "x"), head_field,
                        {"units": "m", "long_name": "pressure head psi(x,z) (water table = psi=0)"}),
            theta_field=(("time", "z", "x"), theta_field,
                         {"units": "-", "long_name": "volumetric water content theta(x,z)"}),
            water_table=(("time", "x"), water_table,
                         {"units": "m", "long_name": "water-table elevation (z where psi=0 per column; "
                                                     "NaN if no crossing)"}),
            drain_discharge=(("time",), drain_disch,
                             {"units": "m^2/day", "long_name": "tile-drain discharge per unit width "
                                                              "(GHB facet flux; + = out)"}),
            cum_drainage=(("time",), cum_drain_t,
                          {"units": "m^2", "long_name": "cumulative tile-drain discharge per unit width"}),
            rainfall=(("time",), rainfall_t, {"units": "m/day", "long_name": "rainfall rate"}),
            soil_water=(("time",), soil_w,
                        {"units": "m^2", "long_name": "subsurface stored water per unit width (int theta)"}),
            surface_water=(("time",), surf_w,
                           {"units": "m^2", "long_name": "surface ponded water per unit width (int d)"}),
            cum_outflow=(("time",), cum_out_t,
                         {"units": "m^2", "long_name": "cumulative surface outlet discharge per unit width"}),
            cum_rain=(("time",), cum_rain_t,
                      {"units": "m^2", "long_name": "cumulative rainfall volume per unit width"}),
            mass_balance_error=(("time",), mbe,
                                {"units": "-", "long_name": "|(total-w0)-(cum_rain-cum_out-cum_drain+clip)|"
                                                            "/cum_rain"}),
        ),
        coords=dict(
            time=("time", out_times, {"units": "day"}),
            x=("x", xu, {"units": "m", "long_name": "horizontal distance (outlet at x=Lx)"}),
            z=("z", zu, {"units": "m", "long_name": "elevation (0=base/drain depth, Dz=top of column)"}),
        ),
        attrs=dict(
            module="m4_hillslope_drain",
            date=DATE,
            scenario=f"2-D vertical hillslope transect, uniform loam, water table at z={Z_WT} m, "
                     f"storm {RATE} m/day for {STORM_DUR} d then recession; a buried tile drain at "
                     f"x in [9.5,10.5] (C={DRAIN_C}/day, head {DRAIN_HEXT} m) pulls a drawdown cone",
            domain=f"{Lx} x {Dz} m vertical cross-section (per unit width)",
            mesh=f"{NX}x{NZ} triangles ({ncell} cells, {3*ndof} DOFs)",
            drain_location=f"x in [9.5,10.5] m at base z=0 ({drain_facets.size} facets)",
            drain_conductance_per_day=float(DRAIN_C),
            drain_external_head_m=float(DRAIN_HEXT),
            surface_slope=float(S0),
            initial_water_table_m=float(Z_WT),
            status="REACHED_T_END" if done else f"PARTIAL_{t_sim:.4f}d",
            t_reached_day=float(t_sim),
            wall_clock_s=float(wall),
            steps=int(nsteps),
            mass_balance_error_final=float(bal_rel),
            mass_balance_error_max=float(mbe.max()),
            cum_rain_m2=float(cum_rain),
            cum_outflow_m2=float(prob.cum_outflow),
            cum_drainage_m2=float(prob.cum_drainage),
            soil_water_final_m2=float(prob.soil_water()),
            water_table_farfield_x2_m=float(wt2_final),
            water_table_cone_min_m=float(wt_cone_min),
            drawdown_m=float(wt2_final - wt_cone_min) if np.isfinite(wt_cone_min) else float("nan"),
            clip_mass_adjust=float(prob.clip_mass_adjust),
            note="ILLUSTRATION on the validated CoupledProblem engine (not a new validated module)",
        ),
    )
    ds.to_netcdf(OUT_NC)
    print(f"WROTE {OUT_NC}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
