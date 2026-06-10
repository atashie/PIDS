"""ILLUSTRATION (not a new validated module): a 2-D vertical hillslope transect with a buried TILE
DRAIN, run on the VALIDATED CoupledProblem engine, identical to scratch/m4_hillslope_drain.py EXCEPT
the subsoil below z=1.0 m is a GENUINE TIGHT CLAY (a real second van-Genuchten curve, ~50x tighter
than the loam topsoil), rather than a single uniform loam.

The contrast this shows: in the pure-loam case the storm infiltrates freely and the base tile drain
pulls a clean ~1.0 m drawdown CONE into the water table. Here the tight clay subsoil (Ks=0.005 m/day,
~12x below the 0.06 m/day storm) cannot accept the infiltrating water, so a PERCHED saturated zone
mounds up above the clay interface (theta -> theta_s, psi >= 0 in the loam just above z=1.0), and the
drain -- though head-controlled and generous -- is STARVED because the clay BULK conductivity limits
how fast water can reach the base: the drawdown is much weaker/shallower than the loam case's ~1.0 m.

Two-layer soil is a duck-typed shim (LayeredSoil pattern from scratch/feasibility_2ha_layered.py) but
with TWO genuine van-Genuchten curves selected by z (interface at z=1.0 m), NOT the Ks-decay loam
approximation. No changes to the validated module code.

Run from forward-model/ with PYTHONPATH=. and *_NUM_THREADS=1 (BLAS pinned -- the tiny coupled systems
oversubscribe multithreaded BLAS). Structure copied verbatim from scratch/m4_hillslope_drain.py.
"""
from __future__ import annotations

import time
import numpy as np
import ufl
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

DATE = "2026-06-09"
DATA_DIR = "../validation/sanity/data"
OUT_NC = f"{DATA_DIR}/m4_hillslope_drain_clay__{DATE}.nc"

Z_IFACE = 1.0   # loam / clay interface elevation (m); loam above, tight clay below


# --------------------------------------------------------------------------- two-layer genuine-vG soil
class TwoLayerSoil:
    """Duck-typed two-layer van-Genuchten with a HARD z interface at z=Z_IFACE (loam above / clay below).

    Same UFL interface CoupledProblem uses (theta_ufl / K_ufl / kirchhoff_ufl) plus numpy closures for
    the viz (theta_of(psi, z), Ks_of_z(z)). Unlike the feasibility LayeredSoil (which decays loam Ks
    toward clay-LIKE values while keeping the loam retention SHAPE), here BOTH branches are GENUINE
    van-Genuchten curves -- a real loam and a real tight clay -- so the clay's flat retention + low Ks
    actually perches water.

    `Ks` is exposed as the SURFACE (loam) scalar for the NCP tau_c = ell_c/Ks: surface infiltration is
    in the loam top, so the land-surface exchange uses the loam scalar. `kirchhoff_ufl` is the CONSTANT
    LOAM matric flux potential (the infiltration leg lives in the loam top; z=z_surf -> loam), matching
    the feasibility LayeredSoil convention -- keeps the Simpson sum a small loam-only expression.
    """

    def __init__(self, mesh, *, z_iface=Z_IFACE,
                 loam=dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25),
                 clay=dict(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.005)):
        self._z = ufl.SpatialCoordinate(mesh)[mesh.geometry.dim - 1]
        self.z_iface = float(z_iface)
        # genuine van-Genuchten objects -- reuse the validated closures branch-by-branch
        self._loam = VanGenuchten(**loam)
        self._clay = VanGenuchten(**clay)
        self.loam, self.clay = loam, clay
        # interface predicate: loam where z >= z_iface, clay below
        self._loam_cond = ufl.ge(self._z, z_iface - 1e-9)
        # exposed scalars: SURFACE = loam (tau_c uses the loam top, infiltration is in the loam)
        self.Ks = loam["Ks"]
        self.theta_r = loam["theta_r"]
        self.theta_s = loam["theta_s"]

    # --- UFL solve-path closures (z-selected genuine vG) ---
    def theta_ufl(self, psi):
        return ufl.conditional(self._loam_cond, self._loam.theta_ufl(psi), self._clay.theta_ufl(psi))

    def K_ufl(self, psi):
        return ufl.conditional(self._loam_cond, self._loam.K_ufl(psi), self._clay.K_ufl(psi))

    # surface infiltration leg lives in the loam top (z=z_surf -> loam branch); use the CONSTANT loam
    # Kirchhoff so the Simpson sum stays a small loam-only expression (no conditional/branch blow-up).
    def kirchhoff_ufl(self, a, b):
        return self._loam.kirchhoff_ufl(a, b)

    # --- numpy closures for the VIZ (z-aware; NOT used in the solve path) ---
    def theta_of(self, psi, z):
        z = np.asarray(z, float)
        in_loam = z >= self.z_iface - 1e-9
        th_loam = self._loam.theta(psi)
        th_clay = self._clay.theta(psi)
        return np.where(in_loam, th_loam, th_clay)

    def Ks_of_z(self, z):
        z = np.asarray(z, float)
        in_loam = z >= self.z_iface - 1e-9
        return np.where(in_loam, self.loam["Ks"], self.clay["Ks"])


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
    order = np.argsort(z_col)
    z = z_col[order]
    p = psi_col[order]
    for k in range(z.size - 1, 0, -1):
        p_up, p_lo = p[k], p[k - 1]
        if p_lo >= 0.0 and p_up < 0.0:
            frac = p_lo / (p_lo - p_up)  # in [0,1]
            return z[k - 1] + frac * (z[k] - z[k - 1])
    if p[-1] >= 0.0:
        return z[-1]          # saturated to the surface
    return np.nan             # never saturated in this column


def _perched_extent(psi_col, z_col, z_iface):
    """Vertical extent of a GENUINE PERCHED saturated zone in the LOAM just above the clay interface.

    A genuine perch = a contiguous saturated band (psi >= 0) sitting ON the clay interface (lowest loam
    node z>=z_iface is saturated) that is CAPPED BY UNSATURATED LOAM ABOVE it (there is a loam node with
    psi < 0 higher up in the same column). That distinguishes a perched MOUND on the clay from a column
    that has simply filled to the surface (no unsaturated cap -> not perching, just full saturation).

    Returns the thickness (m) of the contiguous psi>=0 loam band on the interface, OR 0.0 if the lowest
    loam node is unsaturated (front not yet at the clay) OR if the whole loam column is saturated to the
    top (full saturation, not a perch)."""
    order = np.argsort(z_col)
    z = z_col[order]
    p = psi_col[order]
    loam = z >= z_iface - 1e-9
    if not np.any(loam):
        return 0.0
    zl = z[loam]
    pl = p[loam]
    sat = pl >= 0.0
    if not sat[0]:
        return 0.0            # lowest loam node unsaturated -> front not at the clay yet (no perch)
    if np.all(sat):
        return 0.0            # whole loam column saturated -> full saturation, NOT a perched mound
    # contiguous saturated band from the interface up, capped by an unsaturated loam node above
    top = 0
    for k in range(zl.size):
        if sat[k]:
            top = k
        else:
            break
    return float(zl[top] - zl[0])


def main():
    comm = MPI.COMM_WORLD
    t_build = time.perf_counter()

    # --- domain / mesh (2-D vertical transect; gravity on index 1) -- IDENTICAL to the loam case ---
    Lx, Dz, NX, NZ = 20.0, 2.0, 40, 20
    S0 = 0.02                 # 2% surface slope, descending toward x=Lx
    msh = dmesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Dz]], [NX, NZ], dmesh.CellType.triangle)
    ncell = msh.topology.index_map(msh.topology.dim).size_local

    # --- soil: TWO genuine van-Genuchten layers (loam topsoil / tight clay subsoil at z=1.0 m) ---
    soil = TwoLayerSoil(msh)
    print(f"  SOIL: loam top (Ks={soil.loam['Ks']}, alpha={soil.loam['alpha']}, n={soil.loam['n']})  "
          f"over tight clay (Ks={soil.clay['Ks']}, alpha={soil.clay['alpha']}, n={soil.clay['n']}) "
          f"at z={soil.z_iface} m", flush=True)

    # --- coupled problem (quadrature cap default 8). Try the default 'bt' linesearch first; the clay
    # is stiff + the storm wets toward a saturated wall (memory: wetting/saturated-wall steps want
    # 'cp'), so fall back to a fresh problem with snes_linesearch_type='cp' if Newton stalls. ---
    def build_problem(linesearch=None):
        opts = None
        if linesearch is not None:
            opts = dict(CoupledProblem._DEFAULT_PETSC_OPTIONS)
            opts["snes_linesearch_type"] = linesearch
        return CoupledProblem(msh, soil, n_man=0.05, petsc_options=opts)

    prob = build_problem(None)
    ndof = prob.Vpsi.dofmap.index_map.size_local
    print(f"[build {time.perf_counter()-t_build:5.2f}s] mesh {NX}x{NZ} -> {ncell} tris, "
          f"{ndof} psi-dofs x3 = {3*ndof} DOFs; ell_c={prob.ell_c:.4f} m", flush=True)

    # --- IC: regional water table at z=0.7 m (in the clay) -> psi = 0.7 - z, no ponding ---
    Z_WT = 0.7
    prob.set_initial_condition(lambda x: Z_WT - x[1], d_value=0.0)

    # --- the hillslope: 2% bed descending toward x=Lx; surface outlet at x=Lx ---
    prob.set_topography(lambda x: S0 * (Lx - x[0]))
    prob.add_outflow_bc(lambda x: np.isclose(x[0], Lx), slope=S0)

    # --- THE TILE DRAIN: buried at the base, localized x in [9.5,10.5], GENEROUS head-controlled drain
    # (high conductance) so the CLAY BULK K is the limiter, not the GHB film -- the physical "clay is
    # hard to drain" effect. Same widened band as the loam case to catch the two centered facets. ---
    DRAIN_C, DRAIN_HEXT = 30.0, 0.0
    prob.add_drainage_bc(
        locator=lambda x: np.isclose(x[1], 0.0) & (x[0] > 9.4) & (x[0] < 10.6),
        conductance=DRAIN_C, external_head=DRAIN_HEXT)
    drain_facets = prob._drains[-1][0]
    print(f"  tile drain: C={DRAIN_C}/day, H_ext={DRAIN_HEXT} m, {drain_facets.size} base facets "
          f"tagged near x in [9.5,10.5]", flush=True)

    # --- forcing: storm 0.085 m/day for 1.5 days (17x the clay Ks 0.005), then recession; march to
    # 5.0 d (the clay responds slowly). Rate TUNED for a clean PERCHED-MOUND illustration:
    #   0.06 m/day (first try) wets the 1-m loam too slowly -> the front reached the clay only ~1 d into
    #     RECESSION (perch formed late, one cell thick) -- too weak.
    #   0.12 m/day (second try) over-drives it -> the whole column saturates to the surface (full
    #     saturation + surface runoff, a hydrostatic psi=WT-z profile), which is NOT a perched mound.
    #   0.085 m/day lands in between: the front reaches the clay DURING the storm and mounds a saturated
    #     band on the interface while the loam ABOVE stays unsaturated (a genuine perch), and the clay's
    #     low Ks keeps the drain sluggish -- the intended clay-vs-loam contrast. ---
    rain = prob.add_rain(0.0)
    RATE, STORM_DUR, T_END = 0.085, 1.5, 5.0

    # --- viz extraction: regular (z, x) grid on the P1 vertices ---
    pcoord = prob.Vpsi.tabulate_dof_coordinates()[:, :2]
    xu, zu, ix, iz = _grid_index_xz(pcoord)
    nxc, nzc = xu.size, zu.size
    top_len = Lx          # 2-D: the top edge length = Lx (rain volume = RATE * Lx * dt)

    col_of = {xv: np.where(ix == k)[0] for k, xv in enumerate(xu)}

    N_OUT = 50
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
        theta_field[k][iz, ix] = soil.theta_of(psi, pcoord[:, 1])
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
    stalled = False
    while t_sim < T_END - 1e-12:
        h = min(dt, T_END - t_sim)
        if t_sim < STORM_DUR - 1e-12 and t_sim + h > STORM_DUR:
            h = STORM_DUR - t_sim
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
                print(f"  !! DT COLLAPSE at t={t_sim:.6f} -- Newton stalled with the default "
                      f"linesearch", flush=True)
                stalled = True
                break
    if stalled:
        print("  -> retrying from scratch with snes_linesearch_type='cp' (wetting / saturated wall)",
              flush=True)
        return main_cp()

    # pad any unfilled snapshots with the final state
    while k_out < N_OUT:
        snap(k_out, cum_rain)
        k_out += 1
    rainfall_t = np.where(out_times < STORM_DUR - 1e-12, RATE, 0.0)

    wall = time.perf_counter() - tstart
    done = t_sim >= T_END - 1e-9

    _finish(prob, soil, msh, ncell, ndof, S0, Lx, Dz, NX, NZ, Z_WT, DRAIN_C, DRAIN_HEXT, drain_facets,
            RATE, STORM_DUR, T_END, out_times, head_field, theta_field, water_table, drain_disch,
            cum_drain_t, rainfall_t, soil_w, surf_w, cum_out_t, cum_rain_t, mbe, xu, zu, ix, iz, iz,
            pcoord, w0, cum_rain, wall, nsteps, first_step_s, done, t_sim, "bt")


def main_cp():
    """Identical run but with the 'cp' SNES linesearch (wetting / saturated-wall) -- a clean re-run from
    a fresh CoupledProblem (state is per-problem; cannot swap the linesearch mid-run)."""
    comm = MPI.COMM_WORLD
    Lx, Dz, NX, NZ = 20.0, 2.0, 40, 20
    S0 = 0.02
    msh = dmesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Dz]], [NX, NZ], dmesh.CellType.triangle)
    ncell = msh.topology.index_map(msh.topology.dim).size_local
    soil = TwoLayerSoil(msh)
    opts = dict(CoupledProblem._DEFAULT_PETSC_OPTIONS)
    opts["snes_linesearch_type"] = "cp"
    prob = CoupledProblem(msh, soil, n_man=0.05, petsc_options=opts)
    ndof = prob.Vpsi.dofmap.index_map.size_local
    print(f"[cp build] mesh {NX}x{NZ} -> {ncell} tris, {3*ndof} DOFs; ell_c={prob.ell_c:.4f} m",
          flush=True)

    Z_WT = 0.7
    prob.set_initial_condition(lambda x: Z_WT - x[1], d_value=0.0)
    prob.set_topography(lambda x: S0 * (Lx - x[0]))
    prob.add_outflow_bc(lambda x: np.isclose(x[0], Lx), slope=S0)
    DRAIN_C, DRAIN_HEXT = 30.0, 0.0
    prob.add_drainage_bc(
        locator=lambda x: np.isclose(x[1], 0.0) & (x[0] > 9.4) & (x[0] < 10.6),
        conductance=DRAIN_C, external_head=DRAIN_HEXT)
    drain_facets = prob._drains[-1][0]
    rain = prob.add_rain(0.0)
    RATE, STORM_DUR, T_END = 0.085, 1.5, 5.0

    pcoord = prob.Vpsi.tabulate_dof_coordinates()[:, :2]
    xu, zu, ix, iz = _grid_index_xz(pcoord)
    nxc, nzc = xu.size, zu.size
    top_len = Lx
    col_of = {xv: np.where(ix == k)[0] for k, xv in enumerate(xu)}
    N_OUT = 50
    out_times = np.linspace(0.0, T_END, N_OUT)
    head_field = np.full((N_OUT, nzc, nxc), np.nan)
    theta_field = np.full((N_OUT, nzc, nxc), np.nan)
    water_table = np.full((N_OUT, nxc), np.nan)
    drain_disch = np.zeros(N_OUT); cum_drain_t = np.zeros(N_OUT); rainfall_t = np.zeros(N_OUT)
    soil_w = np.zeros(N_OUT); surf_w = np.zeros(N_OUT); cum_out_t = np.zeros(N_OUT)
    cum_rain_t = np.zeros(N_OUT); mbe = np.zeros(N_OUT)
    w0 = prob.total_water()

    def snap(k, cum_rain):
        psi = prob.psi.x.array
        head_field[k][iz, ix] = psi
        theta_field[k][iz, ix] = soil.theta_of(psi, pcoord[:, 1])
        for kx, xv in enumerate(xu):
            cd = col_of[xv]
            water_table[k, kx] = _water_table(psi[cd], pcoord[cd, 1])
        drain_disch[k] = prob.last_drainage; cum_drain_t[k] = prob.cum_drainage
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water()
        cum_out_t[k] = prob.cum_outflow; cum_rain_t[k] = cum_rain
        expected = cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust
        mbe[k] = abs((prob.total_water() - w0) - expected) / (cum_rain + 1e-30)

    print(f"=== [cp] marching 0 -> {T_END} d (storm {RATE} m/day for {STORM_DUR} d) ===", flush=True)
    snap(0, 0.0)
    dt, t_sim, cum_rain, nsteps, k_out = 1e-3, 0.0, 0.0, 0, 1
    tstart = time.perf_counter(); first_step_s = None
    while t_sim < T_END - 1e-12:
        h = min(dt, T_END - t_sim)
        if t_sim < STORM_DUR - 1e-12 and t_sim + h > STORM_DUR:
            h = STORM_DUR - t_sim
        if k_out < N_OUT and t_sim + h > out_times[k_out]:
            h = out_times[k_out] - t_sim
        r = RATE if t_sim < STORM_DUR - 1e-12 else 0.0
        rain.value = r
        ts = time.perf_counter()
        conv, it = prob.step(h)
        step_s = time.perf_counter() - ts
        if first_step_s is None:
            first_step_s = step_s
            print(f"  [cp] step 1: {step_s:.1f}s converged={conv} iters={it}", flush=True)
        if conv:
            cum_rain += r * top_len * h
            t_sim += h; nsteps += 1
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 0.05)
            while k_out < N_OUT and t_sim >= out_times[k_out] - 1e-12:
                snap(k_out, cum_rain); k_out += 1
            if nsteps % 25 == 0:
                print(f"  [cp] t={t_sim:.4f}/{T_END} dt={dt:.2e} it={it} cum_rain={cum_rain:.4f} "
                      f"cum_drain={prob.cum_drainage:.4f} drain_rate={prob.last_drainage:.4e}",
                      flush=True)
        else:
            dt *= 0.5
            if dt < 1e-9:
                print(f"  !! [cp] DT COLLAPSE at t={t_sim:.6f}", flush=True)
                break
    while k_out < N_OUT:
        snap(k_out, cum_rain); k_out += 1
    rainfall_t = np.where(out_times < STORM_DUR - 1e-12, RATE, 0.0)
    wall = time.perf_counter() - tstart
    done = t_sim >= T_END - 1e-9
    _finish(prob, soil, msh, ncell, ndof, S0, Lx, Dz, NX, NZ, Z_WT, DRAIN_C, DRAIN_HEXT, drain_facets,
            RATE, STORM_DUR, T_END, out_times, head_field, theta_field, water_table, drain_disch,
            cum_drain_t, rainfall_t, soil_w, surf_w, cum_out_t, cum_rain_t, mbe, xu, zu, ix, iz, iz,
            pcoord, w0, cum_rain, wall, nsteps, first_step_s, done, t_sim, "cp")


def _finish(prob, soil, msh, ncell, ndof, S0, Lx, Dz, NX, NZ, Z_WT, DRAIN_C, DRAIN_HEXT, drain_facets,
            RATE, STORM_DUR, T_END, out_times, head_field, theta_field, water_table, drain_disch,
            cum_drain_t, rainfall_t, soil_w, surf_w, cum_out_t, cum_rain_t, mbe, xu, zu, ix, iz, iz2,
            pcoord, w0, cum_rain, wall, nsteps, first_step_s, done, t_sim, linesearch):
    """Shared verification + NetCDF emit (same contract as the loam .nc, plus Ks_profile + clay attrs)."""
    LOAM_DRAWDOWN = 1.0   # the loam case's clean ~1.0 m drawdown cone, for contrast

    psi_finite = bool(np.all(np.isfinite(prob.psi.x.array)))
    theta_all = soil.theta_of(prob.psi.x.array, pcoord[:, 1])
    theta_finite = bool(np.all(np.isfinite(theta_all)))
    bal = (prob.total_water() - w0) - (
        cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust)
    bal_rel = abs(bal) / (cum_rain + 1e-30)
    drains_out = prob.cum_drainage > 0.0

    # --- PERCHED ZONE: a GENUINE perched mound = a saturated (psi>=0) loam band sitting ON the clay
    # interface (z >= Z_IFACE) that is CAPPED by unsaturated loam above (so it is NOT just a column
    # filled to the surface). Track, over all snapshots: the MAX band thickness (where/when), and the
    # BROADEST snapshot (most columns perching simultaneously -- a slope-wide perched water table). ---
    col_of = {xv: np.where(ix == k)[0] for k, xv in enumerate(xu)}
    perch_max, perch_t, perch_x = 0.0, np.nan, np.nan
    theta_perch = np.nan
    perch_ncol_best, perch_ncol_t, perch_ncol_meanthk = 0, np.nan, 0.0
    for k in range(out_times.size):
        psi_grid = head_field[k]
        ncol_k, thk_k = 0, []
        for kx, xv in enumerate(xu):
            cd = col_of[xv]
            zc = pcoord[cd, 1]
            pc = psi_grid[iz[cd], ix[cd]]
            ext = _perched_extent(pc, zc, Z_IFACE)
            if ext > 0.0:
                ncol_k += 1
                thk_k.append(ext)
            if ext > perch_max:
                perch_max = ext
                perch_t = float(out_times[k])
                perch_x = float(xv)
                loam_above = zc >= Z_IFACE - 1e-9
                if np.any(loam_above):
                    z_la = zc[loam_above]
                    p_la = pc[loam_above]
                    j = int(np.argmin(np.abs(z_la - Z_IFACE)))
                    theta_perch = float(soil.theta_of(p_la[j], z_la[j]))
        if ncol_k > perch_ncol_best:
            perch_ncol_best = ncol_k
            perch_ncol_t = float(out_times[k])
            perch_ncol_meanthk = float(np.mean(thk_k)) if thk_k else 0.0
    nxc_total = xu.size
    perch_in_storm = bool(np.isfinite(perch_t) and perch_t <= STORM_DUR + 1e-9)

    # --- DRAWDOWN: far-field table (x~2) vs the cone near the drain (x in [9,11]) at the FINAL
    # snapshot (same metric as the loam case). For the clay case we EXPECT this to be small. ---
    kx2 = int(np.argmin(np.abs(xu - 2.0)))
    kx10 = int(np.argmin(np.abs(xu - 10.0)))
    wt2_final = water_table[-1, kx2]
    wt10_final = water_table[-1, kx10]
    near = (xu >= 9.0) & (xu <= 11.0)
    cone = water_table[-1, near]
    wt_cone_min = float(np.nanmin(cone)) if np.any(np.isfinite(cone)) else np.nan
    clay_drawdown = float(wt2_final - wt_cone_min) if (np.isfinite(wt2_final)
                                                       and np.isfinite(wt_cone_min)) else float("nan")

    print("=== HILLSLOPE-DRAIN CLAY ILLUSTRATION RESULT ===", flush=True)
    print(f"  linesearch          : {linesearch}", flush=True)
    print(f"  status              : {'REACHED t_end' if done else f'PARTIAL ({t_sim:.4f}/{T_END} d)'}",
          flush=True)
    print(f"  wall clock          : {wall:.1f}s over {nsteps} steps "
          f"({wall/max(nsteps,1):.3f}s/step; compile+first {first_step_s:.1f}s)", flush=True)
    print(f"  (1) finite          : psi={psi_finite}  theta={theta_finite}", flush=True)
    print(f"  (2) mass balance    : |bal|/cum_rain = {bal_rel:.3e}  (gate < 1e-3 -> "
          f"{'PASS' if bal_rel < 1e-3 else 'FAIL'})  max={mbe.max():.3e}", flush=True)
    print(f"  (3) drain discharges: cum_drainage = {prob.cum_drainage:.5f} m^2 (>0 -> "
          f"{'PASS' if drains_out else 'FAIL'})", flush=True)
    print(f"  (4) perched zone    : max sat-loam band above clay = {perch_max:.4f} m at x={perch_x} m "
          f"t={perch_t} d (theta_just_above~{theta_perch:.4f}, theta_s_loam={soil.loam['theta_s']}); "
          f"BROADEST = {perch_ncol_best}/{nxc_total} columns perching at t={perch_ncol_t} d "
          f"(mean thk {perch_ncol_meanthk:.3f} m) -> {'PASS' if perch_max > 0.05 else 'FAIL'}",
          flush=True)
    print(f"      perch timing    : peak {'DURING storm' if perch_in_storm else 'in RECESSION'} "
          f"(storm ends {STORM_DUR} d) -- the 1-m loam buffers the storm; the front reaches the clay "
          f"~{max(perch_t-STORM_DUR,0):.1f} d after it ends", flush=True)
    print(f"  (5) drawdown        : wt(x~2)={wt2_final:.4f} m  wt(x=10)={wt10_final:.4f} m  "
          f"cone-min={wt_cone_min:.4f} m  clay_drawdown={clay_drawdown:.4f} m  "
          f"(vs loam ~{LOAM_DRAWDOWN} m -> weaker = {'YES' if clay_drawdown < 0.9 else 'NO'})",
          flush=True)
    print(f"  cum_rain={cum_rain:.5f} m^2  cum_outflow={prob.cum_outflow:.5f} m^2  "
          f"cum_drainage={prob.cum_drainage:.5f} m^2  clip={prob.clip_mass_adjust:.2e}", flush=True)

    # Ks(z) profile on the (z) grid for the viz (loam above / clay below the interface)
    Ks_profile = soil.Ks_of_z(zu)

    ds = xr.Dataset(
        data_vars=dict(
            head_field=(("time", "z", "x"), head_field,
                        {"units": "m", "long_name": "pressure head psi(x,z) (water table = psi=0)"}),
            theta_field=(("time", "z", "x"), theta_field,
                         {"units": "-", "long_name": "volumetric water content theta(x,z) (2-layer)"}),
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
            Ks_profile=(("z",), Ks_profile,
                        {"units": "m/day", "long_name": "Ks(z): loam (0.25) above / tight clay (0.005) "
                                                        "below the interface at z=1.0 m"}),
        ),
        coords=dict(
            time=("time", out_times, {"units": "day"}),
            x=("x", xu, {"units": "m", "long_name": "horizontal distance (outlet at x=Lx)"}),
            z=("z", zu, {"units": "m", "long_name": "elevation (0=base/drain depth, Dz=top of column)"}),
        ),
        attrs=dict(
            module="m4_hillslope_drain_clay",
            date=DATE,
            scenario=f"2-D hillslope transect, loam topsoil over tight CLAY subsoil (interface z={Z_IFACE} m); "
                     f"water table at z={Z_WT} m, storm {RATE} m/day for {STORM_DUR} d then recession; a "
                     f"generous head-controlled tile drain at x in [9.5,10.5] (C={DRAIN_C}/day, head "
                     f"{DRAIN_HEXT} m) -- the clay perches water + starves the drain",
            clay_interface_z=float(Z_IFACE),
            domain=f"{Lx} x {Dz} m vertical cross-section (per unit width)",
            mesh=f"{NX}x{NZ} triangles ({ncell} cells, {3*ndof} DOFs)",
            loam_params="theta_r=0.078 theta_s=0.43 alpha=3.6 n=1.56 Ks=0.25",
            clay_params="theta_r=0.068 theta_s=0.38 alpha=0.8 n=1.09 Ks=0.005",
            drain_location=f"x in [9.5,10.5] m at base z=0 ({drain_facets.size} facets)",
            drain_conductance_per_day=float(DRAIN_C),
            drain_external_head_m=float(DRAIN_HEXT),
            surface_slope=float(S0),
            initial_water_table_m=float(Z_WT),
            snes_linesearch=linesearch,
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
            clay_drawdown_m=clay_drawdown,
            loam_case_drawdown_m=float(LOAM_DRAWDOWN),
            perched_zone_max_thickness_m=float(perch_max),
            perched_zone_x_m=float(perch_x),
            perched_zone_time_day=float(perch_t),
            perched_zone_theta_just_above_interface=float(theta_perch),
            perched_zone_broadest_ncolumns=int(perch_ncol_best),
            perched_zone_total_columns=int(nxc_total),
            perched_zone_broadest_time_day=float(perch_ncol_t),
            perched_zone_broadest_mean_thickness_m=float(perch_ncol_meanthk),
            perched_zone_peak_during_storm=int(perch_in_storm),
            clip_mass_adjust=float(prob.clip_mass_adjust),
            note="ILLUSTRATION on the validated CoupledProblem engine (not a new validated module): "
                 "loam-over-tight-clay contrast vs the pure-loam ~1.0 m drawdown cone",
        ),
    )
    ds.to_netcdf(OUT_NC)
    print(f"WROTE {OUT_NC}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
