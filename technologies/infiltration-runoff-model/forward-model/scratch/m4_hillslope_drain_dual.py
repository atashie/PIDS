"""ILLUSTRATION (not a new validated module): the loam-over-tight-clay hillslope of
scratch/m4_hillslope_drain_clay.py, but with TWO drains replacing the buried base drain:

  (1) an INTERFACE TILE DRAIN sitting ON the clay (the agronomically standard placement): a
      MODFLOW-DRN-style volumetric sink over the one loam cell-row x in [9.5,10.5], z in [1.0,1.1],
      q = C_VOL * kr(psi) * pos(psi + z - z_pipe)  [1/day],  z_pipe = 1.0 (pipe at atmospheric:
      OUTFLOW-ONLY via a smooth max -- an air-filled pipe never injects), kr = K_loam(psi)/Ks_loam;
  (2) a SURFACE "TILE" (grate inlet) over the same x-footprint on the land surface: a linear intake
      on the ponded depth, q = C_SURF * d  [m/day], removing overland water that crosses it.

Both drains use the FIRST-CLASS engine APIs ``add_interior_drain`` / ``add_surface_inlet``
(plan docs/plans/2026-06-10-module4-engine-drain-inlet-apis.md): the GHB (add_drainage_bc) supports
EXTERIOR-boundary facets only and forbids the lambda-coupled top, so these sink types got their own
APIs -- PROMOTED from this run's original script-level form injection (git history 63145e2). The
API == injection equivalence is pinned by tests/test_engine_drains.py::test_api_matches_script_injection;
per-drain accounting is engine-owned (last_sinks/cum_sinks), and the structural mass balance
includes both sinks exactly as for the GHB.

FORCING is deliberately STRONGER than the base-drain clay run (0.085 -> 0.12 m/day, 1.5 -> 2.0 d):
at 0.085 the loam (Ks 0.25) swallows the whole storm and ponding NEVER occurs (proven by the
base-drain run: cum surface outflow ~ 0), so a surface drain would sit at exactly zero all run.
0.12 m/day was observed (clay-run tuning) to saturate the column to the surface: saturation-excess
ponding occurs, the surface inlet intercepts overland run-on, and the interface tile intercepts the
perched mound -- both placements visibly engage and their division of labor is the story.

Run from forward-model/ with PYTHONPATH=. and *_NUM_THREADS=1 (BLAS pinned).
"""
from __future__ import annotations

import time
import numpy as np
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.coupling import CoupledProblem
from scratch.m4_hillslope_drain_clay import (
    TwoLayerSoil, Z_IFACE, _grid_index_xz, _water_table, _perched_extent,
)

DATE = "2026-06-10"
OUT_NC = f"../validation/sanity/data/m4_hillslope_drain_dual__{DATE}.nc"

# --- the two drains (shared x-footprint so placement is the ONLY difference) ---
DRAIN_X0, DRAIN_X1 = 9.5, 10.5
IFACE_ZTOP = Z_IFACE + 0.1   # one 0.1-m mesh cell-row in the loam, sitting ON the clay
C_VOL = 300.0                # 1/(day*m): band-integrated ~ C_VOL*0.1 m^2 = 30 m^2/day per m head
HE_IFACE = Z_IFACE           # pipe invert at the interface; drains where H = psi+z > z_pipe
EPS_POS = 1e-3               # smooth-max width [m] (C-inf for Newton; leak ~C*eps/2 only AT activation)
C_SURF = 500.0               # 1/day grate intake on ponded depth (generous: supply-limited, not intake)

RATE, STORM_DUR, T_END = 0.12, 2.0, 5.0
N_OUT = 50


def run(linesearch: str | None):
    comm = MPI.COMM_WORLD
    t_build = time.perf_counter()
    Lx, Dz, NX, NZ = 20.0, 2.0, 40, 20
    S0 = 0.02
    msh = dmesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Dz]], [NX, NZ], dmesh.CellType.triangle)
    ncell = msh.topology.index_map(msh.topology.dim).size_local
    soil = TwoLayerSoil(msh)
    print(f"  SOIL: loam top (Ks={soil.loam['Ks']}) over tight clay (Ks={soil.clay['Ks']}) at "
          f"z={soil.z_iface} m", flush=True)

    opts = None
    if linesearch is not None:
        opts = dict(CoupledProblem._DEFAULT_PETSC_OPTIONS)
        opts["snes_linesearch_type"] = linesearch
    prob = CoupledProblem(msh, soil, n_man=0.05, petsc_options=opts)
    ndof = prob.Vpsi.dofmap.index_map.size_local
    ls_name = linesearch or "bt"
    print(f"[build {time.perf_counter()-t_build:5.2f}s ls={ls_name}] mesh {NX}x{NZ} -> {ncell} tris, "
          f"{3*ndof} DOFs; ell_c={prob.ell_c:.4f} m", flush=True)

    # --- engine setup (the add_* APIs weave their terms through the central form rebuilds, so
    # ordering is free; kept in the original order for diff-stability vs 63145e2) ---
    Z_WT = 0.7
    prob.set_initial_condition(lambda x: Z_WT - x[1], d_value=0.0)
    prob.set_topography(lambda x: S0 * (Lx - x[0]))
    prob.add_outflow_bc(lambda x: np.isclose(x[0], Lx), slope=S0)
    rain = prob.add_rain(0.0)

    # --- the two drains via the first-class engine APIs (see module docstring). Band/footprint
    # edges align with the 0.5 x 0.1 mesh grid; locators are ALL-VERTEX predicates, so the bounds
    # are INCLUSIVE (the DG-0 indicator then selects exactly the intended whole cells -> exact).
    tol = 1e-6
    prob.add_interior_drain(
        locator=lambda x: (x[0] > DRAIN_X0 - tol) & (x[0] < DRAIN_X1 + tol)
                          & (x[1] > Z_IFACE - tol) & (x[1] < IFACE_ZTOP + tol),
        conductance_density=C_VOL, drain_head=HE_IFACE, eps_act=EPS_POS)
    prob.add_surface_inlet(
        locator=lambda x: (x[0] > DRAIN_X0 - tol) & (x[0] < DRAIN_X1 + tol),
        intake_coeff=C_SURF)
    print(f"  drains: INTERFACE tile x in [{DRAIN_X0},{DRAIN_X1}] z in [{Z_IFACE},{IFACE_ZTOP}] "
          f"(C_VOL={C_VOL}/day/m, He={HE_IFACE} m, outflow-only smooth-DRN) + SURFACE grate inlet "
          f"same x-footprint (C_SURF={C_SURF}/day on d)", flush=True)

    # --- viz extraction grids ---
    pcoord = prob.Vpsi.tabulate_dof_coordinates()[:, :2]
    xu, zu, ix, iz = _grid_index_xz(pcoord)
    nxc, nzc = xu.size, zu.size
    top_len = Lx
    col_of = {xv: np.where(ix == k)[0] for k, xv in enumerate(xu)}
    # surface ponded-depth profile: top dofs of Vd ordered by x
    dcoord = prob.Vd.tabulate_dof_coordinates()[:, :2]
    topdofs = prob._top_dofs(prob.Vd)
    torder = np.argsort(dcoord[topdofs, 0])
    top_d_dofs = topdofs[torder]
    x_surf = dcoord[top_d_dofs, 0]
    assert np.allclose(x_surf, xu), "top Vd dof x-grid must match the psi vertex grid"

    out_times = np.linspace(0.0, T_END, N_OUT)
    head_field = np.full((N_OUT, nzc, nxc), np.nan)
    theta_field = np.full((N_OUT, nzc, nxc), np.nan)
    water_table = np.full((N_OUT, nxc), np.nan)
    ponded = np.zeros((N_OUT, nxc))
    perch_thk = np.zeros((N_OUT, nxc))
    q_if_t = np.zeros(N_OUT); q_sf_t = np.zeros(N_OUT)
    cum_if_t = np.zeros(N_OUT); cum_sf_t = np.zeros(N_OUT)
    soil_w = np.zeros(N_OUT); surf_w = np.zeros(N_OUT)
    cum_out_t = np.zeros(N_OUT); cum_rain_t = np.zeros(N_OUT); mbe = np.zeros(N_OUT)

    w0 = prob.total_water()
    cum_if = cum_sf = 0.0
    q_if_now = q_sf_now = 0.0

    def snap(k, cum_rain):
        psi = prob.psi.x.array
        head_field[k][iz, ix] = psi
        theta_field[k][iz, ix] = soil.theta_of(psi, pcoord[:, 1])
        for kx, xv in enumerate(xu):
            cd = col_of[xv]
            water_table[k, kx] = _water_table(psi[cd], pcoord[cd, 1])
            perch_thk[k, kx] = _perched_extent(psi[cd], pcoord[cd, 1], Z_IFACE)
        ponded[k] = prob.d.x.array[top_d_dofs]
        q_if_t[k] = q_if_now; q_sf_t[k] = q_sf_now
        cum_if_t[k] = cum_if; cum_sf_t[k] = cum_sf
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water()
        cum_out_t[k] = prob.cum_outflow; cum_rain_t[k] = cum_rain
        expected = cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust
        mbe[k] = abs((prob.total_water() - w0) - expected) / (cum_rain + 1e-30)

    print(f"=== [{ls_name}] marching 0 -> {T_END} d (storm {RATE} m/day for {STORM_DUR} d) ===",
          flush=True)
    snap(0, 0.0)
    dt, t_sim, cum_rain, nsteps, k_out = 1e-3, 0.0, 0.0, 0, 1
    tstart = time.perf_counter()
    first_step_s = None
    while t_sim < T_END - 1e-12:
        h = min(dt, T_END - t_sim)
        if t_sim < STORM_DUR - 1e-12 and t_sim + h > STORM_DUR:
            h = STORM_DUR - t_sim
        if k_out < N_OUT and t_sim + h > out_times[k_out]:
            h = out_times[k_out] - t_sim
        rain.value = RATE if t_sim < STORM_DUR - 1e-12 else 0.0
        ts = time.perf_counter()
        conv, it = prob.step(h)
        if first_step_s is None:
            first_step_s = time.perf_counter() - ts
            print(f"  step 1 (compile+solve): {first_step_s:.1f}s converged={conv} iters={it}",
                  flush=True)
        if conv:
            # per-drain split at the SOLVED state -- engine-owned (last_sinks/cum_sinks)
            q_if_now = prob.last_sinks["interior_drain"][0]
            q_sf_now = prob.last_sinks["surface_inlet"][0]
            cum_if = prob.cum_sinks["interior_drain"][0]
            cum_sf = prob.cum_sinks["surface_inlet"][0]
            cum_rain += float(rain.value) * top_len * h
            t_sim += h
            nsteps += 1
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 0.05)
            while k_out < N_OUT and t_sim >= out_times[k_out] - 1e-12:
                snap(k_out, cum_rain)
                k_out += 1
            if nsteps % 25 == 0:
                print(f"  t={t_sim:.4f}/{T_END} dt={dt:.2e} it={it} cum_rain={cum_rain:.4f} "
                      f"cum_out={prob.cum_outflow:.4f} q_iface={q_if_now:.4f} q_surf={q_sf_now:.4f} "
                      f"max_d={prob.surface_depth()*1e3:.2f}mm", flush=True)
        else:
            dt *= 0.5
            if dt < 1e-9:
                print(f"  !! [{ls_name}] DT COLLAPSE at t={t_sim:.6f}", flush=True)
                return None
    while k_out < N_OUT:
        snap(k_out, cum_rain)
        k_out += 1
    rainfall_t = np.where(out_times < STORM_DUR - 1e-12, RATE, 0.0)
    wall = time.perf_counter() - tstart

    return dict(prob=prob, soil=soil, ncell=ncell, ndof=ndof, S0=S0, Lx=Lx, Dz=Dz, NX=NX, NZ=NZ,
                Z_WT=Z_WT, out_times=out_times, head_field=head_field, theta_field=theta_field,
                water_table=water_table, ponded=ponded, perch_thk=perch_thk, q_if_t=q_if_t,
                q_sf_t=q_sf_t, cum_if_t=cum_if_t, cum_sf_t=cum_sf_t, rainfall_t=rainfall_t,
                soil_w=soil_w, surf_w=surf_w, cum_out_t=cum_out_t, cum_rain_t=cum_rain_t, mbe=mbe,
                xu=xu, zu=zu, ix=ix, iz=iz, pcoord=pcoord, w0=w0, cum_rain=cum_rain, wall=wall,
                nsteps=nsteps, first_step_s=first_step_s, t_sim=t_sim, linesearch=ls_name,
                cum_if=cum_if, cum_sf=cum_sf)


def main():
    # bt first; the saturated-wall wetting steps may want cp (memory: opposite-linesearch pairing)
    res = run(None)
    if res is None:
        print("  -> retrying from scratch with snes_linesearch_type='cp'", flush=True)
        res = run("cp")
    if res is None:
        raise SystemExit("both linesearches stalled")
    _finish(**res)


def _finish(prob, soil, ncell, ndof, S0, Lx, Dz, NX, NZ, Z_WT, out_times, head_field, theta_field,
            water_table, ponded, perch_thk, q_if_t, q_sf_t, cum_if_t, cum_sf_t, rainfall_t, soil_w,
            surf_w, cum_out_t, cum_rain_t, mbe, xu, zu, ix, iz, pcoord, w0, cum_rain, wall, nsteps,
            first_step_s, t_sim, linesearch, cum_if, cum_sf):
    done = t_sim >= T_END - 1e-9
    psi_finite = bool(np.all(np.isfinite(prob.psi.x.array)))
    theta_finite = bool(np.all(np.isfinite(soil.theta_of(prob.psi.x.array, pcoord[:, 1]))))
    d_finite = bool(np.all(np.isfinite(prob.d.x.array)))
    bal = (prob.total_water() - w0) - (
        cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust)
    bal_rel = abs(bal) / (cum_rain + 1e-30)
    # split consistency: the per-step split must re-sum to the engine's own cum_drainage
    split_err = abs((cum_if + cum_sf) - prob.cum_drainage) / (prob.cum_drainage + 1e-30)

    # ponding occurrence + the perch contrast at the drain column vs far field
    max_pond = float(ponded.max())
    k_pond, kx_pond = np.unravel_index(int(np.argmax(ponded)), ponded.shape)
    kx10 = int(np.argmin(np.abs(xu - 10.0)))
    kx2 = int(np.argmin(np.abs(xu - 2.0)))
    perch_max = float(perch_thk.max())
    k_pm, kx_pm = np.unravel_index(int(np.argmax(perch_thk)), perch_thk.shape)
    k_end_storm = int(np.argmin(np.abs(out_times - STORM_DUR)))
    perch_drain_es = float(perch_thk[k_end_storm, kx10])
    perch_far_es = float(perch_thk[k_end_storm, kx2])
    wt_drain_final = float(water_table[-1, kx10])
    wt_far_final = float(water_table[-1, kx2])

    print("=== HILLSLOPE DUAL-DRAIN (surface + interface) ILLUSTRATION RESULT ===", flush=True)
    print(f"  linesearch          : {linesearch}", flush=True)
    print(f"  status              : {'REACHED t_end' if done else f'PARTIAL ({t_sim:.4f}/{T_END} d)'}",
          flush=True)
    print(f"  wall clock          : {wall:.1f}s over {nsteps} steps (first {first_step_s:.1f}s)",
          flush=True)
    print(f"  (1) finite          : psi={psi_finite} theta={theta_finite} d={d_finite}", flush=True)
    print(f"  (2) mass balance    : |bal|/cum_rain = {bal_rel:.3e} (gate < 1e-3 -> "
          f"{'PASS' if bal_rel < 1e-3 else 'FAIL'})  max={mbe.max():.3e}; per-drain split re-sum "
          f"err = {split_err:.3e}", flush=True)
    print(f"  (3) INTERFACE tile  : cum = {cum_if:.5f} m^2 (>0 -> {'PASS' if cum_if > 0 else 'FAIL'}); "
          f"peak rate {q_if_t.max():.4f} m^2/day", flush=True)
    print(f"  (4) SURFACE inlet   : cum = {cum_sf:.5f} m^2; peak rate {q_sf_t.max():.4f} m^2/day; "
          f"max ponded depth {max_pond*1e3:.2f} mm at x={xu[kx_pond]:.1f} t={out_times[k_pond]:.2f} d "
          f"-> {'PASS (ponding occurred + inlet captured water)' if (max_pond > 1e-3 and cum_sf > 0) else 'INFO (little/no ponding reached the inlet)'}",
          flush=True)
    print(f"  (5) perch contrast  : max perch {perch_max:.3f} m at x={xu[kx_pm]:.1f} "
          f"t={out_times[k_pm]:.2f} d; end-of-storm thickness at drain col x=10: "
          f"{perch_drain_es:.3f} m vs far-field x=2: {perch_far_es:.3f} m -> "
          f"{'PASS (tile notches the perch locally)' if perch_drain_es < perch_far_es else 'INFO'}",
          flush=True)
    print(f"  water table (final) : drain col {wt_drain_final:.3f} m vs far {wt_far_final:.3f} m", flush=True)
    print(f"  cum_rain={cum_rain:.5f}  cum_outflow={prob.cum_outflow:.5f}  "
          f"cum_drainage={prob.cum_drainage:.5f} (iface {cum_if:.5f} + surface {cum_sf:.5f})  "
          f"clip={prob.clip_mass_adjust:.2e}", flush=True)

    Ks_profile = soil.Ks_of_z(zu)
    ds = xr.Dataset(
        data_vars=dict(
            head_field=(("time", "z", "x"), head_field,
                        {"units": "m", "long_name": "pressure head psi(x,z) (water table = psi=0)"}),
            theta_field=(("time", "z", "x"), theta_field,
                         {"units": "-", "long_name": "volumetric water content theta(x,z) (2-layer)"}),
            water_table=(("time", "x"), water_table,
                         {"units": "m", "long_name": "water-table elevation (z where psi=0; NaN if none)"}),
            ponded_depth=(("time", "x"), ponded,
                          {"units": "m", "long_name": "surface ponded depth d(x) on the top boundary"}),
            perched_thickness=(("time", "x"), perch_thk,
                               {"units": "m", "long_name": "thickness of the perched saturated loam band "
                                                           "on the clay interface (0 = none/full column)"}),
            drain_discharge_iface=(("time",), q_if_t,
                                   {"units": "m^2/day", "long_name": "INTERFACE tile discharge per unit "
                                                                     "width (volumetric DRN band)"}),
            drain_discharge_surface=(("time",), q_sf_t,
                                     {"units": "m^2/day", "long_name": "SURFACE grate-inlet discharge per "
                                                                       "unit width (intake on ponded d)"}),
            cum_drainage_iface=(("time",), cum_if_t,
                                {"units": "m^2", "long_name": "cumulative interface-tile capture"}),
            cum_drainage_surface=(("time",), cum_sf_t,
                                  {"units": "m^2", "long_name": "cumulative surface-inlet capture"}),
            rainfall=(("time",), rainfall_t, {"units": "m/day", "long_name": "rainfall rate"}),
            soil_water=(("time",), soil_w,
                        {"units": "m^2", "long_name": "subsurface stored water per unit width"}),
            surface_water=(("time",), surf_w,
                           {"units": "m^2", "long_name": "surface ponded water per unit width"}),
            cum_outflow=(("time",), cum_out_t,
                         {"units": "m^2", "long_name": "cumulative surface outlet discharge (x=Lx)"}),
            cum_rain=(("time",), cum_rain_t, {"units": "m^2", "long_name": "cumulative rainfall volume"}),
            mass_balance_error=(("time",), mbe,
                                {"units": "-", "long_name": "|(total-w0)-(rain-out-drain+clip)|/cum_rain"}),
            Ks_profile=(("z",), Ks_profile,
                        {"units": "m/day", "long_name": "Ks(z): loam 0.25 above / clay 0.005 below z=1.0"}),
        ),
        coords=dict(
            time=("time", out_times, {"units": "day"}),
            x=("x", xu, {"units": "m", "long_name": "horizontal distance (outlet at x=Lx)"}),
            z=("z", zu, {"units": "m", "long_name": "elevation (0=base, 2=surface)"}),
        ),
        attrs=dict(
            module="m4_hillslope_drain_dual",
            date=DATE,
            scenario=f"2-D hillslope, loam over tight clay (interface z={Z_IFACE} m); storm {RATE} "
                     f"m/day x {STORM_DUR} d then recession to {T_END} d; TWO drains share the "
                     f"x-footprint [{DRAIN_X0},{DRAIN_X1}]: a SURFACE grate inlet on ponded depth and "
                     f"an INTERFACE tile (smooth outflow-only DRN band) on the clay",
            clay_interface_z=float(Z_IFACE),
            domain=f"{Lx} x {Dz} m vertical cross-section (per unit width)",
            mesh=f"{NX}x{NZ} triangles ({ncell} cells, {3*ndof} DOFs)",
            loam_params="theta_r=0.078 theta_s=0.43 alpha=3.6 n=1.56 Ks=0.25",
            clay_params="theta_r=0.068 theta_s=0.38 alpha=0.8 n=1.09 Ks=0.005",
            iface_drain=f"x in [{DRAIN_X0},{DRAIN_X1}], z in [{Z_IFACE},{IFACE_ZTOP}] (one loam "
                        f"cell-row on the clay); q=C*kr*smoothmax(psi+z-{HE_IFACE},0), C_VOL={C_VOL} "
                        f"/day/m (band-integrated ~{C_VOL*0.1:.0f} m^2/day per m head), eps={EPS_POS}",
            surface_drain=f"grate inlet x in [{DRAIN_X0},{DRAIN_X1}] on the land surface; q=C*d, "
                          f"C_SURF={C_SURF} /day (intake of ponded water only)",
            forcing_note=f"storm RAISED vs the base-drain clay run (0.085x1.5d -> {RATE}x{STORM_DUR}d) "
                         f"so saturation-excess ponding occurs and the surface inlet engages; at 0.085 "
                         f"the loam absorbs everything and a surface drain captures exactly 0",
            surface_slope=float(S0),
            initial_water_table_m=float(Z_WT),
            snes_linesearch=linesearch,
            status="REACHED_T_END" if done else f"PARTIAL_{t_sim:.4f}d",
            t_reached_day=float(t_sim),
            wall_clock_s=float(wall),
            steps=int(nsteps),
            mass_balance_error_final=float(bal_rel),
            mass_balance_error_max=float(mbe.max()),
            drain_split_resum_err=float(split_err),
            cum_rain_m2=float(cum_rain),
            cum_outflow_m2=float(prob.cum_outflow),
            cum_drainage_m2=float(prob.cum_drainage),
            cum_drainage_iface_m2=float(cum_if),
            cum_drainage_surface_m2=float(cum_sf),
            max_ponded_depth_m=float(max_pond),
            max_ponded_x_m=float(xu[kx_pond]),
            max_ponded_time_day=float(out_times[k_pond]),
            perched_zone_max_thickness_m=float(perch_max),
            perched_zone_x_m=float(xu[kx_pm]),
            perched_zone_time_day=float(out_times[k_pm]),
            perch_endstorm_draincol_m=float(perch_drain_es),
            perch_endstorm_farfield_m=float(perch_far_es),
            water_table_draincol_final_m=float(wt_drain_final),
            water_table_farfield_final_m=float(wt_far_final),
            base_drain_run_cum_m2=0.054,   # the buried-base-drain clay run, for contrast
            clip_mass_adjust=float(prob.clip_mass_adjust),
            note="ILLUSTRATION on the validated CoupledProblem engine via the FIRST-CLASS "
                 "add_interior_drain/add_surface_inlet APIs (TDD'd, tests/test_engine_drains.py; "
                 "promoted from this run's original script-level injection, git 63145e2) -- the mass "
                 "balance includes both sinks structurally. These are RESOLVED drains, NOT the "
                 "retracted M4 sub-grid embedded feature.",
        ),
    )
    ds.to_netcdf(OUT_NC)
    print(f"WROTE {OUT_NC}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
