"""Module-3 LATERAL-DRAINAGE sweep (2-D coupling): the SAME texture scenarios as run_texture_sweep, now
with a lateral subsurface drainage BC at the downslope toe -- baseline (no drain) vs drained.

Assessment requested 2026-06-07 (Arik), after the subsurface drainage BC + the Codex K(psi)-weighting fix.
Per Arik's constraint: NO base drainage -- the base is IMPERMEABLE (no-flow, the natural BC); subsurface
water leaves ONLY laterally, through a general-head (Cauchy/GHB) boundary on the downslope toe face (x=L,
full height). The flux is relative-permeability-weighted, q_n = C*kr(psi)*(H - H_ext), so it self-limits
to the soil's K-capacity (Codex correction 2026-06-07; see docs/.../subsurface-drainage-bc-spec.md §6).

Lateral boundary parameterization (physically scaled, documented):
  - C = Ks / L  [1/day]  : interface conductance matched to the soil over the hillslope length L (so the
                            lateral flux is the Darcy flux K(psi)*ΔH spread over the slope length).
  - H_ext = 0.0 [m]      : external stream / water table at the BASE elevation (z=0) of the toe. The wet
                            zone (H = psi+z > 0) drains OUT; very dry lower soil (H < 0) can be recharged
                            IN (bidirectional GHB) but kr suppresses it -- a Codex watch-item, reported.

Everything else (hillslope geometry, storm, antecedent head, Manning surface outlet, mesh) is IMPORTED
unchanged from run_texture_sweep so the comparison is apples-to-apples ("same scenarios as before").

Records per soil, for BOTH configs (baseline, drained), the full partition + the drainage hydrograph +
the FULL conservation balance Δtotal = cum_rain − cum_outflow − cum_drainage + clip_mass_adjust, plus the
Codex assessment watch-items (min psi, drainage sign, clip_mass_adjust). Writes one NetCDF per soil with a
`config` dimension to ../validation/sanity/data/ for the separate viz step.

Run from forward-model/ with PYTHONPATH=. :
  PYTHONPATH=. python viz/run_lateral_drainage_sweep.py            # all soils
  PYTHONPATH=. python viz/run_lateral_drainage_sweep.py <soil_key> # one soil by key
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.coupling import CoupledProblem
from viz.run_texture_sweep import (
    SOILS, SOIL_LABEL, ANTECEDENT_PSI, L, H, NX, NZ, S0, RATE, STORM_DUR, T_END, N_MAN,
)

DATE = "2026-06-07"
DATA_DIR = "../validation/sanity/data"
CONFIGS = ["baseline", "drained"]


def _run_one(key, drained, *, n_out=60):
    """One scenario run. drained=False -> texture baseline; True -> add the toe lateral GHB."""
    soil = SOILS[key]
    psi0 = ANTECEDENT_PSI[key]
    C_lat = soil.Ks / L          # interface conductance over the hillslope length
    H_ext = 0.0                  # toe stream / water table at base elevation
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [NX, NZ])
    prob = CoupledProblem(msh, soil, n_man=N_MAN)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)   # surface (Manning) outlet at the toe
    if drained:
        # lateral subsurface GHB on the toe FACE (vertical x=L facets; disjoint from the z=H top surface).
        prob.add_drainage_bc(lambda x: np.isclose(x[0], L), conductance=C_lat, external_head=H_ext)

    coords = prob.Vpsi.tabulate_dof_coordinates()
    colmask = np.isclose(coords[:, 0], L / 2.0)
    zcol = coords[colmask, prob._zaxis]
    zorder = np.argsort(zcol)
    coldofs = np.where(colmask)[0][zorder]
    zs = zcol[zorder]
    nz = zs.size
    topdofs = prob._top_dofs(prob.Vd)
    xtop = prob.Vd.tabulate_dof_coordinates()[topdofs, 0]
    xorder = np.argsort(xtop)
    xs = xtop[xorder]
    nx = xs.size
    one_top = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * prob._ds_top)), op=MPI.SUM)

    out_times = np.linspace(0.0, T_END, n_out)
    psi_zt = np.zeros((n_out, nz)); theta_zt = np.zeros((n_out, nz)); d_xt = np.zeros((n_out, nx))
    outflow_q = np.zeros(n_out); cum_outflow_t = np.zeros(n_out)
    drain_q = np.zeros(n_out); cum_drain_t = np.zeros(n_out)
    soil_w = np.zeros(n_out); surf_w = np.zeros(n_out); tot_w = np.zeros(n_out)
    cum_rain_t = np.zeros(n_out); rainfall = np.zeros(n_out); mbe = np.zeros(n_out)
    min_psi = np.zeros(n_out); clip_t = np.zeros(n_out); iters_rec = np.zeros(n_out)

    def hyeto(t):
        return RATE if t <= STORM_DUR + 1e-12 else 0.0

    psi_zt[0] = prob.psi.x.array[coldofs]; theta_zt[0] = soil.theta(prob.psi.x.array[coldofs])
    d_xt[0] = prob.d.x.array[topdofs][xorder]
    soil_w[0] = prob.soil_water(); surf_w[0] = prob.surface_water(); tot_w[0] = prob.total_water()
    min_psi[0] = float(prob.psi.x.array.min())
    w0 = tot_w[0]
    cum_rain = 0.0
    dt = 1e-6
    last_iters = 0

    def safe_step(h):
        try:
            return prob.step(h)
        except Exception:
            prob.psi.x.array[:] = prob.psi_n.x.array; prob.psi.x.scatter_forward()
            prob.d.x.array[:] = prob.d_n.x.array; prob.d.x.scatter_forward()
            prob.lam.x.array[:] = 0.0; prob.lam.x.scatter_forward()
            prob._problem = None
            return False, 0

    for k in range(1, n_out):
        t = out_times[k - 1]; t_target = out_times[k]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            r_now = hyeto(t + h)
            rain.value = r_now
            converged, it = safe_step(h)
            if converged:
                cum_rain += r_now * one_top * h
                t += h
                last_iters = it
                dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 5e-4)
            else:
                dt *= 0.5
                if dt < 1e-12:
                    raise RuntimeError(f"{key}/{('drained' if drained else 'baseline')}: dt collapse t={t:.5g}")
        psi_zt[k] = prob.psi.x.array[coldofs]; theta_zt[k] = soil.theta(prob.psi.x.array[coldofs])
        d_xt[k] = prob.d.x.array[topdofs][xorder]
        outflow_q[k] = prob.outflow_rate(); cum_outflow_t[k] = prob.cum_outflow
        drain_q[k] = prob.drainage_rate(); cum_drain_t[k] = prob.cum_drainage
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water(); tot_w[k] = prob.total_water()
        cum_rain_t[k] = cum_rain; rainfall[k] = hyeto(t_target)
        min_psi[k] = float(prob.psi.x.array.min()); clip_t[k] = prob.clip_mass_adjust
        # FULL balance (Codex 2026-06-07): Δtotal = cum_rain − cum_outflow − cum_drainage + clip_mass_adjust
        resid = (tot_w[k] - w0) - (cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust)
        mbe[k] = abs(resid) / (cum_rain + 1e-30)
        iters_rec[k] = last_iters

    return dict(
        zs=zs, xs=xs, out_times=out_times, one_top=float(one_top), C_lat=float(C_lat), H_ext=float(H_ext),
        psi_zt=psi_zt, theta_zt=theta_zt, d_xt=d_xt, outflow_q=outflow_q, cum_outflow_t=cum_outflow_t,
        drain_q=drain_q, cum_drain_t=cum_drain_t, soil_w=soil_w, surf_w=surf_w, tot_w=tot_w,
        cum_rain_t=cum_rain_t, rainfall=rainfall, mbe=mbe, min_psi=min_psi, clip_t=clip_t,
        iters_rec=iters_rec, cum_rain=float(cum_rain),
        infiltrated=float(soil_w[-1] - soil_w[0]), ponded=float(surf_w[-1]),
        drained_surf=float(cum_outflow_t[-1]), drained_lat=float(cum_drain_t[-1]),
    )


def run_soil(key):
    base = _run_one(key, drained=False)
    drn = _run_one(key, drained=True)
    soil = SOILS[key]
    cr = drn["cum_rain"] + 1e-30
    # stack configs along a new leading axis
    def st(name):
        return np.stack([base[name], drn[name]], axis=0)

    ds = xr.Dataset(
        data_vars=dict(
            head=(("config", "time", "z"), st("psi_zt"), {"units": "m", "long_name": "mid-column psi(z)"}),
            water_content=(("config", "time", "z"), st("theta_zt"), {"units": "-", "long_name": "theta(z)"}),
            ponding_depth=(("config", "time", "x"), st("d_xt"), {"units": "m", "long_name": "ponding d(x,t)"}),
            outflow=(("config", "time"), st("outflow_q"), {"units": "m2/day", "long_name": "surface outlet q"}),
            cum_outflow=(("config", "time"), st("cum_outflow_t"), {"units": "m2", "long_name": "cum surface outflow"}),
            drainage=(("config", "time"), st("drain_q"), {"units": "m2/day", "long_name": "lateral drainage rate (+out)"}),
            cum_drainage=(("config", "time"), st("cum_drain_t"), {"units": "m2", "long_name": "cum lateral drainage"}),
            soil_water=(("config", "time"), st("soil_w"), {"units": "m2", "long_name": "subsurface stored water"}),
            surface_water=(("config", "time"), st("surf_w"), {"units": "m2", "long_name": "surface stored water"}),
            total_water=(("config", "time"), st("tot_w"), {"units": "m2", "long_name": "total stored water"}),
            cum_rain=(("config", "time"), st("cum_rain_t"), {"units": "m2", "long_name": "cumulative rainfall"}),
            rainfall=(("config", "time"), st("rainfall"), {"units": "m/day", "long_name": "rainfall intensity"}),
            mass_balance_error=(("config", "time"), st("mbe"), {"units": "-", "long_name": "|resid|/rain (full balance)"}),
            min_head=(("config", "time"), st("min_psi"), {"units": "m", "long_name": "min psi over domain (dryness)"}),
            clip_mass_adjust=(("config", "time"), st("clip_t"), {"units": "m2", "long_name": "limiter degenerate-branch mass"}),
            newton_iters=(("config", "time"), st("iters_rec"), {"units": "-", "long_name": "Newton iters (last step)"}),
        ),
        coords=dict(
            config=("config", CONFIGS),
            time=("time", base["out_times"], {"units": "day"}),
            z=("z", base["zs"], {"units": "m", "long_name": "elevation (0=base, top=surface)"}),
            x=("x", base["xs"], {"units": "m", "long_name": "horizontal position (toe at x=L)"}),
        ),
        attrs=dict(
            module="coupling_lateral_drainage", soil_key=key, scenario=SOIL_LABEL[key], date=DATE,
            theta_r=soil.theta_r, theta_s=soil.theta_s, vg_alpha=soil.alpha, vg_n=soil.n, Ks_m_per_day=soil.Ks,
            L=L, H=H, nx=NX, nz=NZ, bed_slope_S0=S0, outlet_slope=S0, antecedent_psi0_m=ANTECEDENT_PSI[key],
            rain_rate_m_per_day=RATE, storm_duration_day=STORM_DUR, t_end_day=T_END, n_manning=N_MAN,
            top_length_m=drn["one_top"], base_bc="impermeable (no-flow)",
            lateral_bc="toe-face GHB x=L, q_n=C*kr(psi)*(psi+z-H_ext)",
            lateral_conductance_C_per_day=drn["C_lat"], lateral_external_head_H_ext_m=drn["H_ext"],
            cum_rain_total_m2=drn["cum_rain"],
            # partition (drained config)
            drained_infiltrated_m2=drn["infiltrated"], drained_ponded_m2=drn["ponded"],
            drained_surface_out_m2=drn["drained_surf"], drained_lateral_out_m2=drn["drained_lat"],
            drained_part_infiltrated_frac=float(drn["infiltrated"] / cr),
            drained_part_ponded_frac=float(drn["ponded"] / cr),
            drained_part_surface_out_frac=float(drn["drained_surf"] / cr),
            drained_part_lateral_out_frac=float(drn["drained_lat"] / cr),
            # partition (baseline config) -- no lateral drain
            baseline_infiltrated_m2=base["infiltrated"], baseline_ponded_m2=base["ponded"],
            baseline_surface_out_m2=base["drained_surf"],
            baseline_part_infiltrated_frac=float(base["infiltrated"] / (base["cum_rain"] + 1e-30)),
            baseline_part_ponded_frac=float(base["ponded"] / (base["cum_rain"] + 1e-30)),
            baseline_part_surface_out_frac=float(base["drained_surf"] / (base["cum_rain"] + 1e-30)),
            # watch-items
            drained_min_psi_m=float(drn["min_psi"].min()), drained_min_drainage_m2_per_day=float(drn["drain_q"].min()),
            drained_clip_mass_adjust_final_m2=float(drn["clip_t"][-1]),
            mass_balance_error_max=float(max(base["mbe"].max(), drn["mbe"].max())),
            max_newton_iters=int(max(base["iters_rec"].max(), drn["iters_rec"].max())),
        ),
    )
    out_nc = f"{DATA_DIR}/coupling_lateraldrain__{key}__{DATE}.nc"
    ds.to_netcdf(out_nc)
    print(f"WROTE {out_nc}\n"
          f"  baseline: infil={base['infiltrated']:.4f} pond={base['ponded']:.4f} surfout={base['drained_surf']:.4f}\n"
          f"  drained : infil={drn['infiltrated']:.4f} pond={drn['ponded']:.4f} surfout={drn['drained_surf']:.4f} "
          f"latout={drn['drained_lat']:.4f}  (cum_rain={drn['cum_rain']:.4f})\n"
          f"  C_lat={drn['C_lat']:.4g} H_ext={drn['H_ext']:.2f}  min_psi={drn['min_psi'].min():.3f} "
          f"min_drain={drn['drain_q'].min():.3e}  max_mbe={max(base['mbe'].max(), drn['mbe'].max()):.2e} "
          f"max_iters={int(max(base['iters_rec'].max(), drn['iters_rec'].max()))}", flush=True)
    return out_nc


def main(only=None):
    for key in SOILS:
        if only and key != only:
            continue
        run_soil(key)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
