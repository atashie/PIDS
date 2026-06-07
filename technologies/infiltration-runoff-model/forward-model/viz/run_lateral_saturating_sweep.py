"""Module-3 LATERAL-DRAINAGE assessment under a SATURATING storm, with a C / H_ext sensitivity sweep.

Requested 2026-06-07 (Arik): the short Hortonian texture-sweep storms never saturate the soil MATRIX (the
runoff there is surface ponding -> Manning outlet; the matrix stays unsaturated), so the lateral
subsurface drain stays idle. To exercise the drain in its real regime -- SATURATION-EXCESS, the PIDS
value-proposition where a lateral channel/tunnel removes subsurface water to prevent saturation/runoff --
this drives the soil to saturation, then sweeps the toe lateral GHB conductance C and external head H_ext.

Scenario (fixed; the SAME hillslope + textures as run_texture_sweep, only the forcing/antecedent change to
make it saturation-driven):
  - antecedent: a hydrostatic WATER TABLE at z = Z_WT (psi0(z) = Z_WT - z) -> the lower profile is already
    SATURATED (psi>0), the thin upper cap unsaturated. Realistic wet initial state with excess to drain.
  - rain: a sustained LOW intensity RATE_SAT < every texture's Ks (so it INFILTRATES rather than running
    off the surface -- saturation-excess, not Hortonian), for STORM_DUR_SAT, then recession to T_END_SAT.
  - base: IMPERMEABLE (no-flow). Subsurface water leaves ONLY laterally, at the downslope toe face (x=L).
  - lateral drain: GHB q_n = C*kr(psi)*(psi+z-H_ext) (relative-perm weighted; Codex 2026-06-07).

Sweep (per texture): a 'baseline' (no lateral drain) plus a set of (C, H_ext) CASES -- a C sweep at fixed
H_ext (weak->strong drain; maps the self-limiting response) crossed with an H_ext sweep at fixed C
(shallow toe stream -> deep tunnel). Writes one NetCDF per texture with a `case` dimension for the viz.

Run from forward-model/ with PYTHONPATH=. :
  PYTHONPATH=. python viz/run_lateral_saturating_sweep.py                 # all textures, all cases
  PYTHONPATH=. python viz/run_lateral_saturating_sweep.py <soil_key>      # one texture
  PYTHONPATH=. python viz/run_lateral_saturating_sweep.py <soil_key> <n>  # one texture, first n cases (probe)
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.coupling import CoupledProblem
from viz.run_texture_sweep import SOILS, SOIL_LABEL, L, H, NX, NZ, S0, N_MAN

DATE = "2026-06-07"
DATA_DIR = "../validation/sanity/data"

# --- saturating forcing (replaces the Hortonian texture-sweep storm) ----------------------------------
Z_WT = 0.7            # antecedent water-table elevation (m): lower 70% saturated, upper 30% unsat cap
RATE_SAT = 0.04       # sustained rain (m/day) -- BELOW clay Ks (0.048) so ALL textures infiltrate
STORM_DUR_SAT = 0.4   # sustained-rain duration (day)
T_END_SAT = 0.6       # + recession so the lateral drainage recession is visible
DRAIN_RAMP = 0.02     # ramp the drain conductance 0 -> C over this (day) to avoid a cold-start shock on
                      # near-saturated STIFF soil (clay/silt) -- a numerical aid; fully accounted for in
                      # the balance (drainage_rate uses the live conductance each step). Negligible vs T_END.

# --- sweep cases: (label, C [1/day], H_ext [m]); C=None -> baseline (no lateral drain) -----------------
# C sweep at H_ext=-0.5 (weak/moderate/strong) crossed with an H_ext sweep at C=0.5 (toe stream / tunnel).
CASES = [
    ("baseline",       None, None),
    ("C0.1_He-0.5",    0.1, -0.5),
    ("C0.5_He-0.5",    0.5, -0.5),
    ("C2.0_He-0.5",    2.0, -0.5),
    ("C0.5_He0.0",     0.5,  0.0),
    ("C0.5_He-1.5",    0.5, -1.5),
]


def _run_case(key, C, H_ext, *, n_out=50):
    soil = SOILS[key]
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [NX, NZ])
    prob = CoupledProblem(msh, soil, n_man=N_MAN)
    prob.set_initial_condition(lambda x: Z_WT - x[prob._zaxis], d_value=0.0)   # hydrostatic water table
    prob.set_topography(lambda x: S0 * (L - x[0]))
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)
    Cdr = None
    if C is not None:   # toe lateral GHB; conductance ramps 0 -> C (Constant) to ease the cold start
        Cdr = prob.add_drainage_bc(lambda x: np.isclose(x[0], L),
                                   conductance=fem.Constant(msh, PETSc.ScalarType(0.0)), external_head=H_ext)

    coords = prob.Vpsi.tabulate_dof_coordinates()
    colmask = np.isclose(coords[:, 0], L / 2.0)
    zcol = coords[colmask, prob._zaxis]; zorder = np.argsort(zcol)
    coldofs = np.where(colmask)[0][zorder]; zs = zcol[zorder]; nz = zs.size
    topdofs = prob._top_dofs(prob.Vd)
    xtop = prob.Vd.tabulate_dof_coordinates()[topdofs, 0]; xorder = np.argsort(xtop)
    xs = xtop[xorder]; nx = xs.size
    one_top = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * prob._ds_top)), op=MPI.SUM)

    out_times = np.linspace(0.0, T_END_SAT, n_out)
    psi_zt = np.zeros((n_out, nz)); theta_zt = np.zeros((n_out, nz)); d_xt = np.zeros((n_out, nx))
    outflow_q = np.zeros(n_out); cum_outflow_t = np.zeros(n_out)
    drain_q = np.zeros(n_out); cum_drain_t = np.zeros(n_out)
    soil_w = np.zeros(n_out); surf_w = np.zeros(n_out); tot_w = np.zeros(n_out)
    cum_rain_t = np.zeros(n_out); rainfall = np.zeros(n_out); mbe = np.zeros(n_out)
    sat_frac = np.zeros(n_out); wt_mid = np.zeros(n_out)
    min_psi = np.zeros(n_out); clip_t = np.zeros(n_out); iters_rec = np.zeros(n_out)

    def hyeto(t):
        return RATE_SAT if t <= STORM_DUR_SAT + 1e-12 else 0.0

    def sat_fraction():
        loc = float(np.count_nonzero(prob.psi.x.array >= 0.0)); n = float(prob.psi.x.array.size)
        return msh.comm.allreduce(loc, op=MPI.SUM) / max(msh.comm.allreduce(n, op=MPI.SUM), 1.0)

    def wt_in_midcol():
        p = prob.psi.x.array[coldofs]   # psi up the mid column (z ascending)
        sat = np.where(p >= 0.0)[0]
        return float(zs[sat[-1]]) if sat.size else 0.0   # highest saturated elevation (water-table proxy)

    psi_zt[0] = prob.psi.x.array[coldofs]; theta_zt[0] = soil.theta(prob.psi.x.array[coldofs])
    d_xt[0] = prob.d.x.array[topdofs][xorder]
    soil_w[0] = prob.soil_water(); surf_w[0] = prob.surface_water(); tot_w[0] = prob.total_water()
    sat_frac[0] = sat_fraction(); wt_mid[0] = wt_in_midcol(); min_psi[0] = float(prob.psi.x.array.min())
    w0 = tot_w[0]; cum_rain = 0.0; dt = 1e-6; last_iters = 0

    def safe_step(h):
        try:
            return prob.step(h)
        except Exception:
            prob.psi.x.array[:] = prob.psi_n.x.array; prob.psi.x.scatter_forward()
            prob.d.x.array[:] = prob.d_n.x.array; prob.d.x.scatter_forward()
            prob.lam.x.array[:] = 0.0; prob.lam.x.scatter_forward(); prob._problem = None
            return False, 0

    for k in range(1, n_out):
        t = out_times[k - 1]; t_target = out_times[k]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t); r_now = hyeto(t + h); rain.value = r_now
            if Cdr is not None:   # ramp the drain conductance on over DRAIN_RAMP
                Cdr.value = C * min(1.0, (t + h) / DRAIN_RAMP)
            converged, it = safe_step(h)
            if converged:
                cum_rain += r_now * one_top * h; t += h; last_iters = it
                dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 2e-3)
            else:
                dt *= 0.5
                if dt < 1e-12:
                    raise RuntimeError(f"{key}/C={C}/He={H_ext}: dt collapse t={t:.5g}")
        psi_zt[k] = prob.psi.x.array[coldofs]; theta_zt[k] = soil.theta(prob.psi.x.array[coldofs])
        d_xt[k] = prob.d.x.array[topdofs][xorder]
        outflow_q[k] = prob.outflow_rate(); cum_outflow_t[k] = prob.cum_outflow
        drain_q[k] = prob.drainage_rate(); cum_drain_t[k] = prob.cum_drainage
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water(); tot_w[k] = prob.total_water()
        cum_rain_t[k] = cum_rain; rainfall[k] = hyeto(t_target)
        sat_frac[k] = sat_fraction(); wt_mid[k] = wt_in_midcol()
        min_psi[k] = float(prob.psi.x.array.min()); clip_t[k] = prob.clip_mass_adjust
        resid = (tot_w[k] - w0) - (cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust)
        mbe[k] = abs(resid) / (cum_rain + 1e-30); iters_rec[k] = last_iters

    return dict(
        zs=zs, xs=xs, out_times=out_times, one_top=float(one_top),
        psi_zt=psi_zt, theta_zt=theta_zt, d_xt=d_xt, outflow_q=outflow_q, cum_outflow_t=cum_outflow_t,
        drain_q=drain_q, cum_drain_t=cum_drain_t, soil_w=soil_w, surf_w=surf_w, tot_w=tot_w,
        cum_rain_t=cum_rain_t, rainfall=rainfall, mbe=mbe, sat_frac=sat_frac, wt_mid=wt_mid,
        min_psi=min_psi, clip_t=clip_t, iters_rec=iters_rec, cum_rain=float(cum_rain),
        infiltrated=float(soil_w[-1] - soil_w[0]), ponded=float(surf_w[-1]),
        surf_out=float(cum_outflow_t[-1]), lat_out=float(cum_drain_t[-1]),
    )


def run_soil(key, cases=None):
    cases = cases or CASES
    soil = SOILS[key]
    res = []
    for label, C, He in cases:
        r = _run_case(key, C, He)
        r["label"] = label; r["C"] = (float(C) if C is not None else np.nan)
        r["He"] = (float(He) if He is not None else np.nan); r["is_base"] = (C is None)
        res.append(r)
        cr = r["cum_rain"] + 1e-30
        print(f"  [{key}/{label}] dSoilW={r['infiltrated']:.4f} pond={r['ponded']:.4f} "
              f"surfout={r['surf_out']:.4f} latout={r['lat_out']:.4f} ({r['lat_out']/cr*100:.1f}% rain) "
              f"sat:{r['sat_frac'][0]*100:.0f}->{r['sat_frac'][-1]*100:.0f}% wt:{r['wt_mid'][0]:.2f}->"
              f"{r['wt_mid'][-1]:.2f} maxq={r['outflow_q'].max():.3e} mbe={r['mbe'].max():.1e} "
              f"it={int(r['iters_rec'].max())}", flush=True)

    labels = [r["label"] for r in res]

    def st(name):
        return np.stack([r[name] for r in res], axis=0)

    ds = xr.Dataset(
        data_vars=dict(
            head=(("case", "time", "z"), st("psi_zt"), {"units": "m", "long_name": "mid-column psi(z)"}),
            water_content=(("case", "time", "z"), st("theta_zt"), {"units": "-"}),
            ponding_depth=(("case", "time", "x"), st("d_xt"), {"units": "m"}),
            outflow=(("case", "time"), st("outflow_q"), {"units": "m2/day", "long_name": "surface outlet q"}),
            cum_outflow=(("case", "time"), st("cum_outflow_t"), {"units": "m2"}),
            drainage=(("case", "time"), st("drain_q"), {"units": "m2/day", "long_name": "lateral drainage (+out)"}),
            cum_drainage=(("case", "time"), st("cum_drain_t"), {"units": "m2"}),
            soil_water=(("case", "time"), st("soil_w"), {"units": "m2"}),
            surface_water=(("case", "time"), st("surf_w"), {"units": "m2"}),
            total_water=(("case", "time"), st("tot_w"), {"units": "m2"}),
            cum_rain=(("case", "time"), st("cum_rain_t"), {"units": "m2"}),
            rainfall=(("case", "time"), st("rainfall"), {"units": "m/day"}),
            mass_balance_error=(("case", "time"), st("mbe"), {"units": "-", "long_name": "|resid|/rain (full balance)"}),
            sat_fraction=(("case", "time"), st("sat_frac"), {"units": "-", "long_name": "fraction of nodes psi>=0"}),
            water_table=(("case", "time"), st("wt_mid"), {"units": "m", "long_name": "mid-column water-table elevation"}),
            min_head=(("case", "time"), st("min_psi"), {"units": "m"}),
            clip_mass_adjust=(("case", "time"), st("clip_t"), {"units": "m2"}),
            newton_iters=(("case", "time"), st("iters_rec"), {"units": "-"}),
            case_C=(("case",), np.array([r["C"] for r in res]), {"units": "1/day"}),
            case_He=(("case",), np.array([r["He"] for r in res]), {"units": "m"}),
        ),
        coords=dict(
            case=("case", labels),
            time=("time", res[0]["out_times"], {"units": "day"}),
            z=("z", res[0]["zs"], {"units": "m", "long_name": "elevation (0=base, top=surface)"}),
            x=("x", res[0]["xs"], {"units": "m", "long_name": "horizontal (toe at x=L)"}),
        ),
        attrs=dict(
            module="coupling_lateral_saturating", soil_key=key, scenario=SOIL_LABEL[key], date=DATE,
            theta_r=soil.theta_r, theta_s=soil.theta_s, vg_alpha=soil.alpha, vg_n=soil.n, Ks_m_per_day=soil.Ks,
            L=L, H=H, nx=NX, nz=NZ, bed_slope_S0=S0, n_manning=N_MAN, base_bc="impermeable (no-flow)",
            antecedent_water_table_m=Z_WT, rain_rate_m_per_day=RATE_SAT, storm_duration_day=STORM_DUR_SAT,
            t_end_day=T_END_SAT, top_length_m=res[0]["one_top"], cum_rain_total_m2=res[0]["cum_rain"],
            lateral_bc="toe-face GHB x=L, q_n=C*kr(psi)*(psi+z-H_ext)",
            cases=";".join(labels),
            baseline_surf_out_m2=float(res[0]["surf_out"]), baseline_sat_final=float(res[0]["sat_frac"][-1]),
            mass_balance_error_max=float(max(r["mbe"].max() for r in res)),
            max_newton_iters=int(max(r["iters_rec"].max() for r in res)),
            min_drainage_any=float(min(r["drain_q"].min() for r in res)),
        ),
    )
    out_nc = f"{DATA_DIR}/coupling_latsat__{key}__{DATE}.nc"
    ds.to_netcdf(out_nc)
    print(f"WROTE {out_nc}", flush=True)
    return out_nc


def main(only=None, ncase=None):
    cases = CASES[:ncase] if ncase else CASES
    for key in SOILS:
        if only and key != only:
            continue
        run_soil(key, cases)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ncase = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(only, ncase)
