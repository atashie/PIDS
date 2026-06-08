"""Emit standardized Tier-3 sanity-run NetCDFs for the Module-3 coupling in 3-D (a HILLSLOPE).

Runs storm + recession scenarios on a tilted 3-D box (the full coupled system: 3-D Richards +
sorptive Kirchhoff infiltration NCP + lateral Manning overland routing + the codim-2 ridge EDGE
outlet + a LATERAL groundwater drainage GHB on the downslope side face) and writes the documented
data contract (governance/visualize-sanity-check-routine.md) so the SEPARATE visualization subagent
(make_coupling_3d_html.py) can build the HTML WITHOUT importing the solver.

Domain: an L x W x H box, bed tilted z_b = S0*(L-x) (slope down toward x=L). Antecedent = a hydro-
static WATER TABLE at z_wt (psi = z_wt - z, so saturated below z_wt). Two routing outlets, both at
the downslope x=L: (i) a SURFACE Manning edge outlet (overland runoff) and (ii) a LATERAL groundwater
GHB on the x=L side FACE (the kr(psi) weight confines it to the saturated zone, so it carries only
groundwater seepage). The texture contrast Arik asked to see:
  - SAND (Ks=7.13 >> rain): rain INFILTRATES, the water table rises and DRAINS LATERALLY (deep
    infiltration + lateral groundwater flow; little/no overland).
  - LOAM (Ks=0.25 < rain): infiltration-EXCESS -> OVERLAND flow to the surface edge (comparison).
Global balance: Delta_total = cum_rain - cum_outflow - cum_drainage + clip_mass_adjust.

Run from forward-model/ with PYTHONPATH=. :
  PYTHONPATH=. python viz/run_coupling_3d_sanity.py            # all scenarios
  PYTHONPATH=. python viz/run_coupling_3d_sanity.py <name>     # one scenario by key
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

DATE = "2026-06-08"
DATA_DIR = "../validation/sanity/data"
# Carsel & Parrish (1988), SI (m, day).
SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
L, W, H = 5.0, 1.0, 1.0
S0 = 0.05      # 5% bed slope toward the x=L downslope outlet
NX, NY, NZ = 16, 6, 8
Z_WT = 0.35    # antecedent water table elevation (m): saturated below, hydrostatic above
H_EXT = 0.20   # downslope groundwater head the lateral GHB drains toward (m)

# Same PERSISTENT storm for both soils; only the texture differs (the comparison Arik asked for).
STORM = dict(rate=0.50, storm_dur=0.30, t_end=0.50)   # 0.5 m/day for 7.2 h, then recession
SCENARIOS = {
    "sand_lateral_gw": dict(soil=SAND, soil_name="sand",
                            label="persistent rain (0.5 m/day, 7.2 h) on slightly-saturated SAND "
                                  "(Ks=7.13, water table z=0.35) -> infiltration + lateral groundwater", **STORM),
    "loam_overland":   dict(soil=LOAM, soil_name="loam",
                            label="SAME storm (0.5 m/day, 7.2 h) on LOAM (Ks=0.25, water table z=0.35) "
                                  "-> infiltration-excess overland flow", **STORM),
}


def _grid_index(coords_xy):
    xr_ = np.round(coords_xy[:, 0], 9); yr_ = np.round(coords_xy[:, 1], 9)
    xu = np.unique(xr_); yu = np.unique(yr_)
    return xu, yu, np.searchsorted(xu, xr_), np.searchsorted(yu, yr_)


def run_scenario(key, spec, *, n_out=55):
    soil = spec["soil"]
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, H]], [NX, NY, NZ])
    prob = CoupledProblem(msh, soil, n_man=0.05)
    prob.set_initial_condition(lambda x: Z_WT - x[2], d_value=0.0)   # hydrostatic water table at Z_WT
    prob.set_topography(lambda x: S0 * (L - x[0]))
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)                 # surface overland edge
    prob.add_drainage_bc(lambda x: np.isclose(x[0], L), conductance=0.5,         # LATERAL groundwater
                         external_head=H_EXT)                                     # (x=L side face; kr-limited)

    topd = prob._top_dofs(prob.Vd)
    xs_u, ys_u, six, siy = _grid_index(prob.Vd.tabulate_dof_coordinates()[topd])
    pcoord = prob.Vpsi.tabulate_dof_coordinates()
    y_levels = np.unique(np.round(pcoord[:, 1], 9))
    y_sec = float(y_levels[np.argmin(np.abs(y_levels - 0.5 * W))])
    sel = np.isclose(pcoord[:, 1], y_sec)
    xc_u = np.unique(np.round(pcoord[sel, 0], 9)); zc_u = np.unique(np.round(pcoord[sel, 2], 9))
    cix = np.searchsorted(xc_u, np.round(pcoord[sel, 0], 9))
    ciz = np.searchsorted(zc_u, np.round(pcoord[sel, 2], 9))

    # top-down theta(x,y) at 3 z-LAYERS (surface / medial / bottom) -- the lateral moisture structure
    z_all = np.unique(np.round(pcoord[:, 2], 9))
    z_layers = np.array([z_all[-1], z_all[np.argmin(np.abs(z_all - 0.5 * H))], z_all[0]])  # surf/mid/base
    layer_sel = [np.isclose(pcoord[:, 2], zl) for zl in z_layers]
    layer_idx = [(_grid_index(pcoord[s][:, :2])[2], _grid_index(pcoord[s][:, :2])[3]) for s in layer_sel]

    top_area = prob.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(prob.mesh, 1.0) * prob._ds_top)), op=MPI.SUM)
    storm_dur, t_end = spec["storm_dur"], spec["t_end"]
    out_times = np.linspace(0.0, t_end, n_out)

    nys, nxs = ys_u.size, xs_u.size
    nzc, nxc = zc_u.size, xc_u.size
    d_map = np.zeros((n_out, nys, nxs))
    theta_xy = np.zeros((n_out, z_layers.size, nys, nxs))   # (time, layer, y, x)
    psi_xs = np.zeros((n_out, nzc, nxc)); th_xs = np.zeros((n_out, nzc, nxc))
    rainfall = np.zeros(n_out); soil_w = np.zeros(n_out); surf_w = np.zeros(n_out)
    outflow = np.zeros(n_out); cum_out = np.zeros(n_out)
    drainage = np.zeros(n_out); cum_dr = np.zeros(n_out)
    cum_rain_t = np.zeros(n_out); mbe = np.zeros(n_out); iters_rec = np.zeros(n_out)

    def snap(k):
        d_map[k][siy, six] = prob.d.x.array[topd]
        for li, (s, (ixl, iyl)) in enumerate(zip(layer_sel, layer_idx)):
            theta_xy[k, li][iyl, ixl] = soil.theta(prob.psi.x.array[s])
        psi_xs[k][ciz, cix] = prob.psi.x.array[sel]
        th_xs[k][ciz, cix] = soil.theta(prob.psi.x.array[sel])
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water()
        outflow[k] = prob.last_outflow; cum_out[k] = prob.cum_outflow
        drainage[k] = prob.last_drainage; cum_dr[k] = prob.cum_drainage

    def hyeto(t):
        return spec["rate"] if t <= storm_dur + 1e-12 else 0.0

    snap(0); w0 = prob.total_water(); cum_rain = 0.0
    dt = 2e-5
    for k in range(1, n_out):
        t, t_target = out_times[k - 1], out_times[k]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            r_now = hyeto(t + h); rain.value = r_now
            converged, it = prob.step(h)
            if converged:
                cum_rain += r_now * top_area * h
                t += h; iters_rec[k] = it
                dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 2e-3)
            else:
                dt *= 0.5
                if dt < 1e-10:
                    raise RuntimeError(f"{key}: dt collapse at t={t:.5g}")
        snap(k); rainfall[k] = hyeto(t_target); cum_rain_t[k] = cum_rain
        expected = cum_rain - prob.cum_outflow - prob.cum_drainage + prob.clip_mass_adjust
        mbe[k] = abs((prob.total_water() - w0) - expected) / (cum_rain + 1e-30)

    ds = xr.Dataset(
        data_vars=dict(
            surface_depth_map=(("time", "y", "x"), d_map,
                               {"units": "m", "long_name": "surface ponding depth d(x,y)"}),
            theta_xy=(("time", "zlayer", "y", "x"), theta_xy,
                      {"units": "-", "long_name": "water content theta(x,y) at the surface/medial/bottom z-layers"}),
            head_xsec=(("time", "z", "xc"), psi_xs,
                       {"units": "m", "long_name": f"pressure head psi on the y={y_sec:.3f} m section "
                                                   "(water table = the psi=0 contour)"}),
            theta_xsec=(("time", "z", "xc"), th_xs,
                        {"units": "-", "long_name": "water content theta on the cross-section"}),
            rainfall=(("time",), rainfall, {"units": "m/day", "long_name": "rainfall intensity"}),
            soil_water=(("time",), soil_w, {"units": "m^3", "long_name": "subsurface stored water (int theta)"}),
            surface_water=(("time",), surf_w, {"units": "m^3", "long_name": "surface stored water (int d)"}),
            outflow=(("time",), outflow, {"units": "m^3/day", "long_name": "surface overland edge discharge"}),
            cum_outflow=(("time",), cum_out, {"units": "m^3", "long_name": "cumulative overland outflow"}),
            drainage=(("time",), drainage,
                      {"units": "m^3/day", "long_name": "lateral groundwater drainage (x=L side, +=out)"}),
            cum_drainage=(("time",), cum_dr, {"units": "m^3", "long_name": "cumulative lateral groundwater outflow"}),
            cum_rain=(("time",), cum_rain_t, {"units": "m^3", "long_name": "cumulative rainfall input"}),
            mass_balance_error=(("time",), mbe, {"units": "-", "long_name": "relative global mass-balance error"}),
            newton_iters=(("time",), iters_rec, {"units": "-", "long_name": "Newton iterations (last step)"}),
        ),
        coords=dict(time=("time", out_times, {"units": "day"}),
                    x=("x", xs_u, {"units": "m", "long_name": "downslope x (outlet at x=L)"}),
                    y=("y", ys_u, {"units": "m", "long_name": "cross-slope y"}),
                    xc=("xc", xc_u, {"units": "m", "long_name": "downslope x (cross-section)"}),
                    z=("z", zc_u, {"units": "m", "long_name": "elevation z (0=base, H=surface)"}),
                    zlayer=("zlayer", z_layers, {"units": "m", "long_name": "z-layer elevation (surface/medial/bottom)"})),
        attrs=dict(
            module="coupling_3d", scenario=spec["label"], date=DATE, soil=spec["soil_name"],
            domain_LxWxH_m=f"{L}x{W}x{H}", mesh_nx_ny_nz=f"{NX}x{NY}x{NZ}", bed_slope=S0,
            y_section_m=y_sec, Ks_m_per_day=soil.Ks, water_table_z0_m=Z_WT, gw_external_head_m=H_EXT,
            rain_rate_m_per_day=spec["rate"], storm_duration_day=storm_dur,
            cum_rain_total_m3=float(cum_rain), peak_surface_depth_m=float(d_map.max()),
            cum_overland_outflow_m3=float(cum_out[-1]), cum_lateral_gw_m3=float(cum_dr[-1]),
            final_subsurface_change_m3=float(soil_w[-1] - soil_w[0]),
            mass_balance_error_max=float(mbe.max()), max_newton_iters=int(iters_rec.max()),
        ),
    )
    out_nc = f"{DATA_DIR}/coupling_3d__{key}__{DATE}.nc"
    ds.to_netcdf(out_nc)
    print(f"WROTE {out_nc}  soil={spec['soil_name']}  peak_d={d_map.max():.4f} m  "
          f"cum_rain={cum_rain:.4f}  overland={cum_out[-1]:.4f}  lateral_gw={cum_dr[-1]:.4f} m^3  "
          f"dSoil={soil_w[-1]-soil_w[0]:+.4f}  max_mbe={mbe.max():.2e}  max_iters={int(iters_rec.max())}",
          flush=True)
    return out_nc


def main(only=None):
    keys = [only] if only else list(SCENARIOS)
    for key in keys:
        run_scenario(key, SCENARIOS[key])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
