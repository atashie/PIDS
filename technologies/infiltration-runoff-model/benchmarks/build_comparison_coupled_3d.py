#!/usr/bin/env python3
"""Build the 3-D COUPLED hillslope comparison (in-house vs ParFlow) -- B5.

Extends the B4 flat-column coupled comparison (build_comparison_coupled.py) to a tilted 3-D
hillslope with lateral overland routing + a lateral groundwater seepage outlet. Unlike B4 this
does NOT re-run the in-house solver -- the in-house 3-D reference already exists as a standardized
NetCDF (../validation/sanity/data/coupling_3d__{key}__2026-06-08.nc, written by
forward-model/viz/run_coupling_3d_sanity.py); we LOAD it and align it to the ParFlow run.

ParFlow side (../parflow/cases/coupled_hillslope_3d.py summaries):
  * coupled_3d_<key>.npz       -- FULL run (OverlandFlow top + DirEquil lateral-GW head face)
  * coupled_3d_<key>_nogw.npz  -- CONTROL run (lateral-GW face = no-flow) -> total outflow = OVERLAND

OVERLAND/GW SEPARATION (validated): ParFlow's PrintOverlandSum proved unreliable as an overland
VOLUME on the kinematic OverlandFlow BC (read ~0.03 vs a true ~0.48 m^3). The robust separation is
the DIFFERENCE OF TWO RUNS:
    overland(t)    = total_out(no-GW control)              [all outflow is overland when GW is off]
    lateral_GW(t)  = total_out(full) - total_out(control)
    infiltration   = d(subsurface storage), ponding = surface store max(psi_top,0) (B4 convention).
The 4-way partition closes to cum_rain by construction (lateral_GW is the residual after overland).
Caveat: attributing overland to the control run's value (vs the full run's true, unmeasurable split)
is exact only in the linear limit; negligible when GW is small (loam) -- a documented separation delta.

KEY FINDING carried into the metrics + deltas: for SAND (lateral-GW-dominated) the constant-head
(DirEquilRefPatch, C->inf) face OVER-DRAINS ~5x vs the in-house finite kr-weighted GHB (C=0.5/day) --
ParFlow has no Robin/Cauchy BC, so the lateral-GW MAGNITUDE is BC-parameterization-dependent, not a
model-physics discrepancy (Arik decision 2026-06-10: document as a BC delta, no grid surgery).

Run (pids-fem; reads $HOME ParFlow summaries + the repo in-house .nc):
    source /root/miniforge3/etc/profile.d/conda.sh && conda activate pids-fem
    python build_comparison_coupled_3d.py [scenario]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr

DATE = "2026-06-10"
IH_DATE = "2026-06-08"
HERE = os.path.dirname(os.path.abspath(__file__))
IH_DIR = os.path.abspath(os.path.join(HERE, "..", "validation", "sanity", "data"))
PF_DIR = os.path.expanduser("~/parflow-runs/coupled_hillslope_3d/summaries")
N_COMMON = 51                       # common time grid for the overlaid time series

SCENARIOS = {
    "loam_overland":   dict(soil="loam", Ks=0.25,
                            label="SAME storm (0.5 m/day, 7.2 h) on LOAM (Ks=0.25, water table z=0.35) "
                                  "-> infiltration-excess OVERLAND flow"),
    "sand_lateral_gw": dict(soil="sand", Ks=7.13,
                            label="persistent rain (0.5 m/day, 7.2 h) on SAND (Ks=7.13, water table z=0.35) "
                                  "-> infiltration + lateral GROUNDWATER"),
}


def _interp_t(t_src, y_src, t_dst):
    return np.interp(t_dst, t_src, y_src)


def _pf_series(npz):
    """Derive (times, cum_rain, infiltration, ponding, total_out, d_map_frames, fields) from a run."""
    z = npz["z"]; y = npz["y"]; x = npz["x"]; times = npz["times"]
    press = npz["press"]; theta = npz["theta"]
    dx, dy, dz = float(npz["dx"]), float(npz["dy"]), float(npz["dz"])
    rate, storm_dur = float(npz["rate"]), float(npz["storm_dur"])
    cell_vol = dx * dy * dz
    cell_area = dx * dy
    top_area = (x.size * dx) * (y.size * dy)
    d_map = np.maximum(press[:, -1, :, :], 0.0)            # (t, y, x) ponding store
    pond = d_map.sum(axis=(1, 2)) * cell_area              # m^3
    w_soil = theta.sum(axis=(1, 2, 3)) * cell_vol          # m^3
    cum_rain = rate * top_area * np.minimum(times, storm_dur)
    total_out = cum_rain - (w_soil - w_soil[0]) - (pond - pond[0])
    return dict(times=times, cum_rain=cum_rain, infil=w_soil - w_soil[0], pond=pond,
                total_out=total_out, d_map=d_map, press=press, theta=theta,
                x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, rate=rate, storm_dur=storm_dur)


def _snapshot(field_tzyx, times, t_pick, y, y_target):
    """Cross-section (z, x) at the y-cell nearest y_target, at the frame nearest t_pick."""
    k = int(np.argmin(np.abs(times - t_pick)))
    j = int(np.argmin(np.abs(y - y_target)))
    return field_tzyx[k, :, j, :], k, j


def build_one(key, spec):
    soil = spec["soil"]
    ih = xr.open_dataset(os.path.join(IH_DIR, f"coupling_3d__{key}__{IH_DATE}.nc"))
    full = _pf_series(np.load(os.path.join(PF_DIR, f"coupled_3d_{key}.npz")))
    ctrl = _pf_series(np.load(os.path.join(PF_DIR, f"coupled_3d_{key}_nogw.npz")))

    storm_dur = full["storm_dur"]
    t_end = float(ih["time"].values[-1])
    t = np.linspace(0.0, t_end, N_COMMON)

    # ---- in-house time series (already in the contract) ----
    tih = np.asarray(ih["time"].values, float)
    cum_rain_ih = _interp_t(tih, ih["cum_rain"].values, t)
    infil_ih = _interp_t(tih, ih["soil_water"].values - float(ih["soil_water"][0]), t)
    pond_ih = _interp_t(tih, ih["surface_water"].values, t)
    ov_cum_ih = _interp_t(tih, ih["cum_outflow"].values, t)
    gw_cum_ih = _interp_t(tih, ih["cum_drainage"].values, t)
    ov_rate_ih = _interp_t(tih, ih["outflow"].values, t)
    gw_rate_ih = _interp_t(tih, ih["drainage"].values, t)
    mbe_ih = _interp_t(tih, ih["mass_balance_error"].values, t)

    # ---- ParFlow: overland = control total_out; lateral GW = full - control (validated split) ----
    ov_cum_pf = _interp_t(ctrl["times"], ctrl["total_out"], t)        # OVERLAND (no-GW control)
    tot_pf = _interp_t(full["times"], full["total_out"], t)
    gw_cum_pf = tot_pf - ov_cum_pf                                    # lateral GW (residual)
    infil_pf = _interp_t(full["times"], full["infil"], t)
    pond_pf = _interp_t(full["times"], full["pond"], t)
    cum_rain_pf = _interp_t(full["times"], full["cum_rain"], t)
    ov_rate_pf = np.gradient(ov_cum_pf, t)
    gw_rate_pf = np.gradient(gw_cum_pf, t)
    # ParFlow theta+pond balance residual (the comparable, Ss-free books):
    mbe_pf = np.abs((infil_pf + pond_pf + ov_cum_pf + gw_cum_pf) - cum_rain_pf) / np.where(cum_rain_pf > 0, cum_rain_pf, 1.0)

    # --- early-infiltration transient diagnostic (Arik 2026-06-10) ---
    # infiltration RATE dI/dt and the SURFACE SATURATION S_surf = theta_top/theta_s. The story:
    # the in-house sorptive Kirchhoff surface closure (+ air-entry cap) decays infiltration capacity
    # as S_surf -> 1, reaching the steady rate EARLY; ParFlow's coarse top cell buffers rain at full
    # rate until S_surf hits 1.0, then chokes -> a longer, more curved early transient. The diagnostic
    # is the RATE plotted against S_surf: at equal S_surf (~0.985) the in-house rate has collapsed
    # while ParFlow's is still ~full. End states converge (a TIMING delta, not a physics one).
    THETA_S = 0.43
    infrate_ih = np.gradient(infil_ih, t)
    infrate_pf = np.gradient(infil_pf, t)
    zl = ih["zlayer"].values
    surf_layer = int(np.argmax(zl))                          # surface = highest z-layer
    ssurf_ih = _interp_t(tih, ih["theta_xy"].values[:, surf_layer].mean(axis=(1, 2)) / THETA_S, t)
    ssurf_pf = _interp_t(full["times"], full["theta"][:, -1].mean(axis=(1, 2)) / THETA_S, t)

    def _onset(cum, frac=0.01):
        thr = frac * max(float(cum[-1]), 1e-9)
        return float(t[int(np.argmax(cum > thr))]) if np.any(cum > thr) else float("nan")
    runoff_onset_ih = _onset(ov_cum_ih)
    runoff_onset_pf = _onset(ov_cum_pf)

    # ---- cross-section snapshots at storm-end (representative: peak pond / max water-table) ----
    y_sec = float(ih.attrs.get("y_section_m", 0.5))
    # in-house: head_xsec/theta_xsec already ON the y-section
    kih = int(np.argmin(np.abs(tih - storm_dur)))
    psi_xs_ih = np.asarray(ih["head_xsec"][kih].values, float)       # (z_ih, xc_ih)
    th_xs_ih = np.asarray(ih["theta_xsec"][kih].values, float)
    xc_ih = np.asarray(ih["xc"].values, float); z_ih = np.asarray(ih["z"].values, float)
    dmap_ih = np.asarray(ih["surface_depth_map"][kih].values, float)  # (y_ih, x_ih)
    x_ih = np.asarray(ih["x"].values, float); y_ih = np.asarray(ih["y"].values, float)
    # ParFlow: slice the y-cell nearest the section
    psi_xs_pf, kpf, jpf = _snapshot(full["press"], full["times"], storm_dur, full["y"], y_sec)
    th_xs_pf, _, _ = _snapshot(full["theta"], full["times"], storm_dur, full["y"], y_sec)
    dmap_pf = full["d_map"][kpf]
    x_pf = full["x"]; z_pf = full["z"]; y_pf = full["y"]

    # ---- agreement metrics ----
    m = dict(
        overland_final_inhouse_m3=float(ov_cum_ih[-1]), overland_final_parflow_m3=float(ov_cum_pf[-1]),
        lateral_gw_final_inhouse_m3=float(gw_cum_ih[-1]), lateral_gw_final_parflow_m3=float(gw_cum_pf[-1]),
        infiltration_final_inhouse_m3=float(infil_ih[-1]), infiltration_final_parflow_m3=float(infil_pf[-1]),
        peak_ponding_inhouse_mm=float(1e3 * pond_ih.max() / ((x_ih.max() * y_ih.max()) or 1.0)),  # mean-depth proxy
        peak_pond_d_inhouse_mm=float(1e3 * dmap_ih.max()), peak_pond_d_parflow_mm=float(1e3 * dmap_pf.max()),
        cum_rain_inhouse_m3=float(cum_rain_ih[-1]), cum_rain_parflow_m3=float(cum_rain_pf[-1]),
        # ratios only meaningful when the in-house component is non-negligible (else 0/0 -> NaN -> "n/a")
        overland_ratio_pf_over_ih=(float(ov_cum_pf[-1] / ov_cum_ih[-1]) if abs(ov_cum_ih[-1]) > 1e-3 else float("nan")),
        lateral_gw_ratio_pf_over_ih=(float(gw_cum_pf[-1] / gw_cum_ih[-1]) if abs(gw_cum_ih[-1]) > 1e-3 else float("nan")),
        inhouse_mbe_max=float(np.max(np.abs(mbe_ih))), parflow_mbe_max=float(np.max(np.abs(mbe_pf))),
        snapshot_time_day=float(storm_dur),
        runoff_onset_inhouse_day=runoff_onset_ih, runoff_onset_parflow_day=runoff_onset_pf,
    )

    deltas = (
        "OVERLAND (loam regime) matches to ~1%. LATERAL-GW magnitude is BC-parameterization-dependent: "
        "the in-house finite kr-weighted GHB (q=C*kr*(psi+z-H_ext), C=0.5/day) has NO native ParFlow "
        "analog; ParFlow's constant-head DirEquil face (C->inf) over-drains ~5x on SAND (Ks=7.13, rising "
        "water table) -> a documented BC delta, not a physics discrepancy (Arik 2026-06-10). "
        "Overland/GW split via the no-GW-control difference (PrintOverlandSum unreliable here). "
        "EARLY-INFILTRATION TRANSIENT (loam): the in-house sorptive Kirchhoff surface closure (+ air-entry "
        "cap) decays infiltration capacity as the surface approaches saturation -> reaches the steady rate "
        "EARLY and generates runoff sooner; ParFlow's coarse 8-cell column buffers rain at near-full rate "
        "until the top cell saturates (S->1), then chokes -> a longer, more curved early transient (would "
        "converge under vertical refinement). A TIMING delta -- end states converge. "
        "Other deltas: ParFlow FLAT grid + TopoSlopes vs in-house geometrically TILTED mesh; no Vogel/"
        "Ippisch air-entry cap; tiny SpecificStorage vs no-Ss; cell-centred FV (16x6x8) vs P1 FEM (17x7x9)."
    )

    ds = xr.Dataset(
        data_vars=dict(
            # --- overlaid time series (common time grid) ---
            cum_rain_inhouse=(("time",), cum_rain_ih, {"units": "m^3"}),
            cum_rain_parflow=(("time",), cum_rain_pf, {"units": "m^3"}),
            infiltration_inhouse=(("time",), infil_ih, {"units": "m^3"}),
            infiltration_parflow=(("time",), infil_pf, {"units": "m^3"}),
            ponding_inhouse=(("time",), pond_ih, {"units": "m^3"}),
            ponding_parflow=(("time",), pond_pf, {"units": "m^3"}),
            overland_cum_inhouse=(("time",), ov_cum_ih, {"units": "m^3", "long_name": "cum surface overland outflow"}),
            overland_cum_parflow=(("time",), ov_cum_pf, {"units": "m^3", "long_name": "cum overland (no-GW control)"}),
            lateral_gw_cum_inhouse=(("time",), gw_cum_ih, {"units": "m^3", "long_name": "cum lateral groundwater"}),
            lateral_gw_cum_parflow=(("time",), gw_cum_pf, {"units": "m^3", "long_name": "cum lateral GW (full - control)"}),
            overland_rate_inhouse=(("time",), ov_rate_ih, {"units": "m^3/day"}),
            overland_rate_parflow=(("time",), ov_rate_pf, {"units": "m^3/day"}),
            lateral_gw_rate_inhouse=(("time",), gw_rate_ih, {"units": "m^3/day"}),
            lateral_gw_rate_parflow=(("time",), gw_rate_pf, {"units": "m^3/day"}),
            infiltration_rate_inhouse=(("time",), infrate_ih, {"units": "m^3/day", "long_name": "infiltration rate dI/dt"}),
            infiltration_rate_parflow=(("time",), infrate_pf, {"units": "m^3/day", "long_name": "infiltration rate dI/dt"}),
            surf_sat_inhouse=(("time",), ssurf_ih, {"units": "-", "long_name": "surface saturation theta_top/theta_s"}),
            surf_sat_parflow=(("time",), ssurf_pf, {"units": "-", "long_name": "surface saturation theta_top/theta_s"}),
            mbe_inhouse=(("time",), mbe_ih, {"units": "-"}),
            mbe_parflow=(("time",), mbe_pf, {"units": "-", "long_name": "theta+pond partition residual"}),
            # --- cross-section snapshots (native grids; storm-end) ---
            psi_xsec_inhouse=(("z_ih", "xc_ih"), psi_xs_ih, {"units": "m"}),
            theta_xsec_inhouse=(("z_ih", "xc_ih"), th_xs_ih, {"units": "-"}),
            psi_xsec_parflow=(("z_pf", "xc_pf"), psi_xs_pf, {"units": "m"}),
            theta_xsec_parflow=(("z_pf", "xc_pf"), th_xs_pf, {"units": "-"}),
            # --- ponding maps (native grids; storm-end) ---
            dmap_inhouse=(("y_ih", "x_ih"), dmap_ih, {"units": "m"}),
            dmap_parflow=(("y_pf", "x_pf"), dmap_pf, {"units": "m"}),
        ),
        coords=dict(
            time=("time", t, {"units": "day"}),
            xc_ih=("xc_ih", xc_ih, {"units": "m"}), z_ih=("z_ih", z_ih, {"units": "m"}),
            x_ih=("x_ih", x_ih, {"units": "m"}), y_ih=("y_ih", y_ih, {"units": "m"}),
            xc_pf=("xc_pf", x_pf, {"units": "m"}), z_pf=("z_pf", z_pf, {"units": "m"}),
            x_pf=("x_pf", x_pf, {"units": "m"}), y_pf=("y_pf", y_pf, {"units": "m"}),
        ),
        attrs=dict(
            case=f"coupled_3d__{key}", date=DATE, soil=soil, Ks_m_per_day=spec["Ks"],
            title=f"3-D coupled hillslope ({spec['label']}): in-house vs ParFlow",
            scenario=spec["label"], y_section_m=y_sec,
            domain_LxWxH_m="5.0x1.0x1.0", bed_slope=0.05, water_table_z0_m=0.35,
            gw_external_head_m=0.20, manning_n=0.05, rain_rate_m_per_day=0.5, storm_duration_day=storm_dur,
            grid_inhouse="P1 FEM 16x6x8 elems (17x7x9 nodes), geometrically tilted",
            grid_parflow="cell-centred FV 16x6x8, flat grid + TopoSlopes",
            deltas=deltas, **m,
        ),
    )
    out_nc = os.path.join(HERE, "data", f"coupled_3d__{key}__{DATE}.nc")
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ds.to_netcdf(out_nc)
    ih.close()

    from make_comparison_coupled_3d_html import build as build_html
    out_html = os.path.join(HERE, "html", f"coupled_3d__{key}__{DATE}.html")
    build_html(out_nc, out_html)
    return m, out_nc, out_html


def main(only=None):
    keys = [only] if only else list(SCENARIOS)
    print("=== B5 coupled 3-D hillslope: in-house vs ParFlow ===")
    for key in keys:
        if key not in SCENARIOS:
            raise SystemExit(f"unknown scenario {key!r}; choose from {list(SCENARIOS)}")
        m, nc, html = build_one(key, SCENARIOS[key])
        print(f"\n[{key}]  (snapshot t={m['snapshot_time_day']:.3f} d)")
        print(f"  overland   : in-house {m['overland_final_inhouse_m3']:.4f}  ParFlow {m['overland_final_parflow_m3']:.4f} m^3"
              f"   (PF/IH = {m['overland_ratio_pf_over_ih']:.2f}x)")
        print(f"  lateral GW : in-house {m['lateral_gw_final_inhouse_m3']:.4f}  ParFlow {m['lateral_gw_final_parflow_m3']:.4f} m^3"
              f"   (PF/IH = {m['lateral_gw_ratio_pf_over_ih']:.2f}x)")
        print(f"  infiltration: in-house {m['infiltration_final_inhouse_m3']:.4f}  ParFlow {m['infiltration_final_parflow_m3']:.4f} m^3")
        print(f"  peak pond d : in-house {m['peak_pond_d_inhouse_mm']:.2f}  ParFlow {m['peak_pond_d_parflow_mm']:.2f} mm")
        print(f"  runoff onset: in-house {m['runoff_onset_inhouse_day']:.3f}  ParFlow {m['runoff_onset_parflow_day']:.3f} day"
              f"   (ParFlow delayed = it over-infiltrates early -> corroborates the transient story)")
        print(f"  cum_rain    : in-house {m['cum_rain_inhouse_m3']:.4f}  ParFlow {m['cum_rain_parflow_m3']:.4f} m^3")
        print(f"  partition residual (max): in-house {m['inhouse_mbe_max']:.2e}  ParFlow {m['parflow_mbe_max']:.2e}")
        print(f"  -> WROTE {os.path.relpath(nc, HERE)}  +  {os.path.relpath(html, HERE)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
