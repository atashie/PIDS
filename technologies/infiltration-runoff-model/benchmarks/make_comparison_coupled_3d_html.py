#!/usr/bin/env python3
"""Side-by-side 3-D COUPLED hillslope benchmark HTML (in-house vs ParFlow) -- B5.

Reads ONE comparison NetCDF (the contract from build_comparison_coupled_3d.py) and emits ONE
self-contained, offline, interactive HTML. Mirrors the in-house 3-D coupling viz
(forward-model/viz/make_coupling_3d_html.py) views but as a TWO-MODEL overlay focused on the
benchmark story:

  row 1: PARTITION -- cumulative infiltration / overland / lateral-GW vs time, in-house (solid)
         vs ParFlow (dashed), with cum_rain reference   |   final-partition grouped BAR chart
  row 2: OUTLET HYDROGRAPHS -- surface overland Q(t)     |   lateral groundwater Q(t)
         (both models)                                       (both models)
  row 3: psi CROSS-SECTION at storm-end (y-section) -- in-house  |  ParFlow, each with the
         water-table (psi=0) contour; SHARED colour range so the water-table heights compare
  row 4: theta CROSS-SECTION at storm-end -- in-house    |   ParFlow (shared theta range)
  row 5: SURFACE PONDING map d(x,y) at storm-end -- in-house  |  ParFlow (shared 0..max mm)
  + METRICS panel (partition agreement, the sand over-drain ratio, the active formulation deltas).

Imports neither solver. Usage:
    python make_comparison_coupled_3d_html.py data/<case>.nc html/<case>.html
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

IN_COLOR = "#1565c0"     # in-house (blue)
PF_COLOR = "#e65100"     # ParFlow (orange)
RAIN_COLOR = "#455a64"   # cumulative rain reference
INFIL_COLOR = "#2e7d32"  # infiltration
OV_COLOR = "#0277bd"     # overland
GW_COLOR = "#6a1b9a"     # lateral groundwater
WT_COLOR = "#111111"     # water-table iso-line


def _attr(ds, key, default=None):
    val = ds.attrs.get(key, default)
    return val.item() if isinstance(val, np.generic) else val


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "n/a"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.3e}"
        return f"{v:.{nd}g}"
    return str(v)


def _watertable_z(psi_zx, z):
    """Water-table elevation (psi=0 crossing, scanning from the surface down) per xc column."""
    nz, nx = psi_zx.shape
    wt = np.full(nx, np.nan)
    for j in range(nx):
        col = psi_zx[:, j]
        found = np.nan
        for i in range(nz - 1, 0, -1):
            a, b = col[i - 1], col[i]
            if np.isnan(a) or np.isnan(b):
                continue
            if (a >= 0.0) and (b < 0.0):
                d = a - b
                found = z[i - 1] + (a / d if d else 0.0) * (z[i] - z[i - 1]); break
            if (a < 0.0) and (b >= 0.0):
                d = b - a
                found = z[i - 1] + ((-a) / d if d else 0.0) * (z[i] - z[i - 1]); break
        if np.isnan(found) and np.all(col >= 0.0):
            found = z[-1]
        wt[j] = found
    return wt


def build(nc_path: str, html_path: str) -> str:
    ds = xr.open_dataset(nc_path)
    t = np.asarray(ds["time"].values, float)

    cr_i = ds["cum_rain_inhouse"].values; cr_p = ds["cum_rain_parflow"].values
    inf_i = ds["infiltration_inhouse"].values; inf_p = ds["infiltration_parflow"].values
    pnd_i = ds["ponding_inhouse"].values; pnd_p = ds["ponding_parflow"].values
    ov_i = ds["overland_cum_inhouse"].values; ov_p = ds["overland_cum_parflow"].values
    gw_i = ds["lateral_gw_cum_inhouse"].values; gw_p = ds["lateral_gw_cum_parflow"].values
    ovr_i = ds["overland_rate_inhouse"].values; ovr_p = ds["overland_rate_parflow"].values
    gwr_i = ds["lateral_gw_rate_inhouse"].values; gwr_p = ds["lateral_gw_rate_parflow"].values
    ifr_i = ds["infiltration_rate_inhouse"].values; ifr_p = ds["infiltration_rate_parflow"].values
    ss_i = ds["surf_sat_inhouse"].values; ss_p = ds["surf_sat_parflow"].values

    psi_i = np.asarray(ds["psi_xsec_inhouse"].values, float); psi_p = np.asarray(ds["psi_xsec_parflow"].values, float)
    th_i = np.asarray(ds["theta_xsec_inhouse"].values, float); th_p = np.asarray(ds["theta_xsec_parflow"].values, float)
    xc_i = ds["xc_ih"].values; z_i = ds["z_ih"].values
    xc_p = ds["xc_pf"].values; z_p = ds["z_pf"].values
    dm_i = np.asarray(ds["dmap_inhouse"].values, float) * 1e3   # mm
    dm_p = np.asarray(ds["dmap_parflow"].values, float) * 1e3
    x_i = ds["x_ih"].values; y_i = ds["y_ih"].values
    x_p = ds["x_pf"].values; y_p = ds["y_pf"].values

    t_snap = _attr(ds, "snapshot_time_day")
    y_sec = _attr(ds, "y_section_m")
    soil = _attr(ds, "soil", "")

    # shared colour ranges for fair side-by-side comparison
    psi_lo = float(min(np.nanmin(psi_i), np.nanmin(psi_p))); psi_hi = float(max(np.nanmax(psi_i), np.nanmax(psi_p)))
    th_lo = float(min(np.nanmin(th_i), np.nanmin(th_p))); th_hi = float(max(np.nanmax(th_i), np.nanmax(th_p)))
    dm_hi = float(max(np.nanmax(dm_i), np.nanmax(dm_p), 1e-6))
    wt_i = _watertable_z(psi_i, z_i); wt_p = _watertable_z(psi_p, z_p)

    fig = make_subplots(
        rows=6, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "secondary_y": True}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        row_heights=[0.185, 0.15, 0.16, 0.17, 0.17, 0.17],
        vertical_spacing=0.058, horizontal_spacing=0.11,
        subplot_titles=(
            "Partition (cumulative): infiltration / overland / lateral GW vs time",
            "Final partition [m&sup3;] (in-house vs ParFlow)",
            "Surface overland discharge Q(t)", "Lateral groundwater discharge Q(t)",
            "Infiltration RATE dI/dt + surface saturation S_surf (the early-transient diagnostic)",
            "Surface saturation S_surf = &theta;_top/&theta;_s",
            f"&psi;(x,z) in-house @ t={_fmt(t_snap)} d (y={_fmt(y_sec)} m; line=water table)",
            f"&psi;(x,z) ParFlow @ t={_fmt(t_snap)} d (line=water table)",
            "&theta;(x,z) in-house", "&theta;(x,z) ParFlow",
            "Surface ponding d(x,y) in-house [mm]", "Surface ponding d(x,y) ParFlow [mm]",
        ),
    )

    # ===== row 1 col 1: partition cumulative lines (in-house solid / ParFlow dashed) =====
    def line(x, y, color, name, row, col, dash=None, width=2.4, legend=True):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash),
                                 name=name, showlegend=legend,
                                 hovertemplate="t=%{x:.3f} d<br>%{y:.4f}<extra>" + name + "</extra>"),
                      row=row, col=col)

    line(t, cr_i, RAIN_COLOR, "cum rain (in-house)", 1, 1, dash="dot", width=1.6)
    line(t, cr_p, RAIN_COLOR, "cum rain (ParFlow)", 1, 1, dash="longdashdot", width=1.2, legend=False)
    line(t, inf_i, INFIL_COLOR, "infiltration in-house", 1, 1)
    line(t, inf_p, INFIL_COLOR, "infiltration ParFlow", 1, 1, dash="dash")
    line(t, ov_i, OV_COLOR, "overland in-house", 1, 1)
    line(t, ov_p, OV_COLOR, "overland ParFlow", 1, 1, dash="dash")
    line(t, gw_i, GW_COLOR, "lateral GW in-house", 1, 1)
    line(t, gw_p, GW_COLOR, "lateral GW ParFlow", 1, 1, dash="dash")

    # ===== row 1 col 2: final-partition grouped bars =====
    cats = ["infiltration", "ponding", "overland", "lateral GW"]
    ih_vals = [float(inf_i[-1]), float(pnd_i[-1]), float(ov_i[-1]), float(gw_i[-1])]
    pf_vals = [float(inf_p[-1]), float(pnd_p[-1]), float(ov_p[-1]), float(gw_p[-1])]
    fig.add_trace(go.Bar(x=cats, y=ih_vals, name="in-house", marker_color=IN_COLOR, showlegend=False,
                         hovertemplate="%{x}<br>in-house=%{y:.4f} m&sup3;<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Bar(x=cats, y=pf_vals, name="ParFlow", marker_color=PF_COLOR, showlegend=False,
                         hovertemplate="%{x}<br>ParFlow=%{y:.4f} m&sup3;<extra></extra>"), row=1, col=2)

    # ===== row 2: outlet hydrographs =====
    line(t, ovr_i, IN_COLOR, "overland in-house", 2, 1, legend=False)
    line(t, ovr_p, PF_COLOR, "overland ParFlow", 2, 1, dash="dash", legend=False)
    line(t, gwr_i, IN_COLOR, "lateral GW in-house", 2, 2, legend=False)
    line(t, gwr_p, PF_COLOR, "lateral GW ParFlow", 2, 2, dash="dash", legend=False)

    # ===== row 3: infiltration RATE dI/dt (+ S_surf twin) | S_surf (the early-transient diagnostic) =====
    fig.add_trace(go.Scatter(x=t, y=ifr_i, mode="lines", line=dict(color=IN_COLOR, width=2.4),
                             name="infil rate in-house", showlegend=False,
                             hovertemplate="t=%{x:.3f} d<br>dI/dt=%{y:.3f} m&sup3;/d<extra>in-house</extra>"),
                  row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=ifr_p, mode="lines", line=dict(color=PF_COLOR, width=2.4, dash="dash"),
                             name="infil rate ParFlow", showlegend=False,
                             hovertemplate="t=%{x:.3f} d<br>dI/dt=%{y:.3f} m&sup3;/d<extra>ParFlow</extra>"),
                  row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=ss_i, mode="lines", line=dict(color=IN_COLOR, width=1.1, dash="dot"),
                             name="S_surf in-house", showlegend=False, opacity=0.65,
                             hovertemplate="t=%{x:.3f} d<br>S_surf=%{y:.3f}<extra>in-house</extra>"),
                  row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=t, y=ss_p, mode="lines", line=dict(color=PF_COLOR, width=1.1, dash="dot"),
                             name="S_surf ParFlow", showlegend=False, opacity=0.65,
                             hovertemplate="t=%{x:.3f} d<br>S_surf=%{y:.3f}<extra>ParFlow</extra>"),
                  row=3, col=1, secondary_y=True)
    line(t, ss_i, IN_COLOR, "S_surf in-house", 3, 2, legend=False)
    line(t, ss_p, PF_COLOR, "S_surf ParFlow", 3, 2, dash="dash", legend=False)

    # ===== rows 4-5: cross-sections (shared ranges) =====
    def heat(x, y, zz, zlo, zhi, scale, row, col, cbar_title, show):
        fig.add_trace(go.Heatmap(x=x, y=y, z=zz, zmin=zlo, zmax=zhi, zauto=False, colorscale=scale,
                                 showscale=show,
                                 colorbar=(dict(title=dict(text=cbar_title, side="right"), thickness=11,
                                                len=0.16, x=1.0) if show else None),
                                 hovertemplate="x=%{x:.3f}<br>z=%{y:.3f}<br>%{z:.4f}<extra></extra>"),
                      row=row, col=col)

    heat(xc_i, z_i, psi_i, psi_lo, psi_hi, "RdBu", 4, 1, "&psi; [m]", False)
    heat(xc_p, z_p, psi_p, psi_lo, psi_hi, "RdBu", 4, 2, "&psi; [m]", True)
    fig.add_trace(go.Scatter(x=xc_i, y=wt_i, mode="lines", line=dict(color=WT_COLOR, width=3), connectgaps=False,
                             name="water table", showlegend=False, hoverinfo="skip"), row=4, col=1)
    fig.add_trace(go.Scatter(x=xc_p, y=wt_p, mode="lines", line=dict(color=WT_COLOR, width=3), connectgaps=False,
                             name="water table", showlegend=False, hoverinfo="skip"), row=4, col=2)
    heat(xc_i, z_i, th_i, th_lo, th_hi, "YlGnBu", 5, 1, "&theta; [-]", False)
    heat(xc_p, z_p, th_p, th_lo, th_hi, "YlGnBu", 5, 2, "&theta; [-]", True)

    # ===== row 6: ponding maps (shared 0..max mm) =====
    heat(x_i, y_i, dm_i, 0.0, dm_hi, "Blues", 6, 1, "d [mm]", False)
    heat(x_p, y_p, dm_p, 0.0, dm_hi, "Blues", 6, 2, "d [mm]", True)

    # ===== axes =====
    fig.update_xaxes(title_text="time [day]", row=1, col=1)
    fig.update_yaxes(title_text="cumulative water [m&sup3;]", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="volume [m&sup3;]", row=1, col=2, rangemode="tozero")
    for c in (1, 2):
        fig.update_xaxes(title_text="time [day]", row=2, col=c)
    fig.update_yaxes(title_text="overland Q [m&sup3;/day]", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="lateral GW Q [m&sup3;/day]", row=2, col=2, rangemode="tozero")
    # row 3: infiltration rate (+ S_surf twin) | S_surf
    fig.update_xaxes(title_text="time [day]", row=3, col=1)
    fig.update_yaxes(title_text="dI/dt [m&sup3;/day]", row=3, col=1, secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="S_surf [-]", range=[0.0, 1.05], row=3, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="time [day]", row=3, col=2)
    fig.update_yaxes(title_text="S_surf = &theta;_top/&theta;_s", range=[0.0, 1.05], row=3, col=2)
    # rows 4-5: cross-sections
    for col, (xc, zc) in ((1, (xc_i, z_i)), (2, (xc_p, z_p))):
        for row in (4, 5):
            fig.update_xaxes(title_text="x downslope [m] (outlet at right)", range=[float(xc.min()), float(xc.max())], row=row, col=col)
            fig.update_yaxes(title_text="z [m] (surface top)", range=[float(zc.min()), float(zc.max())], row=row, col=col)
    # row 6: ponding maps
    for col, (xx, yy) in ((1, (x_i, y_i)), (2, (x_p, y_p))):
        fig.update_xaxes(title_text="x downslope [m]", range=[float(xx.min()), float(xx.max())], constrain="domain", row=6, col=col)
        fig.update_yaxes(title_text="y cross-slope [m]", range=[float(yy.min()), float(yy.max())], row=6, col=col)

    # ===== metrics panel =====
    ov_r = _attr(ds, "overland_ratio_pf_over_ih"); gw_r = _attr(ds, "lateral_gw_ratio_pf_over_ih")
    ov_chip = f" &nbsp;(<b>{_fmt(ov_r)}&times;</b>)" if (ov_r is not None and np.isfinite(ov_r)) else " &nbsp;(both ~0)"
    gw_chip = f" &nbsp;(<b>{_fmt(gw_r)}&times;</b>)" if (gw_r is not None and np.isfinite(gw_r)) else ""
    lines = [
        f"<b>case</b>: {_attr(ds,'case','')} &nbsp; <b>soil</b>: {soil} (Ks={_fmt(_attr(ds,'Ks_m_per_day'))})",
        f"<b>date</b>: {_attr(ds,'date','')} &nbsp; <b>snapshot</b>: t={_fmt(t_snap)} d",
        f"<b>domain</b>: {_attr(ds,'domain_LxWxH_m')} m, slope {_fmt(_attr(ds,'bed_slope'))}, n={_fmt(_attr(ds,'manning_n'))}",
        f"<b>storm</b>: {_fmt(_attr(ds,'rain_rate_m_per_day'))} m/d &times; {_fmt(_attr(ds,'storm_duration_day'))} d",
        "<b>--- partition (final, m&sup3;) ---</b>",
        f"&nbsp;infiltration: IH={_fmt(_attr(ds,'infiltration_final_inhouse_m3'))} PF={_fmt(_attr(ds,'infiltration_final_parflow_m3'))}",
        f"&nbsp;OVERLAND: IH={_fmt(_attr(ds,'overland_final_inhouse_m3'))} PF={_fmt(_attr(ds,'overland_final_parflow_m3'))}"
        + ov_chip,
        f"&nbsp;LATERAL GW: IH={_fmt(_attr(ds,'lateral_gw_final_inhouse_m3'))} PF={_fmt(_attr(ds,'lateral_gw_final_parflow_m3'))}"
        + gw_chip,
        f"&nbsp;peak pond d: IH={_fmt(_attr(ds,'peak_pond_d_inhouse_mm'))} PF={_fmt(_attr(ds,'peak_pond_d_parflow_mm'))} mm",
        f"&nbsp;runoff onset: IH={_fmt(_attr(ds,'runoff_onset_inhouse_day'))} PF={_fmt(_attr(ds,'runoff_onset_parflow_day'))} d"
        " <i>(PF later = over-infiltrates early)</i>",
        f"&nbsp;cum rain: IH={_fmt(_attr(ds,'cum_rain_inhouse_m3'))} PF={_fmt(_attr(ds,'cum_rain_parflow_m3'))}",
        f"<b>partition residual</b>: IH={_fmt(_attr(ds,'inhouse_mbe_max'))} PF={_fmt(_attr(ds,'parflow_mbe_max'))}",
        "<b>--- deltas ---</b>",
        f"<span style='font-size:9px'>{_attr(ds,'deltas','')}</span>",
    ]
    fig.add_annotation(x=1.012, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="top",
                       align="left", showarrow=False, text="<b>METRICS</b><br>" + "<br>".join(lines),
                       font=dict(size=10, color="#222", family="Consolas, monospace"),
                       bordercolor="#888", borderwidth=1, borderpad=8, bgcolor="rgba(245,245,245,0.96)")

    title = _attr(ds, "title", "in-house vs ParFlow")
    fig.update_layout(
        title=dict(text="<b>PIDS benchmark B5 &mdash; 3-D coupled hillslope (in-house vs ParFlow)</b><br>"
                        f"<span style='font-size:12px'>{title} &middot; {_attr(ds,'date','')}</span>",
                   x=0.5, xanchor="center", y=0.995, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.012, xanchor="left", x=0.0, font=dict(size=9)),
        margin=dict(l=70, r=400, t=130, b=60),
        width=1380, height=2120, barmode="group",
        paper_bgcolor="white", plot_bgcolor="#fafafa", hovermode="closest",
    )

    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    fig.write_html(html_path, include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(displaylogo=False, responsive=True))
    ds.close()
    return html_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python make_comparison_coupled_3d_html.py <comparison.nc> <out.html>", file=sys.stderr)
        sys.exit(2)
    out = build(sys.argv[1], sys.argv[2])
    print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
