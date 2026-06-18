#!/usr/bin/env python3
"""Side-by-side tilted-V hydrograph comparison HTML (in-house vs ParFlow vs analytic Q_eq) -- B6.

Reads the comparison NetCDF from build_comparison_tiltedv.py and emits ONE self-contained, offline,
interactive HTML:
  row 1: OUTLET HYDROGRAPH Q(t) -- in-house vs ParFlow, with the analytic Q_eq line + rain hyetograph
         (inverted twin axis). The headline: both reach Q_eq; the equilibrium plateau + recession are
         the clean comparison; the rising limb is cold-start-transient-contaminated (shaded).
  row 2: CUMULATIVE outflow int(Q dt) -- both vs cumulative rain (mass check).
  + METRICS panel (Q_eq, equilibrium plateaus, peaks, in-house run cost, the formulation deltas incl.
    the in-house convergent-routing envelope finding).

Usage: python make_comparison_tiltedv_html.py <comparison.nc> <out.html>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

IN_COLOR = "#1565c0"   # in-house
PF_COLOR = "#e65100"   # ParFlow
QEQ_COLOR = "#2e7d32"  # analytic equilibrium
RAIN_COLOR = "#90a4ae"
CUM_COLOR = "#455a64"


def _attr(ds, k, d=None):
    v = ds.attrs.get(k, d)
    return v.item() if isinstance(v, np.generic) else v


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


def build(nc_path: str, html_path: str) -> str:
    ds = xr.open_dataset(nc_path)
    t = np.asarray(ds["time"].values, float)
    q_ih = np.asarray(ds["q_inhouse"].values, float)
    q_pf = np.asarray(ds["q_parflow"].values, float)
    cum_ih = np.asarray(ds["cum_inhouse"].values, float)
    cum_pf = np.asarray(ds["cum_parflow"].values, float)
    cum_rain = np.asarray(ds["cum_rain"].values, float)
    rain = np.asarray(ds["rain_rate"].values, float)
    Q_eq = float(_attr(ds, "Q_eq_m3day"))
    storm = float(_attr(ds, "storm_min", 90.0)) / (24.0 * 60.0)   # day

    # cold-start transient window (rising limb, ~first 10% of the storm) -- shade + de-emphasize
    t_cold = 0.18 * storm

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "xy", "secondary_y": True}], [{"type": "xy"}]],
        row_heights=[0.6, 0.4], vertical_spacing=0.13,
        subplot_titles=(
            "Outlet hydrograph Q(t): in-house vs ParFlow vs analytic Q_eq (rain inverted; rising limb = cold-start transient)",
            "Cumulative outflow ∫Q dt vs cumulative rain (mass check)",
        ),
    )

    # --- row 1: hydrographs ---
    fig.add_trace(go.Scatter(x=t, y=q_ih, mode="lines", line=dict(color=IN_COLOR, width=2.6),
                             name="in-house (Manning outlet)",
                             hovertemplate="t=%{x:.4f} d<br>Q=%{y:.0f} m³/d<extra>in-house</extra>"),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=q_pf, mode="lines", line=dict(color=PF_COLOR, width=2.6, dash="dash"),
                             name="ParFlow (storage balance)",
                             hovertemplate="t=%{x:.4f} d<br>Q=%{y:.0f} m³/d<extra>ParFlow</extra>"),
                  row=1, col=1, secondary_y=False)
    fig.add_hline(y=Q_eq, line=dict(color=QEQ_COLOR, width=2, dash="dot"), row=1, col=1,
                  annotation_text=f"analytic Q_eq = {Q_eq/86400:.2f} m³/s", annotation_position="top right")
    # rain hyetograph (inverted, top of a twin axis)
    r_top = float(np.max(rain)) * 3.0 if np.max(rain) > 0 else 1.0
    fig.add_trace(go.Bar(x=t, y=rain, base=r_top - rain,
                         marker=dict(color=RAIN_COLOR, line=dict(color="#607d8b", width=0.4)), opacity=0.7,
                         width=(t[1] - t[0]) * 0.9, name="rain",
                         hovertemplate="t=%{x:.4f} d<br>rain<extra></extra>"),
                  row=1, col=1, secondary_y=True)
    # cold-start shading
    fig.add_vrect(x0=0.0, x1=t_cold, fillcolor="rgba(120,120,120,0.10)", line_width=0, row=1, col=1)

    # --- row 2: cumulative ---
    fig.add_trace(go.Scatter(x=t, y=cum_rain, mode="lines", line=dict(color=CUM_COLOR, width=1.8, dash="dot"),
                             name="cumulative rain",
                             hovertemplate="t=%{x:.4f} d<br>%{y:.0f} m³<extra>cum rain</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=cum_ih, mode="lines", line=dict(color=IN_COLOR, width=2.4),
                             name="cum outflow in-house", showlegend=False,
                             hovertemplate="t=%{x:.4f} d<br>%{y:.0f} m³<extra>in-house</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=cum_pf, mode="lines", line=dict(color=PF_COLOR, width=2.4, dash="dash"),
                             name="cum outflow ParFlow", showlegend=False,
                             hovertemplate="t=%{x:.4f} d<br>%{y:.0f} m³<extra>ParFlow</extra>"), row=2, col=1)

    fig.update_xaxes(title_text="time [day]", row=1, col=1)
    fig.update_yaxes(title_text="outlet Q [m³/day]", row=1, col=1, secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="rain (inv)", range=[r_top, 0.0], row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="time [day]", row=2, col=1)
    fig.update_yaxes(title_text="cumulative water [m³]", row=2, col=1, rangemode="tozero")

    lines = [
        f"<b>case</b>: {_attr(ds,'case','')} &nbsp; <b>date</b>: {_attr(ds,'date','')}",
        f"<b>domain</b>: {_attr(ds,'domain_km')} km &nbsp; Sx={_fmt(_attr(ds,'slope_x'))} Sy={_fmt(_attr(ds,'slope_y'))} n={_fmt(_attr(ds,'manning_n'))}",
        f"<b>rain</b>: 3e-6 m/s &times; 90 min + recession",
        "<b>--- known answer ---</b>",
        f"&nbsp;analytic Q_eq = {_fmt(_attr(ds,'Q_eq_m3day'))} m³/d = <b>{_fmt(_attr(ds,'Q_eq_m3s'))} m³/s</b>",
        "<b>--- equilibrium plateau (late storm) ---</b>",
        f"&nbsp;ParFlow:  {_fmt(_attr(ds,'plateau_parflow_over_Qeq'))} Q_eq  <i>(reproduces known answer)</i>",
        f"&nbsp;in-house: {_fmt(_attr(ds,'plateau_inhouse_over_Qeq'))} Q_eq  <i>(reproduces known answer)</i>",
        f"&nbsp;(cold-start peaks: PF {_fmt(_attr(ds,'peak_parflow_over_Qeq'))}×, IH {_fmt(_attr(ds,'peak_inhouse_over_Qeq'))}×)",
        "<b>--- in-house mass books (P0-corrected) ---</b>",
        f"&nbsp;engine ledger gap: {_fmt(100*_attr(ds,'inhouse_ledger_gap_engine_frac'))}% of cum rain "
        f"<i>(per-accepted-step books)</i>",
        f"&nbsp;40-pt trapz residue: {_fmt(100*_attr(ds,'inhouse_trapz_residue_frac'))}% "
        f"<i>(hydrograph sampling, not mass loss)</i>",
        "<b>--- in-house cost (envelope) ---</b>",
        f"&nbsp;{_fmt(_attr(ds,'inhouse_run_minutes'))} min, {_attr(ds,'inhouse_steps')} steps, mesh {_attr(ds,'inhouse_mesh')}",
        f"&nbsp;ParFlow grid {_attr(ds,'parflow_grid')}",
        "<b>--- deltas ---</b>",
        f"<span style='font-size:9px'>{_attr(ds,'deltas','')}</span>",
    ]
    fig.add_annotation(x=1.015, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="top",
                       align="left", showarrow=False, text="<b>METRICS</b><br>" + "<br>".join(lines),
                       font=dict(size=10, color="#222", family="Consolas, monospace"),
                       bordercolor="#888", borderwidth=1, borderpad=8, bgcolor="rgba(245,245,245,0.96)")

    fig.update_layout(
        title=dict(text="<b>PIDS benchmark B6 &mdash; canonical tilted-V catchment (in-house vs ParFlow vs Q_eq)</b><br>"
                        f"<span style='font-size:12px'>{_attr(ds,'title','')} &middot; {_attr(ds,'date','')}</span>",
                   x=0.5, xanchor="center", y=0.985, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0, font=dict(size=10)),
        margin=dict(l=80, r=400, t=110, b=60), width=1320, height=900, barmode="overlay",
        paper_bgcolor="white", plot_bgcolor="#fafafa", hovermode="x unified",
    )

    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    fig.write_html(html_path, include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(displaylogo=False, responsive=True))
    ds.close()
    return html_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python make_comparison_tiltedv_html.py <comparison.nc> <out.html>", file=sys.stderr)
        sys.exit(2)
    out = build(sys.argv[1], sys.argv[2])
    print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
