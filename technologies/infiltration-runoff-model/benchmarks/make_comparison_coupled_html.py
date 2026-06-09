#!/usr/bin/env python3
"""Side-by-side COUPLED benchmark HTML generator (in-house vs ParFlow) -- B4.

Reads ONE combined coupled comparison NetCDF (the contract written by
``build_comparison_coupled.py``) and emits ONE self-contained, offline, interactive HTML.
Extends ``make_comparison_html.py`` (theta/psi profile rows + error panels) with the coupled
SURFACE quantities the bulk comparison lacks:

  row 1: theta(z) -- in-house & ParFlow OVERLAID   | error  d theta(z)
  row 2: psi(z)   -- in-house & ParFlow OVERLAID    | error  d psi(z)
  row 3: surface ponding depth d(t), both           | infiltration PARTITION: cumulative
                                                       infiltrated vs ponded (both) + cum rain
  row 4: mass-balance error vs time, both (log y)               (colspan 2)
  + a TIME SLIDER animating the profile/error panels and a "now" marker on the time-series
    panels, and a METRICS panel (agreement + partition + the active formulation deltas).

Imports neither solver. Usage:
    python make_comparison_coupled_html.py data/<case>.nc html/<case>.html
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

IN_COLOR = "#1565c0"   # in-house (blue)
PF_COLOR = "#e65100"   # ParFlow (orange)
ERR_COLOR = "#00897b"  # error (teal)
RAIN_COLOR = "#455a64"  # cumulative rain reference (slate)


def _attr(ds, key, default=None):
    val = ds.attrs.get(key, default)
    return val.item() if isinstance(val, np.generic) else val


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.3e}"
        return f"{v:.{nd}g}"
    return str(v)


def _sym_range(arr, frac=0.08):
    m = float(np.max(np.abs(arr))) if arr.size else 1.0
    m = m if m > 0 else 1.0
    return [-(1.0 + frac) * m, (1.0 + frac) * m]


def build(nc_path: str, html_path: str) -> str:
    ds = xr.open_dataset(nc_path)
    z = np.asarray(ds["z"].values, float)
    t = np.asarray(ds["time"].values, float)
    th_i = np.asarray(ds["theta_inhouse"].values, float)
    th_p = np.asarray(ds["theta_parflow"].values, float)
    ps_i = np.asarray(ds["head_inhouse"].values, float)
    ps_p = np.asarray(ds["head_parflow"].values, float)
    d_th = np.asarray(ds["dtheta"].values, float)
    d_ps = np.asarray(ds["dhead"].values, float)
    mbe_i = np.asarray(ds["mbe_inhouse"].values, float)
    mbe_p = np.asarray(ds["mbe_parflow"].values, float)
    d_i = np.asarray(ds["surface_depth_inhouse"].values, float)
    d_p = np.asarray(ds["surface_depth_parflow"].values, float)
    inf_i = np.asarray(ds["infiltrated_inhouse"].values, float)
    inf_p = np.asarray(ds["infiltrated_parflow"].values, float)
    cum = np.asarray(ds["cum_rain"].values, float)
    nt = t.shape[0]

    theta_s = _attr(ds, "theta_s")
    theta_r = _attr(ds, "theta_r")

    # fixed axis ranges so the front "moves" rather than rescaling
    th_all = np.concatenate([th_i.ravel(), th_p.ravel()])
    ps_all = np.concatenate([ps_i.ravel(), ps_p.ravel()])
    th_lo, th_hi = float(np.min(th_all)), float(np.max(th_all))
    if theta_r is not None:
        th_lo = min(th_lo, float(theta_r))
    if theta_s is not None:
        th_hi = max(th_hi, float(theta_s))
    th_pad = 0.04 * (th_hi - th_lo + 1e-9)
    th_range = [th_lo - th_pad, th_hi + th_pad]
    ps_pad = 0.05 * (float(np.max(ps_all)) - float(np.min(ps_all)) + 1e-9)
    ps_range = [float(np.min(ps_all)) - ps_pad, float(np.max(ps_all)) + ps_pad]
    dth_range = _sym_range(d_th)
    dps_range = _sym_range(d_ps)
    z_range = [float(z.min()), float(z.max())]
    d_hi = max(float(np.max(d_i)), float(np.max(d_p)), 1e-6)
    d_range = [-0.04 * d_hi, 1.08 * d_hi]
    part_hi = max(float(np.max(inf_i)), float(np.max(inf_p)), float(np.max(cum)), 1e-6)
    part_range = [-0.04 * part_hi, 1.08 * part_hi]

    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        row_heights=[0.28, 0.28, 0.24, 0.20],
        vertical_spacing=0.075, horizontal_spacing=0.12,
        subplot_titles=("theta(z): in-house vs ParFlow",
                        "error  d theta(z) = in-house - ParFlow",
                        "psi(z): in-house vs ParFlow",
                        "error  d psi(z) = in-house - ParFlow",
                        "surface ponding depth d(t)",
                        "infiltration partition: cumulative infiltrated vs ponded",
                        "mass-balance error vs time (both models)"),
    )

    # reference lines on profile panels
    for val, col in ((theta_r, "#8d6e63"), (theta_s, "#2e7d32")):
        if val is not None:
            fig.add_vline(x=float(val), line=dict(color=col, width=1, dash="dash"), row=1, col=1, layer="below")
    fig.add_vline(x=0.0, line=dict(color="#bbb", width=1), row=2, col=1, layer="below")
    for r in (1, 2):
        fig.add_vline(x=0.0, line=dict(color="#999", width=1), row=r, col=2, layer="below")

    def add_profile(x, color, name, row, col, legend):
        fig.add_trace(
            go.Scatter(x=x[0], y=z, mode="lines+markers",
                       line=dict(color=color, width=2.3), marker=dict(size=3, color=color),
                       name=name, showlegend=legend,
                       hovertemplate="%{x:.4f}<br>z=%{y:.3f} m<extra>" + name + "</extra>"),
            row=row, col=col)
        return len(fig.data) - 1

    # row 1: theta | dtheta ; row 2: psi | dpsi  (animated z-profiles)
    i_thi = add_profile(th_i, IN_COLOR, "in-house", 1, 1, True)
    i_thp = add_profile(th_p, PF_COLOR, "ParFlow", 1, 1, True)
    i_dth = add_profile(d_th, ERR_COLOR, "in-house - ParFlow", 1, 2, True)
    i_psi = add_profile(ps_i, IN_COLOR, "in-house", 2, 1, False)
    i_psp = add_profile(ps_p, PF_COLOR, "ParFlow", 2, 1, False)
    i_dps = add_profile(d_ps, ERR_COLOR, "in-house - ParFlow", 2, 2, False)

    # row 3 col 1: surface depth d(t), both (static curves + animated "now" line)
    def ts(x, y, color, name, row, col, dash=None, width=2.0, legend=False):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash),
                                 name=name, showlegend=legend,
                                 hovertemplate="t=%{x:.3f} d<br>%{y:.4f}<extra>" + name + "</extra>"),
                      row=row, col=col)

    ts(t, 1000 * d_i, IN_COLOR, "d in-house", 3, 1)
    ts(t, 1000 * d_p, PF_COLOR, "d ParFlow", 3, 1)
    # row 3 col 2: partition -- cumulative infiltrated (solid) vs ponded (dashed) for both + cum rain
    ts(t, cum, RAIN_COLOR, "cumulative rain", 3, 2, dash="dot", width=1.6)
    ts(t, inf_i, IN_COLOR, "infiltrated in-house", 3, 2)
    ts(t, inf_p, PF_COLOR, "infiltrated ParFlow", 3, 2)
    ts(t, d_i, IN_COLOR, "ponded in-house", 3, 2, dash="dash", width=1.6)
    ts(t, d_p, PF_COLOR, "ponded ParFlow", 3, 2, dash="dash", width=1.6)

    # row 4: mass-balance error vs time, both (log)
    def mbe_clean(a):
        return np.where(np.abs(a) <= 0, np.nan, np.abs(a))
    ts(t, mbe_clean(mbe_i), IN_COLOR, "in-house MBE", 4, 1)
    ts(t, mbe_clean(mbe_p), PF_COLOR, "ParFlow MBE", 4, 1)

    # animated "now" vertical markers on the three time-series panels
    def nowline(row, col, yr):
        fig.add_trace(go.Scatter(x=[t[0], t[0]], y=yr, mode="lines",
                                 line=dict(color="#c0392b", width=1.5, dash="dot"),
                                 name="now", showlegend=False, hoverinfo="skip"), row=row, col=col)
        return len(fig.data) - 1
    mbe_floor = max(1e-16, np.nanmin([np.nanmin(mbe_clean(mbe_i)), np.nanmin(mbe_clean(mbe_p))]))
    mbe_top = np.nanmax([np.nanmax(mbe_clean(mbe_i)), np.nanmax(mbe_clean(mbe_p)), 1e-12])
    i_now_d = nowline(3, 1, d_range)
    i_now_p = nowline(3, 2, part_range)
    i_now_m = nowline(4, 1, [mbe_floor, mbe_top])

    # animation frames: update the 6 profiles + 3 now-lines
    frames = []
    for k in range(nt):
        frames.append(go.Frame(name=str(k),
                               traces=[i_thi, i_thp, i_dth, i_psi, i_psp, i_dps, i_now_d, i_now_p, i_now_m],
                               data=[go.Scatter(x=th_i[k], y=z), go.Scatter(x=th_p[k], y=z),
                                     go.Scatter(x=d_th[k], y=z), go.Scatter(x=ps_i[k], y=z),
                                     go.Scatter(x=ps_p[k], y=z), go.Scatter(x=d_ps[k], y=z),
                                     go.Scatter(x=[t[k], t[k]], y=d_range),
                                     go.Scatter(x=[t[k], t[k]], y=part_range),
                                     go.Scatter(x=[t[k], t[k]], y=[mbe_floor, mbe_top])]))
    fig.frames = frames

    steps = [dict(method="animate", label=f"{t[k]:.2f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(nt)]
    sliders = [dict(active=0, x=0.05, y=-0.04, len=0.9, pad=dict(t=30, b=10),
                    currentvalue=dict(prefix="time = ", suffix="  day", font=dict(size=14)), steps=steps)]
    updatemenus = [dict(type="buttons", direction="left", x=0.05, y=1.10, xanchor="left", showactive=False,
                        buttons=[dict(label="▶ Play", method="animate",
                                      args=[None, dict(mode="immediate", fromcurrent=True,
                                                       frame=dict(duration=200, redraw=True),
                                                       transition=dict(duration=0))]),
                                 dict(label="⏸ Pause", method="animate",
                                      args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=True))])])]

    # axes
    fig.update_yaxes(title_text="z [m]", range=z_range, row=1, col=1)
    fig.update_yaxes(range=z_range, row=1, col=2)
    fig.update_yaxes(title_text="z [m]", range=z_range, row=2, col=1)
    fig.update_yaxes(range=z_range, row=2, col=2)
    fig.update_xaxes(title_text="theta [m3/m3]", range=th_range, row=1, col=1)
    fig.update_xaxes(title_text="d theta [m3/m3]", range=dth_range, row=1, col=2)
    fig.update_xaxes(title_text="psi [m]", range=ps_range, row=2, col=1)
    fig.update_xaxes(title_text="d psi [m]", range=dps_range, row=2, col=2)
    fig.update_xaxes(title_text="time [day]", row=3, col=1)
    fig.update_yaxes(title_text="ponding depth [mm]", range=d_range, row=3, col=1)
    fig.update_xaxes(title_text="time [day]", row=3, col=2)
    fig.update_yaxes(title_text="water [m]", range=part_range, row=3, col=2)
    fig.update_xaxes(title_text="time [day]", row=4, col=1)
    fig.update_yaxes(title_text="|rel. MB error|", type="log", exponentformat="power", row=4, col=1)

    # metrics panel
    lines = [
        f"<b>case</b>: {_attr(ds, 'case', '')}",
        f"<b>date</b>: {_attr(ds, 'date', '')}",
        f"<b>soil</b>: {_attr(ds, 'soil', '')}",
        (f"&nbsp;&nbsp;theta_r={_fmt(theta_r)} theta_s={_fmt(theta_s)} "
         f"alpha={_fmt(_attr(ds,'alpha_per_m'))} n={_fmt(_attr(ds,'n'))} Ks={_fmt(_attr(ds,'Ks_m_per_day'))}"),
        f"<b>forcing</b>: q={_fmt(_attr(ds,'rain_flux_m_per_day'))} m/d for {_fmt(_attr(ds,'storm_duration_day'))} d, "
        f"psi0={_fmt(_attr(ds,'psi0_m'))} m, cum_in={_fmt(_attr(ds,'cumulative_input_m'))} m",
        "<b>--- profile agreement (in-house - ParFlow) ---</b>",
        f"&nbsp;&nbsp;max|d theta|={_fmt(_attr(ds,'max_abs_dtheta'))} &nbsp; RMS={_fmt(_attr(ds,'rms_dtheta'))}",
        f"&nbsp;&nbsp;max|d psi| ={_fmt(_attr(ds,'max_abs_dhead'))} m &nbsp; RMS={_fmt(_attr(ds,'rms_dhead'))} m",
        "<b>--- surface / partition ---</b>",
        f"&nbsp;&nbsp;peak pond d: in-house={_fmt(1000*_attr(ds,'peak_d_inhouse_m'))} mm, "
        f"ParFlow={_fmt(1000*_attr(ds,'peak_d_parflow_m'))} mm",
        f"&nbsp;&nbsp;final infiltrated: in-house={_fmt(_attr(ds,'final_infiltrated_inhouse_m'))} m, "
        f"ParFlow={_fmt(_attr(ds,'final_infiltrated_parflow_m'))} m",
        f"&nbsp;&nbsp;final ponded: in-house={_fmt(_attr(ds,'final_ponded_inhouse_m'))} m, "
        f"ParFlow={_fmt(_attr(ds,'final_ponded_parflow_m'))} m",
        f"&nbsp;&nbsp;max|d d|={_fmt(1000*_attr(ds,'max_abs_dd_m'))} mm &nbsp; RMS={_fmt(1000*_attr(ds,'rms_dd_m'))} mm",
        f"<b>mass balance</b>: in-house max={_fmt(_attr(ds,'inhouse_mbe_max'))}, "
        f"ParFlow max={_fmt(_attr(ds,'parflow_mbe_max'))}",
        f"<b>deltas</b>: {_attr(ds,'deltas','')}",
    ]
    fig.add_annotation(x=1.005, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="top",
                       align="left", showarrow=False, text="<b>METRICS</b><br>" + "<br>".join(lines),
                       font=dict(size=10, color="#222", family="Consolas, monospace"),
                       bordercolor="#888", borderwidth=1, borderpad=8, bgcolor="rgba(245,245,245,0.95)")

    title = _attr(ds, "title", "in-house vs ParFlow")
    fig.update_layout(
        title=dict(text=f"<b>PIDS benchmark - coupled surface&harr;subsurface (in-house vs ParFlow)</b><br>"
                        f"<span style='font-size:13px'>{title} &middot; {_attr(ds,'date','')}</span>",
                   x=0.5, xanchor="center", y=0.99, yanchor="top"),
        sliders=sliders, updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="right", x=1.0, font=dict(size=10)),
        margin=dict(l=70, r=370, t=120, b=70),
        width=1300, height=1180, paper_bgcolor="white", plot_bgcolor="#fafafa", hovermode="closest",
    )

    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    fig.write_html(html_path, include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(displaylogo=False, responsive=True))
    ds.close()
    return html_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python make_comparison_coupled_html.py <comparison.nc> <out.html>", file=sys.stderr)
        sys.exit(2)
    out = build(sys.argv[1], sys.argv[2])
    print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
