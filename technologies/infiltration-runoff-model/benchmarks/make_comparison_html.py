#!/usr/bin/env python3
"""Side-by-side benchmark HTML generator (in-house vs ParFlow).

Reads ONE combined comparison NetCDF (the contract written by the case builders,
e.g. ``build_comparison_column.py``) and emits ONE self-contained, offline,
interactive HTML -- the side-by-side viewer. Imports neither solver.

Layout:
  row 1:  theta(z) -- in-house & ParFlow OVERLAID   |  error  d theta(z) = in-house - ParFlow
  row 2:  psi(z)   -- in-house & ParFlow OVERLAID    |  error  d psi(z)   = in-house - ParFlow
  row 3:  mass-balance error vs time, both models (log y)   (colspan 2)
  + a TIME SLIDER animating every profile/error panel, and a METRICS panel with
    the difference summary and the active formulation deltas.

Usage:
    python make_comparison_html.py data/<case>.nc html/<case>.html

Dependencies: xarray + plotly only (Plotly vendored inline -> opens offline).
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
    d_th = np.asarray(ds["dtheta"].values, float)   # in-house - ParFlow
    d_ps = np.asarray(ds["dhead"].values, float)
    mbe_i = np.asarray(ds["mbe_inhouse"].values, float)
    mbe_p = np.asarray(ds["mbe_parflow"].values, float)
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

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        row_heights=[0.36, 0.36, 0.28],
        vertical_spacing=0.10, horizontal_spacing=0.12,
        subplot_titles=("theta(z): in-house vs ParFlow",
                        "error  d theta(z) = in-house - ParFlow",
                        "psi(z): in-house vs ParFlow",
                        "error  d psi(z) = in-house - ParFlow",
                        "mass-balance error vs time (both models)"),
    )

    # reference lines
    for val, col in ((theta_r, "#8d6e63"), (theta_s, "#2e7d32")):
        if val is not None:
            fig.add_vline(x=float(val), line=dict(color=col, width=1, dash="dash"),
                          row=1, col=1, layer="below")
    fig.add_vline(x=0.0, line=dict(color="#bbb", width=1), row=2, col=1, layer="below")  # psi=0
    for r in (1, 2):  # zero-error reference on the error panels
        fig.add_vline(x=0.0, line=dict(color="#999", width=1), row=r, col=2, layer="below")

    def add(x, color, name, row, col, legend):
        fig.add_trace(
            go.Scatter(x=x[0], y=z, mode="lines+markers",
                       line=dict(color=color, width=2.3), marker=dict(size=3, color=color),
                       name=name, showlegend=legend,
                       hovertemplate="%{x:.4f}<br>z=%{y:.3f} m<extra>" + name + "</extra>"),
            row=row, col=col)
        return len(fig.data) - 1

    # row 1: theta overlaid | dtheta error
    i_thi = add(th_i, IN_COLOR, "in-house", 1, 1, True)
    i_thp = add(th_p, PF_COLOR, "ParFlow", 1, 1, True)
    i_dth = add(d_th, ERR_COLOR, "in-house - ParFlow", 1, 2, True)
    # row 2: psi overlaid | dpsi error
    i_psi = add(ps_i, IN_COLOR, "in-house", 2, 1, False)
    i_psp = add(ps_p, PF_COLOR, "ParFlow", 2, 1, False)
    i_dps = add(d_ps, ERR_COLOR, "in-house - ParFlow", 2, 2, False)

    # row 3: mass-balance error vs time (both); static + animated current-time marker
    def mbe_clean(a):
        return np.where(np.abs(a) <= 0, np.nan, np.abs(a))
    fig.add_trace(go.Scatter(x=t, y=mbe_clean(mbe_i), mode="lines+markers",
                             line=dict(color=IN_COLOR, width=2), name="in-house MBE", showlegend=False,
                             hovertemplate="t=%{x:.3f} d<br>err=%{y:.2e}<extra>in-house</extra>"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=mbe_clean(mbe_p), mode="lines+markers",
                             line=dict(color=PF_COLOR, width=2), name="ParFlow MBE", showlegend=False,
                             hovertemplate="t=%{x:.3f} d<br>err=%{y:.2e}<extra>ParFlow</extra>"),
                  row=3, col=1)
    i_mark = len(fig.data)
    fig.add_trace(go.Scatter(x=[t[0]], y=[1.0], mode="markers",
                             marker=dict(size=11, color="#c0392b", symbol="circle-open",
                                         line=dict(width=3, color="#c0392b")),
                             name="now", showlegend=False,
                             hovertemplate="t=%{x:.3f} d<extra>now</extra>"),
                  row=3, col=1)

    # animation frames
    frames = []
    for k in range(nt):
        ym = max(mbe_clean(mbe_i)[k] if np.isfinite(mbe_clean(mbe_i)[k]) else 1e-16,
                 mbe_clean(mbe_p)[k] if np.isfinite(mbe_clean(mbe_p)[k]) else 1e-16)
        frames.append(go.Frame(name=str(k),
                               traces=[i_thi, i_thp, i_dth, i_psi, i_psp, i_dps, i_mark],
                               data=[go.Scatter(x=th_i[k], y=z), go.Scatter(x=th_p[k], y=z),
                                     go.Scatter(x=d_th[k], y=z), go.Scatter(x=ps_i[k], y=z),
                                     go.Scatter(x=ps_p[k], y=z), go.Scatter(x=d_ps[k], y=z),
                                     go.Scatter(x=[t[k]], y=[ym])]))
    fig.frames = frames

    steps = [dict(method="animate", label=f"{t[k]:.2f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(nt)]
    sliders = [dict(active=0, x=0.05, y=-0.04, len=0.9, pad=dict(t=30, b=10),
                    currentvalue=dict(prefix="time = ", suffix="  day", font=dict(size=14)),
                    steps=steps)]
    updatemenus = [dict(type="buttons", direction="left", x=0.05, y=1.13, xanchor="left",
                        showactive=False,
                        buttons=[dict(label="▶ Play", method="animate",
                                      args=[None, dict(mode="immediate", fromcurrent=True,
                                                       frame=dict(duration=200, redraw=True),
                                                       transition=dict(duration=0))]),
                                 dict(label="⏸ Pause", method="animate",
                                      args=[[None], dict(mode="immediate",
                                                         frame=dict(duration=0, redraw=True))])])]

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
    fig.update_yaxes(title_text="|rel. MB error|", type="log", exponentformat="power", row=3, col=1)

    # metrics panel
    lines = [
        f"<b>case</b>: {_attr(ds, 'case', '')}",
        f"<b>date</b>: {_attr(ds, 'date', '')}",
        f"<b>soil</b>: {_attr(ds, 'soil', '')}",
        (f"&nbsp;&nbsp;theta_r={_fmt(theta_r)} theta_s={_fmt(theta_s)} "
         f"alpha={_fmt(_attr(ds,'alpha_per_m'))} n={_fmt(_attr(ds,'n'))} Ks={_fmt(_attr(ds,'Ks_m_per_day'))}"),
        f"<b>forcing</b>: q={_fmt(_attr(ds,'rain_flux_m_per_day'))} m/d, psi0={_fmt(_attr(ds,'psi0_m'))} m, "
        f"cum_in={_fmt(_attr(ds,'cumulative_input_m'))} m",
        "<b>--- agreement (in-house - ParFlow) ---</b>",
        f"&nbsp;&nbsp;max|d theta|={_fmt(_attr(ds,'max_abs_dtheta'))} &nbsp; RMS={_fmt(_attr(ds,'rms_dtheta'))}",
        f"&nbsp;&nbsp;max|d psi| ={_fmt(_attr(ds,'max_abs_dhead'))} m &nbsp; RMS={_fmt(_attr(ds,'rms_dhead'))} m",
        f"<b>mass balance</b>: in-house max={_fmt(_attr(ds,'inhouse_mbe_max'))}, "
        f"ParFlow max={_fmt(_attr(ds,'parflow_mbe_max'))}",
        f"<b>deltas</b>: {_attr(ds,'deltas','')}",
    ]
    fig.add_annotation(x=1.005, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="top",
                       align="left", showarrow=False, text="<b>METRICS</b><br>" + "<br>".join(lines),
                       font=dict(size=10.5, color="#222", family="Consolas, monospace"),
                       bordercolor="#888", borderwidth=1, borderpad=8, bgcolor="rgba(245,245,245,0.95)")

    title = _attr(ds, "title", "in-house vs ParFlow")
    fig.update_layout(
        title=dict(text=f"<b>PIDS benchmark - in-house vs ParFlow</b><br>"
                        f"<span style='font-size:13px'>{title} &middot; {_attr(ds,'date','')}</span>",
                   x=0.5, xanchor="center", y=0.985, yanchor="top"),
        sliders=sliders, updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=70, r=360, t=130, b=80),
        width=1240, height=940, paper_bgcolor="white", plot_bgcolor="#fafafa", hovermode="closest",
    )

    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    fig.write_html(html_path, include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(displaylogo=False, responsive=True))
    ds.close()
    return html_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python make_comparison_html.py <comparison.nc> <out.html>", file=sys.stderr)
        sys.exit(2)
    out = build(sys.argv[1], sys.argv[2])
    print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
