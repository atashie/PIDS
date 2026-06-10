"""Tier-3 visualizer for the Module-4 §E Phase-3 OFFLINE full-curve fidelity gate.

Reads ONLY the saved gate result (the data contract) -- never imports/runs the solver:
  validation/sanity/data/m4_phase3_gate__2026-06-09.npz   (per-curve t, Iref, Imodel, Iplanar)
  validation/sanity/data/m4_phase3_gate__2026-06-09.json  (closure defns, tolerances, verdict, per-curve rel-L2 + status)
and produces ONE self-contained offline HTML (Plotly vendored inline -> opens by double-click).

What this check is (framed honestly in the HTML):
  The CLOSURE cumulative sorptive uptake I(t) [m, per wall area] is compared against the
  RESOLVED near-field REFERENCE I(t), per soil x geometry x direction.

  * DISPERSE (a-priori, NO knob): cylindrical Green-Ampt F=2z/ln(1+2z) x Parlange S.
    GATING leg -- PASSES at rel-L2 <= 5% (worst 4.1% on SAND/LOAM/SILT). For disperse we also
    overlay the PLANAR clock (F=1, the constant-kappa discriminator) which FAILS the same harness.
  * DRAIN (semi-empirical desorptivity throttle): FLAGGED -- advisory, not gating.
  * CLAY: near-saturated degenerate edge -- FLAG-EXCLUDED.

  (A coupled-embedding claim was RETRACTED post-review; THIS offline gate is the solid result.
   We do NOT visualize any coupled-embedding claim.)

Layout: a 4-soil x 2-geometry grid of subplots. Each panel overlays, on a shared log-t x-axis:
  - disperse reference (markers) + disperse closure (solid line) + disperse PLANAR clock (dashed, the failing comparator)
  - drain reference (open markers) + drain closure (dash-dot line)
PASS vs FLAG is encoded in the panel-title colour (green = PASS gate / grey = FLAG advisory / red = worst CLAY edge).
An embedded METRICS panel surfaces the deciding numbers + the honest scope note.

Usage:  python viz/make_phase3_gate_html.py [out.html]
Run from forward-model/.  Dependencies: numpy + plotly (vendored inline -> offline double-click).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------- paths (relative to forward-model/)
DATA_DIR = "../validation/sanity/data"
NPZ = f"{DATA_DIR}/m4_phase3_gate__2026-06-09.npz"
JSON = f"{DATA_DIR}/m4_phase3_gate__2026-06-09.json"
DEFAULT_OUT = "../validation/sanity/viz/m4_phase3_gate__2026-06-09.html"

SOILS = ["SAND", "LOAM", "SILT", "CLAY"]
GEOMS = ["tunnel", "annulus"]

# Per-DIRECTION colours (so disperse vs drain read at a glance, consistent across panels).
C_DISP_REF = "#1565c0"   # disperse reference  (blue)
C_DISP_MOD = "#1565c0"   # disperse closure    (blue line)
C_DISP_PLAN = "#c0392b"  # disperse PLANAR clock (red dashed -- the FAILING discriminator)
C_DRAIN_REF = "#e08a00"  # drain reference     (amber)
C_DRAIN_MOD = "#e08a00"  # drain closure       (amber line)

# Status -> panel-title colour.
STATUS_COLOR = {"PASS": "#1b7837", "FLAG": "#666666", "FAIL": "#c0392b"}


def _fmt_pct(x):
    return "n/a" if x is None else f"{100.0 * float(x):.2f}%"


def build(out_html: str) -> str:
    npz = dict(np.load(NPZ))
    with open(JSON, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # index the per-curve metrics by (soil, geom, direction)
    cmeta = {(c["soil"], c["geom"], c["direction"]): c for c in meta["curves"]}

    units_t = meta["units"]["t"]      # "day"
    units_I = meta["units"]["I"]      # "m (cumulative uptake per wall area)"

    # ---- panel titles carry the gate verdict per (soil, geom): use the DISPERSE status
    #      (the gating leg) for the title colour; full per-curve numbers go in the metrics table.
    subplot_titles = []
    for s in SOILS:
        for g in GEOMS:
            disp = cmeta[(s, g, "disperse")]
            drn = cmeta[(s, g, "drain")]
            col = STATUS_COLOR.get(disp["status"], "#333")
            tag = "PASS" if disp["status"] == "PASS" else "FLAG"
            subplot_titles.append(
                f"<span style='color:{col}'><b>{s} · {g}</b>  [disperse {tag}]</span>"
                f"<br><span style='font-size:10px;color:#555'>"
                f"disperse L2={_fmt_pct(disp['rel_l2'])} (planar={_fmt_pct(disp.get('planar_l2'))}) · "
                f"drain L2={_fmt_pct(drn['rel_l2'])}</span>"
            )

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.075, horizontal_spacing=0.085,
    )

    # legend de-dup: only show each series label once
    seen = set()

    def _show(label):
        if label in seen:
            return False
        seen.add(label)
        return True

    for ri, s in enumerate(SOILS):
        for ci, g in enumerate(GEOMS):
            row, col = ri + 1, ci + 1
            base = f"{s}_{g}"

            # --- DISPERSE: reference (markers), closure (solid), planar clock (red dashed) ---
            t_d = npz[f"{base}_disperse_t"]
            iref_d = npz[f"{base}_disperse_Iref"]
            imod_d = npz[f"{base}_disperse_Imodel"]
            iplan_d = npz.get(f"{base}_disperse_Iplanar")
            disp = cmeta[(s, g, "disperse")]

            fig.add_trace(go.Scatter(
                x=t_d, y=iref_d, mode="markers", legendgroup="disp_ref",
                name="disperse reference (resolved)", showlegend=_show("disp_ref"),
                marker=dict(color=C_DISP_REF, size=7, symbol="circle"),
                hovertemplate=(f"<b>{s} · {g} · disperse REF</b><br>"
                               "t=%{x:.3e} day<br>I=%{y:.4e} m<extra></extra>"),
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=t_d, y=imod_d, mode="lines", legendgroup="disp_mod",
                name="disperse closure (cyl Green-Ampt, a-priori)", showlegend=_show("disp_mod"),
                line=dict(color=C_DISP_MOD, width=2.5),
                hovertemplate=(f"<b>{s} · {g} · disperse closure</b><br>"
                               f"rel-L2 = {_fmt_pct(disp['rel_l2'])} ({disp['status']})<br>"
                               "t=%{x:.3e} day<br>I=%{y:.4e} m<extra></extra>"),
            ), row=row, col=col)
            if iplan_d is not None:
                fig.add_trace(go.Scatter(
                    x=t_d, y=iplan_d, mode="lines", legendgroup="disp_plan",
                    name="planar clock F=1 (FAILS gate)", showlegend=_show("disp_plan"),
                    line=dict(color=C_DISP_PLAN, width=1.8, dash="dash"),
                    hovertemplate=(f"<b>{s} · {g} · PLANAR clock (F=1)</b><br>"
                                   f"rel-L2 = {_fmt_pct(disp.get('planar_l2'))} (the failing discriminator)<br>"
                                   "t=%{x:.3e} day<br>I=%{y:.4e} m<extra></extra>"),
                ), row=row, col=col)

            # --- DRAIN: reference (open markers), closure (dash-dot) ---
            t_r = npz[f"{base}_drain_t"]
            iref_r = npz[f"{base}_drain_Iref"]
            imod_r = npz[f"{base}_drain_Imodel"]
            drn = cmeta[(s, g, "drain")]

            fig.add_trace(go.Scatter(
                x=t_r, y=iref_r, mode="markers", legendgroup="drain_ref",
                name="drain reference (resolved)", showlegend=_show("drain_ref"),
                marker=dict(color=C_DRAIN_REF, size=7, symbol="circle-open",
                            line=dict(width=1.6, color=C_DRAIN_REF)),
                hovertemplate=(f"<b>{s} · {g} · drain REF</b><br>"
                               "t=%{x:.3e} day<br>I=%{y:.4e} m<extra></extra>"),
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=t_r, y=imod_r, mode="lines", legendgroup="drain_mod",
                name="drain closure (semi-empirical throttle, FLAG advisory)", showlegend=_show("drain_mod"),
                line=dict(color=C_DRAIN_MOD, width=2.0, dash="dashdot"),
                hovertemplate=(f"<b>{s} · {g} · drain closure</b><br>"
                               f"rel-L2 = {_fmt_pct(drn['rel_l2'])} ({drn['status']}, advisory)<br>"
                               "t=%{x:.3e} day<br>I=%{y:.4e} m<extra></extra>"),
            ), row=row, col=col)

            # per-panel axes: log-t (spans decades); log-I reads best across the magnitude range
            fig.update_xaxes(title_text=f"t  [{units_t}]", type="log", row=row, col=col,
                             showgrid=True, gridcolor="#eee")
            fig.update_yaxes(title_text="I(t)  [m]", type="log", row=row, col=col,
                             showgrid=True, gridcolor="#eee")

    # ----------------------------------------------------------------------------- METRICS panel
    v = meta["verdict"]
    tol = meta["tolerance"]
    cl = meta["closure"]

    rows_html = []
    for c in meta["curves"]:
        sc = STATUS_COLOR.get(c["status"], "#333")
        gating = "gate" if c.get("pass_required") else "adv."
        planar = _fmt_pct(c.get("planar_l2")) if c["direction"] == "disperse" else "—"
        rows_html.append(
            f"<tr>"
            f"<td>{c['soil']}</td><td>{c['geom']}</td><td>{c['direction']}</td>"
            f"<td style='text-align:right'>{_fmt_pct(c['rel_l2'])}</td>"
            f"<td style='text-align:right;color:#c0392b'>{planar}</td>"
            f"<td style='color:{sc};font-weight:bold'>{c['status']}</td>"
            f"<td style='color:#777'>{gating}</td>"
            f"</tr>"
        )
    table = (
        "<table style='border-collapse:collapse;font-size:10px;font-family:Consolas,monospace'>"
        "<tr style='border-bottom:1px solid #999'>"
        "<th>soil</th><th>geom</th><th>dir</th>"
        "<th>rel-L2</th><th style='color:#c0392b'>planar</th><th>status</th><th>role</th></tr>"
        + "".join(rows_html) +
        "</table>"
    )

    metrics_html = (
        "<b>METRICS</b><br>"
        f"<b>check</b>: {meta['check']}<br>"
        f"<b>module</b>: {meta['module']}<br>"
        f"<b>date</b>: {meta['date']}<br><br>"
        f"<b style='color:#1b7837'>DISPERSE a-priori gate: {v['disperse_apriori_gate']}</b><br>"
        f"worst PASS-required rel-L2 = <b>{_fmt_pct(v['worst_disperse_pass_l2'])}</b> "
        f"(tol: {tol['disperse_pass']})<br>"
        f"advisory: {tol['advisory']}<br><br>"
        f"<b>metric</b>: {meta['metric']}<br><br>"
        f"<b>closure (disperse, gating)</b>:<br>&nbsp;{cl['disperse']}<br>"
        f"<b>planar (the FAIL)</b>: {cl['planar']}<br>"
        f"<b>drain (flagged)</b>:<br>&nbsp;{cl['drain']}<br><br>"
        + table +
        f"<br><b>scope note</b>: {v['note']}"
    )
    fig.add_annotation(
        x=1.012, y=1.0, xref="paper", yref="paper",
        xanchor="left", yanchor="top", align="left", showarrow=False,
        text=metrics_html,
        font=dict(size=11, color="#222", family="Consolas, monospace"),
        bordercolor="#888", borderwidth=1, borderpad=8,
        bgcolor="rgba(245,245,245,0.97)",
    )

    title = "M4 §E Phase-3 offline fidelity gate · 2026-06-09"
    fig.update_layout(
        title=dict(
            text="<b>PIDS Pillar-2 sanity (Tier-3)</b><br>"
                 f"<span style='font-size:13px'>{title}</span>"
                 "<br><span style='font-size:11px;color:#555'>"
                 "I(t) = cumulative sorptive uptake per wall area · "
                 "blue = disperse (gating, a-priori) · red dashed = planar clock F=1 (fails) · "
                 "amber = drain (advisory)</span>",
            x=0.5, xanchor="center", y=0.985, yanchor="top",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.07,
                    xanchor="center", x=0.42, font=dict(size=10)),
        margin=dict(l=70, r=420, t=120, b=90),
        width=1500, height=1500,
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        hovermode="closest", template="plotly_white",
    )

    # axis-title font size down a touch (dense grid)
    fig.update_xaxes(title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_font=dict(size=10), tickfont=dict(size=9))

    fig.write_html(
        out_html,
        include_plotlyjs=True,   # inline the WHOLE plotly.js -> offline, no CDN
        full_html=True,
        auto_play=False,
        config=dict(displaylogo=False, responsive=True,
                    toImageButtonOptions=dict(format="png", scale=2)),
    )
    return out_html


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    path = build(out)
    print(f"WROTE {path}  ({os.path.getsize(path) / 1e6:.2f} MB)")
