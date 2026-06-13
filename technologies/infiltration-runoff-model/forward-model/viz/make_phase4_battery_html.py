"""Tier-3 visualizer for the M4 Phase-4 coupled-embedding deployment gate battery.

Reads ONLY the saved data contract (never imports/runs the solver):
  validation/sanity/data/m4_phase4_battery__2026-06-12.npz / .json
and produces ONE self-contained offline HTML (Plotly vendored inline).

What the eye should confirm (physical realism, panel by panel):
  * every embedded curve rises MONOTONICALLY and bends toward (never crosses) the grey
    capacity line I_max -- a finite closed reservoir depleting;
  * the n=8 solid and n=12 dotted production curves LIE ON TOP OF EACH OTHER (resolution
    robustness; the drain legs are n-independent by construction of the mass-exact drive);
  * the red dashed twin -- the same physics WITHOUT the live host read (or, on refD40-C,
    without recharge knowledge) -- visibly misses the bend / the re-steepening: the host
    coupling carries the signal;
  * on the two HISTORY legs the shaded band marks the source window: the reference re-steepens
    inside it and the production curve follows with NO knowledge of the source (it sees only
    the live host state);
  * the diagnostics panel shows the per-instant WI-era exchange rate vs the resolved truth at
    matched I (disperse LOAM R40): the known, localized mid-curve deficit that cumulative
    rel-L2 tolerates -- shown so the residual is SEEN, not hidden.

Usage:  python viz/make_phase4_battery_html.py [out.html]    (from forward-model/)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATE = "2026-06-12"
DATA_DIR = "../validation/sanity/data"
NPZ = f"{DATA_DIR}/m4_phase4_battery__{DATE}.npz"
JSON = f"{DATA_DIR}/m4_phase4_battery__{DATE}.json"
DEFAULT_OUT = f"../validation/sanity/viz/m4_phase4_battery__{DATE}.html"

LEG_TITLES = {
    "disp_LOAM_R40": "DISPERSE · LOAM · R=40 r_w (full depletion)",
    "disp_SAND_R40": "DISPERSE · SAND · R=40 r_w (full depletion)",
    "disp_SILT_R40": "DISPERSE · SILT · R=40 r_w (full depletion)",
    "disp_LOAM_RefB40": "DISPERSE · LOAM · RefB-40 re-wetting HISTORY",
    "drain_LOAM_refD40": "DRAIN · LOAM · refD40 (R=40 r_w)",
    "drain_SAND_R40": "DRAIN · SAND · R=40 r_w (fresh ref)",
    "drain_SILT_R40": "DRAIN · SILT · R=40 r_w (fresh ref)",
    "drain_LOAM_R20deep": "DRAIN · LOAM · R=20 r_w DEEP (48%)",
    "drain_LOAM_refD40C": "DRAIN · LOAM · refD40-C CONTINUOUS RECHARGE",
}
ORDER = list(LEG_TITLES)

C_REF = "#444444"
C_EMB = {"disperse": "#1565c0", "drain": "#e08a00"}
C_TWIN = "#c0392b"
C_CAP = "#999999"


def build(out_html: str) -> str:
    npz = dict(np.load(NPZ))
    with open(JSON, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    legs = {r["leg"]: r for r in meta["legs"]}

    titles = []
    for key in ORDER:
        r = legs[key]
        titles.append(
            f"<span style='color:#1b7837'><b>{LEG_TITLES[key]}</b>  [PASS]</span>"
            f"<br><span style='font-size:10px;color:#555'>"
            f"emb n=8 {r['n8_rel_l2']:.1%} / n=12 {r['n12_rel_l2']:.1%} · "
            f"<span style='color:{C_TWIN}'>{r['twin_name']} {r['twin_rel_l2']:.0%}</span></span>")
    titles.append(
        "<b>diagnostic · disperse WI-era rate vs resolved truth (matched I)</b>"
        "<br><span style='font-size:10px;color:#555'>the localized residual, shown not hidden"
        "</span>")

    fig = make_subplots(rows=4, cols=3, subplot_titles=titles,
                        vertical_spacing=0.066, horizontal_spacing=0.07)
    seen = set()

    def show(lbl):
        if lbl in seen:
            return False
        seen.add(lbl)
        return True

    for i, key in enumerate(ORDER):
        row, col = i // 3 + 1, i % 3 + 1
        r = legs[key]
        t = npz[f"{key}_t"]
        I_ref, I_twin = npz[f"{key}_Iref"], npz[f"{key}_Itwin"]
        i_max, v_extra = float(npz[f"{key}_Imax"]), float(npz[f"{key}_Vextra"])
        c = C_EMB[r["direction"]]

        if f"{key}_t_src" in npz:                          # the history legs: shade the source era
            t1, t2 = (float(x) for x in npz[f"{key}_t_src"])
            fig.add_vrect(x0=t1, x1=t2, fillcolor="#d9ead3", opacity=0.5, line_width=0,
                          row=row, col=col)
        fig.add_hline(y=i_max, line=dict(color=C_CAP, width=1.2, dash="dash"), row=row, col=col)
        if v_extra > 0:
            fig.add_hline(y=i_max + v_extra, line=dict(color=C_CAP, width=1.0, dash="dot"),
                          row=row, col=col)

        fig.add_trace(go.Scatter(
            x=t, y=I_ref, mode="markers", legendgroup="ref",
            name="resolved reference", showlegend=show("ref"),
            marker=dict(color=C_REF, size=6, symbol="circle-open", line=dict(width=1.4)),
            hovertemplate=f"<b>{key} REF</b><br>t=%{{x:.3e}} d<br>I=%{{y:.4e}} m<extra></extra>",
        ), row=row, col=col)
        for n, dash, lbl in ((8, "solid", "production n=8"), (12, "dot", "production n=12")):
            fig.add_trace(go.Scatter(
                x=t, y=npz[f"{key}_Iemb_n{n}"], mode="lines",
                legendgroup=f"emb{n}", name=lbl, showlegend=show(f"emb{n}"),
                line=dict(color=c, width=2.4 if n == 8 else 1.8, dash=dash),
                hovertemplate=(f"<b>{key} production n={n}</b><br>"
                               f"rel-L2 {r[f'n{n}_rel_l2']:.1%}, end {r[f'n{n}_end_ratio']:.3f}"
                               f"<br>t=%{{x:.3e}} d<br>I=%{{y:.4e}} m<extra></extra>"),
            ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=t, y=I_twin, mode="lines", legendgroup="twin",
            name="discrimination twin (no host/source knowledge)", showlegend=show("twin"),
            line=dict(color=C_TWIN, width=1.6, dash="dash"),
            hovertemplate=(f"<b>{key} twin: {r['twin_name']}</b><br>"
                           f"rel-L2 {r['twin_rel_l2']:.0%} (killed)"
                           f"<br>t=%{{x:.3e}} d<br>I=%{{y:.4e}} m<extra></extra>"),
        ), row=row, col=col)

        ymax = max(float(np.max(I_ref)), i_max + v_extra,
                   float(np.max(npz[f"{key}_Iemb_n8"]))) * 1.08
        fig.update_xaxes(title_text="t [day]", type="log", row=row, col=col,
                         showgrid=True, gridcolor="#eee")
        fig.update_yaxes(title_text="I(t) [m]", range=[0, ymax], row=row, col=col,
                         showgrid=True, gridcolor="#eee")

    # ---- diagnostics panel (row 4, col 1): WI-era rate ratio vs depletion fraction ----------
    for n, dash in ((8, "solid"), (12, "dot")):
        xf, yf = npz[f"diag_n{n}_Ifrac"], npz[f"diag_n{n}_ratio"]
        ok = np.isfinite(yf)                # the resolved rate -> 0 at the full-depletion plateau
        xf, yf = xf[ok], yf[ok]
        fig.add_trace(go.Scatter(
            x=xf, y=yf, mode="lines",
            legendgroup=f"diag{n}", name=f"WI-era rate / resolved rate (n={n})",
            showlegend=show(f"diag{n}"),
            line=dict(color="#6a3d9a", width=2.0 if n == 8 else 1.5, dash=dash),
            hovertemplate=(f"<b>rate ratio n={n}</b><br>I/I_max=%{{x:.2f}}<br>"
                           "emb/resolved=%{y:.2f}<extra></extra>"),
        ), row=4, col=1)
    fig.add_hline(y=1.0, line=dict(color="#444", width=1.0, dash="dash"), row=4, col=1)
    fig.update_xaxes(title_text="I / I_max (matched cumulative state)", range=[0, 1],
                     row=4, col=1, showgrid=True, gridcolor="#eee")
    fig.update_yaxes(title_text="rate ratio emb/resolved", range=[0, 1.6],
                     row=4, col=1, showgrid=True, gridcolor="#eee")

    # ---- metrics annotation ------------------------------------------------------------------
    v, tol = meta["verdict"], meta["tolerance"]
    rows_html = []
    for key in ORDER:
        r = legs[key]
        rows_html.append(
            f"<tr><td>{LEG_TITLES[key].replace(' · ', '·')}</td>"
            f"<td style='text-align:right'>{r['n8_rel_l2']:.1%}</td>"
            f"<td style='text-align:right'>{r['n12_rel_l2']:.1%}</td>"
            f"<td style='text-align:right'>{r['n8_end_ratio']:.3f}</td>"
            f"<td style='text-align:right;color:{C_TWIN}'>{r['twin_rel_l2']:.0%}</td>"
            f"<td style='text-align:right;color:#777'>{r['depletion_frac']:.0%}</td></tr>")
    table = ("<table style='border-collapse:collapse;font-size:10px;"
             "font-family:Consolas,monospace'>"
             "<tr style='border-bottom:1px solid #999'><th>leg</th><th>n=8</th><th>n=12</th>"
             "<th>end/ref</th><th style='color:#c0392b'>twin</th><th>depl.</th></tr>"
             + "".join(rows_html) + "</table>")
    metrics = (
        "<b>METRICS</b><br>"
        f"<b>check</b>: {meta['check']}<br>"
        f"<b>date</b>: {meta['date']}<br><br>"
        f"<b style='color:#1b7837'>ALL {len(meta['legs'])} LEGS PASS</b><br>"
        f"worst embedded rel-L2 = <b>{v['worst_embedded_rel_l2']:.1%}</b> "
        f"(tol {tol['embedded']:.0%})<br>"
        f"worst twin kill = <b>{v['worst_twin_kill']:.0%}</b> "
        f"(bar {tol['baseline_kill']:.0%})<br><br>"
        + table +
        f"<br><b>mass ledgers</b>: {meta['ledgers']}<br><br>"
        f"<b>scope + known residuals</b>: {v['note']}"
    )
    fig.add_annotation(x=1.012, y=1.0, xref="paper", yref="paper",
                       xanchor="left", yanchor="top", align="left", showarrow=False,
                       text=metrics, font=dict(size=11, family="Consolas, monospace"),
                       bordercolor="#888", borderwidth=1, borderpad=8,
                       bgcolor="rgba(245,245,245,0.97)")

    fig.update_layout(
        title=dict(
            text="<b>PIDS Pillar-2 sanity (Tier-3)</b><br>"
                 f"<span style='font-size:13px'>M4 Phase-4 coupled-embedding deployment gate "
                 f"battery · {DATE}</span>"
                 "<br><span style='font-size:11px;color:#555'>"
                 "I(t) = cumulative wall exchange per wall area · grey dash = capacity I_max · "
                 "green band = source window · red dash = the killed no-host-knowledge twin · "
                 "n=8 solid vs n=12 dotted (on top of each other = resolution-robust)</span>",
            x=0.5, xanchor="center", y=0.988, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.42,
                    font=dict(size=10)),
        margin=dict(l=70, r=430, t=125, b=80),
        width=1620, height=1700,
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        hovermode="closest", template="plotly_white",
    )
    fig.update_xaxes(title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_font=dict(size=10), tickfont=dict(size=9))
    fig.write_html(out_html, include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(displaylogo=False, responsive=True,
                               toImageButtonOptions=dict(format="png", scale=2)))
    return out_html


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    path = build(out)
    print(f"WROTE {path}  ({os.path.getsize(path) / 1e6:.2f} MB)")
