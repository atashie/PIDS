"""Tier-3 visualizer for the Module-4 §E Phase-1 near-field REFERENCE SUITE (curves).

Reads ONLY the two saved reference tables (the result contract) -- never imports/runs the solver:
  scratch/m4_phase1b_disperse_refs.npz   (disperse: tunnel/drain/annulus I(t), 4 soils x 2 geometries)
  scratch/m4_phase1c_drain_refs.npz      (drain: tunnel/annulus/planar I(t))
and the fixed ANALYTICAL Parlange sorptivities (constants, computed once from the van Genuchten closures;
NOT the FEM solver) to overlay the ground-truth comparators. Produces ONE self-contained offline HTML.

The visual story (what to validate by eye before Phase 2):
  1. DISPERSE = Philip absorption: cumulative uptake I(t) ~ S*sqrt(t) -- straight lines vs sqrt(t),
     slope = the sorptivity, matching the independent analytical Parlange S (dashed).
  2. SORPTIVITY check: the extrapolated S0 (pen->0) vs analytical Parlange, all 4 soils (the <1.5% match).
  3. THE ASYMMETRY (headline): desorptivity/sorptivity ratio per soil < 1, coarse soils most asymmetric
     -- a sign-symmetric Kirchhoff sigma cannot serve both directions (the Phase-3 design driver).
  4. DISPERSE vs DRAIN uptake I(t): disperse wets faster than drain drains (same |dH|, opposite sign).
  5. GEOMETRY anchors: tunnel(radial) vs planar desorptivity (->1 validates the r-weighting).

Usage:  python viz/make_phase1_curves_html.py [out.html]
Run from forward-model/.  Dependencies: numpy + plotly (vendored inline -> offline double-click).
"""
from __future__ import annotations

import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

R_W = 0.05
DISP = "scratch/m4_phase1b_disperse_refs.npz"
DRAIN = "scratch/m4_phase1c_drain_refs.npz"
# Analytical Parlange (1975) sorptivities over [-1,0] m (fixed constants; the independent ground truth).
PARLANGE = {"SAND": 0.49068, "LOAM": 0.09806, "SILT": 0.04453, "CLAY": 0.01501}
COLOR = {"SAND": "#d9a441", "LOAM": "#4c9a4c", "SILT": "#4178c0", "CLAY": "#8a5a3b"}
SOILS = ["SAND", "LOAM", "SILT", "CLAY"]


def _intercept(t, I, pen, window=0.6 * R_W):
    """Extrapolate I/sqrt(t) ~ S0*(1 + c*pen/r_w) to zero penetration (the planar-equivalent S)."""
    m = (t > 0) & (pen > 0) & (pen < window)
    if m.sum() < 3:
        return float("nan")
    x = pen[m] / R_W
    y = I[m] / np.sqrt(t[m])
    return float(np.polyfit(x, y, 1)[1])


def build(out_html):
    d = dict(np.load(DISP))
    c = dict(np.load(DRAIN))

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "1. Disperse = Philip absorption:  I(t) vs √t  (tunnel; dashed = analytical Parlange S√t)",
            "2. Sorptivity check:  extrapolated S₀  vs  analytical Parlange",
            "3. THE ASYMMETRY:  desorptivity / sorptivity  (&lt;1; coarse soils most asymmetric)",
            "4. Disperse vs Drain uptake I(t)  (same |ΔH|=1 m, opposite sign; log-log)",
            "5. Cumulative disperse/drain ratio over time  (grows → transient asymmetry)",
            "6. Geometry anchor:  radial(tunnel) / planar desorptivity  (→1 validates r-weighting)",
        ),
        vertical_spacing=0.10, horizontal_spacing=0.09,
    )

    # ---- Panel 1: Philip I vs sqrt(t) (disperse tunnel) ----
    for s in SOILS:
        t = d[f"{s}_t"]; I = d[f"{s}_tunnel_I"]
        rt = np.sqrt(t)
        fig.add_trace(go.Scatter(x=rt, y=I, mode="lines+markers", name=s, legendgroup=s,
                                 line=dict(color=COLOR[s]), marker=dict(size=4)), row=1, col=1)
        # analytical Parlange S*sqrt(t) reference (dashed)
        fig.add_trace(go.Scatter(x=rt, y=PARLANGE[s] * rt, mode="lines", showlegend=False, legendgroup=s,
                                 line=dict(color=COLOR[s], dash="dash", width=1)), row=1, col=1)
    fig.update_xaxes(title_text="√t  (√day)", row=1, col=1)
    fig.update_yaxes(title_text="uptake I  (m)", row=1, col=1)

    # ---- Panel 2: extrapolated S0 vs Parlange (bars) ----
    s0 = [_intercept(d[f"{s}_t"], d[f"{s}_tunnel_I"], d[f"{s}_tunnel_pen"]) for s in SOILS]
    par = [PARLANGE[s] for s in SOILS]
    fig.add_trace(go.Bar(x=SOILS, y=par, name="Parlange S (analytical)", marker_color="#888",
                         showlegend=True), row=1, col=2)
    fig.add_trace(go.Bar(x=SOILS, y=s0, name="extrapolated S₀ (FEM)",
                         marker_color=[COLOR[s] for s in SOILS], showlegend=True), row=1, col=2)
    for i, s in enumerate(SOILS):
        err = 100 * (s0[i] - par[i]) / par[i]
        fig.add_annotation(x=s, y=max(s0[i], par[i]), text=f"{err:+.1f}%", showarrow=False,
                           yshift=10, font=dict(size=9), row=1, col=2)
    fig.update_yaxes(title_text="sorptivity  (m/√day)", type="log", row=1, col=2)

    # ---- Panel 3: desorptivity/sorptivity ratio (the headline) ----
    rt_tun = [_intercept(c[f"{s}_t"], c[f"{s}_tunnel_I"], c[f"{s}_tunnel_pen"]) / PARLANGE[s] for s in SOILS]
    rt_pl = [_intercept(c[f"{s}_t"], c[f"{s}_planar_I"], c[f"{s}_tunnel_pen"]) / PARLANGE[s]
             if f"{s}_planar_I" in c else float("nan") for s in SOILS]
    fig.add_trace(go.Bar(x=SOILS, y=rt_tun, name="desorptivity/sorptivity (tunnel)",
                         marker_color=[COLOR[s] for s in SOILS], showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=SOILS, y=rt_pl, mode="markers", name="planar anchor",
                             marker=dict(symbol="diamond", size=11, color="black")), row=2, col=1)
    fig.add_hline(y=1.0, line=dict(color="red", dash="dot"), row=2, col=1)
    fig.add_annotation(x="CLAY", y=1.0, text="symmetric (=1)", showarrow=False, yshift=8,
                       font=dict(size=9, color="red"), row=2, col=1)
    fig.update_yaxes(title_text="ratio  (<1 ⇒ drying slower)", range=[0, 1.1], row=2, col=1)

    # ---- Panel 4: disperse vs drain I(t) (log-log) ----
    for s in SOILS:
        td = d[f"{s}_t"]; Idisp = d[f"{s}_tunnel_I"]
        tc = c[f"{s}_t"]; Idr = c[f"{s}_tunnel_I"]
        fig.add_trace(go.Scatter(x=td, y=Idisp, mode="lines", name=f"{s} disperse", legendgroup=s,
                                 line=dict(color=COLOR[s])), row=2, col=2)
        fig.add_trace(go.Scatter(x=tc, y=Idr, mode="lines", name=f"{s} drain", legendgroup=s,
                                 line=dict(color=COLOR[s], dash="dash")), row=2, col=2)
    fig.update_xaxes(title_text="t  (day)", type="log", row=2, col=2)
    fig.update_yaxes(title_text="uptake I per wall area  (m)", type="log", row=2, col=2)

    # ---- Panel 5: cumulative disperse/drain ratio over time ----
    for s in SOILS:
        tc = c[f"{s}_t"]; Idr = c[f"{s}_tunnel_I"]
        Idisp_i = np.interp(tc, d[f"{s}_t"], d[f"{s}_tunnel_I"])
        fig.add_trace(go.Scatter(x=tc, y=Idisp_i / Idr, mode="lines+markers", name=s, legendgroup=s,
                                 showlegend=False, line=dict(color=COLOR[s]), marker=dict(size=3)),
                      row=3, col=1)
    fig.update_xaxes(title_text="t  (day)", type="log", row=3, col=1)
    fig.update_yaxes(title_text="I_disperse / I_drain", row=3, col=1)

    # ---- Panel 6: radial/planar anchor ----
    anchor = []
    for s in SOILS:
        st = _intercept(c[f"{s}_t"], c[f"{s}_tunnel_I"], c[f"{s}_tunnel_pen"])
        sp = _intercept(c[f"{s}_t"], c[f"{s}_planar_I"], c[f"{s}_tunnel_pen"]) if f"{s}_planar_I" in c else np.nan
        anchor.append(st / sp if sp else np.nan)
    fig.add_trace(go.Bar(x=SOILS, y=anchor, marker_color=[COLOR[s] for s in SOILS], showlegend=False),
                  row=3, col=2)
    fig.add_hline(y=1.0, line=dict(color="green", dash="dot"), row=3, col=2)
    fig.update_yaxes(title_text="radial S_des / planar S_des", range=[0.9, 1.2], row=3, col=2)

    fig.update_layout(
        title_text="PIDS Module-4 §E — Phase-1 near-field reference suite (sorptivity / desorptivity ground truth)",
        height=1150, width=1350, barmode="group", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.07, font=dict(size=10)),
    )
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    print(f"WROTE {out_html}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "../validation/sanity/viz/m4_phase1_curves__2026-06-08.html"
    build(out)
