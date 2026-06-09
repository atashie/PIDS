"""Tier-3 visualizer for the NO-PIDS vertical-profile baseline (scratch/m4_vertical_baseline.npz).

Reads ONLY the result npz. Shows the space-time evolution of a 1-D layered column under a storm -- the
'before PIDS' control whose theta/psi + ponding + infiltration the eventual WITH-PIDS run overlays:
  Row 1: psi(z,t) heatmap (water table = psi=0 contour; loam/clay split line) -- the perched mound.
  Row 2: theta(z,t) heatmap -- the wetting front + the saturated perch above the clay.
  Row 3: time series -- rainfall (inverted), surface PONDING depth, cumulative INFILTRATION.

Usage:  python viz/make_phase1_vertical_html.py [out.html]   (run from forward-model/)
"""
from __future__ import annotations

import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NPZ = "scratch/m4_vertical_baseline.npz"


def build(out_html):
    d = np.load(NPZ)
    z, t = d["z"], d["t"]
    psi, th = d["psi_zt"].T, d["theta_zt"].T     # -> (z, t)
    zsplit = float(d["z_split"]); storm = float(d["storm"])
    H = float(z.max())

    fig = make_subplots(
        rows=3, cols=1, row_heights=[0.34, 0.34, 0.32], vertical_spacing=0.085,
        subplot_titles=(
            "ψ(z, t)  pressure head [m]  (water table = ψ=0 contour; dashed = loam/clay interface)",
            "θ(z, t)  water content [-]  (saturated perch builds above the clay)",
            "surface ponding & cumulative infiltration  (rainfall inverted on the right axis)"),
        specs=[[{"type": "heatmap"}], [{"type": "heatmap"}], [{"secondary_y": True}]])

    fig.add_trace(go.Heatmap(x=t, y=z, z=psi, colorscale="RdBu", zmid=0.0,
                             colorbar=dict(title="ψ (m)", len=0.30, y=0.85)), row=1, col=1)
    fig.add_trace(go.Contour(x=t, y=z, z=psi, showscale=False, contours=dict(start=0, end=0, coloring="none"),
                             line=dict(color="black", width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=t, y=z, z=th, colorscale="YlGnBu",
                             colorbar=dict(title="θ", len=0.30, y=0.5)), row=2, col=1)
    for r in (1, 2):
        fig.add_hline(y=zsplit, line=dict(color="black", dash="dash", width=1), row=r, col=1)
        fig.update_yaxes(title_text="elevation z (m, surface=top)", row=r, col=1)

    fig.add_trace(go.Scatter(x=t, y=d["ponding"] * 100, name="ponding depth (cm)",
                             line=dict(color="#1f77b4", width=2.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=d["infiltration"] * 100, name="cumulative infiltration (cm-equiv)",
                             line=dict(color="#2ca02c", width=2.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=d["rain"] * 1000, name="rainfall (mm/day)", line=dict(color="#888", dash="dot"),
                             fill="tozeroy", fillcolor="rgba(120,120,120,0.15)"), row=3, col=1, secondary_y=True)
    fig.add_vline(x=storm, line=dict(color="grey", dash="dot"), row=3, col=1)
    fig.add_annotation(x=storm, y=1.0, yref="y3 domain", text="storm ends", showarrow=False,
                       font=dict(size=9), xshift=28)
    fig.update_xaxes(title_text="time (day)", row=3, col=1)
    fig.update_yaxes(title_text="depth (cm)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="rain (mm/day)", autorange="reversed", row=3, col=1, secondary_y=True)

    fig.update_layout(
        title_text="PIDS Pillar-2 — NO-PIDS vertical-profile baseline (loam over clay, 0.3 m/day storm): "
                   "the waterlogging a PIDS vertical feature will address  [Phase-2 = the with-PIDS overlay]",
        height=1080, width=1150, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.06))
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    print(f"WROTE {out_html}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "../validation/sanity/viz/m4_vertical_baseline__2026-06-08.html"
    build(out)
