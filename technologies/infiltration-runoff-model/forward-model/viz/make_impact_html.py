"""Tier-3 visualizer for the Module-4 PIDS IMPACT experiments (scratch/m4_impact_experiments.npz).

Reads ONLY the result npz. Shows the WITH-vs-WITHOUT-PIDS impact of one embedded vertical feature on a
3-D loam block, both modes:
  Row 1: DRAIN  -- θ(x,z) cross-section, WITHOUT (trapped waterlog) vs WITH (drained zone around the feature)
  Row 2: DISPERSE -- θ(x,z), WITHOUT (stays dry) vs WITH (wetted halo -- subsurface irrigation)
  Row 3: cumulative soil-water change Δ(t) (drain removes water; disperse delivers it)
  Row 4: σ (wall-exchange) sensitivity -- the exchange-limited → soil-transport-limited transition

CAVEAT (in the title): Phase-2 SIMPLE constant σ (calibrated Kirchhoff closure = Phase 3), bare host
Richards block (surface inlet + overland = Phase 4). Direction/trend are real; magnitude is uncalibrated.

Usage:  python viz/make_impact_html.py [out.html]   (run from forward-model/)
"""
from __future__ import annotations

import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NPZ = "scratch/m4_impact_experiments.npz"
FEATURE_X = 0.5


def _grid(xz, theta):
    x = np.unique(np.round(xz[:, 0], 6)); z = np.unique(np.round(xz[:, 1], 6))
    G = np.full((z.size, x.size), np.nan)
    ix = np.searchsorted(x, np.round(xz[:, 0], 6)); iz = np.searchsorted(z, np.round(xz[:, 1], 6))
    G[iz, ix] = theta
    return x, z, G


def build(out_html):
    d = np.load(NPZ)

    fig = make_subplots(
        rows=4, cols=2, row_heights=[0.28, 0.28, 0.22, 0.22], vertical_spacing=0.075, horizontal_spacing=0.12,
        subplot_titles=(
            "DRAIN — θ(x,z) WITHOUT feature (trapped waterlog)", "DRAIN — θ(x,z) WITH feature (drained)",
            "DISPERSE — θ(x,z) WITHOUT feature (stays dry)", "DISPERSE — θ(x,z) WITH feature (irrigated halo)",
            "cumulative soil-water change Δ(t)  [L per m³-block]", "",
            "σ (wall-exchange) sensitivity — drain", ""),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}],
               [{"colspan": 2}, None],
               [{"colspan": 2}, None]])

    # theta colour range shared per row from the data
    for row, mode in ((1, "drain"), (2, "disperse")):
        th_all = np.concatenate([d[f"{mode}_without_theta"], d[f"{mode}_with_theta"]])
        cmin, cmax = float(th_all.min()), float(th_all.max())
        for col, wf in ((1, "without"), (2, "with")):
            x, z, G = _grid(d[f"{mode}_{wf}_xz"], d[f"{mode}_{wf}_theta"])
            fig.add_trace(go.Heatmap(x=x, y=z, z=G, colorscale="YlGnBu", zmin=cmin, zmax=cmax,
                                     colorbar=dict(title="θ", len=0.22, y=0.88 - (row - 1) * 0.28, x=1.02)),
                          row=row, col=col)
            fig.add_vline(x=FEATURE_X, line=dict(color="red", width=2, dash="dot"), row=row, col=col)
            fig.update_xaxes(title_text="x (m)", row=row, col=col)
            fig.update_yaxes(title_text="z (m, surface=top)", row=row, col=col)

    # Δsoil-water timeseries
    style = {("drain", "with"): ("#c0392b", None), ("drain", "without"): ("#c0392b", "dash"),
             ("disperse", "with"): ("#2471a3", None), ("disperse", "without"): ("#2471a3", "dash")}
    for (mode, wf), (color, dash) in style.items():
        fig.add_trace(go.Scatter(x=d[f"{mode}_{wf}_t"], y=d[f"{mode}_{wf}_sw"] * 1000,
                                 name=f"{mode} {wf} PIDS", line=dict(color=color, dash=dash, width=2.5)),
                      row=3, col=1)
    fig.update_xaxes(title_text="time (day)", row=3, col=1)
    fig.update_yaxes(title_text="Δ soil water (L)", row=3, col=1)

    # σ sweep
    fig.add_trace(go.Scatter(x=d["sweep_sigma"], y=-d["sweep_dsw"] * 1000, mode="lines+markers",
                             name="drain vs σ", line=dict(color="#117a65", width=2.5),
                             marker=dict(size=8), showlegend=False), row=4, col=1)
    fig.add_annotation(x=np.log10(0.05), y=130, text="knee ~σ=0.05<br>(exchange→transport limited)",
                       showarrow=True, arrowhead=2, ax=60, ay=-25, font=dict(size=9), row=4, col=1)
    fig.update_xaxes(title_text="wall-exchange σ", type="log", row=4, col=1)
    fig.update_yaxes(title_text="water drained (L)", row=4, col=1)

    fig.update_layout(
        title_text="PIDS impact (Module-4 §E, Phase-2 primitive) — one embedded vertical feature in a loam "
                   "block · WITH vs WITHOUT · drain & disperse<br>"
                   "<sub>RELATIVE assessment: simple constant σ (Kirchhoff calibration = Phase 3), bare host "
                   "Richards (surface inlet + overland = Phase 4) — direction/trend real, magnitude uncalibrated</sub>",
        height=1280, width=1180, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.03))
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    print(f"WROTE {out_html}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "../validation/sanity/viz/m4_impact__2026-06-08.html"
    build(out)
