"""Tier-3 field visualizer for the Phase-1 near-field references: the 2-D ANNULUS theta(y,z) fields.

Reads ONLY scratch/m4_phase1_field_snapshots.npz (node y,z + theta for SAND disperse & drain). Shows,
side by side, the moisture field around a single feature:
  * DISPERSE (saturated wall wetting dry soil): a ~RADIALLY SYMMETRIC wetting shell (gravity negligible
    at the dry soil's low K) -- concentric, top == bottom.
  * DRAIN (draining saturated soil): a gravity-ASYMMETRIC desaturation (the saturated K=Ks gravity
    throughflow skews drying top vs bottom) -- the spatial counterpart of drain/tunnel ratio ~1.5.
A third panel plots theta vs polar angle at a fixed radius: disperse ~flat, drain varies (the asymmetry).

Usage:  python viz/make_phase1_fields_html.py [out.html]   (run from forward-model/)
Dependencies: numpy + plotly (inline -> offline).
"""
from __future__ import annotations

import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NPZ = "scratch/m4_phase1_field_snapshots.npz"


def _circle(r, n=200):
    a = np.linspace(0, 2 * np.pi, n)
    return r * np.cos(a), r * np.sin(a)


def _angular(y, z, th, r_target, dr):
    m = np.abs(np.hypot(y, z) - r_target) < dr
    ang = (np.degrees(np.arctan2(z[m], y[m]))) % 360
    o = np.argsort(ang)
    return ang[o], th[m][o]


def build(out_html):
    z = np.load(NPZ)
    r_w, r_out = float(z["r_w"]), float(z["r_out"])
    ths, thr = float(z["theta_s"]), float(z["theta_r"])
    cmin, cmax = thr, ths

    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.36, 0.36, 0.28],
        subplot_titles=("DISPERSE θ(y,z): wall wets dry soil → radially symmetric",
                        "DRAIN θ(y,z): draining saturated soil → gravity-asymmetric",
                        "θ vs polar angle at r = r_w+2cm"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
        horizontal_spacing=0.06,
    )

    for col, tag, title in ((1, "disp", "disperse"), (2, "drain", "drain")):
        y, zc, th = z[f"{tag}_y"], z[f"{tag}_z"], z[f"{tag}_theta"]
        # zoom to the near field (the action is within ~12 cm of the wall)
        view = 0.18
        keep = np.hypot(y, zc) < view
        fig.add_trace(go.Scattergl(
            x=y[keep], y=zc[keep], mode="markers",
            marker=dict(size=4, color=th[keep], colorscale="Viridis", cmin=cmin, cmax=cmax,
                        showscale=(col == 2), colorbar=dict(title="θ", x=0.63, len=0.9)),
            showlegend=False, hovertemplate="y=%{x:.3f} z=%{y:.3f} θ=%{marker.color:.3f}<extra></extra>"),
            row=1, col=col)
        for r, c in ((r_w, "white"), (view, None)):
            cx, cy = _circle(r)
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color="black", width=1.5),
                                     showlegend=False), row=1, col=col)
        fig.update_xaxes(title_text="y (m)", range=[-view, view], row=1, col=col,
                         scaleanchor=f"y{'' if col == 1 else col}", scaleratio=1)
        fig.update_yaxes(title_text="z (m, gravity ↓)", range=[-view, view], row=1, col=col)

    # angular profiles
    for tag, name, color in (("disp", "disperse", "#4178c0"), ("drain", "drain", "#c0504d")):
        ang, th = _angular(z[f"{tag}_y"], z[f"{tag}_z"], z[f"{tag}_theta"], r_w + 0.02, 0.004)
        fig.add_trace(go.Scatter(x=ang, y=th, mode="markers", name=name, marker=dict(color=color, size=4)),
                      row=1, col=3)
    for a, lbl in ((90, "top"), (270, "bottom")):
        fig.add_vline(x=a, line=dict(color="grey", dash="dot"), row=1, col=3)
        fig.add_annotation(x=a, y=1.0, yref="y domain", text=lbl, showarrow=False, font=dict(size=9), row=1, col=3)
    fig.update_xaxes(title_text="polar angle (deg: 90=top, 270=bottom)", row=1, col=3)
    fig.update_yaxes(title_text="θ", row=1, col=3)

    fig.update_layout(
        title_text="PIDS Module-4 §E Phase-1 — near-field θ fields (SAND): disperse symmetry vs drain gravity asymmetry",
        height=560, width=1450, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    print(f"WROTE {out_html}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "../validation/sanity/viz/m4_phase1_fields__2026-06-08.html"
    build(out)
