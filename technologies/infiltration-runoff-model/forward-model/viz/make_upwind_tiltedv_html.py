"""Tier-3 visual for the P2 coupled upwind tilted-V re-baseline (Convergent-flow P2-E1).

Builds a self-contained offline HTML comparing the coupled UPWIND scheme on the idealized KINK
tilted-V (the B5b measure-zero-channel artifact) vs a RESOLVED finite-width SWALE (the real PIDS
geometry), from the `_tiltedv_diag.py OVERLAND_SCHEME=upwind` npz runs. Four panels Arik inspects:

  1. outlet hydrograph q/Q_eq over the storm (the conservation/equilibrium plateau),
  2. dt over the run (log-y) -- the galerkin dt-pin (~1.5e-6) is LIFTED to DT_MAX (1e-3),
  3. per-step max surface undershoot (mm) -- kink cm-scale ARTIFACT vs resolved-swale SUB-MM,
  4. cumulative external mass-balance gap (% of cum rain) -- machine-tight both.

Run (pids-fem):
  PYTHONPATH=. python viz/make_upwind_tiltedv_html.py KINK.npz SWALE.npz OUT.html
"""
from __future__ import annotations

import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load(path):
    d = np.load(path)
    acc = d["accepted"] == 1
    return dict(
        t=d["t"][acc], dt=d["dt"][acc], q=d["qout"][acc] / float(d["Q_eq"]),
        clip=d["clipdepth"][acc] * 1e3,  # mm
        gap=np.cumsum(d["gap"][acc]) / max(float(d["cum_rain"]), 1e-9) * 100.0,  # % cum rain
        scale=float(d["scale"]), nrej=int(d["n_rej"]), nacc=int(d["n_acc"]),
        run_s=float(d["run_s"]), maxclip=float(np.max(d["clipdepth"][acc])) * 1e3,
    )


def main():
    kink_p, swale_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
    K, S = _load(kink_p), _load(swale_p)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Outlet hydrograph  q / Q_eq  (storm)",
            "Timestep dt over the run  [galerkin pin ~1.5e-6 -> DT_MAX 1e-3]",
            "Max surface undershoot per step  [mm]  (positivity)",
            "Cumulative mass-balance gap  [% of cum rain]",
        ),
    )
    series = [("KINK V (artifact)", K, "#d62728"), ("RESOLVED swale (PIDS)", S, "#1f77b4")]
    for name, D, c in series:
        fig.add_trace(go.Scatter(x=D["t"], y=D["q"], name=name, line=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=D["t"], y=D["dt"], name=name, line=dict(color=c),
                                 showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=D["t"], y=D["clip"], name=name, line=dict(color=c),
                                 showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=D["t"], y=D["gap"], name=name, line=dict(color=c),
                                 showlegend=False), row=2, col=2)
    fig.add_hline(y=1.0, line=dict(color="gray", dash="dot"), row=1, col=1)
    fig.add_hline(y=5.0, line=dict(color="green", dash="dash"),
                  annotation_text="tripwire tol 5mm", row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title_text="t [day]", row=2, col=1)
    fig.update_xaxes(title_text="t [day]", row=2, col=2)

    cap = (f"P2 coupled upwind tilted-V (canonical 1.62 km).  "
           f"KINK: undershoot {K['maxclip']:.1f} mm, {K['nrej']} rej, {K['run_s']/60:.1f} min.  "
           f"SWALE: undershoot {S['maxclip']:.2f} mm, {S['nrej']} rej, {S['run_s']/60:.1f} min.  "
           f"Galerkin baseline: 39.5 h / 60k rejections.  Both conserve to ~0%.")
    fig.update_layout(title=dict(text="PIDS P2 — coupled upwind overland: kink-V artifact vs "
                                      "resolved swale<br><sub>" + cap + "</sub>"),
                      height=820, width=1180, legend=dict(orientation="h", y=1.08))
    fig.write_html(out, include_plotlyjs=True)  # offline, double-click
    print(f"[done] -> {out}  (KINK maxclip {K['maxclip']:.1f}mm vs SWALE {S['maxclip']:.2f}mm)")


if __name__ == "__main__":
    main()
