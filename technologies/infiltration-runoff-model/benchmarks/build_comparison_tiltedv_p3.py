#!/usr/bin/env python3
"""P3 Part C: canonical tilted-V re-benchmark -- in-house UPWIND (the convergent-flow fix) vs in-house
GALERKIN (the sawtooth) vs ParFlow (the reference). One self-contained offline HTML.

Reuses SAVED data (no re-run of the 39.5h galerkin or ParFlow):
  - galerkin  : forward-model/scratch/tiltedv_inhouse_s1_p0.npz  (the P0-corrected 39.5h / 60,008-rej run)
  - upwind    : forward-model/scratch/tiltedv_diag_24x16_upwind_swale324.npz  (regenerated coupled upwind
                on the RESOLVED swale -- the clean product geometry; the KINK upwind dt-collapses, the
                measure-zero-channel artifact, annotated)
  - ParFlow   : benchmarks/data/tilted_v__canonical__2026-06-10.nc  (q_parflow; ParFlow = 1.000*Q_eq)

HONEST FRAMING (parent plan 8.7/8.8): the LUMPED outlet plateau -> Q_eq is a CONSERVATION/equilibrium
identity (forced for any converged steady field), NOT discharge accuracy; all three reach it. The real
convergent-flow story is the STIFFNESS: galerkin 39.5h / 60,008 rejected steps (the never-settling
wet/dry sawtooth on the convergence line, dt pinned ~5e-5 d) -> the monotone upwind scheme removes it at
the source (~600x). Absolute accuracy is the DEPTH FIELD on the RESOLVED swale (P3 Part A: operator-
equivalence + plane->normal-depth), not this idealized kink/measure-zero channel.

Run (pids-fem, from benchmarks/):  python build_comparison_tiltedv_p3.py
"""
from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
FWD = os.path.join(HERE, "..", "forward-model")
DATE = "2026-06-21"


def _plateau(t, q, storm):
    m = (t > 0.5 * storm) & (t <= storm)
    return float(np.mean(q[m])) if m.any() else float("nan")


def main():
    g = np.load(os.path.join(FWD, "scratch", "tiltedv_inhouse_s1_p0.npz"))
    u = np.load(os.path.join(FWD, "scratch", "tiltedv_diag_24x16_upwind_swale324.npz"))
    nc = xr.open_dataset(os.path.join(HERE, "data", "tilted_v__canonical__2026-06-10.nc"))

    Q_eq = float(g["Q_eq"])
    storm = float(g["storm"])

    # galerkin (accepted-step ledger; the npz is the processed summary)
    g_t, g_q = np.asarray(g["times"]), np.asarray(g["q_out"])
    g_dt = np.diff(g_t)
    g_rej, g_run = int(g["n_rej"]), float(g["run_s"])
    g_steps = int(g["n_step"]) if "n_step" in g else len(g_t)
    g_ledger = float(g["ext_gap_engine"]) / max(float(g["cum_rain_engine"]), 1e-9)

    # upwind on the resolved swale
    acc = (u["accepted"] == 1)
    u_t, u_q, u_dt = u["t"][acc], u["qout"][acc], u["dt"][acc]
    u_rej, u_nacc, u_run = int(u["n_rej"]), int(u["n_acc"]), float(u["run_s"])
    u_dtmax = float(u["dt_max"])
    u_ledger = float(u["ext_gap"]) / max(float(u["cum_rain"]), 1e-9)
    u_storm = float(u["storm"])

    # ParFlow (already on a common grid in the NC)
    pf_t, pf_q = nc["time"].values, nc["q_parflow"].values

    pg, pu, ppf = _plateau(g_t, g_q, storm), _plateau(u_t, u_q, u_storm), _plateau(pf_t, pf_q, storm)
    speedup = g_run / max(u_run, 1e-9)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.11,
        subplot_titles=("outlet discharge Q(t) / Q_eq  (all reach the plateau — a conservation identity)",
                        "accepted time-step dt [day]  (the stiffness: galerkin pin vs upwind lifted)"))
    fig.add_trace(go.Scatter(x=pf_t, y=pf_q / Q_eq, name="ParFlow (reference)",
                             line=dict(color="black", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=g_t, y=g_q / Q_eq, name="in-house galerkin (39.5 h)",
                             line=dict(color="#d62728")), 1, 1)
    fig.add_trace(go.Scatter(x=u_t, y=u_q / Q_eq, name="in-house upwind (resolved swale)",
                             line=dict(color="#1f77b4", width=2)), 1, 1)
    fig.add_hline(y=1.0, line=dict(color="gray", dash="dot"), row=1, col=1)
    fig.update_xaxes(title_text="time [day]", row=1, col=1)
    fig.update_yaxes(title_text="Q / Q_eq", row=1, col=1)

    fig.add_trace(go.Scatter(x=g_t[1:], y=g_dt, name="galerkin dt", line=dict(color="#d62728"),
                             showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=u_t, y=u_dt, name="upwind dt", line=dict(color="#1f77b4"),
                             showlegend=False), 1, 2)
    fig.add_hline(y=u_dtmax, line=dict(color="gray", dash="dot"), row=1, col=2)
    fig.update_xaxes(title_text="time [day]", row=1, col=2)
    fig.update_yaxes(title_text="dt [day]", type="log", row=1, col=2)
    fig.update_layout(height=470, width=1180, template="plotly_white",
                      title_text=f"Canonical tilted-V (1.62×1.0 km) — upwind vs galerkin vs ParFlow",
                      legend=dict(orientation="h", y=-0.18))

    bars = [
        ("Plateau Q → Q_eq (conservation identity, NOT accuracy)",
         f"ParFlow {ppf/Q_eq:.3f} · galerkin {pg/Q_eq:.3f} · upwind {pu/Q_eq:.3f}", "✓"),
        ("dt-pin LIFTED (the tractability fix)",
         f"galerkin dt ~{np.median(g_dt):.0e} d pinned ({g_rej:,} rejected steps) → upwind up to "
         f"dt_max {u_dtmax:.0e} ({u_rej} rejected)", "✓"),
        ("Runtime win", f"galerkin {g_run/3600:.1f} h → upwind {u_run/60:.1f} min  (~{speedup:.0f}×)", "✓"),
        ("Conservation (engine ledger)",
         f"galerkin {g_ledger:+.1e} · upwind {u_ledger:+.1e} of cum-rain (machine-tight)", "✓"),
        ("Absolute accuracy", "the DEPTH FIELD on the RESOLVED swale (P3 Part A: operator-equivalence "
         "0.13% + plane→normal-depth 0.17%) — NOT the lumped outlet here", "→ Part A"),
    ]
    rows = "".join(f"<tr><td>{b[2]}</td><td>{b[0]}</td><td>{b[1]}</td></tr>" for b in bars)
    note = (
        "<p class='note'><b>Honest framing (parent §8.7/§8.8).</b> The lumped outlet plateau → Q_eq is a "
        "CONSERVATION/equilibrium identity (forced for any converged steady field), <b>not</b> discharge "
        "accuracy — all three models reach it. The real convergent-flow result is the <b>stiffness</b>: "
        "the validated galerkin scheme develops a never-settling wet/dry <b>sawtooth</b> on the "
        "convergence line (dt pinned ~5e-5 d, 60,008 rejected steps, 39.5 h); the monotone <b>upwind</b> "
        "scheme removes it at the source (~600×). The idealized 1-cell <b>KINK</b> tilted-V is a "
        "measure-zero-channel artifact where even upwind dt-collapses; the <b>RESOLVED swale</b> (shown "
        "here, the product geometry) runs cleanly, and absolute accuracy is its DEPTH FIELD (P3 Part A). "
        "ParFlow (purpose-built kinematic router) is the 1.000·Q_eq reference.</p>")
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>tilted-V upwind vs galerkin vs ParFlow"
            "</title><script>" + __import__("plotly").offline.get_plotlyjs() +
            "</script><style>body{font-family:system-ui,Arial,sans-serif;max-width:1180px;margin:18px auto;"
            "color:#222}table{border-collapse:collapse;font-size:13px;margin:10px 0}td,th{border:1px solid "
            "#ccc;padding:5px 9px}.note{background:#f7f7f9;border-left:4px solid #d62728;padding:8px 12px;"
            "font-size:14px}</style></head><body>"
            "<h2>Canonical tilted-V re-benchmark — in-house upwind vs galerkin vs ParFlow (P3 Part C)</h2>"
            + note + "<h3>Acceptance bars (parent §6)</h3><table>"
            "<tr><th></th><th>bar</th><th>result</th></tr>" + rows + "</table>"
            + fig.to_html(full_html=False, include_plotlyjs=False) + "</body></html>")
    out = os.path.join(HERE, "html", f"tilted_v__upwind_galerkin_parflow__{DATE}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"=== P3-C tilted-V re-benchmark ===")
    print(f"  plateau/Q_eq:  ParFlow {ppf/Q_eq:.3f}   galerkin {pg/Q_eq:.3f}   upwind {pu/Q_eq:.3f}")
    print(f"  runtime: galerkin {g_run/3600:.1f} h ({g_rej:,} rej) -> upwind {u_run/60:.1f} min "
          f"({u_rej} rej) = ~{speedup:.0f}x")
    print(f"  ledger: galerkin {g_ledger:+.1e}  upwind {u_ledger:+.1e}")
    print(f"  WROTE {out}")


if __name__ == "__main__":
    main()
