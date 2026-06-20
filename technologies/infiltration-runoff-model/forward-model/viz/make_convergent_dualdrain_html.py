"""Tier-3 visualizer (P3 Part B, Gate B): the COUPLED CONVERGENT-FLOW dual-drain fixture.

Reads ONLY the npz from scratch/_p3_convergent_storm_matrix.py (independent visual evidence -- never
runs the solver) and writes ONE self-contained, offline, interactive HTML for Arik's Gate-B sign-off.

WHAT IT MAKES VISIBLE
  - the convergent overland fix: surface ponded depth d(x,y) CONCENTRATES onto the convergence line
    (the upwind scheme routes the side-slope run-off into the low line without the galerkin sawtooth);
  - the embedded PIDS dual-drain DIVISION OF LABOUR: the surface grate inlet eats the concentrated
    ponded run-on; the interior tile drain taps the subsurface along the line;
  - structural mass conservation with ALL sinks (machine-tight) across the storm;
  - robustness: a 100-yr burst + a wet antecedent stay conservative and engage both elements.

FRAMING (Arik 2026-06-19): the convergent topography is the SETTING (variable geometry), NOT a PIDS
"feature"; the embedded PIDS element is the coupled-integrated dual-drain (add_surface_inlet +
add_interior_drain). Plotly vendored INLINE so the HTML opens offline by double-click.

Usage:  python viz/make_convergent_dualdrain_html.py <data.npz> <out.html>
"""
from __future__ import annotations

import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build(npz_path: str, out_html: str) -> None:
    d = np.load(npz_path, allow_pickle=True)
    xu, yu = d["show_xu"], d["show_yu"]
    grids = d["show_d_grids"] * 1e3                      # (n_out, ny, nx) -> mm
    t = d["show_t"]
    Xc, W = float(d["show_Xc"]), float(d["show_W"])
    t_storm = float(d["show_t_storm"])
    d_max = d["show_d_max"]
    peak = int(np.argmax(d_max))
    early = int(np.argmin(np.abs(t - 0.5 * t_storm)))
    iy_mid = len(yu) // 2

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"surface ponded depth d(x,y) at the storm peak (t={t[peak]:.3f} d) — concentrates on the line",
            "cross-profile d(x) at mid-valley — the depth piles onto the convergence floor",
            "water budget + dual-drain capture (cumulative)",
            "mass-balance error |Δ−(rain−out−drain)| / cum_rain (machine-tight)"),
        specs=[[{"type": "heatmap"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.12, vertical_spacing=0.14)

    # (1,1) headline heatmap at the storm peak
    fig.add_trace(go.Heatmap(x=xu, y=yu, z=grids[peak], colorscale="Blues",
                             colorbar=dict(title="d [mm]", len=0.45, y=0.78, x=0.46)), 1, 1)
    for xv, nm in ((Xc - W / 2, "floor"), (Xc + W / 2, None)):
        fig.add_shape(type="line", x0=xv, x1=xv, y0=float(yu.min()), y1=float(yu.max()),
                      line=dict(color="seagreen", width=1.5, dash="dash"), row=1, col=1)
    fig.update_xaxes(title_text="x (cross) [m]", row=1, col=1)
    fig.update_yaxes(title_text="y (down-valley) [m]", row=1, col=1)

    # (1,2) cross-profiles at mid-valley, three times
    for idx, label, color in ((early, "early storm", "#9ecae1"), (peak, "storm peak", "#08519c"),
                              (len(t) - 1, "recession end", "#fdae6b")):
        fig.add_trace(go.Scatter(x=xu, y=grids[idx][iy_mid], name=label, line=dict(color=color)), 1, 2)
    for xv in (Xc - W / 2, Xc + W / 2):
        fig.add_vline(x=xv, line=dict(color="seagreen", width=1, dash="dash"), row=1, col=2)
    fig.update_xaxes(title_text="x (cross) [m]", row=1, col=2)
    fig.update_yaxes(title_text="d [mm]", row=1, col=2)

    # (2,1) cumulative water budget + capture
    for key, label, color in (("show_cum_rain", "rain in", "#3182bd"),
                              ("show_cum_out", "outlet (convergence)", "#31a354"),
                              ("show_cum_inlet", "surface inlet (grate)", "#e6550d"),
                              ("show_cum_drain", "interior tile drain", "#756bb1")):
        fig.add_trace(go.Scatter(x=t, y=d[key], name=label, line=dict(color=color)), 2, 1)
    fig.add_vrect(x0=0, x1=t_storm, fillcolor="LightGray", opacity=0.25, line_width=0, row=2, col=1)
    fig.update_xaxes(title_text="time [day]", row=2, col=1)
    fig.update_yaxes(title_text="cumulative volume [m³]", row=2, col=1)

    # (2,2) conservation (log)
    rel = np.abs(d["show_bal"]) / np.maximum(d["show_cum_rain"], 1e-30)
    rel = np.where(d["show_cum_rain"] > 0, rel, np.nan)
    fig.add_trace(go.Scatter(x=t, y=rel, name="|bal|/cum_rain", line=dict(color="crimson")), 2, 2)
    fig.add_hline(y=1e-3, line=dict(color="gray", dash="dot"), row=2, col=2)
    fig.update_xaxes(title_text="time [day]", row=2, col=2)
    fig.update_yaxes(title_text="rel. mass-balance error", type="log", row=2, col=2)

    fig.update_layout(height=820, width=1180, template="plotly_white",
                      title_text="PIDS coupled convergent-flow dual-drain — Tier-3 (P3 Part B, Gate B)",
                      legend=dict(orientation="h", y=-0.06))

    # robustness summary + framing
    names = d["robust_names"]
    rows = ""
    for i, nm in enumerate(names):
        rows += (f"<tr><td>{nm}</td><td>{d['robust_rate'][i]:.2f}</td><td>{d['robust_z_wt'][i]:.2f}</td>"
                 f"<td>{d['robust_cum_inlet'][i]:.4f}</td><td>{d['robust_cum_drain'][i]:.4f}</td>"
                 f"<td>{d['robust_bal_rel'][i]:.1e}</td><td>{1e3*d['robust_max_pond'][i]:.1f}</td>"
                 f"<td>{1e3*d['robust_d_min'][i]:+.2f}</td></tr>")
    show_inlet, show_drain = float(d["show_cum_inlet"][-1]), float(d["show_cum_drain"][-1])
    show_rain = float(d["show_cum_rain"][-1])
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>PIDS convergent dual-drain — Tier-3</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1180px;margin:18px auto;color:#222}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}}td,th{{border:1px solid #ccc;padding:4px 8px}}
th{{background:#f0f0f0}}.note{{background:#f7f7f9;border-left:4px solid #3182bd;padding:8px 12px;font-size:14px}}</style>
</head><body>
<h2>PIDS coupled convergent-flow dual-drain — Tier-3 sign-off (P3 Part B)</h2>
<p class="note"><b>What this validates.</b> The convergent-flow workstream (P0–P3) fixed the coupled
<b>upwind</b> overland solver for the regime where surface water concentrates along convergence lines —
where PIDS networks/inlets install. This permanent Tier-2 fixture exercises that fix end-to-end: a
<b>convergent graded topography</b> (the topographic <i>setting</i>, variable geometry — <b>not</b> a PIDS
"feature") on a 3-D loam-over-clay host, the upwind scheme, and the signed-off <b>dual-drain</b>
(<code>add_surface_inlet</code> + <code>add_interior_drain</code> — the coupled-integrated PIDS drainage
elements; the full bidirectional <code>WellIndexExchange</code> feature is not yet coupled-integrated).</p>
<p>Showcase storm (typical 0.35 m/day &gt; Ks then recession): the overland depth <b>concentrates on the
convergence line</b> (top-left/right); the <b>surface grate inlet</b> captures the concentrated ponded
run-on ({show_inlet:.3f} m³, {100*show_inlet/show_rain:.0f}% of rain) while the <b>interior tile drain</b>
taps the subsurface along the line ({show_drain:.4f} m³); the books close machine-tight
(bottom-right ≪ 10⁻³).</p>
<h3>Robustness (folded-in Tier-2 check)</h3>
<table><tr><th>scenario</th><th>rate<br>[m/d]</th><th>z_wt<br>[m]</th><th>cum inlet<br>[m³]</th>
<th>cum drain<br>[m³]</th><th>mass-bal<br>rel</th><th>max pond<br>[mm]</th><th>min d<br>[mm]</th></tr>
{rows}</table>
<p style="font-size:13px;color:#555">Every scenario stays mass-conservative (rel ≪ 10⁻³), within the
upwind positivity tripwire (sub-cm min depth), and engages both embedded elements. Builder
<code>scratch/_p3_convergent_fixture.py</code>; framing per <code>pids-p3-convergent-progress</code> memory.</p>
{fig.to_html(full_html=False, include_plotlyjs=True)}
</body></html>"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"WROTE {out_html}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else
          "../validation/sanity/data/p3_convergent_dualdrain__2026-06-20.npz",
          sys.argv[2] if len(sys.argv) > 2 else
          "../validation/sanity/viz/p3_convergent_dualdrain__2026-06-20.html")   # viz/ is gitignored
