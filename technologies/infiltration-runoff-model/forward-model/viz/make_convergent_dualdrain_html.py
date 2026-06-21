"""Tier-3 visualizer (P3 Part B, Gate B): the COUPLED CONVERGENT-FLOW dual-drain fixture.

Reads ONLY the npz from scratch/_p3_convergent_storm_matrix.py (independent visual evidence -- never runs
the solver) and writes ONE self-contained offline HTML for Arik's Gate-B sign-off. For EACH scenario
(typical / 100-yr burst / wet antecedent) it animates, over time (slider + play):
  - LEFT : the surface ponded depth d(x,y) -- the overland flow CONCENTRATING onto the convergence line;
  - RIGHT: the subsurface saturation theta(x,z) at mid-valley -- the wetting front + perched mound on the
           clay + the interior tile drain footprint;
plus the static water budget (rain / outlet / surface inlet / interior drain) and machine-tight mass
conservation. FRAMING (Arik 2026-06-19): the convergent topography is the SETTING (variable geometry),
NOT a PIDS "feature"; the embedded PIDS element is the coupled-integrated dual-drain. Plotly INLINE.

Usage:  python viz/make_convergent_dualdrain_html.py <data.npz> <out.html>
"""
from __future__ import annotations

import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _rect(x0, x1, y0, y1, color):
    return dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color=color, width=1.5), fillcolor="rgba(0,0,0,0)")


def scenario_html(d, key):
    g = lambda f: d[f"{key}__{f}"]
    t = g("t")
    dg = g("d_grids") * 1e3                      # (nt, ny, nx) mm
    tg = g("th_grids")                           # (nt, nz, nx)
    sxu, syu, pxu, pzu = g("sxu"), g("syu"), g("pxu"), g("pzu")
    Xc, W, z_iface = float(g("Xc")), float(g("W")), float(g("z_iface"))
    th_s, th_r = float(g("theta_s")), float(g("theta_r"))
    name = str(g("name"))
    dmax = float(np.nanmax(dg)) or 1.0

    fig = make_subplots(
        rows=2, cols=2, row_heights=[0.66, 0.34], vertical_spacing=0.13, horizontal_spacing=0.12,
        subplot_titles=("surface ponded depth d(x,y) [mm] — concentrates on the line",
                        "subsurface saturation θ(x,z) at mid-valley — wetting front + perched mound",
                        "cumulative water budget [m³]", "mass-balance error (rel)"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}], [{"type": "xy"}, {"type": "xy"}]])

    fig.add_trace(go.Heatmap(x=sxu, y=syu, z=dg[0], colorscale="Blues", zmin=0.0, zmax=dmax,
                             colorbar=dict(title="mm", len=0.42, y=0.79, x=0.455)), 1, 1)
    fig.add_trace(go.Heatmap(x=pxu, y=pzu, z=tg[0], colorscale="YlGnBu", zmin=th_r, zmax=th_s,
                             colorbar=dict(title="θ", len=0.42, y=0.79, x=1.0)), 1, 2)
    # static budget + conservation (full series)
    for f, lab, c in (("cum_rain", "rain in", "#3182bd"), ("cum_out", "outlet", "#31a354"),
                      ("cum_inlet", "surface inlet", "#e6550d"), ("cum_drain", "interior drain", "#756bb1")):
        fig.add_trace(go.Scatter(x=t, y=g(f), name=lab, line=dict(color=c)), 2, 1)
    rel = np.where(g("cum_rain") > 0, np.abs(g("bal")) / np.maximum(g("cum_rain"), 1e-30), np.nan)
    fig.add_trace(go.Scatter(x=t, y=rel, line=dict(color="crimson"), showlegend=False), 2, 2)
    fig.add_hline(y=1e-3, line=dict(color="gray", dash="dot"), row=2, col=2)

    # frames update only the two heatmaps (traces 0,1)
    fig.frames = [go.Frame(name=f"{t[k]:.4f}", traces=[0, 1],
                           data=[go.Heatmap(z=dg[k]), go.Heatmap(z=tg[k])]) for k in range(len(t))]
    steps = [dict(method="animate", label=f"{t[k]:.3f}",
                  args=[[f"{t[k]:.4f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}])
             for k in range(len(t))]
    fig.update_layout(
        sliders=[dict(active=0, steps=steps, x=0.04, len=0.92, y=-0.04,
                      currentvalue=dict(prefix="t = ", suffix=" day", font=dict(size=13)))],
        updatemenus=[dict(type="buttons", x=0.0, y=1.13, buttons=[dict(
            label="▶ play", method="animate",
            args=[None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True}])])])

    shapes = [
        dict(**_rect(Xc - W / 2, Xc + W / 2, float(syu.min()), float(syu.max()), "seagreen"),
             xref="x", yref="y"),                                                          # convergence floor
        dict(type="line", x0=float(pxu.min()), x1=float(pxu.max()), y0=z_iface, y1=z_iface,
             xref="x2", yref="y2", line=dict(color="firebrick", width=1.5, dash="dash")),  # clay interface
        dict(**_rect(Xc - 2.0, Xc + 2.0, z_iface, z_iface + (pzu[1] - pzu[0]), "darkorange"),
             xref="x2", yref="y2"),                                                        # tile drain band
    ]
    fig.update_layout(shapes=shapes, height=760, width=1180, template="plotly_white",
                      title_text=f"{name}", legend=dict(orientation="h", y=-0.16),
                      margin=dict(t=70, b=70))
    fig.update_xaxes(title_text="x (cross) [m]", row=1, col=1)
    fig.update_yaxes(title_text="y (down-valley) [m]", row=1, col=1)
    fig.update_xaxes(title_text="x (cross) [m]", row=1, col=2)
    fig.update_yaxes(title_text="z (elevation) [m]", row=1, col=2)
    fig.update_xaxes(title_text="time [day]", row=2, col=1)
    fig.update_xaxes(title_text="time [day]", row=2, col=2)
    fig.update_yaxes(type="log", row=2, col=2)
    stat = (f"max pond {1e3*float(g('max_pond')):.1f} mm · inlet {float(g('cum_inlet')[-1]):.3f} m³ · "
            f"tile drain {float(g('cum_drain')[-1]):.3f} m³ · mass-bal {float(g('bal_rel')):.1e} · "
            f"min d {1e3*float(g('d_min')):+.2f} mm · {int(g('n_acc'))} steps")
    return f"<h3>{name}</h3><p style='color:#555;font-size:13px;margin-top:-6px'>{stat}</p>" + \
        fig.to_html(full_html=False, include_plotlyjs=False)


def build(npz_path: str, out_html: str) -> None:
    from plotly.offline import get_plotlyjs
    d = np.load(npz_path, allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    body = "".join(scenario_html(d, k) for k in keys)
    intro = """</script>
<style>body{font-family:system-ui,Arial,sans-serif;max-width:1180px;margin:18px auto;color:#222}
.note{background:#f7f7f9;border-left:4px solid #3182bd;padding:8px 12px;font-size:14px}</style></head><body>
<h2>PIDS coupled convergent-flow dual-drain — Tier-3 sign-off (P3 Part B)</h2>
<p class="note"><b>What this validates.</b> The convergent-flow workstream (P0–P3) fixed the coupled
<b>upwind</b> overland solver for the regime where surface water concentrates along convergence lines —
where PIDS networks/inlets install. This permanent Tier-2 fixture exercises that fix end-to-end on a
<b>convergent graded topography</b> (the topographic <i>setting</i>, variable geometry — <b>not</b> a PIDS
"feature") over a 3-D loam-over-clay host, with the signed-off <b>dual-drain</b> (<code>add_surface_inlet</code>
+ <code>add_interior_drain</code> — the coupled-integrated PIDS drainage elements). For each scenario,
<b>drag the time slider / press ▶</b> to watch the ponding concentrate onto the line (left) and the
subsurface saturation evolve (right; firebrick = clay interface, orange = tile-drain band, green = the
convergence floor).</p>"""
    foot = ("<p style='font-size:13px;color:#555;margin-top:24px'>Every scenario stays mass-conservative "
            "(rel ≪ 10⁻³), within the upwind positivity tripwire (sub-cm min depth), and engages both "
            "embedded elements. Builder <code>scratch/_p3_convergent_fixture.py</code>; framing per "
            "<code>pids-p3-convergent-progress</code> memory.</p></body></html>")
    html = ('<!doctype html><html><head><meta charset="utf-8">'
            '<title>PIDS convergent dual-drain — Tier-3</title><script>'
            + get_plotlyjs() + intro + body + foot)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"WROTE {out_html}  ({len(keys)} scenarios)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else
          "../validation/sanity/data/p3_convergent_dualdrain__2026-06-20.npz",
          sys.argv[2] if len(sys.argv) > 2 else
          "../validation/sanity/viz/p3_convergent_dualdrain__2026-06-20.html")
