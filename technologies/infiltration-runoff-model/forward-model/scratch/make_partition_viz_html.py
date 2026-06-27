"""Build a self-contained HTML visualizing infiltration & runoff for the three soil classes from the
_viz_{sand,loam,clay}.npz produced by seq_partition_viz.py. Three view types per soil (Arik's request):
  COL 1  TOP-DOWN: surface ponding depth d(x,y) on the tilted plane at storm end (outlet at y=0).
  COL 2  PROFILE : vertical water-content theta vs DEPTH at the centre column, several times (wetting front).
  COL 3  TIME SERIES: cumulative rain / infiltration / runoff / ponding depths [mm] vs time (the partition).

matplotlib -> base64 PNG embedded in one offline HTML (double-click to open). No solver import.

Run (any python with matplotlib; WSL pids-fem works):
  python -u scratch/make_partition_viz_html.py
Output: scratch/partition_viz__<date>.html  (date passed in to avoid Date.now in-script)
"""
from __future__ import annotations

import base64
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HERE = os.path.dirname(__file__)
SOILS = ["sand", "loam", "clay"]
SOIL_LABEL = {"sand": "SAND (Ks=1.5)", "loam": "LOAM (Ks=0.25)", "clay": "CLAY (Ks=0.048)"}


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load():
    data = {}
    for s in SOILS:
        p = os.path.join(HERE, f"_viz_{s}.npz")
        if os.path.exists(p):
            data[s] = np.load(p)
    return data


def fig_topdown(z):
    """Surface ponding depth d(x,y) at storm end (snap nearest STORM)."""
    snap_t = z["snap_t"]; storm = float(z["storm"])
    k = int(np.argmin(np.abs(snap_t - storm)))
    x, y, d = z["top_x"], z["top_y"], z["snap_d"][k] * 1000.0   # mm
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    tri = Triangulation(x, y)
    vmax = max(float(np.percentile(d, 99)), 1e-3)
    tcf = ax.tricontourf(tri, d, levels=14, cmap="YlGnBu", vmin=0.0, vmax=vmax)
    ax.plot([x.min(), x.max()], [0, 0], "r-", lw=2)           # outlet edge y=0
    ax.annotate("outlet (y=0)", (0.5 * (x.min() + x.max()), 0), color="r", fontsize=7,
                ha="center", va="bottom")
    fig.colorbar(tcf, ax=ax, label="ponding d [mm]", shrink=0.9)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y (downslope) [m]")
    ax.set_title(f"top-down ponding @ t={snap_t[k]:.2f} d  (peak {d.max():.2f} mm)", fontsize=8)
    ax.set_aspect("equal")
    return _b64(fig)


def fig_profile(z):
    """theta vs depth at the centre column, several snapshot times (the wetting front)."""
    zc = z["col_z"]; depth = (float(z["Lz"]) - zc) * 1000.0     # mm below surface
    th = z["snap_theta"]; snap_t = z["snap_t"]
    thr, ths = float(z["theta_r"]), float(z["theta_s"])
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(snap_t)))
    for i, t in enumerate(snap_t):
        ax.plot(th[i], depth, "-o", ms=2.5, color=cmap[i], label=f"t={t:.2f}d")
    ax.axvline(ths, ls=":", c="gray", lw=1); ax.text(ths, depth.max() * 0.97, " θs", fontsize=6, c="gray")
    ax.axvline(thr, ls=":", c="gray", lw=1); ax.text(thr, depth.max() * 0.97, " θr", fontsize=6, c="gray")
    ax.set_ylim(depth.max(), 0)                                # surface at top
    ax.set_xlabel("water content θ [-]"); ax.set_ylabel("depth below surface [mm]")
    ax.set_title("wetting-front profile (centre)", fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(alpha=0.3)
    return _b64(fig)


def fig_timeseries(z):
    """Cumulative partition depths [mm] vs time: rain / infiltration / runoff / ponding."""
    ts = z["ts"]; A = float(z["top_area"]); sw0 = float(z["sw0"]); storm = float(z["storm"])
    t = ts[:, 0]
    rain = ts[:, 1] / A * 1000.0
    runoff = ts[:, 2] / A * 1000.0
    infil = (ts[:, 4] - sw0 + ts[:, 3]) / A * 1000.0           # soil gain + drainage
    pond = ts[:, 5] / A * 1000.0
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    ax.plot(t, rain, "k--", lw=1.5, label="rain (in)")
    ax.plot(t, infil, "-", c="#2c7", lw=2, label="infiltration")
    ax.plot(t, runoff, "-", c="#36c", lw=2, label="runoff")
    ax.plot(t, pond, "-", c="#c93", lw=1.3, label="ponding (surface store)")
    ax.axvspan(0, storm, color="gray", alpha=0.10)
    ax.annotate("storm", (storm * 0.5, ax.get_ylim()[1] * 0.95), fontsize=6, ha="center", va="top",
                color="gray")
    rr = float(z["routed_R"]); ir = float(z["infil_R"])
    ax.set_xlabel("time [day]"); ax.set_ylabel("cumulative depth [mm]")
    ax.set_title(f"partition: runoff/R={rr:.2f}  infil/R={ir:.2f}", fontsize=8)
    ax.legend(fontsize=6, loc="center right"); ax.grid(alpha=0.3)
    return _b64(fig)


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "undated"
    data = _load()
    if not data:
        print("NO _viz_*.npz found -- run seq_partition_viz.py first.", flush=True)
        return
    rows = []
    for s in SOILS:
        if s not in data:
            continue
        z = data[s]
        td = fig_topdown(z); pr = fig_profile(z); tsd = fig_timeseries(z)
        rr, ir = float(z["routed_R"]), float(z["infil_R"])
        rows.append(f"""
        <tr>
          <td class="lab"><b>{SOIL_LABEL[s]}</b><br><span class="sub">runoff {rr*100:.0f}% /
              infil {ir*100:.0f}%</span></td>
          <td><img src="data:image/png;base64,{td}"></td>
          <td><img src="data:image/png;base64,{pr}"></td>
          <td><img src="data:image/png;base64,{tsd}"></td>
        </tr>""")
    g = data[SOILS[0] if SOILS[0] in data else list(data)[0]]
    storm_mm = float(g["rain"]) * float(g["storm"]) * 1000.0
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Infiltration &amp; runoff by soil class</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:18px;color:#222}}
 h1{{font-size:19px;margin:0 0 2px}} .meta{{color:#555;font-size:12px;margin-bottom:12px}}
 table{{border-collapse:collapse}} td{{border:1px solid #ddd;padding:5px;vertical-align:middle;text-align:center}}
 td.lab{{width:130px;font-size:12px;background:#f7f7f7}} .sub{{color:#777;font-size:11px}}
 th{{font-size:12px;padding:6px;background:#eef;border:1px solid #ddd}}
 img{{display:block}} .note{{font-size:11px;color:#666;margin-top:10px;max-width:1100px;line-height:1.45}}
</style></head><body>
<h1>Infiltration &amp; runoff for three soil classes — resolved tilted plane</h1>
<div class="meta">b1 plane {g['Lx']:.0f}×{g['Ly']:.0f}×{g['Lz']:.0f} m, slope S={float(g['S0']):.2f},
 storm {float(g['rain']):.1f} m/day × {float(g['storm'])*24:.1f} h ≈ {storm_mm:.0f} mm, ψᵢ=−0.4 m,
 RESOLVED surface (graded ~2 mm top cell). Generated {date}.</div>
<table>
 <tr><th>soil</th><th>top-down: surface ponding d(x,y)</th>
     <th>profile: wetting front θ(depth)</th><th>time series: where the rain goes</th></tr>
 {''.join(rows)}
</table>
<div class="note"><b>Read:</b> top-down = where surface water sits/concentrates toward the outlet;
 profile = how deep water penetrates (sand: deep front; clay: shallow); time series = the rain/runoff/
 infiltration partition over the storm. Same storm on all three → the partition is set by the SOIL.
 These use a RESOLVED surface (the corrected physics, §22); the coarse-mesh model over-states runoff by
 under-capturing sorptive infiltration.</div>
</body></html>"""
    out = os.path.join(HERE, f"partition_viz__{date}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}  ({len(rows)} soils)", flush=True)


if __name__ == "__main__":
    main()
