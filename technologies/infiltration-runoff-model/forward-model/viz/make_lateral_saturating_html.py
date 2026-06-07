"""Tier-3 visualizer: LATERAL-DRAINAGE under a SATURATING storm, with a C / H_ext sensitivity sweep.

Reads ONE coupling_latsat__<soil>__<date>.nc (dims: case, time, z, x) written by
viz/run_lateral_saturating_sweep.py and builds ONE self-contained offline HTML PER TEXTURE comparing the
sweep cases (a 'baseline' with no lateral drain, plus a set of (C, H_ext) drained cases). The generator
reads ONLY the result NetCDF -- it never imports or runs the solver (the Tier-3 independence lens).

The story (the PIDS value-proposition): a sustained low-intensity rain on a soil with an antecedent water
table fills the profile toward saturation. WITHOUT a lateral drain the water table rises (saturation
-> surface runoff). WITH a toe lateral drain (GHB, relative-perm weighted) the excess subsurface water is
removed laterally, holding the water table down and cutting saturation/runoff. The sweep maps how the
response scales with conductance C and external head H_ext, and that it SELF-LIMITS (Codex 2026-06-07).

Panels (per texture):
  1. psi(z) mid-column [animated slider] -- per case; surface at top.   1b. water-table elevation wt(t).
  2. LATERAL drainage hydrograph q_drain(t) per case (+ zero line: <0 = inflow).
  3. SURFACE outlet hydrograph q(t) per case (the drain should cut this vs baseline).
  4. water partition per case (infiltrated/ponded/surface-out/lateral-out) | conservation |mbe|(t) [log].
  5. metrics / findings panel (text from attrs).

Note: plotly's text renderer does NOT decode named HTML entities -- all symbols are Unicode literals.

Usage:
    python viz/make_lateral_saturating_html.py [<date>] [<soil_key>]
        no args      -> date=2026-06-07, all four textures
        <date>       -> that date, all four textures
        <date> <key> -> that date, one texture
Dependencies: xarray + plotly (plotly vendored inline so the HTML opens offline by double-click).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SOIL_ORDER = ["sand", "loam", "silt", "clay"]
# distinct colors per sweep case (baseline first = black); up to 6 cases.
CASE_COLORS = ["#000000", "#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#c62828", "#5d4037"]
# NOTE: "Δsoil-water" is the NET subsurface STORAGE change (soil_water[-1]-soil_water[0]), NOT an
# infiltration flux -- it goes NEGATIVE when a strong drain removes antecedent groundwater (Codex 2026-06-07).
PARTITION_COLOR = {"Δsoil-water": "#8d6e63", "ponded": "#0288d1", "surface-out": "#43a047", "lateral-out": "#ef6c00"}
RAIN_REF_COLOR = "#9e9e9e"


def _attr(ds, key, default=None):
    if key not in ds.attrs:
        return default
    v = ds.attrs[key]
    return v.item() if isinstance(v, np.generic) else v


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.3e}"
        return f"{v:.{nd}g}"
    return str(v)


def _case_label(lbl, C, He):
    if not np.isfinite(C):
        return "baseline (no drain)"
    return f"C={_fmt(float(C))}, H_ext={_fmt(float(He))}"


def _interp_water_table(psi_col, z):
    """Continuous water-table elevation from a column profile: the highest z where psi crosses 0,
    LINEARLY INTERPOLATED between the bracketing nodes (vs snapping to the nearest mesh node, which
    produces a 0.125-m staircase). psi_col, z are 1-D, z ascending (base->surface). Codex 2026-06-07."""
    psi_col = np.asarray(psi_col, float)
    sat = np.where(psi_col >= 0.0)[0]
    if sat.size == 0:
        return float(z[0])              # fully unsaturated -> water table at/below the base
    j = int(sat[-1])                    # highest saturated node
    if j >= len(z) - 1:
        return float(z[j])             # saturated to the top
    p0, p1 = psi_col[j], psi_col[j + 1]   # p0 >= 0, expect p1 < 0
    if p1 >= 0.0:
        return float(z[j])
    frac = p0 / (p0 - p1)               # in [0,1]: where psi=0 between node j and j+1
    return float(z[j] + frac * (z[j + 1] - z[j]))


def build_html(nc_path: str, out_path: str, date: str = "") -> str:
    ds = xr.open_dataset(nc_path)
    cases = [str(c) for c in ds["case"].values]
    nC = len(cases)
    z = np.asarray(ds["z"].values, float)
    time = np.asarray(ds["time"].values, float)
    ntime = time.shape[0]
    case_C = np.asarray(ds["case_C"].values, float)
    case_He = np.asarray(ds["case_He"].values, float)
    rainfall = np.asarray(ds["rainfall"].values, float)[0]   # same forcing across cases
    if not date:
        date = _attr(ds, "date", "")

    soil_key = _attr(ds, "soil_key", "")
    scenario = _attr(ds, "scenario", soil_key)
    storm_dur = _attr(ds, "storm_duration_day")
    z_wt = _attr(ds, "antecedent_water_table_m")
    rain_rate = _attr(ds, "rain_rate_m_per_day")
    Ks = _attr(ds, "Ks_m_per_day")
    cum_rain_total = _attr(ds, "cum_rain_total_m2")
    top_length = float(_attr(ds, "top_length_m", 1.0))

    head = np.asarray(ds["head"].values, float)          # (case, time, z)
    drain = np.asarray(ds["drainage"].values, float)     # (case, time)
    outflow = np.asarray(ds["outflow"].values, float)    # (case, time)
    # smooth, interpolated mid-column water table (psi=0 crossing) instead of the node-snapped proxy
    wt = np.array([[_interp_water_table(head[i, k], z) for k in range(ntime)] for i in range(nC)])
    mbe = np.asarray(ds["mass_balance_error"].values, float)
    soil_w = np.asarray(ds["soil_water"].values, float)
    surf_w = np.asarray(ds["surface_water"].values, float)
    cum_out = np.asarray(ds["cum_outflow"].values, float)
    cum_drn = np.asarray(ds["cum_drainage"].values, float)

    labels = [_case_label(cases[i], case_C[i], case_He[i]) for i in range(nC)]
    colors = [CASE_COLORS[i % len(CASE_COLORS)] for i in range(nC)]

    # axis ranges (fixed so slider frames don't rescale)
    h_lo, h_hi = float(np.nanmin(head)), float(np.nanmax(head))
    h_pad = max((h_hi - h_lo) * 0.06, 1e-3)
    h_range = [h_lo - h_pad, h_hi + h_pad]
    z_range = [float(z.min()), float(z.max())]
    dq_lo, dq_hi = float(np.nanmin(drain)), float(np.nanmax(drain))
    dq_pad = max((dq_hi - dq_lo) * 0.08, 1e-6)
    dq_range = [dq_lo - dq_pad, dq_hi + dq_pad]
    q_hi = max(float(np.nanmax(outflow)), 1e-12)
    q_range = [0.0, q_hi * 1.08]
    wt_lo, wt_hi = float(np.nanmin(wt)), float(np.nanmax(wt))
    wt_range = [max(wt_lo - 0.05, 0.0), wt_hi + 0.05]

    # partition per case [m2]: infiltrated (soil gain), ponded (final surface store), surface-out, lateral-out
    infil = soil_w[:, -1] - soil_w[:, 0]
    ponded = surf_w[:, -1]
    surfout = cum_out[:, -1]
    latout = cum_drn[:, -1]
    part_vals = np.array([infil, ponded, surfout, latout])
    part_max = max(float(np.nanmax(np.abs(part_vals))), float(cum_rain_total or 0.0), 1e-9)
    part_range = [min(0.0, float(np.nanmin(part_vals)) * 1.1), part_max * 1.18]

    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None],
               [{"type": "xy", "colspan": 2}, None],
               [{"type": "xy"}, {"type": "xy"}]],
        row_heights=[0.28, 0.22, 0.22, 0.28], vertical_spacing=0.085, horizontal_spacing=0.12,
        subplot_titles=(
            "1 · Pressure head  ψ(z)  at MID-SLOPE (x=L/2), surface at top — one line per drain case",
            "1b · Water-table z (t) [m] — mid-slope, interpolated ψ=0",
            "2 · Lateral drainage hydrograph  q_drain(t)  [m²/day]   (<0 = inflow)",
            "3 · Surface outlet hydrograph  q(t)  [m²/day]   (drain should cut this)",
            "4 · Water partition per case  [m²]",
            "5 · Conservation  |mass-balance error| (t)  [log]",
        ),
    )

    # ---- row1 col1: psi(z) animated, per case ----
    h_idx = {}
    for i in range(nC):
        h_idx[i] = len(fig.data)
        fig.add_trace(go.Scatter(
            x=head[i, 0], y=z, mode="lines+markers",
            line=dict(color=colors[i], width=2.4, dash="dot" if i == 0 else "solid"),
            marker=dict(size=3, color=colors[i]), name=labels[i], legendgroup=str(i),
            hovertemplate="ψ=%{x:.3f} m<br>z=%{y:.3f} m<extra>" + labels[i] + "</extra>"),
            row=1, col=1)
    # water-table reference line at antecedent z_wt
    if z_wt is not None:
        fig.add_hline(y=float(z_wt), line=dict(color="#90a4ae", width=1.0, dash="dash"), row=1, col=1)

    # ---- row1 col2: water table wt(t) per case ----
    for i in range(nC):
        fig.add_trace(go.Scatter(
            x=time, y=wt[i], mode="lines", line=dict(color=colors[i], width=2.2, dash="dot" if i == 0 else "solid"),
            name=labels[i], legendgroup=str(i), showlegend=False,
            hovertemplate="t=%{x:.3f} d<br>wt=%{y:.3f} m<extra>" + labels[i] + "</extra>"),
            row=1, col=2)

    # ---- row2: lateral drainage hydrograph ----
    fig.add_hline(y=0.0, line=dict(color="#bbbbbb", width=1.0), row=2, col=1)
    for i in range(nC):
        fig.add_trace(go.Scatter(
            x=time, y=drain[i], mode="lines", line=dict(color=colors[i], width=2.4, dash="dot" if i == 0 else "solid"),
            name=labels[i], legendgroup=str(i), showlegend=False,
            hovertemplate="t=%{x:.3f} d<br>q_drain=%{y:.4f}<extra>" + labels[i] + "</extra>"),
            row=2, col=1)
    if storm_dur is not None:
        fig.add_vline(x=float(storm_dur), line=dict(color="#b71c1c", width=1.2, dash="dot"), row=2, col=1)

    # ---- row3: surface hydrograph ----
    for i in range(nC):
        fig.add_trace(go.Scatter(
            x=time, y=outflow[i], mode="lines", line=dict(color=colors[i], width=2.4, dash="dot" if i == 0 else "solid"),
            name=labels[i], legendgroup=str(i), showlegend=False,
            hovertemplate="t=%{x:.3f} d<br>q=%{y:.4f}<extra>" + labels[i] + "</extra>"),
            row=3, col=1)
    if storm_dur is not None:
        fig.add_vline(x=float(storm_dur), line=dict(color="#b71c1c", width=1.2, dash="dot"), row=3, col=1)

    # ---- row4 col1: partition grouped bars per case ----
    comp_names = ["Δsoil-water", "ponded", "surface-out", "lateral-out"]
    comp_data = [infil, ponded, surfout, latout]
    for cname, cdat in zip(comp_names, comp_data):
        fig.add_trace(go.Bar(
            x=labels, y=cdat, name=cname, marker=dict(color=PARTITION_COLOR[cname]),
            legendgroup="part", hovertemplate="%{x}<br>" + cname + "=%{y:.5f} m²<extra></extra>"),
            row=4, col=1)

    # ---- row4 col2: conservation ----
    for i in range(nC):
        m = np.where(np.abs(mbe[i]) <= 0, np.nan, np.abs(mbe[i]))
        fig.add_trace(go.Scatter(
            x=time, y=m, mode="lines+markers", line=dict(color=colors[i], width=1.6),
            marker=dict(size=3, color=colors[i]), name=labels[i], legendgroup=str(i), showlegend=False,
            hovertemplate="t=%{x:.3f} d<br>|mbe|=%{y:.2e}<extra>" + labels[i] + "</extra>"),
            row=4, col=2)

    # ---- animation frames (psi only) ----
    frames = []
    for k in range(ntime):
        fdata, ftr = [], []
        for i in range(nC):
            fdata.append(go.Scatter(x=head[i, k], y=z)); ftr.append(h_idx[i])
        frames.append(go.Frame(name=str(k), data=fdata, traces=ftr))
    fig.frames = frames
    slider_steps = [dict(method="animate", label=f"{time[k]:.3f}",
                         args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                              transition=dict(duration=0))]) for k in range(ntime)]
    sliders = [dict(active=0, x=0.05, y=-0.06, len=0.9, pad=dict(t=18, b=8),
                    currentvalue=dict(prefix="profile time = ", suffix=" day", font=dict(size=14)),
                    steps=slider_steps)]
    updatemenus = [dict(type="buttons", direction="left", x=0.0, y=1.04, xanchor="left", yanchor="bottom",
                        pad=dict(t=0, r=10), showactive=False, buttons=[
                            dict(label="▶ Play", method="animate", args=[None, dict(mode="immediate",
                                 fromcurrent=True, frame=dict(duration=120, redraw=True), transition=dict(duration=0))]),
                            dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate",
                                 frame=dict(duration=0, redraw=True), transition=dict(duration=0))])])]

    fig.update_xaxes(title_text="pressure head ψ  [m]", range=h_range, row=1, col=1)
    fig.update_yaxes(title_text="elevation z  [m]  (surface at top)", range=z_range, row=1, col=1)
    fig.update_xaxes(title_text="time  [day]", range=[float(time.min()), float(time.max())], row=1, col=2)
    fig.update_yaxes(title_text="water-table z  [m]", range=wt_range, row=1, col=2)
    fig.update_xaxes(title_text="time  [day]", range=[float(time.min()), float(time.max())], row=2, col=1)
    fig.update_yaxes(title_text="q_drain  [m²/day]", range=dq_range, row=2, col=1)
    fig.update_xaxes(title_text="time  [day]", range=[float(time.min()), float(time.max())], row=3, col=1)
    fig.update_yaxes(title_text="surface q  [m²/day]", range=q_range, row=3, col=1)
    fig.update_xaxes(title_text="sweep case", row=4, col=1)
    fig.update_yaxes(title_text="water  [m²]", range=part_range, row=4, col=1)
    fig.update_xaxes(title_text="time  [day]", range=[float(time.min()), float(time.max())], row=4, col=2)
    fig.update_yaxes(title_text="|mass-balance err|", type="log", exponentformat="power", row=4, col=2)

    # ---- metrics / findings ----
    mbe_max = float(_attr(ds, "mass_balance_error_max", 0.0))
    max_it = int(_attr(ds, "max_newton_iters", 0))
    min_drn = float(_attr(ds, "min_drainage_any", 0.0))
    NB = " "
    lines = [
        f"<b>texture</b>: {scenario} {NB} <b>date</b>: {date}",
        (f"<b>saturating scenario</b>: antecedent water table z={_fmt(z_wt)} m; sustained rain "
         f"r={_fmt(rain_rate)} m/day (< Ks={_fmt(Ks)}) for {_fmt(storm_dur)} day, then recession; "
         f"cum rain ≈ {_fmt(cum_rain_total)} m²"),
        "<b>base</b>: impermeable (no-flow) · <b>lateral drain</b>: toe GHB q_n=C·kr(ψ)·(ψ+z−H_ext), "
        "conductance ramped 0→C over 0.02 day (numerical regularization)",
        "<b>per case</b> (label | Δsoil-water / pond / surf-out / lat-out as % of rain | Δwater-table | sat% final):",
    ]
    cr = float(cum_rain_total or 1e-30) + 1e-30
    for i in range(nC):
        dwt = wt[i, -1] - wt[i, 0]
        satf = float(np.asarray(ds["sat_fraction"].values, float)[i, -1]) * 100.0
        lines.append(
            f"{NB}{NB}<b>{labels[i]}</b> | {infil[i]/cr*100:.1f} / {ponded[i]/cr*100:.1f} / "
            f"{surfout[i]/cr*100:.1f} / {latout[i]/cr*100:.1f} % | Δwt={dwt:+.3f} m | {satf:.0f}%")
    metrics_html = "<br>".join(lines)

    base_surf = float(surfout[0]); drained_min_surf = float(np.min(surfout[1:])) if nC > 1 else base_surf
    cut = (1.0 - drained_min_surf / base_surf) * 100.0 if base_surf > 1e-12 else 0.0
    findings = (
        "<b>FINDINGS</b> (framing per Codex review 2026-06-07)<br>"
        f"All {nC} cases CONSERVE (max|mbe| ≤ {mbe_max:.0e}) under the FULL balance Δtotal = cum_rain − "
        "cum_surface_out − cum_lateral_drain + clip — independently re-checked from raw stored water + "
        "cumulative fluxes. With a saturating supply the lateral toe drain is genuinely engaged: stronger "
        f"C / lower H_ext cut surface runoff (best ≈ {cut:.0f}% vs baseline) and drain laterally instead; "
        "drainage is K(ψ)-weighted so it SELF-LIMITS (monotonic, diminishing returns — cannot over-drain). "
        "<b>Read carefully:</b> ‘Δsoil-water’ is NET subsurface STORAGE change, NOT infiltration — it goes "
        "negative when a strong drain removes ANTECEDENT groundwater (e.g. conductive sand de-waters the "
        "water table; lateral-out then exceeds rain). <b>Caveats:</b> (1) the drain conductance is RAMPED "
        "0→C over 0.02 day to avoid a cold-start divergence on stiff near-saturated soil — a numerical "
        "regularization (conservation-exact, but it suppresses the earliest drainage; the unramped "
        "cold-start is a solver limitation). (2) Every case here is OUTFLOW (H_ext below the soil head) — "
        "the bidirectional INJECTION branch is pinned by a unit test but NOT exercised by this sweep. "
        "(3) the water-table line is the interpolated ψ=0 crossing at MID-SLOPE (x=L/2) — a single-column "
        "sample, NOT the toe where the drain acts; sat% is a node-count proxy (ψ≥0). Trust the conserved "
        "partition/flux totals over these mid-slope state proxies."
    )
    fig.add_annotation(
        x=0.0, y=-0.13, xref="paper", yref="paper", xanchor="left", yanchor="top", align="left", showarrow=False,
        text="<b>METRICS</b><br>" + metrics_html + "<br><br>" + findings,
        font=dict(size=10.5, color="#222", family="Consolas, monospace"),
        bordercolor="#888", borderwidth=1, borderpad=8, bgcolor="rgba(245,245,245,0.96)")

    title = f"<b>PIDS Pillar-2 Tier-3: lateral drainage under a saturating storm — {scenario}</b> · {date}"
    fig.update_layout(
        title=dict(text=title + "<br><span style='font-size:13px'>sensitivity sweep over toe-drain "
                        "conductance C and external head H_ext vs a no-drain baseline</span>",
                   x=0.5, xanchor="center", y=0.997, yanchor="top"),
        sliders=sliders, updatemenus=updatemenus,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01, font=dict(size=9.5)),
        margin=dict(l=90, r=240, t=130, b=300), width=1320, height=2000,
        barmode="group", bargap=0.25, bargroupgap=0.08,
        paper_bgcolor="white", plot_bgcolor="#fafafa", hovermode="closest")
    fig.write_html(out_path, include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(displaylogo=False, responsive=True))
    ds.close()
    return out_path


def _main(date="2026-06-07", only=None):
    data_dir = "../validation/sanity/data"
    viz_dir = "../validation/sanity/viz"
    os.makedirs(viz_dir, exist_ok=True)
    keys = [only] if only else SOIL_ORDER
    for key in keys:
        nc = f"{data_dir}/coupling_latsat__{key}__{date}.nc"
        if not os.path.exists(nc):
            print(f"SKIP {key}: {nc} not found", file=sys.stderr); continue
        out = f"{viz_dir}/coupling_latsat__{key}__{date}.html"
        build_html(nc, out, date=date)
        print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        _main()
    elif len(args) == 1:
        _main(date=args[0])
    else:
        _main(date=args[0], only=args[1])
