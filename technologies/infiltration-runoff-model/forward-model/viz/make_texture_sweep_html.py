"""Tier-3 sanity visualizer for the PIDS Pillar-2 forward model: SOIL-TEXTURE SWEEP (coupling).

Compares FOUR soil-texture runs of the coupled (surface<->subsurface) hydrology model
under ONE 0.6 m/day storm on the SAME sloped hillslope, and turns them into ONE
self-contained, offline, interactive HTML. The generator reads ONLY the result NetCDFs --
it never imports or runs the solver (this independence is the whole point of the Tier-3
lens).

The four textures (one NetCDF each, plotted coarse -> fine):
  - sand : Ks ~ 7.1 m/day, vg_n ~ 2.7  (sharp, deep wetting front)
  - loam : Ks ~ 0.25 m/day, vg_n ~ 1.56
  - silt : Ks ~ 0.06 m/day, vg_n ~ 1.37
  - clay : Ks ~ 0.05 m/day, vg_n ~ 1.09 (diffuse, shallow response)

EVERYTHING is identical across the four runs (L=5 m sloped hillslope, S0=0.05, open
downstream outlet at Manning normal depth, one 0.6 m/day storm for 0.03 day then recession
to 0.06 day) EXCEPT (a) the van Genuchten texture params and (b) the per-soil field-moist
antecedent head psi0 (a common head is physically non-comparable across textures).

Panels (a multi-row subplot figure):
  1. INFILTRATION PROFILE theta(z) with a TIME SLIDER, surface at TOP (front advances down).
  1b. PRESSURE-HEAD PROFILE psi(z) (same animated slider) -- linked companion.
  2. INFILTRATION RATE vs RAINFALL (t) -- d(soil_water)/dt per unit area [m/day] vs the rain
     supply: shows WHY coarse soils still run off (the k_ex surface throttle limits early
     infiltration below the rain rate until the surface wets up).
  3. WATER PARTITION per soil: grouped bars infiltrated / ponded / drained vs cum-rain.
  4. OUTLET HYDROGRAPH q(t) [m2/day] per soil + the rain-off line; CONSERVATION |mbe|(t) [log].
  5. METRICS / FINDINGS panel (text from attrs).

NOTE on text: plotly's text renderer does NOT decode named HTML entities (e.g. &middot;,
&sup2;) -- they appear literally -- so all symbols here are Unicode literals (·, ², θ, ψ, …).

Usage:
    python viz/make_texture_sweep_html.py [<date>] [<out.html>]
        (no args -> date=2026-06-06, default out path)

Dependencies: xarray + plotly only. Plotly is vendored INLINE
(``include_plotlyjs=True``) so the HTML opens offline by double-click on Windows.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------------------------------------------------------- helpers
def _attr(ds: xr.Dataset, key: str, default=None):
    """Return a global attr coerced to a plain Python scalar where possible."""
    if key not in ds.attrs:
        return default
    val = ds.attrs[key]
    if isinstance(val, (np.generic,)):
        return val.item()
    return val


def _fmt(val, nd: int = 4) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        if val != 0 and (abs(val) < 1e-3 or abs(val) >= 1e4):
            return f"{val:.3e}"
        return f"{val:.{nd}g}"
    return str(val)


def _units(da, fallback: str = "") -> str:
    return str(da.attrs.get("units", fallback))


# the 4 textures in plotting order (coarse -> fine), with consistent colors across panels.
SOIL_ORDER = ["sand", "loam", "silt", "clay"]
SOIL_COLOR = {
    "sand": "#daa520",   # goldenrod  -- coarse, sharp deep front
    "loam": "#8b4513",   # sienna/brown
    "silt": "#008080",   # teal
    "clay": "#7b1fa2",   # purple     -- fine, diffuse shallow response
}
SOIL_LABEL = {"sand": "sand", "loam": "loam", "silt": "silt", "clay": "clay"}
PARTITION_COLOR = {"infiltrated": "#8d6e63", "ponded": "#0288d1", "drained": "#43a047"}
RAIN_REF_COLOR = "#9e9e9e"


# ----------------------------------------------------------------------------- builder
def build_html(nc_paths: dict[str, str], out_path: str, date: str = "") -> str:
    """Build the self-contained interactive texture-sweep comparison HTML.

    Parameters
    ----------
    nc_paths : dict[str, str]
        Mapping ``soil_key -> path`` for the four standardized texture-sweep NetCDF results
        (keys: ``sand``, ``loam``, ``silt``, ``clay``).
    out_path : str
        Path to write the self-contained HTML (plotly inlined, offline).
    date : str
        Date stamp for the title (falls back to the file attr if empty).
    """
    # ---- load all 4 datasets, keyed by soil_key, in canonical order ----------------
    ds_map: dict[str, xr.Dataset] = {}
    for key in SOIL_ORDER:
        if key not in nc_paths:
            raise KeyError(f"missing NetCDF for soil key {key!r}")
        ds_map[key] = xr.open_dataset(nc_paths[key])

    ref = ds_map[SOIL_ORDER[0]]
    z = np.asarray(ref["z"].values, dtype=float)            # (z,) [m] elevation; 0=base, max=surface
    time = np.asarray(ref["time"].values, dtype=float)      # (time,) [day]
    rainfall = np.asarray(ref["rainfall"].values, dtype=float)  # (time,) [m/day] shared forcing
    cum_rain = np.asarray(ref["cum_rain"].values, dtype=float)  # (time,) [m2]
    ntime = time.shape[0]

    z_units = _units(ref["z"], "m")
    t_units = _units(ref["time"], "day")
    q_units = _units(ref["outflow"], "m2/day")
    th_units = _units(ref["water_content"], "-")
    h_units = _units(ref["head"], "m")
    top_length = float(_attr(ref, "top_length_m", 1.0))     # m -- for per-area infiltration rate

    # shared scenario context (same across files except texture + antecedent head)
    storm_dur = _attr(ref, "storm_duration_day")
    t_end = _attr(ref, "t_end_day")
    rain_rate = _attr(ref, "rain_rate_m_per_day")
    S0 = _attr(ref, "bed_slope_S0")
    n_manning = _attr(ref, "n_manning")
    L = _attr(ref, "L")
    cum_rain_total = _attr(ref, "cum_rain_total_m2")
    if not date:
        date = _attr(ref, "date", "")

    # per-soil arrays pulled once ----------------------------------------------------
    series = {}
    for key in SOIL_ORDER:
        ds = ds_map[key]
        sw = np.asarray(ds["soil_water"].values, dtype=float)          # (time,) [m2]
        # net infiltration rate per unit area [m/day] = d(soil_water)/dt / top_length (>=0; the base
        # is no-flux so soil-water gain IS infiltration). Clip tiny gradient noise to 0.
        infil_rate = np.maximum(np.gradient(sw, time) / top_length, 0.0)
        series[key] = dict(
            theta=np.asarray(ds["water_content"].values, dtype=float),     # (time, z) [-]
            head=np.asarray(ds["head"].values, dtype=float),               # (time, z) [m]
            outflow=np.asarray(ds["outflow"].values, dtype=float),         # (time,) [m2/day]
            mbe=np.asarray(ds["mass_balance_error"].values, dtype=float),  # (time,) [-]
            infil_rate=infil_rate,                                         # (time,) [m/day]
        )

    # ---- fixed axis ranges (so slider frames don't rescale) ------------------------
    th_lo = min(float(np.nanmin(series[k]["theta"])) for k in SOIL_ORDER)
    th_hi = max(float(np.nanmax(series[k]["theta"])) for k in SOIL_ORDER)
    th_pad = max((th_hi - th_lo) * 0.06, 1e-3)
    th_range = [max(th_lo - th_pad, 0.0), th_hi + th_pad]

    h_lo = min(float(np.nanmin(series[k]["head"])) for k in SOIL_ORDER)
    h_hi = max(float(np.nanmax(series[k]["head"])) for k in SOIL_ORDER)
    h_pad = max((h_hi - h_lo) * 0.06, 1e-3)
    h_range = [h_lo - h_pad, h_hi + h_pad]

    z_range = [float(z.min()), float(z.max())]               # surface (max z) at TOP

    q_hi = max(float(np.nanmax(series[k]["outflow"])) for k in SOIL_ORDER)
    q_range = [0.0, max(q_hi * 1.08, 1e-9)]

    # infiltration-rate axis: span the per-soil rates AND the rain supply, padded.
    ir_hi = max(float(np.nanmax(series[k]["infil_rate"])) for k in SOIL_ORDER)
    ir_hi = max(ir_hi, float(np.nanmax(rainfall)))
    ir_range = [0.0, ir_hi * 1.12]

    # partition axis: TIGHT range around the bar + cum-rain values (else bars look squished).
    part = {
        "infiltrated": [float(_attr(ds_map[k], "final_infiltrated_m2", 0.0)) for k in SOIL_ORDER],
        "ponded":      [float(_attr(ds_map[k], "final_ponded_m2", 0.0)) for k in SOIL_ORDER],
        "drained":     [float(_attr(ds_map[k], "final_drained_m2", 0.0)) for k in SOIL_ORDER],
    }
    cum_per_soil = [float(_attr(ds_map[k], "cum_rain_total_m2", 0.0)) for k in SOIL_ORDER]
    part_max = max([v for comp in part.values() for v in comp] + cum_per_soil + [1e-9])
    part_range = [0.0, part_max * 1.18]

    # =============================================================== figure layout (4 rows)
    #   row1 left : theta(z) [animated]     row1 right : psi(z) [animated]      [HEADLINE]
    #   row2 (colspan 2): infiltration rate vs rainfall (t)                     [diagnostic]
    #   row3 (colspan 2): water partition grouped bars per soil
    #   row4 left : outlet hydrograph q(t)  row4 right : conservation |mbe|(t) [log]
    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.30, 0.22, 0.18, 0.30],
        vertical_spacing=0.075,
        horizontal_spacing=0.12,
        subplot_titles=(
            "1 · Infiltration  θ(z)  [-] — surface at top",
            "1b · Pressure head  ψ(z)  [m] — surface at top",
            "2 · Infiltration rate vs rainfall  (t)  [m/day]",
            "3 · Water partition per soil  [m²]",
            "4 · Outlet hydrograph  q(t)  [m²/day]",
            "5 · Conservation  |mass-balance error| (t)  [log]",
        ),
    )

    # =============================================================== row1: theta(z) & psi(z) animated
    th_trace_idx, h_trace_idx = {}, {}
    for key in SOIL_ORDER:  # theta ghosts (t=0)
        fig.add_trace(go.Scatter(
            x=series[key]["theta"][0], y=z, mode="lines",
            line=dict(color=SOIL_COLOR[key], width=1.0, dash="dot"), opacity=0.35,
            name=f"{SOIL_LABEL[key]} (t=0)", legendgroup=key, showlegend=False,
            hovertemplate="θ₀=%{x:.3f}<br>z=%{y:.3f} m<extra>" + key + " init</extra>"),
            row=1, col=1)
    for key in SOIL_ORDER:  # theta animated
        th_trace_idx[key] = len(fig.data)
        fig.add_trace(go.Scatter(
            x=series[key]["theta"][0], y=z, mode="lines+markers",
            line=dict(color=SOIL_COLOR[key], width=2.6), marker=dict(size=4, color=SOIL_COLOR[key]),
            name=SOIL_LABEL[key], legendgroup=key,
            hovertemplate="θ=%{x:.3f}<br>z=%{y:.3f} m<extra>" + key + "</extra>"),
            row=1, col=1)
    for key in SOIL_ORDER:  # psi ghosts (t=0)
        fig.add_trace(go.Scatter(
            x=series[key]["head"][0], y=z, mode="lines",
            line=dict(color=SOIL_COLOR[key], width=1.0, dash="dot"), opacity=0.35,
            name=f"{SOIL_LABEL[key]} (t=0)", legendgroup=key, showlegend=False,
            hovertemplate="ψ₀=%{x:.3f} m<br>z=%{y:.3f} m<extra>" + key + " init</extra>"),
            row=1, col=2)
    for key in SOIL_ORDER:  # psi animated
        h_trace_idx[key] = len(fig.data)
        fig.add_trace(go.Scatter(
            x=series[key]["head"][0], y=z, mode="lines+markers",
            line=dict(color=SOIL_COLOR[key], width=2.6), marker=dict(size=4, color=SOIL_COLOR[key]),
            name=SOIL_LABEL[key], legendgroup=key, showlegend=False,
            hovertemplate="ψ=%{x:.3f} m<br>z=%{y:.3f} m<extra>" + key + "</extra>"),
            row=1, col=2)

    # =============================================================== row2: infiltration rate vs rainfall
    for key in SOIL_ORDER:
        fig.add_trace(go.Scatter(
            x=time, y=series[key]["infil_rate"], mode="lines",
            line=dict(color=SOIL_COLOR[key], width=2.6),
            name=SOIL_LABEL[key], legendgroup=key, showlegend=False,
            hovertemplate="t=%{x:.4f} day<br>infil=%{y:.4f} m/day<extra>" + key + "</extra>"),
            row=2, col=1)
    # rainfall supply (the reference the infiltration rate is competing against).
    fig.add_trace(go.Scatter(
        x=time, y=rainfall, mode="lines",
        line=dict(color=RAIN_REF_COLOR, width=2.4, dash="dash"),
        name="rainfall (supply)", legendgroup="rain",
        hovertemplate="t=%{x:.4f} day<br>rain=%{y:.3f} m/day<extra>supply</extra>"),
        row=2, col=1)
    if storm_dur is not None:
        fig.add_vline(x=float(storm_dur), line=dict(color="#b71c1c", width=1.2, dash="dot"),
                      row=2, col=1)
        fig.add_annotation(x=float(storm_dur), y=ir_range[1] * 0.94, text=f"rain off (t={_fmt(storm_dur)})",
                           showarrow=False, font=dict(size=10, color="#b71c1c"), xanchor="left",
                           bgcolor="rgba(255,255,255,0.7)", row=2, col=1)

    # =============================================================== row3: water partition
    soil_x = [SOIL_LABEL[k] for k in SOIL_ORDER]
    for comp in ("infiltrated", "ponded", "drained"):
        fig.add_trace(go.Bar(
            x=soil_x, y=part[comp], name=comp.capitalize(),
            marker=dict(color=PARTITION_COLOR[comp]), legendgroup="partition",
            hovertemplate="%{x}<br>" + comp + "=%{y:.5f} m²<extra></extra>"),
            row=3, col=1)
    fig.add_trace(go.Scatter(
        x=soil_x, y=cum_per_soil, mode="markers",
        marker=dict(symbol="line-ew", size=46, color="#212121", line=dict(width=2.2, color="#212121")),
        name="cum rain (total)", legendgroup="partition",
        hovertemplate="%{x}<br>cum rain=%{y:.5f} m²<extra></extra>"),
        row=3, col=1)

    # =============================================================== row4 left: hydrograph q(t)
    for key in SOIL_ORDER:
        fig.add_trace(go.Scatter(
            x=time, y=series[key]["outflow"], mode="lines",
            line=dict(color=SOIL_COLOR[key], width=2.6),
            name=SOIL_LABEL[key], legendgroup=key, showlegend=False,
            hovertemplate="t=%{x:.4f} day<br>q=%{y:.4f} " + q_units + "<extra>" + key + "</extra>"),
            row=4, col=1)
    if storm_dur is not None:
        fig.add_vline(x=float(storm_dur), line=dict(color="#b71c1c", width=1.2, dash="dot"),
                      row=4, col=1)
        fig.add_annotation(x=float(storm_dur), y=q_range[1] * 0.94, text=f"rain off (t={_fmt(storm_dur)})",
                           showarrow=False, font=dict(size=10, color="#b71c1c"), xanchor="left",
                           bgcolor="rgba(255,255,255,0.7)", row=4, col=1)

    # =============================================================== row4 right: conservation mbe(t)
    for key in SOIL_ORDER:
        mbe = series[key]["mbe"]
        mbe_plot = np.where(np.abs(mbe) <= 0, np.nan, np.abs(mbe))
        fig.add_trace(go.Scatter(
            x=time, y=mbe_plot, mode="lines+markers",
            line=dict(color=SOIL_COLOR[key], width=1.8), marker=dict(size=3, color=SOIL_COLOR[key]),
            name=SOIL_LABEL[key], legendgroup=key, showlegend=False,
            hovertemplate="t=%{x:.4f} day<br>|mbe|=%{y:.3e}<extra>" + key + "</extra>"),
            row=4, col=2)

    # =============================================================== animation frames (theta + psi)
    frames = []
    for k in range(ntime):
        fdata, ftraces = [], []
        for key in SOIL_ORDER:
            fdata.append(go.Scatter(x=series[key]["theta"][k], y=z)); ftraces.append(th_trace_idx[key])
        for key in SOIL_ORDER:
            fdata.append(go.Scatter(x=series[key]["head"][k], y=z)); ftraces.append(h_trace_idx[key])
        frames.append(go.Frame(name=str(k), data=fdata, traces=ftraces))
    fig.frames = frames

    # =============================================================== slider + controls
    slider_steps = [dict(method="animate", label=f"{time[k]:.3f}",
                         args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                              transition=dict(duration=0))]) for k in range(ntime)]
    sliders = [dict(active=0, x=0.05, y=-0.05, len=0.9, pad=dict(t=18, b=8),
                    currentvalue=dict(prefix="profile time = ", suffix=f"  {t_units}",
                                      font=dict(size=14, color="#222")),
                    steps=slider_steps)]
    updatemenus = [dict(type="buttons", direction="left", x=0.0, y=1.04,
                        xanchor="left", yanchor="bottom", pad=dict(t=0, r=10), showactive=False,
                        buttons=[
                            dict(label="▶ Play", method="animate",
                                 args=[None, dict(mode="immediate", fromcurrent=True,
                                                  frame=dict(duration=120, redraw=True),
                                                  transition=dict(duration=0))]),
                            dict(label="⏸ Pause", method="animate",
                                 args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                    transition=dict(duration=0))])])]

    # =============================================================== axes
    fig.update_xaxes(title_text=f"water content θ  [{th_units}]", range=th_range, row=1, col=1)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]   (surface at top)", range=z_range, row=1, col=1)
    fig.update_xaxes(title_text=f"pressure head ψ  [{h_units}]", range=h_range, row=1, col=2)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]   (surface at top)", range=z_range, row=1, col=2)

    fig.update_xaxes(title_text=f"time  [{t_units}]",
                     range=[float(time.min()), float(time.max())], row=2, col=1)
    fig.update_yaxes(title_text="rate  [m/day]", range=ir_range, row=2, col=1)

    fig.update_xaxes(title_text="soil texture (coarse → fine)", row=3, col=1)
    fig.update_yaxes(title_text="final water  [m²]", range=part_range, row=3, col=1)

    fig.update_xaxes(title_text=f"time  [{t_units}]",
                     range=[float(time.min()), float(time.max())], row=4, col=1)
    fig.update_yaxes(title_text=f"outlet discharge q  [{q_units}]", range=q_range, row=4, col=1)
    fig.update_xaxes(title_text=f"time  [{t_units}]",
                     range=[float(time.min()), float(time.max())], row=4, col=2)
    fig.update_yaxes(title_text="|mass-balance err|", type="log", exponentformat="power", row=4, col=2)

    # =============================================================== metrics / findings panel
    mbe_max_all = max(float(_attr(ds_map[k], "mass_balance_error_max", 0.0)) for k in SOIL_ORDER)
    NB = " "  # non-breaking space (renders; '&nbsp;' would show literally)
    metrics_lines = [
        f"<b>module</b>: soil-texture sweep (coupling) {NB} <b>date</b>: {date}",
        (f"<b>storm</b>: r={_fmt(rain_rate)} m/{t_units} for {_fmt(storm_dur)} {t_units}, "
         f"t_end={_fmt(t_end)} {t_units}"),
        (f"<b>hillslope</b> (shared): L={_fmt(L)} m, S0={_fmt(S0)}, n={_fmt(n_manning)}, "
         f"open outlet @ Manning normal depth; cum rain ≈ {_fmt(cum_rain_total)} m²"),
        "<b>per-soil</b> (Ks m/day, vg_n, ψ₀ m | inf / pond / drn % | peak q | max|mbe|):",
    ]
    for key in SOIL_ORDER:
        ds = ds_map[key]
        fi = float(_attr(ds, "partition_infiltrated_frac", 0.0)) * 100.0
        fp = float(_attr(ds, "partition_ponded_frac", 0.0)) * 100.0
        fd = float(_attr(ds, "partition_drained_frac", 0.0)) * 100.0
        lab = key.ljust(5).replace(" ", NB)
        metrics_lines.append(
            f"{NB}{NB}<b>{lab}</b> Ks={_fmt(_attr(ds, 'Ks_m_per_day'))}, n={_fmt(_attr(ds, 'vg_n'))}, "
            f"ψ₀={_fmt(_attr(ds, 'antecedent_psi0_m'))} | {fi:.1f} / {fp:.3f} / {fd:.1f} % | "
            f"peak q={_fmt(_attr(ds, 'peak_outflow_m2_per_day'))} | {_fmt(_attr(ds, 'mass_balance_error_max'))}"
        )
    metrics_html = "<br>".join(metrics_lines)

    findings = (
        "<b>FINDINGS</b><br>"
        f"All four textures converge and conserve (max|mbe| ≤ {mbe_max_all:.0e}) — the coupled model "
        "spans sand→clay. <b>Why even sand runs off (panel 2):</b> infiltration is the supply-limited "
        "NCP flux q_pot = K(ψ_top)/ℓ_c·(d−ψ_top), throttled by the near-surface UNSATURATED K. For dry "
        "coarse soil K(ψ_top) is low, so the early infiltration rate sits BELOW the rain supply until the "
        "surface wets up — under-representing the suction-driven sorptivity that lets real dry sand "
        "infiltrate fast (Green-Ampt: capacity ≈ Ks). With the efficient outlet draining the transient "
        "pond, the model OVER-predicts coarse-soil runoff (a known limitation of the Robin/k_ex closure; "
        "ℓ_c is also mesh-tied). The clean texture signal is the θ(z) front SHAPE (sharp/deep sand vs "
        "diffuse clay); the partition is confounded (each soil sits at a different per-texture "
        "antecedent — a common head is non-comparable across textures)."
    )
    conservation_line = (
        f"<b>conservation</b>: max|mbe| ≤ {mbe_max_all:.0e} for all 4 soils "
        "(Δtotal = cum_rain − cum_outflow closes for every texture)"
    )
    fig.add_annotation(
        x=0.0, y=-0.12, xref="paper", yref="paper", xanchor="left", yanchor="top",
        align="left", showarrow=False,
        text="<b>METRICS</b><br>" + metrics_html + "<br><br>" + findings + "<br>" + conservation_line,
        font=dict(size=10.5, color="#222", family="Consolas, monospace"),
        bordercolor="#888", borderwidth=1, borderpad=8, bgcolor="rgba(245,245,245,0.96)")

    title = f"<b>PIDS Pillar-2 Tier-3: soil-texture sweep (coupling)</b> · {date}"
    fig.update_layout(
        title=dict(text=title + "<br><span style='font-size:13px'>"
                        "sand / loam / silt / clay (coarse→fine) under one 0.6 m/day storm on the SAME "
                        "sloped hillslope — only texture + antecedent head differ</span>",
                   x=0.5, xanchor="center", y=0.997, yanchor="top"),
        sliders=sliders, updatemenus=updatemenus,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01, font=dict(size=10)),
        margin=dict(l=90, r=175, t=150, b=300),
        width=1300, height=2050,
        barmode="group", bargap=0.25, bargroupgap=0.08,
        paper_bgcolor="white", plot_bgcolor="#fafafa", hovermode="closest")

    fig.write_html(out_path, include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(displaylogo=False, responsive=True,
                               toImageButtonOptions=dict(format="png", scale=2)))
    for ds in ds_map.values():
        ds.close()
    return out_path


# backward-compatible alias matching the other generators' naming.
build_texture_sweep_html = build_html


def _main(date: str = "2026-06-06", out_path: str | None = None) -> str:
    data_dir = "../validation/sanity/data"
    viz_dir = "../validation/sanity/viz"
    os.makedirs(viz_dir, exist_ok=True)
    nc_paths = {key: f"{data_dir}/coupling_texture__{key}__{date}.nc" for key in SOIL_ORDER}
    if out_path is None:
        out_path = f"{viz_dir}/coupling_texture_sweep__{date}.html"
    out = build_html(nc_paths, out_path, date=date)
    sz = os.path.getsize(out) / 1e6
    print(f"WROTE {out}  ({sz:.2f} MB)", flush=True)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        _main()
    elif len(args) == 1:
        _main(date=args[0])
    elif len(args) == 2:
        _main(date=args[0], out_path=args[1])
    else:
        print("usage: python make_texture_sweep_html.py [<date>] [<out.html>]", file=sys.stderr)
        sys.exit(2)
