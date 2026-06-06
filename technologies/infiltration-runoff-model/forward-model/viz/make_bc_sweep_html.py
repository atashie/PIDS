"""Tier-3 sanity visualizer for the PIDS Pillar-2 forward model: OUTFLOW-BC SWEEP (coupling).

Compares FOUR outflow-boundary-condition runs of the coupled (surface<->subsurface)
hydrology model under ONE 0.6 m/day storm on a sloped hillslope, and turns them into
ONE self-contained, offline, interactive HTML. The generator reads ONLY the result
NetCDFs -- it never imports or runs the solver (this independence is the whole point of
the Tier-3 lens).

The four boundary conditions (one NetCDF each):
  - closed        : no outlet -- water is trapped and only routes downslope.
  - open_matched  : open point outlet at the bed slope S0 = 0.05.
  - open_steep    : open point outlet at 2*S0 = 0.10 (faster drainage).
  - open_shallow  : open point outlet at 0.5*S0 = 0.025 (slower drainage).

The headline story is CLOSED vs OPEN: a closed boundary traps the storm (it ponds into a
downhill wedge ~9 cm deep at the outlet) while ANY open outlet drains ~98 % of the rain
(the surface stays sub-mm). The three open variants nearly overlap -- a real finding: a
free point outlet is supply-limited, so outlet slope barely matters.

Panels (a multi-row subplot figure):
  1. OUTLET HYDROGRAPH q(t) [m2/day], one line per BC, with the storm-off line + a faint
     rainfall hyetograph (inverted bars on a twin axis). THE HEADLINE.
  2. WATER PARTITION per BC: grouped bars infiltrated / ponded / drained vs cum-rain total.
  3. PONDING-DEPTH PROFILE d(x) with a TIME SLIDER, LOG y so the 9 cm closed wedge AND
     the sub-mm open profiles are both readable; t=0 ghost; outlet at right.
  4. TOTAL STORED WATER delta total_water(t) per BC vs cum_rain(t) reference.
  5. CONSERVATION: mass_balance_error(t) per BC on a LOG y-axis (near machine precision).
  6. METRICS / FINDINGS panel (text from attrs).

Usage:
    python viz/make_bc_sweep_html.py [<date>] [<out.html>]
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


# the 4 BCs in plotting order, with consistent colors used across EVERY panel.
BC_ORDER = ["closed", "open_matched", "open_steep", "open_shallow"]
BC_COLOR = {
    "closed":       "#212121",   # dark gray / black -- traps water
    "open_matched": "#1565c0",   # blue  -- outlet @ S0
    "open_steep":   "#e65100",   # orange/red -- outlet @ 2*S0
    "open_shallow": "#2e7d32",   # green -- outlet @ 0.5*S0
}
BC_LABEL = {
    "closed":       "closed (no outlet)",
    "open_matched": "open @ S0=0.05",
    "open_steep":   "open @ 2&middot;S0=0.10",
    "open_shallow": "open @ 0.5&middot;S0=0.025",
}
PARTITION_COLOR = {"infiltrated": "#8d6e63", "ponded": "#0288d1", "drained": "#43a047"}


# ----------------------------------------------------------------------------- builder
def build_html(nc_paths: dict[str, str], out_path: str, date: str = "") -> str:
    """Build the self-contained interactive BC-sweep comparison HTML.

    Parameters
    ----------
    nc_paths : dict[str, str]
        Mapping ``bc_key -> path`` for the four standardized BC-sweep NetCDF results
        (keys: ``closed``, ``open_matched``, ``open_steep``, ``open_shallow``).
    out_path : str
        Path to write the self-contained HTML (plotly inlined, offline).
    date : str
        Date stamp for the title (falls back to the file attr if empty).

    Returns
    -------
    str
        The ``out_path`` written.
    """
    # ---- load all 4 datasets, keyed by bc_key, in canonical order -------------------
    ds_map: dict[str, xr.Dataset] = {}
    for key in BC_ORDER:
        if key not in nc_paths:
            raise KeyError(f"missing NetCDF for BC key {key!r}")
        ds_map[key] = xr.open_dataset(nc_paths[key])

    ref = ds_map[BC_ORDER[0]]
    x = np.asarray(ref["x"].values, dtype=float)            # (x,) [m]; outlet at max x = L
    time = np.asarray(ref["time"].values, dtype=float)      # (time,) [day]
    bed = np.asarray(ref["bed_elevation"].values, dtype=float)  # (x,) [m]
    rain = np.asarray(ref["rainfall"].values, dtype=float)  # (time,) [m/day]
    cum_rain = np.asarray(ref["cum_rain"].values, dtype=float)  # (time,) [m2]
    ntime = time.shape[0]

    x_units = _units(ref["x"], "m")
    t_units = _units(ref["time"], "day")
    q_units = _units(ref["outflow"], "m2/day")
    d_units = _units(ref["ponding_depth"], "m")
    r_units = _units(ref["rainfall"], "m/day")

    # shared scenario context (same across files except the outlet slope)
    storm_dur = _attr(ref, "storm_duration_day")
    t_end = _attr(ref, "t_end_day")
    rain_rate = _attr(ref, "rain_rate_m_per_day")
    Ks = _attr(ref, "Ks_m_per_day")
    S0 = _attr(ref, "bed_slope_S0")
    n_manning = _attr(ref, "n_manning")
    psi0 = _attr(ref, "antecedent_psi0_m")
    L = _attr(ref, "L")
    cum_rain_total = _attr(ref, "cum_rain_total_m2")
    if not date:
        date = _attr(ref, "date", "")

    # per-BC time series pulled once -------------------------------------------------
    series = {}
    for key in BC_ORDER:
        ds = ds_map[key]
        total = np.asarray(ds["total_water"].values, dtype=float)
        series[key] = dict(
            outflow=np.asarray(ds["outflow"].values, dtype=float),         # (time,) [m2/day]
            ponding=np.asarray(ds["ponding_depth"].values, dtype=float),   # (time, x) [m]
            total=total,                                                   # (time,) [m2]
            dtotal=total - total[0],                                       # delta storage [m2]
            cum_rain=np.asarray(ds["cum_rain"].values, dtype=float),       # (time,) [m2]
            mbe=np.asarray(ds["mass_balance_error"].values, dtype=float),  # (time,) [-]
        )

    # ---- fixed axis ranges (so slider frames don't rescale) ------------------------
    q_hi = max(float(np.nanmax(series[k]["outflow"])) for k in BC_ORDER)
    q_range = [0.0, max(q_hi * 1.08, 1e-9)]

    # ponding profile: log floor; both the ~9 cm wedge and sub-mm opens visible.
    D_FLOOR = 1e-6  # m -> 1 micron clip for the log axis
    d_hi = max(float(np.nanmax(series[k]["ponding"])) for k in BC_ORDER)
    d_log_range = [np.log10(D_FLOOR), np.log10(max(d_hi * 1.6, D_FLOOR * 10))]

    # delta-storage range
    dt_hi = max(float(np.nanmax(series[k]["dtotal"])) for k in BC_ORDER)
    dt_hi = max(dt_hi, float(np.nanmax(cum_rain)))
    dt_range = [0.0, dt_hi * 1.08]

    # =============================================================== figure layout
    #   row1 (colspan 2): outlet hydrograph q(t) + rain hyetograph (twin axis)  [HEADLINE]
    #   row2 (colspan 2): water partition grouped bars per BC
    #   row3 (colspan 2): ponding-depth profile d(x) [log-y]  [animated slider]
    #   row4 left : delta total stored water (t)         row4 right : conservation mbe(t) [log]
    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "xy", "colspan": 2, "secondary_y": True}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.26, 0.20, 0.28, 0.26],
        vertical_spacing=0.10,
        horizontal_spacing=0.12,
        subplot_titles=(
            "1 &middot; Outlet hydrograph  q(t)  [m&sup2;/day]",
            "2 &middot; Water partition per BC  [m&sup2;]",
            "3 &middot; Ponding-depth profile  d(x)  [m, log] &mdash; outlet at right",
            "4 &middot; &Delta; total stored water (t)  [m&sup2;]  vs cum-rain",
            "5 &middot; Conservation  |mass-balance error| (t)  [log]",
        ),
    )

    RAIN_COLOR = "#90a4ae"
    RAIN_BORDER = "#546e7a"
    CUM_COLOR = "#37474f"

    # =============================================================== row1: hydrograph q(t)
    for key in BC_ORDER:
        q = series[key]["outflow"]
        fig.add_trace(
            go.Scatter(
                x=time, y=q, mode="lines",
                line=dict(color=BC_COLOR[key], width=2.6),
                name=BC_LABEL[key], legendgroup=key,
                hovertemplate=("t=%{x:.4f} day<br>q=%{y:.4f} " + q_units
                               + "<extra>" + key + "</extra>"),
            ),
            row=1, col=1, secondary_y=False,
        )
    # rainfall hyetograph: inverted bars hanging from the top of a twin axis.
    r_hi = float(np.nanmax(rain))
    r_top = (r_hi if r_hi > 0 else 1.0) * 1.05
    r_axis_top = r_top * 3.0   # rain occupies only the top third (clear of the q lines)
    fig.add_trace(
        go.Bar(
            x=time, y=rain, base=r_axis_top - rain,
            marker=dict(color=RAIN_COLOR, line=dict(color=RAIN_BORDER, width=0.5)),
            opacity=0.75,
            width=(time[1] - time[0]) * 0.9 if ntime > 1 else 0.005,
            name=f"rainfall  [{r_units}]", legendgroup="rain",
            hovertemplate="t=%{x:.4f} day<br>r=%{base:.3f} " + r_units + "<extra>rain</extra>",
        ),
        row=1, col=1, secondary_y=True,
    )
    # rain-off marker (vertical line + annotation) at t = storm_duration.
    if storm_dur is not None:
        fig.add_vline(
            x=float(storm_dur), line=dict(color="#b71c1c", width=1.4, dash="dot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=float(storm_dur), y=q_range[1] * 0.96, xref="x", yref="y",
            text=f"rain off (t={_fmt(storm_dur)})", showarrow=False,
            font=dict(size=10, color="#b71c1c"), xanchor="left",
            bgcolor="rgba(255,255,255,0.7)",
        )

    # =============================================================== row2: water partition
    # grouped bars: one x-group per BC, three bars (infiltrated / ponded / drained).
    bc_x = [BC_LABEL[k].replace("&middot;", "·") for k in BC_ORDER]
    part = {
        "infiltrated": [float(_attr(ds_map[k], "final_infiltrated_m2", 0.0)) for k in BC_ORDER],
        "ponded":      [float(_attr(ds_map[k], "final_ponded_m2", 0.0)) for k in BC_ORDER],
        "drained":     [float(_attr(ds_map[k], "final_drained_m2", 0.0)) for k in BC_ORDER],
    }
    for comp in ("infiltrated", "ponded", "drained"):
        fig.add_trace(
            go.Bar(
                x=bc_x, y=part[comp], name=comp.capitalize(),
                marker=dict(color=PARTITION_COLOR[comp]),
                legendgroup="partition",
                hovertemplate="%{x}<br>" + comp + "=%{y:.5f} m&sup2;<extra></extra>",
            ),
            row=2, col=1,
        )
    # reference: cum-rain total per BC as a thin black outline marker.
    cum_per_bc = [float(_attr(ds_map[k], "cum_rain_total_m2", 0.0)) for k in BC_ORDER]
    fig.add_trace(
        go.Scatter(
            x=bc_x, y=cum_per_bc, mode="markers",
            marker=dict(symbol="line-ew", size=46, color="#212121",
                        line=dict(width=2.2, color="#212121")),
            name="cum rain (total)", legendgroup="partition",
            hovertemplate="%{x}<br>cum rain=%{y:.5f} m&sup2;<extra></extra>",
        ),
        row=2, col=1,
    )

    # =============================================================== row3: ponding profile d(x) [log]
    # t=0 ghost (initial state, faint) per BC, then the animated current-time line per BC.
    def _clip(arr):
        return np.maximum(arr, D_FLOOR)

    prof_trace_idx = {}
    for key in BC_ORDER:
        pond = series[key]["ponding"]
        # faint ghost initial profile (t=0)
        fig.add_trace(
            go.Scatter(
                x=x, y=_clip(pond[0]), mode="lines",
                line=dict(color=BC_COLOR[key], width=1.0, dash="dot"),
                opacity=0.35,
                name=f"{BC_LABEL[key]} (t=0)", legendgroup=key, showlegend=False,
                hovertemplate="x=%{x:.3f} m<br>d&#8320;=%{y:.3e} m<extra>" + key + " init</extra>",
            ),
            row=3, col=1,
        )
    for key in BC_ORDER:
        pond = series[key]["ponding"]
        prof_trace_idx[key] = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=x, y=_clip(pond[0]), mode="lines+markers",
                line=dict(color=BC_COLOR[key], width=2.6),
                marker=dict(size=4, color=BC_COLOR[key]),
                name=BC_LABEL[key], legendgroup=key, showlegend=False,
                hovertemplate="x=%{x:.3f} m<br>d=%{y:.3e} m<extra>" + key + "</extra>",
            ),
            row=3, col=1,
        )

    # =============================================================== row4 left: delta total water
    fig.add_trace(
        go.Scatter(
            x=time, y=cum_rain, mode="lines",
            line=dict(color=CUM_COLOR, width=3.0, dash="dash"),
            name="cumulative rain  [m&sup2;]", legendgroup="cumref",
            hovertemplate="t=%{x:.4f} day<br>cum rain=%{y:.5f} m&sup2;<extra></extra>",
        ),
        row=4, col=1,
    )
    for key in BC_ORDER:
        fig.add_trace(
            go.Scatter(
                x=time, y=series[key]["dtotal"], mode="lines",
                line=dict(color=BC_COLOR[key], width=2.4),
                name=BC_LABEL[key], legendgroup=key, showlegend=False,
                hovertemplate=("t=%{x:.4f} day<br>&Delta;stored=%{y:.5f} m&sup2;"
                               "<extra>" + key + "</extra>"),
            ),
            row=4, col=1,
        )

    # =============================================================== row4 right: conservation mbe(t)
    for key in BC_ORDER:
        mbe = series[key]["mbe"]
        mbe_plot = np.where(np.abs(mbe) <= 0, np.nan, np.abs(mbe))  # guard zeros for log
        fig.add_trace(
            go.Scatter(
                x=time, y=mbe_plot, mode="lines+markers",
                line=dict(color=BC_COLOR[key], width=1.8),
                marker=dict(size=3, color=BC_COLOR[key]),
                name=BC_LABEL[key], legendgroup=key, showlegend=False,
                hovertemplate="t=%{x:.4f} day<br>|mbe|=%{y:.3e}<extra>" + key + "</extra>",
            ),
            row=4, col=2,
        )

    # =============================================================== animation frames (row3 profile)
    frames = []
    for k in range(ntime):
        frame_data = []
        frame_traces = []
        for key in BC_ORDER:
            frame_data.append(go.Scatter(x=x, y=_clip(series[key]["ponding"][k])))
            frame_traces.append(prof_trace_idx[key])
        frames.append(go.Frame(name=str(k), data=frame_data, traces=frame_traces))
    fig.frames = frames

    # =============================================================== slider + controls
    slider_steps = [
        dict(
            method="animate",
            label=f"{time[k]:.3f}",
            args=[[str(k)], dict(mode="immediate",
                                 frame=dict(duration=0, redraw=True),
                                 transition=dict(duration=0))],
        )
        for k in range(ntime)
    ]
    sliders = [dict(
        active=0,
        x=0.05, y=-0.05, len=0.9,
        pad=dict(t=18, b=8),
        currentvalue=dict(prefix="profile time = ", suffix=f"  {t_units}",
                          font=dict(size=14, color="#222")),
        steps=slider_steps,
    )]
    updatemenus = [dict(
        type="buttons", direction="left",
        x=0.0, y=1.04, xanchor="left", yanchor="bottom",
        pad=dict(t=0, r=10),
        showactive=False,
        buttons=[
            dict(label="&#9654; Play", method="animate",
                 args=[None, dict(mode="immediate",
                                  fromcurrent=True,
                                  frame=dict(duration=120, redraw=True),
                                  transition=dict(duration=0))]),
            dict(label="&#9208; Pause", method="animate",
                 args=[[None], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0))]),
        ],
    )]

    # =============================================================== axes
    # row1: hydrograph q(t) + rain (inverted, twin axis)
    fig.update_xaxes(title_text=f"time  [{t_units}]",
                     range=[float(time.min()), float(time.max())], row=1, col=1)
    fig.update_yaxes(title_text=f"outlet discharge q  [{q_units}]", range=q_range,
                     row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"rain  [{r_units}]  (inverted)",
                     range=[r_axis_top, 0.0], row=1, col=1, secondary_y=True, showgrid=False)

    # row2: partition bars
    fig.update_xaxes(title_text="boundary condition", row=2, col=1)
    fig.update_yaxes(title_text="final water  [m&sup2;]", row=2, col=1, rangemode="tozero")

    # row3: ponding profile d(x) log-y; outlet at right (x increases to the outlet at L)
    fig.update_xaxes(title_text=f"horizontal position x  [{x_units}]   (outlet at right, x=L)",
                     range=[float(x.min()), float(x.max())], row=3, col=1)
    fig.update_yaxes(title_text=f"ponding depth d  [{d_units}, log]", type="log",
                     range=d_log_range, exponentformat="power", row=3, col=1)

    # row4 left: delta total stored water
    fig.update_xaxes(title_text=f"time  [{t_units}]",
                     range=[float(time.min()), float(time.max())], row=4, col=1)
    fig.update_yaxes(title_text="&Delta; stored water  [m&sup2;]", range=dt_range,
                     row=4, col=1)

    # row4 right: conservation mbe(t) log
    fig.update_xaxes(title_text=f"time  [{t_units}]",
                     range=[float(time.min()), float(time.max())], row=4, col=2)
    fig.update_yaxes(title_text="|mass-balance err|", type="log",
                     exponentformat="power", row=4, col=2)

    # =============================================================== metrics / findings panel
    peak_qs = [float(_attr(ds_map[k], "peak_outflow_m2_per_day", 0.0))
               for k in ("open_matched", "open_steep", "open_shallow")]
    q_spread = (max(peak_qs) - min(peak_qs)) / max(min(peak_qs), 1e-30) * 100.0
    mbe_max_all = max(float(_attr(ds_map[k], "mass_balance_error_max", 0.0)) for k in BC_ORDER)

    metrics_lines = [
        f"<b>module</b>: BC sweep (coupling) &nbsp; <b>date</b>: {date}",
        (f"<b>storm</b>: r={_fmt(rain_rate)} m/{t_units} for {_fmt(storm_dur)} {t_units}, "
         f"t_end={_fmt(t_end)} {t_units}"),
        (f"<b>hillslope</b>: L={_fmt(L)} m, S0={_fmt(S0)}, Ks={_fmt(Ks)} m/{t_units}, "
         f"n={_fmt(n_manning)}, &psi;&#8320;={_fmt(psi0)} m"),
        f"<b>cum rain total</b> &asymp; {_fmt(cum_rain_total)} m&sup2;",
    ]
    metrics_html = "<br>".join(metrics_lines)

    findings = (
        "<b>FINDINGS</b><br>"
        "Closed traps water (pond &rarr; ~9 cm downhill wedge); any open outlet drains "
        "~98% (surface stays sub-mm). Outlet slope (&frac12;&times;&ndash;2&times; S0) barely "
        f"discriminates (~{q_spread:.0f}% spread in peak q) &mdash; a free point outlet is "
        "supply-limited (q_out passes ~the rain supply at sub-mm depth), matching Module-2's "
        "1-D kinematic outlet."
    )
    conservation_line = (
        f"<b>conservation</b>: max|mbe| &le; {mbe_max_all:.0e} for all 4 BCs "
        "(&Delta;total = cum_rain &minus; cum_outflow closes)"
    )

    fig.add_annotation(
        x=0.0, y=-0.13, xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        align="left", showarrow=False,
        text="<b>METRICS</b><br>" + metrics_html
             + "<br><br>" + findings
             + "<br>" + conservation_line,
        font=dict(size=10.5, color="#222", family="Consolas, monospace"),
        bordercolor="#888", borderwidth=1, borderpad=8,
        bgcolor="rgba(245,245,245,0.96)",
    )

    title = ("<b>PIDS Pillar-2 Tier-3: outflow-BC sweep (coupling)</b> &middot; "
             f"{date}")
    fig.update_layout(
        title=dict(text=title + "<br><span style='font-size:13px'>"
                        "closed vs open point outlet (matched / steep / shallow) under one "
                        "0.6 m/day storm on a sloped hillslope</span>",
                   x=0.5, xanchor="center", y=0.995, yanchor="top"),
        sliders=sliders,
        updatemenus=updatemenus,
        legend=dict(orientation="v", yanchor="top", y=1.0,
                    xanchor="left", x=1.01, font=dict(size=10)),
        margin=dict(l=90, r=175, t=150, b=300),
        width=1300, height=1750,
        barmode="group",
        bargap=0.25, bargroupgap=0.08,
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        hovermode="closest",
    )

    # ---- write self-contained HTML (plotly inlined for offline use) ----
    fig.write_html(
        out_path,
        include_plotlyjs=True,          # inline the WHOLE plotly.js -> offline, no CDN
        full_html=True,
        auto_play=False,
        config=dict(displaylogo=False, responsive=True,
                    toImageButtonOptions=dict(format="png", scale=2)),
    )
    for ds in ds_map.values():
        ds.close()
    return out_path


# backward-compatible alias matching the other generators' naming.
build_bc_sweep_html = build_html


def _main(date: str = "2026-06-06", out_path: str | None = None) -> str:
    data_dir = "../validation/sanity/data"
    viz_dir = "../validation/sanity/viz"
    os.makedirs(viz_dir, exist_ok=True)
    nc_paths = {key: f"{data_dir}/coupling_bc__{key}__{date}.nc" for key in BC_ORDER}
    if out_path is None:
        out_path = f"{viz_dir}/coupling_bc_sweep__{date}.html"
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
        print("usage: python make_bc_sweep_html.py [<date>] [<out.html>]", file=sys.stderr)
        sys.exit(2)
