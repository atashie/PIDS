"""Reusable Tier-3 sanity visualizer for the PIDS Pillar-2 forward model (Module 2: overland flow).

Turns a standardized overland sanity-run NetCDF result (the data contract of
``governance/visualize-sanity-check-routine.md``) into ONE self-contained,
offline, interactive HTML. The generator reads ONLY the result file -- it never
imports or runs the solver.

For a storm-on-a-hillslope (overland sheet flow) the standard views (per the viz
catalog) are:
  1. SURFACE-DEPTH ANIMATION: the overland sheet depth d(x) vs distance, animated
     with a TIME SLIDER + play/pause so the sheet is seen building during the
     storm then receding.
  2. HYDROGRAPH + HYETOGRAPH (twin-axis): outlet outflow(t) as a line, rainfall(t)
     as inverted bars from the top, with the kinematic equilibrium r*L marked and
     a current-time marker that tracks the slider.
  3. DIAGNOSTICS: mass-balance error (log-y), Newton iterations, and max velocity
     vs time.
  4. METRICS panel: the deciding numbers (module/scenario/date, Manning n, slope,
     length, rain peak, equilibrium r*L vs peak outflow, peak velocity, MAX MBE).

Usage:
    python viz/make_overland_html.py <result.nc> <out.html>

Dependencies: xarray + plotly only. Plotly is vendored INLINE
(``include_plotlyjs=True``) so the HTML opens offline by double-click.
"""
from __future__ import annotations

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


# ----------------------------------------------------------------------------- builder
def build_overland_html(nc_path: str, html_path: str) -> str:
    """Build the self-contained interactive overland sanity HTML.

    Parameters
    ----------
    nc_path : str
        Path to the standardized overland sanity-run NetCDF result file.
    html_path : str
        Path to write the self-contained HTML (plotly inlined, offline).

    Returns
    -------
    str
        The ``html_path`` written.
    """
    ds = xr.open_dataset(nc_path)

    # --- coords / fields (item-access so ``ds['x']`` is the var, not a method) ---
    x = np.asarray(ds["x"].values, dtype=float)                       # (x,)  [m]
    time = np.asarray(ds["time"].values, dtype=float)                 # (time,) [day]
    depth = np.asarray(ds["surface_depth"].values, dtype=float)       # (time, x) [m]
    bed = np.asarray(ds["bed_elevation"].values, dtype=float)         # (x,) [m]
    outflow = np.asarray(ds["outflow"].values, dtype=float)          # (time,) [m2/day]
    rain = np.asarray(ds["rainfall"].values, dtype=float)            # (time,) [m/day]
    mbe = np.asarray(ds["mass_balance_error"].values, dtype=float)    # (time,) [-]
    niter = np.asarray(ds["newton_iters"].values, dtype=float)       # (time,) [-]
    vmax = np.asarray(ds["max_velocity"].values, dtype=float)        # (time,) [m/s]

    x_units = _units(ds["x"], "m")
    t_units = _units(ds["time"], "day")
    q_units = _units(ds["outflow"], "m2/day")
    r_units = _units(ds["rainfall"], "m/day")
    v_units = _units(ds["max_velocity"], "m/s")

    # depth is tiny (~mm) -> plot in mm with clear labelling.
    depth_mm = depth * 1000.0

    # --- global attrs / metrics ---
    module = _attr(ds, "module", "overland")
    scenario = _attr(ds, "scenario", "storm hydrograph")
    date = _attr(ds, "date", "")
    n_man = _attr(ds, "n_man")
    slope = _attr(ds, "slope")
    length_m = _attr(ds, "length_m")
    rain_peak = _attr(ds, "rain_peak_m_per_day")
    storm_dur = _attr(ds, "storm_duration_day")
    eq_rL = _attr(ds, "equilibrium_outflow_rL")
    peak_q = _attr(ds, "peak_outflow_m2_per_day")
    peak_v = _attr(ds, "peak_velocity_m_per_s")
    mbe_max = _attr(ds, "mass_balance_error_max", float(np.max(np.abs(mbe))))

    ntime = time.shape[0]

    # --- fixed axis ranges so the sheet "moves" rather than rescaling per frame ---
    depth_hi = float(np.nanmax(depth_mm))
    depth_pad = 0.06 * (depth_hi + 1e-12)
    depth_range = [0.0, depth_hi + depth_pad]

    q_hi = max(float(np.nanmax(outflow)), float(eq_rL) if eq_rL is not None else 0.0)
    q_pad = 0.08 * (q_hi + 1e-12)
    q_range = [0.0, q_hi + q_pad]

    # hyetograph drawn as bars descending from the top of the right axis.
    r_hi = float(np.nanmax(rain))
    r_top = (r_hi if r_hi > 0 else 1.0) * 1.05
    r_range = [0.0, r_top * 3.0]  # rain bars occupy only the top third (clear from hydrograph)

    # --- subplot layout ---
    #   row1 (colspan 2): surface-depth profile d(x)  [animated]
    #   row2 (colspan 2): hydrograph + hyetograph twin-axis  [current-time marker]
    #   row3 left: mass-balance error (log-y) + Newton iters ; row3 right: max velocity
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2, "secondary_y": True}, None],
            [{"type": "xy"}, {"type": "xy", "secondary_y": True}],
        ],
        row_heights=[0.36, 0.34, 0.30],
        vertical_spacing=0.11,
        horizontal_spacing=0.16,
        subplot_titles=(
            "Overland sheet depth  d(x)  along the hillslope  [mm]",
            "Outlet hydrograph (line) + hyetograph (inverted bars)",
            "Diagnostics: mass-balance error (log)",
            "Diagnostics: max velocity &amp; Newton iterations",
        ),
    )

    DEPTH_COLOR = "#1565c0"
    Q_COLOR = "#0277bd"
    RAIN_COLOR = "#90a4ae"
    EQ_COLOR = "#c62828"
    MBE_COLOR = "#e67e22"
    ITER_COLOR = "#6a1b9a"
    VEL_COLOR = "#2e7d32"
    MARK_COLOR = "#c0392b"

    # =============================================================== row1: depth profile
    i_depth = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=x, y=depth_mm[0], mode="lines",
            line=dict(color=DEPTH_COLOR, width=2.5),
            fill="tozeroy", fillcolor="rgba(21,101,192,0.18)",
            name="sheet depth d(x)",
            hovertemplate="x=%{x:.1f} m<br>d=%{y:.4f} mm<extra></extra>",
        ),
        row=1, col=1,
    )

    # =============================================================== row2: hydro + hyeto
    fig.add_trace(
        go.Scatter(
            x=time, y=outflow, mode="lines",
            line=dict(color=Q_COLOR, width=2.5),
            name=f"outflow  [{q_units}]",
            hovertemplate="t=%{x:.4f} day<br>Q=%{y:.4f} " + q_units + "<extra></extra>",
        ),
        row=2, col=1, secondary_y=False,
    )
    # equilibrium kinematic reference r*L
    if eq_rL is not None:
        fig.add_hline(
            y=float(eq_rL), line=dict(color=EQ_COLOR, width=1.6, dash="dash"),
            annotation_text=f"equilibrium r*L = {_fmt(float(eq_rL))} {q_units}",
            annotation_position="top left",
            annotation_font=dict(size=11, color=EQ_COLOR),
            row=2, col=1, secondary_y=False,
        )
    # hyetograph: inverted bars from the top (use base so they hang down from r_range top)
    fig.add_trace(
        go.Bar(
            x=time, y=rain, base=r_range[1] - rain,
            marker=dict(color=RAIN_COLOR), opacity=0.85,
            width=(time[1] - time[0]) * 0.9 if ntime > 1 else 0.005,
            name=f"rainfall  [{r_units}]",
            hovertemplate="t=%{x:.4f} day<br>r=%{base:.3f} " + r_units + "<extra></extra>",
        ),
        row=2, col=1, secondary_y=True,
    )
    # current-time marker on the hydrograph (tracks slider)
    i_qmark = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[outflow[0]], mode="markers",
            marker=dict(size=12, color=MARK_COLOR, symbol="circle-open",
                        line=dict(width=3, color=MARK_COLOR)),
            name="current time",
            hovertemplate="t=%{x:.4f} day<br>Q=%{y:.4f} " + q_units + "<extra>current</extra>",
        ),
        row=2, col=1, secondary_y=False,
    )

    # =============================================================== row3 left: MBE (single axis)
    mbe_plot = np.where(np.abs(mbe) <= 0, np.nan, np.abs(mbe))  # guard zeros for log axis
    fig.add_trace(
        go.Scatter(
            x=time, y=mbe_plot, mode="lines+markers",
            line=dict(color=MBE_COLOR, width=2),
            marker=dict(size=4, color=MBE_COLOR),
            name="|mass-balance error|",
            hovertemplate="t=%{x:.4f} day<br>err=%{y:.3e}<extra></extra>",
        ),
        row=3, col=1,
    )

    # ====================================== row3 right: max velocity (left) + Newton iters (right)
    # Newton iters share the velocity panel on its FAR-right secondary axis, so its label sits at
    # the figure edge (clear of col-1) rather than colliding with the velocity axis in the gap.
    fig.add_trace(
        go.Scatter(
            x=time, y=vmax, mode="lines+markers",
            line=dict(color=VEL_COLOR, width=2),
            marker=dict(size=4, color=VEL_COLOR),
            name=f"max velocity  [{v_units}]",
            hovertemplate="t=%{x:.4f} day<br>v_max=%{y:.4f} " + v_units + "<extra></extra>",
        ),
        row=3, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=niter, mode="lines+markers",
            line=dict(color=ITER_COLOR, width=1.6, dash="dot"),
            marker=dict(size=4, color=ITER_COLOR),
            name="Newton iters",
            hovertemplate="t=%{x:.4f} day<br>iters=%{y:.0f}<extra></extra>",
        ),
        row=3, col=2, secondary_y=True,
    )

    # =============================================================== animation frames
    frames = []
    for k in range(ntime):
        frames.append(
            go.Frame(
                name=str(k),
                data=[
                    go.Scatter(x=x, y=depth_mm[k]),       # i_depth
                    go.Scatter(x=[time[k]], y=[outflow[k]]),  # i_qmark
                ],
                traces=[i_depth, i_qmark],
            )
        )
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
        x=0.05, y=-0.02, len=0.9,
        pad=dict(t=30, b=10),
        currentvalue=dict(prefix="time = ", suffix=f"  {t_units}",
                          font=dict(size=14, color="#222")),
        steps=slider_steps,
    )]
    updatemenus = [dict(
        type="buttons", direction="left",
        x=0.05, y=1.10, xanchor="left", yanchor="top",
        pad=dict(t=0, r=10),
        showactive=False,
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(mode="immediate",
                                  fromcurrent=True,
                                  frame=dict(duration=140, redraw=True),
                                  transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0))]),
        ],
    )]

    # =============================================================== axes
    # row1: depth profile
    fig.update_xaxes(title_text=f"distance along hillslope  x  [{x_units}]  (0 = top, outlet at x = L)",
                     range=[float(x.min()), float(x.max())], row=1, col=1)
    fig.update_yaxes(title_text="sheet depth d  [mm]", range=depth_range, row=1, col=1,
                     zeroline=True, zerolinecolor="#bbb")

    # row2: hydrograph + hyetograph
    fig.update_xaxes(title_text=f"time  [{t_units}]",
                     range=[float(time.min()), float(time.max())], row=2, col=1)
    fig.update_yaxes(title_text=f"outflow Q  [{q_units}]", range=q_range,
                     row=2, col=1, secondary_y=False)
    # rain axis inverted-from-top: reversed so larger rain reaches further down.
    fig.update_yaxes(title_text=f"rainfall r  [{r_units}]", range=[r_range[1], r_range[0]],
                     row=2, col=1, secondary_y=True, showgrid=False)

    # row3 left: MBE (log), single axis
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=3, col=1)
    fig.update_yaxes(title_text="|mass-balance err|", type="log",
                     exponentformat="power", row=3, col=1)
    # row3 right: max velocity (left) + Newton iters (far-right secondary)
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=3, col=2)
    fig.update_yaxes(title_text=f"v_max  [{v_units}]", row=3, col=2, secondary_y=False,
                     rangemode="tozero")
    fig.update_yaxes(title_text="Newton iters", row=3, col=2, secondary_y=True,
                     showgrid=False, rangemode="tozero")

    # =============================================================== metrics panel
    metrics_lines = [
        f"<b>module</b>: {module}",
        f"<b>scenario</b>: {scenario}",
        f"<b>date</b>: {date}",
        (f"<b>Manning n</b> = {_fmt(n_man)} &nbsp; <b>slope</b> = {_fmt(slope)} "
         f"&nbsp; <b>L</b> = {_fmt(length_m)} m"),
        (f"<b>rain peak</b> = {_fmt(rain_peak)} m/{t_units} "
         f"&nbsp; <b>storm dur</b> = {_fmt(storm_dur)} {t_units}"),
        (f"<b>equilibrium r*L</b> = {_fmt(eq_rL)} {q_units}<br>"
         f"&nbsp;&nbsp;<b>peak outflow</b> = {_fmt(peak_q)} {q_units}"),
        f"<b>peak velocity</b> = {_fmt(peak_v)} {v_units}",
        f"<b>MAX mass-balance error</b>: {_fmt(float(mbe_max))}  (machine ~ 1e-14)",
    ]
    metrics_html = "<br>".join(metrics_lines)
    fig.add_annotation(
        x=1.005, y=1.0, xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        align="left", showarrow=False,
        text="<b>METRICS</b><br>" + metrics_html,
        font=dict(size=11, color="#222", family="Consolas, monospace"),
        bordercolor="#888", borderwidth=1, borderpad=8,
        bgcolor="rgba(245,245,245,0.95)",
    )

    title = f"{module} &middot; {scenario} &middot; {date}"
    fig.update_layout(
        title=dict(text="<b>PIDS Pillar-2 sanity (Tier-3)</b><br>"
                        f"<span style='font-size:13px'>{title}</span>",
                   x=0.5, xanchor="center", y=0.985, yanchor="top"),
        sliders=sliders,
        updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.015,
                    xanchor="right", x=1.0),
        margin=dict(l=75, r=330, t=140, b=80),
        width=1220, height=1060,
        barmode="overlay",
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        hovermode="closest",
    )

    # ---- write self-contained HTML (plotly inlined for offline use) ----
    fig.write_html(
        html_path,
        include_plotlyjs=True,          # inline the WHOLE plotly.js -> offline, no CDN
        full_html=True,
        auto_play=False,
        config=dict(displaylogo=False, responsive=True,
                    toImageButtonOptions=dict(format="png", scale=2)),
    )
    ds.close()
    return html_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python make_overland_html.py <result.nc> <out.html>", file=sys.stderr)
        sys.exit(2)
    out = build_overland_html(sys.argv[1], sys.argv[2])
    import os
    print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
