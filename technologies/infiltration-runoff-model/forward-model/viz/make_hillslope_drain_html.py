"""Tier-3 sanity visualizer: 2-D hillslope cross-section with a buried tile drain.

Turns the standardized hillslope-drain sanity NetCDF (the data contract of
``governance/visualize-sanity-check-routine.md``) into ONE self-contained,
offline, interactive HTML. The generator reads ONLY the result file -- it never
imports or runs the solver (independent visual evidence).

The headline view is a (x, z) cross-section HEATMAP of pressure head psi over the
20 m x 2 m vertical transect, animated with a TIME SLIDER + play/pause. On top:
the WATER TABLE (psi = 0) contour and the buried-drain footprint at the base.
The viewer should SEE the drawdown cone carve down over the drain (x ~ 10 m) as
the storm progresses, then hold through recession. A second panel animates the
water-table elevation profile water_table(x); a third panel shows the time-series
diagnostics (drain discharge, cumulative drainage, rainfall hyetograph,
mass-balance error) with a moving current-time marker. A paper-space metrics block
surfaces the deciding numbers and the honest framing note.

Honest framing: this drain is the VALIDATED Module-3 *resolved* drainage BC (a
drain modelled the expensive, fully-resolved way), NOT the Module-4 sub-grid
embedded feature (that coupled claim was retracted). The title and metrics say so.

Usage:
    python viz/make_hillslope_drain_html.py <result.nc> <out.html>

Dependencies: xarray + plotly only. Plotly is vendored INLINE
(``include_plotlyjs=True``) so the HTML opens offline by double-click on Windows.
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
    if isinstance(val, (np.generic,)):
        val = val.item()
    if isinstance(val, float):
        if val != 0 and (abs(val) < 1e-3 or abs(val) >= 1e4):
            return f"{val:.3e}"
        return f"{val:.{nd}g}"
    return str(val)


def _units(da, fallback: str = "") -> str:
    return str(da.attrs.get("units", fallback))


# ----------------------------------------------------------------------------- builder
def build_hillslope_drain_html(nc_path: str, html_path: str) -> str:
    """Build the self-contained interactive hillslope tile-drain HTML.

    Parameters
    ----------
    nc_path : str
        Path to the standardized hillslope-drain sanity NetCDF result.
    html_path : str
        Path to write the self-contained HTML (plotly inlined, offline).

    Returns
    -------
    str
        The ``html_path`` written.
    """
    ds = xr.open_dataset(nc_path)

    # --- coords ---
    x = np.asarray(ds["x"].values, dtype=float)            # (x,)  0 -> 20, outlet at 20
    z = np.asarray(ds["z"].values, dtype=float)            # (z,)  0 = base/drain depth -> 2 = top
    time = np.asarray(ds["time"].values, dtype=float)      # (time,) day

    # --- fields ---
    head = np.asarray(ds["head_field"].values, dtype=float)    # (time, z, x)  psi [m]
    theta = np.asarray(ds["theta_field"].values, dtype=float)  # (time, z, x)  theta [-]
    wt = np.asarray(ds["water_table"].values, dtype=float)     # (time, x)  z of psi=0 [m] (NaN below base)

    q_drain = np.asarray(ds["drain_discharge"].values, dtype=float)  # (time,) m^2/day
    cum_drain = np.asarray(ds["cum_drainage"].values, dtype=float)   # (time,) m^2
    rain = np.asarray(ds["rainfall"].values, dtype=float)           # (time,) m/day
    mbe = np.asarray(ds["mass_balance_error"].values, dtype=float)  # (time,) [-]

    x_units = _units(ds["x"], "m")
    z_units = _units(ds["z"], "m")
    t_units = _units(ds["time"], "day")
    head_units = _units(ds["head_field"], "m")
    q_units = _units(ds["drain_discharge"], "m^2/day")
    cum_units = _units(ds["cum_drainage"], "m^2")
    rain_units = _units(ds["rainfall"], "m/day")

    ntime = time.shape[0]

    # --- global attrs / metrics ---
    module = _attr(ds, "module", "m4_hillslope_drain")
    scenario = _attr(ds, "scenario", "hillslope tile drain")
    date = _attr(ds, "date", "")
    domain = _attr(ds, "domain", "")
    mesh = _attr(ds, "mesh", "")
    drain_loc = _attr(ds, "drain_location", "")
    drain_C = _attr(ds, "drain_conductance_per_day")
    drain_head = _attr(ds, "drain_external_head_m")
    slope = _attr(ds, "surface_slope")
    init_wt = _attr(ds, "initial_water_table_m")
    drawdown = _attr(ds, "drawdown_m")
    wt_far = _attr(ds, "water_table_farfield_x2_m")
    wt_cone = _attr(ds, "water_table_cone_min_m")
    cum_drain_tot = _attr(ds, "cum_drainage_m2")
    cum_rain_tot = _attr(ds, "cum_rain_m2")
    cum_outflow = _attr(ds, "cum_outflow_m2")
    mbe_final = _attr(ds, "mass_balance_error_final", float(mbe[-1]))
    mbe_max = _attr(ds, "mass_balance_error_max", float(np.max(np.abs(mbe))))
    note = _attr(ds, "note", "")
    wall_clock = _attr(ds, "wall_clock_s")
    steps = _attr(ds, "steps")
    status = _attr(ds, "status", "")

    # --- drain footprint x-extent (parse from coords: x in [9.5, 10.5]) ---
    DRAIN_X0, DRAIN_X1 = 9.5, 10.5

    # ----------------------------------------------------------------- color scaling
    # psi heatmap: fixed across frames so the cone "carves" rather than rescales.
    psi_lo = float(np.nanmin(head))
    psi_hi = float(np.nanmax(head))
    # symmetric-ish around 0 so the water table (psi=0) sits at a clear color break.
    psi_abs = max(abs(psi_lo), abs(psi_hi))
    psi_range = [-psi_abs, psi_abs]

    # ----------------------------------------------------------------- subplot layout
    # row1: cross-section heatmap (colspan 2)
    # row2: water-table profile (col 1)  |  diagnostics: discharge + cum (col 2)
    # row3: rainfall hyetograph (col 1)  |  mass-balance error log-y (col 2)
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy", "secondary_y": True}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.50, 0.27, 0.23],
        vertical_spacing=0.085,
        horizontal_spacing=0.11,
        subplot_titles=(
            f"Cross-section: pressure head psi(x, z)  [{head_units}]  "
            f"&mdash; water table = psi 0 contour; drain at base x in [9.5, 10.5] m",
            "Water-table elevation profile  water_table(x)",
            "Drain discharge &amp; cumulative drainage",
            "Rainfall hyetograph",
            "Diagnostics: |mass-balance error| vs time",
        ),
    )

    # ============================================================ ROW 1: cross-section
    # static water-table fill band illustrating "saturated below the table" is implicit
    # in psi>0; we draw the psi heatmap then overlay the water-table contour line.
    i_heat = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            x=x, y=z, z=head[0],
            zmin=psi_range[0], zmax=psi_range[1],
            colorscale="RdBu",
            colorbar=dict(
                title=dict(text=f"psi [{head_units}]", side="right"),
                len=0.46, y=0.78, yanchor="middle", x=1.005, thickness=14,
            ),
            hovertemplate="x=%{x:.2f} m<br>z=%{y:.2f} m<br>psi=%{z:.3f} m<extra></extra>",
            name="psi",
            zsmooth="best",
        ),
        row=1, col=1,
    )

    # water-table contour (psi = 0 crossing per column). Drawn as a line over the heatmap.
    # connectgaps=False so the NaN drawdown column (centerline below base) shows a break.
    i_wt_cross = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=x, y=wt[0], mode="lines",
            line=dict(color="#000000", width=3),
            connectgaps=False,
            name="water table (psi=0)",
            hovertemplate="x=%{x:.2f} m<br>water table z=%{y:.3f} m<extra></extra>",
        ),
        row=1, col=1,
    )

    # drain footprint marker: a heavy band at the base over x in [9.5,10.5].
    fig.add_shape(
        type="rect",
        x0=DRAIN_X0, x1=DRAIN_X1, y0=-0.04, y1=0.06,
        line=dict(color="#00c853", width=2),
        fillcolor="rgba(0,200,83,0.65)",
        row=1, col=1, layer="above",
    )
    fig.add_annotation(
        x=0.5 * (DRAIN_X0 + DRAIN_X1), y=0.18, ax=0, ay=0,
        xref="x", yref="y",
        text="<b>tile drain</b>", showarrow=False,
        font=dict(size=11, color="#00501f"),
        row=1, col=1,
    )

    # ============================================================ ROW 2 col1: WT profile
    i_wt_prof = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=x, y=wt[0], mode="lines+markers",
            line=dict(color="#1565c0", width=2.5),
            marker=dict(size=4, color="#1565c0"),
            connectgaps=False,
            name="water table z(x)",
            hovertemplate="x=%{x:.2f} m<br>z=%{y:.3f} m<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )
    # initial water-table reference (flat) + base line
    if init_wt is not None:
        fig.add_hline(
            y=float(init_wt), line=dict(color="#9e9e9e", width=1.2, dash="dash"),
            row=2, col=1, layer="below",
            annotation_text=f"initial table {float(init_wt):.2f} m",
            annotation_position="top left",
            annotation_font=dict(size=9, color="#757575"),
        )
    fig.add_vrect(
        x0=DRAIN_X0, x1=DRAIN_X1, fillcolor="rgba(0,200,83,0.18)",
        line_width=0, row=2, col=1, layer="below",
    )

    # ============================================================ ROW 2 col2: discharge + cum (twin-y)
    fig.add_trace(
        go.Scatter(
            x=time, y=q_drain, mode="lines",
            line=dict(color="#00897b", width=2),
            name="drain discharge",
            hovertemplate="t=%{x:.3f} day<br>q=%{y:.4f} " + q_units + "<extra></extra>",
        ),
        row=2, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=cum_drain, mode="lines",
            line=dict(color="#5e35b1", width=2, dash="dot"),
            name="cumulative drainage",
            hovertemplate="t=%{x:.3f} day<br>cum=%{y:.4f} " + cum_units + "<extra></extra>",
        ),
        row=2, col=2, secondary_y=True,
    )
    i_mark_q = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[q_drain[0]], mode="markers",
            marker=dict(size=11, color="#c62828", symbol="circle-open",
                        line=dict(width=3, color="#c62828")),
            name="current time", showlegend=True,
            hovertemplate="t=%{x:.3f} day<extra>now</extra>",
        ),
        row=2, col=2, secondary_y=False,
    )

    # ============================================================ ROW 3 col1: rainfall hyetograph
    # bar width ~ time step
    dt = float(np.median(np.diff(time))) if ntime > 1 else 0.1
    fig.add_trace(
        go.Bar(
            x=time, y=rain, width=0.9 * dt,
            marker=dict(color="#1e88e5"),
            name="rainfall",
            hovertemplate="t=%{x:.3f} day<br>rain=%{y:.3f} " + rain_units + "<extra></extra>",
            showlegend=False,
        ),
        row=3, col=1,
    )
    i_mark_rain = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[rain[0]], mode="markers",
            marker=dict(size=11, color="#c62828", symbol="line-ns",
                        line=dict(width=3, color="#c62828")),
            name="now", showlegend=False,
            hovertemplate="t=%{x:.3f} day<extra>now</extra>",
        ),
        row=3, col=1,
    )

    # ============================================================ ROW 3 col2: mass-balance error
    mbe_abs = np.abs(mbe)
    mbe_plot = np.where(mbe_abs <= 0, np.nan, mbe_abs)  # guard zeros for log axis
    fig.add_trace(
        go.Scatter(
            x=time, y=mbe_plot, mode="lines+markers",
            line=dict(color="#e67e22", width=2),
            marker=dict(size=4, color="#e67e22"),
            name="|mass-balance error|", showlegend=False,
            hovertemplate="t=%{x:.3f} day<br>err=%{y:.3e}<extra></extra>",
        ),
        row=3, col=2,
    )
    t0_err = mbe_plot[0] if np.isfinite(mbe_plot[0]) else np.nanmin(mbe_plot)
    i_mark_mbe = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[t0_err], mode="markers",
            marker=dict(size=11, color="#c62828", symbol="circle-open",
                        line=dict(width=3, color="#c62828")),
            name="now", showlegend=False,
            hovertemplate="t=%{x:.3f} day<extra>now</extra>",
        ),
        row=3, col=2,
    )

    # ----------------------------------------------------------------- animation frames
    frames = []
    for k in range(ntime):
        cur_err = mbe_plot[k] if np.isfinite(mbe_plot[k]) else np.nanmin(mbe_plot)
        frames.append(
            go.Frame(
                name=str(k),
                data=[
                    go.Heatmap(z=head[k]),
                    go.Scatter(x=x, y=wt[k]),          # cross-section WT contour
                    go.Scatter(x=x, y=wt[k]),          # WT profile panel
                    go.Scatter(x=[time[k]], y=[q_drain[k]]),
                    go.Scatter(x=[time[k]], y=[rain[k]]),
                    go.Scatter(x=[time[k]], y=[cur_err]),
                ],
                traces=[i_heat, i_wt_cross, i_wt_prof, i_mark_q, i_mark_rain, i_mark_mbe],
            )
        )
    fig.frames = frames

    # ----------------------------------------------------------------- slider + controls
    slider_steps = [
        dict(
            method="animate",
            label=f"{time[k]:.2f}",
            args=[[str(k)], dict(mode="immediate",
                                 frame=dict(duration=0, redraw=True),
                                 transition=dict(duration=0))],
        )
        for k in range(ntime)
    ]
    sliders = [dict(
        active=0,
        x=0.05, y=-0.045, len=0.9,
        pad=dict(t=28, b=8),
        currentvalue=dict(prefix="time = ", suffix=f"  {t_units}",
                          font=dict(size=14, color="#222")),
        steps=slider_steps,
    )]
    updatemenus = [dict(
        type="buttons", direction="left",
        x=0.05, y=1.085, xanchor="left", yanchor="top",
        pad=dict(t=0, r=10),
        showactive=False,
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(mode="immediate", fromcurrent=True,
                                  frame=dict(duration=200, redraw=True),
                                  transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0))]),
        ],
    )]

    # ----------------------------------------------------------------- axes
    # row1 cross-section
    fig.update_xaxes(
        title_text=f"x  [{x_units}]  (downslope; surface outlet at 20 m)",
        range=[float(x.min()), float(x.max())], row=1, col=1,
        constrain="domain",
    )
    fig.update_yaxes(
        title_text=f"z  [{z_units}]  (0 = drain depth / base; top = surface)",
        range=[float(z.min()), float(z.max())], row=1, col=1,
    )
    # row2 col1 WT profile
    fig.update_xaxes(title_text=f"x  [{x_units}]", range=[float(x.min()), float(x.max())],
                     row=2, col=1)
    fig.update_yaxes(title_text=f"water-table z  [{z_units}]",
                     range=[float(z.min()), float(z.max())], row=2, col=1)
    # row2 col2 discharge twin
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=2, col=2)
    fig.update_yaxes(title_text=f"discharge  [{q_units}]", row=2, col=2,
                     secondary_y=False, color="#00695c")
    fig.update_yaxes(title_text=f"cum  [{cum_units}]", row=2, col=2,
                     secondary_y=True, color="#5e35b1", showgrid=False)
    # row3 col1 rainfall
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=3, col=1)
    fig.update_yaxes(title_text=f"rain  [{rain_units}]", row=3, col=1,
                     rangemode="tozero")
    # row3 col2 mbe (log)
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=3, col=2)
    fig.update_yaxes(title_text="|mass-balance error|  (-)",
                     type="log", exponentformat="power", row=3, col=2)

    # ----------------------------------------------------------------- metrics panel
    metrics_lines = [
        f"<b>module</b>: {module}",
        f"<b>scenario</b>: 2-D hillslope transect, uniform loam",
        f"<b>domain</b>: {domain}",
        f"<b>mesh</b>: {mesh}",
        f"<b>drain</b>: {drain_loc}",
        (f"&nbsp;&nbsp;conductance C = {_fmt(drain_C)} /day &nbsp; "
         f"external head = {_fmt(drain_head)} m"),
        f"<b>surface slope</b>: {_fmt(slope)} &nbsp; initial table = {_fmt(init_wt)} m",
        "<b>&mdash; headline numbers &mdash;</b>",
        (f"&nbsp;&nbsp;drawdown at drain = <b>{_fmt(drawdown)} m</b> "
         f"(far-field {_fmt(wt_far)} m → cone {_fmt(wt_cone)} m)"),
        f"&nbsp;&nbsp;cumulative drainage = {_fmt(cum_drain_tot)} {cum_units}",
        f"&nbsp;&nbsp;cumulative rain = {_fmt(cum_rain_tot)} {cum_units}",
        (f"&nbsp;&nbsp;surface runoff = {_fmt(cum_outflow)} {cum_units} "
         f"(≈ 0: rain &lt; Ks, all infiltrates)"),
        (f"&nbsp;&nbsp;mass-balance error: final {_fmt(mbe_final)}, "
         f"max {_fmt(mbe_max)}  (machine ~ 1e-13)"),
        f"&nbsp;&nbsp;status {status} in {_fmt(steps)} steps, {_fmt(wall_clock)} s",
        "<b>&mdash; framing (read me) &mdash;</b>",
        ("<span style='color:#b71c1c'>This is the VALIDATED Module-3 <i>resolved</i> "
         "drainage BC<br>&nbsp;&nbsp;(a drain modelled the expensive, fully-resolved "
         "way),<br>&nbsp;&nbsp;NOT the Module-4 sub-grid embedded feature<br>"
         "&nbsp;&nbsp;(that coupled claim was retracted). Illustration only.</span>"),
    ]
    metrics_html = "<br>".join(metrics_lines)
    fig.add_annotation(
        x=1.004, y=0.49, xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        align="left", showarrow=False,
        text="<b>METRICS</b><br>" + metrics_html,
        font=dict(size=10.5, color="#222", family="Consolas, monospace"),
        bordercolor="#888", borderwidth=1, borderpad=8,
        bgcolor="rgba(245,245,245,0.97)",
    )

    title = f"{module} &middot; {scenario[:60]}… &middot; {date}"
    fig.update_layout(
        title=dict(
            text="<b>PIDS Pillar-2 sanity (Tier-3) &middot; hillslope tile-drain (resolved) "
                 "&middot; 2026-06-09</b><br>"
                 "<span style='font-size:12px'>buried tile drain carves a water-table "
                 "drawdown cone in a 20 m &times; 2 m loam hillslope "
                 "&mdash; resolved Module-3 BC, not the retracted M4 embedded feature</span>",
            x=0.5, xanchor="center", y=0.985, yanchor="top",
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.018,
                    xanchor="right", x=1.0, font=dict(size=10)),
        margin=dict(l=75, r=430, t=140, b=80),
        width=1320, height=1080,
        paper_bgcolor="white",
        plot_bgcolor="#f7f7f7",
        bargap=0.1,
        hovermode="closest",
    )

    # ----------------------------------------------------------------- write HTML
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
        print("usage: python make_hillslope_drain_html.py <result.nc> <out.html>",
              file=sys.stderr)
        sys.exit(2)
    out = build_hillslope_drain_html(sys.argv[1], sys.argv[2])
    import os
    print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
