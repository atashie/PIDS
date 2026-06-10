"""Tier-3 sanity visualizer: hillslope tile drain over a TIGHT-CLAY subsoil.

Companion to ``make_hillslope_drain_html.py`` (the pure-loam drawdown-cone case).
This generator turns the loam-over-tight-clay hillslope-drain sanity NetCDF into
ONE self-contained, offline, interactive HTML. It reads ONLY the result file --
it never imports or runs the solver (independent visual evidence).

THE PHYSICS to make visible
---------------------------
A 20 m x 2 m vertical hillslope transect with a LOAM topsoil (Ks 0.25 m/day,
theta_s 0.43) over a TIGHT CLAY subsoil (Ks 0.005 m/day, theta_s 0.38), split at
the interface z = 1.0 m. A 0.085 m/day storm runs for 1.5 d then recesses. Because
the rain rate >> the clay Ks (0.085 >> 0.005), water cannot drain down through the
clay: it PERCHES on top of the interface and the loam SATURATES from the interface
up (a perched saturated band, theta -> theta_s = 0.43 just above z = 1.0). A
generous head-controlled tile drain sits buried in the clay at the base
(x in [9.5, 10.5] m, z = 0), but the clay starves it: it barely dents the table.

Contrast with the pure-loam run: loam drew a clean ~1.0 m drawdown cone over the
drain; HERE the water table actually RISES (0.7 m -> ~1.46 m, waterlogging the
root zone) and the only "drawdown" is the 0.085 m the drain column sits below the
far-field table (far-field 1.459 m vs cone 1.374 m). The clay is the drainage
BOTTLENECK.

VIEWS
-----
- Headline psi(x, z) cross-section HEATMAP with a TIME SLIDER + play; the water
  table (psi = 0) contour; the buried-drain footprint marked; and a horizontal
  line at the CLAY INTERFACE z = 1.0 (loam above / tight clay below). The viewer
  SEES the table rise and the weak drain notch.
- theta(x, z) cross-section HEATMAP (same slider) -- the perched SATURATED band
  (theta -> 0.43) builds in the loam directly above the clay interface; this reads
  the waterlogging far more clearly than psi alone.
- Animated water-table(x) elevation profile (the table rising, not coning down).
- Diagnostics (time series + moving marker): drain discharge + cumulative
  drainage; the rainfall hyetograph; |mass-balance error| on log-y.
- A paper-space METRICS panel with the layering, the loam-vs-clay CONTRAST, the
  perched-zone note, and the honest framing.

Honest framing: this is an ILLUSTRATION on the VALIDATED Module-3 *resolved*
drainage BC (a drain modelled the expensive, fully-resolved way) running on the
validated CoupledProblem engine -- it is NOT a newly validated module, and NOT the
Module-4 sub-grid embedded feature (that coupled claim was retracted).

Usage:
    python viz/make_hillslope_drain_clay_html.py <result.nc> <out.html>

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
def build_hillslope_drain_clay_html(nc_path: str, html_path: str) -> str:
    """Build the self-contained interactive loam-over-tight-clay hillslope HTML.

    Parameters
    ----------
    nc_path : str
        Path to the standardized hillslope-drain (clay subsoil) sanity NetCDF.
    html_path : str
        Path to write the self-contained HTML (plotly inlined, offline).

    Returns
    -------
    str
        The ``html_path`` written.
    """
    ds = xr.open_dataset(nc_path)

    # --- coords ---
    x = np.asarray(ds["x"].values, dtype=float)            # (x,)  0 -> 20
    z = np.asarray(ds["z"].values, dtype=float)            # (z,)  0 = base/drain -> 2 = surface
    time = np.asarray(ds["time"].values, dtype=float)      # (time,) day

    # --- fields ---
    head = np.asarray(ds["head_field"].values, dtype=float)    # (time, z, x)  psi [m]
    theta = np.asarray(ds["theta_field"].values, dtype=float)  # (time, z, x)  theta [-]
    wt = np.asarray(ds["water_table"].values, dtype=float)     # (time, x)  z of psi=0 [m]

    q_drain = np.asarray(ds["drain_discharge"].values, dtype=float)  # (time,) m^2/day
    cum_drain = np.asarray(ds["cum_drainage"].values, dtype=float)   # (time,) m^2
    rain = np.asarray(ds["rainfall"].values, dtype=float)           # (time,) m/day
    mbe = np.asarray(ds["mass_balance_error"].values, dtype=float)  # (time,) [-]
    Ks_profile = np.asarray(ds["Ks_profile"].values, dtype=float)   # (z,) m/day

    x_units = _units(ds["x"], "m")
    z_units = _units(ds["z"], "m")
    t_units = _units(ds["time"], "day")
    head_units = _units(ds["head_field"], "m")
    theta_units = _units(ds["theta_field"], "-")
    q_units = _units(ds["drain_discharge"], "m^2/day")
    cum_units = _units(ds["cum_drainage"], "m^2")
    rain_units = _units(ds["rainfall"], "m/day")

    ntime = time.shape[0]

    # --- global attrs / metrics ---
    module = _attr(ds, "module", "m4_hillslope_drain_clay")
    scenario = _attr(ds, "scenario", "hillslope tile drain, clay subsoil")
    date = _attr(ds, "date", "2026-06-09")
    domain = _attr(ds, "domain", "")
    mesh = _attr(ds, "mesh", "")
    drain_loc = _attr(ds, "drain_location", "")
    drain_C = _attr(ds, "drain_conductance_per_day")
    drain_head = _attr(ds, "drain_external_head_m")
    slope = _attr(ds, "surface_slope")
    init_wt = _attr(ds, "initial_water_table_m")

    clay_iface = _attr(ds, "clay_interface_z", 1.0)
    clay_drawdown = _attr(ds, "clay_drawdown_m")
    loam_drawdown = _attr(ds, "loam_case_drawdown_m", 1.0)
    clay_params = _attr(ds, "clay_params", "")
    loam_params = _attr(ds, "loam_params", "")
    wt_far = _attr(ds, "water_table_farfield_x2_m")
    wt_cone = _attr(ds, "water_table_cone_min_m")

    cum_drain_tot = _attr(ds, "cum_drainage_m2", float(cum_drain[-1]))
    cum_rain_tot = _attr(ds, "cum_rain_m2")
    cum_outflow = _attr(ds, "cum_outflow_m2")
    mbe_final = _attr(ds, "mass_balance_error_final", float(mbe[-1]))
    mbe_max = _attr(ds, "mass_balance_error_max", float(np.max(np.abs(mbe))))

    perch_max_thk = _attr(ds, "perched_zone_max_thickness_m")
    perch_max_t = _attr(ds, "perched_zone_time_day")
    perch_broad_thk = _attr(ds, "perched_zone_broadest_mean_thickness_m")
    perch_broad_ncol = _attr(ds, "perched_zone_broadest_ncolumns")
    perch_tot_col = _attr(ds, "perched_zone_total_columns")
    perch_theta = _attr(ds, "perched_zone_theta_just_above_interface")

    note = _attr(ds, "note", "")
    wall_clock = _attr(ds, "wall_clock_s")
    steps = _attr(ds, "steps")
    status = _attr(ds, "status", "")
    snes_ls = _attr(ds, "snes_linesearch", "")

    # --- drain footprint x-extent (from coords / drain_location: x in [9.5, 10.5]) ---
    DRAIN_X0, DRAIN_X1 = 9.5, 10.5

    # --- Ks layer split (clay below interface, loam above) ---
    Ks_clay = float(np.nanmin(Ks_profile))   # tight clay
    Ks_loam = float(np.nanmax(Ks_profile))   # loam topsoil

    # ----------------------------------------------------------------- color scaling
    # psi heatmap: fixed across frames so the rise/notch evolve rather than rescale.
    psi_lo = float(np.nanmin(head))
    psi_hi = float(np.nanmax(head))
    psi_abs = max(abs(psi_lo), abs(psi_hi))
    psi_range = [-psi_abs, psi_abs]
    # theta heatmap: fixed across frames; clamp to data span so the saturated band
    # (theta -> theta_s) is a clear colour break.
    th_lo = float(np.nanmin(theta))
    th_hi = float(np.nanmax(theta))

    # ----------------------------------------------------------------- subplot layout
    # row1: psi(x,z) cross-section heatmap          (colspan 2)
    # row2: theta(x,z) cross-section heatmap        (colspan 2)  <- perched sat. band
    # row3: water-table profile (col 1)  |  drain discharge + cum (col 2, twin-y)
    # row4: rainfall hyetograph (col 1)  |  |mass-balance error| log-y (col 2)
    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy", "secondary_y": True}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.32, 0.30, 0.20, 0.18],
        vertical_spacing=0.075,
        horizontal_spacing=0.11,
        subplot_titles=(
            f"Cross-section: pressure head psi(x, z)  [{head_units}]  "
            f"&mdash; water table = psi 0 contour; clay interface z={_fmt(clay_iface)} m; "
            f"drain at base x in [9.5, 10.5] m",
            f"Cross-section: water content theta(x, z)  [{theta_units}]  "
            f"&mdash; PERCHED saturated band (theta &rarr; theta_s) builds on the clay",
            "Water-table elevation profile  water_table(x)  (it RISES, no cone)",
            "Drain discharge &amp; cumulative drainage  (clay starves the drain)",
            "Rainfall hyetograph  (0.085 m/day &raquo; clay Ks 0.005)",
            "Diagnostics: |mass-balance error| vs time",
        ),
    )

    # ============================================================ ROW 1: psi cross-section
    i_heat = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            x=x, y=z, z=head[0],
            zmin=psi_range[0], zmax=psi_range[1],
            colorscale="RdBu",
            colorbar=dict(
                title=dict(text=f"psi [{head_units}]", side="right"),
                len=0.30, y=0.875, yanchor="middle", x=1.005, thickness=13,
            ),
            hovertemplate="x=%{x:.2f} m<br>z=%{y:.2f} m<br>psi=%{z:.3f} m<extra></extra>",
            name="psi",
            zsmooth="best",
        ),
        row=1, col=1,
    )
    # water-table contour (psi = 0). connectgaps=False so a NaN column breaks.
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

    # ============================================================ ROW 2: theta cross-section
    i_theta = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            x=x, y=z, z=theta[0],
            zmin=th_lo, zmax=th_hi,
            colorscale="YlGnBu",
            colorbar=dict(
                title=dict(text=f"theta [{theta_units}]", side="right"),
                len=0.28, y=0.55, yanchor="middle", x=1.005, thickness=13,
            ),
            hovertemplate="x=%{x:.2f} m<br>z=%{y:.2f} m<br>theta=%{z:.3f}<extra></extra>",
            name="theta",
            zsmooth="best",
        ),
        row=2, col=1,
    )
    # water-table contour echoed on the theta panel too (reads the saturation top)
    i_wt_theta = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=x, y=wt[0], mode="lines",
            line=dict(color="#c62828", width=2.5, dash="dot"),
            connectgaps=False,
            name="water table (on theta)",
            hovertemplate="x=%{x:.2f} m<br>water table z=%{y:.3f} m<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- shared cross-section annotations (drain footprint + clay interface) on rows 1 & 2 ---
    for r in (1, 2):
        # drain footprint marker: a heavy band at the base over x in [9.5,10.5].
        fig.add_shape(
            type="rect",
            x0=DRAIN_X0, x1=DRAIN_X1, y0=-0.04, y1=0.06,
            line=dict(color="#00c853", width=2),
            fillcolor="rgba(0,200,83,0.75)",
            row=r, col=1, layer="above",
        )
        # clay interface line at z = clay_iface across the whole transect
        fig.add_shape(
            type="line",
            x0=float(x.min()), x1=float(x.max()),
            y0=float(clay_iface), y1=float(clay_iface),
            line=dict(color="#6d4c41", width=2.5, dash="dash"),
            row=r, col=1, layer="above",
        )
    # interface labels (loam above / tight clay below) -- once, on the psi panel
    fig.add_annotation(
        x=1.4, y=float(clay_iface) + 0.30, xref="x", yref="y",
        text=f"<b>LOAM topsoil</b> (Ks {_fmt(Ks_loam)} m/d)", showarrow=False,
        font=dict(size=10.5, color="#3e2723"), row=1, col=1,
        bgcolor="rgba(255,255,255,0.55)",
    )
    fig.add_annotation(
        x=1.4, y=float(clay_iface) - 0.32, xref="x", yref="y",
        text=f"<b>TIGHT CLAY subsoil</b> (Ks {_fmt(Ks_clay)} m/d)", showarrow=False,
        font=dict(size=10.5, color="#3e2723"), row=1, col=1,
        bgcolor="rgba(255,255,255,0.55)",
    )
    fig.add_annotation(
        x=float(x.max()) - 0.3, y=float(clay_iface) + 0.04, xref="x", yref="y",
        text=f"clay interface z={_fmt(clay_iface)} m", showarrow=False,
        font=dict(size=9, color="#6d4c41"), xanchor="right", row=1, col=1,
    )
    # drain labels on both panels
    for r in (1, 2):
        fig.add_annotation(
            x=0.5 * (DRAIN_X0 + DRAIN_X1), y=0.20, ax=0, ay=0,
            xref="x", yref="y",
            text="<b>tile drain</b>", showarrow=False,
            font=dict(size=10, color="#00501f"),
            row=r, col=1,
        )
    # perched-band callout on theta panel
    fig.add_annotation(
        x=15.0, y=float(clay_iface) + 0.25, xref="x", yref="y",
        text="<b>perched saturated band</b><br>(loam waterlogs on the clay)",
        showarrow=True, arrowhead=2, arrowcolor="#004d40", arrowwidth=1.5,
        ax=0, ay=-26,
        font=dict(size=9.5, color="#004d40"), align="center",
        bgcolor="rgba(255,255,255,0.6)",
        row=2, col=1,
    )

    # ============================================================ ROW 3 col1: WT profile
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
        row=3, col=1,
    )
    if init_wt is not None:
        fig.add_hline(
            y=float(init_wt), line=dict(color="#9e9e9e", width=1.2, dash="dash"),
            row=3, col=1, layer="below",
            annotation_text=f"initial table {float(init_wt):.2f} m",
            annotation_position="bottom left",
            annotation_font=dict(size=9, color="#757575"),
        )
    # clay interface reference on the WT profile, too
    fig.add_hline(
        y=float(clay_iface), line=dict(color="#6d4c41", width=1.2, dash="dot"),
        row=3, col=1, layer="below",
        annotation_text=f"clay interface {float(clay_iface):.2f} m",
        annotation_position="top left",
        annotation_font=dict(size=9, color="#6d4c41"),
    )
    fig.add_vrect(
        x0=DRAIN_X0, x1=DRAIN_X1, fillcolor="rgba(0,200,83,0.18)",
        line_width=0, row=3, col=1, layer="below",
    )

    # ============================================================ ROW 3 col2: discharge + cum (twin-y)
    fig.add_trace(
        go.Scatter(
            x=time, y=q_drain, mode="lines",
            line=dict(color="#00897b", width=2),
            name="drain discharge",
            hovertemplate="t=%{x:.3f} day<br>q=%{y:.5f} " + q_units + "<extra></extra>",
        ),
        row=3, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=cum_drain, mode="lines",
            line=dict(color="#5e35b1", width=2, dash="dot"),
            name="cumulative drainage",
            hovertemplate="t=%{x:.3f} day<br>cum=%{y:.5f} " + cum_units + "<extra></extra>",
        ),
        row=3, col=2, secondary_y=True,
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
        row=3, col=2, secondary_y=False,
    )

    # ============================================================ ROW 4 col1: rainfall hyetograph
    dt = float(np.median(np.diff(time))) if ntime > 1 else 0.1
    fig.add_trace(
        go.Bar(
            x=time, y=rain, width=0.9 * dt,
            marker=dict(color="#1e88e5"),
            name="rainfall",
            hovertemplate="t=%{x:.3f} day<br>rain=%{y:.3f} " + rain_units + "<extra></extra>",
            showlegend=False,
        ),
        row=4, col=1,
    )
    # clay-Ks reference line: rain >> clay Ks is the whole story
    fig.add_hline(
        y=Ks_clay, line=dict(color="#b71c1c", width=1.5, dash="dash"),
        row=4, col=1, layer="above",
        annotation_text=f"clay Ks {_fmt(Ks_clay)}",
        annotation_position="bottom right",
        annotation_font=dict(size=9, color="#b71c1c"),
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
        row=4, col=1,
    )

    # ============================================================ ROW 4 col2: mass-balance error
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
        row=4, col=2,
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
        row=4, col=2,
    )

    # ----------------------------------------------------------------- animation frames
    frames = []
    for k in range(ntime):
        cur_err = mbe_plot[k] if np.isfinite(mbe_plot[k]) else np.nanmin(mbe_plot)
        frames.append(
            go.Frame(
                name=str(k),
                data=[
                    go.Heatmap(z=head[k]),                 # psi cross-section
                    go.Scatter(x=x, y=wt[k]),              # psi-panel WT contour
                    go.Heatmap(z=theta[k]),                # theta cross-section
                    go.Scatter(x=x, y=wt[k]),              # theta-panel WT contour
                    go.Scatter(x=x, y=wt[k]),              # WT profile panel
                    go.Scatter(x=[time[k]], y=[q_drain[k]]),
                    go.Scatter(x=[time[k]], y=[rain[k]]),
                    go.Scatter(x=[time[k]], y=[cur_err]),
                ],
                traces=[i_heat, i_wt_cross, i_theta, i_wt_theta, i_wt_prof,
                        i_mark_q, i_mark_rain, i_mark_mbe],
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
        x=0.05, y=-0.035, len=0.9,
        pad=dict(t=28, b=8),
        currentvalue=dict(prefix="time = ", suffix=f"  {t_units}",
                          font=dict(size=14, color="#222")),
        steps=slider_steps,
    )]
    updatemenus = [dict(
        type="buttons", direction="left",
        x=0.05, y=1.06, xanchor="left", yanchor="top",
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
    # row1 psi cross-section
    fig.update_xaxes(
        title_text=f"x  [{x_units}]  (downslope; surface outlet at 20 m)",
        range=[float(x.min()), float(x.max())], row=1, col=1, constrain="domain",
    )
    fig.update_yaxes(
        title_text=f"z  [{z_units}]  (0 = drain depth / base; top = surface)",
        range=[float(z.min()), float(z.max())], row=1, col=1,
    )
    # row2 theta cross-section
    fig.update_xaxes(
        title_text=f"x  [{x_units}]",
        range=[float(x.min()), float(x.max())], row=2, col=1, constrain="domain",
    )
    fig.update_yaxes(
        title_text=f"z  [{z_units}]",
        range=[float(z.min()), float(z.max())], row=2, col=1,
    )
    # row3 col1 WT profile
    fig.update_xaxes(title_text=f"x  [{x_units}]", range=[float(x.min()), float(x.max())],
                     row=3, col=1)
    fig.update_yaxes(title_text=f"water-table z  [{z_units}]",
                     range=[float(z.min()), float(z.max())], row=3, col=1)
    # row3 col2 discharge twin
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=3, col=2)
    fig.update_yaxes(title_text=f"discharge  [{q_units}]", row=3, col=2,
                     secondary_y=False, color="#00695c")
    fig.update_yaxes(title_text=f"cum  [{cum_units}]", row=3, col=2,
                     secondary_y=True, color="#5e35b1", showgrid=False)
    # row4 col1 rainfall
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=4, col=1)
    fig.update_yaxes(title_text=f"rain  [{rain_units}]", row=4, col=1,
                     rangemode="tozero")
    # row4 col2 mbe (log)
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=4, col=2)
    fig.update_yaxes(title_text="|mass-balance error|  (-)",
                     type="log", exponentformat="power", row=4, col=2)

    # ----------------------------------------------------------------- metrics panel
    drawdown_str = (
        f"{_fmt(clay_drawdown)} m" if clay_drawdown is not None else "n/a"
    )
    metrics_lines = [
        f"<b>module</b>: {module}",
        "<b>scenario</b>: 2-D hillslope, LOAM topsoil over",
        "&nbsp;&nbsp;TIGHT-CLAY subsoil (interface z=1.0 m)",
        f"<b>domain</b>: {domain}",
        f"<b>mesh</b>: {mesh}",
        "<b>&mdash; layering / Ks &mdash;</b>",
        f"&nbsp;&nbsp;loam: {loam_params}",
        f"&nbsp;&nbsp;clay: {clay_params}",
        (f"&nbsp;&nbsp;Ks loam {_fmt(Ks_loam)} vs clay {_fmt(Ks_clay)} m/day "
         f"(<b>50x</b>)"),
        f"<b>drain</b>: {drain_loc}",
        (f"&nbsp;&nbsp;conductance C = {_fmt(drain_C)} /day &nbsp; "
         f"external head = {_fmt(drain_head)} m"),
        f"<b>storm</b>: 0.085 m/day for 1.5 d, then recession",
        f"&nbsp;&nbsp;initial table = {_fmt(init_wt)} m; slope {_fmt(slope)}",
        "<b>&mdash; the CONTRAST (read me) &mdash;</b>",
        ("<span style='color:#b71c1c'>rain 0.085 &raquo; clay Ks 0.005 &rArr; "
         "water<br>&nbsp;&nbsp;PERCHES on the clay; the loam saturates.</span>"),
        (f"&nbsp;&nbsp;clay drawdown at drain = <b>{drawdown_str}</b><br>"
         f"&nbsp;&nbsp;<b>vs pure-loam cone {_fmt(loam_drawdown)} m</b> "
         f"(~12x weaker)"),
        (f"&nbsp;&nbsp;(far-field table {_fmt(wt_far)} m &minus; "
         f"drain {_fmt(wt_cone)} m)"),
        (f"&nbsp;&nbsp;table RISES {_fmt(init_wt)}&rarr;~{_fmt(wt_far)} m "
         f"(waterlogs root zone)"),
        (f"&nbsp;&nbsp;perched band: theta&rarr;{_fmt(perch_theta)} just above "
         f"interface;<br>&nbsp;&nbsp;max thickness {_fmt(perch_max_thk)} m "
         f"(t={_fmt(perch_max_t)} d),<br>&nbsp;&nbsp;broadest over "
         f"{_fmt(perch_broad_ncol)}/{_fmt(perch_tot_col)} columns"),
        f"&nbsp;&nbsp;cumulative drainage = {_fmt(cum_drain_tot)} {cum_units}",
        f"&nbsp;&nbsp;cumulative rain = {_fmt(cum_rain_tot)} {cum_units}",
        (f"&nbsp;&nbsp;surface ponding = {_fmt(cum_outflow)} {cum_units} "
         f"(&asymp; 0)"),
        "<b>&mdash; diagnostics &mdash;</b>",
        (f"&nbsp;&nbsp;mass-balance error: final {_fmt(mbe_final)},<br>"
         f"&nbsp;&nbsp;max {_fmt(mbe_max)}  (machine ~ 1e-13)"),
        (f"&nbsp;&nbsp;status {status} in {_fmt(steps)} steps "
         f"(linesearch {snes_ls}), {_fmt(wall_clock)} s"),
        "<b>&mdash; framing (read me) &mdash;</b>",
        ("<span style='color:#b71c1c'>ILLUSTRATION on the VALIDATED Module-3 "
         "<i>resolved</i><br>&nbsp;&nbsp;drainage BC (drain modelled the expensive, "
         "fully-<br>&nbsp;&nbsp;resolved way) on the validated CoupledProblem<br>"
         "&nbsp;&nbsp;engine &mdash; NOT a new validated module, and NOT<br>"
         "&nbsp;&nbsp;the retracted Module-4 sub-grid embedded feature.</span>"),
    ]
    metrics_html = "<br>".join(metrics_lines)
    fig.add_annotation(
        x=1.004, y=1.0, xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        align="left", showarrow=False,
        text="<b>METRICS</b><br>" + metrics_html,
        font=dict(size=9.8, color="#222", family="Consolas, monospace"),
        bordercolor="#888", borderwidth=1, borderpad=7,
        bgcolor="rgba(245,245,245,0.97)",
    )

    fig.update_layout(
        title=dict(
            text="<b>PIDS Pillar-2 sanity (Tier-3) &middot; hillslope tile-drain, "
                 "tight-clay subsoil &middot; 2026-06-09</b><br>"
                 "<span style='font-size:12px'>rain 0.085 m/day &raquo; clay Ks 0.005 "
                 "&rArr; water PERCHES on the clay (loam waterlogs from the interface "
                 "up) and the buried drain barely moves the table "
                 "&mdash; clay drawdown 0.085 m vs the pure-loam ~1.0 m cone</span>",
            x=0.5, xanchor="center", y=0.992, yanchor="top",
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.012,
                    xanchor="right", x=1.0, font=dict(size=10)),
        margin=dict(l=75, r=470, t=130, b=70),
        width=1360, height=1320,
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
        print("usage: python make_hillslope_drain_clay_html.py <result.nc> <out.html>",
              file=sys.stderr)
        sys.exit(2)
    out = build_hillslope_drain_clay_html(sys.argv[1], sys.argv[2])
    import os
    print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
