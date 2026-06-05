"""Reusable Tier-3 sanity visualizer for the PIDS Pillar-2 forward model (Module 3: surface<->subsurface COUPLING).

Turns a standardized coupling sanity-run NetCDF result (the data contract of
``governance/visualize-sanity-check-routine.md``) into ONE self-contained,
offline, interactive HTML. The generator reads ONLY the result file -- it never
imports or runs the solver (this independence is the whole point of the Tier-3 lens).

For a storm + recession on a 1-D soil column (z up, surface at the top) the
standard coupled views are:
  1. SOIL PROFILE with a TIME SLIDER: psi(z) and theta(z) vs elevation z, animated
     so the wetting front is seen advancing from the surface downward during the
     storm (two side-by-side panels; y = elevation z [m], surface at the TOP).
     This is the key physical view.
  2. SURFACE + EXCHANGE time series: surface depth d(t) [m] and exchange flux
     lambda(t) [m/day] as lines, with rainfall(t) [m/day] as INVERTED bars on a
     twin top axis (hyetograph). Shows ponding + infiltration vs the storm + recession.
  3. PARTITIONING: cum_rain(t) vs cumulative infiltration (soil_water(t) - soil_water(0))
     vs surface_water(t) [all m]. infiltration + ponding should track cum_rain.
  4. DIAGNOSTICS: mass-balance error (log-y) + Newton iterations.
  5. METRICS panel (text from attrs): the deciding numbers.

Usage:
    python viz/make_coupling_html.py <result.nc> <out.html>

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
    if isinstance(val, float):
        if val != 0 and (abs(val) < 1e-3 or abs(val) >= 1e4):
            return f"{val:.3e}"
        return f"{val:.{nd}g}"
    return str(val)


def _units(da, fallback: str = "") -> str:
    return str(da.attrs.get("units", fallback))


# ----------------------------------------------------------------------------- builder
def build_html(nc_path: str, out_path: str) -> str:
    """Build the self-contained interactive coupling sanity HTML.

    Parameters
    ----------
    nc_path : str
        Path to the standardized coupling sanity-run NetCDF result file.
    out_path : str
        Path to write the self-contained HTML (plotly inlined, offline).

    Returns
    -------
    str
        The ``out_path`` written.
    """
    ds = xr.open_dataset(nc_path)

    # --- coords / fields (item-access so ``ds['head']`` is the var, not .head()) ---
    z = np.asarray(ds["z"].values, dtype=float)                          # (z,)  [m elevation; surface at max]
    time = np.asarray(ds["time"].values, dtype=float)                    # (time,) [day]
    head = np.asarray(ds["head"].values, dtype=float)                    # (time, z) [m psi]
    theta = np.asarray(ds["water_content"].values, dtype=float)          # (time, z) [-]
    d_surf = np.asarray(ds["surface_depth"].values, dtype=float)         # (time,) [m]
    lam = np.asarray(ds["exchange_flux"].values, dtype=float)            # (time,) [m/day]
    rain = np.asarray(ds["rainfall"].values, dtype=float)               # (time,) [m/day]
    soil_w = np.asarray(ds["soil_water"].values, dtype=float)           # (time,) [m]
    surf_w = np.asarray(ds["surface_water"].values, dtype=float)        # (time,) [m]
    cum_rain = np.asarray(ds["cum_rain"].values, dtype=float)           # (time,) [m]
    mbe = np.asarray(ds["mass_balance_error"].values, dtype=float)       # (time,) [-]
    niter = np.asarray(ds["newton_iters"].values, dtype=float)          # (time,) [-]

    z_units = _units(ds["z"], "m")
    t_units = _units(ds["time"], "day")
    psi_units = _units(ds["head"], "m")
    lam_units = _units(ds["exchange_flux"], "m/day")
    r_units = _units(ds["rainfall"], "m/day")

    # cumulative infiltration (change in subsurface storage relative to t=0).
    infil = soil_w - soil_w[0]

    ntime = time.shape[0]

    # --- global attrs / metrics ---
    module = _attr(ds, "module", "coupling")
    scenario = _attr(ds, "scenario", "storm + recession")
    date = _attr(ds, "date", "")
    psi0 = _attr(ds, "antecedent_psi0_m")
    rain_rate = _attr(ds, "rain_rate_m_per_day")
    storm_dur = _attr(ds, "storm_duration_day")
    column_depth = _attr(ds, "column_depth_m")
    Ks = _attr(ds, "Ks_m_per_day")
    cum_rain_total = _attr(ds, "cum_rain_total_m")
    peak_d = _attr(ds, "peak_surface_depth_m")
    peak_lam = _attr(ds, "peak_exchange_flux_m_per_day")
    final_infil = _attr(ds, "final_infiltrated_m")
    final_pond = _attr(ds, "final_ponded_m")
    mbe_max = _attr(ds, "mass_balance_error_max", float(np.max(np.abs(mbe))))
    max_iters = _attr(ds, "max_newton_iters", int(np.max(niter)))

    # --- fixed axis ranges so the front "advances" rather than rescaling per frame ---
    psi_lo = float(np.nanmin(head)); psi_hi = float(np.nanmax(head))
    psi_pad = 0.04 * (psi_hi - psi_lo + 1e-12)
    psi_range = [psi_lo - psi_pad, psi_hi + psi_pad]

    th_lo = float(np.nanmin(theta)); th_hi = float(np.nanmax(theta))
    th_pad = 0.04 * (th_hi - th_lo + 1e-12)
    th_range = [th_lo - th_pad, th_hi + th_pad]

    z_range = [float(z.min()), float(z.max())]

    # =============================================================== figure layout
    #   row1 left : psi(z) profile  [animated]   row1 right : theta(z) profile [animated]
    #   row2 (colspan 2): surface depth + exchange (left axis) / rain inverted (right axis)
    #   row3 (colspan 2): partitioning -- cum_rain vs cumulative infiltration vs ponding
    #   row4 left : mass-balance error (log-y)   row4 right : Newton iterations
    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2, "secondary_y": True}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.34, 0.24, 0.22, 0.20],
        vertical_spacing=0.085,
        horizontal_spacing=0.13,
        subplot_titles=(
            "Soil pressure head  &psi;(z)   [m]   (surface at top)",
            "Water content  &theta;(z)   [-]   (surface at top)",
            "Surface ponding d &amp; infiltration flux &lambda;  vs storm (inverted bars)",
            "Partitioning: cumulative rain = infiltration + ponding  [m]",
            "Diagnostics: mass-balance error (log)",
            "Diagnostics: Newton iterations",
        ),
    )

    PSI_COLOR = "#1565c0"
    TH_COLOR = "#2e7d32"
    D_COLOR = "#0277bd"
    LAM_COLOR = "#ef6c00"
    RAIN_COLOR = "#90a4ae"
    RAIN_BORDER = "#546e7a"
    CUM_COLOR = "#37474f"
    INFIL_COLOR = "#2e7d32"
    POND_COLOR = "#0277bd"
    MBE_COLOR = "#e67e22"
    ITER_COLOR = "#6a1b9a"
    SURF_MARK = "#c62828"

    # =============================================================== row1: psi(z) & theta(z) profiles
    # static "initial" ghost (t=0) so the moving front is read against the antecedent state.
    fig.add_trace(
        go.Scatter(
            x=head[0], y=z, mode="lines",
            line=dict(color="#b0bec5", width=1.4, dash="dot"),
            name="initial ψ (t=0)", legendgroup="init", showlegend=True,
            hovertemplate="ψ₀=%{x:.3f} m<br>z=%{y:.3f} m<extra></extra>",
        ),
        row=1, col=1,
    )
    i_psi = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=head[0], y=z, mode="lines",
            line=dict(color=PSI_COLOR, width=2.6),
            name="ψ(z)  [m]",
            hovertemplate="ψ=%{x:.3f} m<br>z=%{y:.3f} m<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=theta[0], y=z, mode="lines",
            line=dict(color="#a5d6a7", width=1.4, dash="dot"),
            name="initial θ (t=0)", legendgroup="init", showlegend=True,
            hovertemplate="θ₀=%{x:.4f}<br>z=%{y:.3f} m<extra></extra>",
        ),
        row=1, col=2,
    )
    i_theta = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=theta[0], y=z, mode="lines",
            line=dict(color=TH_COLOR, width=2.6),
            name="θ(z)  [-]",
            hovertemplate="θ=%{x:.4f}<br>z=%{y:.3f} m<extra></extra>",
        ),
        row=1, col=2,
    )

    # =============================================================== row2: surface depth + exchange / rain
    fig.add_trace(
        go.Scatter(
            x=time, y=d_surf, mode="lines",
            line=dict(color=D_COLOR, width=2.6),
            name=f"surface depth d  [{psi_units}]",
            hovertemplate="t=%{x:.4f} day<br>d=%{y:.5f} m<extra></extra>",
        ),
        row=2, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=lam, mode="lines",
            line=dict(color=LAM_COLOR, width=2.2, dash="dash"),
            name=f"exchange flux λ  [{lam_units}]",
            hovertemplate="t=%{x:.4f} day<br>λ=%{y:.4f} " + lam_units + "<extra></extra>",
        ),
        row=2, col=1, secondary_y=True,
    )
    # hyetograph: inverted bars hanging from the top of the rain (secondary) axis.
    r_hi = float(np.nanmax(rain))
    r_top = (r_hi if r_hi > 0 else 1.0) * 1.05
    r_axis_top = r_top * 3.0  # rain occupies only the top third (clear of the d / lambda lines)
    fig.add_trace(
        go.Bar(
            x=time, y=rain, base=r_axis_top - rain,
            marker=dict(color=RAIN_COLOR, line=dict(color=RAIN_BORDER, width=0.5)),
            opacity=0.8,
            width=(time[1] - time[0]) * 0.9 if ntime > 1 else 0.005,
            name=f"rainfall  [{r_units}]",
            hovertemplate="t=%{x:.4f} day<br>r=%{base:.3f} " + r_units + "<extra></extra>",
        ),
        row=2, col=1, secondary_y=True,
    )
    # current-time marker on the surface-depth line (tracks slider).
    i_dmark = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[d_surf[0]], mode="markers",
            marker=dict(size=12, color=SURF_MARK, symbol="circle-open",
                        line=dict(width=3, color=SURF_MARK)),
            name="current time",
            hovertemplate="t=%{x:.4f} day<br>d=%{y:.5f} m<extra>current</extra>",
        ),
        row=2, col=1, secondary_y=False,
    )

    # =============================================================== row3: partitioning
    fig.add_trace(
        go.Scatter(
            x=time, y=cum_rain, mode="lines",
            line=dict(color=CUM_COLOR, width=3.0),
            name="cumulative rain  [m]",
            hovertemplate="t=%{x:.4f} day<br>cum rain=%{y:.5f} m<extra></extra>",
        ),
        row=3, col=1,
    )
    # stacked: infiltration (bottom) + ponding (on top) should sum to cum_rain.
    fig.add_trace(
        go.Scatter(
            x=time, y=infil, mode="lines",
            line=dict(color=INFIL_COLOR, width=0.5),
            fill="tozeroy", fillcolor="rgba(46,125,50,0.30)",
            name="cumulative infiltration  [m]",
            hovertemplate="t=%{x:.4f} day<br>infiltrated=%{y:.5f} m<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=infil + surf_w, mode="lines",
            line=dict(color=POND_COLOR, width=0.5),
            fill="tonexty", fillcolor="rgba(2,119,189,0.30)",
            name="+ ponding (surface water)  [m]",
            customdata=surf_w,
            hovertemplate="t=%{x:.4f} day<br>ponded=%{customdata:.5f} m<br>infil+pond=%{y:.5f} m<extra></extra>",
        ),
        row=3, col=1,
    )

    # =============================================================== row4: diagnostics
    mbe_plot = np.where(np.abs(mbe) <= 0, np.nan, np.abs(mbe))  # guard zeros for log axis
    fig.add_trace(
        go.Scatter(
            x=time, y=mbe_plot, mode="lines+markers",
            line=dict(color=MBE_COLOR, width=2),
            marker=dict(size=4, color=MBE_COLOR),
            name="|mass-balance error|",
            hovertemplate="t=%{x:.4f} day<br>err=%{y:.3e}<extra></extra>",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=niter, mode="lines+markers",
            line=dict(color=ITER_COLOR, width=1.8),
            marker=dict(size=4, color=ITER_COLOR),
            name="Newton iters",
            hovertemplate="t=%{x:.4f} day<br>iters=%{y:.0f}<extra></extra>",
        ),
        row=4, col=2,
    )

    # =============================================================== animation frames
    frames = []
    for k in range(ntime):
        frames.append(
            go.Frame(
                name=str(k),
                data=[
                    go.Scatter(x=head[k], y=z),        # i_psi
                    go.Scatter(x=theta[k], y=z),       # i_theta
                    go.Scatter(x=[time[k]], y=[d_surf[k]]),  # i_dmark
                ],
                traces=[i_psi, i_theta, i_dmark],
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
        x=0.05, y=-0.025, len=0.9,
        pad=dict(t=30, b=10),
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
                 args=[None, dict(mode="immediate",
                                  fromcurrent=True,
                                  frame=dict(duration=120, redraw=True),
                                  transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0))]),
        ],
    )]

    # =============================================================== axes
    # row1: psi(z) & theta(z) profiles (elevation y; surface at top means y increases upward;
    #       z already runs base=0 -> surface=max, so default increasing-y puts the surface on top).
    fig.update_xaxes(title_text=f"pressure head ψ  [{psi_units}]", range=psi_range, row=1, col=1)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]  (surface at top)", range=z_range,
                     row=1, col=1)
    fig.update_xaxes(title_text="water content θ  [-]", range=th_range, row=1, col=2)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]", range=z_range, row=1, col=2)

    # row2: surface depth (left) + exchange flux & rain (right)
    fig.update_xaxes(title_text=f"time  [{t_units}]",
                     range=[float(time.min()), float(time.max())], row=2, col=1)
    fig.update_yaxes(title_text=f"surface depth d  [{psi_units}]", row=2, col=1,
                     secondary_y=False, rangemode="tozero")
    # rain axis reversed-from-top so inverted bars hang down; lambda also lives on this axis but
    # stays in the lower two-thirds (rain only fills the top third).
    fig.update_yaxes(title_text=f"flux  [{lam_units}]  (rain inverted)",
                     range=[r_axis_top, 0.0], row=2, col=1, secondary_y=True, showgrid=False)

    # row3: partitioning
    fig.update_xaxes(title_text=f"time  [{t_units}]",
                     range=[float(time.min()), float(time.max())], row=3, col=1)
    fig.update_yaxes(title_text="depth-equivalent water  [m]", row=3, col=1, rangemode="tozero")

    # row4: diagnostics
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=4, col=1)
    fig.update_yaxes(title_text="|mass-balance err|", type="log",
                     exponentformat="power", row=4, col=1)
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=4, col=2)
    fig.update_yaxes(title_text="Newton iters", row=4, col=2, rangemode="tozero")

    # =============================================================== metrics panel
    metrics_lines = [
        f"<b>module</b>: {module}",
        f"<b>scenario</b>: {scenario}",
        f"<b>date</b>: {date}",
        (f"<b>antecedent ψ₀</b> = {_fmt(psi0)} m &nbsp; "
         f"<b>column</b> = {_fmt(column_depth)} m &nbsp; <b>Ks</b> = {_fmt(Ks)} m/{t_units}"),
        (f"<b>rain rate</b> = {_fmt(rain_rate)} m/{t_units} &nbsp; "
         f"<b>storm dur</b> = {_fmt(storm_dur)} {t_units}"),
        f"<b>cum rain total</b> = {_fmt(cum_rain_total)} m",
        f"<b>final infiltrated</b> = {_fmt(final_infil)} m",
        f"<b>final ponded</b> = {_fmt(final_pond)} m",
        f"<b>peak surface depth</b> = {_fmt(peak_d)} m",
        f"<b>peak exchange flux</b> = {_fmt(peak_lam)} m/{t_units}",
        f"<b>MAX mass-balance error</b> = {_fmt(float(mbe_max))}  (machine ~ 1e-14)",
        f"<b>max Newton iters</b> = {_fmt(max_iters)}",
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
        title=dict(text="<b>PIDS Pillar-2 sanity (Tier-3): surface&harr;subsurface coupling</b><br>"
                        f"<span style='font-size:13px'>{title}</span>",
                   x=0.5, xanchor="center", y=0.99, yanchor="top"),
        sliders=sliders,
        updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.012,
                    xanchor="right", x=1.0, font=dict(size=10)),
        margin=dict(l=80, r=340, t=150, b=90),
        width=1240, height=1320,
        barmode="overlay",
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
    ds.close()
    return out_path


# backward-compatible alias matching the overland generator's naming.
build_coupling_html = build_html


def _main_all(date: str = "2026-06-05"):
    """Loop all 6 coupling keys -> one self-contained HTML each."""
    import os

    keys = ["normal_on_dry", "normal_on_normal", "normal_on_wet",
            "extreme_on_dry", "extreme_on_normal", "extreme_on_wet"]
    data_dir = "../validation/sanity/data"
    viz_dir = "../validation/sanity/viz"
    os.makedirs(viz_dir, exist_ok=True)
    written = []
    for key in keys:
        nc = f"{data_dir}/coupling__{key}__{date}.nc"
        out = f"{viz_dir}/coupling__{key}__{date}.html"
        build_html(nc, out)
        sz = os.path.getsize(out) / 1e6
        written.append((out, sz))
        print(f"WROTE {out}  ({sz:.2f} MB)", flush=True)
    return written


if __name__ == "__main__":
    if len(sys.argv) == 3:
        import os
        out = build_html(sys.argv[1], sys.argv[2])
        print(f"WROTE {out}  ({os.path.getsize(out) / 1e6:.2f} MB)")
    elif len(sys.argv) == 1:
        _main_all()
    else:
        print("usage: python make_coupling_html.py [<result.nc> <out.html>]  "
              "(no args -> build all 6)", file=sys.stderr)
        sys.exit(2)
