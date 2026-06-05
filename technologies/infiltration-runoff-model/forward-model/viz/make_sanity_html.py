"""Reusable Tier-3 sanity visualizer for the PIDS Pillar-2 forward model.

Turns a standardized sanity-run NetCDF result (the data contract of
``governance/visualize-sanity-check-routine.md``) into ONE self-contained,
offline, interactive HTML. The generator reads ONLY the result file -- it never
imports or runs the solver.

For a 1-D Richards column the standard view (per the viz catalog) is a PROFILE
plot of water content theta(z) and pressure head psi(z) vs elevation z, animated
with a TIME SLIDER so the infiltration front is visible advancing upward to
saturation/ponding at the top. A diagnostics panel shows mass-balance error vs
time, and a metrics panel surfaces the deciding numbers.

Usage:
    python viz/make_sanity_html.py <result.nc> <out.html>

Dependencies: xarray + plotly only. Plotly is vendored INLINE
(``include_plotlyjs=True``) so the HTML opens offline by double-click.
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr
import plotly.graph_objects as go


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
def build_profile_html(nc_path: str, html_path: str) -> str:
    """Build the self-contained interactive profile+diagnostics HTML.

    Parameters
    ----------
    nc_path : str
        Path to the standardized sanity-run NetCDF result file.
    html_path : str
        Path to write the self-contained HTML (plotly inlined, offline).

    Returns
    -------
    str
        The ``html_path`` written.
    """
    ds = xr.open_dataset(nc_path)

    # --- pull coords / fields (use item-access so ``ds['head']`` is the var,
    #     not the pandas-style ``.head`` method). ---
    z = np.asarray(ds["z"].values, dtype=float)
    time = np.asarray(ds["time"].values, dtype=float)
    head = np.asarray(ds["head"].values, dtype=float)          # (time, z)  pressure head psi
    theta = np.asarray(ds["water_content"].values, dtype=float)  # (time, z) water content
    mbe = np.asarray(ds["mass_balance_error"].values, dtype=float)  # (time,)

    z_units = _units(ds["z"], "m")
    t_units = _units(ds["time"], "day")
    head_units = _units(ds["head"], "m")
    theta_units = _units(ds["water_content"], "m3/m3")

    # --- global attrs / metrics ---
    module = _attr(ds, "module", "subsurface")
    scenario = _attr(ds, "scenario", "sanity run")
    date = _attr(ds, "date", "")
    soil = _attr(ds, "soil", "")
    theta_r = _attr(ds, "theta_r")
    theta_s = _attr(ds, "theta_s")
    Ks = _attr(ds, "Ks_m_per_day")
    rain = _attr(ds, "rain_flux_m_per_day")
    cum_in = _attr(ds, "cumulative_input_m")
    mbe_max = _attr(ds, "mass_balance_error_max", float(np.max(np.abs(mbe))))

    ntime = time.shape[0]

    # axis ranges (fixed across frames so the front "moves" rather than rescaling)
    theta_lo = float(np.nanmin(theta))
    theta_hi = float(np.nanmax(theta))
    if theta_r is not None:
        theta_lo = min(theta_lo, float(theta_r))
    if theta_s is not None:
        theta_hi = max(theta_hi, float(theta_s))
    theta_pad = 0.04 * (theta_hi - theta_lo + 1e-9)
    theta_range = [theta_lo - theta_pad, theta_hi + theta_pad]

    head_lo = float(np.nanmin(head))
    head_hi = float(np.nanmax(head))
    head_pad = 0.05 * (head_hi - head_lo + 1e-9)
    head_range = [head_lo - head_pad, head_hi + head_pad]

    # --- subplot layout: row1 = theta(z) | psi(z) profiles (animated),
    #     row2 = mass-balance error vs time (static diagnostic) ---
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
        ],
        row_heights=[0.62, 0.38],
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
        subplot_titles=(
            f"Water content theta(z)  [{theta_units}]",
            f"Pressure head psi(z)  [{head_units}]",
            f"Diagnostics: relative mass-balance error vs time",
        ),
    )

    # ---- static reference lines on the theta panel (theta_r, theta_s) ----
    if theta_r is not None:
        fig.add_vline(
            x=float(theta_r), line=dict(color="#c0392b", width=1, dash="dot"),
            row=1, col=1,
            annotation_text=f"theta_r={_fmt(theta_r)}",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#c0392b"),
        )
    if theta_s is not None:
        fig.add_vline(
            x=float(theta_s), line=dict(color="#1f6f3d", width=1, dash="dot"),
            row=1, col=1,
            annotation_text=f"theta_s={_fmt(theta_s)} (saturation)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#1f6f3d"),
        )

    # ---- initial (t=0) profile traces ----
    THETA_COLOR = "#1565c0"
    HEAD_COLOR = "#6a1b9a"

    fig.add_trace(
        go.Scatter(
            x=theta[0], y=z, mode="lines+markers",
            line=dict(color=THETA_COLOR, width=2.5),
            marker=dict(size=4, color=THETA_COLOR),
            name="theta(z)",
            hovertemplate="theta=%{x:.4f} m3/m3<br>z=%{y:.3f} m<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=head[0], y=z, mode="lines+markers",
            line=dict(color=HEAD_COLOR, width=2.5),
            marker=dict(size=4, color=HEAD_COLOR),
            name="psi(z)",
            hovertemplate="psi=%{x:.4f} m<br>z=%{y:.3f} m<extra></extra>",
        ),
        row=1, col=2,
    )

    # ---- diagnostics: mass-balance error vs time (static, full series) ----
    # guard against zeros for log axis
    mbe_plot = np.where(np.abs(mbe) <= 0, np.nan, np.abs(mbe))
    fig.add_trace(
        go.Scatter(
            x=time, y=mbe_plot, mode="lines+markers",
            line=dict(color="#e67e22", width=2),
            marker=dict(size=5, color="#e67e22"),
            name="|mass-balance error|",
            hovertemplate="t=%{x:.4f} day<br>err=%{y:.3e}<extra></extra>",
        ),
        row=2, col=1,
    )
    # moving time-marker on the diagnostics panel (animates with the slider)
    t0_err = mbe_plot[0] if np.isfinite(mbe_plot[0]) else np.nanmin(mbe_plot)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[t0_err], mode="markers",
            marker=dict(size=12, color="#c0392b", symbol="circle-open",
                        line=dict(width=3, color="#c0392b")),
            name="current time",
            hovertemplate="t=%{x:.4f} day<extra>current</extra>",
        ),
        row=2, col=1,
    )

    # indices of the animated traces (theta profile=0, psi profile=1, time-marker=3)
    # trace order: 0 theta, 1 psi, 2 mbe series, 3 current-time marker
    # ---- animation frames ----
    frames = []
    for k in range(ntime):
        cur_err = mbe_plot[k] if np.isfinite(mbe_plot[k]) else np.nanmin(mbe_plot)
        frames.append(
            go.Frame(
                name=str(k),
                data=[
                    go.Scatter(x=theta[k], y=z),     # -> trace 0
                    go.Scatter(x=head[k], y=z),      # -> trace 1
                    go.Scatter(x=[time[k]], y=[cur_err]),  # -> trace 3
                ],
                traces=[0, 1, 3],
            )
        )
    fig.frames = frames

    # ---- slider + play/pause controls ----
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
        currentvalue=dict(prefix=f"time = ", suffix=f"  {t_units}",
                          font=dict(size=14, color="#222")),
        steps=slider_steps,
    )]
    updatemenus = [dict(
        type="buttons", direction="left",
        x=0.05, y=1.18, xanchor="left", yanchor="top",
        pad=dict(t=0, r=10),
        showactive=False,
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(mode="immediate",
                                  fromcurrent=True,
                                  frame=dict(duration=180, redraw=True),
                                  transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0))]),
        ],
    )]

    # ---- axes ----
    fig.update_xaxes(title_text=f"theta  [{theta_units}]", range=theta_range,
                     row=1, col=1, zeroline=False)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]  (0 = bottom, top = surface)",
                     range=[float(z.min()), float(z.max())], row=1, col=1)
    fig.update_xaxes(title_text=f"psi  [{head_units}]", range=head_range,
                     row=1, col=2, zeroline=True, zerolinecolor="#bbb")
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]",
                     range=[float(z.min()), float(z.max())], row=1, col=2)
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=2, col=1)
    # Power notation (10^-14) instead of Plotly's default SI prefixes (which render
    # tiny values as e.g. "10f" femto -- confusing for an error magnitude).
    fig.update_yaxes(title_text="|relative mass-balance error|  (dimensionless)",
                     type="log", exponentformat="power", row=2, col=1)

    # ---- metrics panel (paper-space annotation block) ----
    metrics_lines = [
        f"<b>module</b>: {module}",
        f"<b>scenario</b>: {scenario}",
        f"<b>date</b>: {date}",
        f"<b>soil</b>: {soil}",
        (f"&nbsp;&nbsp;theta_r = {_fmt(theta_r)} &nbsp; theta_s = {_fmt(theta_s)} "
         f"&nbsp; Ks = {_fmt(Ks)} {head_units}/{t_units}"),
        f"<b>net top flux</b>: {_fmt(rain)} m/{t_units}",
        f"<b>cumulative input</b>: {_fmt(cum_in)} m  over {_fmt(float(time[-1]))} {t_units}",
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
        title=dict(text=f"<b>PIDS Pillar-2 sanity (Tier-3)</b><br>"
                        f"<span style='font-size:13px'>{title}</span>",
                   x=0.5, xanchor="center", y=0.975, yanchor="top"),
        sliders=sliders,
        updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0),
        margin=dict(l=70, r=320, t=130, b=80),
        width=1180, height=860,
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
        print("usage: python make_sanity_html.py <result.nc> <out.html>", file=sys.stderr)
        sys.exit(2)
    out = build_profile_html(sys.argv[1], sys.argv[2])
    import os
    print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
