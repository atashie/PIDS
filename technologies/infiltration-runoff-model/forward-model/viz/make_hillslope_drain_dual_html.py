"""Tier-3 sanity visualizer: hillslope DUAL drain -- surface inlet vs interface tile.

Companion to ``make_hillslope_drain_clay_html.py`` (single buried base drain in the
clay). This generator turns the dual-drain hillslope sanity NetCDF into ONE
self-contained, offline, interactive HTML. It reads ONLY the result file -- it
never imports or runs the solver (independent visual evidence).

THE PHYSICS to make visible
---------------------------
Same 20 m x 2 m loam-over-tight-clay hillslope as the clay-subsoil figure (loam
Ks 0.25 m/day above z = 1.0 m, tight clay Ks 0.005 below; 2% slope, outlet at
x = 20; initial water table z = 0.7 m). But the buried base drain is REPLACED by
TWO drains sharing one x-footprint [9.5, 10.5] m, so PLACEMENT is the only
difference:

1. INTERFACE TILE DRAIN sitting ON the clay (z in [1.0, 1.1] -- the agronomically
   standard placement): an outflow-only smooth-DRN sink (pipe at atmospheric;
   never injects).
2. SURFACE "TILE" (grate inlet) on the land surface at the same x: a linear
   intake on ponded depth d.

Storm: 0.12 m/day for 2.0 d, then recession to 5.0 d. Honest note: the storm was
deliberately RAISED vs the base-drain clay run (0.085 x 1.5 d) so saturation-
excess ponding actually occurs -- at 0.085 the loam absorbs everything and a
surface drain captures exactly 0.

THE PUNCHLINE: the interface tile on the clay does the heavy lifting (cum 0.76
m^2, peak 0.54 m^2/day, locally ELIMINATES the 0.90 m perched band at its
column); the surface grate inlet only engages during the saturation-excess
ponding window (cum 0.16 m^2) intercepting a sub-mm sheet-flow film (max ponded
depth 0.78 mm). Total dual capture 0.92 m^2 vs the buried-base-drain run's 0.054
m^2 (different forcing -- caveat stated in panel).

VIEWS
-----
- Headline psi(x, z) cross-section HEATMAP with TIME SLIDER + play; water-table
  (psi = 0) contour; BOTH drain footprints marked (green tile band on the clay,
  orange grate band at the surface); clay-interface dashed line with LOAM /
  TIGHT CLAY labels.
- theta(x, z) cross-section HEATMAP (same slider) -- the perched saturated band
  builds on the clay everywhere EXCEPT the tile column (the notch).
- Animated water-table(x) profile + a light filled PERCH band (z = 1.0 to
  1.0 + perched_thickness(x)) + the shared drain footprint shaded.
- BOTH drain hydrographs (left axis) + cumulatives (right axis) with the storm
  window shaded and a moving time marker -- the division of labor.
- Animated ponded-depth profile d(x) in mm (NEW vs the template): the thin sheet
  of overland water arriving at and being eaten by the grate inlet.
- |mass-balance error| log-y with moving marker.
- A paper-space METRICS panel: layering, both drain specs, forcing (with the
  raised-storm caveat), division of labor, perch contrast, base-drain contrast,
  mass balance, the NCP-residue fine print, and the honest framing.

Honest framing: this is an ILLUSTRATION on the validated CoupledProblem engine
with SCRIPT-LEVEL drain sinks (the engine GHB cannot tag interior/top facets;
the sinks are wired into the engine's drainage accounting so the balance closes
structurally) -- NOT a new validated module, NOT the retracted Module-4 sub-grid
embedded feature.

Usage:
    python viz/make_hillslope_drain_dual_html.py <result.nc> <out.html>

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


def _wrap(text: str, width: int = 50, indent: str = "&nbsp;&nbsp;") -> str:
    """Wrap a long attr string into indented <br>-joined lines for the panel."""
    if not text:
        return indent
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    return "<br>".join(indent + ln for ln in lines)


# ----------------------------------------------------------------------------- builder
def build_hillslope_drain_dual_html(nc_path: str, html_path: str) -> str:
    """Build the self-contained interactive dual-drain hillslope HTML.

    Parameters
    ----------
    nc_path : str
        Path to the standardized dual-drain hillslope sanity NetCDF.
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
    z = np.asarray(ds["z"].values, dtype=float)            # (z,)  0 = base -> 2 = surface
    time = np.asarray(ds["time"].values, dtype=float)      # (time,) day

    # --- fields ---
    head = np.asarray(ds["head_field"].values, dtype=float)    # (time, z, x)  psi [m]
    theta = np.asarray(ds["theta_field"].values, dtype=float)  # (time, z, x)  theta [-]
    wt = np.asarray(ds["water_table"].values, dtype=float)     # (time, x)  z of psi=0 [m]
    pond = np.asarray(ds["ponded_depth"].values, dtype=float)  # (time, x)  d [m]
    perch = np.asarray(ds["perched_thickness"].values, dtype=float)  # (time, x) [m]

    q_iface = np.asarray(ds["drain_discharge_iface"].values, dtype=float)    # m^2/day
    q_surf = np.asarray(ds["drain_discharge_surface"].values, dtype=float)   # m^2/day
    cum_iface = np.asarray(ds["cum_drainage_iface"].values, dtype=float)     # m^2
    cum_surf = np.asarray(ds["cum_drainage_surface"].values, dtype=float)    # m^2
    rain = np.asarray(ds["rainfall"].values, dtype=float)           # (time,) m/day
    mbe = np.asarray(ds["mass_balance_error"].values, dtype=float)  # (time,) [-]
    Ks_profile = np.asarray(ds["Ks_profile"].values, dtype=float)   # (z,) m/day

    x_units = _units(ds["x"], "m")
    z_units = _units(ds["z"], "m")
    t_units = _units(ds["time"], "day")
    head_units = _units(ds["head_field"], "m")
    theta_units = _units(ds["theta_field"], "-")
    q_units = _units(ds["drain_discharge_iface"], "m^2/day")
    cum_units = _units(ds["cum_drainage_iface"], "m^2")

    ntime = time.shape[0]
    pond_mm = pond * 1000.0  # [mm] for the sheet-flow panel

    # --- global attrs / metrics ---
    module = _attr(ds, "module", "m4_hillslope_drain_dual")
    date = _attr(ds, "date", "2026-06-10")
    domain = _attr(ds, "domain", "")
    mesh = _attr(ds, "mesh", "")
    loam_params = _attr(ds, "loam_params", "")
    clay_params = _attr(ds, "clay_params", "")
    clay_iface = _attr(ds, "clay_interface_z", 1.0)
    iface_drain_spec = _attr(ds, "iface_drain", "")
    surface_drain_spec = _attr(ds, "surface_drain", "")
    forcing_note = _attr(ds, "forcing_note", "")
    slope = _attr(ds, "surface_slope")
    init_wt = _attr(ds, "initial_water_table_m")

    cum_rain_tot = _attr(ds, "cum_rain_m2")
    cum_outflow = _attr(ds, "cum_outflow_m2")
    cum_drain_tot = _attr(ds, "cum_drainage_m2",
                          float(cum_iface[-1] + cum_surf[-1]))
    cum_iface_tot = _attr(ds, "cum_drainage_iface_m2", float(cum_iface[-1]))
    cum_surf_tot = _attr(ds, "cum_drainage_surface_m2", float(cum_surf[-1]))
    split_resum_err = _attr(ds, "drain_split_resum_err")

    mbe_final = _attr(ds, "mass_balance_error_final", float(mbe[-1]))
    mbe_max = _attr(ds, "mass_balance_error_max", float(np.max(np.abs(mbe))))

    max_pond_m = _attr(ds, "max_ponded_depth_m", float(np.nanmax(pond)))
    max_pond_x = _attr(ds, "max_ponded_x_m")
    max_pond_t = _attr(ds, "max_ponded_time_day")
    perch_max_thk = _attr(ds, "perched_zone_max_thickness_m",
                          float(np.nanmax(perch)))
    perch_max_x = _attr(ds, "perched_zone_x_m")
    perch_max_t = _attr(ds, "perched_zone_time_day")
    perch_end_drain = _attr(ds, "perch_endstorm_draincol_m")
    perch_end_far = _attr(ds, "perch_endstorm_farfield_m")
    wt_drain_final = _attr(ds, "water_table_draincol_final_m")
    wt_far_final = _attr(ds, "water_table_farfield_final_m")
    base_drain_cum = _attr(ds, "base_drain_run_cum_m2")

    note = _attr(ds, "note", "")
    wall_clock = _attr(ds, "wall_clock_s")
    steps = _attr(ds, "steps")
    status = _attr(ds, "status", "")
    snes_ls = _attr(ds, "snes_linesearch", "")

    # --- shared drain x-footprint (both drains: x in [9.5, 10.5]) ---
    DRAIN_X0, DRAIN_X1 = 9.5, 10.5
    TILE_Z0, TILE_Z1 = float(clay_iface), float(clay_iface) + 0.1  # on the clay
    z_top = float(z.max())                                          # land surface

    # --- Ks layer split (clay below interface, loam above) ---
    Ks_clay = float(np.nanmin(Ks_profile))
    Ks_loam = float(np.nanmax(Ks_profile))
    rain_peak = float(np.nanmax(rain))

    # --- storm window from the rainfall series (midpoint of last-wet/first-dry) ---
    wet = np.where(rain > 0)[0]
    if wet.size and wet[-1] + 1 < ntime:
        t_storm_end = 0.5 * (time[wet[-1]] + time[wet[-1] + 1])
    elif wet.size:
        t_storm_end = float(time[wet[-1]])
    else:
        t_storm_end = 0.0

    # --- surface-inlet engagement window (on the saved grid) + NCP residue ---
    on = np.where(q_surf > 1e-3)[0]
    if on.size:
        t_inlet_on, t_inlet_off = float(time[on[0]]), float(time[on[-1]])
        ncp_residue_cum = float(cum_surf[max(on[0] - 1, 0)])
        q_residue = float(np.max(q_surf[: on[0]])) if on[0] > 0 else 0.0
    else:
        t_inlet_on = t_inlet_off = float("nan")
        ncp_residue_cum = q_residue = 0.0

    # ----------------------------------------------------------------- color scaling
    # psi heatmap: fixed, symmetric across frames so the perch/notch evolve.
    psi_abs = max(abs(float(np.nanmin(head))), abs(float(np.nanmax(head))))
    psi_range = [-psi_abs, psi_abs]
    # theta heatmap: fixed to the data span so the saturated band is a clear break.
    th_lo = float(np.nanmin(theta))
    th_hi = float(np.nanmax(theta))
    pond_mm_max = float(np.nanmax(pond_mm))

    # drain colors: GREEN = interface tile (as the old drain marker), ORANGE = grate.
    C_TILE = "#00c853"
    C_TILE_DARK = "#00501f"
    C_GRATE = "#ef6c00"
    C_GRATE_DARK = "#bf360c"

    # ----------------------------------------------------------------- subplot layout
    # row1: psi(x,z) cross-section heatmap          (colspan 2)
    # row2: theta(x,z) cross-section heatmap        (colspan 2)  <- perch + notch
    # row3: WT profile + perch band (col 1) | dual hydrographs + cum (col 2, twin-y)
    # row4: ponded depth d(x) [mm] (col 1)  | |mass-balance error| log-y (col 2)
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
            f"&mdash; TWO drains share x in [9.5, 10.5] m: interface TILE on the "
            f"clay (z in [{_fmt(TILE_Z0)}, {_fmt(TILE_Z1)}]) + surface GRATE inlet",
            f"Cross-section: water content theta(x, z)  [{theta_units}]  "
            f"&mdash; the perched band builds on the clay EXCEPT at the tile "
            f"column (the notch)",
            "Water table + PERCHED band profile  (tile kills the perch at x=10)",
            "BOTH drain hydrographs &amp; cumulatives  (division of labor)",
            f"Ponded depth d(x)  [mm]  &mdash; sub-mm sheet flow feeds the grate "
            f"(max {1000.0 * float(max_pond_m):.2f} mm)",
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

    # --- shared cross-section annotations (BOTH drains + clay interface) rows 1 & 2 ---
    for r in (1, 2):
        # INTERFACE TILE: a green band sitting ON the clay, z in [1.0, 1.1].
        fig.add_shape(
            type="rect",
            x0=DRAIN_X0, x1=DRAIN_X1, y0=TILE_Z0, y1=TILE_Z1,
            line=dict(color=C_TILE, width=2),
            fillcolor="rgba(0,200,83,0.75)",
            row=r, col=1, layer="above",
        )
        # SURFACE GRATE INLET: an orange band at the land-surface edge, same x.
        fig.add_shape(
            type="rect",
            x0=DRAIN_X0, x1=DRAIN_X1, y0=z_top - 0.05, y1=z_top + 0.04,
            line=dict(color=C_GRATE, width=2),
            fillcolor="rgba(239,108,0,0.80)",
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
        # drain labels on both cross-sections
        fig.add_annotation(
            x=0.5 * (DRAIN_X0 + DRAIN_X1), y=TILE_Z0 - 0.18, ax=0, ay=0,
            xref="x", yref="y",
            text=f"<b>interface tile</b> (ON the clay)", showarrow=False,
            font=dict(size=10, color=C_TILE_DARK),
            bgcolor="rgba(255,255,255,0.55)",
            row=r, col=1,
        )
        fig.add_annotation(
            x=0.5 * (DRAIN_X0 + DRAIN_X1) + 3.4, y=z_top - 0.16, ax=0, ay=0,
            xref="x", yref="y",
            text="<b>surface grate inlet</b>", showarrow=False,
            font=dict(size=10, color=C_GRATE_DARK),
            bgcolor="rgba(255,255,255,0.55)",
            row=r, col=1,
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
    # perched-band callout on the theta panel (and the notch at the tile)
    fig.add_annotation(
        x=15.5, y=float(clay_iface) + 0.25, xref="x", yref="y",
        text="<b>perched saturated band</b><br>(loam waterlogs on the clay)",
        showarrow=True, arrowhead=2, arrowcolor="#004d40", arrowwidth=1.5,
        ax=0, ay=-26,
        font=dict(size=9.5, color="#004d40"), align="center",
        bgcolor="rgba(255,255,255,0.6)",
        row=2, col=1,
    )
    fig.add_annotation(
        x=0.5 * (DRAIN_X0 + DRAIN_X1), y=float(clay_iface) + 0.45, xref="x", yref="y",
        text="<b>the notch</b>: tile column<br>stays unperched",
        showarrow=True, arrowhead=2, arrowcolor=C_TILE_DARK, arrowwidth=1.5,
        ax=0, ay=-30,
        font=dict(size=9.5, color=C_TILE_DARK), align="center",
        bgcolor="rgba(255,255,255,0.6)",
        row=2, col=1,
    )

    # ============================================================ ROW 3 col1: WT + perch band
    # perch band: invisible baseline at the clay interface, then a filled trace
    # up to z = clay_iface + perched_thickness(x). MUST stay consecutive in
    # fig.data (fill="tonexty" binds to the PREVIOUS trace).
    fig.add_trace(
        go.Scatter(
            x=x, y=np.full_like(x, float(clay_iface)), mode="lines",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
            name="clay interface (perch base)",
        ),
        row=3, col=1,
    )
    i_perch = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=x, y=float(clay_iface) + perch[0], mode="lines",
            line=dict(color="#0277bd", width=1),
            fill="tonexty", fillcolor="rgba(2,119,189,0.20)",
            name="perched band (on clay)",
            hovertemplate=("x=%{x:.2f} m<br>perch top z=%{y:.3f} m"
                           "<extra>perched band</extra>"),
        ),
        row=3, col=1,
    )
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
    fig.add_hline(
        y=float(clay_iface), line=dict(color="#6d4c41", width=1.2, dash="dot"),
        row=3, col=1, layer="below",
        annotation_text=f"clay interface {float(clay_iface):.2f} m",
        annotation_position="top left",
        annotation_font=dict(size=9, color="#6d4c41"),
    )
    # the SHARED x-footprint of both drains (they sit at the same x)
    fig.add_vrect(
        x0=DRAIN_X0, x1=DRAIN_X1, fillcolor="rgba(0,200,83,0.16)",
        line_width=0, row=3, col=1, layer="below",
    )
    fig.add_annotation(
        x=0.5 * (DRAIN_X0 + DRAIN_X1), y=float(z.max()) - 0.12, xref="x", yref="y",
        text="shared drain<br>footprint", showarrow=False,
        font=dict(size=8.5, color=C_TILE_DARK), align="center",
        row=3, col=1,
    )

    # ============================================================ ROW 3 col2: dual hydrographs
    # storm-window shading first (layer below) so the timing split reads at a
    # glance. NOTE: plotly 6.x add_vrect(row=, col=) is a SILENT NO-OP on a
    # secondary_y subplot, so derive the axis refs and add the shape explicitly.
    sp_hydro = fig.get_subplot(3, 2)
    xref_hydro = sp_hydro.xaxis.plotly_name.replace("axis", "")   # e.g. 'x4'
    yref_hydro = sp_hydro.yaxis.plotly_name.replace("axis", "")   # e.g. 'y4'
    fig.add_shape(
        type="rect",
        x0=float(time.min()), x1=t_storm_end, y0=0, y1=1,
        xref=xref_hydro, yref=f"{yref_hydro} domain",
        fillcolor="rgba(30,136,229,0.10)", line_width=0, layer="below",
    )
    fig.add_annotation(
        x=float(time.min()) + 0.05, y=0.985,
        xref=xref_hydro, yref=f"{yref_hydro} domain",
        text=f"storm {_fmt(rain_peak)} m/day", showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(size=9, color="#1565c0"),
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=q_iface, mode="lines",
            line=dict(color="#00897b", width=2.2),
            name="interface tile q",
            hovertemplate="t=%{x:.3f} day<br>q=%{y:.5f} " + q_units + "<extra>tile</extra>",
        ),
        row=3, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=q_surf, mode="lines",
            line=dict(color=C_GRATE, width=2.2),
            name="surface inlet q",
            hovertemplate="t=%{x:.3f} day<br>q=%{y:.5f} " + q_units + "<extra>inlet</extra>",
        ),
        row=3, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=cum_iface, mode="lines",
            line=dict(color="#004d40", width=2, dash="dash"),
            name="cum tile",
            hovertemplate="t=%{x:.3f} day<br>cum=%{y:.5f} " + cum_units + "<extra>tile</extra>",
        ),
        row=3, col=2, secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=cum_surf, mode="lines",
            line=dict(color=C_GRATE_DARK, width=2, dash="dot"),
            name="cum inlet",
            hovertemplate="t=%{x:.3f} day<br>cum=%{y:.5f} " + cum_units + "<extra>inlet</extra>",
        ),
        row=3, col=2, secondary_y=True,
    )
    i_mark_qi = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[q_iface[0]], mode="markers",
            marker=dict(size=11, color="#c62828", symbol="circle-open",
                        line=dict(width=3, color="#c62828")),
            name="current time", showlegend=True,
            hovertemplate="t=%{x:.3f} day<extra>now (tile)</extra>",
        ),
        row=3, col=2, secondary_y=False,
    )
    i_mark_qs = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[q_surf[0]], mode="markers",
            marker=dict(size=9, color="#c62828", symbol="diamond-open",
                        line=dict(width=2.5, color="#c62828")),
            name="now (inlet)", showlegend=False,
            hovertemplate="t=%{x:.3f} day<extra>now (inlet)</extra>",
        ),
        row=3, col=2, secondary_y=False,
    )
    # inlet engagement window note (computed on the saved output grid)
    if np.isfinite(t_inlet_on):
        fig.add_annotation(
            x=t_inlet_off + 0.15, y=float(np.nanmax(q_surf)), xref="x", yref="y",
            text=(f"inlet fires ONLY in the ponding window<br>"
                  f"t &asymp; {t_inlet_on:.2f}&ndash;{t_inlet_off:.2f} d "
                  f"(saved grid)"),
            showarrow=False, xanchor="left",
            font=dict(size=8.5, color=C_GRATE_DARK), align="left",
            row=3, col=2,
        )

    # ============================================================ ROW 4 col1: ponded depth d(x)
    i_pond = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=x, y=pond_mm[0], mode="lines",
            line=dict(color="#0288d1", width=2.2),
            fill="tozeroy", fillcolor="rgba(2,136,209,0.18)",
            name="ponded depth d(x)",
            hovertemplate="x=%{x:.2f} m<br>d=%{y:.4f} mm<extra></extra>",
            showlegend=False,
        ),
        row=4, col=1,
    )
    # grate-inlet footprint (orange) -- where the sheet flow gets eaten
    fig.add_vrect(
        x0=DRAIN_X0, x1=DRAIN_X1, fillcolor="rgba(239,108,0,0.18)",
        line_width=0, row=4, col=1, layer="below",
    )
    fig.add_annotation(
        x=0.5 * (DRAIN_X0 + DRAIN_X1), y=0.92 * 1.25 * pond_mm_max,
        xref="x", yref="y",
        text="grate inlet", showarrow=False,
        font=dict(size=8.5, color=C_GRATE_DARK),
        row=4, col=1,
    )
    fig.add_hline(
        y=1000.0 * float(max_pond_m),
        line=dict(color="#b71c1c", width=1.2, dash="dash"),
        row=4, col=1, layer="above",
        annotation_text=(f"max {1000.0 * float(max_pond_m):.2f} mm "
                         f"(x={_fmt(max_pond_x)}, t={_fmt(max_pond_t, 3)} d) "
                         f"&mdash; thin-film sheet flow"),
        annotation_position="bottom left",
        annotation_font=dict(size=9, color="#b71c1c"),
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
                    go.Heatmap(z=head[k]),                         # psi cross-section
                    go.Scatter(x=x, y=wt[k]),                      # psi-panel WT contour
                    go.Heatmap(z=theta[k]),                        # theta cross-section
                    go.Scatter(x=x, y=wt[k]),                      # theta-panel WT contour
                    go.Scatter(x=x, y=float(clay_iface) + perch[k]),  # perch band top
                    go.Scatter(x=x, y=wt[k]),                      # WT profile panel
                    go.Scatter(x=[time[k]], y=[q_iface[k]]),       # now-marker tile q
                    go.Scatter(x=[time[k]], y=[q_surf[k]]),        # now-marker inlet q
                    go.Scatter(x=x, y=pond_mm[k]),                 # ponded-depth profile
                    go.Scatter(x=[time[k]], y=[cur_err]),          # now-marker mbe
                ],
                traces=[i_heat, i_wt_cross, i_theta, i_wt_theta, i_perch,
                        i_wt_prof, i_mark_qi, i_mark_qs, i_pond, i_mark_mbe],
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
        title_text=f"z  [{z_units}]  (0 = base; top = surface)",
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
    # row3 col1 WT + perch profile
    fig.update_xaxes(title_text=f"x  [{x_units}]", range=[float(x.min()), float(x.max())],
                     row=3, col=1)
    fig.update_yaxes(title_text=f"z  [{z_units}]",
                     range=[float(z.min()), float(z.max())], row=3, col=1)
    # row3 col2 dual hydrographs twin
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=3, col=2)
    fig.update_yaxes(title_text=f"discharge  [{q_units}]", row=3, col=2,
                     secondary_y=False, color="#00695c", rangemode="tozero")
    fig.update_yaxes(title_text=f"cum  [{cum_units}]", row=3, col=2,
                     secondary_y=True, color="#4e342e", showgrid=False,
                     rangemode="tozero")
    # row4 col1 ponded depth
    fig.update_xaxes(title_text=f"x  [{x_units}]", range=[float(x.min()), float(x.max())],
                     row=4, col=1)
    fig.update_yaxes(title_text="ponded depth d  [mm]",
                     range=[0.0, 1.25 * pond_mm_max], row=4, col=1)
    # row4 col2 mbe (log)
    fig.update_xaxes(title_text=f"time  [{t_units}]", row=4, col=2)
    fig.update_yaxes(title_text="|mass-balance error|  (-)",
                     type="log", exponentformat="power", row=4, col=2)

    # ----------------------------------------------------------------- metrics panel
    pct_iface = 100.0 * cum_iface_tot / cum_drain_tot if cum_drain_tot else 0.0
    pct_surf = 100.0 * cum_surf_tot / cum_drain_tot if cum_drain_tot else 0.0
    ratio_tile = (cum_iface_tot / base_drain_cum) if base_drain_cum else None
    ratio_tot = (cum_drain_tot / base_drain_cum) if base_drain_cum else None
    metrics_lines = [
        f"<b>module</b>: {module}",
        "<b>scenario</b>: DUAL drain, one x-footprint &mdash;",
        "&nbsp;&nbsp;PLACEMENT is the only difference",
        f"<b>domain</b>: {domain}",
        f"<b>mesh</b>: {mesh}",
        "<b>&mdash; layering / Ks &mdash;</b>",
        f"&nbsp;&nbsp;loam: {loam_params}",
        f"&nbsp;&nbsp;clay: {clay_params}",
        (f"&nbsp;&nbsp;Ks loam {_fmt(Ks_loam)} vs clay {_fmt(Ks_clay)} m/day "
         f"(<b>{Ks_loam / Ks_clay:.0f}x</b>)"),
        f"<b>drain 1 &mdash; INTERFACE TILE (on the clay)</b>:",
        _wrap(iface_drain_spec),
        f"<b>drain 2 &mdash; SURFACE GRATE inlet</b>:",
        _wrap(surface_drain_spec),
        (f"<b>storm</b>: {_fmt(rain_peak)} m/day for {t_storm_end:.1f} d, "
         f"then recession to {float(time.max()):.1f} d"),
        f"&nbsp;&nbsp;initial table = {_fmt(init_wt)} m; slope {_fmt(slope)}",
        "<span style='color:#b71c1c'><b>&nbsp;&nbsp;forcing caveat:</b></span>",
        "<span style='color:#b71c1c'>" + _wrap(forcing_note) + "</span>",
        f"<b>&mdash; division of labor (cum, {cum_units}) &mdash;</b>",
        (f"&nbsp;&nbsp;interface tile&nbsp;&nbsp;<b>{_fmt(cum_iface_tot)}</b> "
         f"({pct_iface:.0f}% of drain total)"),
        (f"&nbsp;&nbsp;surface inlet&nbsp;&nbsp;&nbsp;<b>{_fmt(cum_surf_tot)}</b> "
         f"({pct_surf:.0f}%)"),
        f"&nbsp;&nbsp;drain total&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{_fmt(cum_drain_tot)}",
        (f"&nbsp;&nbsp;outlet runoff (x=20)&nbsp;{_fmt(cum_outflow)}; "
         f"rain {_fmt(cum_rain_tot)}"),
        "<b>&mdash; the PERCH contrast (read me) &mdash;</b>",
        (f"&nbsp;&nbsp;end-of-storm perched band: <b>{_fmt(perch_end_far)} m</b> "
         f"far-field (x=2)<br>&nbsp;&nbsp;vs <b>{_fmt(perch_end_drain)} m</b> at "
         f"the drain column (x=10)<br>&nbsp;&nbsp;&rArr; the tile locally "
         f"ELIMINATES the perch"),
        (f"&nbsp;&nbsp;max perch {_fmt(perch_max_thk)} m at x={_fmt(perch_max_x)}, "
         f"t={_fmt(perch_max_t, 3)} d<br>&nbsp;&nbsp;(loam column is 1.0 m: nearly "
         f"fully saturated<br>&nbsp;&nbsp;&rarr; that's why it ponds)"),
        (f"&nbsp;&nbsp;final table: <b>{_fmt(wt_drain_final, 3)} m</b> drain col "
         f"vs <b>{_fmt(wt_far_final, 3)} m</b> far-field"),
        "<b>&mdash; vs the BURIED-base-drain clay run &mdash;</b>",
        (f"&nbsp;&nbsp;base drain in the clay captured "
         f"<b>{_fmt(base_drain_cum)} {cum_units}</b>;"),
        (f"&nbsp;&nbsp;here tile alone {_fmt(cum_iface_tot)} "
         f"(~{ratio_tile:.0f}x), total {_fmt(cum_drain_tot)} "
         f"(~{ratio_tot:.0f}x)" if ratio_tile else
         "&nbsp;&nbsp;(no base-drain reference attr)"),
        ("<span style='color:#b71c1c'>&nbsp;&nbsp;CAVEAT: that run had a WEAKER "
         "storm (0.085 x 1.5 d)<br>&nbsp;&nbsp;&mdash; not apples-to-apples "
         "forcing</span>"),
        "<b>&mdash; diagnostics &mdash;</b>",
        (f"&nbsp;&nbsp;|bal|/cum_rain: final {_fmt(mbe_final)},<br>"
         f"&nbsp;&nbsp;max {_fmt(mbe_max)}  (machine)"),
        f"&nbsp;&nbsp;per-drain split re-sum err {_fmt(split_resum_err)}",
        (f"&nbsp;&nbsp;status {status} in {_fmt(steps)} steps "
         f"(linesearch {snes_ls}), {_fmt(wall_clock)} s"),
        "<b>&mdash; fine print (inlet trace) &mdash;</b>",
        (f"&nbsp;&nbsp;the ~{_fmt(q_residue)} {q_units} standing value BEFORE<br>"
         f"&nbsp;&nbsp;ponding is the NCP smoothing residue of d<br>"
         f"&nbsp;&nbsp;(numerical, negligible: ~{_fmt(ncp_residue_cum)} "
         f"{cum_units} cumulative),<br>&nbsp;&nbsp;NOT physical capture"),
        "<b>&mdash; framing (read me) &mdash;</b>",
        "<span style='color:#b71c1c'>" + _wrap(note) + "</span>",
    ]
    metrics_html = "<br>".join(metrics_lines)
    fig.add_annotation(
        x=1.004, y=1.0, xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        align="left", showarrow=False,
        text="<b>METRICS</b><br>" + metrics_html,
        font=dict(size=9.2, color="#222", family="Consolas, monospace"),
        bordercolor="#888", borderwidth=1, borderpad=7,
        bgcolor="rgba(245,245,245,0.97)",
    )

    fig.update_layout(
        title=dict(
            text=(f"<b>PIDS Pillar-2 sanity (Tier-3) &middot; hillslope DUAL "
                  f"drain: surface inlet vs interface tile &middot; {date}</b><br>"
                  f"<span style='font-size:12px'>the interface tile ON the clay "
                  f"does the heavy lifting ({_fmt(cum_iface_tot, 3)} m&sup2;, "
                  f"kills the perch locally); the surface inlet only engages "
                  f"during saturation-excess ponding "
                  f"({_fmt(cum_surf_tot, 3)} m&sup2;)</span>"),
            x=0.5, xanchor="center", y=0.992, yanchor="top",
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.012,
                    xanchor="right", x=1.0, font=dict(size=10)),
        margin=dict(l=75, r=480, t=130, b=70),
        width=1380, height=1380,
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
        print("usage: python make_hillslope_drain_dual_html.py <result.nc> <out.html>",
              file=sys.stderr)
        sys.exit(2)
    out = build_hillslope_drain_dual_html(sys.argv[1], sys.argv[2])
    import os
    print(f"WROTE {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
