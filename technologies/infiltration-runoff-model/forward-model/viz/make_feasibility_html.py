"""Tier-3 sanity visualizer for the PIDS Pillar-2 FIELD-SCALE FEASIBILITY run.

Turns the standardized feasibility NetCDF result (the data contract of
``governance/visualize-sanity-check-routine.md``) into ONE self-contained,
offline, interactive HTML. The generator reads ONLY the result file -- it never
imports or runs the solver (``pids_forward``); that independence is the whole
point of the Tier-3 lens.

This visualizes a 2-ha, HETEROGENEOUS, LAYERED-soil 3-D hillslope feasibility
run. The point of the viz is to let the human VISUALLY VALIDATE that the
heterogeneous layered run behaved correctly:

  * SOIL STRUCTURE (static, row 1): the STATIC Ks(z) profile on a LOG x-axis vs
    elevation z (surface at top). This SHOWS the heterogeneity -- the loam->clay
    Ks decay with depth and the SAND capillary-barrier spike at z in [0.50,0.51].
  * psi CROSS-SECTION (row 2, full width, animated): head_xsec(xc,z) with the
    surface at the TOP, the WATER TABLE drawn as the psi=0 contour, and a thin
    horizontal line marking the sand barrier at z~0.505 -- the wetting front
    advancing into the layered profile + any perching above the sand.
  * theta CROSS-SECTION (row 3, full width, animated): theta_xsec(xc,z) with the
    sand-layer line -- the moisture front + the sand layer's distinct theta.
  * SURFACE PONDING d(x,y) (row 4, animated, FIXED 0..4 mm range in mm): the thin
    overland sheet routing toward the x=210 outlet edge (marked).
  * TOP-DOWN theta z-LAYERS (row 5, four small maps, animated, shared theta
    range): surface / above-sand / below-sand / base -- lateral wetting / perching
    at depth.
  * OUTLET HYDROGRAPH (row 6): outflow(t) with the rain hyetograph inverted on a
    twin axis + a slider-tracking current-time marker.
  * PARTITION / CONSERVATION (row 7, full width): cum_rain vs cumulative
    infiltration (soil_water - soil_water[0]) vs ponding (surface_water) vs
    cum_outflow (they sum to cum_rain).
  * DIAGNOSTICS (row 8): mass-balance error (log-y) + Newton iterations.
  * METRICS panel (text from attrs): the deciding numbers.

The two WIDE cross-section heatmaps each own a FULL-WIDTH row (stacked
vertically) so their axis titles never collide with a neighbour.

Usage:
    python viz/make_feasibility_html.py <result.nc> <out.html>

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


def _watertable_z(psi_zx: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Locate the water-table elevation (psi=0 iso-line) per column of a
    cross-section ``psi_zx`` shaped (z, xc).

    For each xc column we scan from the surface (top) downward for the first
    sign change and linearly interpolate the elevation where psi crosses 0 (the
    top of the saturated zone). Columns with no crossing return NaN so the
    contour line simply breaks there.

    Returns an array shaped (xc,) of water-table elevations [same units as z].
    """
    nz, nxc = psi_zx.shape
    wt = np.full(nxc, np.nan, dtype=float)
    for j in range(nxc):
        col = psi_zx[:, j]
        found = np.nan
        for i in range(nz - 1, 0, -1):
            a, b = col[i - 1], col[i]      # a is lower z, b is upper z
            if np.isnan(a) or np.isnan(b):
                continue
            if (a >= 0.0) and (b < 0.0):
                denom = (a - b)
                frac = a / denom if denom != 0 else 0.0
                found = z[i - 1] + frac * (z[i] - z[i - 1])
                break
            if (a < 0.0) and (b >= 0.0):
                denom = (b - a)
                frac = (-a) / denom if denom != 0 else 0.0
                found = z[i - 1] + frac * (z[i] - z[i - 1])
                break
        if np.isnan(found):
            if np.all(col >= 0.0):
                found = z[-1]
        wt[j] = found
    return wt


# ----------------------------------------------------------------------------- builder
def build_html(nc_path: str, out_path: str) -> str:
    """Build the self-contained interactive feasibility sanity HTML.

    Parameters
    ----------
    nc_path : str
        Path to the standardized feasibility-run NetCDF result file.
    out_path : str
        Path to write the self-contained HTML (plotly inlined, offline).

    Returns
    -------
    str
        The ``out_path`` written.
    """
    ds = xr.open_dataset(nc_path)

    # --- coords (item-access so ``ds['x']`` is the var, not a method) ---
    x = np.asarray(ds["x"].values, dtype=float)          # (x,)  downslope [m], outlet at max
    y = np.asarray(ds["y"].values, dtype=float)          # (y,)  cross-slope [m]
    xc = np.asarray(ds["xc"].values, dtype=float)        # (xc,) downslope for the cross-section [m]
    z = np.asarray(ds["z"].values, dtype=float)          # (z,)  GRADED elevation [m], 0=base..1=surface
    zlayer = np.asarray(ds["zlayer"].values, dtype=float)  # (zlayer,) 4 elevations [m]
    time = np.asarray(ds["time"].values, dtype=float)    # (time,) [day]

    # --- 3-D / 2-D fields ---
    d_map = np.asarray(ds["surface_depth_map"].values, dtype=float)   # (time, y, x) [m]
    psi_xs = np.asarray(ds["head_xsec"].values, dtype=float)          # (time, z, xc) [m]
    th_xs = np.asarray(ds["theta_xsec"].values, dtype=float)          # (time, z, xc) [-]
    theta_xy = np.asarray(ds["theta_xy"].values, dtype=float)         # (time, zlayer, y, x) [-]
    Ks_prof = np.asarray(ds["Ks_profile"].values, dtype=float)        # (z,) [m/day] STATIC

    # --- time series ---
    rain = np.asarray(ds["rainfall"].values, dtype=float)            # (time,) [m/day]
    soil_w = np.asarray(ds["soil_water"].values, dtype=float)        # (time,) [m^3]
    surf_w = np.asarray(ds["surface_water"].values, dtype=float)     # (time,) [m^3]
    outflow = np.asarray(ds["outflow"].values, dtype=float)         # (time,) [m^3/day]
    cum_outflow = np.asarray(ds["cum_outflow"].values, dtype=float)  # (time,) [m^3]
    cum_rain = np.asarray(ds["cum_rain"].values, dtype=float)        # (time,) [m^3]
    mbe = np.asarray(ds["mass_balance_error"].values, dtype=float)    # (time,) [-]
    niter = np.asarray(ds["newton_iters"].values, dtype=float)      # (time,) [-]

    x_units = _units(ds["x"], "m")
    y_units = _units(ds["y"], "m")
    z_units = _units(ds["z"], "m")
    zlayer_units = _units(ds["zlayer"], "m")
    t_units = _units(ds["time"], "day")
    psi_units = _units(ds["head_xsec"], "m")
    ks_units = _units(ds["Ks_profile"], "m/day")
    thxy_units = _units(ds["theta_xy"], "-")
    r_units = _units(ds["rainfall"], "m/day")
    q_units = _units(ds["outflow"], "m^3/day")
    vol_units = _units(ds["cum_rain"], "m^3")

    # cumulative infiltration = change in subsurface storage relative to t=0 [m^3].
    infil = soil_w - soil_w[0]

    ntime = time.shape[0]
    L = float(x.max())  # outlet edge (downslope, x=210)
    H = float(z.max())  # surface elevation (=1)

    # --- SAND capillary-barrier layer: locate from the Ks profile (the spike) ---
    # The contract states a thin sand layer at z in [0.50, 0.51]; mark its centre.
    ks_med = float(np.median(Ks_prof))
    sand_mask = Ks_prof > (5.0 * max(ks_med, 1e-9))
    if np.any(sand_mask):
        z_sand_lo = float(np.min(z[sand_mask]))
        z_sand_hi = float(np.max(z[sand_mask]))
        z_sand = 0.5 * (z_sand_lo + z_sand_hi)
        ks_sand = float(np.max(Ks_prof))
    else:
        z_sand_lo, z_sand_hi, z_sand, ks_sand = 0.50, 0.51, 0.505, float(np.max(Ks_prof))

    # --- water-table (psi=0) iso-line per frame, for the cross-section overlay ---
    wt_z = np.vstack([_watertable_z(psi_xs[k], z) for k in range(ntime)])  # (time, xc)

    # --- global attrs / metrics ---
    module = _attr(ds, "module", "feasibility_2ha_layered")
    scenario = _attr(ds, "scenario", "2-ha heterogeneous layered hillslope")
    date = _attr(ds, "date", "")
    domain = _attr(ds, "domain")
    mesh = _attr(ds, "mesh")
    status = _attr(ds, "status", "")
    y_section = _attr(ds, "y_section_m")
    wall_clock = _attr(ds, "wall_clock_s")
    steps = _attr(ds, "steps")
    sec_per_step = _attr(ds, "sec_per_step")
    cum_rain_tot = _attr(ds, "cum_rain_m3")
    cum_outflow_tot = _attr(ds, "cum_outflow_m3")
    runoff_frac = _attr(ds, "runoff_fraction")
    soil_w_final = _attr(ds, "soil_water_final_m3")
    mbe_max = _attr(ds, "mass_balance_error_max", float(np.max(np.abs(mbe))))
    clip_mass = _attr(ds, "clip_mass_adjust")
    peak_d_attr = _attr(ds, "peak_surface_depth_m")

    # --- fixed color ranges so features MOVE / GROW rather than rescale per frame ---
    psi_lo = float(np.nanmin(psi_xs)); psi_hi = float(np.nanmax(psi_xs))
    th_lo = float(np.nanmin(th_xs)); th_hi = float(np.nanmax(th_xs))

    # PONDING colour range: FIXED, physically-meaningful, anchored at 0. The storm
    # peak overland sheet is ~3.7 mm; a fixed 0..4 mm full-scale keeps the thin
    # sheet visible and identical across all frames. Display in mm (data untouched).
    POND_ZMAX_MM = 4.0
    d_map_mm = d_map * 1e3
    d_peak = float(np.nanmax(d_map))           # this run's true storm-peak [m]

    # THETA(x,y) LAYER colour range: ONE shared fixed range across the 4 layers so
    # saturation is comparable everywhere. Anchor to the residual/saturated band
    # but expand to data extremes if needed.
    THXY_LO = min(0.045, float(np.nanmin(theta_xy)))
    THXY_HI = max(0.43, float(np.nanmax(theta_xy)))

    # Label the 4 z-layers by descending elevation: surface / above-sand /
    # below-sand / base.
    order = np.argsort(-zlayer)                  # surface (high z) first
    if zlayer.shape[0] == 4:
        layer_names = ["surface", "above-sand", "below-sand", "base"]
    else:
        layer_names = [f"layer {i}" for i in range(zlayer.shape[0])]
    layer_specs = [(int(idx), layer_names[i],
                    (0.0 if float(zlayer[idx]) == 0.0 else float(zlayer[idx])))
                   for i, idx in enumerate(order)]

    # =============================================================== figure layout
    #   row1 col1     : SOIL STRUCTURE -- static Ks(z) profile, LOG x, surface at top
    #   row1 col2-3   : (reserved spacer for metrics readability)  -> we use col1 only
    #   row2 (colspan): psi cross-section psi(xc,z) [animated] + psi=0 water table
    #                   + sand-barrier line
    #   row3 (colspan): theta cross-section theta(xc,z) [animated] + sand-barrier line
    #   row4 col1     : surface ponding map d(x,y) [animated, fixed 0..4 mm]
    #   row4 col2-3   : outlet hydrograph (outflow) + rain inverted (twin axis)
    #   row5 col1-4   : 4 top-down theta(x,y) layer maps (surface/above/below/base)
    #   row6 (colspan): partition -- cum_rain = infil + ponding + outflow
    #   row7 col1     : mass-balance error (log-y)   row7 col2-3 : Newton iterations
    #
    # The two WIDE cross-section heatmaps each own a FULL-WIDTH row. Row 5 holds the
    # 4 theta-layer maps across 4 columns with generous spacing. The grid is 4
    # columns; full-width rows use colspan=4 and the hydrograph/Newton panels span
    # the remaining columns.
    fig = make_subplots(
        rows=7,
        cols=4,
        specs=[
            # row1: soil structure (col1-2) ; col3-4 left empty (metrics panel sits to the right)
            [{"type": "xy", "colspan": 2}, None, None, None],
            # row2: psi cross-section, full width
            [{"type": "xy", "colspan": 4}, None, None, None],
            # row3: theta cross-section, full width
            [{"type": "xy", "colspan": 4}, None, None, None],
            # row4: ponding map (col1-2) + hydrograph (col3-4, twin y)
            [{"type": "xy", "colspan": 2}, None, {"type": "xy", "secondary_y": True, "colspan": 2}, None],
            # row5: 4 theta-layer top-down maps
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            # row6: partition, full width
            [{"type": "xy", "colspan": 4}, None, None, None],
            # row7: diagnostics -- mbe (col1-2) + newton (col3-4)
            [{"type": "xy", "colspan": 2}, None, {"type": "xy", "colspan": 2}, None],
        ],
        row_heights=[0.135, 0.135, 0.135, 0.135, 0.12, 0.16, 0.13],
        vertical_spacing=0.052,
        horizontal_spacing=0.075,
        subplot_titles=(
            # row1
            f"SOIL STRUCTURE (static): Ks(z)  [{ks_units}]  log-x vs elevation z "
            f"(surface at top) &mdash; loam&rarr;clay decay + SAND barrier",
            # row2
            f"Pressure head &psi;(x,z)  [{psi_units}]  on y={_fmt(y_section)} m  "
            f"(surface at top; black = water table &psi;=0; orange = sand barrier)",
            # row3
            f"Water content &theta;(x,z)  [-]  on y={_fmt(y_section)} m  "
            f"(surface at top; orange = sand barrier @ z&asymp;{_fmt(z_sand)} m)",
            # row4
            f"Surface ponding d(x,y) [mm]  &mdash; fixed 0&ndash;{POND_ZMAX_MM:g} mm  "
            f"(thin overland sheet &rarr; outlet x={_fmt(L)} m)",
            "Outlet hydrograph: overland Q vs rain (hyetograph inverted)",
            # row5: 4 theta-layer titles
            (f"&theta;(x,y) {layer_specs[0][1]}  z={_fmt(layer_specs[0][2])} {zlayer_units}"),
            (f"&theta;(x,y) {layer_specs[1][1]}  z={_fmt(layer_specs[1][2])} {zlayer_units}"),
            (f"&theta;(x,y) {layer_specs[2][1]}  z={_fmt(layer_specs[2][2])} {zlayer_units}"),
            (f"&theta;(x,y) {layer_specs[3][1]}  z={_fmt(layer_specs[3][2])} {zlayer_units}"),
            # row6
            "Partition: cum rain = infiltration + ponding + overland outflow  [m&sup3;]",
            # row7
            "Diagnostics: mass-balance error (log)",
            "Diagnostics: Newton iterations",
        ),
    )

    KS_COLOR = "#5d4037"
    SAND_COLOR = "#ef6c00"      # sand-barrier marker line (warm, distinct)
    Q_COLOR = "#0277bd"
    RAIN_COLOR = "#90a4ae"
    RAIN_BORDER = "#546e7a"
    CUM_COLOR = "#37474f"
    INFIL_COLOR = "#2e7d32"
    POND_COLOR = "#00838f"
    OUT_FILL = "#0277bd"
    MBE_COLOR = "#e67e22"
    ITER_COLOR = "#6a1b9a"
    WT_COLOR = "#111111"

    # =============================================================== row1: SOIL STRUCTURE Ks(z) (static)
    # Ks on a LOG x-axis vs elevation z (y). z runs surface (top) to base (bottom),
    # which is the natural reading of a soil profile. STATIC (no animation).
    fig.add_trace(
        go.Scatter(
            x=Ks_prof, y=z, mode="lines+markers",
            line=dict(color=KS_COLOR, width=2.6, shape="vh"),
            marker=dict(size=5, color=KS_COLOR),
            name=f"Ks(z)  [{ks_units}]",
            hovertemplate="z=%{y:.3f} m<br>Ks=%{x:.4g} " + ks_units + "<extra></extra>",
        ),
        row=1, col=1,
    )
    # sand-barrier horizontal marker on the Ks panel.
    fig.add_hline(y=z_sand, line=dict(color=SAND_COLOR, width=2, dash="dot"), row=1, col=1)
    fig.add_annotation(
        x=np.log10(max(ks_sand, 1e-9)), y=z_sand,
        xref="x", yref="y", xanchor="right", yanchor="bottom",
        align="right", showarrow=False,
        text=f"SAND barrier z&asymp;{_fmt(z_sand)} m<br>Ks&asymp;{_fmt(ks_sand)} {ks_units}",
        font=dict(size=9.5, color=SAND_COLOR),
        bgcolor="rgba(255,243,224,0.92)", bordercolor=SAND_COLOR, borderwidth=1, borderpad=3,
    )
    fig.add_annotation(
        x=np.log10(max(float(Ks_prof[0]), 1e-9)), y=float(z[0]),
        xref="x", yref="y", xanchor="left", yanchor="top",
        align="left", showarrow=False,
        text="loam&rarr;clay: Ks decays toward the impermeable base",
        font=dict(size=9, color=KS_COLOR),
        bgcolor="rgba(239,235,233,0.9)", bordercolor="#a1887f", borderwidth=1, borderpad=3,
    )

    # =============================================================== row2: psi cross-section (FULL WIDTH) + water table + sand line
    i_psi = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            x=xc, y=z, z=psi_xs[0],
            zmin=psi_lo, zmax=psi_hi, zauto=False,
            colorscale="RdBu",
            colorbar=dict(title=dict(text=f"&psi; [{psi_units}]", side="right"),
                          len=0.115, y=0.78, x=1.0, thickness=12),
            hovertemplate="x=%{x:.3f} m<br>z=%{y:.3f} m<br>&psi;=%{z:.4f} m<extra></extra>",
            name="&psi; x-section",
        ),
        row=2, col=1,
    )
    i_wt = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=xc, y=wt_z[0], mode="lines",
            line=dict(color=WT_COLOR, width=3.0),
            connectgaps=False,
            name="water table (&psi;=0)",
            hovertemplate="x=%{x:.3f} m<br>water table z=%{y:.4f} m<extra>&psi;=0</extra>",
        ),
        row=2, col=1,
    )
    # sand-barrier horizontal line on the psi cross-section.
    fig.add_hline(y=z_sand, line=dict(color=SAND_COLOR, width=1.6, dash="dot"), row=2, col=1)

    # =============================================================== row3: theta cross-section (FULL WIDTH) + sand line
    i_theta = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            x=xc, y=z, z=th_xs[0],
            zmin=th_lo, zmax=th_hi, zauto=False,
            colorscale="YlGnBu",
            colorbar=dict(title=dict(text="&theta; [-]", side="right"),
                          len=0.115, y=0.645, x=1.0, thickness=12),
            hovertemplate="x=%{x:.3f} m<br>z=%{y:.3f} m<br>&theta;=%{z:.4f}<extra></extra>",
            name="&theta; x-section",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=z_sand, line=dict(color=SAND_COLOR, width=1.6, dash="dot"), row=3, col=1)

    # =============================================================== row4 col1: surface ponding map d(x,y)
    i_dmap = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            x=x, y=y, z=d_map_mm[0],
            zmin=0.0, zmax=POND_ZMAX_MM, zauto=False,
            colorscale="Blues",
            colorbar=dict(title=dict(text="d [mm]", side="right"),
                          len=0.11, y=0.475, x=0.46, thickness=12),
            hovertemplate="x=%{x:.3f} m<br>y=%{y:.3f} m<br>d=%{z:.3e} mm<extra></extra>",
            name="surface depth",
        ),
        row=4, col=1,
    )

    # =============================================================== row4 col3: outlet hydrograph + rain inverted
    fig.add_trace(
        go.Scatter(
            x=time, y=outflow, mode="lines",
            line=dict(color=Q_COLOR, width=2.8),
            name=f"overland Q  [{q_units}]",
            legendgroup="hydro",
            hovertemplate="t=%{x:.4f} day<br>Q=%{y:.4e} " + q_units + "<extra></extra>",
        ),
        row=4, col=3, secondary_y=False,
    )
    r_hi = float(np.nanmax(rain))
    r_top = (r_hi if r_hi > 0 else 1.0) * 1.05
    r_axis_top = r_top * 3.0  # rain occupies only the top third (clear of the Q line)
    fig.add_trace(
        go.Bar(
            x=time, y=rain, base=r_axis_top - rain,
            marker=dict(color=RAIN_COLOR, line=dict(color=RAIN_BORDER, width=0.5)),
            opacity=0.8,
            width=(time[1] - time[0]) * 0.9 if ntime > 1 else 0.005,
            name=f"rainfall  [{r_units}]",
            hovertemplate="t=%{x:.4f} day<br>r=%{base:.3f} " + r_units + "<extra></extra>",
        ),
        row=4, col=3, secondary_y=True,
    )
    i_qmark = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[outflow[0]], mode="markers",
            marker=dict(size=11, color=Q_COLOR, symbol="circle-open",
                        line=dict(width=3, color=Q_COLOR)),
            name="current time", showlegend=False,
            hovertemplate="t=%{x:.4f} day<br>Q=%{y:.4e} " + q_units + "<extra>current</extra>",
        ),
        row=4, col=3, secondary_y=False,
    )

    # =============================================================== row5: 4 top-down theta(x,y) layer maps
    i_thxy = []  # trace indices in plotted column order (surface, above, below, base)
    for col_i, (zidx, lname, zval) in enumerate(layer_specs, start=1):
        show_cbar = (col_i == len(layer_specs))
        cbar = (dict(title=dict(text=f"&theta; [{thxy_units}]", side="right"),
                     len=0.10, y=0.32, x=1.0, thickness=12)
                if show_cbar else None)
        i_thxy.append(len(fig.data))
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=theta_xy[0, zidx],
                zmin=THXY_LO, zmax=THXY_HI, zauto=False,
                colorscale="YlGnBu",
                showscale=show_cbar,
                colorbar=cbar,
                hovertemplate=("x=%{x:.3f} m<br>y=%{y:.3f} m<br>"
                               "&theta;=%{z:.4f}<extra>" + lname +
                               f" z={_fmt(zval)} {zlayer_units}</extra>"),
                name=f"theta {lname}",
            ),
            row=5, col=col_i,
        )

    # =============================================================== row6: partition (conservation, FULL WIDTH)
    fig.add_trace(
        go.Scatter(
            x=time, y=cum_rain, mode="lines",
            line=dict(color=CUM_COLOR, width=3.0),
            name=f"cumulative rain  [{vol_units}]",
            hovertemplate="t=%{x:.4f} day<br>cum rain=%{y:.4f} " + vol_units + "<extra></extra>",
        ),
        row=6, col=1,
    )
    # stacked partition: infiltration -> + ponding -> + overland outflow == cum_rain.
    s1 = infil
    s2 = infil + surf_w
    s3 = infil + surf_w + cum_outflow
    fig.add_trace(
        go.Scatter(
            x=time, y=s1, mode="lines",
            line=dict(color=INFIL_COLOR, width=0.5),
            fill="tozeroy", fillcolor="rgba(46,125,50,0.35)",
            name="infiltration (&Delta; subsurface storage)",
            hovertemplate="t=%{x:.4f} day<br>infiltrated=%{y:.4f} " + vol_units + "<extra></extra>",
        ),
        row=6, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=s2, mode="lines",
            line=dict(color=POND_COLOR, width=0.5),
            fill="tonexty", fillcolor="rgba(0,131,143,0.35)",
            name="+ ponding (surface water)",
            customdata=surf_w,
            hovertemplate="t=%{x:.4f} day<br>ponded=%{customdata:.5f} " + vol_units + "<extra></extra>",
        ),
        row=6, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=s3, mode="lines",
            line=dict(color=OUT_FILL, width=0.5),
            fill="tonexty", fillcolor="rgba(2,119,189,0.32)",
            name="+ overland outflow (cum)",
            customdata=cum_outflow,
            hovertemplate="t=%{x:.4f} day<br>cum overland=%{customdata:.4f} " + vol_units +
                          "<br>partition sum=%{y:.4f} " + vol_units + "<extra></extra>",
        ),
        row=6, col=1,
    )

    # =============================================================== row7: diagnostics
    mbe_plot = np.where(np.abs(mbe) <= 0, np.nan, np.abs(mbe))  # guard zeros for log axis
    fig.add_trace(
        go.Scatter(
            x=time, y=mbe_plot, mode="lines+markers",
            line=dict(color=MBE_COLOR, width=2),
            marker=dict(size=4, color=MBE_COLOR),
            name="|mass-balance error|",
            hovertemplate="t=%{x:.4f} day<br>err=%{y:.3e}<extra></extra>",
        ),
        row=7, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=niter, mode="lines+markers",
            line=dict(color=ITER_COLOR, width=1.8),
            marker=dict(size=4, color=ITER_COLOR),
            name="Newton iters",
            hovertemplate="t=%{x:.4f} day<br>iters=%{y:.0f}<extra></extra>",
        ),
        row=7, col=3,
    )

    # =============================================================== animation frames
    frames = []
    for k in range(ntime):
        fdata = [
            go.Heatmap(z=psi_xs[k]),                  # i_psi
            go.Scatter(x=xc, y=wt_z[k]),              # i_wt (water table)
            go.Heatmap(z=th_xs[k]),                   # i_theta
            go.Heatmap(z=d_map_mm[k]),                # i_dmap (mm display)
            go.Scatter(x=[time[k]], y=[outflow[k]]),  # i_qmark
        ]
        ftraces = [i_psi, i_wt, i_theta, i_dmap, i_qmark]
        for j, (zidx, _lname, _zval) in enumerate(layer_specs):
            fdata.append(go.Heatmap(z=theta_xy[k, zidx]))
            ftraces.append(i_thxy[j])
        frames.append(go.Frame(name=str(k), data=fdata, traces=ftraces))
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
        x=0.05, y=-0.04, len=0.9,
        pad=dict(t=30, b=10),
        currentvalue=dict(prefix="time = ", suffix=f"  {t_units}",
                          font=dict(size=14, color="#222")),
        steps=slider_steps,
    )]
    updatemenus = [dict(
        type="buttons", direction="left",
        x=0.05, y=1.045, xanchor="left", yanchor="top",
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
    STANDOFF = 16

    # row1: SOIL STRUCTURE Ks(z) -- LOG x, z surface at top (so do NOT reverse z;
    # z already runs base..surface, and we want surface at top -> y range high..?).
    # z increases base(0)..surface(1); "surface at top" is the default (larger z up).
    fig.update_xaxes(title_text=f"Ks  [{ks_units}]  (log)", title_standoff=STANDOFF,
                     type="log", exponentformat="power", row=1, col=1)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]  (surface at top)",
                     title_standoff=STANDOFF,
                     range=[float(z.min()), float(z.max())], row=1, col=1)

    # row2: psi cross-section (FULL WIDTH; xc horizontal, z up -> surface on top).
    fig.update_xaxes(title_text=f"x downslope  [{x_units}]  (outlet x={_fmt(L)} m at right)",
                     title_standoff=STANDOFF,
                     range=[float(xc.min()), float(xc.max())], row=2, col=1)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]  (surface at top)", title_standoff=STANDOFF,
                     range=[float(z.min()), float(z.max())], row=2, col=1)
    # row3: theta cross-section (FULL WIDTH).
    fig.update_xaxes(title_text=f"x downslope  [{x_units}]", title_standoff=STANDOFF,
                     range=[float(xc.min()), float(xc.max())], row=3, col=1)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]  (surface at top)", title_standoff=STANDOFF,
                     range=[float(z.min()), float(z.max())], row=3, col=1)

    # row4 col1: surface ponding map.
    fig.update_xaxes(title_text=f"x downslope  [{x_units}]  (outlet x={_fmt(L)} m)",
                     title_standoff=STANDOFF,
                     range=[float(x.min()), float(x.max())], constrain="domain", row=4, col=1)
    fig.update_yaxes(title_text=f"y cross-slope  [{y_units}]", title_standoff=STANDOFF,
                     range=[float(y.min()), float(y.max())], row=4, col=1)
    # row4 col3: hydrograph.
    fig.update_xaxes(title_text=f"time  [{t_units}]", title_standoff=STANDOFF,
                     range=[float(time.min()), float(time.max())], row=4, col=3)
    fig.update_yaxes(title_text=f"discharge  [{q_units}]", title_standoff=STANDOFF, row=4, col=3,
                     secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text=f"rain [{r_units}] (inv)", title_standoff=STANDOFF,
                     range=[r_axis_top, 0.0], row=4, col=3, secondary_y=True, showgrid=False)

    # row5: the 4 top-down theta(x,y) layer maps (short axis titles -> no collisions).
    for col_i in range(1, len(layer_specs) + 1):
        fig.update_xaxes(title_text=f"x  [{x_units}]", title_standoff=STANDOFF,
                         range=[float(x.min()), float(x.max())],
                         constrain="domain", row=5, col=col_i)
        fig.update_yaxes(title_text=(f"y  [{y_units}]" if col_i == 1 else ""),
                         title_standoff=STANDOFF,
                         range=[float(y.min()), float(y.max())], row=5, col=col_i)

    # row6: partition (FULL WIDTH).
    fig.update_xaxes(title_text=f"time  [{t_units}]", title_standoff=STANDOFF,
                     range=[float(time.min()), float(time.max())], row=6, col=1)
    fig.update_yaxes(title_text=f"cumulative water  [{vol_units}]", title_standoff=STANDOFF,
                     row=6, col=1, rangemode="tozero")
    # row7: diagnostics.
    fig.update_xaxes(title_text=f"time  [{t_units}]", title_standoff=STANDOFF, row=7, col=1)
    fig.update_yaxes(title_text="|MB err|", type="log", title_standoff=STANDOFF,
                     exponentformat="power", row=7, col=1)
    fig.update_xaxes(title_text=f"time  [{t_units}]", title_standoff=STANDOFF, row=7, col=3)
    fig.update_yaxes(title_text="Newton iters", title_standoff=STANDOFF, row=7, col=3, rangemode="tozero")

    # =============================================================== outlet-edge markers
    fig.add_vline(x=L, line=dict(color="#c62828", width=2, dash="dash"), row=4, col=1)
    fig.add_vline(x=float(xc.max()), line=dict(color="#c62828", width=1.5, dash="dot"), row=2, col=1)
    fig.add_vline(x=float(xc.max()), line=dict(color="#c62828", width=1.5, dash="dot"), row=3, col=1)

    # ponding-panel note: the thin overland sheet (storm peak ~mm) routes to the outlet.
    pond_note = (f"storm-peak d = {_fmt(d_peak * 1e3)} mm: thin overland sheet<br>"
                 f"routing to the x={_fmt(L)} m outlet (fixed 0&ndash;{POND_ZMAX_MM:g} mm scale)")
    fig.add_annotation(
        x=float(x.min()), y=float(y.max()),
        xref="x4", yref="y4",
        xanchor="left", yanchor="bottom",
        align="left", showarrow=False,
        text=pond_note,
        font=dict(size=9, color="#01579b"),
        bgcolor="rgba(225,245,254,0.94)", bordercolor="#0277bd", borderwidth=1, borderpad=3,
    )

    # =============================================================== metrics panel
    metrics_lines = [
        f"<b>module</b>: {module}",
        f"<b>status</b>: {status} &nbsp; <b>date</b>: {date}",
        f"<b>scenario</b>: {scenario}",
        f"<b>domain</b> = {domain}",
        f"<b>mesh</b> = {mesh}",
        f"<b>y-section</b> = {_fmt(y_section)} m",
        (f"<b>wall clock</b> = {_fmt(wall_clock)} s &nbsp; <b>steps</b> = {_fmt(steps)} &nbsp; "
         f"<b>sec/step</b> = {_fmt(sec_per_step)}"),
        f"<b>cum rain</b> = {_fmt(cum_rain_tot)} {vol_units}",
        f"<b>cum overland outflow</b> = {_fmt(cum_outflow_tot)} {vol_units}",
        f"<b>runoff fraction</b> = {_fmt(runoff_frac)}",
        f"<b>final subsurface water</b> = {_fmt(soil_w_final)} {vol_units}",
        f"<b>peak surface depth</b> = {_fmt(peak_d_attr)} m (final-frame)",
        f"<b>storm-peak ponding</b> = {_fmt(d_peak * 1e3)} mm",
        f"<b>SAND barrier</b> @ z&asymp;{_fmt(z_sand)} m, Ks&asymp;{_fmt(ks_sand)} {ks_units}",
        f"<b>MAX mass-balance error</b> = {_fmt(float(mbe_max))}",
        f"<b>clip mass adjust</b> = {_fmt(clip_mass)} {vol_units}",
        f"<b>ponding map scale</b> = fixed 0&ndash;{POND_ZMAX_MM:g} mm",
    ]
    metrics_html = "<br>".join(metrics_lines)
    fig.add_annotation(
        x=1.005, y=1.0, xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        align="left", showarrow=False,
        text="<b>METRICS</b><br>" + metrics_html,
        font=dict(size=10.5, color="#222", family="Consolas, monospace"),
        bordercolor="#888", borderwidth=1, borderpad=8,
        bgcolor="rgba(245,245,245,0.95)",
    )

    title = ("field-scale feasibility &middot; 2-ha heterogeneous layered hillslope "
             "&middot; 2026-06-08")
    fig.update_layout(
        title=dict(text="<b>PIDS Pillar-2 sanity (Tier-3): field-scale feasibility "
                        "(heterogeneous layered 3-D hillslope)</b><br>"
                        f"<span style='font-size:13px'>{title}</span>",
                   x=0.5, xanchor="center", y=0.994, yanchor="top"),
        sliders=sliders,
        updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.012,
                    xanchor="right", x=1.0, font=dict(size=10)),
        margin=dict(l=90, r=400, t=180, b=90),
        width=1480, height=2240,
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


# backward-compatible alias matching the other generators' naming.
build_feasibility_html = build_html


if __name__ == "__main__":
    if len(sys.argv) == 3:
        import os
        out = build_html(sys.argv[1], sys.argv[2])
        print(f"WROTE {out}  ({os.path.getsize(out) / 1e6:.2f} MB)")
    else:
        print("usage: python make_feasibility_html.py <result.nc> <out.html>", file=sys.stderr)
        sys.exit(2)
