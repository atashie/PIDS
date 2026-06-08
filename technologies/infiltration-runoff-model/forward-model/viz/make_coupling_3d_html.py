"""Reusable Tier-3 sanity visualizer for the PIDS Pillar-2 forward model
(Module 3: surface<->subsurface COUPLING) in **3-D**.

Turns a standardized 3-D coupling sanity-run NetCDF result (the data contract of
``governance/visualize-sanity-check-routine.md``) into ONE self-contained,
offline, interactive HTML. The generator reads ONLY the result file -- it never
imports or runs the solver (this independence is the whole point of the Tier-3 lens).

For a storm + recession on a sloping 3-D hillslope block (x downslope with the
outlet at x=L, y cross-slope, z up with the surface at the top) the coupled 3-D
views are arranged so the WIDE cross-section heatmaps each get their own
FULL-WIDTH row (stacked vertically). This is the layout fix: side-by-side wide
heatmaps used to collide their y-axis titles with the neighbouring subplot
title. Full-width stacking gives every axis title room to breathe.

  1. SURFACE PONDING MAP d(x,y) [m] (row 1, left) with a TIME SLIDER: heatmap on
     the top face, animated so ponding builds during the storm and concentrates
     toward the downstream outlet edge x=L. The outlet edge x=L is marked. The
     colour range is FIXED at [0, 1 mm] for BOTH scenarios so they are directly
     comparable: on sand all rain infiltrates (peak d ~3e-7 m, sub-micron) so the
     panel reads as a flat ~0 floor (annotated as the NCP smoothing floor =
     effectively NO ponding); on loam the real sub-mm overland sheet shows.
     OUTLET HYDROGRAPHS (row 1, right): BOTH the surface OVERLAND discharge
     (``outflow``) and the LATERAL GROUNDWATER outflow (``drainage``) vs time,
     with the rain hyetograph inverted on a twin axis. The contrast is the point
     (sand: groundwater line dominates; loam: overland line dominates).
  2. THREE TOP-DOWN theta(x,y) LAYER MAPS (row 2, three columns): animated
     heatmaps of water content theta(x,y) at three z-elevations
     (``zlayer`` = surface / medial / bottom), all sharing ONE fixed theta colour
     range (theta_r..theta_s ~ 0.045..0.43) so saturation is comparable across
     layers AND across scenarios. These reveal the lateral moisture / water-table
     structure at depth: on sand the bottom layer is saturated and the medial
     layer saturates as the water table rises; the downslope tilt shows the
     lateral groundwater shape.
  3. psi CROSS-SECTION head_xsec(xc, z) [m] (row 3, FULL WIDTH), animated, with
     the SURFACE (z=H) at the TOP, and the WATER TABLE drawn as the psi=0 contour
     so the rising / tilting water table -- i.e. lateral groundwater flow toward
     x=L -- is visible.
  4. theta CROSS-SECTION theta_xsec(xc, z) [-] (row 4, FULL WIDTH), animated.
  5. PARTITION / CONSERVATION (row 5, FULL WIDTH): where the rain goes --
     cum_rain vs cumulative infiltration (soil_water - soil_water[0]) vs ponding
     (surface_water) vs cum OVERLAND outflow (cum_outflow) vs cum LATERAL
     GROUNDWATER (cum_drainage); they sum to cum_rain.
  6. DIAGNOSTICS (row 6): mass-balance error (log-y) + Newton iterations.
  7. METRICS panel (text from attrs): the deciding numbers.

Usage:
    python viz/make_coupling_3d_html.py <result.nc> <out.html>

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

    For each xc column we walk up in z and linearly interpolate the elevation
    where psi crosses 0 (the highest such crossing -- the top of the saturated
    zone). Columns with no crossing (fully unsaturated or fully saturated within
    the slab) return NaN so the contour line simply breaks there.

    Returns an array shaped (xc,) of water-table elevations [same units as z].
    """
    nz, nxc = psi_zx.shape
    wt = np.full(nxc, np.nan, dtype=float)
    for j in range(nxc):
        col = psi_zx[:, j]
        # scan from the surface (top) downward for the first sign change to >=0.
        found = np.nan
        for i in range(nz - 1, 0, -1):
            a, b = col[i - 1], col[i]      # a is lower z, b is upper z
            if np.isnan(a) or np.isnan(b):
                continue
            if (a >= 0.0) and (b < 0.0):
                # crossing between z[i-1] (sat) and z[i] (unsat): interpolate psi=0.
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
            # no interior crossing: if the whole column is saturated, table is at/above
            # the top; if entirely unsaturated, leave NaN (no table in slab).
            if np.all(col >= 0.0):
                found = z[-1]
        wt[j] = found
    return wt


# ----------------------------------------------------------------------------- builder
def build_html(nc_path: str, out_path: str) -> str:
    """Build the self-contained interactive 3-D coupling sanity HTML.

    Parameters
    ----------
    nc_path : str
        Path to the standardized 3-D coupling sanity-run NetCDF result file.
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
    z = np.asarray(ds["z"].values, dtype=float)          # (z,)  elevation [m], 0=base, H=surface
    time = np.asarray(ds["time"].values, dtype=float)    # (time,) [day]

    # --- 3-D / 2-D fields ---
    d_map = np.asarray(ds["surface_depth_map"].values, dtype=float)   # (time, y, x) [m]
    psi_xs = np.asarray(ds["head_xsec"].values, dtype=float)          # (time, z, xc) [m]
    th_xs = np.asarray(ds["theta_xsec"].values, dtype=float)          # (time, z, xc) [-]

    # --- NEW: top-down theta(x,y) at three z-layers (surface / medial / bottom) ---
    has_theta_xy = ("theta_xy" in ds) and ("zlayer" in ds.coords)
    if has_theta_xy:
        theta_xy = np.asarray(ds["theta_xy"].values, dtype=float)     # (time, zlayer, y, x) [-]
        zlayer = np.asarray(ds["zlayer"].values, dtype=float)         # (zlayer,) elevations [m]
        zlayer_units = _units(ds["zlayer"], "m")
        thxy_units = _units(ds["theta_xy"], "-")
    else:
        theta_xy = None
        zlayer = np.array([], dtype=float)
        zlayer_units = "m"
        thxy_units = "-"

    # --- time series ---
    rain = np.asarray(ds["rainfall"].values, dtype=float)            # (time,) [m/day]
    soil_w = np.asarray(ds["soil_water"].values, dtype=float)        # (time,) [m^3]
    surf_w = np.asarray(ds["surface_water"].values, dtype=float)     # (time,) [m^3]
    outflow = np.asarray(ds["outflow"].values, dtype=float)         # (time,) [m^3/day]  SURFACE OVERLAND
    cum_outflow = np.asarray(ds["cum_outflow"].values, dtype=float)  # (time,) [m^3]      cum overland
    drainage = np.asarray(ds["drainage"].values, dtype=float)       # (time,) [m^3/day]  LATERAL GROUNDWATER
    cum_drainage = np.asarray(ds["cum_drainage"].values, dtype=float)  # (time,) [m^3]   cum lateral gw
    cum_rain = np.asarray(ds["cum_rain"].values, dtype=float)        # (time,) [m^3]
    mbe = np.asarray(ds["mass_balance_error"].values, dtype=float)    # (time,) [-]
    niter = np.asarray(ds["newton_iters"].values, dtype=float)      # (time,) [-]

    x_units = _units(ds["x"], "m")
    y_units = _units(ds["y"], "m")
    z_units = _units(ds["z"], "m")
    t_units = _units(ds["time"], "day")
    d_units = _units(ds["surface_depth_map"], "m")
    psi_units = _units(ds["head_xsec"], "m")
    th_units = _units(ds["theta_xsec"], "-")
    r_units = _units(ds["rainfall"], "m/day")
    q_units = _units(ds["outflow"], "m^3/day")
    vol_units = _units(ds["cum_rain"], "m^3")

    # cumulative infiltration = change in subsurface storage relative to t=0 [m^3].
    infil = soil_w - soil_w[0]

    ntime = time.shape[0]
    L = float(x.max())  # outlet edge (downslope)
    H = float(z.max())  # surface elevation

    # --- water-table (psi=0) iso-line per frame, for the cross-section overlay ---
    wt_z = np.vstack([_watertable_z(psi_xs[k], z) for k in range(ntime)])  # (time, xc)

    # --- global attrs / metrics (new / renamed contract attrs) ---
    module = _attr(ds, "module", "coupling_3d")
    scenario = _attr(ds, "scenario", "storm + recession")
    date = _attr(ds, "date", "")
    soil = _attr(ds, "soil", "")
    domain = _attr(ds, "domain_LxWxH_m")
    mesh = _attr(ds, "mesh_nx_ny_nz")
    bed_slope = _attr(ds, "bed_slope")
    y_section = _attr(ds, "y_section_m")
    Ks = _attr(ds, "Ks_m_per_day")
    water_table_z0 = _attr(ds, "water_table_z0_m")
    gw_external_head = _attr(ds, "gw_external_head_m")
    rain_rate = _attr(ds, "rain_rate_m_per_day")
    storm_dur = _attr(ds, "storm_duration_day")
    cum_rain_total = _attr(ds, "cum_rain_total_m3")
    peak_d = _attr(ds, "peak_surface_depth_m")
    cum_overland_tot = _attr(ds, "cum_overland_outflow_m3", _attr(ds, "cum_outflow_m3"))
    cum_lateral_gw_tot = _attr(ds, "cum_lateral_gw_m3", _attr(ds, "cum_drainage_m3"))
    final_sub_change = _attr(ds, "final_subsurface_change_m3", _attr(ds, "final_infiltrated_m3"))
    mbe_max = _attr(ds, "mass_balance_error_max", float(np.max(np.abs(mbe))))
    max_iters = _attr(ds, "max_newton_iters", int(np.max(niter)))

    # --- fixed color ranges so features MOVE / GROW rather than rescale per frame ---
    psi_lo = float(np.nanmin(psi_xs)); psi_hi = float(np.nanmax(psi_xs))
    th_lo = float(np.nanmin(th_xs)); th_hi = float(np.nanmax(th_xs))

    # PONDING colour range (reviewer fix #2): a FIXED, physically-meaningful range
    # anchored at 0 and identical for both scenarios so they are comparable. The
    # smoothed-NCP floor on sand sits at peak d ~3e-7 m (sub-micron) = effectively
    # NO ponding; auto-scaling to that range would stretch the floor into fake
    # ponding. 1 mm full-scale lets sand read as a uniform ~0 floor while loam's
    # real sub-mm sheet (peak ~0.7 mm) still shows. Data is NOT altered.
    POND_ZMAX_M = 1.0e-3          # 1 mm full-scale, shared by both scenarios
    POND_ZMAX_MM = POND_ZMAX_M * 1e3
    d_map_mm = d_map * 1e3        # display in mm (data untouched; this is display-only)
    d_peak = float(np.nanmax(d_map))                     # this scenario's true peak [m]
    NCP_FLOOR = 1.0e-6           # ~1 micron: below this, ponding is the smoothing floor
    pond_is_floor = d_peak < NCP_FLOOR                   # True on sand (all rain infiltrates)

    # THETA(x,y) LAYER colour range (reviewer fix #3): ONE fixed theta_r..theta_s
    # range shared by all three layers AND both scenarios so saturation is
    # comparable everywhere. Use the contract-stated residual/saturated bounds.
    THXY_LO, THXY_HI = 0.045, 0.43
    if has_theta_xy:
        # clamp to data extremes only if they fall outside the nominal band.
        THXY_LO = min(THXY_LO, float(np.nanmin(theta_xy)))
        THXY_HI = max(THXY_HI, float(np.nanmax(theta_xy)))
    # label each layer surface/medial/bottom by descending elevation.
    if has_theta_xy:
        order = np.argsort(-zlayer)                       # surface (high z) first
        layer_names = ["surface", "medial", "bottom"]
        # if there are not exactly 3 layers, fall back to generic names.
        if zlayer.shape[0] != 3:
            layer_names = [f"layer {i}" for i in range(zlayer.shape[0])]
        layer_specs = [(int(idx), layer_names[i],
                        (0.0 if float(zlayer[idx]) == 0.0 else float(zlayer[idx])))  # drop -0.0
                       for i, idx in enumerate(order)]
    else:
        layer_specs = []

    # =============================================================== figure layout
    #   row1 col1     : surface ponding map d(x,y)   [animated heatmap, FIXED 0..1mm range]
    #   row1 col2-3   : outlet hydrographs (overland + lateral gw) + rain inverted (twin axis)
    #   row2 col1/2/3 : theta(x,y) TOP-DOWN maps at surface / medial / bottom z-layers
    #                   [animated heatmaps, ONE shared fixed theta range]   <-- NEW VIEW
    #   row3 (colspan): psi cross-section psi(xc,z)  [animated heatmap, surface at top]
    #                   + psi=0 water-table contour line overlay (animated)
    #   row4 (colspan): theta cross-section th(xc,z) [animated heatmap, surface at top]
    #   row5 (colspan): partition -- cum_rain = infil + ponding + overland + lateral gw
    #   row6 col1     : mass-balance error (log-y)   row6 col2-3 : Newton iterations
    #
    # LAYOUT: the two WIDE cross-section heatmaps each own a FULL-WIDTH row so their
    # y-axis titles cannot collide with a neighbour. The three top-down theta-layer
    # maps share row 2 in three columns with GENEROUS horizontal spacing so each
    # short per-layer title stays readable. The grid is 3 columns throughout;
    # full-width rows use colspan=3 and the hydrograph / Newton panels colspan=2.
    H_SP = 0.085   # generous horizontal spacing between the 3 theta-layer columns
    fig = make_subplots(
        rows=6,
        cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy", "secondary_y": True, "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 3}, None, None],
            [{"type": "xy", "colspan": 3}, None, None],
            [{"type": "xy", "colspan": 3}, None, None],
            [{"type": "xy"}, {"type": "xy", "colspan": 2}, None],
        ],
        row_heights=[0.175, 0.135, 0.155, 0.155, 0.175, 0.155],
        vertical_spacing=0.072,
        horizontal_spacing=H_SP,
        subplot_titles=(
            # row1
            f"Surface ponding d(x,y) [mm]  &mdash; fixed 0&ndash;{POND_ZMAX_MM:g} mm scale",
            "Outlet hydrographs: overland Q vs lateral groundwater (storm inverted)",
            # row2: three theta-layer titles (filled below; placeholders here keep indices aligned)
            (f"&theta;(x,y) {layer_specs[0][1]}  z={_fmt(layer_specs[0][2])} {zlayer_units}"
             if layer_specs else "&theta;(x,y) surface"),
            (f"&theta;(x,y) {layer_specs[1][1]}  z={_fmt(layer_specs[1][2])} {zlayer_units}"
             if len(layer_specs) > 1 else "&theta;(x,y) medial"),
            (f"&theta;(x,y) {layer_specs[2][1]}  z={_fmt(layer_specs[2][2])} {zlayer_units}"
             if len(layer_specs) > 2 else "&theta;(x,y) bottom"),
            # row3
            f"Pressure head &psi;(x,z)  [{psi_units}]  on y={_fmt(y_section)} m  "
            f"(surface at top; black line = water table, &psi;=0)",
            # row4
            f"Water content &theta;(x,z)  [-]  on y={_fmt(y_section)} m  (surface at top)",
            # row5
            "Partition: cum rain = infiltration + ponding + overland outflow + lateral groundwater  [m&sup3;]",
            # row6
            "Diagnostics: mass-balance error (log)",
            "Diagnostics: Newton iterations",
        ),
    )

    Q_COLOR = "#0277bd"        # surface overland outflow
    DRAIN_COLOR = "#6a1b9a"    # lateral groundwater
    RAIN_COLOR = "#90a4ae"
    RAIN_BORDER = "#546e7a"
    CUM_COLOR = "#37474f"
    INFIL_COLOR = "#2e7d32"
    POND_COLOR = "#00838f"
    OUT_FILL = "#0277bd"
    DRAIN_FILL = "#6a1b9a"
    MBE_COLOR = "#e67e22"
    ITER_COLOR = "#6a1b9a"
    WT_COLOR = "#111111"       # water-table iso-line (high contrast on RdBu)

    # =============================================================== row1 col1: surface ponding map d(x,y)
    # FIXED range [0, POND_ZMAX_MM] in mm, shared by both scenarios (reviewer fix #2).
    i_dmap = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            x=x, y=y, z=d_map_mm[0],
            zmin=0.0, zmax=POND_ZMAX_MM, zauto=False,
            colorscale="Blues",
            colorbar=dict(title=dict(text="d [mm]", side="right"),
                          len=0.135, y=0.915, x=0.255, thickness=12),
            hovertemplate="x=%{x:.3f} m<br>y=%{y:.3f} m<br>d=%{z:.3e} mm<extra></extra>",
            name="surface depth",
        ),
        row=1, col=1,
    )

    # =============================================================== row1 col2: outlet hydrographs (overland + lateral gw)
    fig.add_trace(
        go.Scatter(
            x=time, y=outflow, mode="lines",
            line=dict(color=Q_COLOR, width=2.8),
            name=f"surface overland Q  [{q_units}]",
            legendgroup="hydro",
            hovertemplate="t=%{x:.4f} day<br>overland Q=%{y:.4e} " + q_units + "<extra></extra>",
        ),
        row=1, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=drainage, mode="lines",
            line=dict(color=DRAIN_COLOR, width=2.8, dash="dash"),
            name=f"lateral groundwater  [{q_units}]",
            legendgroup="hydro",
            hovertemplate="t=%{x:.4f} day<br>lateral gw=%{y:.4e} " + q_units + "<extra></extra>",
        ),
        row=1, col=2, secondary_y=False,
    )
    # hyetograph: inverted bars hanging from the top of the rain (secondary) axis.
    r_hi = float(np.nanmax(rain))
    r_top = (r_hi if r_hi > 0 else 1.0) * 1.05
    r_axis_top = r_top * 3.0  # rain occupies only the top third (clear of the Q lines)
    fig.add_trace(
        go.Bar(
            x=time, y=rain, base=r_axis_top - rain,
            marker=dict(color=RAIN_COLOR, line=dict(color=RAIN_BORDER, width=0.5)),
            opacity=0.8,
            width=(time[1] - time[0]) * 0.9 if ntime > 1 else 0.005,
            name=f"rainfall  [{r_units}]",
            hovertemplate="t=%{x:.4f} day<br>r=%{base:.3f} " + r_units + "<extra></extra>",
        ),
        row=1, col=2, secondary_y=True,
    )
    # current-time markers on BOTH hydrograph lines (track slider).
    i_qmark = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[outflow[0]], mode="markers",
            marker=dict(size=11, color=Q_COLOR, symbol="circle-open",
                        line=dict(width=3, color=Q_COLOR)),
            name="current (overland)", showlegend=False,
            hovertemplate="t=%{x:.4f} day<br>overland Q=%{y:.4e} " + q_units + "<extra>current</extra>",
        ),
        row=1, col=2, secondary_y=False,
    )
    i_dmark = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[time[0]], y=[drainage[0]], mode="markers",
            marker=dict(size=11, color=DRAIN_COLOR, symbol="circle-open",
                        line=dict(width=3, color=DRAIN_COLOR)),
            name="current (lateral gw)", showlegend=False,
            hovertemplate="t=%{x:.4f} day<br>lateral gw=%{y:.4e} " + q_units + "<extra>current</extra>",
        ),
        row=1, col=2, secondary_y=False,
    )

    # =============================================================== row2: theta(x,y) top-down layer maps (NEW)
    # Three animated heatmaps of water content theta(x,y) at the surface / medial /
    # bottom z-layers, sharing ONE fixed theta colour range so saturation is
    # comparable across layers and scenarios. Only the rightmost shows a colorbar.
    i_thxy = []  # trace indices, in plotted column order (surface, medial, bottom)
    if has_theta_xy:
        for col_i, (zidx, lname, zval) in enumerate(layer_specs, start=1):
            show_cbar = (col_i == len(layer_specs))
            cbar = (dict(title=dict(text=f"&theta; [{thxy_units}]", side="right"),
                         len=0.115, y=0.74, x=1.0, thickness=12)
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
                row=2, col=col_i,
            )

    # =============================================================== row3: psi cross-section (FULL WIDTH) + water table
    i_psi = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            x=xc, y=z, z=psi_xs[0],
            zmin=psi_lo, zmax=psi_hi,
            colorscale="RdBu",
            colorbar=dict(title=dict(text=f"&psi; [{psi_units}]", side="right"),
                          len=0.135, y=0.555, x=1.0, thickness=12),
            hovertemplate="x=%{x:.3f} m<br>z=%{y:.3f} m<br>ψ=%{z:.4f} m<extra></extra>",
            name="ψ x-section",
        ),
        row=3, col=1,
    )
    # WATER TABLE: psi=0 iso-line overlaid as a thick scatter line (animated per frame).
    i_wt = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=xc, y=wt_z[0], mode="lines",
            line=dict(color=WT_COLOR, width=3.0),
            connectgaps=False,
            name="water table (ψ=0)",
            legendgroup="wt",
            hovertemplate="x=%{x:.3f} m<br>water table z=%{y:.4f} m<extra>ψ=0</extra>",
        ),
        row=3, col=1,
    )

    # =============================================================== row4: theta cross-section (FULL WIDTH)
    i_theta = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            x=xc, y=z, z=th_xs[0],
            zmin=th_lo, zmax=th_hi,
            colorscale="YlGnBu",
            colorbar=dict(title=dict(text="&theta; [-]", side="right"),
                          len=0.135, y=0.385, x=1.0, thickness=12),
            hovertemplate="x=%{x:.3f} m<br>z=%{y:.3f} m<br>θ=%{z:.4f}<extra></extra>",
            name="θ x-section",
        ),
        row=4, col=1,
    )

    # =============================================================== row5: partition (conservation, FULL WIDTH)
    fig.add_trace(
        go.Scatter(
            x=time, y=cum_rain, mode="lines",
            line=dict(color=CUM_COLOR, width=3.0),
            name=f"cumulative rain  [{vol_units}]",
            hovertemplate="t=%{x:.4f} day<br>cum rain=%{y:.5f} " + vol_units + "<extra></extra>",
        ),
        row=5, col=1,
    )
    # stacked partition: infiltration -> + ponding -> + overland -> + lateral gw == cum_rain.
    s1 = infil
    s2 = infil + surf_w
    s3 = infil + surf_w + cum_outflow
    s4 = infil + surf_w + cum_outflow + cum_drainage
    fig.add_trace(
        go.Scatter(
            x=time, y=s1, mode="lines",
            line=dict(color=INFIL_COLOR, width=0.5),
            fill="tozeroy", fillcolor="rgba(46,125,50,0.35)",
            name="infiltration (Δ subsurface storage)",
            hovertemplate="t=%{x:.4f} day<br>infiltrated=%{y:.5f} " + vol_units + "<extra></extra>",
        ),
        row=5, col=1,
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
        row=5, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=s3, mode="lines",
            line=dict(color=OUT_FILL, width=0.5),
            fill="tonexty", fillcolor="rgba(2,119,189,0.32)",
            name="+ overland outflow (cum)",
            customdata=cum_outflow,
            hovertemplate="t=%{x:.4f} day<br>cum overland=%{customdata:.5f} " + vol_units + "<extra></extra>",
        ),
        row=5, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=s4, mode="lines",
            line=dict(color=DRAIN_FILL, width=0.5),
            fill="tonexty", fillcolor="rgba(106,27,154,0.32)",
            name="+ lateral groundwater (cum)",
            customdata=cum_drainage,
            hovertemplate="t=%{x:.4f} day<br>cum lateral gw=%{customdata:.6f} " + vol_units +
                          "<br>partition sum=%{y:.5f} " + vol_units + "<extra></extra>",
        ),
        row=5, col=1,
    )

    # =============================================================== row6: diagnostics
    mbe_plot = np.where(np.abs(mbe) <= 0, np.nan, np.abs(mbe))  # guard zeros for log axis
    fig.add_trace(
        go.Scatter(
            x=time, y=mbe_plot, mode="lines+markers",
            line=dict(color=MBE_COLOR, width=2),
            marker=dict(size=4, color=MBE_COLOR),
            name="|mass-balance error|",
            hovertemplate="t=%{x:.4f} day<br>err=%{y:.3e}<extra></extra>",
        ),
        row=6, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=niter, mode="lines+markers",
            line=dict(color=ITER_COLOR, width=1.8),
            marker=dict(size=4, color=ITER_COLOR),
            name="Newton iters",
            hovertemplate="t=%{x:.4f} day<br>iters=%{y:.0f}<extra></extra>",
        ),
        row=6, col=2,
    )

    # =============================================================== animation frames
    frames = []
    for k in range(ntime):
        fdata = [
            go.Heatmap(z=d_map_mm[k]),                    # i_dmap (mm display)
            go.Scatter(x=[time[k]], y=[outflow[k]]),      # i_qmark
            go.Scatter(x=[time[k]], y=[drainage[k]]),     # i_dmark
        ]
        ftraces = [i_dmap, i_qmark, i_dmark]
        # three theta(x,y) top-down layer maps (in plotted column order).
        for j, (zidx, _lname, _zval) in enumerate(layer_specs):
            fdata.append(go.Heatmap(z=theta_xy[k, zidx]))
            ftraces.append(i_thxy[j])
        fdata += [
            go.Heatmap(z=psi_xs[k]),                      # i_psi
            go.Scatter(x=xc, y=wt_z[k]),                  # i_wt (water table)
            go.Heatmap(z=th_xs[k]),                       # i_theta
        ]
        ftraces += [i_psi, i_wt, i_theta]
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
        x=0.05, y=-0.045, len=0.9,
        pad=dict(t=30, b=10),
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
    STANDOFF = 18  # px gap between axis title and tick labels -> keeps titles off neighbours

    # row1 col1: surface ponding map (x downslope horizontal, y cross-slope vertical).
    fig.update_xaxes(title_text=f"x downslope  [{x_units}]  (outlet x=L)",
                     title_standoff=STANDOFF,
                     range=[float(x.min()), float(x.max())], constrain="domain", row=1, col=1)
    fig.update_yaxes(title_text=f"y cross-slope  [{y_units}]", title_standoff=STANDOFF,
                     range=[float(y.min()), float(y.max())], row=1, col=1)
    # row1 col2: hydrographs (overland + lateral gw on left axis) + rain (right, inverted).
    fig.update_xaxes(title_text=f"time  [{t_units}]", title_standoff=STANDOFF,
                     range=[float(time.min()), float(time.max())], row=1, col=2)
    fig.update_yaxes(title_text=f"discharge  [{q_units}]", title_standoff=STANDOFF, row=1, col=2,
                     secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text=f"rain [{r_units}] (inv)", title_standoff=STANDOFF,
                     range=[r_axis_top, 0.0], row=1, col=2, secondary_y=True, showgrid=False)

    # row2: the three top-down theta(x,y) layer maps (short axis titles -> no collisions).
    if has_theta_xy:
        for col_i in range(1, len(layer_specs) + 1):
            fig.update_xaxes(title_text=f"x  [{x_units}]", title_standoff=STANDOFF,
                             range=[float(x.min()), float(x.max())],
                             constrain="domain", row=2, col=col_i)
            fig.update_yaxes(title_text=(f"y  [{y_units}]" if col_i == 1 else ""),
                             title_standoff=STANDOFF,
                             range=[float(y.min()), float(y.max())], row=2, col=col_i)

    # row3: psi cross-section (FULL WIDTH; xc horizontal, z up so surface on top).
    fig.update_xaxes(title_text=f"x downslope  [{x_units}]  (outlet x=L at right)",
                     title_standoff=STANDOFF,
                     range=[float(xc.min()), float(xc.max())], row=3, col=1)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]  (surface at top)", title_standoff=STANDOFF,
                     range=[float(z.min()), float(z.max())], row=3, col=1)
    # row4: theta cross-section (FULL WIDTH).
    fig.update_xaxes(title_text=f"x downslope  [{x_units}]", title_standoff=STANDOFF,
                     range=[float(xc.min()), float(xc.max())], row=4, col=1)
    fig.update_yaxes(title_text=f"elevation z  [{z_units}]  (surface at top)", title_standoff=STANDOFF,
                     range=[float(z.min()), float(z.max())], row=4, col=1)
    # row5: partition (FULL WIDTH).
    fig.update_xaxes(title_text=f"time  [{t_units}]", title_standoff=STANDOFF,
                     range=[float(time.min()), float(time.max())], row=5, col=1)
    fig.update_yaxes(title_text=f"cumulative water  [{vol_units}]", title_standoff=STANDOFF,
                     row=5, col=1, rangemode="tozero")
    # row6: diagnostics (the only 2-col row -> short titles + standoff so nothing collides).
    fig.update_xaxes(title_text=f"time  [{t_units}]", title_standoff=STANDOFF, row=6, col=1)
    fig.update_yaxes(title_text="|MB err|", type="log", title_standoff=STANDOFF,
                     exponentformat="power", row=6, col=1)
    fig.update_xaxes(title_text=f"time  [{t_units}]", title_standoff=STANDOFF, row=6, col=2)
    fig.update_yaxes(title_text="Newton iters", title_standoff=STANDOFF, row=6, col=2, rangemode="tozero")

    # =============================================================== outlet-edge marker on the ponding map
    # vertical dashed line at x=L on row1-col1 + at x=L (right edge) on the cross-sections.
    fig.add_vline(x=L, line=dict(color="#c62828", width=2, dash="dash"), row=1, col=1)
    fig.add_vline(x=float(xc.max()), line=dict(color="#c62828", width=1.5, dash="dot"), row=3, col=1)
    fig.add_vline(x=float(xc.max()), line=dict(color="#c62828", width=1.5, dash="dot"), row=4, col=1)

    # NCP-floor annotation on the ponding panel (reviewer fix #2): when all rain
    # infiltrates (sand) the panel is the smoothed-NCP FLOOR, NOT real ponding.
    if pond_is_floor:
        pond_note = (f"peak d = {_fmt(d_peak)} m (sub-micron)<br>"
                     "= NCP smoothing floor &rarr; effectively NO ponding")
        pond_bg = "rgba(255,243,205,0.94)"   # warm = "this is ~zero, read as no ponding"
    else:
        pond_note = (f"peak d = {_fmt(d_peak * 1e3)} mm: real sub-mm overland sheet<br>"
                     f"(shown on fixed 0&ndash;{POND_ZMAX_MM:g} mm scale)")
        pond_bg = "rgba(225,245,254,0.94)"
    fig.add_annotation(
        x=float(x.min()), y=float(y.max()),
        xref="x", yref="y",            # data coords of the row1-col1 ponding panel
        xanchor="left", yanchor="bottom",
        align="left", showarrow=False,
        text=pond_note,
        font=dict(size=9.5, color="#5d4037"),
        bgcolor=pond_bg, bordercolor="#a1887f", borderwidth=1, borderpad=3,
    )

    # =============================================================== metrics panel
    metrics_lines = [
        f"<b>module</b>: {module}",
        f"<b>scenario</b>: {scenario}",
        f"<b>date</b>: {date} &nbsp; <b>soil</b>: {soil}",
        (f"<b>domain LxWxH</b> = {domain} m &nbsp; <b>mesh</b> = {mesh}"),
        (f"<b>bed slope</b> = {_fmt(bed_slope)} &nbsp; <b>y-section</b> = {_fmt(y_section)} m &nbsp; "
         f"<b>Ks</b> = {_fmt(Ks)} m/{t_units}"),
        (f"<b>water table z₀</b> = {_fmt(water_table_z0)} m &nbsp; "
         f"<b>ext. GW head</b> = {_fmt(gw_external_head)} m"),
        (f"<b>rain rate</b> = {_fmt(rain_rate)} m/{t_units} &nbsp; "
         f"<b>storm dur</b> = {_fmt(storm_dur)} {t_units}"),
        f"<b>cum rain total</b> = {_fmt(cum_rain_total)} {vol_units}",
        f"<b>Δ subsurface storage</b> = {_fmt(final_sub_change)} {vol_units}",
        f"<b>cum OVERLAND outflow</b> = {_fmt(cum_overland_tot)} {vol_units}",
        f"<b>cum LATERAL groundwater</b> = {_fmt(cum_lateral_gw_tot)} {vol_units}",
        f"<b>peak surface depth</b> = {_fmt(peak_d)} m"
        + ("  <i>(= NCP floor &rarr; no real ponding)</i>" if pond_is_floor else ""),
        f"<b>ponding map scale</b> = fixed 0&ndash;{POND_ZMAX_MM:g} mm (both scenarios)",
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
        title=dict(text="<b>PIDS Pillar-2 sanity (Tier-3): surface&harr;subsurface coupling (3-D)</b><br>"
                        f"<span style='font-size:13px'>{title}</span>",
                   x=0.5, xanchor="center", y=0.992, yanchor="top"),
        sliders=sliders,
        updatemenus=updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.008,
                    xanchor="right", x=1.0, font=dict(size=10)),
        margin=dict(l=90, r=380, t=170, b=90),
        width=1360, height=2040,
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
build_coupling_3d_html = build_html


def _main_all(date: str = "2026-06-08"):
    """Loop the 3-D coupling keys -> one self-contained HTML each."""
    import os

    keys = ["sand_lateral_gw", "loam_overland"]
    data_dir = "../validation/sanity/data"
    viz_dir = "../validation/sanity/viz"
    os.makedirs(viz_dir, exist_ok=True)
    written = []
    for key in keys:
        nc = f"{data_dir}/coupling_3d__{key}__{date}.nc"
        out = f"{viz_dir}/coupling_3d__{key}__{date}.html"
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
        print("usage: python make_coupling_3d_html.py [<result.nc> <out.html>]  "
              "(no args -> build all)", file=sys.stderr)
        sys.exit(2)
