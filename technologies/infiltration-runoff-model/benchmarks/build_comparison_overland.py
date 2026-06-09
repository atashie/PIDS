#!/usr/bin/env python3
"""Build the overland 1-D hillslope storm comparison summary (in-house vs ParFlow).

Runs the in-house diffusion-wave OverlandProblem on the SAME hillslope storm as the
ParFlow deck (../parflow/cases/overland_hillslope.py), loads ParFlow's exported
profiles, aligns both on a common x grid at matched times, and writes ONE
self-describing comparison NetCDF to ``data/`` for ``make_comparison_overland_html.py``.

Run in the in-house env (DOLFINx); from WSL:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      /root/miniforge3/bin/conda run -n pids-fem python build_comparison_overland.py

Matched setup (both engines): 100 m hillslope, 2% slope, Manning n=0.05 (SI),
0.30 m/day storm for 0.06 day then recession to 0.18 day, dry start, impermeable bed.
Comparison quantities: ponded depth profile d(x,t) and the outlet hydrograph Q(t),
the latter derived for BOTH from the storage balance Q = rain*L - dW/dt (model-agnostic).
"""
from __future__ import annotations

import os

import numpy as np
import xarray as xr

L = 100.0
NCELL = 100
S0 = 0.02
N_MAN = 0.05
Q_RAIN = 0.30
STORM = 0.06
T_STOP = 0.18
DT_OUT = 0.003
DATE = "2026-06-08"
CASE = "overland__hillslope_storm"

PARFLOW_NPZ = os.path.expanduser("~/parflow-runs/overland_hillslope/output/overland_hillslope_profiles.npz")
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "forward-model")))
OUT_NC = os.path.join(HERE, "data", f"{CASE}__{DATE}.nc")

TIMES = np.round(np.arange(0.0, T_STOP + 1e-9, DT_OUT), 6)   # match ParFlow dump times
XC = np.linspace(0.0, L, 101)                                # common comparison grid (m)


def run_inhouse():
    """Run the in-house overland hillslope storm; return (x, depth(t,x), W(t))."""
    from mpi4py import MPI
    from dolfinx import mesh as dmesh
    from pids_forward.physics.overland import OverlandProblem

    msh = dmesh.create_interval(MPI.COMM_WORLD, NCELL, [0.0, L])
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    coords = prob.V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(coords)
    xs = coords[order]

    def hyeto(t):
        return Q_RAIN if t <= STORM + 1e-12 else 0.0

    depth = [prob.d.x.array[order].copy()]
    W = [prob.total_water()]
    dt = 1e-5
    for k in range(1, TIMES.size):
        t, t_target = float(TIMES[k - 1]), float(TIMES[k])
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            rain.value = hyeto(t + h)
            converged, it = prob.step(h)
            if converged:
                t += h
                dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 5e-3)
            else:
                dt *= 0.5
                if dt < 1e-9:
                    raise RuntimeError(f"in-house overland dt collapse at t={t:.5g}")
        depth.append(prob.d.x.array[order].copy())
        W.append(prob.total_water())
    return xs, np.array(depth), np.array(W)


def manning_outlet(depth_tx):
    """Outlet discharge Q(t) = 86400/n * d_outlet^(5/3) * sqrt(S) (m^2/day per width) --
    the in-house add_outflow_bc definition, applied to EACH model's outlet depth (x=L) so the
    hydrograph comparison uses one consistent, infiltration-independent definition."""
    d_L = np.maximum(depth_tx[:, -1], 0.0)
    return 86400.0 / N_MAN * d_L ** (5.0 / 3.0) * np.sqrt(S0)


def interp_tx(x_src, arr_tx):
    return np.array([np.interp(XC, x_src, arr_tx[k]) for k in range(arr_tx.shape[0])])


def main():
    # in-house
    xs, d_ih, W_ih = run_inhouse()
    depth_ih = interp_tx(xs, d_ih)
    q_ih = manning_outlet(depth_ih)

    # ParFlow
    pf = np.load(PARFLOW_NPZ)
    if not np.allclose(pf["times"], TIMES, atol=1e-6):
        raise SystemExit(f"time mismatch: ParFlow {pf['times'][:3]}... vs {TIMES[:3]}...")
    depth_pf = interp_tx(pf["x"], pf["depth"])
    q_pf = manning_outlet(depth_pf)
    rain = np.where(TIMES <= STORM + 1e-9, Q_RAIN, 0.0)
    eq_rL = float(Q_RAIN * L)

    ddepth = depth_ih - depth_pf
    max_abs_dd = float(np.max(np.abs(ddepth)))
    rms_dd = float(np.sqrt(np.mean(ddepth ** 2)))
    max_abs_dq = float(np.max(np.abs(q_ih - q_pf)))

    ds = xr.Dataset(
        data_vars=dict(
            depth_inhouse=(("time", "x"), depth_ih, {"units": "m", "long_name": "ponded depth d (in-house)"}),
            depth_parflow=(("time", "x"), depth_pf, {"units": "m", "long_name": "ponded depth d (ParFlow)"}),
            ddepth=(("time", "x"), ddepth, {"units": "m", "long_name": "depth difference (in-house - ParFlow)"}),
            outflow_inhouse=(("time",), q_ih, {"units": "m2/day", "long_name": "outlet discharge per width (in-house, Manning normal-depth at outlet)"}),
            outflow_parflow=(("time",), q_pf, {"units": "m2/day", "long_name": "outlet discharge per width (ParFlow, Manning normal-depth at outlet)"}),
            rain=(("time",), rain, {"units": "m/day", "long_name": "rainfall intensity"}),
        ),
        coords=dict(
            time=("time", TIMES, {"units": "day"}),
            x=("x", XC, {"units": "m", "long_name": "distance along hillslope (0 = top, L = outlet)"}),
        ),
        attrs=dict(
            case=CASE, date=DATE,
            title="Overland 1-D hillslope storm: in-house (diffusion-wave) vs ParFlow (OverlandDiffusive)",
            length_m=L, slope=S0, n_man_SI=N_MAN,
            rain_peak_m_per_day=Q_RAIN, storm_duration_day=STORM, duration_day=T_STOP,
            equilibrium_outflow_rL=eq_rL,
            grid_inhouse="P1 FEM, 100 cells on [0,L]", grid_parflow="cell-centred FV, 100 cells on [0,L]",
            common_grid=f"{XC.size} uniform points on [0,L] (both interpolated)",
            deltas=("both diffusion-wave (in-house) vs ParFlow OverlandDiffusive; ParFlow ponded depth read "
                    "from top-cell pressure; outlet Q = Manning normal-depth 86400/n*d^(5/3)*sqrt(S) at the "
                    "outlet for both; ParFlow IC saturated to surface (impermeable-bed match); n_PF=n_SI/86400"),
            peak_depth_inhouse_mm=float(1000 * depth_ih.max()),
            peak_depth_parflow_mm=float(1000 * depth_pf.max()),
            peak_outflow_inhouse=float(q_ih.max()), peak_outflow_parflow=float(q_pf.max()),
            max_abs_ddepth_mm=float(1000 * max_abs_dd), rms_ddepth_mm=float(1000 * rms_dd),
            max_abs_doutflow=max_abs_dq,
        ),
    )
    os.makedirs(os.path.dirname(OUT_NC), exist_ok=True)
    ds.to_netcdf(OUT_NC)
    print(f"WROTE {OUT_NC}")
    print(f"peak depth: in-house {1000*depth_ih.max():.3f} mm  ParFlow {1000*depth_pf.max():.3f} mm")
    print(f"peak outflow: in-house {q_ih.max():.3f}  ParFlow {q_pf.max():.3f}  (eq r*L={eq_rL:.3f}) m2/day")
    print(f"depth diff: max|dd|={1000*max_abs_dd:.3f} mm  RMS={1000*rms_dd:.3f} mm   max|dQ|={max_abs_dq:.3f} m2/day")


if __name__ == "__main__":
    main()
