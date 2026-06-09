#!/usr/bin/env python3
"""Build the subsurface SWEEP comparison summaries (in-house vs ParFlow).

Loops the non-ponding antecedent-wetness x intensity matrix from
../parflow/cases/column_sweep.py: for each scenario it runs the in-house Richards column
(ponding BC + storm/recession, matching run_subsurface_sanity), loads ParFlow's exported
profiles, aligns both on a common z grid at matched times, and writes one comparison NetCDF
per scenario to data/ (same contract as build_comparison_column.py, so make_comparison_html.py
renders each unchanged). Prints an across-scenario agreement table.

Run (pids-fem):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      /root/miniforge3/bin/conda run -n pids-fem python build_comparison_sweep.py
"""
from __future__ import annotations

import os

import numpy as np
import xarray as xr

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)
THETA_S = LOAM["theta_s"]
NCELL = 60
NFRAMES = 41
DATE = "2026-06-09"

SCENARIOS = {
    "mesic_q010": dict(psi0=-1.0, q=0.10, T_storm=0.50, label="mesic antecedent, typical storm"),
    "mesic_q002": dict(psi0=-1.0, q=0.02, T_storm=0.50, label="mesic antecedent, small storm"),
    "dry_q010":   dict(psi0=-5.0, q=0.10, T_storm=0.50, label="dry antecedent, typical storm"),
    "dry_q002":   dict(psi0=-5.0, q=0.02, T_storm=0.50, label="dry antecedent, small storm"),
}

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "forward-model")))
PF_DIR = os.path.expanduser("~/parflow-runs/column_sweep/summaries")
ZC = np.linspace(0.0, 1.0, 101)
DZ_PF = 1.0 / NCELL


def run_inhouse(psi0, q, T_storm, times):
    """Run the in-house column (ponding BC + storm/recession). Return z, psi(t,z), theta(t,z), W(t)."""
    from mpi4py import MPI
    from dolfinx import mesh as dmesh
    from pids_forward.physics.constitutive import VanGenuchten
    from pids_forward.physics.richards import RichardsProblem

    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, NCELL)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0])
    prob.add_ponding_bc(lambda x: np.isclose(x[0], 1.0), q)   # constant storm rain (storm-only)

    z = prob.V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(z)
    z = z[order]

    psi_rec = [prob.psi.x.array[order].copy()]
    theta_rec = [prob.theta_array()[order].copy()]
    W = [prob.total_water()]
    prev = 0.0
    for tk in times[1:]:
        prob.advance(t_end=float(tk - prev), dt=0.005)
        prev = float(tk)
        psi_rec.append(prob.psi.x.array[order].copy())
        theta_rec.append(prob.theta_array()[order].copy())
        W.append(prob.total_water())
    return z, np.array(psi_rec), np.array(theta_rec), np.array(W)


def interp_tz(z_src, arr):
    return np.array([np.interp(ZC, z_src, arr[k]) for k in range(arr.shape[0])])


def mbe_series(W, W0, q, T_storm, times):
    """Relative theta-balance error per time: |dW - cumulative rain| / cumulative rain."""
    out = [0.0]
    for k in range(1, times.size):
        cum = q * min(float(times[k]), T_storm)
        out.append(abs((W[k] - W0) - cum) / cum if cum > 0 else 0.0)
    return np.array(out)


def build_one(key, cfg):
    psi0, q, T_storm = cfg["psi0"], cfg["q"], cfg["T_storm"]
    times = np.round(np.linspace(0.0, T_storm, NFRAMES), 6)   # storm-only, matched to ParFlow

    z_ih, psi_ih, th_ih, W_ih = run_inhouse(psi0, q, T_storm, times)
    head_ih = interp_tz(z_ih, psi_ih)
    theta_ih = interp_tz(z_ih, th_ih)
    mbe_ih = mbe_series(W_ih, W_ih[0], q, T_storm, times)

    pf = np.load(os.path.join(PF_DIR, f"column_sweep_{key}.npz"))
    if not np.allclose(pf["times"], times, atol=1e-6):
        raise SystemExit(f"{key}: time mismatch ParFlow vs expected")
    head_pf = interp_tz(pf["z"], pf["head"])
    theta_pf = interp_tz(pf["z"], pf["theta"])
    W_pf = pf["theta"].sum(axis=1) * DZ_PF          # ParFlow theta-storage on its native grid
    mbe_pf = mbe_series(W_pf, W_pf[0], q, T_storm, times)

    dhead = head_ih - head_pf
    dtheta = theta_ih - theta_pf
    m = dict(max_abs_dtheta=float(np.max(np.abs(dtheta))), rms_dtheta=float(np.sqrt(np.mean(dtheta**2))),
             max_abs_dhead=float(np.max(np.abs(dhead))), rms_dhead=float(np.sqrt(np.mean(dhead**2))),
             inhouse_mbe_max=float(np.max(mbe_ih)), parflow_mbe_max=float(np.max(mbe_pf)))

    ds = xr.Dataset(
        data_vars=dict(
            head_inhouse=(("time", "z"), head_ih, {"units": "m", "long_name": "pressure head psi (in-house)"}),
            head_parflow=(("time", "z"), head_pf, {"units": "m", "long_name": "pressure head psi (ParFlow)"}),
            theta_inhouse=(("time", "z"), theta_ih, {"units": "m3/m3", "long_name": "water content theta (in-house)"}),
            theta_parflow=(("time", "z"), theta_pf, {"units": "m3/m3", "long_name": "water content theta (ParFlow)"}),
            dhead=(("time", "z"), dhead, {"units": "m", "long_name": "psi difference (in-house - ParFlow)"}),
            dtheta=(("time", "z"), dtheta, {"units": "m3/m3", "long_name": "theta difference (in-house - ParFlow)"}),
            mbe_inhouse=(("time",), mbe_ih, {"units": "-", "long_name": "relative mass-balance error (in-house, theta)"}),
            mbe_parflow=(("time",), mbe_pf, {"units": "-", "long_name": "relative mass-balance error (ParFlow, theta-equiv)"}),
        ),
        coords=dict(time=("time", times, {"units": "day"}),
                    z=("z", ZC, {"units": "m", "long_name": "elevation (0 = bottom, 1 = top surface)"})),
        attrs=dict(
            case=f"subsurface__{key}", date=DATE,
            title=f"Subsurface 1-D column ({cfg['label']}): in-house vs ParFlow",
            soil="Carsel & Parrish (1988) loam",
            theta_r=LOAM["theta_r"], theta_s=LOAM["theta_s"], alpha_per_m=LOAM["alpha"],
            n=LOAM["n"], Ks_m_per_day=LOAM["Ks"],
            psi0_m=psi0, rain_flux_m_per_day=q, duration_day=float(T_storm),
            storm_duration_day=T_storm, cumulative_input_m=float(q * T_storm),
            grid_inhouse="P1 FEM, 60 cells / 61 nodes on [0,1] m",
            grid_parflow="cell-centred FV, 60 cells on [0,1] m",
            common_grid=f"{ZC.size} uniform points on [0,1] m (both interpolated)",
            deltas=("ParFlow: no Vogel/Ippisch air-entry cap (theta, K, near-sat tangent); tiny "
                    "SpecificStorage vs in-house no-Ss; cell-centred FV vs P1 FEM; sub-Ks (no ponding)"),
            **m,
        ),
    )
    out_nc = os.path.join(HERE, "data", f"subsurface__{key}__{DATE}.nc")
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ds.to_netcdf(out_nc)
    from make_comparison_html import build as build_html  # same dir; generate the HTML too
    build_html(out_nc, os.path.join(HERE, "html", f"subsurface__{key}__{DATE}.html"))
    return m


def main():
    print("=== subsurface sweep: in-house vs ParFlow ===")
    print(f"{'scenario':<12} {'max|dθ|':>9} {'RMS dθ':>9} {'max|dψ| m':>10} {'RMS dψ m':>9} "
          f"{'MBE_ih':>9} {'MBE_pf':>9}")
    for key, cfg in SCENARIOS.items():
        m = build_one(key, cfg)
        print(f"{key:<12} {m['max_abs_dtheta']:>9.2e} {m['rms_dtheta']:>9.2e} "
              f"{m['max_abs_dhead']:>10.2e} {m['rms_dhead']:>9.2e} "
              f"{m['inhouse_mbe_max']:>9.1e} {m['parflow_mbe_max']:>9.1e}")
    print(f"WROTE data/subsurface__*__{DATE}.nc")


if __name__ == "__main__":
    main()
