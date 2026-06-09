#!/usr/bin/env python3
"""Build the subsurface 1-D column comparison summary (in-house vs ParFlow).

Runs the in-house DOLFINx Richards solver on the SAME clean column as the ParFlow
deck (../parflow/cases/column_1d.py), loads ParFlow's exported profiles, interpolates
both onto a common grid at matched times, and writes ONE self-describing comparison
NetCDF to ``data/`` for the side-by-side HTML generator (``make_comparison_html.py``).

Run in the in-house env (DOLFINx); from WSL:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      /root/miniforge3/bin/conda run -n pids-fem python build_comparison_column.py

Matched setup (both engines): 1 m loam column, 60 cells, uniform IC psi0=-1.0 m,
constant top influx q=0.10 m/day (< Ks, no ponding), no-flow bottom, T=1.0 day,
output at 21 equally spaced times. Loam = Carsel & Parrish (1988):
theta_r=0.078, theta_s=0.43, alpha=3.6 /m, n=1.56, Ks=0.2496 m/day.
"""
from __future__ import annotations

import os

import numpy as np
import xarray as xr

# --- matched case parameters (identical to ../parflow/cases/column_1d.py) ----
LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)
NCELL = 60
PSI0 = -1.0
Q = 0.10
T_STOP = 1.0
NOUT = 21
DATE = "2026-06-08"
CASE = "subsurface__column_1d"

PARFLOW_NPZ = os.path.expanduser("~/parflow-runs/column_1d/output/column_1d_profiles.npz")
HERE = os.path.dirname(os.path.abspath(__file__))
# make the in-house package importable without modifying the conda env
import sys
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "forward-model")))
OUT_NC = os.path.join(HERE, "data", f"{CASE}__{DATE}.nc")

TIMES = np.round(np.linspace(0.0, T_STOP, NOUT), 6)
ZC = np.linspace(0.0, 1.0, 101)  # common comparison grid (elevation, m; 0=bottom,1=top)


# --- in-house run ------------------------------------------------------------
def run_inhouse():
    """Run the in-house Richards column; return (z, psi(t,z), theta(t,z), mbe(t))."""
    from mpi4py import MPI
    from dolfinx import mesh as dmesh
    from pids_forward.physics.constitutive import VanGenuchten
    from pids_forward.physics.richards import RichardsProblem

    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, NCELL)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: PSI0 + 0.0 * x[0])
    prob.add_flux_bc(lambda x: np.isclose(x[0], 1.0), Q)  # constant influx at the top

    z = prob.V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(z)
    z = z[order]
    s0 = prob.total_water()

    psi_rec = [prob.psi.x.array[order].copy()]
    theta_rec = [prob.theta_array()[order].copy()]
    mbe_rec = [0.0]
    prev = 0.0
    for tk in TIMES[1:]:
        prob.advance(t_end=float(tk - prev), dt=0.005)
        prev = float(tk)
        psi_rec.append(prob.psi.x.array[order].copy())
        theta_rec.append(prob.theta_array()[order].copy())
        dS = prob.total_water() - s0
        cum = Q * float(tk)
        mbe_rec.append(abs(dS - cum) / cum if cum > 0 else 0.0)
    return z, np.array(psi_rec), np.array(theta_rec), np.array(mbe_rec)


def interp_tz(z_src, arr_tz):
    """Interpolate (nt, nz_src) profiles onto the common grid ZC -> (nt, len(ZC))."""
    return np.array([np.interp(ZC, z_src, arr_tz[k]) for k in range(arr_tz.shape[0])])


def main():
    # in-house
    z_ih, psi_ih, th_ih, mbe_ih = run_inhouse()
    head_ih = interp_tz(z_ih, psi_ih)
    theta_ih = interp_tz(z_ih, th_ih)

    # ParFlow (exported profiles)
    pf = np.load(PARFLOW_NPZ)
    z_pf, t_pf = pf["z"], pf["times"]
    if not np.allclose(t_pf, TIMES, atol=1e-6):
        raise SystemExit(f"time mismatch: ParFlow {t_pf} vs expected {TIMES}")
    head_pf = interp_tz(z_pf, pf["head"])
    theta_pf = interp_tz(z_pf, pf["theta"])
    # ParFlow per-time water-content (theta-equivalent) mass-balance, same definition as in-house
    dz_pf = 1.0 / z_pf.size
    w_pf = (pf["theta"].sum(axis=1) * dz_pf)
    mbe_pf = np.array([0.0] + [abs((w_pf[k] - w_pf[0]) - Q * TIMES[k]) / (Q * TIMES[k])
                               for k in range(1, NOUT)])

    # differences on the common grid
    dhead = head_ih - head_pf
    dtheta = theta_ih - theta_pf
    max_abs_dtheta = float(np.max(np.abs(dtheta)))
    rms_dtheta = float(np.sqrt(np.mean(dtheta ** 2)))
    max_abs_dhead = float(np.max(np.abs(dhead)))
    rms_dhead = float(np.sqrt(np.mean(dhead ** 2)))

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
        coords=dict(
            time=("time", TIMES, {"units": "day"}),
            z=("z", ZC, {"units": "m", "long_name": "elevation (0 = bottom, 1 = top surface)"}),
        ),
        attrs=dict(
            case=CASE, date=DATE,
            title="Subsurface 1-D loam infiltration column: in-house vs ParFlow",
            soil="Carsel & Parrish (1988) loam",
            theta_r=LOAM["theta_r"], theta_s=LOAM["theta_s"], alpha_per_m=LOAM["alpha"],
            n=LOAM["n"], Ks_m_per_day=LOAM["Ks"],
            psi0_m=PSI0, rain_flux_m_per_day=Q, duration_day=T_STOP,
            cumulative_input_m=float(Q * T_STOP),
            grid_inhouse="P1 FEM, 60 cells / 61 nodes on [0,1] m",
            grid_parflow="cell-centred FV, 60 cells on [0,1] m",
            common_grid=f"{ZC.size} uniform points on [0,1] m (both interpolated)",
            deltas=("ParFlow: no Vogel/Ippisch air-entry cap (affects theta, K, near-sat tangent); "
                    "tiny SpecificStorage (S*Ss*psi) vs in-house no-Ss; cell-centred FV vs P1 FEM"),
            max_abs_dtheta=max_abs_dtheta, rms_dtheta=rms_dtheta,
            max_abs_dhead=max_abs_dhead, rms_dhead=rms_dhead,
            inhouse_mbe_max=float(np.max(mbe_ih)), parflow_mbe_max=float(np.max(mbe_pf)),
        ),
    )
    os.makedirs(os.path.dirname(OUT_NC), exist_ok=True)
    ds.to_netcdf(OUT_NC)
    print(f"WROTE {OUT_NC}")
    print(f"max|dtheta|={max_abs_dtheta:.4e}  RMS dtheta={rms_dtheta:.4e}")
    print(f"max|dhead| ={max_abs_dhead:.4e} m  RMS dhead={rms_dhead:.4e} m")
    print(f"in-house MBE max={np.max(mbe_ih):.2e}   ParFlow MBE max={np.max(mbe_pf):.2e}")


if __name__ == "__main__":
    main()
