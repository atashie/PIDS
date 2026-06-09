#!/usr/bin/env python3
"""Build the COUPLED surface<->subsurface comparison summaries (in-house vs ParFlow) -- B4.

Loops the in-house coupling matrix (forward-model/viz/run_coupling_sanity.py): rain x antecedent
= {normal, extreme} x {dry, normal, wet} = 6 scenarios on a 2 m / 80-cell loam column with a
CLOSED no-flux base (so total water = soil int-theta + surface pond d closes to cum rainfall).
For each scenario it RE-RUNS the in-house CoupledProblem (storm + recession hyetograph, matching
run_coupling_sanity), loads ParFlow's coupled-column profiles
(../parflow/cases/coupled_column.py), aligns both on a common z grid and the in-house output
times, and writes one comparison NetCDF per scenario to data/.

The contract EXTENDS build_comparison_sweep.py's (so the profile rows render unchanged) with the
coupled surface quantities -- surface ponding depth d(t), cumulative infiltration partition
(infiltrated vs ponded), and the infiltration rate lambda(t) = d(soil storage)/dt -- which the
coupled viewer (make_comparison_coupled_html.py) renders as an extra surface/partition panel.

ParFlow's mass-consistent OverlandFlow surface store is d = max(psi_top, 0) (verified: closes the
closed-column theta+pond balance to ~1e-6 across the matrix; the DZ/2 offset used for the
STANDALONE overland case -- README S5a -- is WRONG for the coupled store and breaks the balance by
exactly DZ/2 once ponded).

Run (pids-fem):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      /root/miniforge3/bin/conda run -n pids-fem python build_comparison_coupled.py [scenario]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)  # SE-Piedmont loam (coupled SOIL)
THETA_S = LOAM["theta_s"]
NCELL = 80
DEPTH = 2.0
N_OUT = 60
DATE = "2026-06-09"

ANTECEDENT = {"dry": -3.0, "normal": -1.0, "wet": -0.15}
STORMS = {
    "normal":  dict(rate=0.30, storm_dur=0.30, t_end=1.0),
    "extreme": dict(rate=3.00, storm_dur=0.05, t_end=0.40),
}
SCENARIOS = {
    f"{sk}_on_{ak}": dict(
        psi0=psi0, rate=s["rate"], storm_dur=s["storm_dur"], t_end=s["t_end"],
        label=f"{sk} rain ({s['rate']} m/day, {s['storm_dur']*24:.1f} h) on {ak} soil (psi0={psi0} m)")
    for sk, s in STORMS.items() for ak, psi0 in ANTECEDENT.items()
}

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "forward-model")))
PF_DIR = os.path.expanduser("~/parflow-runs/coupled_column/summaries")
ZC = np.linspace(0.0, DEPTH, 101)


def run_inhouse(spec):
    """Re-run the in-house CoupledProblem (storm+recession). Mirrors run_coupling_sanity.run_scenario.

    Returns z (ascending, 0=base), and per-out-time arrays: head(t,z), theta(t,z), d(t), lam(t),
    soil_w(t), surf_w(t), cum_rain(t), mbe(t).
    """
    from mpi4py import MPI
    from dolfinx import mesh as dmesh
    from pids_forward.physics.constitutive import VanGenuchten
    from pids_forward.physics.coupling import CoupledProblem

    soil = VanGenuchten(**LOAM)
    msh = dmesh.create_interval(MPI.COMM_WORLD, NCELL, [0.0, DEPTH])
    prob = CoupledProblem(msh, soil)
    prob.set_initial_condition(lambda x: spec["psi0"] + 0.0 * x[0], d_value=0.0)
    rain = prob.add_rain(0.0)

    z = prob.Vpsi.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(z)
    zs = z[order]
    storm_dur, t_end = spec["storm_dur"], spec["t_end"]
    out_times = np.linspace(0.0, t_end, N_OUT)
    nz = zs.size

    head = np.zeros((N_OUT, nz)); theta = np.zeros((N_OUT, nz))
    d_surf = np.zeros(N_OUT); lam = np.zeros(N_OUT)
    soil_w = np.zeros(N_OUT); surf_w = np.zeros(N_OUT)
    cum_rain_t = np.zeros(N_OUT); mbe = np.zeros(N_OUT)

    def hyeto(t):
        return spec["rate"] if t <= storm_dur + 1e-12 else 0.0

    head[0] = prob.psi.x.array[order]; theta[0] = soil.theta(prob.psi.x.array[order])
    d_surf[0] = prob.surface_depth(); soil_w[0] = prob.soil_water(); surf_w[0] = prob.surface_water()
    w0 = prob.total_water()
    cum_rain = 0.0
    dt = 1e-4
    for k in range(1, N_OUT):
        t = out_times[k - 1]; t_target = out_times[k]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            r_now = hyeto(t + h)
            rain.value = r_now
            converged, it = prob.step(h)
            if converged:
                cum_rain += r_now * h
                t += h
                dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 2e-2)
            else:
                dt *= 0.5
                if dt < 1e-9:
                    raise RuntimeError(f"dt collapse at t={t:.5g}")
        head[k] = prob.psi.x.array[order]; theta[k] = soil.theta(prob.psi.x.array[order])
        d_surf[k] = prob.surface_depth(); lam[k] = prob.exchange_flux()
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water()
        cum_rain_t[k] = cum_rain
        mbe[k] = abs((prob.total_water() - w0) - cum_rain) / (cum_rain + 1e-30)
    return zs, head, theta, d_surf, lam, soil_w, surf_w, cum_rain_t, mbe


def interp_z(z_src, arr2d):
    """Interpolate a (time, z_src) array onto the common ZC grid in z."""
    return np.array([np.interp(ZC, z_src, arr2d[k]) for k in range(arr2d.shape[0])])


def interp_t(t_src, arr, t_dst):
    """Interpolate along time (axis 0) onto t_dst. arr is (t_src,) or (t_src, z)."""
    if arr.ndim == 1:
        return np.interp(t_dst, t_src, arr)
    return np.array([np.interp(t_dst, t_src, arr[:, j]) for j in range(arr.shape[1])]).T


def load_parflow(key, out_times, storm_dur, rate):
    """Load ParFlow coupled summary, build d/soil_w/partition, interpolate onto (out_times, ZC)."""
    pf = np.load(os.path.join(PF_DIR, f"coupled_{key}.npz"))
    z_pf = pf["z"]; t_pf = pf["times"]; head_pf = pf["head"]; theta_pf = pf["theta"]; dz = float(pf["dz"])
    ptop = head_pf[:, -1]
    d_pf = np.maximum(ptop, 0.0)                     # mass-consistent OverlandFlow surface store
    soil_w_pf = theta_pf.sum(axis=1) * dz            # subsurface theta-storage (m, unit area)

    # interpolate profiles: z onto ZC, then time onto out_times
    head_zc = interp_t(t_pf, interp_z(z_pf, head_pf), out_times)
    theta_zc = interp_t(t_pf, interp_z(z_pf, theta_pf), out_times)
    d_t = interp_t(t_pf, d_pf, out_times)
    soil_w_t = interp_t(t_pf, soil_w_pf, out_times)

    cum = rate * np.minimum(out_times, storm_dur)
    mbe = np.abs((soil_w_t - soil_w_t[0]) + d_t - cum) / np.where(cum > 0, cum, 1.0)
    lam_t = np.gradient(soil_w_t, out_times)         # infiltration rate = d(soil storage)/dt
    return head_zc, theta_zc, d_t, soil_w_t, lam_t, mbe


def build_one(key, spec):
    psi0, rate, storm_dur, t_end = spec["psi0"], spec["rate"], spec["storm_dur"], spec["t_end"]
    out_times = np.linspace(0.0, t_end, N_OUT)

    z_ih, head_ih, theta_ih, d_ih, lam_ih, soilw_ih, surfw_ih, cum_ih, mbe_ih = run_inhouse(spec)
    head_i = interp_z(z_ih, head_ih)
    theta_i = interp_z(z_ih, theta_ih)
    infil_ih = soilw_ih - soilw_ih[0]                # cumulative infiltrated (m)

    head_p, theta_p, d_p, soilw_p, lam_p, mbe_p = load_parflow(key, out_times, storm_dur, rate)
    infil_p = soilw_p - soilw_p[0]

    dhead = head_i - head_p
    dtheta = theta_i - theta_p
    dd = d_ih - d_p
    cum_rain = rate * np.minimum(out_times, storm_dur)

    m = dict(
        max_abs_dtheta=float(np.max(np.abs(dtheta))), rms_dtheta=float(np.sqrt(np.mean(dtheta**2))),
        max_abs_dhead=float(np.max(np.abs(dhead))), rms_dhead=float(np.sqrt(np.mean(dhead**2))),
        max_abs_dd_m=float(np.max(np.abs(dd))), rms_dd_m=float(np.sqrt(np.mean(dd**2))),
        inhouse_mbe_max=float(np.max(mbe_ih)), parflow_mbe_max=float(np.max(mbe_p)),
        peak_d_inhouse_m=float(d_ih.max()), peak_d_parflow_m=float(d_p.max()),
        final_infiltrated_inhouse_m=float(infil_ih[-1]), final_infiltrated_parflow_m=float(infil_p[-1]),
        final_ponded_inhouse_m=float(d_ih[-1]), final_ponded_parflow_m=float(d_p[-1]),
    )

    ds = xr.Dataset(
        data_vars=dict(
            head_inhouse=(("time", "z"), head_i, {"units": "m", "long_name": "pressure head psi (in-house)"}),
            head_parflow=(("time", "z"), head_p, {"units": "m", "long_name": "pressure head psi (ParFlow)"}),
            theta_inhouse=(("time", "z"), theta_i, {"units": "m3/m3", "long_name": "water content theta (in-house)"}),
            theta_parflow=(("time", "z"), theta_p, {"units": "m3/m3", "long_name": "water content theta (ParFlow)"}),
            dhead=(("time", "z"), dhead, {"units": "m", "long_name": "psi difference (in-house - ParFlow)"}),
            dtheta=(("time", "z"), dtheta, {"units": "m3/m3", "long_name": "theta difference (in-house - ParFlow)"}),
            mbe_inhouse=(("time",), mbe_ih, {"units": "-", "long_name": "relative mass-balance error (in-house)"}),
            mbe_parflow=(("time",), mbe_p, {"units": "-", "long_name": "relative mass-balance error (ParFlow, theta+pond)"}),
            # --- coupled surface quantities ---
            surface_depth_inhouse=(("time",), d_ih, {"units": "m", "long_name": "surface ponding depth d (in-house)"}),
            surface_depth_parflow=(("time",), d_p, {"units": "m", "long_name": "surface ponding store max(psi_top,0) (ParFlow)"}),
            infiltrated_inhouse=(("time",), infil_ih, {"units": "m", "long_name": "cumulative infiltrated water (in-house)"}),
            infiltrated_parflow=(("time",), infil_p, {"units": "m", "long_name": "cumulative infiltrated water (ParFlow)"}),
            lambda_inhouse=(("time",), lam_ih, {"units": "m/day", "long_name": "infiltration flux lambda (in-house exchange flux)"}),
            lambda_parflow=(("time",), lam_p, {"units": "m/day", "long_name": "infiltration rate d(soil storage)/dt (ParFlow)"}),
            cum_rain=(("time",), cum_rain, {"units": "m", "long_name": "cumulative rainfall input (both)"}),
        ),
        coords=dict(time=("time", out_times, {"units": "day"}),
                    z=("z", ZC, {"units": "m", "long_name": "elevation (0 = base, 2 = top surface)"})),
        attrs=dict(
            case=f"coupled__{key}", date=DATE,
            title=f"Coupled surface<->subsurface ({spec['label']}): in-house vs ParFlow",
            soil="Carsel & Parrish (1988) / SE-Piedmont loam",
            theta_r=LOAM["theta_r"], theta_s=LOAM["theta_s"], alpha_per_m=LOAM["alpha"],
            n=LOAM["n"], Ks_m_per_day=LOAM["Ks"],
            psi0_m=psi0, rain_flux_m_per_day=rate, duration_day=float(t_end),
            storm_duration_day=storm_dur, cumulative_input_m=float(cum_rain[-1]),
            grid_inhouse="P1 FEM, 80 cells / 81 nodes on [0,2] m + surface ponding store",
            grid_parflow="cell-centred FV, 80 cells on [0,2] m + OverlandFlow surface store",
            common_grid=f"{ZC.size} uniform points on [0,2] m (both interpolated); ParFlow interp to in-house times",
            deltas=("ParFlow OverlandFlow surface store = max(psi_top,0) (mass-consistent; carries a +DZ/2~12.5mm "
                    "cell-centred hydrostatic offset vs the in-house surface-node depth d). ParFlow: no Vogel/Ippisch "
                    "air-entry cap (active here -- the column SATURATES); tiny SpecificStorage vs in-house no-Ss; "
                    "cell-centred FV vs P1 FEM; two-phase storm+recession restart"),
            **m,
        ),
    )
    out_nc = os.path.join(HERE, "data", f"coupled__{key}__{DATE}.nc")
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ds.to_netcdf(out_nc)
    from make_comparison_coupled_html import build as build_html
    build_html(out_nc, os.path.join(HERE, "html", f"coupled__{key}__{DATE}.html"))
    return m


def main(only=None):
    keys = [only] if only else list(SCENARIOS)
    print("=== B4 coupled: in-house vs ParFlow ===")
    print(f"{'scenario':<18} {'max|dθ|':>9} {'RMS dθ':>9} {'max|dψ|m':>9} {'peak_d_ih':>10} {'peak_d_pf':>10} "
          f"{'MBE_ih':>9} {'MBE_pf':>9}")
    for key in keys:
        if key not in SCENARIOS:
            raise SystemExit(f"unknown scenario {key!r}; choose from {list(SCENARIOS)}")
        m = build_one(key, SCENARIOS[key])
        print(f"{key:<18} {m['max_abs_dtheta']:>9.2e} {m['rms_dtheta']:>9.2e} {m['max_abs_dhead']:>9.2e} "
              f"{m['peak_d_inhouse_m']:>10.4f} {m['peak_d_parflow_m']:>10.4f} "
              f"{m['inhouse_mbe_max']:>9.1e} {m['parflow_mbe_max']:>9.1e}")
    print(f"WROTE data/coupled__*__{DATE}.nc + html/coupled__*__{DATE}.html")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
