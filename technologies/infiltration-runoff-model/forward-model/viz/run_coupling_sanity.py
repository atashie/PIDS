"""Emit standardized Tier-3 sanity-run NetCDFs for the Module-3 surface<->subsurface coupling (1-D).

Runs storm + recession scenarios across a rain x antecedent matrix (normal/extreme rain on
dry/normal/wet soil) and writes the documented data contract
(governance/visualize-sanity-check-routine.md) so the SEPARATE visualization subagent
(make_coupling_html.py) can build the HTML WITHOUT importing the solver.

Domain: a 2 m soil column (z up, surface at z=2), CLOSED no-flux base, so total water (soil ∫θ +
surface store d) changes by exactly the cumulative rainfall -- a clean coupled mass-balance check.
The land-surface exchange partitions rain between infiltration (λ into the soil) and ponding (d):
dry/normal soil under normal rain should infiltrate (d≈0, advancing wetting front); wet soil or
extreme rain should pond (infiltration-excess). A recession (rain off) then drains any pond into
the soil. Each NetCDF records the ψ/θ depth profiles (time x z), the surface depth d, the exchange
flux λ, rainfall, the soil/surface storages, the partitioning, mass-balance error, and Newton iters.

Run from forward-model/ with PYTHONPATH=. :
  PYTHONPATH=. python viz/run_coupling_sanity.py            # generate ALL scenarios
  PYTHONPATH=. python viz/run_coupling_sanity.py <name>     # one scenario by key
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

DATE = "2026-06-05"
DATA_DIR = "../validation/sanity/data"
SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)  # SE-Piedmont loam

# antecedent soil states (uniform initial pressure head, m) and storm intensities (m/day).
ANTECEDENT = {"dry": -3.0, "normal": -1.0, "wet": -0.15}
# normal = moderate SE-Piedmont storm; extreme = 100-yr Atlas-14-style sub-hourly burst.
STORMS = {
    "normal":  dict(rate=0.30, storm_dur=0.30, t_end=1.0),
    "extreme": dict(rate=3.00, storm_dur=0.05, t_end=0.40),
}


def _scenarios():
    scen = {}
    for sk, s in STORMS.items():
        for ak, psi0 in ANTECEDENT.items():
            scen[f"{sk}_on_{ak}"] = dict(
                label=f"{sk} rain ({s['rate']} m/day, {s['storm_dur']*24:.1f} h) on {ak} soil "
                      f"(ψ0={psi0} m)",
                psi0=psi0, rate=s["rate"], storm_dur=s["storm_dur"], t_end=s["t_end"])
    return scen


def run_scenario(key, spec, *, n_out=60, ncell=80, depth_m=2.0):
    msh = dmesh.create_interval(MPI.COMM_WORLD, ncell, [0.0, depth_m])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: spec["psi0"] + 0.0 * x[0], d_value=0.0)
    rain = prob.add_rain(0.0)

    z = prob.Vpsi.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(z)
    zs = z[order]  # elevation (0 = base, depth_m = surface)
    storm_dur, t_end = spec["storm_dur"], spec["t_end"]

    out_times = np.linspace(0.0, t_end, n_out)
    nz = zs.size
    head = np.zeros((n_out, nz)); theta = np.zeros((n_out, nz))
    d_surf = np.zeros(n_out); lam = np.zeros(n_out); rainfall = np.zeros(n_out)
    soil_w = np.zeros(n_out); surf_w = np.zeros(n_out)
    cum_rain_t = np.zeros(n_out); mbe = np.zeros(n_out); iters_rec = np.zeros(n_out)

    def hyeto(t):
        return spec["rate"] if t <= storm_dur + 1e-12 else 0.0

    head[0] = prob.psi.x.array[order]; theta[0] = SOIL.theta(prob.psi.x.array[order])
    d_surf[0] = prob.surface_depth(); soil_w[0] = prob.soil_water(); surf_w[0] = prob.surface_water()
    w0 = prob.total_water()
    cum_rain = 0.0
    dt = 1e-4
    last_iters = 0
    for k in range(1, n_out):
        t = out_times[k - 1]; t_target = out_times[k]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            r_now = hyeto(t + h)
            rain.value = r_now
            converged, it = prob.step(h)
            if converged:
                cum_rain += r_now * h  # 1-D surface is a point (unit area): cum rain = ∫r dt
                t += h
                last_iters = it
                dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 2e-2)
            else:
                dt *= 0.5
                if dt < 1e-9:
                    raise RuntimeError(f"{key}: dt collapse at t={t:.5g}")
        head[k] = prob.psi.x.array[order]; theta[k] = SOIL.theta(prob.psi.x.array[order])
        d_surf[k] = prob.surface_depth(); lam[k] = prob.exchange_flux()
        rainfall[k] = hyeto(t_target)
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water()
        cum_rain_t[k] = cum_rain
        mbe[k] = abs((prob.total_water() - w0) - cum_rain) / (cum_rain + 1e-30)
        iters_rec[k] = last_iters

    ds = xr.Dataset(
        data_vars=dict(
            head=(("time", "z"), head, {"units": "m", "long_name": "soil pressure head psi"}),
            water_content=(("time", "z"), theta, {"units": "-", "long_name": "volumetric water content theta"}),
            surface_depth=(("time",), d_surf, {"units": "m", "long_name": "surface ponding depth d"}),
            exchange_flux=(("time",), lam, {"units": "m/day", "long_name": "land-surface infiltration flux lambda"}),
            rainfall=(("time",), rainfall, {"units": "m/day", "long_name": "rainfall intensity"}),
            soil_water=(("time",), soil_w, {"units": "m", "long_name": "subsurface stored water (int theta)"}),
            surface_water=(("time",), surf_w, {"units": "m", "long_name": "surface stored water (int d)"}),
            cum_rain=(("time",), cum_rain_t, {"units": "m", "long_name": "cumulative rainfall input"}),
            mass_balance_error=(("time",), mbe, {"units": "-", "long_name": "relative global mass-balance error"}),
            newton_iters=(("time",), iters_rec, {"units": "-", "long_name": "Newton iterations (last step)"}),
        ),
        coords=dict(time=("time", out_times, {"units": "day"}),
                    z=("z", zs, {"units": "m", "long_name": "elevation (0=base, top=surface)"})),
        attrs=dict(
            module="coupling", scenario=spec["label"], date=DATE,
            antecedent_psi0_m=spec["psi0"], rain_rate_m_per_day=spec["rate"],
            storm_duration_day=storm_dur, column_depth_m=depth_m, Ks_m_per_day=SOIL.Ks,
            cum_rain_total_m=float(cum_rain), peak_surface_depth_m=float(d_surf.max()),
            peak_exchange_flux_m_per_day=float(lam.max()),
            final_infiltrated_m=float(soil_w[-1] - soil_w[0]),
            final_ponded_m=float(surf_w[-1]),
            mass_balance_error_max=float(mbe.max()),
            max_newton_iters=int(iters_rec.max()),
        ),
    )
    out_nc = f"{DATA_DIR}/coupling__{key}__{DATE}.nc"
    ds.to_netcdf(out_nc)
    print(f"WROTE {out_nc}  peak_d={d_surf.max():.4f} m  infiltrated={soil_w[-1]-soil_w[0]:.4f} m "
          f"ponded_final={surf_w[-1]:.4f} m  cum_rain={cum_rain:.4f} m  max_mbe={mbe.max():.2e} "
          f"max_iters={int(iters_rec.max())}", flush=True)
    return out_nc


def main(only=None):
    scen = _scenarios()
    keys = [only] if only else list(scen)
    for key in keys:
        run_scenario(key, scen[key])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
