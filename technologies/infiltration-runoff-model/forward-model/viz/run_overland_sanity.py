"""Emit a standardized Tier-3 sanity-run NetCDF for the overland (diffusion-wave) module.

Runs a representative storm-on-a-hillslope scenario and writes the documented data contract
(governance/visualize-sanity-check-routine.md) so the SEPARATE visualization subagent can
build the HTML without importing the solver. Produces:
  coords: time [day], x [m]
  vars:   surface_depth(time, x) [m], bed_elevation(x) [m]
          + time series: outflow [m2/day], rainfall [m/day], mass_balance_error [-],
            newton_iters [-], max_velocity [m/s]
  attrs:  module, scenario, date, n_man, slope, length_m, peak rainfall, equilibrium r*L,
          peak outflow, max mass-balance error, peak velocity.

Run from forward-model/ with PYTHONPATH=. :
  PYTHONPATH=. python viz/run_overland_sanity.py [out.nc]
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland import OverlandProblem

DATE = "2026-06-05"
L = 100.0        # hillslope length [m]
S0 = 0.02        # 2% bed slope
N_MAN = 0.05     # vegetated overland Manning [s m^-1/3]
R_PEAK = 0.30    # storm intensity [m/day]
T_STORM = 0.06   # storm duration [day] (~1.4 h)
T_END = 0.18     # total run [day] (storm + recession)
N_OUT = 60       # number of output snapshots
NCELL = 100


def main(out_path: str) -> str:
    msh = dmesh.create_interval(MPI.COMM_WORLD, NCELL, [0.0, L])
    prob = OverlandProblem(msh, n_man=N_MAN)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    prob.set_initial_condition(lambda x: 0.0 * x[0])  # dry antecedent
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    coords = prob.V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(coords)
    xs = coords[order]
    zb = (S0 * (L - xs))

    out_times = np.linspace(0.0, T_END, N_OUT)
    depth = np.zeros((N_OUT, xs.size))
    outflow = np.zeros(N_OUT)
    rainfall = np.zeros(N_OUT)
    mbe = np.zeros(N_OUT)
    iters_rec = np.zeros(N_OUT)
    maxvel = np.zeros(N_OUT)

    # record the initial (dry) state
    depth[0] = prob.d.x.array[order]
    cum_rain = cum_out = 0.0
    w0 = prob.total_water()

    dt = 1e-4
    last_iters = 0
    for k in range(1, N_OUT):
        t_target = out_times[k]
        t = out_times[k - 1]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            r_now = R_PEAK if (t + h) <= T_STORM + 1e-12 else 0.0
            rain.value = r_now
            converged, it = prob.step(h)
            if converged:
                cum_rain += r_now * L * h
                cum_out += prob.last_outflow * h
                t += h
                last_iters = it
                dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 5e-3)
            else:
                dt *= 0.5
                if dt < 1e-9:
                    raise RuntimeError(f"dt collapse at t={t:.5g}")
        depth[k] = prob.d.x.array[order]
        outflow[k] = prob.outflow_rate()
        rainfall[k] = R_PEAK if t_target <= T_STORM + 1e-12 else 0.0
        mbe[k] = abs((prob.total_water() - w0) - (cum_rain - cum_out)) / (cum_rain + 1e-30)
        iters_rec[k] = last_iters
        maxvel[k] = float(np.max(np.abs(prob.velocity().x.array)))

    ds = xr.Dataset(
        data_vars=dict(
            surface_depth=(("time", "x"), depth, {"units": "m", "long_name": "overland water depth"}),
            bed_elevation=(("x",), zb, {"units": "m", "long_name": "bed elevation z_b"}),
            outflow=(("time",), outflow, {"units": "m2/day", "long_name": "outlet discharge per width"}),
            rainfall=(("time",), rainfall, {"units": "m/day", "long_name": "rainfall intensity"}),
            mass_balance_error=(("time",), mbe, {"units": "-", "long_name": "relative global mass-balance error"}),
            newton_iters=(("time",), iters_rec, {"units": "-", "long_name": "Newton iterations (last step)"}),
            max_velocity=(("time",), maxvel, {"units": "m/s", "long_name": "domain-max Manning velocity"}),
        ),
        coords=dict(
            time=("time", out_times, {"units": "day"}),
            x=("x", xs, {"units": "m"}),
        ),
        attrs=dict(
            module="overland",
            scenario="storm hydrograph on a 100 m hillslope (2% slope) + recession",
            date=DATE,
            n_man=N_MAN,
            slope=S0,
            length_m=L,
            rain_peak_m_per_day=R_PEAK,
            storm_duration_day=T_STORM,
            equilibrium_outflow_rL=R_PEAK * L,
            peak_outflow_m2_per_day=float(outflow.max()),
            peak_velocity_m_per_s=float(maxvel.max()),
            mass_balance_error_max=float(mbe.max()),
        ),
    )
    ds.to_netcdf(out_path)
    print(f"WROTE {out_path}  peak_outflow={outflow.max():.3f} (eq r*L={R_PEAK*L:.1f}) "
          f"peak_vel={maxvel.max():.3f} m/s  max_mbe={mbe.max():.2e}")
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "../validation/sanity/data/overland__storm__2026-06-05.nc"
    main(out)
