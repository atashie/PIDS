"""Emit standardized Tier-3 sanity-run NetCDFs for the overland (diffusion-wave) module.

Runs storm-on-a-hillslope scenarios and writes the documented data contract
(governance/visualize-sanity-check-routine.md) so the SEPARATE visualization subagent /
generator (make_overland_html.py) can build the HTML without importing the solver.

The toy domain is a 1-D HILLSLOPE: ``ncell`` cells over ``L`` metres (NOT a single cell);
water routes laterally downslope to a free-drainage (normal-depth) outlet at x = L. The bed
is impermeable -- standalone overland has NO infiltration (that is the Module-3 coupling);
so rain either stores as surface depth or leaves the outlet, and mass closes as
rain = d(storage) + outflow.

Run from forward-model/ with PYTHONPATH=. :
  PYTHONPATH=. python viz/run_overland_sanity.py            # generate ALL scenarios
  PYTHONPATH=. python viz/run_overland_sanity.py <name>     # one scenario by key
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland import OverlandProblem

DATE = "2026-06-05"
DATA_DIR = "../validation/sanity/data"


def _block(rate, window):
    """Block hyetograph: constant ``rate`` (m/day) for t in [0, window], then 0."""
    return lambda t: rate if t <= window + 1e-12 else 0.0


def _pulses(edges, rates):
    """Piecewise-constant hyetograph: ``rates[i]`` on [edges[i], edges[i+1])."""
    def f(t):
        for i in range(len(rates)):
            if t < edges[i + 1] - 1e-12:
                return rates[i]
        return 0.0
    return f


# scenario matrix: 4 storm types on baseline terrain (S0=0.02, n=0.05), then slope/roughness
# sensitivity on the typical storm. Each: (label, L, S0, n_man, hyeto, rain_peak, storm_dur, t_end)
def _scenarios():
    L = 100.0
    typ = _block(0.30, 0.06)  # the "typical" storm reused for the terrain/roughness sweep
    return {
        # --- storm types (baseline 2% slope, n=0.05) ---
        "small_storm":    dict(label="small storm (0.05 m/day) on a 2% hillslope",
                               L=L, S0=0.02, n=0.05, hyeto=_block(0.05, 0.06), rain_peak=0.05, storm_dur=0.06, t_end=0.18),
        "typical_storm":  dict(label="typical storm (0.3 m/day) on a 2% hillslope",
                               L=L, S0=0.02, n=0.05, hyeto=typ, rain_peak=0.30, storm_dur=0.06, t_end=0.18),
        "severe_storm":   dict(label="severe storm (2.0 m/day burst) on a 2% hillslope",
                               L=L, S0=0.02, n=0.05, hyeto=_block(2.0, 0.03), rain_peak=2.0, storm_dur=0.03, t_end=0.12),
        "variable_storm": dict(label="variable storm (0.1->0.5->0.1->0.5->0) on a 2% hillslope",
                               L=L, S0=0.02, n=0.05,
                               hyeto=_pulses([0.0, 0.03, 0.06, 0.09, 0.12, 1e9], [0.1, 0.5, 0.1, 0.5, 0.0]),
                               rain_peak=0.5, storm_dur=0.12, t_end=0.24),
        # --- terrain / roughness sensitivity (typical 0.3 m/day storm) ---
        "steep_slope":    dict(label="typical storm on a STEEP 10% slope",
                               L=L, S0=0.10, n=0.05, hyeto=typ, rain_peak=0.30, storm_dur=0.06, t_end=0.18),
        "shallow_slope":  dict(label="typical storm on a SHALLOW 0.5% slope",
                               L=L, S0=0.005, n=0.05, hyeto=typ, rain_peak=0.30, storm_dur=0.06, t_end=0.18),
        "low_manning":    dict(label="typical storm, LOW roughness n=0.02 (smooth)",
                               L=L, S0=0.02, n=0.02, hyeto=typ, rain_peak=0.30, storm_dur=0.06, t_end=0.18),
        "high_manning":   dict(label="typical storm, HIGH roughness n=0.15 (dense veg)",
                               L=L, S0=0.02, n=0.15, hyeto=typ, rain_peak=0.30, storm_dur=0.06, t_end=0.18),
    }


def run_scenario(key, spec, *, n_out=60, ncell=100):
    L, S0, n_man = spec["L"], spec["S0"], spec["n"]
    hyeto, t_end = spec["hyeto"], spec["t_end"]
    msh = dmesh.create_interval(MPI.COMM_WORLD, ncell, [0.0, L])
    prob = OverlandProblem(msh, n_man=n_man)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    prob.set_initial_condition(lambda x: 0.0 * x[0])  # dry antecedent
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    coords = prob.V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(coords)
    xs = coords[order]
    zb = S0 * (L - xs)

    out_times = np.linspace(0.0, t_end, n_out)
    depth = np.zeros((n_out, xs.size))
    outflow = np.zeros(n_out); rainfall = np.zeros(n_out); mbe = np.zeros(n_out)
    iters_rec = np.zeros(n_out); maxvel = np.zeros(n_out)
    depth[0] = prob.d.x.array[order]
    rainfall[0] = hyeto(0.0)
    cum_rain = cum_out = 0.0
    w0 = prob.total_water()
    dt = 1e-5
    last_iters = 0
    for k in range(1, n_out):
        t = out_times[k - 1]
        t_target = out_times[k]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            r_now = hyeto(t + h)
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
                    raise RuntimeError(f"{key}: dt collapse at t={t:.5g}")
        depth[k] = prob.d.x.array[order]
        outflow[k] = prob.outflow_rate()
        rainfall[k] = hyeto(t_target)
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
        coords=dict(time=("time", out_times, {"units": "day"}), x=("x", xs, {"units": "m"})),
        attrs=dict(
            module="overland", scenario=spec["label"], date=DATE,
            n_man=n_man, slope=S0, length_m=L,
            rain_peak_m_per_day=spec["rain_peak"], storm_duration_day=spec["storm_dur"],
            equilibrium_outflow_rL=spec["rain_peak"] * L,
            peak_outflow_m2_per_day=float(outflow.max()),
            peak_velocity_m_per_s=float(maxvel.max()),
            mass_balance_error_max=float(mbe.max()),
        ),
    )
    out_nc = f"{DATA_DIR}/overland__{key}__{DATE}.nc"
    ds.to_netcdf(out_nc)
    print(f"WROTE {out_nc}  peak_out={outflow.max():.3f} (eq r*L={spec['rain_peak']*L:.2f}) "
          f"peak_vel={maxvel.max():.3f} m/s  max_mbe={mbe.max():.2e}", flush=True)
    return out_nc


def main(only=None):
    scen = _scenarios()
    keys = [only] if only else list(scen)
    for key in keys:
        run_scenario(key, scen[key])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
