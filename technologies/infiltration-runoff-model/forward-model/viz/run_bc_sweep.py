"""Module-3 OUTFLOW boundary-condition sweep (2-D coupling): one hillslope, one storm, vary the outlet.

Assessment requested 2026-06-06 (Arik): isolate the effect of the lateral-outflow BC. ONE sloped 2-D
hillslope cross-section + ONE heavy storm + ONE antecedent state, holding everything fixed EXCEPT the
downstream outlet boundary condition:

  closed         -- no outlet (natural no-flux; ponded water is trapped, only routes downslope)
  open_matched   -- free-drainage outlet at x=L, friction slope = bed slope S0   (normal-depth)
  open_steep     -- outlet friction slope = 2*S0                                  (faster drainage)
  open_shallow   -- outlet friction slope = 0.5*S0                                (slower drainage)

Each run records the outlet hydrograph q(t), the water partition (infiltrated / ponded / drained), the
ponding profile d(x,t), the storages, and the global mass-balance closure Δtotal = cum_rain-cum_outflow.
Writes one NetCDF per BC to ../validation/sanity/data/ following the documented data contract
(governance/visualize-sanity-check-routine.md), so the SEPARATE visualization subagent builds the HTML
WITHOUT importing the solver.

Run from forward-model/ with PYTHONPATH=. :
  PYTHONPATH=. python viz/run_bc_sweep.py            # all 4 BCs
  PYTHONPATH=. python viz/run_bc_sweep.py <bc_key>   # one BC by key
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

DATE = "2026-06-06"
DATA_DIR = "../validation/sanity/data"
SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)  # SE-Piedmont loam

# Fixed hillslope + storm (vary ONLY the outlet BC below).
L, H = 5.0, 1.0            # cross-section: x in [0,L], elevation z in [0,H]; surface = top edge z=H
NX, NZ = 12, 5            # mesh resolution (kept small -- overland is stiff)
S0 = 0.05                 # bed slope (5%), draining toward the x=L outlet
PSI0 = -0.8               # moderate antecedent soil moisture (uniform initial head, m)
RATE = 0.6                # heavy storm intensity (m/day) > Ks -> infiltration-excess ponding
STORM_DUR = 0.03          # storm duration (day) ~0.72 h
T_END = 0.06              # total simulated time (day): storm + recession
N_MAN = 0.05              # Manning roughness

# BC variants: (key, label, outlet slope or None for closed)
BCS = [
    ("closed",       "closed (no outlet -- trapped, routes downslope only)", None),
    ("open_matched", f"open outlet @ bed slope S0={S0}",                     S0),
    ("open_steep",   f"open outlet @ 2*S0={2*S0} (faster drainage)",          2 * S0),
    ("open_shallow", f"open outlet @ 0.5*S0={0.5*S0} (slower drainage)",      0.5 * S0),
]


def run_bc(key, label, outlet_slope, *, n_out=60):
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [NX, NZ])
    prob = CoupledProblem(msh, SOIL, n_man=N_MAN)
    prob.set_initial_condition(lambda x: PSI0 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: S0 * (L - x[0]))   # surface tilts down toward x=L
    rain = prob.add_rain(0.0)
    if outlet_slope is not None:
        prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=outlet_slope)

    # top-surface x-nodes (sorted) for the ponding profile d(x,t)
    topdofs = prob._top_dofs(prob.Vd)
    xtop = prob.Vd.tabulate_dof_coordinates()[topdofs, 0]
    xorder = np.argsort(xtop)
    xs = xtop[xorder]
    nx = xs.size
    zb_x = (S0 * (L - xs))  # bed elevation along the surface

    out_times = np.linspace(0.0, T_END, n_out)
    d_xt = np.zeros((n_out, nx))            # ponding depth profile d(x, t)
    outflow_q = np.zeros(n_out)             # outlet discharge hydrograph (m^2/day per width)
    cum_outflow_t = np.zeros(n_out)         # cumulative drained volume
    soil_w = np.zeros(n_out); surf_w = np.zeros(n_out); tot_w = np.zeros(n_out)
    cum_rain_t = np.zeros(n_out); rainfall = np.zeros(n_out); mbe = np.zeros(n_out)
    d_max = np.zeros(n_out); iters_rec = np.zeros(n_out)

    one_top = None  # top length for cum-rain bookkeeping (per unit width)
    from dolfinx import fem
    import ufl
    one_top = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * prob._ds_top)), op=MPI.SUM)

    def hyeto(t):
        return RATE if t <= STORM_DUR + 1e-12 else 0.0

    d_xt[0] = prob.d.x.array[topdofs][xorder]
    soil_w[0] = prob.soil_water(); surf_w[0] = prob.surface_water(); tot_w[0] = prob.total_water()
    w0 = tot_w[0]
    cum_rain = 0.0
    dt = 5e-5
    last_iters = 0
    for k in range(1, n_out):
        t = out_times[k - 1]; t_target = out_times[k]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            r_now = hyeto(t + h)
            rain.value = r_now
            converged, it = prob.step(h)
            if converged:
                cum_rain += r_now * one_top * h   # rain volume over the top edge (per unit width)
                t += h
                last_iters = it
                dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 1e-3)
            else:
                dt *= 0.5
                if dt < 1e-10:
                    raise RuntimeError(f"{key}: dt collapse at t={t:.5g}")
        d_xt[k] = prob.d.x.array[topdofs][xorder]
        outflow_q[k] = prob.outflow_rate()
        cum_outflow_t[k] = prob.cum_outflow
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water(); tot_w[k] = prob.total_water()
        cum_rain_t[k] = cum_rain; rainfall[k] = hyeto(t_target)
        d_max[k] = float(prob.d.x.array[topdofs].max())
        # global closure with the open outlet: Δtotal must equal cum_rain - cum_outflow
        mbe[k] = abs((tot_w[k] - w0) - (cum_rain - prob.cum_outflow)) / (cum_rain + 1e-30)
        iters_rec[k] = last_iters

    infiltrated = float(soil_w[-1] - soil_w[0])
    ponded = float(surf_w[-1])
    drained = float(cum_outflow_t[-1])
    ds = xr.Dataset(
        data_vars=dict(
            ponding_depth=(("time", "x"), d_xt, {"units": "m", "long_name": "surface ponding depth d(x,t)"}),
            outflow=(("time",), outflow_q, {"units": "m2/day", "long_name": "outlet discharge per unit width"}),
            cum_outflow=(("time",), cum_outflow_t, {"units": "m2", "long_name": "cumulative drained volume"}),
            soil_water=(("time",), soil_w, {"units": "m2", "long_name": "subsurface stored water (int theta)"}),
            surface_water=(("time",), surf_w, {"units": "m2", "long_name": "surface stored water (int d)"}),
            total_water=(("time",), tot_w, {"units": "m2", "long_name": "total stored water"}),
            cum_rain=(("time",), cum_rain_t, {"units": "m2", "long_name": "cumulative rainfall input"}),
            rainfall=(("time",), rainfall, {"units": "m/day", "long_name": "rainfall intensity"}),
            max_depth=(("time",), d_max, {"units": "m", "long_name": "max surface ponding depth"}),
            mass_balance_error=(("time",), mbe, {"units": "-", "long_name": "|Δtotal-(cum_rain-cum_outflow)|/cum_rain"}),
            newton_iters=(("time",), iters_rec, {"units": "-", "long_name": "Newton iterations (last step)"}),
            bed_elevation=(("x",), zb_x, {"units": "m", "long_name": "surface bed elevation z_b(x)"}),
        ),
        coords=dict(time=("time", out_times, {"units": "day"}),
                    x=("x", xs, {"units": "m", "long_name": "horizontal position (outlet at x=L)"})),
        attrs=dict(
            module="coupling_bc_sweep", bc_key=key, scenario=label, date=DATE,
            outlet_slope=(-1.0 if outlet_slope is None else float(outlet_slope)),
            is_closed=int(outlet_slope is None),
            L=L, H=H, nx=NX, nz=NZ, bed_slope_S0=S0, antecedent_psi0_m=PSI0,
            rain_rate_m_per_day=RATE, storm_duration_day=STORM_DUR, t_end_day=T_END,
            Ks_m_per_day=SOIL.Ks, n_manning=N_MAN, top_length_m=float(one_top),
            cum_rain_total_m2=float(cum_rain),
            final_infiltrated_m2=infiltrated, final_ponded_m2=ponded, final_drained_m2=drained,
            partition_infiltrated_frac=float(infiltrated / (cum_rain + 1e-30)),
            partition_ponded_frac=float(ponded / (cum_rain + 1e-30)),
            partition_drained_frac=float(drained / (cum_rain + 1e-30)),
            peak_outflow_m2_per_day=float(outflow_q.max()),
            peak_surface_depth_m=float(d_max.max()),
            mass_balance_error_max=float(mbe.max()),
            max_newton_iters=int(iters_rec.max()),
        ),
    )
    out_nc = f"{DATA_DIR}/coupling_bc__{key}__{DATE}.nc"
    ds.to_netcdf(out_nc)
    print(f"WROTE {out_nc}\n  partition: infiltrated={infiltrated:.4f} ponded={ponded:.4f} "
          f"drained={drained:.4f}  (cum_rain={cum_rain:.4f})  peak_q={outflow_q.max():.3e} "
          f"peak_d={d_max.max():.4f}  max_mbe={mbe.max():.2e}  max_iters={int(iters_rec.max())}",
          flush=True)
    return out_nc


def main(only=None):
    for key, label, slope in BCS:
        if only and key != only:
            continue
        run_bc(key, label, slope)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
