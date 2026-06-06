"""Module-3 SOIL-TEXTURE sweep (2-D coupling): one hillslope + one storm + one open outlet, vary soil.

Assessment requested 2026-06-06 (Arik): drainage and infiltration profiles under different soil
textures. ONE sloped 2-D hillslope cross-section with an OPEN downstream outlet, ONE heavy storm, and
ONE antecedent head -- holding everything fixed EXCEPT the soil hydraulic properties (van Genuchten /
Mualem, standard Carsel-Parrish textures in the model's units: alpha [1/m], Ks [m/day]):

  sand : Ks=7.13  -> rain << Ks: all infiltrates, deep front, little/no runoff
  loam : Ks=0.25  (the reference soil used elsewhere)
  silt : Ks=0.06  -> infiltration-excess: ponds + drains out the outlet
  clay : Ks=0.048 -> strongest infiltration-excess (very stiff: VG n=1.09)

Each run records the INFILTRATION PROFILE theta(z)/psi(z) at the mid-slope column, the DRAINAGE
(outlet hydrograph + partition infiltrated/ponded/drained), the ponding profile d(x,t), and the global
mass-balance closure. Writes one NetCDF per soil to ../validation/sanity/data/ (data contract of
governance/visualize-sanity-check-routine.md) so the SEPARATE viz subagent builds the HTML.

Run from forward-model/ with PYTHONPATH=. :
  PYTHONPATH=. python viz/run_texture_sweep.py            # all soils
  PYTHONPATH=. python viz/run_texture_sweep.py <soil_key> # one soil by key
"""
from __future__ import annotations

import sys

import numpy as np
import ufl
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

DATE = "2026-06-06"
DATA_DIR = "../validation/sanity/data"

# Fixed hillslope + storm + outlet (vary ONLY the soil below).
L, H = 5.0, 1.0
NX, NZ = 10, 8            # finer vertical resolution for the infiltration profile
S0 = 0.05                 # bed slope, drains toward x=L
# PER-SOIL field-moist antecedent head (m). A COMMON head is physically NON-comparable across textures
# (at psi=-2 clay is Se~0.92 wet but sand is Se~0.003 bone-dry, K->0 -> intractable Richards). Instead
# each soil starts at a realistic field-moist state for THAT texture (coarse soils sit at lower suction).
ANTECEDENT_PSI = {"sand": -0.12, "loam": -1.0, "silt": -0.6, "clay": -1.5}
RATE = 0.6                # storm intensity (m/day)
STORM_DUR = 0.03          # storm duration (day)
T_END = 0.06              # storm + recession
N_MAN = 0.05

# Carsel-Parrish van Genuchten textures (alpha 1/m, Ks m/day).
SOILS = {
    "sand": VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.128),
    "loam": VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6,  n=1.56, Ks=0.25),
    "silt": VanGenuchten(theta_r=0.034, theta_s=0.46, alpha=1.6,  n=1.37, Ks=0.060),
    "clay": VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8,  n=1.09, Ks=0.048),
}
SOIL_LABEL = {
    "sand": "sand (Ks=7.13 m/d, n=2.68)",
    "loam": "loam (Ks=0.25 m/d, n=1.56)",
    "silt": "silt (Ks=0.06 m/d, n=1.37)",
    "clay": "clay (Ks=0.048 m/d, n=1.09)",
}


def run_soil(key, *, n_out=60):
    soil = SOILS[key]
    psi0 = ANTECEDENT_PSI[key]
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [NX, NZ])
    prob = CoupledProblem(msh, soil, n_man=N_MAN)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    # mid-slope vertical column (x = L/2) for the infiltration profile psi(z)/theta(z).
    coords = prob.Vpsi.tabulate_dof_coordinates()
    colmask = np.isclose(coords[:, 0], L / 2.0)
    zcol = coords[colmask, prob._zaxis]
    zorder = np.argsort(zcol)
    coldofs = np.where(colmask)[0][zorder]
    zs = zcol[zorder]            # elevation 0=base .. H=surface
    nz = zs.size

    # top-surface x-nodes for d(x,t)
    topdofs = prob._top_dofs(prob.Vd)
    xtop = prob.Vd.tabulate_dof_coordinates()[topdofs, 0]
    xorder = np.argsort(xtop)
    xs = xtop[xorder]
    nx = xs.size

    one_top = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * prob._ds_top)), op=MPI.SUM)

    out_times = np.linspace(0.0, T_END, n_out)
    psi_zt = np.zeros((n_out, nz)); theta_zt = np.zeros((n_out, nz))
    d_xt = np.zeros((n_out, nx))
    outflow_q = np.zeros(n_out); cum_outflow_t = np.zeros(n_out)
    soil_w = np.zeros(n_out); surf_w = np.zeros(n_out); tot_w = np.zeros(n_out)
    cum_rain_t = np.zeros(n_out); rainfall = np.zeros(n_out); mbe = np.zeros(n_out)
    iters_rec = np.zeros(n_out)

    def hyeto(t):
        return RATE if t <= STORM_DUR + 1e-12 else 0.0

    psi_zt[0] = prob.psi.x.array[coldofs]; theta_zt[0] = soil.theta(prob.psi.x.array[coldofs])
    d_xt[0] = prob.d.x.array[topdofs][xorder]
    soil_w[0] = prob.soil_water(); surf_w[0] = prob.surface_water(); tot_w[0] = prob.total_water()
    w0 = tot_w[0]
    cum_rain = 0.0
    dt = 1e-6          # tiny cold-start step (sand's Ks is ~150x clay's -> fast, stiff dry front)
    last_iters = 0

    def safe_step(h):
        """prob.step(h), but a SINGULAR/blown-up solve (PETSc raises, e.g. dry-sand K->0 zero pivot)
        is caught and treated as non-convergence: restore the last accepted state so dt can be cut."""
        try:
            return prob.step(h)
        except Exception:
            prob.psi.x.array[:] = prob.psi_n.x.array; prob.psi.x.scatter_forward()
            prob.d.x.array[:] = prob.d_n.x.array; prob.d.x.scatter_forward()
            prob.lam.x.array[:] = 0.0; prob.lam.x.scatter_forward()  # reset (re-solved next step)
            prob._problem = None  # rebuild the SNES after the failed solve
            return False, 0

    for k in range(1, n_out):
        t = out_times[k - 1]; t_target = out_times[k]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            r_now = hyeto(t + h)
            rain.value = r_now
            converged, it = safe_step(h)
            if converged:
                cum_rain += r_now * one_top * h
                t += h
                last_iters = it
                dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 5e-4)
            else:
                dt *= 0.5
                if dt < 1e-12:
                    raise RuntimeError(f"{key}: dt collapse at t={t:.5g}")
        psi_zt[k] = prob.psi.x.array[coldofs]; theta_zt[k] = soil.theta(prob.psi.x.array[coldofs])
        d_xt[k] = prob.d.x.array[topdofs][xorder]
        outflow_q[k] = prob.outflow_rate(); cum_outflow_t[k] = prob.cum_outflow
        soil_w[k] = prob.soil_water(); surf_w[k] = prob.surface_water(); tot_w[k] = prob.total_water()
        cum_rain_t[k] = cum_rain; rainfall[k] = hyeto(t_target)
        mbe[k] = abs((tot_w[k] - w0) - (cum_rain - prob.cum_outflow)) / (cum_rain + 1e-30)
        iters_rec[k] = last_iters

    infiltrated = float(soil_w[-1] - soil_w[0]); ponded = float(surf_w[-1]); drained = float(cum_outflow_t[-1])
    ds = xr.Dataset(
        data_vars=dict(
            head=(("time", "z"), psi_zt, {"units": "m", "long_name": "mid-column pressure head psi(z)"}),
            water_content=(("time", "z"), theta_zt, {"units": "-", "long_name": "mid-column theta(z)"}),
            ponding_depth=(("time", "x"), d_xt, {"units": "m", "long_name": "surface ponding depth d(x,t)"}),
            outflow=(("time",), outflow_q, {"units": "m2/day", "long_name": "outlet discharge per unit width"}),
            cum_outflow=(("time",), cum_outflow_t, {"units": "m2", "long_name": "cumulative drained volume"}),
            soil_water=(("time",), soil_w, {"units": "m2", "long_name": "subsurface stored water"}),
            surface_water=(("time",), surf_w, {"units": "m2", "long_name": "surface stored water"}),
            total_water=(("time",), tot_w, {"units": "m2", "long_name": "total stored water"}),
            cum_rain=(("time",), cum_rain_t, {"units": "m2", "long_name": "cumulative rainfall input"}),
            rainfall=(("time",), rainfall, {"units": "m/day", "long_name": "rainfall intensity"}),
            mass_balance_error=(("time",), mbe, {"units": "-", "long_name": "|d total-(rain-outflow)|/rain"}),
            newton_iters=(("time",), iters_rec, {"units": "-", "long_name": "Newton iterations (last step)"}),
        ),
        coords=dict(time=("time", out_times, {"units": "day"}),
                    z=("z", zs, {"units": "m", "long_name": "elevation (0=base, top=surface)"}),
                    x=("x", xs, {"units": "m", "long_name": "horizontal position (outlet at x=L)"})),
        attrs=dict(
            module="coupling_texture_sweep", soil_key=key, scenario=SOIL_LABEL[key], date=DATE,
            theta_r=soil.theta_r, theta_s=soil.theta_s, vg_alpha=soil.alpha, vg_n=soil.n,
            Ks_m_per_day=soil.Ks, L=L, H=H, nx=NX, nz=NZ, bed_slope_S0=S0, outlet_slope=S0,
            antecedent_psi0_m=psi0, rain_rate_m_per_day=RATE, storm_duration_day=STORM_DUR,
            t_end_day=T_END, n_manning=N_MAN, top_length_m=float(one_top),
            cum_rain_total_m2=float(cum_rain),
            final_infiltrated_m2=infiltrated, final_ponded_m2=ponded, final_drained_m2=drained,
            partition_infiltrated_frac=float(infiltrated / (cum_rain + 1e-30)),
            partition_ponded_frac=float(ponded / (cum_rain + 1e-30)),
            partition_drained_frac=float(drained / (cum_rain + 1e-30)),
            peak_outflow_m2_per_day=float(outflow_q.max()), peak_surface_depth_m=float(d_xt.max()),
            mass_balance_error_max=float(mbe.max()), max_newton_iters=int(iters_rec.max()),
        ),
    )
    out_nc = f"{DATA_DIR}/coupling_texture__{key}__{DATE}.nc"
    ds.to_netcdf(out_nc)
    print(f"WROTE {out_nc}\n  infiltrated={infiltrated:.4f} ponded={ponded:.4f} drained={drained:.4f} "
          f"(cum_rain={cum_rain:.4f})  peak_q={outflow_q.max():.3e}  max_mbe={mbe.max():.2e} "
          f"max_iters={int(iters_rec.max())}", flush=True)
    return out_nc


def main(only=None):
    for key in SOILS:
        if only and key != only:
            continue
        run_soil(key)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
