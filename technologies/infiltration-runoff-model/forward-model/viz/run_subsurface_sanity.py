"""Sanity run: subsurface storm-infiltration scenario -> standardized NetCDF result.

Produces a self-describing result file (xarray / NetCDF -- the viz data contract of
governance/visualize-sanity-check-routine.md) for the Tier-3 interactive visualization.
The viz generator reads ONLY this file and never imports the solver.

Scenario: a 1-D loam column, initial psi = -2 m, driven by an intense rainfall flux
(q = 1 m/day ~ 4x Ks) so a wetting front advances to saturation/ponding at the top.
"""
from __future__ import annotations

import os

import numpy as np
import xarray as xr
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import RichardsProblem

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)
DATE = "2026-06-04"


def run(out_dir: str) -> str:
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 60)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0])
    q = 1.0  # m/day intense storm (~4x Ks)
    prob.add_flux_bc(lambda x: np.isclose(x[0], 1.0), q)

    z = prob.V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(z)
    z = z[order]
    times = np.round(np.linspace(0.0, 0.2, 21), 6)
    s0 = prob.total_water()

    psi_rec = [prob.psi.x.array[order].copy()]
    theta_rec = [prob.theta_array()[order].copy()]
    mb_rec = [0.0]
    prev = 0.0
    for tk in times[1:]:
        prob.advance(t_end=float(tk - prev), dt=0.01)
        prev = tk
        psi_rec.append(prob.psi.x.array[order].copy())
        theta_rec.append(prob.theta_array()[order].copy())
        dS = prob.total_water() - s0
        mb_rec.append(abs(dS - q * tk) / (q * tk))

    ds = xr.Dataset(
        data_vars=dict(
            head=(("time", "z"), np.array(psi_rec),
                  {"units": "m", "long_name": "pressure head psi"}),
            water_content=(("time", "z"), np.array(theta_rec),
                           {"units": "m3/m3", "long_name": "volumetric water content theta"}),
            mass_balance_error=(("time",), np.array(mb_rec),
                                {"units": "-", "long_name": "relative open-system mass-balance error"}),
        ),
        coords=dict(
            time=("time", times, {"units": "day"}),
            z=("z", z, {"units": "m", "long_name": "elevation (0 = bottom, 1 = top)"}),
        ),
        attrs=dict(
            module="subsurface (mixed-form Richards)",
            scenario="intense storm infiltration (q=1 m/day ~ 4x Ks) into a psi=-2 m loam column",
            date=DATE,
            soil="Carsel & Parrish (1988) loam",
            theta_r=LOAM["theta_r"], theta_s=LOAM["theta_s"], Ks_m_per_day=LOAM["Ks"],
            rain_flux_m_per_day=q,
            cumulative_input_m=float(q * times[-1]),
            mass_balance_error_max=float(np.max(mb_rec)),
        ),
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"subsurface__infiltration__{DATE}.nc")
    ds.to_netcdf(path)
    print(f"WROTE {path}  (mass_balance_error_max={float(np.max(mb_rec)):.2e})")
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "..", "validation", "sanity", "data"))
    run(out)
