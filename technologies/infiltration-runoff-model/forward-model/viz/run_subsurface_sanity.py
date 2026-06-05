"""Sanity runs: subsurface forcing scenarios -> standardized NetCDF results.

Produces self-describing result files (xarray / NetCDF -- the viz data contract of
governance/visualize-sanity-check-routine.md) for the Tier-3 interactive viz. The
viz generator (make_sanity_html.py) reads ONLY these files and never imports the solver.

All scenarios use the PONDING boundary condition: rainfall infiltrates, and any excess
the soil cannot take accumulates as a ponded depth that raises the surface pressure head
(vertical accumulation only -- lateral routing is the overland module). Mass-conserving:
rainfall = soil-storage change + ponded-depth change.

Four scenarios spanning antecedent wetness x event size on a 1-D loam column:
  typical-mesic : typical storm on mesic (moderately-moist) soil
  intense-dry   : extreme intense storm on dry soil
  intense-wet   : extreme intense storm on wet soil (saturation-excess -> ponding)
  small-mesic   : very small event on mesic soil
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

# psi0 (m) = antecedent head; q (m/day) = rainfall flux at the top; T (day) = duration.
SCENARIOS = {
    "typical-mesic": dict(psi0=-1.0, q=0.10, T=0.50,
                          antecedent="mesic (psi=-1 m)",
                          event="typical storm (0.10 m/day, below Ks)"),
    "intense-dry": dict(psi0=-5.0, q=2.00, T=0.12,
                        antecedent="dry (psi=-5 m)",
                        event="extreme intense storm (2.0 m/day ~ 8x Ks)"),
    "intense-wet": dict(psi0=-0.3, q=2.00, T=0.12,
                        antecedent="wet (psi=-0.3 m)",
                        event="extreme intense storm (2.0 m/day ~ 8x Ks)"),
    "small-mesic": dict(psi0=-1.0, q=0.02, T=0.50,
                        antecedent="mesic (psi=-1 m)",
                        event="very small event (0.02 m/day)"),
}


def run_scenario(key: str, cfg: dict, out_dir: str) -> str:
    psi0, q, T = cfg["psi0"], cfg["q"], cfg["T"]
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 60)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0])
    prob.add_ponding_bc(lambda x: np.isclose(x[0], 1.0), q)  # rain + ponding at the top

    z = prob.V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(z)
    z = z[order]
    times = np.round(np.linspace(0.0, T, 21), 6)
    s0 = prob.total_water() + prob.ponded_depth()

    psi_rec = [prob.psi.x.array[order].copy()]
    theta_rec = [prob.theta_array()[order].copy()]
    pond_rec = [prob.ponded_depth()]
    mb_rec = [0.0]
    prev = 0.0
    for tk in times[1:]:
        prob.advance(t_end=float(tk - prev), dt=0.005)
        prev = tk
        psi_rec.append(prob.psi.x.array[order].copy())
        theta_rec.append(prob.theta_array()[order].copy())
        pond = prob.ponded_depth()
        pond_rec.append(pond)
        dS = (prob.total_water() + pond) - s0
        mb_rec.append(abs(dS - q * tk) / (q * tk))

    ds = xr.Dataset(
        data_vars=dict(
            head=(("time", "z"), np.array(psi_rec),
                  {"units": "m", "long_name": "pressure head psi"}),
            water_content=(("time", "z"), np.array(theta_rec),
                           {"units": "m3/m3", "long_name": "volumetric water content theta"}),
            ponded_depth=(("time",), np.array(pond_rec),
                          {"units": "m", "long_name": "surface ponded depth max(psi,0)"}),
            mass_balance_error=(("time",), np.array(mb_rec),
                                {"units": "-", "long_name": "relative mass-balance error (soil + pond)"}),
        ),
        coords=dict(
            time=("time", times, {"units": "day"}),
            z=("z", z, {"units": "m", "long_name": "elevation (0 = bottom, 1 = top)"}),
        ),
        attrs=dict(
            module="subsurface (mixed-form Richards)",
            scenario=f"{cfg['event']} on {cfg['antecedent']} soil",
            date=DATE,
            soil="Carsel & Parrish (1988) loam",
            theta_r=LOAM["theta_r"], theta_s=LOAM["theta_s"], Ks_m_per_day=LOAM["Ks"],
            rain_flux_m_per_day=q,
            cumulative_input_m=float(q * T),
            ponded_depth_max_m=float(np.max(pond_rec)),
            mass_balance_error_max=float(np.max(mb_rec)),
        ),
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"subsurface__{key}__{DATE}.nc")
    ds.to_netcdf(path)
    print(f"WROTE {path}  (pond_max={float(np.max(pond_rec)):.4f} m, mbe_max={float(np.max(mb_rec)):.2e})")
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "..", "validation", "sanity", "data"))
    for key, cfg in SCENARIOS.items():
        run_scenario(key, cfg, out)
