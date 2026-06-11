#!/usr/bin/env python3
"""Build the tilted-V catchment comparison (in-house vs ParFlow vs the analytic known answer) -- B6.

The canonical Di Giammarco (1996) / Kollet-Maxwell (2006) tilted-V (IH-MIP2 benchmark): two hillslope
planes converging to a central channel that routes to an outlet. Both models run the SAME canonical
1.62 km x 1.0 km case (Sx=5%, Sy=2%, single Manning n=0.015, near-impermeable bed, rain 3e-6 m/s for
90 min then recession) so the outlet hydrograph can be compared to each other AND to the analytic
equilibrium discharge Q_eq = rain * area = 4.86 m^3/s (the known answer for a surface-runoff catchment).

Inputs (npz, in $HOME/parflow-runs/tilted_v/summaries/):
  * ParFlow: tiltedv_s1_diff_1n.npz  (OverlandDiffusive; outlet by STORAGE BALANCE -- ParFlow uses a
    FREE-OUTFLOW overland boundary that evacuates at ~0 depth, so boundary-flux extraction fails, and
    PrintOverlandSum is unreliable -- B5).
  * in-house: tiltedv_inhouse_s1.npz  (diffusion-wave CoupledProblem; outlet by the Manning NORMAL-DEPTH
    add_outflow_bc -> outflow_rate() is direct). A documented outlet-BC formulation delta; both -> Q_eq.

B6 ENVELOPE FINDINGS carried into the metrics/deltas:
  - ParFlow reproduces the known answer (Q -> Q_eq = 4.86 m^3/s exactly) + clean recession.
  - The in-house engine ROUTES the tilted-V but the CONVERGENT channel routing makes the coupled overland
    solve stiff at ANY scale (dt pins ~1e-4, ~1-1.5 hr/run on a 24x16x3 mesh -- NOT a km-scale effect;
    B5's single planar slope was fast). A characterized field-scale-FEM-vs-watershed envelope limit.
  - Both show a cold-start transient (the pre-saturated catchment ponds before steady routing).

Run (pids-fem): python build_comparison_tiltedv.py
"""
from __future__ import annotations

import os

import numpy as np
import xarray as xr

DATE = "2026-06-10"
HERE = os.path.dirname(os.path.abspath(__file__))
PF_DIR = os.path.expanduser("~/parflow-runs/tilted_v/summaries")
N_COMMON = 121

DELTAS = (
    "Both run the canonical tilted-V (1.62x1.0 km, Sx=5% Sy=2%, n=0.015, near-impermeable, rain 3e-6 m/s "
    "90 min + recession) -> both reach the analytic Q_eq = rain*area = 4.86 m^3/s. KEY DELTAS: "
    "(1) OUTLET BC -- in-house imposes a Manning NORMAL-DEPTH outlet (boundary ponds; outflow_rate() is "
    "direct), ParFlow uses a FREE-OUTFLOW overland boundary (evacuates at ~0 depth, outflow only via "
    "storage balance; the B3 small-Manning behaviour, re-confirmed at watershed scale). "
    "(2) WAVE -- both diffusion-wave here (in-house diffusion-wave; ParFlow OverlandDiffusive). "
    "(3) COLD START -- the pre-saturated catchment ponds transiently before steady routing (in-house "
    "wave-arrival overshoot; ParFlow storage-balance spike) -> the rising limb is transient-contaminated; "
    "the EQUILIBRIUM plateau + recession + Q_eq are the clean comparison. "
    "(4) IN-HOUSE ENVELOPE -- two limits found on this steep convergent watershed: (a) the convergent "
    "channel routing makes the coupled overland solve stiff at ANY scale (dt pins ~1e-4, ~20 min/run on "
    "24x16x3; B5's single slope was fast -> it's the convergence, not km scale); (b) MASS CONSERVATION -- "
    "on the steep V the diffusion-wave undershoots d and the overland positivity LIMITER hits its "
    "degenerate dry-the-cell branch, LEAKING ~20-30% of the input, so the in-house plateaus at ~0.68 Q_eq "
    "(leak-corrupted, NOT physics). The in-house is a field-scale drainage FEM; steep km-watershed routing "
    "is outside its robustness envelope. ParFlow (purpose-built kinematic-wave watershed router) handles it."
)


def _load(name):
    z = np.load(os.path.join(PF_DIR, name))
    return {k: z[k] for k in z.files}


def main():
    pf = _load("tiltedv_s1_diff_1n.npz")
    ih = _load("tiltedv_inhouse_s1.npz")
    Q_eq = float(pf["Q_eq"])
    storm = float(pf["storm"]); t_end = float(pf["t_end"])
    t = np.linspace(0.0, t_end, N_COMMON)

    q_pf = np.interp(t, pf["times"], pf["q_out"])
    q_ih = np.interp(t, ih["times"], ih["q_out"])
    rain_area = float(pf["rain"]) * float(pf["LX"]) * float(pf["LY"])
    rain_t = np.where(t <= storm + 1e-9, rain_area, 0.0)
    # cumulative outflow (mass check): both should approach cum_rain - (small infiltration/storage)
    cum_pf = np.concatenate([[0.0], np.cumsum(0.5 * (q_pf[1:] + q_pf[:-1]) * np.diff(t))])
    cum_ih = np.concatenate([[0.0], np.cumsum(0.5 * (q_ih[1:] + q_ih[:-1]) * np.diff(t))])
    cum_rain = rain_area * np.minimum(t, storm)

    # equilibrium plateau (mean over the late-storm window, away from the cold-start transient)
    pl = (t > 0.5 * storm) & (t <= storm)
    plat_pf = float(np.mean(q_pf[pl])); plat_ih = float(np.mean(q_ih[pl]))

    # in-house MASS-BALANCE check (cum_rain = cum_out + surf storage + soil storage + LEAK). On the steep
    # convergent V the diffusion-wave undershoots d and the overland POSITIVITY LIMITER hits its degenerate
    # "dry the cell" branch -> mass loss; the 0.68 Q_eq plateau is ~20-30% mass-leak-corrupted, not physics.
    surf_ih = np.interp(t, ih["times"], ih["surface_water"])
    soil_ih = np.interp(t, ih["times"], ih["soil_dw"])
    i_se = int(np.argmin(np.abs(t - storm)))
    leak_se = float(cum_rain[i_se] - cum_ih[i_se] - surf_ih[i_se] - soil_ih[i_se])
    inhouse_leak_frac = leak_se / max(cum_rain[i_se], 1e-9)

    m = dict(
        Q_eq_m3day=Q_eq, Q_eq_m3s=Q_eq / 86400.0,
        plateau_parflow_m3day=plat_pf, plateau_inhouse_m3day=plat_ih,
        plateau_parflow_over_Qeq=plat_pf / Q_eq, plateau_inhouse_over_Qeq=plat_ih / Q_eq,
        peak_parflow_over_Qeq=float(np.max(q_pf) / Q_eq), peak_inhouse_over_Qeq=float(np.max(q_ih) / Q_eq),
        inhouse_mass_leak_frac=inhouse_leak_frac,
        inhouse_run_minutes=float(ih.get("run_s", np.array(0.0))) / 60.0,
        inhouse_steps=int(ih.get("n_step", np.array(0))),
        inhouse_mesh=f"{int(ih['NX'])}x{int(ih['NY'])}x{int(ih['NZ'])}",
        parflow_grid=f"{int(pf['NX'])}x{int(pf['NY'])}",
    )

    ds = xr.Dataset(
        data_vars=dict(
            q_inhouse=(("time",), q_ih, {"units": "m^3/day", "long_name": "outlet discharge (in-house, Manning outlet)"}),
            q_parflow=(("time",), q_pf, {"units": "m^3/day", "long_name": "outlet discharge (ParFlow, storage balance)"}),
            cum_inhouse=(("time",), cum_ih, {"units": "m^3"}),
            cum_parflow=(("time",), cum_pf, {"units": "m^3"}),
            cum_rain=(("time",), cum_rain, {"units": "m^3"}),
            rain_rate=(("time",), rain_t, {"units": "m^3/day"}),
        ),
        coords=dict(time=("time", t, {"units": "day"})),
        attrs=dict(
            case="tilted_v_canonical", date=DATE,
            title="Canonical tilted-V catchment (1.62x1.0 km): in-house vs ParFlow vs analytic Q_eq",
            domain_km="1.62x1.0", slope_x=0.05, slope_y=0.02, manning_n=0.015,
            rain_m_per_s=3e-6, storm_min=90.0, deltas=DELTAS, **m,
        ),
    )
    out_nc = os.path.join(HERE, "data", f"tilted_v__canonical__{DATE}.nc")
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ds.to_netcdf(out_nc)

    from make_comparison_tiltedv_html import build as build_html
    out_html = os.path.join(HERE, "html", f"tilted_v__canonical__{DATE}.html")
    build_html(out_nc, out_html)

    print("=== B6 tilted-V: in-house vs ParFlow vs Q_eq ===")
    print(f"  Q_eq = {Q_eq:.0f} m^3/day (~{Q_eq/86400:.3f} m^3/s)")
    print(f"  equilibrium plateau (late storm):  ParFlow {plat_pf/Q_eq:.3f} Q_eq   in-house {plat_ih/Q_eq:.3f} Q_eq")
    print(f"  in-house MASS LEAK at storm-end: {inhouse_leak_frac*100:.1f}% (positivity-limiter on the steep V "
          f"-> plateau is leak-corrupted)")
    print(f"  in-house run: {m['inhouse_run_minutes']:.1f} min, {m['inhouse_steps']} steps, mesh {m['inhouse_mesh']}")
    print(f"  -> WROTE {os.path.relpath(out_nc, HERE)}  +  {os.path.relpath(out_html, HERE)}")


if __name__ == "__main__":
    main()
