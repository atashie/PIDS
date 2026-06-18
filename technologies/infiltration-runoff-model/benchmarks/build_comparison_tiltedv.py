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

B6 ENVELOPE FINDINGS carried into the metrics/deltas (P0-CORRECTED 2026-06-11, see the
stabilization plan SS9 -- the original "0.676 Q_eq plateau / 20% mass-ledger gap" was NOT
reproducible from the committed deck and is RETRACTED):
  - ParFlow reproduces the known answer (Q -> Q_eq = 4.86 m^3/s exactly) + clean recession.
  - The in-house engine ALSO reaches the known answer (plateau ~1.0 Q_eq) with the ENGINE-EXACT
    per-step ledger closed to ~roundoff (the npz carries cum_rain/cum_out/ext_gap from the
    accepted-step books; the 40-point trapz reconstruction below differs by a few % -- SAMPLING
    of the wave-arrival spike, not mass loss).
  - What remains real on the convergent V: scale-independent STIFFNESS (dt pins ~5e-5..1e-4 d;
    measured mechanism = wet/dry sawtooth undershoots fire the global-rescale positivity limiter
    every step and Newton re-equilibrates the perturbation -- the P1 upwind flux removes the
    undershoots at the source), plus a cold-start transient in both models.

Run (pids-fem): python build_comparison_tiltedv.py
"""
from __future__ import annotations

import os

import numpy as np
import xarray as xr

DATE = "2026-06-11"  # the P0-correction date (the 2026-06-10 artifact embedded the retracted numbers)
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
    "the EQUILIBRIUM plateau + recession + Q_eq are the clean comparison. The in-house CUMULATIVE-outflow "
    "trace here is the 40-point trapz reconstruction -- it under/over-samples that spike by a few %; the "
    "ENGINE per-accepted-step ledger (metrics) is the honest mass statement. "
    "(4) IN-HOUSE ENVELOPE (P0-corrected 2026-06-11, adversarially reviewed) -- the earlier B6 text "
    "attributed a 20% mass-ledger gap to the positivity limiter's degenerate branch and a 0.676 Q_eq "
    "plateau; P0 of the stabilization plan could NOT reproduce either from the committed deck (engine "
    "storm-window books ~1e-12 of cum rain under BOTH dt-controller settings, full-window spike runs "
    "<=1e-11; plateau ~1.0 Q_eq; the published npz recorded NET soil DRYING (-26 m^3) under sustained "
    "ponded rain, sign-opposite to the committed-deck rerun (+47 m^3) = corrupted booked states; "
    "RETRACTED -- provenance unknown: all npz-recorded params match deck defaults, the unrecorded "
    "variables are controller knobs / comm size / in-session code; original preserved as "
    "tiltedv_inhouse_s1_pre_p0_corrupt.npz; runner npz now records the knobs). What REMAINS real on "
    "the convergent V: (a) scale-independent STIFFNESS -- dt pins ~5e-5..1e-4 d at 1.6 km AND 162 m: "
    "the wet/dry sawtooth-and-clip state never settles, so every step costs ~4-6+ Newton iterations "
    "and the controller's growth threshold pins dt (causal control: bypassing the limiter does NOT "
    "drop iterations and Newton then fails -- the limiter is load-bearing; 2x dt costs +1 iteration, "
    "a throughput cost not a wall; the planned P1 upwind-mobility flux removes the sawtooth at the "
    "source); (b) FIELD-SCALE ACCURACY -- at 162 m the clips (~1.3 cm) exceed the ~3 mm equilibrium "
    "sheet: coupled 24x16 plateau 0.876 Q_eq with closed books, healing to 1.01 at 48x30 standalone "
    "-> the upwind flux is accuracy-critical for the PIDS swale regime; (c) step-acceptance hardening "
    "landed (stagnation verdicts bookable only at the residual floor -- dirty stalled-line-search "
    "states, |F| up to ~3e-3 observed here, are now honest rejections). ParFlow (purpose-built "
    "kinematic watershed router) needs neither limiter nor rejection."
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

    # in-house MASS BOOKS. The honest statement is the ENGINE per-accepted-step ledger saved in the
    # npz (cum_rain_engine = cum_out_engine + dw_engine + ext_gap_engine, ext_gap ~ roundoff after the
    # P0 acceptance hardening). The 40-point trapz reconstruction (cum_rain - trapz(q) - dstorage at
    # storm end) is ALSO reported -- its few-% residue is SAMPLING of the wave-arrival spike, not mass
    # loss (P0-corrected 2026-06-11; the earlier 20%-leak attribution is retracted).
    surf_ih = np.interp(t, ih["times"], ih["surface_water"])
    soil_ih = np.interp(t, ih["times"], ih["soil_dw"])
    i_se = int(np.argmin(np.abs(t - storm)))
    leak_se = float(cum_rain[i_se] - cum_ih[i_se] - surf_ih[i_se] - soil_ih[i_se])
    inhouse_trapz_residue_frac = leak_se / max(cum_rain[i_se], 1e-9)
    cr_eng = float(ih.get("cum_rain_engine", np.array(np.nan)))
    inhouse_ledger_gap_engine_frac = float(ih.get("ext_gap_engine", np.array(np.nan))) / max(cr_eng, 1e-9)

    m = dict(
        Q_eq_m3day=Q_eq, Q_eq_m3s=Q_eq / 86400.0,
        plateau_parflow_m3day=plat_pf, plateau_inhouse_m3day=plat_ih,
        plateau_parflow_over_Qeq=plat_pf / Q_eq, plateau_inhouse_over_Qeq=plat_ih / Q_eq,
        peak_parflow_over_Qeq=float(np.max(q_pf) / Q_eq), peak_inhouse_over_Qeq=float(np.max(q_ih) / Q_eq),
        inhouse_ledger_gap_engine_frac=inhouse_ledger_gap_engine_frac,
        inhouse_trapz_residue_frac=inhouse_trapz_residue_frac,
        inhouse_clip_mass_adjust_m3=float(ih.get("clip_mass_adjust", np.array(np.nan))),
        inhouse_rejected_steps=int(ih.get("n_rej", np.array(-1))),
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

    print("=== B6 tilted-V: in-house vs ParFlow vs Q_eq (P0-corrected) ===")
    print(f"  Q_eq = {Q_eq:.0f} m^3/day (~{Q_eq/86400:.3f} m^3/s)")
    print(f"  equilibrium plateau (late storm):  ParFlow {plat_pf/Q_eq:.3f} Q_eq   in-house {plat_ih/Q_eq:.3f} Q_eq")
    print(f"  in-house ENGINE ledger gap: {inhouse_ledger_gap_engine_frac*100:+.2e}% of cum rain "
          f"(clip_mass_adjust={m['inhouse_clip_mass_adjust_m3']:.3f} m^3, {m['inhouse_rejected_steps']} rejected steps)")
    print(f"  in-house 40-pt trapz reconstruction residue at storm-end: {inhouse_trapz_residue_frac*100:+.1f}% "
          f"(hydrograph SAMPLING, not mass loss)")
    print(f"  in-house run: {m['inhouse_run_minutes']:.1f} min, {m['inhouse_steps']} steps, mesh {m['inhouse_mesh']}")
    print(f"  -> WROTE {os.path.relpath(out_nc, HERE)}  +  {os.path.relpath(out_html, HERE)}")


if __name__ == "__main__":
    main()
