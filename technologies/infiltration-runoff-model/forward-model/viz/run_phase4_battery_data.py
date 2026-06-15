"""Emit the standardized result data for the M4 Phase-4 COUPLED-EMBEDDING deployment gate battery.

THE quantitative validation series for the 2026-06-12 state of the embedded feature exchange
(production pids_forward/physics/wi_exchange.py: disperse rate-clock+WI, drain theta-mean+Heun
PSS): EVERY committed deployment leg, embedded at BOTH n=8 and n=12, scored against its resolved
reference AND its discrimination twin, with the physical-realism bounds asserted:

  DISPERSE (dry soil psi_i=-1, saturated wall 0):    twin = the offline cyl-GA clock
    LOAM R40 full depletion / SAND R40 / SILT R40 / LOAM RefB-40 re-wetting history
  DRAIN (near-saturated bulk -0.03, wall -1):        twin = the fixed-drive PSS clock
    LOAM refD40 / SAND R40 / SILT R40 / LOAM R20 DEEP (48%) /
    LOAM refD40-C CONTINUOUS RECHARGE                twin = the recharge-BLIND water-balance PSS

  Asserted per leg: embedded rel-L2 <= EMBEDDED_TOL (0.10) at BOTH n; twin rel-L2 >=
  BASELINE_KILL (0.20); embedded I(t) monotone; capacity bound I_end <= I_max (+ injected, for
  the history legs); both mass ledgers hold machine-tight on every sample INSIDE the harness.

Writes the .npz (all curves) + JSON sidecar (metrics/verdict) data contract; the separate
visualizer (viz/make_phase4_battery_html.py) reads ONLY these files.

Run from forward-model/ (WSL):  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python viz/run_phase4_battery_data.py        (~1 h: 18 closed-box FEM runs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FM))

from scratch.m4_phase4_embedded_harness import run_embedded, R_W, SOILS                 # noqa: E402
from scratch.m4_phase4_drain_desorptivity import bruce_klute_desorptivity, pss_drain    # noqa: E402
from pids_forward.physics.wi_exchange import WellIndexExchange                          # noqa: E402
from pids_forward.physics.sorptive_closure import F_cylindrical, sorptive_clock, rel_l2  # noqa: E402

DATE = "2026-06-14"
FIX = FM / "tests" / "data"
SCR = FM / "scratch"
OUT = FM.parent / "validation" / "sanity" / "data"
OUT.mkdir(parents=True, exist_ok=True)
EMBEDDED_TOL, BASELINE_KILL = 0.10, 0.20
DISPERSE_TOL = 0.03     # item-A pre-registered target (2026-06-14): the resolved-ring WI read
NS = (8, 12)


def clock_twin(t, npz_S, npz_dth):
    return sorptive_clock(t, float(npz_S), float(npz_dth), R_W, F_cylindrical)


def fixed_drive_twin(soil_name, t, R_factor, i_max):
    geo = np.log(float(R_factor)) - 0.75
    rate = float(SOILS[soil_name].kirchhoff(-1.0, -0.03)) / (R_W * geo)
    return np.minimum(rate * (t - t[0]), i_max)


def main():
    a = np.load(FIX / "m4_phase4_refA_disperse.npz")
    a40 = np.load(FIX / "m4_phase4_refA40_sand_silt.npz")
    b40 = np.load(FIX / "m4_phase4_refB40_disperse.npz")
    refD = np.load(SCR / "m4_phase4_refD40_drain.npz")
    fresh = np.load(SCR / "m4_phase4_drain_fresh_refs.npz")
    silt = np.load(SCR / "m4_phase4_silt_drain_ref.npz")
    refC = np.load(SCR / "m4_phase4_refD40C_drain.npz")

    S_bk_loam = bruce_klute_desorptivity(SOILS["LOAM"], -1.0, -0.03)[0]

    b40_pulse = (float(b40["LOAM_t_pulse"][0]), float(b40["LOAM_t_pulse"][1]),
                 float(b40["LOAM_V_pulse_per_wall_area"]) * R_W * 2 * np.pi)
    b40_band = tuple(float(x) for x in b40["LOAM_band"])
    c_pulse = (float(refC["LOAM_t_src"][0]), float(refC["LOAM_t_src"][1]),
               float(refC["LOAM_V_per_wall_area"]) * 2 * np.pi * R_W)
    c_band = tuple(float(x) for x in refC["LOAM_band"])
    tC = refC["LOAM_t"]
    c_twin = pss_drain(tC, SOILS["LOAM"], -0.03, -1.0, R_W, 40 * R_W,
                       S_bk_loam * np.sqrt(tC[0]), float(tC[0]))

    # leg = (key, direction, soil, R_factor, t, I_ref, I_max, injected_extra, pulse, band,
    #        twin_name, I_twin)
    LEGS = [
        ("disp_LOAM_R40", "disperse", "LOAM", 40, a["LOAM_R40_t"], a["LOAM_R40_I"],
         float(a["LOAM_R40_Imax"]), 0.0, None, None,
         "offline cyl-GA clock", clock_twin(a["LOAM_R40_t"], a["LOAM_S"], a["LOAM_dtheta"])),
        ("disp_SAND_R40", "disperse", "SAND", 40, a40["SAND_R40_t"], a40["SAND_R40_I"],
         float(a40["SAND_R40_Imax"]), 0.0, None, None,
         "offline cyl-GA clock", clock_twin(a40["SAND_R40_t"], a40["SAND_S"], a40["SAND_dtheta"])),
        ("disp_SILT_R40", "disperse", "SILT", 40, a40["SILT_R40_t"], a40["SILT_R40_I"],
         float(a40["SILT_R40_Imax"]), 0.0, None, None,
         "offline cyl-GA clock", clock_twin(a40["SILT_R40_t"], a40["SILT_S"], a40["SILT_dtheta"])),
        ("disp_LOAM_RefB40", "disperse", "LOAM", 40, b40["LOAM_t"], b40["LOAM_I"],
         float(b40["LOAM_Imax"]), float(b40["LOAM_V_pulse_per_wall_area"]), b40_pulse, b40_band,
         "offline cyl-GA clock", clock_twin(b40["LOAM_t"], a["LOAM_S"], a["LOAM_dtheta"])),
        ("drain_LOAM_refD40", "drain", "LOAM", 40, refD["LOAM_t"], refD["LOAM_I"],
         float(refD["LOAM_Imax"]), 0.0, None, None,
         "fixed-drive PSS clock",
         fixed_drive_twin("LOAM", refD["LOAM_t"], 40, float(refD["LOAM_Imax"]))),
        ("drain_SAND_R40", "drain", "SAND", 40, fresh["SAND_R40_t"], fresh["SAND_R40_I"],
         float(fresh["SAND_R40_Imax"]), 0.0, None, None,
         "fixed-drive PSS clock",
         fixed_drive_twin("SAND", fresh["SAND_R40_t"], 40, float(fresh["SAND_R40_Imax"]))),
        ("drain_SILT_R40", "drain", "SILT", 40, silt["SILT_R40_t"], silt["SILT_R40_I"],
         float(silt["SILT_R40_Imax"]), 0.0, None, None,
         "fixed-drive PSS clock",
         fixed_drive_twin("SILT", silt["SILT_R40_t"], 40, float(silt["SILT_R40_Imax"]))),
        ("drain_LOAM_R20deep", "drain", "LOAM", 20, fresh["LOAM_R20_t"], fresh["LOAM_R20_I"],
         float(fresh["LOAM_R20_Imax"]), 0.0, None, None,
         "fixed-drive PSS clock",
         fixed_drive_twin("LOAM", fresh["LOAM_R20_t"], 20, float(fresh["LOAM_R20_Imax"]))),
        ("drain_LOAM_refD40C", "drain", "LOAM", 40, tC, refC["LOAM_I"],
         float(refC["LOAM_Imax"]), float(refC["LOAM_V_per_wall_area"]), c_pulse, c_band,
         "recharge-BLIND water-balance PSS", c_twin),
    ]

    # the disperse t0 seed (S*sqrt(t0), ~2e-4 of capacity): part of the documented mass identity
    # I_total*A == injected + seed*A -- it is start-of-clock water already in the ground, so the
    # capacity bound for the disperse legs is I_max + seed (+ injected pulse); drain has no seed
    S_BY_LEG = {"disp_LOAM_R40": float(a["LOAM_S"]), "disp_SAND_R40": float(a40["SAND_S"]),
                "disp_SILT_R40": float(a40["SILT_S"]), "disp_LOAM_RefB40": float(a["LOAM_S"])}

    arrays, rows = {}, []
    for (key, direction, soil, Rf, t, I_ref, i_max, v_extra, pulse, band,
         twin_name, I_twin) in LEGS:
        seed = S_BY_LEG.get(key, 0.0) * np.sqrt(float(t[0]))
        arrays[f"{key}_t"] = t
        arrays[f"{key}_Iref"] = np.asarray(I_ref)
        arrays[f"{key}_Itwin"] = np.asarray(I_twin)
        arrays[f"{key}_Imax"] = np.array(i_max)
        arrays[f"{key}_Vextra"] = np.array(v_extra)
        if pulse is not None:
            arrays[f"{key}_t_src"] = np.array(pulse[:2])
        e_twin = rel_l2(I_twin, I_ref)
        row = dict(leg=key, direction=direction, soil=soil, R_factor=Rf,
                   twin_name=twin_name, twin_rel_l2=round(float(e_twin), 4),
                   depletion_frac=round(float(I_ref[-1] / i_max), 4))
        for n in NS:
            kw = dict(direction=direction)
            if pulse is not None:
                kw.update(pulse=pulse, pulse_band=band)
            out = run_embedded(WellIndexExchange(direction=direction), soil, Rf * R_W, n, t, **kw)
            assert out is not None, f"{key} n={n}: run did not complete"
            I_emb = out["I"]
            e = rel_l2(I_emb, I_ref)
            tol = DISPERSE_TOL if direction == "disperse" else EMBEDDED_TOL
            assert e <= tol, f"{key} n={n}: relL2 {e:.1%} > {tol:.0%}"
            assert np.all(np.diff(I_emb) >= -1e-12 * i_max), f"{key} n={n}: not monotone"
            # the box's ACTUAL capacity: the harness box matches the annulus REMOVABLE volume
            # (L^2 = pi*(R^2-r_w^2)) but has no excluded feature core, so it holds an extra
            # r_w^2/(R^2-r_w^2) of I_max (~0.06% at R40); plus the disperse seed + any pulse
            core = i_max * R_W ** 2 / ((Rf * R_W) ** 2 - R_W ** 2)
            assert I_emb[-1] <= (i_max + core + v_extra + seed) * (1 + 1e-9), \
                f"{key} n={n}: capacity violated"
            arrays[f"{key}_Iemb_n{n}"] = I_emb
            row[f"n{n}_rel_l2"] = round(float(e), 4)
            row[f"n{n}_end_ratio"] = round(float(I_emb[-1] / I_ref[-1]), 4)
            print(f"  [{key}] n={n}: relL2={e:.1%}  end={I_emb[-1]/I_ref[-1]:.3f}  "
                  f"(twin {twin_name}: {e_twin:.1%})", flush=True)
        assert e_twin >= BASELINE_KILL, f"{key}: twin kill lost ({e_twin:.1%})"
        row["status"] = "PASS"
        rows.append(row)

    worst_emb = max(max(r["n8_rel_l2"], r["n12_rel_l2"]) for r in rows)
    worst_disp = max((max(r["n8_rel_l2"], r["n12_rel_l2"]) for r in rows
                      if r["direction"] == "disperse"), default=0.0)
    worst_kill = min(r["twin_rel_l2"] for r in rows)
    meta = dict(
        check="M4 Phase-4 coupled-embedding deployment gate battery (production WellIndexExchange)",
        module="Module 4 (sec E) Phase 4 + follow-ups + item A", date=DATE,
        metric="rel-L2 of cumulative wall-exchange I(t) vs the resolved closed-domain reference",
        tolerance=dict(embedded=EMBEDDED_TOL, disperse=DISPERSE_TOL, baseline_kill=BASELINE_KILL),
        verdict=dict(all_pass=all(r["status"] == "PASS" for r in rows),
                     worst_embedded_rel_l2=round(float(worst_emb), 4),
                     worst_disperse_rel_l2=round(float(worst_disp), 4),
                     worst_twin_kill=round(float(worst_kill), 4),
                     note=("Host control: drain = live theta-mean water-balance drive (every "
                           "twin without the live host read fails); disperse = WI era only "
                           "(>=80-91% of I; the clock era is a prescribed-rate closure). The "
                           "disperse WI-era residual (5.7% with the on-ridge read) is RESOLVED "
                           "(item A, 2026-06-14): the WI-era bridge reads the RESOLVED host field "
                           "at the catchment-radius midpoint R_out/2 (Heun-corrected prescribed "
                           "rate + live recharge-aware capacity throttle), dropping the disperse "
                           "worst to <=2.6% across n=8/12 + the soil triad + the RefB40 history "
                           "leg. Scope: serial meshes, homogeneous isotropic soils, saturated "
                           "disperse wall, positive-WI regime (h > ~5.5 r_w). The drain '+2-4% end "
                           "over-bias' is RESOLVED (item B, 2026-06-15): it was a REFERENCE "
                           "first-order backward-Euler under-count (~4%), NOT a law model-form error "
                           "-- the closed-drain refs were regenerated at a converged dt cap, and the "
                           "offline PSS law is accurate/slightly conservative (the small Jensen "
                           "volume-average under).")),
        legs=rows,
        ledgers=("both mass ledgers asserted machine-tight on every sample inside the harness: "
                 "host theta-gain == exchanged + injected (<=1e-6 rel); scheme I*A == "
                 "extracted/injected + reservoir (<=1e-4 rel)"),
    )
    np.savez(OUT / f"m4_phase4_battery__{DATE}.npz", **arrays)
    with open(OUT / f"m4_phase4_battery__{DATE}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\nALL {len(rows)} LEGS PASS (worst embedded {worst_emb:.1%}, worst disperse "
          f"{worst_disp:.1%}, worst kill {worst_kill:.1%})")
    print(f"Saved -> {OUT}/m4_phase4_battery__{DATE}.npz + .json")


if __name__ == "__main__":
    main()
