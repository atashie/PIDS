"""Ref A-40 SAND + SILT: the deployment-regime disperse references for the SOIL-GENERALITY axis.

Follow-up #2a of the 2026-06-12 list ([[pids-m4-phase4-followups-kickoff]]): the committed R40
deployment legs are LOAM-only (the 2026-06-10 axis decision -- "LOAM carries this axis"); the
production WellIndexExchange disperse gate evidence is therefore single-soil. This generator adds
SAND and SILT at R = 40 r_w with the SAME machinery and window auto-tune rules as the committed
Ref-A generator (scratch/m4_phase4_refA_disperse.py: full-depletion discriminator -- final
>= 97% of I_max, 90% crossing in the first 3/4 of the window; early window must match the
signed-off Phase-1b infinite-domain fixture < 1.5%), saved ADDITIVELY to a new npz so the
committed byte-stable fixtures are never regenerated (the pre-registration review's provenance
property). Window first guesses scale the LOAM-R40 70-d window by the per-soil Phase-1b ratios
(SAND 3e-3/6e-2 -> ~3.5 d, SILT 4e-1/6e-2 -> ~470 d).

Printed for the record (the discrimination battery on the new legs): offline-clock and
capacity-clamped-clock rel-L2 -- the raw clock must FAIL (the depleting reservoir bends, the
clock does not), the clamp documents the known refA-alone evasion (killed by the drain legs).

Run from forward-model/ (WSL):
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scratch/m4_phase4_refA40_sand_silt.py
"""
from __future__ import annotations

import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import scratch.m4_phase1b_disperse_reference as dz
from scratch.m4_phase4_refA_disperse import (
    run_depleting, R_W, PSI_I, PSI_WALL, N_SAMP, T90_FRAC_MAX)
from pids_forward.physics.sorptive_closure import F_cylindrical, sorptive_clock, rel_l2

R_FACTOR = 40
# SAND first guess 3.5 d landed t90/t_end = 0.65 (within the <= 0.75 rule) but left the clock
# kill at 19.5% -- BELOW the 0.20 bar (the bend was too late-window; the discrimination lives in
# the deep bend + plateau). 14 d puts the 90% crossing mid-window, the retune rule's design
# target, mirroring the LOAM-R40 structure (clock kill 26.6% there). Chosen BEFORE the embedded
# scheme was scored on the regenerated leg (the scheme had already passed the SHORTER window at
# 2.2-2.8%, so the extension tightens the gate, it cannot protect the scheme).
T_END0 = {"SAND": 14.0, "SILT": 470.0}
HERE = pathlib.Path(__file__).parent


if __name__ == "__main__":
    fix = np.load("tests/data/m4_phase1b_disperse_refs.npz")
    out_path = HERE / "m4_phase4_refA40_sand_silt.npz"
    names = tuple(sys.argv[1:]) or ("SAND", "SILT")
    saved = dict(np.load(out_path)) if out_path.exists() else {}   # selective re-gen merges
    saved["r_w"] = np.array(R_W)
    r_out = R_FACTOR * R_W
    print("=" * 88)
    print(f"Ref A-40 {'+'.join(names)} (deployment regime, R={R_FACTOR} r_w, full-depletion windows)")
    print("=" * 88)
    for name in names:
        soil = dz.SOILS[name]
        S_an = dz.parlange_sorptivity(soil, PSI_I)
        dth = float(soil.theta(PSI_WALL) - soil.theta(PSI_I))
        t_start = (dth * 0.1 * R_W / S_an) ** 2
        i_max = dth * (r_out ** 2 - R_W ** 2) / (2.0 * R_W)
        t_end = T_END0[name]
        print(f"\n{name}: S_an={S_an:.5f} m/day^0.5  dtheta={dth:.4f}  I_max={i_max:.3f} m")
        for attempt in range(4):
            t = np.geomspace(t_start, t_end, N_SAMP)
            I = run_depleting(soil, r_out, t, label=f"{name} R40 (t_end={t_end:.3g})")
            frac = I[-1] / i_max
            if frac < 0.97:
                t_end *= 3.0
                print(f"   depleted {frac:.1%} < 97% -> t_end x3 = {t_end:.3g}", flush=True)
                continue
            t90 = float(np.interp(0.90, I / i_max, t))
            if t90 / t_end > T90_FRAC_MAX:
                t_end = t90 / 0.5
                print(f"   t90/t_end={t90/t_end:.2f} > {T90_FRAC_MAX} -> t_end={t_end:.3g}", flush=True)
                continue
            break
        else:
            raise SystemExit(f"{name} R40: window auto-tune failed")

        assert np.all(np.diff(I) >= -1e-12 * i_max), f"{name} R40: I not monotone"
        assert I[-1] <= i_max * (1.0 + 1e-9), f"{name} R40: capacity violated"
        tf, If = fix[f"{name}_t"], fix[f"{name}_tunnel_I"]
        early = (I < 0.3 * i_max) & (t >= tf[0]) & (t <= tf[-1])
        dev = (np.abs(I[early] - np.interp(t[early], tf, If)) / np.interp(t[early], tf, If)
               if np.any(early) else np.array([np.inf]))
        assert np.any(early) and dev.max() < 0.015, \
            f"{name} R40: early-window mismatch vs Phase-1b fixture (max {dev.max():.2%})"
        clk = sorptive_clock(t, S_an, dth, R_W, F_cylindrical)
        print(f"   R40: I_end={I[-1]:.4e}  depleted={frac:.1%}  early-dev={dev.max():.2%}"
              f"(n={int(early.sum())})  OFFLINE-CLOCK relL2={rel_l2(clk, I):.1%}  "
              f"CLAMPED relL2={rel_l2(np.minimum(clk, i_max), I):.1%}", flush=True)
        saved[f"{name}_R40_t"], saved[f"{name}_R40_I"] = t, I
        saved[f"{name}_R40_Imax"] = np.array(i_max)
        saved[f"{name}_S"], saved[f"{name}_dtheta"] = np.array(S_an), np.array(dth)

    np.savez(out_path, **saved)
    print(f"\nSaved R40 disperse refs ({'+'.join(names)}) -> {out_path}")
    print("=" * 88)
