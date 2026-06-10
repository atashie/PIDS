"""Emit the standardized result data for the Module-4 Phase-3 OFFLINE full-curve fidelity gate.

Computes, per (soil, geometry, direction), the closure clock I(t) and the resolved reference I(t), and the
per-curve rel-L2 + PASS/FLAG status -- the SOLID, adversarially-upheld result (the coupled embedding was
retracted; commit d42432c). Writes a self-describing .npz + JSON sidecar (the viz data contract for
small/1-D results); a SEPARATE viz subagent turns it into the Tier-3 HTML without importing the solver.

Run from forward-model/:  PYTHONPATH=. python viz/run_phase3_gate_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import (
    F_cylindrical, F_throttle, throttle_params, parlange_sorptivity, sorptive_clock, rel_l2, R_W_DEFAULT,
)

DATE = "2026-06-09"
FM = Path(__file__).resolve().parent.parent
DATA_DIR = FM / "scratch"          # committed gate fixtures live in tests/data
FIX = FM / "tests" / "data"
OUT = FM.parent / "validation" / "sanity" / "data"
OUT.mkdir(parents=True, exist_ok=True)

R_W = R_W_DEFAULT
SOILS = {
    "SAND": VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13),
    "LOAM": VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25),
    "SILT": VanGenuchten(theta_r=0.034, theta_s=0.46, alpha=1.6, n=1.37, Ks=0.06),
    "CLAY": VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.048),
}
COARSE = {"SAND", "LOAM", "SILT"}
GEOM_KEY = {"tunnel": "tunnel_I", "annulus": "drain_I"}
DISP_TOL = 0.05


def _dtheta(soil):
    return abs(float(soil.theta(0.0) - soil.theta(-1.0)))


def _empirical_desorptivity(t, I, k=3):
    return float(np.mean(I[1:1 + k] / np.sqrt(t[1:1 + k])))


def main():
    disp = np.load(FIX / "m4_phase1b_disperse_refs.npz")
    drain = np.load(FIX / "m4_phase1c_drain_refs.npz")
    arrays, rows = {}, []
    for name, soil in SOILS.items():
        S = parlange_sorptivity(soil, -1.0, 0.0)
        dth = _dtheta(soil)
        for geom in ("tunnel", "annulus"):
            t = disp[f"{name}_t"]; Iref = disp[f"{name}_{GEOM_KEY[geom]}"]
            Imod = sorptive_clock(t, S, dth, R_W, F_cylindrical)              # the a-priori closure
            Ipl = sorptive_clock(t, S, dth, R_W, lambda z: 1.0)              # the planar clock that FAILS
            err = rel_l2(Imod, Iref)
            req = name in COARSE
            status = ("PASS" if err <= DISP_TOL else "FAIL") if req else "FLAG"
            key = f"{name}_{geom}_disperse"
            arrays[f"{key}_t"] = t; arrays[f"{key}_Iref"] = Iref
            arrays[f"{key}_Imodel"] = Imod; arrays[f"{key}_Iplanar"] = Ipl
            rows.append(dict(soil=name, geom=geom, direction="disperse", S=round(S, 5),
                             rel_l2=round(err, 4), planar_l2=round(rel_l2(Ipl, Iref), 4),
                             status=status, pass_required=req))
            # drain (semi-empirical throttle, flagged)
            t = drain[f"{name}_t"]; Iref = drain[f"{name}_{GEOM_KEY[geom]}"]
            S_des = _empirical_desorptivity(t, Iref)
            z0, k = throttle_params(dth)
            Imod = sorptive_clock(t, S_des, dth, R_W, lambda z: F_throttle(z, z0, k))
            err = rel_l2(Imod, Iref)
            key = f"{name}_{geom}_drain"
            arrays[f"{key}_t"] = t; arrays[f"{key}_Iref"] = Iref; arrays[f"{key}_Imodel"] = Imod
            rows.append(dict(soil=name, geom=geom, direction="drain", S=round(S_des, 5),
                             rel_l2=round(err, 4), status="FLAG", pass_required=False))

    npz_path = OUT / f"m4_phase3_gate__{DATE}.npz"
    np.savez(npz_path, **arrays)
    disp_pass = [r for r in rows if r["direction"] == "disperse" and r["pass_required"]]
    meta = dict(
        module="pids-features (M4 §E)", check="Phase-3 offline full-curve fidelity gate (C-004, disperse a-priori)",
        date=DATE, units=dict(t="day", I="m (cumulative uptake per wall area)"),
        closure=dict(disperse="dI/dt=(S^2/2I)*F_cyl(zeta), F_cyl=2z/ln(1+2z), S=a-priori Parlange sorptivity, NO knob",
                     drain="dI/dt=(S_des^2/2I)*exp(-(zeta/z0)^k), SEMI-EMPIRICAL desorptivity (flagged)",
                     planar="F=1 (the constant-kappa clock that FAILS the gate)"),
        tolerance=dict(disperse_pass="rel-L2 <= 5%", advisory="drain + CLAY flagged"),
        metric="rel-L2 over the I(t) domain-integral; *_pen front diagnostics never used",
        verdict=dict(disperse_apriori_gate="PASS" if all(r["status"] == "PASS" for r in disp_pass) else "FAIL",
                     worst_disperse_pass_l2=max(r["rel_l2"] for r in disp_pass),
                     note="Coupled embedding RETRACTED post-review (vacuous/non-general); this OFFLINE gate is the solid result."),
        curves=rows,
    )
    json_path = OUT / f"m4_phase3_gate__{DATE}.json"
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"WROTE {npz_path}  ({npz_path.stat().st_size/1e3:.1f} KB)")
    print(f"WROTE {json_path}")
    print(f"disperse a-priori gate verdict: {meta['verdict']['disperse_apriori_gate']} "
          f"(worst PASS-required rel-L2 = {meta['verdict']['worst_disperse_pass_l2']:.1%})")
    for r in rows:
        print(f"  {r['soil']:5s} {r['geom']:7s} {r['direction']:8s}  rel-L2={r['rel_l2']:6.1%}  {r['status']}")


if __name__ == "__main__":
    main()
