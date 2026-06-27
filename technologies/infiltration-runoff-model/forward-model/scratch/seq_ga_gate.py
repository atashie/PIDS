"""Task 3, 1-D GATE -- does a Green-Ampt CAPACITY (no ell_c -> mesh-objective by construction) reproduce
the RESOLVED Richards infiltration I(t)? If yes, GA-as-surface-flux on a COARSE cell gives the resolved
sorptive uptake regardless of cell size (the production fix).

GA cumulative infiltration under ponding: I - B*ln(1+I/B) = Ks*t, B = psi_f*dtheta,
psi_f = |kirchhoff(psi_i,0)|/Ks (validated effective front suction, SS19), dtheta = theta_s - theta(psi_i).
Compared to the RESOLVED add_ponding_bc Richards reference (fine nz) from seq_sorptivity_meshconv (cached).
Dry IC psi_i = -3 m (the meshconv setup). Times [0.005, 0.02, 0.08] day.

Run: python -u scratch/seq_ga_gate.py
"""
from __future__ import annotations

import numpy as np

from pids_forward.physics.constitutive import VanGenuchten

SOILS = {
    "sand": dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=1.5),
    "loam": dict(theta_r=0.078, theta_s=0.43, alpha=3.6,  n=1.56, Ks=0.25),
    "clay": dict(theta_r=0.068, theta_s=0.38, alpha=0.8,  n=1.09, Ks=0.048),
}
PSI_I = -3.0
TIMES = [0.005, 0.02, 0.08]
# RESOLVED add_ponding_bc Richards reference [mm] (finest converged nz), from seq_sorptivity_meshconv runs:
RESOLVED = {
    "loam": {0.005: 6.88,  0.02: 18.61, 0.08: 52.30},   # nz=40
    "clay": {0.005: 1.63,  0.02: 3.70,  0.08: 8.66},     # nz=40
    "sand": {0.005: 19.80, 0.02: 74.64, 0.08: 280.76},   # nz=24 (converged at storm scale)
}


def green_ampt(s, psi_i, times):
    soil = VanGenuchten(**s)
    Ks = s["Ks"]
    dth = s["theta_s"] - float(soil.theta(np.array([psi_i]))[0])
    psi_f = abs(soil.kirchhoff(psi_i, 0.0)) / Ks
    B = psi_f * dth
    out = {}
    for t in times:
        I = max(Ks * t + B, 1e-9)
        for _ in range(100):
            f = I - B * np.log1p(I / B) - Ks * t
            fp = 1.0 - B / (B + I)
            I = I - f / fp
        out[t] = I
    return out, psi_f, dth


def main():
    print("#" * 96)
    print("Task 3 1-D GATE -- Green-Ampt CAPACITY vs RESOLVED Richards infiltration I(t) [mm] (psi_i=-3m)")
    print("  GA has NO ell_c -> mesh-objective; GATE: ratio GA/resolved in ~0.9-1.1 (=> GA models the")
    print("  resolved sorptive uptake, so GA-on-a-coarse-cell would be mesh-objective)")
    print("#" * 96)
    print(f"{'soil':>5} {'psi_f[m]':>8} {'dtheta':>7} | " +
          "  ".join(f"t={t}" for t in TIMES) + "   (GA mm / resolved mm = ratio)")
    for name in ("sand", "loam", "clay"):
        ga, pf, dth = green_ampt(SOILS[name], PSI_I, TIMES)
        cells = "  ".join(f"{ga[t]*1000:6.1f}/{RESOLVED[name][t]:6.1f}={ga[t]*1000/RESOLVED[name][t]:.2f}"
                          for t in TIMES)
        print(f"{name:>5} {pf:>8.3f} {dth:>7.3f} | {cells}")
    print("#" * 96, flush=True)


if __name__ == "__main__":
    main()
