"""Extended-range gate test (Codex/Arik review 2026-06-09): the committed gate used Carsel-Parrish texture
MEANS spanning only ~2 orders of K_s (sand 7.13 -> clay 0.048 m/day). Real PIDS soils span ~9-10 orders
(coarse gravel ~1e2-1e4 m/day; compacted clay liner ~1e-4-1e-6 m/day). This generates RESOLVED disperse-
tunnel references for two EXTREMES and checks whether the a-priori cylindrical Green-Ampt closure
(F=2z/ln(1+2z) x Parlange S, NO knob) still reproduces them -- i.e. does the gate's "a-priori" claim
survive outside the benign agricultural band, or does the closure break (e.g. gravity-dominated coarse
material, or FEM stiffness at very low K).

Resolved reference reuses the Phase-1 radial-tunnel machinery (1-D cylindrical, gravity-free; the cleanest
a-priori case). Run from forward-model/:
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scratch/m4_extended_range_refs.py
"""
from __future__ import annotations

import numpy as np

import scratch.m4_phase1b_disperse_reference as dz   # radial_tunnel, _run, R_W, R_OUT, cp linesearch
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import (
    F_cylindrical, parlange_sorptivity, sorptive_clock, rel_l2, R_W_DEFAULT,
)

R_W = R_W_DEFAULT
# the four committed soils for reference + two EXTREMES bracketing the real range
EXTRA = {
    "GRAVEL":    VanGenuchten(theta_r=0.030, theta_s=0.38, alpha=20.0, n=3.00, Ks=500.0),    # coarse, free-draining
    "TIGHTCLAY": VanGenuchten(theta_r=0.100, theta_s=0.45, alpha=0.50, n=1.10, Ks=1.0e-3),   # tight clay / weak liner
}


def main():
    out, summary = {}, []
    print("=" * 90)
    print("EXTENDED-RANGE disperse-tunnel gate test (a-priori cyl Green-Ampt + Parlange S, NO knob)")
    print(f"  K_s span now: TIGHTCLAY {EXTRA['TIGHTCLAY'].Ks:.0e} -> GRAVEL {EXTRA['GRAVEL'].Ks:.0e} m/day "
          f"(~{np.log10(EXTRA['GRAVEL'].Ks/EXTRA['TIGHTCLAY'].Ks):.1f} orders)")
    print("=" * 90)
    for name, soil in EXTRA.items():
        S = parlange_sorptivity(soil, -1.0, 0.0)
        dth = abs(float(soil.theta(0.0) - soil.theta(-1.0)))
        # window: first sample ~0.1 r_w penetration; t_end so the front reaches ~3 r_w (capped so a glacial
        # tight-clay run stays tractable -- a small window still tests the early a-priori sqrt-t coefficient).
        t_start = (dth * 0.1 * R_W / max(S, 1e-12)) ** 2
        t_end = min((0.2 * dth / max(S, 1e-12)) ** 2, 5.0)
        samples = list(np.geomspace(t_start, t_end, 24))
        print(f"\n{'-'*90}\n{name}: Ks={soil.Ks:g} m/day  alpha={soil.alpha} n={soil.n}  "
              f"S(Parlange)={S:.5f}  dtheta={dth:.4f}  t_end={t_end:.3e} d", flush=True)
        tun = dz.radial_tunnel(soil, -1.0, samples, label=name)
        t = np.array([r[0] for r in tun]); Iref = np.array([r[1] for r in tun]); pen = np.array([r[2] for r in tun])
        Imod = sorptive_clock(t, S, dth, R_W, F_cylindrical)
        err = rel_l2(Imod, Iref)
        end_pen = pen[-1] / R_W
        out[f"{name}_t"] = t; out[f"{name}_tunnel_I"] = Iref; out[f"{name}_Imodel"] = Imod
        summary.append((name, soil.Ks, S, err, end_pen))
        print(f"   -> closure rel-L2 = {err:.1%}  (front reached {end_pen:.1f} r_w; {'PASS<=5%' if err<=0.05 else 'OUTSIDE 5% gate'})")

    np.savez("scratch/m4_extended_range_refs.npz", **out)
    print("\n" + "=" * 90)
    print(f"{'soil':10s} {'Ks(m/d)':>10s} {'Parlange_S':>11s} {'closure_relL2':>14s} {'front(r_w)':>11s}")
    for name, Ks, S, err, ep in summary:
        print(f"{name:10s} {Ks:10.3g} {S:11.5f} {err:13.1%} {ep:11.1f}")
    print("saved -> scratch/m4_extended_range_refs.npz")
    print("=" * 90)


if __name__ == "__main__":
    main()
