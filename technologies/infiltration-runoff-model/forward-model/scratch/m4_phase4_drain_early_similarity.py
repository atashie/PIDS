"""The drain law's own +2-4% end over-bias: the EARLY-SIMILARITY composite -- REFUTED 2026-06-12.

OUTCOME (pre-registered bars, measured): the min-composite FAILS catastrophically -- relL2
84.3/85.4/76.1% on refD40/SAND_R40/LOAM_R20, ending at 0.141/0.133/0.239 of the refs (pure PSS:
3.3/3.4/3.9% at 1.020/1.031/1.042). The law STAYS pure PSS. Post-mortem (kept because the
mechanism is the lesson):
  * The min LOCKS onto the similarity branch: S_BK^2/(2I) only DECAYS with I, so once it drops
    below the PSS rate (refD40: t ~ 1e-3 d) the integrator can never escape -- the rate
    starves itself. The hypothesized "similarity era bottleneck until t_x ~ 0.58 d" is wrong:
    the TRUE curve follows the PSS rate from ~1e-3 d on (if the truth were similarity-limited
    through 0.58 d, pure PSS would carry a +25%-of-end surplus; it measures +2%).
  * The early era carries NO measurable mass here: the max-composite (the correct matched form
    for a high-transient -> global-steady transition) differs from pure PSS by the integral of
    (sim - pss) over [t_0, 1e-3 d] ~ 1e-4 m ~ 0.01% of end-I. Not worth running.
  * CONCLUSION: the +2-4% end over-bias does NOT come from the early similarity era. That
    attribution (2026-06-12 progress notes' suspect) is REFUTED by this check. Remaining
    candidates, unattributed: the Dietz -3/4 linearization on the nonlinear Kirchhoff problem;
    the resolved refs' own documented ~1.3% BE temporal accuracy.

THE ORIGINAL HYPOTHESIS (kept for the record): for t below the similarity-exit time the front
is local and the wall flux follows Bruce-Klute, I = S_BK*sqrt(t) -> rate = S_BK^2/(2 I);
composite rate(I) = min(similarity, PSS), zero knobs.

PRE-REGISTERED BARS (stated before running): on each of the three resolved refs the composite
must (a) cut the END ratio to within 1.5% of the ref and (b) have relL2 <= the pure-PSS relL2.
A failure is recorded as a failure (the law stays pure PSS). <- THE BARS FAILED; see OUTCOME.

Run (WSL): PYTHONPATH=. OMP_NUM_THREADS=1 python scratch/m4_phase4_drain_early_similarity.py
"""
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scratch.m4_phase4_drain_desorptivity import (
    bruce_klute_desorptivity, make_psi_of_theta, pss_drain)
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import rel_l2

HERE = pathlib.Path(__file__).parent
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)
PSI_I, PSI_WALL = -0.03, -1.0
R_W = 0.05


def composite_drain(t, soil, R, S_bk, n_sub=400):
    """The early-similarity composite: pss_drain's integrator with the min-rate."""
    psi_of_th = make_psi_of_theta(soil, PSI_WALL, PSI_I)
    th_i = float(soil.theta(PSI_I))
    geo = np.log(R / R_W) - 0.75
    I = np.empty_like(t)
    x = S_bk * np.sqrt(t[0])
    I[0] = x
    for k in range(1, t.size):
        sub = np.linspace(t[k - 1], t[k], n_sub)
        for j in range(1, sub.size):
            th_b = th_i - x * 2.0 * R_W / (R * R - R_W * R_W)
            pss = float(soil.kirchhoff(PSI_WALL, float(psi_of_th(th_b)))) / (R_W * geo)
            sim = S_bk ** 2 / (2.0 * max(x, 1e-12))
            x += min(sim, pss) * (sub[j] - sub[j - 1])
        I[k] = x
    return I


if __name__ == "__main__":
    refD = np.load(HERE / "m4_phase4_refD40_drain.npz")
    fresh = np.load(HERE / "m4_phase4_drain_fresh_refs.npz")
    legs = [
        ("refD40 LOAM R40", LOAM, 40, refD["LOAM_t"], refD["LOAM_I"]),
        ("fresh SAND R40", SAND, 40, fresh["SAND_R40_t"], fresh["SAND_R40_I"]),
        ("fresh LOAM R20 (deep 48%)", LOAM, 20, fresh["LOAM_R20_t"], fresh["LOAM_R20_I"]),
    ]
    print("leg                          pure-PSS relL2/end     COMPOSITE relL2/end   (bars: end<=1.5%, relL2<=PSS)")
    for name, soil, Rf, t, I_ref in legs:
        S_bk, _, _, _ = bruce_klute_desorptivity(soil, PSI_WALL, PSI_I)
        R = Rf * R_W
        I_pss = pss_drain(t, soil, PSI_I, PSI_WALL, R_W, R, S_bk * np.sqrt(t[0]), float(t[0]))
        I_cmp = composite_drain(t, soil, R, S_bk)
        e_p, e_c = rel_l2(I_pss, I_ref), rel_l2(I_cmp, I_ref)
        r_p, r_c = I_pss[-1] / I_ref[-1], I_cmp[-1] / I_ref[-1]
        ok = abs(r_c - 1.0) <= 0.015 and e_c <= e_p + 1e-12
        print(f"{name:28s}  {e_p:5.1%} / {r_p:.3f}        {e_c:5.1%} / {r_c:.3f}    "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
