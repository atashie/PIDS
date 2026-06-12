"""Phase-4 drain-closure hypothesis #1: AIR-ENTRY-REGULARIZED A-PRIORI DESORPTIVITY.

Phase-3's a-priori desorptivity attempts died on the D = K/C -> inf singularity at the saturated
psi=0 start (Bruce-Klute NaN, Parlange-desorption ~2x over). The DEPLOYMENT drain scenario (refD40)
starts at psi_i = -0.03, BELOW the Ippisch air entry h_s = -0.02, where C(-0.03) ~ 0.2 1/m is
finite: D is finite everywhere on [psi_wall, psi_i] = [-1, -0.03] and the Bruce-Klute similarity
BVP is regular. This script:

1. Solves the EXACT Bruce-Klute desorption similarity BVP by shooting (no approximation):
       (D(theta) theta_l')' = -(lambda/2) theta_l',  theta(0) = theta_b (wall), theta(inf) = theta_i,
   with the internal identity check  S = 2 D theta'(0) = int (theta_i - theta) dlambda.
2. Derives S_des A-PRIORI -- no refD40 input anywhere in the derivation.
3. Integrates the cylindrical-GA drain clock dI/dt = (S^2/2I) F(zeta), F = 2z/ln(1+2z),
   zeta = I/(dtheta r_w) (the disperse machinery, direction-reversed) against the resolved
   closed-box refD40 (scratch/m4_phase4_refD40_drain.npz). ZERO tuning: pass/fail as-is.
4. Reports the Phase-3 comparison points for context (S_sorp 109% over, 0.46*S_sorp 42% under)
   and where S_BK/S_sorp lands relative to the TAINTED 0.74 (a prediction check, not a target:
   0.74 was implied BY refD40; S_BK is derived with no refD40 input, so agreement is evidence,
   disagreement is just disagreement).

Run (WSL): conda activate pids-fem && OMP_NUM_THREADS=1 python scratch/m4_phase4_drain_desorptivity.py
"""
import numpy as np
from scipy.integrate import solve_ivp

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import (
    parlange_sorptivity, F_cylindrical, rel_l2)

LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
PSI_WALL, PSI_I = -1.0, -0.03
R_W = 0.05


def make_D_theta(soil, psi_lo, psi_hi, npts=120000):
    """D(theta) = K/C on the regular band [psi_lo, psi_hi] (both < h_s) via psi-grid inversion."""
    psi = np.linspace(psi_lo, min(psi_hi, soil.h_s - 1e-9), npts)
    th, K, C = soil.theta(psi), soil.K(psi), soil.capacity(psi)
    assert np.all(C > 0), "capacity must be positive on the whole band (regularized start)"
    D = K / C
    order = np.argsort(th)
    uth, idx = np.unique(th[order], return_index=True)
    uD = D[order][idx]
    return (lambda t: np.interp(np.clip(t, uth[0], uth[-1]), uth, uD)), uth[0], uth[-1]


def bruce_klute_desorptivity(soil, psi_wall, psi_i, lam_max=80.0, tol=1e-12):
    """Exact desorption similarity solution by shooting on the wall flux.

    State y = (theta, g) with g = D(theta) dtheta/dlambda:
        theta' = g / D(theta),   g' = -(lambda/2) theta'.
    theta(0) = theta_b; shoot g0 = D theta'(0) > 0 so theta -> theta_i without overshoot.
    Returns (S, profile-integral S, lambda grid, theta profile)."""
    Dfun, th_lo, th_hi = make_D_theta(soil, psi_wall, psi_i)
    th_b, th_i = th_lo, th_hi                          # dry wall boundary, wet initial

    def rhs(lam, y):
        th, g = y
        dth = g / max(Dfun(th), 1e-30)
        return [dth, -(lam / 2.0) * dth]

    def overshoot(lam, y):                             # theta passes theta_i -> g0 too big
        return th_i - y[0]
    overshoot.terminal, overshoot.direction = True, -1.0

    def stalled(lam, y):                               # flux dead before reaching theta_i -> too small
        return y[1] - 1e-14
    stalled.terminal, stalled.direction = True, -1.0

    def classify(g0):
        sol = solve_ivp(rhs, [0.0, lam_max], [th_b, g0], events=[overshoot, stalled],
                        rtol=1e-10, atol=1e-14, dense_output=True, max_step=lam_max / 200)
        if sol.t_events[0].size:                       # overshot theta_i
            return 1, sol
        return -1, sol                                 # stalled / ran out: undershoot

    lo, hi = 1e-8, 1.0                                 # bracket g0 [m/sqrt(day)-ish units]
    while classify(hi)[0] < 0:
        hi *= 4.0
        assert hi < 1e3, "failed to bracket the shooting flux"
    while classify(lo)[0] > 0:
        lo /= 4.0
    for _ in range(200):                               # bisection to machine-tight bracket
        mid = 0.5 * (lo + hi)
        if classify(mid)[0] > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * max(1.0, hi):
            break
    g0 = 0.5 * (lo + hi)
    _, sol = classify(g0)
    lam = np.linspace(0.0, sol.t[-1], 4000)
    th = sol.sol(lam)[0]
    S_flux = 2.0 * g0                                  # cumulative = 2 g0 sqrt(t)
    S_prof = np.trapezoid(th_i - th, lam)              # mass-balance identity
    return S_flux, S_prof, lam, th


def make_psi_of_theta(soil, psi_lo, psi_hi, npts=120000):
    """Inverse retention psi(theta) on the band (both ends < h_s -> strictly monotone)."""
    psi = np.linspace(psi_lo, min(psi_hi, soil.h_s - 1e-9), npts)
    th = soil.theta(psi)
    return lambda x: np.interp(np.clip(x, th[0], th[-1]), th, psi)


def pss_drain(t, soil, psi_i, psi_wall, r_w, R, I0, t0, three_quarters=True,
              I_seed_curve=None):
    """HYPOTHESIS #2 (derived form): pseudo-steady-state closed-reservoir radial depletion.

    Quasi-steady Kirchhoff radial flow from the closed bulk into the wall:
        q_len = 2*pi*[Phi(psi_bulk) - Phi(psi_wall)] / (ln(R/r_w) - 3/4),
    the -3/4 being the DERIVED volumetric-average PSS constant (Dietz; well-testing
    pseudo-steady state) -- set three_quarters=False for the plain steady log (sensitivity,
    both forms derived, neither tuned). psi_bulk(I) from the closed-box water balance:
        theta_bulk = theta_i - I * 2 r_w / (R^2 - r_w^2),   psi_bulk = retention^-1(theta_bulk).
    dI/dt = q_len/(2 pi r_w). Starts from (t0, I0)."""
    psi_of_th = make_psi_of_theta(soil, psi_wall, psi_i)
    th_i = float(soil.theta(psi_i))
    geo = np.log(R / r_w) - (0.75 if three_quarters else 0.0)
    I = np.empty_like(t)
    started = False
    for k in range(t.size):
        if t[k] <= t0:
            I[k] = I_seed_curve[k] if I_seed_curve is not None else I0
            continue
        x = I0 if not started else I[k - 1]
        t_prev = t0 if not started else t[k - 1]
        started = True
        sub = np.linspace(t_prev, t[k], 400)
        for j in range(1, sub.size):
            th_b = th_i - x * 2.0 * r_w / (R * R - r_w * r_w)
            dPhi = float(soil.kirchhoff(psi_wall, float(psi_of_th(th_b))))
            x += dPhi / (r_w * geo) * (sub[j] - sub[j - 1])
        I[k] = x
    return I


def lam_support(lam, th, th_i, frac=0.99):
    """Boltzmann coordinate containing `frac` of the desorbed mass (the profile's leading edge)."""
    m = np.cumsum((th_i - th) * np.gradient(lam))
    return float(np.interp(frac * m[-1], m, lam))


def drain_clock(t, S, dth, r_w):
    """Cylindrical-GA drain clock (the disperse machinery, reversed): explicit substepped ODE."""
    I = np.empty_like(t)
    I[0] = S * np.sqrt(t[0])                           # planar similarity seed at the first sample
    for k in range(1, t.size):
        sub = np.linspace(t[k - 1], t[k], 400)
        x = I[k - 1]
        for j in range(1, sub.size):
            x += (S * S / (2.0 * x)) * F_cylindrical(x / (dth * r_w)) * (sub[j] - sub[j - 1])
        I[k] = x
    return I


if __name__ == "__main__":
    ref = np.load(pathlib.Path(__file__).parent / "m4_phase4_refD40_drain.npz")
    t, I_ref = ref["LOAM_t"], ref["LOAM_I"]
    dth_ref, Imax = float(ref["LOAM_dtheta"]), float(ref["LOAM_Imax"])

    # --- the a-priori derivation (no refD40 input) -----------------------------------------
    S_bk, S_prof, lam, th = bruce_klute_desorptivity(LOAM, PSI_WALL, PSI_I)
    S_sorp = parlange_sorptivity(LOAM, PSI_WALL, PSI_I)    # the wetting-pair S over the same band
    dth = float(LOAM.theta(PSI_I) - LOAM.theta(PSI_WALL))
    print(f"Bruce-Klute (regularized [-1, -0.03]):  S_des = {S_bk:.5f} m/sqrt(day)")
    print(f"  identity check: flux-form {S_bk:.6f} vs profile-form {S_prof:.6f} "
          f"(rel dev {abs(S_bk-S_prof)/S_bk:.2e})")
    print(f"  S_sorp (Parlange, same band) = {S_sorp:.5f};  S_BK/S_sorp = {S_bk/S_sorp:.3f}  "
          f"(tainted refD40-implied ratio was ~0.74; Phase-3 semi-empirical 0.46)")
    print(f"  dtheta = {dth:.5f} (fixture {dth_ref:.5f})")

    # --- the no-knob clock vs the resolved closed-box reference ----------------------------
    for label, S in (("S_BK (a-priori, THIS hypothesis)", S_bk),
                     ("S_sorp (Phase-3 point: 109% over)", S_sorp),
                     ("0.46*S_sorp (Phase-3 point: 42% under)", 0.46 * S_sorp)):
        I = drain_clock(t, S, dth, R_W)
        e = rel_l2(I, I_ref)
        print(f"  cyl-GA clock with {label:42s}: relL2 = {e:7.1%}  "
              f"end I/ref = {I[-1]/I_ref[-1]:.3f}")
    print(f"  (ref drains {I_ref[-1]:.3f} m = {I_ref[-1]/Imax:.1%} of I_max in {t[-1]:.1f} d)")

    # --- the deficit's time-shape (diagnosis: front-similarity vs global depletion) ---------
    I_bk = drain_clock(t, S_bk, dth, R_W)
    q = np.searchsorted(t, [0.5, 2.0, 5.0, 10.0])
    print("  S_BK clock / ref at t = 0.5/2/5/10/20 d:",
          " ".join(f"{I_bk[i]/I_ref[i]:.3f}" for i in [*q, len(t) - 1]))

    # --- HYPOTHESIS #2 (derived, no knob): pseudo-steady closed-reservoir depletion --------
    R_out = 40 * R_W
    lam99 = lam_support(lam, th, float(LOAM.theta(PSI_I)))
    t_x = ((R_out - R_W) / lam99) ** 2                 # boundary-influence time (derived)
    print(f"\nPSS depletion era (hypothesis #2): lam99 = {lam99:.3f} m/sqrt(day) -> "
          f"boundary-influence t_x = {t_x:.2f} d")
    for label, t0_, I0_, seed in (
            ("pure PSS from the first sample      ", t[0], I_ref[0] * 0 + S_bk * np.sqrt(t[0]), None),
            ("BK similarity era -> PSS at t_x     ", t_x, None, I_bk)):
        I0v = float(np.interp(t0_, t, I_bk)) if I0_ is None else I0_
        for tq in (True, False):
            I_pss = pss_drain(t, LOAM, PSI_I, PSI_WALL, R_W, R_out, I0v, t0_,
                              three_quarters=tq, I_seed_curve=seed)
            e = rel_l2(I_pss, I_ref)
            print(f"  {label} (ln(R/r_w){' - 3/4' if tq else '      '}): "
                  f"relL2 = {e:7.1%}  end I/ref = {I_pss[-1]/I_ref[-1]:.3f}")
