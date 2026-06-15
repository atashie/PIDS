"""ITEM (B) attribution probe -- the offline PSS drain law's own +2-4% END BIAS.

PURE-PYTHON, no FEM (the constitutive float closures + sorptive_closure are dolfinx-free at import;
the FEM references are loaded from the committed-on-disk npz). Read-only diagnostic.

POST-REGEN NOTE (2026-06-15): item B was RESOLVED (candidate 2) and the drain refs were THEN
regenerated at a converged dt cap. This probe loads the on-disk refs, which are now the CONVERGED
refs -- so re-running it prints end_FE ~0.97-0.99 (the law slightly UNDER, the Jensen gap), NOT the
pre-cap "+2-4% over" this header describes. The durable PRE-CAP evidence (recorded ends
1.020/1.031/1.042/1.010; ref-BE share +3.7-4.8%) lives in
validation/sanity/m4_phase4_drain_endbias_attribution__2026-06-15.md.

THE BUG under investigation: the offline closure pss_drain
    dI/dt = [Phi(psi_bulk(I)) - Phi(psi_wall)] / (r_w*(ln(R/r_w) - 3/4)),
    theta_bulk = theta_i - I*2*r_w/(R^2 - r_w^2),  psi_bulk = retention^-1(theta_bulk)
over-predicts cumulative I at end-of-curve by +2-4% vs the resolved FEM refs (recorded ends
refD40 1.019, SAND-R40 1.030, LOAM-R20 1.040). Early-similarity already REFUTED (ea5e352).

THE NUMERICAL ASYMMETRY (the key framing): the LAW integrates with FORWARD EULER (400 substeps/
sample; holds the higher start-rate on a decaying ODE -> biases I HIGH), while the REFS use
adaptive BACKWARD EULER (decaying drive -> biases cumulative I LOW). So law/ref is pushed ABOVE 1
by BOTH schemes, on top of any Dietz -3/4 model-form error. This probe removes the LAW's FE share
(re-integrate the SAME model form to dt->0) so the residual law_exact/ref isolates
(model-form + ref-BE). The ref-BE share is measured separately in Part B (FEM dt_max Richardson).

CANDIDATES (kickoff): (1) Dietz -3/4 = a STEADY linearization on the nonlinear transient Kirchhoff
depletion; (2) the refs' own ~1.3% backward-Euler temporal accuracy.
DISCRIMINATOR built in here: does the bias (and the effective constant c_eff) SCALE WITH DEPLETION
DEPTH / soil nonlinearity? -> model-form. Or sit ~flat regardless of depth? -> uniform ref-BE.

Run (WSL pids-fem, from forward-model/, no threading pins needed -- no FEM):
    python scratch/_b_drain_endbias.py
"""
import sys, pathlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import rel_l2
from scratch.m4_phase4_drain_desorptivity import (
    bruce_klute_desorptivity, pss_drain, make_psi_of_theta, LOAM)

SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)
SILT = VanGenuchten(theta_r=0.034, theta_s=0.46, alpha=1.6, n=1.37, Ks=0.06)
PSI_I, PSI_WALL, R_W = -0.03, -1.0, 0.05
HERE = pathlib.Path(__file__).parent


def law_exact(t, soil, R, I0, t0, three_quarters=0.75):
    """SAME model form as pss_drain, integrated to dt->0 (stiff Radau, rtol 1e-11). The ONLY change
    vs pss_drain is the time integration: identical psi_of_th interpolant + kirchhoff quadrature +
    closed-box water balance. three_quarters = the geometric constant c in (ln(R/r_w) - c)."""
    psi_of_th = make_psi_of_theta(soil, PSI_WALL, PSI_I)
    th_i = float(soil.theta(PSI_I))
    th_wall = float(soil.theta(PSI_WALL))
    geo = np.log(R / R_W) - three_quarters
    A = 2.0 * R_W / (R * R - R_W * R_W)

    def rhs(tt, y):
        th_b = th_i - max(y[0], 0.0) * A
        if th_b <= th_wall:
            return [0.0]
        dphi = float(soil.kirchhoff(PSI_WALL, float(psi_of_th(th_b))))
        return [max(dphi, 0.0) / (R_W * geo)]

    mask = t > t0
    sol = solve_ivp(rhs, [t0, float(t[-1])], [I0], t_eval=t[mask],
                    method="Radau", rtol=1e-11, atol=1e-14, dense_output=False)
    assert sol.success, sol.message
    I = np.empty_like(t, dtype=float)
    I[~mask] = I0
    I[mask] = sol.y[0]
    return I


def m1_truth(t, soil, R, I0, t0):
    """A-PRIORI MECHANISM CHECK (M1: the nonlinear volume-average gap). The law drives with
    Phi(theta_mean); the correct Dietz PSS drives with the VOLUME-MEAN potential <Phi>. Both are
    divided by the SAME (ln(R/r_w)-3/4), so the constant cancels in the rate ratio -- this isolates
    the Phi(theta_mean) vs <Phi> gap, nothing else. Profile assumed = the steady Kirchhoff log in Phi
    (Phi_w := 0), scaled by Phi_R so its volume-mean theta matches the material-balance theta_mean.
    Returns I_truth(t) (dt->0). If I_truth ~ ref, M1 IS the mechanism; the leftover ref/I_truth is
    (ref backward-Euler + non-uniform-depletion M2 + spatial)."""
    psg = np.linspace(PSI_WALL, PSI_I, 4001)
    Phi = np.array([soil.kirchhoff(PSI_WALL, float(p)) for p in psg])   # >=0, increasing in psi
    th = soil.theta(psg)                                                # increasing in psi
    th_i, th_w = float(soil.theta(PSI_I)), float(soil.theta(PSI_WALL))
    Phi_i = float(Phi[-1])
    geo = np.log(R / R_W) - 0.75
    A = 2.0 * R_W / (R * R - R_W * R_W)
    lnRr = np.log(R / R_W)
    rg = np.geomspace(R_W, R, 3000)
    wsum = np.trapezoid(rg, rg)                                         # int r dr (2*pi cancels)
    PhiR_grid = np.linspace(Phi_i, 1e-7, 600)
    thm_g, pbar_g = np.empty_like(PhiR_grid), np.empty_like(PhiR_grid)
    for j, PhiR in enumerate(PhiR_grid):
        Phir = PhiR * np.log(rg / R_W) / lnRr                           # steady log profile, Phi_w=0
        thr = np.interp(Phir, Phi, th)                                  # theta(Phi)
        thm_g[j] = np.trapezoid(thr * rg, rg) / wsum
        pbar_g[j] = np.trapezoid(Phir * rg, rg) / wsum
    order = np.argsort(thm_g)
    thm_s, pbar_s = thm_g[order], pbar_g[order]

    def rhs(tt, y):
        thm = th_i - max(y[0], 0.0) * A
        if thm <= th_w:
            return [0.0]
        pbar = float(np.interp(thm, thm_s, pbar_s))                     # <Phi>(theta_mean)
        return [max(pbar, 0.0) / (R_W * geo)]

    mask = t > t0
    sol = solve_ivp(rhs, [t0, float(t[-1])], [I0], t_eval=t[mask],
                    method="Radau", rtol=1e-10, atol=1e-14)
    assert sol.success, sol.message
    I = np.empty_like(t, dtype=float)
    I[~mask] = I0
    I[mask] = sol.y[0]
    return I


def c_effective(t, soil, R, I0, t0, ref_end):
    """The geometric constant c in (ln(R/r_w) - c) for which law_exact's END equals ref_end.
    c > 0.75 -> the law needs MORE resistance than Dietz -3/4 (it over-predicts at -3/4)."""
    def f(c):
        return law_exact(t, soil, R, I0, t0, three_quarters=c)[-1] - ref_end
    # larger c -> smaller geo -> larger rate -> larger end. f INCREASING in c. Bracket wide.
    lo, hi = -2.0, np.log(R / R_W) - 0.05    # c < ln(R/r_w) keeps geo > 0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:                          # same sign -> root not bracketed
        return float("nan")
    return brentq(f, lo, hi, xtol=1e-7)


def leg(name, soil, R_factor, t, ref):
    R = R_factor * R_W
    S_bk = bruce_klute_desorptivity(soil, PSI_WALL, PSI_I)[0]
    I0 = S_bk * np.sqrt(t[0])
    dth = float(soil.theta(PSI_I) - soil.theta(PSI_WALL))
    i_max = dth * (R ** 2 - R_W ** 2) / (2.0 * R_W)

    I_fe = pss_drain(t, soil, PSI_I, PSI_WALL, R_W, R, I0, t[0])          # production (FE 400-substep)
    I_ex = law_exact(t, soil, R, I0, t0=t[0])                            # same form, dt->0
    I_fe_nolog = pss_drain(t, soil, PSI_I, PSI_WALL, R_W, R, I0, t[0], three_quarters=False)
    I_m1 = m1_truth(t, soil, R, I0, t0=t[0])                             # a-priori M1 mechanism truth

    e_fe = rel_l2(I_fe, ref)
    e_ex = rel_l2(I_ex, ref)
    ce = c_effective(t, soil, R, I0, t[0], ref[-1])
    return dict(
        name=name, soil=soil, R_factor=R_factor, dth=dth, S_bk=S_bk, i_max=i_max,
        depl=ref[-1] / i_max, n=len(t), window=float(t[-1]),
        end_fe=I_fe[-1] / ref[-1], end_ex=I_ex[-1] / ref[-1],
        fe_share=I_fe[-1] / I_ex[-1] - 1.0,            # the LAW's own forward-Euler over-injection
        end_nolog=I_fe_nolog[-1] / ref[-1],            # the -3/4 lever (plain log)
        relL2_fe=e_fe, relL2_ex=e_ex, c_eff=ce,
        end_m1=I_ex[-1] / I_m1[-1],                    # M1-PREDICTED model-form end bias (law/truth)
        leftover=I_m1[-1] / ref[-1],                   # truth/ref = ref-BE + M2 + spatial (unexplained)
    )


def main():
    legs = []

    # --- refD40: LOAM R40, ~20% depletion (the screening target; recorded end 1.019) ---
    d = np.load(HERE / "m4_phase4_refD40_drain.npz")
    legs.append(leg("refD40  LOAM R40", LOAM, 40, d["LOAM_t"], d["LOAM_I"]))

    # --- fresh refs: SAND R40 (~20%), LOAM R20 (~50% DEEP) (recorded ends 1.030, 1.040) ---
    fr = np.load(HERE / "m4_phase4_drain_fresh_refs.npz")
    legs.append(leg("SAND    SAND R40", SAND, 40, fr["SAND_R40_t"], fr["SAND_R40_I"]))
    legs.append(leg("LOAMR20 LOAM R20", LOAM, 20, fr["LOAM_R20_t"], fr["LOAM_R20_I"]))

    # --- SILT R40 generality leg (recorded end 1.010), if the ref npz exposes a t/I pair ---
    sp = HERE / "m4_phase4_silt_drain_ref.npz"
    if sp.exists():
        s = np.load(sp)
        k = {x.lower(): x for x in s.files}
        tk = next((s.files[i] for i, x in enumerate(s.files) if x.endswith("_t")), None)
        ik = next((s.files[i] for i, x in enumerate(s.files) if x.endswith("_I")), None)
        print(f"[silt npz keys] {list(s.files)}  -> t={tk} I={ik}", flush=True)
        if tk and ik and s[tk].shape == s[ik].shape:
            legs.append(leg("SILT    SILT R40", SILT, 40, s[tk], s[ik]))

    print("\n" + "=" * 108)
    print("ITEM (B): offline PSS drain end-bias attribution -- LAW forward-Euler share removed (Part A)")
    print("=" * 108)
    hdr = (f"{'leg':17s} {'depl%':>6s} {'n':>3s} {'win[d]':>7s} | "
           f"{'end_FE':>7s} {'end_dt0':>7s} {'FEshare':>8s} | {'relL2_FE':>9s} {'relL2_dt0':>10s} | "
           f"{'c_eff':>7s} {'end_nolog':>9s}")
    print(hdr)
    print("-" * 108)
    for r in legs:
        print(f"{r['name']:17s} {r['depl']*100:6.1f} {r['n']:3d} {r['window']:7.2f} | "
              f"{r['end_fe']:7.3f} {r['end_ex']:7.3f} {r['fe_share']*100:+7.2f}% | "
              f"{r['relL2_fe']*100:8.2f}% {r['relL2_ex']*100:9.2f}% | "
              f"{r['c_eff']:7.3f} {r['end_nolog']:9.3f}")
    print("-" * 108)
    print("LEGEND: end_FE = recorded offline-law/ref end ratio (forward-Euler, 400 substep/sample).")
    print("        end_dt0 = SAME model form integrated to dt->0 (Radau) / ref end. Drop FE->dt0 = the")
    print("                  LAW's own forward-Euler over-injection (FEshare). end_dt0 = (Dietz")
    print("                  model-form bias) x (ref backward-Euler bias) -- split in Part B (FEM).")
    print("        c_eff   = the constant c in (ln(R/r_w)-c) that lands law_exact END on the ref.")
    print("                  c_eff trending with depl% -> model-form; c_eff ~flat ~0.75 -> uniform/ref.")
    print("        end_nolog = the -3/4 lever: end ratio with the PLAIN log (c=0), FE. (geo sensitivity.)")
    print("\nDIETZ reference constant = 0.75.  ln(R/r_w): R40 -> %.3f, R20 -> %.3f"
          % (np.log(40.0), np.log(20.0)))

    print("\n" + "=" * 108)
    print("A-PRIORI MECHANISM CHECK (M1: Phi(theta_mean) vs <Phi> nonlinear volume-average gap)")
    print("=" * 108)
    print(f"{'leg':17s} {'depl%':>6s} | {'measured':>9s} {'M1 pred':>9s} {'M1/meas':>8s} | "
          f"{'leftover':>9s}   (measured = law_dt0/ref;  M1 pred = law_dt0/truth)")
    print("-" * 108)
    for r in legs:
        meas = r['end_ex'] - 1.0
        pred = r['end_m1'] - 1.0
        frac = pred / meas if abs(meas) > 1e-6 else float('nan')
        print(f"{r['name']:17s} {r['depl']*100:6.1f} | {meas*100:+8.2f}% {pred*100:+8.2f}% "
              f"{frac*100:7.0f}% | {(r['leftover']-1.0)*100:+8.2f}%")
    print("-" * 108)
    print("M1/meas ~100% -> the nonlinear volume-average gap IS the mechanism (model-form).")
    print("leftover = truth/ref - 1 = the UNEXPLAINED remainder (ref backward-Euler + non-uniform")
    print("           depletion M2 + spatial); if small, M1 closes the attribution with no FEM.")


if __name__ == "__main__":
    main()
