"""Phase-4 drain closure: FRESH-REFERENCE validation of the PSS depletion closure (hypothesis #2).

TAINT DISCIPLINE (the kickoff's hard rule): refD40 was the diagnostic/screening target, so the
candidate closure must be validated on references it has NEVER seen, with its predictions
PRE-REGISTERED before the FEM references are generated. This script does exactly that, in order:

1. THE CLOSURE (fixed form, locked by scratch/m4_phase4_drain_desorptivity.py on refD40, no
   per-leg freedom): pure pseudo-steady-state closed-reservoir Kirchhoff radial depletion
       dI/dt = [Phi(psi_bulk(I)) - Phi(psi_wall)] / (r_w * (ln(R/r_w) - 3/4)),
       theta_bulk = theta_i - I*2*r_w/(R^2 - r_w^2),  psi_bulk = retention^-1(theta_bulk),
   I(t[0]) = S_BK * sqrt(t[0]) (the a-priori Bruce-Klute desorptivity seed; negligible).
   Every quantity is geometry/constitutive-derived; the -3/4 is the Dietz volumetric-average
   pseudo-steady constant. ZERO free parameters.
2. FRESH LEGS (new soil, new geometry, new depletion depth -- windows sized by the closure's own
   prediction to hit the target depletion fractions, which fixes the regime, not the answer):
       FRESH-1: SAND,  R = 40 r_w, psi_i = -0.03, window -> ~20% of I_max (soil generality)
       FRESH-2: LOAM,  R = 20 r_w, psi_i = -0.03, window -> ~50% of I_max (geometry + DEEP depletion)
3. Saves the predictions npz FIRST (pre-registration), then generates the resolved 1-D radial
   closed-box FEM references (same machinery as refD40: scratch/m4_phase4_refAB_drain.py), then
   scores. PASS bar stated up front: relL2 <= 10% on BOTH legs (the screening result was 3.3%);
   a failure is recorded as a failure.

Run (WSL): conda activate pids-fem && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scratch/m4_phase4_drain_fresh_refs.py
"""
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scratch.m4_phase4_drain_desorptivity import (
    bruce_klute_desorptivity, pss_drain, LOAM as LOAM_SOIL)
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import rel_l2

SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)
PSI_I, PSI_WALL = -0.03, -1.0
R_W = 0.05
N_SAMP = 32
LEGS = (  # (name, soil, R_factor, target depletion fraction)
    ("SAND_R40", SAND, 40, 0.20),
    ("LOAM_R20", LOAM_SOIL, 20, 0.50),
)
HERE = pathlib.Path(__file__).parent
PRED_NPZ = HERE / "m4_phase4_drain_fresh_predictions.npz"
REF_NPZ = HERE / "m4_phase4_drain_fresh_refs.npz"


def predict_leg(soil, R_factor, target_frac):
    """Size the window so the CLOSURE predicts ~target_frac depletion, then emit the
    pre-registered prediction on the leg's sample grid."""
    R = R_factor * R_W
    S_bk, _, _, _ = bruce_klute_desorptivity(soil, PSI_WALL, PSI_I)
    dth = float(soil.theta(PSI_I) - soil.theta(PSI_WALL))
    i_max = dth * (R ** 2 - R_W ** 2) / (2.0 * R_W)
    t_start = (dth * 0.1 * R_W / S_bk) ** 2           # the refD40 grid-form, a-priori S
    # bisect the window length on the closure's own depletion prediction
    lo, hi = t_start * 10, 1.0
    def end_frac(t_end):
        t = np.geomspace(t_start, t_end, N_SAMP)
        I = pss_drain(t, soil, PSI_I, PSI_WALL, R_W, R, S_bk * np.sqrt(t[0]), t[0])
        return I[-1] / i_max, t, I
    while end_frac(hi)[0] < target_frac:
        hi *= 2.0
        assert hi < 1e4
    for _ in range(60):
        mid = np.sqrt(lo * hi)                        # geometric bisection (t spans decades)
        f, _, _ = end_frac(mid)
        if f < target_frac:
            lo = mid
        else:
            hi = mid
    t_end = np.sqrt(lo * hi)
    f, t, I_pred = end_frac(t_end)
    return t, I_pred, i_max, S_bk, dth, f


if __name__ == "__main__":
    # ---- step 1+2: pre-register the predictions (BEFORE any FEM) ---------------------------
    pred = {}
    for name, soil, Rf, frac in LEGS:
        t, I_pred, i_max, S_bk, dth, f = predict_leg(soil, Rf, frac)
        pred[f"{name}_t"] = t
        pred[f"{name}_Ipred"] = I_pred
        pred[f"{name}_Imax"] = np.array(i_max)
        pred[f"{name}_Sbk"] = np.array(S_bk)
        pred[f"{name}_dtheta"] = np.array(dth)
        print(f"[predict] {name}: R={Rf}r_w  S_BK={S_bk:.5f}  dth={dth:.5f}  I_max={i_max:.3f} m  "
              f"window={t[-1]:.3f} d -> predicted end depletion {f:.1%}", flush=True)
    np.savez(PRED_NPZ, **pred)
    print(f"PRE-REGISTERED predictions saved -> {PRED_NPZ.name}\n", flush=True)

    # ---- step 3: generate the resolved references (the expensive FEM part) -----------------
    import scratch.m4_phase4_refAB_drain as gen
    refs = {}
    for name, soil, Rf, _ in LEGS:
        t = pred[f"{name}_t"]
        print(f"[FEM] {name}: 1-D radial closed box, R={Rf}r_w, psi_i={PSI_I}, "
              f"wall {PSI_WALL}, {t[-1]:.3f} d, {N_SAMP} samples", flush=True)
        I_ref = gen.run_drain_A(soil, Rf * R_W, t, label=name, psi_i=PSI_I)
        i_max = float(pred[f"{name}_Imax"])
        assert np.all(np.diff(I_ref) >= -1e-12 * i_max) and I_ref[-1] <= i_max * (1 + 1e-9)
        refs[f"{name}_t"] = t
        refs[f"{name}_I"] = I_ref
        refs[f"{name}_Imax"] = np.array(i_max)
        np.savez(REF_NPZ, **refs)                      # save incrementally
        e = rel_l2(pred[f"{name}_Ipred"], I_ref)
        print(f"[score] {name}: PRE-REGISTERED PSS prediction relL2 = {e:.1%}  "
              f"end pred/ref = {pred[f'{name}_Ipred'][-1]/I_ref[-1]:.3f}  "
              f"ref end depletion = {I_ref[-1]/i_max:.1%}  -> "
              f"{'PASS' if e <= 0.10 else 'FAIL'} (bar 10%)", flush=True)
    print(f"References saved -> {REF_NPZ.name}")
