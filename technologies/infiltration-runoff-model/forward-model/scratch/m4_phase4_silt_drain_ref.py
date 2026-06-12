"""SILT R40 drain reference -- the last missing embedded-drain soil (fresh-refs recipe).

Completes the drain soil-generality axis (LOAM refD40 + SAND R40 fresh ref are gate-asserted;
SILT had no resolved drain reference). Same TAINT DISCIPLINE as
scratch/m4_phase4_drain_fresh_refs.py: the offline PSS closure's prediction (zero knobs,
S_BK seed) is sized to ~20% depletion and SAVED FIRST, then the resolved 1-D radial closed FEM
reference is generated (run_drain_A machinery, bt linesearch), then scored. PASS bar 10%
(the LOAM/SAND/LOAM-R20 record: 3.3-3.9%). A failure is recorded as a failure.

Run (WSL): PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scratch/m4_phase4_silt_drain_ref.py
"""
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scratch.m4_phase4_drain_fresh_refs import predict_leg, PSI_I, PSI_WALL, R_W
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import rel_l2

SILT = VanGenuchten(theta_r=0.034, theta_s=0.46, alpha=1.6, n=1.37, Ks=0.06)
HERE = pathlib.Path(__file__).parent
OUT = HERE / "m4_phase4_silt_drain_ref.npz"

if __name__ == "__main__":
    t, I_pred, i_max, S_bk, dth, f = predict_leg(SILT, 40, 0.20)
    print(f"[predict] SILT_R40: S_BK={S_bk:.5f}  dth={dth:.5f}  I_max={i_max:.3f} m  "
          f"window={t[-1]:.3f} d -> predicted end depletion {f:.1%}", flush=True)
    np.savez(OUT, SILT_R40_t=t, SILT_R40_Ipred=I_pred, SILT_R40_Imax=np.array(i_max),
             SILT_R40_Sbk=np.array(S_bk), SILT_R40_dtheta=np.array(dth))
    print(f"PRE-REGISTERED prediction saved -> {OUT.name}", flush=True)

    import scratch.m4_phase4_refAB_drain as gen
    I_ref = gen.run_drain_A(SILT, 40 * R_W, t, label="SILT_R40 drain", psi_i=PSI_I)
    assert np.all(np.diff(I_ref) >= -1e-12 * i_max) and I_ref[-1] <= i_max * (1 + 1e-9)
    data = dict(np.load(OUT))
    data["SILT_R40_I"] = I_ref
    np.savez(OUT, **data)
    e = rel_l2(I_pred, I_ref)
    print(f"[score] SILT_R40: PRE-REGISTERED PSS prediction relL2 = {e:.1%}  "
          f"end pred/ref = {I_pred[-1]/I_ref[-1]:.3f}  ref end depletion = {I_ref[-1]/i_max:.1%}"
          f"  -> {'PASS' if e <= 0.10 else 'FAIL'} (bar 10%)", flush=True)
    print(f"Reference saved -> {OUT.name}")
