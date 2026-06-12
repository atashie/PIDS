"""Phase-4 EMBEDDED drain scheme probe: the PSS depletion closure with a LIVE HOST DRIVE.

The offline result (scratch/m4_phase4_drain_desorptivity.py + the pre-registered fresh-ref
validation): closed-domain deployment drain is quasi-steady-state radial Kirchhoff depletion,
    dI/dt = [Phi(psi_bulk) - Phi(psi_wall)] / (r_w * (ln(R/r_w) - 3/4)),
3.3-3.9% on three independent resolved references, zero knobs. THE EMBEDDED FORM probed here:
the same law as a RATE-PRESCRIBED ridge sink whose drive psi_bulk is the LIVE volumetric-mean
host state (the PSS volumetric average -- the same average the -3/4 constant is derived against).

HOST CONTROL is load-bearing here, not decorative: the depletion bend exists ONLY because the
host's mean psi falls as it drains -- freeze the drive at psi_i (the offline twin below) and the
curve goes straight and misses the reference. This is the discriminating mechanism the disperse
clock era never exercised (2026-06-11 review attack a), exercised on every drain leg by the
depletion itself; a replenishment-pulse leg (refD40-B) additionally discriminates history.

Scheme contract: harness-compatible (scratch/m4_phase4_embedded_harness.py, direction="drain":
psi_i=-0.03, wall H_f=-1, bt linesearch). The prescribed rate returns NEGATIVE (host sink); Omega
stays 0 (no WI era in this v1 probe -- the desat front is sub-cell for whole deployment windows,
the regime the WI bridge measurably misses: 10.9/16.8% degrading, commit 919218d).

Run (WSL): conda activate pids-fem && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
    python scratch/m4_phase4_drain_embed_probe.py [n ...]
"""
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pids_forward.physics.sorptive_closure import rel_l2


class DrainPSSScheme:
    """PSS depletion closure as a host-driven prescribed-rate ridge sink (drain deployment v1).

    Every number is derived: geo = ln(R/r_w) - 3/4 (Dietz pseudo-steady, R = the closed domain's
    capacity-equivalent cylinder radius from ctx); the drive = kirchhoff(H_f_wall, mean(psi_host))
    (the live volumetric-average host state); seed = S_BK*sqrt(t0) with the a-priori Bruce-Klute
    desorptivity (regularized band, no refD40 input)."""

    def __init__(self, S_bk):
        self.S_bk = float(S_bk)

    def setup(self, feat, soil, ctx):
        self.soil, self.feat = soil, feat
        R = float(ctx["R_out"])
        self._r_w = feat._r_w
        self.geo = np.log(R / self._r_w) - 0.75
        self._p_len = feat._perimeter * feat.length
        self.h_f = -1.0                                    # the harness drain wall head
        self.seed_I = self.S_bk * np.sqrt(float(ctx["t0"]))
        self.inj = 0.0                                     # cumulative EXTRACTED volume [m^3], >= 0
        self._last = 0.0

    def pre_step(self, feat, psi, t):
        feat.Omega.x.array[:] = 0.0
        feat.Omega.x.scatter_forward()
        psi_bar = float(psi.x.array.mean())                # the PSS volumetric-average drive
        if psi_bar <= self.h_f:
            self._last = 0.0
            return 0.0                                     # bulk at/below the wall head: no drive
        dPhi = float(self.soil.kirchhoff(self.h_f, psi_bar))
        rate_vol = dPhi / (self._r_w * self.geo) * self._p_len   # [m^3/day] extracted
        self._last = rate_vol
        return -rate_vol                                   # ridge SINK on the host

    def post_step(self, feat, psi, t, dt):
        self.inj += self._last * dt

    def I_total(self, feat=None) -> float:
        return self.seed_I + self.inj / self._p_len

    def reservoir(self, feat=None, injected=None) -> float:
        return self.seed_I * self._p_len                   # only the t0 seed (all live, host-ward)


if __name__ == "__main__":
    from scratch.m4_phase4_embedded_harness import run_embedded
    from scratch.m4_phase4_drain_desorptivity import (
        bruce_klute_desorptivity, pss_drain, LOAM, PSI_WALL, PSI_I, R_W)

    ref = np.load("scratch/m4_phase4_refD40_drain.npz")
    t, I_ref = ref["LOAM_t"], ref["LOAM_I"]
    Imax = float(ref["LOAM_Imax"])
    S_bk, _, _, _ = bruce_klute_desorptivity(LOAM, PSI_WALL, PSI_I)

    # the OFFLINE fixed-drive twin (discrimination): freeze the drive at psi_i -- prescribed-rate
    # schemes are exactly reproducible offline, so the twin needs no FEM run. It must FAIL.
    geo = np.log(40.0) - 0.75
    rate_fixed = float(LOAM.kirchhoff(-1.0, PSI_I)) / (R_W * geo)
    I_fixed = np.minimum(S_bk * np.sqrt(t[0]) + rate_fixed * (t - t[0]), Imax)
    print(f"fixed-drive twin (offline, must FAIL): relL2 = {rel_l2(I_fixed, I_ref):.1%}  "
          f"end I/ref = {I_fixed[-1]/I_ref[-1]:.3f}")
    # the offline PSS curve on the same grid (the 3.3% screening, for side-by-side context)
    I_off = pss_drain(t, LOAM, PSI_I, PSI_WALL, R_W, 40 * R_W, S_bk * np.sqrt(t[0]), t[0])
    print(f"offline PSS (true-bulk drive, context):  relL2 = {rel_l2(I_off, I_ref):.1%}")

    for n in [int(a) for a in (sys.argv[1:] or [8, 12])]:
        out = run_embedded(DrainPSSScheme(S_bk), "LOAM", 40 * R_W, n, t, direction="drain")
        if out is None:
            print(f"  EMBEDDED PSS n={n}: dt collapse")
            continue
        e = rel_l2(out["I"], I_ref)
        print(f"  EMBEDDED PSS-drive n={n:2d}: relL2 = {e:.1%}  end I/ref = "
              f"{out['I'][-1]/I_ref[-1]:.3f}  (EMBEDDED_TOL=0.10)", flush=True)
