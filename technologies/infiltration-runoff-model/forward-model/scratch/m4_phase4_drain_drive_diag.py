"""Follow-up #4: the drain +8%-end drive bias -- instrumented candidate-read comparison.

The embedded drain ends ~1.08x ref on refD40 (n=8/12 consistent) and ~1.10x on SAND R40, while
the OFFLINE PSS with the water-balance bulk ends 1.02-1.04x: the production drive read (plain
dof-mean psi) is biased WET vs the bulk state the validated PSS law is defined on. Candidate
DERIVED reads, logged per step on the refD40 leg by subclassing the production class (pre_step
receives the live host field; the harness stays untouched):

  (1) dof-mean psi (CURRENT): plain average of vertex values -- over-weights the (wetter)
      domain-boundary vertices (53% of dofs at n=8 carry 1/2-1/8 cell volumes) AND averages psi
      (the wrong functional: the law's bulk state is theta-based).
  (2) volume-weighted mean psi (lumped vertex-rule weights): fixes the weighting, keeps the
      wrong functional (Jensen gap on the nonlinear retention across the drawdown cone).
  (3) psi(volume-mean theta) (inverse retention of the lumped theta mean): the closed-box
      water balance IS the volume-mean theta (harness ledger-1 asserts the discrete theta-gain
      == extracted to 1e-6), and psi_bulk = retention^-1(theta_bulk) is EXACTLY the drive the
      validated offline PSS closure uses. Recharge-aware by construction (the theta field
      carries injected water -- the refD40-C property).

TRUTH at matched extraction (no regen needed -- mass conservation): theta_bar(I) = theta_i -
I*2*r_w/(R^2-r_w^2), psi_true = retention^-1(theta_bar). PRE-REGISTERED predictions (BEFORE the
run): candidate (3) matches psi_true to discretization noise (conservation makes it exact);
candidate (1) reads WET by an amount whose dPhi excess ~ the +8% end bias; candidate (2) closes
part of the gap. If (3) wins, production _drain_rate switches to it (TDD) and ALL drain legs
re-gate with pre-registered end-bias <= 1.05.

Run (WSL, from forward-model/): PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scratch/m4_phase4_drain_drive_diag.py
"""
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pids_forward.physics.wi_exchange import WellIndexExchange
from pids_forward.physics.sorptive_closure import rel_l2

HERE = pathlib.Path(__file__).parent
R_W, R_OUT = 0.05, 2.0


def psi_of_theta_vg(soil, th):
    """Closed-form inverse of the air-entry-modified retention (exact for psi < h_s)."""
    se = (np.asarray(th, dtype=float) - soil.theta_r) / (soil.theta_s - soil.theta_r)
    se = np.clip(se, 1e-12, 1.0)
    se_vg = np.minimum(se * soil.Sc, 1.0 - 1e-15)
    u = (se_vg ** (-1.0 / soil.m) - 1.0) ** (1.0 / soil.n)
    return np.where(se >= 1.0, soil.h_s, -u / soil.alpha)


class InstrumentedDrain(WellIndexExchange):
    """Production drain + per-step candidate-read logging (behavior identical: the returned
    rate is the production rate; only observation is added)."""

    def __init__(self):
        super().__init__(direction="drain")
        self.log = {k: [] for k in ("t", "I", "psi_dof", "psi_vol", "psi_th")}

    def setup(self, feat, soil, ctx):
        super().setup(feat, soil, ctx)
        import ufl
        from dolfinx import fem
        v = ufl.TestFunction(feat.V)
        dxs = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
        w = fem.assemble_vector(fem.form(v * dxs)).array.copy()
        self._w = w / w.sum()
        self._soil_diag = soil
        return self

    def pre_step(self, feat, psi, t, dt=0.0):
        rate = super().pre_step(feat, psi, t, dt)
        a = psi.x.array
        self.log["t"].append(t)
        self.log["I"].append(self.I_total(feat))
        self.log["psi_dof"].append(float(a.mean()))
        self.log["psi_vol"].append(float((self._w * a).sum()))
        th_bar = float((self._w * self._soil_diag.theta(a)).sum())
        self.log["psi_th"].append(float(psi_of_theta_vg(self._soil_diag, th_bar)))
        return rate


if __name__ == "__main__":
    from scratch.m4_phase4_embedded_harness import run_embedded, SOILS

    soil = SOILS["LOAM"]
    ref = np.load(HERE / "m4_phase4_refD40_drain.npz")
    t, I_ref = ref["LOAM_t"], ref["LOAM_I"]
    sch = InstrumentedDrain()
    out = run_embedded(sch, "LOAM", R_OUT, 8, t, direction="drain", label="drive-diag n=8")
    assert out is not None
    print(f"\nproduction (dof-mean drive) refD40 n=8: relL2={rel_l2(out['I'], I_ref):.1%}  "
          f"end I/ref={out['I'][-1]/I_ref[-1]:.3f}")

    g = {k: np.array(v) for k, v in sch.log.items()}
    th_i = float(soil.theta(-0.03))
    th_bar_true = th_i - g["I"] * 2.0 * R_W / (R_OUT ** 2 - R_W ** 2)
    psi_true = psi_of_theta_vg(soil, th_bar_true)
    dphi = lambda p: np.array([float(soil.kirchhoff(-1.0, max(x, -1.0 + 1e-12))) for x in p])
    d_true = dphi(psi_true)
    i_frac = g["I"] / float(ref["LOAM_Imax"])
    print("\ncandidate drive dPhi deviation vs the water-balance truth, by depletion band:")
    for lo, hi in ((0.0, 0.05), (0.05, 0.12), (0.12, 0.20)):
        m = (i_frac >= lo) & (i_frac < hi) & (d_true > 0)
        if not np.any(m):
            continue
        row = [f"I/Imax {lo:.2f}-{hi:.2f} (n={m.sum():4d}):"]
        for name, key in (("(1) dof-mean", "psi_dof"), ("(2) vol-mean", "psi_vol"),
                          ("(3) psi(th-mean)", "psi_th")):
            dev = np.median(dphi(g[key][m]) / d_true[m] - 1.0)
            row.append(f"{name} {dev:+7.1%}")
        print("   " + "   ".join(row), flush=True)
