"""Phase-4 production integration (DISPERSE-ONLY): the rate-clock + well-index exchange.

Pins `pids_forward.physics.wi_exchange.WellIndexExchange` -- the production form of the Phase-4
prototype that passed the deployment-regime discriminating gate (RefA-40 2.2-5.7%, RefB-40 history
5.0-5.4%, control 2.1-6.0%; offline clock 27-34%, retracted dual-scale 13-39%). SCOPE: disperse,
positive-WI deployment regime (h > ~5.5 r_w); the drain direction is REFUSED (its sub-grid closure
for closed domains is open research -- see scratch/m4_phase4_wi_probe.py header) and the
resolved-wall regime is REFUSED (negative-log bridge = transiently unstable). Coupled wording is
prototype-validated; adversarial review pending (the Phase-4 protocol gate before any claim).
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.feature import EmbeddedFeature
from pids_forward.physics.wi_exchange import WellIndexExchange
from pids_forward.physics.sorptive_closure import (
    F_cylindrical, sorptive_clock, rel_l2, R0_OVER_H_P1, R_W_DEFAULT)

COMM = MPI.COMM_WORLD
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
_ON_FEAT = lambda x: np.isclose(x[1], 0.5) & np.isclose(x[2], 0.5)


def _feat(n):
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [n, n, n])
    feat = EmbeddedFeature(msh, _ON_FEAT, tangent=(1.0, 0.0, 0.0),
                           K_feat=1.0, area=np.pi * R_W_DEFAULT ** 2, porosity=0.4)
    feat.configure_sorptive(LOAM, psi_i=-1.0, psi_wall=0.0)
    return feat


def test_wi_constant_from_measured_r0():
    """WI = 2*pi/ln(r_0/r_w) with r_0 = 0.1986*h, h auto-measured from the lattice."""
    feat = _feat(2)                                        # h = 0.5 m = 10 r_w: deployment regime
    x = WellIndexExchange()
    x.setup(feat, LOAM, {"t0": 1e-4})
    assert abs(x.h - 0.5) < 1e-12
    assert abs(x.WI - 2.0 * np.pi / np.log(R0_OVER_H_P1 * 0.5 / R_W_DEFAULT)) < 1e-12
    assert x.WI > 0.0


def test_resolved_wall_regime_is_refused():
    """h <= ~5.5 r_w -> negative-log bridge (transiently unstable, analyzed 2026-06-10) -> refuse."""
    feat = _feat(8)                                        # h = 0.125 m = 2.5 r_w
    with pytest.raises(ValueError, match="regime"):
        WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4})


def test_drain_direction_is_refused():
    """The drain sub-grid closure for closed domains is OPEN research (no a-priori form fits:
    throttle 98%, cyl+S_des 42%, cyl+S_sorp 109% vs refD40) -- the production class refuses it."""
    with pytest.raises(NotImplementedError, match="drain"):
        WellIndexExchange(direction="drain")


def test_subgrid_era_reduces_to_the_validated_clock():
    """With the host held at psi_i (scale = 1), the sub-grid era's prescribed rate must integrate to
    the validated offline clock (the gate's pre-handover property; explicit-Euler stepping on a fine
    grid vs the 400-substep clock)."""
    from dolfinx import fem
    feat = _feat(2)
    x = WellIndexExchange()
    x.setup(feat, LOAM, {"t0": 1e-4})
    x.seed(1e-4)
    psi = fem.Function(feat.V)
    psi.x.array[:] = -1.0
    t = np.geomspace(1e-4, 5e-3, 400)                      # short window, well inside I_fill
    I = [x.I_total(feat)]
    for i in range(1, t.size):
        rate = x.pre_step(feat, psi, t[i - 1])
        assert rate > 0.0 and x.in_subgrid_era
        x.post_step(feat, psi, t[i - 1], t[i] - t[i - 1])
        I.append(x.I_total(feat))
    clk = sorptive_clock(t, feat.S_disp, feat.dth_disp, R_W_DEFAULT, F_cylindrical)
    assert rel_l2(np.array(I), clk) < 0.01


@pytest.mark.parametrize("n", [8])
def test_gate_refA40_closed_box(n):
    """THE discriminating gate, production class through the closed-box harness: the embedded I(t)
    tracks the resolved depleting-reservoir reference (LOAM, R=40 r_w, full depletion) within the
    pre-registered EMBEDDED_TOL while the offline clock fails the same leg; both mass ledgers are
    asserted every sample inside the harness. (~3 min; the full n/RefB sweep lives in
    scratch/m4_phase4_wi_probe.py + the Tier-3 record.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded
    refA = np.load("tests/data/m4_phase4_refA_disperse.npz")
    t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
    out = run_embedded(WellIndexExchange(), "LOAM", 40 * R_W_DEFAULT, n, t)
    assert out is not None, "embedded closed-box run did not complete"
    e = rel_l2(out["I"], I_ref)
    assert e <= 0.10, f"gate failed: relL2={e:.1%}"
    S, dth = float(refA["LOAM_S"]), float(refA["LOAM_dtheta"])
    clk = sorptive_clock(t, S, dth, R_W_DEFAULT, F_cylindrical)
    assert rel_l2(clk, I_ref) >= 0.20, "discrimination twin lost: the offline clock passes this leg"
