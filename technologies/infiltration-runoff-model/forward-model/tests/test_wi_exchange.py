"""Phase-4 production integration (DISPERSE-ONLY): the rate-clock + well-index exchange.

Pins `pids_forward.physics.wi_exchange.WellIndexExchange` -- the production form of the Phase-4
prototype that passed the deployment-regime discriminating gate (RefA-40 2.2-5.7%, RefB-40 history
5.0-5.4%, control 2.1-6.0%; offline clock 27-34%, retracted dual-scale 13-39%). SCOPE: disperse,
positive-WI deployment regime (h > ~5.5 r_w); the drain direction is REFUSED (its sub-grid closure
for closed domains is open research -- see scratch/m4_phase4_wi_probe.py header) and the
resolved-wall regime is REFUSED (negative-log bridge = transiently unstable). ADVERSARIALLY
REVIEWED 2026-06-11 (validation/sanity/m4_phase4_coupled_review__2026-06-11.md): host control is
established for the WI era; the clock era is a prescribed-rate closure; within this disperse-only
scope the capacity-clamped clock passive is killed by NO leg (its killers are the refused drain
legs) -- the gate discriminates v2 from the RAW clock and the dual-scale, not from capacity-aware
passives.
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


def _feat(n, psi_wall=0.0):
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [n, n, n])
    feat = EmbeddedFeature(msh, _ON_FEAT, tangent=(1.0, 0.0, 0.0),
                           K_feat=1.0, area=np.pi * R_W_DEFAULT ** 2, porosity=0.4)
    feat.configure_sorptive(LOAM, psi_i=-1.0, psi_wall=psi_wall)
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


def test_drain_requires_catchment_radius():
    """The drain mode is the PSS depletion closure (2026-06-12): it needs the closed/catchment
    radius R_out (a physical geometry input, ln(R/r_w)-3/4); setup without it must refuse."""
    feat = _feat(2)
    with pytest.raises(ValueError, match="R_out"):
        WellIndexExchange(direction="drain").setup(feat, LOAM, {"t0": 1e-4})


def test_drain_rate_zero_when_bulk_at_wall_head():
    """The drain drive vanishes when the host bulk mean reaches the live wall head: the prescribed
    extraction must hard-zero (capacity safety: the scheme cannot overdraw a depleted host)."""
    from dolfinx import fem
    feat = _feat(2)
    x = WellIndexExchange(direction="drain").setup(feat, LOAM, {"t0": 1e-4, "R_out": 2.0})
    feat.Hf.x.array[:] = -1.0
    feat.Hf.x.scatter_forward()
    psi = fem.Function(feat.V)
    for host in (-1.0, -1.5):                              # bulk at the wall head; bulk below it
        psi.x.array[:] = host
        assert x.pre_step(feat, psi, 0.0) == 0.0


def test_drain_prescribed_rate_is_the_pss_law():
    """The drain pre_step returns the PSS depletion rate as a NEGATIVE ridge source (host sink):
    -2*pi*[Phi(psi_bar) - Phi(Hf_bar)]/(ln(R/r_w) - 3/4) * length -- every factor derived."""
    from dolfinx import fem
    feat = _feat(2)
    R_out = 2.0
    x = WellIndexExchange(direction="drain").setup(feat, LOAM, {"t0": 1e-4, "R_out": R_out})
    feat.Hf.x.array[:] = -1.0
    feat.Hf.x.scatter_forward()
    psi = fem.Function(feat.V)
    psi.x.array[:] = -0.03
    rate = x.pre_step(feat, psi, 0.0)
    geo = np.log(R_out / R_W_DEFAULT) - 0.75
    want = -float(LOAM.kirchhoff(-1.0, -0.03)) / (R_W_DEFAULT * geo) * feat._perimeter * feat.length
    assert rate < 0.0, "drain must be a host sink (negative ridge source)"
    assert abs(rate - want) < 1e-12 * abs(want)
    x.post_step(feat, psi, 0.0, 0.5)                       # accounting: I_total positive, ledger form
    assert x.I_total(feat) > 0.0
    assert abs(x.I_total(feat) * feat._perimeter * feat.length - abs(rate) * 0.5) < 1e-12


@pytest.mark.parametrize("n", [8])
def test_drain_gate_refD40_closed_box(n):
    """THE drain deployment gate leg, production class through the closed-box harness: the embedded
    I(t) tracks the resolved refD40 (LOAM, R=40 r_w, psi_i=-0.03, 20-d window) within the
    pre-registered EMBEDDED_TOL while the FIXED-DRIVE twin (no host read -- the same law with the
    drive frozen at psi_i, exactly reproducible offline for a prescribed-rate scheme) fails by
    >= BASELINE_KILL: the host read is load-bearing (probe: embedded 7.4/6.9% at n=8/12,
    twin 74.2%). (~4 min; prior failures: WI-bridge drain 10.9/16.8% degrading, throttle 98%.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded
    ref = np.load("scratch/m4_phase4_refD40_drain.npz")
    t, I_ref = ref["LOAM_t"], ref["LOAM_I"]
    out = run_embedded(WellIndexExchange(direction="drain"), "LOAM", 40 * R_W_DEFAULT, n, t,
                       direction="drain")
    assert out is not None, "embedded closed-box drain run did not complete"
    e = rel_l2(out["I"], I_ref)
    assert e <= 0.10, f"drain gate failed: relL2={e:.1%}"
    geo = np.log(40.0) - 0.75
    rate_fixed = float(LOAM.kirchhoff(-1.0, -0.03)) / (R_W_DEFAULT * geo)
    I_twin = np.minimum(rate_fixed * (t - t[0]), float(ref["LOAM_Imax"]))
    assert rel_l2(I_twin, I_ref) >= 0.20, \
        "discrimination twin lost: the fixed-drive PSS clock passes this leg"


def test_nonzero_wall_head_refused():
    """The gate evidence is saturated-wall (psi_wall=0) only; the class must refuse a feature
    configured with any other wall head rather than silently scaling the clock era against the
    wrong Kirchhoff reference (2026-06-11 adversarial review)."""
    feat = _feat(2, psi_wall=-0.1)
    with pytest.raises(ValueError, match="wall head"):
        WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4})


def test_clock_rate_zero_when_ring_at_wall_head():
    """The disperse drive vanishes when the front ring sits at/above the wall head: the clock-era
    rate must hard-zero, not inject on a magnitude-only Kirchhoff drop (2026-06-11 review)."""
    from dolfinx import fem
    feat = _feat(2)
    x = WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4})
    psi = fem.Function(feat.V)
    for host in (0.0, 0.5):                                # saturated to the wall head; ponded above
        psi.x.array[:] = host
        assert x.pre_step(feat, psi, 0.0) == 0.0 and x.in_subgrid_era


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
