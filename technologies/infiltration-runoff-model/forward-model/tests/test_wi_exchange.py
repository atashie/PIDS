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

ITEM (B) NOTE (2026-06-15): the drain-leg docstrings below recorded an "end I/ref = 1.019/1.030/
1.040/1.010 = the offline PSS law's own +2-4%". Item B REFUTED that model-form reading: the closed-
drain references carried a ~3.7-4.8% first-order backward-Euler temporal UNDER-count, so the law
only APPEARED to over-predict. The refs were regenerated at a converged dt cap (window/2048,
scratch/m4_phase4_refAB_drain.py::DT_MAX_DIV); against them the embedded scheme now scores (n=8)
refD40 4.4%/end 0.979, SAND 3.6%/0.986, LOAM-R20 2.7%/0.992, SILT 5.5%/0.973, refD40-C 2.9%/0.983
-- i.e. the over-bias FLIPPED to a small UNDER (the law's own Jensen volume-average gap, the law is
accurate/slightly conservative). The gate bars (<=0.10, twin>=0.20) hold unchanged. The historical
numbers in the per-test docstrings predate the regeneration; binding record:
validation/sanity/m4_phase4_drain_endbias_attribution__2026-06-15.md.
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
    x.setup(feat, LOAM, {"t0": 1e-4, "R_out": 1.0})        # R_out only sizes the WI-era ring read
    assert abs(x.h - 0.5) < 1e-12
    assert abs(x.WI - 2.0 * np.pi / np.log(R0_OVER_H_P1 * 0.5 / R_W_DEFAULT)) < 1e-12
    assert x.WI > 0.0


def _feat_box(R_out, n):
    """A harness-style closed box (ridge along x at the y-z centre) sized to R_out, so the WI-era
    ring at R_out/2 lands on a resolved vertex shell (the [0,1]^3 _feat box is too small for that)."""
    L = float(np.sqrt(np.pi * (R_out ** 2 - R_W_DEFAULT ** 2)))
    h = L / n
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [4 * h, L, L]], [4, n, n])
    feat = EmbeddedFeature(msh, lambda x: np.isclose(x[1], L / 2) & np.isclose(x[2], L / 2),
                           tangent=(1.0, 0.0, 0.0), K_feat=1.0,
                           area=np.pi * R_W_DEFAULT ** 2, porosity=0.4)
    feat.configure_sorptive(LOAM, psi_i=-1.0, psi_wall=0.0)
    return feat


def test_resolved_wall_validated_band_auto_allowed():
    """Item C honest fence (2026-06-16): the post-item-A driver is r_0-independent, and the resolved-wall
    SWEEP (scratch/m4_phase4_resolved_wall.py) validated the realistic deployment band (large R_out
    catchment + fine mesh). A disperse mesh inside it -- R_out >= 40 r_w AND h >= 2.2 r_w -- is now
    AUTO-ALLOWED with the DEFAULT constructor (no opt-in): the validation is the warrant."""
    feat = _feat_box(2.0, 16)                              # R_out = 40 r_w, h = 4.43 r_w: validated band
    x = WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4, "R_out": 2.0})
    assert x.allow_resolved_wall is False                 # no opt-in needed
    assert x.r0 <= 1.1 * R_W_DEFAULT and x.h < 5.5 * R_W_DEFAULT   # genuinely resolved-wall
    assert x.h >= 2.2 * R_W_DEFAULT


def test_resolved_wall_below_rout_floor_refused():
    """Disperse keys on R_out: the WI-era ring read at R_out/2 degrades when R_out/2 is shallow (the
    refinement budget shrinks with R_out -- R10 degraded 2.4->19.1% as h:2.2->0.49). A resolved-wall
    mesh with R_out < 40 r_w is REFUSED by default, and allow_resolved_wall=True forces past it."""
    feat = _feat_box(0.5, 16)                              # R_out = 10 r_w (< 40), h = 1.10 r_w
    with pytest.raises(ValueError, match="VALIDATED band"):
        WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4, "R_out": 0.5})
    x = WellIndexExchange(allow_resolved_wall=True).setup(feat, LOAM, {"t0": 1e-4, "R_out": 0.5})
    assert x.allow_resolved_wall is True                  # the probe/escape hatch still works


def test_resolved_wall_below_h_floor_refused():
    """Both directions auto-allow only down to the validated floor h >= 2.2 r_w; finer is unvalidated
    (the embedded relL2 grows with refinement) and REFUSED unless opted in. (ctx['h'] override drives
    the fence to a sub-floor h on a coarse mesh so this stays a fast setup-only test.)"""
    feat = _feat_box(2.0, 16)                              # R_out = 40 r_w (ok); ctx h forces sub-floor
    with pytest.raises(ValueError, match="VALIDATED band"):
        WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4, "R_out": 2.0, "h": 2.0 * R_W_DEFAULT})


def test_resolved_wall_drain_band_and_floor():
    """Drain is mass-based (psi(volume-mean theta), no R_out/2 ring read) -> R_out-robust, validated at
    R20+R40, so its fence keys on the LOWER R_out >= 20 r_w floor (+ h >= 2.2 r_w). Below 20 r_w is
    refused by default; the validated band is auto-allowed."""
    ok = _feat_box(1.0, 8)                                 # R_out = 20 r_w, h = 4.42 r_w: validated band
    WellIndexExchange(direction="drain").setup(ok, LOAM, {"t0": 1e-4, "R_out": 1.0})   # no raise
    bad = _feat_box(0.5, 8)                                # R_out = 10 r_w (< 20), h = 4.42 r_w
    with pytest.raises(ValueError, match="VALIDATED band"):
        WellIndexExchange(direction="drain").setup(bad, LOAM, {"t0": 1e-4, "R_out": 0.5})


def test_disperse_requires_catchment_radius():
    """Item A (2026-06-13): the WI-era bridge now reads the resolved field at the catchment-radius
    midpoint R_out/2 (the on-ridge read was -7..-18% WET -- the localized WI-era residual;
    scratch/m4_phase4_wi_ring_derivation.py). Disperse setup must REQUIRE R_out (symmetric with drain), AFTER the
    regime/wall-head guards so those keep their own messages."""
    feat = _feat_box(2.0, 8)
    with pytest.raises(ValueError, match="R_out"):
        WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4})


def test_disperse_wi_era_reads_resolved_ring_at_Rout_half():
    """Item A: in the WI era the disperse exchange is a PRESCRIBED ridge rate driven by the resolved
    field at r_ring = R_out/2 (mean psi over the vertex shell there), via the steady cylindrical
    Kirchhoff resistance  q = 2*pi*[Phi(H_f) - Phi(psi_ring)]/ln(r_ring/r_w) * length  -- NOT the
    implicit on-ridge Omega bridge. Omega is 0 and the rate is returned as a ridge source (the
    clock-era path). A uniform host makes psi_ring exact, so the rate is checked to machine eps."""
    from dolfinx import fem
    R_out = 2.0
    feat = _feat_box(R_out, 8)
    x = WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4, "R_out": R_out})
    assert abs(x.r_ring_target - 0.5 * R_out) < 1e-12
    assert 0.4 * R_out < x.r_ring < 0.6 * R_out            # the captured shell sits near R_out/2
    x.in_subgrid_era = False                               # force the WI era
    psi = fem.Function(feat.V)
    psi.x.array[:] = -0.5                                  # uniform host -> psi_ring = -0.5 exactly
    rate = x.pre_step(feat, psi, 1.0)
    assert np.allclose(feat.Omega.x.array, 0.0), "WI era must be prescribed-rate (Omega = 0)"
    want = float(LOAM.kirchhoff(-0.5, 0.0)) / (R_W_DEFAULT * np.log(x.r_ring / R_W_DEFAULT)) \
        * feat._perimeter * feat.length
    assert rate > 0.0
    assert abs(rate - want) < 1e-9 * abs(want), f"ring-read rate {rate:.6e} != want {want:.6e}"


def test_disperse_wi_ring_rate_zero_when_ring_at_wall_head():
    """Capacity/saturation safety: when the resolved ring sits at/above the wall head the WI-era
    prescribed rate must hard-zero (no injection into an already-saturated annulus), mirroring the
    clock-era front-ring guard."""
    from dolfinx import fem
    R_out = 2.0
    x = WellIndexExchange().setup(_feat_box(R_out, 8), LOAM, {"t0": 1e-4, "R_out": R_out})
    x.in_subgrid_era = False
    psi = fem.Function(x.feat.V)
    for host in (0.0, 0.5):                                # at the wall head; ponded above it
        psi.x.array[:] = host
        assert x.pre_step(x.feat, psi, 1.0) == 0.0         # dt=0 (bare) and ...
        assert x.pre_step(x.feat, psi, 1.0, 20.0) == 0.0   # ... dt>0 (r0==0 early return)
    # cap=0 edge: a fully-SATURATED bulk (theta_bulk == theta_s) with a sub-wall-head ring read
    # (a nonzero bare bridge) must still return 0 -- the live-capacity throttle hard-zeros it.
    psi.x.array[:] = LOAM.h_s                              # air entry: theta == theta_s, psi < H_f
    assert x._ring_bridge(LOAM.h_s) > 0.0                  # the bare bridge is nonzero here
    assert x.pre_step(x.feat, psi, 1.0, 20.0) == 0.0       # but the cap (theta_s - theta_s) zeros it


def test_disperse_wi_heun_reduces_the_explicit_lag():
    """Item A (2026-06-13 debug): the prescribed WI-era rate held over the big WI-era steps is a
    first-order explicit lag (over-injection). With dt>0 the scheme returns the HEUN rate
    (r0+r1)/2 with r1 = the bridge at the ring head PREDICTED after the step's mean theta rise
    r0*dt/V_box -- strictly below the start-state rate on a wetting step (no knobs)."""
    from dolfinx import fem
    R_out = 2.0
    feat = _feat_box(R_out, 8)
    x = WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4, "R_out": R_out})
    x.in_subgrid_era = False
    psi = fem.Function(feat.V)
    psi.x.array[:] = -0.5                                  # uniform, far from saturation (cap slack)
    r0 = x.pre_step(feat, psi, 1.0)                        # dt omitted -> bare bridge
    dt = 0.5
    rate = x.pre_step(feat, psi, 1.0, dt)                  # Heun-corrected
    th_pred = float(LOAM.theta(-0.5)) + r0 * dt / x._vol
    se = (th_pred - LOAM.theta_r) / (LOAM.theta_s - LOAM.theta_r) * LOAM.Sc
    psi_pred = -((se ** (-1.0 / LOAM.m) - 1.0) ** (1.0 / LOAM.n)) / LOAM.alpha
    r1 = float(LOAM.kirchhoff(psi_pred, 0.0)) / (R_W_DEFAULT * np.log(x.r_ring / R_W_DEFAULT)) \
        * feat._perimeter * feat.length
    assert abs(rate - 0.5 * (r0 + r1)) < 1e-9 * abs(rate), f"Heun mismatch {rate} vs {0.5*(r0+r1)}"
    assert rate < r0, "Heun must reduce the start-state rate on a wetting step"


def test_disperse_wi_live_capacity_throttle_is_recharge_aware():
    """Item A (2026-06-13 debug): the infinite-domain ring bridge has no outer-boundary knowledge
    and would over-inject past saturation (+17% end overshoot, ledger break). The rate is throttled
    by the LIVE remaining capacity (theta_s - theta_bulk)*V_box/dt read from the current field --
    mass-exact and recharge-aware. A near-saturated bulk with a dry ring read must return the
    capacity cap, not the large bridge rate."""
    from dolfinx import fem
    R_out = 2.0
    feat = _feat_box(R_out, 8)
    x = WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4, "R_out": R_out})
    x.in_subgrid_era = False
    psi = fem.Function(feat.V)
    psi.x.array[:] = -0.05                                 # uniform NEAR saturation (tiny capacity)
    dt = 20.0                                              # long step -> the small capacity binds
    rate = x.pre_step(feat, psi, 1.0, dt)
    th_bulk = float((x._wvol * LOAM.theta(psi.x.array)).sum())
    cap = (LOAM.theta_s - th_bulk) * x._vol / dt
    r0 = x._ring_bridge(-0.05)
    assert cap < r0, "test setup: the capacity cap must bind below the (uncapped) bridge rate"
    assert rate > 0.0 and abs(rate - cap) < 1e-9 * abs(cap), f"throttle not applied: {rate} vs {cap}"


def test_parallel_comm_refused():
    """2026-06-12 Codex review finding 1: every host read in this class (dof/theta means, the
    front-ring mean, the lumped weight vector) is RANK-LOCAL -- partition-dependent and silently
    wrong on a distributed mesh (pre-existing on main for the dof-mean reads; the branch added
    the volume weights). All gate evidence is serial. The class must REFUSE a parallel comm
    (the repo's scope-guard pattern) rather than degrade silently."""
    class _Comm:
        size = 2
    class _Mesh:
        comm = _Comm()
    class _V:
        mesh = _Mesh()
    class _Feat:
        V = _V()
    for direction in ("disperse", "drain"):
        with pytest.raises(ValueError, match="serial"):
            WellIndexExchange(direction=direction).setup(_Feat(), LOAM, {"t0": 1e-4, "R_out": 2.0})


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


def test_drain_drive_is_inverse_retention_of_the_theta_mean():
    """Follow-up #4 (2026-06-12): the drain drive must be psi(volume-mean theta) -- the
    closed-box WATER BALANCE the validated PSS closure is defined on (the lumped theta mean is
    mass-conserving on the discrete field, so it is exact independent of how badly the coarse
    grid resolves the drawdown cone, and it is recharge-aware: injected water raises the theta
    field) -- NOT the plain dof-mean psi (biased WET: boundary-vertex over-weighting + the
    Jensen gap of averaging psi across the cone; measured +8-10% end bias on the drain legs).
    Non-uniform field: the rate must equal the PSS law on the theta-mean drive (independently
    recomputed here) and measurably differ from the dof-mean drive."""
    import ufl
    from dolfinx import fem
    feat = _feat(2)
    R_out = 2.0
    x = WellIndexExchange(direction="drain").setup(feat, LOAM, {"t0": 1e-4, "R_out": R_out})
    feat.Hf.x.array[:] = -1.0
    feat.Hf.x.scatter_forward()
    psi = fem.Function(feat.V)
    xc = feat.V.tabulate_dof_coordinates()
    psi.x.array[:] = -0.03 - 0.4 * xc[:, 0]               # wet-to-dry along the feature axis
    rate = x.pre_step(feat, psi, 0.0)
    v = ufl.TestFunction(feat.V)
    w = fem.assemble_vector(fem.form(v * ufl.dx(
        metadata={"quadrature_rule": "vertex", "quadrature_degree": 1}))).array
    th_bar = float((w * LOAM.theta(psi.x.array)).sum() / w.sum())
    se_vg = (th_bar - LOAM.theta_r) / (LOAM.theta_s - LOAM.theta_r) * LOAM.Sc
    psi_bar = -((se_vg ** (-1.0 / LOAM.m) - 1.0) ** (1.0 / LOAM.n)) / LOAM.alpha
    geo = np.log(R_out / R_W_DEFAULT) - 0.75
    want = -float(LOAM.kirchhoff(-1.0, psi_bar)) / (R_W_DEFAULT * geo) \
        * feat._perimeter * feat.length
    assert abs(rate - want) < 1e-9 * abs(want), \
        f"drain drive is not psi(theta-mean): rate={rate:.6e} want={want:.6e}"
    dof_drive = -float(LOAM.kirchhoff(-1.0, float(psi.x.array.mean()))) / (R_W_DEFAULT * geo) \
        * feat._perimeter * feat.length
    assert abs(rate - dof_drive) > 1e-3 * abs(rate), "test field fails to separate the reads"


def test_drain_rate_heun_predictor_with_dt():
    """Follow-up #4 part 2 (2026-06-12): the prescribed drain rate evaluated at the START state
    and held over the step is a first-order EXPLICIT LAG -- measured +4.7% end excess on the
    refD40 mark grid (1 substep/mark; x1.020 law-bias x 1.047 lag = the embedded 1.068 EXACTLY;
    scratch/_c_lag_check evidence in the commit). With dt known before the solve, pre_step takes
    an optional dt and returns the HEUN (trapezoid) rate in the scheme's own mass variable:
    rate = (r0 + r1)/2 with r1 the PSS rate at theta_bar - r0*dt/V_box. dt omitted/0 degenerates
    to r0 (backward compatible; second-order in dt otherwise, no knobs)."""
    from dolfinx import fem
    import ufl
    feat = _feat(2)
    R_out = 2.0
    x = WellIndexExchange(direction="drain").setup(feat, LOAM, {"t0": 1e-4, "R_out": R_out})
    feat.Hf.x.array[:] = -1.0
    feat.Hf.x.scatter_forward()
    psi = fem.Function(feat.V)
    psi.x.array[:] = -0.03
    r0 = x.pre_step(feat, psi, 0.0)                       # dt omitted -> the plain PSS rate
    geo = np.log(R_out / R_W_DEFAULT) - 0.75
    want0 = -float(LOAM.kirchhoff(-1.0, -0.03)) / (R_W_DEFAULT * geo) \
        * feat._perimeter * feat.length
    assert abs(r0 - want0) < 1e-9 * abs(want0)
    dt = 0.5
    v = ufl.TestFunction(feat.V)
    w = fem.assemble_vector(fem.form(v * ufl.dx(
        metadata={"quadrature_rule": "vertex", "quadrature_degree": 1}))).array
    th_pred = float(LOAM.theta(-0.03)) - abs(want0) * dt / w.sum()
    se_vg = (th_pred - LOAM.theta_r) / (LOAM.theta_s - LOAM.theta_r) * LOAM.Sc
    psi_pred = -((se_vg ** (-1.0 / LOAM.m) - 1.0) ** (1.0 / LOAM.n)) / LOAM.alpha
    r1 = -float(LOAM.kirchhoff(-1.0, psi_pred)) / (R_W_DEFAULT * geo) \
        * feat._perimeter * feat.length
    want_heun = 0.5 * (want0 + r1)
    rate = x.pre_step(feat, psi, 0.0, dt)
    assert abs(rate - want_heun) < 1e-9 * abs(want_heun), \
        f"Heun predictor mismatch: rate={rate:.8e} want={want_heun:.8e}"
    assert abs(rate) < abs(r0), "the predictor must reduce the start-state rate on depletion"


@pytest.mark.parametrize("n", [8])
def test_drain_gate_refD40_closed_box(n):
    """THE drain deployment gate leg, production class through the closed-box harness: the embedded
    I(t) tracks the resolved refD40 (LOAM, R=40 r_w, psi_i=-0.03, 20-d window) within the
    pre-registered EMBEDDED_TOL while the FIXED-DRIVE twin (no host read -- the same law with the
    drive frozen at psi_i, exactly reproducible offline for a prescribed-rate scheme) fails by
    >= BASELINE_KILL: the host read is load-bearing (probe with the theta-mean + Heun drive,
    follow-up #4: 3.4/3.4% at n=8/12 -- n-INDEPENDENT, end I/ref=1.019 = the offline law's own
    1.020; the original dof-mean read scored 7.4/6.9% at end 1.085; twin 74.2%). (~4 min; prior
    failures: WI-bridge drain 10.9/16.8% degrading, throttle 98%.)"""
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


@pytest.mark.parametrize("n", [8])
def test_drain_gate_loamR20_deep_depletion_closed_box(n):
    """DEEP-DEPLETION drain gate (2026-06-12, the geometry-generality + deepest-bend leg): the
    production drain tracks the LOAM R20 fresh ref (48% depletion -- the deepest committed
    bend; the reference was PRE-REGISTERED-predicted by the offline PSS at 3.9% before it
    existed) within EMBEDDED_TOL while the fixed-drive twin fails at 135.8%. Probe with the
    theta-mean + Heun drive: 3.8/3.8% at n=8/12, n-INDEPENDENT, end 1.040 = the offline law's
    own 1.042 on this ref (the embedded discretization adds nothing). (~2 min.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded
    fresh = np.load("scratch/m4_phase4_drain_fresh_refs.npz")
    t, I_ref = fresh["LOAM_R20_t"], fresh["LOAM_R20_I"]
    out = run_embedded(WellIndexExchange(direction="drain"), "LOAM", 20 * R_W_DEFAULT, n, t,
                       direction="drain")
    assert out is not None, "embedded closed-box R20 drain run did not complete"
    e = rel_l2(out["I"], I_ref)
    assert e <= 0.10, f"R20 deep-depletion drain gate failed: relL2={e:.1%}"
    geo = np.log(20.0) - 0.75
    rate_fixed = float(LOAM.kirchhoff(-1.0, -0.03)) / (R_W_DEFAULT * geo)
    I_twin = np.minimum(rate_fixed * (t - t[0]), float(fresh["LOAM_R20_Imax"]))
    assert rel_l2(I_twin, I_ref) >= 0.20, \
        "discrimination twin lost: the fixed-drive PSS clock passes this leg"


@pytest.mark.parametrize("n", [8])
def test_drain_gate_siltR40_closed_box(n):
    """SILT drain gate (2026-06-12, completing the drain SOIL TRIAD: LOAM + SAND + SILT all
    gate-asserted): the production drain tracks the new SILT R40 resolved ref
    (scratch/m4_phase4_silt_drain_ref.py, fresh-refs taint discipline -- the offline PSS
    prediction was saved BEFORE the FEM reference and scored 4.1%, the law's FOURTH independent
    fresh-ref pass) within EMBEDDED_TOL while the fixed-drive twin fails at 53.2%. Probe with
    the theta-mean + Heun drive: 4.3/4.3% at n=8/12, n-INDEPENDENT, end 1.010 = the offline
    law exactly. (~2 min.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded, SOILS
    ref = np.load("scratch/m4_phase4_silt_drain_ref.npz")
    t, I_ref = ref["SILT_R40_t"], ref["SILT_R40_I"]
    out = run_embedded(WellIndexExchange(direction="drain"), "SILT", 40 * R_W_DEFAULT, n, t,
                       direction="drain")
    assert out is not None, "embedded closed-box SILT drain run did not complete"
    e = rel_l2(out["I"], I_ref)
    assert e <= 0.10, f"SILT drain gate failed: relL2={e:.1%}"
    geo = np.log(40.0) - 0.75
    rate_fixed = float(SOILS["SILT"].kirchhoff(-1.0, -0.03)) / (R_W_DEFAULT * geo)
    I_twin = np.minimum(rate_fixed * (t - t[0]), float(ref["SILT_R40_Imax"]))
    assert rel_l2(I_twin, I_ref) >= 0.20, \
        "discrimination twin lost: the fixed-drive PSS clock passes this leg"


@pytest.mark.parametrize("n", [8])
def test_drain_history_gate_refD40C_continuous_recharge(n):
    """THE discriminating drain HISTORY leg (refD40-C, 2026-06-12): CONTINUOUS recharge into the
    [24,32] r_w band over [3,18] d (V=1.8 m at ~0.85x the band->wall PSS throughput, 30-d horizon
    -- the band re-drains while being refilled, so the injection is throughput-limited, not
    storage-capped). Designed because the v1 single deficit-aware pulse (refD40-B) measurably
    CANNOT discriminate: a one-shot refill is capped at ~one band deficit (~6% of end-I), moved
    the end state only +2.0%, and the recharge-blind twin passed it at 1.5%. Here the recharge
    shifts the end state +54.7%, and the production drain -- live volumetric-mean host read, ZERO
    recharge knowledge (the recharge reaches it only THROUGH the host) -- must track the recharged
    resolved reference within the pre-registered EMBEDDED_TOL while the recharge-blind
    water-balance PSS twin (the strongest no-recharge-knowledge competitor: the validated PSS
    closure tracking its OWN extraction ledger; pre-registered prediction was production ~7%,
    probe measured 3.0/2.7% at n=8/12 with the dof-mean drive, 2.7/2.7% end 0.996 with the
    follow-up-4 theta-mean + Heun drive, twin 31.8%) fails by >= BASELINE_KILL. (~10 min.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded
    from scratch.m4_phase4_drain_desorptivity import bruce_klute_desorptivity, pss_drain
    ref = np.load("scratch/m4_phase4_refD40C_drain.npz")
    t, I_ref = ref["LOAM_t"], ref["LOAM_I"]
    t1, t2 = (float(x) for x in ref["LOAM_t_src"])
    v_len = float(ref["LOAM_V_per_wall_area"]) * 2.0 * np.pi * R_W_DEFAULT
    out = run_embedded(WellIndexExchange(direction="drain"), "LOAM", 40 * R_W_DEFAULT, n, t,
                       direction="drain", pulse=(t1, t2, v_len),
                       pulse_band=tuple(float(x) for x in ref["LOAM_band"]))
    assert out is not None, "embedded closed-box recharged drain run did not complete"
    e = rel_l2(out["I"], I_ref)
    assert e <= 0.10, f"drain history gate failed: relL2={e:.1%}"
    S_bk, _, _, _ = bruce_klute_desorptivity(LOAM, -1.0, -0.03)
    I_twin = pss_drain(t, LOAM, -0.03, -1.0, R_W_DEFAULT, 40 * R_W_DEFAULT,
                       S_bk * np.sqrt(t[0]), float(t[0]))
    assert rel_l2(I_twin, I_ref) >= 0.20, \
        "discrimination twin lost: the recharge-blind water-balance PSS clock passes this leg"


@pytest.mark.parametrize("soil", ["SAND", "SILT"])
def test_gate_R40_soil_generality_closed_box(soil):
    """SOIL GENERALITY of the disperse deployment gate (follow-up #2a, 2026-06-12): the
    production exchange -- no soil-specific constants anywhere; S/dtheta/Kirchhoff all derived
    from the constitutive -- passes the NEW SAND and SILT R40 full-depletion legs while the raw
    offline clock fails them (SAND 39.4%, SILT 214.5%). Pre-registered <= EMBEDDED_TOL at
    n=8/12; probe measured SAND 2.4/1.9%, SILT 2.7/4.0% (SILT's n=8->12 rise = the known WI-era
    systematic growing with the WI-era share, the LOAM-review adjudication, not mesh
    degradation). (~2-3 min each.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded
    a40 = np.load("tests/data/m4_phase4_refA40_sand_silt.npz")
    t, I_ref = a40[f"{soil}_R40_t"], a40[f"{soil}_R40_I"]
    out = run_embedded(WellIndexExchange(), soil, 40 * R_W_DEFAULT, 8, t)
    assert out is not None, f"embedded closed-box {soil} run did not complete"
    e = rel_l2(out["I"], I_ref)
    assert e <= DISPERSE_TOL, f"{soil} R40 gate failed: relL2={e:.1%} > {DISPERSE_TOL:.0%}"
    clk = sorptive_clock(t, float(a40[f"{soil}_S"]), float(a40[f"{soil}_dtheta"]),
                         R_W_DEFAULT, F_cylindrical)
    assert rel_l2(clk, I_ref) >= 0.20, "discrimination twin lost: the offline clock passes"


@pytest.mark.parametrize("n", [8])
def test_drain_gate_sandR40_closed_box(n):
    """SOIL GENERALITY of the drain deployment gate (follow-up #2b, 2026-06-12): the production
    PSS depletion drain passes the SAND R40 leg against the 2026-06-12 PRE-REGISTERED fresh-ref
    resolved curve (scratch/m4_phase4_drain_fresh_refs.npz, ~20% depletion window; the offline
    PSS predicted it at 3.4% BEFORE that reference existed) while the fixed-drive twin (drive
    frozen at psi_i -- no host read) fails at 82.1%. Probe with the theta-mean + Heun drive
    (follow-up #4): 3.5/3.5% at n=8/12, n-INDEPENDENT, end bias +3.0% (the dof-mean read scored
    8.8/8.3% with end +10.2/9.6%). (~2 min.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded, SOILS
    fresh = np.load("scratch/m4_phase4_drain_fresh_refs.npz")
    t, I_ref = fresh["SAND_R40_t"], fresh["SAND_R40_I"]
    out = run_embedded(WellIndexExchange(direction="drain"), "SAND", 40 * R_W_DEFAULT, n, t,
                       direction="drain")
    assert out is not None, "embedded closed-box SAND drain run did not complete"
    e = rel_l2(out["I"], I_ref)
    assert e <= 0.10, f"SAND drain gate failed: relL2={e:.1%}"
    geo = np.log(40.0) - 0.75
    rate_fixed = float(SOILS["SAND"].kirchhoff(-1.0, -0.03)) / (R_W_DEFAULT * geo)
    I_twin = np.minimum(rate_fixed * (t - t[0]), float(fresh["SAND_R40_Imax"]))
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
    x = WellIndexExchange().setup(feat, LOAM, {"t0": 1e-4, "R_out": 1.0})
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
    x.setup(feat, LOAM, {"t0": 1e-4, "R_out": 1.0})
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


def test_dualscale_kill_baseline_regression():
    """The RETRACTED dual-scale baseline must keep FAILING the gate (kill-map regression,
    follow-up #5 of the 2026-06-12 list): the frozen DualScaleScheme reimplementation in the
    harness (behavior-identity verified against the excised production code, 27.8% on LOAM R3
    n=8) must fail the depleting-reservoir leg by >= BASELINE_KILL. If this ever PASSES, STOP:
    the gate no longer discriminates passive accumulators (the harness docstring's warning,
    asserted instead of remembered). (~1-2 min FEM smoke.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded, DualScaleScheme
    refA = np.load("tests/data/m4_phase4_refA_disperse.npz")
    t, I_ref = refA["LOAM_R3_t"], refA["LOAM_R3_I"]
    out = run_embedded(DualScaleScheme("disperse"), "LOAM", 3 * R_W_DEFAULT, 8, t)
    assert out is not None, "dual-scale baseline run did not complete"
    e = rel_l2(out["I"], I_ref)
    assert e >= 0.20, \
        f"the retracted dual-scale baseline PASSES the gate ({e:.1%}) -- discrimination lost, STOP"


DISPERSE_TOL = 0.03     # item-A pre-registered target (2026-06-13): the resolved-ring WI read drops
#                         the disperse worst from 5.7% (on-ridge) to <=3% across n=8/12 + the soil
#                         triad + the RefB40 history leg (scratch/m4_phase4_wi_ring_derivation.py derivation; the
#                         Heun-corrected prescribed rate + live recharge-aware capacity throttle).


@pytest.mark.parametrize("n", [8, 12])
def test_gate_refA40_closed_box(n):
    """THE discriminating gate, production class through the closed-box harness: the embedded I(t)
    tracks the resolved depleting-reservoir reference (LOAM, R=40 r_w, full depletion) within the
    item-A DISPERSE_TOL at BOTH meshes while the offline clock fails the same leg; both mass ledgers
    are asserted every sample inside the harness. (~3 min each; measured 2.0%/2.6% at n=8/12 -- the
    resolved-ring read, vs the on-ridge 2.2%/5.7%.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded
    refA = np.load("tests/data/m4_phase4_refA_disperse.npz")
    t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
    out = run_embedded(WellIndexExchange(), "LOAM", 40 * R_W_DEFAULT, n, t)
    assert out is not None, "embedded closed-box run did not complete"
    e = rel_l2(out["I"], I_ref)
    assert e <= DISPERSE_TOL, f"gate failed: relL2={e:.1%} > {DISPERSE_TOL:.0%}"
    S, dth = float(refA["LOAM_S"]), float(refA["LOAM_dtheta"])
    clk = sorptive_clock(t, S, dth, R_W_DEFAULT, F_cylindrical)
    assert rel_l2(clk, I_ref) >= 0.20, "discrimination twin lost: the offline clock passes this leg"


RW_EMBEDDED_TOL = 0.10  # item-C pre-registered (2026-06-16, locked before the resolved-wall sweep): the
#                         realistic deployment band (large R_out + fine mesh) gate bar; the established
#                         gate constant for a NEW regime (the 0.03 deployment polish does not apply).


@pytest.mark.parametrize("n", [16])
def test_resolved_wall_gate_refA40_auto_allowed(n):
    """Item C (2026-06-16): in the VALIDATED resolved-wall band (R_out=40 r_w, fine mesh h=4.43 r_w <
    5.5 r_w -- the regime the DEPLOYMENT gate above REFUSED) the DEFAULT constructor (auto-allowed, NO
    opt-in -- the honest fence's warrant) tracks RefA40 within RW_EMBEDDED_TOL while the offline clock
    fails. End-to-end proof the fence's auto-allow works. (resolved-wall sweep: n16/24/32 = 2.7/3.5/4.3%;
    Codex-reviewed -- the discrimination is ensemble [+ RefB40 + drain], the analytic mode-2 fence
    refuted, the R10 small-R_out failure is refinement-budget, not deployment-scale.) (~3 min.)"""
    from scratch.m4_phase4_embedded_harness import run_embedded
    refA = np.load("tests/data/m4_phase4_refA_disperse.npz")
    t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
    out = run_embedded(WellIndexExchange(), "LOAM", 40 * R_W_DEFAULT, n, t)   # DEFAULT ctor: auto-allow
    assert out is not None, "embedded resolved-wall run did not complete"
    assert out["h"] < 5.5 * R_W_DEFAULT, "test must be in the resolved-wall regime (h < 5.5 r_w)"
    e = rel_l2(out["I"], I_ref)
    assert e <= RW_EMBEDDED_TOL, f"resolved-wall gate failed: relL2={e:.1%} > {RW_EMBEDDED_TOL:.0%}"
    clk = sorptive_clock(t, float(refA["LOAM_S"]), float(refA["LOAM_dtheta"]), R_W_DEFAULT, F_cylindrical)
    assert rel_l2(clk, I_ref) >= 0.20, "discrimination twin lost: the offline clock passes this leg"
