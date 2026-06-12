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
    assert e <= 0.10, f"{soil} R40 gate failed: relL2={e:.1%}"
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
