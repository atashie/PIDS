"""Module 4 (§E) Phase-3: the FULL-CURVE FIDELITY GATE (claim C-004), built test-first.

The embedded sorptive WALL leg must reproduce the RESOLVED Phase-1 near-field references I(t) over the FULL
curve (not merely the early sqrt-t coefficient) on >=2 geometries. This gate forward-integrates the closure
clock on each reference t-grid and asserts per-curve relative-L2 vs the resolved I(t).

PASS-REQUIRED (parameter-free / a-priori): the 6 DISPERSE curves SAND/LOAM/SILT x {tunnel, annulus}, with
the cylindrical Green-Ampt closure  dI/dt = (S^2/2I)*F_cyl(zeta),  F_cyl = 2z/ln(1+2z),  S = a-priori
PARLANGE sorptivity, NO calibration knob (C=1) -- must be <=5% rel-L2. The constant-kappa PLANAR clock
(F=1) is shown to FAIL the SAME harness (the documented -29..-56% under-prediction of the super-sqrt-t
radial references): the gate discriminates.

FLAGGED (advisory, NON-blocking): (a) CLAY -- the near-saturated degenerate edge (dtheta=0.014, 95%
saturated even at psi=-1), where the sorptive leg is physically negligible (clay-lined features are
conveyance-only); (b) the DRAIN direction -- a VALIDATED sub-sqrt-t conductivity-throttle FORM
(exp(-(z/z0)^k)) but currently SEMI-EMPIRICAL (fitted (z0,k); no robust a-priori desorptivity, the
saturated psi=0 start being a D=K/C->inf singularity). See pids-drain-usecase / pids-module4-starting-note.

Metric: rel-L2 over the I(t) DOMAIN-INTEGRAL arrays ONLY ({SOIL}_tunnel_I / _drain_I / _planar_I). The
{SOIL}_*_pen arrays are node-quantized front DIAGNOSTICS and are NEVER consumed. References are committed
fixtures (tests/data/, from scratch/m4_phase1{b,c}_*_reference.py, adversarially reviewed). Pure numpy
(no DOLFINx) -- the same closure RHS feeds the embedded-feature UFL residual (Phase-4 integration).
"""
from pathlib import Path

import numpy as np
import pytest

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import (
    F_cylindrical,
    F_throttle,
    parlange_sorptivity,
    rel_l2,
    sorptive_clock,
    throttle_params,
    R_W_DEFAULT,
)

DATA = Path(__file__).parent / "data"
R_W = R_W_DEFAULT

# Carsel & Parrish (1988) van Genuchten textures (alpha 1/m, Ks m/day) -- identical to the Phase-1 refs.
SOILS = {
    "SAND": VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13),
    "LOAM": VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25),
    "SILT": VanGenuchten(theta_r=0.034, theta_s=0.46, alpha=1.6, n=1.37, Ks=0.06),
    "CLAY": VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.048),
}
COARSE = ["SAND", "LOAM", "SILT"]          # the non-degenerate soils (CLAY = near-saturated edge, flagged)
GEOM_KEY = {"tunnel": "tunnel_I", "annulus": "drain_I", "planar": "planar_I"}

DISP_TOL = 0.05        # a-priori disperse PASS-REQUIRED rel-L2 threshold
ADVISORY_TOL = 0.20    # soft documented ceiling for the flagged (drain) curves


def _disp():
    return np.load(DATA / "m4_phase1b_disperse_refs.npz")


def _drain():
    return np.load(DATA / "m4_phase1c_drain_refs.npz")


def _dtheta(soil, psi_i, psi_w):
    return abs(float(soil.theta(psi_w) - soil.theta(psi_i)))


def _disperse_clock(soil_name, t, F):
    """Forward-integrate the disperse clock with shape ``F`` (a-priori Parlange S, C=1)."""
    soil = SOILS[soil_name]
    S = parlange_sorptivity(soil, -1.0, 0.0)
    return sorptive_clock(t, S, _dtheta(soil, -1.0, 0.0), R_W, F)


def _empirical_desorptivity(t, I, k=3):
    """Drain clock anchor = the reference's OWN early limb mean(I/sqrt(t)) over the first k nodes. NEVER
    the *_pen arrays. (No robust a-priori desorptivity: the saturated psi=0 start is a D=K/C->inf
    singularity -- Bruce-Klute NaN-fails, Parlange-desorption over-predicts ~2x.)"""
    return float(np.mean(I[1 : 1 + k] / np.sqrt(t[1 : 1 + k])))


# --------------------------------------------------------------------------------------------------
# PASS-REQUIRED: the a-priori disperse gate
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("geom", ["tunnel", "annulus"])
@pytest.mark.parametrize("soil_name", COARSE)
def test_disperse_apriori_gate(soil_name, geom):
    """The parameter-free cylindrical Green-Ampt closure (F_cyl + a-priori Parlange S, NO calibration)
    reproduces the resolved disperse reference -- BOTH the z-invariant tunnel AND the gravity-bearing
    annulus -- to <=5% full-curve rel-L2. This is the C-004 gate for the disperse direction."""
    d = _disp()
    t, Iref = d[f"{soil_name}_t"], d[f"{soil_name}_{GEOM_KEY[geom]}"]
    err = rel_l2(_disperse_clock(soil_name, t, F_cylindrical), Iref)
    assert err <= DISP_TOL, f"{soil_name} {geom} disperse rel-L2 {err:.2%} > {DISP_TOL:.0%}"


@pytest.mark.parametrize("geom", ["tunnel", "annulus"])
@pytest.mark.parametrize("soil_name", COARSE)
def test_planar_clock_fails_disperse_gate(soil_name, geom):
    """The constant-kappa PLANAR clock (F=1) UNDER-predicts the super-sqrt-t radial references by a wide
    margin (rel-L2 >> 5%, signed end-error negative) on the SAME harness -- only the shape F differs. This
    is the documented failure the cylindrical correction fixes; it proves the gate is not vacuous."""
    d = _disp()
    t, Iref = d[f"{soil_name}_t"], d[f"{soil_name}_{GEOM_KEY[geom]}"]
    Imod = _disperse_clock(soil_name, t, lambda z: 1.0)
    err = rel_l2(Imod, Iref)
    assert err > DISP_TOL, f"{soil_name} {geom}: planar clock rel-L2 {err:.2%} unexpectedly within tol"
    assert Imod[-1] < Iref[-1], f"{soil_name} {geom}: planar clock should UNDER-predict the radial ref"


# --------------------------------------------------------------------------------------------------
# FLAGGED (advisory, non-blocking): CLAY edge + the drain direction
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("geom", ["tunnel", "annulus"])
def test_clay_disperse_is_flagged_not_apriori(geom):
    """CLAY disperse is the near-saturated degenerate edge: it stays ~sqrt-t (dtheta=0.014 can't build the
    radial gradient) while the cylindrical correction over-bends, so it does NOT meet the a-priori bar and
    is correctly FLAGGED (clay-lined features are conveyance-only -> sorptive exchange is negligible)."""
    d = _disp()
    t, Iref = d["CLAY_t"], d[f"CLAY_{GEOM_KEY[geom]}"]
    err = rel_l2(_disperse_clock("CLAY", t, F_cylindrical), Iref)
    assert err > DISP_TOL, f"CLAY {geom} unexpectedly within a-priori tol ({err:.2%}) -- revisit flagging"


def _drain_errs(soil_name, geom):
    """(throttle, planar, cyl-up) full-curve rel-L2 for a drain reference; S anchored on its early limb."""
    dr = _drain()
    t, Iref = dr[f"{soil_name}_t"], dr[f"{soil_name}_{GEOM_KEY[geom]}"]
    dth = _dtheta(SOILS[soil_name], 0.0, -1.0)
    S_des = _empirical_desorptivity(t, Iref)
    z0, k = throttle_params(dth)
    e_throttle = rel_l2(sorptive_clock(t, S_des, dth, R_W, lambda z: F_throttle(z, z0, k)), Iref)
    e_planar = rel_l2(sorptive_clock(t, S_des, dth, R_W, lambda z: 1.0), Iref)
    e_cylup = rel_l2(sorptive_clock(t, S_des, dth, R_W, F_cylindrical), Iref)
    return e_throttle, e_planar, e_cylup


@pytest.mark.parametrize("soil_name", list(SOILS))
def test_drain_radial_throttle_form_validated(soil_name):
    """FLAGGED/advisory: the RADIAL (tunnel) drain is sub-sqrt-t (near-wall conductivity throttling as the
    soil desaturates toward psi_wall). The validated throttle FORM exp(-(z/z0)^k) (semi-empirical: (z0,k)
    fitted, S anchored on the reference early limb) beats BOTH the planar clock and the wrong-sign
    cylindrical up-bend on the clean radial geometry, within the advisory band, for EVERY soil. Documents
    the closure FORM; it is NOT a-priori, so it does not gate the build."""
    e_throttle, e_planar, e_cylup = _drain_errs(soil_name, "tunnel")
    assert e_throttle < e_planar, f"{soil_name} tunnel: throttle {e_throttle:.2%} !< planar {e_planar:.2%}"
    assert e_throttle < e_cylup, f"{soil_name} tunnel: throttle {e_throttle:.2%} !< cyl-up {e_cylup:.2%}"
    assert e_throttle <= ADVISORY_TOL, f"{soil_name} tunnel: throttle {e_throttle:.2%} > advisory {ADVISORY_TOL:.0%}"


@pytest.mark.parametrize("soil_name", list(SOILS))
def test_drain_annulus_within_advisory(soil_name):
    """The gravity-bearing ANNULUS (horizontal drain) stays within the advisory band with the same
    throttle. Gravity adds a downward-plume up-bend (the disperse-like effect) that partially counteracts
    the throttle -- a recorded, non-gating limitation of the semi-empirical drain leg."""
    e_throttle, _, _ = _drain_errs(soil_name, "annulus")
    assert e_throttle <= ADVISORY_TOL, f"{soil_name} annulus: throttle {e_throttle:.2%} > advisory {ADVISORY_TOL:.0%}"


def test_drain_sand_annulus_is_gravity_dominated_exception():
    """SAND drain annulus is the documented GRAVITY exception: the coarse-soil horizontal drain up-bends
    (gravity plume), so its late slope (~0.55) is far less sub-sqrt-t than the tunnel (~0.35) and the
    radial throttle OVER-suppresses -- the cylindrical up-bend fits it better. Locks the finding (a gravity
    add-back for coarse horizontal drains is the noted future fix; see pids-drain-usecase)."""
    e_throttle, e_planar, e_cylup = _drain_errs("SAND", "annulus")
    assert e_cylup < e_throttle, f"SAND annulus: expected gravity-dominated (cyl-up {e_cylup:.2%} < throttle {e_throttle:.2%})"
    assert e_planar < e_throttle, f"SAND annulus: expected gravity-dominated (planar {e_planar:.2%} < throttle {e_throttle:.2%})"


# --------------------------------------------------------------------------------------------------
# Summary table (informational; also asserts the headline a-priori verdict)
# --------------------------------------------------------------------------------------------------
def test_gate_summary_table(capsys):
    """Print the full per-curve table and assert the headline: all 6 a-priori disperse curves PASS <=5%."""
    d, dr = _disp(), _drain()
    rows, disperse_pass = [], True
    for name in SOILS:
        # disperse (a-priori cyl)
        for geom in ("tunnel", "annulus"):
            t, Iref = d[f"{name}_t"], d[f"{name}_{GEOM_KEY[geom]}"]
            err = rel_l2(_disperse_clock(name, t, F_cylindrical), Iref)
            req = name in COARSE
            status = ("PASS" if err <= DISP_TOL else "FAIL") if req else "FLAG"
            if req and err > DISP_TOL:
                disperse_pass = False
            rows.append((name, geom, "disperse", err, status))
        # drain (semi-empirical throttle)
        for geom in ("tunnel", "annulus"):
            t, Iref = dr[f"{name}_t"], dr[f"{name}_{GEOM_KEY[geom]}"]
            dth = _dtheta(SOILS[name], 0.0, -1.0)
            z0, k = throttle_params(dth)
            err = rel_l2(sorptive_clock(t, _empirical_desorptivity(t, Iref), dth, R_W,
                                        lambda z: F_throttle(z, z0, k)), Iref)
            rows.append((name, geom, "drain", err, "FLAG"))
    with capsys.disabled():
        print("\n  PHASE-3 FULL-CURVE FIDELITY GATE (rel-L2 over I(t) domain integral; *_pen NEVER used)")
        print(f"  disperse a-priori PASS-REQUIRED <= {DISP_TOL:.0%} (SAND/LOAM/SILT); CLAY + drain FLAGGED")
        print("  " + "-" * 60)
        print(f"  {'soil':5s} {'geom':8s} {'dir':9s} {'rel-L2':>8s}  status")
        for name, geom, direc, err, status in rows:
            print(f"  {name:5s} {geom:8s} {direc:9s} {err:7.2%}  {status}")
    assert disperse_pass, "a PASS-REQUIRED disperse curve exceeded the a-priori tolerance"
