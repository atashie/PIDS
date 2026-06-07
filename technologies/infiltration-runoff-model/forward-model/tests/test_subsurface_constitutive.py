"""Tier-1 sanity (subsurface constitutive closures): van Genuchten-Mualem.

These pure-function closures theta(psi) and K(psi) underpin the mixed-form
Richards residual (Module 1). Per governance/claude-sanity-check-routine.md they
are validated against the closed-form / known limits before any solver is built.

Soil: Carsel & Parrish (1988) loam, SI units (length m, time day).
"""
import numpy as np
import pytest

from pids_forward.physics.constitutive import VanGenuchten

# Carsel & Parrish (1988) loam, SI (length m, TIME = DAYS). Ks = 0.2496 m/day
# (= 1.04 cm/hr); an earlier 0.0104 was m/hr mislabelled as m/day (a 24x error).
LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)
# Carsel & Parrish sand -- steep K(psi) rise near psi=0 (the stiff sorptivity case).
SAND = dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.128)


def test_saturated_at_nonnegative_pressure_head():
    """At psi >= 0 the soil is fully saturated: Se == 1 and theta == theta_s exactly."""
    soil = VanGenuchten(**LOAM)
    assert soil.effective_saturation(0.0) == 1.0
    assert soil.theta(0.0) == LOAM["theta_s"]
    # Positive (ponded) pressure head is still saturated, not super-saturated.
    assert soil.effective_saturation(0.05) == 1.0
    assert soil.theta(0.05) == LOAM["theta_s"]


def test_unsaturated_retention_follows_van_genuchten():
    """For psi < 0, Se = (1 + |alpha*psi|^n)^-m with m = 1 - 1/n; theta -> theta_r dry."""
    soil = VanGenuchten(**LOAM)
    # Hand reference at psi = -1.0 m (van Genuchten Se, renormalized by air-entry Sc):
    #   Se_vg = (1 + 3.6^1.56)^-0.358974 = 0.46626 ; Sc(h_s=-0.02) = 0.99414
    #   Se = Se_vg / Sc = 0.46901
    assert soil.effective_saturation(-1.0) == pytest.approx(0.4690, abs=1e-3)
    # Dry limit: Se -> 0, theta -> theta_r. NOTE the approach is a SLOW power law
    # (Se ~ (alpha|psi|)^-(n-1) = (alpha|psi|)^-0.56 here), so reaching < 1e-6
    # requires an extreme suction (~1e12 m); at -1e6 m Se is still ~2e-4.
    assert soil.effective_saturation(-1.0e12) == pytest.approx(0.0, abs=1e-6)
    assert soil.theta(-1.0e12) == pytest.approx(LOAM["theta_r"], abs=1e-6)
    # Monotone increasing toward saturation; strictly within (0, 1].
    se = [soil.effective_saturation(p) for p in (-100.0, -10.0, -1.0, -0.1, -0.01)]
    assert all(a < b for a, b in zip(se, se[1:]))
    assert all(0.0 < s <= 1.0 for s in se)


def test_mualem_conductivity():
    """K(psi) = Ks*Se^L*(1-(1-Se^(1/m))^m)^2 (Mualem 1976, L=0.5); K(psi>=0)=Ks."""
    soil = VanGenuchten(**LOAM)
    # Saturated branch is exactly Ks.
    assert soil.K(0.0) == LOAM["Ks"]
    assert soil.K(0.1) == LOAM["Ks"]
    # Hand reference at psi = -1.0 m, air-entry (Ippisch 2006) form, Sc = 0.99414:
    #   K = Ks * Se^0.5 * (num/den)^2, Se=0.46901, num=0.04460, den=0.77212
    #     = 0.2496 * 0.68484 * (0.05776)^2 = 5.70e-4 m/day
    assert soil.K(-1.0) == pytest.approx(5.70e-4, rel=1e-2)
    # Drier => lower K; strictly within (0, Ks].
    ks = [soil.K(p) for p in (-100.0, -10.0, -1.0, -0.1, -0.01)]
    assert all(a < b for a, b in zip(ks, ks[1:]))
    assert all(0.0 < k <= LOAM["Ks"] for k in ks)


def test_specific_moisture_capacity():
    """C(psi) = dtheta/dpsi (1/m): 0 in the saturated zone; analytic vG derivative else."""
    soil = VanGenuchten(**LOAM)
    # Saturated: theta is constant, so capacity is exactly 0.
    assert soil.capacity(0.0) == 0.0
    assert soil.capacity(0.5) == 0.0
    # Hand reference at psi = -1.0 m (air-entry: C = C_vanGenuchten / Sc):
    #   C_vg = 0.352*3.6*0.56*2.0490*0.055665 = 0.0809 ; C = 0.0809/0.99414 = 0.0814 1/m
    assert soil.capacity(-1.0) == pytest.approx(0.0814, rel=1e-2)
    # Independent check: analytic C must match a central difference of theta(psi).
    h = 1e-6
    for psi in (-0.2, -1.0, -5.0):
        fd = (soil.theta(psi + h) - soil.theta(psi - h)) / (2.0 * h)
        assert soil.capacity(psi) == pytest.approx(fd, rel=1e-4)
    # Non-negative everywhere (wetter => more water).
    assert all(soil.capacity(p) >= 0.0 for p in (-100.0, -10.0, -1.0, -0.1, 0.0, 0.5))


def test_air_entry_saturation_point():
    """Vogel/Ippisch air entry: the soil saturates at h_s (Se=1, K=Ks), not at psi=0."""
    soil = VanGenuchten(**LOAM)
    hs = soil.h_s
    assert soil.effective_saturation(hs) == pytest.approx(1.0, abs=1e-12)
    assert soil.K(hs) == pytest.approx(soil.Ks, rel=1e-9)
    assert soil.theta(hs) == pytest.approx(LOAM["theta_s"], rel=1e-9)
    # Just drier than the air-entry head, the soil is no longer fully saturated.
    assert soil.effective_saturation(hs * 2.0) < 1.0


# --- Kirchhoff matric-flux-potential (the sorptive soil-exchange leg, design §D spec 2026-06-06) ---

def _dense_kirchhoff(soil, a, b, m=40000):
    """High-resolution reference for the matric-flux-potential difference int_a^b K(psi) dpsi."""
    s = np.linspace(a, b, m + 1)
    k = soil.K(s)
    return float(np.sum(0.5 * (k[:-1] + k[1:]) * np.diff(s)))  # trapezoid (np.trapz removed in numpy 2)


def test_kirchhoff_matric_flux_potential():
    """``kirchhoff(a, b) = int_a^b K(psi) dpsi`` (the matric flux potential difference the sorptive
    exchange leg needs): accurate vs a dense reference across ranges -- including the STEEP sand rise
    near psi=0 and ranges crossing into the ponded (psi>0 -> K=Ks) zone -- and FTC-consistent
    (d/db int_a^b K = K(b)). The quadrature order/grading is chosen here, NOT in the residual.
    """
    for params in (LOAM, SAND):
        soil = VanGenuchten(**params)
        for a, b in [(-1.0, 0.0), (-0.5, 0.02), (-2.0, -0.1), (-0.1, 0.0), (-3.0, 0.05)]:
            approx = soil.kirchhoff(a, b)
            ref = _dense_kirchhoff(soil, a, b)
            assert abs(approx - ref) <= 1e-2 * abs(ref) + 1e-12, \
                f"Ks={params['Ks']} [{a},{b}]: kirchhoff {approx:.6e} vs ref {ref:.6e}"
        # Fundamental theorem of calculus: kirchhoff(psi, psi+h)/h -> K(psi).
        for psi in (-0.8, -0.3, -0.05):
            h = 1e-5
            assert abs(soil.kirchhoff(psi, psi + h) / h - soil.K(psi)) <= 1e-3 * soil.K(psi) + 1e-9
        # Exfiltration SIGN: soil wetter than the surface (a > b) gives a NEGATIVE matric potential
        # difference (flux soil->surface); infiltration (a < b) gives positive. (The graded rule
        # clusters near the wet end b, so it is not exactly antisymmetric -- the exchange always passes
        # the wetter head as b, i.e. kirchhoff(psi_top, d); only the sign matters for direction.)
        assert soil.kirchhoff(-0.1, -0.6) < 0.0 < soil.kirchhoff(-0.6, -0.1)


def test_kirchhoff_ufl_matches_numpy():
    """The UFL ``kirchhoff_ufl`` (used in the coupled residual) equals the numpy ``kirchhoff`` to
    machine precision (same quadrature rule), so the convergence test above certifies both."""
    import ufl
    from mpi4py import MPI
    from dolfinx import mesh as dmesh, fem

    soil = VanGenuchten(**LOAM)
    msh = dmesh.create_interval(MPI.COMM_WORLD, 1, [0.0, 1.0])  # unit cell -> int over dx == the value
    for a, b in [(-0.7, 0.01), (-2.0, -0.2)]:
        ca, cb = fem.Constant(msh, float(a)), fem.Constant(msh, float(b))
        val = fem.assemble_scalar(fem.form(soil.kirchhoff_ufl(ca, cb) * ufl.dx))
        assert abs(val - soil.kirchhoff(a, b)) <= 1e-12 * abs(soil.kirchhoff(a, b)) + 1e-14
