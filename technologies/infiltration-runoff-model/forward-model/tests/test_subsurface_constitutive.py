"""Tier-1 sanity (subsurface constitutive closures): van Genuchten-Mualem.

These pure-function closures theta(psi) and K(psi) underpin the mixed-form
Richards residual (Module 1). Per governance/claude-sanity-check-routine.md they
are validated against the closed-form / known limits before any solver is built.

Soil: Carsel & Parrish (1988) loam, SI units (length m, time day).
"""
import pytest

from pids_forward.physics.constitutive import VanGenuchten

# Carsel & Parrish (1988) loam, SI (length m, TIME = DAYS). Ks = 0.2496 m/day
# (= 1.04 cm/hr); an earlier 0.0104 was m/hr mislabelled as m/day (a 24x error).
LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)


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
    # Hand reference at psi = -1.0 m:
    #   m = 1 - 1/1.56 = 0.358974 ; (alpha|psi|)^n = 3.6^1.56 = 7.3762
    #   Se = (1 + 7.3762)^-0.358974 = 0.46626
    assert soil.effective_saturation(-1.0) == pytest.approx(0.4663, abs=1e-3)
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
    # Hand reference at psi = -1.0 m (Se = 0.46626, m = 0.358974, L = 0.5):
    #   K = 0.2496 * 0.46626^0.5 * (1 - (1 - 0.46626^(1/m))^m)^2 = 3.390e-4 m/day
    assert soil.K(-1.0) == pytest.approx(3.39e-4, rel=1e-2)
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
    # Hand reference at psi = -1.0 m (u = alpha|psi| = 3.6; m*n = n-1 = 0.56):
    #   C = (theta_s-theta_r)*alpha*m*n*u^(n-1)*(1+u^n)^(-m-1)
    #     = 0.352*3.6*0.56*2.0490*0.055665 = 0.0809 1/m
    assert soil.capacity(-1.0) == pytest.approx(0.0809, rel=1e-2)
    # Independent check: analytic C must match a central difference of theta(psi).
    h = 1e-6
    for psi in (-0.2, -1.0, -5.0):
        fd = (soil.theta(psi + h) - soil.theta(psi - h)) / (2.0 * h)
        assert soil.capacity(psi) == pytest.approx(fd, rel=1e-4)
    # Non-negative everywhere (wetter => more water).
    assert all(soil.capacity(p) >= 0.0 for p in (-100.0, -10.0, -1.0, -0.1, 0.0, 0.5))
