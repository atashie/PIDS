"""Tier-1 sanity (subsurface constitutive closures): van Genuchten-Mualem.

These pure-function closures theta(psi) and K(psi) underpin the mixed-form
Richards residual (Module 1). Per governance/claude-sanity-check-routine.md they
are validated against the closed-form / known limits before any solver is built.

Soil: Carsel & Parrish (1988) loam, SI units (length m, time day).
"""
import pytest

from pids_forward.physics.constitutive import VanGenuchten

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.0104)


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
