"""van Genuchten (1980) retention and Mualem (1976) conductivity closures.

Pure functions of pressure head ``psi`` (m). Sign convention: ``psi < 0`` in the
unsaturated zone, ``psi >= 0`` saturated/ponded. SI units (length m, time day).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VanGenuchten:
    """van Genuchten-Mualem soil with parameters ``theta_r, theta_s, alpha, n, Ks``."""

    theta_r: float
    theta_s: float
    alpha: float  # inverse air-entry scale (1/m)
    n: float  # pore-size distribution index (> 1)
    Ks: float  # saturated hydraulic conductivity (m/day)
    L: float = 0.5  # Mualem pore-connectivity / tortuosity exponent
    h_s: float = -0.02  # air-entry head (m, <= 0); Vogel/Ippisch saturation cutoff

    @property
    def m(self) -> float:
        return 1.0 - 1.0 / self.n

    @property
    def Sc(self) -> float:
        """van Genuchten saturation at the air-entry head h_s (Ippisch 2006 cutoff).

        Sc == 1 when h_s == 0 (recovers the unmodified van Genuchten-Mualem model).
        With h_s < 0 the curves are renormalized so the soil saturates at h_s rather
        than at h = 0, which keeps K and dK/dh finite *through* saturation.
        """
        u_s = self.alpha * (-self.h_s)
        return (1.0 + u_s**self.n) ** (-self.m)

    def effective_saturation(self, psi):
        """Air-entry-modified effective saturation Se in [0, 1]. Scalar or array.

        van Genuchten (1980) with the Vogel/Ippisch air-entry cutoff: Se reaches 1 at
        psi = h_s (not psi = 0), so the soil is saturated for psi >= h_s.
        """
        u = self.alpha * np.maximum(-psi, 0.0)
        se_vg = (1.0 + u**self.n) ** (-self.m)
        return np.minimum(se_vg / self.Sc, 1.0)

    def theta(self, psi: float) -> float:
        """Volumetric water content."""
        return self.theta_r + self.effective_saturation(psi) * (self.theta_s - self.theta_r)

    def K(self, psi):
        """Air-entry-modified Mualem (1976) conductivity (m/day); K(psi >= h_s) == Ks.

        Ippisch et al. (2006) form: the pore-size integral is cut off at the air-entry
        saturation Sc, so dK/dpsi stays finite through saturation (no Se->1 singularity).
        """
        se = self.effective_saturation(psi)
        sc = self.Sc
        num = 1.0 - (1.0 - (se * sc) ** (1.0 / self.m)) ** self.m
        den = 1.0 - (1.0 - sc ** (1.0 / self.m)) ** self.m
        return self.Ks * se**self.L * (num / den) ** 2

    def capacity(self, psi):
        """Specific moisture capacity C = dtheta/dpsi (1/m). Scalar or array.

        Air-entry-modified: C = C_vanGenuchten / Sc for psi < h_s, and 0 for the
        saturated zone psi >= h_s (theta is constant there).
        """
        u = self.alpha * np.maximum(-psi, 0.0)
        c_vg = (
            (self.theta_s - self.theta_r)
            * self.alpha
            * self.m
            * self.n
            * u ** (self.n - 1.0)
            * (1.0 + u**self.n) ** (-self.m - 1.0)
        )
        return np.where(psi < self.h_s, c_vg / self.Sc, 0.0)

    # --- UFL-symbolic forms (for the Richards residual + auto-diff Jacobian) ---
    # These mirror the float closures above but accept a UFL expression ``psi``.
    # ``max_value(-psi, 0)`` gives the saturated branch (Se=1 at psi>=0) smoothly,
    # avoiding a kink-prone ufl.conditional whose unselected branch can still
    # contaminate the auto-differentiated Jacobian.

    def se_ufl(self, psi):
        import ufl

        u = self.alpha * ufl.max_value(-psi, 0.0)
        se_vg = (1.0 + u**self.n) ** (-self.m)
        return ufl.min_value(se_vg / self.Sc, 1.0)

    def theta_ufl(self, psi):
        return self.theta_r + self.se_ufl(psi) * (self.theta_s - self.theta_r)

    def K_ufl(self, psi):
        se = self.se_ufl(psi)
        sc = self.Sc
        num = 1.0 - (1.0 - (se * sc) ** (1.0 / self.m)) ** self.m
        den = 1.0 - (1.0 - sc ** (1.0 / self.m)) ** self.m
        return self.Ks * se**self.L * (num / den) ** 2

    def capacity_ufl(self, psi):
        """UFL specific moisture capacity C = dtheta/dpsi (1/m); 0 for psi >= h_s."""
        import ufl

        u = self.alpha * ufl.max_value(-psi, 0.0)
        c_vg = (
            (self.theta_s - self.theta_r)
            * self.alpha
            * self.m
            * self.n
            * u ** (self.n - 1.0)
            * (1.0 + u**self.n) ** (-self.m - 1.0)
        )
        return ufl.conditional(ufl.lt(psi, self.h_s), c_vg / self.Sc, 0.0)

    # --- Kirchhoff matric flux potential (sorptive soil-exchange leg; design §D spec 2026-06-06) ---
    # The film Darcy flux uses the FILM-INTEGRAL of K (the matric flux potential), not the dry cell
    # value K(psi_top): when the interface is ponded/saturated, K rises steeply to Ks across the film,
    # which the dry value misses (under-infiltrating dry soil ~50x at coarse resolution). The matric
    # flux potential difference is ``kirchhoff(a, b) = int_a^b K(psi) dpsi``; the sorptive leg uses
    # ``q_pot = kirchhoff(psi_top, d) / ell_c``. Evaluated by a GRADED composite-Simpson rule with nodes
    # clustered near the wet end ``b`` (where K is largest/steepest) via psi(s) = b - (b-a) s^p,
    # s in [0,1] -> a modest order resolves the near-saturation rise even for steep coarse soils. K_ufl
    # is C1 (Vogel air-entry cap -> K=Ks for psi>=h_s automatically integrates the saturated/ponded
    # psi>0 part), so kirchhoff is SMOOTH in (a, b): no max()/conditional kink (plain-Newton friendly).
    # p=3 integrates the substitution Jacobian (3 s^2) exactly, so a constant K is integrated exactly.
    _KIRCHHOFF_N = 32      # composite-Simpson sub-intervals (even); converged to <1% vs dense ref
    _KIRCHHOFF_P = 4.0     # grading exponent (cluster nodes near the wet end b); p=4 -> 4 s^3 Jacobian,
    #                        still a cubic so Simpson integrates a constant K exactly

    def _kirchhoff_quad(self, a, b, K_fn):
        n, p = self._KIRCHHOFF_N, self._KIRCHHOFF_P
        span = b - a
        total = 0.0
        for j in range(n + 1):
            s = j / n
            psi_j = b - span * s**p
            w = 1.0 if j in (0, n) else (4.0 if j % 2 == 1 else 2.0)
            total = total + w * K_fn(psi_j) * (p * s ** (p - 1.0))
        return span * total / (3.0 * n)

    def kirchhoff(self, a, b):
        """Matric flux potential difference int_a^b K(psi) dpsi (m^2/day); numpy (scalar a, b)."""
        return self._kirchhoff_quad(a, b, self.K)

    def kirchhoff_ufl(self, a, b):
        """UFL matric flux potential difference int_a^b K(psi) dpsi, for the coupled residual."""
        return self._kirchhoff_quad(a, b, self.K_ufl)
