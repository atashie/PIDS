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

    @property
    def m(self) -> float:
        return 1.0 - 1.0 / self.n

    def effective_saturation(self, psi):
        """Effective saturation Se in [0, 1] (van Genuchten, 1980). Scalar or array.

        ``max(-psi, 0)`` selects the saturated branch (Se=1 at psi>=0) smoothly.
        """
        u = self.alpha * np.maximum(-psi, 0.0)
        return (1.0 + u**self.n) ** (-self.m)

    def theta(self, psi: float) -> float:
        """Volumetric water content."""
        return self.theta_r + self.effective_saturation(psi) * (self.theta_s - self.theta_r)

    def K(self, psi: float) -> float:
        """Mualem (1976) hydraulic conductivity (m/day); K(psi >= 0) == Ks."""
        se = self.effective_saturation(psi)
        return self.Ks * se**self.L * (1.0 - (1.0 - se ** (1.0 / self.m)) ** self.m) ** 2

    def capacity(self, psi):
        """Specific moisture capacity C = dtheta/dpsi (1/m); 0 in the saturated zone.

        Scalar or array. At psi >= 0, u = 0 so u**(n-1) = 0 and C = 0 naturally.
        """
        u = self.alpha * np.maximum(-psi, 0.0)
        return (
            (self.theta_s - self.theta_r)
            * self.alpha
            * self.m
            * self.n
            * u ** (self.n - 1.0)
            * (1.0 + u**self.n) ** (-self.m - 1.0)
        )

    # --- UFL-symbolic forms (for the Richards residual + auto-diff Jacobian) ---
    # These mirror the float closures above but accept a UFL expression ``psi``.
    # ``max_value(-psi, 0)`` gives the saturated branch (Se=1 at psi>=0) smoothly,
    # avoiding a kink-prone ufl.conditional whose unselected branch can still
    # contaminate the auto-differentiated Jacobian.

    def se_ufl(self, psi):
        import ufl

        u = self.alpha * ufl.max_value(-psi, 0.0)
        return (1.0 + u**self.n) ** (-self.m)

    def theta_ufl(self, psi):
        return self.theta_r + self.se_ufl(psi) * (self.theta_s - self.theta_r)

    def K_ufl(self, psi):
        se = self.se_ufl(psi)
        return self.Ks * se**self.L * (1.0 - (1.0 - se ** (1.0 / self.m)) ** self.m) ** 2
