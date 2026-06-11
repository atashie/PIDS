"""A-priori sorptive-exchange CLOSURES for the embedded PIDS feature wall (Module 4 §E, Phase 3).

The wall exchange flux per unit wall area is a sorptivity CLOCK in the cumulative uptake state ``I`` (a
per-face state, NOT explicit time):

    q = dI/dt = C * (S^2 / (2 I)) * F(zeta),     zeta = I / (dtheta * r_w)

with ``dtheta = |theta(0) - theta(psi_i)|`` and ``r_w`` the feature radius. ``F`` is the geometry/direction
SHAPE factor and ``S`` the (de)sorptivity coefficient:

* **DISPERSE** (feature wetter than soil; capillary sorption into the matrix) --
  ``F = F_cylindrical(zeta) = 2*zeta / ln(1 + 2*zeta)`` and ``S = parlange_sorptivity`` (a-priori).
  ``F_cylindrical`` is DERIVED, not fitted: cylindrical Green-Ampt mass balance
  ``I = dtheta*(R^2 - r_w^2)/(2*r_w)`` (=> ``R^2/r_w^2 = 1 + 2*zeta``) closed with the STEADY radial
  Kirchhoff (matric-flux-potential) resistance ``q = dPhi/(r_w*ln(R/r_w))`` gives exactly
  ``dI/dt = (S_WS^2/2I) * 2*zeta/ln(1+2*zeta)``; ``F -> 1`` as ``zeta -> 0`` recovers the planar limb
  ``I = S*sqrt(t)``. This leg is PARAMETER-FREE (``C = 1``) and reproduces the resolved tunnel + annulus
  references to <=5% full-curve (the Phase-3 C-004 gate, ``tests/test_sorptive_closure_gate.py``).
  A small per-geometry ``C`` (``C_tunnel ~ 1.008``, ``C_annulus ~ 1.043``: the horizontal-feature gravity
  asymmetry) tightens both to <=2% but is optional polish, not needed to pass.

* **DRAIN** (soil wetter than feature; matrix desorption of the drainable porosity between saturation and
  field capacity -- a CORE PIDS use case) -- ``F = F_throttle(zeta) = exp(-(zeta/z0)^k)`` and ``S`` the
  desorptivity. The reference is SUB-sqrt-t: as the near-wall soil desaturates toward ``psi_wall`` its
  conductivity collapses, presenting a growing series resistance that throttles the flux below the planar
  ``sqrt(t)`` law. This leg is SEMI-EMPIRICAL: ``(z0, k)`` are fitted globals (``throttle_params``) and no
  robust a-priori desorptivity exists -- the saturated ``psi = 0`` start is a ``D = K/C -> inf`` similarity
  singularity (Bruce-Klute NaN-fails; the Parlange-desorption integral over-predicts ~2x). Closing the
  drain a-priori gap is documented future work (see the pids-drain-usecase memory).

The disperse and drain shapes are of OPPOSITE curvature (super- vs sub-sqrt-t) and CANNOT be unified into
one direction-independent factor -- this is the genuine sorptive sign-ASYMMETRY the resolved references
showed (a single direction-independent sigma is FALSIFIED; the leg needs a direction switch).

Pure numpy (a ``VanGenuchten``-like soil exposes ``.theta``/``.K``/``.kirchhoff``). The same RHS (`dIdt`)
feeds both the offline gate (forward integration on a reference t-grid) and the live embedded-feature UFL
residual (Phase-4 integration). Design: docs/plans/2026-06-08-module4-features-plan.md (Phase 3).
"""
from __future__ import annotations

import numpy as np

R_W_DEFAULT = 0.05  # default feature (wall) radius [m] -- matches the Phase-1 references

# The MEASURED discrete well index of the P1 ridge source (Peaceman-for-FEM, Phase-4 Task 5,
# 2026-06-10, scratch/m4_phase4_well_index.py + tests/test_well_index_p1.py): a vertex-line source on
# the structured create_box TET lattice (and identically on the create_rectangle TRIANGLE lattice --
# the x-invariant restriction of the 3-D operator IS the 2-D one) produces the analytic log field
# beyond ~2h (slope dev <1.1% at n>=12) with well-block value u_h(Gamma) = -(1/2pi)*ln(r_0),
# r_0 = R0_OVER_H_P1 * h, h-independent to 0.05% across n=8..64 (FD-Peaceman analog: 0.208).
# The wall->cell exchange bridges through WI = 2*pi/ln(r_0(h)/r_w); note r_0 < r_w throughout the
# Phase-4 harness regime (negative-log bridge: the discrete ridge value legitimately OVERSHOOTS the
# wall potential; total bridge+host resistance is exactly the analytic ln(R/r_w) by construction).
R0_OVER_H_P1 = 0.1986

# Semi-empirical drain-throttle constants (joint tunnel+annulus optimum, Phase-3 design wf 2026-06-09):
# z0 = _Z0_A + _Z0_S*dtheta, k = _THROTTLE_K. NOT a-priori (see module docstring).
_Z0_A = 2.41
_Z0_S = -5.35
_THROTTLE_K = 7.46
_Z0_FLOOR = 0.1  # positive floor: z0=2.41-5.35*dtheta goes <=0 for dtheta>=0.45 (coarse soil, dry antecedent),
#                  which would make exp(-(zeta/z0)^k) NaN; floor it (the throttle just saturates there).


def parlange_sorptivity(soil, psi_i, psi_w=0.0, n=20001):
    """A-priori Parlange (1975) sorptivity ``S`` [m/day^0.5] -- the disperse early-time coefficient.

    ``S^2 = int_{psi_i}^{psi_w} (theta0 + theta(psi) - 2*theta_i) * K(psi) dpsi``, with ``theta0 =
    theta(psi_w)`` (the wet/wall end) and ``theta_i = theta(psi_i)`` (the dry/initial soil). Integrated in
    PSI-space (NOT theta-space): the substitution ``D dtheta = K dpsi`` removes the ``D = K/C -> inf``
    saturated-end singularity that makes a theta-space quadrature blow up. ``psi_i`` is the ACTUAL field
    initial head (the calibration is conditioned on it); the non-hysteretic van Genuchten model is assumed.
    """
    psi = np.linspace(psi_i, psi_w, n)
    th = soil.theta(psi)
    th_i, th0 = float(soil.theta(psi_i)), float(soil.theta(psi_w))
    integrand = (th0 + th - 2.0 * th_i) * soil.K(psi)
    return float(np.sqrt(max(float(np.trapezoid(integrand, psi)), 0.0)))


def F_cylindrical(zeta):
    """Disperse cylindrical Green-Ampt shape ``F = 2*zeta / ln(1 + 2*zeta)`` (-> 1 as zeta -> 0). Scalar
    or array. DERIVED (see module docstring); the best a-priori curvature factor (beats free polynomials).
    """
    z = np.asarray(zeta, dtype=float)
    zc = np.where(z > 1e-12, z, 1.0)                       # guard the unused 0/0 branch (no warning)
    val = np.where(z > 1e-12, 2.0 * zc / np.log1p(2.0 * zc), 1.0)
    return float(val) if np.ndim(zeta) == 0 else val


def F_throttle(zeta, z0, k):
    """Drain conductivity-throttle shape ``F = exp(-(zeta/z0)^k)`` (-> 1 as zeta -> 0, sub-sqrt-t for
    zeta>0). Semi-empirical ``(z0, k)`` from :func:`throttle_params`. Scalar or array. ``zeta`` is clamped
    to >=0 (a transient forward-Euler overshoot to zeta<0 would give ``(neg)^k`` = NaN for non-integer k;
    the zeta=0 limit F=1 is the safe value); ``z0`` must be > 0 (guaranteed by :func:`throttle_params`)."""
    z = np.maximum(np.asarray(zeta, dtype=float), 0.0)
    return np.exp(-((z / z0) ** k))


def throttle_params(dtheta):
    """Semi-empirical drain-throttle constants ``(z0, k)`` for a soil with storable contrast ``dtheta``.
    ``z0 = max(2.41 - 5.35*dtheta, 0.1)``, ``k = 7.46`` (fitted; NOT a-priori -- see module docstring). The
    positive floor keeps ``F_throttle`` finite for high-contrast soils (``dtheta >= 0.45`` would make the raw
    ``z0`` non-positive)."""
    return max(_Z0_A + _Z0_S * float(dtheta), _Z0_FLOOR), _THROTTLE_K


def des_sorp_ratio(dtheta):
    """Semi-empirical desorptivity/sorptivity ratio ``S_des/S_sorp`` from the storable contrast ``dtheta``,
    so the drain leg has a soil-derivable (if not a-priori) desorptivity for production use. Linear fit to
    the Phase-1 references (SAND dtheta=0.381 -> 0.33; LOAM 0.187 -> 0.46; SILT 0.106 -> 0.56; CLAY 0.014 ->
    0.64): ``ratio = clip(0.66 - 0.86*dtheta, 0.2, 0.9)``. NOT a-priori (no robust closed-form desorptivity;
    the saturated psi=0 start is a D=K/C->inf singularity) -- a documented v1 default; pass an explicit value
    to override (e.g. a measured desorptivity / sorptivity)."""
    return float(np.clip(0.66 - 0.86 * float(dtheta), 0.2, 0.9))


def dIdt(I, S, dtheta, r_w, F, C=1.0):
    """The clock RHS ``C*(S^2/2I)*F(zeta)`` at cumulative state ``I`` (the live per-face exchange flux)."""
    return C * (S * S) / (2.0 * I) * F(I / (dtheta * r_w))


def well_index(I, S, dtheta, F, dPhi_ref, r_w=R_W_DEFAULT, C=1.0):
    """The cumulative-state geometric WELL-INDEX ``Omega(I) = C*S^2*F(zeta) / (2*I*dPhi_ref)`` [1/m], the
    soil/geometry factor of the Kirchhoff exchange flux ``q = Omega(I) * dPhi_live`` (the live driving
    potential ``dPhi_live = Phi(H_f) - Phi(psi_cell)``). ``dPhi_ref`` is the REFERENCE Kirchhoff drop the
    (de)sorptivity ``S`` was evaluated over; at ``dPhi_live == dPhi_ref`` the flux recovers the validated
    clock ``dI/dt = (S^2/2I)*F`` exactly, and it scales linearly with the live potential otherwise. This
    is the form the EmbeddedFeature UFL residual uses (``Omega`` a lagged-state coefficient, ``dPhi_live``
    the differentiable ``kirchhoff_ufl`` difference)."""
    return C * (S * S) * F(I / (dtheta * r_w)) / (2.0 * I * dPhi_ref)


def sorptive_clock(t, S, dtheta, r_w, F, C=1.0, nsub=400):
    """Forward-integrate the clock on the time grid ``t``, seeded with the planar early limb
    ``I[0] = S*sqrt(t[0])`` (avoids the 1/I singularity at I=0). ``nsub`` sub-steps per reported interval
    (substep-converged: 400 == RK45 to <0.01%). Returns ``I`` at each node of ``t``. This is the offline
    gate integrator; the live feature uses :func:`dIdt` inside the FEM time step."""
    t = np.asarray(t, dtype=float)
    if t.size and t[0] <= 0.0:
        raise ValueError("sorptive_clock needs t[0] > 0 (seed I[0]=S*sqrt(t[0]); t[0]=0 -> I=0 -> 1/I blow-up).")
    I = np.empty_like(t)
    I[0] = S * np.sqrt(t[0])
    for i in range(1, t.size):
        Ii, h = I[i - 1], (t[i] - t[i - 1]) / nsub
        for _ in range(nsub):
            Ii = Ii + h * dIdt(max(Ii, 1e-300), S, dtheta, r_w, F, C)   # clamp guards a transient I->0
        I[i] = Ii
    return I


def rel_l2(model, ref):
    """Relative L2 error ``sqrt(sum((model-ref)^2) / sum(ref^2))`` over the curve nodes (the gate metric)."""
    model, ref = np.asarray(model, dtype=float), np.asarray(ref, dtype=float)
    return float(np.sqrt(np.sum((model - ref) ** 2) / np.sum(ref ** 2)))
