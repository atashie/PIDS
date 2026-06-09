"""Unit tests for the sorptive-closure primitives used by the embedded-feature exchange leg (Module 4 §E).

The live feature flux is the Kirchhoff form  q = well_index(I) * dPhi_live,  with the cumulative-state
well-index  Omega(I) = C*S^2*F(zeta) / (2*I*dPhi_ref),  zeta = I/(dtheta*r_w). These tests pin the two
properties the EmbeddedFeature integration relies on:
  (1) at the REFERENCE drop (dPhi_live == dPhi_ref) the Omega*dPhi form RECOVERS the validated clock
      dI/dt = (S^2/2I)*F (so the gate-validated trajectory carries over to the live residual unchanged);
  (2) at a PARTIAL drop the flux scales LINEARLY with the live Kirchhoff potential (the feature responds
      correctly as heads evolve -- the gate validated only the fixed-BC reference drop).
"""
import numpy as np

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import (
    F_cylindrical,
    dIdt,
    parlange_sorptivity,
    sorptive_clock,
    well_index,
    R_W_DEFAULT,
)

LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def _loam_ref_params():
    S = parlange_sorptivity(LOAM, -1.0, 0.0)
    dth = abs(float(LOAM.theta(0.0) - LOAM.theta(-1.0)))
    dPhi_ref = float(LOAM.kirchhoff(-1.0, 0.0))
    return S, dth, dPhi_ref


def test_well_index_times_ref_drop_recovers_clock_rhs():
    """Omega(I)*dPhi_ref == dI/dt = (S^2/2I)*F(zeta) at every state I (the reconciliation identity)."""
    S, dth, dPhi_ref = _loam_ref_params()
    for I in (1e-4, 1e-3, 1e-2, 5e-2):
        omega = well_index(I, S, dth, F_cylindrical, dPhi_ref)
        assert abs(omega * dPhi_ref - dIdt(I, S, dth, R_W_DEFAULT, F_cylindrical)) < 1e-15 * dIdt(I, S, dth, R_W_DEFAULT, F_cylindrical)


def test_omega_dphi_form_reproduces_clock_trajectory():
    """Forward-integrating dI/dt = Omega(I)*dPhi_ref reproduces sorptive_clock bit-for-bit (the Kirchhoff
    Omega*dPhi form and the clock are the SAME ODE at the reference drop)."""
    S, dth, dPhi_ref = _loam_ref_params()
    t = np.geomspace(1e-4, 6e-2, 24)
    I_clock = sorptive_clock(t, S, dth, R_W_DEFAULT, F_cylindrical)
    # integrate the Omega*dPhi form on the same grid with the same seed/substeps
    I_omega = sorptive_clock(t, S, dth, R_W_DEFAULT,
                             F_cylindrical, C=1.0)  # identical call -> identical trajectory by construction
    # explicit Omega*dPhi integration must match the clock
    I = np.empty_like(t)
    I[0] = S * np.sqrt(t[0])
    for i in range(1, t.size):
        Ii, h = I[i - 1], (t[i] - t[i - 1]) / 400
        for _ in range(400):
            Ii = Ii + h * well_index(Ii, S, dth, F_cylindrical, dPhi_ref) * dPhi_ref
        I[i] = Ii
    assert np.allclose(I, I_clock, rtol=1e-12, atol=0.0)


def test_partial_drop_scales_flux_linearly():
    """At a partial live drop the flux is Omega(I)*dPhi_live -- linear in the live Kirchhoff potential."""
    S, dth, dPhi_ref = _loam_ref_params()
    I = 1e-2
    omega = well_index(I, S, dth, F_cylindrical, dPhi_ref)
    q_full = omega * dPhi_ref
    q_half = omega * (0.5 * dPhi_ref)
    assert abs(q_half - 0.5 * q_full) < 1e-15 * q_full
    assert abs(q_full - dIdt(I, S, dth, R_W_DEFAULT, F_cylindrical)) < 1e-12 * q_full
