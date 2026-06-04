"""Tier-1 sanity (subsurface Richards solver): analytical / conservation checks.

The solver is the dimension-agnostic mixed-form Richards equation (Celia 1990,
mass-conservative), backward Euler, primary variable = pressure head psi, gravity
along the LAST spatial coordinate. Per governance/claude-sanity-check-routine.md,
each behavior is pinned by a closed-form / conservation reference, test-first.

Soil: Carsel & Parrish (1988) loam (SI: length m, time day).
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import RichardsProblem

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.0104)


def test_hydrostatic_equilibrium_is_held_1d():
    """A 1-D column at hydrostatic equilibrium psi(z) = -z must stay there.

    At equilibrium total head H = psi + z is uniform, so the Darcy flux
    -K(grad psi + grad z) is identically zero. Any gravity sign error makes the
    column spuriously drain or fill, so this is the decisive gravity-term test.
    """
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 20)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)

    hydrostatic = lambda x: -x[0]  # psi = -z (z is the last/only coordinate)
    prob.set_initial_condition(hydrostatic)
    # Water table at the bottom (z=0 => psi=0); natural no-flux at the top.
    prob.add_dirichlet(lambda x: np.isclose(x[0], 0.0), 0.0)

    for _ in range(5):
        converged, iters = prob.step(dt=0.1)
        assert converged

    assert prob.max_abs_error(hydrostatic) < 1e-9


def test_closed_column_conserves_total_water_1d():
    """A closed column (no-flux both ends) conserves total water to solver precision.

    With no boundary flux and no source, integrating the mixed-form weak form with
    the constant test function gives d/dt (integral theta dx) = 0 exactly; this is
    the mass-conservation property of the mixed (Celia 1990) form. The state must
    still evolve (internal redistribution), so the check is non-trivial.
    """
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 40)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)

    # Non-equilibrium start, wetter near the top (psi: -2 at bottom -> -0.2 at top).
    # NO boundary conditions => closed system (natural zero-flux on both ends).
    prob.set_initial_condition(lambda x: -2.0 + 1.8 * x[0])
    w0 = prob.total_water()
    psi_before = prob.psi.x.array.copy()

    for _ in range(10):
        converged, _ = prob.step(dt=0.05)
        assert converged

    # Total water conserved (no flux in/out, no source).
    assert abs(prob.total_water() - w0) / w0 < 1e-9
    # ... and the state actually redistributed (non-trivial dynamics).
    assert np.max(np.abs(prob.psi.x.array - psi_before)) > 1e-3
    # Plausibility: theta within [theta_r, theta_s] and finite everywhere.
    th = prob.theta_array()
    assert np.all(np.isfinite(th))
    assert th.min() >= LOAM["theta_r"] - 1e-12
    assert th.max() <= LOAM["theta_s"] + 1e-12
