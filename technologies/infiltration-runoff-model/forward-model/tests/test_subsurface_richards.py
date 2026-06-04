"""Tier-1 sanity (subsurface Richards solver): analytical / conservation checks.

The solver is the dimension-agnostic mixed-form Richards equation (Celia 1990,
mass-conservative), backward Euler, primary variable = pressure head psi, gravity
along the LAST spatial coordinate. Per governance/claude-sanity-check-routine.md,
each behavior is pinned by a closed-form / conservation reference, test-first.

Soil: Carsel & Parrish (1988) loam (SI: length m, time day).
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import RichardsProblem

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)  # m/day


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
        assert 0 <= iters <= 50  # real SNES iteration count (no -1 sentinel)

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

    # Total water conserved (no flux in/out, no source) to the routine's mass-balance
    # criterion (<1e-6); the mixed form's actual closure is far tighter (~1e-10 here).
    assert abs(prob.total_water() - w0) / w0 < 1e-6
    # ... and the state actually redistributed (non-trivial dynamics).
    assert np.max(np.abs(prob.psi.x.array - psi_before)) > 1e-3
    # Plausibility: theta within [theta_r, theta_s] and finite everywhere.
    th = prob.theta_array()
    assert np.all(np.isfinite(th))
    assert th.min() >= LOAM["theta_r"] - 1e-12
    assert th.max() <= LOAM["theta_s"] + 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_closed_box_conserves_total_water_2d_3d(dim):
    """The SAME solver, unchanged, conserves + stays plausible in 2-D and 3-D.

    Validates the dimension-agnostic decision: only the gravity unit vector differs
    between 1-D / 2-D / 3-D; the residual and code path are identical.
    """
    if dim == 2:
        msh = dmesh.create_unit_square(MPI.COMM_WORLD, 12, 12)
    else:
        msh = dmesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)

    last = dim - 1  # elevation = last coordinate; wetter at the top
    prob.set_initial_condition(lambda x: -2.0 + 1.8 * x[last])
    w0 = prob.total_water()
    psi_before = prob.psi.x.array.copy()

    for _ in range(5):
        converged, _ = prob.step(dt=0.05)
        assert converged

    assert abs(prob.total_water() - w0) / w0 < 1e-6  # routine mass-balance criterion
    assert np.max(np.abs(prob.psi.x.array - psi_before)) > 1e-3
    th = prob.theta_array()
    assert np.all(np.isfinite(th))
    assert th.min() >= LOAM["theta_r"] - 1e-12
    assert th.max() <= LOAM["theta_s"] + 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_gravity_acts_along_last_coordinate(dim):
    """Pins e_g to the LAST axis: psi=-x[last] holds equilibrium, while the same
    linear profile along a NON-gravity axis (x[0]) drives flow.

    Closed-box conservation alone passes for ANY e_g (the constant test function
    annihilates the stiffness term), so this is the only check that actually pins
    the gravity direction in 2-D / 3-D.
    """
    make = dmesh.create_unit_square if dim == 2 else dmesh.create_unit_cube
    args = (12, 12) if dim == 2 else (6, 6, 6)
    soil = VanGenuchten(**LOAM)
    last = dim - 1

    # (a) hydrostatic along gravity (last axis) is held to ~machine precision.
    msh = make(MPI.COMM_WORLD, *args)
    along_g = lambda x: -x[last]
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(along_g)
    prob.add_dirichlet(lambda x: np.isclose(x[last], 0.0), 0.0)
    for _ in range(3):
        conv, _ = prob.step(dt=0.1)
        assert conv
    assert prob.max_abs_error(along_g) < 1e-9

    # (b) an UNSATURATED profile varying along x[0] (NOT the gravity axis): total
    # head -0.3-x[0]+x[last] varies, so flow IS driven. If gravity wrongly pointed
    # along x[0], H = psi + x[0] = -0.3 would be constant => no flow, failing this.
    # Kept strictly unsaturated to isolate the gravity-direction check from the
    # saturated-zone regularization (A1/Ss), which is tracked and built separately.
    msh2 = make(MPI.COMM_WORLD, *args)
    prob2 = RichardsProblem(msh2, soil)
    prob2.set_initial_condition(lambda x: -0.3 - x[0])
    prob2.add_dirichlet(lambda x: np.isclose(x[0], 0.0), -0.3)
    before = prob2.psi.x.array.copy()
    for _ in range(3):
        conv, _ = prob2.step(dt=0.5)
        assert conv
    assert np.max(np.abs(prob2.psi.x.array - before)) > 1e-4


def test_solver_is_deterministic_1d():
    """Fixed inputs => bit-identical output (serial LU/preonly is deterministic)."""
    soil = VanGenuchten(**LOAM)

    def run():
        msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 30)
        p = RichardsProblem(msh, soil)
        p.set_initial_condition(lambda x: -2.0 + 1.8 * x[0])
        p.add_dirichlet(lambda x: np.isclose(x[0], 0.0), 0.0)
        for _ in range(5):
            conv, _ = p.step(dt=0.05)
            assert conv
        return p.psi.x.array.copy(), p.total_water()

    psi_a, w_a = run()
    psi_b, w_b = run()
    assert np.array_equal(psi_a, psi_b)
    assert w_a == w_b


def test_infiltration_to_saturation_converges():
    """Wetting a column into saturation/ponding now converges.

    Before the Vogel air-entry fix, the Mualem-K Jacobian singularity at Se->1
    produced 0*inf=NaN whenever a node reached saturation and stalled Newton. A
    ponded (psi>0) top drives a wetting front down to full saturation; the solve
    must converge and stay physically plausible.
    """
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 30)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0])  # uniformly unsaturated
    # Ponded top (x=1, psi>0 => saturated) drives infiltration; a fixed unsaturated
    # head at the bottom anchors the column.
    prob.add_dirichlet(lambda x: np.isclose(x[0], 1.0), 0.05)
    prob.add_dirichlet(lambda x: np.isclose(x[0], 0.0), -1.0)
    # Adaptive stepping auto-cuts the stiff initial step (fixed dt=0.05 diverges here).
    nsteps = prob.advance(t_end=0.4, dt=0.05)
    assert nsteps > 0

    th = prob.theta_array()
    assert np.all(np.isfinite(th))
    assert th.min() >= LOAM["theta_r"] - 1e-12
    assert th.max() <= LOAM["theta_s"] + 1e-12
    # the ponded top is fully saturated, and infiltration wetted the column overall.
    assert th.max() == pytest.approx(LOAM["theta_s"], rel=1e-6)
    assert th.mean() > 0.25  # wetter than the initial uniform theta(-1) ~ 0.243
