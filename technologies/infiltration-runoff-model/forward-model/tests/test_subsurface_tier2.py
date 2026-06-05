"""Tier-2 sanity (subsurface): synthetic-forcing runs (typical + end-member extremes).

Per governance/claude-sanity-check-routine.md the module is driven by synthetic
forcing and must stay mass-conservative + physically plausible while showing the
expected qualitative behaviour. Rainfall / evaporation enter as a Neumann flux at
the top; the bottom is closed (natural zero-flux) so storage tracks net input.

Soil: Carsel & Parrish (1988) loam, SI (length m, time DAYS). Ks = 0.2496 m/day.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import RichardsProblem

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)


def test_open_system_mass_balance_constant_rain():
    """Constant rainfall into a closed-bottom column: storage increase == cumulative
    input (the routine's open-system |dStorage - net_flux|/scale < 1e-6 gate, and a
    sign/magnitude check on the flux BC)."""
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 40)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: -3.0 + 0.0 * x[0])  # dry column

    q_rain = 0.05  # m/day, well below Ks so it all infiltrates (no ponding/runoff)
    prob.add_flux_bc(lambda x: np.isclose(x[0], 1.0), q_rain)  # rain at the top
    # bottom: no flux (natural) => closed; all input is stored.

    s0 = prob.total_water()
    T = 2.0
    prob.advance(t_end=T, dt=0.1)
    dS = prob.total_water() - s0

    assert dS == pytest.approx(q_rain * T, rel=1e-4)  # = 0.10 m of water
    th = prob.theta_array()
    assert np.all(np.isfinite(th))
    assert th.min() >= LOAM["theta_r"] - 1e-12
    assert th.max() <= LOAM["theta_s"] + 1e-12


def test_intense_storm_saturates_top_and_conserves_mass():
    """100-yr-style intense storm: a rainfall flux ABOVE the infiltration capacity wets
    a column to saturation/ponding at the top, exercises adaptive stepping, conserves
    mass, and stays plausible (an end-member extreme per the routine)."""
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 40)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0])  # moderately dry
    q_storm = 1.0  # m/day ~ 4x Ks: exceeds infiltration capacity -> top saturates/ponds
    prob.add_flux_bc(lambda x: np.isclose(x[0], 1.0), q_storm)

    s0 = prob.total_water()
    th0_mean = soil.theta(-2.0)
    T = 0.2
    prob.advance(t_end=T, dt=0.02)
    th = prob.theta_array()

    assert (prob.total_water() - s0) == pytest.approx(q_storm * T, rel=1e-4)  # mass conserved
    assert np.all(np.isfinite(th))
    assert th.min() >= LOAM["theta_r"] - 1e-12
    assert th.max() <= LOAM["theta_s"] + 1e-12
    assert th.max() == pytest.approx(LOAM["theta_s"], rel=1e-3)  # top saturated
    assert th.mean() > th0_mean  # wetting front advanced into the column


def test_drought_dries_column_monotonically():
    """100-yr-drought end-member: a steady (sustainable) evaporative outflux with no
    rain dries the column monotonically and conserves mass (storage loss == cumulative
    evaporation). NOTE: a fixed evaporative flux is only valid while it stays below the
    soil's supply capacity; once the surface dries enough that K can't deliver the
    demand, evaporation becomes soil-limited -- a supply-limited (atmospheric/Robin) BC,
    deferred to the forcing/vegetation module. Here the flux is gentle and sustainable.
    """
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 40)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: -0.5 + 0.0 * x[0])  # moderately wet
    q_evap = -0.002  # m/day evaporative OUTFLUX (gentle, stays below supply capacity)
    prob.add_flux_bc(lambda x: np.isclose(x[0], 1.0), q_evap)

    mean0 = soil.theta(-0.5)
    s0 = prob.total_water()
    T = 2.0
    prob.advance(t_end=T / 2, dt=0.1)
    s_half = prob.total_water()  # advances the state another T/2
    prob.advance(t_end=T / 2, dt=0.1)
    s_end = prob.total_water()

    assert s0 > s_half > s_end  # monotone drying
    assert (s_end - s0) == pytest.approx(q_evap * T, rel=1e-3)  # = -0.004 m
    th = prob.theta_array()
    assert np.all(np.isfinite(th))
    assert th.min() >= LOAM["theta_r"] - 1e-12
    assert th.mean() < mean0  # column lost water overall


def test_intense_rain_on_wet_soil_ponds_and_conserves():
    """Saturation-excess regime: extreme intense rain on already-wet soil saturates the
    column; the excess PONDS (surface pressure head rises) rather than stalling the solve.
    Vertical ponding store only (lateral routing is the overland module). Mass balance:
    rainfall = soil-storage change + ponded-depth change."""
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 40)
    soil = VanGenuchten(**LOAM)
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: -0.3 + 0.0 * x[0])  # wet antecedent
    q = 2.0  # m/day, extreme intense (~8x Ks)
    prob.add_ponding_bc(lambda x: np.isclose(x[0], 1.0), q)

    s0 = prob.total_water() + prob.ponded_depth()  # soil + pond
    T = 0.15
    prob.advance(t_end=T, dt=0.005)

    th = prob.theta_array()
    pond = prob.ponded_depth()
    assert np.all(np.isfinite(th))
    assert th.min() >= LOAM["theta_r"] - 1e-12
    assert th.max() <= LOAM["theta_s"] + 1e-12
    assert th.max() == pytest.approx(LOAM["theta_s"], rel=1e-3)  # column saturated
    assert pond > 0.0  # water is now ponding (surface head increased)
    # mass conserved across soil + pond.
    assert (prob.total_water() + pond - s0) == pytest.approx(q * T, rel=1e-3)
