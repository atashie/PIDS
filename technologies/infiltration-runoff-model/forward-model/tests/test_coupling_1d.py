"""Tier-1 sanity (Module 3 coupling, 1-D): land-surface exchange ψ↔d.

Validates the monolithic two-way coupling of subsurface Richards (ψ) and a surface water store
(d) via the land-surface exchange q_ls = k_ex·(d − ψ_top) (design §D.3, Robin form B), in the
1-D limit where the surface is the top point (no lateral overland). The coupling is built test-
first per governance/claude-sanity-check-routine.md: conservation, then continuity/datum, then
partitioning/recession.

1-D is the realization-agnostic validation of the COUPLING PHYSICS (the cross-mesh submesh
machinery of realization S only bites in 2-D/3-D lateral overland).
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

# A loam (van Genuchten/Mualem); Ks in m/day.
SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def test_closed_column_mass_balance_under_rain_1d():
    """Closed 1-D column under rain: Δ(soil storage)+Δ(surface store) = ∫rain, to the 1e-6 gate.

    The decisive coupling invariant. The land-surface exchange is sign-paired (water leaving the
    surface store enters the soil top), so with a no-flux base and no outflow the total water
    (∫θ over the column + the surface depth) changes by EXACTLY the cumulative rainfall. This
    forces the full coupled solve (Richards block + surface store + ψ↔d exchange, one monolithic
    Newton per step) to exist and close the books regardless of how the rain partitions between
    infiltration and surface storage.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 1.0])  # 1 m column; z = x[0] in [0,1]
    prob = CoupledProblem(msh, SOIL)
    # uniformly unsaturated soil, dry surface; closed base (default no-flux), no outflow.
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    w0 = prob.total_water()

    rate = 0.1  # m/day rainfall onto the surface store
    rain = prob.add_rain(rate)
    t_end = 0.5
    prob.advance(t_end=t_end, dt=1e-3, dt_max=0.05)

    cum_rain = rate * t_end  # closed column: all rain stays in the system (soil + surface)
    assert abs((prob.total_water() - w0) - cum_rain) / cum_rain < 1e-6
    # SUPPLY-LIMITED: rain (0.1) < Ks (0.25) on unsaturated soil -> all rain infiltrates, NO ponding
    # (the NCP must keep d ~ 0, NOT let the dry soil over-draw an empty store into negative depth).
    assert prob.surface_depth() < 1e-3, f"spurious ponding: d={prob.surface_depth():.3e} (should be ~0)"
    assert prob.surface_water() >= -1e-12  # plausibility: depth non-negative
    assert np.all(np.isfinite(prob.psi.x.array))
