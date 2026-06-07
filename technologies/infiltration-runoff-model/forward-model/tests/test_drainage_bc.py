"""Tier-1 sanity: subsurface Darcy/head drainage BC (general-head / Cauchy / MODFLOW GHB).

q_n = C*(H - H_ext), H = psi + z (z = elevation, last axis); outward Darcy flux on a domain boundary.
Lets the soil matrix exchange water with an external reservoir (lateral groundwater outflow, soil-
moisture drainage, deep percolation) -- distinct from the surface Manning outlet. Spec:
docs/plans/2026-06-07-subsurface-drainage-bc-spec.md.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import RichardsProblem
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def test_drainage_analytical_steady_darcy():
    """Saturated column, Dirichlet head at top + GHB at the base -> steady uniform Darcy flux equal to
    BOTH Ks*(H_top-H_base)/L and C*(H_base-H_ext). Pins the GHB physics + the H = psi + z convention.

    Numbers: Ks=0.25, L=1, psi_top=0.1 (H_top=1.1), C=0.5, H_ext=0.2 -> H_base=0.5, q=0.15, psi_base=0.5.
    """
    L, Ks, C, H_ext, psi_top = 1.0, SOIL.Ks, 0.5, 0.2, 0.1
    H_top = psi_top + L
    a = Ks / L
    H_base = (a * H_top + C * H_ext) / (a + C)        # = 0.5
    q_ana = C * (H_base - H_ext)                        # = 0.15
    msh = dmesh.create_interval(MPI.COMM_WORLD, 40, [0.0, L])
    prob = RichardsProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: 0.3 + 0.0 * x[0])     # saturated start (psi>0 -> K=Ks)
    prob.add_dirichlet(lambda x: np.isclose(x[0], L), psi_top)  # head datum at top
    prob.add_drainage_bc(lambda x: np.isclose(x[0], 0.0), conductance=C, external_head=H_ext)
    prob.advance(t_end=5.0, dt=0.5, dt_max=2.0)                # saturated -> elliptic -> steady fast

    assert prob.drainage_rate() == pytest.approx(q_ana, rel=2e-2), \
        f"drainage {prob.drainage_rate():.4f} vs analytical {q_ana:.4f}"
    zc = prob.V.tabulate_dof_coordinates()[:, 0]
    psi_base = float(prob.psi.x.array[np.argmin(zc)])
    assert psi_base == pytest.approx(H_base, rel=2e-2), f"psi_base {psi_base:.4f} vs {H_base:.4f}"


def test_drainage_noflow_and_dirichlet_limits():
    """C=0 -> NO drainage (rate exactly 0). Large C -> the boundary head -> H_ext (Dirichlet limit)."""
    L, H_ext = 1.0, 0.2
    msh = dmesh.create_interval(MPI.COMM_WORLD, 20, [0.0, L])
    p0 = RichardsProblem(msh, SOIL)
    p0.set_initial_condition(lambda x: 0.3 + 0.0 * x[0])
    p0.add_dirichlet(lambda x: np.isclose(x[0], L), 0.1)
    p0.add_drainage_bc(lambda x: np.isclose(x[0], 0.0), conductance=0.0, external_head=H_ext)
    p0.advance(t_end=2.0, dt=0.5, dt_max=2.0)
    assert abs(p0.drainage_rate()) <= 1e-12   # C=0 -> no-flow

    msh2 = dmesh.create_interval(MPI.COMM_WORLD, 20, [0.0, L])
    pb = RichardsProblem(msh2, SOIL)
    pb.set_initial_condition(lambda x: 0.3 + 0.0 * x[0])
    pb.add_dirichlet(lambda x: np.isclose(x[0], L), 0.1)
    pb.add_drainage_bc(lambda x: np.isclose(x[0], 0.0), conductance=1e4, external_head=H_ext)
    pb.advance(t_end=2.0, dt=0.5, dt_max=2.0)
    zc = pb.V.tabulate_dof_coordinates()[:, 0]
    psi_base = float(pb.psi.x.array[np.argmin(zc)])
    assert psi_base == pytest.approx(H_ext, abs=3e-3)   # large C -> H_base -> H_ext (z_base=0)


def test_drainage_conservation_coupled():
    """A CoupledProblem column draining through a base GHB (no rain, no surface outlet): total water
    decreases by EXACTLY cum_drainage -- Delta_total = -cum_drainage to solver precision."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [4, 8])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -0.2 + 0.0 * x[0], d_value=0.0)   # moist-ish soil
    prob.add_drainage_bc(lambda x: np.isclose(x[1], 0.0), conductance=0.5, external_head=-1.0)
    w0 = prob.total_water()
    prob.advance(t_end=0.5, dt=1e-2, dt_max=0.05)

    assert prob.cum_drainage > 0.0   # H = psi+z (~ -0.2 at base) > H_ext (-1.0) -> drains OUT
    assert abs((prob.total_water() - w0) + prob.cum_drainage) / abs(prob.cum_drainage) < 1e-6
    assert np.all(np.isfinite(prob.psi.x.array))


def test_drainage_bc_rejects_negative_conductance():
    msh = dmesh.create_interval(MPI.COMM_WORLD, 8, [0.0, 1.0])
    prob = RichardsProblem(msh, SOIL)
    with pytest.raises(ValueError):
        prob.add_drainage_bc(lambda x: np.isclose(x[0], 0.0), conductance=-1.0, external_head=0.0)
