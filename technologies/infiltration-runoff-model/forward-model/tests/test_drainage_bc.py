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
    # full balance Δtotal = cum_rain − cum_outflow − cum_drainage + clip_mass_adjust (Codex 2026-06-07:
    # the degenerate-limiter branch's clip_mass_adjust is part of the total change). Here no rain/outlet/
    # clipping, so it reduces to Δtotal = −cum_drainage.
    bal = (prob.total_water() - w0) - (-prob.cum_drainage + prob.clip_mass_adjust)
    assert abs(bal) / abs(prob.cum_drainage) < 1e-6
    assert np.all(np.isfinite(prob.psi.x.array))


def test_drainage_relative_permeability_weighting():
    """Unsaturated drainage is K(psi)-weighted (relative permeability kr=K(psi)/Ks), so it SELF-LIMITS
    as the boundary dries: q_n = C*kr(psi)*(H - H_ext). A constant-C GHB would over-drain unsaturated
    soil (driving the boundary head down to deliver an unphysical flux). Codex review 2026-06-07.
    """
    L, C, H_ext, psi0 = 1.0, 0.5, -1.0, -0.5
    msh = dmesh.create_interval(MPI.COMM_WORLD, 10, [0.0, L])
    prob = RichardsProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0])   # uniform UNSATURATED
    prob.add_drainage_bc(lambda x: np.isclose(x[0], 0.0), conductance=C, external_head=H_ext)
    kr = SOIL.K(psi0) / SOIL.Ks
    assert kr < 0.1, f"test soil not unsaturated enough (kr={kr:.3f}) to exercise the weighting"
    q_expected = C * kr * (psi0 + 0.0 - H_ext)               # z_base = 0; the K-weighted GHB flux
    assert prob.drainage_rate() == pytest.approx(q_expected, rel=1e-6), \
        f"drainage {prob.drainage_rate():.5e} vs K-weighted {q_expected:.5e} (constant-C would be {C*(psi0-H_ext):.5e})"


def test_drainage_bc_rejects_overlapping_facets():
    """A drainage locator overlapping the top surface (or a prior drain) would double-tag a facet in
    the shared F_psi meshtags (ambiguous) -- reject it. Codex review 2026-06-07."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [4, 4])
    prob = CoupledProblem(msh, SOIL)
    with pytest.raises(ValueError):   # x[1]=1.0 IS the top surface (λ-coupling facets)
        prob.add_drainage_bc(lambda x: np.isclose(x[1], 1.0), conductance=0.5, external_head=0.0)


def test_drainage_conductance_accepts_constant_and_is_rampable():
    """CoupledProblem.add_drainage_bc accepts a fem.Constant conductance (time-varying drain), and
    drainage_rate() responds to .value updates: C.value=0 -> no drainage; raise it -> drains. Lets a
    strong drain ease on gently to avoid a cold-start shock on near-saturated stiff soil (used by the
    saturating-storm sweep). Added 2026-06-07."""
    from dolfinx import fem
    from petsc4py import PETSc
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [4, 6])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: 0.2 + 0.0 * x[0], d_value=0.0)   # saturated (psi>0 -> kr=1)
    Cc = fem.Constant(msh, PETSc.ScalarType(0.0))
    ret = prob.add_drainage_bc(lambda x: np.isclose(x[1], 0.0), conductance=Cc, external_head=-1.0)
    assert ret is Cc                              # returns the Constant handle for ramping
    assert abs(prob.drainage_rate()) <= 1e-14     # C=0 -> no drainage (drain inert)
    Cc.value = 0.5
    assert prob.drainage_rate() > 0.0             # raised -> drains OUT (H = 0.2 + z > H_ext = -1)


def test_drainage_bc_bidirectional_inflow():
    """The GHB is BIDIRECTIONAL: an external head ABOVE the soil head draws water IN (drainage_rate < 0,
    an injector) and total water INCREASES, conserving as Δtotal = −cum_drainage (cum_drainage < 0).
    Pins the inflow branch -- the lateral-drainage sweep only exercised outflow (Codex 2026-06-07)."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [4, 6])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.0)   # unsaturated: H = -0.5 + z
    # GHB on the BASE only (z=0, where H=-0.5); H_ext=0.2 > -0.5 -> gentle INFLOW (injector).
    prob.add_drainage_bc(lambda x: np.isclose(x[1], 0.0), conductance=0.3, external_head=0.2)
    assert prob.drainage_rate() < 0.0     # net INWARD (the "drain" is injecting)
    w0 = prob.total_water()
    prob.advance(t_end=0.03, dt=1e-3, dt_max=3e-3)
    assert prob.total_water() > w0        # water genuinely entered the domain
    assert prob.cum_drainage < 0.0        # cumulative drainage is negative (inflow)
    bal = (prob.total_water() - w0) - (-prob.cum_drainage + prob.clip_mass_adjust)
    assert abs(bal) / abs(prob.cum_drainage) < 1e-5   # conserves on the inflow branch too


def test_drainage_bc_rejects_negative_conductance():
    msh = dmesh.create_interval(MPI.COMM_WORLD, 8, [0.0, 1.0])
    prob = RichardsProblem(msh, SOIL)
    with pytest.raises(ValueError):
        prob.add_drainage_bc(lambda x: np.isclose(x[0], 0.0), conductance=-1.0, external_head=0.0)
