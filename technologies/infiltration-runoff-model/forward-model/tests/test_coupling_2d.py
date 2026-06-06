"""Tier-1 sanity (Module 3 coupling, 2-D): the co-located realization with LATERAL overland.

2-D extends the 1-D coupling (same supply-limited NCP land-surface exchange) by adding lateral
surface routing: the Manning diffusion-wave as a TANGENTIAL-gradient surface PDE on the host top
facets (`ds_top`). Realization A (co-located d,λ on the host; the design-intended submesh
realization S is deferred pending an upstream FFCX fix -- see
docs/plans/2026-06-05-module3-realization-ffcx-bug.md). The exchange physics is dimension-agnostic;
the only new behavior here is lateral conveyance of ponded water downslope.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def _top_length(prob):
    import ufl
    from dolfinx import fem
    one = fem.Constant(prob.mesh, 1.0)
    return prob.mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * prob._ds_top)), op=MPI.SUM)


def _surface_xd(prob):
    """(x, d) at the top-surface dofs, sorted by x -- to inspect the lateral ponding profile."""
    dofs = prob._top_dofs(prob.Vd)
    xc = prob.Vd.tabulate_dof_coordinates()[dofs, 0]
    dv = prob.d.x.array[dofs]
    order = np.argsort(xc)
    return xc[order], dv[order]


def test_2d_flat_closed_conservation():
    """2-D flat closed domain under rain: total water grows by exactly the cumulative rainfall.

    Verifies the co-located NCP exchange is dimension-agnostic -- the same monolithic [ψ,d,λ] solve,
    now on a 2-D host. Closed (no-flux base, no outflow), so Δtotal = ∫rain over the top edge to the
    1e-6 gate, regardless of how the rain partitions between infiltration and ponding (the partition
    depends on the k_ex film ~ ℓ_c = top-cell half-height, so it is NOT pinned here -- conservation
    is the dimension-agnostic invariant). Guards the 2-D ℓ_c computation (the np.unique-on-noisy-z
    bug that drove k_ex → ∞ and broke conservation).
    """
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 2.0]], [12, 16])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    w0 = prob.total_water()
    rate = 0.1
    prob.add_rain(rate)
    prob.advance(t_end=0.5, dt=1e-3, dt_max=0.05)

    cum_rain = rate * _top_length(prob) * 0.5
    assert abs((prob.total_water() - w0) - cum_rain) / cum_rain < 1e-6   # closed: conserves
    assert prob.total_water() - w0 > 0.5 * cum_rain    # rain genuinely entered the system
    assert prob.surface_depth() >= -1e-12              # plausibility: non-negative depth
    assert np.all(np.isfinite(prob.psi.x.array))


def test_2d_lateral_redistribution_downslope():
    """Heavy rain on a SLOPED 2-D hillslope ponds, then the Manning overland term ROUTES the ponded
    water downslope -- it accumulates at the downhill end. Closed domain conserves.

    Drives the new 2-D machinery: set_topography(z_b) + the tangential-gradient Manning overland flux
    on ds_top. Without lateral routing the pond would be uniform along the top; with it, the downhill
    half holds more water. (RED until the overland term + set_topography exist.)
    """
    L = 5.0
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, 1.0]], [10, 5])
    prob = CoupledProblem(msh, SOIL, n_man=0.05)
    prob.set_initial_condition(lambda x: -0.8 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: 0.05 * (L - x[0]))  # 5% slope down toward x=L
    rate = 0.6  # > Ks -> infiltration-excess ponding
    prob.add_rain(rate)
    w0 = prob.total_water()

    # the Manning diffusion-wave is stiff -> small adaptive steps (overland is fast); short + coarse
    # to keep the test quick while the downslope routing signal is already strong.
    prob.advance(t_end=0.02, dt=5e-5, dt_max=1e-3)

    xc, dv = _surface_xd(prob)
    assert dv.max() > 0.02, f"no ponding occurred (max d={dv.max():.3e})"
    n = dv.size
    uphill = dv[: n // 2].mean()      # x near 0 (high ground)
    downhill = dv[n // 2:].mean()     # x near L (low ground)
    assert downhill > 1.5 * uphill, f"no downslope routing: uphill={uphill:.3e} downhill={downhill:.3e}"
    # closed domain conserves (no outflow yet); the limiter preserves the surface budget.
    cum_rain = rate * _top_length(prob) * 0.02
    assert abs((prob.total_water() - w0) - cum_rain) / cum_rain < 1e-6
    assert prob.surface_depth() >= -1e-12
    assert prob.d.x.array.min() >= -1e-12   # limiter holds d >= 0 everywhere
    assert np.all(np.isfinite(prob.psi.x.array))
