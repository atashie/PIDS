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
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.overland import overland_conveyance

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def test_2d_overland_operator_matches_module2():
    """The co-located lateral overland operator (tangential gradient on the host ds_top) is the SAME
    operator as standalone Module-2 overland (grad on the surface mesh) -- to machine precision.

    Realization A puts d co-located on the host (pinned 0 below the top) and uses grad_T = grad −
    (grad·n)n on ds_top. This pins down (adversarial-review concern) that grad_T recovers the true
    SURFACE gradient and is NOT polluted by the artificial vertical structure of the pinned field:
    we assemble the Manning FLUX residual K_s·grad_T(H_s)·grad_T(v) on the host top facets and the
    Module-2 flux K_s·grad(H_s)·grad(v) on the matching 1-D surface mesh, for identical d(x), z_b(x),
    and require them equal at the interior surface nodes. Both use the SAME overland_conveyance.
    """
    L, Hd, nx, nz, n_man, eps_S = 4.0, 1.0, 16, 6, 0.05, 1e-3
    d_expr = lambda x: 0.06 + 0.02 * np.sin(np.pi * x[0] / L)   # smooth, strictly positive
    zb_expr = lambda x: 0.05 * (L - x[0])                       # sloped bed -> grad H_s != 0

    # (a) coupled: grad_T overland flux on the 2-D host top facets
    host = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, Hd]], [nx, nz])
    fdim = host.topology.dim - 1
    host.topology.create_connectivity(fdim, host.topology.dim)
    top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[1], Hd)))
    mt = dmesh.meshtags(host, fdim, top, np.ones(top.size, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=host, subdomain_data=mt)(1)
    Vd = fem.functionspace(host, ("Lagrange", 1))
    dH = fem.Function(Vd); dH.interpolate(d_expr)
    zbH = fem.Function(Vd); zbH.interpolate(zb_expr)
    vH = ufl.TestFunction(Vd)
    nv = ufl.FacetNormal(host)
    gT = lambda f: ufl.grad(f) - ufl.dot(ufl.grad(f), nv) * nv
    HsH, KsH = overland_conveyance(dH, zbH, n_man, eps_S, grad=gT)
    bH = assemble_vector(fem.form(KsH * ufl.dot(gT(HsH), gT(vH)) * ds_top)); bH.assemble()
    topdofs = fem.locate_dofs_geometrical(Vd, lambda x: np.isclose(x[1], Hd))
    xtop = Vd.tabulate_dof_coordinates()[topdofs, 0]
    oH = np.argsort(xtop); xH, fH = xtop[oH], bH.getArray()[topdofs][oH]

    # (b) Module 2: grad overland flux on the matching 1-D surface mesh
    surf = dmesh.create_interval(MPI.COMM_WORLD, nx, [0.0, L])
    Vs = fem.functionspace(surf, ("Lagrange", 1))
    dS = fem.Function(Vs); dS.interpolate(d_expr)
    zbS = fem.Function(Vs); zbS.interpolate(zb_expr)
    vS = ufl.TestFunction(Vs)
    HsS, KsS = overland_conveyance(dS, zbS, n_man, eps_S)
    bS = assemble_vector(fem.form(KsS * ufl.dot(ufl.grad(HsS), ufl.grad(vS)) * ufl.dx)); bS.assemble()
    xc = Vs.tabulate_dof_coordinates()[:, 0]; oS = np.argsort(xc); xS, fS = xc[oS], bS.getArray()[oS]

    assert np.allclose(xH, xS)
    # compare interior nodes (the two ends differ by boundary treatment: ds_top facet ends vs the
    # 1-D mesh's natural no-flux ends).
    rel = np.abs(fH[1:-1] - fS[1:-1]) / (np.abs(fS[1:-1]).max() + 1e-30)
    assert rel.max() < 1e-9, f"co-located overland operator != Module-2 overland (max rel {rel.max():.2e})"


def _top_length(prob):
    import ufl
    from dolfinx import fem
    one = fem.Constant(prob.mesh, 1.0)
    return prob.mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * prob._ds_top)), op=MPI.SUM)


def _psi_top(prob):
    """ψ at the top surface (max over the top-facet ψ-dofs; uniform on a flat 2-D column)."""
    zc = prob.Vpsi.tabulate_dof_coordinates()[:, prob._zaxis]
    td = np.isclose(zc, zc.max())
    return float(prob.psi.x.array[td].max())


def test_2d_flat_reduces_to_1d_column():
    """A 2-D flat-top coupled column under uniform rain stays x-uniform AND matches the validated
    1-D coupled column quantitatively (ψ_top, infiltrated water, surface depth).

    The strongest reduction check: with a flat surface there is no lateral flow, so each x-column of
    the 2-D solve must reproduce the 1-D coupling. Same vertical resolution -> same ℓ_c -> same k_ex.
    """
    nz, rate, t_end, psi0 = 16, 0.05, 0.3, -2.0  # rain << Ks -> supply-limited, converges fast
    # 1-D reference column
    m1 = dmesh.create_interval(MPI.COMM_WORLD, nz, [0.0, 1.0])
    p1 = CoupledProblem(m1, SOIL)
    p1.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    p1.add_rain(rate)
    p1.advance(t_end=t_end, dt=1e-3, dt_max=0.05)
    psi_top_1d, soil_1d, d_1d = _psi_top(p1), p1.soil_water(), p1.surface_depth()

    # 2-D flat column (same vertical resolution; flat z_b = default 0)
    m2 = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [6, nz])
    p2 = CoupledProblem(m2, SOIL)
    p2.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    p2.add_rain(rate)
    p2.advance(t_end=t_end, dt=1e-3, dt_max=0.05)

    # x-uniform (no lateral structure on a flat surface) -- exact: the columns are identical
    d2 = p2.d.x.array[p2._top_dofs(p2.Vd)]
    assert d2.std() < 1e-6, f"2-D flat column not x-uniform (d std {d2.std():.2e})"
    # quantitative match to the 1-D column. Tolerance ~3% absorbs only the 2-D-triangle vs
    # 1-D-interval vertical-discretization difference (physics/ℓ_c/Kirchhoff leg are identical); a
    # wrong operator would differ by far more. (The sorptive Kirchhoff leg makes infiltration sharper,
    # which amplifies the triangle-vs-interval discretization gap from ~2% to ~2.3% -- 2026-06-06.)
    # soil_water is per-unit-width (domain width 1).
    assert abs(_psi_top(p2) - psi_top_1d) < 0.03, f"ψ_top {_psi_top(p2):.4f} vs 1-D {psi_top_1d:.4f}"
    assert abs(p2.soil_water() - soil_1d) / abs(soil_1d) < 0.03
    assert abs(p2.surface_depth() - d_1d) < 3e-3


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


def test_2d_surface_balance_is_structural():
    """Tolerance-INDEPENDENT structural-conservation guard (Codex review 2026-06-06): the surface
    (d-block) residual tested against v=1 over the FREE top rows must EXACTLY equal the physical balance
    (storage + sign-paired λ), to machine precision -- proving the eps_diag diagonal allocation
    contributes ZERO on the free rows (it sits on the pinned interior vertices) and no spurious VOLUME
    (dx) term leaks. This is a pure residual ASSEMBLY at a set state, so it does NOT depend on the
    Newton solve tolerance -- it is the real leak tripwire (the numeric gate below only tracks solver
    tolerance and is too loose to catch the historical ~1.18e-12 whole-domain eps_diag*dx leak).
    """
    from dolfinx.fem.petsc import assemble_vector
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [5, 4])
    prob = CoupledProblem(msh, SOIL)
    prob.dt.value = 0.1
    prob.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.0)
    # nontrivial state: d on the top, λ on the top (every surface term active)
    topd = prob._top_dofs(prob.Vd); topl = prob._top_dofs(prob.Vlam)
    prob.d.x.array[topd] = 0.03; prob.d.x.scatter_forward()
    prob.lam.x.array[topl] = 0.2; prob.lam.x.scatter_forward()
    prob._ensure_problem()  # build the forms

    b = assemble_vector(fem.form(prob.F_d)); b.assemble()
    assembled_free_sum = float(np.sum(b.getArray()[topd]))  # = <F_d, v=1> over the FREE top rows
    # physical balance for v=1 (flat, no rain/overland-contribution/outlet): storage + λ. overland flux
    # ~ grad_T(v)=grad_T(1)=0; eps_diag is on pinned dofs (excluded from topd). Computed independently.
    storage = fem.assemble_scalar(fem.form(((prob.d - prob.d_n) / prob.dt) * prob._ds_top))
    lam_term = fem.assemble_scalar(fem.form(prob.lam * prob._ds_top))
    expected = float(storage + lam_term)
    assert abs(assembled_free_sum - expected) <= 1e-13 * (abs(expected) + 1.0), \
        f"surface balance not structural: assembled {assembled_free_sum:.6e} vs physical {expected:.6e}"


def test_2d_closed_conservation_is_structural():
    """Closed flat column under smooth sub-Ks rain: total water grows by the rainfall to solver
    precision. The TIGHT structural guard is test_2d_surface_balance_is_structural (tolerance-free);
    this complementary end-to-end check confirms the SOLVED run conserves -- its gate (1e-11) tracks the
    Newton tolerance (snes_atol=1e-12; with the more-nonlinear Kirchhoff q_pot the closure sits at
    ~2.5e-12 rel), NOT a structural leak. Smooth (rate < Ks => no ponding) so the limiter is a no-op.
    """
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 2.0]], [12, 16])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    w0 = prob.total_water()
    rate = 0.1  # < Ks=0.25 -> supply-limited, smooth, no ponding/clipping
    prob.add_rain(rate)
    prob.advance(t_end=0.5, dt=1e-3, dt_max=0.05)

    cum_rain = rate * _top_length(prob) * 0.5
    assert prob.max_clip_seen == 0.0  # no clipping -> limiter is a no-op
    assert abs((prob.total_water() - w0) - cum_rain) / cum_rain < 1e-11  # solver-tolerance-limited


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
    # limiter does only LIGHT work (front undershoots are mm/cm) -> the accepted λ stays
    # near-consistent with the clipped d; and no degenerate drying fired (clip_mass_adjust ~ 0).
    assert prob.max_clip_seen < 0.05, f"heavy limiter clipping ({prob.max_clip_seen:.3e}) -> λ staleness"
    assert abs(prob.clip_mass_adjust) < 1e-9


def test_2d_limiter_degenerate_dries_and_tracks():
    """The coupled limiter must not SILENTLY create water when the surface-undershoot total <= 0.

    Mirrors the Module-2 regression (the limiter's degenerate branch): if the negative-depth
    undershoots outweigh all positive surface water (∫d ds_top <= 0) while a positive node survives
    clipping, a naive clip would jump the total from <=0 to >0 -- silent mass creation. The coupled
    limiter instead dries the surface (d>=0, ∫d ds_top -> 0) and records the unavoidable adjustment
    in clip_mass_adjust (never silent).
    """
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [4, 4])
    prob = CoupledProblem(msh, SOIL)
    topdofs = prob._top_dofs(prob.Vd)
    arr = prob.d.x.array
    arr[topdofs] = -0.4          # surface mostly negative (pathological near-dry undershoot)
    arr[topdofs[0]] = 0.05       # ... but one positive node would survive clipping
    prob.d.x.scatter_forward()
    oldtotal = prob.surface_water()
    assert oldtotal < 0.0                       # precondition: deficit outweighs the positive water

    prob._enforce_positivity()

    assert prob.d.x.array.min() >= -1e-12       # non-negative
    assert prob.surface_water() <= 1e-12        # dried, NOT left with created positive water
    assert prob.clip_mass_adjust == pytest.approx(-oldtotal, rel=1e-6)  # adjustment tracked, not silent


# ---------------------------------------------------------------------------------------------
# Lateral OUTFLOW boundary condition (free-drainage outlet) -- co-located realization A.
#
# In realization A the overland PDE lives on the host top facets (ds_top, codim-1), so the outlet
# (downstream end of the surface) is the BOUNDARY of ds_top -- a codim-2 host entity: in this 2-D
# cross-section, the downstream-TOP CORNER VERTEX. The outlet imposes the Manning normal-depth point
# discharge q_out = SECONDS_PER_DAY*(1/n_man)*d^{5/3}*sqrt(slope) [m^2/day per unit width] via a
# single-mesh VERTEX integral (dP) -- FFCX-native on stock 0.10 (NOT the mixed-dim codim-0 path that
# blocked realization S; verified in scratch/m3_outflow_spike.py). Same (locator, slope) API as
# Module-2 OverlandProblem.add_outflow_bc. [3-D extends the outlet to a perimeter CURVE = codim-2
# edges; that is a separate spike -- the dimension-agnostic normalized-band variant is the fallback.]
# ---------------------------------------------------------------------------------------------
def test_2d_outflow_bc_rejects_nonpositive_slope():
    """slope<=0 is a caller error (slope=0 dams the outlet; slope<0 gives sqrt(<0)=NaN)."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [2.0, 1.0]], [4, 4])
    prob = CoupledProblem(msh, SOIL)
    with pytest.raises(ValueError):
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 2.0), slope=0.0)
    with pytest.raises(ValueError):
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 2.0), slope=-0.01)


def test_2d_outflow_bc_rejects_empty_outlet():
    """A locator matching NO top-boundary vertex is a caller error (typo) -- fail loudly, not silently
    no-op (Codex review 2026-06-06). Uses a GLOBAL count, so a parallel-legitimately-empty rank does
    not falsely raise (the outlet vertex may live on another rank)."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [2.0, 1.0]], [4, 4])
    prob = CoupledProblem(msh, SOIL)
    with pytest.raises(ValueError):
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 123456.0), slope=0.05)  # matches nothing


def test_2d_outflow_point_discharge_magnitude():
    """The outlet discharge equals the Manning normal-depth value at the outlet node, in m^2/day.

    Pins the EXACT mechanism + the day<->second factor (hard-coded 86400, not imported): set a known
    uniform surface depth d0 on a known slope, add the outflow BC at x=L, and require outflow_rate()
    == 86400*(1/n)*d0^{5/3}*sqrt(slope). In the 2-D cross-section the outlet is a single top corner
    vertex, so the vertex integral equals the nodal point discharge exactly (no time-stepping needed).
    """
    L, n_man, S0, d0 = 4.0, 0.05, 0.05, 0.06
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, 1.0]], [16, 6])
    prob = CoupledProblem(msh, SOIL, n_man=n_man)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=d0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    expected = 86400.0 * (1.0 / n_man) * d0 ** (5.0 / 3.0) * np.sqrt(S0)  # one outlet corner -> point Q
    assert prob.outflow_rate() == pytest.approx(expected, rel=1e-6)


def test_2d_outflow_drains_and_conserves():
    """Heavy rain on a sloped hillslope with an OPEN downstream outlet: ponded water routes downhill
    and DRAINS through the outlet; the global books close as Δtotal = cum_rain - cum_outflow.

    The open-domain analogue of test_2d_lateral_redistribution_downslope (which is closed). Drives the
    coupled blocked Newton WITH the vertex-measure outflow term, the residual-consistent outflow
    accounting (last_outflow recorded pre-limiter; cum_outflow accumulated in step), and proves water
    genuinely leaves the system (cum_outflow > 0 and total grows by LESS than the full rainfall).
    """
    L = 5.0
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, 1.0]], [10, 5])
    prob = CoupledProblem(msh, SOIL, n_man=0.05)
    prob.set_initial_condition(lambda x: -0.8 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: 0.05 * (L - x[0]))   # 5% slope down toward the x=L outlet
    rate = 0.6  # > Ks -> infiltration-excess ponding
    prob.add_rain(rate)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=0.05)  # normal-depth free drain
    w0 = prob.total_water()

    prob.advance(t_end=0.02, dt=5e-5, dt_max=1e-3)

    cum_rain = rate * _top_length(prob) * 0.02
    # water genuinely left through the outlet, and a non-trivial fraction of the rain did so.
    assert prob.cum_outflow > 0.05 * cum_rain, f"negligible outflow ({prob.cum_outflow:.3e})"
    assert prob.outflow_rate() >= 0.0
    # global mass balance WITH the open outlet (residual-consistent, pre-limiter outflow).
    assert abs((prob.total_water() - w0) - (cum_rain - prob.cum_outflow)) / cum_rain < 1e-6
    # open domain retains LESS than it would closed (the outflow removed cum_outflow).
    assert (prob.total_water() - w0) < cum_rain
    assert prob.surface_depth() >= -1e-12
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.psi.x.array))
    assert abs(prob.clip_mass_adjust) < 1e-9
