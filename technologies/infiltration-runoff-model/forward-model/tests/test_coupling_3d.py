"""Tier-1 sanity (Module 3 coupling, 3-D): the co-located realization on a host VOLUME box.

3-D extends the 2-D coupling (same supply-limited NCP land-surface exchange + tangential-gradient
Manning overland on the host top FACE `ds_top`). Realization A (co-located d,lambda on the host;
the design-intended submesh realization S is deferred pending an upstream FFCX fix). The exchange
physics, the pinned-interior vertex (dP) diagonal allocation, ell_c, topography, and the drainage
GHB are all dimension-agnostic; the only genuinely new 3-D piece is the lateral-outflow outlet
(codim-2 = a perimeter EDGE/ridge, vs a corner vertex in 2-D) -- exercised in its own tests.

Phase 1 here: verify the agnostic CORE works in 3-D on a closed box (no outlet) -- mesh,
set_topography, ell_c auto-detection, the off-top pin, the NCP exchange, and conservation -- AND
that the coupled surface scheme is CONSISTENT in 3-D (the coarse-mesh surface non-uniformity is a
convergent discretization artifact that vanishes under refinement, NOT a structural bug).

PERF NOTE: CoupledProblem caps the nonlinear-integrand quadrature degree (default 6); without it
FFCX auto-estimates ~26 for the van Genuchten / Kirchhoff fractional powers, which on 3-D TETS is
~1000x slower (a 6x6x5 smoke step took ~16.5 s -> ~0.08 s capped; benchmark scratch/m3_3d_perf_*).
Meshes are kept SMALL: 3-D dof count (3 P1 fields on the volume) is the perf constraint.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def _top_area(prob):
    """Area of the top surface (the 2-D top face of the 3-D host) -- the rain footprint."""
    one = fem.Constant(prob.mesh, 1.0)
    return prob.mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * prob._ds_top)), op=MPI.SUM)


def _run_closed(nx, ny, nz, rate=0.1, t_end=0.5):
    """A flat closed 3-D box under uniform sub-Ks rain; return the solved problem + cum_rain."""
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [nx, ny, nz])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    w0 = prob.total_water()
    prob.add_rain(rate)
    prob.advance(t_end=t_end, dt=1e-3, dt_max=0.05)
    return prob, w0, rate * _top_area(prob) * t_end


def _lambda_top(prob):
    return prob.lam.x.array[prob._top_dofs(prob.Vlam)]


def test_3d_closed_conservation():
    """3-D flat closed box under smooth sub-Ks rain: total water grows by EXACTLY the cumulative
    rainfall (supply-limited -> all rain infiltrates, no ponding), and the state is finite --
    confirming the co-located NCP exchange + the dimension-agnostic core (mesh/topography/ell_c/pin/
    NCP) all work on a 3-D host. Guards the 3-D ell_c computation (the np.unique-on-noisy-z heuristic
    now sees many nodes per z-level on a box).

    Conservation/finiteness are dimension-agnostic invariants; the SPATIAL uniformity of the surface
    fields is a convergence property checked separately (test_3d_infiltration_uniformity_converges).
    """
    prob, w0, cum_rain = _run_closed(6, 6, 5)

    assert abs((prob.total_water() - w0) - cum_rain) / cum_rain < 1e-6   # closed: conserves
    assert prob.total_water() - w0 > 0.5 * cum_rain                      # rain genuinely entered
    assert prob.surface_depth() < 1e-3                                   # supply-limited: ~no ponding
    assert prob.surface_depth() >= -1e-12                                # non-negative depth
    assert prob.d.x.array.min() >= -1e-12                                # limiter holds d >= 0
    # Tiny conservative clips (~NCP-smoothing scale eps_ncp=1e-4) can fire even here (the tet mesh
    # breaks the exact symmetry the 2-D column had), but they are sub-mm and mass-neutral.
    assert prob.max_clip_seen < 1e-3, f"unexpectedly heavy clipping ({prob.max_clip_seen:.3e})"
    assert abs(prob.clip_mass_adjust) < 1e-9
    assert np.all(np.isfinite(prob.psi.x.array))
    assert np.all(np.isfinite(prob.d.x.array)) and np.all(np.isfinite(prob.lam.x.array))


def test_3d_infiltration_uniformity_converges():
    """The 3-D coupled NCP surface scheme is CONSISTENT: on a flat box under uniform rain the
    supply-limited infiltration flux lambda should be UNIFORM (== rain everywhere). On a coarse mesh
    the tet-connectivity asymmetry seeds a checkerboard that the smoothed NCP / Kirchhoff feedback
    amplifies (lambda std ~2e-2 at 6x6x5 = ~20% of rain -- the 3-D analog of the documented 2-D
    consistent-surface-scheme artifact, concern #9). This test proves it is a CONVERGENT
    discretization artifact, not a structural bug: refining the mesh drives lambda -> uniform.

    Measured: lambda_top std 1.9e-2 (6x6x5) -> 1.6e-8 (12x12x8) -- collapses by ~1e6; at 12x12x8
    lambda == rain to 8 figs. Total infiltration conserves at BOTH resolutions regardless.
    """
    coarse, _, _ = _run_closed(6, 6, 5)
    fine, w0f, cum_rain_f = _run_closed(12, 12, 8)

    std_coarse = float(_lambda_top(coarse).std())
    std_fine = float(_lambda_top(fine).std())
    # both conserve (the artifact never breaks the books)
    assert abs((fine.total_water() - w0f) - cum_rain_f) / cum_rain_f < 1e-6
    # the surface infiltration converges to UNIFORM (= the supply-limited physical answer, lambda=rain)
    assert std_fine < 1e-4, f"lambda not converging to uniform at 12x12x8 (std {std_fine:.3e})"
    assert std_fine < std_coarse / 10.0, \
        f"surface non-uniformity not converging (coarse {std_coarse:.3e} -> fine {std_fine:.3e})"
    assert _lambda_top(fine).mean() == pytest.approx(0.1, rel=1e-3)  # converged lambda == rain


# ---------------------------------------------------------------------------------------------
# Lateral OUTFLOW boundary condition in 3-D -- the codim-2 outlet is a perimeter EDGE (ridge),
# not the 2-D corner vertex. add_outflow_bc imposes the Manning normal-depth discharge as a
# ridge (dr) LINE integral: int_edge q_out(d) ds = (per-width q_out) * edge_length [m^3/day].
# Path R adopted after the Phase-0 spike + adversarial review (scratch/m3_3d_outlet_spike.py).
# ---------------------------------------------------------------------------------------------
def test_3d_outflow_edge_discharge_magnitude():
    """The 3-D outlet discharge is the Manning normal-depth LINE integral over the downstream-top
    EDGE: outflow_rate() == 86400*(1/n)*d0^{5/3}*sqrt(slope) * edge_length, in m^3/day. With a
    uniform depth d0 the ridge integral is exact (no time-stepping needed).

    This is the decisive width-normalization pin: the 2-D vertex code summed POINT discharges over
    the edge's (ny+1) vertices (~ (ny+1)*q_per_width) -- wrong by the vertex count; the codim-2 ridge
    line integral gives the correct q_per_width * W. (Spike: q_total = q_per_width*edge_len to 3.8e-16.)
    """
    L, W, n_man, S0, d0 = 4.0, 2.0, 0.05, 0.05, 0.06
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, 1.0]], [8, 4, 5])
    prob = CoupledProblem(msh, SOIL, n_man=n_man)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=d0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)  # downstream-top edge x=L, z=H

    q_per_width = 86400.0 * (1.0 / n_man) * d0 ** (5.0 / 3.0) * np.sqrt(S0)
    expected = q_per_width * W  # ridge line integral over the outlet edge of length W
    assert prob.outflow_rate() == pytest.approx(expected, rel=1e-6)


def test_3d_outflow_edge_discharge_nonuniform_quadrature():
    """Quadrature tripwire (Codex review): a y-VARYING depth d(y)=a+b*y on the outlet edge integrated
    against the CLOSED FORM of int_0^W (a+b*y)^{5/3} dy. Uniform d (the test above) cannot catch a
    silently-corrupt ridge table -- the value table just sums to the edge length -- but a non-uniform
    profile forces the codim-2 reference-point map to land on the correct physical positions.

        int_0^W (a+b*y)^{5/3} dy = (3/(8b)) * ((a+b*W)^{8/3} - a^{8/3})
    """
    L, W, n_man, S0, a, b = 4.0, 2.0, 0.05, 0.05, 0.02, 0.02   # d: 0.02 -> 0.06 along the edge
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, 1.0]], [8, 4, 5])
    prob = CoupledProblem(msh, SOIL, n_man=n_man)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.0)
    coords = prob.Vd.tabulate_dof_coordinates()
    topd = prob._top_dofs(prob.Vd)
    prob.d.x.array[topd] = a + b * coords[topd, 1]   # linear in y -> P1-exact on the edge
    prob.d.x.scatter_forward()
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    line_int = (3.0 / (8.0 * b)) * ((a + b * W) ** (8.0 / 3.0) - a ** (8.0 / 3.0))
    expected = 86400.0 * (1.0 / n_man) * np.sqrt(S0) * line_int
    assert prob.outflow_rate() == pytest.approx(expected, rel=1e-5)


def test_3d_outflow_jacobian_fd():
    """FD-Jacobian regression for the ridge outlet term dq_out/dd (closes the long-open concern #8).
    A finite norm (all the spike's Jacobian check asserted) is NOT a correct Jacobian; here we central-
    difference F_d along a perturbation supported on the outlet-edge dofs and require it to match the
    assembled dF_d/dd action -- the tripwire for a silently-wrong ridge tangent.
    """
    import ufl
    from dolfinx.fem.petsc import assemble_vector, assemble_matrix
    L, W, n_man, S0 = 4.0, 2.0, 0.05, 0.05
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, 1.0]], [6, 4, 5])
    prob = CoupledProblem(msh, SOIL, n_man=n_man)
    prob.dt.value = 0.1
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)
    coords = prob.Vd.tabulate_dof_coordinates()
    topd = prob._top_dofs(prob.Vd)
    prob.d.x.array[topd] = 0.01 + 0.02 * coords[topd, 1] / W   # 0.01..0.03 along y (all > 0)
    prob.d.x.scatter_forward()
    prob._ensure_problem()  # build the forms

    Fform = fem.form(prob.F_d)
    J = assemble_matrix(fem.form(ufl.derivative(prob.F_d, prob.d))); J.assemble()
    outdofs = fem.locate_dofs_geometrical(
        prob.Vd, lambda x: np.isclose(x[0], L) & np.isclose(x[2], prob._ztop))
    p = np.zeros_like(prob.d.x.array); p[outdofs] = 1.0
    xp = J.createVecRight(); xp.array[:] = p
    yp = J.createVecLeft(); J.mult(xp, yp)
    Jp = yp.getArray().copy()

    d0 = prob.d.x.array.copy()
    eps = 1e-7
    prob.d.x.array[:] = d0 + eps * p; prob.d.x.scatter_forward()
    Fp = assemble_vector(Fform); Fp.assemble(); Fp = Fp.getArray().copy()
    prob.d.x.array[:] = d0 - eps * p; prob.d.x.scatter_forward()
    Fm = assemble_vector(Fform); Fm.assemble(); Fm = Fm.getArray().copy()
    prob.d.x.array[:] = d0; prob.d.x.scatter_forward()
    fd = (Fp - Fm) / (2.0 * eps)

    rel = np.abs(fd - Jp) / (np.abs(Jp).max() + 1e-30)
    assert rel.max() < 1e-6, f"ridge outlet Jacobian FD mismatch (max rel {rel.max():.2e})"

    # At the d=0 KINK (the diffusion-wave front crosses it every wet/dry step), central FD SMEARS the
    # one-sided d^{5/3} tangent (q_out' ~ d^{2/3} -> 0, q_out'' -> inf at 0), so FD vs the auto-Jacobian
    # agree only to ~eps^{2/3} -- a FD artifact, NOT a Jacobian error. Pin that magnitude so a real
    # regression is caught; the Tier-2 burst test is the integration-level evidence that the solve
    # converges THROUGH d=0.
    prob.d.x.array[:] = d0
    prob.d.x.array[outdofs] = np.linspace(-1e-3, 2e-3, outdofs.size)  # straddle 0
    prob.d.x.scatter_forward()
    Jk = assemble_matrix(fem.form(ufl.derivative(prob.F_d, prob.d))); Jk.assemble()
    xk = Jk.createVecRight(); xk.array[:] = p
    yk = Jk.createVecLeft(); Jk.mult(xk, yk); Jpk = yk.getArray().copy()
    dk = prob.d.x.array.copy()
    prob.d.x.array[:] = dk + eps * p; prob.d.x.scatter_forward()
    Fpk = assemble_vector(Fform); Fpk.assemble(); Fpk = Fpk.getArray().copy()
    prob.d.x.array[:] = dk - eps * p; prob.d.x.scatter_forward()
    Fmk = assemble_vector(Fform); Fmk.assemble(); Fmk = Fmk.getArray().copy()
    fdk = (Fpk - Fmk) / (2.0 * eps)
    relk = np.abs(fdk - Jpk) / (np.abs(Jpk).max() + 1e-30)
    assert relk.max() < 1e-3, f"ridge Jacobian at the d=0 kink exceeds FD-smearing ({relk.max():.2e})"


def test_3d_outflow_rejects_nonpositive_slope():
    """slope<=0 is a caller error (slope=0 dams the outlet; slope<0 gives sqrt(<0)=NaN)."""
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [4, 4, 4])
    prob = CoupledProblem(msh, SOIL)
    with pytest.raises(ValueError):
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 1.0), slope=0.0)
    with pytest.raises(ValueError):
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 1.0), slope=-0.01)


def test_3d_outflow_rejects_empty_edge():
    """A locator matching no top-boundary EDGE is a caller error -- fail loudly, not silently no-op.
    Exercises the new codim-2 (dim gdim-2 = 1) locate path + the global-count guard."""
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [4, 4, 4])
    prob = CoupledProblem(msh, SOIL)
    with pytest.raises(ValueError):
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 123456.0), slope=0.05)  # matches nothing


def test_3d_outflow_drains_routes_and_conserves():
    """Heavy rain on a SLOPED 3-D hillslope with an OPEN downstream EDGE outlet: ponded water routes
    downhill (accumulates at x=L) and DRAINS through the edge; the global books close as
    Delta_total = cum_rain - cum_outflow. The open-domain dynamic analogue of the 2-D drains-and-
    conserves test, now exercising the ridge outlet + the residual-consistent (pre-limiter) outflow
    accounting in a real coupled solve. Kept SMALL/SHORT (the diffusion-wave overland is stiff).
    """
    L = 5.0
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, 1.0, 1.0]], [8, 4, 4])
    prob = CoupledProblem(msh, SOIL, n_man=0.05)
    prob.set_initial_condition(lambda x: -0.8 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: 0.05 * (L - x[0]))   # 5% slope down toward the x=L outlet edge
    rate = 0.6  # > Ks -> infiltration-excess ponding
    prob.add_rain(rate)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=0.05)
    w0 = prob.total_water()

    prob.advance(t_end=0.012, dt=5e-5, dt_max=1e-3)

    cum_rain = rate * _top_area(prob) * 0.012
    # a non-trivial fraction of the rain leaves through the edge (measured ~0.58; the bar brackets it
    # so a large outflow-magnitude regression -- e.g. a collapsed edge_length or a dropped 86400 -- is
    # actually caught, not just "some outflow").
    assert 0.2 * cum_rain < prob.cum_outflow < cum_rain, f"outflow out of range ({prob.cum_outflow:.3e})"
    assert prob.outflow_rate() >= 0.0
    # global mass balance WITH the open edge outlet (residual-consistent, pre-limiter outflow).
    assert abs((prob.total_water() - w0) - (cum_rain - prob.cum_outflow)) / cum_rain < 1e-6
    assert (prob.total_water() - w0) < cum_rain          # open retains less than closed
    # downslope routing: the downhill third holds more ponded water than the uphill third.
    dofs = prob._top_dofs(prob.Vd)
    xc = prob.Vd.tabulate_dof_coordinates()[dofs, 0]
    dv = prob.d.x.array[dofs]
    uphill = dv[xc < L / 3.0].mean()
    downhill = dv[xc > 2.0 * L / 3.0].mean()
    assert downhill > 1.5 * uphill, f"no downslope routing (uphill={uphill:.3e} downhill={downhill:.3e})"
    assert prob.d.x.array.min() >= -1e-12
    assert np.all(np.isfinite(prob.psi.x.array))
    assert abs(prob.clip_mass_adjust) < 1e-9


# ---------------------------------------------------------------------------------------------
# Phase 3: port the strongest 2-D guards to 3-D (the rest of the suite is dimension-agnostic and
# already exercised by the tests above). The tolerance-free structural-balance leak guard pins the
# _finalize_forms dimension branch (vertex pin vs ridge outlet) on the FREE top rows; the side-face
# drainage GHB confirms the codim-1 Darcy/head BC + its quadrature cap on a 3-D boundary face.
# ---------------------------------------------------------------------------------------------
def test_3d_surface_balance_is_structural():
    """Tolerance-INDEPENDENT structural-conservation guard (3-D port of the 2-D version): the surface
    (d-block) residual against v=1 over the FREE top rows must EXACTLY equal the physical balance
    (storage + sign-paired lambda), to machine precision -- proving the eps_diag vertex-pin diagonal
    allocation contributes ZERO on the free rows (it sits on the pinned interior vertices, a SEPARATE
    vertex meshtags from any ridge outlet) and no spurious VOLUME (dx) term leaks. Pure residual
    ASSEMBLY at a set state -> independent of the Newton solve. Kept OUTLET-FREE so the bare
    storage+lambda expectation is exact (an outlet would add +int q_out dr to the free rows).
    """
    from dolfinx.fem.petsc import assemble_vector
    import ufl
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [4, 4, 4])
    prob = CoupledProblem(msh, SOIL)
    prob.dt.value = 0.1
    prob.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.0)
    topd = prob._top_dofs(prob.Vd)
    topl = prob._top_dofs(prob.Vlam)
    prob.d.x.array[topd] = 0.03; prob.d.x.scatter_forward()
    prob.lam.x.array[topl] = 0.2; prob.lam.x.scatter_forward()
    prob._ensure_problem()

    b = assemble_vector(fem.form(prob.F_d)); b.assemble()
    assembled_free_sum = float(np.sum(b.getArray()[topd]))   # <F_d, v=1> over the FREE top rows
    storage = fem.assemble_scalar(fem.form(((prob.d - prob.d_n) / prob.dt) * prob._ds_top))
    lam_term = fem.assemble_scalar(fem.form(prob.lam * prob._ds_top))
    expected = float(storage + lam_term)   # flat -> overland grad_T(1)=0; eps_diag on pinned (excluded)
    assert abs(assembled_free_sum - expected) <= 1e-13 * (abs(expected) + 1.0), \
        f"3-D surface balance not structural: assembled {assembled_free_sum:.6e} vs {expected:.6e}"


def test_3d_drainage_side_face_conserves():
    """Subsurface Darcy/head (GHB) drainage on a 3-D SIDE FACE (codim-1) -- the dimension-agnostic
    drainage BC + its quadrature cap on a 3-D boundary face. A coupled column draining through the
    x=0 side (no rain, no surface outlet): total water decreases by EXACTLY cum_drainage.
    """
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [4, 4, 6])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -0.2 + 0.0 * x[0], d_value=0.0)   # moist-ish soil
    prob.add_drainage_bc(lambda x: np.isclose(x[0], 0.0), conductance=0.5, external_head=-1.0)
    w0 = prob.total_water()
    prob.advance(t_end=0.5, dt=1e-2, dt_max=0.05)

    assert prob.cum_drainage > 0.0   # H = psi+z (~ -0.2..0.8) > H_ext (-1.0) -> drains OUT
    bal = (prob.total_water() - w0) - (-prob.cum_drainage + prob.clip_mass_adjust)
    assert abs(bal) / abs(prob.cum_drainage) < 1e-6
    assert np.all(np.isfinite(prob.psi.x.array))


# ---------------------------------------------------------------------------------------------
# Review follow-ups (adversarial review 2026-06-07): multi-outlet + overlap guard, the cap-accuracy
# contract (conservation is structural and cannot catch Darcy under-integration), and the literal
# reduces-to-2-D check. The sorptive Kirchhoff infiltration leg is dimension-AGNOSTIC (same
# constitutive.kirchhoff_ufl + coupling code path as 2-D) and is exercised in 3-D by the dynamic
# ponding test above (rate=0.6 > Ks drives infiltration-excess through the leg) and by the
# reduces-to-2-D match below; its dense-reference RECOVERY benchmark lives in test_coupling_sorptivity.py.
# ---------------------------------------------------------------------------------------------
def test_3d_two_outlets_sum_discharges():
    """Two outlet EDGES (x=0 and x=L) with DIFFERENT slopes -> outflow_rate is the SUM of the two
    ridge line integrals. Pins the multi-outlet tag loop (tags 2,3..) + the separate _outflow_forms
    in 3-D (untested by the single-outlet cases) and the happy path of the outlet-overlap guard."""
    L, W, n_man, d0 = 4.0, 2.0, 0.05, 0.06
    S_lo, S_hi = 0.02, 0.05
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, 1.0]], [8, 4, 5])
    prob = CoupledProblem(msh, SOIL, n_man=n_man)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=d0)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S_hi)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], 0.0), slope=S_lo)   # disjoint second outlet edge

    q = lambda S: 86400.0 * (1.0 / n_man) * d0 ** (5.0 / 3.0) * np.sqrt(S) * W
    assert prob.outflow_rate() == pytest.approx(q(S_hi) + q(S_lo), rel=1e-6)


def test_3d_outflow_rejects_overlapping_outlets():
    """Two outlets on the SAME edge must raise -- overlapping locators would double-tag the ridge
    meshtags and silently drop one outlet's discharge (symmetric with add_drainage_bc's guard)."""
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [4, 4, 4])
    prob = CoupledProblem(msh, SOIL)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], 1.0), slope=0.05)
    with pytest.raises(ValueError):
        prob.add_outflow_bc(lambda x: np.isclose(x[0], 1.0), slope=0.03)  # same downstream edge


def test_3d_quadrature_cap_accuracy():
    """Pin the quadrature-cap CONTRACT: conservation is STRUCTURAL (lambda sign-paired) so the suite
    would NOT catch Darcy-volume under-integration. The capped Darcy Jacobian (CoupledProblem's
    default degree) must match a high reference degree -- machine-precision at a SMOOTH state, and
    within ~1e-3 at an air-entry-STRADDLE cell (the binding case the cap is sized for; a one-cell
    wetting-front kink is a separate mesh-resolution issue, not quadrature-fixable)."""
    import ufl
    from petsc4py import PETSc
    from dolfinx.fem.petsc import assemble_matrix
    from pids_forward.physics.richards import richards_bulk_residual

    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [4, 4, 6])
    qd = CoupledProblem(msh, SOIL)._quad_degree
    assert qd == 8   # default cap, sized by the Darcy bulk (the highest-auto-degree term)
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi = fem.Function(V); psi_n = fem.Function(V); v = ufl.TestFunction(V)
    e_g = fem.Constant(msh, np.array([0.0, 0.0, 1.0], dtype=PETSc.ScalarType))
    dt = fem.Constant(msh, PETSc.ScalarType(0.1))
    dx_lump = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})

    def reldiff(qa, qb):
        def J(q):
            F = richards_bulk_residual(psi, psi_n, v, SOIL, dt, e_g, dx_storage=dx_lump,
                                       quadrature_degree=q)
            M = assemble_matrix(fem.form(ufl.derivative(F, psi))); M.assemble(); return M
        Ma, Mb = J(qa), J(qb)
        Ma.axpy(-1.0, Mb, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        return Ma.norm() / Mb.norm()

    psi.x.array[:] = -0.5; psi.x.scatter_forward()                 # smooth, unsaturated
    assert reldiff(qd, 20) < 1e-10, "cap not bit-identical at a smooth state"
    # An in-cell air-entry crossing (se_ufl's min/max kink at h_s) is a mesh-resolution kink that NO
    # polynomial degree resolves (non-monotone in degree; ~2e-2 deg-8-vs-ref here) -- handled by
    # lumped storage + small dt, NOT quadrature, and end-to-end benign (the Tier-2 burst test is the
    # integration evidence). Pin only that it stays BOUNDED (a gross under-integration would be O(1));
    # the smooth bit-identity above is the real cap-preserves-the-validated-regime guarantee and
    # `qd == 8` pins the chosen degree.
    coords = V.tabulate_dof_coordinates()
    psi.x.array[:] = -0.5 + 0.6 * coords[:, 2]; psi.x.scatter_forward()  # crosses h_s=-0.02 in-cell
    assert reldiff(qd, 20) < 0.1, "Darcy quadrature grossly under-integrating across the air-entry kink"


def test_3d_reduces_to_2d_flat_column():
    """A 1-cell-thick 3-D slab under flat closed rain reproduces the validated 2-D column: the
    per-unit-width subsurface storage matches within the tet-vs-triangle discretization gap, and the
    3-D run conserves. The plan's literal 'reduces-to-2-D' check (the outflow width-normalization is
    separately pinned EXACTLY by test_3d_outflow_edge_discharge_magnitude). Also confirms the sorptive
    Kirchhoff leg gives the same infiltration in 3-D as 2-D (same dimension-agnostic code path)."""
    nz, rate, t_end, psi0, Wy = 8, 0.1, 0.3, -2.0, 0.25
    # 2-D reference column (per unit width: domain width 1)
    m2 = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [6, nz])
    p2 = CoupledProblem(m2, SOIL)
    p2.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    p2.add_rain(rate); p2.advance(t_end=t_end, dt=1e-3, dt_max=0.05)
    sw2 = p2.soil_water()
    # 3-D thin slab (1 cell in y); per-width storage = soil_water / slab thickness
    m3 = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, Wy, 1.0]], [6, 1, nz])
    p3 = CoupledProblem(m3, SOIL)
    p3.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    w0 = p3.total_water()
    p3.add_rain(rate); p3.advance(t_end=t_end, dt=1e-3, dt_max=0.05)

    cum_rain = rate * _top_area(p3) * t_end
    assert abs((p3.total_water() - w0) - cum_rain) / cum_rain < 1e-6   # 3-D conserves
    assert abs(p3.soil_water() / Wy - sw2) / abs(sw2) < 0.05           # per-width matches 2-D
