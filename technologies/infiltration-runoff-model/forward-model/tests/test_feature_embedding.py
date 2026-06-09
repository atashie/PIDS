"""Module 4 (§E) Phase-2: the co-located embedded-feature PRIMITIVE, built test-first.

A PIDS feature = a 1-D vector Γ embedded in the 3-D Richards host, carrying its OWN head ``H_f`` (a P1
field co-located on the host, pinned to 0 off Γ -- realization A, the Phase-0 spike's validated route),
contributing CONVEYANCE (1-D Darcy along the centerline, tangential projection ∇H_f·t̂), STORAGE, and a
sign-paired EXCHANGE with the host soil (σ·(H_f−ψ) into the feature, −into the host). These tests pin
each primitive in isolation before Phase 3's sorptive σ closure and Phase 4's CoupledProblem integration.
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
import dolfinx.fem.petsc as fp
from petsc4py import PETSc

from pids_forward.physics.feature import EmbeddedFeature

_ON_FEAT = lambda x: np.isclose(x[1], 0.5) & np.isclose(x[2], 0.5)

COMM = MPI.COMM_WORLD
_LU = {
    "snes_type": "newtonls", "snes_linesearch_type": "basic",
    "snes_rtol": 1e-12, "snes_atol": 1e-14, "snes_max_it": 25,
    "ksp_type": "preonly", "pc_type": "lu",
}


def _box_feature(n=6, tangent=(1.0, 0.0, 0.0), **params):
    """Host unit box with a straight feature Γ = the centerline along x at y=z=0.5 (L=1)."""
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [n, n, n])
    feat = EmbeddedFeature(msh, _ON_FEAT, tangent=tangent, **params)
    return msh, feat


def _end_dofs(feat, x0):
    return fem.locate_dofs_geometrical(
        feat.V, lambda x: np.isclose(x[0], x0) & np.isclose(x[1], 0.5) & np.isclose(x[2], 0.5))


def _solve(feat, F, bcs):
    prob = NonlinearProblem(F, feat.Hf, bcs=bcs, petsc_options_prefix="feat_", petsc_options=_LU)
    prob.solve()
    return prob.solver.getConvergedReason() > 0


def test_conveyance_recovers_analytical_1d_darcy_current():
    """A straight embedded feature with a head drop ΔH across its ends conveys the EXACT 1-D Darcy
    current q = K_feat·A·ΔH/L -- the granular-conveyance primitive. The tangential projection ∇H_f·t̂
    is decisive: the full ∇·∇ (cell-trace-dependent) gives the wrong current (Phase-0 spike).
    """
    K_feat, A, L, dH = 2.0, 1.5, 1.0, 1.0
    msh, feat = _box_feature(n=6, K_feat=K_feat, area=A, porosity=0.3)
    v = ufl.TestFunction(feat.V)
    bcs = [feat.pin_bc(),
           fem.dirichletbc(PETSc.ScalarType(dH), _end_dofs(feat, 0.0), feat.V),
           fem.dirichletbc(PETSc.ScalarType(0.0), _end_dofs(feat, 1.0), feat.V)]
    assert _solve(feat, feat.hf_residual(v), bcs), "conveyance solve did not converge"
    q = feat.conveyance_current()
    assert abs(abs(q) - K_feat * A * dH / L) < 1e-10, \
        f"conveyance current {q:.6e} != analytical {K_feat * A * dH / L:.6e}"


def test_storage_accounting_fills_to_phi_area_length():
    """Feature storage = ∫_Γ φ·A·H_f dΓ; a unit fill (H_f≡1) holds exactly φ·A·|Γ| (the capacity)."""
    phi, A = 0.3, 1.5
    msh, feat = _box_feature(n=6, K_feat=2.0, area=A, porosity=phi)
    feat.Hf.x.array[:] = 1.0
    feat.Hf.x.scatter_forward()
    assert abs(feat.stored_water() - phi * A * feat.length) < 1e-12, \
        f"stored {feat.stored_water():.6e} != φ·A·|Γ| = {phi * A * feat.length:.6e}"


def test_exchange_is_structurally_conservative():
    """Sign-paired exchange σ·(H_f−ψ): what leaves the host enters the feature, to MACHINE precision
    (tolerance-free) -- the same term on the same dΓ with opposite sign in the two blocks.
    """
    msh, feat = _box_feature(n=6, K_feat=2.0, area=1.5, porosity=0.3, sigma=0.7)
    feat.Hf.x.array[:] = 0.5
    feat.Hf.x.scatter_forward()
    psi = fem.Function(feat.V)
    psi.x.array[:] = -0.2
    psi.x.scatter_forward()
    vf, wh = ufl.TestFunction(feat.V), ufl.TestFunction(feat.V)
    Rf = fp.assemble_vector(fem.form(feat.exchange_into_feature(vf, psi))); Rf.assemble()
    Rh = fp.assemble_vector(fem.form(feat.exchange_into_host(wh, psi))); Rh.assemble()
    assert abs(float(Rf.array.sum()) + float(Rh.array.sum())) < 1e-14, "exchange not sign-paired"


def test_pin_is_conservation_neutral():
    """The off-Γ pin can hold ANY value (it is overwritten Dirichlet); every feature integral lives on
    dΓ over Γ, so off-Γ H_f does not perturb the conserved feature quantities."""
    msh, feat = _box_feature(n=6, K_feat=2.0, area=1.5, porosity=0.3)
    gdofs = fem.locate_dofs_geometrical(feat.V, _ON_FEAT)
    feat.Hf.x.array[:] = 0.0
    feat.Hf.x.array[gdofs] = 0.5
    feat.Hf.x.scatter_forward()
    s0 = feat.stored_water()
    offdofs = fem.locate_dofs_geometrical(feat.V, lambda x: ~_ON_FEAT(x))
    feat.Hf.x.array[offdofs] = 99.0   # garbage off Γ
    feat.Hf.x.scatter_forward()
    assert abs(feat.stored_water() - s0) < 1e-13, "off-Γ values leaked into a feature integral"

    # the dP diagonal-allocation must touch ONLY off-Γ rows (scoping): Γ rows get zero diagonal from it,
    # off-Γ rows get a nonzero diagonal (the slot the Dirichlet pin overwrites).
    dHf = ufl.TrialFunction(feat.V)
    M = fp.assemble_matrix(fem.form(ufl.derivative(feat.Hf * ufl.TestFunction(feat.V) * feat._dPoff,
                                                   feat.Hf, dHf)))
    M.assemble()
    diag = M.getDiagonal().array
    assert np.allclose(diag[feat._gamma_dofs], 0.0), "pin diagonal allocation touched a Γ row"
    assert np.all(diag[feat._off_dofs] > 0.0), "pin diagonal allocation missed an off-Γ row"


def test_exchange_flux_magnitude_and_sealed_reduces_to_nothing():
    """The net soil↔feature exchange flux = ∫_Γ σ·(H_f−ψ) dΓ = σ·(H_f−ψ)·|Γ| (the simple-σ primitive),
    and a clay-SEALED feature (σ=0, conveyance-only) exchanges NOTHING -- it reduces to nothing."""
    Hf_val, psi_val = 0.5, -0.2
    msh, feat = _box_feature(n=6, K_feat=2.0, area=1.5, porosity=0.3, sigma=0.7)
    feat.Hf.x.array[:] = Hf_val
    feat.Hf.x.scatter_forward()
    psi = fem.Function(feat.V)
    psi.x.array[:] = psi_val
    psi.x.scatter_forward()
    expected = 0.7 * (Hf_val - psi_val) * feat.length
    assert abs(feat.host_exchange_flux(psi) - expected) < 1e-12, "exchange flux magnitude wrong"

    msh0, sealed = _box_feature(n=6, K_feat=2.0, area=1.5, porosity=0.3, sigma=0.0)
    sealed.Hf.x.array[:] = Hf_val
    sealed.Hf.x.scatter_forward()
    psi0 = fem.Function(sealed.V)
    psi0.x.array[:] = psi_val
    psi0.x.scatter_forward()
    assert sealed.host_exchange_flux(psi0) == 0.0, "sealed σ=0 feature still exchanged"


def test_conveyance_current_is_mesh_independent():
    """The conveyance current = K_feat·A·ΔH/L EXACTLY at every host refinement (1-D Darcy is linear ⇒
    P1-exact; the tangential embedding adds no spurious cell-trace / refinement dependence)."""
    for n in (4, 6, 10):
        msh, feat = _box_feature(n=n, K_feat=2.0, area=1.5, porosity=0.3)
        v = ufl.TestFunction(feat.V)
        bcs = [feat.pin_bc(),
               fem.dirichletbc(PETSc.ScalarType(1.0), _end_dofs(feat, 0.0), feat.V),
               fem.dirichletbc(PETSc.ScalarType(0.0), _end_dofs(feat, 1.0), feat.V)]
        assert _solve(feat, feat.hf_residual(v), bcs), f"n={n} did not converge"
        assert abs(abs(feat.conveyance_current()) - 2.0 * 1.5 * 1.0 / 1.0) < 1e-10, f"n={n}"


def test_blocked_newton_exchange_conserves_total_water():
    """A monolithic [ψ, H_f] backward-Euler step with ONLY the sign-paired exchange coupling (closed,
    no external flux) conserves the total water  C_h·∫ψ dx + ∫_Γ φ·A·H_f dΓ  to the solver gate -- the
    'blocked Newton working' + structural conservation. The feature starts WETTER than the soil, so it
    DISPERSES into the soil (H_f drops, ψ rises); the closed total is invariant.
    """
    from dolfinx.fem.petsc import NonlinearProblem
    msh, feat = _box_feature(n=6, K_feat=2.0, area=1.5, porosity=0.3, sigma=2.0)
    Vpsi = fem.functionspace(msh, ("Lagrange", 1))
    psi, psi_n = fem.Function(Vpsi), fem.Function(Vpsi)
    psi.x.array[:] = -0.5; psi_n.x.array[:] = -0.5
    feat.Hf.x.array[:] = 0.8; feat.Hf_n.x.array[:] = 0.8       # feature wetter -> disperses
    C_h = 0.4
    dt = fem.Constant(msh, PETSc.ScalarType(1.0))
    w, v = ufl.TestFunction(Vpsi), ufl.TestFunction(feat.V)
    F_psi = C_h * (psi - psi_n) / dt * w * ufl.dx + feat.exchange_into_host(w, psi)
    F_hf = (feat.storage_form(v, dt) + feat.conveyance_form(v)
            + feat.exchange_into_feature(v, psi) + feat.Hf * v * feat._dPoff)

    def total():
        h = C_h * psi * ufl.dx
        host = msh.comm.allreduce(fem.assemble_scalar(fem.form(h)), op=MPI.SUM)
        return host + feat.stored_water()

    w0 = total()
    prob = NonlinearProblem([F_psi, F_hf], [psi, feat.Hf], bcs=[feat.pin_bc()],
                            petsc_options_prefix="blk_", petsc_options={**_LU, "pc_factor_mat_solver_type": "mumps"},
                            kind="mpi")
    prob.solve()
    assert prob.solver.getConvergedReason() > 0, "blocked Newton did not converge"
    assert feat.host_exchange_flux(psi) > 1e-6, "no exchange happened (test is vacuous)"
    assert abs(total() - w0) / abs(w0) < 1e-9, f"total water not conserved: {total():.6e} vs {w0:.6e}"
    # DIRECTION (not just conservation): feature wetter (H_f=0.8 > ψ=-0.5) ⇒ it DISPERSES -> H_f drops,
    # ψ rises. A sign-reversed-but-paired coupling would still conserve; this rules it out.
    gdofs = fem.locate_dofs_geometrical(feat.V, _ON_FEAT)
    assert float(feat.Hf.x.array[gdofs].mean()) < 0.8 - 1e-9, "feature did not disperse (H_f should drop)"
    assert float(psi.x.array.mean()) > -0.5 + 1e-9, "soil did not receive water (ψ should rise)"


# --- hardening (Codex adversarial review 2026-06-08) ---------------------------------------------
import pytest


def test_rejects_empty_feature():
    """A locator that selects no feature edges must fail LOUDLY (not crash in endpoint bookkeeping)."""
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [6, 6, 6])
    with pytest.raises(ValueError, match="no feature edges"):
        EmbeddedFeature(msh, lambda x: np.full(x.shape[1], False), tangent=(1.0, 0.0, 0.0),
                        K_feat=2.0, area=1.5, porosity=0.3)


def test_rejects_degree_above_one():
    """The vertex dP diagonal allocation only covers P1 vertex dofs; degree>1 must be rejected."""
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [6, 6, 6])
    with pytest.raises(ValueError):
        _box_feature(n=6, K_feat=2.0, area=1.5, porosity=0.3, degree=2)


def test_rejects_boundary_feature():
    """A feature edge ON the domain boundary is silently mis-measured by the interior ridge -> reject."""
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [6, 6, 6])
    with pytest.raises((ValueError, NotImplementedError)):
        EmbeddedFeature(msh, lambda x: np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0),
                        tangent=(1.0, 0.0, 0.0), K_feat=2.0, area=1.5, porosity=0.3)


def test_nonunit_tangent_gives_same_current_as_unit():
    """A non-unit tangent must NOT scale the conveyance current by |t̂|² -- the tangent is normalized."""
    K_feat, A = 2.0, 1.5
    msh, feat = _box_feature(n=6, K_feat=K_feat, area=A, porosity=0.3, tangent=(2.0, 0.0, 0.0))
    v = ufl.TestFunction(feat.V)
    bcs = [feat.pin_bc(),
           fem.dirichletbc(PETSc.ScalarType(1.0), _end_dofs(feat, 0.0), feat.V),
           fem.dirichletbc(PETSc.ScalarType(0.0), _end_dofs(feat, 1.0), feat.V)]
    assert _solve(feat, feat.hf_residual(v), bcs)
    assert abs(abs(feat.conveyance_current()) - K_feat * A * 1.0 / 1.0) < 1e-10, "non-unit tangent scaled the current"
