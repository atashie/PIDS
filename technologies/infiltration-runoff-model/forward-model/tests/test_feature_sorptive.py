"""Module 4 (§E): the sorptive Kirchhoff wall-exchange FORMS on EmbeddedFeature.

The exchange is q = Omega*[Phi(H_f) - Phi(psi)] per wall area (the Kirchhoff form), with ``Omega``
a lagged coefficient set by an external driver -- the validated production driver is
``wi_exchange.WellIndexExchange`` (tests/test_wi_exchange.py). These tests pin the FORM layer:
the direction/sign of the flux, structural (sign-paired) conservation on the same dGamma, and
total-water conservation through a monolithic blocked-Newton step.

HISTORY: the Phase-3 EXPERIMENTAL dual-scale machinery (seed_clock/advance_clock/update_well_index
+ the shell read, storage gate and I_disp/I_drain accumulators) was EXCISED 2026-06-12 after the
Phase-4 adversarial review (it measured 13-39% against the discriminating gate vs the production
scheme's 2.2-5.7%; record validation/sanity/m4_phase4_coupled_review__2026-06-11.md). Its frozen
failure-baseline reimplementation lives in scratch/m4_phase4_embedded_harness.py (DualScaleScheme);
the clock-reduction tests went with the machinery (the offline closures stay pinned by
tests/test_sorptive_closure_gate.py).
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
import dolfinx.fem.petsc as fp
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.feature import EmbeddedFeature

COMM = MPI.COMM_WORLD
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
_ON_FEAT = lambda x: np.isclose(x[1], 0.5) & np.isclose(x[2], 0.5)
_LU = {"snes_type": "newtonls", "snes_linesearch_type": "basic",
       "snes_rtol": 1e-12, "snes_atol": 1e-14, "snes_max_it": 25, "ksp_type": "preonly", "pc_type": "lu"}


def _box_feature(n=6, **params):
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [n, n, n])
    feat = EmbeddedFeature(msh, _ON_FEAT, tangent=(1.0, 0.0, 0.0), **params)
    return msh, feat


def _const_psi(feat, val):
    psi = fem.Function(feat.V)
    psi.x.array[:] = val
    psi.x.scatter_forward()
    return psi


def _set_omega(feat, val):
    feat.Omega.x.array[:] = 0.0
    feat.Omega.x.array[feat._gamma_dofs] = val
    feat.Omega.x.scatter_forward()


def test_direction_and_flux_sign():
    """The bidirectional Kirchhoff flux q = Omega*[Phi(H_f)-Phi(psi)]: feature WETTER (H_f>psi) ->
    disperse (q>0, feature loses to soil); feature DRIER (H_f<psi) -> drain (q<0, feature gains).
    Omega is direction-agnostic (a positive lagged coefficient); the sign lives in the Kirchhoff
    difference."""
    msh, feat = _box_feature(K_feat=1.0, area=1e-3, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0)
    _set_omega(feat, 0.05)
    # disperse: H_f=0 > psi=-1
    psi = _const_psi(feat, -1.0)
    feat.Hf.x.array[:] = 0.0; feat.Hf.x.scatter_forward()
    assert feat.host_sorptive_flux(psi) > 0.0, "feature wetter than soil should disperse (q>0 into host)"
    # drain: H_f=-1 < psi=0
    psi2 = _const_psi(feat, 0.0)
    feat.Hf.x.array[:] = -1.0; feat.Hf.x.scatter_forward()
    assert feat.host_sorptive_flux(psi2) < 0.0, "feature drier than soil should drain (q<0, feature gains)"


def test_sorptive_exchange_is_structurally_conservative():
    """The sorptive exchange is sign-paired on the SAME dGamma: what leaves the host enters the feature
    to machine precision (the Kirchhoff term, like the Phase-2 constant-sigma leg)."""
    msh, feat = _box_feature(K_feat=1.0, area=1e-3, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0)
    _set_omega(feat, 0.05)
    psi = _const_psi(feat, -0.3)
    feat.Hf.x.array[:] = 0.0; feat.Hf.x.scatter_forward()
    vf, wh = ufl.TestFunction(feat.V), ufl.TestFunction(feat.V)
    Rf = fp.assemble_vector(fem.form(feat.sorptive_into_feature(vf, psi))); Rf.assemble()
    Rh = fp.assemble_vector(fem.form(feat.sorptive_into_host(wh, psi))); Rh.assemble()
    assert abs(float(Rf.array.sum()) + float(Rh.array.sum())) < 1e-14, "sorptive exchange not sign-paired"


def test_sorptive_blocked_newton_conserves_total_water():
    """A monolithic [psi, H_f] backward-Euler step with the sorptive exchange (closed, no external flux)
    conserves total water C_h*int psi dx + int_Gamma phi*A*H_f dGamma to the solver gate, with the lagged
    Omega frozen in the residual (the real coupled code path + structural conservation)."""
    msh, feat = _box_feature(K_feat=2.0, area=1.5, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0)
    _set_omega(feat, 0.05)                                  # frozen lagged coefficient for the step
    Vpsi = fem.functionspace(msh, ("Lagrange", 1))          # SEPARATE psi space (distinct dofmap from feat.V)
    psi, psi_n = fem.Function(Vpsi), fem.Function(Vpsi)
    psi.x.array[:] = -0.5; psi_n.x.array[:] = -0.5
    feat.Hf.x.array[:] = 0.8; feat.Hf_n.x.array[:] = 0.8     # feature wetter -> disperses
    C_h = 0.4
    dt = fem.Constant(msh, PETSc.ScalarType(1e-2))
    w, v = ufl.TestFunction(Vpsi), ufl.TestFunction(feat.V)
    F_psi = C_h * (psi - psi_n) / dt * w * ufl.dx + feat.sorptive_into_host(w, psi)
    F_hf = (feat.storage_form(v, dt) + feat.conveyance_form(v)
            + feat.sorptive_into_feature(v, psi) + feat.Hf * v * feat._dPoff)

    def total():
        host = msh.comm.allreduce(fem.assemble_scalar(fem.form(C_h * psi * ufl.dx)), op=MPI.SUM)
        return host + feat.stored_water()

    w0 = total()
    prob = NonlinearProblem([F_psi, F_hf], [psi, feat.Hf], bcs=[feat.pin_bc()],
                            petsc_options_prefix="sblk_",
                            petsc_options={**_LU, "pc_factor_mat_solver_type": "mumps"}, kind="mpi")
    prob.solve()
    assert prob.solver.getConvergedReason() > 0, "blocked Newton (sorptive) did not converge"
    assert abs(feat.host_sorptive_flux(psi)) > 1e-9, "no exchange happened (test is vacuous)"
    assert abs(total() - w0) / abs(w0) < 1e-9, f"total water not conserved: {total():.6e} vs {w0:.6e}"
