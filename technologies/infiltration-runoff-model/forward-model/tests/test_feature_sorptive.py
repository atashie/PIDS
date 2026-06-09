"""Module 4 (§E) Phase-3: the sorptive-exchange CLOSURE wired into EmbeddedFeature, built test-first.

The constant-sigma wall exchange (Phase 2) is upgraded to the gate-validated sorptivity CLOCK: the flux per
wall area is q = Omega(I)*[Phi(H_f) - Phi(psi_cell)] (the Kirchhoff form), with a per-face cumulative-uptake
state I advanced each step. Per Arik (2026-06-09): SEPARATE disperse/drain accumulators (no reset on
reversal). These tests pin the machinery in isolation -- that the embedded advance REDUCES TO the validated
clock under a fixed far field, that the exchange stays structurally conservative, the direction switch picks
the right branch, and the two accumulators are independent -- before the coupled FEM acceptance test.
"""
import numpy as np
import ufl
import pytest
from pathlib import Path
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
import dolfinx.fem.petsc as fp
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.feature import EmbeddedFeature
from pids_forward.physics.sorptive_closure import (
    F_cylindrical, F_throttle, throttle_params, parlange_sorptivity, sorptive_clock, rel_l2, R_W_DEFAULT,
)

DATA = Path(__file__).parent / "data"
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


def _gamma_mean(func, feat):
    return float(func.x.array[feat._gamma_dofs].mean())


def test_disperse_state_advance_reduces_to_clock():
    """With the soil far field held at psi_i and the wall saturated (H_f=0) -- the gate's disperse
    reference scenario -- the per-face clock advance must reproduce the validated sorptive_clock I(t)
    (cylindrical Green-Ampt + a-priori Parlange S) on the reference time grid."""
    msh, feat = _box_feature(K_feat=1.0, area=1e-3, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0)
    t = np.load(DATA / "m4_phase1b_disperse_refs.npz")["LOAM_t"]
    feat.seed_clock(t[0])
    psi = _const_psi(feat, -1.0)
    feat.Hf.x.array[:] = 0.0
    feat.Hf.x.scatter_forward()
    I = [_gamma_mean(feat.I_disp, feat)]
    for i in range(1, t.size):
        feat.advance_clock(psi, t[i] - t[i - 1])
        I.append(_gamma_mean(feat.I_disp, feat))
    I_clock = sorptive_clock(t, feat.S_disp, feat.dth_disp, R_W_DEFAULT, F_cylindrical)
    assert rel_l2(np.array(I), I_clock) < 1e-6, "disperse advance did not reduce to the validated clock"
    assert abs(_gamma_mean(feat.I_drain, feat) - feat.S_drain * np.sqrt(t[0])) < 1e-15, \
        "drain accumulator moved during a disperse-only run (should stay at its seed)"


def test_drain_state_advance_reduces_to_throttle_clock():
    """With the soil saturated (psi=0) and the wall low (H_f=-1) -- the drain reference scenario -- the
    per-face advance reproduces the validated sub-sqrt-t throttle clock with the feature's own (semi-
    empirical) desorptivity. Self-consistency of the drain machinery (the closure FORM, not a-priori)."""
    msh, feat = _box_feature(K_feat=1.0, area=1e-3, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0)
    t = np.load(DATA / "m4_phase1c_drain_refs.npz")["LOAM_t"]
    feat.seed_clock(t[0])
    psi = _const_psi(feat, 0.0)
    feat.Hf.x.array[:] = -1.0
    feat.Hf.x.scatter_forward()
    I = [_gamma_mean(feat.I_drain, feat)]
    for i in range(1, t.size):
        feat.advance_clock(psi, t[i] - t[i - 1])
        I.append(_gamma_mean(feat.I_drain, feat))
    z0, k = throttle_params(feat.dth_drain)
    I_clock = sorptive_clock(t, feat.S_drain, feat.dth_drain, R_W_DEFAULT, lambda z: F_throttle(z, z0, k))
    assert rel_l2(np.array(I), I_clock) < 1e-6, "drain advance did not reduce to the throttle clock"
    assert abs(_gamma_mean(feat.I_disp, feat) - feat.S_disp * np.sqrt(t[0])) < 1e-15, \
        "disperse accumulator moved during a drain-only run (should stay at its seed)"


def test_direction_switch_and_flux_sign():
    """The bidirectional Kirchhoff flux q = Omega*[Phi(H_f)-Phi(psi)]: feature WETTER (H_f>psi) -> disperse
    (q>0, feature loses to soil); feature DRIER (H_f<psi) -> drain (q<0, feature gains). The well-index uses
    the matching branch (disperse cyl / drain throttle)."""
    msh, feat = _box_feature(K_feat=1.0, area=1e-3, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0, r_eq=R_W_DEFAULT)   # r_eq=r_w -> I_fill=0 -> gate open
    feat.seed_clock(1e-3)
    # disperse: H_f=0 > psi=-1
    psi = _const_psi(feat, -1.0)
    feat.Hf.x.array[:] = 0.0; feat.Hf.x.scatter_forward()
    feat.update_well_index(psi)
    q_disp = feat.host_sorptive_flux(psi)
    assert q_disp > 0.0, "feature wetter than soil should disperse (q>0 into host)"
    # drain: H_f=-1 < psi=0
    psi2 = _const_psi(feat, 0.0)
    feat.Hf.x.array[:] = -1.0; feat.Hf.x.scatter_forward()
    feat.update_well_index(psi2)
    q_drain = feat.host_sorptive_flux(psi2)
    assert q_drain < 0.0, "feature drier than soil should drain (q<0, feature gains)"


def test_sorptive_exchange_is_structurally_conservative():
    """The sorptive exchange is sign-paired on the SAME dGamma: what leaves the host enters the feature to
    machine precision (the Kirchhoff term, like the Phase-2 constant-sigma leg)."""
    msh, feat = _box_feature(K_feat=1.0, area=1e-3, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0, r_eq=R_W_DEFAULT)   # gate open (I_fill=0)
    feat.seed_clock(1e-3)
    psi = _const_psi(feat, -0.3)
    feat.Hf.x.array[:] = 0.0; feat.Hf.x.scatter_forward()
    feat.update_well_index(psi)
    vf, wh = ufl.TestFunction(feat.V), ufl.TestFunction(feat.V)
    Rf = fp.assemble_vector(fem.form(feat.sorptive_into_feature(vf, psi))); Rf.assemble()
    Rh = fp.assemble_vector(fem.form(feat.sorptive_into_host(wh, psi))); Rh.assemble()
    assert abs(float(Rf.array.sum()) + float(Rh.array.sum())) < 1e-14, "sorptive exchange not sign-paired"


def test_host_source_is_storage_gated():
    """Dual-scale: the host source is GATED OFF until the sub-grid annulus [r_w, r_eq] fills (I >= I_fill).
    Below I_fill the near-wall uptake is held in the I-clock reservoir (host flux 0); above it the host
    receives the through-flow (flux > 0). The default r_eq gives a finite I_fill on a coarse host."""
    msh, feat = _box_feature(K_feat=1.0, area=1e-3, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0)                # default r_eq -> finite I_fill
    assert feat._I_fill > 0.0, "default r_eq should give a finite sub-grid capacity on this coarse mesh"
    feat.seed_clock(1e-3)
    psi = _const_psi(feat, -1.0)
    feat.Hf.x.array[:] = 0.0; feat.Hf.x.scatter_forward()
    feat.update_well_index(psi)                              # I_disp(seed) << I_fill -> gated off
    assert feat.host_sorptive_flux(psi) == 0.0, "host source not gated off during the sub-grid fill"
    feat.I_disp.x.array[feat._gamma_dofs] = 1.5 * feat._I_fill
    feat.I_disp.x.scatter_forward()
    feat.update_well_index(psi)                              # now past I_fill -> gate opens
    assert feat.host_sorptive_flux(psi) > 0.0, "host source did not open after the sub-grid filled"


def test_separate_accumulators_persist_across_reversal():
    """SEPARATE disperse/drain accumulators (Arik 2026-06-09): a face that disperses then reverses to drain
    keeps its disperse history -- I_disp is NOT reset, and the subsequent drain grows I_drain independently."""
    msh, feat = _box_feature(K_feat=1.0, area=1e-3, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0)
    feat.seed_clock(1e-3)
    # phase 1: disperse
    psi = _const_psi(feat, -1.0)
    feat.Hf.x.array[:] = 0.0; feat.Hf.x.scatter_forward()
    feat.advance_clock(psi, 1e-2)
    I_disp_after = _gamma_mean(feat.I_disp, feat)
    assert I_disp_after > feat.S_disp * np.sqrt(1e-3), "disperse did not accumulate"
    # phase 2: reverse to drain
    psi2 = _const_psi(feat, 0.0)
    feat.Hf.x.array[:] = -1.0; feat.Hf.x.scatter_forward()
    feat.advance_clock(psi2, 1e-2)
    assert abs(_gamma_mean(feat.I_disp, feat) - I_disp_after) < 1e-15, "I_disp was disturbed by drain (should persist)"
    assert _gamma_mean(feat.I_drain, feat) > feat.S_drain * np.sqrt(1e-3), "drain did not accumulate after reversal"


def test_sorptive_blocked_newton_conserves_total_water():
    """A monolithic [psi, H_f] backward-Euler step with the sorptive exchange (closed, no external flux)
    conserves total water C_h*int psi dx + int_Gamma phi*A*H_f dGamma to the solver gate, with the lagged
    well-index in the residual (the real coupled code path + structural conservation)."""
    msh, feat = _box_feature(K_feat=2.0, area=1.5, porosity=0.3)
    feat.configure_sorptive(LOAM, psi_i=-1.0, r_eq=R_W_DEFAULT)   # gate open (I_fill=0)
    feat.seed_clock(1e-3)
    Vpsi = fem.functionspace(msh, ("Lagrange", 1))          # SEPARATE psi space (distinct dofmap from feat.V)
    psi, psi_n = fem.Function(Vpsi), fem.Function(Vpsi)
    psi.x.array[:] = -0.5; psi_n.x.array[:] = -0.5
    feat.Hf.x.array[:] = 0.8; feat.Hf_n.x.array[:] = 0.8     # feature wetter -> disperses
    feat.update_well_index(psi)                              # lag the well-index at the step start
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
