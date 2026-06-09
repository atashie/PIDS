"""Module 4 (§E) Phase-3: the COUPLED EMBEDDING-FIDELITY acceptance test (claim C-004, coupled).

The sub-grid sorptive closure must reproduce the RESOLVED near-field reference I(t) when the feature is
embedded in a COARSE host that does NOT resolve the sorption front. This is the make-or-break embedding
claim: a single feature (saturated disperse wall, H_f=0) as a clock-driven source in a coarse-host Richards
solve (LOAM, box, far faces psi_i=-1) must match the resolved tunnel reference.

The NAIVE co-located coupling fails this (~3x under-prediction, worsening with refinement: the wall flux
saturates the near-Gamma cell, ψ(Γ)->0, the drive collapses). The DUAL-SCALE closure (built into
EmbeddedFeature.configure_sorptive/update_well_index/advance_clock) fixes it: the clock reads the off-Γ
SHELL ψ_cell (far field), and the host source is GATED off until the front reaches r_eq -- validated to
≤2% across host resolutions (scratch/_zdualscale_probe.py). This test pins it at one resolution.

Slow-ish (a coupled Richards march); kept small (n=8). Pure production API -- the dual-scale is internal.
"""
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.feature import EmbeddedFeature
from pids_forward.physics.sorptive_closure import rel_l2, R_W_DEFAULT

DATA = Path(__file__).parent / "data"
COMM = MPI.COMM_WORLD
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
R_W = R_W_DEFAULT
_ON_FEAT = lambda x: np.isclose(x[1], 0.5) & np.isclose(x[2], 0.5)
_LU = {"snes_type": "newtonls", "snes_linesearch_type": "cp", "snes_rtol": 1e-9, "snes_atol": 1e-12,
       "snes_max_it": 40, "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def _vertex_dx():
    return ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})


def test_coupled_disperse_embedding_reproduces_reference():
    """A feature embedded in a COARSE host reproduces the resolved disperse reference I(t) to <=5%
    full-curve rel-L2 -- the dual-scale sub-grid closure (off-Γ ψ_cell + storage gate) coupled to a real
    Richards host. (The naive co-located coupling under-predicts ~55-78% here.)"""
    ref = np.load(DATA / "m4_phase1b_disperse_refs.npz")
    T, I_TUN = ref["LOAM_t"], ref["LOAM_tunnel_I"]
    n = 8
    msh = dmesh.create_box(COMM, [[0., 0., 0.], [1., 1., 1.]], [n, n, n])
    feat = EmbeddedFeature(msh, _ON_FEAT, tangent=(1., 0., 0.),
                           K_feat=1.0, area=np.pi * R_W**2, porosity=0.4)
    feat.configure_sorptive(LOAM, psi_i=-1.0)            # auto shell + storage gate (dual-scale)
    feat.seed_clock(T[0])
    V = feat.V
    psi, psi_n = fem.Function(V), fem.Function(V)
    psi.x.array[:] = -1.0; psi_n.x.array[:] = -1.0
    feat.Hf.x.array[:] = 0.0; feat.Hf.x.scatter_forward()  # saturated disperse source (fixed wall)
    w = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    dxs, dxq = _vertex_dx(), ufl.dx(metadata={"quadrature_degree": 8})
    th, thn, K = LOAM.theta_ufl(psi), LOAM.theta_ufl(psi_n), LOAM.K_ufl(psi)
    eg = fem.Constant(msh, np.array([0., 0., 1.], dtype=PETSc.ScalarType))
    F = ((th - thn) / dt_c) * w * dxs + K * ufl.dot(ufl.grad(psi) + eg, ufl.grad(w)) * dxq \
        + feat.sorptive_into_host(w, psi)
    lat = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], 1.)
                                      | np.isclose(x[2], 0.) | np.isclose(x[2], 1.))
    bcs = [fem.dirichletbc(PETSc.ScalarType(-1.0), lat, V)]
    prob = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="embfid_", petsc_options=_LU)

    g = feat._gamma_dofs
    I_emb = [float(feat.I_disp.x.array[g].mean())]
    dt, tprev = 1e-7, T[0]
    for ts in T[1:]:
        t = tprev
        while t < ts - 1e-15:
            h = min(dt, ts - t); dt_c.value = h
            feat.update_well_index(psi)                  # lag the gated well-index
            prob.solve()
            if prob.solver.getConvergedReason() > 0:
                feat.advance_clock(psi, h)               # advance the clock against the far-field ψ_cell
                psi_n.x.array[:] = psi.x.array; psi_n.x.scatter_forward()
                t += h
                it = int(prob.solver.getIterationNumber())
                dt = dt * 1.5 if it <= 4 else (dt * 0.6 if it >= 9 else dt)
            else:
                psi.x.array[:] = psi_n.x.array; psi.x.scatter_forward(); dt *= 0.5
                assert dt > 1e-12, f"dt collapse at t={t:.2e}"
        tprev = ts
        I_emb.append(float(feat.I_disp.x.array[g].mean()))

    err = rel_l2(np.array(I_emb), I_TUN)
    assert err < 0.05, f"coupled embedding rel-L2 {err:.1%} > 5% (reference end {I_TUN[-1]:.3e}, emb {I_emb[-1]:.3e})"
