"""Module 4 (§E) Phase-0 spike: is the 1-D-in-3-D feature embedding FFCX-feasible on stock 0.10?

A PIDS feature = a 1-D vector Γ embedded in the 3-D Richards host, with its OWN dof H_f, contributing
  conveyance  ∫_Γ K·A·∇ₛH_f·∇ₛv_f ds      (1-D Darcy along the centerline; ∇ₛ = tangential)
  exchange    Σ_face ∫_Γ σ·(H_f − ψ)·v ds  (sign-paired: into the feature, out of the host)
  storage     ∫_Γ φ·A·∂H_f/∂t·v_f ds
The coupling is MIXED-DIMENSIONAL (codim-2 line Γ ↔ 3-D host) — structurally like realization S,
which is BLOCKED by the FFCX 0.10 codim-0 codegen bug (a submesh self-Jacobian's missing `else`).
§E is WORSE than S: codim-2 (edges) not codim-1 (facets), and the conveyance self-term has a GRADIENT
(exactly the submesh self-Jacobian that broke S's overland operator). This spike DECIDES the
representation by testing what compiles/assembles on stock FFCX:
  A. design-intended 1-D SUBMESH + entity_maps (conveyance self-term on the submesh dx_feat; exchange
     on the PARENT ridge measure -- probe4 idiom but codim-2).
  B. single-mesh CO-LOCATED H_f on the host + an INTERIOR-edge `ridge` (dr) integral (the M3-ridge
     precedent, but INTERIOR not boundary; and can we get ∇ₛ for the conveyance?).

Run from forward-model/ with PYTHONPATH=. and *_NUM_THREADS=1.
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import dolfinx.fem.petsc as fp

COMM = MPI.COMM_WORLD


def _try(label, fn):
    try:
        detail = fn() or ""
        print(f"  [OK]   {label}{('  -> ' + detail) if detail else ''}", flush=True)
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {type(e).__name__}: {str(e)[:170]}", flush=True)
        return False


def _vec(form_ufl, em=None):
    b = fp.assemble_vector(fem.form(form_ufl, entity_maps=em) if em else fem.form(form_ufl))
    b.assemble()
    return f"|r|={b.norm():.3e}"


def _mat(form_ufl, em=None):
    M = fp.assemble_matrix(fem.form(form_ufl, entity_maps=em) if em else fem.form(form_ufl))
    M.assemble()
    return f"|J|={M.norm():.3e}"


N = 6
host = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [N, N, N])
tdim = host.topology.dim          # 3
host.topology.create_connectivity(1, tdim)
host.topology.create_connectivity(tdim, 1)
host.topology.create_connectivity(1, 0)

# feature Γ = the horizontal centerline edges along x at y=0.5, z=0.5 (INTERIOR, mesh-aligned).
on_feat = lambda x: np.isclose(x[1], 0.5) & np.isclose(x[2], 0.5)
feat_edges = np.sort(dmesh.locate_entities(host, 1, on_feat)).astype(np.int32)
print(f"feature Γ: {feat_edges.size} interior edges (expect {N} along x at y=z=0.5)")
print("=" * 80)

# ============================================================================ TEST A: submesh + entity_maps
print("TEST A: design-intended 1-D SUBMESH + entity_maps (codim-2 cross-mesh)")
try:
    submesh, s2p, _v, _g = dmesh.create_submesh(host, 1, feat_edges)
    print(f"  create_submesh OK: {submesh.topology.index_map(submesh.topology.dim).size_local} 1-D cells")
    Vs = fem.functionspace(host, ("Lagrange", 1))      # ψ (soil)
    Vf = fem.functionspace(submesh, ("Lagrange", 1))   # H_f (feature)
    psi = fem.Function(Vs); Hf = fem.Function(Vf)
    vs = ufl.TestFunction(Vs); vf = ufl.TestFunction(Vf)
    dpsi = ufl.TrialFunction(Vs); dHf = ufl.TrialFunction(Vf)
    dxf = ufl.Measure("dx", domain=submesh)            # 1-D submesh cell measure (∇ on it = ∇ₛ)
    ft = dmesh.meshtags(host, 1, feat_edges, np.ones(feat_edges.size, dtype=np.int32))
    dG = ufl.Measure("ridge", domain=host, subdomain_data=ft)(1)   # PARENT codim-2 coupling measure
    em = [s2p]

    Rf_conv = ufl.dot(ufl.grad(Hf), ufl.grad(vf)) * dxf            # conveyance self-term (∇ₛ native on 1-D submesh)
    Rf_exch = (Hf - psi) * vf * dG                                  # exchange, cross-mesh on PARENT ridge
    Rs = ufl.dot(ufl.grad(psi), ufl.grad(vs)) * ufl.dx - (Hf - psi) * vs * dG

    _try("A2 conveyance self-Jacobian (submesh dx, ∇) compiles  [the S-bug analog]",
         lambda: (fem.form(ufl.derivative(Rf_conv, Hf, dHf)), "")[1])
    _try("A3 exchange cross-Jacobian ∂/∂ψ (parent ridge, entity_maps) compiles",
         lambda: (fem.form(ufl.derivative(Rf_exch, psi, dpsi), entity_maps=em), "")[1])
    _try("A3 exchange residual assembles (codim-2 cross-mesh)", lambda: _vec(Rf_exch, em))

    def _coupled():
        a_ss = fem.form(ufl.derivative(Rs, psi, dpsi))
        a_sf = fem.form(ufl.derivative(Rs, Hf, dHf), entity_maps=em)
        a_fs = fem.form(ufl.derivative(Rf_exch, psi, dpsi), entity_maps=em)
        a_ff = fem.form(ufl.derivative(Rf_conv + Rf_exch, Hf, dHf), entity_maps=em)
        M = fp.assemble_matrix([[a_ss, a_sf], [a_fs, a_ff]], kind="mpi"); M.assemble()
        return f"|J|={M.norm():.3e}"
    _try("A4 FULL coupled blocked Jacobian assembles  [the S coexistence trigger]", _coupled)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  TEST A setup FAILED: {type(e).__name__}: {str(e)[:200]}")
print("=" * 80)

# ============================================================================ TEST B: co-located interior ridge
print("TEST B: single-mesh CO-LOCATED H_f + INTERIOR-edge ridge integral (M3-ridge precedent)")
Vh = fem.functionspace(host, ("Lagrange", 1))
psi_b = fem.Function(Vh); Hf_b = fem.Function(Vh)
vb = ufl.TestFunction(Vh); dHb = ufl.TrialFunction(Vh)
ftb = dmesh.meshtags(host, 1, feat_edges, np.ones(feat_edges.size, dtype=np.int32))
dGb = ufl.Measure("ridge", domain=host, subdomain_data=ftb)(1)
one = fem.Constant(host, 1.0)
Hf_b.x.array[:] = 0.5; psi_b.x.array[:] = -0.2

_try("B1 interior ridge length ∫_Γ 1 ds (expect ~1.0)",
     lambda: f"length={COMM.allreduce(fem.assemble_scalar(fem.form(one * dGb)), op=MPI.SUM):.4f}")
_try("B2 exchange residual (H_f−ψ)·v ∫_Γ ds assembles", lambda: _vec((Hf_b - psi_b) * vb * dGb))
_try("B2 exchange self-Jacobian ∂/∂H_f compiles+assembles",
     lambda: _mat(ufl.derivative((Hf_b - psi_b) * vb * dGb, Hf_b, dHb)))
_try("B3 conveyance ∫_Γ ∇H_f·∇v ds compiles+assembles (full ∇; tangential projection TBD)",
     lambda: _mat(ufl.derivative(ufl.dot(ufl.grad(Hf_b), ufl.grad(vb)) * dGb, Hf_b, dHb)))
print("=" * 80)
print("SPIKE DONE")
