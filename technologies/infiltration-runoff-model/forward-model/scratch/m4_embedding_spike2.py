"""Module 4 (§E) embedding spike v2 -- the follow-up Codex demanded before trusting path B.

(1) Kill A's LAST stock route: exchange on the 1-D submesh `dx_f` with the host psi restricted via
    entity_maps (the only codim-2 submesh path not yet tested; expected to trip the codim assert).
(2) Validate B's CONVEYANCE with the TANGENTIAL projection (the full grad-dot-grad spike number was
    meaningless): manufactured H_f = b*s on a straight axis-aligned feature -> the tangential bilinear
    energy must equal the exact 1-D value K*A*b^2*L, while the FULL grad form must DIFFER (negative test).
(3) Quick checks of the Medium concerns: storage couples only Gamma dofs; sign-paired exchange is
    structurally conservative.

Run from forward-model/ with PYTHONPATH=. and *_NUM_THREADS=1.
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import dolfinx.fem.petsc as fp

COMM = MPI.COMM_WORLD
N = 6
host = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [N, N, N])
tdim = host.topology.dim
for d in (0, 1):
    host.topology.create_connectivity(d, tdim)
    host.topology.create_connectivity(tdim, d)
on_feat = lambda x: np.isclose(x[1], 0.5) & np.isclose(x[2], 0.5)
feat_edges = np.sort(dmesh.locate_entities(host, 1, on_feat)).astype(np.int32)
print(f"feature Γ: {feat_edges.size} interior edges; L=1.0")
print("=" * 80)


# ===================================================================== (1) A's last route: exchange on dx_f
print("A5: exchange on the 1-D submesh dx_f + host psi restricted via entity_maps")
try:
    submesh, s2p, _v, _g = dmesh.create_submesh(host, 1, feat_edges)
    Vs = fem.functionspace(host, ("Lagrange", 1))
    Vf = fem.functionspace(submesh, ("Lagrange", 1))
    psi = fem.Function(Vs); Hf = fem.Function(Vf)
    vf = ufl.TestFunction(Vf)
    dxf = ufl.Measure("dx", domain=submesh)
    Rf = (Hf - psi) * vf * dxf            # form on the SUBMESH; host psi appears (codim 1-3 = -2)
    try:
        f = fem.form(Rf, entity_maps=[s2p])
        b = fp.assemble_vector(f); b.assemble()
        print(f"  [OK]   A5 dx_f + entity_maps WORKS (|r|={b.norm():.3e}) -> A NOT dead after all!")
    except Exception as e:
        print(f"  [FAIL] A5 dx_f + entity_maps: {type(e).__name__}: {str(e)[:150]}")
except Exception as e:
    print(f"  A5 setup failed: {type(e).__name__}: {str(e)[:150]}")
print("=" * 80)


# ===================================================================== (2) B conveyance: tangential vs full
print("B-conv: tangential projection ∇H_f·t̂ vs the (wrong) full ∇·∇")
Vh = fem.functionspace(host, ("Lagrange", 1))
Hf = fem.Function(Vh); v = ufl.TestFunction(Vh); dH = ufl.TrialFunction(Vh)
ft = dmesh.meshtags(host, 1, feat_edges, np.ones(feat_edges.size, dtype=np.int32))
dG = ufl.Measure("ridge", domain=host, subdomain_data=ft)(1)
t_hat = ufl.as_vector([1.0, 0.0, 0.0])      # feature tangent (axis-aligned along x)
K, Aa, bcoef = 2.0, 1.5, 0.7

# manufactured H_f = b*x on Γ, pinned 0 off Γ
coords = Vh.tabulate_dof_coordinates()
gdofs = fem.locate_dofs_geometrical(Vh, on_feat)
Hf.x.array[:] = 0.0
Hf.x.array[gdofs] = bcoef * coords[gdofs, 0]
Hf.x.scatter_forward()


def _energy(form_ufl):
    M = fp.assemble_matrix(fem.form(form_ufl)); M.assemble()
    x = M.createVecRight(); x.array[:] = Hf.x.array
    y = M.createVecLeft(); M.mult(x, y)
    return float(x.dot(y))


e_tang = _energy(K * Aa * ufl.dot(ufl.grad(dH), t_hat) * ufl.dot(ufl.grad(v), t_hat) * dG)
e_full = _energy(K * Aa * ufl.dot(ufl.grad(dH), ufl.grad(v)) * dG)
expected = K * Aa * bcoef ** 2 * 1.0        # ∫_Γ K·A·(∂H_f/∂s)^2 ds = K·A·b^2·L
print(f"  expected exact 1-D conveyance energy K·A·b²·L = {expected:.6f}")
print(f"  TANGENTIAL ∇·t̂ energy = {e_tang:.6f}   rel-err {abs(e_tang-expected)/expected:.2e}  "
      f"{'[OK correct]' if abs(e_tang-expected)/expected < 1e-10 else '[WRONG]'}")
print(f"  FULL ∇·∇ energy      = {e_full:.6f}   ({'DIFFERS (full grad is wrong, as expected)' if abs(e_full-expected)/expected > 1e-3 else 'matches?!'})")
print("=" * 80)


# ===================================================================== (3) storage + conservation quick checks
print("storage + conservation quick checks")
phi = 0.3
# storage couples ONLY Γ dofs: M_stor rows for off-Γ dofs must be all-zero.
Mst = fp.assemble_matrix(fem.form(phi * Aa * dH * v * dG)); Mst.assemble()
ai, aj, av = Mst.getValuesCSR()
nz_rows = np.where(np.diff(ai) > 0)[0]
gset = set(int(g) for g in gdofs)
storage_only_gamma = set(int(r) for r in nz_rows).issubset(gset)
print(f"  storage matrix nonzero rows ⊆ Γ dofs: {storage_only_gamma}  "
      f"({len(nz_rows)} nz rows vs {len(gdofs)} Γ dofs)")

# sign-paired exchange structurally conservative: <σ(H_f−ψ),1>_feature + <−σ(H_f−ψ),1>_host = 0
psi_b = fem.Function(Vh); psi_b.x.array[:] = -0.2; psi_b.x.scatter_forward()
q_feat = COMM.allreduce(fem.assemble_scalar(fem.form(1.0 * (Hf - psi_b) * dG)), op=MPI.SUM)
q_host = COMM.allreduce(fem.assemble_scalar(fem.form(-1.0 * (Hf - psi_b) * dG)), op=MPI.SUM)
print(f"  exchange flux into feature {q_feat:+.6e} + into host {q_host:+.6e} = {q_feat+q_host:+.1e} "
      f"({'conservative' if abs(q_feat+q_host) < 1e-14 else 'LEAK'})")
print("=" * 80)
print("SPIKE2 DONE")
