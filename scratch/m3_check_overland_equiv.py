"""Shore-up #2 (Codex): is the co-located overland flux operator the SAME as Module-2's overland?

Realization A puts the Manning diffusion-wave on the host top facets with a TANGENTIAL gradient
grad_T = grad - (grad.n)n (because d is co-located over the volume, pinned 0 below the top). Codex's
worry: does grad_T on the pinned host field recover the true SURFACE gradient, or is it polluted by
the artificial vertical structure? Direct test: assemble the overland FLUX residual for the SAME
d(x), z_b(x) on (a) the coupled host ds_top with grad_T, and (b) a standalone Module-2 OverlandProblem
on the matching 1-D surface mesh with grad -- compare the assembled flux at the interior surface
nodes. If they match, the operators are equivalent (independent of the rest of the coupling).
"""
import numpy as np, ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector
from pids_forward.physics.overland import overland_conveyance, SECONDS_PER_DAY

COMM = MPI.COMM_WORLD
L, H, nx, nz = 4.0, 1.0, 16, 6
n_man, eps_S = 0.05, 1e-3
# smooth, strictly-positive surface depth + a sloped bed (so grad_T H_s != 0 everywhere).
d_expr = lambda x: 0.06 + 0.02 * np.sin(np.pi * x[0] / L)
zb_expr = lambda x: 0.05 * (L - x[0])

# ---- (a) COUPLED: grad_T overland flux on the 2-D host top facets ----
host = dmesh.create_rectangle(COMM, [[0.0, 0.0], [L, H]], [nx, nz])
fdim = host.topology.dim - 1
host.topology.create_connectivity(fdim, host.topology.dim)
top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[1], H)))
mt = dmesh.meshtags(host, fdim, top, np.ones(top.size, dtype=np.int32))
ds_top = ufl.Measure("ds", domain=host, subdomain_data=mt)(1)
Vd = fem.functionspace(host, ("Lagrange", 1))
dH = fem.Function(Vd); dH.interpolate(d_expr)
zbH = fem.Function(Vd); zbH.interpolate(zb_expr)
vH = ufl.TestFunction(Vd)
nv = ufl.FacetNormal(host)
gT = lambda f: ufl.grad(f) - ufl.dot(ufl.grad(f), nv) * nv
HsH = zbH + dH
gHsH = gT(HsH)
slopeH = (ufl.dot(gHsH, gHsH) + eps_S**2) ** 0.25
KsH = SECONDS_PER_DAY * ufl.max_value(dH, 0.0) ** (5.0 / 3.0) / (n_man * slopeH)
bH = assemble_vector(fem.form(KsH * ufl.dot(gHsH, gT(vH)) * ds_top)); bH.assemble()
xH = Vd.tabulate_dof_coordinates()
topdofs = fem.locate_dofs_geometrical(Vd, lambda x: np.isclose(x[1], H))
xtop = xH[topdofs, 0]; ftop = bH.getArray()[topdofs]

# ---- (b) MODULE 2: grad overland flux on the matching 1-D surface mesh ----
surf = dmesh.create_interval(COMM, nx, [0.0, L])
Vs = fem.functionspace(surf, ("Lagrange", 1))
dS = fem.Function(Vs); dS.interpolate(d_expr)
zbS = fem.Function(Vs); zbS.interpolate(zb_expr)
vS = ufl.TestFunction(Vs)
HsS, KsS = overland_conveyance(dS, zbS, n_man, eps_S)
bS = assemble_vector(fem.form(KsS * ufl.dot(ufl.grad(HsS), ufl.grad(vS)) * ufl.dx)); bS.assemble()
xS = Vs.tabulate_dof_coordinates()[:, 0]; fS = bS.getArray()

# ---- compare at matching interior nodes (exclude the two ends: boundary treatment differs) ----
oH = np.argsort(xtop); xH_s, fH_s = xtop[oH], ftop[oH]
oS = np.argsort(xS); xS_s, fS_s = xS[oS], fS[oS]
assert np.allclose(xH_s, xS_s), "surface x-nodes do not match"
interior = slice(1, -1)
xi = xH_s[interior]; a = fH_s[interior]; b = fS_s[interior]
rel = np.abs(a - b) / (np.abs(b).max() + 1e-30)
print(f"interior nodes: {xi.size}")
print(f"coupled grad_T flux range: [{a.min():+.4e}, {a.max():+.4e}]")
print(f"M2      grad   flux range: [{b.min():+.4e}, {b.max():+.4e}]")
print(f"max |coupled - M2| / max|M2| (interior) = {rel.max():.3e}")
print("OPERATOR MATCH:", "YES (<1e-9)" if rel.max() < 1e-9 else
      ("close (<1e-3)" if rel.max() < 1e-3 else "MISMATCH"))
# show a few node-by-node values
for i in range(0, xi.size, max(1, xi.size // 6)):
    print(f"  x={xi[i]:.3f}  coupled={a[i]:+.5e}  M2={b[i]:+.5e}  reldiff={rel[i]:.2e}")
