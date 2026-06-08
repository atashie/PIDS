"""Adversarial audit of the quadrature_degree CAP (=6) across the regimes the review flags.

For each term that the cap touches (Darcy volume across psi regimes; the overland Manning surface
flux at intense ponding with a stiff slope; the NCP Fischer-Burmeister leg; the Kirchhoff
infiltration leg; the drainage GHB), assemble the RESIDUAL VECTOR and the JACOBIAN MATRIX on a real
3-D box at a representative state, then sweep quadrature_degree and report the relative deviation of
the capped value (deg 6) from a high-degree reference (deg 20). A cap is SAFE for a term iff the
deviation is at/near machine precision; a non-trivial deviation means deg 6 under-integrates that
term and shifts the discrete operator (and hence the solution / Newton tangent).

Run:
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate pids-fem
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  PYTHONPATH=. python scratch/m3_quad_cap_audit.py
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.overland import overland_conveyance, SECONDS_PER_DAY
from pids_forward.physics.richards import richards_bulk_residual

COMM = MPI.COMM_WORLD
SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
DEGREES = [2, 4, 6, 8, 10, 14, 20]
REF = 20


def _box(nx=6, ny=6, nz=5, L=4.0, W=2.0, H=1.0):
    return dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [L, W, H]], [nx, ny, nz])


def report(label, vals_by_deg):
    ref = vals_by_deg[REF]
    print(f"\n=== {label} (reference deg {REF}) ===", flush=True)
    for qd in DEGREES:
        v = vals_by_deg[qd]
        rel = abs(v - ref) / (abs(ref) + 1e-300)
        mark = "  <-- CAP" if qd == 6 else ""
        print(f"  deg={qd:>2}: {v: .12e}   rel-dev={rel:.2e}{mark}", flush=True)


def sweep_scalar(label, build_form_fn):
    """build_form_fn(qd) -> a UFL scalar form; report assemble_scalar across degrees."""
    out = {}
    for qd in DEGREES:
        out[qd] = COMM.allreduce(fem.assemble_scalar(fem.form(build_form_fn(qd))), op=MPI.SUM)
    report(label, out)


def sweep_vec(label, build_form_fn):
    """build_form_fn(qd) -> residual UFL form; report its L2 norm across degrees."""
    out = {}
    for qd in DEGREES:
        b = assemble_vector(fem.form(build_form_fn(qd))); b.assemble()
        out[qd] = b.norm()
    report(label, out)


def sweep_mat(label, build_form_fn):
    """build_form_fn(qd) -> bilinear (Jacobian) UFL form; report its norm across degrees."""
    out = {}
    for qd in DEGREES:
        M = assemble_matrix(fem.form(build_form_fn(qd))); M.assemble()
        out[qd] = M.norm()
    report(label, out)


def meta(qd):
    return {"quadrature_degree": qd}


# ---------------------------------------------------------------------------------------------
# 1) DARCY VOLUME (Richards bulk) across psi regimes -- the review's "dry / near-saturation / front"
# ---------------------------------------------------------------------------------------------
print("#" * 90)
print("# DARCY VOLUME Jacobian + residual across pressure-head regimes")
print("#" * 90, flush=True)
msh = _box()
V = fem.functionspace(msh, ("Lagrange", 1))
dt = fem.Constant(msh, 1.0)
e_g = fem.Constant(msh, np.array([0.0, 0.0, 1.0]))
dx_lump = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
v = ufl.TestFunction(V)
coords = V.tabulate_dof_coordinates()


def set_psi(arr_fn):
    p = fem.Function(V); p.x.array[:] = arr_fn(coords); p.x.scatter_forward()
    pn = fem.Function(V); pn.x.array[:] = p.x.array
    return p, pn


regimes = {
    "uniform dry psi=-2.0": lambda c: -2.0 + 0.0 * c[:, 0],
    "moderate psi=-0.5 (perf-probe state)": lambda c: -0.5 + 0.0 * c[:, 0],
    "near-air-entry psi=-0.05 (h_s=-0.02)": lambda c: -0.05 + 0.0 * c[:, 0],
    "straddle-saturation psi in [-0.5, +0.1]": lambda c: -0.5 + 0.6 * c[:, 2] / c[:, 2].max(),
    "sharp wetting FRONT (-3 -> +0.05 in z)": lambda c: -3.0 + 3.05 * (c[:, 2] / c[:, 2].max()) ** 4,
}
for name, fn in regimes.items():
    psi, psi_n = set_psi(fn)
    F = lambda qd: richards_bulk_residual(psi, psi_n, v, SOIL, dt, e_g,
                                          dx=ufl.dx(metadata=meta(qd)), dx_storage=dx_lump)
    sweep_mat(f"DARCY J  [{name}]", lambda qd: ufl.derivative(F(qd), psi))
    sweep_vec(f"DARCY R  [{name}]", lambda qd: F(qd))

# ---------------------------------------------------------------------------------------------
# 2) OVERLAND Manning surface flux on ds_top at intense ponding + stiff slope (the tangential leg)
# ---------------------------------------------------------------------------------------------
print("\n" + "#" * 90)
print("# OVERLAND Manning surface flux on ds_top (intense ponding, varying depth + slope)")
print("#" * 90, flush=True)
ztop = float(coords[:, 2].max())
fdim = msh.topology.dim - 1
msh.topology.create_connectivity(fdim, msh.topology.dim)
topf = np.sort(dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], ztop)))
ft = dmesh.meshtags(msh, fdim, topf, np.ones(topf.size, dtype=np.int32))
Vd = fem.functionspace(msh, ("Lagrange", 1))
vd = ufl.TestFunction(Vd)
zb = fem.Function(Vd)  # flat
n_vec = ufl.FacetNormal(msh)
gT = lambda f: ufl.grad(f) - ufl.dot(ufl.grad(f), n_vec) * n_vec
dc = Vd.tabulate_dof_coordinates()

pond_states = {
    "deep uniform-ish d~0.3 + x-slope": lambda c: 0.3 + 0.2 * c[:, 0] / c[:, 0].max(),
    "intense ponding d~1.0 steep front": lambda c: 0.05 + 1.0 * (c[:, 0] / c[:, 0].max()) ** 3,
    "thin film d~0.01 + slope": lambda c: 0.005 + 0.01 * c[:, 0] / c[:, 0].max(),
}
for name, fn in pond_states.items():
    d = fem.Function(Vd); d.x.array[:] = fn(dc); d.x.scatter_forward()

    def ovl(qd):
        ds_top = ufl.Measure("ds", domain=msh, subdomain_data=ft, metadata=meta(qd))(1)
        H_s, K_s = overland_conveyance(d, zb, 0.05, 1e-3, grad=gT)
        return K_s * ufl.dot(gT(H_s), gT(vd)) * ds_top
    sweep_mat(f"OVERLAND J  [{name}]", lambda qd: ufl.derivative(ovl(qd), d))
    sweep_vec(f"OVERLAND R  [{name}]", lambda qd: ovl(qd))

# ---------------------------------------------------------------------------------------------
# 3) NCP Fischer-Burmeister leg on ds_top (the sqrt; the smoothed complementarity)
#    AND 4) Kirchhoff infiltration leg (the 33-term graded-Simpson sum inside q_pot)
# ---------------------------------------------------------------------------------------------
print("\n" + "#" * 90)
print("# NCP (Fischer-Burmeister) + KIRCHHOFF infiltration leg on ds_top")
print("#" * 90, flush=True)
Vpsi = fem.functionspace(msh, ("Lagrange", 1))
Vlam = fem.functionspace(msh, ("Lagrange", 1))
vlam = ufl.TestFunction(Vlam)
ell_c = 0.1
tau_c = ell_c / SOIL.Ks
eps_ncp = 1e-4


def fb(a, b, eps):
    return a + b - ufl.sqrt(a * a + b * b + 2.0 * eps * eps)


ncp_states = {
    "ponded d=0.05, dry soil psi=-2 (steep Kirchhoff)": (0.05, -2.0, 0.0),
    "near-saturation psi=-0.05, d=0.02": (0.02, -0.05, 0.05),
    "supply-limited d~0, psi=-1, lam=rain": (1e-4, -1.0, 0.1),
}
for name, (dval, pval, lval) in ncp_states.items():
    d = fem.Function(Vd); d.x.array[:] = dval; d.x.scatter_forward()
    psi = fem.Function(Vpsi); psi.x.array[:] = pval; psi.x.scatter_forward()
    lam = fem.Function(Vlam); lam.x.array[:] = lval; lam.x.scatter_forward()
    q_pot = SOIL.kirchhoff_ufl(psi, d) / ell_c
    g = q_pot - lam

    def F_lam(qd):
        ds_top = ufl.Measure("ds", domain=msh, subdomain_data=ft, metadata=meta(qd))(1)
        return fb(d, tau_c * g, eps_ncp) * vlam * ds_top
    sweep_vec(f"NCP R  [{name}]", lambda qd: F_lam(qd))
    sweep_mat(f"NCP J wrt psi (Kirchhoff)  [{name}]", lambda qd: ufl.derivative(F_lam(qd), psi))

    def F_psi_kirch(qd):
        # the lambda-influx leg carries q_pot into psi via the coupled balance; here isolate the
        # Kirchhoff potential surface integral directly
        ds_top = ufl.Measure("ds", domain=msh, subdomain_data=ft, metadata=meta(qd))(1)
        return q_pot * vlam * ds_top
    sweep_scalar(f"KIRCHHOFF int q_pot ds  [{name}]", lambda qd: SOIL.kirchhoff_ufl(psi, d) / ell_c
                 * ufl.Measure("ds", domain=msh, subdomain_data=ft, metadata=meta(qd))(1))

# ---------------------------------------------------------------------------------------------
# 5) DRAINAGE GHB on a 3-D side face (kr-weighted Darcy/head)
# ---------------------------------------------------------------------------------------------
print("\n" + "#" * 90)
print("# DRAINAGE GHB on a side face")
print("#" * 90, flush=True)
sf = np.sort(dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0)))
sft = dmesh.meshtags(msh, fdim, sf, np.ones(sf.size, dtype=np.int32))
psi = fem.Function(Vpsi); psi.x.array[:] = -0.2 + 0.0 * coords[:, 0]; psi.x.scatter_forward()
vpsi = ufl.TestFunction(Vpsi)
z = ufl.SpatialCoordinate(msh)[2]
for name, pval in [("moist psi=-0.2", -0.2), ("near-sat psi=-0.03", -0.03), ("dry psi=-3", -3.0)]:
    psi.x.array[:] = pval; psi.x.scatter_forward()

    def F_drain(qd):
        ds_d = ufl.Measure("ds", domain=msh, subdomain_data=sft, metadata=meta(qd))(1)
        kr = SOIL.K_ufl(psi) / SOIL.Ks
        q_n = 0.5 * kr * (psi + z - (-1.0))
        return q_n * vpsi * ds_d
    sweep_vec(f"DRAINAGE R  [{name}]", lambda qd: F_drain(qd))
    sweep_mat(f"DRAINAGE J  [{name}]", lambda qd: ufl.derivative(F_drain(qd), psi))

print("\nAUDIT DONE", flush=True)
