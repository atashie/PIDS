"""Module 3 lateral-OUTFLOW mechanism spike (realization A, co-located).

In realization A the overland PDE lives on the host top facets (ds_top, codim-1). The OUTFLOW
boundary -- the downstream end of the surface -- is therefore the BOUNDARY OF ds_top, a codim-2
host entity (a corner vertex in this 2-D cross-section; a perimeter curve in 3-D). Module 2 had it
easy: its mesh WAS the surface, so the outlet was an FFCX-native `ds` (vertices of a 1-D mesh /
perimeter edges of a 2-D mesh). Here we must find a mechanism that (a) compiles on stock FFCX 0.10
(NOT the mixed-dim codim-0 path that crashed realization S), (b) assembles the correct Manning
normal-depth point discharge q_out = SECONDS_PER_DAY*(1/n)*d^{5/3}*sqrt(S0), and (c) yields a finite
auto-Jacobian dq_out/dd.

Candidates, in order of physical fidelity:
  A. POINT/VERTEX measure (dP) at the downstream-top corner -- exact point discharge if supported.
  B. NORMALIZED DOWNSTREAM-BAND sink on ds_top: q_out*(1/w)*chi_band*v*ds_top -> point discharge as
     band width w->0. FFCX-native (a plain ds_top integral with a spatial mask); the safe fallback.

Run:
  source /root/miniforge3/etc/profile.d/conda.sh && conda activate pids-fem
  python scratch/m3_outflow_spike.py
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix

COMM = MPI.COMM_WORLD
L, H, nx, nz = 4.0, 1.0, 16, 6
n_man, S0 = 0.05, 0.05
SECONDS_PER_DAY = 86400.0

# host cross-section; surface = top edge z=H; outlet = downstream-top corner (x=L, z=H)
host = dmesh.create_rectangle(COMM, [[0.0, 0.0], [L, H]], [nx, nz])
fdim = host.topology.dim - 1
host.topology.create_connectivity(fdim, host.topology.dim)
Vd = fem.functionspace(host, ("Lagrange", 1))
d = fem.Function(Vd, name="d")
d_const = 0.06                     # uniform depth so the expected point discharge is exact + simple
d.interpolate(lambda x: d_const + 0.0 * x[0])
v = ufl.TestFunction(Vd)

# q_out(d) Manning normal-depth discharge [m^2/day per unit width]
d_pos = ufl.max_value(d, 0.0)
q_out = SECONDS_PER_DAY * (1.0 / n_man) * d_pos ** (5.0 / 3.0) * np.sqrt(S0)
q_expected = SECONDS_PER_DAY * (1.0 / n_man) * d_const ** (5.0 / 3.0) * np.sqrt(S0)
print(f"expected point discharge q_out(d={d_const}) = {q_expected:.6e} m^2/day")

# locate the outlet corner DOF (for checking which row should carry the contribution)
corner = fem.locate_dofs_geometrical(Vd, lambda x: np.isclose(x[0], L) & np.isclose(x[1], H))
print(f"outlet corner dof(s): {corner}  coord(s): {Vd.tabulate_dof_coordinates()[corner]}")
print("=" * 78)

# ----------------------------------------------------------------------------------------------
# CANDIDATE A: point/vertex integral (dP) at the corner vertex
# ----------------------------------------------------------------------------------------------
print("CANDIDATE A: point/vertex measure (dP) at the downstream-top corner")
for mtype in ("vertex", "point"):
    try:
        cverts = dmesh.locate_entities_boundary(
            host, 0, lambda x: np.isclose(x[0], L) & np.isclose(x[1], H))
        vt = dmesh.meshtags(host, 0, np.sort(cverts), np.ones(cverts.size, dtype=np.int32))
        dP = ufl.Measure(mtype, domain=host, subdomain_data=vt)(1)
        F = q_out * v * dP
        bvec = assemble_vector(fem.form(F)); bvec.assemble()
        arr = bvec.getArray()
        nz_idx = np.nonzero(np.abs(arr) > 1e-30)[0]
        print(f"  [{mtype}] residual assembled. nonzero rows={nz_idx.tolist()} "
              f"value(s)={arr[nz_idx]}  (expect ~{q_expected:.4e} at row {corner})")
        # auto-Jacobian
        J = ufl.derivative(F, d)
        Jm = assemble_matrix(fem.form(J)); Jm.assemble()
        print(f"  [{mtype}] Jacobian assembled OK, norm={Jm.norm():.4e}")
        print(f"  [{mtype}] ====> WORKS")
    except Exception as e:
        print(f"  [{mtype}] FAILED: {type(e).__name__}: {str(e)[:160]}")
print("=" * 78)

# ----------------------------------------------------------------------------------------------
# CANDIDATE B: normalized downstream-band sink on ds_top  (FFCX-native fallback)
# ----------------------------------------------------------------------------------------------
print("CANDIDATE B: normalized downstream-band sink on ds_top  q_out*(1/w)*chi_band*v*ds_top")
top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[1], H)))
mt = dmesh.meshtags(host, fdim, top, np.ones(top.size, dtype=np.int32))
ds_top = ufl.Measure("ds", domain=host, subdomain_data=mt)(1)
xufl = ufl.SpatialCoordinate(host)
hx = L / nx  # top-cell width
try:
    for w in (1.0 * hx, 2.0 * hx, 4.0 * hx):
        # smooth band indicator concentrated within distance w of the outlet x=L, normalized so
        # integral over the top edge ~ 1 (=> the term ~ q_out at the outlet, a point discharge).
        chi = ufl.conditional(ufl.gt(xufl[0], L - w), 1.0, 0.0)
        norm = COMM.allreduce(fem.assemble_scalar(fem.form(chi * ds_top)), op=MPI.SUM)  # ~ w
        Fb = q_out * (chi / norm) * v * ds_top
        bvec = assemble_vector(fem.form(Fb)); bvec.assemble()
        # integrated discharge that leaves through the band (should ~ q_expected, independent of w)
        Qband = COMM.allreduce(fem.assemble_scalar(fem.form(q_out * (chi / norm) * ds_top)), op=MPI.SUM)
        arr = bvec.getArray()
        nz_idx = np.nonzero(np.abs(arr) > 1e-30)[0]
        # auto-Jacobian
        J = ufl.derivative(Fb, d)
        Jm = assemble_matrix(fem.form(J)); Jm.assemble()
        print(f"  w={w:.3f} ({w/hx:.0f} cells): band_len={norm:.4f}  integrated Q={Qband:.4e} "
              f"(expect {q_expected:.4e}, rel {abs(Qband-q_expected)/q_expected:.2e})  "
              f"#nz_rows={nz_idx.size}  Jnorm={Jm.norm():.3e}")
    print("  ====> WORKS (FFCX-native); integrated discharge ~ point value, w-independent")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
print("=" * 78)
print("SPIKE DONE")
