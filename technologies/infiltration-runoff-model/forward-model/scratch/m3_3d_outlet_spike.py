"""Module 3 3-D lateral-OUTFLOW (codim-2 outlet) mechanism spike -- Phase 0 of the 3-D plan.

In realization A the overland PDE lives on the host TOP FACE (ds_top, codim-1). The OUTFLOW
boundary -- the downstream edge of that surface -- is the BOUNDARY of ds_top, a codim-2 host
entity. In the 2-D cross-section that was a corner VERTEX (handled by a vertex/dP integral,
shipped + validated). In 3-D it becomes the downstream-top EDGE: a codim-2 perimeter curve
(a 1-D *ridge* of the 3-D mesh).

Question this spike answers: does a single-mesh codim-2 RIDGE integral compile + assemble on
stock DOLFINx 0.10 / FFCX (residual + auto-Jacobian), giving the Manning normal-depth LINE
discharge int_edge q_out(d) ds ?  The realization-S FFCX bug was a mixed-dim codim-0 CELL path;
a pure single-mesh ridge integral should hit FFCX's `entity_type=="ridge"` branch (which DOES
assign the element table), so it MAY be fine -- this spike decides.

Candidate R (preferred): codim-2 ridge measure on the downstream-top edge.
Candidate B (fallback) : normalized downstream-BAND sink on ds_top (dimension-agnostic, robust).

Run from the forward-model dir:
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate pids-fem
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  python scratch/m3_3d_outlet_spike.py
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix

COMM = MPI.COMM_WORLD
L, W, H = 4.0, 2.0, 1.0          # x (downslope), y (cross-slope), z (vertical/gravity = last axis)
nx, ny, nz = 6, 4, 5
n_man, S0 = 0.05, 0.05
SECONDS_PER_DAY = 86400.0
ZAX = 2                           # gravity / elevation on the last axis (gdim - 1)

host = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [L, W, H]], [nx, ny, nz])
tdim = host.topology.dim         # 3
fdim = tdim - 1                  # 2  (faces)
rdim = tdim - 2                  # 1  (edges = ridges = the codim-2 outlet entity)
host.topology.create_connectivity(fdim, tdim)
host.topology.create_connectivity(rdim, tdim)

Vd = fem.functionspace(host, ("Lagrange", 1))
d = fem.Function(Vd, name="d")
d_const = 0.06                    # uniform depth -> the expected line discharge is exact + simple
d.interpolate(lambda x: d_const + 0.0 * x[0])
v = ufl.TestFunction(Vd)
one = fem.Constant(host, 1.0)

# Manning normal-depth discharge per unit width [m^2/day]; integrated over the outlet edge -> m^3/day
d_pos = ufl.max_value(d, 0.0)
q_out = SECONDS_PER_DAY * (1.0 / n_man) * d_pos ** (5.0 / 3.0) * np.sqrt(S0)
q_per_width = SECONDS_PER_DAY * (1.0 / n_man) * d_const ** (5.0 / 3.0) * np.sqrt(S0)
edge_len = W                       # downstream-top edge runs in y from 0 to W
q_total_expected = q_per_width * edge_len
print(f"q per unit width = {q_per_width:.6e} m^2/day ; outlet edge length = {edge_len} m ; "
      f"expected TOTAL line discharge = {q_total_expected:.6e} m^3/day")

# locate the codim-2 outlet entities (edges) on x=L AND z=H
on_edge = lambda x: np.isclose(x[0], L) & np.isclose(x[ZAX], H)
oedges = np.sort(dmesh.locate_entities_boundary(host, rdim, on_edge)).astype(np.int32)
print(f"located {oedges.size} outlet ridge-edges (dim={rdim})")
print("=" * 80)

# ----------------------------------------------------------------------------------------------
# CANDIDATE R: codim-2 ridge integral on the downstream-top edge
# ----------------------------------------------------------------------------------------------
print("CANDIDATE R: codim-2 ridge integral on the downstream-top edge")
try:
    import ufl.measure as _m
    print("  ufl integral_type -> measure name:",
          getattr(_m, "integral_type_to_measure_name", "<n/a>"))
except Exception as _e:
    print("  (could not introspect ufl integral types:", _e, ")")

for mname in ("ridge", "exterior_facet_ridge", "ridge_facet"):
    try:
        rt = dmesh.meshtags(host, rdim, oedges, np.ones(oedges.size, dtype=np.int32))
        dR = ufl.Measure(mname, domain=host, subdomain_data=rt)(1)
        length = COMM.allreduce(fem.assemble_scalar(fem.form(one * dR)), op=MPI.SUM)
        F = q_out * v * dR
        bvec = assemble_vector(fem.form(F)); bvec.assemble()
        Q = COMM.allreduce(fem.assemble_scalar(fem.form(q_out * dR)), op=MPI.SUM)
        Jm = assemble_matrix(fem.form(ufl.derivative(F, d))); Jm.assemble()
        nzrows = int(np.nonzero(np.abs(bvec.getArray()) > 1e-30)[0].size)
        print(f"  [{mname}] OK: edge length={length:.4f} (expect {edge_len}); "
              f"integrated Q={Q:.6e} (expect {q_total_expected:.6e}, "
              f"rel {abs(Q - q_total_expected) / q_total_expected:.2e}); "
              f"#nz residual rows={nzrows}; Jacobian norm={Jm.norm():.4e}")
        print(f"  [{mname}] ====> WORKS")
    except Exception as e:
        print(f"  [{mname}] FAILED: {type(e).__name__}: {str(e)[:220]}")
print("=" * 80)

# ----------------------------------------------------------------------------------------------
# CANDIDATE B: normalized downstream-band sink on the top face ds_top  (FFCX-native fallback)
# ----------------------------------------------------------------------------------------------
print("CANDIDATE B: normalized downstream-band sink on ds_top  q_out*(1/xwidth)*chi_band*v*ds_top")
top = np.sort(dmesh.locate_entities_boundary(host, fdim, lambda x: np.isclose(x[ZAX], H)))
mt = dmesh.meshtags(host, fdim, top, np.ones(top.size, dtype=np.int32))
ds_top = ufl.Measure("ds", domain=host, subdomain_data=mt)(1)
xufl = ufl.SpatialCoordinate(host)
hx = L / nx
try:
    for wf in (1.0 * hx, 2.0 * hx, 4.0 * hx):
        # band within distance wf of the outlet x=L over the full y span. Normalize by the band's
        # X-WIDTH (band_area / edge_len) so the term integrates to the LINE discharge q_out*W.
        chi = ufl.conditional(ufl.gt(xufl[0], L - wf), 1.0, 0.0)
        band_area = COMM.allreduce(fem.assemble_scalar(fem.form(chi * ds_top)), op=MPI.SUM)  # ~ wf*W
        xwidth = band_area / edge_len                                                          # ~ wf
        Fb = q_out * (chi / xwidth) * v * ds_top
        bvec = assemble_vector(fem.form(Fb)); bvec.assemble()
        Q = COMM.allreduce(fem.assemble_scalar(fem.form(q_out * (chi / xwidth) * ds_top)), op=MPI.SUM)
        Jm = assemble_matrix(fem.form(ufl.derivative(Fb, d))); Jm.assemble()
        print(f"  w={wf:.3f} ({wf / hx:.0f} cells): band_area={band_area:.4f} xwidth={xwidth:.4f} "
              f"integrated Q={Q:.6e} (expect {q_total_expected:.6e}, "
              f"rel {abs(Q - q_total_expected) / q_total_expected:.2e}) Jnorm={Jm.norm():.3e}")
    print("  ====> WORKS (FFCX-native fallback); integrated Q ~ line discharge, w-independent")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:220]}")
print("=" * 80)

# ----------------------------------------------------------------------------------------------
# CANDIDATE R COEXISTENCE GUARD: ridge + cell + exterior_facet + vertex in ONE coupled form set.
# The realization-S FFCX bug appeared ONLY when the offending integral COEXISTED with the other
# coupled forms (it compiled fine in isolation). So the decisive test for R is that the ridge
# integral compiles + assembles when it shares a form with dx (cell), ds_top (facet), and dP
# (vertex) -- the exact integral-type mix the production F_d / F_psi / F_lam will contain, including
# a cross-field (psi) coupling term so the block Jacobian codegen crosses fields like the real solve.
# ----------------------------------------------------------------------------------------------
print("CANDIDATE R COEXISTENCE: ridge + cell + facet + vertex in ONE form (S-style coexistence guard)")
try:
    Vpsi = fem.functionspace(host, ("Lagrange", 1))
    psi = fem.Function(Vpsi); psi.x.array[:] = -0.5
    below = lambda x: x[ZAX] < H - 0.25 * (H / nz)
    host.topology.create_connectivity(0, tdim)
    pin = np.sort(dmesh.locate_entities(host, 0, below)).astype(np.int32)
    vt = dmesh.meshtags(host, 0, pin, np.full(pin.size, 1, dtype=np.int32))
    dP = ufl.Measure("vertex", domain=host, subdomain_data=vt)(1)
    rt = dmesh.meshtags(host, rdim, oedges, np.ones(oedges.size, dtype=np.int32))
    dR = ufl.Measure("ridge", domain=host, subdomain_data=rt)(1)
    # F_d-like form mixing ALL FOUR integral types + a cross-field (psi) ds_top coupling
    Fmix = (d * v * ds_top                                     # facet  (surface storage-like)
            + ufl.dot(ufl.grad(d), ufl.grad(v)) * ufl.dx       # cell   (a bulk term)
            + 1e-10 * d * v * dP                               # vertex (pin diagonal allocation)
            + q_out * v * dR                                   # ridge  (the outlet)
            + psi * v * ds_top)                                # cross-field facet coupling
    bvec = assemble_vector(fem.form(Fmix)); bvec.assemble()
    Jdd = assemble_matrix(fem.form(ufl.derivative(Fmix, d))); Jdd.assemble()      # diagonal block
    Jdp = assemble_matrix(fem.form(ufl.derivative(Fmix, psi))); Jdp.assemble()    # cross-field block
    print(f"  coexistence form compiled + assembled OK: |b|={bvec.norm():.4e} "
          f"|dF/dd|={Jdd.norm():.4e} |dF/dpsi|={Jdp.norm():.4e}")
    print("  ====> R is FFCX-safe in the coupled integral-type mix (no S-style coexistence crash)")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  COEXISTENCE FAILED: {type(e).__name__}: {str(e)[:240]}")
print("=" * 80)
print("SPIKE DONE")
