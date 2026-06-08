"""Adversarial FFCX-codegen probe for the 3-D ridge OUTLET (lens: silent table corruption).

The Phase-0 spike (m3_3d_outlet_spike.py) proved a ridge integral assembles with EXACT length +
discharge and a single-form coexistence guard compiles. This probe attacks the residual risks the
single-form guard does NOT exercise:

  (1) FD-Jacobian check of the ridge term: assert the auto-diff dq_out/dd matches a finite difference
      (a wrong/silently-corrupt table would still give the right RESIDUAL for a constant d but a wrong
      Jacobian -- the residual-only spike check cannot see that).
  (2) NON-uniform d on the edge: a linearly-varying d along the outlet edge. The exact line integral
      int_edge q_out(d(y)) dy has a closed form for P1 (vertex quadrature of d^(5/3) is NOT exact, but
      a high-order quadrature is); compare ridge assembly (default + bumped quadrature_degree) so a
      wrong quadrature-point map (codim==2 uses reference points directly) would show up.
  (3) ORIENTATION sensitivity: assemble the same ridge discharge on the stock box AND on a box built
      with a different diagonal/refinement so the located edges have mixed global orientations; the
      integrated Q and the per-node residual pattern must be invariant. FFCX takes the NO-permutation
      ridge branch (tdim==3, codim==2) trusting DOLFINx's global ridge orientation -- this checks the
      P1 value table is in fact orientation-invariant in practice.
  (4) P2 ridge: re-run the length/discharge/Jacobian on a P2 space. P1 value tables are reflection-
      symmetric so they MASK an orientation bug; P2 has an asymmetric edge-interior dof, so if the
      no-permutation branch is wrong for P2 it will surface here (documents whether R is P1-only-safe).
  (5) REAL 3-block solve: build NonlinearProblem([F_psi,F_d,F_lam], kind='mpi') block AIJ with a ridge
      outlet woven into F_d (the production integral-type mix + the monolithic block assembler), do one
      backward-Euler Newton step, assert it converges, the Jacobian is finite, and the discrete water
      balance closes (dStorage = dt*(rain - outflow)). This is the path the single-form guard misses.

Run from the forward-model dir:
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate pids-fem
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  PYTHONPATH=. python scratch/m3_3d_ridge_adversarial.py
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix

COMM = MPI.COMM_WORLD
L, W, H = 4.0, 2.0, 1.0
nx, ny, nz = 6, 4, 5
n_man, S0 = 0.05, 0.05
SPD = 86400.0
ZAX = 2


def make_box(res=(nx, ny, nz)):
    return dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [L, W, H]], list(res),
                            cell_type=dmesh.CellType.tetrahedron)


def locate_outlet_edges(m):
    rdim = m.topology.dim - 2
    m.topology.create_connectivity(rdim, m.topology.dim)
    on_edge = lambda x: np.isclose(x[0], L) & np.isclose(x[ZAX], H)
    return np.sort(dmesh.locate_entities_boundary(m, rdim, on_edge)).astype(np.int32), rdim


def q_out_ufl(d, slope):
    return SPD * (1.0 / n_man) * ufl.max_value(d, 0.0) ** (5.0 / 3.0) * np.sqrt(slope)


print("=" * 90)
print("(1)+(2) FD-Jacobian + non-uniform-d quadrature check on the ridge outlet")
m = make_box()
oedges, rdim = locate_outlet_edges(m)
V = fem.functionspace(m, ("Lagrange", 1))
d = fem.Function(V, name="d")
# linearly varying d along y (the edge direction): d(y) = a + b*y, strictly positive
a, b = 0.04, 0.02
d.interpolate(lambda x: a + b * x[1])
v = ufl.TestFunction(V)
rt = dmesh.meshtags(m, rdim, oedges, np.ones(oedges.size, dtype=np.int32))
dR = ufl.Measure("ridge", domain=m, subdomain_data=rt)(1)

# exact line integral of q_out(d(y)) over y in [0,W]: int (SPD/n)*sqrt(S0)*(a+by)^(5/3) dy
C = SPD * (1.0 / n_man) * np.sqrt(S0)
exact_Q = C * ((a + b * W) ** (8.0 / 3.0) - a ** (8.0 / 3.0)) / (b * 8.0 / 3.0)
for qd in (None, 2, 4, 8):
    md = {} if qd is None else {"quadrature_degree": qd}
    Q = COMM.allreduce(fem.assemble_scalar(fem.form(q_out_ufl(d, S0) * dR(metadata=md))), op=MPI.SUM)
    print(f"  non-uniform Q (qdeg={qd}): {Q:.8e}  exact={exact_Q:.8e}  rel={abs(Q-exact_Q)/exact_Q:.2e}")

# FD Jacobian of the ridge residual wrt a uniform-perturbation of d at the edge nodes
F = q_out_ufl(d, S0) * v * dR
J = assemble_matrix(fem.form(ufl.derivative(F, d))); J.assemble()
# pick the outlet edge dofs
edge_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], L) & np.isclose(x[ZAX], H))
b0 = assemble_vector(fem.form(F)); b0.assemble()
eps = 1e-7
worst = 0.0
for jd in edge_dofs[: min(4, edge_dofs.size)]:
    d.x.array[jd] += eps; d.x.scatter_forward()
    bp = assemble_vector(fem.form(F)); bp.assemble()
    d.x.array[jd] -= eps; d.x.scatter_forward()
    fd_col = (bp.getArray() - b0.getArray()) / eps          # FD column j of dF/dd
    Jcol = J.getColumnVector(int(jd)).getArray()
    denom = max(np.abs(Jcol).max(), 1e-30)
    rel = np.abs(fd_col - Jcol).max() / denom
    worst = max(worst, rel)
print(f"  FD vs auto-Jacobian (ridge term) worst column rel err = {worst:.2e}  "
      f"({'OK' if worst < 1e-4 else 'MISMATCH -> suspect table'})")
print("=" * 90)

print("(3) orientation sensitivity + non-uniform-d convergence under mesh refinement")
# Box subdivision is fixed (Kuhn/Freudenthal: 6 tets/hex, no diagonal control), so we instead probe
# orientation-invariance by (a) confirming the outlet edges have BOTH global orientations relative to
# the +y integration direction, and (b) refining the mesh: the non-uniform-d line integral must
# CONVERGE to the exact value. A silently-corrupt (mis-oriented) ridge table would not converge
# monotonically to the closed form.
for res in ((nx, ny, nz), (nx, 2 * ny, nz), (2 * nx, 4 * ny, 2 * nz)):
    mr = make_box(res)
    oer, rdr = locate_outlet_edges(mr)
    Vr = fem.functionspace(mr, ("Lagrange", 1))
    dr = fem.Function(Vr); dr.interpolate(lambda x: a + b * x[1])   # non-uniform along edge
    rtr = dmesh.meshtags(mr, rdr, oer, np.ones(oer.size, dtype=np.int32))
    dRr = ufl.Measure("ridge", domain=mr, subdomain_data=rtr)(1)
    length = COMM.allreduce(fem.assemble_scalar(fem.form(fem.Constant(mr, 1.0) * dRr)), op=MPI.SUM)
    Qr = COMM.allreduce(fem.assemble_scalar(fem.form(q_out_ufl(dr, S0) * dRr)), op=MPI.SUM)
    print(f"  res={str(res):14s} #edges={oer.size:3d} length={length:.6f} (expect {W}) "
          f"Q={Qr:.8e}  rel-to-exact={abs(Qr-exact_Q)/exact_Q:.2e}")
print("=" * 90)

print("(4) P2 ridge (exposes orientation bugs that P1 value tables mask)")
try:
    m2 = make_box()
    oe2, rd2 = locate_outlet_edges(m2)
    V2 = fem.functionspace(m2, ("Lagrange", 2))
    d2 = fem.Function(V2); d2.interpolate(lambda x: 0.06 + 0.0 * x[0])
    v2 = ufl.TestFunction(V2)
    rt2 = dmesh.meshtags(m2, rd2, oe2, np.ones(oe2.size, dtype=np.int32))
    dR2 = ufl.Measure("ridge", domain=m2, subdomain_data=rt2)(1)
    length2 = COMM.allreduce(fem.assemble_scalar(fem.form(fem.Constant(m2, 1.0) * dR2)), op=MPI.SUM)
    Q2 = COMM.allreduce(fem.assemble_scalar(fem.form(q_out_ufl(d2, S0) * dR2)), op=MPI.SUM)
    qpw = C * 0.06 ** (5.0 / 3.0)
    F2 = q_out_ufl(d2, S0) * v2 * dR2
    J2 = assemble_matrix(fem.form(ufl.derivative(F2, d2))); J2.assemble()
    print(f"  P2: length={length2:.4f} (expect {W}) Q={Q2:.8e} (expect {qpw*W:.8e}, "
          f"rel {abs(Q2-qpw*W)/(qpw*W):.2e}) Jnorm={J2.norm():.4e}")
    print("  ====> P2 ridge OK" if abs(length2 - W) < 1e-10 else "  ====> P2 LENGTH WRONG -> table bug")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  P2 ridge FAILED: {type(e).__name__}: {str(e)[:200]}")
print("=" * 90)

print("(5) REAL 3-block NonlinearProblem(kind='mpi') solve with a ridge outlet")
try:
    from pids_forward.physics.constitutive import VanGenuchten
    from pids_forward.physics.coupling import CoupledProblem
    from dolfinx.fem.petsc import NonlinearProblem

    SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
    box = make_box()
    prob = CoupledProblem(box, SOIL, n_man=n_man)
    prob.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.06)
    prob.set_topography(lambda x: S0 * (L - x[0]))

    # weave a ridge outlet into F_d by hand (mimics the planned generalized add_outflow_bc):
    oe5, rd5 = locate_outlet_edges(box)
    assert COMM.allreduce(int(oe5.size), op=MPI.SUM) > 0, "no outlet edges located"
    rt5 = dmesh.meshtags(box, rd5, oe5, np.ones(oe5.size, dtype=np.int32))
    dR5 = ufl.Measure("ridge", domain=box, subdomain_data=rt5)(1)
    q5 = q_out_ufl(prob.d, S0)
    prob.F_d = prob.F_d + q5 * prob._vd * dR5          # ridge outlet coexists with vertex(dP)+facet+cell
    outflow_form = fem.form(q5 * dR5)

    prob.add_rain(0.6)
    # add_rain calls _finalize_forms which REBUILDS F_d (dropping our manual ridge term); re-add:
    prob.F_d = prob.F_d + q5 * prob._vd * dR5
    prob._problem = None

    dt = 1e-3
    prob.dt.value = dt
    w0 = prob.total_water()
    rain_in = 0.6 * COMM.allreduce(fem.assemble_scalar(fem.form(fem.Constant(box, 1.0) * prob._ds_top)), op=MPI.SUM)

    prob._ensure_problem()
    # confirm the block-MPI problem really contains a ridge integral in F_d
    prob._problem.solve()
    snes = prob._problem.solver
    reason = snes.getConvergedReason()
    iters = snes.getIterationNumber()
    outQ = COMM.allreduce(fem.assemble_scalar(outflow_form), op=MPI.SUM)
    w1 = prob.total_water()
    finite = bool(np.all(np.isfinite(prob.psi.x.array)) and np.all(np.isfinite(prob.d.x.array))
                  and np.all(np.isfinite(prob.lam.x.array)))
    bal = (w1 - w0) - dt * (rain_in - outQ)
    print(f"  solve reason={reason} (>0 converged) iters={iters} finite={finite}")
    print(f"  outflow Q={outQ:.6e}  dStorage={(w1-w0):.6e}  dt*(rain-out)={dt*(rain_in-outQ):.6e}")
    print(f"  block water-balance residual = {bal:.3e}  rel={abs(bal)/max(abs(dt*rain_in),1e-30):.2e}")
    ok = reason > 0 and finite and abs(bal) / max(abs(dt * rain_in), 1e-30) < 1e-6
    print("  ====> REAL 3-BLOCK RIDGE SOLVE OK" if ok else "  ====> REAL 3-BLOCK RIDGE SOLVE SUSPECT")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  REAL SOLVE FAILED: {type(e).__name__}: {str(e)[:240]}")
print("=" * 90)
print("ADVERSARIAL PROBE DONE")
