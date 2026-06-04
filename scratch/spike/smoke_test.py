"""
DOLFINx install spike — validate the FEM stack BEFORE committing the architecture.

Two capabilities decide whether the Option-B build can proceed on this machine:

  PART A  DOLFINx + PETSc linear solve works
          -> Poisson with a manufactured solution u = 1 + x^2 + 2y^2 (f = -6);
             solved with P2 (quadratic) elements, which CONTAIN this exact
             quadratic, so Galerkin reproduces it and the L2 error must be
             ~machine-small. (P1/linear would give the correct-but-nonzero
             O(h^2) interpolation error, not a clean stack-validation signal.)

  PART B  PETSc variational-inequality (complementarity) solver works
          -> this is the EXACT path the reverse PIDS catch-valve needs
             (vinewtonrsls / SNESVI). A small bound-constrained nonlinear
             system with a non-trivial active set must converge and satisfy
             the bounds + complementarity.

Run inside WSL2/conda:  conda activate pids-fem && python smoke_test.py
Exit code 0 = both capabilities confirmed.
"""
import sys
import numpy as np

results = {}


def part_a_dolfinx_poisson():
    from mpi4py import MPI
    import ufl
    import dolfinx
    from dolfinx import mesh, fem
    from dolfinx.fem.petsc import LinearProblem
    from petsc4py import PETSc

    print(f"  DOLFINx {dolfinx.__version__}")
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
    # P2: the manufactured quadratic lives in this space -> exact reproduction.
    V = fem.functionspace(domain, ("Lagrange", 2))

    uD = fem.Function(V)
    uD.interpolate(lambda x: 1.0 + x[0] ** 2 + 2.0 * x[1] ** 2)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(-6.0))
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    # DOLFINx 0.10: LinearProblem requires a (keyword-only) petsc_options_prefix
    # to namespace its KSP options in the global PETSc options DB.
    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options_prefix="smoke_poisson_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    V2 = fem.functionspace(domain, ("Lagrange", 2))
    uex = fem.Function(V2)
    uex.interpolate(lambda x: 1.0 + x[0] ** 2 + 2.0 * x[1] ** 2)
    err_local = fem.assemble_scalar(fem.form((uh - uex) ** 2 * ufl.dx))
    err_L2 = np.sqrt(domain.comm.allreduce(err_local, op=MPI.SUM))
    print(f"  Poisson L2 error = {err_L2:.3e} (expect < 1e-9 for an exact quadratic)")
    return err_L2 < 1e-9


def part_b_petsc_vi():
    """Bound-constrained nonlinear solve via SNES vinewtonrsls (the catch-valve's exact path)."""
    from petsc4py import PETSc
    print(f"  PETSc {'.'.join(str(v) for v in PETSc.Sys.getVersion())}")

    n = 20
    A = PETSc.Mat().createAIJ([n, n])
    A.setUp()
    for i in range(n):
        A.setValue(i, i, 2.0)
        if i > 0:
            A.setValue(i, i - 1, -1.0)
        if i < n - 1:
            A.setValue(i, i + 1, -1.0)
    A.assemble()

    # b: positive forcing on the first half (inactive), negative on the second
    # half (drives u<0 -> the u>=0 bound activates) => non-trivial active set.
    b = PETSc.Vec().createSeq(n)
    arr = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    b.setArray(arr)
    b.assemble()

    def formF(snes, x, F):          # residual F(u) = A u - b
        A.mult(x, F)
        F.axpy(-1.0, b)

    def formJ(snes, x, J, P):       # Jacobian = A (linear here)
        A.copy(P, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        if J != P:
            J.assemble()

    x = PETSc.Vec().createSeq(n); x.set(0.0)
    F = x.duplicate()
    J = A.duplicate(copy=True)

    snes = PETSc.SNES().create()
    snes.setFunction(formF, F)
    snes.setJacobian(formJ, J, J)
    snes.setType("vinewtonrsls")
    lb = x.duplicate(); lb.set(0.0)
    ub = x.duplicate(); ub.set(PETSc.INFINITY)
    snes.setVariableBounds(lb, ub)
    snes.setTolerances(rtol=1e-10, max_it=50)
    snes.setFromOptions()
    snes.solve(None, x)

    reason = snes.getConvergedReason()
    xa = x.getArray()
    Fv = x.duplicate(); formF(snes, x, Fv); Fa = Fv.getArray()
    bounds_ok = bool((xa >= -1e-9).all())
    # complementarity: where strictly interior (u>tol) residual ~0; where at bound, residual >= 0
    comp_ok = bool(np.all((xa > 1e-7) <= (np.abs(Fa) < 1e-6)) and np.all(Fa >= -1e-7))
    active = int(np.sum(xa < 1e-9))
    print(f"  SNES vinewtonrsls converged_reason={reason} (>0 ok); "
          f"active-set size={active}/{n}; bounds_ok={bounds_ok}; complementarity_ok={comp_ok}")
    return reason > 0 and bounds_ok and comp_ok and 0 < active < n


print("== DOLFINx install spike ==")
print("[A] DOLFINx + PETSc linear (Poisson, manufactured solution)")
try:
    results["A_dolfinx_poisson"] = part_a_dolfinx_poisson()
except Exception as e:
    results["A_dolfinx_poisson"] = False
    print(f"  ERROR: {type(e).__name__}: {e}")

print("[B] PETSc variational-inequality (vinewtonrsls — the catch-valve's exact path)")
try:
    results["B_petsc_vi"] = part_b_petsc_vi()
except Exception as e:
    results["B_petsc_vi"] = False
    print(f"  ERROR: {type(e).__name__}: {e}")

print("\n== RESULT ==")
for k, v in results.items():
    print(f"  {k:24} {'PASS' if v else 'FAIL'}")
ok = all(results.values())
print("\n==> SPIKE", "PASS — DOLFINx + PETSc-VI confirmed; the Option-B architecture is feasible on this machine."
      if ok else "FAIL — see errors above before committing the architecture.")
sys.exit(0 if ok else 1)
