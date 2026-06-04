"""Dimension-agnostic mixed-form Richards solver (Module 1, subsurface).

Mixed form (Celia et al. 1990, mass-conservative: storage kept as d(theta)/dt,
not C*d(psi)/dt), backward Euler in time, primary variable = pressure head psi,
gravity along the LAST spatial coordinate (elevation z = x[gdim-1]). The same UFL
residual serves 1-D / 2-D / 3-D meshes; only the gravity unit vector e_g is
dimension-aware. The Jacobian is auto-differentiated from the residual (full
Newton via PETSc SNES), which both supplies the consistent tangent and recovers
Celia's mass-conservative modified-Picard behaviour.

Design: docs/plans/2026-06-04-pids-forward-model-architecture-design.md (Sec. B/H).
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc


class RichardsProblem:
    """A mixed-form Richards initial-boundary-value problem on a given mesh.

    Admissibility note (unconfined, no specific storage): in a region that is
    *fully* saturated the capacity ``C = dtheta/dh`` is zero, so storage drops out
    and the operator is purely elliptic there. The pressure field is then unique
    only if the problem retains a head datum (a Dirichlet condition) or an
    unsaturated region; an all-saturated, pure-Neumann state is determined only up
    to a hydrostatic constant. Provide a datum for saturated scenarios.
    """

    # Default PETSc solver options: direct LU is robust for the small verification
    # meshes; override (e.g. gmres + hypre/gamg) for large 2-D/3-D production runs.
    _DEFAULT_PETSC_OPTIONS = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    def __init__(self, mesh, soil, *, degree: int = 1, source=None, petsc_options=None):
        self.mesh = mesh
        self.soil = soil
        self.V = fem.functionspace(mesh, ("Lagrange", degree))
        self.psi = fem.Function(self.V, name="psi")  # current step (n+1)
        self.psi_n = fem.Function(self.V, name="psi_n")  # previous step (n)

        gdim = mesh.geometry.dim
        e_g = np.zeros(gdim, dtype=PETSc.ScalarType)
        e_g[-1] = 1.0  # gravity / elevation gradient = unit vector along last axis
        self.e_g = fem.Constant(mesh, e_g)
        self.dt = fem.Constant(mesh, PETSc.ScalarType(1.0))

        v = ufl.TestFunction(self.V)
        theta = soil.theta_ufl(self.psi)
        theta_n = soil.theta_ufl(self.psi_n)
        K = soil.K_ufl(self.psi)
        # Mixed-form backward-Euler residual; Darcy flux q = -K (grad psi + e_g).
        self.F = ((theta - theta_n) / self.dt) * v * ufl.dx + K * ufl.dot(
            ufl.grad(self.psi) + self.e_g, ufl.grad(v)
        ) * ufl.dx
        # Optional volumetric source/sink (e.g. an MMS forcing term or root uptake).
        if source is not None:
            self.F = self.F - source * v * ufl.dx

        self._bcs: list = []
        self._petsc_options = dict(petsc_options or self._DEFAULT_PETSC_OPTIONS)
        self._problem: NonlinearProblem | None = None

    # -- problem setup --------------------------------------------------------
    def set_initial_condition(self, expr) -> None:
        self.psi.interpolate(expr)
        self.psi_n.interpolate(expr)

    def add_dirichlet(self, locator, value) -> None:
        dofs = fem.locate_dofs_geometrical(self.V, locator)
        bc = fem.dirichletbc(PETSc.ScalarType(value), dofs, self.V)
        self._bcs.append(bc)
        self._problem = None  # force rebuild with the new BC set

    def _ensure_problem(self) -> None:
        if self._problem is None:
            self._problem = NonlinearProblem(
                self.F,
                self.psi,
                bcs=self._bcs,
                petsc_options_prefix="richards_",
                petsc_options=self._petsc_options,
            )

    # -- time stepping --------------------------------------------------------
    def step(self, dt: float):
        """Advance one backward-Euler step. Returns (converged: bool, iters: int).

        Convergence is read directly from the PETSc SNES (converged reason > 0).
        DOLFINx 0.10 ``NonlinearProblem.solve()`` returns the solution Function, not
        a status tuple, so we must NOT infer success from its return value -- an
        unconverged solve returns ``converged=False`` so callers fail loudly rather
        than silently marching a diverged state.
        """
        self.dt.value = dt
        self._ensure_problem()
        self._problem.solve()  # updates self.psi in place
        snes = self._problem.solver
        converged = snes.getConvergedReason() > 0
        iters = int(snes.getIterationNumber())
        if converged:
            self.psi_n.x.array[:] = self.psi.x.array  # accept the step
            self.psi_n.x.scatter_forward()  # keep ghost DOFs consistent (MPI-safe)
        else:
            self.psi.x.array[:] = self.psi_n.x.array  # restore last accepted state (retry-safe)
            self.psi.x.scatter_forward()
        return converged, iters

    def advance(self, t_end, dt, *, dt_min=1e-6, dt_max=None, grow=1.5, cut=0.5,
                target_low=3, target_high=8, shrink=0.7, max_steps=100000):
        """March to ``t_end`` with adaptive backward-Euler steps.

        Cut ``dt`` and retry on a non-converged solve (stiff fronts / storm peaks).
        On accepted steps, grow ``dt`` when Newton converges easily
        (``iters <= target_low``) and shrink it on an expensive-but-accepted step
        (``iters >= target_high``), so the controller reacts to costly steps, not
        only to outright failures. Returns the number of accepted steps. Raises if
        ``dt`` must drop below ``dt_min`` to converge.
        """
        t = 0.0
        nsteps = 0
        while t < t_end - 1e-12:
            if nsteps >= max_steps:
                raise RuntimeError(f"advance exceeded max_steps={max_steps} at t={t:.4g}")
            h = min(dt, t_end - t)
            converged, iters = self.step(h)
            if converged:
                t += h
                nsteps += 1
                if iters <= target_low:
                    dt = dt * grow
                elif iters >= target_high:
                    dt = dt * shrink
                if dt_max is not None:
                    dt = min(dt, dt_max)
            else:
                dt = h * cut
                if dt < dt_min:
                    raise RuntimeError(f"advance: dt {dt:.2e} < dt_min at t={t:.4g} (not converging)")
        return nsteps

    # -- diagnostics ----------------------------------------------------------
    def max_abs_error(self, exact) -> float:
        ex = fem.Function(self.V)
        ex.interpolate(exact)
        diff = np.abs(self.psi.x.array - ex.x.array)
        return self.mesh.comm.allreduce(float(diff.max()), op=MPI.MAX)

    def total_water(self) -> float:
        """Total stored water, integral of theta(psi) over the domain."""
        form = fem.form(self.soil.theta_ufl(self.psi) * ufl.dx)
        local = fem.assemble_scalar(form)
        return self.mesh.comm.allreduce(local, op=MPI.SUM)

    def theta_array(self) -> np.ndarray:
        """Nodal water content theta(psi) at the (P1) degrees of freedom."""
        return self.soil.theta(self.psi.x.array)

    def l2_error(self, exact_ufl) -> float:
        """L2 norm ||psi - exact||, with exact a UFL expression on this mesh."""
        form = fem.form((self.psi - exact_ufl) ** 2 * ufl.dx(metadata={"quadrature_degree": 6}))
        local = fem.assemble_scalar(form)
        return float(np.sqrt(self.mesh.comm.allreduce(local, op=MPI.SUM)))
