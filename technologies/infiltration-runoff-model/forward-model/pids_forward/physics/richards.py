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
    """A mixed-form Richards initial-boundary-value problem on a given mesh."""

    def __init__(self, mesh, soil, *, degree: int = 1, source=None):
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
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1e-10,
                    "snes_atol": 1e-12,
                    "snes_max_it": 50,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
            )

    # -- time stepping --------------------------------------------------------
    def step(self, dt: float):
        """Advance one backward-Euler step. Returns (converged: bool, iters: int)."""
        self.dt.value = dt
        self._ensure_problem()
        result = self._problem.solve()  # solves in place on self.psi
        converged, iters = self._convergence(result)
        self.psi_n.x.array[:] = self.psi.x.array
        return converged, iters

    @staticmethod
    def _convergence(result):
        # DOLFINx 0.10 NonlinearProblem.solve() -> (u, converged_reason, n_iters).
        if isinstance(result, tuple) and len(result) >= 3:
            return int(result[1]) > 0, int(result[2])
        return True, -1

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
