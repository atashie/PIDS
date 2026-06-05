"""Dimension-agnostic diffusion-wave overland-flow solver (Module 2, overland).

2-D diffusion-wave approximation of the shallow-water equations (design Sec. C):
gravity-friction balance, local + convective inertia dropped, leaving a degenerate
parabolic nonlinear diffusion in the surface (ponding) depth ``d``:

    d(d)/dt + div(q) = sources,   q = -K_s(d) grad(H_s),   H_s = z_b + d
    K_s(d) = SECONDS_PER_DAY * d^{5/3} / ( n_man |grad H_s|^{1/2} + eps_S )   [m^2/day]

Primary variable = ponding depth ``d`` >= 0, enforced *inside the solve* by the PETSc
SNES variational-inequality solver (``vinewtonrsls``, proven in the install spike) --
NOT by post-solve clamping, which would break mass conservation. Backward Euler in
time. The same UFL residual serves 1-D / 2-D / 3-D-top meshes: the ONLY dimension-
aware quantity is the bed slope ``grad z_b`` (a known field), so no gravity unit
vector is needed -- unlike subsurface Richards (gravity carried as e_g), here gravity
enters through the topography. The Jacobian is auto-differentiated from the residual.

Units: length m, time day, slope dimensionless. ``n_man`` is the SI Manning roughness
(s.m^{-1/3}); the ``SECONDS_PER_DAY`` factor converts the SI Manning conveyance to
m^2/day so depths/timesteps stay consistent with subsurface Richards (m/day).

Regularization note (design C.3, C.4): the additive slope floor is applied INSIDE the
root as ``|grad H_s|^{1/2} -> (|grad H_s|^2 + eps_S^2)^{1/4}``, which keeps BOTH the
denominator non-zero AND the auto-differentiated Jacobian finite at exactly-zero bed
slope (a perfectly flat lake), where the design's bare additive ``+ eps_S`` would still
leave a singular d/dg of |grad H_s|^{1/2}. As |grad H_s| -> 0 this gives ordinary
linear diffusion K_s -> SECONDS_PER_DAY d^{5/3}/(n_man sqrt(eps_S)) (the lake-at-rest /
flat-terrain regime). The lake-at-rest Tier-1 test confirms spurious flux stays below
the 1e-6 conservation gate at this magnitude.

Design: docs/plans/2026-06-04-pids-forward-model-architecture-design.md (Sec. C/H).
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

SECONDS_PER_DAY = 86400.0


class OverlandProblem:
    """A diffusion-wave overland-flow initial-boundary-value problem on a mesh.

    The mesh is the *surface* itself: a 1-D line for a hillslope cross-section, a 2-D
    mesh for a catchment (standalone module tests), or the top facets of a 3-D host
    (coupling, Module 3). Depth positivity ``d >= 0`` is imposed by SNES-VI.
    """

    # Direct LU is robust + deterministic for the small verification meshes; override
    # (gmres + fieldsplit) for large 2-D/3-D production runs (design H.3).
    _DEFAULT_PETSC_OPTIONS = {
        "snes_type": "vinewtonrsls",  # variational inequality: enforces d >= 0
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    def __init__(self, mesh, n_man: float, *, degree: int = 1, eps_S: float = 1e-3,
                 source=None, petsc_options=None, lumped: bool = True):
        self.mesh = mesh
        self.n_man = float(n_man)
        self.eps_S = float(eps_S)
        self.V = fem.functionspace(mesh, ("Lagrange", degree))
        self.d = fem.Function(self.V, name="d")  # current step (n+1), ponding depth
        self.d_n = fem.Function(self.V, name="d_n")  # previous step (n)
        self.z_b = fem.Function(self.V, name="z_b")  # bed topography (default flat = 0)

        self.dt = fem.Constant(mesh, PETSc.ScalarType(1.0))

        # Mass-lumped storage (vertex quadrature) suppresses wet/dry-front oscillation
        # and keeps the conserved quantity = what the integrator conserves (design C.6).
        self._dx_storage = (
            ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
            if lumped
            else ufl.dx
        )
        v = ufl.TestFunction(self.V)
        H_s = self.z_b + self.d  # surface (hydraulic) head
        g = ufl.grad(H_s)
        # |grad H_s|^{1/2} with the slope floor inside the root -> finite Jacobian at g=0.
        slope_sqrt = (ufl.dot(g, g) + self.eps_S**2) ** 0.25
        d_pos = ufl.max_value(self.d, 0.0)  # guard fractional power against tiny d<0 iterates
        K_s = SECONDS_PER_DAY * d_pos ** (5.0 / 3.0) / (self.n_man * slope_sqrt)

        # Backward-Euler diffusion-wave residual; flux q = -K_s grad(H_s) integrated by
        # parts (natural no-flux boundary unless a Dirichlet/flux BC is added).
        self.F = ((self.d - self.d_n) / self.dt) * v * self._dx_storage \
            + K_s * ufl.dot(g, ufl.grad(v)) * ufl.dx
        if source is not None:
            self.F = self.F - source * v * ufl.dx

        self._v = v
        self._K_s = K_s  # retained for velocity / bed-shear diagnostics
        self._H_s = H_s
        self._bcs: list = []
        self._flux_tags: list = []
        self._petsc_options = dict(petsc_options or self._DEFAULT_PETSC_OPTIONS)
        self._problem: NonlinearProblem | None = None
        self._bounds = None  # (lb, ub) PETSc vecs kept alive for the VI solver

    # -- problem setup --------------------------------------------------------
    def set_topography(self, expr) -> None:
        """Set the bed elevation z_b(x) (a known P1 field; default flat z_b = 0)."""
        self.z_b.interpolate(expr)

    def set_initial_condition(self, expr) -> None:
        self.d.interpolate(expr)
        self.d_n.interpolate(expr)

    def add_dirichlet(self, locator, value) -> None:
        dofs = fem.locate_dofs_geometrical(self.V, locator)
        bc = fem.dirichletbc(PETSc.ScalarType(value), dofs, self.V)
        self._bcs.append(bc)
        self._problem = None  # force rebuild with the new BC set

    def _ensure_problem(self) -> None:
        if self._problem is None:
            self._problem = NonlinearProblem(
                self.F,
                self.d,
                bcs=self._bcs,
                petsc_options_prefix="overland_",
                petsc_options=self._petsc_options,
            )
            snes = self._problem.solver
            snes.setType("vinewtonrsls")  # belt-and-suspenders with the petsc_option
            lb = self.d.x.petsc_vec.duplicate()
            lb.set(0.0)  # depth lower bound d >= 0 (wet/dry complementarity)
            ub = self.d.x.petsc_vec.duplicate()
            ub.set(PETSc.INFINITY)
            snes.setVariableBounds(lb, ub)
            self._bounds = (lb, ub)

    # -- time stepping --------------------------------------------------------
    def step(self, dt: float):
        """Advance one backward-Euler SNES-VI step. Returns (converged: bool, iters).

        Convergence is read from the PETSc SNES converged reason (>0); an unconverged
        solve restores the last accepted state so callers can cut dt and retry.
        """
        self.dt.value = dt
        self._ensure_problem()
        self._problem.solve()  # updates self.d in place (subject to d >= 0)
        snes = self._problem.solver
        converged = snes.getConvergedReason() > 0
        iters = int(snes.getIterationNumber())
        if converged:
            self.d_n.x.array[:] = self.d.x.array  # accept the step
            self.d_n.x.scatter_forward()
        else:
            self.d.x.array[:] = self.d_n.x.array  # restore last accepted state
            self.d.x.scatter_forward()
        return converged, iters

    # -- diagnostics ----------------------------------------------------------
    def max_abs_error(self, exact) -> float:
        ex = fem.Function(self.V)
        ex.interpolate(exact)
        diff = np.abs(self.d.x.array - ex.x.array)
        return self.mesh.comm.allreduce(float(diff.max()), op=MPI.MAX)

    def total_water(self) -> float:
        """Total surface water = integral of d, using the storage quadrature."""
        form = fem.form(self.d * self._dx_storage)
        local = fem.assemble_scalar(form)
        return self.mesh.comm.allreduce(local, op=MPI.SUM)

    def l2_error(self, exact_ufl) -> float:
        """L2 norm ||d - exact||, with exact a UFL expression on this mesh."""
        form = fem.form((self.d - exact_ufl) ** 2 * ufl.dx(metadata={"quadrature_degree": 6}))
        local = fem.assemble_scalar(form)
        return float(np.sqrt(self.mesh.comm.allreduce(local, op=MPI.SUM)))
