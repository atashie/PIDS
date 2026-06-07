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
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc


def richards_bulk_residual(psi, psi_n, v, soil, dt, e_g, *, dx=ufl.dx, dx_storage=None,
                           source=None):
    """Mixed-form Richards BULK residual (storage + Darcy), design B.1/B.4.

    The composable core of Module 1, factored out so both ``RichardsProblem`` (standalone) and the
    Module-3 coupling assembler build the *same* UFL. Returns the weak residual

        ((theta(psi) - theta(psi_n)) / dt) * v * dx_storage
        + K(psi) * (grad(psi) + e_g) . grad(v) * dx            [ - source * v * dx ]

    with ``theta``/``K`` the van Genuchten-Mualem closures (mass-conservative mixed form), gravity
    unit vector ``e_g`` (last-axis), and the storage term on ``dx_storage`` (pass a vertex-lumped
    measure for the production path; defaults to ``dx``). Boundary/exchange terms (Neumann flux,
    ponding, the §D land-surface coupling) are added by the caller on top of this bulk residual.
    """
    if dx_storage is None:
        dx_storage = dx
    theta = soil.theta_ufl(psi)
    theta_n = soil.theta_ufl(psi_n)
    K = soil.K_ufl(psi)
    F = ((theta - theta_n) / dt) * v * dx_storage + K * ufl.dot(
        ufl.grad(psi) + e_g, ufl.grad(v)
    ) * dx
    if source is not None:
        F = F - source * v * dx
    return F


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

    def __init__(self, mesh, soil, *, degree: int = 1, source=None, petsc_options=None,
                 lumped: bool = True):
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

        # Mass-lumped storage (vertex quadrature) suppresses wetting-front
        # oscillation (design B.4); pass lumped=False for clean MMS L2 order.
        self._dx_storage = (
            ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
            if lumped
            else ufl.dx
        )
        v = ufl.TestFunction(self.V)
        # Mixed-form backward-Euler bulk residual (storage + Darcy + optional source); the
        # composable builder is shared with the Module-3 coupling assembler.
        self.F = richards_bulk_residual(
            self.psi, self.psi_n, v, soil, self.dt, self.e_g,
            dx_storage=self._dx_storage, source=source,
        )

        self._v = v
        self._bcs: list = []
        self._flux_tags: list = []
        self._drainage_forms: list = []  # compiled q_n*ds GHB forms for drainage_rate()
        self._drainage_facets = np.empty(0, dtype=np.int32)  # disjointness guard across GHB boundaries
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

    def add_flux_bc(self, locator, flux):
        """Prescribe a normal INFLUX (m/day, positive INTO the domain) on a boundary.

        ``flux`` may be a float or a ``fem.Constant``; for time-dependent forcing
        (rainfall / evaporation) pass a Constant and update its ``.value`` each step.
        Returns the Constant so the caller can drive it. Enters the weak form as the
        natural Neumann term ``-flux * v * ds`` (zero-flux is the default if unset).
        """
        fdim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        facets = np.sort(dmesh.locate_entities_boundary(self.mesh, fdim, locator))
        tag = len(self._flux_tags) + 1
        self._flux_tags.append(tag)
        ft = dmesh.meshtags(
            self.mesh, fdim, facets, np.full(facets.shape, tag, dtype=np.int32)
        )
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=ft)
        q = (
            flux
            if isinstance(flux, fem.Constant)
            else fem.Constant(self.mesh, PETSc.ScalarType(float(flux)))
        )
        self.F = self.F - q * self._v * ds(tag)
        self._problem = None  # force rebuild with the new flux term
        return q

    def _boundary_ds(self, locator):
        fdim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        facets = np.sort(dmesh.locate_entities_boundary(self.mesh, fdim, locator))
        tag = len(self._flux_tags) + 1
        self._flux_tags.append(tag)
        ft = dmesh.meshtags(
            self.mesh, fdim, facets, np.full(facets.shape, tag, dtype=np.int32)
        )
        return ufl.Measure("ds", domain=self.mesh, subdomain_data=ft)(tag)

    def add_ponding_bc(self, locator, rain):
        """Top boundary with a surface PONDING store (vertical accumulation only).

        Rainfall ``rain`` (m/day) enters the top; whatever the soil cannot infiltrate
        accumulates as a ponded depth ``d = max(psi, 0)`` that raises the surface
        pressure head -- with NO lateral surface flow (that is the overland module).
        Mass-conserving: ``rain = infiltration + d(pond)/dt``. The pond store also gives
        a saturated surface node a nonzero storage diagonal, so over-saturating storms
        (intense rain on wet soil -> rapid saturation) converge instead of stalling.
        ``rain`` may be a float or a fem.Constant (time-dependent). Returns the Constant.
        """
        ds_t = self._boundary_ds(locator)
        q = (
            rain
            if isinstance(rain, fem.Constant)
            else fem.Constant(self.mesh, PETSc.ScalarType(float(rain)))
        )
        pond = ufl.max_value(self.psi, 0.0)
        pond_n = ufl.max_value(self.psi_n, 0.0)
        # + d(pond)/dt   (surface storage)   - rain   (influx)
        self.F = self.F + ((pond - pond_n) / self.dt) * self._v * ds_t - q * self._v * ds_t
        self._problem = None
        return q

    def add_drainage_bc(self, locator, conductance, external_head):
        """General-head (Cauchy / MODFLOW GHB) subsurface drainage on a boundary.

        Relative-permeability-weighted outward Darcy flux ``q_n = conductance * kr(psi) * (H -
        external_head)``, ``kr = K(psi)/Ks``, hydraulic head ``H = psi + z`` (z = elevation, last axis).
        The kr(psi) weight makes the GHB the physical Darcy flux through a boundary film of conductivity
        K(psi): it SELF-LIMITS as the boundary dries (a constant-C GHB would over-drain unsaturated soil
        by driving the boundary head down to deliver an unphysical flux -- Codex review 2026-06-07).
        Saturated boundary (kr=1) -> standard GHB. Lets the soil matrix exchange water with an external
        reservoir -- lateral groundwater outflow (a side), deep percolation (the base), or a drain.
        BIDIRECTIONAL: drains OUT when ``H > external_head``, draws IN when ``H < external_head``; reduces
        to no-flow as conductance -> 0 and to a saturated-Dirichlet head as conductance -> inf. Enters the
        residual as the standard exterior-facet term ``+ q_n * v * ds`` (since the bulk residual's Darcy
        term is K(grad psi + e_g).grad v with K grad H . n = -q_n). Smooth (C^1) Jacobian via K_ufl. This
        is Darcy/head physics, NOT the surface Manning law. Drain boundaries must be disjoint.
        """
        if conductance < 0.0:
            raise ValueError(f"add_drainage_bc requires conductance >= 0 [1/day]; got {conductance!r}")
        fdim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        facets = np.sort(dmesh.locate_entities_boundary(self.mesh, fdim, locator)).astype(np.int32)
        ov = int(np.intersect1d(facets, self._drainage_facets).size)
        if self.mesh.comm.allreduce(ov, op=MPI.SUM) > 0:
            raise ValueError("add_drainage_bc: locator overlaps an existing drainage boundary "
                             "(would double-count the flux); GHB boundaries must be disjoint.")
        self._drainage_facets = np.union1d(self._drainage_facets, facets).astype(np.int32)
        tag = len(self._flux_tags) + 1
        self._flux_tags.append(tag)
        ft = dmesh.meshtags(self.mesh, fdim, facets, np.full(facets.shape, tag, dtype=np.int32))
        ds_d = ufl.Measure("ds", domain=self.mesh, subdomain_data=ft)(tag)
        z = ufl.SpatialCoordinate(self.mesh)[self.mesh.geometry.dim - 1]
        kr = self.soil.K_ufl(self.psi) / self.soil.Ks   # relative perm: self-limits unsaturated drainage
        q_n = conductance * kr * (self.psi + z - external_head)
        self.F = self.F + q_n * self._v * ds_d
        self._drainage_forms.append(fem.form(q_n * ds_d))
        self._problem = None
        return None

    def drainage_rate(self) -> float:
        """Net outward subsurface drainage across all GHB boundaries (+ = net out of the domain)."""
        total = 0.0
        for form in self._drainage_forms:
            total += self.mesh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)
        return total

    def ponded_depth(self) -> float:
        """Ponded water depth max(psi, 0) at the top (highest-elevation) boundary node."""
        zc = self.V.tabulate_dof_coordinates()[:, self.mesh.geometry.dim - 1]
        top = np.isclose(zc, zc.max())
        local = float(np.max(np.maximum(self.psi.x.array[top], 0.0))) if np.any(top) else 0.0
        return self.mesh.comm.allreduce(local, op=MPI.MAX)

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
        """Total stored water = integral of theta(psi), using the storage quadrature.

        Uses the same (lumped) measure as the storage term, so the closed-system
        conserved quantity matches what the time integrator actually conserves.
        """
        form = fem.form(self.soil.theta_ufl(self.psi) * self._dx_storage)
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
