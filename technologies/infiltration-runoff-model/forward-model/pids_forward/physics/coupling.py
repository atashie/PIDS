"""Monolithic surface<->subsurface coupling (Module 3, design §D + the 2026-06-05 NCP spec).

Composes the subsurface Richards block (ψ, §B) and a surface water store (d ≥ 0, §C) into ONE
blocked residual solved by a single Newton (PETSc SNES) per backward-Euler step, linked by the
SUPPLY-LIMITED land-surface exchange. The unconstrained Robin law q_pot = k_ex·(d − ψ_top) (design
§D.3) is INCOMPLETE -- it lets dry soil over-draw an empty surface store (d < 0). The fix (spec
docs/plans/2026-06-05-module3-landsurface-ncp-spec.md, Codex-reviewed) makes the actual exchanged
flux λ the single sign-paired unknown, with a smooth complementarity selecting the regime:

    (a) surface:   d(d)/dt + div(q_ovl) + λ = rain          [q_ovl = 0 in 1-D: surface is a point]
    (b) soil top:  Richards top influx = λ                   [sign-paired with (a) -> conservation structural]
    (c) regime:    d ≥ 0,  g ≥ 0,  d·g = 0,   g = q_pot − λ  [smooth Fischer-Burmeister NCP]
                   q_pot = k_ex·(d − ψ_top),  k_ex = K(ψ_top)/ℓ_c

Ponded (d>0) ⇒ g=0 ⇒ λ=q_pot (the §D.3 Robin). Supply-limited (g>0) ⇒ d=0 ⇒ λ=rain (all rain
infiltrates). Solved with PLAIN Newton (the NCP is C^∞-smoothed -- no PETSc VI, sidestepping the M2
stiff-overland VI failure). Conservation is structural: λ enters both blocks with opposite sign, so
total water changes by exactly (rain − outflow) for ANY active set or smoothing. Reduces to Module 1
``add_ponding_bc`` (pond = max(ψ_top,0)) in the no-lateral-flow limit; recovers ψ_top → d as k_ex→∞.

REALIZATION (1-D first). 1-D validates the exchange PHYSICS where the surface is the top point and
there is no lateral overland; d and λ are co-located on the host (P1 fields pinned to 0 below the
surface), so the exchange is a host ``ds_top`` facet term and the blocked Newton is the high-level
DOLFINx solver with an exact auto-Jacobian. The 2-D/3-D lateral overland uses the design-intended
top-facet SUBMESH (realization S); those tests will drive that path. The exchange physics, mass
accounting, and accessors here are realization-agnostic.

Design: docs/plans/2026-06-04-pids-forward-model-architecture-design.md (Sec. D) +
docs/plans/2026-06-05-module3-landsurface-ncp-spec.md.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from .richards import richards_bulk_residual


def _fischer_burmeister(a, b, eps):
    """Smoothed Fischer-Burmeister NCP function: Φ=0 ⟺ a≥0, b≥0, a·b=eps² (→ a·b=0 as eps→0).

    C^∞ in (a,b), so the coupled Newton stays smooth (no variational inequality). ``a`` and ``b``
    must share units (the caller scales the flux leg by a timescale τ_c so both are depths).
    """
    return a + b - ufl.sqrt(a * a + b * b + 2.0 * eps * eps)


class CoupledProblem:
    """Monolithic [ψ, d, λ] surface<->subsurface coupling on a host mesh (1-D realization).

    ``mesh`` is the subsurface host (a 1-D column for the exchange-physics validation); ``soil`` a
    ``VanGenuchten``. The surface store ``d`` and exchange flux ``λ`` are co-located on the host and
    constrained to the top surface; ``ell_c`` (coupling film thickness) defaults to the top-cell
    half-height. ``eps_ncp`` is the NCP smoothing depth [m].
    """

    _DEFAULT_PETSC_OPTIONS = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    def __init__(self, mesh, soil, *, ell_c: float | None = None, eps_ncp: float = 1e-4,
                 degree: int = 1, lumped: bool = True, petsc_options=None):
        self.mesh = mesh
        self.soil = soil
        self.eps_ncp = float(eps_ncp)
        gdim = mesh.geometry.dim
        self._zaxis = gdim - 1  # gravity / elevation along the last spatial axis

        self.Vpsi = fem.functionspace(mesh, ("Lagrange", degree))
        self.Vd = fem.functionspace(mesh, ("Lagrange", degree))    # surface store (1-D: co-located)
        self.Vlam = fem.functionspace(mesh, ("Lagrange", degree))  # exchange flux (1-D: co-located)
        self.psi = fem.Function(self.Vpsi, name="psi")
        self.psi_n = fem.Function(self.Vpsi, name="psi_n")
        self.d = fem.Function(self.Vd, name="d")
        self.d_n = fem.Function(self.Vd, name="d_n")
        self.lam = fem.Function(self.Vlam, name="lambda")  # instantaneous flux: no previous-step value

        e_g = np.zeros(gdim, dtype=PETSc.ScalarType)
        e_g[-1] = 1.0
        self.e_g = fem.Constant(mesh, e_g)
        self.dt = fem.Constant(mesh, PETSc.ScalarType(1.0))

        # top (max-elevation) surface facets + the surface coupling measure.
        zc = self.Vpsi.tabulate_dof_coordinates()[:, self._zaxis]
        self._ztop = mesh.comm.allreduce(float(zc.max()) if zc.size else -np.inf, op=MPI.MAX)
        fdim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(fdim, mesh.topology.dim)
        top_facets = np.sort(dmesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[self._zaxis], self._ztop)))
        ft = dmesh.meshtags(mesh, fdim, top_facets, np.ones(top_facets.size, dtype=np.int32))
        self._ds_top = ufl.Measure("ds", domain=mesh, subdomain_data=ft)(1)

        if ell_c is None:
            zu = np.sort(np.unique(zc))
            local = 0.5 * float(zu[-1] - zu[-2]) if zu.size >= 2 else np.inf
            ell_c = mesh.comm.allreduce(local, op=MPI.MIN)  # top-cell half-height
        self.ell_c = float(ell_c)
        self._tau_c = self.ell_c / soil.Ks  # NCP unit scaling: τ_c·(flux) has depth units

        self._dx_storage = (
            ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
            if lumped else ufl.dx
        )

        vpsi = ufl.TestFunction(self.Vpsi)
        vd = ufl.TestFunction(self.Vd)
        vlam = ufl.TestFunction(self.Vlam)
        self._vd = vd

        # potential infiltration (design §D.3 Robin) and the smooth complementarity g = q_pot − λ.
        q_pot = (soil.K_ufl(self.psi) / self.ell_c) * (self.d - self.psi)
        g = q_pot - self.lam
        eps_diag = 1e-10  # allocates interior d/λ matrix diagonals for the off-top Dirichlet pin

        # ψ block: Richards bulk + infiltration influx (−λ into the soil top, flux-BC sign).
        self.F_psi = richards_bulk_residual(
            self.psi, self.psi_n, vpsi, soil, self.dt, self.e_g, dx_storage=self._dx_storage
        ) - self.lam * vpsi * self._ds_top
        # d block: surface storage + λ leaving the store (sign-paired) [+ rain via add_rain].
        self.F_d = ((self.d - self.d_n) / self.dt) * vd * self._ds_top \
            + self.lam * vd * self._ds_top \
            + eps_diag * self.d * vd * ufl.dx
        # λ block: the smooth NCP picks the regime (ponded λ=q_pot vs supply-limited d=0).
        self.F_lam = _fischer_burmeister(self.d, self._tau_c * g, self.eps_ncp) * vlam * self._ds_top \
            + eps_diag * self.lam * vlam * ufl.dx

        # constrain d and λ to the top surface: pin all non-surface DOFs to 0.
        below = lambda x: x[self._zaxis] < self._ztop - 0.25 * self.ell_c
        self._bcs = [
            fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_geometrical(self.Vd, below), self.Vd),
            fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_geometrical(self.Vlam, below), self.Vlam),
        ]

        self._petsc_options = dict(petsc_options or self._DEFAULT_PETSC_OPTIONS)
        self._problem: NonlinearProblem | None = None
        self._rain = None

    # -- problem setup --------------------------------------------------------
    def set_initial_condition(self, psi_expr, d_value: float = 0.0) -> None:
        """Set the subsurface IC ``psi_expr`` and a uniform surface depth ``d_value`` (on the top)."""
        self.psi.interpolate(psi_expr)
        self.psi_n.interpolate(psi_expr)
        for f in (self.d, self.d_n, self.lam):
            f.x.array[:] = 0.0
        topdofs = self._top_dofs(self.Vd)
        self.d.x.array[topdofs] = d_value
        self.d_n.x.array[topdofs] = d_value
        for f in (self.d, self.d_n, self.lam):
            f.x.scatter_forward()

    def add_dirichlet_psi(self, locator, value) -> None:
        dofs = fem.locate_dofs_geometrical(self.Vpsi, locator)
        self._bcs.append(fem.dirichletbc(PETSc.ScalarType(value), dofs, self.Vpsi))
        self._problem = None

    def add_rain(self, rate):
        """Rainfall onto the surface store (m/day, positive = water in). Returns the Constant."""
        r = rate if isinstance(rate, fem.Constant) else fem.Constant(
            self.mesh, PETSc.ScalarType(float(rate)))
        self.F_d = self.F_d - r * self._vd * self._ds_top
        self._rain = r
        self._problem = None
        return r

    def _top_dofs(self, V):
        return fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[self._zaxis], self._ztop))

    def _ensure_problem(self) -> None:
        if self._problem is None:
            self._problem = NonlinearProblem(
                [self.F_psi, self.F_d, self.F_lam], [self.psi, self.d, self.lam], bcs=self._bcs,
                petsc_options_prefix="coupled_", petsc_options=self._petsc_options,
                kind="mpi",  # monolithic block AIJ so a single LU factorizes the coupled operator
            )

    # -- time stepping --------------------------------------------------------
    def step(self, dt: float):
        """Advance one monolithic backward-Euler Newton step. Returns (converged, iters)."""
        self.dt.value = dt
        self._ensure_problem()
        self._problem.solve()
        snes = self._problem.solver
        converged = snes.getConvergedReason() > 0
        iters = int(snes.getIterationNumber())
        if converged:
            self.psi_n.x.array[:] = self.psi.x.array
            self.psi_n.x.scatter_forward()
            self.d_n.x.array[:] = self.d.x.array
            self.d_n.x.scatter_forward()
        else:
            self.psi.x.array[:] = self.psi_n.x.array
            self.psi.x.scatter_forward()
            self.d.x.array[:] = self.d_n.x.array
            self.d.x.scatter_forward()
        return converged, iters

    def advance(self, t_end, dt, *, dt_min=1e-6, dt_max=None, grow=1.5, cut=0.5,
                target_low=3, target_high=8, shrink=0.7, max_steps=100000):
        """March to ``t_end`` with adaptive backward-Euler steps (cut-and-retry on non-convergence)."""
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
    def soil_water(self) -> float:
        """Subsurface stored water = integral of theta(psi) over the column (storage quadrature)."""
        form = fem.form(self.soil.theta_ufl(self.psi) * self._dx_storage)
        return self.mesh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)

    def surface_water(self) -> float:
        """Surface stored water = integral of d over the top surface (a point in 1-D)."""
        form = fem.form(self.d * self._ds_top)
        return self.mesh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)

    def total_water(self) -> float:
        """Total stored water = subsurface + surface (the closed-system conserved quantity)."""
        return self.soil_water() + self.surface_water()

    def surface_depth(self) -> float:
        """Ponding depth d at the surface (max over the top nodes)."""
        topdofs = self._top_dofs(self.Vd)
        local = float(np.max(self.d.x.array[topdofs])) if topdofs.size else -np.inf
        return self.mesh.comm.allreduce(local, op=MPI.MAX)

    def exchange_flux(self) -> float:
        """Land-surface infiltration flux λ at the surface (m/day; max over the top nodes)."""
        topdofs = self._top_dofs(self.Vlam)
        local = float(np.max(self.lam.x.array[topdofs])) if topdofs.size else -np.inf
        return self.mesh.comm.allreduce(local, op=MPI.MAX)
