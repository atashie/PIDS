"""Dimension-agnostic diffusion-wave overland-flow solver (Module 2, overland).

2-D diffusion-wave approximation of the shallow-water equations (design Sec. C):
gravity-friction balance, local + convective inertia dropped, leaving a degenerate
parabolic nonlinear diffusion in the surface (ponding) depth ``d``:

    d(d)/dt + div(q) = sources,   q = -K_s(d) grad(H_s),   H_s = z_b + d
    K_s(d) = SECONDS_PER_DAY * d^{5/3} / ( n_man |grad H_s|^{1/2} + eps_S )   [m^2/day]

Backward Euler in time, primary variable = ponding depth ``d`` >= 0. The same UFL
residual serves 1-D / 2-D / 3-D-top meshes: the ONLY dimension-aware quantity is the
bed slope ``grad z_b`` (a known field), so -- unlike subsurface Richards (gravity as a
unit vector e_g) -- gravity here enters through the topography. The Jacobian is
auto-differentiated from the residual.

Depth positivity (design §C.4, REVISED 2026-06-05 -- see governance/decision-log.md):
the locked "SNES-VI (vinewtonrsls) for d>=0" mechanism was found by adversarial Tier-1
testing to be UNUSABLE on this stiff, nonsymmetric diffusion-wave system (it walls within
a handful of steps -- raw PETSc and via DOLFINx NonlinearProblem, all line searches,
both vinewtonrsls and vinewtonssls -- and violates conservation when the bound binds; the
install spike's VI success was on a *linear symmetric* system and does not transfer). The
robust path is a smooth back-tracking Newton (``newtonls``), which converges and conserves
mass to ~1e-15 but undershoots ``d`` by ~1-3 cm at wet/dry fronts. Those undershoots are
removed by a post-step CONSERVATIVE positivity limiter (``_enforce_positivity``): clip
negatives to 0, then rescale the remaining positive water so the lumped water budget is
preserved to machine precision (Arik signed off this approach 2026-06-05). The limiter is
a no-op when no negatives are present, so smooth scenarios (lake-at-rest, MMS) are exact.

Units: length m, time day, slope dimensionless. ``n_man`` is the SI Manning roughness
(s.m^{-1/3}); the ``SECONDS_PER_DAY`` factor converts the SI Manning conveyance to m^2/day
so depths/timesteps stay consistent with subsurface Richards (m/day).

Regularization (design C.3, C.4): the slope floor is applied INSIDE the root as
``|grad H_s|^{1/2} -> (|grad H_s|^2 + eps_S^2)^{1/4}``, keeping BOTH the denominator
non-zero AND the auto-differentiated Jacobian finite at exactly-zero bed slope (a
perfectly flat lake), where the design's bare additive ``+ eps_S`` would still leave a
singular d/dg of |grad H_s|^{1/2}. As |grad H_s| -> 0 this gives ordinary linear diffusion
(the flat-terrain regime). ``eps_S`` is DIMENSIONLESS (|grad H_s| is a slope, m/m), and the
effective additive floor on the denominator at zero slope is ``sqrt(eps_S)`` (~0.032 for
eps_S=1e-3) -- ~30x the design's nominal additive-outside-the-root eps_S, a deliberate,
more-robust Jacobian-finiteness choice (so tune against ``sqrt(eps_S)``, not the design's
'1e-3..1e-2 m^{1/2}'). NOTE: lake-at-rest (uniform surface) gives flux = 0 *structurally*
regardless of eps_S; the eps_S-dependent spurious-flux-below-1e-6 claim is pinned instead by
the near-flat NON-uniform Tier-1 test (test_near_flat_no_spurious_flux_1d).

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
    (coupling, Module 3). Depth positivity ``d >= 0`` is maintained by a smooth Newton
    solve plus a conservative post-step positivity limiter (see module docstring).
    """

    # Smooth back-tracking Newton; direct LU is robust + deterministic for the small
    # verification meshes (override with gmres + fieldsplit for large 3-D, design H.3).
    _DEFAULT_PETSC_OPTIONS = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
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
        self._outflow_forms: list = []  # compiled q_out*ds forms for outflow_rate()
        self._petsc_options = dict(petsc_options or self._DEFAULT_PETSC_OPTIONS)
        self._problem: NonlinearProblem | None = None
        self.last_clip = 0.0  # diagnostic: largest negative depth clipped on the last step
        self.max_clip_seen = 0.0  # diagnostic: largest negative depth clipped over all steps
        self.last_outflow = 0.0  # outflow discharge at the SOLVED state (pre-limiter), for accounting
        self.clip_mass_adjust = 0.0  # cumulative UNAVOIDABLE mass change from the limiter
        # (0 in the normal rescale branch; > 0 only when a degenerate non-positive-total state
        # is dried -- should stay ~0 in well-behaved runs; a growing value signals trouble)

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

    def add_rain(self, rate):
        """Add a spatially-uniform rainfall (net) source r (m/day, positive = water IN).

        Enters the residual as the volumetric source ``-r v dx``. ``rate`` may be a float
        or a ``fem.Constant`` (drive ``.value`` for a time-varying hyetograph). Returns
        the Constant. (Infiltration / feature exchange are owned by the coupling module,
        Sec. D; standalone overland sees rainfall and boundary conditions only.)
        """
        r = (
            rate
            if isinstance(rate, fem.Constant)
            else fem.Constant(self.mesh, PETSc.ScalarType(float(rate)))
        )
        self.F = self.F - r * self._v * ufl.dx
        self._problem = None  # force rebuild with the new source term
        return r

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

    def add_outflow_bc(self, locator, slope: float):
        """Free-drainage outlet: water leaves at the Manning NORMAL-DEPTH discharge.

        ``q_out = SECONDS_PER_DAY * (1/n_man) * d^{5/3} * sqrt(slope)`` (m^2/day per unit
        width), i.e. the friction slope at the outlet is taken as the bed ``slope`` -- the
        standard kinematic/normal-depth outflow condition (vs the natural no-flux boundary,
        which would dam the outlet). Enters the residual as ``+q_out v ds`` (a boundary
        sink). ``outflow_rate()`` reports the integrated discharge across all such outlets.

        ``slope`` must be strictly positive: ``slope = 0`` would silently turn the outlet into
        a no-flux wall (damming the reach) and ``slope < 0`` injects ``sqrt(<0)`` = NaN into the
        residual; both are caller errors, so we reject them up front.
        """
        if slope <= 0.0:
            raise ValueError(
                f"add_outflow_bc requires slope > 0 (normal-depth friction slope); got {slope!r}. "
                "slope=0 dams the outlet; slope<0 gives sqrt(<0)=NaN."
            )
        ds_out = self._boundary_ds(locator)
        d_pos = ufl.max_value(self.d, 0.0)
        q_out = SECONDS_PER_DAY * (1.0 / self.n_man) * d_pos ** (5.0 / 3.0) * ufl.sqrt(slope)
        self.F = self.F + q_out * self._v * ds_out
        self._outflow_forms.append(fem.form(q_out * ds_out))
        self._problem = None
        return None

    def outflow_rate(self) -> float:
        """Total discharge leaving through all outflow boundaries (m^2/day per unit width)."""
        total = 0.0
        for form in self._outflow_forms:
            total += self.mesh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)
        return total

    def _ensure_problem(self) -> None:
        if self._problem is None:
            self._problem = NonlinearProblem(
                self.F,
                self.d,
                bcs=self._bcs,
                petsc_options_prefix="overland_",
                petsc_options=self._petsc_options,
            )

    # -- positivity limiter ---------------------------------------------------
    def _enforce_positivity(self) -> float:
        """Conservatively clip d to >= 0, preserving the lumped water budget.

        The smooth Newton solve can undershoot to small (~cm) negative depths at wet/dry
        fronts. We clip those to zero and rescale the remaining positive water by
        ``oldtotal / posvol`` so total storage (the lumped integral of d) is unchanged to
        machine precision. No-op when there are no negatives. Returns the largest negative
        depth that was clipped (0.0 if none) for diagnostics. Mass-conserving by
        construction (unlike a naive clamp): the rescale factor <= 1 only reduces already-
        positive nodes, never creating new negatives.
        """
        local_min = float(self.d.x.array.min()) if self.d.x.array.size else 0.0
        gmin = self.mesh.comm.allreduce(local_min, op=MPI.MIN)
        if gmin >= 0.0:
            return 0.0
        oldtotal = self.total_water()  # lumped budget BEFORE clipping (includes the negatives)
        np.maximum(self.d.x.array, 0.0, out=self.d.x.array)  # clip negatives -> 0
        self.d.x.scatter_forward()
        posvol = self.total_water()  # positive water after clipping (always >= oldtotal)
        # factor <= 1 rescales the clipped positives back to the pre-clip budget, conserving
        # exactly when oldtotal > 0. DEGENERATE case oldtotal <= 0 (undershoot deficit outweighs
        # ALL positive water -- a numerically pathological near-dry state): we cannot have d >= 0
        # AND a non-positive total, so factor clamps to 0 (dry the domain). Without the clamp,
        # oldtotal/posvol would be negative and FLIP every sign (re-introducing negatives); the
        # earlier `oldtotal > 0` guard avoided that but silently CREATED water by leaving the
        # clipped positives. We instead dry it and TRACK the unavoidable adjustment below.
        factor = (oldtotal / posvol) if posvol > 0.0 else 0.0
        if factor < 0.0:
            factor = 0.0
        if posvol > 0.0:
            self.d.x.array[:] *= factor
            self.d.x.scatter_forward()
        self.clip_mass_adjust += self.total_water() - oldtotal  # 0 normally; > 0 when drying
        return -gmin

    # -- time stepping --------------------------------------------------------
    def step(self, dt: float):
        """Advance one backward-Euler Newton step. Returns (converged: bool, iters).

        After a converged solve the conservative positivity limiter runs before the step
        is accepted, so ``d_n`` (and hence the next step's start) is non-negative and the
        water budget is preserved. An unconverged solve restores the last accepted state
        so callers can cut dt and retry.
        """
        self.dt.value = dt
        self._ensure_problem()
        self._problem.solve()  # updates self.d in place
        snes = self._problem.solver
        converged = snes.getConvergedReason() > 0
        iters = int(snes.getIterationNumber())
        if converged:
            # Record the outflow at the SOLVED state (before the limiter rescales d): this is
            # the residual-consistent discharge that balances the storage change exactly
            # (W^{n+1}-W^n = dt*(r*L - last_outflow)). The limiter conserves total water but
            # perturbs the boundary depth, so post-limiter outflow_rate() would not close the
            # books to machine precision.
            self.last_outflow = self.outflow_rate()
            self.last_clip = self._enforce_positivity()
            self.max_clip_seen = max(self.max_clip_seen, self.last_clip)
            self.d_n.x.array[:] = self.d.x.array  # accept the (positivity-cleaned) step
            self.d_n.x.scatter_forward()
        else:
            self.d.x.array[:] = self.d_n.x.array  # restore last accepted state
            self.d.x.scatter_forward()
        return converged, iters

    def advance(self, t_end, dt, *, dt_min=1e-9, dt_max=None, grow=1.5, cut=0.5,
                target_low=3, target_high=8, shrink=0.7, max_steps=100000):
        """March to ``t_end`` with adaptive backward-Euler steps.

        Cut ``dt`` and retry on a non-converged solve (stiff wet/dry fronts / storm
        peaks). On accepted steps, grow ``dt`` when Newton converges easily
        (``iters <= target_low``) and shrink it on an expensive-but-accepted step
        (``iters >= target_high``). Returns the number of accepted steps. Raises if
        ``dt`` must drop below ``dt_min`` to converge. ``dt_min`` defaults small because
        overland flow is fast (sub-second crossing times) -- the cold-start transient on a
        steep thin film genuinely needs tiny steps before ``dt`` can climb.
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

    # -- erosion-threshold diagnostics (design C.5) ---------------------------
    def velocity(self):
        """Cell-wise (DG0) flow-velocity VECTOR u in SI m/s, for the §G erosion check.

        Manning sheet-flow velocity ``u = -(1/n_man) d^{2/3} grad(H_s) / |grad H_s|^{1/2}``
        (magnitude ``(1/n) d^{2/3} S_f^{1/2}``, directed downslope ``-grad H_s/|grad H_s|``).
        SI m/s (NOT the model's m/day) because erosion thresholds are physical velocities;
        the same slope floor as the residual keeps it finite at zero slope, and the
        ``d^{2/3}`` factor makes it vanish (not blow up) at dry cells -- no d_min needed.
        """
        gdim = self.mesh.geometry.dim
        Vv = fem.functionspace(self.mesh, ("DG", 0, (gdim,)))
        g = ufl.grad(self.z_b + self.d)
        slope_sqrt = (ufl.dot(g, g) + self.eps_S**2) ** 0.25
        d_pos = ufl.max_value(self.d, 0.0)
        u_expr = -(1.0 / self.n_man) * d_pos ** (2.0 / 3.0) * g / slope_sqrt
        u = fem.Function(Vv, name="velocity")
        u.interpolate(fem.Expression(u_expr, Vv.element.interpolation_points))
        return u

    def bed_shear(self, rho: float = 1000.0, g_accel: float = 9.81):
        """Cell-wise (DG0) bed shear stress ``tau = rho g d S_f`` in Pa (``S_f=|grad H_s|``).

        The scalar the erosion check differences against a threshold (alongside |u|). Pa is
        time-unit-independent (S_f is dimensionless), so no day<->second conversion here.
        """
        Vs = fem.functionspace(self.mesh, ("DG", 0))
        g = ufl.grad(self.z_b + self.d)
        Sf = ufl.sqrt(ufl.dot(g, g))
        d_pos = ufl.max_value(self.d, 0.0)
        tau_expr = rho * g_accel * d_pos * Sf
        tau = fem.Function(Vs, name="bed_shear")
        tau.interpolate(fem.Expression(tau_expr, Vs.element.interpolation_points))
        return tau
