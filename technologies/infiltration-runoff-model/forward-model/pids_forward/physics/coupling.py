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
from .overland import overland_conveyance, SECONDS_PER_DAY


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
        # stol pinned explicitly (PETSc default): CONVERGED_SNORM_RELATIVE (4) is the STAGNATION
        # verdict -- the iterate stopped moving. Legitimate at the residual floor (states whose
        # assembly floor sits above atol; stol=0 turns those into max-it grinds + dt death
        # spirals), but it certifies NOTHING about balance: a stalled line search far from the
        # root also returns 4, and booking it injects the unbalanced residual into the mass
        # budget (B6 convergent-V P0). step() books reason 4 only below the absolute bar
        # ``stall_accept_fnorm``; dirty stalls become honest rejections (dt cut).
        "snes_stol": 1e-8,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    def __init__(self, mesh, soil, *, ell_c: float | None = None, eps_ncp: float = 1e-4,
                 n_man: float = 0.05, eps_S: float = 1e-3,
                 degree: int = 1, lumped: bool = True, quadrature_degree: int = 8,
                 petsc_options=None):
        self.mesh = mesh
        self.soil = soil
        self.eps_ncp = float(eps_ncp)
        self.n_man = float(n_man)   # Manning roughness for the lateral overland flux (SI s.m^-1/3)
        self.eps_S = float(eps_S)   # diffusion-wave slope floor
        # CAP the nonlinear (van Genuchten / Manning / Kirchhoff) integration degree. FFCX's auto
        # estimate balloons on these fractional-power integrands (residual ~26, Jacobian ~41); on 3-D
        # TETRAHEDRA that is O(deg^3) quadrature points -> ~1000x slower assembly (a 6x6x5 box smoke
        # step: 16.5 s -> 0.08 s; benchmark scratch/m3_3d_perf_probe2.py). The cap is sized by the
        # Darcy VOLUME term (the highest-auto-degree term); the surface/NCP/Kirchhoff/drainage terms
        # are lower-degree and generously covered. Accuracy vs the auto degree (Codex/agent audit
        # 2026-06-07): BIT-IDENTICAL (~1e-14) for SMOOTH cells, including ponded/supply-limited states;
        # for a cell whose psi STRADDLES the air-entry head it deviates O(1e-4) at degree 6 and ~1.6e-5
        # at degree 8 (-> default 8, free at these sizes); a one-cell unsaturated->ponded WETTING FRONT
        # has an in-cell kink (se_ufl's air-entry min/max) that NO polynomial quadrature resolves
        # (handled instead by lumped storage + small dt, not by raising the degree). End-to-end this
        # does not measurably move the solution or Newton convergence (verified: dpsi < 1e-3, identical
        # iteration counts cap=8 vs cap=30 on a dry-start heavy-rain ponding solve). Applied to the
        # Darcy volume dx, the ds_top surface, and the drainage GHB facets; the vertex-lumped storage
        # measure is unaffected. Pinned by test_3d_quadrature_cap_accuracy; re-audit (scratch/
        # m3_quad_cap_audit.py) if a future surface term raises the ds_top auto-degree.
        self._quad_degree = int(quadrature_degree)
        assert mesh.geometry.dim == mesh.topology.dim, (
            "CoupledProblem assumes a full-dimensional host (gdim == tdim); the _zaxis (gdim-1) and "
            "_codim2_dim (tdim-2) conventions diverge on an embedded manifold mesh.")
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
        self.z_b = fem.Function(self.Vd, name="z_b")  # surface topography for lateral overland (default flat)

        e_g = np.zeros(gdim, dtype=PETSc.ScalarType)
        e_g[-1] = 1.0
        self.e_g = fem.Constant(mesh, e_g)
        self.dt = fem.Constant(mesh, PETSc.ScalarType(1.0))

        # top (max-elevation) surface facets + the surface coupling measure.
        zc = self.Vpsi.tabulate_dof_coordinates()[:, self._zaxis]
        self._ztop = mesh.comm.allreduce(float(zc.max()) if zc.size else -np.inf, op=MPI.MAX)
        fdim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(fdim, mesh.topology.dim)
        self._fdim = fdim
        # codim-2 outlet entity dim: the lateral-outflow outlet is the BOUNDARY of the top surface --
        # a corner VERTEX in the 2-D cross-section (tdim-2 = 0) and a perimeter EDGE/ridge in 3-D
        # (tdim-2 = 1). add_outflow_bc locates entities of this dim; _finalize_forms imposes the
        # discharge on the matching measure (vertex dP in 2-D, ridge dr in 3-D). Create the
        # (codim2, tdim) connectivity locate_entities_boundary needs for the 3-D edge case.
        self._codim2_dim = mesh.topology.dim - 2
        if self._codim2_dim >= 1:
            mesh.topology.create_connectivity(self._codim2_dim, mesh.topology.dim)
        top_facets = np.sort(dmesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[self._zaxis], self._ztop))).astype(np.int32)
        self._top_facets = top_facets
        ft = dmesh.meshtags(mesh, fdim, top_facets, np.ones(top_facets.size, dtype=np.int32))
        self._ds_top = ufl.Measure(
            "ds", domain=mesh, subdomain_data=ft,
            metadata={"quadrature_degree": self._quad_degree})(1)

        if ell_c is None:
            # top-cell half-height = half the spacing between the top two DISTINCT z-levels.
            # ASSUMES a FLAT, LAYERED top (also assumed by the top-facet detection at z=ztop and the
            # off-top pin below). Round first: 2-D/3-D meshes have many nodes per level with O(1e-15)
            # float noise, so a raw np.unique returns near-duplicates and the spacing collapses to ~0
            # (-> k_ex blows up). For warped/sloped/unstructured tops this heuristic is unreliable --
            # GUARDED below to fail LOUDLY (pass ell_c explicitly there). [shore-up #4; per-facet
            # local ℓ_c is the future fix when non-flat tops are supported.]
            zu = np.unique(np.round(zc, 9))
            local = 0.5 * float(zu[-1] - zu[-2]) if zu.size >= 2 else np.inf
            ell_c = mesh.comm.allreduce(local, op=MPI.MIN)  # top-cell half-height
            zmin = mesh.comm.allreduce(float(zc.min()) if zc.size else self._ztop, op=MPI.MIN)
            z_extent = max(self._ztop - zmin, 1e-30)
            if not np.isfinite(ell_c) or ell_c < 1e-6 * z_extent:
                raise ValueError(
                    f"ell_c auto-detection failed (got {ell_c:.2e} vs z-extent {z_extent:.2e}); the "
                    "top-cell-half-height heuristic assumes a flat, layered top -- pass ell_c explicitly.")
        self.ell_c = float(ell_c)
        self._tau_c = self.ell_c / soil.Ks  # NCP unit scaling: τ_c·(flux) has depth units

        self._dx_storage = (
            ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
            if lumped else ufl.dx
        )

        vpsi = ufl.TestFunction(self.Vpsi)
        vd = ufl.TestFunction(self.Vd)
        vlam = ufl.TestFunction(self.Vlam)
        self._vpsi = vpsi
        self._vd = vd
        self._vlam = vlam

        # potential infiltration and the smooth complementarity g = q_pot − λ. SORPTIVE leg (design §D
        # spec 2026-06-06): the cross-film conductance uses the FILM-INTEGRAL (Kirchhoff matric flux
        # potential) of K, q_pot = [∫_{ψ_top}^{d} K(ψ)dψ]/ℓ_c, NOT the dry cell value K(ψ_top) — the
        # surface saturates (K→Ks) when ponded, which the dry value misses (under-infiltrating dry soil
        # ~50× at coarse resolution; benchmark scratch/m3_sorptivity_benchmark.py). kirchhoff_ufl is C¹
        # (no max kink), captures the saturated ψ>0 part via K_ufl's air-entry cap, and reduces to the
        # old form / head continuity as ℓ_c→0. The transient is carried by the Richards solve below.
        q_pot = soil.kirchhoff_ufl(self.psi, self.d) / self.ell_c
        g = q_pot - self.lam

        # ψ block: Richards VOLUME bulk; the surface λ influx (−λ into the soil top) and any subsurface
        # drainage GHB terms are woven onto ONE shared facet meshtags by _build_F_psi (DOLFINx requires a
        # single subdomain_data per integral type per form -- the λ-top and drainage facets must share it).
        self._F_psi_bulk = richards_bulk_residual(
            self.psi, self.psi_n, vpsi, soil, self.dt, self.e_g, dx_storage=self._dx_storage,
            quadrature_degree=self._quad_degree)
        # LATERAL overland: Manning diffusion-wave as a TANGENTIAL-gradient surface PDE on ds_top.
        # grad_T = grad − (grad·n)n is the surface gradient; in 1-D the top facet is a point so
        # grad_T ≡ 0 and this term vanishes (dimension-agnostic). H_s = z_b + d.
        n_vec = ufl.FacetNormal(mesh)
        gT = lambda f: ufl.grad(f) - ufl.dot(ufl.grad(f), n_vec) * n_vec
        # Manning conveyance SHARED with Module 2 (overland_conveyance); the co-located grad_T operator
        # is provably identical to standalone overland (test_2d_overland_operator_matches_module2).
        H_s, K_s = overland_conveyance(self.d, self.z_b, self.n_man, self.eps_S, grad=gT)
        overland_flux = K_s * ufl.dot(gT(H_s), gT(vd)) * self._ds_top
        # d / λ BULK residuals: surface storage + lateral overland flux + sign-paired λ; the NCP regime
        # selector. The vertex diagonal-allocation, rainfall, and outlet terms are added by
        # _finalize_forms (which shares ONE vertex meshtags across all dP integrals -- DOLFINx requires
        # a single subdomain_data per integral type within a form).
        # NOTE (sawtooth, 2026-06-06): the CONSISTENT surface storage/coupling produce a small odd-even
        # oscillation at the advancing thin-film wet/dry front, which the positivity limiter clips to 0
        # (a mm-scale, mass-conserving cosmetic SAWTOOTH confined to the near-dry upslope; the physical
        # downslope ponding is smooth). Mass-lumping the surface storage/λ only halved it; the residual
        # is unstabilized Galerkin advection of the kinematic (slope) flux and needs a stabilized/monotone
        # scheme -- deferred (docs/plans/2026-06-05-module3-realization-ffcx-bug.md §8). Kept consistent
        # here (clean, validated) since lumping did not cleanly resolve it.
        self._F_d_bulk = ((self.d - self.d_n) / self.dt) * vd * self._ds_top \
            + overland_flux \
            + self.lam * vd * self._ds_top
        self._F_lam_bulk = _fischer_burmeister(self.d, self._tau_c * g, self.eps_ncp) * vlam * self._ds_top

        # Pinned interior d/λ vertices: their rows are Dirichlet-pinned to 0 but need a MATRIX DIAGONAL
        # to overwrite (ds_top contributes none for interior rows -> zero pivot, MUMPS INFOG(1)=-3). A
        # tiny VERTEX (dP) integral on EXACTLY these vertices (tag 1, added in _finalize_forms) adds the
        # diagonal slot and is CONSERVATION-NEUTRAL: it touches only the overwritten pinned rows, never
        # the free top rows -> the surface-water balance is structural. (The earlier whole-domain
        # eps_diag*d*vd*dx leaked a ~1e-12 spurious sink into the top rows -- Codex review 2026-06-06.)
        below = lambda x: x[self._zaxis] < self._ztop - 0.25 * self.ell_c
        mesh.topology.create_connectivity(0, mesh.topology.dim)
        self._pinned_verts = np.sort(dmesh.locate_entities(mesh, 0, below)).astype(np.int32)
        self._eps_diag = 1e-10  # magnitude immaterial (these rows are overwritten by the Dirichlet pin)

        # constrain d and λ to the top surface: pin all non-surface DOFs to 0 (same `below` as the
        # diagonal-allocation tag, so every pinned row has a diagonal slot to overwrite).
        self._bcs = [
            fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_geometrical(self.Vd, below), self.Vd),
            fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_geometrical(self.Vlam, below), self.Vlam),
        ]

        self._petsc_options = dict(petsc_options or self._DEFAULT_PETSC_OPTIONS)
        self._problem: NonlinearProblem | None = None
        self._rain = None
        self._outlets: list = []        # (outlet_vertices, slope) per add_outflow_bc call
        self._outflow_forms: list = []  # compiled q_out*dP forms for outflow_rate()
        self.last_clip = 0.0      # largest negative surface depth clipped on the last step
        self.max_clip_seen = 0.0  # largest negative depth clipped over the whole run (heavy clipping
        # => the accepted λ is increasingly stale vs the clipped d; should stay ~mm/cm-small)
        self.clip_mass_adjust = 0.0  # cumulative UNAVOIDABLE surface-mass change from the degenerate
        # (oldtotal<=0) drying branch; 0 in the normal conservative-rescale branch
        self.last_outflow = 0.0   # outlet discharge at the SOLVED state (pre-limiter), for accounting
        self.cum_outflow = 0.0    # cumulative outflow volume ∫ outflow dt over accepted steps
        self.last_reason = 0      # SNES converged reason of the last step() solve (audit trail)
        self.last_fnorm = np.nan  # |F| at that solve's exit: what residual the books accepted
        self.stall_accept_fnorm = 1e-5  # reason-4 (stagnation) bookable only if |F| <= this
        # ABSOLUTE bar: admits the measured legitimate-floor population (<= ~1.2e-6, Tier-1
        # MMS/near-flat) with margin, rejects the stiff-V stalled-line-search population
        # (|F| ~ 1e-5..3e-3, B6 P0). Override per problem if its residual floor differs.
        self._drains: list = []          # (drain_facets, conductance, external_head) per add_drainage_bc
        self._drainage_forms: list = []  # compiled q_n*ds GHB forms (subsurface Darcy/head drainage)
        self.last_drainage = 0.0  # net subsurface drainage at the SOLVED state (+ = out)
        self.cum_drainage = 0.0   # cumulative subsurface drainage volume ∫ drainage dt
        self._build_F_psi()       # ψ block: bulk + λ-top influx (+ drainage GHB) on one facet meshtags
        self._finalize_forms()    # build self.F_d, self.F_lam (+ outflow forms) with the shared dP

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

    def set_topography(self, expr) -> None:
        """Set the surface bed elevation z_b(x) for the lateral overland flux (default flat z_b=0)."""
        self.z_b.interpolate(expr)
        self._problem = None

    def add_dirichlet_psi(self, locator, value) -> None:
        dofs = fem.locate_dofs_geometrical(self.Vpsi, locator)
        self._bcs.append(fem.dirichletbc(PETSc.ScalarType(value), dofs, self.Vpsi))
        self._problem = None

    def add_rain(self, rate):
        """Rainfall onto the surface store (m/day, positive = water in). Returns the Constant.

        Sets the (single) rainfall source; drive ``.value`` on the returned Constant for a time-varying
        hyetograph (e.g. rain.value = 0 for recession). Calling again REPLACES the source.
        """
        r = rate if isinstance(rate, fem.Constant) else fem.Constant(
            self.mesh, PETSc.ScalarType(float(rate)))
        self._rain = r
        self._finalize_forms()
        return r

    def _finalize_forms(self) -> None:
        """(Re)build F_d, F_lam and the outflow forms, weaving in the pin diagonal-allocation and any
        outlets on the correct codim-2 measure.

        The pinned-interior diagonal allocation is ALWAYS a vertex (``dP``, tag 1) integral -- pinned
        d/λ rows are 0-dim dofs in any dimension. The OUTLET is the codim-2 boundary of the surface:
        - 2-D: a VERTEX (tdim-2 = 0). Same integral type as the pin, so DOLFINx's one-subdomain_data-
          per-integral-type rule forces pin + outlets into ONE shared vertex meshtags (tags 1, 2..).
        - 3-D: a ridge EDGE (tdim-2 = 1). A DIFFERENT integral type from the vertex pin, so the
          outlets get their OWN ``ridge`` (``dr``) meshtags (tags 2..) that coexists with the pin's
          vertex meshtags in the same F_d (verified FFCX-safe: scratch/m3_3d_outlet_spike.py). The
          ridge line integral ∫ q_out dr = (per-width q_out)·edge_length is the exact outlet discharge.
        Called at construction and whenever rainfall or outlets change.
        """
        if self._codim2_dim == 0 and self._outlets:
            # 2-D: pin (tag 1) + outlets (tags 2..) share ONE vertex meshtags (same integral type).
            ents = [self._pinned_verts]
            tags = [np.full(self._pinned_verts.size, 1, dtype=np.int32)]
            for k, (overts, _slope) in enumerate(self._outlets, start=2):
                ents.append(overts)
                tags.append(np.full(overts.size, k, dtype=np.int32))
            allv = np.concatenate(ents).astype(np.int32)
            allt = np.concatenate(tags)
            order = np.argsort(allv)
            vt = dmesh.meshtags(self.mesh, 0, allv[order], allt[order])
            dP = ufl.Measure("vertex", domain=self.mesh, subdomain_data=vt)
            out_measure = dP  # outlets ride the same vertex measure
        else:
            # pin alone on a vertex meshtags (2-D with no outlets, 3-D, or degenerate 1-D)...
            vt = dmesh.meshtags(self.mesh, 0, np.sort(self._pinned_verts),
                                np.full(self._pinned_verts.size, 1, dtype=np.int32))
            dP = ufl.Measure("vertex", domain=self.mesh, subdomain_data=vt)
            out_measure = None
            if self._outlets:  # 3-D: outlets on a SEPARATE ridge (codim-2 edge) meshtags
                rents, rtags = [], []
                for k, (oedges, _slope) in enumerate(self._outlets, start=2):
                    rents.append(oedges)
                    rtags.append(np.full(oedges.size, k, dtype=np.int32))
                allr = np.concatenate(rents).astype(np.int32)
                allrt = np.concatenate(rtags)
                order = np.argsort(allr)
                rt = dmesh.meshtags(self.mesh, self._codim2_dim, allr[order], allrt[order])
                out_measure = ufl.Measure("ridge", domain=self.mesh, subdomain_data=rt,
                                          metadata={"quadrature_degree": self._quad_degree})

        # d block: bulk + diagonal-allocation (tag 1) [+ rain] [+ outlets on out_measure].
        F_d = self._F_d_bulk + self._eps_diag * self.d * self._vd * dP(1)
        if self._rain is not None:
            F_d = F_d - self._rain * self._vd * self._ds_top
        self._outflow_forms = []
        for k, (_oents, slope) in enumerate(self._outlets, start=2):
            d_pos = ufl.max_value(self.d, 0.0)
            q_out = SECONDS_PER_DAY * (1.0 / self.n_man) * d_pos ** (5.0 / 3.0) * ufl.sqrt(slope)
            F_d = F_d + q_out * self._vd * out_measure(k)
            self._outflow_forms.append(fem.form(q_out * out_measure(k)))
        self.F_d = F_d
        # λ block: bulk + diagonal-allocation (tag 1).
        self.F_lam = self._F_lam_bulk + self._eps_diag * self.lam * self._vlam * dP(1)
        self._problem = None

    def add_outflow_bc(self, locator, slope: float):
        """Free-drainage surface outlet: ponded water leaves at the Manning NORMAL-DEPTH discharge.

        ``q_out = SECONDS_PER_DAY*(1/n_man)*d^{5/3}*sqrt(slope)`` [m^2/day per unit width] -- the
        friction slope at the outlet taken as the bed ``slope`` (standard kinematic/normal-depth
        outlet, vs the natural no-flux boundary that would dam it). Same (locator, slope) API as the
        standalone Module-2 ``OverlandProblem.add_outflow_bc``.

        REALIZATION A: the surface lives on the host top facets, so the outlet -- the boundary of that
        surface -- is a codim-2 host entity (``tdim-2``). In the 2-D cross-section it is the
        downstream-TOP corner VERTEX, imposed as a single-mesh VERTEX integral (``dP``); in 3-D it is
        the downstream-top EDGE (a perimeter curve), imposed as a codim-2 RIDGE integral (``dr``) so
        that ``∫_edge q_out dr = (per-width q_out)·edge_length`` is the exact line discharge [m^3/day].
        Both are single-mesh, FFCX-native on stock 0.10 (NOT the mixed-dim codim-0 path that blocked
        realization S -- spikes scratch/m3_outflow_spike.py (2-D) + m3_3d_outlet_spike.py (3-D ridge,
        incl. an S-style coexistence guard)) with clean auto-Jacobians. ``locator`` is intersected
        with the top surface (z=ztop) so only surface boundary entities become outlets (below-top d is
        pinned 0 anyway, but the intersection keeps the outlet unambiguous). The outlet entities are
        recorded and woven into F_d by ``_finalize_forms`` (the vertex pin and the 3-D ridge outlet are
        different integral types -> separate meshtags); ``outflow_rate()`` reports the integrated
        discharge across all outlets. (The normalized downstream-band sink on ds_top is the documented
        dimension-agnostic fallback if a future curved/unstructured outlet ever defeats the ridge path.)

        ``slope`` must be strictly positive: ``slope=0`` silently turns the outlet into a no-flux wall
        (damming) and ``slope<0`` injects ``sqrt(<0)``=NaN; both are caller errors, rejected up front.

        MPI caveat (deferred; verification is serial): ``locate_entities_boundary`` at the codim-2 dim
        "will not necessarily mark all exterior entities" in parallel, and ``outflow_rate``'s
        allreduce-SUM could double-count an outlet edge shared across a partition boundary -- WORSE for
        3-D ridges (edges straddle partitions more often than corner vertices). Needs ownership-aware
        tagging + a 2-rank regression before any MPI outflow claim.
        """
        if slope <= 0.0:
            raise ValueError(
                f"add_outflow_bc requires slope > 0 (normal-depth friction slope); got {slope!r}. "
                "slope=0 dams the outlet; slope<0 gives sqrt(<0)=NaN.")
        if self._codim2_dim < 0:
            raise ValueError(
                "add_outflow_bc: no lateral outlet in 1-D -- the surface is a single point (the "
                "vertical ponding store), so there is no codim-2 outlet boundary to drain through.")
        on_top = lambda x: locator(x) & np.isclose(x[self._zaxis], self._ztop)
        oents = np.sort(dmesh.locate_entities_boundary(
            self.mesh, self._codim2_dim, on_top)).astype(np.int32)
        if self.mesh.comm.allreduce(int(oents.size), op=MPI.SUM) == 0:
            raise ValueError(
                "add_outflow_bc: the locator matched no top-surface boundary entity (vertex in 2-D, "
                "edge in 3-D) -- the outlet would be a silent no-op. Check the locator (it is "
                "intersected with the top z=ztop).")
        # guard against outlet-vs-outlet overlap (symmetric with add_drainage_bc): overlapping
        # locators would put the SAME entity under two tags in the shared outlet meshtags, silently
        # dropping/mis-tagging one outlet's discharge and breaking the cum_outflow books.
        for prior_oents, _ in self._outlets:
            ov = int(np.intersect1d(oents, prior_oents).size)
            if self.mesh.comm.allreduce(ov, op=MPI.SUM) > 0:
                raise ValueError(
                    "add_outflow_bc: locator overlaps an existing outlet; outlets must be disjoint "
                    "(they share one codim-2 meshtags in _finalize_forms).")
        self._outlets.append((oents, float(slope)))
        self._finalize_forms()  # rebuild F_d: vertex pin (tag 1) + outlet (vertex dP / ridge dr)
        return None

    def outflow_rate(self) -> float:
        """Total discharge leaving through all surface outlets -- the assembled outlet sink (a TOTAL,
        not per-unit-width). ``q_out`` is the per-width Manning integrand; the measure supplies the
        geometric factor, so the assembled value is m^2/day at the 2-D point outlet (unit-width
        cross-section) and m^3/day at the 3-D edge outlet (∫ q_out dr = per-width q_out · edge_length).
        """
        total = 0.0
        for form in self._outflow_forms:
            total += self.mesh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)
        return total

    def _top_dofs(self, V):
        return fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[self._zaxis], self._ztop))

    def add_drainage_bc(self, locator, conductance, external_head):
        """Subsurface Darcy/head drainage on a boundary (general-head / Cauchy / MODFLOW GHB).

        Relative-permeability-weighted outward Darcy flux ``q_n = conductance·kr(ψ)·(H − external_head)``,
        ``kr = K(ψ)/Ks``, ``H = ψ + z`` (z = elevation), into an external reservoir at head
        ``external_head`` through SATURATED boundary conductance ``conductance`` [1/day]. The kr(ψ) weight
        makes the GHB the physical Darcy flux through a boundary film of conductivity K(ψ): it SELF-LIMITS
        as the boundary dries (a constant-C GHB would over-drain unsaturated soil by driving the boundary
        head down to deliver an unphysical flux -- Codex review 2026-06-07). Saturated boundary (kr=1) →
        standard GHB. Lets the SOIL MATRIX drain -- lateral groundwater outflow (a side), deep percolation
        (the base), or a drain -- distinct from the surface Manning outlet. BIDIRECTIONAL (a high
        external_head makes it an injector); reduces to no-flow (conductance→0) / saturated-Dirichlet head
        (conductance→∞). Enters F_psi as the standard exterior-facet term ``+q_n·v_ψ·ds`` (a codim-1 facet
        integral; no vertex-measure machinery). Records to ``drainage_rate``/``cum_drainage`` for the
        balance Δtotal = cum_rain − cum_outflow − cum_drainage + clip_mass_adjust. Drain boundaries must be
        disjoint from the top surface and from each other (one shared facet meshtags in _build_F_psi).
        """
        C_const = conductance if isinstance(conductance, fem.Constant) else fem.Constant(
            self.mesh, PETSc.ScalarType(float(conductance)))
        if float(C_const.value) < 0.0:
            raise ValueError(f"add_drainage_bc requires conductance >= 0 [1/day]; got {conductance!r}")
        self.mesh.topology.create_connectivity(self._fdim, self.mesh.topology.dim)
        facets = np.sort(dmesh.locate_entities_boundary(self.mesh, self._fdim, locator)).astype(np.int32)
        if self.mesh.comm.allreduce(int(facets.size), op=MPI.SUM) == 0:
            raise ValueError("add_drainage_bc: the locator matched no boundary facet (silent no-op).")
        # guard against double-tagging in the shared F_psi meshtags: a drain may not overlap the top
        # λ-coupling surface or a previously-added drain (Codex review 2026-06-07).
        tagged = [self._top_facets] + [f for f, _C, _He in self._drains]
        for prior in tagged:
            ov = int(np.intersect1d(facets, prior).size)
            if self.mesh.comm.allreduce(ov, op=MPI.SUM) > 0:
                raise ValueError(
                    "add_drainage_bc: locator overlaps the top surface or an existing drainage boundary; "
                    "GHB boundaries must be disjoint (they share one facet meshtags in F_psi)."
                )
        self._drains.append((facets, C_const, float(external_head)))
        self._build_F_psi()   # rebuild F_psi: bulk + λ-top + drainage, sharing ONE facet meshtags
        return C_const        # handle for a time-varying / ramped conductance (drive .value)

    def _build_F_psi(self) -> None:
        """(Re)build F_psi = ψ-volume bulk − λ·v·ds_top + Σ drainage GHB terms, with ALL exterior-facet
        integrals on ONE shared facet meshtags (tag 1 = top λ-coupling; tags 2.. = drainage boundaries)
        -- DOLFINx requires a single subdomain_data per integral type per form. (F_d keeps its own
        _ds_top; the λ term over the SAME top facets cancels F_d's +λ term in the conservation balance.)
        """
        groups = [(1, self._top_facets)]
        for k, (facets, _C, _He) in enumerate(self._drains, start=2):
            groups.append((k, facets))
        allf = np.concatenate([f for _, f in groups]).astype(np.int32)
        allt = np.concatenate([np.full(f.size, t, dtype=np.int32) for t, f in groups])
        order = np.argsort(allf)
        ft = dmesh.meshtags(self.mesh, self._fdim, allf[order], allt[order])
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=ft,
                         metadata={"quadrature_degree": self._quad_degree})
        z = ufl.SpatialCoordinate(self.mesh)[self._zaxis]

        F_psi = self._F_psi_bulk - self.lam * self._vpsi * ds(1)   # λ influx into the soil top
        self._drainage_forms = []
        for k, (_facets, C, He) in enumerate(self._drains, start=2):
            kr = self.soil.K_ufl(self.psi) / self.soil.Ks   # relative perm: self-limits unsaturated drainage
            q_n = C * kr * (self.psi + z - He)
            F_psi = F_psi + q_n * self._vpsi * ds(k)
            self._drainage_forms.append(fem.form(q_n * ds(k)))
        self.F_psi = F_psi
        self._problem = None

    def drainage_rate(self) -> float:
        """Net outward subsurface drainage across all GHB boundaries (+ = out) -- the assembled facet
        flux total: m^2/day in the 2-D unit-width cross-section, m^3/day on a 3-D boundary face."""
        total = 0.0
        for form in self._drainage_forms:
            total += self.mesh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)
        return total

    def _ensure_problem(self) -> None:
        if self._problem is None:
            self._problem = NonlinearProblem(
                [self.F_psi, self.F_d, self.F_lam], [self.psi, self.d, self.lam], bcs=self._bcs,
                petsc_options_prefix="coupled_", petsc_options=self._petsc_options,
                kind="mpi",  # monolithic block AIJ so a single LU factorizes the coupled operator
            )

    # -- positivity limiter ---------------------------------------------------
    def _enforce_positivity(self) -> float:
        """Conservatively clip d>=0 on the surface, preserving the surface budget ∫d ds_top.

        The lateral overland (Manning diffusion-wave) undershoots to small negative d at advancing
        wet/dry fronts (same as standalone Module 2). Left in d_n, those negatives accumulate and
        eventually stall the coupled Newton. Post-step we clip them to 0 and rescale the remaining
        positive surface water by oldtotal/posvol so the surface storage ∫d ds_top is preserved to
        machine precision (no-op when d>=0). The off-top DOFs are pinned to 0 -> unaffected.
        """
        arr = self.d.x.array
        gmin = self.mesh.comm.allreduce(float(arr.min()) if arr.size else 0.0, op=MPI.MIN)
        if gmin >= 0.0:
            return 0.0
        oldtotal = self.surface_water()              # ∫d ds_top including the negatives
        np.maximum(arr, 0.0, out=arr); self.d.x.scatter_forward()
        posvol = self.surface_water()                # positive surface water after clipping
        factor = (oldtotal / posvol) if posvol > 0.0 else 0.0
        if factor < 0.0:
            factor = 0.0
        if posvol > 0.0:
            arr[:] *= factor; self.d.x.scatter_forward()
        # normal rescale branch is conservative (surface_water restored); the degenerate oldtotal<=0
        # branch dries the domain -> track the unavoidable surface-mass change so it is never silent.
        self.clip_mass_adjust += self.surface_water() - oldtotal
        return -gmin

    # -- time stepping --------------------------------------------------------
    def step(self, dt: float):
        """Advance one monolithic backward-Euler Newton step. Returns (converged, iters)."""
        self.dt.value = dt
        self._ensure_problem()
        self._problem.solve()
        snes = self._problem.solver
        self.last_reason = int(snes.getConvergedReason())
        self.last_fnorm = float(snes.getFunctionNorm())
        # bookable = residual-tested (reasons 2/3), or stagnation AT the residual floor: reason 4
        # only certifies the iterate stopped moving. With a numerically tiny leftover residual that
        # is the legitimate floor exit; far from balance it is a stalled line search whose booking
        # injects the unbalanced residual into the mass budget (B6 P0) -- those must be honest
        # rejections so the caller cuts dt.
        converged = self.last_reason > 0 and (
            self.last_reason != 4 or self.last_fnorm <= self.stall_accept_fnorm)
        iters = int(snes.getIterationNumber())
        if converged:
            # Record the outlet discharge at the SOLVED state (before the limiter rescales d): this
            # is the residual-consistent discharge that balances the storage change exactly
            # (W^{n+1}-W^n = dt*(rain - last_outflow)). The limiter conserves total water but perturbs
            # the boundary depth, so a post-limiter outflow_rate() would not close the books.
            self.last_outflow = self.outflow_rate()
            self.cum_outflow += dt * self.last_outflow
            # subsurface Darcy/head drainage (on ψ; unaffected by the d-limiter): Δtotal = cum_rain
            # − cum_outflow − cum_drainage.
            self.last_drainage = self.drainage_rate()
            self.cum_drainage += dt * self.last_drainage
            self.last_clip = self._enforce_positivity()  # keep d_n >= 0 (lateral-overland undershoots)
            self.max_clip_seen = max(self.max_clip_seen, self.last_clip)
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

    def advance(self, t_end, dt, *, dt_min=1e-9, dt_max=None, grow=1.5, cut=0.5,
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
