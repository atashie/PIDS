"""Sequential operator-split surface<->subsurface coupling (B4+B5).

PRODUCTION extraction of the validated spike ``scratch/overland_split_spike.py`` ``win`` mode
(``run_case_win``; conservation proven to ~5e-12 and parent-verified -- sanity note
``validation/sanity/overland_split_spike__2026-06-22.md`` CONSERVATION-PROOF section). The
sequential-split alternative to the monolithic ``CoupledProblem``: implicit Richards is solved ALONE
each step, with the surface water redistributed by an explicit Manning rate-limited routing sweep
(``overland_routing.route_excess``), coupled through a water-LEVEL handoff. It STRUCTURALLY removes
both pathologies the monolithic Manning schemes hit on convergent / stiff-clay storms: the
[psi,d,lam] Newton's dt-collapse and the kinematic-advection sawtooth (the stiff Richards solve never
shares a Jacobian with the surface routing).

THE DESIGN (locked from the verified spike; do NOT redesign):
  * The surface store IS the pond ``max(psi,0)`` carried IN psi continuously (this is the validated
    ``RichardsProblem.add_ponding_bc`` physics: ``rain_eff = infiltration + d(pond)/dt``). The soil
    draws the carried pond at its natural Darcy rate -- genuinely SELF-LIMITING (takes only what it
    can absorb; the un-infiltrated remainder raises the pond). NO co-solved surface unknown, NO hard
    Dirichlet pin (which over-infiltrates high-K sand), NO post-solve write-back of psi (which is a
    bad Newton restart and dt-collapses). All three alternatives were tried in the spike and FAILED;
    pond-in-psi is what stays robust.
  * The LATERAL routing enters the Richards solve as a Neumann SOURCE (run-on +, run-off -), iterated
    to a Picard fixed point. The source is omega-relaxed with omega=relax (default 1.0 = no standing
    under-relaxation; halved on a failed inner solve as the ROBUSTNESS fallback), and the routing sweep
    is sub-stepped ``route_substeps`` times per step (default 4) to resolve the intra-step travel -- the
    transport-RATE calibration (see the class docstring + ``validation/sanity/
    overland_transport_calibration__2026-06-23.md``). A single under-relaxed sweep (the spike's original
    omega~=0.5, rs=1) throttles transport ~40-50x; it conserves either way.
  * The conserved ledger is ``total = int theta dV + int max(psi,0) ds_top`` -- the surface pond is a
    REAL stored quantity NOT in ``int theta`` (theta is flat at theta_s for psi>=h_s, so a ponded
    node's pond depth is carried by the boundary pond-storage term, not the volume integral).

THE LOAD-BEARING FIXES (reproduced EXACTLY -- these are what make it conserve / run):
  1. MATCHED QUADRATURE (the conservation fix, 18% -> 5e-12): the pond-storage / rain / lateral-source
     surface terms AND the pond ledger use the SAME lumped VERTEX quadrature as the routing's
     ``sum d_i A_i``. A degree-8 ``ds`` integrates ``max(psi,0)`` differently and reintroduces the
     leak. (Pinned: ``int max(psi,0) ds_top == sum_i d_i A_i`` to ~1e-12.)
  2. PRESSURE-HEAD BC = ponded DEPTH, not z_b+depth. The mesh top is FLAT; z_b is the routing
     topography ONLY (it enters the surface head ``H = z_b + d`` in the routing, never the Richards
     pressure BC -- the pond is just ``max(psi,0)``, no elevation term).
  3. QUADRATURE CAP: the Richards bulk residual is built with ``quadrature_degree=8`` (FFCX auto
     degree on van Genuchten makes the first 3-D solve ~55 s vs ~1 s capped).
  4. BAND dt-controller in ``advance()``: ``it<=4 -> dt*1.4``, ``it>=12 -> dt*0.7``, else hold (a
     naive grow/shrink soft-collapses dt during the ponding-onset transient where the 'basic' Newton
     legitimately needs ~8-10 iters).
  5. 'basic' SNES linesearch for the standalone Richards path (bt/cp stall at SNES -5 on ponding onset).
  6. Non-singular buffer (a fixture concern, not code): the near-impermeable thin slab singularizes
     once fully saturated in the unconfined (no-Ss) Richards -- keep a deep unsaturated buffer.

CONSERVATION IS OMEGA-INDEPENDENT (key structural fact): the converged residual gives, per step,
``d(int theta + int pond) = rain*area*dt + (sum lat_i A_i)*dt - drain*dt = rain - omega*outflow -
drain``, and we book ``cum_outflow += omega*outflow`` -> the balance closes to solver tolerance for
ANY omega. omega only sets the lateral transport RATE; the un-applied (1-omega) routing is a bounded
lag in psi's pond (tracked as ``cum_handoff_imbalance``), NOT a leak.

Public surface mirrors ``CoupledProblem`` where sensible: ``set_initial_condition``,
``set_topography``, ``add_rain``, ``add_outflow_bc``, ``add_drainage_bc``, ``step``, ``advance``,
``total_water`` / ``surface_water`` / ``soil_water``, the ledger (``cum_rain``, ``cum_outflow``,
``cum_drainage``, ``cum_handoff_imbalance``) + ``balance()``. The B6 sinks ``add_interior_drain``
(subsurface tile drain) + ``add_surface_inlet`` (surface grate) + the per-sink split
(``sink_rates`` / ``last_sinks`` / ``cum_sinks``) are wired in (F2 evaluation-state contract on the
class). Embedded-feature exchange (``add_embedded_exchange`` / ``WellIndexExchange``) stays DEFERRED
pending the unified-feature redesign.

Restriction: serial-only (the top-facet routing graph is not ownership-aware -- guarded loudly).

Design / validation: docs/plans/2026-06-22-overland-flow-sequential-coupling-decision.md;
validation/sanity/overland_split_spike__2026-06-22.md.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from .richards import RichardsProblem, richards_bulk_residual
from .overland_edge_kernel import build_top_facet_edge_graph
from .overland_routing import build_adjacency, node_widths, route_excess

SECONDS_PER_DAY = 86400.0


class SequentialCoupledProblem:
    """Sequential operator-split surface<->subsurface coupling on a (3-D / 2-D) host mesh.

    Implicit Richards solved ALONE each step (pond carried IN psi as ``max(psi,0)``, self-limiting),
    with the surface water redistributed by an explicit Manning rate-limited routing sweep injected
    as an under-relaxed Neumann SOURCE, iterated to a Picard fixed point. The PRODUCTION extraction
    of the validated spike ``run_case_win`` (conservation ~5e-12).

    THE F2 SINK-EVALUATION-STATE MODEL (a stated contract -- where each sink is evaluated within the
    operator split, so no sink is left to "whatever state happens to be current"):
      * SUBSURFACE sinks -- the GHB boundary drain (``add_drainage_bc``) and the interior tile drain
        (``add_interior_drain``) -- are evaluated INSIDE the implicit Richards solve: they enter the
        Richards residual (``+ q * v * ds`` / ``+ q_vol * v * dx``), so the Newton resolves them
        against the SOLVED psi each step, and their booked rate is that residual term assembled at the
        solved psi. This is the same state the monolithic ``CoupledProblem`` evaluates them at.
      * SURFACE sinks -- the Manning outlet (``add_outflow_bc``) and the grate inlet
        (``add_surface_inlet``) -- are evaluated DURING / AFTER the routing sweep on the POST-Richards
        ponded field ``max(psi,0)``: the outlet books its off-domain discharge in the routing sweep,
        and the inlet removes ponded depth in the surface update AFTER the Richards solve accepts
        (NOT as a co-solved ``ds_top`` term inside the Newton -- there is no surface unknown here).
        Their booked rate is read on that post-Richards pond.
    Per-sink accounting (``last_sinks`` / ``cum_sinks`` keyed by ``'ghb'`` / ``'interior_drain'`` /
    ``'surface_inlet'``, plus ``sink_rates()``) splits the total; the flat sum equals
    ``last_drainage`` / ``cum_drainage``.

    DEFERRED: embedded-feature exchange (``add_embedded_exchange`` / ``WellIndexExchange``) is NOT
    provided -- the embedded-feature API is being replaced by a separate redesign (the unified
    feature work); it will be wired in once that lands.

    TRANSPORT-RATE CALIBRATION (record ``validation/sanity/overland_transport_calibration__2026-06-23``):
    a SINGLE Manning sweep per Richards step under-resolves the intra-step surface travel and throttles
    lateral transport ~40-50x vs the resolved ``CoupledProblem(overland_scheme="upwind")`` reference.
    The DOMINANT fix is ``route_substeps`` (nsub sweeps each over dt/nsub) -- NOT omega (omega 0.5->1.0
    buys only ~2.4x). ``route_substeps=4`` matches the resolved reference drain timing (1.0x);
    ``route_substeps >= 8`` OVERSHOOTS (the fully sub-stepped explicit Manning is a kinematic wave that
    outruns the reference diffusion-wave), so 4 is the optimum, not "more is better". omega is now
    1.0 by default (no standing under-relaxation -- the right transport setting); under-relaxation is
    the ROBUSTNESS FALLBACK only (``step`` resets omega=relax each step, then HALVES it on a failed
    inner solve before cutting dt). Conservation is omega/substep-INDEPENDENT at ~1e-12 throughout
    (a pure accuracy knob; the spike's CONSERVATION PROOF holds because each sub-sweep telescopes).

    ORTHOGONAL FRAGILITY (not the transport question): a near-saturated column with a ~zero pond can
    dt-collapse the standalone-Richards path on the no-Ss near-saturation singularity -- carry the pond
    as a COMFORTABLY-POSITIVE head (as the validated cases do), not a near-zero one.

    Parameters
    ----------
    mesh : the subsurface host (FLAT top at z=ztop; topography is carried by ``z_b`` via
        ``set_topography``, never the Richards pressure BC).
    soil : a ``VanGenuchten`` (or duck-typed layered soil exposing ``theta_ufl``, ``K_ufl``, ``Ks``).
    n_man : Manning roughness for the lateral routing (SI s.m^-1/3).
    picard_iters : max Picard inner iterates per step (route -> source -> solve).
    relax : the source under-relaxation omega (default 1.0 = no standing under-relaxation; ``step``
        resets to this each step and HALVES it on a failed inner solve as the robustness fallback).
    route_substeps : Manning sub-sweeps per Richards step (default 4 = the resolved-reference transport
        match; >= 8 overshoots). The lateral-transport-RATE knob; conservation-neutral. Must be >= 1.
    quadrature_degree : the Darcy-volume (and GHB) integration-degree cap (van Genuchten fractional
        powers; auto degree balloons ~1000x on 3-D tets).
    """

    # 'basic' SNES linesearch (full Newton, no linesearch) for the standalone Richards ponding path:
    # bt/cp STALL at SNES -5 on the ponding-onset step (spike solver_probe). LU/MUMPS direct solve is
    # robust on the verification meshes (same as RichardsProblem); override for large production runs.
    _DEFAULT_PETSC_OPTIONS = {
        **RichardsProblem._DEFAULT_PETSC_OPTIONS,
        "snes_linesearch_type": "basic",
    }

    def __init__(self, mesh, soil, *, n_man: float = 0.05, picard_iters: int = 4,
                 relax: float = 1.0, route_substeps: int = 4, eps_head: float = 1e-9,
                 quadrature_degree: int = 8, degree: int = 1, lumped: bool = True,
                 petsc_options=None):
        if mesh.comm.size > 1:
            raise NotImplementedError(
                "SequentialCoupledProblem is SERIAL-ONLY: the top-facet routing graph "
                "(build_top_facet_edge_graph + the descending-head sweep) is not ownership-aware. "
                f"Got mesh.comm.size = {mesh.comm.size}. Run on one MPI rank.")
        if degree != 1:
            raise NotImplementedError(
                "SequentialCoupledProblem requires degree=1 (P1): the routing graph identifies a top "
                "vertex with its dof and the pond ledger is vertex-lumped (the matched-quadrature "
                f"conservation fix). Got degree = {degree}.")
        if int(route_substeps) < 1:
            raise ValueError(
                f"route_substeps must be >= 1 (Manning sub-sweeps per Richards step); "
                f"got {route_substeps!r}.")
        self.mesh = mesh
        self.soil = soil
        self.n_man = float(n_man)
        self.picard_iters = int(picard_iters)
        self.relax = float(relax)
        self.route_substeps = int(route_substeps)
        # head-DROP floor for the routing sweep [m]: a node treats a neighbour as a downslope receiver
        # only when (H_i - H_j) > eps_head. Without it, ULP-level head noise (~1e-16*depth) on a FLAT
        # lake invents spurious downslope directions whose Manning cap, amplified across the
        # descending-head cascade (each receiver's H rises by what it gets -> a real gradient out of
        # nothing), DRAINS the lake (a classic flat-water instability; route_excess's bare s>0 is fine
        # on real slopes but not at rest). 1e-9 m is far below any physical pond gradient (mm-scale
        # drops >> it, so the validated sloped cases are unchanged) yet far above the float noise ->
        # lake-at-rest is EXACTLY stationary. Same role as the upwind scheme's eps_H/eps_S floors.
        self.eps_head = float(eps_head)
        self._quad_degree = int(quadrature_degree)
        gdim = mesh.geometry.dim
        self._zaxis = gdim - 1

        # the standalone vertical solver the split reuses (the validated Richards physics). 'basic'
        # linesearch + the degree-8 Darcy cap (rebuild self.F at the cap, as run_case_win does).
        petsc = dict(petsc_options or self._DEFAULT_PETSC_OPTIONS)
        rp = RichardsProblem(mesh, soil, degree=degree, lumped=lumped, petsc_options=petsc)
        rp.F = richards_bulk_residual(rp.psi, rp.psi_n, rp._v, soil, rp.dt, rp.e_g,
                                      dx_storage=rp._dx_storage, quadrature_degree=self._quad_degree)
        self._rp = rp
        self.psi = rp.psi          # alias the carried field (pond = max(psi,0))
        self.psi_n = rp.psi_n

        # top (max-elevation) surface facets.
        zc = rp.V.tabulate_dof_coordinates()[:, self._zaxis]
        self._ztop = float(zc.max()) if zc.size else -np.inf
        self._coords = rp.V.tabulate_dof_coordinates()
        self._n_dofs = rp.V.dofmap.index_map.size_local
        self._top_dofs_arr = np.where(np.isclose(zc, self._ztop))[0].astype(np.int64)

        # bed elevation z_b for the lateral routing (default FLAT; set via set_topography). The
        # routing surface head is H = z_b + d; z_b NEVER enters the Richards pressure BC.
        self.z_b = np.zeros(self._n_dofs, dtype=np.float64)

        # rain influx [m/day] (a Constant on ds_top; drive .value for a hyetograph) -- 0 until add_rain.
        self._rain_c = fem.Constant(mesh, PETSc.ScalarType(0.0))
        # P1 lateral run-on/off source [m/day] (lagged, nonzero only on top dofs).
        self._lat_src = fem.Function(rp.V, name="lat_src")

        # deferred surface/subsurface sink registrations (forms (re)built by _finalize_forms).
        self._outlets: list = []   # (outlet_dofs, slope) per add_outflow_bc
        self._drains: list = []    # (drain_facets, conductance, external_head) per add_drainage_bc
        # B6 sinks. INTERIOR tile drains (SUBSURFACE, F2: in the Richards solve) -- volumetric DG-0 sink
        # terms re-woven into rp.F by _finalize_forms (mirror CoupledProblem). SURFACE grate inlets
        # (SURFACE, F2: on the post-Richards pond) -- removed in the surface update, NOT a residual term.
        self._V0 = None                   # lazy DG-0 space for the cell-sharp interior-drain indicator
        self._interior_drains: list = []  # (cells, conductance_density, drain_head, eps_act) per add
        self._inlets: list = []           # (inlet_dofs, intake_coeff) per add_surface_inlet (top dofs)

        # the surface routing graph (built lazily once topography + sinks are known, in _ensure_built).
        self._built = False
        self._A_i = None
        self._adj = None
        self._W = None
        self._outlet_mask = None
        self._outlet_slope_node = None
        self._pond_ledger = None   # int max(psi,0) ds_top on the VERTEX measure (== sum d_i A_i)
        self._lat_src_ledger = None  # int lat_src ds_top (== sum lat_i A_i; the source consistency check)
        self._drain_forms: list = []          # compiled GHB scalar rate forms (degree-8 ds), per drain
        self._interior_forms: list = []       # compiled interior-drain scalar rate forms (plain dx)

        # -- ledger (B5) ------------------------------------------------------
        self.cum_rain = 0.0
        self.cum_outflow = 0.0
        self.cum_drainage = 0.0
        # per-sink accounting (B6), keyed by kind, each list in add-order; the flat sum equals
        # last_drainage / cum_drainage (mirror CoupledProblem). 'ghb' = boundary GHB drains, evaluated
        # IN the Richards solve; 'interior_drain' = volumetric tile drains, also IN the Richards solve;
        # 'surface_inlet' = grate intakes, evaluated on the POST-Richards pond in the surface update
        # (the F2 model -- see the class docstring). (No 'feature' key: embedded-feature exchange is
        # DEFERRED pending the redesign.)
        self.last_sinks = {"ghb": [], "interior_drain": [], "surface_inlet": []}
        self.cum_sinks = {"ghb": [], "interior_drain": [], "surface_inlet": []}
        # The handoff CONSISTENCY term, tracked NOT hidden: the per-step residual between what the
        # lateral SOURCE actually removed from the system (the assembled int lat_src ds_top, = the
        # under-relaxed routed-pond change the solver saw) and what we BOOKED as outflow
        # (omega*outflow). For the win design these are equal to machine precision -- the routing
        # conserves (m_pre = m_post + outflow) and the source is exactly omega*(routed change) -- so
        # this term is ~0 (a genuine leak-detector: if the booked outflow were ever sourced from soil
        # rather than removed at the surface, the residual would be nonzero). The (1-omega) deferred
        # routing is NOT booked here: it stays in psi's pond (counted in d(total)) and is re-routed
        # next step, so it needs no balance credit -- the closure rain - outflow(applied) - drainage
        # is structurally exact (omega-independent; the spike's CONSERVATION PROOF). balance() carries
        # this ~0 term so the closure statement is complete and the consistency is auditable.
        self.cum_handoff_imbalance = 0.0
        self.last_handoff_resid = 0.0   # the per-step source-vs-booking residual (~0; tracked)
        self._w0 = None            # int theta at t=0 (set on first build)
        self._surf0 = None         # int max(psi,0) ds_top at t=0
        # per-step diagnostics (mirrors CoupledProblem's last_* accounting style).
        self.last_outflow = 0.0
        self.last_drainage = 0.0
        self.last_routing_resid = 0.0
        self.last_reason = 0
        self.last_fnorm = np.nan
        self.last_omega = self.relax
        # honest reject gate (mirror CoupledProblem): book a reason-4 (SNES stagnation) inner solve
        # only if |F| is below this absolute bar; dirty stalls become honest rejections (dt cut), so
        # the books never absorb an unconverged residual. In assembled-residual units (override per
        # problem if its floor differs; mis-set LOW fails loudly as a reject, never a silent booking).
        self.stall_accept_fnorm = 3e-6
        # FALSIFICATION hook (mirrors the spike's PIDS_WIN_LEAK): deliberately mis-book outflow by
        # this fraction to prove the conservation close is a genuine detector, not a tautology. 0 in
        # production (a no-op). The conservation test asserts a 10% setting breaks the balance by ~10%.
        self.outflow_leak_frac = 0.0
        self._t = 0.0

    # -- problem setup --------------------------------------------------------
    def set_initial_condition(self, psi_expr) -> None:
        """Set the subsurface IC ``psi_expr`` (the pond, if any, is its positive part at the top)."""
        self._rp.set_initial_condition(psi_expr)
        self._built = False   # the IC may change the pond -> re-snapshot the ledger baseline on build

    def set_topography(self, expr) -> None:
        """Set the surface bed elevation z_b(x) for the lateral routing (default flat z_b=0).

        z_b carries the routing surface head ``H = z_b + d`` ONLY; it NEVER enters the Richards
        pressure BC (the mesh top is flat; the pond is ``max(psi,0)``, no elevation term).
        """
        zb = fem.Function(self._rp.V)
        zb.interpolate(expr)
        self.z_b = zb.x.array.astype(np.float64).copy()
        self._built = False

    def add_rain(self, rate):
        """Rainfall onto the surface pond (m/day, positive = in). Returns the Constant.

        Drive ``.value`` on the returned Constant for a time-varying hyetograph (e.g. rain.value = 0
        for recession). Calling again REPLACES the source.
        """
        r = rate if isinstance(rate, fem.Constant) else fem.Constant(
            self.mesh, PETSc.ScalarType(float(rate)))
        self._rain_c = r
        self._built = False
        return r

    def add_outflow_bc(self, locator, slope: float):
        """Free-drainage surface outlet: ponded water leaves at the Manning normal-depth discharge.

        The routing sweep books the off-domain outflow at the located top-surface nodes using
        ``slope`` as the outlet pseudo-receiver hydraulic slope (``weight = sqrt(slope)``; the Manning
        cap ``Vcap`` applies). Same (locator, slope) API shape as ``CoupledProblem.add_outflow_bc``;
        here the outlet acts in the explicit routing (not a UFL boundary term). ``slope`` must be
        strictly positive (slope=0 dams the outlet; slope<0 gives sqrt(<0)=NaN).
        """
        if slope <= 0.0:
            raise ValueError(
                f"add_outflow_bc requires slope > 0 (normal-depth friction slope); got {slope!r}. "
                "slope=0 dams the outlet; slope<0 gives sqrt(<0)=NaN.")
        self._outlets.append((locator, float(slope)))
        self._built = False
        return None

    def add_drainage_bc(self, locator, conductance, external_head):
        """General-head (Cauchy / MODFLOW GHB) subsurface drainage on a boundary.

        Relative-permeability-weighted outward Darcy flux ``q_n = conductance * kr(psi) * (psi + z -
        external_head)``, ``kr = K(psi)/Ks`` (self-limits as the boundary dries -- exactly the
        ``RichardsProblem.add_drainage_bc`` / ``CoupledProblem.add_drainage_bc`` physics). Enters the
        Richards residual as ``+ q_n * v * ds`` on the located facets and is recorded into
        ``cum_drainage`` / the balance. Drain boundaries must be disjoint from the top surface and
        from each other (they share ONE facet meshtags with the top in ``_finalize_forms``).
        """
        if conductance < 0.0:
            raise ValueError(f"add_drainage_bc requires conductance >= 0 [1/day]; got {conductance!r}")
        self._drains.append((locator, float(conductance), float(external_head)))
        self._built = False
        return None

    def add_interior_drain(self, locator, conductance_density, drain_head, *,
                           eps_act: float = 1e-3):
        """Volumetric OUTFLOW-ONLY tile drain (MODFLOW-DRN analogue) over the CELLS matched by
        ``locator`` -- a SUBSURFACE sink. INTERIOR horizons allowed (e.g. a pipe on a clay interface)
        which the exterior-facet GHB (``add_drainage_bc``) cannot represent. **Mirrors
        ``CoupledProblem.add_interior_drain`` faithfully:**

            q_vol = conductance_density * kr(psi) * pos(psi + z - drain_head)      [1/day]
            pos(u) = 1/2 (u + sqrt(u^2 + eps_act^2))                               (C^inf smooth max)

        Per the F2 model (class docstring) this is a SUBSURFACE sink: it enters the Richards residual
        as ``+ q_vol * v * dx`` (degree-8 ``dx``) and so is evaluated INSIDE the implicit Richards
        solve, exactly like the GHB. ``conductance_density`` [1/(day*m)]; ``drain_head`` = the pipe
        invert elevation (drains where ``psi+z`` exceeds it). OUTFLOW-ONLY by construction (an
        air-filled pipe never injects; the smooth max leaks <= density*eps_act/2 per unit volume AT
        activation only, ``eps_act`` > 0). ``kr = K(psi)/Ks`` self-limits unsaturated extraction.
        Localized by a cell-sharp DG-0 indicator on plain ``dx`` (no meshtags; NOTE ``locator`` is an
        ALL-vertex predicate -- the band bounds must strictly contain the intended cell vertices).
        Wired into ``cum_drainage`` / the balance; the per-sink split is ``sink_rates()`` /
        ``cum_sinks['interior_drain']``. Returns the conductance ``Constant`` (drive ``.value`` to
        ramp). Promoted from the signed-off dual-drain illustration (commit 63145e2).
        """
        C_const = conductance_density if isinstance(conductance_density, fem.Constant) else \
            fem.Constant(self.mesh, PETSc.ScalarType(float(conductance_density)))
        if not float(C_const.value) >= 0.0:   # NaN-proof (NaN fails every comparison)
            raise ValueError("add_interior_drain requires conductance_density >= 0 [1/(day*m)]; "
                             f"got {conductance_density!r}")
        if not eps_act > 0.0:
            raise ValueError(f"add_interior_drain requires eps_act > 0 [m]; got {eps_act!r}")
        tdim = self.mesh.topology.dim
        cells = np.sort(dmesh.locate_entities(self.mesh, tdim, locator)).astype(np.int32)
        if self.mesh.comm.allreduce(int(cells.size), op=MPI.SUM) == 0:
            raise ValueError("add_interior_drain: the locator matched no cell (it is an ALL-vertex "
                             "predicate -- widen the bounds to strictly contain the cell vertices).")
        for prior, _C, _He, _eps in self._interior_drains:
            ov = int(np.intersect1d(cells, prior).size)
            if self.mesh.comm.allreduce(ov, op=MPI.SUM) > 0:
                raise ValueError("add_interior_drain: locator overlaps an existing interior drain "
                                 "(an ambiguous double sink on shared cells).")
        self._interior_drains.append((cells, C_const, float(drain_head), float(eps_act)))
        self._built = False
        return C_const

    def add_surface_inlet(self, locator, intake_coeff):
        """Grate / catch-basin intake on the PONDED depth over a top-surface footprint -- a SURFACE
        sink: ``q_in = intake_coeff * d`` [m/day], ``d = max(psi,0)``, removes surface water only.
        **Mirrors ``CoupledProblem.add_surface_inlet`` in SPIRIT but adapted to the sequential
        design:** there is no co-solved surface unknown ``d`` here, so per the F2 model (class
        docstring) the inlet is evaluated on the POST-Richards pond and removed DURING THE SURFACE
        UPDATE (like the Manning outlet), NOT as a ``ds_top`` term inside the Richards Newton.

        Each accepted step removes ``intake_coeff * d_i * A_i * dt`` of ponded volume at each footprint
        node (clamped to the available pond ``d_i * A_i`` so ``psi`` never crosses 0), lowering the
        carried pond ``max(psi,0)``; the volume ACTUALLY removed (= the unclamped intake when
        ``intake_coeff*dt < 1``, the normal regime) is booked into ``cum_drainage`` and the
        ``surface_inlet`` per-sink split, on the SAME lumped VERTEX measure as the pond ledger -- so
        ``balance()`` stays closed UNCONDITIONALLY (matched-quadrature consistent, even if the clamp
        bites a huge coefficient). The footprint = the TOP dofs matched by ``locator`` (non-top matches
        ignored; raises if NO top node matches). Returns the intake ``Constant`` (drive ``.value``).
        Serial-only (the top-dof gather is not ownership-aware;
        the engine's standing serial waiver applies). Promoted from the dual-drain illustration (63145e2).
        """
        C_const = intake_coeff if isinstance(intake_coeff, fem.Constant) else \
            fem.Constant(self.mesh, PETSc.ScalarType(float(intake_coeff)))
        if not float(C_const.value) >= 0.0:   # NaN-proof (NaN fails every comparison)
            raise ValueError(f"add_surface_inlet requires intake_coeff >= 0 [1/day]; "
                             f"got {intake_coeff!r}")
        top = self._top_dofs_arr
        sel = top[locator(self._coords[top].T)]               # the matched TOP dofs (footprint)
        if self.mesh.comm.allreduce(int(sel.size), op=MPI.SUM) == 0:
            raise ValueError("add_surface_inlet: the locator matched no TOP-surface node -- the "
                             "inlet footprint must lie on the land surface.")
        for prior, _C in self._inlets:
            ov = int(np.intersect1d(sel, prior).size)
            if self.mesh.comm.allreduce(ov, op=MPI.SUM) > 0:
                raise ValueError("add_surface_inlet: footprint overlaps an existing inlet.")
        self._inlets.append((np.sort(sel).astype(np.int64), C_const))
        self._built = False
        return C_const

    # -- form assembly --------------------------------------------------------
    def _dg0_indicator(self, cells, name: str):
        """Cell-sharp DG-0 indicator (1 on ``cells``, 0 elsewhere) for an interior-drain sink, with no
        meshtags (sidesteps the one-subdomain_data-per-integral-type rule). Quadrature-exact for the
        selected cell set (constant per cell); the realized sink geometry is quantized to WHOLE cells.
        Mirrors ``CoupledProblem._dg0_indicator``."""
        if self._V0 is None:
            self._V0 = fem.functionspace(self.mesh, ("DG", 0))
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, tdim)  # locate_dofs_topological needs (tdim,tdim)
        chi = fem.Function(self._V0, name=name)
        chi.x.array[:] = 0.0
        dofs = fem.locate_dofs_topological(self._V0, tdim, cells)
        chi.x.array[dofs] = 1.0
        chi.x.scatter_forward()
        return chi

    def _finalize_forms(self) -> None:
        """(Re)build the Richards residual surface/GHB terms + the routing graph.

        Assembles the ponding-storage/rain/lateral-source terms (on the lumped VERTEX measure -- the
        load-bearing quadrature match) and the GHB drainage terms (degree-8 ds) onto ONE unified facet
        meshtags (top=1, drains=2+) -- DOLFINx 0.10 requires a single subdomain_data per integral type
        per form, and ``RichardsProblem`` builds a fresh meshtags per BC call, so we weave them here
        directly (the spike's ``run_case_win`` plumbing). Builds the top-facet routing graph + outlet
        masks. Snapshots the ledger baseline (``int theta``, ``int max(psi,0) ds_top``) at the IC.
        """
        rp = self._rp
        msh = self.mesh
        fdim = msh.topology.dim - 1
        msh.topology.create_connectivity(fdim, msh.topology.dim)

        # rebuild F from the capped bulk (drop any previously-woven surface/GHB terms on re-finalize).
        rp.F = richards_bulk_residual(rp.psi, rp.psi_n, rp._v, self.soil, rp.dt, rp.e_g,
                                      dx_storage=rp._dx_storage, quadrature_degree=self._quad_degree)

        top_facets = np.sort(dmesh.locate_entities_boundary(
            msh, fdim, lambda x: np.isclose(x[self._zaxis], self._ztop))).astype(np.int32)
        all_f = [top_facets]
        all_t = [np.full(top_facets.size, 1, dtype=np.int32)]
        drain_specs = []
        for k, (loc, C, H) in enumerate(self._drains, start=2):
            df = np.sort(dmesh.locate_entities_boundary(msh, fdim, loc)).astype(np.int32)
            if msh.comm.allreduce(int(df.size), op=MPI.SUM) == 0:
                raise ValueError("add_drainage_bc: the locator matched no boundary facet (silent no-op).")
            ov = int(np.intersect1d(df, np.concatenate(all_f)).size)
            if msh.comm.allreduce(ov, op=MPI.SUM) > 0:
                raise ValueError(
                    "add_drainage_bc: locator overlaps the top surface or an existing drainage "
                    "boundary; GHB boundaries must be disjoint (they share one facet meshtags).")
            all_f.append(df)
            all_t.append(np.full(df.size, k, dtype=np.int32))
            drain_specs.append((k, C, H))
        ents = np.concatenate(all_f)
        tags = np.concatenate(all_t)
        order = np.argsort(ents)
        ft = dmesh.meshtags(msh, fdim, ents[order], tags[order])

        # the GHB facets ride a degree-8 ds; the ponding surface terms + ledger ride the lumped
        # VERTEX ds (matched to the routing store's sum d_i A_i -- the conservation fix).
        ds = ufl.Measure("ds", domain=msh, subdomain_data=ft,
                         metadata={"quadrature_degree": self._quad_degree})
        ds_top_v = ufl.Measure("ds", domain=msh, subdomain_data=ft,
                               metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})(1)
        z = ufl.SpatialCoordinate(msh)[self._zaxis]

        # ponding storage + rain influx + lateral source (run-on +, run-off -) on the top facet,
        # all on the vertex measure. The pond is max(psi,0) -- DEPTH only, no z_b (flat top).
        pond = ufl.max_value(rp.psi, 0.0)
        pond_n = ufl.max_value(rp.psi_n, 0.0)
        rp.F = rp.F + ((pond - pond_n) / rp.dt) * rp._v * ds_top_v \
            - self._rain_c * rp._v * ds_top_v \
            - self._lat_src * rp._v * ds_top_v
        # the surface-pond ledger (lumped vertex quadrature -> sum over top nodes of max(psi,0)*A_i,
        # bit-consistent with the routing store's sum d_i A_i; pinned by the matched-quadrature test).
        self._pond_ledger = fem.form(ufl.max_value(rp.psi, 0.0) * ds_top_v)
        # the lateral-source integral int lat_src ds_top_v == sum_i lat_i A_i [m^3/day] -- exactly what
        # the residual injected. Used for the handoff CONSISTENCY residual (it must equal -omega*outflow
        # to machine precision; a nonzero residual would mean booked outflow came from soil, not surface).
        self._lat_src_ledger = fem.form(self._lat_src * ds_top_v)
        # GHB drainage terms on tags 2.. (degree-8 ds); recorded for the cum_drainage books.
        self._drain_forms = []
        for (k, C, H) in drain_specs:
            kr = self.soil.K_ufl(rp.psi) / self.soil.Ks
            q_n = C * kr * (rp.psi + z - H)
            rp.F = rp.F + q_n * rp._v * ds(k)
            self._drain_forms.append(fem.form(q_n * ds(k)))
        # INTERIOR tile drains (B6, SUBSURFACE -- F2: evaluated IN the Richards solve): volumetric
        # OUTFLOW-ONLY DG-0 sink on plain degree-8 dx, mirroring CoupledProblem.add_interior_drain.
        dxq = ufl.dx(metadata={"quadrature_degree": self._quad_degree})
        self._interior_forms = []
        for j, (cells, C, H, eps_act) in enumerate(self._interior_drains):
            chi = self._dg0_indicator(cells, f"chi_drain{j}")
            u = rp.psi + z - H
            pos = 0.5 * (u + ufl.sqrt(u * u + eps_act * eps_act))   # C^inf smooth max (outflow-only)
            kr = self.soil.K_ufl(rp.psi) / self.soil.Ks             # self-limits unsaturated extraction
            q_vol = C * kr * pos * chi
            rp.F = rp.F + q_vol * rp._v * dxq
            self._interior_forms.append(fem.form(q_vol * dxq))
        rp._problem = None   # force a NonlinearProblem rebuild with the new F

        # build the top-facet routing graph (edges/lengths/transmissibility/areas) + adjacency/widths.
        edges, L_e, T_e, A_i = build_top_facet_edge_graph(rp.V, msh, top_facets)
        self._A_i = A_i
        self._adj = build_adjacency(edges, L_e, self._n_dofs)
        self._W = node_widths(edges, L_e, T_e, self._n_dofs)
        self._top_area = float(A_i.sum())

        # outlet masks + per-node slope (route_excess takes a scalar slope; we use a per-node max,
        # consistent with the spike's multi-outlet handling, applied via a per-node-slope sweep).
        self._outlet_mask = np.zeros(self._n_dofs, dtype=bool)
        self._outlet_slope_node = np.zeros(self._n_dofs, dtype=np.float64)
        top_dofs = self._top_dofs_arr
        for (loc, slope) in self._outlets:
            sel = top_dofs[loc(self._coords[top_dofs].T)]
            self._outlet_mask[sel] = True
            self._outlet_slope_node[sel] = np.maximum(self._outlet_slope_node[sel], slope)

        # snapshot the conserved-ledger baseline (int theta + int max(psi,0) ds_top) ONCE, on the FIRST
        # build, and NEVER re-take on a later rebuild (the one-shot fix): a setup mutator called AFTER
        # stepping flips _built=False and re-finalizes here, but re-snapshotting the baseline at the
        # already-advanced state would collapse d(total) and silently corrupt balance(). The baseline
        # is the IC by construction; set_initial_condition is the only legitimate way to move it (it
        # runs before any step), so we tie the one-shot to "_w0 is None" -- a fresh problem or a
        # never-built one. (A re-IC after a step is a misuse the ledger cannot represent anyway.)
        if self._w0 is None:
            self._w0 = rp.total_water()
            self._surf0 = self._surf_pond()
        self._built = True

    def _ensure_built(self) -> None:
        if not self._built:
            self._finalize_forms()

    # -- routing sweep (per-node outlet slope; conserving) --------------------
    def _route(self, d, dt):
        """Manning routing over ``dt``, sub-stepped ``route_substeps`` times (the transport-RATE knob).

        Runs ``self.route_substeps`` descending-head sweeps (``_route_once``) each over ``dt/nsub``,
        recomputing receivers from the LIVE depth between sub-sweeps and ACCUMULATING the off-domain
        outflow -- the calibration prototype ``_install_route_substeps``. A single sweep advances the
        Manning cascade only ~one-to-a-few cells per Richards step, UNDER-resolving the intra-step
        surface travel and throttling transport ~40-50x vs the resolved upwind reference; sub-stepping
        marches the full intra-step distance and closes the gap (rs=4 = the match; record
        ``validation/sanity/overland_transport_calibration__2026-06-23.md``). Conservation is
        sub-step-INDEPENDENT: each sub-sweep telescopes (``sum d_i A_i + outflow`` conserved to
        machine precision), so the accumulated ``(d_new, outflow)`` conserves over the whole ``dt``,
        and ``step`` still books ``cum_outflow += omega*outflow`` exactly. Returns ``(d_new, outflow)``;
        ``d`` is never mutated.
        """
        nsub = self.route_substeps
        if nsub <= 1:
            return self._route_once(d, dt)
        d_cur = d
        outflow = 0.0
        hsub = dt / nsub
        for _ in range(nsub):
            d_cur, of = self._route_once(d_cur, hsub)
            outflow += of
        return d_cur, outflow

    def _route_once(self, d, dt):
        """ONE Manning rate-limited descending-head routing sweep with PER-NODE outlet slopes.

        Wraps the validated ``overland_routing.route_excess`` law but lets each outlet node use its
        OWN slope (``_outlet_slope_node``) -- ``route_excess`` takes a single scalar outlet slope,
        whereas a multi-outlet problem (e.g. a downslope edge AND a channel-mouth outlet at different
        bed slopes) needs per-node values. Bit-identical to ``route_excess`` when all outlet slopes
        are equal; conserves ``sum_i d_i A_i + outflow`` to machine precision (telescoping). Returns
        ``(d_new, outflow)``; ``d`` is never mutated.
        """
        A_i, z_b, adj, W = self._A_i, self.z_b, self._adj, self._W
        top_dofs, outlet_mask, oslope = self._top_dofs_arr, self._outlet_mask, self._outlet_slope_node
        eps_head = self.eps_head
        d_new = d.copy()
        outflow = 0.0
        Cman = SECONDS_PER_DAY / self.n_man
        order = top_dofs[np.argsort(-(z_b + d_new)[top_dofs])]
        for i in order:
            i = int(i)
            di = d_new[i]
            if di <= 0.0:
                continue
            Hi = z_b[i] + di
            recv_j, recv_w = [], []
            smax = 0.0
            for (j, L) in adj[i]:
                dH = Hi - (z_b[j] + d_new[j])
                if dH > eps_head:                  # head-drop floor: ignore sub-eps_head (lake-at-rest)
                    s = dH / L
                    recv_j.append(j)
                    recv_w.append(np.sqrt(s))
                    if s > smax:
                        smax = s
            out_w = 0.0
            os_ = oslope[i]
            if outlet_mask[i] and os_ > 0.0 and di > eps_head:
                out_w = np.sqrt(os_)
                if os_ > smax:
                    smax = os_
            wsum = float(np.sum(recv_w)) + out_w
            if wsum <= 0.0 or smax <= 0.0:
                continue
            Vcap = Cman * di ** (5.0 / 3.0) * np.sqrt(smax) * W[i] * dt
            Vout = min(di * A_i[i], Vcap)
            if Vout <= 0.0:
                continue
            d_new[i] -= Vout / A_i[i]
            if out_w > 0.0:
                outflow += Vout * (out_w / wsum)
            for k, j in enumerate(recv_j):
                d_new[j] += (Vout * (recv_w[k] / wsum)) / A_i[j]
        return d_new, outflow

    def _surf_pond(self) -> float:
        """int max(psi,0) ds_top on the lumped VERTEX measure (== sum_i d_i A_i; the surface store)."""
        return self.mesh.comm.allreduce(fem.assemble_scalar(self._pond_ledger), op=MPI.SUM)

    # -- time stepping --------------------------------------------------------
    def step(self, dt: float):
        """Advance one sequential-split backward-Euler step. Returns ``(converged, iters)``.

        The Picard loop (route -> under-relaxed lateral source -> implicit Richards solve), reproduced
        from the verified ``run_case_win``:
          1. route the CURRENT pond ``d_cur = max(psi_entry,0)`` over dt (``_route``, sub-stepped
             ``route_substeps`` times) -> lateral source ``lat = omega*(d_routed - d_cur)/dt`` over the
             WHOLE dt, held fixed within the Richards solve.
          2. solve Richards ALONE with the pond-storage term + rain influx + lat source (self-limiting;
             the soil draws the carried pond at its natural Darcy rate). On a non-converged inner
             solve, HALVE omega (the robustness fallback; omega resets to ``relax`` at step entry)
             before reporting failure.
          3. accept: psi carries the redistributed pond; book ``cum_outflow += omega*outflow`` and the
             deferred ``(1-omega)*outflow`` into ``cum_handoff_imbalance``.
        On failure the caller (or ``advance``) cuts dt and retries; the entry state is restored.
        Conservation is omega-INDEPENDENT (see the module docstring). The honest reject gate books a
        SNES reason-4 (stagnation) inner solve only when |F| <= ``stall_accept_fnorm``.
        """
        self._ensure_built()
        rp = self._rp
        rp.dt.value = dt
        top_dofs, A_i = self._top_dofs_arr, self._A_i

        psi_entry = rp.psi.x.array.copy()
        pond_entry = self._surf_pond()
        omega = self.relax
        ok = False
        of = 0.0
        it = 0
        reason = 0
        fnorm = np.nan
        d_cur = d_routed = None
        for _pic in range(self.picard_iters):
            rp.psi.x.array[:] = psi_entry
            rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry
            rp.psi_n.x.scatter_forward()
            d_cur = np.maximum(psi_entry, 0.0)
            d_routed, of = self._route(d_cur, dt)
            lat = np.zeros(self._n_dofs, dtype=np.float64)
            lat[top_dofs] = omega * (d_routed[top_dofs] - d_cur[top_dofs]) / dt
            self._lat_src.x.array[:] = lat
            self._lat_src.x.scatter_forward()
            rp._ensure_problem()
            rp._problem.solve()
            snes = rp._problem.solver
            reason = int(snes.getConvergedReason())
            it = int(snes.getIterationNumber())
            fnorm = float(snes.getFunctionNorm())
            # accept a clean convergence (reason > 0), but only book a reason-4 stagnation if the
            # residual is genuinely small (honest gate); otherwise treat as a failed inner solve.
            conv = reason > 0 and (reason != 4 or fnorm <= self.stall_accept_fnorm)
            if not conv:
                omega *= 0.5
                if omega < 1e-4:
                    break
                continue
            ok = True
            break

        self.last_reason = reason
        self.last_fnorm = fnorm
        self.last_omega = omega
        if not ok:
            # restore the entry state so the caller can cut dt and retry (retry-safe).
            rp.psi.x.array[:] = psi_entry
            rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry
            rp.psi_n.x.scatter_forward()
            return False, it

        # accept: psi carries the redistributed pond (the SOLVED Richards state). Read the SUBSURFACE
        # sink rates (GHB + interior drain) NOW, on the solved psi -- BEFORE the surface inlet mutates
        # any top dof (F2: subsurface sinks are evaluated at the solved psi the Newton resolved them
        # against; assembling them after the surface update would perturb a drain band that reaches the
        # top facet). These are the residual terms the Newton balanced, so they are the books-consistent
        # rates (same convention as CoupledProblem reading sinks at the solved state).
        ghb_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                     for f in self._drain_forms]
        interior_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                          for f in self._interior_forms]

        # SURFACE-INLET removal (B6, F2: a SURFACE sink on the POST-Richards pond) happens HERE, on the
        # solved field, BEFORE psi_n := psi -- like the Manning outlet, it removes ponded water in the
        # surface update (not a residual term). Each footprint node loses q_in = intake_coeff*d depth,
        # i.e. intake_coeff*d_i*A_i*dt of volume, clamped to the available pond d_i*A_i so psi never
        # crosses 0 (the carried pond stays >= 0 and the lumped ledger drops by exactly the booked
        # volume). The rate is read on the post-Richards pond (the state the inlet sees).
        inlet_rates = []          # per-inlet rate [m^3/day] at the post-Richards pond (for the books)
        psi_arr = rp.psi.x.array
        for (dofs, C_const) in self._inlets:
            d_post = np.maximum(psi_arr[dofs], 0.0)
            coeff = float(C_const.value)
            # remove depth, clamped per-node to the available pond (depth basis: psi never below 0).
            remove_depth = np.minimum(coeff * d_post * dt, d_post)
            psi_arr[dofs] -= remove_depth
            # BOOK the volume ACTUALLY removed (sum_i remove_depth_i*A_i)/dt, not the unclamped
            # intake_coeff*d rate: in the normal regime (intake_coeff*dt < 1) these are identical, but
            # booking the removed volume keeps balance() closed EXACTLY even if a huge coefficient
            # would otherwise drain a node in one step (the clamp bites). The ledger drops by exactly
            # this volume, so cum_drainage tracks it -- conservation is unconditional.
            inlet_rates.append(float(np.sum(remove_depth * A_i[dofs])) / dt if dt > 0 else 0.0)
        if self._inlets:
            rp.psi.x.scatter_forward()

        # psi_n := psi (post-inlet): the removed pond must NOT reappear as next step's entry pond.
        rp.psi_n.x.array[:] = rp.psi.x.array
        rp.psi_n.x.scatter_forward()

        # outflow APPLIED this step = the omega-scaled routed share (the reference's exact booking).
        # The deferred (1-omega) routing stays in psi's pond (counted in d(total)) and is re-routed
        # next step -> the closure rain - outflow(applied) - drainage is structurally exact (the
        # spike's omega-INDEPENDENT CONSERVATION PROOF), so the (1-omega) part needs NO balance credit.
        # FALSIFICATION hook scales only the BOOKED outflow by (1+leak) to prove the close is a real
        # detector (the handoff consistency term below uses the APPLIED outflow, so it does NOT absorb
        # the injected leak -> the balance breaks by the full leak fraction).
        of_applied = omega * of
        of_booked = of_applied * (1.0 + self.outflow_leak_frac)
        self.last_outflow = of_applied
        self.cum_outflow += of_booked
        # HANDOFF CONSISTENCY (tracked NOT hidden): the residual between what the lateral SOURCE
        # actually removed (assembled int lat_src ds_top = sum lat_i A_i, over dt) and the APPLIED
        # outflow. Routing conserves (m_pre = m_post + outflow) and lat = omega*(routed change), so
        # sum lat_i A_i*dt == -omega*outflow to machine precision -> this residual is ~0 (it would be
        # nonzero only if the booked outflow were sourced from soil rather than removed at the surface,
        # the hybrid's old 18% leak). Carried in balance() so the closure is complete + auditable.
        lat_int = self.mesh.comm.allreduce(fem.assemble_scalar(self._lat_src_ledger), op=MPI.SUM)
        self.last_handoff_resid = lat_int * dt + of_applied
        self.cum_handoff_imbalance += self.last_handoff_resid

        # drainage accounting (B6), per-sink split in last_sinks/cum_sinks; the flat sum is
        # last_drainage / cum_drainage. F2: the GHB + interior-drain rates were read on the solved psi
        # ABOVE (subsurface, pre-inlet); the surface inlet is the SURFACE rate read on the post-Richards
        # pond. Lists AUTO-EXTEND (mirror CoupledProblem) so a sink added after stepping gets a slot.
        total_rate = 0.0
        sink_sources = (("ghb", ghb_rates),
                        ("interior_drain", interior_rates),
                        ("surface_inlet", inlet_rates))
        for kind, rates in sink_sources:
            while len(self.last_sinks[kind]) < len(rates):
                self.last_sinks[kind].append(0.0)
                self.cum_sinks[kind].append(0.0)
            for i, r in enumerate(rates):
                self.last_sinks[kind][i] = r
                self.cum_sinks[kind][i] += dt * r
                total_rate += r
        self.last_drainage = total_rate
        self.cum_drainage += dt * total_rate

        # rain influx over dt (the surface area is the lumped top area sum_i A_i).
        self.cum_rain += float(self._rain_c.value) * self._top_area * dt

        # routing-store conservation residual this step (sum d_i A_i before/after + outflow).
        m_pre = float(np.sum(d_cur[top_dofs] * A_i[top_dofs]))
        m_post = float(np.sum(d_routed[top_dofs] * A_i[top_dofs]))
        self.last_routing_resid = abs(m_pre - (m_post + of))

        self._t += dt
        return True, it

    def advance(self, t_end, dt, *, storm_dur=None, storm_rain=None, dt_max=0.03, dt_min=1e-11,
                ctrl_low=4, ctrl_high=12, grow=1.4, shrink=0.7, cut=0.5, max_steps=200000):
        """March to ``t_end`` with the BAND dt-controller (fix #4), optionally with a storm hyetograph.

        The band controller (``it<=ctrl_low -> dt*grow``, ``it>=ctrl_high -> dt*shrink``, else hold)
        with ``ctrl_high`` ABOVE the ponding-onset transient's iteration spike keeps dt stable where a
        naive grow/shrink soft-collapses it during the ponding onset. On a non-converged step, cut dt
        and retry; raise if dt must drop below ``dt_min``.

        If ``storm_dur`` and ``storm_rain`` (a rate, m/day) are given, the rain Constant is driven to
        ``storm_rain`` for ``t < storm_dur`` and to 0 after (and a step is clipped to land exactly on
        ``storm_dur``) -- so a single ``advance`` call runs a storm-then-recession. ``add_rain`` must
        have been called first. Returns the number of accepted steps.
        """
        self._ensure_built()
        if storm_dur is not None and storm_rain is None:
            raise ValueError("advance: pass storm_rain (the storm rain rate, m/day) with storm_dur.")
        t = 0.0
        nstep = 0
        while t < t_end - 1e-12:
            if nstep >= max_steps:
                raise RuntimeError(f"advance exceeded max_steps={max_steps} at t={t:.4g}")
            h = min(dt, t_end - t)
            if storm_dur is not None:
                if t < storm_dur - 1e-12 and t + h > storm_dur:
                    h = storm_dur - t                       # land exactly on the storm end
                self._rain_c.value = storm_rain if t < storm_dur - 1e-12 else 0.0
            conv, it = self.step(h)
            if not conv:
                dt = h * cut
                if dt < dt_min:
                    raise RuntimeError(f"advance: dt {dt:.2e} < dt_min at t={t:.4g} (not converging)")
                continue
            t += h
            nstep += 1
            if it <= ctrl_low:
                dt = min(dt * grow, dt_max)
            elif it >= ctrl_high:
                dt = dt * shrink
        return nstep

    # -- diagnostics / ledger -------------------------------------------------
    def soil_water(self) -> float:
        """Total stored SOIL water = int theta(psi) dV (the lumped storage quadrature)."""
        return self._rp.total_water()

    def surface_water(self) -> float:
        """Total surface PONDED water = int max(psi,0) ds_top (the lumped vertex measure == sum d_i A_i)."""
        self._ensure_built()
        return self._surf_pond()

    def total_water(self) -> float:
        """The conserved total = int theta dV + int max(psi,0) ds_top (soil + surface pond)."""
        return self.soil_water() + self.surface_water()

    def sink_rates(self) -> dict:
        """Per-sink drainage rates from the LAST accepted step, keyed by kind ('ghb',
        'interior_drain', 'surface_inlet'), each a list in add-order; the flat sum equals
        ``last_drainage`` and the flat sum of ``cum_sinks`` equals ``cum_drainage`` (the per-sink
        split is exhaustive). Returns the BOOKED per-step rates (a snapshot of ``last_sinks``), NOT a
        live re-assembly: unlike ``CoupledProblem.sink_rates`` (which re-assembles UFL forms at the
        current state), the SURFACE INLET here is an APPLIED removal on the post-Richards pond -- after
        the step the live pond is already post-removal, so a live re-read would not reproduce the
        booked figure. The last-booked snapshot is the residual-consistent rate the balance uses."""
        return {kind: list(vals) for kind, vals in self.last_sinks.items()}

    def balance(self) -> float:
        """Global mass-balance closure (B5): the residual of

            d(total) = cum_rain - cum_outflow - cum_drainage + cum_handoff_imbalance

        where d(total) = (int theta + int pond)_now - _start, every term measured INDEPENDENTLY (no
        fudge bucket; the handoff term is the bounded, tracked time-lag, not a fitted credit). The
        validated win design closes this to ~5e-12 (omega-independent); a real leak shows up at full
        magnitude (pinned by the falsification test). Returns the SIGNED residual.
        """
        self._ensure_built()
        dtotal = self.total_water() - (self._w0 + self._surf0)
        return dtotal - (self.cum_rain - self.cum_outflow - self.cum_drainage
                         + self.cum_handoff_imbalance)
