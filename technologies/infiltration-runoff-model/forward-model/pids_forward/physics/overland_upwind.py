"""O1 upwind-mobility two-point edge-flux overland solver (Convergent-flow P1, Part B).

A standalone Module-2 spike of a MONOTONE, WELL-BALANCED diffusion-wave overland scheme, built
to fix the convergent-flow regime where the validated galerkin ``OverlandProblem`` develops a
sawtooth (the "Defect A" Galerkin instability on convergence lines, B6 P0). This class does NOT
touch ``overland.py`` -- that galerkin path stays the MMS/regression reference; this is a
separate scheme on its own class so the two can be compared head-to-head.

P1 SPIKE VERDICT (B6, 2026-06-16): the scheme PASSES every parent-plan §5 P1 gate. On the canonical
2-D tilted-V it FIXES the convergent-flow pathology -- the sawtooth is GONE (oscillation RMS 0.013%,
mesh-convergent), the dt-pin is LIFTED (median dt at DT_MAX, 0 rejected steps, 0.4 s vs the galerkin
V's 39.5 h / 60k rejections), the scheme is monotone (NO limiter; the V's measured run-min depth was
-0.0) and conservative (books gap <=1e-13*cum_rain), and field-scale (SCALE=0.1) it resolves where the
galerkin path is under-resolved. CONSERVATION/EQUILIBRIUM vs ACCURACY -- the honest framing (B5b):
the LUMPED plateau == 1.000*Q_eq is a mass-conservation identity (the lumped outflow_rate() shares the
outlet sink's weights, so a converged steady field FORCES Q_out == rain*area == Q_eq for any shape;
machine-tight books, ParFlow-comparable) -- it confirms CONSERVATION, NOT discharge accuracy (the
galerkin lumped plateau is ~1.0 for the same reason). The genuine accuracy measure is the CONSISTENT
ds-integral discharge: on the idealized KINK V ~0.85*Q_eq (a measure-zero 1-cell-d^(5/3)-spike channel
artifact a smooth P1 functional can't integrate -- shared by BOTH schemes, following the Manning
thin-channel normal-depth law exactly, NOT an upwind-flux defect), and on a RESOLVED finite-width swale
(the real PIDS use case) it converges UPWARD to ~0.99 (<=~1% error). The one characterized POSITIVITY
caveat is the sub-mm geometry-dependent mild-front undershoot (Positivity section below). Full verdict
(gate-by-gate) + the corrected accuracy framing + P2 readiness/risks = parent plan
`docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` §8.7 (the B5b-reconciled close; §8.6
is the pre-B5b verdict, accuracy line corrected to point here). NEXT = P2 (productionize the same edge
scheme on the realization-A top-facet ridge graph of ``CoupledProblem``; galerkin limiter demoted to a
tripwire assert).

The scheme (per the P1 plan, docs/plans/2026-06-14-overland-convergent-flow-P1.md, Part B).
Surface head ``H = z_b + d``. The lateral conveyance is assembled as a finite-volume two-point
flux on an EDGE GRAPH: for each edge ``e = (i, j)`` with transmissibility ``T_e``,

    Q_e = T_e * M(d_up) * (H_i - H_j)                                  [T_e = 1/L_e, unit width]
    M(d) = SECONDS_PER_DAY * max(d, 0)^{5/3}
           / ( n_man * ( ((H_i - H_j)/L_e)^2 + eps_S^2 )^{1/4} )       [Manning mobility]
    d_up = w * d_i + (1 - w) * d_j,   w = 0.5 * (1 + tanh((H_i - H_j)/eps_H))   [C1 upwind]

so the mobility is evaluated at the UPSTREAM (higher-head) depth -- a smoothed, C1 selection
(tanh) that keeps the finite-difference Newton Jacobian well-behaved (the hard ``sign`` selector
is not UFL-expressible nor FD-friendly). The Manning conveyance form + ``n_man`` + ``eps_S``
match ``overland.overland_conveyance`` (the slope is taken from the EDGE head-drop here, vs the
galerkin elementwise ``grad H``). ``SECONDS_PER_DAY`` converts the SI Manning conveyance to
m^2/day so depths/timesteps stay in model units (m/day), exactly as in ``overland.py``.

Node residual (lumped storage; backward Euler):

    R_i = (d_i - d_n,i) * A_i / dt + sum_{e: i in e} (+/- Q_e) - r * A_i

with edge sign ``+Q_e`` for the i-row and ``-Q_e`` for the j-row (telescoping). ``A_i`` is the
nodal control volume (FV dual: half the sum of the lengths of the cells incident to node i, so
``sum_i A_i = domain length`` = the ``total_water`` measure). ``r`` is the uniform rain
(m/day). Two structural properties fall out of this construction:

  * WELL-BALANCED (lake-at-rest): a uniform surface head makes every edge head-drop
    ``H_i - H_j = 0`` -- to ROUNDOFF in ``z_b + d`` (the head sums two separately-interpolated
    fields, so the cancellation is ~machine-eps, ~1e-16, NOT bit-exact; this is independent of
    ``eps_S``/``eps_H``). The conveyance amplifies that ~1e-16 head-drop into a ~1e-8 m^2/day
    residual flux, which the Newton solve then drives below tolerance -- so a still pond holds its
    DEPTH to machine precision and cannot drain spuriously down the bed slope. (The fluxes are not
    *structurally* zero to machine precision; well-balancedness comes from differencing H on edges,
    and the exactness of the held depth comes from the solve converging that residual away -- vs
    the galerkin UFL ``grad(z_b + d)``.)
  * CONSERVATIVE (discrete, structural at the root): each interior edge adds ``+Q_e`` to one row
    and ``-Q_e`` to the other, so summing all node residuals telescopes the entire flux network to
    zero. AT A RESIDUAL-CONVERGED ROOT (R=0) this is exact:
    ``sum_i (d_i - d_n,i) A_i = dt * r * sum_i A_i`` (closed domain, no rain => total water
    invariant). NB B1 books ANY ``getConvergedReason() > 0`` (incl. reason-4 SNORM
    stagnation-at-floor), so on a STALLED step the balance holds only to the exit ``||F||``, not
    exactly -- the ``stall_accept_fnorm`` floor gate ``OverlandProblem`` uses for exactly this
    (the B6 P0 hazard of booking a dirty reason-4 stall) is deferred to B2 with the conservation test.

Index space (Decision 1). The edge graph lives in the P1 *dof* index space: edges come from
``V.dofmap.cell_dofs(c)`` (1-D: each cell -> one edge ``(dof_a, dof_b)``). The SNES unknown
vector IS the P1 dof array ``d.x.array``, so the residual, ``total_water()``, and the
IC/topography interpolation all share ONE index space -- no vertex<->dof remap (more robust than
raw topology vertex ids). For P1 elements the dof coordinates equal the vertex coordinates, so
edge lengths come straight from ``V.tabulate_dof_coordinates()`` indexed by the cell dofs.

Solver (Decisions 3, 5). ``UpwindOverlandProblem`` drives a ``petsc4py.PETSc.SNES`` directly
(no DOLFINx ``NonlinearProblem`` -- the upwind selection is not UFL-expressible). The Jacobian
is a finite-difference Jacobian -- PETSc's internal coloring FD, selected via ``snes.setUseFD(True)``
with J's sparsity preset from the edge graph -- and a direct LU solve, i.e. true Newton + direct
solve, the most robust choice for this stiff degenerate diffusion and trivially cheap at 1-D sizes
(a hand/analytic Jacobian is a later optimization, the ParFlow ``UseJacobian False`` precedent
being that the monotone flux, not an exact J, is what gives robustness). Boundaries are
NO-FLUX (closed) by default: a boundary node simply has fewer incident edges, so its residual
omits the missing flux -- the natural no-flux condition. B2 adds an OPTIONAL Manning normal-depth
``add_outflow_bc`` outlet (a free-drainage sink at located boundary nodes; see that method).

Positivity (B2, the monotone-scheme payoff). The upwind scheme carries NO positivity limiter (no
clip/rescale machinery -- contrast the galerkin ``OverlandProblem._enforce_positivity``), so when
it holds ``d >= 0`` that is the monotone construction, not a post-step clip. CONDITIONALITY, stated
honestly: the smoothed (tanh) selector is monotone STRICTLY only when it is SHARP relative to the
front head-drop -- i.e. when the wet/dry-front head-drop greatly exceeds ``eps_H``. The selector
width that best balances this against Newton/accuracy was fixed empirically in B3 (see
"Regularization (Decision 4)" below). At the chosen ``eps_H=1e-3``: the STEEP 5%-slope slump holds
strict ``d >= -1e-12`` (head-drop >> eps_H -- exactly where the galerkin path must engage its clip);
the MILD 2%-slope regime (the B5 valley) is **GEOMETRY-DEPENDENT SUB-MILLIMETER** -- strictly
positive on smooth mounds, but adversarially sharp 2% mounds undershoot up to ~0.9 mm (controller
adjudication sweep 2026-06-16, 24 mild-2% geometries: worst run-min -0.93 mm; B2's earlier
"~1.5 mm" was high and B3's "<=0.36 mm" was low -- the honest figure is a geometry-dependent range
0 .. ~0.9 mm). The undershoot is governed by ``eps_H`` relative to the front head-drop, not the
slope alone, and grows at a LOOSER ``eps_H`` (~2.4 mm at ``eps_H=1e-2``). Two things are robust
regardless of the residual undershoot: it is SUB-MM (vs the galerkin limiter's cm-scale
clip-and-rescale pathology it replaces), and CONSERVATION is machine-tight (the telescoping flux
network balances at any residual-converged root, ~2e-16). The B2 positivity gate pins the STEEP
regime (clean + adversarial: galerkin-would-clip there); the conservation gate uses a closed
scenario that also stays positive at the default width.

Regularization (Decision 4). ``eps_S`` (slope floor inside the conveyance root, dimensionless,
default 1e-3, matching ``overland.py``) keeps the mobility + its FD Jacobian finite at zero edge
slope. ``eps_H`` (smoothed-upwind head width, m, default 1e-3) sets the tanh selector width; the
lake-at-rest gate is INDEPENDENT of ``eps_H`` (structural -- uniform H zeroes the head drop the
selector multiplies), so the gate cannot be tuned to pass.

  eps_H DECISION (B3, chosen empirically -- ``scratch/_upwind_selector_probe.py``, 2026-06-15).
  Swept ``eps_H in {1e-2, 1e-3, 1e-4, 1e-5, 1e-6}`` over three adversarial 1-D scenarios (the B2
  STEEP 5% front, a MILD 2% front == the B5 valley slope, and the kinematic-plane equilibrium),
  recording Newton health (steps/iters + converged-reason histogram), run-min ``d`` (positivity),
  and accuracy (front position + kinematic ``d/d_eq``). The trade (steep|mild = the front
  scenarios, 60 cells, t_end 0.03 day; kin = the 100 m plane):

    eps_H | steep min_d | mild undershoot | Newton (steep: steps/iters) | kin d/d_eq | front_x
    ------+-------------+-----------------+-----------------------------+------------+--------
    1e-2  | -1.91e-3 m  | -2.40 mm  (NOT  | 54 / 278                    |   1.000    |  20.00
          |  (undershoot)|  strict)       |                             |            |
    1e-3  | +6e-34 m    |  0.000 mm (STR- | 47 / 257                    |   1.000    |  20.00   <== CHOSEN
          |  (strict)    |  ICT*)         |                             |            |
    1e-4  | +6e-34 m    |  0.000 mm (str) | 50 / 253                    |   1.000    |  20.00
    1e-5  | +6e-34 m    |  0.000 mm (str) | 106 / 498  (2x cost)        |   1.000    |  20.00
    1e-6  | +6e-34 m    |  0.000 mm (str) | 843 / 3680 (~15x cost)      |   1.000    |  20.00

  (* the table's mild-front row is the CANONICAL mound (sigma=1.5/peak=0.25), strictly positive
  in B3's setup; the mild-2% undershoot is GEOMETRY-DEPENDENT: a controller adjudication sweep
  (2026-06-16, 24 geometries) found 0 .. ~0.9 mm at 1e-3 (worst -0.93 mm on a sharp mound), vs
  ~2.4 mm at the looser 1e-2 -- so 1e-3 is ~3x better than 1e-2 but NOT bit-strict on the sharpest
  mild mounds. Conservation machine-tight throughout.)

  CHOICE = ``eps_H = 1e-3`` (the B1/B2 default is KEPT). Reasoning + the honest trade:
   * Positivity does NOT need a sharper selector -- it needs the selector NOT to be LOOSER. At the
     default 1e-3 the steep 5% front is strictly monotone (>= -1e-12) and the mild 2% front is
     sub-mm (0 .. ~0.9 mm, geometry-dependent); the looser 1e-2 is clearly worse (-1.9 mm steep,
     -2.4 mm mild). (The B2 "~1.5 mm at default" was high for typical mounds AND B3's first
     "<=0.36 mm" was low -- both superseded by the 0 .. ~0.9 mm range above; corrected here + in
     the Positivity section + the B2 tests/docstrings.)
   * Accuracy is eps_H-INVARIANT from 1e-3 downward: front position (20.00 m), post-run peak
     depth, and the kinematic equilibrium ``d/d_eq = 1.000`` do not move -- sharpening buys NO
     accuracy, so there is no accuracy reason to go sharper.
   * Newton robustness DEGRADES sharply below 1e-3: 1e-5 doubles the step count and 1e-6 is a
     ~15x blow-up (843 vs 47 steps, the extra all reason-4 SNORM stagnation as ``tanh`` -> the
     non-smooth ``sign`` and the FD-Jacobian Newton stalls). Going sharper to chase the last sub-mm
     of mild-front positivity would hit this cliff for no accuracy gain -- so 1e-3 is the balance.
   * 1e-4 ties 1e-3 on every axis; 1e-3 is kept as the established default with the larger margin
     to the Newton cliff and the green suite.
   IMPLICATION FOR B5 (the V's 2% valley): at ``eps_H=1e-3`` O1 is positive-viable on the V --
   sub-mm undershoot (0 .. ~0.9 mm, geometry-dependent) with machine-tight conservation, vs the
   galerkin limiter's cm-scale clip-and-rescale that drives the convergent-V pathology. This is NOT
   bit-strict positivity on the sharpest mild fronts, so **B5 must MEASURE the actual undershoot on
   the V** (not assume d>=0). IF the V needed ``eps_H`` sharper than ~1e-4 for acceptable positivity,
   the Newton cost would become prohibitive and SEMISMOOTH NEWTON is the P2 fallback (the SCHEME is
   the deliverable; this smoothed-selector tuning is secondary) -- not indicated by anything probed.

Units: length m, time day, slope dimensionless. ``n_man`` is SI s.m^{-1/3}; ``SECONDS_PER_DAY``
converts to m^2/day. Plan: docs/plans/2026-06-14-overland-convergent-flow-P1.md (Part B, B1).
"""
from __future__ import annotations

import numpy as np
import ufl  # B4: lumped-mass A_i assembly; B5: per-node outlet-edge control length (line outlet)
from dolfinx import fem
from dolfinx import mesh as dmesh  # B5: locate_entities_boundary + meshtags for the outlet ds
from mpi4py import MPI
from petsc4py import PETSc

SECONDS_PER_DAY = 86400.0


class UpwindOverlandProblem:
    """Diffusion-wave overland flow via a monotone upwind-mobility edge flux + custom SNES.

    Mirrors the PUBLIC interface of ``overland.OverlandProblem`` (``set_topography``,
    ``set_initial_condition``, ``add_rain``, ``add_outflow_bc``, ``outflow_rate``, ``step``,
    ``total_water``) so later tasks/tests use the two solvers interchangeably; the INTERNALS (edge
    graph, finite-difference SNES) follow the P1 locked decisions. B1 = 1-D, closed (no-flux)
    boundaries; B2 adds the optional ``add_outflow_bc`` Manning normal-depth outlet (a free-drainage
    nodal sink) plus the positivity/conservation/kinematic gates. See the module docstring for the
    scheme, the structural well-balanced/conservation properties, the positivity conditionality, and
    the index space.
    """

    def __init__(self, mesh, n_man: float, *, degree: int = 1, eps_S: float = 1e-3,
                 eps_H: float = 1e-3):
        if degree != 1:
            raise ValueError(
                f"UpwindOverlandProblem is a P1 edge-graph scheme; degree must be 1, got {degree}."
            )
        self.mesh = mesh
        self.n_man = float(n_man)
        self.eps_S = float(eps_S)
        self.eps_H = float(eps_H)  # smoothed-upwind head width [m] (Decision 4; B3 tunes it)

        self.V = fem.functionspace(mesh, ("Lagrange", degree))
        # fem.Functions hold d / d_n / z_b for convenient IC + topography interpolation and the
        # total_water integral; the SNES exchanges the raw dof array with d.x.array (Decision 1).
        self.d = fem.Function(self.V, name="d")      # current step (n+1), ponding depth
        self.d_n = fem.Function(self.V, name="d_n")  # previous step (n)
        self.z_b = fem.Function(self.V, name="z_b")  # bed topography (default flat = 0)

        self.dt = 1.0          # backward-Euler step (set per step())
        self.rain = 0.0        # uniform net rain source r [m/day]
        # Outflow outlets (B2; B5 adds the per-node boundary control length B for the 2-D LINE
        # outlet): each is (dofs, slope, B) -- a Manning normal-depth free-drainage sink at the
        # located boundary node(s). ``B[k]`` is the boundary-edge control LENGTH carried by outlet
        # node ``dofs[k]`` (= int phi_k ds over the located outlet facets): 1.0 for a 1-D point
        # outlet (point measure -> length-weighting is a no-op, the B2 1-D values are preserved),
        # and the node's share of the outlet-edge length in 2-D (sum B = outlet length), so the
        # per-unit-width Manning flux q_out [m^2/day] integrates to a VOLUMETRIC line discharge
        # [m^3/day] that telescopes with the volumetric storage (d*A_i/dt) and edge fluxes. Empty
        # = closed (no-flux) everywhere (the B1 default).
        self._outflows: list[tuple[np.ndarray, float, np.ndarray]] = []
        self.last_reason = 0   # SNES converged reason of the last step() solve (audit trail)
        self.last_iters = 0    # SNES iteration count of the last solve
        self.last_fnorm = np.nan  # ||F|| at the last solve's exit

        self._build_edge_graph()
        self._setup_snes()

    # -- edge graph (P1-dof index space, Decision 1/2; B4 adds the 2-D branch) -
    def _build_edge_graph(self) -> None:
        """Build the P1-dof edge list, edge transmissibilities T_e, edge lengths L_e, areas A_i.

        Dispatches on ``mesh.topology.dim``: the 1-D path (B1) is unchanged; the 2-D path (B4)
        is additive (a separate builder). Both populate the SAME attributes the residual reads --
        ``self.edges`` (dof pairs), ``self.L_e`` (edge lengths, for the Manning friction slope),
        ``self.T_e`` (the geometric two-point transmissibility), ``self.A_i`` (nodal areas), and
        ``self.n_edges`` -- so ``_assemble_residual`` / ``_setup_snes`` are dimension-agnostic.

        1-D: every cell is one edge ``(dof_a, dof_b)`` from ``V.dofmap.cell_dofs(c)``. Edge
        length ``L_e`` = |x_a - x_b| from the dof coordinates (P1: dof coords == vertex coords).
        ``T_e = 1/L_e`` (FD-Laplacian transmissibility, unit cross-section, Decision 2). ``A_i``
        = half the sum of the lengths of the cells incident to node i (standard FV dual), so
        ``sum_i A_i = domain length`` = the ``total_water`` measure (Decision 2).
        """
        tdim = self.mesh.topology.dim
        self.n_dofs = self.V.dofmap.index_map.size_local
        if tdim == 2:
            self._build_edge_graph_2d()
            return
        if tdim != 1:
            raise NotImplementedError(
                "UpwindOverlandProblem supports 1-D interval and 2-D triangle meshes (got "
                f"topology.dim = {tdim}; 3-D top-facet ridge graph is the P2 productionization)."
            )
        n_cells = self.mesh.topology.index_map(tdim).size_local
        coords = self.V.tabulate_dof_coordinates()[:, 0]  # P1: dof x-coordinate per dof

        edges = np.empty((n_cells, 2), dtype=np.int32)
        L_e = np.empty(n_cells, dtype=np.float64)
        A_i = np.zeros(self.n_dofs, dtype=np.float64)
        for c in range(n_cells):
            cd = self.V.dofmap.cell_dofs(c)  # 1-D cell -> two endpoint dofs
            a, b = int(cd[0]), int(cd[1])
            L = abs(coords[a] - coords[b])
            edges[c] = (a, b)
            L_e[c] = L
            # FV dual: split each cell length to its two endpoint control volumes.
            A_i[a] += 0.5 * L
            A_i[b] += 0.5 * L

        self.edges = edges                 # (n_edges, 2) int32 dof pairs (i, j)
        self.L_e = L_e                     # (n_edges,) edge lengths [m]
        self.T_e = 1.0 / L_e               # (n_edges,) transmissibility 1/L_e (Decision 2)
        self.A_i = A_i                     # (n_dofs,) nodal control volumes [m]; sum = domain length
        self.n_edges = n_cells

    def _build_edge_graph_2d(self) -> None:
        """B4: the 2-D triangle-mesh edge graph + cotangent transmissibility + M-matrix guard.

        Only the edge graph, ``T_e``, and ``A_i`` change vs 1-D -- the residual telescoping,
        smoothed-upwind selector, Manning mobility, SNES, and ``eps_H``/``eps_S`` are identical.

        EDGE GRAPH. Mesh EDGES are the unique P1-dof pairs: ``create_connectivity(1, 0)`` gives
        edge -> the two vertices, and ``create_connectivity(1, tdim)`` gives edge -> its (one or two)
        adjacent triangles. We map vertices -> dofs via the cell-local correspondence (for P1 the
        ``c2v`` cell-vertex order matches the ``cell_dofs`` order, the most robust map; verified on a
        tiny mesh in ``scratch/_b4_cotan_probe.py``, coord match ~1e-16) so the graph lives in the
        P1-dof index space exactly as 1-D and ``d.x.array`` stays the SNES unknown.

        TRANSMISSIBILITY (the crux). ``T_e`` is the COTANGENT / FV dual-mesh weight -- the standard
        monotone two-point coefficient -- equal to the NEGATED P1 stiffness off-diagonal for edge ij:
        ``T_e = 1/2 sum_tri cot(theta_opp)`` over the triangle(s) sharing ij, where ``theta_opp`` is
        the angle OPPOSITE edge ij (a boundary edge has one term). ``cot = cos/sin = (u.w)/|u x w|``
        from the two other triangle-edge vectors at the opposite vertex. This equals the FV
        perpendicular-bisector dual length / L_e on a Delaunay mesh, and is bit-identical to
        DOLFINx's assembled ``-int grad(phi_i).grad(phi_j) dx`` (pinned in
        ``test_cotangent_T_e_equals_negated_stiffness_offdiagonal_2d``; probe found max|diff|=0).
        ``L_e`` (the actual edge length) is retained SEPARATELY for the Manning friction slope
        ``S_f = |H_i - H_j| / L_e`` inside the mobility -- only the OUTER geometric factor changes
        from ``1/L_e`` (1-D) to the cotangent ``T_e`` (2-D).

        M-MATRIX GUARD (loud). Monotonicity REQUIRES ``T_e >= 0``. On a structured ``create_rectangle``
        box the right-triangle split gives non-negative weights: the DIAGONAL (hypotenuse) edges are
        opposite the right angle so cot(90 deg)=0 (verified: T_e=0 on all diagonals), and the AXIS
        edges are opposite acute angles so they carry the positive cotangents (T_e in [0.5, 1.0] on a
        unit-square grid). If any ``T_e < -1e-14`` (obtuse triangles / a bad split)
        we RAISE, naming the offending edge -- the scheme is non-monotone otherwise. MESH RESTRICTION:
        structured box / Delaunay, NON-OBTUSE. (Unstructured/obtuse transmissibility is P2/P3.)

        AREA. ``A_i`` = the lumped P1 mass ``int phi_i dx`` (vertex-quadrature row sum), the simplest
        correct 2-D control area; ``sum_i A_i = domain area`` so ``total_water = sum d_i A_i = int d dx``,
        matching the galerkin ``OverlandProblem.total_water`` lumped integral exactly.
        """
        from dolfinx.fem.petsc import assemble_vector

        tdim = self.mesh.topology.dim
        top = self.mesh.topology
        top.create_connectivity(tdim, 0)   # cell -> vertices (for the dof map + cotangents)
        top.create_connectivity(1, 0)      # edge -> vertices
        top.create_connectivity(1, tdim)   # edge -> cells (1 or 2 triangles per edge)
        c2v = top.connectivity(tdim, 0)
        e2v = top.connectivity(1, 0)
        e2c = top.connectivity(1, tdim)
        n_cells = top.index_map(tdim).size_local
        n_verts = top.index_map(0).size_local
        n_edges = top.index_map(1).size_local
        x = self.mesh.geometry.x  # vertex coordinates (gdim columns)

        # vertex -> P1 dof, from the cell-local correspondence (robust; matches cell_dofs order).
        vtx_to_dof = np.full(n_verts, -1, dtype=np.int64)
        for c in range(n_cells):
            verts = c2v.links(c)
            dofs = self.V.dofmap.cell_dofs(c)
            for k in range(len(verts)):
                vtx_to_dof[verts[k]] = dofs[k]

        def _cot_opposite(tri, vi, vj):
            """cotangent of the angle at the third (opposite) vertex of ``tri`` for edge vi-vj."""
            vk = [v for v in tri if v != vi and v != vj]
            k = int(vk[0])
            u = x[vi] - x[k]
            w = x[vj] - x[k]
            cross = u[0] * w[1] - u[1] * w[0]      # 2-D cross-product z-component (= 2*area, signed)
            dot = u[0] * w[0] + u[1] * w[1]
            return dot / abs(cross)                # cot = cos/sin = (u.w)/|u x w|

        edges = np.empty((n_edges, 2), dtype=np.int32)
        L_e = np.empty(n_edges, dtype=np.float64)
        T_e = np.zeros(n_edges, dtype=np.float64)
        for e in range(n_edges):
            vi, vj = (int(v) for v in e2v.links(e))
            di, dj = int(vtx_to_dof[vi]), int(vtx_to_dof[vj])
            edges[e] = (di, dj)
            L_e[e] = float(np.linalg.norm(x[vi] - x[vj]))
            s = 0.0
            for c in e2c.links(e):
                s += 0.5 * _cot_opposite(c2v.links(c), vi, vj)
            T_e[e] = s

        # M-MATRIX GUARD: loud failure on any negative transmissibility (obtuse triangle / bad split).
        if T_e.min() < -1e-14:
            bad = int(np.argmin(T_e))
            i, j = int(edges[bad, 0]), int(edges[bad, 1])
            raise ValueError(
                "M-matrix property violated: the cotangent transmissibility is NEGATIVE on edge "
                f"(dofs {i}, {j}), T_e = {T_e[bad]:.6e} < -1e-14. The monotone upwind scheme REQUIRES "
                "T_e >= 0 for every edge (an obtuse opposite angle gives cot < 0 -> an anti-diffusive "
                "edge that breaks the discrete maximum principle). UpwindOverlandProblem is restricted "
                "to structured-box / Delaunay NON-OBTUSE triangulations; remesh (a structured "
                "create_rectangle box, or a Delaunay mesh) so all opposite angles are <= 90 degrees."
            )

        # A_i = lumped P1 mass int phi_i dx (vertex quadrature); sum_i A_i = domain area.
        v = ufl.TestFunction(self.V)
        mass_form = fem.form(
            v * ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
        )
        bm = assemble_vector(mass_form)
        bm.assemble()
        A_i = bm.getArray()[: self.n_dofs].copy()
        bm.destroy()

        self.edges = edges                 # (n_edges, 2) int32 dof pairs (i, j)
        self.L_e = L_e                     # (n_edges,) edge lengths [m] (for the Manning slope)
        self.T_e = T_e                     # (n_edges,) cotangent transmissibility [-] (>= 0, guarded)
        self.A_i = A_i                     # (n_dofs,) lumped P1 control areas [m^2]; sum = domain area
        self.n_edges = n_edges

    # -- custom PETSc SNES (FD Jacobian + LU, Decision 3) ---------------------
    def _setup_snes(self) -> None:
        comm = self.mesh.comm
        n = self.n_dofs
        self._b = PETSc.Vec().createWithArray(np.zeros(n), comm=comm)  # residual workspace
        self._x = PETSc.Vec().createWithArray(np.zeros(n), comm=comm)  # solution workspace

        snes = PETSc.SNES().create(comm)
        snes.setFunction(self._assemble_residual, self._b)

        # Finite-difference Jacobian (Decision 3): no analytic/UFL Jacobian for the upwind selector
        # -- the ParFlow ``UseJacobian False`` precedent. ``setJacobian(None, J)`` + ``setUseFD(True)``
        # selects PETSc's internal coloring finite-difference Jacobian, assembled into J. (This stack
        # is petsc4py 3.25.2, which exposes ``setUseFD`` but not ``computeJacobianDefault``.) Coloring
        # needs J's SPARSITY PATTERN preset, so we insert the structural nonzeros from the edge graph: a
        # diagonal entry per node (storage) plus the two symmetric off-diagonals per edge (each
        # edge couples its two endpoints). Direct LU on the assembled FD Jacobian = true Newton;
        # trivial cost at these mesh sizes. A colored/hand analytic Jacobian is a later (P2) optimization.
        # Per-row nnz from the edge graph: diagonal + the (variable) edge-neighbour count -- 2 in 1-D
        # (tridiagonal), up to ~6-8 in 2-D (interior triangle-mesh valence), so we count it exactly.
        nnz = np.ones(n, dtype=np.int32)  # 1 diagonal per row
        for e in range(self.n_edges):
            a, b = int(self.edges[e, 0]), int(self.edges[e, 1])
            nnz[a] += 1
            nnz[b] += 1
        J = PETSc.Mat().createAIJ([n, n], comm=comm)
        J.setPreallocationNNZ(nnz)  # diagonal + per-row edge-neighbour count (1-D: 3; 2-D: valence+1)
        J.setUp()
        for k in range(n):
            J.setValue(k, k, 0.0)  # diagonal (storage term) -- always present
        for e in range(self.n_edges):
            a, b = int(self.edges[e, 0]), int(self.edges[e, 1])
            J.setValue(a, b, 0.0)
            J.setValue(b, a, 0.0)  # symmetric coupling structure (values filled by FD)
        J.assemble()

        snes.setJacobian(None, J)
        snes.setUseFD(True)
        snes.getKSP().setType("preonly")
        snes.getKSP().getPC().setType("lu")
        snes.setTolerances(rtol=1e-10, atol=1e-12, stol=1e-8, max_it=50)
        snes.setFromOptions()
        self._snes = snes
        self._J = J

    def _assemble_residual(self, snes, x, b) -> None:
        """SNES residual callback: read depths from ``x``, write node residuals ``R_i`` into ``b``.

        Computes every edge flux ``Q_e = T_e M(d_up)(H_i - H_j)`` (smoothed-upwind mobility) and
        telescopes it into the lumped backward-Euler node residual
        ``R_i = (d_i - d_n,i) A_i/dt + sum_{e in i} +/-Q_e - r A_i`` (Decision 2). Edge sign is
        ``+Q_e`` for the i-row, ``-Q_e`` for the j-row. No-flux boundaries are automatic: a
        boundary node has fewer incident edges, so its residual simply omits the absent flux.
        """
        d = x.getArray(readonly=True)
        z_b = self.z_b.x.array
        d_n = self.d_n.x.array

        # Lumped storage + rain source (per node). Edge fluxes are accumulated on top below.
        R = (d - d_n) * self.A_i / self.dt - self.rain * self.A_i

        i = self.edges[:, 0]
        j = self.edges[:, 1]
        H_i = z_b[i] + d[i]
        H_j = z_b[j] + d[j]
        dH = H_i - H_j                                   # edge head drop (i - j)
        slope = dH / self.L_e                            # signed edge slope

        # Smoothed C1 upstream weight: w -> 1 when H_i >> H_j (take d_i), -> 0 when H_j >> H_i.
        w = 0.5 * (1.0 + np.tanh(dH / self.eps_H))
        d_up = w * d[i] + (1.0 - w) * d[j]
        d_up_pos = np.maximum(d_up, 0.0)                 # guard the fractional power
        slope_root = (slope * slope + self.eps_S ** 2) ** 0.25  # Manning slope floor (inside root)
        M = SECONDS_PER_DAY * d_up_pos ** (5.0 / 3.0) / (self.n_man * slope_root)
        Q_e = self.T_e * M * dH                          # edge flux (>0: i -> j)

        # Telescoping accumulation: +Q_e on the i-row, -Q_e on the j-row (np.add.at handles the
        # repeated indices of interior nodes shared by two edges).
        np.add.at(R, i, Q_e)
        np.add.at(R, j, -Q_e)

        # Outflow sinks (B2; B5 length-weights for the 2-D LINE outlet): water LEAVES the located
        # outlet node(s), so q_out enters that node's residual with a PLUS sign (mirroring the
        # galerkin ``+q_out v ds`` boundary sink, whose ``ds`` supplies exactly this length factor).
        # q_out is the per-unit-width Manning normal-depth discharge [m^2/day] at the node's depth;
        # multiplying by the node's boundary control length ``B`` makes the sink VOLUMETRIC
        # [m^3/day] in 2-D (B = the node's outlet-edge length) and leaves the 1-D point outlet
        # unchanged (B = 1.0). ``max(d,0)`` guards the fractional power.
        for dofs, slope, B in self._outflows:
            q_out = (SECONDS_PER_DAY * (1.0 / self.n_man)
                     * np.maximum(d[dofs], 0.0) ** (5.0 / 3.0) * np.sqrt(slope)) * B
            np.add.at(R, dofs, q_out)

        b.setArray(R)

    # -- problem setup (mirror OverlandProblem's public interface) ------------
    def set_topography(self, expr) -> None:
        """Set the bed elevation z_b(x) (a known P1 field; default flat z_b = 0)."""
        self.z_b.interpolate(expr)

    def set_initial_condition(self, expr) -> None:
        self.d.interpolate(expr)
        self.d_n.interpolate(expr)

    def add_rain(self, rate) -> float:
        """Add a spatially-uniform rainfall (net) source r (m/day, positive = water IN).

        Enters each node residual as ``-r * A_i`` (the volumetric source over the node control
        volume). Returns the stored rate. (Infiltration / feature exchange are owned by the
        coupling module; standalone overland sees rainfall + boundary conditions only.)
        """
        self.rain = float(rate)
        return self.rain

    def add_outflow_bc(self, locator, slope: float):
        """Free-drainage outlet: water leaves at the Manning NORMAL-DEPTH discharge.

        ``q_out = SECONDS_PER_DAY * (1/n_man) * max(d,0)^{5/3} * sqrt(slope)`` (m^2/day per unit
        width) at the located boundary node(s), i.e. the friction slope at the outlet is taken as
        the bed ``slope`` -- the standard kinematic/normal-depth outflow condition (vs the natural
        no-flux boundary, which would dam the reach). Mirrors ``OverlandProblem.add_outflow_bc``:
        there the sink enters the weak form as ``+q_out v ds``; here, in the lumped node-residual
        formulation, it enters each located node's residual with a ``+q_out * B_k`` sign -- water
        LEAVING that control volume.

        2-D LINE OUTLET (B5; the subtlety that makes the V's ``Q_out`` correct). For a 1-D POINT
        outlet (a single node) the discharge is just the nodal ``q_out`` and the residual unit is
        m^2/day. But the V outlet is a LINE of nodes along ``y=LY``: the total discharge is the
        INTEGRAL of the per-unit-width flux ``q_out`` over the outlet edge length, so each node must
        carry its share ``B_k`` of the outlet-edge control LENGTH (exactly what the galerkin
        ``ds`` measure supplies via ``int phi_k ds``). We assemble that boundary mass once here
        (``int phi_k ds`` over the located outlet facets) and store it per outlet; ``B_k`` then
        weights the residual sink AND ``outflow_rate()``. ``B_k = 1.0`` for a 1-D point facet (point
        measure), so length-weighting is a NO-OP in 1-D and the B2 1-D outflow values are preserved;
        in 2-D ``sum_k B_k = outlet length`` and the sink is volumetric m^3/day, telescoping with
        the volumetric storage (``d*A_i/dt``) and edge fluxes. (Verified on the canonical V outlet:
        ``scratch/_b5_outlet_probe.py`` -- the length-weighted nodal sum matches the galerkin
        ``ds``-integral and the analytic ``q*LX`` to ~1e-6; the naive un-weighted B2 sum was ~33x
        wrong on a 48-node outlet.)

        The sink depends only on the outlet node's OWN depth, so its finite-difference Jacobian
        contribution is purely diagonal -- already covered by the preset sparsity pattern (no
        edge-graph/SNES change needed).

        ``slope`` must be strictly positive: ``slope = 0`` would silently turn the outlet into a
        no-flux wall (damming the reach) and ``slope < 0`` injects ``sqrt(<0)`` = NaN into the
        residual; both are caller errors, so we reject them up front (same guard as the galerkin
        path).
        """
        if slope <= 0.0:
            raise ValueError(
                f"add_outflow_bc requires slope > 0 (normal-depth friction slope); got {slope!r}. "
                "slope=0 dams the outlet; slope<0 gives sqrt(<0)=NaN."
            )
        dofs = fem.locate_dofs_geometrical(self.V, locator)
        dofs = dofs[dofs < self.n_dofs].astype(np.int64)  # owned dofs only (MPI-safe)
        B = self._boundary_control_length(locator)[dofs]   # per-node outlet-edge length (1.0 in 1-D)
        self._outflows.append((dofs, float(slope), B))
        return None

    def _boundary_control_length(self, locator) -> np.ndarray:
        """Per-dof boundary control length ``B_i = int phi_i ds`` over the located outlet facets.

        This is the lumped boundary "mass" the galerkin outlet sink carries implicitly through its
        ``ds`` measure: assembling ``v * ds_locator`` puts each node's share of the located outlet
        edge length on row ``i`` (sum over the outlet = the outlet length in 2-D; 1.0 at a 1-D point
        facet). Used to length-weight the outlet sink + ``outflow_rate()`` for the 2-D line outlet
        (B5). Returns the full per-dof array (indexed by the caller with its owned outlet dofs).
        """
        from dolfinx.fem.petsc import assemble_vector

        fdim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        facets = np.sort(dmesh.locate_entities_boundary(self.mesh, fdim, locator))
        ft = dmesh.meshtags(self.mesh, fdim, facets, np.full(facets.shape, 1, dtype=np.int32))
        ds_out = ufl.Measure("ds", domain=self.mesh, subdomain_data=ft)(1)
        v = ufl.TestFunction(self.V)
        bvec = assemble_vector(fem.form(v * ds_out))
        bvec.assemble()
        B = bvec.getArray()[: self.n_dofs].copy()
        bvec.destroy()
        return B

    def outflow_rate(self) -> float:
        """Total discharge leaving through all outflow boundaries.

        The boundary integral of the per-unit-width Manning flux ``q_out`` over each outlet, i.e.
        ``sum_k q_out(d_k) * B_k`` with ``B_k`` the node's outlet-edge control length -- the exact
        discrete analogue of the galerkin ``int q_out ds``. 1-D: ``B_k = 1.0`` (point facet), so
        this is the nodal ``q_out`` and reports m^2/day per unit width (matching the galerkin
        point-facet value + ``test_outflow_discharge_absolute_magnitude_1d``). 2-D: ``B_k`` is the
        node's share of the outlet length, so this is the VOLUMETRIC line discharge m^3/day (the
        quantity compared to ``Q_eq = rain*area`` at equilibrium). Evaluated at the current solved
        depth state.
        """
        d = self.d.x.array
        local = 0.0
        for dofs, slope, B in self._outflows:
            local += float(np.sum(
                SECONDS_PER_DAY * (1.0 / self.n_man)
                * np.maximum(d[dofs], 0.0) ** (5.0 / 3.0) * np.sqrt(slope) * B
            ))
        return self.mesh.comm.allreduce(local, op=MPI.SUM)

    # -- time stepping --------------------------------------------------------
    def step(self, dt: float):
        """Advance one backward-Euler Newton step. Returns ``(converged: bool, iters)``.

        Solves the upwind node-residual system with the FD-Jacobian SNES. On a converged solve
        (``getConvergedReason() > 0``) the new depth is accepted into ``d_n``; on failure the
        last accepted state is restored so callers can cut dt and retry. There is NO positivity
        limiter -- the monotone upwind scheme is meant to hold ``d >= 0`` on its own (the B2
        gate); B1 only requires the lake-at-rest exactness + a live non-flat-flow sanity check.
        """
        self.dt = float(dt)
        # Seed the SNES from the current depth iterate (warm start at d_n's value).
        self._x.setArray(self.d.x.array)
        self._snes.solve(None, self._x)

        self.last_reason = int(self._snes.getConvergedReason())
        self.last_iters = int(self._snes.getIterationNumber())
        self.last_fnorm = float(self._snes.getFunctionNorm())
        # B1 books ANY positive reason, INCLUDING reason-4 (SNORM stagnation-at-floor): on such a
        # step conservation/positivity hold only to the exit ||F||, not exactly. The residual-floor
        # acceptance gate (OverlandProblem.stall_accept_fnorm, the B6 P0 hazard) is deferred to B2.
        converged = self.last_reason > 0
        if converged:
            self.d.x.array[:] = self._x.getArray(readonly=True)
            self.d.x.scatter_forward()
            self.d_n.x.array[:] = self.d.x.array  # accept the step
            self.d_n.x.scatter_forward()
        else:
            self.d.x.array[:] = self.d_n.x.array  # restore last accepted state
            self.d.x.scatter_forward()
        return converged, self.last_iters

    # -- diagnostics ----------------------------------------------------------
    def total_water(self) -> float:
        """Total surface water = sum_i d_i * A_i (the FV-dual lumped integral of d).

        Uses the same nodal control volumes A_i as the residual storage term, so the conserved
        quantity is exactly what the integrator conserves (``sum_i A_i = domain length``, matching
        ``int d dx``). Kept consistent with ``OverlandProblem.total_water`` (the lumped integral).
        """
        local = float(np.dot(self.d.x.array[: self.n_dofs], self.A_i))
        return self.mesh.comm.allreduce(local, op=MPI.SUM)
