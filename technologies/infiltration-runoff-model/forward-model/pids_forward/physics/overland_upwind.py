"""O1 upwind-mobility two-point edge-flux overland solver (Convergent-flow P1, Part B).

A standalone Module-2 spike of a MONOTONE, WELL-BALANCED diffusion-wave overland scheme, built
to fix the convergent-flow regime where the validated galerkin ``OverlandProblem`` develops a
sawtooth (the "Defect A" Galerkin instability on convergence lines, B6 P0). This class does NOT
touch ``overland.py`` -- that galerkin path stays the MMS/regression reference; this is a
separate scheme on its own class so the two can be compared head-to-head.

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

  * WELL-BALANCED (lake-at-rest exact): a uniform surface head makes every ``H_i - H_j = 0`` so
    every ``Q_e = 0`` *structurally* -- independent of ``eps_S``/``eps_H``. A still pond holds
    to machine precision; it cannot drain spuriously down the bed slope (the gate the galerkin
    scheme also passes, but here it is by edge construction on H, not UFL ``grad(z_b + d)``).
  * CONSERVATIVE (discrete, structural): each interior edge contributes ``+Q_e`` to one row and
    ``-Q_e`` to the other, so summing all node residuals telescopes the entire flux network to
    zero -- ``sum_i (d_i - d_n,i) A_i = dt * r * sum_i A_i`` exactly (closed domain, no rain =>
    total water invariant). No flux leaves a node-pair without entering its neighbour.

Index space (Decision 1). The edge graph lives in the P1 *dof* index space: edges come from
``V.dofmap.cell_dofs(c)`` (1-D: each cell -> one edge ``(dof_a, dof_b)``). The SNES unknown
vector IS the P1 dof array ``d.x.array``, so the residual, ``total_water()``, and the
IC/topography interpolation all share ONE index space -- no vertex<->dof remap (more robust than
raw topology vertex ids). For P1 elements the dof coordinates equal the vertex coordinates, so
edge lengths come straight from ``V.tabulate_dof_coordinates()`` indexed by the cell dofs.

Solver (Decisions 3, 5). ``UpwindOverlandProblem`` drives a ``petsc4py.PETSc.SNES`` directly
(no DOLFINx ``NonlinearProblem`` -- the upwind selection is not UFL-expressible). The Jacobian
is a BRUTE finite-difference (``snes.computeJacobianDefault``) with a direct LU solve -- true
Newton + direct solve, the most robust choice for this stiff degenerate diffusion and trivially
cheap at 1-D sizes (a colored FD / hand Jacobian is a later optimization). Boundaries are
NO-FLUX (closed) only for B1: a boundary node simply has fewer incident edges, so its residual
omits the missing flux -- the natural no-flux condition (outflow BC is deferred to B2).

Regularization (Decision 4). ``eps_S`` (slope floor inside the conveyance root, dimensionless,
default 1e-3, matching ``overland.py``) keeps the mobility + its FD Jacobian finite at zero edge
slope. ``eps_H`` (smoothed-upwind head width, m, default 1e-3) sets the tanh selector width; the
lake-at-rest gate is INDEPENDENT of ``eps_H`` (structural -- uniform H zeroes the head drop the
selector multiplies), so the gate cannot be tuned to pass. The empirical ``eps_H`` width
decision is a later task (B3); 1e-3 m is the documented B1 default.

Units: length m, time day, slope dimensionless. ``n_man`` is SI s.m^{-1/3}; ``SECONDS_PER_DAY``
converts to m^2/day. Plan: docs/plans/2026-06-14-overland-convergent-flow-P1.md (Part B, B1).
"""
from __future__ import annotations

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

SECONDS_PER_DAY = 86400.0


class UpwindOverlandProblem:
    """Diffusion-wave overland flow via a monotone upwind-mobility edge flux + custom SNES.

    Mirrors the PUBLIC interface of ``overland.OverlandProblem`` (``set_topography``,
    ``set_initial_condition``, ``add_rain``, ``step``, ``total_water``) so later tasks/tests use
    the two solvers interchangeably; the INTERNALS (edge graph, finite-difference SNES) follow
    the P1 locked decisions. B1 = 1-D, closed (no-flux) boundaries only. See the module docstring
    for the scheme, the structural well-balanced/conservation properties, and the index space.
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
        self.last_reason = 0   # SNES converged reason of the last step() solve (audit trail)
        self.last_iters = 0    # SNES iteration count of the last solve
        self.last_fnorm = np.nan  # ||F|| at the last solve's exit

        self._build_edge_graph()
        self._setup_snes()

    # -- edge graph (P1-dof index space, Decision 1/2) ------------------------
    def _build_edge_graph(self) -> None:
        """Build the P1-dof edge list, edge transmissibilities T_e, and nodal areas A_i.

        1-D: every cell is one edge ``(dof_a, dof_b)`` from ``V.dofmap.cell_dofs(c)``. Edge
        length ``L_e`` = |x_a - x_b| from the dof coordinates (P1: dof coords == vertex coords).
        ``T_e = 1/L_e`` (FD-Laplacian transmissibility, unit cross-section, Decision 2). ``A_i``
        = half the sum of the lengths of the cells incident to node i (standard FV dual), so
        ``sum_i A_i = domain length`` = the ``total_water`` measure (Decision 2).
        """
        tdim = self.mesh.topology.dim
        if tdim != 1:
            raise NotImplementedError(
                "UpwindOverlandProblem B1 supports 1-D interval meshes only (2-D edge graph = B4)."
            )
        n_cells = self.mesh.topology.index_map(tdim).size_local
        self.n_dofs = self.V.dofmap.index_map.size_local
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

    # -- custom PETSc SNES (FD Jacobian + LU, Decision 3) ---------------------
    def _setup_snes(self) -> None:
        comm = self.mesh.comm
        n = self.n_dofs
        self._b = PETSc.Vec().createWithArray(np.zeros(n), comm=comm)  # residual workspace
        self._x = PETSc.Vec().createWithArray(np.zeros(n), comm=comm)  # solution workspace

        snes = PETSc.SNES().create(comm)
        snes.setFunction(self._assemble_residual, self._b)

        # Brute finite-difference Jacobian (Decision 3): no analytic/UFL Jacobian for the upwind
        # selector -- the ParFlow ``UseJacobian False`` precedent. ``setJacobian(None, J)`` +
        # ``setUseFD(True)`` drives PETSc's coloring finite-difference Jacobian assembly into J
        # (the petsc4py 0.10 spelling of the removed ``computeJacobianDefault``). Coloring needs J's
        # SPARSITY PATTERN preset, so we insert the structural nonzeros from the edge graph: a
        # diagonal entry per node (storage) plus the two symmetric off-diagonals per edge (each
        # edge couples its two endpoints). Direct LU on the assembled FD Jacobian = true Newton;
        # trivial cost at 1-D sizes. A colored/hand analytic Jacobian is a later (B4+) optimization.
        J = PETSc.Mat().createAIJ([n, n], comm=comm)
        J.setPreallocationNNZ(3)  # tridiagonal: diagonal + up to 2 edge neighbours per row
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
