"""Pure-NumPy core of the sequential operator-split overland ROUTING SWEEP (B1-B3).

Extracted FAITHFULLY (behaviour-preserving) from the validated spike
``scratch/overland_split_spike.py`` (its ``kernel`` mode: conservation residual 0.0, min depth 0.0,
a 10 cm pulse moves only 2.77% to the far outlet in one step). This is the explicit, mass-conserving,
Manning-rate-limited downslope routing that the sequential-split coupling runs AFTER the implicit
Richards solve each timestep: read the leftover ponded depth at the surface (top-facet) nodes, push
it downhill over the surface flow-direction graph (rate-capped by Manning so water cannot teleport),
and hand the redistributed depths back as the next step's run-on.

The surface graph is the SAME top-facet edge graph the upwind scheme uses
(``overland_edge_kernel.build_top_facet_edge_graph`` -> ``edges, L_e, T_e, A_i``): P1-dof edge pairs,
planar edge lengths ``L_e``, the cotangent two-point transmissibility ``T_e`` (>= 0, M-matrix guarded
at graph-build), and the lumped surface control areas ``A_i`` (``sum_i d_i A_i`` = the lumped surface
water integral). Topography is carried by the bed-elevation field ``z_b``; the surface head is
``H = z_b + d``.

THE LAW (one sweep). Process the surface nodes in DESCENDING head ``H_i`` (upslope before downslope --
the richdem-style single dependency pass, so a node's run-on is already in place when it is visited).
For a node ``i`` with depth ``d_i > 0``:
  * downslope receivers ``j`` = interior neighbours with ``H_j < H_i``; multiple-flow-direction (MFD)
    weight ``w_j = sqrt(slope_ij)``, ``slope_ij = (H_i - H_j)/L_ij``. An off-domain OUTLET
    pseudo-receiver (weight ``sqrt(outlet_slope)``) at boundary outlet nodes removes water from the
    domain. Weights are normalized over all receivers (interior + outlet).
  * Manning volumetric cap over ``dt``:
        ``Vcap_i = (SECONDS_PER_DAY/n_man) * d_i^(5/3) * sqrt(S_i) * W_i * dt``,
    with ``S_i`` = the node's MAX receiver slope and ``W_i`` = the node's geometry-derived Manning
    width (``node_widths``). This is the volume the node can physically pass in ``dt``.
  * ``V_out_i = min(d_i*A_i, Vcap_i)`` (never more than the inventory -> positivity), distributed to
    receivers by MFD weight: each receiver gains ``V_recv/A_recv`` in depth; the outlet's share is
    booked as off-domain outflow.

Conservation: the interior transfers telescope (what leaves a node arrives at its receivers), so
``sum_i d_i A_i`` (interior) plus the cumulative outlet outflow is invariant to machine precision.
The Manning cap is LOAD-BEARING: without it a node dumps its whole inventory to the outlet in one step
(teleportation) and run-on infiltration cannot form.

Units: length m, time day, slope dimensionless; ``n_man`` SI s.m^{-1/3}; ``SECONDS_PER_DAY`` converts
the SI Manning conveyance to volume/day. Pure functions -- inputs are NEVER mutated.

Restriction (shared with the upwind edge kernel): serial-only; structured-box / Delaunay NON-OBTUSE
top triangulation (the cotangent ``T_e >= 0`` M-matrix property, guarded at graph-build).
"""
from __future__ import annotations

import numpy as np

SECONDS_PER_DAY = 86400.0


def build_adjacency(edges, L_e, n_dofs):
    """Per-node undirected adjacency list ``adj[i] = [(j, L_ij), ...]`` over the surface graph.

    ``edges`` is the ``(n_edges, 2)`` int dof-pair array and ``L_e`` the ``(n_edges,)`` planar edge
    lengths from ``build_top_facet_edge_graph``. Each undirected edge contributes both directions so
    the routing sweep can find a node's downslope receivers among all its graph neighbours. Interior
    (non-surface) dofs simply get an empty list.
    """
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_dofs)]
    for e in range(edges.shape[0]):
        i, j = int(edges[e, 0]), int(edges[e, 1])
        L = float(L_e[e])
        adj[i].append((j, L))
        adj[j].append((i, L))
    return adj


def node_widths(edges, L_e, T_e, n_dofs):
    """Characteristic Manning flow width ``W_i`` [m] per node = ``sum_{e ni i} T_e * L_e``.

    ``T_e`` = cotangent transmissibility = (dual-mesh face length)/(edge length), so ``T_e * L_e`` is
    the dual-mesh face length carried by that edge; summing the incident faces gives a geometry-derived
    width that scales with the node's lateral cross-section. NO tuning constant. Surface (top-facet)
    dofs get a positive width; interior dofs are 0 (``T_e`` there is 0). Used as the ``W_i`` factor in
    the Manning volumetric cap.
    """
    W = np.zeros(n_dofs, dtype=np.float64)
    contrib = T_e * L_e
    for e in range(edges.shape[0]):
        i, j = int(edges[e, 0]), int(edges[e, 1])
        W[i] += contrib[e]
        W[j] += contrib[e]
    return W


def topo_order(H, top_dofs):
    """Surface processing order = the ``top_dofs`` sorted by DESCENDING head ``H`` (upslope first).

    The richdem-style single dependency sweep: visiting high-head nodes before low-head ones means a
    node's upslope run-on has already been deposited by the time it is processed, so ONE pass routes
    the excess correctly down a monotone flow path. ``H = z_b + d`` is the surface head. Returns an
    ``int64`` array permuting ``top_dofs``. (``route_excess`` recomputes this internally from the live
    head each sweep; this helper exposes the same order for inspection/tests.)
    """
    top_dofs = np.asarray(top_dofs)
    return top_dofs[np.argsort(-H[top_dofs])]


def build_receivers(H, adj, top_dofs):
    """Per-node downslope receivers + normalized MFD weights from a STATIC head snapshot ``H``.

    For each node in ``top_dofs``, its receivers are the graph neighbours ``j`` with strictly lower
    head (``slope_ij = (H_i - H_j)/L_ij > 0``); the multiple-flow-direction weight is
    ``w_j = sqrt(slope_ij)`` NORMALIZED so ``sum_j w_j = 1`` (a flat or pit node gets an empty list ->
    no spurious direction). Returns a dict ``{i: (js, ws)}`` with ``js`` an int array of receiver dofs
    and ``ws`` the matching float weight array.

    This is the static-snapshot view of the receiver logic. NOTE the validated ``route_excess`` sweep
    recomputes receivers DYNAMICALLY from the live (partially-routed) depth as it visits each node --
    its in-loop receiver set is what the headline numbers come from; this helper mirrors the same
    sqrt(slope) MFD rule on a fixed ``H`` for receiver/order inspection and tests.
    """
    recv: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in np.asarray(top_dofs):
        i = int(i)
        Hi = H[i]
        js, ws = [], []
        for (j, L) in adj[i]:
            s = (Hi - H[j]) / L
            if s > 0.0:
                js.append(j)
                ws.append(np.sqrt(s))
        if ws:
            w = np.asarray(ws, dtype=np.float64)
            w /= w.sum()
            recv[i] = (np.asarray(js, dtype=np.int64), w)
        else:
            recv[i] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
    return recv


def route_excess(d, z_b, A_i, adj, top_dofs, W, n_man, dt, outlet_mask, outlet_slope):
    """ONE explicit Manning-rate-limited downslope routing sweep over the surface graph.

    Faithful extraction of the validated spike kernel. Processes ``top_dofs`` in descending surface
    head ``H_i = z_b_i + d_i``; for each node with ``d_i > 0`` it finds its downslope receivers
    (interior neighbours with lower head -> MFD weight ``sqrt(slope)``, plus an off-domain outlet
    pseudo-receiver at outlet nodes with weight ``sqrt(outlet_slope)``), caps the outgoing volume at
    the Manning rate ``Vcap = (SECONDS_PER_DAY/n_man) * d_i^(5/3) * sqrt(S_i) * W_i * dt`` (``S_i`` =
    max receiver slope), and moves ``V_out = min(d_i*A_i, Vcap)`` to the receivers by normalized MFD
    weight (each receiver's depth += V_recv/A_recv; the outlet's share is removed from the domain).

    Receivers are recomputed PER NODE from the live ``d_new`` (so a node's run-on, deposited by an
    already-visited upslope node, is seen). Local pits / flats (no positive-slope receiver) keep their
    water (see ``fill_depressions``, reserved-but-unused here).

    Parameters
    ----------
    d : (n_dofs,) float -- current ponded depth at every dof (interior dofs ignored). NOT mutated.
    z_b : (n_dofs,) float -- bed elevation (topography); surface head is ``z_b + d``.
    A_i : (n_dofs,) float -- lumped surface control areas (``build_top_facet_edge_graph``).
    adj : list -- undirected adjacency from ``build_adjacency``.
    top_dofs : int array -- the surface (top-facet) dofs to process.
    W : (n_dofs,) float -- per-node Manning width from ``node_widths``.
    n_man : float -- Manning roughness (SI s.m^{-1/3}).
    dt : float -- timestep [day].
    outlet_mask : (n_dofs,) bool -- True at boundary outlet nodes that drain off-domain.
    outlet_slope : float -- the (scalar) hydraulic slope used for the outlet pseudo-receiver.

    Returns
    -------
    (d_new, outflow) : the routed depth array (>= 0) and the total volume removed at outlets this
    sweep. Conserves ``sum_i d_i A_i`` (interior) + ``outflow`` to machine precision (telescoping).
    """
    d_new = d.copy()
    outflow = 0.0
    Cman = SECONDS_PER_DAY / n_man
    H0 = z_b + d_new
    order = np.asarray(top_dofs)[np.argsort(-H0[np.asarray(top_dofs)])]

    for i in order:
        i = int(i)
        di = d_new[i]
        if di <= 0.0:
            continue
        Hi = z_b[i] + di
        recv_j, recv_w = [], []
        smax = 0.0
        for (j, L) in adj[i]:
            s = (Hi - (z_b[j] + d_new[j])) / L
            if s > 0.0:
                recv_j.append(j)
                recv_w.append(np.sqrt(s))
                if s > smax:
                    smax = s
        out_w = 0.0
        if outlet_mask[i] and outlet_slope > 0.0:
            out_w = np.sqrt(outlet_slope)
            if outlet_slope > smax:
                smax = outlet_slope
        wsum = float(np.sum(recv_w)) + out_w
        if wsum <= 0.0 or smax <= 0.0:
            continue  # local pit / flat: water stays put (no spurious direction)
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


def fill_depressions(z, adj, top_dofs, open_nodes):
    """PriorityFlood depression filling on the surface graph (RESERVED -- the validated sweep does
    NOT need it on the tested geometries; the descending-head sweep sufficed).

    Minimal correct Barnes-2014 Priority-Flood: starting from the ``open_nodes`` (the boundary /
    outlet spill set), grow over the surface graph with a min-priority queue, raising each newly
    reached node to ``max(z_node, spill_level)`` where ``spill_level`` is the highest elevation seen
    along the lowest path to an open node. The result fills every interior pit up to its lowest rim so
    a continuous non-ascending drainage path to an outlet exists, NEVER lowers any node, and returns a
    pit-free field unchanged. ``z`` is the bed (or surface-head) field to condition; only ``top_dofs``
    are processed; interior dofs are copied through untouched.

    Provided for completeness / future use (e.g. genuinely pitted field topographies). It is not on
    the routing hot path: ``route_excess`` leaves water in a local pit by construction, which is the
    physically-correct behaviour when no fill is applied.
    """
    import heapq

    zf = z.astype(np.float64).copy()
    top_set = set(int(i) for i in np.asarray(top_dofs))
    closed = {i: True for i in top_set}  # True = not yet finalized
    heap: list[tuple[float, int]] = []
    for i in np.asarray(open_nodes):
        i = int(i)
        if i in closed:
            closed[i] = False
            heapq.heappush(heap, (float(zf[i]), i))
    while heap:
        level, i = heapq.heappop(heap)
        for (j, _L) in adj[i]:
            j = int(j)
            if j not in closed or not closed[j]:
                continue  # not a top node, or already finalized
            closed[j] = False
            # raise j to the spill level (the highest point on the lowest path so far)
            if zf[j] < level:
                zf[j] = level
            heapq.heappush(heap, (float(zf[j]), j))
    return zf
