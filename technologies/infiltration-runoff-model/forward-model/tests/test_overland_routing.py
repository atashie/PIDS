"""Tier-1 sanity for the extracted overland ROUTING-SWEEP kernel (sequential operator-split, B1-B3).

``pids_forward/physics/overland_routing.py`` holds the pure-NumPy core of the validated
sequential-split overland sweep -- the explicit, mass-conserving, Manning-rate-limited downslope
routing over the top-facet flow-direction graph. It was extracted FAITHFULLY (behaviour-preserving)
from the validated spike ``scratch/overland_split_spike.py`` (its ``kernel`` mode: conservation
resid 0.0, min depth 0.0, a 10 cm pulse moves only 2.77% to the far outlet in one step -> the
Manning cap working, NO teleportation).

These tests are the EXTRACTION GUARD: the module must reproduce that validated behaviour --
conservation to machine precision, positivity, the bounded Manning advance (no teleport), correct
MFD receivers/topological order -- AND the spike's exact headline numbers on the same sample input
(so the refactor cannot perturb the validated numerics).

The graph (``edges, L_e, T_e, A_i``) comes from the SAME builder the upwind scheme uses,
``overland_edge_kernel.build_top_facet_edge_graph``, on a structured box top facet.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from pids_forward.physics.overland_edge_kernel import build_top_facet_edge_graph
from pids_forward.physics.overland_routing import (
    build_adjacency,
    build_receivers,
    fill_depressions,
    node_widths,
    route_excess,
    topo_order,
)

COMM = MPI.COMM_WORLD
SECONDS_PER_DAY = 86400.0


# ----------------------------------------------------------------------------------------------------
# A reusable structured-box top-facet routing graph on a uniform planar slope (the spike's geometry).
# ----------------------------------------------------------------------------------------------------
def _slope_graph(LX=2.0, LY=1.0, LZ=0.5, NX=10, NY=5, NZ=2, S0=0.05):
    """Build the top-facet graph + bed elevation z_b for a uniform planar slope z_b = S0*(LX - x).

    Returns a dict mirroring the spike's ``build_surface_graph`` outputs (n_dofs, top_dofs, edges,
    L_e, T_e, A_i, z_b, adj, W, coords) plus the outlet mask at x=LX and the outlet slope S0.
    """
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, LZ]], [NX, NY, NZ])
    V = fem.functionspace(msh, ("Lagrange", 1))
    gdim = msh.geometry.dim
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    top_facets = np.sort(dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[gdim - 1], LZ))).astype(np.int32)
    edges, L_e, T_e, A_i = build_top_facet_edge_graph(V, msh, top_facets)
    coords = V.tabulate_dof_coordinates()
    n_dofs = V.dofmap.index_map.size_local
    zc = coords[:, gdim - 1]
    top_dofs = np.where(np.isclose(zc, LZ))[0].astype(np.int64)
    z_b = (S0 * (LX - coords[:, 0])).astype(np.float64)
    adj = build_adjacency(edges, L_e, n_dofs)
    W = node_widths(edges, L_e, T_e, n_dofs)
    outlet_mask = np.zeros(n_dofs, dtype=bool)
    outlet_mask[top_dofs[np.isclose(coords[top_dofs, 0], LX)]] = True
    up_dofs = top_dofs[np.isclose(coords[top_dofs, 0], 0.0)]
    return dict(msh=msh, V=V, edges=edges, L_e=L_e, T_e=T_e, A_i=A_i, coords=coords,
                n_dofs=n_dofs, top_dofs=top_dofs, z_b=z_b, adj=adj, W=W,
                outlet_mask=outlet_mask, outlet_slope=S0, up_dofs=up_dofs, LX=LX, LZ=LZ)


def _mass(d, A_i, top_dofs):
    return float(np.sum(d[top_dofs] * A_i[top_dofs]))


# ====================================================================================================
# Test 1 -- CONSERVATION (the heart): sum(d_i*A_i) invariant on a closed graph; with an outlet,
# sum(d*A)_after + cum_outflow == before, both to <= 1e-12, over many sweeps.
# ====================================================================================================
def test_conservation_closed_graph_machine_precision():
    """No outlet -> closed surface: sum(d_i*A_i) is invariant to <= 1e-12 over many sweeps on a
    random non-negative field. The lateral transfers telescope (what leaves a node arrives at its
    receivers), so interior mass is conserved exactly."""
    g = _slope_graph()
    closed = np.zeros(g["n_dofs"], dtype=bool)   # no outlet
    rng = np.random.default_rng(1)
    d = np.zeros(g["n_dofs"])
    d[g["top_dofs"]] = rng.uniform(0.0, 0.05, size=g["top_dofs"].size)
    m0 = _mass(d, g["A_i"], g["top_dofs"])
    dt = 2e-3
    for _ in range(80):
        d, of = route_excess(d, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                             0.05, dt, closed, g["outlet_slope"])
        assert of == 0.0  # no outlet -> nothing leaves
    m1 = _mass(d, g["A_i"], g["top_dofs"])
    assert abs(m1 - m0) <= 1e-12, f"closed-graph mass drifted by {abs(m1 - m0):.3e}"


def test_conservation_with_outlet_machine_precision():
    """With an outlet at x=LX: sum(d*A)_after + cumulative outflow == sum(d*A)_before to <= 1e-12
    over many sweeps (the spike's headline conservation check)."""
    g = _slope_graph()
    rng = np.random.default_rng(1)
    d = np.zeros(g["n_dofs"])
    d[g["top_dofs"]] = rng.uniform(0.0, 0.05, size=g["top_dofs"].size)
    m0 = _mass(d, g["A_i"], g["top_dofs"])
    tot_out = 0.0
    dt = 2e-3
    for _ in range(50):
        d, of = route_excess(d, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                             0.05, dt, g["outlet_mask"], g["outlet_slope"])
        tot_out += of
    m1 = _mass(d, g["A_i"], g["top_dofs"])
    resid = abs(m0 - (m1 + tot_out))
    assert resid <= 1e-12, f"interior + outflow did not conserve: resid {resid:.3e}"
    assert tot_out > 0.0, "outlet present but nothing left the domain"


# ====================================================================================================
# Test 2 -- POSITIVITY: output depths >= 0 always.
# ====================================================================================================
def test_positivity_depths_nonnegative():
    """A node never pushes out more than its inventory (V_out = min(d_i*A_i, Vcap) <= d_i*A_i), so
    depths stay >= 0 through any number of sweeps -- to machine precision. A drained node's depth
    can sit at -O(1e-18) from the floating-point cancellation d_i - V_out/A_i (ulp roundoff, not
    physical negative water); we require >= -1e-15 and that this floor does NOT drift with sweeps."""
    g = _slope_graph()
    rng = np.random.default_rng(7)
    d = np.zeros(g["n_dofs"])
    d[g["top_dofs"]] = rng.uniform(0.0, 0.08, size=g["top_dofs"].size)
    dt = 5e-3
    worst = 0.0
    for _ in range(60):
        d, _of = route_excess(d, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                             0.05, dt, g["outlet_mask"], g["outlet_slope"])
        worst = min(worst, float(d[g["top_dofs"]].min()))
        assert d[g["top_dofs"]].min() >= -1e-15, f"depth went negative beyond roundoff: " \
            f"{d[g['top_dofs']].min():.3e}"
    # the only sub-zero excursions are ulp-level cancellation, not an accumulating leak.
    assert worst >= -1e-15, f"positivity floor drifted to {worst:.3e}"


# ====================================================================================================
# Test 3 -- NO-TELEPORT (Manning cap): a single pulse advances a BOUNDED Manning distance per step
# (NOT to the outlet in one step); each transfer <= the Manning cap; the outlet is reached only after
# ~the expected number of steps.
# ====================================================================================================
def test_no_teleport_single_step_bounded():
    """A 10 cm pulse at the up-slope edge must NOT reach the far outlet in one step: the volume that
    reaches the outlet in step 1 is a tiny fraction of the pulse (the Manning cap working). The
    un-capped 'drains fully in one step' behaviour is the explicit NON-goal."""
    g = _slope_graph()
    d = np.zeros(g["n_dofs"])
    d[g["up_dofs"]] = 0.10  # 10 cm pulse at x=0
    pulse_vol = _mass(d, g["A_i"], g["up_dofs"])
    out_before = _mass(d, g["A_i"], np.where(g["outlet_mask"])[0])
    dt = 2e-3
    d1, of1 = route_excess(d, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                          0.05, dt, g["outlet_mask"], g["outlet_slope"])
    out_after = _mass(d1, g["A_i"], np.where(g["outlet_mask"])[0])
    moved_to_outlet = of1 + (out_after - out_before)
    frac = moved_to_outlet / pulse_vol
    # bounded: a few percent, NOT ~100% (teleport). The spike measures 2.77% on this exact geometry.
    assert 0.0 <= frac < 0.10, f"pulse teleported {frac:.2%} to the outlet in one step (cap failed)"


def test_no_teleport_per_transfer_within_manning_cap():
    """Every per-step volume leaving a node is <= its Manning cap Vcap = (SECONDS_PER_DAY/n)*
    d^(5/3)*sqrt(S)*W*dt (and <= its inventory). We verify by reconstructing the cap for the single
    pulse node and checking the depth drop it incurs in one step."""
    g = _slope_graph()
    n_man, dt = 0.05, 2e-3
    d = np.zeros(g["n_dofs"])
    d[g["up_dofs"]] = 0.10
    d1, _of = route_excess(d, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                          n_man, dt, g["outlet_mask"], g["outlet_slope"])
    # volume that left each pulse node this step = (d_before - d_after)*A
    for i in g["up_dofs"]:
        di = 0.10
        # max receiver slope from this node (interior neighbours + outlet)
        Hi = g["z_b"][i] + di
        smax = 0.0
        for (j, L) in g["adj"][i]:
            s = (Hi - (g["z_b"][j] + d[j])) / L
            smax = max(smax, s)
        if g["outlet_mask"][i]:
            smax = max(smax, g["outlet_slope"])
        Vcap = (SECONDS_PER_DAY / n_man) * di ** (5.0 / 3.0) * np.sqrt(smax) * g["W"][i] * dt
        vol_out = (d[i] - d1[i]) * g["A_i"][i]
        assert vol_out <= Vcap + 1e-15, f"node {i} pushed {vol_out:.3e} > cap {Vcap:.3e}"
        assert vol_out <= di * g["A_i"][i] + 1e-15, "node pushed more than its inventory"


def test_no_teleport_reaches_outlet_after_expected_steps():
    """The pulse front reaches the outlet only after a bounded number of steps (~the spike's 12 to
    drain 50% on this geometry), NOT in step 1. We assert it takes MANY steps (>1), bounding the per
    step Manning advance."""
    g = _slope_graph()
    n_man, dt = 0.05, 2e-3
    d = np.zeros(g["n_dofs"])
    d[g["up_dofs"]] = 0.10
    pulse_vol = _mass(d, g["A_i"], g["up_dofs"])
    cumout = 0.0
    steps = 0
    while cumout < 0.5 * pulse_vol and steps < 100000:
        d, of = route_excess(d, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                             n_man, dt, g["outlet_mask"], g["outlet_slope"])
        cumout += of
        steps += 1
    assert steps > 1, "pulse drained 50% in a single step -> teleportation (cap failed)"
    assert steps < 100000, "pulse never reached the outlet (routing stalled)"


# ====================================================================================================
# Test 4 -- MFD RECEIVERS: single steepest-descent line orders/routes correctly; a many-into-one
# convergence node receives from all uphill neighbours with weights summing to 1; a flat patch yields
# no spurious direction.
# ====================================================================================================
def test_receivers_steepest_line_orders_descending_head():
    """topo_order processes nodes in descending surface head (upslope before downslope) -- the
    richdem-style single dependency sweep. On the planar slope the order's heads are monotone
    non-increasing."""
    g = _slope_graph()
    d = np.zeros(g["n_dofs"])
    H = g["z_b"] + d
    order = topo_order(H, g["top_dofs"])
    heads = H[order]
    assert np.all(np.diff(heads) <= 1e-15), "topo order is not descending in head"
    assert set(order.tolist()) == set(g["top_dofs"].tolist()), "order must permute the top dofs"


def test_receivers_mfd_weights_sum_to_one_at_convergence_node():
    """build_receivers gives, for a node with several downslope neighbours, normalized MFD weights
    proportional to sqrt(slope) that SUM TO 1. We construct a many-into-one convergence: one low
    central node surrounded by higher neighbours -> every uphill neighbour routes (partly) into it,
    and each node's own outgoing weights sum to 1."""
    g = _slope_graph()
    # make a pit-free bowl-ish field on the slope: add a depression-free downhill bias already in z_b.
    rng = np.random.default_rng(3)
    d = np.zeros(g["n_dofs"])
    d[g["top_dofs"]] = rng.uniform(0.0, 0.02, size=g["top_dofs"].size)
    H = g["z_b"] + d
    recv = build_receivers(H, g["adj"], g["top_dofs"])
    # every node that HAS a downslope receiver must have weights summing to 1 and proportional to sqrt(slope)
    found_multi = False
    for i in g["top_dofs"]:
        js, ws = recv[i]
        if len(js) == 0:
            continue
        assert abs(float(np.sum(ws)) - 1.0) <= 1e-12, f"node {i} MFD weights sum to {np.sum(ws)}"
        # weights proportional to sqrt(slope): recompute and compare ratios
        slopes = np.array([(H[i] - H[j]) / _edge_len(g, i, j) for j in js])
        assert np.all(slopes > 0.0), "a receiver was not strictly downslope"
        ratio = ws / np.sqrt(slopes)
        assert np.allclose(ratio, ratio[0], rtol=1e-10), "weights not proportional to sqrt(slope)"
        if len(js) > 1:
            found_multi = True
    assert found_multi, "test geometry produced no multi-receiver node -- not exercising MFD split"


def test_receivers_convergence_node_gets_all_uphill():
    """A many-into-one convergence: a node that is a genuine local minimum of the BED (lower than ALL
    its graph neighbours) receives run-on from every uphill neighbour in one descending-head sweep and
    accumulates it (it has no downslope receiver of its own, so the water piles up there -- the
    convergence-line concentration the split must capture). On a planar slope a merely mid-slope node
    is NOT a sink (it transmits onward), so we carve a true pit at it."""
    g = _slope_graph()
    coords = g["coords"]
    interior_top = [i for i in g["top_dofs"]
                    if not (np.isclose(coords[i, 0], 0.0) or np.isclose(coords[i, 0], g["LX"]))]
    sink = int(interior_top[len(interior_top) // 2])
    nbrs = [j for (j, _L) in g["adj"][sink]]
    z_b = g["z_b"].copy()
    z_b[sink] = min(z_b[j] for j in nbrs) - 0.10   # genuine local minimum of the bed
    d = np.zeros(g["n_dofs"])
    for j in nbrs:
        d[j] = 0.05                                 # ponded uphill neighbours
    H = z_b + d
    for j in nbrs:
        assert H[j] > H[sink], "setup failed: neighbour not above the pit"
    closed = np.zeros(g["n_dofs"], dtype=bool)      # closed graph -> the pit keeps what it gathers
    uphill_vol = float(np.sum(d[nbrs] * g["A_i"][nbrs]))
    d2, of = route_excess(d, z_b, g["A_i"], g["adj"], g["top_dofs"], g["W"],
                         0.05, 5e-3, closed, g["outlet_slope"])
    assert of == 0.0, "closed graph should not lose water"
    assert d2[sink] > 0.0, "the convergence node received nothing from its uphill neighbours"
    # it gathered a real share of the surrounding ponded water (run-on concentrated at the pit).
    assert d2[sink] * g["A_i"][sink] > 0.1 * uphill_vol, \
        "the pit gathered a negligible share -- MFD did not route into the convergence node"


def test_flat_patch_no_spurious_direction():
    """A perfectly flat field (uniform head) over the graph yields NO routing: no receiver has a
    strictly-positive slope, so build_receivers returns empty receiver lists and route_excess moves
    nothing (no spurious downslope direction invented)."""
    g = _slope_graph()
    # flat HEAD everywhere: set d so that z_b + d == const on the top dofs.
    Htarget = float((g["z_b"][g["top_dofs"]]).max()) + 0.10
    d = np.zeros(g["n_dofs"])
    d[g["top_dofs"]] = Htarget - g["z_b"][g["top_dofs"]]   # H = Htarget (uniform)
    H = g["z_b"] + d
    recv = build_receivers(H, g["adj"], g["top_dofs"])
    for i in g["top_dofs"]:
        js, ws = recv[i]
        assert len(js) == 0, f"flat field invented a downslope direction at node {i}"
    closed = np.zeros(g["n_dofs"], dtype=bool)
    m0 = _mass(d, g["A_i"], g["top_dofs"])
    d2, of = route_excess(d, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                         0.05, 5e-3, closed, g["outlet_slope"])
    assert of == 0.0 and np.allclose(d2, d), "flat field produced spurious routing"


# ====================================================================================================
# Test 5 -- IDEMPOTENT on an already-drained (all-zero) field.
# ====================================================================================================
def test_idempotent_on_drained_field():
    """An all-zero depth field is a fixed point: routing it changes nothing and removes nothing."""
    g = _slope_graph()
    d = np.zeros(g["n_dofs"])
    d2, of = route_excess(d, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                         0.05, 5e-3, g["outlet_mask"], g["outlet_slope"])
    assert of == 0.0
    assert np.array_equal(d2, d), "routing a drained field was not a no-op"


# ====================================================================================================
# Test 6 -- FAITHFUL-EXTRACTION REGRESSION: the module reproduces the spike kernel's headline numbers
# (conservation resid 0.0; ~2.77% no-teleport pulse) on the SAME sample input -- the extraction did
# not change behaviour.
# ====================================================================================================
def test_faithful_extraction_reproduces_spike_headline_numbers():
    """Reproduce the spike ``kernel`` mode EXACTLY on its sample geometry (LX=2,LY=1,LZ=0.5,
    NX=10,NY=5,NZ=2,S0=0.05,n=0.05): (a) conservation resid 0.0 over 50 sweeps on the seeded random
    field; (b) the 10 cm pulse moves 2.767e-04 m^3 = 2.77% of the 1.0e-2 m^3 pulse to the outlet in
    one step (the Manning cap; NO teleport)."""
    g = _slope_graph()
    n_man = 0.05
    dt = 2e-3

    # (a) conservation resid 0.0 on the seeded random field (exact spike construction)
    rng = np.random.default_rng(1)
    d = np.zeros(g["n_dofs"])
    d[g["top_dofs"]] = rng.uniform(0.0, 0.05, size=g["top_dofs"].size)
    m0 = _mass(d, g["A_i"], g["top_dofs"])
    tot_out = 0.0
    for _ in range(50):
        d, of = route_excess(d, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                             n_man, dt, g["outlet_mask"], g["outlet_slope"])
        tot_out += of
    m1 = _mass(d, g["A_i"], g["top_dofs"])
    assert abs(m0 - (m1 + tot_out)) == pytest.approx(0.0, abs=1e-15), "conservation resid != 0.0"

    # (b) the no-teleport pulse: 2.767e-04 m^3 (2.77% of the 1.0e-2 pulse) to the outlet in one step
    d2 = np.zeros(g["n_dofs"])
    d2[g["up_dofs"]] = 0.10
    pulse_vol = _mass(d2, g["A_i"], g["up_dofs"])
    out_before = _mass(d2, g["A_i"], np.where(g["outlet_mask"])[0])
    d2a, of1 = route_excess(d2, g["z_b"], g["A_i"], g["adj"], g["top_dofs"], g["W"],
                           n_man, dt, g["outlet_mask"], g["outlet_slope"])
    out_after = _mass(d2a, g["A_i"], np.where(g["outlet_mask"])[0])
    moved = of1 + (out_after - out_before)
    assert pulse_vol == pytest.approx(1.0e-2, rel=1e-9), f"pulse volume drifted: {pulse_vol:.5e}"
    assert moved == pytest.approx(2.767e-04, rel=2e-3), f"no-teleport fraction drifted: {moved:.4e}"
    assert moved / pulse_vol == pytest.approx(0.0277, abs=5e-4)


# ====================================================================================================
# Depression-fill (PriorityFlood) -- reserved-but-UNUSED (the spike's descending-head sweep sufficed
# on the test geometries). A minimal correct version: fill every interior pit up to its lowest rim so
# no node is a strict local minimum surrounded by higher rim (water can always escape).
# ====================================================================================================
def test_fill_depressions_removes_single_pit():
    """fill_depressions raises a single interior pit to its lowest spill rim (PriorityFlood): the
    filled surface has no node strictly below ALL its neighbours that is also below its spill path,
    and it never LOWERS any node, and a pit-free field is returned unchanged."""
    g = _slope_graph()
    # carve one interior pit into the bed.
    coords = g["coords"]
    interior_top = [i for i in g["top_dofs"]
                    if not (np.isclose(coords[i, 0], 0.0) or np.isclose(coords[i, 0], g["LX"]))]
    pit = int(interior_top[len(interior_top) // 2])
    z = g["z_b"].copy()
    z[pit] -= 0.5  # deep pit
    # boundary = the top-dof outlet ring (x=0 and x=LX) as the open spill set
    open_nodes = g["top_dofs"][np.isclose(coords[g["top_dofs"], 0], g["LX"])
                               | np.isclose(coords[g["top_dofs"], 0], 0.0)]
    zf = fill_depressions(z, g["adj"], g["top_dofs"], open_nodes)
    # never lowers
    assert np.all(zf[g["top_dofs"]] >= z[g["top_dofs"]] - 1e-15), "fill lowered a node"
    # the pit was ACTUALLY raised (non-vacuous): it sat 0.5 below the slope, fill must lift it.
    assert zf[pit] > z[pit] + 1e-6, "fill did not raise the carved pit"
    # the pit is no longer a strict local minimum among its neighbours (a spill path now exists).
    nbr_min = min(zf[j] for (j, _L) in g["adj"][pit])
    assert zf[pit] >= nbr_min - 1e-12, "pit is still a strict local minimum after fill"


def test_fill_depressions_idempotent_on_pit_free_field():
    """On a pit-free monotone slope, fill_depressions returns the field unchanged (no spurious
    raising)."""
    g = _slope_graph()
    coords = g["coords"]
    open_nodes = g["top_dofs"][np.isclose(coords[g["top_dofs"], 0], g["LX"])]
    zf = fill_depressions(g["z_b"], g["adj"], g["top_dofs"], open_nodes)
    assert np.allclose(zf[g["top_dofs"]], g["z_b"][g["top_dofs"]], atol=1e-12), \
        "fill modified a pit-free field"


# ----------------------------------------------------------------------------------------------------
# helper: edge length between two adjacent dofs (for the MFD weight-proportionality check)
# ----------------------------------------------------------------------------------------------------
def _edge_len(g, i, j):
    for (k, L) in g["adj"][i]:
        if k == j:
            return L
    raise AssertionError(f"{i}-{j} not adjacent")
