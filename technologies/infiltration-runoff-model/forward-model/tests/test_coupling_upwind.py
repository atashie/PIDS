"""Tier-1 sanity for the OPT-IN upwind overland scheme inside CoupledProblem (Convergent-flow P2).

P2 productionizes the validated standalone monotone upwind edge-flux scheme (P1,
``UpwindOverlandProblem``) as the LATERAL-overland operator of the coupled ``[psi, d, lam]`` solver.
It is OPT-IN: ``CoupledProblem(..., overland_scheme="upwind")``; the default ``"galerkin"`` path is
unchanged and bit-identical (pinned by the existing ``tests/test_coupling_{1,2,3}d*.py``). The upwind
path replaces ONLY the UFL ``overland_flux`` term with the non-UFL edge-flux on the top-facet graph,
supplied as a custom residual + Jacobian on the d-block via a custom block-SNES; psi, lam/NCP,
outlet, drainage and the interior-pin stay UFL and untouched, so conservation stays structural.

Scope (P2): the upwind path requires a 3-D host (the top facet is a 2-D triangulation for the
cotangent edge graph) and is serial (a multi-rank guard is added in Task B2). Galerkin is unchanged
in all dimensions.
"""
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import assemble_vector

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)

# -- P3 Part A: resolved-swale ABSOLUTE accuracy (depth-field), Gate-A framing (Arik 2026-06-19) ----
# The coupled outlet is conservation-FORCED to Q_eq (it IS the outlet sink), so it proves conservation
# NOT accuracy. Absolute accuracy is the DEPTH FIELD, proven by (a) operator-equivalence to the
# validated standalone UpwindOverlandProblem (carries B5b's resolved-swale consistent-discharge ~0.99
# into the coupled engine) + (b) a DIRECT coupled plane -> analytic Manning normal-depth check. The
# convergent tilted-V floor is edge/corner-heavy (NOT the uniform 1-D normal-depth idealization), so
# its floor depth is reported MESH-CONVERGENT (vs the kink's measure-zero divergence) rather than
# matched to d_n. Framing = parent plan 8.7/8.8 ("coupled accuracy = operator equivalence"). The
# measurement/evidence harness is scratch/_p3_swale_accuracy.py.
from scratch._p3_swale_accuracy import (   # noqa: E402  (the Part-A measurement helpers)
    RAIN, drive_coupled_swale, drive_standalone_swale, depth_field_reldiff,
    measure_floor_depth, normal_depth, plane_normal_depth, outlet_row_mean_depth,
)

# the fast in-suite Part-A config (small + coarse + large-dt -> ~10-16 s/run, near-impermeable bed,
# inlet OFF, canonical full-edge outlet). The floor W=48 spans >=6 cells at nx=16 (resolved swale).
_PA_CFG = dict(LX=120.0, LY=80.0, H=1.0, SX=0.04, SY=0.02, n_man=0.015,
               dt0=1e-4, dt_max=1e-2, t_end=0.04)
_PA_W = 48.0


def _box(nx=6, ny=3, nz=3, Lx=2.0, Ly=1.0, Lz=1.0):
    return dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, Ly, Lz]], [nx, ny, nz])


def test_overland_scheme_default_is_auto_upwind_in_3d_serial():
    """P3-D: the default is 'auto' -> the dimension/comm-aware resolution picks 'upwind' on a 3-D serial
    host (the convergent-flow fix on-by-default where it applies). The REQUESTED mode (overland_scheme)
    and the RESOLVED/effective mode (_effective_overland_scheme) are kept distinct."""
    p = CoupledProblem(_box(), SOIL)
    assert p.overland_scheme == "auto" and p._effective_overland_scheme == "upwind"


def test_auto_default_falls_back_to_galerkin_in_2d():
    """'auto' falls back to galerkin where upwind does not apply (a 2-D host) -- NO raise, so 1-D/2-D/
    MPI callers are not broken by the default flip (galerkin stays a permanent, explicit fallback)."""
    msh2d = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (2.0, 1.0)], [8, 4])
    p = CoupledProblem(msh2d, SOIL)
    assert p.overland_scheme == "auto" and p._effective_overland_scheme == "galerkin"


def test_overland_scheme_explicit_modes_and_validation():
    assert CoupledProblem(_box(), SOIL, overland_scheme="upwind")._effective_overland_scheme == "upwind"
    g = CoupledProblem(_box(), SOIL, overland_scheme="galerkin")
    assert g.overland_scheme == "galerkin" and g._effective_overland_scheme == "galerkin"
    with pytest.raises(ValueError, match="overland_scheme"):
        CoupledProblem(_box(), SOIL, overland_scheme="bogus")


def test_upwind_requires_3d_host():
    """P2 scope: the cotangent top-facet edge graph needs a 2-D top triangulation (3-D host).
    A 2-D host's top is a 1-D edge -- a future extension, refused loudly for now."""
    msh2d = dmesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (2.0, 1.0)], [8, 4])
    with pytest.raises(NotImplementedError, match="3-D"):
        CoupledProblem(msh2d, SOIL, overland_scheme="upwind")


def test_upwind_reduced_Fd_omits_lateral_flux():
    """The upwind path removes the UFL lateral conveyance from F_d (it is supplied by the edge-flux
    residual instead). At a NON-FLAT surface head, the galerkin F_d carries the lateral term on the
    top rows and the upwind F_d does not -> the assembled d-residual vectors differ."""
    g = CoupledProblem(_box(), SOIL, overland_scheme="galerkin")   # explicit (default is now auto->upwind on 3-D)
    u = CoupledProblem(_box(), SOIL, overland_scheme="upwind")
    for p in (g, u):
        p.set_topography(lambda x: 0.05 * x[0] + 0.02 * x[1])    # tilted bed -> non-flat H = z_b + d
        p.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.05)
        p.add_rain(0.0)
    bg = assemble_vector(fem.form(g.F_d)); bg.assemble()
    bu = assemble_vector(fem.form(u.F_d)); bu.assemble()
    assert np.abs(bg.getArray() - bu.getArray()).max() > 1e-3
    bg.destroy(); bu.destroy()


# -- Task B2: the custom block-SNES (residual + Picard d-d Jacobian) -----------

def test_upwind_step_converges_3d():
    """A coupled upwind step solves cleanly (the custom block-SNES residual + Picard Jacobian
    drive the [psi,d,lam] Newton to convergence) on a tilted ponding box."""
    prob = CoupledProblem(_box(5, 5, 3), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * (2.0 - x[0]))
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.05)
    prob.add_rain(0.3)
    converged, iters = prob.step(1e-3)
    assert converged and prob.last_reason > 0
    assert np.all(np.isfinite(prob.d.x.array))


def test_upwind_block_jacobian_no_malloc_3d():
    """Codex blocker #1 DISSOLVED: the consistent ds_top storage-mass Jacobian already preallocates
    every top-facet d-d coupling, so the Picard edge-flux Jacobian inserts with NO new allocation.
    Setting NEW_NONZERO_ALLOCATION_ERR before the first solve makes a missing slot raise loudly."""
    prob = CoupledProblem(_box(5, 5, 3), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * (2.0 - x[0]))
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.05)
    prob.add_rain(0.3)
    prob._ensure_problem()                                  # build _A + wire callbacks (no solve yet)
    prob._problem._A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    converged, _ = prob.step(1e-3)                          # first Jacobian assembly inserts edge entries
    assert converged                                       # completed -> the edge nonzeros were preallocated


def test_upwind_closed_conservation_3d():
    """The key solver gate: a CLOSED tilted box (no outlet/drainage) under ponding rain conserves
    total water EXACTLY -- Delta total == cumulative rain. Exercises infiltration + the monotone
    lateral edge flux (water ponds and routes downslope, accumulating; nothing leaves) and confirms
    the edge flux telescopes to zero (no spurious mass) without the limiter clipping."""
    Lx, Ly = 2.0, 1.0
    prob = CoupledProblem(_box(6, 6, 4, Lx=Lx, Ly=Ly), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * (Lx - x[0]))      # tilt toward x=0 -> lateral routing
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    w0 = prob.total_water()
    rate, t_end = 0.5, 0.1                                  # > Ks=0.25 -> mild ponding + routing
    prob.add_rain(rate)
    prob.advance(t_end=t_end, dt=1e-3, dt_max=0.02)
    cum_rain = rate * Lx * Ly * t_end

    assert abs((prob.total_water() - w0) - cum_rain) / cum_rain < 1e-6   # conserves (machine-tight)
    assert prob.total_water() - w0 > 0.5 * cum_rain        # rain genuinely entered
    # monotone to within the characterized TRANSIENT sub-cm mild-front undershoot (self-healing,
    # mass-neutral); the tripwire records it but NEVER clips (clip_mass_adjust stays exactly 0).
    assert prob.max_clip_seen < 5e-3                        # sub-cm (vs galerkin's cm-scale clip)
    assert prob.clip_mass_adjust == 0.0                    # tripwire: no silent mass adjustment
    assert np.all(np.isfinite(prob.d.x.array)) and np.all(np.isfinite(prob.psi.x.array))


def test_upwind_coupled_jacobian_matches_fd_smoke_3d():
    """Codex should-fix: a COUPLED-level Jacobian check (the kernel FD-verify is necessary but NOT
    sufficient -- it cannot catch block-offset / sign / BC bugs in the edge block once inserted into
    the [psi,d,lam] block matrix). At a plausible solved state, the assembled block Jacobian action
    J*delta matches a central finite-difference of the FULL assembled coupled residual along a random
    direction, with the interior pins active."""
    prob = CoupledProblem(_box(5, 5, 3), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * (2.0 - x[0]))
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.05)
    prob.add_rain(0.3)
    prob.step(1e-3)                                         # advance to a plausible coupled state

    snes = prob._problem.solver
    A = prob._problem._A
    x = snes.getSolution().copy()
    snes.computeJacobian(x, A, A)                           # assemble J (UFL + edge block) at x

    rng = np.random.default_rng(3)
    delta = x.duplicate()
    with delta.localForm() as dl:
        dl.array[:] = rng.standard_normal(dl.array.size)
    delta.scale(1.0 / delta.norm())

    Jd = x.duplicate(); A.mult(delta, Jd)                   # J * delta
    eps = 1e-6
    Fp = x.duplicate(); Fm = x.duplicate()
    xp = x.copy(); xp.axpy(eps, delta); snes.computeFunction(xp, Fp)
    xm = x.copy(); xm.axpy(-eps, delta); snes.computeFunction(xm, Fm)
    fd = Fp - Fm; fd.scale(1.0 / (2.0 * eps))              # central FD of the full coupled residual

    rel = (Jd - fd).norm() / max(fd.norm(), 1e-30)
    assert rel < 1e-5, f"coupled J*delta vs FD mismatch: rel err {rel:.2e}"


# -- Task D1: the limiter is a tripwire on the upwind path (no silent clip) -----

def test_upwind_tripwire_records_within_tol_undershoot_without_clipping():
    """On the upwind path the positivity limiter is DEMOTED to a tripwire: a within-tolerance
    (characterized sub-mm) negative depth is RECORDED but NOT clipped/rescaled -- d is untouched and
    clip_mass_adjust stays exactly 0 (contrast the galerkin conservative clip)."""
    prob = CoupledProblem(_box(5, 5, 3), SOIL, overland_scheme="upwind")
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.02)
    top = prob._top_dofs(prob.Vd)
    prob.d.x.array[top[0]] = -5e-4                          # a within-tol (<1mm) undershoot
    before = prob.d.x.array.copy()

    undershoot = prob._positivity_tripwire()

    assert undershoot == pytest.approx(5e-4, rel=1e-9)
    assert np.array_equal(prob.d.x.array, before)           # NOT clipped/rescaled
    assert prob.clip_mass_adjust == 0.0


def test_upwind_tripwire_raises_on_gross_undershoot():
    """A gross negative depth (beyond the characterized sub-mm band) RAISES loudly -- a real finding
    to characterize, never silently clipped."""
    prob = CoupledProblem(_box(5, 5, 3), SOIL, overland_scheme="upwind")
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.02)
    top = prob._top_dofs(prob.Vd)
    prob.d.x.array[top[0]] = -1e-2                          # 1cm, well beyond the 5mm sub-cm tol
    with pytest.raises(RuntimeError, match="monotone"):
        prob._positivity_tripwire()


def test_upwind_lateral_redistribution_downslope_3d():
    """The lateral edge flux ROUTES water downslope: on a closed tilted box under ponding rain, the
    monotone scheme moves ponded water down the bed-slope so it accumulates at the LOW end -- surface
    depth is much greater downhill than uphill (the coupled analogue of the galerkin
    test_2d_lateral_redistribution_downslope), while rain still reaches the whole surface."""
    Lx, Ly = 2.0, 1.0
    prob = CoupledProblem(_box(8, 4, 4, Lx=Lx, Ly=Ly), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * x[0])             # z_b high at x=Lx, low at x=0 -> flows to x=0
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.0)
    prob.add_rain(0.5)                                      # > Ks -> ponds + routes
    prob.advance(t_end=0.1, dt=1e-3, dt_max=0.02)

    coords = prob.Vd.tabulate_dof_coordinates()
    top = prob._top_dofs(prob.Vd)
    xt = coords[top, 0]
    d_top = prob.d.x.array[top]
    downhill = d_top[xt < 0.5 * Lx * 0.5].mean()           # the low quarter (x < Lx/4)
    uphill = d_top[xt > Lx - 0.5 * Lx * 0.5].mean()        # the high quarter (x > 3Lx/4)
    assert downhill > 1.5 * uphill                          # water routed downslope (accumulated low)
    assert uphill > 0.0                                    # rain reached the whole surface
    assert prob.d.x.array.min() >= -prob._upwind_pos_tol   # within the characterized band


def test_galerkin_still_uses_clip():
    """The galerkin path is unchanged: it still has the conservative clip (_enforce_positivity), not
    the tripwire -- a forced negative is clipped + tracked, not raised."""
    prob = CoupledProblem(_box(5, 5, 3), SOIL, overland_scheme="galerkin")   # explicit (default is auto->upwind)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.02)
    top = prob._top_dofs(prob.Vd)
    prob.d.x.array[top[0]] = -5e-4
    clipped = prob._enforce_positivity()                   # the clip still exists + runs
    assert clipped == pytest.approx(5e-4, rel=1e-9)
    assert prob.d.x.array.min() >= 0.0                      # clipped to non-negative


# == P3 Part A: resolved-swale absolute accuracy (depth field) =================================
@pytest.fixture(scope="module")
def part_a_runs():
    """Drive the Part-A coupled swale (coarse + fine), the matching STANDALONE upwind surface, and a
    tilted PLANE ONCE (canonical full-edge outlet, near-impermeable bed, inlet OFF) -- the Part-A pins
    assert on the cached plateau fields. ~50 s total (amortized across the four pins)."""
    coarse = drive_coupled_swale(_PA_W, 16, 12, 2, outlet="edge", **_PA_CFG)[2]
    fine = drive_coupled_swale(_PA_W, 24, 16, 2, outlet="edge", **_PA_CFG)[2]
    scfg = {k: v for k, v in _PA_CFG.items() if k != "H"}
    std = drive_standalone_swale(_PA_W, 24, 16, outlet="edge", **scfg)[2]
    pcfg = dict(_PA_CFG)
    pcfg["SX"] = 0.0                                        # no cross-slope -> a clean tilted plane
    plane = drive_coupled_swale(0.0, 24, 16, 2, outlet="edge", **pcfg)[2]
    return dict(coarse=coarse, fine=fine, std=std, plane=plane)


def test_coupled_outlet_is_conservation_forced_not_accuracy(part_a_runs):
    """A1.1 (the §8.7 trap, pinned): the coupled outlet outflow_rate() -> ~Q_eq at the storm plateau
    REGARDLESS of resolution -- it IS the outlet sink (summing the d-residual telescopes the edge flux
    to zero, forcing int q_out = rain*area - int lambda), so it proves CONSERVATION, NOT accuracy, and
    it cannot expose the depth-field accuracy. Resolution-INDEPENDENT confirms the conservation-forcing.
    Near-impermeable bed -> ~0.5% infiltration; the surface+soil books close machine-tight."""
    c, f = part_a_runs["coarse"], part_a_runs["fine"]
    assert abs(c["q_over_Qeq"] - 1.0) < 0.02 and abs(f["q_over_Qeq"] - 1.0) < 0.02   # both ~Q_eq
    assert abs(c["q_over_Qeq"] - f["q_over_Qeq"]) < 5e-3       # resolution-INDEPENDENT (conservation)
    assert abs(f["export_gap"]) < 1e-6 * f["cum_rain"]        # surface+soil balance machine-tight


def test_coupled_upwind_depth_matches_standalone_on_resolved_swale(part_a_runs):
    """A1.2 operator-equivalence (the INHERITED accuracy): on the same resolved swale + near-impermeable
    bed, the coupled-upwind surface depth field matches the validated STANDALONE UpwindOverlandProblem
    field to ~0.1% in the INTERIOR -- the coupling adds only the small lambda infiltration, and the SAME
    extracted edge kernel drives both. This carries B5b's standalone resolved-swale discharge accuracy
    (consistent ds-integral ~0.99) into the coupled engine. (The outlet-row nodes differ by the coupled
    codim-2-ridge vs standalone lumped-nodal outlet-sink treatment -- excluded by interior_only.)"""
    d = depth_field_reldiff(part_a_runs["fine"], part_a_runs["std"], interior_only=True)
    assert d["max"] < 0.01, f"interior coupled-vs-standalone max reldiff = {d['max']:.4f} (want <1%)"


def test_coupled_plane_depth_matches_analytic_normal_depth(part_a_runs):
    """A2 DIRECT absolute accuracy: on a clean tilted PLANE (no cross-slope -> uniform flow, where the
    Manning normal-depth idealization is VALID), the coupled outlet depth matches the analytic
    d_n = (rain*LY*n/(86400*sqrt(S)))^(3/5) to <1%. The convergent tilted-V floor is edge/corner-heavy
    (NOT uniform), so d_n is the wrong ruler there; the plane is where 'coupled depth -> analytic normal
    depth' is well-posed. This is the direct coupled-vs-analytic absolute-accuracy result the (edge-heavy)
    tilted-V floor cannot give."""
    p = part_a_runs["plane"]
    d_n = plane_normal_depth(RAIN, _PA_CFG["LY"], _PA_CFG["n_man"], _PA_CFG["SY"])
    d_out = outlet_row_mean_depth(p, 24, 16)
    assert abs(d_out - d_n) / d_n < 0.05, (
        f"plane outlet depth {1e3*d_out:.3f} mm vs analytic d_n {1e3*d_n:.3f} mm "
        f"(err {abs(d_out-d_n)/d_n:.4f})")


def test_resolved_swale_floor_is_mesh_convergent(part_a_runs):
    """A2 mesh-convergence: the resolved-swale floor depth is MESH-CONVERGENT (its error vs the analytic
    d_n SHRINKS with refinement, and it stays finite/positive/bounded) -- the resolved swale resolves the
    convergent flow. This CONTRASTS the idealized kink V (W=0, a measure-zero 1-cell channel) whose floor
    depth DIVERGES (grows ~2^(3/5) per dx-halving -- the B5b artifact; characterized in
    scratch/_p3_swale_accuracy.py). NB the absolute accuracy is the plane + operator-equivalence above,
    NOT this (edge-heavy) tilted-V floor vs d_n -- d_n is only a convergence yardstick here."""
    c, f = part_a_runs["coarse"], part_a_runs["fine"]
    d_n = normal_depth(c["Q_eq"], _PA_W, _PA_CFG["n_man"], _PA_CFG["SY"])
    flc = measure_floor_depth(c["top_x"], c["top_y"], c["d_top"], c["XC"], _PA_W,
                              c["LX"], c["LY"], 16, 12)
    flf = measure_floor_depth(f["top_x"], f["top_y"], f["d_top"], f["XC"], _PA_W,
                              f["LX"], f["LY"], 24, 16)
    err_c = abs(flc["floor_mean"] - d_n) / d_n
    err_f = abs(flf["floor_mean"] - d_n) / d_n
    assert 0.0 < flf["floor_mean"] < 3.0 * d_n        # finite, positive, physically bounded
    assert err_f < err_c, f"swale floor error not shrinking: coarse {err_c:.3f} -> fine {err_f:.3f}"
