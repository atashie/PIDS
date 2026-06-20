"""Tier-1 sanity for the permanent COUPLED CONVERGENT-FLOW dual-drain fixture (P3 Part B).

The convergent-flow workstream (P0-P3) fixed the coupled UPWIND overland solver for the regime where
surface water concentrates along convergence lines -- where PIDS networks/inlets install. This fixture
is the permanent Tier-2 regression for that fix: a convergent graded topography (the topographic
SETTING, variable geometry -- NOT a PIDS "feature") on a 3-D loam-over-clay host, the upwind scheme, and
the signed-off dual-drain (add_surface_inlet + add_interior_drain -- the coupled-integrated PIDS drainage
elements) extended to 3-D convergent flow. Builder/harness: scratch/_p3_convergent_fixture.py;
scope benchmarks/convergent_dualdrain_scope.md. Framing: Arik 2026-06-19 (see pids-p3-convergent-progress).

Tier-1 (this file, fast): the fixture is well-posed -- structural conservation with ALL sinks,
positivity within the upwind tripwire, BOTH embedded elements capture water (and the inlet rises with
ponding), and the convergent topography concentrates the overland depth onto the line. The full storm
matrix is Tier-2 (scratch/_p3_convergent_storm_matrix.py).
"""
import numpy as np
import pytest

from scratch._p3_convergent_fixture import make_convergent_dualdrain, drive_phase, balance_residual

# small + short fast config (the coupled upwind convergent ponding is stiff -> keep the in-suite drive
# minimal); floor W=12 spans 6 cells at NX=20 on LX=40. Water table z_wt just above the interface tile.
_LX, _LY, _NX, _NY, _NZ, _W = 40.0, 24.0, 20, 12, 4, 12.0
_RATE, _STORM_T, _RECESS_T = 0.35, 0.012, 0.006   # > Ks=0.25 -> infiltration-excess convergence ponding


@pytest.fixture(scope="module")
def conv_run():
    """Drive the convergent dual-drain fixture through a short storm + recession ONCE; the Tier-1 pins
    assert on the cached state (amortizes the stiff coupled run across the four checks)."""
    prob, info, _drain, _inlet = make_convergent_dualdrain(
        _NX, _NY, _NZ, Lx=_LX, Ly=_LY, W=_W)
    rain = prob.add_rain(0.0)
    w0 = prob.total_water()
    accum = [0.0]
    drive_phase(prob, rain, _RATE, _STORM_T, info["top_area"], accum)        # storm: pond + capture
    topdofs = prob._top_dofs(prob.Vd)
    coords = prob.Vd.tabulate_dof_coordinates()[topdofs]
    snap = dict(
        d_storm=prob.d.x.array[topdofs].copy(), top_x=coords[:, 0].copy(),
        q_inlet_storm=prob.last_sinks["surface_inlet"][0],
        q_drain_storm=prob.last_sinks["interior_drain"][0], d_peak=prob.surface_depth())
    drive_phase(prob, rain, 0.0, _RECESS_T, info["top_area"], accum)         # recession
    return dict(prob=prob, info=info, w0=w0, cum_rain=accum[0],
                q_inlet_recess=prob.last_sinks["surface_inlet"][0],
                cum_inlet=prob.cum_sinks["surface_inlet"][0],
                cum_drain=prob.cum_sinks["interior_drain"][0], **snap)


def test_convergent_dualdrain_conserves(conv_run):
    """Structural mass balance with ALL sinks: Delta_total == cum_rain - cum_outflow - cum_drainage
    (+ clip=0 on the upwind tripwire), to solver precision -- across the storm + recession, the open
    convergence-floor outlet, the surface inlet, AND the interior tile drain."""
    prob = conv_run["prob"]
    assert balance_residual(prob, conv_run["w0"], conv_run["cum_rain"]) <= 1e-6 * conv_run["cum_rain"]
    assert prob.clip_mass_adjust == 0.0           # upwind tripwire never clips (no silent mass change)


def test_convergent_dualdrain_positivity(conv_run):
    """The monotone upwind scheme holds d within the characterized sub-cm tripwire band on the
    convergent ponding front (never the galerkin cm-scale clip)."""
    prob = conv_run["prob"]
    assert prob.d.x.array.min() >= -prob._upwind_pos_tol     # within the 5mm tripwire
    assert prob.max_clip_seen < 5e-3                          # sub-cm
    assert np.all(np.isfinite(prob.psi.x.array)) and np.all(np.isfinite(prob.d.x.array))


def test_convergent_dualdrain_both_elements_capture(conv_run):
    """BOTH embedded PIDS drainage elements engage: the surface grate inlet captures the concentrated
    ponded run-on AND the interior tile drain captures subsurface water (cum_sinks > 0 for each); and
    the inlet capture RISES WITH PONDING (storm rate > recession rate)."""
    assert conv_run["cum_inlet"] > 0.0, "surface inlet captured nothing"
    assert conv_run["cum_drain"] > 0.0, "interior tile drain captured nothing"
    assert conv_run["q_inlet_storm"] > conv_run["q_inlet_recess"]   # inlet rises with ponding
    assert conv_run["d_peak"] > 1e-3                                # genuine convergence ponding (>1mm)


def test_convergent_overland_concentrates_on_the_line(conv_run):
    """The convergent topography + upwind scheme CONCENTRATE the overland depth onto the convergence
    line: at the storm peak the mean ponded depth on the floor (|x-Xc|<W/2) exceeds the mean on the
    side slopes -- water routes down the side slopes INTO the low line (the regime the fix targets)."""
    info, x, d = conv_run["info"], conv_run["top_x"], conv_run["d_storm"]
    on_floor = np.abs(x - info["Xc"]) < 0.5 * info["W"]
    on_sides = np.abs(x - info["Xc"]) > 0.5 * info["W"] + info["Lx"] / _NX
    assert d[on_floor].mean() > d[on_sides].mean()      # depth concentrates on the convergence line
