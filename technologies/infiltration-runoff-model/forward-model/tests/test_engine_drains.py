"""Tier-1 sanity: first-class INTERIOR tile drain + SURFACE grate inlet on CoupledProblem.

Plan: docs/plans/2026-06-10-module4-engine-drain-inlet-apis.md (promoting the signed-off dual-drain
illustration's script-level sinks, commit 63145e2, into validated engine APIs).

- ``add_interior_drain(locator, conductance_density, drain_head, *, eps_act)``: a MODFLOW-DRN-style
  VOLUMETRIC, OUTFLOW-ONLY sink over the CELLS matched by ``locator`` (interior horizons allowed --
  the facet GHB cannot tag interior facets or the lambda-coupled top):
      q_vol = C * kr(psi) * pos(psi + z - drain_head)   [1/day],  pos = smooth max (C-inf)
  An air-filled pipe at atmospheric NEVER injects (contrast: the GHB is bidirectional by design).
- ``add_surface_inlet(locator, intake_coeff)``: a grate/catch-basin intake on the PONDED depth over
  a top-surface footprint, q = C*d [m/day] -- removes surface water only.
Both are wired into the engine's drainage accounting (cum_drainage and the structural balance
include them); ``sink_rates()`` / ``cum_sinks`` expose the per-sink split the dual-drain
illustration had to reconstruct by hand.
"""
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def _rect(nx, nz, Lx=1.0, Lz=1.0):
    return dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Lz]], [nx, nz],
                                  dmesh.CellType.triangle)


# --------------------------------------------------------------------- interior drain: physics
def test_interior_drain_conserves_and_drains():
    """Closed box (no rain/outlet), saturated band over the drain: water leaves ONLY through the
    interior drain and Delta_total = -cum_drainage to solver precision (structural conservation)."""
    prob = CoupledProblem(_rect(4, 8), LOAM)
    prob.set_initial_condition(lambda x: 0.5 - x[1], d_value=0.0)   # water table at z=0.5
    prob.add_interior_drain(lambda x: x[1] < 0.13, conductance_density=4.0, drain_head=0.0)
    w0 = prob.total_water()
    prob.advance(t_end=0.3, dt=2e-3, dt_max=2e-2)
    assert prob.cum_drainage > 1e-4                       # the drain genuinely discharges
    bal = (prob.total_water() - w0) - (-prob.cum_drainage + prob.clip_mass_adjust)
    assert abs(bal) / prob.cum_drainage < 1e-8
    assert np.all(np.isfinite(prob.psi.x.array))


def test_interior_drain_exact_rate_and_ghb_limit():
    """Saturated box (kr=1, pos exact): the instantaneous DRN rate equals the analytical band
    integral C_dens * Int_band (psi0 + z) dV to ~1e-4, and a thin base-hugging band approaches the
    equivalent base GHB (same band-integrated conductance) within the band-thickness bias."""
    h, C_dens, psi0 = 0.0625, 4.0, 0.2
    prob = CoupledProblem(_rect(4, 16), LOAM)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)   # saturated everywhere
    prob.add_interior_drain(lambda x: x[1] < h + 1e-3, conductance_density=C_dens, drain_head=0.0)
    rate = prob.sink_rates()["interior_drain"][0]
    rate_ana = C_dens * (psi0 * h + 0.5 * h * h) * 1.0    # Int over [0,1]x[0,h] of (psi0+z)
    assert rate == pytest.approx(rate_ana, rel=1e-4)
    # GHB limit: equivalent base GHB conductance C = C_dens*h; DRN exceeds it only by the band's
    # mean-z bias (h/2 of head) -- within 20% here.
    prob2 = CoupledProblem(_rect(4, 16), LOAM)
    prob2.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    prob2.add_drainage_bc(lambda x: np.isclose(x[1], 0.0), conductance=C_dens * h,
                          external_head=0.0)
    assert rate == pytest.approx(prob2.drainage_rate(), rel=0.2)


def test_interior_drain_kr_weighting_unsaturated_active():
    """An UNSATURATED band with head ABOVE the invert (drain_head well below it): the DRN rate is
    kr-WEIGHTED, q = C·kr(psi0)·Int(psi0+z−He) -- pins the relative-permeability factor directly
    (the saturated analytic test has kr=1 and cannot catch a dropped kr; mirrors
    test_drainage_relative_permeability_weighting for the GHB)."""
    h, C_dens, psi0, He = 0.125, 4.0, -0.1, -0.5
    prob = CoupledProblem(_rect(4, 8), LOAM)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    prob.add_interior_drain(lambda x: x[1] < h + 1e-3, conductance_density=C_dens, drain_head=He)
    kr = LOAM.K(psi0) / LOAM.Ks
    assert 0.05 < kr < 0.9                                # genuinely unsaturated AND active
    rate_ana = C_dens * kr * ((psi0 - He) * h + 0.5 * h * h)   # Int over [0,1]x[0,h] of (psi0+z-He)
    assert prob.sink_rates()["interior_drain"][0] == pytest.approx(rate_ana, rel=1e-4)


def test_interior_drain_outflow_only_never_injects():
    """An UNSATURATED band (head below the pipe) gives only the documented smooth-max leak -- the
    DRN never injects (rate >= 0 always), unlike the bidirectional GHB inflow branch."""
    prob = CoupledProblem(_rect(4, 8), LOAM)
    prob.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.0)  # uniformly unsaturated
    prob.add_interior_drain(lambda x: (x[1] > 0.36) & (x[1] < 0.64),
                            conductance_density=300.0, drain_head=0.5)
    rate = prob.sink_rates()["interior_drain"][0]
    assert rate >= 0.0                                    # NEVER negative (no injection)
    assert rate < 1e-4                                    # only the eps_act activation leak
    prob.advance(t_end=0.05, dt=2e-3, dt_max=5e-3)
    assert prob.cum_drainage < 1e-5                       # stays inert while unsaturated


def test_interior_drain_removes_perched_water():
    """Two-layer mini-slab with a perched saturated loam band on a tight-clay interface: WITHOUT a
    drain the perch persists (the clay leaks ~nothing in the window); WITH an interface drain the
    band visibly de-saturates -- the validated dual-run physics as an engine regression."""
    z_if = 0.5

    class TwoLayer:
        """Minimal duck-typed loam-over-clay (the UFL surface the engine consumes)."""

        def __init__(self, mesh):
            self._loam = LOAM
            self._clay = VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.005)
            zc = ufl.SpatialCoordinate(mesh)[mesh.geometry.dim - 1]
            self._cond = ufl.ge(zc, z_if - 1e-9)
            self.Ks, self.theta_r, self.theta_s = LOAM.Ks, LOAM.theta_r, LOAM.theta_s

        def theta_ufl(self, psi):
            return ufl.conditional(self._cond, self._loam.theta_ufl(psi), self._clay.theta_ufl(psi))

        def K_ufl(self, psi):
            return ufl.conditional(self._cond, self._loam.K_ufl(psi), self._clay.K_ufl(psi))

        def kirchhoff_ufl(self, a, b):
            return self._loam.kirchhoff_ufl(a, b)   # infiltration leg lives in the loam top

    # perched IC: saturated loam band z in [0.5, 0.7] on the clay; dry-ish above; deep clay table
    ic = lambda x: np.where(x[1] >= z_if, 0.7 - x[1], 0.1 - x[1])

    def run(with_drain):
        msh = _rect(4, 16)
        prob = CoupledProblem(msh, TwoLayer(msh))
        prob.set_initial_condition(ic, d_value=0.0)
        if with_drain:
            prob.add_interior_drain(lambda x: (x[1] > z_if - 1e-3) & (x[1] < z_if + 0.064),
                                    conductance_density=300.0, drain_head=z_if)
        w0 = prob.total_water()
        prob.advance(t_end=0.15, dt=5e-4, dt_max=5e-3)
        bal = (prob.total_water() - w0) - (-prob.cum_drainage + prob.clip_mass_adjust)
        assert abs(bal) <= max(1e-10, 1e-6 * max(prob.cum_drainage, 1e-12))
        zc = prob.Vpsi.tabulate_dof_coordinates()[:, 1]
        probe = np.argmin((prob.Vpsi.tabulate_dof_coordinates()[:, 0] - 0.5) ** 2
                          + (zc - (z_if + 0.0625)) ** 2)
        return float(prob.psi.x.array[probe]), prob.cum_drainage

    psi_no, cum_no = run(False)
    psi_dr, cum_dr = run(True)
    assert cum_no == 0.0 and cum_dr > 1e-4               # only the drain removes water
    assert psi_no > -0.05                                 # perch persists undrained
    assert psi_dr < psi_no - 0.05                         # the drain visibly de-saturates the band


# --------------------------------------------------------------------- surface inlet: physics
def test_surface_inlet_captures_ponded_water():
    """Infiltration-excess storm (rain >> Ks) on a flat closed top: ponding develops, the grate
    inlet eats it, the balance Delta_total = cum_rain - cum_drainage closes structurally."""
    prob = CoupledProblem(_rect(8, 8, Lx=2.0), LOAM)
    prob.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.0)
    prob.add_surface_inlet(lambda x: (x[0] > 0.74) & (x[0] < 1.26), intake_coeff=500.0)
    rain = prob.add_rain(0.6)                              # 2.4x Ks -> infiltration-excess ponds
    w0 = prob.total_water()
    prob.advance(t_end=0.05, dt=1e-3, dt_max=5e-3)
    cum_inlet = prob.cum_sinks["surface_inlet"][0]
    assert cum_inlet > 1e-5                                # the inlet captured ponded water
    assert prob.sink_rates()["surface_inlet"][0] >= 0.0    # intake never reverses
    cum_rain = 0.6 * 2.0 * 0.05
    bal = (prob.total_water() - w0) - (cum_rain - prob.cum_drainage + prob.clip_mass_adjust)
    assert abs(bal) / cum_rain < 1e-8
    assert float(np.min(prob.d.x.array)) >= 0.0


def test_surface_inlet_dry_floor_only():
    """No rain, dry soil: the inlet sees only the bounded NCP-smoothing floor of d (documented
    residue), not physical capture."""
    prob = CoupledProblem(_rect(4, 6), LOAM)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.0)
    prob.add_surface_inlet(lambda x: (x[0] > 0.24) & (x[0] < 0.76), intake_coeff=500.0)
    assert prob.sink_rates()["surface_inlet"][0] == 0.0    # d = 0 exactly at the IC
    prob.advance(t_end=0.02, dt=1e-3, dt_max=2e-3)
    assert 0.0 <= prob.cum_sinks["surface_inlet"][0] < 1e-6


# --------------------------------------------------------------------- accounting + coexistence
def test_sink_accounting_split_sums_to_total():
    """GHB + interior drain + surface inlet + rain + outlet in ONE problem: per-sink rates sum to
    drainage_rate(), per-sink cums sum to cum_drainage (machine), and the full balance closes."""
    prob = CoupledProblem(_rect(8, 8, Lx=2.0), LOAM)
    prob.set_initial_condition(lambda x: 0.4 - x[1], d_value=0.0)
    prob.set_topography(lambda x: 0.02 * (2.0 - x[0]))
    prob.add_outflow_bc(lambda x: np.isclose(x[0], 2.0), slope=0.02)
    prob.add_drainage_bc(lambda x: np.isclose(x[1], 0.0) & (x[0] < 0.3),
                         conductance=0.5, external_head=0.0)
    prob.add_interior_drain(lambda x: (x[0] > 0.99) & (x[0] < 1.51) & (x[1] < 0.13),
                            conductance_density=30.0, drain_head=0.0)
    prob.add_surface_inlet(lambda x: (x[0] > 0.99) & (x[0] < 1.51), intake_coeff=500.0)
    rain = prob.add_rain(0.5)
    w0 = prob.total_water()
    cum_rain, t, dt = 0.0, 0.0, 2e-3
    for _ in range(15):
        conv, _it = prob.step(dt)
        assert conv
        cum_rain += 0.5 * 2.0 * dt
        t += dt
    rates = prob.sink_rates()
    flat = [r for kind in ("ghb", "interior_drain", "surface_inlet") for r in rates[kind]]
    assert len(flat) == 3
    assert sum(flat) == pytest.approx(prob.drainage_rate(), rel=1e-12, abs=1e-14)
    cum_flat = [c for kind in ("ghb", "interior_drain", "surface_inlet")
                for c in prob.cum_sinks[kind]]
    assert sum(cum_flat) == pytest.approx(prob.cum_drainage, rel=1e-12, abs=1e-14)
    bal = (prob.total_water() - w0) - (cum_rain - prob.cum_outflow - prob.cum_drainage
                                       + prob.clip_mass_adjust)
    assert abs(bal) / cum_rain < 1e-8
    assert np.all(np.isfinite(prob.psi.x.array))


def test_api_matches_script_injection():
    """The new APIs reproduce the SIGNED-OFF dual-run injection pattern (UFL coordinate-conditional
    indicators + manual _drainage_forms append, commit 63145e2) on aligned bands: same psi/d fields
    and the same cum_drainage after an identical fixed-step march."""
    def setup(api: bool):
        msh = _rect(8, 8)
        prob = CoupledProblem(msh, LOAM)
        prob.set_initial_condition(lambda x: 0.5 - x[1], d_value=0.0)
        rain = prob.add_rain(0.3)
        if api:
            prob.add_interior_drain(lambda x: (x[1] > 0.24) & (x[1] < 0.38),
                                    conductance_density=300.0, drain_head=0.25, eps_act=1e-3)
            prob.add_surface_inlet(lambda x: (x[0] > 0.24) & (x[0] < 0.76), intake_coeff=500.0)
        else:
            # the dual-run script pattern, verbatim mechanics. Bounds sit EXACTLY on cell edges
            # (0.25/0.375, 0.25/0.75) so the coordinate conditional selects the same whole cells
            # as the API's DG-0 indicator (quadrature points are strictly interior).
            xc = ufl.SpatialCoordinate(msh)
            chi_band = ufl.conditional(
                ufl.And(ufl.gt(xc[1], 0.25), ufl.lt(xc[1], 0.375)), 1.0, 0.0)
            chi_top = ufl.conditional(
                ufl.And(ufl.gt(xc[0], 0.25), ufl.lt(xc[0], 0.75)), 1.0, 0.0)
            kr = LOAM.K_ufl(prob.psi) / LOAM.Ks
            u = prob.psi + xc[1] - 0.25
            pos = 0.5 * (u + ufl.sqrt(u * u + 1e-3 * 1e-3))
            q_vol = 300.0 * kr * pos * chi_band
            dxq = ufl.dx(metadata={"quadrature_degree": prob._quad_degree})
            prob.F_psi = prob.F_psi + q_vol * prob._vpsi * dxq
            prob._drainage_forms.append(fem.form(q_vol * dxq))
            q_in = 500.0 * prob.d * chi_top
            prob.F_d = prob.F_d + q_in * prob._vd * prob._ds_top
            prob._drainage_forms.append(fem.form(q_in * prob._ds_top))
            prob._problem = None
        return prob

    pa, pb = setup(True), setup(False)
    for _ in range(12):
        ca, _ = pa.step(2e-3)
        cb, _ = pb.step(2e-3)
        assert ca and cb
    assert np.allclose(pa.psi.x.array, pb.psi.x.array, atol=1e-8)
    assert np.allclose(pa.d.x.array, pb.d.x.array, atol=1e-10)
    assert pa.cum_drainage == pytest.approx(pb.cum_drainage, rel=1e-8)
    assert pa.cum_drainage > 1e-5


def test_3d_smoke_both_sinks():
    """3-D box with an interior base band + a top-footprint inlet: steps converge, fields finite,
    balance closes (the APIs are dimension-agnostic like the rest of the engine)."""
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [3, 3, 4],
                           dmesh.CellType.tetrahedron)
    prob = CoupledProblem(msh, LOAM)
    prob.set_initial_condition(lambda x: 0.5 - x[2], d_value=0.0)
    prob.add_interior_drain(lambda x: x[2] < 0.26, conductance_density=4.0, drain_head=0.0)
    prob.add_surface_inlet(lambda x: x[0] < 0.4, intake_coeff=200.0)
    rain = prob.add_rain(0.05)
    w0 = prob.total_water()
    cum_rain = 0.0
    for _ in range(3):
        conv, _it = prob.step(2e-3)
        assert conv
        cum_rain += 0.05 * 1.0 * 2e-3
    assert np.all(np.isfinite(prob.psi.x.array))
    assert prob.cum_drainage > 0.0
    bal = (prob.total_water() - w0) - (cum_rain - prob.cum_outflow - prob.cum_drainage
                                       + prob.clip_mass_adjust)
    assert abs(bal) / max(cum_rain, prob.cum_drainage) < 1e-6


# --------------------------------------------------------------------- guards
def test_interior_drain_guards():
    prob = CoupledProblem(_rect(4, 4), LOAM)
    with pytest.raises(ValueError):   # no cell matched (locators are ALL-vertex predicates)
        prob.add_interior_drain(lambda x: x[1] > 99.0, conductance_density=1.0, drain_head=0.0)
    with pytest.raises(ValueError):   # negative conductance
        prob.add_interior_drain(lambda x: x[1] < 0.3, conductance_density=-1.0, drain_head=0.0)
    with pytest.raises(ValueError):   # non-positive smoothing width
        prob.add_interior_drain(lambda x: x[1] < 0.3, conductance_density=1.0, drain_head=0.0,
                                eps_act=0.0)
    prob.add_interior_drain(lambda x: x[1] < 0.3, conductance_density=1.0, drain_head=0.0)
    with pytest.raises(ValueError):   # overlapping interior drains = ambiguous double sink
        prob.add_interior_drain(lambda x: x[1] < 0.3, conductance_density=2.0, drain_head=0.1)


def test_surface_inlet_guards():
    prob = CoupledProblem(_rect(4, 4), LOAM)
    with pytest.raises(ValueError):   # footprint off the top surface
        prob.add_surface_inlet(lambda x: np.isclose(x[1], 0.0), intake_coeff=10.0)
    with pytest.raises(ValueError):   # negative intake
        prob.add_surface_inlet(lambda x: x[0] < 0.6, intake_coeff=-1.0)
    prob.add_surface_inlet(lambda x: x[0] < 0.6, intake_coeff=10.0)
    with pytest.raises(ValueError):   # overlapping inlets
        prob.add_surface_inlet(lambda x: x[0] < 0.3, intake_coeff=5.0)
