"""Tier-1 sanity: the SORPTIVE (Kirchhoff matric-flux-potential) land-surface exchange leg.

The dry cell-value conductance ``q_pot = K(psi_top)/ell_c*(d-psi_top)`` under-infiltrates dry soil ~50x
at coarse resolution because it ignores the steep K rise (->Ks) across the wetting film (design
analysis docs/plans/2026-06-06-sorptivity-exchange-design-analysis.md; spec ...-spec.md). The fix
replaces the dry value with the FILM-INTEGRAL (Kirchhoff) mean: ``q_pot = kirchhoff(psi_top, d)/ell_c``.

These pin the §D fix: (A) quasi-steady CAPACITY recovery (a coarse column infiltrates a substantial
fraction of a heavy storm), and (B) TRANSIENT tracking of a resolved Richards reference through time
(not just the end state) -- the Richards solve below the surface carries the wetting-front dynamics, so
the surface leg + Richards reproduces the transient uptake. Benchmark: scratch/m3_sorptivity_benchmark.py.
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import ufl

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.richards import RichardsProblem

LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
H, RAIN, T_END = 1.0, 0.6, 0.04


def _march(prob, restore, t_end=T_END):
    t, dt = 0.0, 1e-6
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        try:
            conv, it = prob.step(h)
        except Exception:
            conv, it = False, 0
            restore()
            prob._problem = None
        if conv:
            t += h
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 1e-3)
        else:
            dt *= 0.5
            assert dt > 1e-11, "dt collapse"


def test_sorptivity_recovers_infiltration_into_dry_soil():
    """RED gate: a coarse 1-D coupled column on DRY loam under a heavy storm must infiltrate a
    substantial fraction of the rain. The dry K(psi_top) conductance throttles to ~1.6%; the
    Kirchhoff film-integral leg recovers to ~49%. Threshold 0.30 separates them cleanly.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 8, [0.0, H])   # coarse -> ell_c = top-cell half-height
    prob = CoupledProblem(msh, LOAM)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.0)
    prob.add_rain(RAIN)
    w0 = prob.soil_water()

    def restore():
        prob.psi.x.array[:] = prob.psi_n.x.array; prob.psi.x.scatter_forward()
        prob.d.x.array[:] = prob.d_n.x.array; prob.d.x.scatter_forward()
        prob.lam.x.array[:] = 0.0; prob.lam.x.scatter_forward()
    _march(prob, restore)

    cum_rain = RAIN * T_END   # 1-D surface is a point (unit area)
    infiltrated = prob.soil_water() - w0
    assert infiltrated / cum_rain >= 0.30, \
        f"infiltration throttled: {infiltrated / cum_rain:.3f} of rain (dry-K conductance ~0.016)"


def test_sorptivity_tracks_resolved_richards_transient():
    """TRANSIENT: the coupled column's cumulative infiltration I(t) tracks a RESOLVED Richards
    reference (fine mesh + add_ponding_bc, which captures the sorptive uptake) through time -- i.e. the
    surface leg + the Richards solve reproduce the wetting dynamics, not just the final capacity. The
    coarse coupling must reach >= 0.5 of the fine resolved I(t) at sampled times (the residual is bulk
    coarse-mesh error). The dry-K conductance fails this (~0.03).
    """
    # resolved reference: fine Richards + ponding BC
    rmsh = dmesh.create_interval(MPI.COMM_WORLD, 80, [0.0, H])
    ref = RichardsProblem(rmsh, LOAM)
    ref.set_initial_condition(lambda x: -1.0 + 0.0 * x[0])
    ref.add_ponding_bc(lambda x: np.isclose(x[0], H), RAIN)
    rw0 = rmsh.comm.allreduce(fem.assemble_scalar(fem.form(LOAM.theta_ufl(ref.psi) * ufl.dx)), op=MPI.SUM)

    def rref():
        ref.psi.x.array[:] = ref.psi_n.x.array; ref.psi.x.scatter_forward()

    # coarse coupling
    cmsh = dmesh.create_interval(MPI.COMM_WORLD, 8, [0.0, H])
    cpl = CoupledProblem(cmsh, LOAM)
    cpl.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.0)
    cpl.add_rain(RAIN)
    cw0 = cpl.soil_water()

    def cref():
        cpl.psi.x.array[:] = cpl.psi_n.x.array; cpl.psi.x.scatter_forward()
        cpl.d.x.array[:] = cpl.d_n.x.array; cpl.d.x.scatter_forward()
        cpl.lam.x.array[:] = 0.0; cpl.lam.x.scatter_forward()

    prev = 0.0
    for t_chk in (0.01, 0.02, 0.04):
        _march(ref, rref, t_chk - prev)   # _march advances by a DURATION from the current state
        _march(cpl, cref, t_chk - prev)
        prev = t_chk
        I_ref = rmsh.comm.allreduce(
            fem.assemble_scalar(fem.form(LOAM.theta_ufl(ref.psi) * ufl.dx)), op=MPI.SUM) - rw0
        I_cpl = cpl.soil_water() - cw0
        assert I_cpl >= 0.5 * I_ref, f"t={t_chk}: coupling I={I_cpl:.4e} < 0.5*resolved {I_ref:.4e}"
