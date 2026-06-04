"""Tier-1 sanity (subsurface MMS): method of manufactured solutions.

Verifies the mixed-form Richards discretization converges at the theoretical order
(P1 Lagrange -> 2nd order in L2 for a smooth solution). A manufactured steady
solution psi*(x) is substituted into the steady Richards operator to give a forcing
f* = -div(K(psi*)(grad psi* + e_g)); the discrete solution must then converge to
psi* as O(h^2) under mesh refinement.
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import RichardsProblem

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.2496)  # m/day


def _manufactured_1d(msh, soil):
    """psi*(x) = -1.0 - 0.5 sin(pi x): smooth, stays unsaturated; psi* = -1 on the ends."""
    x = ufl.SpatialCoordinate(msh)
    psi_star = -1.0 - 0.5 * ufl.sin(ufl.pi * x[0])
    e_g = ufl.as_vector([1.0])  # gravity along the single (vertical) axis
    flux_star = soil.K_ufl(psi_star) * (ufl.grad(psi_star) + e_g)
    f_star = -ufl.div(flux_star)
    return psi_star, f_star


def test_mms_spatial_convergence_order_1d():
    soil = VanGenuchten(**LOAM)
    hs, errors = [], []
    for ncells in (10, 20, 40, 80):
        msh = dmesh.create_unit_interval(MPI.COMM_WORLD, ncells)
        psi_star, f_star = _manufactured_1d(msh, soil)
        # Consistent mass (not lumped) so the spatial L2 order is the clean ~2
        # (mass-lumping degrades the order; it is the production / front-stability path).
        prob = RichardsProblem(msh, soil, source=f_star, lumped=False)
        prob.set_initial_condition(lambda X: -1.0 - 0.5 * np.sin(np.pi * X[0]))
        on_ends = lambda X: np.isclose(X[0], 0.0) | np.isclose(X[0], 1.0)
        prob.add_dirichlet(on_ends, -1.0)
        converged, _ = prob.step(dt=1.0e8)  # huge dt -> steady problem
        assert converged
        hs.append(1.0 / ncells)
        errors.append(prob.l2_error(psi_star))

    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(errors) - 1)
    ]
    # P1 Lagrange => asymptotic L2 order ~2 (allow mild pre-asymptotic slack).
    assert orders[-1] > 1.85, f"finest order={orders[-1]:.3f}; orders={orders}; errors={errors}"
    assert min(orders) > 1.6, f"orders={orders}; errors={errors}"


def test_mms_temporal_convergence_order_1d():
    """Backward-Euler => ~1st-order convergence in dt (time MMS, fixed fine mesh).

    psi*(x,t) = -1 - 0.5 sin(pi x) exp(-t): smooth, unsaturated, psi* = -1 on the ends
    for all t. The transient source f* = dtheta(psi*)/dt - div(K(psi*)(grad psi* + e_g))
    makes psi* exact; refining dt at fixed (fine) h gives temporal order ~1.
    """
    soil = VanGenuchten(**LOAM)
    T = 0.5
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 120)
    x = ufl.SpatialCoordinate(msh)
    time = fem.Constant(msh, 0.0)
    psi_star = -1.0 - 0.5 * ufl.sin(ufl.pi * x[0]) * ufl.exp(-time)
    dpsi_dt = 0.5 * ufl.sin(ufl.pi * x[0]) * ufl.exp(-time)  # analytic d(psi*)/dt
    e_g = ufl.as_vector([1.0])
    f_star = soil.capacity_ufl(psi_star) * dpsi_dt - ufl.div(
        soil.K_ufl(psi_star) * (ufl.grad(psi_star) + e_g)
    )

    def run(dt):
        time.value = 0.0
        prob = RichardsProblem(msh, soil, source=f_star, lumped=False)
        prob.set_initial_condition(lambda X: -1.0 - 0.5 * np.sin(np.pi * X[0]))
        on_ends = lambda X: np.isclose(X[0], 0.0) | np.isclose(X[0], 1.0)
        prob.add_dirichlet(on_ends, -1.0)
        t = 0.0
        while t < T - 1e-12:
            t = min(t + dt, T)
            time.value = t  # backward-Euler: source at the new time level
            converged, _ = prob.step(dt)
            assert converged
        time.value = T
        return prob.l2_error(psi_star)

    dts = [T / 5, T / 10, T / 20, T / 40]
    errs = [run(dt) for dt in dts]
    orders = [
        np.log(errs[i] / errs[i + 1]) / np.log(dts[i] / dts[i + 1])
        for i in range(len(errs) - 1)
    ]
    # Backward-Euler => order ~1.
    assert orders[-1] > 0.9, f"finest temporal order={orders[-1]:.3f}; orders={orders}; errs={errs}"
    assert min(orders) > 0.75, f"orders={orders}; errs={errs}"
