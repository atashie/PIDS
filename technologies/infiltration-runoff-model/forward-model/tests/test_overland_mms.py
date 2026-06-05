"""Tier-1 sanity (overland MMS): method of manufactured solutions.

Verifies the diffusion-wave discretization converges at the theoretical order under
refinement. A manufactured d*(x[,t]) > 0 is substituted into the diffusion-wave operator
to give a forcing f*. The regularized Manning conveyance is C-infinity for eps_S > 0, so a
smooth d* yields a smooth problem with clean convergence order. Consistent mass
(lumped=False), matching the subsurface MMS convention (mass-lumping degrades spatial order;
it is the production/front-stability path, gated on conservation + plausibility instead).

NOTE (spatial): the manufactured d* uses a strictly-positive gradient (exp), NOT sin --
a solution with grad d* = 0 at an interior point makes the slope floor
|grad H_s|^{1/2} -> (|grad H_s|^2 + eps_S^2)^{1/4} create a sharp ~eps_S-wide dip in the
conveyance there that the mesh cannot resolve, degrading the order. With grad d* bounded
away from 0 the conveyance is smooth and well-resolved -> clean P1 order ~2.

NOTE (temporal): the overland equation is strongly diffusion-dominated (Manning conveyance
~1e6 m^2/day vs a tiny storage term), so the backward-Euler temporal error is orders of
magnitude below the spatial-error floor -- an MMS-vs-exact comparison cannot isolate it.
We therefore measure temporal order by SELF-CONVERGENCE (the shared spatial error cancels in
||d_h(dt) - d_h(dt/2)||), which recovers BE's first order.
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from pids_forward.physics.overland import OverlandProblem, SECONDS_PER_DAY

N_MAN = 0.03
EPS_S = 1e-3


def _Ks_ufl(d_expr, n_man, eps_S):
    """Manning conveyance K_s(d) on a flat bed (H_s = d), matching overland.py exactly."""
    g = ufl.grad(d_expr)
    slope_sqrt = (ufl.dot(g, g) + eps_S**2) ** 0.25
    return SECONDS_PER_DAY * d_expr ** (5.0 / 3.0) / (n_man * slope_sqrt)


def test_mms_spatial_convergence_order_1d():
    """Steady MMS: d*(x) = 1 + 0.5 exp(x) (>0, smooth, grad never 0 -> clean order)."""
    e = float(np.e)
    hs, errors = [], []
    for ncells in (10, 20, 40, 80):
        msh = dmesh.create_unit_interval(MPI.COMM_WORLD, ncells)
        x = ufl.SpatialCoordinate(msh)
        d_star = 1.0 + 0.5 * ufl.exp(x[0])  # flat bed: H_s = d_star; grad = 0.5 exp(x) > 0
        f_star = -ufl.div(_Ks_ufl(d_star, N_MAN, EPS_S) * ufl.grad(d_star))
        prob = OverlandProblem(msh, n_man=N_MAN, eps_S=EPS_S, source=f_star, lumped=False)
        prob.set_initial_condition(lambda X: 1.0 + 0.5 * np.exp(X[0]))
        prob.add_dirichlet(lambda X: np.isclose(X[0], 0.0), 1.0 + 0.5)         # d*(0) = 1.5
        prob.add_dirichlet(lambda X: np.isclose(X[0], 1.0), 1.0 + 0.5 * e)     # d*(1)
        converged, _ = prob.step(dt=1.0e8)  # huge dt -> steady problem
        assert converged
        hs.append(1.0 / ncells)
        errors.append(prob.l2_error(d_star))

    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(errors) - 1)
    ]
    assert orders[-1] > 1.85, f"finest order={orders[-1]:.3f}; orders={orders}; errors={errors}"
    assert min(orders) > 1.7, f"orders={orders}; errors={errors}"


def test_mms_temporal_convergence_order_1d():
    """Backward Euler => ~1st-order temporal convergence, by SELF-convergence (fixed mesh).

    d*(x,t) = 1 + 0.5 sin(pi x) (1 + 0.3 sin(2 pi t)): smooth, positive, d* = 1 on the ends
    for all t (constant-in-time Dirichlet). With the manufactured transient source, the
    discrete d_h(dt) converges to the semi-discrete solution at O(dt); consecutive halvings
    give ||d_h(dt)-d_h(dt/2)|| ratios -> 2 (order 1). Self-convergence cancels the
    (dominant, diffusion-driven) spatial error that an MMS-vs-exact test cannot.
    """
    T = 0.5
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 120)
    x = ufl.SpatialCoordinate(msh)
    time = fem.Constant(msh, 0.0)
    gt = 1.0 + 0.3 * ufl.sin(2.0 * ufl.pi * time)
    dgt = 0.3 * 2.0 * ufl.pi * ufl.cos(2.0 * ufl.pi * time)
    d_star = 1.0 + 0.5 * ufl.sin(ufl.pi * x[0]) * gt
    dd_dt = 0.5 * ufl.sin(ufl.pi * x[0]) * dgt
    f_star = dd_dt - ufl.div(_Ks_ufl(d_star, N_MAN, EPS_S) * ufl.grad(d_star))

    def run(dt):
        time.value = 0.0
        prob = OverlandProblem(msh, n_man=N_MAN, eps_S=EPS_S, source=f_star, lumped=False)
        prob.set_initial_condition(lambda X: 1.0 + 0.5 * np.sin(np.pi * X[0]))
        on_ends = lambda X: np.isclose(X[0], 0.0) | np.isclose(X[0], 1.0)
        prob.add_dirichlet(on_ends, 1.0)
        t = 0.0
        while t < T - 1e-12:
            t = min(t + dt, T)
            time.value = t  # backward Euler: source at the new time level
            converged, _ = prob.step(dt)
            assert converged
        return prob.d.x.array.copy()

    sols = [run(T / n) for n in (5, 10, 20, 40)]
    diffs = [float(np.sqrt(np.sum((sols[i] - sols[i + 1]) ** 2))) for i in range(len(sols) - 1)]
    # consecutive self-difference ratios -> 2^order; for BE (order 1) the ratio -> 2.
    ratios = [diffs[i] / diffs[i + 1] for i in range(len(diffs) - 1)]
    orders = [np.log2(r) for r in ratios]
    assert orders[-1] > 0.85, f"finest temporal order={orders[-1]:.3f}; orders={orders}; diffs={diffs}"
    assert min(orders) > 0.7, f"orders={orders}; diffs={diffs}"
