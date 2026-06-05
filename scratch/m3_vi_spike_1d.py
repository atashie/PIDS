"""Spike: does SNES-VI (d>=0 lower bound) fix the 1-D coupled supply-limit over-draw?

Finding (from m3_debug_1d): the smooth Robin land-surface exchange drives d ~ psi_top, and an
unsaturated draining top sits slightly negative, so d goes negative (dry soil over-draws an empty
surface store). The physical answer for rain < Ks on unsaturated soil is d=0, ALL rain infiltrates
(supply-limited). Enforce d>=0 as a complementarity via PETSc SNES-VI (vinewtonrsls), bounding ONLY
the surface DOFs. (VI was retired for the STIFF overland diffusion-wave (M2); 1-D has none, and
design D.5 reserves VI for exactly overland d>=0.)

Compares plain newtonls (negative d) vs vinewtonrsls (should give d=0, all rain infiltrates,
conservation preserved).
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh as dmesh
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def run(use_vi):
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 1.0])
    opts = dict(CoupledProblem._DEFAULT_PETSC_OPTIONS)
    if use_vi:
        opts["snes_type"] = "vinewtonrsls"
    prob = CoupledProblem(msh, SOIL, petsc_options=opts)
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    prob.add_rain(0.1)
    w0 = prob.total_water()
    prob._ensure_problem()

    if use_vi:
        # lower bound: -inf on psi block, 0 on d block; upper: +inf everywhere.
        # block "mpi" layout is [psi_dofs ; d_dofs] concatenated.
        n_psi = prob.Vpsi.dofmap.index_map.size_local * prob.Vpsi.dofmap.index_map_bs
        lb = prob._problem.solver.getSolution().duplicate()
        ub = lb.duplicate()
        lo = lb.getArray()
        lo[:n_psi] = PETSc.NINFINITY
        lo[n_psi:] = 0.0          # d >= 0
        lb.setArray(lo); lb.assemble()
        ub.set(PETSc.PINFINITY); ub.assemble()
        prob._problem.solver.setVariableBounds(lb, ub)

    prob.advance(t_end=0.5, dt=1e-3, dt_max=0.05)
    cum_rain = 0.1 * 0.5
    dW = prob.total_water() - w0
    return dict(d=prob.surface_depth(), surf=prob.surface_water(),
                soil_gain=prob.soil_water() - (w0 - 0.0),  # approx; w0 ~ all soil
                massbal_err=abs(dW - cum_rain) / cum_rain, dW=dW, cum_rain=cum_rain)


if __name__ == "__main__":
    print("=== plain newtonls (expect d<0, soil over-draws) ===")
    a = run(False)
    print(f"  d_surf={a['d']:.5f}  surface_water={a['surf']:.5f}  "
          f"dTotal={a['dW']:.5f} (rain={a['cum_rain']:.5f})  massbal_err={a['massbal_err']:.2e}")
    print("=== SNES-VI vinewtonrsls (d>=0; expect d~0, all rain infiltrates) ===")
    try:
        b = run(True)
        print(f"  d_surf={b['d']:.5f}  surface_water={b['surf']:.5f}  "
              f"dTotal={b['dW']:.5f} (rain={b['cum_rain']:.5f})  massbal_err={b['massbal_err']:.2e}")
        ok = (b['surf'] >= -1e-9) and (b['massbal_err'] < 1e-6)
        print(f"  VI VERDICT: {'PASS (d>=0 AND conserves)' if ok else 'FAIL'}")
    except Exception as e:
        import traceback
        print(f"  VI raised: {type(e).__name__}: {e}")
        print(traceback.format_exc().strip().splitlines()[-1][:120])
