"""SORPTIVITY gap, 4-WAY matrix (Codex review 2026-06-06: separate CLOSURE error from MESH error).

Same dry 1-D column, same heavy rain, infiltration capacity-limited. 2x2:
  solver  in {ponding  = RichardsProblem + add_ponding_bc (resolved head-continuity infiltration),
              conduct  = CoupledProblem  (the K(psi_top)/ell_c conductance NCP -- the MODEL closure)}
  mesh    in {coarse = 8 cells, fine = 80 cells}

Key comparisons:
  * conduct-coarse vs ponding-COARSE  -> CLOSURE error at the SAME resolution (the clean isolation).
  * conduct-fine   vs ponding-fine    -> does the conductance closure RECOVER when refined (ell_c->0)?
  * ponding-coarse vs ponding-fine    -> the bulk MESH error of the reference itself.

SAME storage metric for every cell: I(t) = int theta(psi) dx (consistent quadrature) - I(0).

  PYTHONPATH=. python scratch/m3_sorptivity_benchmark.py   (run from forward-model/)
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import RichardsProblem
from pids_forward.physics.coupling import CoupledProblem

SOILS = {
    "loam": VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25),
    "sand": VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.128),
}
PSI0 = {"loam": -1.0, "sand": -0.5}
H, RAIN, T_END = 1.0, 0.6, 0.04
MESHES = {"coarse": 8, "fine": 80}


def theta_int(mesh, soil, psi):
    """int theta(psi) dx with CONSISTENT quadrature -- the one metric used for every cell."""
    return mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(soil.theta_ufl(psi) * ufl.dx)), op=MPI.SUM)


def march(prob, mesh, soil, restore, n_out=24):
    times = np.linspace(0.0, T_END, n_out)
    I = np.zeros(n_out)
    w0 = theta_int(mesh, soil, prob.psi)
    dt = 1e-6
    for k in range(1, n_out):
        t, tt = times[k - 1], times[k]
        while t < tt - 1e-12:
            h = min(dt, tt - t)
            try:
                conv, it = prob.step(h)
            except Exception:
                conv, it = False, 0
                restore(); prob._problem = None
            if conv:
                t += h; dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 1e-3)
            else:
                dt *= 0.5
                if dt < 1e-11:
                    return times, I, False
        I[k] = theta_int(mesh, soil, prob.psi) - w0
    return times, I, True


def run_ponding(soil, psi0, nz):
    msh = dmesh.create_interval(MPI.COMM_WORLD, nz, [0.0, H])
    prob = RichardsProblem(msh, soil)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0])
    prob.add_ponding_bc(lambda x: np.isclose(x[0], H), RAIN)

    def restore():
        prob.psi.x.array[:] = prob.psi_n.x.array; prob.psi.x.scatter_forward()
    return march(prob, msh, soil, restore)


def run_conduct(soil, psi0, nz):
    msh = dmesh.create_interval(MPI.COMM_WORLD, nz, [0.0, H])
    prob = CoupledProblem(msh, soil)
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    prob.add_rain(RAIN)

    def restore():
        prob.psi.x.array[:] = prob.psi_n.x.array; prob.psi.x.scatter_forward()
        prob.d.x.array[:] = prob.d_n.x.array; prob.d.x.scatter_forward()
        prob.lam.x.array[:] = 0.0; prob.lam.x.scatter_forward()
    return march(prob, msh, soil, restore)


for name, soil in SOILS.items():
    cum_rain = RAIN * T_END
    print(f"\n========== {name}: Ks={soil.Ks} m/day, psi0={PSI0[name]} m, rain={RAIN} (cum {cum_rain:.4f} m) ==========")
    res = {}
    for mname, nz in MESHES.items():
        for sname, fn in (("ponding", run_ponding), ("conduct", run_conduct)):
            t, I, ok = fn(soil, PSI0[name], nz)
            res[(sname, mname)] = (I[-1], ok)
            tag = "" if ok else "  (dt COLLAPSE - partial)"
            print(f"  {sname:8s} {mname:6s} (nz={nz:3d}): infiltrated {I[-1]:.4e} m "
                  f"= {100*I[-1]/cum_rain:5.1f}% of rain{tag}")
    pc = res[("ponding", "coarse")][0]; pf = res[("ponding", "fine")][0]
    cc = res[("conduct", "coarse")][0]; cf = res[("conduct", "fine")][0]
    print(f"  --- CLOSURE error @ coarse (conduct/ponding, same nz=8): {cc/max(pc,1e-12):.2f}")
    print(f"  --- CLOSURE error @ fine   (conduct/ponding, same nz=80): {cf/max(pf,1e-12):.2f}  "
          f"(->1 means the closure RECOVERS when refined => resolution problem; <<1 => closure problem)")
    print(f"  --- MESH error of reference (ponding coarse/fine): {pc/max(pf,1e-12):.2f}")
