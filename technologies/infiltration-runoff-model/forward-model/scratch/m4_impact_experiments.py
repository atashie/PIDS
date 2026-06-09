"""Module 4 §E -- PIDS IMPACT experiments (Phase-2 primitive coupled into a 3-D host Richards block).

Quantifies the impact of ONE embedded vertical feature on the surrounding soil, WITH vs WITHOUT the
feature, in two PIDS modes:
  * DRAIN   -- a waterlogged (near-saturated) block; the feature (low-head outlet at the base) DRAINS it.
  * DISPERSE-- a dry block; the feature (charged inlet at the top) DELIVERS water (subsurface irrigation).
plus a sigma (wall-exchange) SENSITIVITY sweep on the drain case.

SCOPE / CAVEAT: the exchange uses the Phase-2 SIMPLE CONSTANT sigma (the calibrated Kirchhoff sorptive
closure is Phase 3) and a bare host Richards block (the surface inlet + overland are Phase 4). So the
DIRECTION and TREND are real physics; the absolute magnitude is uncalibrated. This is a relative impact
assessment, clearly labelled as such in the viz.

Run from forward-model/:  PYTHONPATH=. OMP_NUM_THREADS=1 ... python scratch/m4_impact_experiments.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import richards_bulk_residual
from pids_forward.physics.feature import EmbeddedFeature

COMM = MPI.COMM_WORLD
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
LX, LZ, NX, NZ = 1.0, 1.5, 8, 12          # 1x1x1.5 m block; feature = vertical centerline (x=y=0.5)
_BT = {"snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_rtol": 1e-9, "snes_atol": 1e-11,
       "snes_max_it": 40, "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def run(direction, with_feature, *, sigma=0.5, K_feat=10.0, area=0.008, porosity=0.4,
        t_end=3.0, n_snap=25):
    """One coupled run. Returns dict(times, soil_water, flux, xz, theta_xz_final)."""
    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LX, LZ]], [NX, NX, NZ])
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi, psi_n = fem.Function(V), fem.Function(V)
    psi_i = -0.05 if direction == "drain" else -1.2      # waterlogged vs dry
    psi.x.array[:] = psi_i; psi_n.x.array[:] = psi_i
    psi.x.scatter_forward(); psi_n.x.scatter_forward()
    eg = fem.Constant(msh, PETSc.ScalarType([0.0, 0.0, 1.0]))
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-3))
    lumped = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
    w = ufl.TestFunction(V)
    F_psi = richards_bulk_residual(psi, psi_n, w, LOAM, dt_c, eg, dx_storage=lumped, quadrature_degree=8)

    feat = None
    Fs, funcs, bcs = [F_psi], [psi], []
    if with_feature:
        feat = EmbeddedFeature(msh, lambda x: np.isclose(x[0], LX / 2) & np.isclose(x[1], LX / 2),
                               tangent=(0.0, 0.0, 1.0), K_feat=K_feat, area=area, porosity=porosity,
                               sigma=sigma)
        F_psi = F_psi + feat.exchange_into_host(w, psi)
        v = ufl.TestFunction(feat.V)
        F_hf = (feat.storage_form(v, dt_c) + feat.conveyance_form(v)
                + feat.exchange_into_feature(v, psi) + feat.Hf * v * feat._dPoff)
        if direction == "drain":
            h_end, z_end = -2.0, 0.0          # low-head OUTLET at the base
        else:
            h_end, z_end = 0.3, LZ            # charged INLET at the top
        feat.Hf.x.array[:] = h_end; feat.Hf_n.x.array[:] = h_end
        feat.Hf.x.scatter_forward(); feat.Hf_n.x.scatter_forward()
        end_dofs = fem.locate_dofs_geometrical(
            feat.V, lambda x: np.isclose(x[0], LX / 2) & np.isclose(x[1], LX / 2) & np.isclose(x[2], z_end))
        bcs = [feat.pin_bc(), fem.dirichletbc(PETSc.ScalarType(h_end), end_dofs, feat.V)]
        Fs, funcs = [F_psi, F_hf], [psi, feat.Hf]

    if with_feature:
        prob = NonlinearProblem(Fs, funcs, bcs=bcs, petsc_options_prefix="imp_", petsc_options=_BT, kind="mpi")
    else:
        prob = NonlinearProblem(F_psi, psi, bcs=bcs, petsc_options_prefix="imp_", petsc_options=_BT)

    soil_water_form = fem.form(LOAM.theta_ufl(psi) * lumped)

    def soil_water():
        return COMM.allreduce(fem.assemble_scalar(soil_water_form), op=MPI.SUM)

    # cross-section extraction (y = LX/2 plane) for the final theta field
    xc = V.tabulate_dof_coordinates()
    sel = np.isclose(xc[:, 1], LX / 2)
    xz = xc[sel][:, [0, 2]]

    times, sw, flux = [], [], []
    snaps = np.linspace(0.0, t_end, n_snap)
    t, dt, k = 0.0, 1e-3, 0
    w0 = soil_water()
    times.append(0.0); sw.append(0.0); flux.append(0.0)
    k = 1
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if k < n_snap and t + h > snaps[k]:
            h = snaps[k] - t
        dt_c.value = h
        prob.solve()
        if prob.solver.getConvergedReason() > 0:
            psi_n.x.array[:] = psi.x.array; psi_n.x.scatter_forward()
            if feat is not None:
                feat.Hf_n.x.array[:] = feat.Hf.x.array; feat.Hf_n.x.scatter_forward()
            t += h
            it = prob.solver.getIterationNumber()
            if k < n_snap and t >= snaps[k] - 1e-12:
                times.append(t); sw.append(soil_water() - w0)
                flux.append(feat.host_exchange_flux(psi) if feat is not None else 0.0)
                k += 1
            dt = min(dt * (1.4 if it <= 4 else 0.7 if it >= 9 else 1.0), 0.1)
        else:
            dt *= 0.5
            assert dt > 1e-8, f"dt collapse at t={t:.4f} ({direction}, feature={with_feature})"
    while k < n_snap:
        times.append(t_end); sw.append(soil_water() - w0); flux.append(flux[-1]); k += 1

    theta_xz = LOAM.theta(psi.x.array[sel])
    return dict(times=np.array(times), soil_water=np.array(sw), flux=np.array(flux),
                xz=xz, theta_xz=theta_xz, w0=w0)


if __name__ == "__main__":
    out = {}
    for direction in ("drain", "disperse"):
        for wf in (False, True):
            tag = f"{direction}_{'with' if wf else 'without'}"
            print(f"running {tag} ...", flush=True)
            r = run(direction, wf)
            out[f"{tag}_t"] = r["times"]; out[f"{tag}_sw"] = r["soil_water"]; out[f"{tag}_flux"] = r["flux"]
            out[f"{tag}_xz"] = r["xz"]; out[f"{tag}_theta"] = r["theta_xz"]
            print(f"  {tag}: final Δsoil_water = {r['soil_water'][-1]*1000:.2f} L/m³-equiv "
                  f"(w0={r['w0']:.3f})", flush=True)
    # sigma sensitivity sweep on the drain case: span the exchange-limited -> transport-limited transition
    sweep = [0.002, 0.01, 0.05, 0.25, 1.0]
    sw_final = []
    for s in sweep:
        r = run("drain", True, sigma=s)
        sw_final.append(float(r["soil_water"][-1]))
        print(f"  drain sweep sigma={s}: Δsoil_water = {r['soil_water'][-1]*1000:.2f}", flush=True)
    out["sweep_sigma"] = np.array(sweep); out["sweep_dsw"] = np.array(sw_final)
    np.savez("scratch/m4_impact_experiments.npz", **out)
    print("WROTE scratch/m4_impact_experiments.npz", flush=True)
