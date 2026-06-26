"""SCRATCH BENCHMARK v3 -- the CLEAN mesh-convergence test (3rd Codex review, §20).

Fixes the §20 mis-framing: the v2 ell_c sweep was a FREE-PARAMETER sweep on a fixed coarse mesh, comparing
a coarse-mesh monolith to a fine-mesh Dirichlet (two discretizations that can cross by cancellation). The
clean test (Codex-specified):
  * REAL monolith `CoupledProblem` on an nz LADDER {8,16,24,40}, ell_c LOCKED to dz_top/2 (auto) -- so the
    mesh AND the q_pot film scale refine TOGETHER (the actual production discretization).
  * vs a RAIN-DRIVEN `add_ponding_bc` Richards column (the repo-native no-film reference,
    tests/test_coupling_1d.py:110-141) on the SAME mesh, SAME forcing -- the ell_c->0 limit (no film cap).
  * SAME forcing for both: rain = 10*Ks (ponding regime); consistent deg-8 INT theta; report I(t), the
    surface pond d(t), ponding time t_p, and solver health (converged reason / total-water closure).
The question: does the monolith I(t) collapse toward the add_ponding_bc curve as dz->0 (=> the coarse
q_pot throttle is a genuine surface-RESOLUTION artifact), or plateau short (=> a true closure gap)?

DRY IC: fixed dry head -3 m (finite/physical for all soils; soil-specific S_e is the honest reality --
clay holds water tenaciously). 1-D-style column (flat, no routing -> overland inert).

Run (WSL pids-fem) -- LIVE to a file (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_sorptivity_meshconv.py <soil>'
"""
from __future__ import annotations

import sys
import time

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.richards import RichardsProblem

COMM = MPI.COMM_WORLD
SOILS = {
    "loam": dict(theta_r=0.078, theta_s=0.43, alpha=3.6,  n=1.56, Ks=0.25),
    "sand": dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=1.5),
    "silt": dict(theta_r=0.067, theta_s=0.45, alpha=2.0,  n=1.41, Ks=0.10),
    "clay": dict(theta_r=0.068, theta_s=0.38, alpha=0.8,  n=1.09, Ks=0.048),
}
LZ = 1.0
PSI_DRY = -3.0
TIMES = [0.005, 0.02, 0.08]
NZ_LADDER = [8, 16, 24, 40]


def _col(nz):
    return dmesh.create_box(COMM, [np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, LZ])],
                            [2, 2, nz], cell_type=dmesh.CellType.tetrahedron)


def _theta(prob, soil):
    dxq = ufl.dx(metadata={"quadrature_degree": 8})
    return prob.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(soil.theta_ufl(prob.psi) * dxq)), op=MPI.SUM)


def _pond_vol_form(prob, want_d):
    """Conserved pond DEPTH = (INT max(field,0) ds_top)/area -- the physical ledger value (NOT a node-max,
    which catches Newton transients). field = the monolith's d, or max(psi,0) for add_ponding_bc."""
    msh = prob.mesh
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    tf = np.sort(dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], LZ))).astype(np.int32)
    ft = dmesh.meshtags(msh, fdim, tf, np.ones(tf.size, dtype=np.int32))
    dst = ufl.Measure("ds", domain=msh, subdomain_data=ft, metadata={"quadrature_degree": 8})(1)
    field = prob.d if want_d else ufl.max_value(prob.psi, 0.0)
    return fem.form(field * dst)


def _march(prob, soil, *, want_d=False, dt0=2e-5, dt_max=2e-3):
    """March to TIMES[-1]; return (I dict, ponding-time, max d, n bad-solves). want_d => prob has .d/.psi."""
    th0 = _theta(prob, soil)
    pond_form = _pond_vol_form(prob, want_d)
    targets = list(TIMES); out = {}
    t, dt, nstep, nbad = 0.0, dt0, 0, 0
    t_pond = None; d_max = 0.0
    while t < TIMES[-1] - 1e-12 and nstep < 40000:
        h = min(dt, TIMES[-1] - t)
        if targets and t + h > targets[0]:
            h = targets[0] - t
        conv, it = prob.step(h)
        if not conv:
            nbad += 1
            dt *= 0.5
            if dt < 1e-9:
                break
            continue
        t += h; nstep += 1
        # pond DEPTH = conserved ledger volume / area (NOT a node-max -> no Newton-transient spikes).
        d_now = prob.mesh.comm.allreduce(fem.assemble_scalar(pond_form), op=MPI.SUM) / 0.01
        d_max = max(d_max, d_now)
        if t_pond is None and d_now > 1e-5:
            t_pond = t
        if it <= 3:
            dt = min(dt * 1.4, dt_max)
        elif it >= 8:
            dt *= 0.7
        while targets and t >= targets[0] - 1e-12:
            out[targets.pop(0)] = (_theta(prob, soil) - th0) / 0.01
    for tt in targets:
        out[tt] = (_theta(prob, soil) - th0) / 0.01
    return out, t_pond, d_max, nbad


def run_monolith(soil_name, nz, rain):
    soil = VanGenuchten(**SOILS[soil_name])
    msh = _col(nz)
    prob = CoupledProblem(msh, soil, overland_scheme="galerkin")     # ell_c AUTO = dz_top/2 (locked)
    prob.set_initial_condition(lambda x: PSI_DRY + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: 0.0 * x[0])
    prob.add_rain(rain)
    t0 = time.perf_counter()
    I, tp, dmx, nbad = _march(prob, soil, want_d=True)
    return dict(I=I, t_pond=tp, d_max=dmx, nbad=nbad, ell_c=prob.ell_c, wall=time.perf_counter() - t0)


def run_ponding(soil_name, nz, rain):
    """RAIN-DRIVEN add_ponding_bc Richards column (no film cap = the ell_c->0 reference)."""
    soil = VanGenuchten(**SOILS[soil_name])
    msh = _col(nz)
    rp = RichardsProblem(msh, soil)
    rp.set_initial_condition(lambda x: PSI_DRY + 0.0 * x[0])
    rp.add_ponding_bc(lambda x: np.isclose(x[2], LZ), rain)
    t0 = time.perf_counter()
    I, tp, dmx, nbad = _march(rp, soil, want_d=False)
    return dict(I=I, t_pond=tp, d_max=dmx, nbad=nbad, wall=time.perf_counter() - t0)


def main():
    np.set_printoptions(precision=4, suppress=True)
    soil_name = sys.argv[1] if len(sys.argv) > 1 else "loam"
    Ks = SOILS[soil_name]["Ks"]
    soil = VanGenuchten(**SOILS[soil_name])
    th_d = float(soil.theta(np.array([PSI_DRY]))[0])
    se = (th_d - SOILS[soil_name]["theta_r"]) / (SOILS[soil_name]["theta_s"] - SOILS[soil_name]["theta_r"])
    rain = 10.0 * Ks
    print("#" * 100)
    print(f"CLEAN MESH-CONVERGENCE -- {soil_name} (Ks={Ks}); ell_c LOCKED to dz_top/2; rain={rain} (10xKs)")
    print(f"  DRY IC psi=-3m -> theta={th_d:.4f} (S_e={se:.3f});  I(t)[mm] consistent deg-8 INT theta")
    print(f"  MONOLITH (q_pot film, ell_c=dz/2) vs RAIN-DRIVEN add_ponding_bc (no film = ell_c->0 ref)")
    print("#" * 100, flush=True)
    print(f"\n  {'scheme':>22} {'nz':>4} {'ell_c[mm]':>9} | " +
          "".join(f"{t:>8.3f} " for t in TIMES) + "| t_pond  d_max[mm] nbad wall")
    pond_ref = {}
    for nz in NZ_LADDER:
        rp = run_ponding(soil_name, nz, rain)
        pond_ref[nz] = rp["I"]
        tp = f"{rp['t_pond']:.4f}" if rp["t_pond"] else "  none"
        print(f"  {'add_ponding_bc':>22} {nz:>4} {'--':>9} | " +
              "".join(f"{rp['I'][t]*1000:>8.2f} " for t in TIMES) +
              f"| {tp:>6} {rp['d_max']*1000:>8.2f} {rp['nbad']:>4} {rp['wall']:>4.0f}s", flush=True)
    for nz in NZ_LADDER:
        mo = run_monolith(soil_name, nz, rain)
        tp = f"{mo['t_pond']:.4f}" if mo["t_pond"] else "  none"
        ratio = "".join(f"{mo['I'][t]/pond_ref[nz][t]:>8.2f} " for t in TIMES)
        print(f"  {'monolith q_pot':>22} {nz:>4} {mo['ell_c']*1000:>9.2f} | " +
              "".join(f"{mo['I'][t]*1000:>8.2f} " for t in TIMES) +
              f"| {tp:>6} {mo['d_max']*1000:>8.2f} {mo['nbad']:>4} {mo['wall']:>4.0f}s", flush=True)
        print(f"  {'(ratio mono/ponding)':>22} {nz:>4} {'':>9} | " + ratio, flush=True)
    print("\nREAD: if monolith/ponding -> 1.0 as nz grows (ell_c=dz/2 -> 0), the q_pot throttle is a genuine")
    print("      surface-RESOLUTION artifact (both converge to the same sorptive uptake). If it plateaus")
    print("      < 1, q_pot is a true sub-grid closure that does real work (a film resistance).")
    print("#" * 100, flush=True)


if __name__ == "__main__":
    main()
