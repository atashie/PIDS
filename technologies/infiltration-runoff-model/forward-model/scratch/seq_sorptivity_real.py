"""SCRATCH BENCHMARK v2 -- the DECISIVE sorptivity test (Codex's recommendation, §19).

Fixes the three §19 hits in seq_sorptivity_benchmark.py:
  (1) ★ uses the REAL monolith `CoupledProblem` (co-solved q_pot=kirchhoff(psi,d)/ell_c, the d=actual
      pond, NCP lambda), swept over ell_c (coarse half-cell -> ~1mm) -- NOT a hand-rolled frozen-Neumann
      fixed-h_ref surrogate. Answers: is the q_pot under-capture a COARSE-FILM artifact (vanishes as
      ell_c->0) or a TRUE closure gap?
  (2) consistent-quadrature INT theta (degree-8) for the truth measure (not only the lumped ledger), AND
      a real nz-convergence ladder for the ponded-Dirichlet reference (demonstrate convergence, not assert).
  (3) soil-matched DRY initial psi via a target effective saturation S_e (so "dry" is comparable across
      soils -- clay at psi=-0.4 was barely dry).

Setup: 1-D-style column (overland inert: a 2x2 horizontal box, flat, no slope -> no lateral routing), dry
IC, surface driven to PONDING by a high rain rate (>> Ks) so the monolith's surface saturates and we read
the infiltration-capacity I(t). Compare to a converged ponded-Dirichlet Richards column (the resolved
sorptive truth).

Run (WSL pids-fem) -- LIVE to a file (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_sorptivity_real.py <soil>'
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
TIMES = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
# DRY antecedent IC: a fixed DRY matric head (what "dry soil before a storm" physically means -- each soil
# takes its NATURAL water content at that suction; NOT a forced equal S_e, which is unphysical for clay:
# clay's flat retention needs psi~-2e8 m for S_e=0.18). -3 m is comfortably dry (well below field
# capacity ~-3.3 m is the classic FC head; -3 m ~ near FC, genuinely unsaturated for all four soils) and
# is finite/physical for every soil. Capped via S_e>=0.02 guard so no soil starts numerically at theta_r.
PSI_DRY = -3.0


def psi_for_se(soil_name, se):
    """van Genuchten psi giving effective saturation S_e = se (used only as a dryness DIAGNOSTIC/guard)."""
    s = SOILS[soil_name]; n = s["n"]; m = 1.0 - 1.0 / n; alpha = s["alpha"]
    return -(1.0 / alpha) * (se ** (-1.0 / m) - 1.0) ** (1.0 / n)


def psi_dry(soil_name):
    """The dry IC head for a soil: a fixed dry matric head PSI_DRY, but not so dry that S_e < 2% (guard)."""
    return max(PSI_DRY, psi_for_se(soil_name, 0.02))   # max -> the LESS negative (wetter) of the two


def _col(nz, nx=2, ny=2):
    return dmesh.create_box(COMM, [np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, LZ])],
                            [nx, ny, nz], cell_type=dmesh.CellType.tetrahedron)


def _theta_consistent(prob, soil):
    """INT theta(psi) with CONSISTENT (degree-8) quadrature -- the physical measure, not the lumped ledger."""
    dxq = ufl.dx(metadata={"quadrature_degree": 8})
    psi = prob.psi
    return prob.mesh.comm.allreduce(fem.assemble_scalar(fem.form(soil.theta_ufl(psi) * dxq)), op=MPI.SUM)


def _march(prob, soil, psi0, step_rain=None, dt0=2e-5, dt_max=2e-3, t_end=None):
    """March; record consistent I(t)=Delta(INT theta)/A at TIMES. step_rain: callable(prob) set each step
    (monolith) or None (Dirichlet). Returns {t: I}."""
    t_end = t_end or TIMES[-1]
    th0 = _theta_consistent(prob, soil)
    targets = list(TIMES); out = {}
    t, dt, nstep = 0.0, dt0, 0
    while t < t_end - 1e-12 and nstep < 30000:
        h = min(dt, t_end - t)
        if targets and t + h > targets[0]:
            h = targets[0] - t
        conv, it = prob.step(h)
        if not conv:
            dt *= 0.5
            if dt < 1e-9:
                break
            continue
        t += h; nstep += 1
        if it <= 3:
            dt = min(dt * 1.4, dt_max)
        elif it >= 8:
            dt *= 0.7
        while targets and t >= targets[0] - 1e-12:
            out[targets.pop(0)] = (_theta_consistent(prob, soil) - th0) / 0.01
    for tt in targets:
        out[tt] = (_theta_consistent(prob, soil) - th0) / 0.01
    return out, nstep


def run_dirichlet(soil_name, nz, psi0):
    soil = VanGenuchten(**SOILS[soil_name])
    rp = RichardsProblem(_col(nz), soil)
    rp.set_initial_condition(lambda x: psi0 + 0.0 * x[0])
    rp.add_dirichlet(lambda x: np.isclose(x[2], LZ), 0.0)
    t0 = time.perf_counter()
    I, ns = _march(rp, soil, psi0)
    return I, ns, time.perf_counter() - t0


def run_monolith(soil_name, nz, psi0, ell_c, rain):
    soil = VanGenuchten(**SOILS[soil_name])
    msh = _col(nz)
    prob = CoupledProblem(msh, soil, ell_c=ell_c, overland_scheme="galerkin")  # 1-D-ish: lateral inert
    prob.set_initial_condition(lambda x: psi0 + 0.0 * x[0], d_value=0.0)
    prob.set_topography(lambda x: 0.0 * x[0])
    prob.add_rain(rain)
    t0 = time.perf_counter()
    I, ns = _march(prob, soil, psi0)
    return I, ns, time.perf_counter() - t0


def main():
    np.set_printoptions(precision=4, suppress=True)
    soil_name = sys.argv[1] if len(sys.argv) > 1 else "loam"
    Ks = SOILS[soil_name]["Ks"]
    psi0 = psi_dry(soil_name)
    soil_d = VanGenuchten(**SOILS[soil_name])
    th_d = float(soil_d.theta(np.array([psi0]))[0])
    se_d = (th_d - SOILS[soil_name]["theta_r"]) / (SOILS[soil_name]["theta_s"] - SOILS[soil_name]["theta_r"])
    rain = 10.0 * Ks            # >> Ks -> surface ponds quickly -> infiltration-capacity regime
    print("#" * 100)
    print(f"DECISIVE SORPTIVITY (real CoupledProblem, ell_c sweep) -- {soil_name} (Ks={Ks})")
    print(f"  DRY IC: psi_i={psi0:.3f} m -> theta={th_d:.4f} (S_e={se_d:.3f});  ponding rain={rain} (=10xKs)")
    print(f"  I(t) [mm], consistent (deg-8) INT theta;  ell_c sweep vs converged ponded-Dirichlet")
    print("#" * 100, flush=True)

    # (2) nz-convergence ladder for the Dirichlet reference (DEMONSTRATE the truth).
    print("\n--- ponded-Dirichlet nz-convergence (the resolved sorptive truth) ---", flush=True)
    refs = {}
    for nz in (60, 120, 240):
        I, ns, w = run_dirichlet(soil_name, nz, psi0)
        refs[nz] = I
        print(f"  Dirichlet nz={nz:3d} | " + "".join(f"{I[t]*1000:>8.2f} " for t in TIMES) +
              f"  [{ns} steps {w:.0f}s]", flush=True)
    ref = refs[240]
    conv = max(abs(refs[120][t] - refs[240][t]) / max(refs[240][t], 1e-9) for t in TIMES)
    print(f"  => nz120->240 max rel change = {conv*100:.2f}%  ({'CONVERGED' if conv < 0.03 else 'NOT yet'})",
          flush=True)

    # (1) the REAL monolith, ell_c swept coarse->fine (the decisive control).
    print("\n--- REAL monolith CoupledProblem, ell_c sweep (nz=8 fixed; ell_c is the FILM scale) ---",
          flush=True)
    print(f"  {'ell_c[mm]':>10} | " + "".join(f"{t:>8.3f} " for t in TIMES) + " | ratio to Dirichlet-truth")
    for ell_c in (0.0625, 0.02, 0.005, 0.001):
        I, ns, w = run_monolith(soil_name, 8, psi0, ell_c, rain)
        rr = "".join(f"{I[t]/ref[t]:>5.2f} " for t in TIMES)
        print(f"  {ell_c*1000:>10.2f} | " + "".join(f"{I[t]*1000:>8.2f} " for t in TIMES) +
              f" | {rr} [{ns}st {w:.0f}s]", flush=True)
    print(f"\n  REFERENCE (Dirichlet nz240) | " + "".join(f"{ref[t]*1000:>8.2f} " for t in TIMES))
    print("\nREAD: if the monolith ratio -> 1.0 as ell_c->1mm, the q_pot gap is a COARSE-FILM ARTIFACT")
    print("      (refine the surface -> correct sorptivity). If it plateaus < 1, it is a TRUE closure gap.")
    print("#" * 100, flush=True)


if __name__ == "__main__":
    main()
