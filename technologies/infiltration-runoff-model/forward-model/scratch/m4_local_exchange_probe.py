"""PROBE (2026-06-22 unified-feature redesign): does a LOCAL sub-grid exchange reproduce the validated
radial references, and does its conductance need the bounded local funnel factor 1/ln(r_inner/r_w)?

CONTEXT (the design question). We are weighing whether to retire the WellIndexExchange clock/ring/PSS
apparatus and unify disperse+drain into ONE structure that exchanges with the soil via a simple
potential-driven flux. The structure stays SUB-GRID (a line), but the host soil around it is resolved
at a cell scale r_inner -- NOT down to the 5 cm wall. The open question (the line-source funnel): if the
flux reads the soil pressure at the local cell r_inner, does it (a) reproduce the truth and (b) stay
consistent as the mesh changes, or does it need a geometric correction?

THE MODEL UNDER TEST. The resolved soil lives on [r_inner, R_out]; the structure (inside r_inner) is
sub-grid. Water crosses into the resolved soil at r_inner as an IMPLICIT Robin flux (reduced cylindrical
units, the references' convention):
    f_in = G * kirchhoff(psi_inner, H_f) = G * [Phi(H_f) - Phi(psi_inner)]      (>0 disperse, <0 drain)
with psi_inner the SOLVED soil head at the inner cell. Two conductance models:
    WI   : G = 1 / ln(r_inner / r_w)        -- the bounded LOCAL funnel factor (well-index, read LOCALLY)
    CONST: G frozen at the r_inner = 4 r_w calibration (= the bare "C*area*dpsi" idea, no size factor)
As r_inner -> r_w, ln -> 0 and WI -> a Dirichlet wall = the reference itself.

WHY 1-D RADIAL. Both validated references are 1-D cylindrical radial closed-box solves
(scratch/m4_phase4_refA_disperse.py disperse, scratch/m4_phase4_refAB_drain.py drain), so this radial
probe is the EXACT axially-uniform cross-section of the 3-D line feature. Meshing [r_inner, R_out] finely
ISOLATES the sub-grid-shell read: the only approximation under test is the inner exchange; outer-mesh
(host) discretization is ordinary FEM convergence, handled elsewhere.

REFERENCES (the fully-resolved truths, wall meshed down to r_w):
    disperse RefA40 : tests/data/m4_phase4_refA_disperse.npz  (LOAM, R=40 r_w, psi_i=-1, wall 0)
    drain    refD40 : scratch/m4_phase4_refD40_drain.npz      (LOAM, R=40 r_w, psi_i=-0.03, wall -1)

NOTE (early-time): the probe's I is the uptake into the RESOLVED soil (r>=r_inner); the reference I also
includes water stored in the sub-grid shell r_w..r_inner (= the structure's OWN storage in the full
model). That shell capacity is printed as context -- it is the expected early-time offset, not an error.

Run (WSL): conda activate pids-fem && cd .../forward-model && PYTHONPATH=. \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scratch/m4_local_exchange_probe.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

import scratch.m4_phase1b_disperse_reference as dz
from pids_forward.physics.sorptive_closure import rel_l2

COMM = MPI.COMM_WORLD
R_W = dz.R_W                               # 0.05 m structure radius
LOAM = dz.SOILS["LOAM"]
CELL = 0.5 / 800.0                         # outer resolved-soil cell (the far field is smooth; the
#                                            sharp near-wall front lives in the UNmeshed sub-grid shell)
R_INNER_OVER_RW = (8.0, 4.0, 2.0)          # host cell scale around the structure: h = 0.4 / 0.2 / 0.1 m
G_CAL_OVER_RW = 4.0                        # CONST model frozen at this r_inner/r_w calibration point

_LU_CP = dict(dz._LU, snes_linesearch_type="cp")    # disperse saturated-wall step wants cp
_LU_BT = dict(dz._LU, snes_linesearch_type="bt")    # drain saturated-start step wants bt


def run_local(soil, r_out, r_inner, G, samples, *, psi_i, h_f, direction,
              cell=CELL, dt_max=np.inf, linesearch_opts=None):
    """1-D cylindrical radial solve on [r_inner, R_out], NO-FLOW outer, sub-grid structure entering at
    r_inner as the implicit Robin flux f_in = G*kirchhoff(psi_inner, h_f). Returns I(t) per wall area:
    disperse = soil GAIN /r_w; drain = soil LOSS /r_w (the references' metric)."""
    n = max(int(round((r_out - r_inner) / cell)), 40)
    msh = dmesh.create_interval(COMM, n, [r_inner, r_out])
    r = ufl.SpatialCoordinate(msh)[0]
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi, psi_n = fem.Function(V), fem.Function(V)
    psi.x.array[:] = psi_i; psi_n.x.array[:] = psi_i
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    Hf_c = fem.Constant(msh, PETSc.ScalarType(h_f))
    dxs = dz._vertex_dx()
    dxq = ufl.dx(metadata={"quadrature_degree": 8})
    theta, theta_n, K = soil.theta_ufl(psi), soil.theta_ufl(psi_n), soil.K_ufl(psi)
    # tag ONLY the inner boundary vertex; the outer (r_out) stays untagged -> natural no-flow
    inner_facets = dmesh.locate_entities_boundary(msh, 0, lambda x: np.isclose(x[0], r_inner))
    ft = dmesh.meshtags(msh, 0, np.sort(inner_facets).astype(np.int32),
                        np.ones(inner_facets.size, dtype=np.int32))
    ds_in = ufl.Measure("ds", domain=msh, subdomain_data=ft)(1)
    # bulk radial Richards (reduced cylindrical, measure r*dr) MINUS the implicit inner inflow
    F = (((theta - theta_n) / dt_c) * v * r * dxs
         + K * ufl.dot(ufl.grad(psi), ufl.grad(v)) * r * dxq
         - G * soil.kirchhoff_ufl(psi, Hf_c) * v * ds_in)
    opts = linesearch_opts or dz._LU
    problem = NonlinearProblem(F, psi, bcs=[], petsc_options_prefix="m4lx_", petsc_options=opts)

    th_i = float(soil.theta(psi_i))
    sgn = 1.0 if direction == "disperse" else -1.0          # disperse: soil gains; drain: soil loses
    stored = fem.form(sgn * (theta - th_i) * r * dxs)        # >= 0 for both
    out, dt, t_prev = [], 1e-8, 0.0
    for t_s in samples:
        dt = dz._solve_to(problem, psi, psi_n, dt_c, t_prev, float(t_s), dt, dt_max=dt_max)
        t_prev = float(t_s)
        out.append(float(COMM.allreduce(fem.assemble_scalar(stored), op=MPI.SUM) / R_W))
    return np.array(out)


def _shell_capacity(soil, r_inner, psi_i, h_f):
    """Upper-bound uptake stored in the UNmeshed sub-grid shell r_w..r_inner /r_w (the wall side fully
    at h_f): the expected early-time offset between probe (resolved soil only) and reference."""
    dth = abs(float(soil.theta(h_f) - soil.theta(psi_i)))
    return dth * (r_inner ** 2 - R_W ** 2) / (2.0 * R_W)


def _score(I_loc, I_ref, Imax):
    full = rel_l2(I_loc, I_ref)
    early = I_ref < 0.3 * Imax
    late = I_ref >= 0.3 * Imax
    e_early = rel_l2(I_loc[early], I_ref[early]) if np.any(early) else float("nan")
    e_late = rel_l2(I_loc[late], I_ref[late]) if np.any(late) else float("nan")
    return full, e_early, e_late


def run_direction(name, ref_t, ref_I, Imax, *, psi_i, h_f, direction, linesearch_opts, dt_div=None):
    r_out = 40.0 * R_W
    dt_max = (float(ref_t[-1]) / dt_div) if dt_div else np.inf
    G_cal = 1.0 / np.log(G_CAL_OVER_RW)                     # CONST model: frozen at r_inner = 4 r_w
    print(f"\n{'='*94}\n{name}: LOAM R=40 r_w  (psi_i={psi_i}, wall H_f={h_f}, {len(ref_t)} samples, "
          f"window {ref_t[-1]:.3g} d, I_max={Imax:.4f} m)\n{'='*94}")
    print(f"{'r_inner/r_w':>11} {'h [m]':>7} {'shell/Imax':>11} | "
          f"{'WI G':>7} {'WI full':>8} {'WI early':>9} {'WI late':>8} | "
          f"{'CONST full':>10} {'CONST early':>11} {'CONST late':>10}")
    rows = []
    for ratio in R_INNER_OVER_RW:
        r_inner = ratio * R_W
        G_wi = 1.0 / np.log(ratio)
        shell = _shell_capacity(LOAM, r_inner, psi_i, h_f)
        try:
            I_wi = run_local(LOAM, r_out, r_inner, G_wi, ref_t, psi_i=psi_i, h_f=h_f,
                             direction=direction, dt_max=dt_max, linesearch_opts=linesearch_opts)
            wf, we, wl = _score(I_wi, ref_I, Imax)
        except Exception as e:
            I_wi, wf, we, wl = None, float("nan"), float("nan"), float("nan")
            print(f"  [WI r_inner={ratio}r_w FAILED: {e}]", flush=True)
        try:
            I_c = run_local(LOAM, r_out, r_inner, G_cal, ref_t, psi_i=psi_i, h_f=h_f,
                            direction=direction, dt_max=dt_max, linesearch_opts=linesearch_opts)
            cf, ce, cl = _score(I_c, ref_I, Imax)
        except Exception as e:
            cf, ce, cl = float("nan"), float("nan"), float("nan")
            print(f"  [CONST r_inner={ratio}r_w FAILED: {e}]", flush=True)
        print(f"{ratio:>11.0f} {r_inner:>7.2f} {shell/Imax:>10.1%} | "
              f"{G_wi:>7.3f} {wf:>7.1%} {we:>8.1%} {wl:>7.1%} | "
              f"{cf:>9.1%} {ce:>10.1%} {cl:>9.1%}", flush=True)
        rows.append((ratio, G_wi, wf, we, wl, cf, ce, cl, I_wi))
    return rows


if __name__ == "__main__":
    disp = np.load("tests/data/m4_phase4_refA_disperse.npz")
    drain = np.load("scratch/m4_phase4_refD40_drain.npz")

    print("PROBE: local sub-grid exchange vs the resolved radial references")
    print("  WI    = read the local cell + the bounded local funnel factor 1/ln(r_inner/r_w)")
    print(f"  CONST = freeze the conductance at the r_inner={G_CAL_OVER_RW:.0f} r_w value (no size factor)")
    print("  'early' = ref I < 0.3 I_max (the sub-grid-shell-dominated start); 'late' = the bend/plateau")

    run_direction("DISPERSE RefA40", disp["LOAM_R40_t"], disp["LOAM_R40_I"],
                  float(disp["LOAM_R40_Imax"]), psi_i=-1.0, h_f=0.0,
                  direction="disperse", linesearch_opts=_LU_CP)

    run_direction("DRAIN refD40", drain["LOAM_t"], drain["LOAM_I"], float(drain["LOAM_Imax"]),
                  psi_i=float(drain["LOAM_psi_i"]), h_f=-1.0, direction="drain",
                  linesearch_opts=_LU_BT, dt_div=2048)   # match the ref's converged BE dt cap

    print(f"\n{'='*94}")
    print("READ: if WI 'late'/'full' stays low + flat across r_inner while CONST drifts -> the local read")
    print("needs the bounded funnel factor (one line), and with it reproduces the truth mesh-robustly.")
    print("'early' carries the expected sub-grid-shell offset (the structure's own storage in the full model).")
