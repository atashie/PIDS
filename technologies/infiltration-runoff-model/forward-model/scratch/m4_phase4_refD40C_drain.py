"""refD40-C: the DISCRIMINATING drain HISTORY reference (LOAM, R=40 r_w, CONTINUOUS recharge).

Why v1 (refD40-B, the single deficit-aware pulse) measurably CANNOT discriminate (2026-06-12):
a one-shot band refill is STORAGE-capped at ~100% of the band's standing deficit (band psi must
stay below the h_s = -0.02 air entry), which is ~6% of the leg's end-I; the committed v1 leg moved
the end state only +2.0%, and the recharge-blind water-balance PSS twin -- the strongest
no-recharge-knowledge competitor, pss_drain tracking its OWN extraction ledger -- passes it at
1.5% while production drain tracks at 6.8%. Nothing to assert.

CONTINUOUS recharge breaks the storage cap: with the source active over [T1, T2] = [3, 18] d the
band RE-DRAINS WHILE BEING REFILLED, so the binding limit is the band->wall PSS THROUGHPUT
(dPhi(-1, ~-0.02)/(r_w ln(24)) ~ 0.141 m/day of wall-area-equivalent): the cumulative injection
can be MANY band deficits and the end-state gap vs the unpulsed curve is sized by V, not by one
deficit. The recharge-blind twin integrates the unperturbed depletion (its bulk-psi read only
ever falls) and misses the entire re-steepening; the embedded production drain (live
volumetric-mean host read) sees the recharged host and tracks. Note rel-L2 bounds the achievable
kill: an end-gap-G under-predictor can never exceed relL2 ~ G/(1+G), so the kickoff's first
suggestion (+15-25% gap over a [5,15] d window on the 20-d horizon) measurably CANNOT reach the
pre-registered BASELINE_KILL = 0.20 (offline design table: V=1.2 there gives twin relL2 18.8% at
0.85x throughput -- maxed out). The design that can: inject near the throughput edge over a
LONGER window and extend the horizon so more of the injected water is recovered through the wall
-- [3, 18] d, V = 1.8 m, horizon 30 d -> predicted twin relL2 = 32.4%, end gap = +56% at 0.85x
throughput ("hover the band just below theta_i" = the kickoff's regime; the FEM band guard
arbitrates validity). V is sized on the offline source-aware PSS design model (the PSS mechanism
is validated 3.3-3.9% on three drain refs) so the PREDICTED twin failure is >= TWIN_PRED_TARGET
= 0.30; the measured twin failure vs the FEM reference is asserted >= 0.20 (bounded retunes:
band-validity down-tune x0.8 / twin-kill up-tune x1.3).

Band [24, 32] r_w (the refB40/refD40B outer-bulk band -- host-mediated transmission to the
near-wall drive); source = CONSTANT rate (exactly replayable by the embedded harness's
volume-matched pulse machinery). Sample grid = the refD40 grid UNION a linear [6, 30] d
refinement (the discrimination era carries the signal; the unpulsed leg keeps its 32-pt grid).
Bookkeeping (closed domain): I_wall = [ integral (theta_i - theta) r dr + cum_source ] / r_w.
Validity: pre-window identity with the committed refD40 (same marching path -> ~machine zero),
band psi_max <= -0.02 (air entry) throughout, I monotone, end I <= I_max + V.

Run (WSL): conda activate pids-fem && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
    python scratch/m4_phase4_refD40C_drain.py [design]
("design" prints the offline sizing table and exits without FEM.)
"""
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import scratch.m4_phase1b_disperse_reference as dz
import scratch.m4_phase4_refAB_drain as gen
from scratch.m4_phase4_drain_desorptivity import (
    bruce_klute_desorptivity, make_psi_of_theta, pss_drain)
from pids_forward.physics.sorptive_closure import rel_l2

R_W = gen.R_W
R_OUT = 40 * R_W
PSI_I40, PSI_WALL = -0.03, -1.0
BAND = (24 * R_W, 32 * R_W)
T1, T2 = 3.0, 18.0
T_END = 30.0
V_START = 1.8                         # m (wall-area units), from the offline design table
TWIN_PRED_TARGET = 0.30               # design the leg so the PREDICTED twin failure is >= this
TWIN_MEASURED_MIN = 0.24              # up-tune V if the MEASURED failure is below this
HERE = pathlib.Path(__file__).parent


_SBK = {}


def sbk(soil):
    if id(soil) not in _SBK:
        _SBK[id(soil)] = bruce_klute_desorptivity(soil, PSI_WALL, PSI_I40)[0]
    return _SBK[id(soil)]


def pss_with_source(t, soil, V_wall, t1=T1, t2=T2, n_sub=200):
    """Source-AWARE offline PSS depletion (the design model; V_wall=0 reduces to the blind twin):
    theta_bulk = theta_i - (I - V_src(t)) * 2 r_w / (R^2 - r_w^2), V_src ramping linearly over
    [t1, t2] to V_wall (constant-rate source, wall-area units)."""
    psi_of_th = make_psi_of_theta(soil, PSI_WALL, PSI_I40)
    th_i = float(soil.theta(PSI_I40))
    geo = np.log(R_OUT / R_W) - 0.75
    I = np.empty_like(t)
    x = sbk(soil) * np.sqrt(t[0])
    I[0] = x
    for k in range(1, t.size):
        sub = np.linspace(t[k - 1], t[k], n_sub)
        for j in range(1, sub.size):
            v_src = V_wall * np.clip((sub[j - 1] - t1) / (t2 - t1), 0.0, 1.0)
            th_b = th_i - (x - v_src) * 2.0 * R_W / (R_OUT ** 2 - R_W ** 2)
            dPhi = float(soil.kirchhoff(PSI_WALL, float(psi_of_th(th_b))))
            x += dPhi / (R_W * geo) * (sub[j] - sub[j - 1])
        I[k] = x
    return I


def design(t, soil, V=V_START):
    """Verify on the offline model that V's PREDICTED recharge-blind twin rel-L2 reaches
    TWIN_PRED_TARGET (scan upward if not). Returns (V, predicted relL2, predicted end gap)."""
    twin = pss_with_source(t, soil, 0.0)
    print(f"[design] blind twin end I = {twin[-1]:.4f} m (PSS model, {t[-1]:.0f}-d horizon)")
    for _ in range(6):
        aware = pss_with_source(t, soil, V)
        e = rel_l2(twin, aware)
        gap = (aware[-1] - twin[-1]) / twin[-1]
        print(f"[design] V = {V:.3f} m -> predicted twin relL2 = {e:.1%}, end gap = +{gap:.1%}")
        if e >= TWIN_PRED_TARGET:
            return V, e, gap
        V *= 1.3
    raise AssertionError(f"design failed to reach {TWIN_PRED_TARGET:.0%}")


def run_fem(t_grid, soil, V_wall):
    """The resolved 1-D radial closed FEM reference with the constant band source (the refD40B
    machinery with the pulse replaced by the [T1,T2] constant-rate source sized a-priori)."""
    import ufl
    from mpi4py import MPI
    from dolfinx import mesh as dmesh, fem
    from dolfinx.fem.petsc import NonlinearProblem
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD
    n = max(int(round((R_OUT - R_W) / gen.CELL)), 80)
    msh = dmesh.create_interval(comm, n, [R_W, R_OUT])
    r = ufl.SpatialCoordinate(msh)[0]
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi, psi_n = fem.Function(V), fem.Function(V)
    psi.x.array[:] = PSI_I40
    psi_n.x.array[:] = PSI_I40
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    s_c = fem.Constant(msh, PETSc.ScalarType(0.0))
    dxs, dxq = dz._vertex_dx(), ufl.dx(metadata={"quadrature_degree": 8})
    theta, theta_n = soil.theta_ufl(psi), soil.theta_ufl(psi_n)
    K = soil.K_ufl(psi)
    band_ind = ufl.conditional(ufl.And(ufl.ge(r, BAND[0]), ufl.le(r, BAND[1])), 1.0, 0.0)
    F = (((theta - theta_n) / dt_c) * v * r * dxs
         + K * ufl.dot(ufl.grad(psi), ufl.grad(v)) * r * dxq
         - s_c * band_ind * v * r * dxq)
    wall_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], R_W))
    bcs = [fem.dirichletbc(PETSc.ScalarType(PSI_WALL), wall_dofs, V)]
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4p4dc_",
                               petsc_options=gen._DRAIN_LU)
    th_i = float(soil.theta(PSI_I40))
    removed = fem.form((th_i - theta) * r * dxs)
    xc = V.tabulate_dof_coordinates()[:, 0]
    in_band = (xc >= BAND[0]) & (xc <= BAND[1])
    Bw = (BAND[1] ** 2 - BAND[0] ** 2) / 2.0
    s_rate = V_wall * R_W / ((T2 - T1) * Bw)              # per-radian volume V_wall*r_w over [T1,T2]

    marks = np.unique(np.concatenate([t_grid, [T1, T2]]))
    I_wall, band_max = [], -np.inf
    dt, t_prev = 1e-8, 0.0
    for t_s in marks:
        active = (t_prev >= T1 - 1e-15) and (t_s <= T2 + 1e-15)
        s_c.value = s_rate if active else 0.0
        dt = dz._solve_to(problem, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        if active:
            band_max = max(band_max, float(psi.x.array[in_band].max()))
        if t_s in t_grid:
            loss = comm.allreduce(fem.assemble_scalar(removed), op=MPI.SUM)
            cum_src = s_rate * Bw * min(max(t_s - T1, 0.0), T2 - T1)
            I_wall.append((loss + cum_src) / R_W)
            print(f"  t={t_s:.3e}  I={I_wall[-1]:.4e}", flush=True)
    return np.array(I_wall), band_max


def main(design_only=False):
    soil = dz.SOILS["LOAM"]
    ref = np.load(HERE / "m4_phase4_refD40_drain.npz")
    t_unp, I_unp = ref["LOAM_t"], ref["LOAM_I"]
    t = np.unique(np.concatenate([t_unp, np.linspace(6.0, T_END, 10)]))

    # throughput-vs-injection feasibility (the design's validity condition, printed for the record;
    # the leg deliberately runs NEAR the edge -- "hover the band just below theta_i" -- and the
    # band air-entry guard on the FEM run arbitrates)
    cap = float(soil.kirchhoff(PSI_WALL, -0.021)) / (R_W * np.log(BAND[0] / R_W))
    V_wall, e_pred, gap_pred = design(t, soil)
    inj_rate = V_wall / (T2 - T1)
    print(f"[design] CHOSEN V = {V_wall:.3f} m over [{T1},{T2}] d -> inj {inj_rate:.3f} m/day "
          f"vs band->wall PSS throughput ~{cap:.3f} m/day (ratio {inj_rate/cap:.2f})")
    assert inj_rate < 0.9 * cap, "injection exceeds the band's re-drain throughput"
    if design_only:
        return

    twin = pss_drain(t, soil, PSI_I40, PSI_WALL, R_W, R_OUT, sbk(soil) * np.sqrt(t[0]), float(t[0]))
    dth = float(soil.theta(PSI_I40) - soil.theta(PSI_WALL))
    i_max = dth * (R_OUT ** 2 - R_W ** 2) / (2.0 * R_W)

    for attempt in range(3):
        I, band_max = run_fem(t, soil, V_wall)
        e_twin = rel_l2(twin, I)
        print(f"refD40-C (V={V_wall:.3f}): MEASURED recharge-blind twin relL2 = {e_twin:.1%} "
              f"(predicted {e_pred:.1%}); band psi_max = {band_max:.4f}")
        if band_max > -0.02:                       # validity first: back off the air-entry edge
            V_wall *= 0.8
            print(f"  band crossed air entry -> down-tune V x0.8 = {V_wall:.3f}")
            continue
        if e_twin < TWIN_MEASURED_MIN and attempt < 2:
            V_wall *= 1.3
            assert V_wall / (T2 - T1) < 0.9 * cap, "twin up-tune would exceed throughput"
            print(f"  twin failure below {TWIN_MEASURED_MIN:.0%} -> up-tune V x1.3 = {V_wall:.3f}")
            continue
        break

    # validity gates
    assert band_max <= -0.02 + 1e-12, f"band crossed air entry (psi_max {band_max:.4f})"
    assert np.all(np.diff(I) >= -1e-12 * i_max), "I not monotone"
    assert I[-1] <= (i_max + V_wall) * (1 + 1e-9), "capacity violated (I_max + injected)"
    pre = t < T1
    shared = np.isin(t[pre], t_unp)
    dev_pre = np.abs(I[pre][shared] - np.interp(t[pre][shared], t_unp, I_unp)) \
        / np.interp(t[pre][shared], t_unp, I_unp)
    assert dev_pre.max() < 1e-6, f"pre-window mismatch vs refD40: {dev_pre.max():.2e}"
    assert e_twin >= 0.20, f"twin kill below BASELINE_KILL after retune: {e_twin:.1%}"
    gap = (I[-1] - twin[-1]) / twin[-1]            # vs the twin's same-horizon unperturbed end
    print(f"refD40-C: end I = {I[-1]:.4f} m vs recharge-blind {twin[-1]:.4f} m (gap +{gap:.1%}); "
          f"pre-window dev {dev_pre.max():.1e}; twin relL2 {e_twin:.1%} (kill bar 0.20)")

    np.savez(HERE / "m4_phase4_refD40C_drain.npz",
             LOAM_t=t, LOAM_I=I, LOAM_t_src=np.array([T1, T2]),
             LOAM_V_per_wall_area=np.array(V_wall), LOAM_band=np.array(BAND),
             LOAM_band_psi_max=np.array(band_max), LOAM_Imax=np.array(i_max),
             LOAM_twin_relL2=np.array(e_twin), LOAM_gap=np.array(gap), r_w=np.array(R_W))
    print("Saved -> scratch/m4_phase4_refD40C_drain.npz")


if __name__ == "__main__":
    main(design_only=(len(sys.argv) > 1 and sys.argv[1] == "design"))
