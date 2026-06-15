"""ITEM (B) Part B -- the PRE-REGISTERED discriminator: dt-refinement of the FEM drain refs.

POST-REGEN NOTE (2026-06-15): after item B resolved, the drain refs were regenerated at a converged
dt cap, so the I_committed this probe loads is NOW the converged ref -> the "ref-BE share =
I_true/I_committed - 1" it prints will read ~0 (the committed ref already IS the dt->0 limit). The
original measurement -- committed refs ~3.7-4.8% LOW, the ~4% first-order backward-Euler under-count
that WAS the recorded "+2-4% over" -- is preserved in
validation/sanity/m4_phase4_drain_endbias_attribution__2026-06-15.md.

The free analysis (scratch/_b_drain_endbias.py) ruled out the law's forward-Euler integration
(+0.01%) and REFUTED the static volume-average gap (M1 predicts -1%, wrong sign). The +2-4% over
must be either (a) transient-PSS model-form (the -3/4 is a uniform-depletion constant; real
depletion is wall-concentrated -> true effective constant smaller -> law over-predicts the PHYSICS)
or (b) the refs' backward-Euler under-count (refs read low -> law only APPEARS to over-predict).

This re-runs each closed-drain ref (1-D radial, the SAME machinery as m4_phase4_refAB_drain via its
_setup) with the BE step hard-capped at dt_max, Richardson-extrapolates I_ref to dt->0, and reports
law_end / I_ref(dt->0).

PRE-REGISTERED (written before running):
  (b) ref-BE dominant  -> I_ref RISES with finer dt; law/I_true collapses to ~1.00 (or ~0.99 per M1).
  (a) model-form       -> I_ref barely moves; law/I_true stays ~1.03-1.04 (the recorded bias).
The ref-BE SHARE = I_true/I_committed - 1 (how much the committed ref was low). first-order BE -> the
level-to-level I changes should shrink ~4x per dt/4 level (convergence witness).

Run (WSL pids-fem, from forward-model/):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
      python scratch/_b_drain_refbe.py
"""
import sys, pathlib, time
import numpy as np
from mpi4py import MPI
from dolfinx import fem

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import scratch.m4_phase4_refAB_drain as gen
import scratch.m4_phase1b_disperse_reference as dz
from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.sorptive_closure import rel_l2
from scratch.m4_phase4_drain_desorptivity import bruce_klute_desorptivity, pss_drain, LOAM

COMM = MPI.COMM_WORLD
R_W = 0.05
PSI_I, PSI_WALL = -0.03, -1.0
SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)
SILT = VanGenuchten(theta_r=0.034, theta_s=0.46, alpha=1.6, n=1.37, Ks=0.06)
HERE = pathlib.Path(__file__).parent


def solve_capped(problem, psi, psi_n, dt_c, t0, t_target, dt, dt_max):
    """dz._solve_to with a hard dt_max cap on every BE step (adaptive controller still drops BELOW
    dt_max on Newton stiffness, never rises above it)."""
    t, nstep = t0, 0
    while t < t_target - 1e-15:
        h = min(dt, t_target - t, dt_max)
        dt_c.value = h
        problem.solve()
        snes = problem.solver
        if snes.getConvergedReason() > 0:
            psi_n.x.array[:] = psi.x.array
            psi_n.x.scatter_forward()
            t += h
            nstep += 1
            it = int(snes.getIterationNumber())
            dt = dt * 1.5 if it <= 4 else (dt * 0.7 if it >= 8 else dt)
            dt = min(dt, dt_max)
        else:
            psi.x.array[:] = psi_n.x.array
            psi.x.scatter_forward()
            dt *= 0.5
            assert dt > 1e-13, f"dt collapse near t={t:.3e}"
    return dt, nstep


def run_capped(soil, r_out, samples, psi_i, dt_max):
    _, _, psi, psi_n, dt_c, _, problem, removed, _ = gen._setup(soil, r_out, gen.CELL, psi_i)
    out, dt, t_prev, total = [], 1e-8, 0.0, 0
    for t_s in samples:
        dt, ns = solve_capped(problem, psi, psi_n, dt_c, t_prev, t_s, dt, dt_max)
        t_prev = t_s
        total += ns
        out.append(float(COMM.allreduce(fem.assemble_scalar(removed), op=MPI.SUM) / R_W))
    return np.array(out), total


def law_end(soil, R, t):
    S_bk = bruce_klute_desorptivity(soil, PSI_WALL, PSI_I)[0]
    I0 = S_bk * np.sqrt(t[0])
    return float(pss_drain(t, soil, PSI_I, PSI_WALL, R_W, R, I0, t[0])[-1])


LEGS = []
d = np.load(HERE / "m4_phase4_refD40_drain.npz")
LEGS.append(("refD40 LOAM R40", LOAM, 40, d["LOAM_t"], d["LOAM_I"]))
fr = np.load(HERE / "m4_phase4_drain_fresh_refs.npz")
LEGS.append(("SAND R40", SAND, 40, fr["SAND_R40_t"], fr["SAND_R40_I"]))
LEGS.append(("LOAM R20 (deep)", LOAM, 20, fr["LOAM_R20_t"], fr["LOAM_R20_I"]))
s = np.load(HERE / "m4_phase4_silt_drain_ref.npz")
LEGS.append(("SILT R40", SILT, 40, s["SILT_R40_t"], s["SILT_R40_I"]))


def study(name, soil, Rf, t, ref_committed):
    R = Rf * R_W
    W = float(t[-1])
    Ilaw = law_end(soil, R, t)
    print(f"\n{'='*100}\n{name}: R={Rf}r_w  window={W:.3f} d  committed I_end={ref_committed[-1]:.5e}  "
          f"law_end={Ilaw:.5e}  (recorded law/committed={Ilaw/ref_committed[-1]:.4f})", flush=True)
    levels = [("baseline(adaptive)", np.inf), ("W/256", W / 256), ("W/1024", W / 1024),
              ("W/4096", W / 4096)]
    ends = {}
    print(f"  {'level':20s} {'steps':>7s} {'I_end':>13s} {'reproL2':>9s} {'I/commit':>9s} "
          f"{'law/I':>8s}", flush=True)
    for lab, dtm in levels:
        t0 = time.time()
        I, nst = run_capped(soil, R, t, PSI_I, dtm)
        ends[lab] = I[-1]
        rl2 = rel_l2(I, ref_committed)
        print(f"  {lab:20s} {nst:7d} {I[-1]:13.6e} {rl2*100:8.3f}% {I[-1]/ref_committed[-1]:9.4f} "
              f"{Ilaw/I[-1]:8.4f}   ({time.time()-t0:.1f}s)", flush=True)
    # Richardson (first-order BE) from the two finest (factor-4 apart): I_true = I_f + (I_f - I_c)/3
    Ic, If = ends["W/1024"], ends["W/4096"]
    I_true = If + (If - Ic) / 3.0
    # convergence witness: successive changes should shrink ~4x
    c1 = ends["W/1024"] - ends["W/256"]
    c2 = ends["W/4096"] - ends["W/1024"]
    print(f"  -> level changes: (W/256->W/1024)={c1:+.3e}  (W/1024->W/4096)={c2:+.3e}  "
          f"ratio={c1/c2 if c2 else float('nan'):.2f} (first-order BE ~4)", flush=True)
    print(f"  -> Richardson I_true(dt->0) = {I_true:.6e}   ref-BE share = "
          f"{(I_true/ref_committed[-1]-1)*100:+.2f}%   LAW/I_true = {Ilaw/I_true:.4f} "
          f"({(Ilaw/I_true-1)*100:+.2f}%)", flush=True)
    return dict(name=name, depl=ref_committed[-1], law=Ilaw, committed=ref_committed[-1],
                I_true=I_true, refbe=I_true / ref_committed[-1] - 1, model=Ilaw / I_true - 1)


if __name__ == "__main__":
    rows = [study(*leg) for leg in LEGS]
    print(f"\n{'='*100}\nITEM (B) ATTRIBUTION SUMMARY (offline-law end bias = ref-BE x model-form)")
    print(f"{'='*100}")
    print(f"{'leg':20s} {'recorded bias':>13s} {'ref-BE share':>13s} {'model-form':>12s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:20s} {(r['law']/r['committed']-1)*100:+12.2f}% "
              f"{r['refbe']*100:+12.2f}% {r['model']*100:+11.2f}%")
    print("-" * 100)
    print("recorded bias = law/committed-1 (what the gate sees). ref-BE share = how much the committed")
    print("ref was LOW (backward-Euler under-count). model-form = law/I_true-1 (the law vs the dt->0")
    print("physics): >0 -> law genuinely over-predicts (candidate a); ~0/<0 -> law accurate (candidate b).")
