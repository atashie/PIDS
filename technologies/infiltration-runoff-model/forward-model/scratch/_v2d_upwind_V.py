"""P1-B5 (THE decisive gate): the canonical 2-D tilted-V with the O1 UpwindOverlandProblem.

Does the monotone upwind edge-flux scheme FIX the convergent tilted-V where the validated galerkin
``OverlandProblem`` fails? This runner is the head-to-head: it reuses the EXACT geometry/forcing of
the galerkin diagnostic ``scratch/_v2d_overland_diag.py`` (canonical tilted-V, LX,LY = 1620,1000 m *
SCALE; SX=0.05 cross-slope, SY=0.02 valley; z_b = SY*(LY-y) + SX*|x-XC|; n=0.015; rain 0.2592 m/day
for STORM=0.0625 d then recession; outflow on the y=LY edge, slope SY), but drives the UPWIND class.

The galerkin baseline to beat (P0, parent §8.3): canonical plateau ~0.676-0.996*Q_eq with dt PINNED
~5e-5 (the limiter<->Newton churn); FIELD-SCALE (SCALE=0.1) 0.876*Q_eq (under-resolved by the
oscillatory scheme). The O1 target (parent §5 P1 gate): plateau -> Q_eq +-3%, mesh-convergent,
oscillation <=2% RMS, dt-pin LIFTED, field-scale ~1.0.

THE 2-D OUTLET SUBTLETY (handled in the engine, B5): the V outlet is a LINE of nodes; the discharge
is the INTEGRAL of the per-unit-width Manning flux over the outlet length, so each outlet node carries
its boundary-edge control length (``UpwindOverlandProblem.add_outflow_bc`` assembles ``int phi_k ds``;
1.0 at a 1-D point, the node's outlet-edge length in 2-D). Verified: ``scratch/_b5_outlet_probe.py``.

Metrics recorded (printed + npz): plateau Q/Q_eq (late-storm mean, away from cold-start), plateau
oscillation RMS, mesh convergence (run 48x30 vs 96x60), dt distribution (vs the galerkin ~5e-5 pin),
books gap (cum_rain - cum_out - dStorage), min(d) over the V (the actual undershoot -- per B3 the 2%
valley is a MILD front, sub-mm expected, MEASURED not assumed), runtime.

Env knobs:
  NX,NY (also argv[1:3])  mesh (default 48x30 = the ParFlow grid)
  SCALE                   1.0 canonical / 0.1 field-scale (162x100 m)
  T_END (argv[3])         total time, day (default = STORM = storm window only, enough for the plateau)
  DT_MAX, GROW_AT, SHRINK_AT  adaptive-controller knobs (mirroring the galerkin diag)
  OUT                     npz path

Run (pids-fem): PYTHONPATH=. python scratch/_v2d_upwind_V.py [NX NY T_END]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland_upwind import UpwindOverlandProblem

# canonical tilted-V (Di Giammarco 1996 / Kollet-Maxwell 2006) -- IDENTICAL numbers to the galerkin
# diagnostic _v2d_overland_diag.py (SCALE=0.1 = the 162 m field-scale variant, same shape/slopes).
SCALE = float(os.environ.get("SCALE", "1.0"))
LX, LY = 1620.0 * SCALE, 1000.0 * SCALE
XC = LX / 2.0
SX, SY = 0.05, 0.02
N_MAN = 0.015
RAIN = 0.2592                  # m/day = 3e-6 m/s
STORM = 0.0625                 # day (90 min)
AREA = LX * LY
Q_EQ = RAIN * AREA             # equilibrium discharge: all rain runs off [m^3/day]

# Default T_END = the storm window: the rising limb saturates to the plateau within the storm, so
# the storm window is enough to MEASURE the plateau (and is much cheaper than storm+recession at
# the FD-Jacobian SNES cost). Override via argv[3] for the recession tail if wanted.
T_END = STORM

NX, NY = 48, 30                # ParFlow's B6 grid
if len(sys.argv) >= 3:
    NX, NY = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) >= 4:
    T_END = float(sys.argv[3])
NX = int(os.environ.get("NX", NX)); NY = int(os.environ.get("NY", NY))
T_END = float(os.environ.get("T_END", T_END))

DT_MAX = float(os.environ.get("DT_MAX", "1e-3"))
GROW_AT = int(os.environ.get("GROW_AT", "3"))
SHRINK_AT = int(os.environ.get("SHRINK_AT", "8"))
DT0 = float(os.environ.get("DT0", "1e-5"))


def z_b(x):
    return SY * (LY - x[1]) + SX * np.abs(x[0] - XC)


def main():
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [LX, LY]], [NX, NY])
    prob = UpwindOverlandProblem(msh, N_MAN)
    prob.set_topography(z_b)
    prob.set_initial_condition(lambda x: 0.0 * x[0])  # dry start
    prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)  # the y=LY LINE outlet

    print(f"[setup] O1 UPWIND 2-D V {NX}x{NY} on {LX:g}x{LY:g} m  n={N_MAN}  "
          f"Q_eq={Q_EQ:.1f} m^3/day  SCALE={SCALE:g}  T_END={T_END:g}d  area={AREA:.1f} m^2",
          flush=True)

    rec = {k: [] for k in ("t", "dt", "reason", "iters", "fnorm", "qout", "dW", "mind", "accepted")}
    t, dt = 0.0, DT0
    n_acc = n_rej = 0
    W = prob.total_water()
    cum_rain = cum_out = 0.0
    run_min_d = float(prob.d.x.array.min())
    t0 = time.time()
    next_print = 0.0
    while t < T_END - 1e-12:
        h = min(dt, T_END - t)
        r = RAIN if (t + h) <= STORM + 1e-12 else 0.0
        prob.add_rain(r)
        converged, it = prob.step(h)
        reason = int(prob.last_reason)
        fnorm = float(prob.last_fnorm)
        if converged:
            Wn = prob.total_water()
            qout = prob.outflow_rate()  # VOLUMETRIC line discharge [m^3/day] (length-weighted, B5)
            rA = r * AREA
            cum_rain += h * rA
            cum_out += h * qout
            mind = float(prob.d.x.array.min())
            run_min_d = min(run_min_d, mind)
            rec["t"].append(t + h); rec["dt"].append(h); rec["reason"].append(reason)
            rec["iters"].append(it); rec["fnorm"].append(fnorm); rec["qout"].append(qout)
            rec["dW"].append(Wn - W); rec["mind"].append(mind); rec["accepted"].append(1)
            W = Wn
            t += h
            n_acc += 1
            dt = min(dt * (1.5 if it <= GROW_AT else 0.7 if it >= SHRINK_AT else 1.0), DT_MAX)
            if t >= next_print:
                print(f"  t={t:9.6f} q/Qeq={qout / Q_EQ:7.4f} dt={h:9.2e} it={it:2d} "
                      f"reason={reason:2d} |F|={fnorm:9.2e} min_d={mind:+.2e}", flush=True)
                next_print = t + T_END / 30.0
        else:
            n_rej += 1
            dt = h * 0.5
            if dt < 1e-12:
                print(f"  !! dt collapse at t={t:.6g}", flush=True)
                break
    run_s = time.time() - t0

    a = {k: np.asarray(v) for k, v in rec.items()}
    acc = a["accepted"] == 1
    tt = a["t"][acc]
    q = a["qout"][acc] / Q_EQ
    dts = a["dt"][acc]

    # PLATEAU: the late-storm window, away from the cold-start rising limb. Take steps with
    # t in [0.6*STORM, STORM] (the rising limb saturates well before then) AND t <= T_END.
    plateau_lo, plateau_hi = 0.6 * min(STORM, T_END), min(STORM, T_END)
    pm = acc & (a["t"] >= plateau_lo) & (a["t"] <= plateau_hi + 1e-12)
    qp = a["qout"][pm] / Q_EQ
    plateau_mean = float(np.mean(qp)) if qp.size else float("nan")
    plateau_rms = float(np.sqrt(np.mean((qp - plateau_mean) ** 2))) if qp.size else float("nan")
    plateau_last = float(q[-1]) if q.size else float("nan")

    dW_total = W - 0.0  # started dry
    books_gap = cum_rain - cum_out - dW_total

    reasons, counts = np.unique(a["reason"][acc], return_counts=True)
    print(f"\n[plateau] window t in [{plateau_lo:.5f},{plateau_hi:.5f}] d, n={int(pm.sum())} steps",
          flush=True)
    print(f"[plateau] Q/Q_eq mean = {plateau_mean:.4f}   last-step Q/Q_eq = {plateau_last:.4f}",
          flush=True)
    print(f"[plateau] oscillation RMS = {plateau_rms:.4f} ({100*plateau_rms:.2f}%)  "
          f"[bar <=2%]", flush=True)
    print(f"[dt] over the run: min={dts.min():.2e} median={np.median(dts):.2e} max={dts.max():.2e}  "
          f"[galerkin pin ~5e-5]", flush=True)
    print(f"[dt] plateau-window dt: min={a['dt'][pm].min():.2e} median={np.median(a['dt'][pm]):.2e} "
          f"max={a['dt'][pm].max():.2e}", flush=True)
    print(f"[books] cum_rain={cum_rain:.2f}  cum_out={cum_out:.2f}  dStorage={dW_total:.2f}  "
          f"gap={books_gap:+.3e} m^3 ({100*books_gap/max(cum_rain,1e-9):+.4f}% of rain)", flush=True)
    print(f"[min_d] run-minimum depth on the V = {run_min_d:+.3e} m  "
          f"({'sub-mm' if abs(run_min_d) < 1e-3 else 'OVER 1mm'} undershoot)  [final min_d="
          f"{float(prob.d.x.array.min()):+.3e}]", flush=True)
    print(f"[steps] accepted={n_acc} rejected={n_rej}  run={run_s:.1f}s  "
          f"finite={bool(np.all(np.isfinite(prob.d.x.array)))}", flush=True)
    print(f"[reasons] accepted-step histogram: " +
          "  ".join(f"reason {int(rr)}: {c}" for rr, c in zip(reasons, counts)), flush=True)

    sfx = f"_s{SCALE:g}" if SCALE != 1.0 else ""
    out = os.environ.get("OUT", f"scratch/v2d_upwind_{NX}x{NY}{sfx}.npz")
    np.savez(out, **a, cum_rain=cum_rain, cum_out=cum_out, dW_total=dW_total, books_gap=books_gap,
             plateau_mean=plateau_mean, plateau_rms=plateau_rms, plateau_last=plateau_last,
             run_min_d=run_min_d, n_acc=n_acc, n_rej=n_rej, run_s=run_s, NX=NX, NY=NY, Q_eq=Q_EQ,
             storm=STORM, t_end=T_END, scale=SCALE, area=AREA)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
