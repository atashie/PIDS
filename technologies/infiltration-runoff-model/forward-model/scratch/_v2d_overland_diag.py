"""P0 minimal reproducer: standalone Module-2 OverlandProblem on the 2-D tilted-V (no Richards/NCP).

Plan: docs/plans/2026-06-11-overland-convergent-flow-stabilization.md, Phase P0. Isolates Defect A
(unstabilized Galerkin advection at the convergence line) from the coupling, and attributes the
mass-ledger gap per accepted step:

    gap_k = [W(t_{k+1}) - W(t_k)] - dt*(rain*A - Q_out) - dclip_k

with W = total_water() (lumped storage = what the residual integrates), Q_out the SOLVED-state
outflow (pre-limiter, residual-consistent), and dclip_k the limiter's tracked degenerate-branch
adjustment. If the booked step truly converged (residual ~ atol), gap_k ~ 0 structurally; a large
gap_k on an accepted step is an UNBALANCED-RESIDUAL injection -> correlate with the SNES converged
reason (B1: reason 4 = SNORM_RELATIVE = stalled line search) and ||F|| at acceptance.

Candidates (plan SS2, Defect B): B1 snorm-acceptance | B2 limiter degenerate branch (dclip) |
B3 limiter<->NCP staleness (no NCP here -- the standalone isolates it away).

Env knobs:
  NX,NY (also argv[1:3])  mesh (default 48x30 = the ParFlow grid)
  T_END (argv[3])         total time, day (default 0.125 = storm+recession)
  STOL0=1                 apply O5 acceptance hardening (snes_stol=0) -- the pre/post probe
  DT_MAX, GROW_AT, SHRINK_AT  adaptive-controller knobs (spike defaults)
  OUT                     npz path (default scratch/v2d_diag_<nx>x<ny>[_stol0].npz)

Run (pids-fem): PYTHONPATH=. python scratch/_v2d_overland_diag.py [NX NY T_END]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem import petsc as fpetsc

from pids_forward.physics.overland import OverlandProblem, SECONDS_PER_DAY

# canonical tilted-V (Di Giammarco 1996 / Kollet-Maxwell 2006), same numbers as _tiltedv_spike.py
LX, LY = 1620.0, 1000.0
XC = LX / 2.0
SX, SY = 0.05, 0.02
N_MAN = 0.015
RAIN = 0.2592                  # m/day = 3e-6 m/s
STORM = 0.0625                 # day (90 min)
T_END = 0.125
AREA = LX * LY
Q_EQ = RAIN * AREA

NX, NY = 48, 30                # ParFlow's B6 grid
if len(sys.argv) >= 3:
    NX, NY = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) >= 4:
    T_END = float(sys.argv[3])
NX = int(os.environ.get("NX", NX)); NY = int(os.environ.get("NY", NY))

STOL0 = os.environ.get("STOL0", "0") == "1"
DT_MAX = float(os.environ.get("DT_MAX", "1e-3"))
GROW_AT = int(os.environ.get("GROW_AT", "3"))
SHRINK_AT = int(os.environ.get("SHRINK_AT", "8"))


def z_b(x):
    return SY * (LY - x[1]) + SX * np.abs(x[0] - XC)


def main():
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [LX, LY]], [NX, NY])
    opts = dict(OverlandProblem._DEFAULT_PETSC_OPTIONS)
    if STOL0:
        opts["snes_stol"] = 0.0   # O5: forbid SNORM_RELATIVE (reason 4) acceptance
    prob = OverlandProblem(msh, N_MAN, petsc_options=opts)
    prob.set_topography(z_b)
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)

    print(f"[setup] 2-D V {NX}x{NY} on {LX:g}x{LY:g} m  n={N_MAN}  Q_eq={Q_EQ:.0f} m^3/day  "
          f"STOL0={int(STOL0)}", flush=True)

    # DEEP decomposition machinery: the booked per-step gap splits EXACTLY as
    #   gap_k = h*rowsum_k + h*(Q_booked - Q_res(post-limiter))
    # where rowsum_k = sum of ALL residual rows assembled at the BOOKED state pair (d_post, d_n_old)
    # (the true mass injection the books absorbed: storage rows sum to dW/h, rain rows to -rA,
    # conveyance rows to 0 structurally, outlet rows to +Q under the RESIDUAL form's quadrature),
    # and the remainder is the OUTLET-QUADRATURE BOOKING MISMATCH (outflow_rate()'s standalone
    # scalar form vs the residual form's auto degree) + (clip steps only) the limiter's outlet-depth
    # perturbation. Q20 (degree-20 quadrature) approximates the true outlet integral.
    F_form = fem.form(prob.F)                 # final residual (rain + outlet woven in)
    Fvec = fpetsc.create_vector(prob.V)       # dolfinx 0.10: from the function space
    fdim = msh.topology.dim - 1
    ofacets = np.sort(dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], LY)))
    oft = dmesh.meshtags(msh, fdim, ofacets, np.full(ofacets.size, 1, dtype=np.int32))
    ds20 = ufl.Measure("ds", domain=msh, subdomain_data=oft,
                       metadata={"quadrature_degree": 20})(1)
    d_pos = ufl.max_value(prob.d, 0.0)
    Q20_form = fem.form(SECONDS_PER_DAY * (1.0 / N_MAN) * d_pos ** (5.0 / 3.0) * ufl.sqrt(SY) * ds20)

    def deep_probe(d_prev_arr):
        """Assemble (rowsum, |F|_2, Q_default, Q_deg20) at the booked state (d_post, d_n_old)."""
        dn_save = prob.d_n.x.array.copy()
        prob.d_n.x.array[:] = d_prev_arr
        prob.d_n.x.scatter_forward()
        with Fvec.localForm() as lf:
            lf.set(0.0)
        fpetsc.assemble_vector(Fvec, F_form)
        Fvec.assemble()
        rowsum = float(Fvec.sum())
        fnorm2 = float(Fvec.norm())
        q_def = float(fem.assemble_scalar(prob._outflow_forms[0]))
        q_20 = float(fem.assemble_scalar(Q20_form))
        prob.d_n.x.array[:] = dn_save
        prob.d_n.x.scatter_forward()
        return rowsum, fnorm2, q_def, q_20

    # per-step records
    rec = {k: [] for k in ("t", "dt", "reason", "iters", "fnorm", "rainA", "qout", "dW",
                           "dclip", "gap", "clipdepth", "accepted",
                           "rowsum", "fnorm2post", "qdef", "q20")}
    t, dt = 0.0, 1e-5
    n_acc = n_rej = 0
    W = prob.total_water()
    cum_rain = cum_out = cum_gap = 0.0
    clip_prev = prob.clip_mass_adjust
    snes = None
    t0 = time.time()
    next_print = 0.0
    while t < T_END - 1e-12:
        h = min(dt, T_END - t)
        rain.value = RAIN if (t + h) <= STORM + 1e-12 else 0.0
        d_prev = prob.d_n.x.array.copy()           # booked-state pair needs the OLD d_n
        converged, it = prob.step(h)
        if snes is None:
            snes = prob._problem.solver            # scratch probe: read the SNES directly
        reason = int(snes.getConvergedReason())
        fnorm = float(snes.getFunctionNorm())
        if converged:
            Wn = prob.total_water()
            dclip = prob.clip_mass_adjust - clip_prev
            clip_prev = prob.clip_mass_adjust
            rA = float(rain.value) * AREA
            gap = (Wn - W) - h * (rA - prob.last_outflow) - dclip
            rowsum, fnorm2post, q_def, q_20 = deep_probe(d_prev)
            cum_rain += h * rA
            cum_out += h * prob.last_outflow
            cum_gap += gap
            rec["t"].append(t + h); rec["dt"].append(h); rec["reason"].append(reason)
            rec["iters"].append(it); rec["fnorm"].append(fnorm); rec["rainA"].append(rA)
            rec["qout"].append(prob.last_outflow); rec["dW"].append(Wn - W)
            rec["dclip"].append(dclip); rec["gap"].append(gap)
            rec["clipdepth"].append(prob.last_clip); rec["accepted"].append(1)
            rec["rowsum"].append(rowsum); rec["fnorm2post"].append(fnorm2post)
            rec["qdef"].append(q_def); rec["q20"].append(q_20)
            W = Wn
            t += h
            n_acc += 1
            dt = min(dt * (1.5 if it <= GROW_AT else 0.7 if it >= SHRINK_AT else 1.0), DT_MAX)
            if t >= next_print:
                print(f"  t={t:8.5f} q/Qeq={prob.last_outflow/Q_EQ:6.3f} dt={h:8.2e} it={it:2d} "
                      f"reason={reason:2d} |F|={fnorm:9.2e} gap={gap:+10.3e} "
                      f"h*rowsum={h*rowsum:+10.3e} qmis={gap-h*rowsum:+10.3e}", flush=True)
                next_print = t + T_END / 40.0
        else:
            rec["t"].append(t + h); rec["dt"].append(h); rec["reason"].append(reason)
            rec["iters"].append(it); rec["fnorm"].append(fnorm); rec["rainA"].append(0.0)
            rec["qout"].append(0.0); rec["dW"].append(0.0); rec["dclip"].append(0.0)
            rec["gap"].append(0.0); rec["clipdepth"].append(0.0); rec["accepted"].append(0)
            rec["rowsum"].append(0.0); rec["fnorm2post"].append(0.0)
            rec["qdef"].append(0.0); rec["q20"].append(0.0)
            n_rej += 1
            dt = h * 0.5
            if dt < 1e-11:
                print(f"  !! dt collapse at t={t:.6g}", flush=True)
                break
    run_s = time.time() - t0

    a = {k: np.asarray(v) for k, v in rec.items()}
    acc = a["accepted"] == 1
    reasons, counts = np.unique(a["reason"][acc], return_counts=True)
    # external ledger (storm-relevant): cum_rain - cum_out - dW_total  vs engine pieces
    dW_total = W - 0.0  # started dry (W0 = 0 surface water)
    ext_gap = cum_rain - cum_out - dW_total
    print(f"\n[ledger] cum_rain={cum_rain:.1f}  cum_out={cum_out:.1f}  dW={dW_total:.1f}  "
          f"clip_mass_adjust={prob.clip_mass_adjust:+.3f}", flush=True)
    print(f"[ledger] EXTERNAL gap = cum_rain - cum_out - dW = {ext_gap:+.3f} m^3 "
          f"({100.0*ext_gap/max(cum_rain,1e-9):+.2f}% of cum rain)", flush=True)
    print(f"[ledger] sum per-step booked gaps = {cum_gap:+.3f} m^3 "
          f"({100.0*cum_gap/max(cum_rain,1e-9):+.2f}%)  [identity: ext_gap = -sum_gap - clip_adjust: "
          f"{-cum_gap - prob.clip_mass_adjust:+.3f}]", flush=True)
    hrs = a["dt"][acc] * a["rowsum"][acc]
    qmis = a["gap"][acc] - hrs
    print(f"[decompose] TRUE unbalanced-residual injection sum(h*rowsum) = {hrs.sum():+.3f} m^3 "
          f"({100.0*hrs.sum()/max(cum_rain,1e-9):+.3f}%)", flush=True)
    print(f"[decompose] outlet-booking mismatch sum(gap - h*rowsum)      = {qmis.sum():+.3f} m^3 "
          f"({100.0*qmis.sum()/max(cum_rain,1e-9):+.3f}%)", flush=True)
    nc = acc & (a["clipdepth"] == 0.0)
    print(f"[decompose] no-clip accepted steps: {int(nc.sum())}/{int(acc.sum())}  "
          f"(on these, qmis = pure quadrature mismatch of the booked outflow form)", flush=True)
    qd = a["qdef"][acc]; q2 = a["q20"][acc]
    big = qd > 0.01 * Q_EQ
    if big.any():
        rel = (qd[big] - q2[big]) / q2[big]
        print(f"[decompose] booked-form vs deg-20 outlet integral (post states, Q>1%Qeq): "
              f"median rel {np.median(rel):+.2e}  max |rel| {np.max(np.abs(rel)):.2e}", flush=True)
    print(f"[steps] accepted={n_acc} rejected={n_rej}  run={run_s:.1f}s  "
          f"max_clip={prob.max_clip_seen:.4f} m", flush=True)
    print(f"[reasons] accepted-step histogram: " +
          "  ".join(f"reason {int(r)}: {c}" for r, c in zip(reasons, counts)), flush=True)
    for r in reasons:
        m = acc & (a["reason"] == r)
        print(f"  reason {int(r)}: n={int(m.sum())}  sum_gap={a['gap'][m].sum():+10.3e}  "
              f"max|F|={a['fnorm'][m].max():9.2e}  med|F|={np.median(a['fnorm'][m]):9.2e}", flush=True)
    worst = np.argsort(-np.abs(a["gap"]))[:10]
    print("[worst gap steps]  t        dt        reason it  |F|       gap        h*rowsum   qmis", flush=True)
    for i in worst:
        print(f"  {a['t'][i]:9.5f} {a['dt'][i]:9.2e} {int(a['reason'][i]):3d} "
              f"{int(a['iters'][i]):3d} {a['fnorm'][i]:9.2e} {a['gap'][i]:+10.3e} "
              f"{a['dt'][i]*a['rowsum'][i]:+10.3e} {a['gap'][i]-a['dt'][i]*a['rowsum'][i]:+10.3e}",
              flush=True)

    out = os.environ.get("OUT", f"scratch/v2d_diag_{NX}x{NY}{'_stol0' if STOL0 else ''}.npz")
    np.savez(out, **a, cum_rain=cum_rain, cum_out=cum_out, dW_total=dW_total, ext_gap=ext_gap,
             clip_mass_adjust=prob.clip_mass_adjust, n_acc=n_acc, n_rej=n_rej, run_s=run_s,
             NX=NX, NY=NY, Q_eq=Q_EQ, storm=STORM, t_end=T_END, stol0=int(STOL0))
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
