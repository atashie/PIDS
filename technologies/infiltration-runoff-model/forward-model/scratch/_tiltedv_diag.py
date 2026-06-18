"""P0 coupled tilted-V diagnostic run: _tiltedv_spike.py + per-step ENGINE-EXACT ledger attribution.

Plan: docs/plans/2026-06-11-overland-convergent-flow-stabilization.md, Phase P0. Same canonical
case/mesh/controller as the B6 spike, but every accepted step books

    gap_k = [dSoil_k + dSurf_k] - dt*(rain*A - Q_out) - dclip_k

(W in the residual-consistent storage measures; Q_out the SOLVED-state pre-limiter outflow;
dclip_k the limiter's tracked degenerate-branch adjustment), plus the SNES converged REASON and
||F|| at acceptance. Attribution: B1 (snorm-stall acceptance, reason 4 / large ||F||) vs B2
(limiter degenerate branch, dclip) vs sampling artifacts in the old 40-point trapz harness ledger.

Env knobs: NX NY T_END (argv), STOL0=1 (O5 hardening probe), DT_MAX/GROW_AT/SHRINK_AT, KS, OUT.

Run (pids-fem): PYTHONPATH=. python scratch/_tiltedv_diag.py [NX NY T_END]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem

SCALE = float(os.environ.get("SCALE", "1.0"))
LX, LY, H = 1620.0 * SCALE, 1000.0 * SCALE, 2.0
XC = LX / 2.0
SX, SY = 0.05, 0.02
N_MAN = float(os.environ.get("N_MAN", "0.015"))
RAIN = 0.2592
STORM, T_END = 0.0625, 0.125
KS = float(os.environ.get("KS", "1.0e-3"))
SOIL = VanGenuchten(theta_r=0.10, theta_s=0.40, alpha=2.0, n=2.0, Ks=KS)
AREA = LX * LY
Q_EQ = RAIN * AREA

NX, NY, NZ = 24, 16, 3
if len(sys.argv) >= 3:
    NX, NY = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) >= 4:
    T_END = float(sys.argv[3])

STOL0 = os.environ.get("STOL0", "0") == "1"
SCHEME = os.environ.get("OVERLAND_SCHEME", "galerkin")  # "galerkin" (default) or "upwind" (P2 re-baseline)
DT_MAX = float(os.environ.get("DT_MAX", "1e-3"))
GROW_AT = int(os.environ.get("GROW_AT", "3"))
SHRINK_AT = int(os.environ.get("SHRINK_AT", "8"))


def z_b(x):
    return SY * (LY - x[1]) + SX * np.abs(x[0] - XC)


def main():
    t0 = time.time()
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [LX, LY, H]], [NX, NY, NZ])
    opts = dict(CoupledProblem._DEFAULT_PETSC_OPTIONS)
    if STOL0:
        opts["snes_stol"] = 0.0   # O5: forbid SNORM_RELATIVE (reason 4) acceptance
    prob = CoupledProblem(msh, SOIL, n_man=N_MAN, petsc_options=opts, overland_scheme=SCHEME)
    if SCHEME == "upwind" and os.environ.get("POS_TOL"):
        # relax the positivity tripwire for the DIAGNOSTIC (kink-V is the B5b measure-zero-channel
        # artifact worst case; this lets the run complete so we can CHARACTERIZE the max undershoot
        # + confirm the dt-pin/conservation, instead of aborting). NOT a production setting.
        prob._upwind_pos_tol = float(os.environ["POS_TOL"])
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[2], d_value=0.0)
    prob.set_topography(z_b)
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)
    print(f"[setup] mesh {NX}x{NY}x{NZ} on {LX:g}x{LY:g}x{H:g} m  Ks={KS:g}  ell_c={prob.ell_c:.3f}  "
          f"Q_eq={Q_EQ:.0f} m^3/day  scheme={SCHEME}  STOL0={int(STOL0)}  setup={time.time()-t0:.1f}s",
          flush=True)

    rec = {k: [] for k in ("t", "dt", "reason", "iters", "fnorm", "rainA", "qout", "dW",
                           "dsoil", "dsurf", "dclip", "gap", "clipdepth", "accepted")}
    t, dt = 0.0, 1e-5
    n_acc = n_rej = 0
    Wsoil = prob.soil_water()
    Wsurf = prob.surface_water()
    w0_soil = Wsoil
    cum_rain = cum_out = cum_gap = 0.0
    clip_prev = prob.clip_mass_adjust
    t_run = time.time()
    next_print = 0.0
    while t < T_END - 1e-12:
        h = min(dt, T_END - t)
        rain.value = RAIN if (t + h) <= STORM + 1e-12 else 0.0
        converged, it = prob.step(h)
        # ENGINE's audited verdict, not snes.getFunctionNorm() directly: step() recomputes
        # last_fnorm at the returned iterate on reason-4 exits (PETSc can cache the previous
        # iterate's norm on a failed line search) -- Codex P0 review 2026-06-12.
        reason = int(prob.last_reason)
        fnorm = float(prob.last_fnorm)
        if converged:
            Wsoil_n = prob.soil_water()
            Wsurf_n = prob.surface_water()
            dclip = prob.clip_mass_adjust - clip_prev
            clip_prev = prob.clip_mass_adjust
            rA = float(rain.value) * AREA
            dW = (Wsoil_n - Wsoil) + (Wsurf_n - Wsurf)
            gap = dW - h * (rA - prob.last_outflow) - dclip
            cum_rain += h * rA
            cum_out += h * prob.last_outflow
            cum_gap += gap
            rec["t"].append(t + h); rec["dt"].append(h); rec["reason"].append(reason)
            rec["iters"].append(it); rec["fnorm"].append(fnorm); rec["rainA"].append(rA)
            rec["qout"].append(prob.last_outflow); rec["dW"].append(dW)
            rec["dsoil"].append(Wsoil_n - Wsoil); rec["dsurf"].append(Wsurf_n - Wsurf)
            rec["dclip"].append(dclip); rec["gap"].append(gap)
            rec["clipdepth"].append(prob.last_clip); rec["accepted"].append(1)
            Wsoil, Wsurf = Wsoil_n, Wsurf_n
            t += h
            n_acc += 1
            dt = min(dt * (1.5 if it <= GROW_AT else 0.7 if it >= SHRINK_AT else 1.0), DT_MAX)
            if t >= next_print:
                print(f"  t={t:8.5f} q/Qeq={prob.last_outflow/Q_EQ:6.3f} dt={h:8.2e} it={it:2d} "
                      f"reason={reason:2d} |F|={fnorm:9.2e} gap={gap:+10.3e} dclip={dclip:+9.2e} "
                      f"clip={prob.last_clip:7.4f}", flush=True)
                next_print = t + T_END / 50.0
        else:
            rec["t"].append(t + h); rec["dt"].append(h); rec["reason"].append(reason)
            rec["iters"].append(it); rec["fnorm"].append(fnorm); rec["rainA"].append(0.0)
            rec["qout"].append(0.0); rec["dW"].append(0.0); rec["dsoil"].append(0.0)
            rec["dsurf"].append(0.0); rec["dclip"].append(0.0); rec["gap"].append(0.0)
            rec["clipdepth"].append(0.0); rec["accepted"].append(0)
            n_rej += 1
            dt = h * 0.5
            if dt < 1e-11:
                print(f"  !! dt collapse at t={t:.6g}", flush=True)
                break
    run_s = time.time() - t_run

    a = {k: np.asarray(v) for k, v in rec.items()}
    acc = a["accepted"] == 1
    reasons, counts = np.unique(a["reason"][acc], return_counts=True)
    dW_total = (Wsoil - w0_soil) + Wsurf  # started with Wsurf0 = 0
    ext_gap = cum_rain - cum_out - dW_total
    print(f"\n[ledger] cum_rain={cum_rain:.1f}  cum_out={cum_out:.1f}  dW={dW_total:.1f} "
          f"(soil {Wsoil-w0_soil:+.1f}, surf {Wsurf:+.1f})  "
          f"clip_mass_adjust={prob.clip_mass_adjust:+.3f}", flush=True)
    print(f"[ledger] EXTERNAL gap = cum_rain - cum_out - dW = {ext_gap:+.3f} m^3 "
          f"({100.0*ext_gap/max(cum_rain,1e-9):+.2f}% of cum rain)", flush=True)
    print(f"[ledger] sum per-step RESIDUAL gaps (B1) = {cum_gap:+.3f} m^3 "
          f"({100.0*cum_gap/max(cum_rain,1e-9):+.2f}%)  [identity check ext = -(sum_gap) - clip: "
          f"{-cum_gap - prob.clip_mass_adjust:+.3f}]", flush=True)
    print(f"[steps] accepted={n_acc} rejected={n_rej}  run={run_s/60.0:.1f} min  "
          f"max_clip={prob.max_clip_seen:.4f} m", flush=True)
    print(f"[reasons] accepted-step histogram: " +
          "  ".join(f"reason {int(r)}: {c}" for r, c in zip(reasons, counts)), flush=True)
    for r in reasons:
        m = acc & (a["reason"] == r)
        print(f"  reason {int(r)}: n={int(m.sum())}  sum_gap={a['gap'][m].sum():+10.3e}  "
              f"max|F|={a['fnorm'][m].max():9.2e}  med|F|={np.median(a['fnorm'][m]):9.2e}", flush=True)
    worst = np.argsort(-np.abs(a["gap"]))[:10]
    print("[worst gap steps]  t        dt        reason it  |F|       gap        dclip", flush=True)
    for i in worst:
        print(f"  {a['t'][i]:9.5f} {a['dt'][i]:9.2e} {int(a['reason'][i]):3d} "
              f"{int(a['iters'][i]):3d} {a['fnorm'][i]:9.2e} {a['gap'][i]:+10.3e} "
              f"{a['dclip'][i]:+9.2e}", flush=True)

    out = os.environ.get(
        "OUT", f"scratch/tiltedv_diag_{NX}x{NY}_{SCHEME}{'_stol0' if STOL0 else ''}.npz")
    np.savez(out, **a, cum_rain=cum_rain, cum_out=cum_out, dW_total=dW_total, ext_gap=ext_gap,
             clip_mass_adjust=prob.clip_mass_adjust, n_acc=n_acc, n_rej=n_rej, run_s=run_s,
             NX=NX, NY=NY, NZ=NZ, Q_eq=Q_EQ, storm=STORM, t_end=T_END, stol0=int(STOL0),
             Ks=KS, scale=SCALE, scheme=SCHEME,
             grow_at=GROW_AT, shrink_at=SHRINK_AT, dt_max=DT_MAX, comm_size=MPI.COMM_WORLD.size)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
