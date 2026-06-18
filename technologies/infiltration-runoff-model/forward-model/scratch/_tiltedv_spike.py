"""B6 SPIKE (matchability gate): can the in-house CoupledProblem route a tilted-V catchment?

Tests the TWO unknowns before committing to the tilted-V benchmark:
  (1) CAPABILITY -- does the overland operator route convergently into the central channel and
      down the channel to a point-ish outlet (B5 only did a single planar slope -> straight edge)?
  (2) FEASIBILITY -- the canonical tilted-V is 1.62 km x 1.0 km, a ~300x scale jump over anything
      the in-house FEM has run (<=5 m). Does it stay conditioned + solve at acceptable cost?

Canonical geometry (Di Giammarco 1996 / Kollet-Maxwell 2006): two 800 m hillslopes (cross-slope
Sx=5%) + a central channel (valley slope Sy=2%) draining to an outlet at the channel end. Rain
3e-6 m/s = 0.2592 m/day for 90 min (0.0625 d), then recession. Single Manning n=0.015 (spike;
the canonical hillslope/channel contrast 0.015/0.15 needs a spatial-n model change). Near-
impermeable bed (low Ks) so this is the surface-runoff-dominated case: Q_eq = rain*area
= 0.2592 * 1.62e6 ~= 419,900 m^3/day (~4.86 m^3/s).

Run: PYTHONPATH=. python scratch/_tiltedv_spike.py [NX NY]
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

SCALE = float(os.environ.get("SCALE", "1.0"))   # 1.0 canonical (1.62 km), 0.1 field-scale (162 m)
LX, LY, H = 1620.0 * SCALE, 1000.0 * SCALE, 2.0  # m; channel center xc = LX/2
XC = LX / 2.0
SX, SY = 0.05, 0.02                  # hillslope cross-slope, channel/valley slope
N_MAN = float(os.environ.get("N_MAN", "0.015"))  # single Manning (0.15 = rough head-to-head variant)
RAIN = 0.2592                        # m/day (= 3e-6 m/s)
STORM, T_END = 0.0625, 0.125         # day (90 min storm, 90 min recession)
CHAN_HALF = 70.0 * SCALE             # outlet half-width around the channel center at y=LY

NX, NY, NZ = 24, 16, 3
if len(sys.argv) >= 3:
    NX, NY = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) >= 4:
    T_END = float(sys.argv[3])       # override total sim time (for a quick timed gate run)

# subsurface Ks (m/day): env KS overrides. 1e-3 = near-impermeable (surface-runoff, but tau_c=ell_c/Ks
# is huge -> stiff NCP); a moderate Ks (e.g. 0.1) is closer to IH-MIP2 AND less stiff. The probe.
KS = float(os.environ.get("KS", "1.0e-3"))
SOIL = VanGenuchten(theta_r=0.10, theta_s=0.40, alpha=2.0, n=2.0, Ks=KS)
AREA = LX * LY
Q_EQ = RAIN * AREA                   # equilibrium outlet discharge if all rain runs off [m^3/day]


def z_b(x):
    # valley floor falls toward the outlet at y=LY (Sy); hillslopes rise away from the channel (Sx)
    return SY * (LY - x[1]) + SX * np.abs(x[0] - XC)


def main():
    t0 = time.time()
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [LX, LY, H]], [NX, NY, NZ])
    prob = CoupledProblem(msh, SOIL, n_man=N_MAN)
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[2], d_value=0.0)   # unsaturated, no antecedent pond
    prob.set_topography(z_b)
    rain = prob.add_rain(0.0)
    # outlet = the WHOLE downslope y=LY edge (consistent with TopoSlopesY=-Sy everywhere, matching ParFlow's
    # whole-edge drainage). friction slope = Sy. (A channel-band-only outlet dams the hillslope toes and
    # under-drains -> 0.62 Q_eq; the whole-edge outlet reaches Q_eq, the apples-to-apples config.)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)
    setup_s = time.time() - t0
    print(f"[setup] mesh {NX}x{NY}x{NZ} on {LX}x{LY}x{H} m  Ks={KS:g}  ell_c={prob.ell_c:.3f}  "
          f"tau_c=ell_c/Ks={prob._tau_c:.1f}  Q_eq={Q_EQ:.0f} m^3/day (~{Q_EQ/86400:.2f} m^3/s)  "
          f"setup={setup_s:.1f}s", flush=True)

    n_out = 40
    out_times = np.linspace(0.0, T_END, n_out)

    def hyeto(t):
        return RAIN if t <= STORM + 1e-12 else 0.0

    # tunable adaptive controller (probe whether the dt-pin is genuine stiffness or controller-conservatism)
    GROW_AT = int(os.environ.get("GROW_AT", "3"))    # grow dt when iters <= this
    SHRINK_AT = int(os.environ.get("SHRINK_AT", "8"))  # shrink dt when iters >= this
    DT_MAX = float(os.environ.get("DT_MAX", "1e-3"))
    t_run = time.time()
    dt = 1e-5
    n_step = 0
    n_rej = 0
    cum_rain_exact = 0.0  # engine-exact rain booking (rain.value * h per ACCEPTED step)
    print(f"{'t[day]':>8} {'Q_out[m3/d]':>12} {'Q/Q_eq':>7} {'surf_W':>10} {'soil_dW':>10} {'iters':>5} {'dt':>9}")
    w0_soil = prob.soil_water()
    t_hist, q_hist, surf_hist, soil_hist = [0.0], [0.0], [float(prob.surface_water())], [0.0]  # hydrograph
    for k in range(1, n_out):
        t, t_target = out_times[k - 1], out_times[k]
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            rain.value = hyeto(t + h)
            converged, it = prob.step(h)
            n_step += 1
            if converged:
                t += h
                cum_rain_exact += h * float(rain.value) * AREA
                dt = min(dt * (1.5 if it <= GROW_AT else 0.7 if it >= SHRINK_AT else 1.0), DT_MAX)
            else:
                n_rej += 1
                dt *= 0.5
                if dt < 1e-11:
                    print(f"  !! dt collapse at t={t:.6g} -- CONDITIONING/SCALE FAILURE", flush=True)
                    return
        q = prob.outflow_rate()
        t_hist.append(t_target); q_hist.append(q)
        surf_hist.append(float(prob.surface_water())); soil_hist.append(float(prob.soil_water() - w0_soil))
        print(f"{t_target:8.4f} {q:12.0f} {q/Q_EQ:7.3f} {prob.surface_water():10.1f} "
              f"{prob.soil_water()-w0_soil:10.1f} {int(it):5d} {dt:9.2e}", flush=True)
    run_s = time.time() - t_run
    qf = prob.outflow_rate()
    # ENGINE-EXACT ledger (per accepted step, NOT the 40-point trapz the harness reconstructs --
    # which mis-samples the wave-arrival spike): every accepted step is residual-tested, so
    # cum_rain = cum_outflow + dW + ext_gap must close to ~roundoff (P0 acceptance hardening).
    dW_eng = (prob.soil_water() - w0_soil) + prob.surface_water()
    ext_gap_eng = cum_rain_exact - prob.cum_outflow - dW_eng
    print(f"  [mass] ENGINE ledger: cum_rain={cum_rain_exact:.1f}  cum_out={prob.cum_outflow:.1f}  "
          f"dW={dW_eng:.1f}  ext_gap={ext_gap_eng:+.3e} m^3 "
          f"({100.0*ext_gap_eng/max(cum_rain_exact,1e-9):+.2e}% of cum rain)", flush=True)
    print(f"  [mass] clip_mass_adjust={getattr(prob, 'clip_mass_adjust', float('nan')):.1f} m^3  "
          f"max_clip={getattr(prob, 'max_clip_seen', float('nan')):.4f} m  rejected_steps={n_rej}", flush=True)
    out = os.environ.get("OUT", f"scratch/tiltedv_inhouse_s{SCALE:g}.npz")
    np.savez(out, times=np.array(t_hist), q_out=np.array(q_hist), surface_water=np.array(surf_hist),
             soil_dw=np.array(soil_hist), Q_eq=Q_EQ, LX=LX, LY=LY, NX=NX, NY=NY, NZ=NZ, scale=SCALE,
             n_man=N_MAN, Ks=KS, storm=STORM, t_end=T_END, sx=SX, sy=SY, rain=RAIN, run_s=run_s, n_step=n_step,
             n_rej=n_rej, cum_rain_engine=cum_rain_exact, cum_out_engine=prob.cum_outflow,
             dw_engine=dW_eng, ext_gap_engine=ext_gap_eng, clip_mass_adjust=prob.clip_mass_adjust,
             max_clip=prob.max_clip_seen,
             # FULL run provenance (P0 lesson: the unrecorded controller knobs were exactly the
             # ambiguity that made the retracted 2026-06-10 run irreproducible) --
             grow_at=GROW_AT, shrink_at=SHRINK_AT, dt_max=DT_MAX, comm_size=MPI.COMM_WORLD.size)
    print(f"\n[done] {n_step} steps  run={run_s:.1f}s ({run_s/max(n_step,1)*1000:.0f} ms/step)  "
          f"final Q={qf:.0f} ({qf/Q_EQ:.2f} Q_eq)  peak approached Q_eq: {'YES' if qf/Q_EQ > 0.5 else 'NO/PARTIAL'}  "
          f"-> WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
