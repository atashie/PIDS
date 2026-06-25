"""SCRATCH STUDY 2 -- h_ref CLOSURE at ADEQUATE dt resolution (the corrected run).

The first closure study (seq_href_closure_study) used too-coarse dt (dt_max=0.02, ctrl_high=12) -> the
0.08-day storm onset was under-resolved -> garbage/non-monotone partitions. The dt-check
(seq_href_dt_check) proved B is dt-CONVERGENT by dt_max~0.004 and that b1_base h_ref*~2mm (2mm->0.546 vs
monolith 0.547). This run finds h_ref* for the GENERALITY discriminators (steep slope, coarse mesh) at
dt_max=0.004, reusing the (dt-robust, cross-checked) galerkin-monolith targets from the first run.

Question: is h_ref* a universal CONSTANT (~2mm), DERIVABLE from a physical scale (Manning sheet d_M, or
the monolith sheet d_mono), or MESH-dependent (moves with ell_c)?

Run (WSL pids-fem) -- guarded, live (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_href_closure2.py'
"""
from __future__ import annotations

import time

import numpy as np
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from scratch.seq_href_cap_spike import HrefCappedPondInPsi
from scratch.seq_iterative_prototype import _top_area_ds
from scratch.seq_href_closure_study import make_box, _march, manning_normal_depth

LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
DT_MAX = 0.004      # the dt-check's adequately-resolved level (b1: 116-171 steps)

# (target routed/R, d_mono peak sheet [m], ell_c [m]) from the first closure study's monolith runs
# (galerkin; b1_base cross-checks the cached upwind 0.5466 -> the monolith is dt/scheme-robust).
CASES = {
    "b1_base":   dict(S0=0.03, mesh=(30, 20, 8), RAIN=0.5, target=0.5470, d_mono=0.436e-3),
    "b1_steep":  dict(S0=0.10, mesh=(30, 20, 8), RAIN=0.5, target=0.5508, d_mono=0.304e-3),
    "b1_coarse": dict(S0=0.03, mesh=(20, 14, 5), RAIN=0.5, target=0.6145, d_mono=0.474e-3),
}
# b1_base is already pinned by the dt-check (1mm->0.520, 2mm->0.546, 4mm->0.664 at dt_max=0.004).
B1_BASE_SWEEP = [(0.001, 0.5203), (0.002, 0.5457), (0.004, 0.6637)]
H_GRID = {"b1_steep": [0.001, 0.002, 0.004], "b1_coarse": [0.001, 0.002, 0.004]}
Lx, Ly, Lz, PSI_I, STORM, TEND, NMAN = 8.0, 5.0, 1.0, -0.4, 0.08, 0.45, 0.05


def run_B(case, h_ref):
    soil = VanGenuchten(**LOAM)
    msh = make_box(*case["mesh"], Lx, Ly, Lz)
    prob = HrefCappedPondInPsi(msh, soil, n_man=NMAN, route_substeps=4, h_ref=h_ref)
    prob.set_initial_condition(lambda x: PSI_I + 0.0 * x[0])
    prob.set_topography(lambda x: case["S0"] * x[1])
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=case["S0"])
    top_area = _top_area_ds(msh, Lz)
    R_in = case["RAIN"] * top_area * STORM
    t0 = time.perf_counter()
    ns, coll, tend, _ = _march(prob, rain_c, storm_dur=STORM, storm_rain=case["RAIN"], t_end=TEND,
                               dt0=DT_MAX / 4.0, dt_max=DT_MAX, max_steps=600)
    ok = (not coll) and tend >= TEND - 1e-9
    return dict(routed=prob.cum_outflow / R_in, bal=abs(prob.balance()) / prob.cum_rain,
                ns=ns, ok=ok, wall=time.perf_counter() - t0)


def interp_hstar(rows, target):
    hs = np.array([r[0] for r in rows]); rr = np.array([r[1] for r in rows])
    for i in range(len(hs) - 1):
        if (rr[i] - target) * (rr[i + 1] - target) <= 0 and rr[i + 1] != rr[i]:
            w = (target - rr[i]) / (rr[i + 1] - rr[i])
            return hs[i] + w * (hs[i + 1] - hs[i])
    return np.nan


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("#" * 90)
    print(f"h_ref CLOSURE at dt_max={DT_MAX} (adequately resolved)")
    print("#" * 90, flush=True)
    sweeps = {"b1_base": B1_BASE_SWEEP}
    print(f"\nb1_base [from dt-check]: " +
          " ".join(f"{h*1000:.0f}mm->{r:.4f}" for h, r in B1_BASE_SWEEP), flush=True)
    for name in ("b1_steep", "b1_coarse"):
        case = dict(CASES[name]); rows = []
        print(f"\n=== {name}: S={case['S0']} mesh={case['mesh']} target={case['target']:.4f} ===",
              flush=True)
        for h in H_GRID[name]:
            r = run_B(case, h)
            rows.append((h, r["routed"]))
            print(f"    B h_ref={h*1000:.1f}mm -> routed/R={r['routed']:.4f} bal={r['bal']:.1e} "
                  f"ok={r['ok']} [{r['ns']} steps {r['wall']:.0f}s]", flush=True)
        sweeps[name] = rows

    print("\n" + "#" * 90)
    print("CLOSURE TABLE (depths in mm)")
    print(f"{'case':>10} | {'target':>7} | {'h_ref*':>7} | {'d_M':>6} | {'d_mono':>6} | {'ell_c':>6} "
          f"| {'h*/d_M':>6} | {'h*/dmono':>8} | {'h*':>5}")
    for name in ("b1_base", "b1_steep", "b1_coarse"):
        case = CASES[name]
        ell_c = 0.5 * Lz / case["mesh"][2]
        d_M = manning_normal_depth(case["RAIN"], LOAM["Ks"], NMAN, case["S0"], Ly)
        hstar = interp_hstar(sweeps[name], case["target"])
        print(f"{name:>10} | {case['target']:>7.4f} | {hstar*1000:>7.3f} | {d_M*1000:>6.3f} | "
              f"{case['d_mono']*1000:>6.3f} | {ell_c*1000:>6.1f} | "
              f"{hstar/d_M if d_M>0 else np.nan:>6.2f} | "
              f"{hstar/case['d_mono'] if case['d_mono']>0 else np.nan:>8.2f} | {hstar*1000:>5.2f}")
    print("\nREAD: h_ref* ~constant => universal constant; h*/d_M or h*/d_mono ~constant => derivable;")
    print("      h_ref*(base) vs h_ref*(coarse) differ => MESH-dependent (ell_c in the rule).")
    print("#" * 90, flush=True)


if __name__ == "__main__":
    main()
