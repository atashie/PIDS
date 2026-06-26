"""SCRATCH DRIVER -- co-cycled sub-stepping (CoCycledCappedSplit) sanity + CONSERVATION GATE.

Tasks 2-3 of docs/plans/2026-06-25-iterated-capped-split-conservation-rearchitecture.md.

  smoke : Task 2 Step 2 -- a quick few-step b1_base run (short t_end), confirm it RUNS (no crash).
  gate  : Task 3 -- b1_base + b1_steep @ K=6, h_ref=2e-3, dt_max=0.004.  GATE: bal/rain <= ~5e-11 on
          BOTH.  Then falsification: outflow_leak_frac=0.10 on b1_base must break the ledger ~10%.

Run (WSL pids-fem, threads pinned) -- LIVE to a file (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_cocycled_gate.py gate > scratch/_cocycled_gate_out.txt 2>&1'
"""
from __future__ import annotations

import sys
import time

import numpy as np

from pids_forward.physics.constitutive import VanGenuchten
from scratch.seq_href_iterated import CoCycledCappedSplit, COMMON, CASES, LOAM, run
from scratch.seq_href_closure_study import make_box, _march
from scratch.seq_iterative_prototype import _top_area_ds


def smoke(K=4, dt_max=0.01, t_end=0.10):
    """Task 2 Step 2: a short run to confirm CoCycledCappedSplit RUNS + conserves over a few steps."""
    c = COMMON
    soil = VanGenuchten(**LOAM)
    case = CASES["b1_base"]
    msh = make_box(*c["MESH"], c["Lx"], c["Ly"], c["Lz"])
    prob = CoCycledCappedSplit(msh, soil, n_man=c["NMAN"], route_substeps=4, h_ref=2e-3, K=K)
    prob.set_initial_condition(lambda x: c["PSI_I"] + 0.0 * x[0])
    prob.set_topography(lambda x: case["S0"] * x[1])
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=case["S0"])
    top_area = _top_area_ds(msh, c["Lz"])
    R_in = c["RAIN"] * top_area * c["STORM"]
    t0 = time.perf_counter()
    ns, coll, tend, _ = _march(prob, rain_c, storm_dur=c["STORM"], storm_rain=c["RAIN"],
                               t_end=t_end, dt0=dt_max / 4.0, dt_max=dt_max, max_steps=120)
    bal = abs(prob.balance()) / prob.cum_rain if prob.cum_rain > 0 else float("nan")
    print("#" * 80)
    print(f"SMOKE CoCycledCappedSplit K={K}: ns={ns} t={tend:.3f}/{t_end} coll={coll} "
          f"routed/R={prob.cum_outflow / R_in:.4f} bal/rain={bal:.2e} "
          f"[{time.perf_counter()-t0:.0f}s]  => {'RUNS' if (not coll and ns > 0) else 'FAIL'}")
    print("#" * 80, flush=True)


def gate(K=6, h_ref=2e-3, dt_max=0.004):
    """Task 3: conservation gate on b1_base + b1_steep, then falsification on b1_base."""
    print("#" * 92)
    print(f"CONSERVATION GATE -- CoCycledCappedSplit K={K} h_ref={h_ref*1000:.0f}mm dt_max={dt_max}")
    print(f"  GATE: bal/rain <= ~5e-11 on BOTH (matched-quadrature exact, like route-first B).")
    print("#" * 92, flush=True)
    results = {}
    for name in ("b1_base", "b1_steep"):
        r = run(CoCycledCappedSplit, CASES[name], h_ref, dt_max=dt_max, K=K)
        results[name] = r
        gp = r["routed"] - CASES[name]["target"]
        verdict = "PASS" if (r["bal"] <= 5e-11 and r["ok"]) else "FAIL"
        print(f"\n  {name}: bal/rain={r['bal']:.2e}  [{verdict}]   "
              f"routed/R={r['routed']:.4f} vs monolith {CASES[name]['target']:.4f} "
              f"(gap {gp*100:+.1f}pp, partition is Task 4 -- not gated here)")
        print(f"     ok={r['ok']} ns={r['ns']} wall={r['wall']:.0f}s "
              f"newton/sub-step avg={r['pic_avg']:.1f}/max={r['pic_max']}", flush=True)

    # falsification: 10% outflow mis-book on b1_base must break the ledger ~10% (|ratio|~1).
    print("\n  --- falsification (10% outflow mis-book) on b1_base ---", flush=True)
    rf = run(CoCycledCappedSplit, CASES["b1_base"], h_ref, dt_max=dt_max, K=K, leak=0.10)
    # |bal|/rain should be ~ leak * outflow-share (>> 1e-11), NOT clean.
    fired = rf["bal"] > 1e-3
    print(f"  leak=10%: |bal|/rain={rf['bal']:.2e}  => detector {'FIRES' if fired else 'DID NOT FIRE'} "
          f"(clean was ~1e-11; a real break ~0.05-0.10)", flush=True)

    base_ok = results["b1_base"]["bal"] <= 5e-11
    steep_ok = results["b1_steep"]["bal"] <= 5e-11
    print("\n" + "#" * 92)
    print(f"GATE VERDICT: b1_base {'PASS' if base_ok else 'FAIL'} | "
          f"b1_steep {'PASS' if steep_ok else 'FAIL'} | falsification {'PASS' if fired else 'FAIL'}")
    print("  => co-cycled conservation is EXACT (no reconstruction)." if (base_ok and steep_ok and fired)
          else "  => investigate the failing gate before partition (Task 4).")
    print("#" * 92, flush=True)


def one(case_name, K=6, h_ref=2e-3, dt_max=0.004, leak=0.0):
    """Run a SINGLE case (for concurrent single-process launches; aggregate the files afterward)."""
    r = run(CoCycledCappedSplit, CASES[case_name], h_ref, dt_max=dt_max, K=K, leak=leak)
    gp = r["routed"] - CASES[case_name]["target"]
    tag = f"{case_name}{'+leak0.10' if leak else ''}"
    print("#" * 92)
    print(f"ONE {tag}  K={K} h_ref={h_ref*1000:.0f}mm dt_max={dt_max}")
    print(f"  bal/rain = {r['bal']:.3e}   routed/R = {r['routed']:.4f} vs monolith "
          f"{CASES[case_name]['target']:.4f} (gap {gp*100:+.1f}pp)")
    print(f"  ok={r['ok']} ns={r['ns']} wall={r['wall']:.0f}s newton/sub-step "
          f"avg={r['pic_avg']:.1f}/max={r['pic_max']}")
    print("#" * 92, flush=True)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if mode == "smoke":
        smoke()
    elif mode == "one":
        # python seq_cocycled_gate.py one <case_name> [K] [leak]
        cname = sys.argv[2]
        Kc = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        leakc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        one(cname, K=Kc, leak=leakc)
    else:
        gate()
