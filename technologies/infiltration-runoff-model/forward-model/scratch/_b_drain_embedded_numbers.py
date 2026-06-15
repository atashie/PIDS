"""ITEM (B): collect the embedded drain-gate relL2 + end ratios against the REGENERATED (converged)
refs, for the docstring refresh. Mirrors the test_wi_exchange.py drain gate setup; prints only."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pids_forward.physics.wi_exchange import WellIndexExchange
from pids_forward.physics.sorptive_closure import rel_l2, R_W_DEFAULT as RW
from scratch.m4_phase4_embedded_harness import run_embedded
HERE = pathlib.Path(__file__).parent
N = 8


def line(name, I_emb, I_ref):
    print(f"{name:18s} relL2={rel_l2(I_emb, I_ref)*100:5.2f}%  end emb/ref={I_emb[-1]/I_ref[-1]:.3f}",
          flush=True)


d = np.load(HERE / "m4_phase4_refD40_drain.npz")
o = run_embedded(WellIndexExchange(direction="drain"), "LOAM", 40 * RW, N, d["LOAM_t"], direction="drain")
line("refD40 LOAM R40", o["I"], d["LOAM_I"])

fr = np.load(HERE / "m4_phase4_drain_fresh_refs.npz")
o = run_embedded(WellIndexExchange(direction="drain"), "SAND", 40 * RW, N, fr["SAND_R40_t"], direction="drain")
line("SAND R40", o["I"], fr["SAND_R40_I"])
o = run_embedded(WellIndexExchange(direction="drain"), "LOAM", 20 * RW, N, fr["LOAM_R20_t"], direction="drain")
line("LOAM R20 (deep)", o["I"], fr["LOAM_R20_I"])

s = np.load(HERE / "m4_phase4_silt_drain_ref.npz")
o = run_embedded(WellIndexExchange(direction="drain"), "SILT", 40 * RW, N, s["SILT_R40_t"], direction="drain")
line("SILT R40", o["I"], s["SILT_R40_I"])

c = np.load(HERE / "m4_phase4_refD40C_drain.npz")
t1, t2 = (float(x) for x in c["LOAM_t_src"])
v_len = float(c["LOAM_V_per_wall_area"]) * 2.0 * np.pi * RW
o = run_embedded(WellIndexExchange(direction="drain"), "LOAM", 40 * RW, N, c["LOAM_t"], direction="drain",
                 pulse=(t1, t2, v_len), pulse_band=tuple(float(x) for x in c["LOAM_band"]))
line("refD40-C recharge", o["I"], c["LOAM_I"])
print("DONE", flush=True)
