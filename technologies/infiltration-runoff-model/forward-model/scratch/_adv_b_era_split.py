"""Adversarial review, attack (b): era-split forensics on the deployment gate legs.

Pure-numpy on the committed fixtures -- NO FEM. Quantifies, per leg and per n:
  - the clock-era share of cumulative I, of time, of samples, of the relL2 DENOMINATOR sum(ref^2);
  - WHERE the pure offline clock's failure mass (the relL2 numerator) accumulates: clock era vs WI era;
  - the clock-vs-resolved-ref agreement WITHIN the clock era (the defense's "<1.5%" claim);
  - RefB-40: handover time vs the pulse window (is the pulse response carried by the WI era?);
  - control leg: era weight split + the implied WI-era bias needed to explain v2's 6.0% at n=12.
"""
import numpy as np
import sys

sys.path.insert(0, ".")
from pids_forward.physics.sorptive_closure import sorptive_clock, F_cylindrical, rel_l2

R_W = 0.05
R = 40 * R_W
L = float(np.sqrt(np.pi * (R ** 2 - R_W ** 2)))

a = np.load("tests/data/m4_phase4_refA_disperse.npz")
b = np.load("tests/data/m4_phase4_refB40_disperse.npz")

S, dth = float(a["LOAM_S"]), float(a["LOAM_dtheta"])


def split(tag, t, I, ns, t_marks=()):
    clk = sorptive_clock(t, S, dth, R_W, F_cylindrical)
    err2 = (clk - I) ** 2
    den = float(np.sum(I ** 2))
    print(f"\n=== {tag}: {len(t)} samples, t=[{t[0]:.3g},{t[-1]:.3g}] d, "
          f"I_end={I[-1]:.4f} m, clock relL2={rel_l2(clk, I):.1%}")
    # where does the CLOCK's failure mass live in time?  cumulative err^2 quantiles
    c = np.cumsum(err2) / np.sum(err2)
    for q in (0.05, 0.25, 0.50, 0.95):
        i = int(np.searchsorted(c, q))
        print(f"    clock err^2 {q:.0%} quantile at t={t[i]:.2f} d, I_ref={I[i]:.3f} m "
              f"({I[i]/I[-1]:.0%} of I_end)")
    for n in ns:
        h = L / n
        I_fill = dth * ((2 * h) ** 2 - R_W ** 2) / (2.0 * R_W)
        m = I < I_fill                      # clock era (by the ref curve; v2 tracks ref to ~1.00)
        t_h = float(np.interp(I_fill, I, t)) if I_fill < I[-1] else np.inf
        nclk = int(m.sum())
        fr_den = float(np.sum(I[m] ** 2) / den)
        fr_err = float(np.sum(err2[m]) / np.sum(err2))
        # clock-vs-ref agreement WITHIN the clock era (the defense's "<1.5%")
        if nclk:
            rel_within = float(np.sqrt(np.sum(err2[m]) / np.sum(I[m] ** 2)))
            max_pt = float(np.max(np.abs(clk[m] - I[m]) / I[m]))
        else:
            rel_within, max_pt = 0.0, 0.0
        # leg-metric contribution of clock-era samples: sqrt(sum_clk err^2 / den)
        leg_contrib = float(np.sqrt(np.sum(err2[m]) / den))
        extra = "  ".join(f"{nm}={'PRE' if t_h < tv else 'IN-CLOCK-ERA' if t_h>=tv else '?'}"
                          for nm, tv in t_marks)
        print(f"  n={n:2d}: I_fill={I_fill:.3f} m ({I_fill/I[-1]:.1%} of I_end) "
              f"handover t≈{t_h:.2f} d ({t_h/t[-1]:.1%} of window) samples {nclk}/{len(t)}")
        print(f"        clock-era share: denominator {fr_den:.1%}, clock-failure-mass {fr_err:.2%} "
              f"(leg-metric contribution {leg_contrib:.2%})")
        print(f"        clock-vs-ref WITHIN clock era: relL2 {rel_within:.2%}, max pointwise {max_pt:.2%}")
        for nm, tv in t_marks:
            print(f"        {nm} (t={tv} d) is {'AFTER handover -> WI era' if tv > t_h else 'inside the CLOCK era'}")
    return clk


# ---- Ref A-40 ----
t, I = a["LOAM_R40_t"], a["LOAM_R40_I"]
imax = float(a["LOAM_R40_Imax"])
print(f"LOAM R40: I_max={imax:.3f} m, S={S:.4f}, dth={dth:.4f}, L={L:.4f} m")
print(f"depletion: I_end/I_max = {I[-1]/imax:.3f}")
clk = split("RefA-40 (full leg)", t, I, (6, 8, 10, 12))

# the bend landmarks
for frac in (0.5, 0.9, 0.99):
    print(f"  I_ref reaches {frac:.0%} I_end at t={float(np.interp(frac*I[-1], I, t)):.2f} d")

# ---- Ref B-40 ----
tb, Ib = b["LOAM_t"], b["LOAM_I"]
t1, t2 = float(b["LOAM_t_pulse"][0]), float(b["LOAM_t_pulse"][1])
split("RefB-40 (history leg)", tb, Ib, (8, 12), t_marks=(("pulse_start", t1), ("pulse_end", t2)))
print(f"  pulse window = [{t1}, {t2}] d")

# ---- control leg (truncated to I < 0.3 I_max) ----
K = int(np.searchsorted(I, 0.3 * imax))
tc, Ic = t[:K], I[:K]
clkc = sorptive_clock(tc, S, dth, R_W, F_cylindrical)
denc = float(np.sum(Ic ** 2))
print(f"\n=== CONTROL leg: {K} samples, t_end={tc[-1]:.2f} d, clock relL2={rel_l2(clkc, Ic):.2%}")
for n in (8, 12):
    h = L / n
    I_fill = dth * ((2 * h) ** 2 - R_W ** 2) / (2.0 * R_W)
    m = Ic < I_fill
    fr_den = float(np.sum(Ic[m] ** 2) / denc)
    # implied uniform relative bias e in the WI era to produce the measured v2 relL2,
    # assuming the clock era is exact:  v2relL2 = e * sqrt(1 - fr_den)
    v2 = {8: 0.021, 12: 0.060}[n]
    e_impl = v2 / np.sqrt(max(1.0 - fr_den, 1e-12))
    print(f"  n={n:2d}: clock-era denominator share {fr_den:.1%} (WI era {1-fr_den:.1%}); "
          f"v2 measured {v2:.1%} -> implied uniform WI-era bias ≈ {e_impl:.1%}")
print("  (same arithmetic on the FULL RefA-40 leg:)")
for n, v2 in ((6, 0.022), (8, 0.051), (10, 0.056), (12, 0.057)):
    h = L / n
    I_fill = dth * ((2 * h) ** 2 - R_W ** 2) / (2.0 * R_W)
    m = I < I_fill
    fr_den = float(np.sum(I[m] ** 2) / float(np.sum(I ** 2)))
    e_impl = v2 / np.sqrt(max(1.0 - fr_den, 1e-12))
    print(f"  n={n:2d}: clock-era den share {fr_den:.2%}; v2 {v2:.1%} -> implied WI-era bias {e_impl:.1%}")

# ---- what would a scheme score that is EXACT in the WI era and pure-clock in the clock era? ----
print("\n=== hybrid bound: pure clock INSIDE the clock era only, exact after (per n) ===")
for n in (6, 8, 10, 12):
    h = L / n
    I_fill = dth * ((2 * h) ** 2 - R_W ** 2) / (2.0 * R_W)
    m = I < I_fill
    hyb = np.where(m, clk, I)
    print(f"  n={n:2d}: relL2(clock-era-only clock) = {rel_l2(hyb, I):.2%}")
