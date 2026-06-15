# M4 Phase-4 — item (B): the offline PSS drain law's "+2–4% end bias" ATTRIBUTED

**Date:** 2026-06-15 · **Scope:** the drain-closure follow-up open item (B) from
`pids-m4-phase4-followups-kickoff`. **Status:** RESOLVED. **Binding wording lives here.**

## Question

The offline pseudo-steady-state (PSS) drain closure
`dI/dt = [Φ(ψ_bulk(I)) − Φ(ψ_wall)] / (r_w·(ln(R/r_w) − 3/4))`
(`scratch/m4_phase4_drain_desorptivity.py::pss_drain`; `ψ_bulk = retention⁻¹(θ_mean)`,
`θ_mean` from the closed-box water balance) was recorded as over-predicting cumulative `I` at
end-of-curve by **+2–4%** vs the resolved 1-D closed-drain FEM references (recorded ends:
refD40 1.020, SAND-R40 1.031, LOAM-R20 1.042, SILT-R40 1.010). The early-similarity composite was
already REFUTED (`ea5e352`). Two candidates remained (kickoff): **(1)** the Dietz −3/4 is a *steady,
uniform-depletion* linearization applied to a *nonlinear transient* Kirchhoff depletion (genuine
model-form over-prediction); **(2)** the references' own backward-Euler (BE) temporal under-count
(the law only *appears* to over-predict).

## Method (probes; both committed under `scratch/_b_drain_*.py`)

1. **Free / law-side (`_b_drain_endbias.py`).** Re-integrated the SAME model form to dt→0 (stiff
   Radau, rtol 1e-11) vs the production forward-Euler (FE, 400 substeps/sample); the effective
   geometric constant `c_eff` in `(ln(R/r_w) − c)` that lands the law's end on the ref; and an
   a-priori mechanism check **M1** = the nonlinear volume-average gap `Φ(θ_mean)` vs `⟨Φ⟩`.
2. **FEM / reference-side (`_b_drain_refbe.py`), the pre-registered discriminator.** Re-ran each
   1-D closed-drain ref (same `m4_phase4_refAB_drain._setup`) with the BE step hard-capped at
   `dt_max ∈ {W/256, W/1024, W/4096}` (W = window), Richardson-extrapolated `I_ref` to dt→0
   (first-order BE), and computed `law_end / I_ref(dt→0)`.
   **Pre-registered:** (2) ref-BE ⇒ `I_ref` rises with finer dt, `law/I_true` collapses to ~1.00
   (or ~0.99 per M1); (1) model-form ⇒ `I_ref` ~unchanged, `law/I_true` stays ~1.03–1.04.

## Findings

**Law's own FE integration is innocent** — `law_FE/law_dt→0 = +0.01%` on every leg (the 400-substep
FE is bit-exact). A third, un-enumerated candidate (law time-integration) is ruled out.

**M1 (static volume-average gap) is small and the WRONG sign** — predicts the law should *under*-
predict by −0.84…−1.29% (Φ is convex in θ ⇒ `Φ(θ_mean) ≤ ⟨Φ⟩`, rigorous regardless of profile).
So the static volume-averaging is not the over-bias mechanism; it works slightly in the law's favor.

**dt-refinement of the refs (the decisive result):** the committed refs are **~4% LOW** from
first-order BE; once corrected the recorded "+2–4% over" flips to a small **under**-prediction:

| leg | recorded (law/committed) | **ref-BE share** (committed low by) | **true model-form** (law/I_dt→0) | M1 prediction |
|---|---|---|---|---|
| refD40 LOAM R40 | +2.00% | **+4.08%** | **−1.99%** | −1.05% |
| SAND R40 | +3.11% | **+4.44%** | **−1.27%** | −0.84% |
| LOAM R20 (deep) | +4.21% | **+4.82%** | **−0.58%** | −1.05% |
| SILT R40 | +1.02% | **+3.73%** | **−2.62%** | −1.29% |

Three independent consistency checks make this airtight:
- **First-order BE confirmed** — level-to-level `I_end` changes shrink ×3.76–3.87 (≈4 for dt/4).
- **`recorded = (1+ref-BE)·(1+model-form)` exactly** (e.g. LOAM-R20: 1.0482 × 0.9942 = 1.0420).
- **M1 vindicated** — the dt-corrected model-form (−0.6…−2.6%) matches the independent a-priori M1
  Jensen prediction (−0.8…−1.3%) in sign and magnitude. M1 only looked "wrong" because it had been
  measured against BE-biased refs.

## Conclusion (binding)

**The recorded "+2–4% offline PSS drain end bias" is a REFERENCE ARTIFACT — the committed 1-D
closed-drain FEM refs carry a ~3.7–4.8% first-order backward-Euler temporal under-count — NOT a law
model-form error.** Candidate (2) wins decisively; candidate (1) (Dietz −3/4 over-prediction) is
**refuted**. The Dietz −3/4 is sound; the offline PSS law is accurate and in fact slightly
**conservative** (under by ~0.6–2.6% vs the dt→0 physics), as predicted a-priori by the convex-Φ
volume-average (Jensen) gap. **The law stays pure PSS, unchanged, no knob.**

Corollary: the refAB docstring's "~1.3% BE temporal accuracy" UNDER-states the true temporal error
on these legs (~4%). That check (`m4_phase4_refAB_drain.py`, LOAM R3) only inserted geometric
sample midpoints — a weak densification that barely refines the *adaptive* dt — so it measured a
small relL2 between two coarse-dt curves, not the dt→0 error.

## Caveats / scope

- The dt-refinement holds the mesh fixed at `gen.CELL`; the model-form numbers carry an additional
  ~<1% spatial uncertainty (committed mesh-conv was <1%, cell/2 on R3). The headline (ref-BE ~4%
  dominant; law accurate-to-conservative) is robust to it; the exact model-form sub-percent is not.
- All gate tests pass either way: the law scores 3.3–4.3% relL2 against the (slightly-low) committed
  refs, well within the 10% bar. **No production behavior change is implied.** The embedded
  production drain scheme reuses the same PSS form, so its recorded end ratios vs these refs inherit
  the same ~4% reference artifact (gate margins unaffected).

## Resolution: drain refs REGENERATED at converged dt (Arik-directed, 2026-06-15)

Arik chose the thorough path ("regenerate all drain refs, full consistency"). Mechanism: a `dt_max`
cap added to the shared adaptive BE stepper (`m4_phase1b_disperse_reference._solve_to`, default ∞ →
disperse refs unchanged), with the drain generators capping every step at **window/2048**
(`m4_phase4_refAB_drain.DT_MAX_DIV`; the dt-ladder showed window/1024 already <0.05% of the dt→0
limit). refD40-C explicitly shares refD40's cap so its machine-zero pre-window identity holds
(measured dev = 0.0e+00 post-regen).

**Regenerated (committed) fixtures:** `scratch/m4_phase4_refD40_drain.npz`, `..._drain_fresh_refs.npz`
(SAND-R40, LOAM-R20), `..._silt_drain_ref.npz`, `..._refD40C_drain.npz`, `..._refD40B_drain.npz`,
and `tests/data/m4_phase4_refAB_drain.npz` (R3/R5/R10 + RefB). `..._drain_fresh_predictions.npz`
unchanged (pre-registration; the offline-law prediction is dt-independent). refD40 end 1.41283 →
1.4702 (+4.06%, = the dt-ladder Richardson limit). refAB convergence asserts: mesh cell/2 0.001%,
dt-dense **0.019%** (was 1.26% pre-cap — now temporally converged).

**Corrected offline-law pred/ref ends** (was 1.020/1.031/1.042/1.010): now **0.987 (SAND), 0.994
(LOAM-R20), 0.974 (SILT), 0.980 (refD40)** — the law slightly conservative, matching the Part-B
`law/I_true` and the M1 Jensen prediction.

**Re-validation (all green):** `test_coupled_gate_refs.py` 10/10 (kill-map margins widened as
predicted); `test_wi_exchange.py` 26/26 (all 5 drain gates + disperse + units). New embedded
n=8 emb/ref: refD40 4.4%/0.979, SAND 3.6%/0.986, LOAM-R20 2.7%/0.992, SILT 5.5%/0.973, refD40-C
2.9%/0.983 — the recorded "+2–4% over" flipped to the small Jensen under; gates `≤0.10`/twin `≥0.20`
unchanged.

**Docstrings/wording corrected:** `wi_exchange.py` drain section, `viz/run_phase4_battery_data.py`
battery annotation, `m4_phase4_refAB_drain.py` ("1.3% BE" → converged), `tests/test_wi_exchange.py`
module note. Probes retained: `scratch/_b_drain_endbias.py`, `_b_drain_refbe.py`,
`_b_drain_embedded_numbers.py`.
