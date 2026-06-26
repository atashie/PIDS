# Codex critical review — the 1-D sorptivity benchmark (2026-06-26)

Skeptical Codex review (gpt-5-codex, effort=high) of `scratch/seq_sorptivity_benchmark.py` + its loam
results, commissioned to stress-test the §18 conclusion before it drives the overland-scheme decision.
Prompt: `scratch/_codex_sorptivity_review_prompt.txt`. **Verdict: directionally useful, but NOT yet strong
enough for the full claim — three fair hits to fix.**

## What the benchmark DOES support (Codex)
- The coarse `switch coarse8` early overshoot (ratio 2.10 @ t=0.002) is a **real top-cell resolution
  artifact** (125 mm cell can't resolve the wetting front), cured by refinement — `switch 2mm-skin ≈
  refined` (0.99–1.01). ⟹ "don't put ψ=0 on a 125 mm top cell and expect correct sorptivity."
- A **resolved ponded-Dirichlet** boundary gives much larger early uptake than the **coarse `q_pot` film
  surrogate**. Directionally: ψ=0 + surface refinement converges; the coarse film throttles.

## What it does NOT yet support — the fair hits (Codex)
1. **★ LOAD-BEARING: the benchmarked `q_pot` is NOT the real monolith law.** Benchmark applies it as a
   HARD NEUMANN flux from the **mean** top head with a **fixed `h_ref=1mm`** and coarse `ell_c=62.5mm`
   (`seq_sorptivity_benchmark.py:59-79,108-116`). The real monolith co-solves λ with `kirchhoff(ψ, ACTUAL
   d)/ell_c` (`coupling.py:223-231`). As ψ→0 the benchmarked law → `Ks·h_ref/ell_c ≈ 1.6%·Ks` — a film
   throttle, so under-capture is BUILT IN. ⟹ **"q_pot captures only 57%" is true ONLY for this coarse
   fixed-`h_ref` surrogate, NOT a clean statement about the monolith.** `tests/test_coupling_1d.py:120-141`
   shows the real coupled model → `add_ponding_bc` as `ell_c→1mm`.
2. **`nz=240` is ASSERTED truth, not demonstrated.** No mesh/time convergence sweep; the metric is the
   LUMPED `∫θ` (internal-comparison-valid, the quantity the solver conserves, but degrades order vs a
   consistent `∫θ` postprocess — cf. the older `scratch/m3_sorptivity_benchmark.py` used consistent
   quadrature). Times too coarse to robustly establish the √t limb.
3. **The clay setup ≠ "dry clay".** Same `psi_i=−0.4 m` across soils ≠ same antecedent dryness: clay's
   retention is flat near saturation (`α=0.8, n=1.09`), so `dtheta=0.006` — barely dry, **storage-limited**,
   not sorptivity-limited. The lead's "dry clay → prompt runoff" intuition needs a soil-specific dry `psi_i`
   (e.g. matched effective saturation S_e), else the clay result is mis-specified.
- Green-Ampt: `psi_f=|kirchhoff(psi_i,0)|/Ks` is a reasonable EFFECTIVE mapping, not canonical GA suction;
  a loose secondary cross-check (GA ~13% high by t=0.08), NOT ground truth.

## Codex's best single additional test (the fix)
Run the ACTUAL 1-D `CoupledProblem` (the real monolith) and **sweep `ell_c` from the coarse half-cell down
to ~1 mm** against a converged ponded-Dirichlet / `add_ponding_bc` curve, with a CONSISTENT `∫θ dz`
postprocess + a mesh/time convergence check. That separates CLOSURE error from MESH error and directly says
whether the "57% gap" is a coarse-film artifact or a true closure failure.

## Disposition (mine)
The §18 DIRECTION holds (resolved ψ=0 sorptive uptake ≫ coarse film; the switching BC is the right closure
at resolution), and the loam `2mm-skin ≈ refined` (0.99–1.01) is solid evidence the error is
top-resolution-driven. BUT I OVER-STATED the q_pot comparison: the benchmark's q_pot is a coarse
fixed-`h_ref` surrogate, so "monolith captures 57%" is RETRACTED as a monolith claim — it is a statement
about the coarse film surrogate only. FIXES before any decision: (a) benchmark the REAL `CoupledProblem`
with an `ell_c` sweep + consistent `∫θ`; (b) demonstrate nz convergence (not assert nz=240); (c) re-spec
clay with a soil-matched dry `psi_i`. Only then is "which closure captures sorptivity" decision-grade.
