# Codex review — the decisive 3-D top-refinement verdict §22 (2026-06-27)

5th Codex review (gpt-5-codex, high) of `seq_partition_topref.py` + the §22 "0.547 is a coarse-resolution
artifact / the original bug had the comparison backwards" verdict — the most consequential claim of the
investigation. Prompt: `scratch/_codex_3d_review_prompt.txt`. **Verdict: directionally important + NOT
unsound, but STILL OVER-CLAIMED on three points.**

## What Codex AFFIRMS (solid)
- **The collapse is REAL.** routed/R 0.547→0.266 as `p` refines, fixed lateral mesh/forcing/soil/scheme;
  `ell_c` auto-locks to top-cell half-height (`coupling.py:191-209`) and `q_pot=kirchhoff/ell_c` directly
  changes (`:223-230`). Monotone, both slopes move together, clean solves (41-51 steps, max_steps not
  binding, limiter negligible). For `p=2.5` the top 4 cells span ~17.7 cm so the ~17 cm wetting front is
  still mostly INSIDE the refined band → **not dismissible as a pure deep-cell artifact.** Matches the
  independent 1-D UNIFORM mesh-conv (§21).
- **Enough to REJECT 0.547 as a mesh-objective validation target** and to reopen the sequential-vs-monolith
  framing.

## What Codex says is OVER-CLAIMED (must soften)
1. **"converges to ~0.27" — NOT proven.** `p=2.0→2.5` still moved 0.287→0.266 (−2.1 pp), trend still
   downward at the finest rung. Defensible: **"runoff at least HALVED vs 0.547, still decreasing, NOT yet
   converged"** — the evidence supports "not 0.55" far more strongly than "definitely 0.27".
2. **The grading is NOT literally "refine only the top".** `make_graded_box` warps ALL z-levels (top-biased
   redistribution of 8 layers; deep cells COARSEN). The test does NOT separate top-interface resolution
   from bulk-Richards resolution. **Needs a UNIFORM-nz cross-check.**
3. **"ParFlow was under-resolved" stated as FACT — not warranted** without re-running ParFlow at fine dz.
   Repo has a plausible mechanism (coarse top cell buffers rain, `build_comparison_coupled_3d.py:121-127`,
   `benchmarks/README.md:346`) but B5 only proves coarse-grid code-to-code agreement. Defensible: **"B5 no
   longer validates 0.55 as continuum truth; it validates agreement at the shared coarse vertical resolution."**
- Minor: **`clip≤1e-16` is NOT a conservation proof** (it's only `abs(clip_mass_adjust)`, not global
  closure); the ~0.4% `1−(infil+routed)` is plausibly final surface storage (~0.17 mm mean ≈ peak mean
  sheet 0.337) and/or lumped-vs-deg8 postprocess mismatch — don't present as proven-tight. **"peak sheet"
  → "peak MEAN sheet"** (`surface_water/top_area`, not max local depth). Slope-insensitivity is SUPPORTING
  evidence, not independent proof; and 0.27 ≠ (rain−Ks)/rain=0.5 — 0.27 means MORE infiltration than steady
  Hortonian (29 mm infiltrated vs Ks·storm=20 mm; the extra ~9 mm IS the short-storm sorptivity), which is
  consistent but I mis-framed the "0.5 Hortonian" tie.

## Codex's single highest-value confirmatory test
**`b1_base` UNIFORM vertical mesh nz=64** (dz_top=15.6 mm, ell_c=7.8 mm — matches graded `p=2.0` WITHOUT
deep-cell coarsening), same forcing/scheme/ell_c-auto. **If it lands near routed/R ≈ 0.29 (= graded p=2.0's
0.287) → the grading-confound argument largely DIES (collapse is genuine resolution). If it rebounds toward
0.55 → §22 is in trouble.** (A finer graded rung tests convergence; a fine-dz ParFlow rerun tests the
ParFlow inference; but the uniform-nz cross-check is THE cleanest discriminator for §22.)

## Codex's most-defensible rewrite (adopted into §22)
"In the real 3-D routed loam case, the monolithic partition is highly sensitive to near-surface vertical
resolution: a top-biased graded nz=8 ladder drops routed/R from 0.547 to 0.266-0.269 with fixed lateral
mesh and forcing, while infiltration rises and solves stay clean. This is enough to REJECT 0.547 as a
mesh-objective validation target and reopen the sequential-vs-monolith framing. It is NOT yet enough to
claim convergence at ~0.27, to validate the sequential closures, or to state as fact that ParFlow is
under-resolved."

## Disposition (mine)
Adopt the rewrite. RUNNING the uniform-nz {16,32,64} cross-check now (the discriminator). If nz=64 ≈ 0.29,
the resolution conclusion is robust (grading confound dead); then the remaining open items are (a) one
finer rung for the converged value, (b) reconcile ParFlow B5 at fine dz before any "ParFlow under-resolved"
statement, (c) the subgrid capacity closure. The collapse + "0.547 is not mesh-objective truth" stand; the
"~0.27" endpoint and the ParFlow inference are softened to hypotheses.
