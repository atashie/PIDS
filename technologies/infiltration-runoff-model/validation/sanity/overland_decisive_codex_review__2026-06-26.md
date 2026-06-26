# Codex review — the "decisive" sorptivity result (2026-06-26) — VERDICT: MIS-FRAMED

Third skeptical Codex review (gpt-5-codex, high) of `scratch/seq_sorptivity_real.py` + the §-mechanics
"q_pot gap is a coarse-film artifact" conclusion. Prompt: `scratch/_codex_decisive_review_prompt.txt`.
**Verdict: MIS-FRAMED — the test does NOT prove what it claimed; a clean test is specified.**

## The valid hits (all upheld — my errors)

1. **The `ell_c` sweep is NOT a resolution test — it's a free-closure-parameter sweep on a FIXED coarse
   mesh.** `CoupledProblem` documents + auto-detects `ell_c = top-cell half-height` (`coupling.py:64,
   191-209`); the benchmark FIXES nz=8 (top cell ~125 mm) and sweeps `ell_c` to 1 mm
   (`seq_sorptivity_real.py:157-165`). Since `q_pot = kirchhoff(ψ,d)/ell_c`, shrinking `ell_c` below the
   cell is just "reduce the film resistance by hand." A legitimate asymptotic PROBE, NOT a claim that the
   production discretization converges.
2. **The "96%" is NOT convergence of the same problem.** Monolith side = nz=8, rain-driven, NCP-coupled;
   reference side = nz=60/120/240, permanently-ponded Dirichlet from t=0. **Two different meshes AND two
   different top-BC problems can cross by error cancellation.** Missing: same-mesh refinement with
   `ell_c = dz_top/2`. Also never printed solver health (`last_reason/fnorm/clip`) or total-water closure
   → "96% without a degraded solve" is unproven.
3. **Early-time ratios are contaminated by finite ponding onset.** Closed flat no-routing column, finite
   rain = 10·Ks: by t=0.002 total rain ≈ 5.0 mm ≈ the Dirichlet 5.03 — not yet a ponded-capacity
   asymptote. Un-infiltrated water is surface STORAGE, not runoff. Without `d(t)`/ponding-time, the 17%/30%
   early ratios over-interpret finite pond buildup.
4. **ψ_top=0 Dirichlet is NOT the sole "truth"** — it assumes INSTANT ponding from t=0. The repo's own
   thin-film reduction test (`tests/test_coupling_1d.py:110-141`) compares the coupled model to
   `add_ponding_bc` ON THE SAME MESH, not to pure Dirichlet on a different mesh. For the storm question the
   cleaner control is a RAIN-DRIVEN ponding/switching column; Dirichlet is an upper envelope after ponding.
5. **The dry-IC framing is inconsistent + confounded.** Docstring still says "soil-matched S_e" but the
   code uses fixed `PSI_DRY=-3 m` (`:47-63`) AND scales rain by each soil's Ks (`:137`) → cross-soil claims
   ("dryness is soil-specific", "dry clay → prompt runoff") are confounded TWICE (different storage deficit
   AND different storm intensity). Fixed ψ = "same suction", NOT a clean "dry clay" test when clay is still
   ~90% saturated at −3 m.
6. **The big inference is too broad.** This isolates VERTICAL sorptivity in a NO-ROUTING column, not the
   3-D runoff partition. It supports only: coarse `ell_c/dz` strongly throttles uptake, and 0.547 is
   plausibly contaminated by coarse-film/coarse-cell effects. It does NOT establish scheme choice is
   settled, NOR that "switch BC and refined monolith agree at resolution" (the switching BC is NOT RUN in
   this benchmark — that claim was stitched from the separate §18 setup).
7. **The practical implication is NOT "just refine."** If correct sorptivity needs mm-scale surface cells,
   that is not a tractable field-scale 3-D operating point. And if ParFlow used the same coarse vertical
   resolution, 0.547 is "the truth of that under-resolved benchmark," not continuum truth. **The
   engineering lesson: the coarse production answer is not decision-grade without a MESH-OBJECTIVE SUBGRID
   infiltration closure** (a corrected q_pot / Green-Ampt capacity model — the GSSHA/tRIBS family).

## The clean test Codex specifies (the real decisive test)
Run the real `CoupledProblem` on an **nz ladder {8,16,32,64,...} with `ell_c` LOCKED to the actual mesh
(`ell_c = dz_top/2`, auto)** — so mesh AND film scale refine together — and compare ON EACH SAME MESH to a
RAIN-DRIVEN `add_ponding_bc` column with the SAME forcing, consistent `∫θ`, `d(t)`, and solver-health
diagnostics. **If the monolith collapses toward the ponding curve as dz→0, the gap is genuinely a
surface-resolution artifact. If not, the `ell_c=1mm` result was parameter tuning / cross-discretization
cancellation, not convergence.**

## Disposition (mine)
The "q_pot gap is a coarse-film artifact" conclusion is **RETRACTED as proven** — it is a PLAUSIBLE
HYPOTHESIS the mis-framed test cannot establish. What survives: the real monolith is strongly `ell_c`-
sensitive, so the coarse production 0.547 very likely carries a large coarse-film/coarse-cell component.
NEXT = the locked-`ell_c` nz-ladder + rain-driven `add_ponding_bc` same-mesh comparison with full
diagnostics. AND keep Codex's #7 in view: even if it IS a resolution artifact, the field-scale answer is
likely a mesh-objective subgrid infiltration closure, not brute refinement.
