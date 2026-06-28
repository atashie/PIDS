# Codex review — the graded near-surface package result + recommendation (2026-06-28)

7th Codex review (gpt-5-codex, high) of §29's "r=3 geometric near-surface package = the solution." Prompt:
`scratch/_codex_gradedpkg_result_review_prompt.txt`. **Verdict: OVER-CLAIMED — `r=3` is the best-tested
default for b1, NOT a universal solution. Do not ship as "the solution"; require a 2nd-geometry stress
test first.** All hits accepted.

## Valid hits
1. **loam-long FAILS the gate, and it's the deliberate deeper-front stress case** — −1.6 pp (outside ±1 pp),
   and r=2 doesn't fix it (−1.9 pp). NOT benign: it is exactly the case meant to test where a fixed shallow
   package should fail. r=2-barely-changing-loam-long ⟹ the residual is NOT "just add depth" — it points to
   the handoff into the coarse cells below, or a longer-duration distributed resistance a fixed shallow
   package doesn't represent.
2. **"Deeper is WORSE" is a COST claim, not physics.** On loam-MODERATE, r=2 (0.259) is actually CLOSER to
   ref (0.264) than r=3 (0.270). r=2 is worse only on sand/loam-long + much slower. Correct framing: **r=3
   is the better cost/accuracy COMPROMISE**, not "finer is physically worse." (The sand r=2 run is healthy
   at the surface — ok, clip 6e-16 — but ns=859/6403 s = severe stiffness, not proof of a real 80 mm optimum.)
3. **Generality is the biggest over-reach.** §25 said the sensitivity is geometry/discretization-driven and
   UN-isolated (slope/aspect/lateral-res confounded); §29 validated only on b1 and called it "generalizes."
   That does NOT support a FIXED 80 mm package across geometries — the needed depth could scale with the
   un-isolated geometry factor.
4. **Reference NOT fatally circular** (§22 uniform-vs-graded agree ~1 pp at matched ell_c defends it), and
   ±1 pp IS meaningful (ref uncertainty ~0.08–0.17 pp). But this doesn't earn "universal closure."
5. **L90 sound only as a NEGATIVE result** (end-of-storm L90 is the wrong anchor). The "partition set by the
   early/upper front" positive statement is interpretive; it does NOT yet predict WHEN 80 mm fails, and
   r=2-not-helping-loam-long says the trigger is not a simple "front depth > 80 mm" rule.
6. **Modularity real only in the narrowest tested sense** — only "uniform-below" tested
   (`seq_skin_split.py:65` builds the package on a uniform subsurface; "mesh-agnostic" is design INTENT
   only). NO heterogeneous/layered/already-fine subsurface tested — i.e. the lead's stated use cases are
   UNVALIDATED.

## Highest-value next test (Codex)
**One materially different geometry from b1, run the loam-long dry-soil stress case with {coarse, converged
ref p2.75/p3.0, fixed r=3}.** This attacks BOTH unclosed risks at once: the §25 geometry dependence AND the
§29 loam-long gate failure.

## Most defensible statement (adopt)
"`r=3` (2,6,18,54 mm + uniform below) is the best-tested default modular Richards-only near-surface package
FOR the b1 geometry and the tested soil/storm matrix — good cost/accuracy, no solver pathology. It is NOT
a universal solution: it MISSES the deeper-front loam-long gate (−1.6 pp, and deepening doesn't fix it),
and it is untested on a 2nd sensitive geometry and on non-uniform subsurfaces. Ship (if at all) as an
EXPERIMENTAL default behind an explicit caveat; require the 2nd-geometry loam-long transfer test before
productionizing."

## Disposition (mine)
Accept fully. §29 "solution/generalizes" is RETRACTED to "best-tested b1 default, not universal." Temper the
record; the next step is the 2nd-geometry × loam-long stress test (+ a non-uniform-subsurface check) before
any productionization. The CORE findings remain solid: the 0.547 is a coarse-vertical-resolution artifact;
the converged partition is ~0.265 (b1 loam); ParFlow reconciled; a near-surface graded package is the
right DIRECTION within pure Richards — just not yet proven universal at a fixed depth.
