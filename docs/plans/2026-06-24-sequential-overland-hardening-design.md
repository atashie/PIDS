# Sequential Overland Scheme — Hardening Test Design

> **Status:** DESIGN (approved section-by-section, 2026-06-24). Next: write the bite-sized
> implementation plan (superpowers:writing-plans) and execute (superpowers:subagent-driven-development).

**Goal:** Harden the merged sequential operator-split overland coupling
(`SequentialCoupledProblem`, `f7c48d8`) by validating it beyond the deliberately-shrunk
fixtures — at field scale, on real run-on accuracy, against the monolith, at the
no-Ss saturation edge, and for performance.

**Context:** The sequential scheme (implicit Richards solved alone + explicit Manning
rate-limited downslope routing + water-LEVEL infiltration handoff) is COMPLETE and merged
to `main` with conservation proven to machine precision and a Tier-3 visual sign-off
(decision record `docs/plans/2026-06-22-overland-flow-sequential-coupling-decision.md`,
memory `pids-overland-flow-rethink`). Everything validated so far runs on small cases: the
largest is the sign-off's sand-channel-in-clay box — **8×5×1 m at 20×12×5 (~1.2k cells,
~4k DOFs), t_end=1.2 d, storm 0.15 m/d**; the suite fixtures are smaller (6×3×1 m,
t_end=0.3 d). This effort takes the scheme to the scales and regimes PIDS actually
operates in.

**Scope decision (Arik, 2026-06-24):** option **A — harden the new overland scheme**.
Priority order locked: **Test 1 + Test 2 first** (existential + core-job), then Tests 3–5.

---

## Known limits this design is testing against

From the redesign sign-off, the sequential scheme's documented limits are:

- **Kinematic-timing, not flood-depth** — accurate redistribution/timing, not peak flood
  stage (galerkin/upwind remain the depth fallback). The hardening targets the
  redistribution+infiltration story PIDS needs, not hydrograph peaks.
- **Near-saturation no-Ss fragility** — PIDS models an unconfined aquifer with no specific
  storage (`pids-forward-model-assumptions`), so capacity C=dθ/dψ→0 at saturation; the
  mitigation is carrying the pond as positive head. Test 4 makes this precise.
- **Serial-only** — all runs single-process (BLAS pinned to 1 thread per
  `pids-fem-blas-threading`).

---

## Test 1 — Field-scale robustness (TOP PRIORITY)

**Scenario.** A convergent hillslope with a mid-slope conveyance feature (the canonical
PIDS setting): downslope **~50 m × cross-slope ~30 m × soil depth ~2 m**, with a tilted-V
or swale convergence line down the fall axis — a ~0.15 ha hillslope (vs the 0.004 ha
sign-off box). Native low-K **CLAY** (`Ks=0.048`, the regime that dt-collapsed the old
monolith) so it ponds and runs on. Design storm stepped up from the gentle 0.15 m/d to a
realistic **~1.0–1.5 m/d (~40–60 mm/hr) burst over ~2–4 hr, then recession to ~0.5 d** —
exercising the ledger over *thousands* of steps.

**The test is a refinement ladder.** The old scheme's dt-collapse was *refinement-driven*
(item-C finding), so run the same scenario at **coarse → medium → fine** horizontal
resolution (~2 m → ~1 m → ~0.5 m cells; roughly **12k → 30k → 70k cells**).

**Assert at every rung:**
1. **Completes to T_END, no dt-collapse** (the headline claim of the redesign).
2. **Conservation holds and does not drift** — `|balance|/cum_rain < 1e-6` sustained
   across the long run (not just at the end).
3. **Routing store resid ≤ 1e-12** each step (no accumulation).
4. **Peak pond stays physical** — bounded and ≥ 0.

Wall-time per rung is recorded and feeds Test 5. The fine rung is the existential check:
does it still hold when we stop shrinking the cases?

---

## Test 2 — Run-on accuracy (TOP PRIORITY)

Run-on means two things must be right *together*: water redistributes downslope/into the
convergence line, and it infiltrates *there*. The pond-release calibration only checked
transport with infiltration off; this closes that gap.

**Headline metric — the infiltration partition:** of the rain that fell, how much
infiltrated vs routed out the outlet vs remains ponded, **and where** it infiltrated (the
spatial field).

**The oracle is regime-split** (the validated monolith only converges on mild cases):

- **Case A — oracle-anchored (non-stiff).** A milder hillslope (**LOAM**, moderate slope +
  storm) where `CoupledProblem(overland_scheme="upwind")` converges. Assert sequential vs
  upwind agree on (i) the global infiltrate/runoff/pond partition to a few %, and (ii) the
  convergence-line infiltration footprint. This is the true external accuracy anchor.
- **Case B — target regime (stiff convergent CLAY, upwind collapses).** No monolith oracle,
  so accuracy is established three ways:
  - (i) **self-convergence** of the partition under mesh + route_substeps refinement;
  - (ii) the **run-on signature** — the convergence line infiltrates *strictly more* than a
    no-lateral-transport twin (each column infiltrating only its own rain), proving run-on
    is physically happening, not merely conserved;
  - (iii) **route_substeps sensitivity** — confirm `rs=4` gives a resolved partition
    (`rs=8` within tolerance, not still climbing).

Point (iii) hardens the `route_substeps=4` default on *infiltrating* run-on, extending the
idealized pond-release calibration
(`validation/sanity/overland_transport_calibration__2026-06-23.md`) to the real use case.

---

## Test 3 — Sequential-vs-monolith agreement battery

Broad-confidence check across a small matrix where *both* schemes converge; generalizes
Test 2's Case-A anchor and characterizes the operator-split error magnitude.

**The matrix (small — YAGNI).** Two mid-range soils where the monolith is well-behaved —
**LOAM** (`Ks=0.25`) and **SILT** (codebase definition / standard Carsel–Parrish) — crossed
with two geometries — a **planar hillslope** (no convergence) and a **gentle swale** (mild
convergence the monolith still resolves) — under a **ponding storm**. That's **4 coupled
cells**. Plus one **sub-infiltration no-pond case** (rain < Ks → no surface water → the
split reduces to the monolith's vertical Richards) as the tight-agreement anchor. The sharp
tilted-V / convergent-clay is deliberately excluded — that's Case B, where the monolith
collapses and there is no oracle.

**Metric is regime-aware.** Both schemes march independent dt sequences, so compare
*physical invariants over a fixed horizon*, not node values: the infiltrate/route-out/pond
partition + the sink volumes. Tolerance **~1e-3** on the no-pond anchor, a **few %** on the
ponding cells — and **record the discrepancy in every cell**, yielding a characterized map
of how far operator-split lag moves the answer from monolithic simultaneity across soil and
geometry, not a bare pass/fail.

---

## Test 4 — Near-saturation no-Ss fragility (characterization)

With no specific storage, the moisture capacity C=dθ/dψ→0 at full saturation, so the system
goes singular/stiff at ψ=0. The sign-off saw the flip side: a deep berm pond (positive head)
*regularized* the singularity and let even the old upwind monolith limp through. So the
danger zone is **ψ_top ≈ 0⁺ with no pond buffer**.

**This is characterization, not a pass/fail gate.** Deliberately provoke the singular regime
the three ways realistic run-on visits it:
- (a) **ponding onset** — a storm tuned so the surface just barely saturates (ψ crosses 0
  from below);
- (b) **recession tail** — a pond infiltrating fully down to ψ_top → 0⁺;
- (c) **thin sheet on a shedding flank** — routing sheds water as fast as it arrives,
  holding a near-zero saturated film.

For each, measure the **symptoms**: Newton iteration spikes, how far dt must shrink, and —
the real question — **whether conservation and completion survive the transition**.

**Then test the mitigation and decide.** Confirm the design's "carry pond as +head" relieves
it and quantify how much head is enough. **Outcome forks:**
- realistic run-on naturally keeps a buffer → **document the safe envelope and move on**; or
- it bites in a real scenario → **a guard is needed.** Guard options — minimum-pond
  regularization, a saturation-aware dt limiter, or a tiny numerical Ss floor — differ in
  whether they touch the no-Ss physics assumption, so **if it comes to a guard, the fork
  goes to Arik** rather than being pre-decided.

---

## Test 5 — Performance sanity

The routing sweep is a pure-Python loop (receivers → topo-order → `route_excess`) ×
`route_substeps` over the surface nodes; the Richards solve is a MUMPS LU on tens-of-thousands
of DOFs. **Reusing Test 1's refinement-ladder runs (no new runs — DRY)**, instrument a
**wall-time breakdown**: Richards solve vs routing sweep vs assembly/ledger, at each rung.

**Check two things:** which cost *dominates* at field scale, and whether total wall-time
scales *as expected* with cell count or **super-linearly** (a red flag). Also confirm the
routing graph (receivers/topo-order) isn't needlessly rebuilt every step.

This is a sanity check, not an optimization project: if routing is negligible (likely) note
it and stop; if it dominates or scales badly, flag the pure-NumPy kernel for vectorization.
**YAGNI — don't optimize what isn't slow.**

---

## Where the tests live (two homes)

- **pytest suite** — `tests/test_sequential_hardening.py` (new). Cheap, deterministic
  guards only: a *coarse* refinement-robustness check, one vs-monolith cell, the
  rs-sensitivity assertion, a fast near-saturation provoke. Must be suite-fast (the existing
  fixtures are coarse for exactly this reason).
- **viz/sanity data-run** — `viz/run_sequential_overland_hardening.py` +
  `viz/make_sequential_overland_hardening_html.py`. The *expensive* field-scale ladder, the
  accuracy battery (Case A/B), and the timing breakdown. Tier-3 HTML for visual sign-off.
  Gitignore the heavy npz/HTML (sign-off pattern); commit the code.

**Verdict record** — `validation/sanity/sequential_overland_hardening__<date>.md`:
field-scale pass, accuracy numbers, the vs-monolith discrepancy map, the near-saturation
envelope/decision, the perf breakdown.

---

## Process & gate

- **Subagent-driven TDD** (superpowers:subagent-driven-development): fresh implementer per
  task + two-stage review (spec, then code quality).
- **Adversarial Codex review** of the assembled battery.
- **Commits handled by the parent** (subagents do not commit); verify `git branch` = `main`
  before every commit (the investor-branch gotcha, `pids-embedded-coupled-integration`).
- **Closes on Arik's Tier-3 visual sign-off**, with the near-saturation guard fork surfaced
  if it triggers.
- **Reference soils** (from the sign-off run): SAND `theta_r=0.045,theta_s=0.43,alpha=14.5,
  n=2.68,Ks=7.13`; CLAY `0.068,0.38,0.8,1.09,0.048`; LOAM `0.078,0.43,3.6,1.56,0.25`; SILT
  = codebase definition / standard Carsel–Parrish.
- **Run pattern (WSL):** `pids-fem` conda env, `PYTHONPATH=.`,
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, serial.

---

## Open decisions (carried to execution)

1. **Near-saturation guard** (Test 4) — only if the fragility bites a realistic scenario;
   fork brought to Arik.
2. **Exact field-scale mesh counts** — the ~12k/30k/70k ladder is a target; the implementer
   tunes to keep the fine rung within a sane wall-time while still being a genuine step up.
