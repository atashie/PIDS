# Sequential Overland Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this
> plan task-by-task (same session, fresh implementer subagent per task + two-stage review).

**Goal:** Validate the merged `SequentialCoupledProblem` beyond the shrunk fixtures — field-scale
robustness, run-on accuracy, sequential-vs-monolith agreement, the no-Ss saturation edge, and
performance — per the approved design `docs/plans/2026-06-24-sequential-overland-hardening-design.md`.

**Architecture:** Two homes. Cheap deterministic guards in the pytest suite
(`tests/test_sequential_hardening.py`); the heavy field-scale ladder + accuracy battery + timing in a
viz/sanity data-run (`viz/run_sequential_overland_hardening.py`) → Tier-3 HTML + a sanity record.

**Tech Stack:** DOLFINx 0.10 / PETSc (MUMPS LU), pure-NumPy routing kernel, pytest. WSL2 conda env
`pids-fem`, serial, BLAS pinned.

**Working directory:** `technologies/infiltration-runoff-model/forward-model/`

**WSL run pattern (every heavy run / suite invocation):**
```bash
cd .../forward-model && export PATH=.../pids-fem/bin:$PATH && PYTHONPATH=. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u <script-or-pytest>
```

---

## CRITICAL FRAMING — these are acceptance tests against MERGED code

`SequentialCoupledProblem` is already merged and validated on small cases. These tests do **not**
follow red-green TDD (there is no new feature to make pass). The discipline is:

1. Author the test with a **pre-registered scenario, metric, and tolerance** (write the tolerance
   into the assertion + a comment explaining where it comes from).
2. Run it under the WSL pattern.
3. **PASS** → the scheme is hardened on that axis; record the numbers.
4. **FAIL** → **STOP. Surface the finding to the parent (→ Arik).** A failing hardening test is a
   real finding — a genuine physical limit, a tolerance that needs justification, or a bug. **Do NOT
   loosen the tolerance to force green** without sign-off. The parent decides: accept-and-document,
   tighten/justify, or open a fix task.

The subagent-driven two-stage review (spec compliance, then code quality) still applies to every
test-authoring task. **Subagents do NOT commit — the parent commits** (verifying `git branch` =
`main` first; investor-branch gotcha).

**Partition metric (shared definition, used throughout).** For a marched problem, the water partition
of `cum_rain` is the balance decomposition:
`cum_rain == Δ(soil storage) + surface_store + cum_outflow + cum_drainage`
where `Δ(soil storage) = soil_water() - soil_water_0`, `surface_store = surface_water()` (still
ponded), `cum_outflow` (surface routed out), `cum_drainage` (sinks). These four fractions are the
"infiltrate / pond / route-out / drain" partition. For `CoupledProblem` the same quantities exist
(soil store via `∫θ`, surface via `∫d`, plus `cum_outflow`/`cum_drainage`). Each test/run defines a
small `water_partition(prob)` helper locally (the codebase duplicates fixtures across the test/viz
boundary by convention — DRY within a file, not across it).

---

## SUITE TASKS (cheap guards — `tests/test_sequential_hardening.py`)

Each suite task adds tests to the new file. Keep every test **suite-fast** (target ≤ ~15 s each;
coarse meshes, short horizons). Run the specific test(s) after authoring:
`pytest tests/test_sequential_hardening.py::<name> -v`.

### Task 1: Field-scale robustness — coarse guard

**Files:** Create `tests/test_sequential_hardening.py`.

**Spec.** A convergent-clay hillslope (CLAY `theta_r=0.068,theta_s=0.38,alpha=0.8,n=1.09,Ks=0.048`,
a tilted-V topography + a downslope outlet, ponding storm `rain≈1.0 m/d` for a short burst then
recession) at **two coarse resolutions** that still differ enough to exercise refinement
(e.g. `16×10×4` and `28×18×5`). For **each** rung assert:
- completes to `t_end` with **no dt-collapse** (the marching loop never drives `dt` below `1e-9`);
- conservation holds **and does not drift**: track `|balance|/cum_rain` every accepted step and assert
  the **max over the run** `< 1e-6` (not only the final value);
- routing store resid (`prob.last_routing_resid`) ≤ `1e-12` every step;
- peak pond (`max(max(psi,0))`) is finite and ≥ 0.
- guard: `cum_rain > 0` and `cum_outflow > 0` (the storm genuinely ran on and routed — not vacuous).

**Steps:**
1. Write `test_field_scale_robustness_coarse_two_rungs` per the spec (helper `_convergent_clay_hillslope(nx,ny,nz,...)` building the problem; a marcher recording `max|balance|/cum_rain`).
2. Run it. PASS → record per-rung (steps, max-bal, wall). FAIL → surface (esp. any dt-collapse or balance drift — that contradicts the merged claim and must go to Arik).
3. Parent commits.

**Acceptance:** both rungs complete, conserve (max `<1e-6`), routing-tight; finer rung shows no
refinement-driven degradation at suite scale.

### Task 2: Run-on accuracy — suite guards

**Files:** Modify `tests/test_sequential_hardening.py`.

**Spec — three tests:**
- `test_caseA_runon_partition_matches_upwind_coarse`: a **mild LOAM** hillslope (gentle slope, storm
  where the monolith converges) run with BOTH `SequentialCoupledProblem` and
  `CoupledProblem(overland_scheme="upwind")` over the same horizon. Assert the global partition
  (infiltrate/pond/route-out) agrees within a few % (pre-register, e.g. `rel < 0.05` on each non-tiny
  fraction). Guard: upwind genuinely converged (reached `t_end`, did not collapse).
- `test_caseB_runon_signature_vs_no_transport_twin`: the convergent-clay case + a **no-lateral-transport
  twin** built by setting `n_man` huge (e.g. `1e6` → routing velocity ~0, water infiltrates where it
  fell). Assert (i) the twin genuinely does not route (`cum_outflow ≈ 0`, surface stays put), and
  (ii) the real run infiltrates **strictly more** into the convergence-line column footprint than the
  twin (`Δθ_convergence_real > Δθ_convergence_twin` by a real margin) — proves run-on physically moves
  water to where it soaks in.
- `test_route_substeps_partition_resolved_rs4_vs_rs8`: the same run-on case at `route_substeps=4` and
  `=8`; assert the partition agrees within tolerance (e.g. `rel < 0.05`) — `rs=4` is resolved, not
  still climbing. (Complements the pond-release `rs=1` vs `rs=4` throttle test already in
  `test_sequential_coupling.py`.)

**Steps:** author each test (local `water_partition` + a convergence-footprint `∫θ` over the column
band); run; PASS → record; FAIL → surface (a Case-A miss vs upwind is an accuracy finding for Arik).
Parent commits.

**Acceptance:** Case-A matches upwind within the registered tol; the run-on signature is positive;
`rs=4` is resolved.

### Task 3: Sequential-vs-monolith agreement battery

**Files:** Modify `tests/test_sequential_hardening.py`.

**Spec.** A parameterized `test_seq_vs_monolith_partition[soil-geom]` over the **4+1 matrix**:
`{LOAM, SILT} × {planar hillslope, gentle swale}` under a ponding storm (4 cells) + one
**sub-infiltration no-pond** case (`rain < Ks`, dry start) as the tight anchor. For each cell run
`SequentialCoupledProblem` and `CoupledProblem(overland_scheme="upwind")` (or `"galerkin"` where
upwind is unnecessary) over a fixed horizon; compare the partition + sink volumes. Tolerance:
`rel < 1e-3` on the no-pond anchor (reduces to vertical Richards), a **few %** on the 4 ponding cells.
**Record the discrepancy** for every cell (print + assert), building the operator-split error map.
SILT params: codebase definition if present, else standard Carsel–Parrish silt.

**Steps:** author the parameterized test + a small results table printed at the end; run; PASS →
record the discrepancy map; FAIL → surface. Parent commits.

**Acceptance:** all 5 cells within their registered tolerance; the discrepancy map is recorded.

### Task 4: Near-saturation no-Ss provoke (characterization)

**Files:** Modify `tests/test_sequential_hardening.py`.

**Spec.** Deliberately drive `ψ_top` through/near 0 the three ways from the design and assert the
**robustness invariants survive**, while **recording the symptoms** (this is characterization, not a
tight gate):
- `test_near_saturation_ponding_onset_survives`: a storm tuned so the surface just reaches saturation
  (ψ crosses 0 from below). Assert: completes (no dt-collapse) and `|balance|/cum_rain < 1e-6` through
  the transition. Record: peak Newton its, min dt at the crossing.
- `test_near_saturation_recession_tail_survives`: pond infiltrates fully down to ψ_top→0⁺ (rain off,
  permeable enough to draw the pond to ~0). Assert: completes + conserves; record min dt / max its at
  the tail.
- (optional, if cheap) a thin-sheet shedding-flank variant.

Each test also asserts the **mitigation direction**: a twin carrying a small positive pond buffer
(seed +ε head) does **not** show worse dt/Newton behaviour than the bare-zero case (confirms "carry
pond as +head" relieves it). If any provoke FAILS (dt-collapse or balance break at ψ≈0) → **STOP and
surface the guard fork to Arik** (min-pond regularization vs saturation-aware dt limiter vs numerical
Ss floor; the last touches the no-Ss assumption).

**Steps:** author the provoke tests + symptom recording; run; record the envelope; surface any
failure as the guard fork. Parent commits.

**Acceptance:** the realistic provokes complete + conserve, OR a failure is surfaced as a decision
(not silently passed by loosening).

---

## VIZ / SANITY DATA-RUN TASKS (heavy — field scale)

These run in WSL at real wall-time. The parent checkpoints with Arik rather than blocking silently;
long runs go in the background and incremental-save per stage (the sign-off `_stage` pattern) so a
killed run never loses completed stages.

### Task 5: Field-scale refinement ladder + timing (Tests 1 & 5)

**Files:** Create `viz/run_sequential_overland_hardening.py`.

**Spec.** The convergent-clay hillslope (~50 m × ~30 m × ~2 m, tilted-V convergence line + outlet,
design storm ~1.0–1.5 m/d burst → recession to ~0.5 d) at **coarse → medium → fine** resolution.
Target counts (implementer **tunes** to keep the fine rung within a sane wall-time while remaining a
genuine step up over the ~1.2k-cell sign-off): **~9k → ~24k → ~60k cells**. For each rung capture:
- the dt-vs-t timeline (headline: stays up, no collapse);
- `|balance|/cum_rain` every accepted step (machine-tight, no drift);
- peak pond, routing resid;
- **wall-time breakdown** (Test 5): instrument the time in (a) the Richards solve, (b) the routing
  sweep, (c) assembly/ledger, per rung — and confirm the routing graph isn't rebuilt needlessly.
- Report whether total wall-time scales as-expected with cell count (flag super-linear).

Incremental-save to `scratch/sequential_overland_hardening.npz` after each rung.

**Steps:** write the run; launch in background under the WSL pattern; on completion record the ladder
verdict + the timing breakdown. If a rung dt-collapses or drifts → surface (contradicts the merged
claim). Parent commits the script (not the npz).

**Acceptance:** all rungs complete + conserve; a timing breakdown identifying the dominant cost +
scaling exists.

### Task 6: Accuracy battery — Case A & Case B at field scale (Test 2)

**Files:** Modify `viz/run_sequential_overland_hardening.py` (add stages).

**Spec.**
- **Case A** (oracle-anchored): a mild LOAM field-scale hillslope where `upwind` converges; run both
  schemes; record the partition + the convergence-line infiltration footprint side by side + the
  agreement.
- **Case B** (target convergent-clay): self-convergence of the partition across the Task-5 ladder
  rungs (reuse them) + `route_substeps ∈ {1,4,8}` sensitivity at one rung + the run-on signature vs
  the high-`n_man` no-transport twin.

Incremental-save the battery arrays.

**Steps:** add the stages; run (background); record Case-A agreement + Case-B convergence/signature.
Surface a Case-A miss vs upwind. Parent commits.

**Acceptance:** Case-A agreement quantified; Case-B self-converges; run-on signature positive;
`rs` sensitivity shows `rs=4` resolved.

### Task 7: Tier-3 HTML builder

**Files:** Create `viz/make_sequential_overland_hardening_html.py`.

**Spec.** Read `scratch/sequential_overland_hardening.npz` and render a self-contained Tier-3 HTML
(inlined Plotly, sign-off pattern; degrade gracefully if a stage is absent): the ladder dt-vs-t (all
rungs holding), the balance traces, the partition-convergence + Case-A-vs-upwind panels, the timing
breakdown bars, and the near-saturation symptom summary. Output
`viz/sequential_overland_hardening__<date>.html`.

**Steps:** write the builder; generate the HTML from the npz; eyeball it renders. Parent commits the
builder (HTML is gitignored).

**Acceptance:** a single self-contained HTML renders every present stage.

### Task 8: gitignore + sanity record + end-to-end + verdict

**Files:** Modify `.gitignore`; create `validation/sanity/sequential_overland_hardening__<date>.md`.

**Spec.**
- `.gitignore`: add `…/viz/sequential_overland_hardening__*.html` and
  `…/scratch/sequential_overland_hardening.npz` (mirror the existing sign-off entries).
- Run the full data-run + HTML build end-to-end (WSL).
- Write the **verdict record**: field-scale pass (per-rung), the accuracy numbers (Case A/B), the
  vs-monolith discrepancy map (from Task 3), the near-saturation envelope + any guard decision, the
  perf breakdown + scaling. State tolerances and where they came from. Link the design doc.

**Steps:** edit `.gitignore`; run end-to-end; author the record. Parent commits.

**Acceptance:** record committed; heavy artifacts gitignored; the suite (`pytest tests/ -q`) is green.

---

## FINAL — review & sign-off

### Task 9: Adversarial Codex review + Arik Tier-3 sign-off

- Dispatch the assembled battery (suite + run + record) to **Codex** (`codex:codex-rescue`) for an
  adversarial pass: are the tolerances honest, the oracles valid, the no-transport twin a fair
  discriminator, the timing instrumentation correct, any vacuous assertion?
- Address findings (fix-then-ship); parent commits.
- Present the Tier-3 HTML + the verdict record to **Arik** for visual sign-off. Surface the
  near-saturation guard fork if it triggered. **Do not merge/close the workstream until signed off.**

---

## Task ordering

Priority-locked: **Tasks 1 → 2** (existential + core-job, suite) first, then **5 → 6** (their
field-scale runs), then **3, 4** (battery + saturation), then **7 → 8** (viz + record), then **9**
(review + sign-off). Tasks 5/6 are long; launch in background and proceed with suite tasks while they
run.
