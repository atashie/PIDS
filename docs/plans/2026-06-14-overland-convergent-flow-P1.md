# Convergent-Flow P1 Implementation Plan — scale-invariant acceptance gate + O1 upwind-flux spike

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** (A) Replace the non-portable absolute acceptance bar with a scale-invariant *residual-normalized* gate, then (B) spike the **O1 upwind-mobility two-point edge flux** on standalone Module 2 to prove it is monotone (positive without the limiter), conservative, and convergent on the 2-D tilted-V — the production fix for the convergence-line regime.

**Architecture:** Part A is a surgical change to `step()` in both solver classes: the reason-4 (stagnation) booking decision moves from `‖F‖₂ ≤ stall_accept_fnorm` (absolute, mesh/area-dependent) to `‖F‖₂ ≤ tol_rel · R_scale` where `R_scale` is a problem-natural residual scale that scales *with* the problem (dimensionless `tol_rel`). Part B is a **new standalone class** `UpwindOverlandProblem` (it does NOT touch the validated `OverlandProblem`) that assembles the lateral conveyance as an edge-graph two-point flux with upstream-weighted mobility and solves it with a PETSc SNES using a finite-difference (colored) Jacobian — the same "no analytic overland Jacobian" choice ParFlow ships (`UseJacobian False`). A smoothed C¹ upwind selector keeps the FD-Jacobian Newton well-behaved.

**Tech Stack:** Python, DOLFINx 0.10 + PETSc/petsc4py (WSL2 conda env `pids-fem`), pytest. Run pattern (every command below assumes this): from `technologies/infiltration-runoff-model/forward-model/`, with `PATH=/root/miniforge3/envs/pids-fem/bin:$PATH` (conda is NOT on PATH non-interactively; `gcc` from the env is needed for FFCX JIT), `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, and `PYTHONPATH=.`. The package is NOT pip-installed.

---

## Background & context (read before starting)

- **Parent plan:** `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md`. P0 is done (§8 = the results/mechanism note). This plan executes **P1** (§5, the P1 bullet + its PREREQUISITE sub-bullet) and the acceptance bars in §6.
- **Why Part A first (Codex P0 review, F1):** `stall_accept_fnorm = 3e-6` is an absolute L2-residual bar. `‖F‖₂` scales with mesh row-count and per-row flux/measure, so the value that separates the *measured* populations (Tier-1 columns vs the km² V) is not portable to P1's new scheme or P3's swale fixture. The gate must become scale-invariant before O1 introduces a new scheme/scale.
- **The two solver classes and the current gate:**
  - `pids_forward/physics/overland.py` — `OverlandProblem` (single field `d`; standalone Module 2). `step()` is at lines ~307-345; the reason-4 norm-recompute + gate at ~319-337. `stall_accept_fnorm` is set in `__init__` (search `self.stall_accept_fnorm`).
  - `pids_forward/physics/coupling.py` — `CoupledProblem` (block `[ψ, d, λ]`; Module 3). `step()` reason-4 handling + gate is the mirror of overland's (search `last_reason == 4`).
- **CORRECTION to the parent plan + Codex F1 — a "mass-relative gate" is WRONG; the fix is a scale-invariant RESIDUAL gate (measured 2026-06-14, before writing this plan).** The parent §5/§8.4 and Codex F1 both proposed replacing the absolute `‖F‖` bar with a *mass-balance* gate (`reject reason-4 iff the booked mass error exceeds ε·throughput`). Drafting found this BREAKS the MMS tests: a manufactured-source MMS solve has its source balanced by **Dirichlet through-flux**, which no physical ledger or free-row-sum accounts for, so a correctly-converged MMS floor state (reason-4, `‖F‖≈1.1e-7`) shows a mass-balance gap of **~1e15** and the mass gate would WRONGLY REJECT it → `step(dt=1e8)` returns not-converged → MMS death-spiral. Probe (reproducible, `OverlandProblem` MMS-spatial nc=20/40): `int(f*) ≈ −1.0e7`, `mass_gap ≈ +1.0e15`, `‖F‖ ≈ 1.1e-7`. **Conclusion: the residual `‖F‖` IS the correct convergence invariant (small at the MMS floor); F1's real requirement is to make the `‖F‖` criterion SCALE-INVARIANT, not to swap it for mass-balance.** Mass-balance only works where there is no Dirichlet through-flux (the V, the coupled column) — not general, rejected as THE gate.
- **The replacement — a forcing/operator-normalized residual gate (leading candidate; Task A1 data fixes the exact scale):** accept a reason-4 stall iff `‖F‖ ≤ tol_rel · R_scale`, with `R_scale` a problem-natural residual scale that scales WITH the problem (so the same physical state at two mesh/domain scales gets the same decision — the F1 fix). Candidate `R_scale = max(‖b_forcing‖, ‖b_storage‖, atol_floor)`:
  - `‖b_forcing‖` = norm of the assembled rain/source residual vector (`assemble_vector(form(source·v·dx [+ rain·v·ds]))`). Huge for MMS (`f*` ~1e7) → its floor accepts the MMS state; this is exactly why MMS works.
  - `‖b_storage‖` = norm of the lumped storage residual `((d−d_n)/dt)·v·dx_storage` assembled at the candidate state (covers closed-slump / no-forcing cases where `‖b_forcing‖=0`).
  - `atol_floor` (~1e-12) a tiny absolute backstop for the fully-quiescent step.
  - `tol_rel` (~1e-3, DIMENSIONLESS) is the only tunable. **Task A1 MEASURES `‖F‖`, `‖b_forcing‖`, `‖b_storage‖` across EVERY reason-4 state in the suite + the V before fixing the `R_scale` form and `tol_rel`** — do not hardcode without the data.
- **Efficiency hypothesis to test (not assume):** if `R_scale` for the V is large enough that some currently-rejected dirty stalls fall under `tol_rel·R_scale`, the gate may accept mass-conserving stalls the absolute bar over-rejected (the 39.5 h / 60k-rejection cost in §8.4) and lift the dt-pin. Measure it (Task A5). If the V dirty stalls have `‖F‖/R_scale ≫ tol_rel`, the gate is "only" the portability fix and the dt-pin is pure Defect-A stiffness for O1 — report honestly either way.

**Working location:** worktree `C:\Users\arikt\Documents\GitHub\PIDS-b6-docs` on branch `b6-tilted-v-convergent-flow` (HEAD = the P0 work). All commits land here. Do NOT touch the main checkout `C:\Users\arikt\Documents\GitHub\PIDS` (concurrent M4 session).

**Baseline check (Task 0 — do this first):**
Run: `PATH=/root/miniforge3/envs/pids-fem/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/ -q --no-header`
Expected: `136 passed`. If not, STOP and reconcile before starting.

---

# Part A — scale-invariant (residual-normalized) acceptance gate

**Acceptance for Part A:** full suite green; the gate is provably scale-invariant (a test asserts the same physical state at two domain scales gets the same decision, and the old absolute bar provably failed it); the `R_scale` form + `tol_rel` are DATA-DRIVEN from the reason-4 census (not guessed); B4/B5/B6 conservation reproduced; efficiency measured.

### Task A1: Census the reason-4 population — the diagnostic that picks `R_scale` and `tol_rel`

**Files:** Create `scratch/_reason4_census.py`. No engine change yet.

**Why first:** the gate's correctness hinges on a scale that cleanly separates *floor* reason-4 states (MMS, near-flat, lake — must ACCEPT) from *dirty* reason-4 stalls (the convergent V — must REJECT), across all problem types AND mesh scales. Guessing `R_scale`/`tol_rel` without seeing the populations is how the absolute bar got stuck. Measure first.

**Step 1 — write the census.** For each of these states, drive one `step()` that lands on reason-4, and record `(case, scale, ‖F‖, ‖b_forcing‖, ‖b_storage‖, ‖F‖/R_scale)`:
- MMS spatial steady (nc = 20, 40, 80) — floor, Dirichlet source.
- MMS temporal (one stalled step) — floor.
- near-flat 1-D (`test_near_flat_no_spurious_flux` setup) — floor, no forcing.
- lake-at-rest — floor, no forcing.
- the 2-D convergent V at storm plateau, scale = 1.0 AND 0.1 (reuse `_v2d_overland_diag.py`) — DIRTY, two scales.
- the coupled column under rain — floor/working.
Assemble `‖b_forcing‖`/`‖b_storage‖` with `dolfinx.fem.petsc.assemble_vector(fem.form(...))` on the relevant source/storage UFL (the census script builds these directly; it is the prototype for the engine helper in A2).

**Step 2 — run and READ the table.**
Run: `... python scratch/_reason4_census.py`
Confirm: every FLOOR state has `‖F‖/R_scale ≤ 1e-3` with margin; every DIRTY V state has `‖F‖/R_scale ≫ 1e-3`; and the V's ratio is the SAME at scale 1.0 and 0.1 (scale-invariance, the F1 property the absolute bar lacks). If the candidate `R_scale = max(‖b_forcing‖,‖b_storage‖,atol)` does NOT separate them, adjust the form (e.g. include a conveyance-flux scale) and re-run BEFORE coding the gate. Record the chosen `R_scale` form + the `tol_rel` that sits in the gap (geometric mean of the two populations' `‖F‖/R_scale`).

**Step 3 — commit the census + the chosen constants** (documented in the script header):
```bash
git add scratch/_reason4_census.py
git commit -m "P1-A1: reason-4 census -> R_scale form + tol_rel chosen from data (floor vs dirty separation)"
```

### Task A2: Add the `R_scale` helper to both solver classes (no gate change yet)

**Files:** `pids_forward/physics/overland.py`, `pids_forward/physics/coupling.py`, `tests/test_step_acceptance.py`.

**Step 1 — failing test:** the helper returns the same `R_scale` the census measured for one known state, and is positive.
```python
def test_residual_scale_positive_and_matches_census():
    prob = _overland_blob()
    prob.step(1e-3)
    Rs = prob._residual_scale()          # NEW (Task A2)
    assert Rs > 0.0
    # forcing-dominated here (rain on the blob? if none, storage-dominated) -- assert it tracks
    # the assembled forcing/storage norms, the same quantities the census used:
    assert np.isfinite(Rs)
```

**Step 2 — run, verify fail** (`_residual_scale` missing).

**Step 3 — implement `_residual_scale()`** on both classes, returning `max(‖b_forcing‖, ‖b_storage‖, atol_floor)` from the A1-chosen form. Reuse compiled forms: in `__init__` build `self._storage_form = fem.form(((d - d_n)/dt)*v*dx_storage)` and a forcing form from the stored source/rain (overland: store `self._rain` in `add_rain` per Task A2b below; coupled: `self._rain` already stored, plus the `ds_top` rain term). Assemble with `assemble_vector`; norm via `vec.norm()`. Store `self._atol_floor`, `self._tol_rel` (from A1) as attributes.
- **A2b (overland source plumbing):** `OverlandProblem.add_rain` currently returns `r` without storing — add `self._rain = r` (keep the return). Constructor `source=` (MMS): store `self._source = source` so the forcing form can include it. The forcing form = `fem.form((rain_term + source_term)·v·measure)` rebuilt when rain/source changes.

**Step 4 — run, verify pass + full acceptance suite green** (helper added, gate unchanged):
Run: `... pytest tests/test_step_acceptance.py -q`

**Step 5 — commit:**
```bash
git commit -am "P1-A2: _residual_scale() helper (forcing/storage-normalized) on both solvers"
```

### Task A3: Switch the reason-4 criterion to `‖F‖ ≤ tol_rel·R_scale` (the F1 fix)

**Files:** `pids_forward/physics/overland.py`, `pids_forward/physics/coupling.py`, `tests/test_step_acceptance.py`.

**Step 1 — the behavioural tests (pin the INVARIANT, not a constant):**
```python
def test_floor_stall_accepted_residual_gate():
    """MMS / near-flat reason-4 (||F|| << R_scale) must BOOK -- the case mass-balance broke."""
    prob = _overland_mms_steady(nc=40)             # reason-4, ||F||~1.7e-7, ||forcing||~1e7
    converged, _ = prob.step(1e8)
    assert prob.last_reason == 4
    assert prob.last_fnorm <= prob._tol_rel * prob._residual_scale()
    assert converged

def test_dirty_V_stall_rejected_residual_gate():
    prob = _overland_blob(petsc_options=_STALL_OPTIONS)
    converged, _ = prob.step(1e-3)
    assert prob.last_reason == 4
    assert prob.last_fnorm > prob._tol_rel * prob._residual_scale()
    assert not converged

def test_gate_is_scale_invariant():
    """THE F1 fix: same physical state, two domain scales -> SAME decision.

    The 2-D V at scale 1.0 and 0.1 (geometrically similar, same slopes/rain) -- a reason-4 step's
    accept/reject must not depend on scale. The OLD absolute bar flips here (assert it would);
    the residual-normalized gate does not.
    """
    r1 = _run_one_stalled_step(scale=1.0)          # (reason, accepted, fnorm, Rscale)
    r0 = _run_one_stalled_step(scale=0.1)
    assert r1[0] == r0[0] == 4
    assert r1[1] == r0[1]                           # SAME decision
    # and prove the OLD bar would have flipped: the two ||F|| straddle a single absolute 3e-6:
    assert (r1[2] <= 3e-6) != (r0[2] <= 3e-6)
```
(Provide helpers `_overland_mms_steady`, `_run_one_stalled_step` in the test file.)

**Step 2 — run, verify fail** (criterion still absolute; the scale-invariance + floor tests fail).

**Step 3 — implement.** Add `self._last_Rscale = 0.0`. Replace the criterion in both `step()`:
```python
# reason 2/3 = residual-tested -> bookable. reason 4 = STAGNATION: bookable iff the residual is
# small RELATIVE TO the problem's natural residual scale (forcing/storage norm) -- SCALE-INVARIANT,
# unlike the old absolute stall_accept_fnorm bar (Codex P0 review F1; mass-balance gate rejected --
# it breaks Dirichlet-source MMS, see plan 2026-06-14 Part A background).
self._last_Rscale = self._residual_scale()
floor_ok = self.last_fnorm <= self._tol_rel * self._last_Rscale
converged = self.last_reason > 0 and (self.last_reason != 4 or floor_ok)
```
Keep `stall_accept_fnorm` and `last_fnorm` as recorded diagnostics (do not delete; the §8 narrative + audit reference them) — they no longer gate.

**Step 4 — run the acceptance suite.** New tests PASS. Update (don't delete) the P0 tests: `test_floor_stagnation_books` and `test_*_snorm_stall_is_rejected_not_booked` → assert via `_tol_rel·_residual_scale()`; `test_acceptance_contract_defaults` → DROP the `stall_accept_fnorm == 3e-6` constant-pin (F1: it locked the magic number), keep the behavioural assertions + assert `_tol_rel` is the A1 value.

**Step 5 — commit:**
```bash
git commit -am "P1-A3: reason-4 booking gated on ||F|| <= tol_rel*R_scale, scale-invariant (Codex F1)"
```

### Task A4: Full-suite regression + B4/B5/B6 decision-reproduction

**Files:** `scratch/_gate_ab.py`; fixes to engine/tests if a regression appears.

**Step 1 — full suite.**
Run: `... pytest tests/ -q --no-header`
Expected: `>=139 passed` (P0's 136 + the new tests; minus any merged-away). A conservation/coupling regression means the gate mis-decides on a real problem — debug via the A1 census, NOT by loosening `tol_rel`.

**Step 2 — A/B decision-reproduction probe.** `scratch/_gate_ab.py`: run the storm-window coupled V (reuse `_tiltedv_diag.py`) under the OLD absolute bar and the NEW residual gate; print per-step where the two decisions differ + confirm the engine books close in BOTH. Capture the newly-accepted-step count (feeds A5's efficiency read).

**Step 3 — commit:**
```bash
git add scratch/_gate_ab.py && git commit -m "P1-A4: A/B probe -- residual gate vs absolute bar decision diff (books close in both)"
```

### Task A5: Measure the efficiency effect (the hypothesis) — do NOT skip or assume

**Files:** scratch + a parent-plan §8.6 write-up.

**Step 1 — standalone v2d (≈30 s) + field-scale coupled (≈75 min) under the new gate.**
`scratch/_v2d_overland_diag.py 48 30 0.125`; `SCALE=0.1 ... scratch/_tiltedv_spike.py OUT=scratch/tiltedv_inhouse_s0.1_resgate.npz`. Record accepted/rejected, dt distribution, runtime, engine ledger gap; compare to P0 (field-scale −8.6e-12 books, §8.4).

**Step 2 — decide on the canonical re-run.** If rejections drop sharply (the dt-pin was partly the over-rejection), schedule ONE canonical full-window run (`scratch/_tiltedv_spike.py`, hours, background) as the P3 confirmation — do NOT block P1. If unchanged, the pin is pure Defect-A stiffness (O1's job); say so.

**Step 3 — write parent §8.6 "Residual-gate efficiency result" + commit:**
```bash
git add docs/plans/2026-06-11-overland-convergent-flow-stabilization.md && git commit -m "P1-A5: residual-gate efficiency measurement (field-scale + standalone)"
```

**>>> CHECKPOINT (Gate A): report to the architect.** Suite green, gate scale-invariant + data-driven + decision-reproducing, efficiency measured. Get sign-off before Part B.

---

# Part B — O1 upwind-mobility two-point edge-flux spike (standalone Module 2)

**Nature of Part B:** this is a **research spike**, not strict TDD — the scheme's exact form (slope treatment, selector width) has empirical degrees of freedom the spike resolves. Each task therefore has a **validation gate** (an analytic/reference comparison) rather than a pre-written unit test, and `eps_H`/`eps_S` choices are spike OUTPUTS. Keep everything in a NEW module so the validated `OverlandProblem` (galerkin path) is untouched and stays the MMS/regression reference.

**The scheme (the math the residual implements).** Surface head `H = z_b + d`. For each surface mesh EDGE `e=(i,j)` with geometric transmissibility `T_e ≥ 0`:
```
Q_e = T_e · M(d_up) · (H_i − H_j),   d_up = smoothed-upwind(d_i, d_j; H_i−H_j)
M(d) = SECONDS_PER_DAY · max(d,0)^{5/3} / ( n · ( ((H_i−H_j)/L_e)² + eps_S² )^{1/4} )    [Manning mobility, slope from the edge]
smoothed-upwind: w = ½(1 + tanh((H_i−H_j)/eps_H));  d_up = w·d_i + (1−w)·d_j   (C¹ in d -> FD-Jacobian-friendly)
```
Node residual (lumped storage, partition of unity): `R_i = (d_i − d_n,i)·A_i/dt + Σ_{e∋i} ±Q_e − r·A_i + outflow_i`, `A_i` the nodal control area, edge sign `+` for the `i`-row, `−` for `j` (telescoping → exact discrete conservation, structural). **Monotonicity** needs `T_e ≥ 0` (M-matrix): holds on structured box meshes; GUARD it (`assert T_e.min() >= -1e-14`) and document the obtuse-triangle caveat. Solve with PETSc `SNES` + residual callback + a **finite-difference colored Jacobian** (the ParFlow `UseJacobian False` precedent) — proving the scheme, not the Jacobian. Hand-Jacobian is a P2 optimization.

### Task B1: 1-D upwind core + SNES scaffold; lake-at-rest exact

**Files:** Create `pids_forward/physics/overland_upwind.py` (class `UpwindOverlandProblem`); Test `tests/test_overland_upwind.py`.

1-D first (interval mesh): `T_e = A_e/L_e` unambiguous (FD Laplacian), no M-matrix worry. Build the edge list from `mesh.topology` (`create_connectivity(1,0)`; in 1-D the cells ARE the edges). Implement `set_topography`, `set_initial_condition`, `add_rain`, `step`, `total_water`. **Validation gate (the test):** a flat lake `H = const` (set `z_b`+`d` so `z_b+d` uniform) stays EXACTLY at rest — `Q_e=0` structurally, `max|d−d₀| < 1e-14` after a step. Proves well-balancedness (H-differences, not d-differences).
```bash
git commit -am "P1-B1: UpwindOverlandProblem 1-D core + SNES(FD-Jac); lake-at-rest exact"
```

### Task B2: 1-D positivity WITHOUT a limiter + conservation + kinematic rising limb

**Files:** `pids_forward/physics/overland_upwind.py`, `tests/test_overland_upwind.py`.

Validation gates:
- **Positivity, no limiter:** a 1-D slump / wetting-front advance keeps `d.min() ≥ −1e-12` across the run with NO limiter in the class (there is none — the point). Contrast: galerkin `OverlandProblem` needs the limiter here.
- **Conservation:** closed domain (no rain/outflow) conserves `total_water()` to ~1e-13.
- **Kinematic rising limb:** 1-D plane + constant rain + normal-depth outlet reproduces the analytic kinematic-wave rising limb (reuse `test_kinematic_wave_plane_hydrograph_1d`'s reference) to the scheme's O(h).
```bash
git commit -am "P1-B2: 1-D upwind positivity-without-limiter + conservation + kinematic rising limb"
```

### Task B3: FD-Jacobian Newton health + the smoothed-selector width decision

**Files:** `pids_forward/physics/overland_upwind.py`, `scratch/_upwind_selector_probe.py`.

Empirical spike question (parent §5: smoothed vs semismooth). Probe Newton iterations + convergence across `eps_H ∈ {coarse→tight}` on the 1-D front. **Gate:** pick the `eps_H` giving robust Newton with minimal smearing; record the choice + trade in the module docstring. If smoothed+FD cannot converge the front, document it and note semismooth as the P2 fallback (the SCHEME is the deliverable; solver tuning is secondary).
```bash
git commit -am "P1-B3: smoothed upwind-selector width chosen empirically (eps_H); Newton health recorded"
```

### Task B4: 2-D extension — edge graph on the triangle V + M-matrix guard

**Files:** `pids_forward/physics/overland_upwind.py`, `tests/test_overland_upwind.py`.

Extend to 2-D (`create_rectangle`): edge→vertex graph; `T_e` (FV dual-mesh `T_e = (dual edge length)/L_e` for the structured box V); assert `T_e ≥ −1e-14` (M-matrix guard, loud otherwise). **Gates:** 2-D lake-at-rest exact; 2-D closed slump conserves + positive without a limiter (the `test_tilted_v_catchment_conserves_2d` scenario — assert the upwind path conserves AND has no negative excursions).
```bash
git commit -am "P1-B4: 2-D upwind edge-flux on the triangle V + M-matrix guard; lake/slump gates"
```

### Task B5: The 2-D V plateau → Q_eq, mesh convergence, dt-pin lifted (the decisive gate)

**Files:** `scratch/_v2d_upwind_V.py`, `tests/test_overland_upwind.py`.

Parent §5 P1 gates: **V plateau → Q_eq ± 3%, mesh-convergent; dt no longer pinned (measure).** Run the canonical 2-D V (the `_v2d_overland_diag.py` geometry/forcing) with `UpwindOverlandProblem` at 48×30 and 96×60. Record plateau Q/Q_eq, plateau oscillation RMS (parent §6.3 ≤2%), dt distribution (vs galerkin ~5e-5 pin), books gap, min(d). **Field-scale check:** SCALE=0.1 — galerkin gave 0.876·Q_eq (under-resolved); upwind should hold ≈1.0 at the same resolution (the accuracy claim, §8.3/F10). Add one Tier-1 test pinning the V plateau within ±3% Q_eq at 48×30.
```bash
git commit -am "P1-B5: 2-D V plateau ->Q_eq +-3% mesh-convergent, dt-pin lifted, field-scale accuracy"
```

### Task B6: Spike decision + parent-plan write-up

**Files:** `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` (new §8.7 "O1 spike result"); module docstring; memory.

Write the spike verdict against the parent §5 P1 gate list (each: pass/fail + number). State P2 productionization readiness (same edge scheme on the realization-A top-facet ridge graph in `CoupledProblem`; limiter → tripwire). Record carried decisions (eps_H, eps_S, `T_e` form, FD-vs-hand Jacobian) and any gate that did NOT meet bar (honest fail, characterized — the P0/M4 standard). Update memory `pids-convergent-flow-priority.md` (NEXT = P2).
```bash
git commit -am "P1-B6: O1 spike verdict vs the P1 gate list + P2 readiness; parent plan 8.7"
```

**>>> CHECKPOINT (Gate B / P1 complete):** use superpowers:finishing-a-development-branch.

---

## Acceptance bars (P1 definition of done)

**Part A (gate):**
1. Reason-4 booking gated on `‖F‖ ≤ tol_rel·R_scale` (dimensionless `tol_rel`, `R_scale` a problem-natural residual scale); `stall_accept_fnorm` demoted to a recorded diagnostic.
2. `R_scale` form + `tol_rel` chosen from the A1 census (floor vs dirty separation shown), NOT guessed.
3. Scale-invariance test passes (same physical state, two scales → same decision); the old absolute bar provably flipped.
4. MMS + near-flat + lake floor states still ACCEPT (the case mass-balance broke); full suite green; B4/B5/B6 conservation reproduced.
5. Efficiency measured (field-scale + standalone) and reported honestly.

**Part B (O1 spike, standalone Module 2):**
6. Lake-at-rest + near-flat exact (well-balanced); closed conservation machine-tight.
7. **d ≥ −1e-12 WITHOUT any limiter** (1-D and 2-D) — the monotonicity headline.
8. 1-D kinematic rising-limb analytic matched (to O(h)).
9. 2-D V plateau → Q_eq ± 3% at ≈48×30, mesh-convergent, oscillation ≤2% RMS; field-scale holds ≈1.0 where galerkin gave 0.876.
10. dt no longer pinned (measured vs galerkin ~5e-5).
11. The validated `OverlandProblem` galerkin path UNTOUCHED (new scheme = separate class); full suite green.

## Risks / open questions
- **`R_scale` may need a flux term.** If the A1 census shows `max(‖b_forcing‖,‖b_storage‖)` does not separate floor from dirty on some case (e.g. a no-rain dynamic redistribution at the plateau), add a conveyance-flux-norm term to `R_scale`. A1 is exactly where this is caught — do not skip it.
- **`tol_rel = ~1e-3` is a constant — but DIMENSIONLESS** (a fraction of the natural residual scale), so it is scale-invariant by construction; that is the F1 fix. Tune only against the floor-vs-dirty ratio, never an absolute norm.
- **Mass-balance is NOT used** (it breaks Dirichlet-source MMS, measured). Anyone tempted to "just check the books close" on reason-4 must re-read the Part A background.
- **T_e ≥ 0 (M-matrix) off structured meshes:** spike runs structured box V where it holds; guard fails loudly otherwise. Unstructured/obtuse-triangle transmissibility is P2/P3 (parent §7).
- **FD-Jacobian cost** at the 2-D V: fine for the spike's small meshes; the hand Jacobian (parent §4 O1) is the P2 performance item, not a P1 blocker.
- **O(h) upwind smearing** on the mm sheet: quantify in the B5 mesh-convergence study; FCT (parent O3) is the escape hatch only if it measurably hurts the swale answer (P4).

## Artifacts
- This plan; parent `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` (§8 P0 results, §8.6/§8.7 to be added).
- New: `pids_forward/physics/overland_upwind.py`, `tests/test_overland_upwind.py`; scratch `_reason4_census.py`, `_gate_ab.py`, `_upwind_selector_probe.py`, `_v2d_upwind_V.py`.
- Touched: `pids_forward/physics/{overland,coupling}.py` (Part A gate), `tests/test_step_acceptance.py`.
