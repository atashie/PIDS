# Convergent-Flow P3 Implementation Plan — resolved-swale absolute accuracy + the permanent PIDS-swale fixture + the default-flip

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the convergent-flow workstream by proving the coupled upwind overland scheme is *absolutely accurate* on the geometry the product actually ships into (a RESOLVED finite-width swale, not the idealized kink-V), establishing that swale as a permanent Tier-2 benchmark fixture (with a surface inlet), re-benchmarking the tilted-V against ParFlow with upwind, and — once accuracy is confirmed — flipping the `CoupledProblem` default from galerkin to upwind.

**Architecture:** P3 is validation + productionization, not new physics. (A) Add a CONSISTENT ds-integral discharge diagnostic (the genuine accuracy metric, vs the lumped conservation-identity) and show it → Q_eq, mesh-convergent, on a resolved coupled swale — the absolute-accuracy claim P1/P2 deferred (§8.7/§8.8). (B) Build the permanent PIDS-swale Tier-2 fixture (~50×30 m field, 2–5% side slopes → a 1–2% swale line, loam, `add_surface_inlet`, the storm matrix) with Tier-1/2 tests + a Tier-3 HTML. (C) Re-benchmark the tilted-V (canonical + field) upwind-vs-galerkin-vs-ParFlow. (D) Flip the default to upwind (Arik-gated) + update benchmarks README / memory / a parent §8.9 verdict.

**Tech Stack:** Python, DOLFINx 0.10 + PETSc/petsc4py (WSL2 conda env `pids-fem`), pytest, Plotly (offline HTML). Run pattern (every command assumes this): from `technologies/infiltration-runoff-model/forward-model/`, with `PATH=/root/miniforge3/envs/pids-fem/bin:$PATH` (conda not on PATH non-interactively; `gcc` needed for FFCX JIT), `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, `PYTHONPATH=.`. The full suite's terminal summary line is suppressed by an M4 session-finish print — confirm via the process EXIT CODE (0 = green).

---

## Background & context (read before starting)

- **Parent plan:** `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` — §5 P3 bullet = the scope; §8.7 = the corrected P1 verdict (the accuracy framing); §8.8 = the P2 productionization verdict + the E1 disposition (the kink-V vs resolved-swale finding). **P2 plan:** `docs/plans/2026-06-16-overland-convergent-flow-P2.md`. **P1 plan:** `docs/plans/2026-06-14-overland-convergent-flow-P1.md`.
- **What P1/P2 established (so P3 inherits, not re-derives):**
  - The coupled upwind scheme (`CoupledProblem(overland_scheme="upwind")`) is SHIPPED, opt-in, galerkin-bit-identical, conserving, monotone, and ~600× faster than galerkin on the canonical tilted-V. Merged to `main` (`29cdbe6`); Tier-3 signed off (Arik 2026-06-18).
  - **The accuracy framing (CRITICAL — do NOT repeat the over-claim):** the LUMPED outlet discharge (`outflow_rate()`) is a CONSERVATION/EQUILIBRIUM identity (`Σ q·B = rain·area = Q_eq` for ANY converged steady field) — NOT discharge accuracy. **The genuine accuracy measure is the CONSISTENT ds-integral discharge** `∫ 86400·(1/n)·d^(5/3)·√S ds` over the outlet (a P1-interpolated functional of the depth field).
  - **The kink-V vs resolved-swale finding (B5b + P2-E1):** on the idealized tilted-V the valley is a measure-ZERO 1-cell kink → the consistent integral reads ~0.85 and DIVERGES under refinement, AND the coupled positivity undershoot is cm-scale (28.5 mm canonical) — a characterized ARTIFACT shared by galerkin, NOT a scheme defect. On a **RESOLVED finite-width swale** the consistent integral heals UPWARD to ~0.99 (≤~1%, B5b standalone: 0.971→0.982→0.990 over 48×30→96×60→192×120) and the undershoot drops to sub-mm (P2-E1: 0.5 mm). **P3's job is to confirm this in the COUPLED engine, on the product geometry, as a permanent fixture, and to make it the accuracy gate the default-flip rests on.**
  - The consistent-discharge methodology is in `scratch/_b5b_valley_concentration.py` (standalone): `make_zb(W)` = a flat-bottomed valley `z_b = SY·(LY−y) + SX·max(|x−XC|−W/2, 0)`; the consistent functional = the galerkin `OverlandProblem.outflow_rate()` (= the ds-integral). For the upwind COUPLED path, P3 adds a consistent-discharge diagnostic (the upwind `outflow_rate()` is the LUMPED one).
- **The product geometry (parent §5 P3):** ~50×30 m field, 2–5% side slopes converging to a 1–2% swale line, realistic loam, the standard storm matrix, an `add_surface_inlet` in the swale — "the geometry the product actually ships into." This is SMALL (vs the 1.62 km tilted-V) → the swale is finite-width + naturally resolved.
- **The coupled APIs P3 uses (all on `main` post-merge):** `CoupledProblem(mesh, soil, overland_scheme="upwind", n_man=…)`; `set_topography`, `set_initial_condition`, `add_rain`, `add_outflow_bc`, `add_surface_inlet`, `add_drainage_bc`, `step`/`advance`; accounting `outflow_rate`/`sink_rates`/`cum_sinks`/`total_water`. The Tier-2 coupled pattern to mirror: `tests/test_coupling_3d_tier2.py` (storm phases on a 3-D hillslope) + the benchmark runners `viz/run_coupling_3d_sanity.py` + `benchmarks/{build,make}_comparison_*`.

**Decision points (resolve with Arik / refine during the work):**
- **(DP-A) The absolute-accuracy bar.** Parent §6 bar-3 is "±3% of Q_eq, error shrinking with h, oscillation ≤2% RMS." For the consistent ds-integral on the resolved swale, propose **≤3% from Q_eq at the product resolution, converging upward toward ~1% with refinement** (matching B5b). Task A1 measures it; do not hardcode the bar without the data.
- **(DP-B) The swale fixture specifics.** Parent gives ranges (~50×30 m, 2–5% sides, 1–2% swale, loam). Task B1 commits to concrete numbers (proposed: 50×30 m, SX=0.03 side, SY=0.015 valley, swale floor width W≈6–10 m so it spans several cells at the product mesh, loam Ks≈0.25, the `test_coupling_3d_tier2.py` storm matrix). Confirm with Arik — this becomes the PERMANENT fixture.
- **(DP-C) ParFlow-swale scope.** ParFlow has the tilted-V deck (`parflow/cases/tilted_v_catchment.py`) — Part C re-benchmarks THAT with upwind. A ParFlow *swale* deck is NEW work; propose deferring it (validate the swale against Q_eq + the consistent-integral convergence + the existing 3-D coupled ParFlow hillslope `coupled_hillslope_3d.py`), with a ParFlow-swale deck as optional follow-up.
- **(DP-D) The default-flip (the workstream's closing act).** Flip `CoupledProblem` default `galerkin → "upwind"` ONLY after Part A/B confirm the resolved-swale accuracy + Arik signs off; keep galerkin as a PERMANENT fallback mode. Part D proposes it; Arik decides.

**Working location.** Per Arik's "work from main": P3 executes on `main` in `C:\Users\arikt\Documents\GitHub\PIDS` (no worktree). ⚠️ **`main` is the pushed trunk** — committing WIP directly to it is unusual; keep every commit green (TDD), or (RECOMMENDED, confirm with Arik) use a short-lived `p3-…` branch merged when green. Either way, ONE folder. (Cleanup still pending from the P2 consolidation: the orphaned `PIDS-b6-docs` folder + the M4 untracked scratch + `M README.md` — unrelated to P3.)

**Baseline check (Task 0 — do this first):**
Run: `… python -m pytest tests/ -q --no-header -p no:cacheprovider` → expect **227 passed** (exit 0). If not, STOP and reconcile.

---

# Part A — the resolved-swale ABSOLUTE accuracy (the deferred P1/P2 validation)

**Why first:** this is the claim the whole default-flip rests on — that the upwind scheme is *accurate* (not just conserving) on a resolved swale, in the COUPLED engine. It's a measurement + a Tier-1 pin, no new physics.

### Task A1: A consistent ds-integral discharge diagnostic for the coupled path

**Files:** `pids_forward/physics/coupling.py` (add `consistent_outflow_rate()`); `tests/test_coupling_upwind.py`.

**Why:** the coupled `outflow_rate()` is the LUMPED conservation-identity discharge (§8.8). The accuracy metric is the CONSISTENT functional `∫ 86400·(1/n)·d_pos^(5/3)·√slope ds` over the outlet codim-2 measure (the depth field interpolated, à la the galerkin `OverlandProblem.outflow_rate`). Add it as a diagnostic (read-only; does not change the solve).

**Step 1 — failing test:** on a SMOOTH coupled hillslope at a ponded plateau, `consistent_outflow_rate()` is positive, finite, and within a few % of the lumped `outflow_rate()` (they agree when the depth field is well-resolved/smooth; they diverge only on a 1-cell channel).
```python
def test_consistent_outflow_rate_matches_lumped_on_smooth_hillslope():
    prob = CoupledProblem(_box(8, 4, 4), SOIL, overland_scheme="upwind")
    prob.set_topography(lambda x: 0.05 * (2.0 - x[0]))
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[0], d_value=0.0)
    prob.add_rain(0.4); prob.add_outflow_bc(lambda x: np.isclose(x[0], 2.0), slope=0.05)
    prob.advance(t_end=0.1, dt=1e-3, dt_max=0.02)
    qc = prob.consistent_outflow_rate(); ql = prob.outflow_rate()
    assert qc > 0 and np.isfinite(qc)
    assert abs(qc - ql) / max(ql, 1e-12) < 0.15   # smooth: consistent ≈ lumped (no 1-cell spike)
```

**Step 2 — run, verify fail** (method missing).

**Step 3 — implement `consistent_outflow_rate()`** on `CoupledProblem`: assemble `∫ q_out·ds` over the outlet codim-2 measure(s) with `q_out = SECONDS_PER_DAY·(1/n_man)·max(d,0)^(5/3)·√slope` (the SAME integrand as `_finalize_forms`' outlet sink, but as a STANDALONE diagnostic functional of the SOLVED `d`, not the residual sink). Reuse the stored `_outlets` (locator/slope) + `out_measure`. Mirror the galerkin `OverlandProblem.outflow_rate` ds-integral.

**Step 4 — run, verify pass.**

**Step 5 — commit:** `git commit -am "P3-A1: consistent ds-integral discharge diagnostic on CoupledProblem (the accuracy metric)"`

### Task A2: Resolved-swale accuracy — consistent discharge → Q_eq, mesh-convergent (coupled)

**Files:** `scratch/_p3_swale_accuracy.py`; `tests/test_coupling_upwind.py`.

**Validation gate (the decisive P3 accuracy result):** on a COUPLED resolved swale (a flat-bottomed valley, `z_b` à la B5b `make_zb(W)` with W spanning several cells, near-impermeable bed so ≈all rain runs off), drive to the storm plateau and measure the CONSISTENT discharge / Q_eq at increasing cells-across-floor. **Expect it to converge UPWARD toward ~1.0 (≤~3% at the product resolution, → ~1% refined)** — the B5b standalone result reproduced in the coupled engine. Contrast the kink-V (W=0) which stays ~0.85 and diverges. Record the convergence table.
```python
def test_resolved_swale_consistent_discharge_approaches_Qeq():
    # coupled upwind, flat-floor valley resolved by >= ~6 cells, near-impermeable bed, storm plateau:
    q_over_Qeq = _run_coupled_swale_plateau(W=..., nx=..., ny=...)   # consistent / Q_eq
    assert 0.95 < q_over_Qeq < 1.03           # resolved swale: accurate (vs kink-V ~0.85)
```

**Step 3 — implement** `scratch/_p3_swale_accuracy.py` (the W / resolution sweep, the convergence table) + the Tier-1 pin at the product resolution.
```bash
git add scratch/_p3_swale_accuracy.py && git commit -am "P3-A2: coupled resolved-swale consistent discharge -> Q_eq, mesh-convergent (the absolute-accuracy gate)"
```

**>>> CHECKPOINT (Gate A): report.** The absolute accuracy on a resolved swale (consistent discharge → Q_eq, mesh-convergent) is the result the default-flip rests on — get Arik's read before building the permanent fixture.

---

# Part B — the permanent PIDS-swale Tier-2 fixture (the product geometry)

### Task B1: The swale fixture geometry + a reusable builder

**Files:** `pids_forward/physics/` or `viz/` helper `make_pids_swale(...)`; `tests/test_pids_swale.py` (new); `benchmarks/pids_swale_scope.md` (new, mirror `B5_coupled_3d_scope.md`).

Commit to the DP-B specifics (confirm with Arik): a ~50×30 m field, 2–5% side slopes converging to a 1–2% swale line of finite floor width W (several cells), loam, a host 3-D box meshed so the top facet RESOLVES the swale floor. A builder returns a configured `CoupledProblem(overland_scheme="upwind")` with `set_topography` (the swale `z_b`), the loam soil, an outlet at the swale's downstream end, and an `add_surface_inlet` grate in the swale. **Tier-1 tests:** conservation (closed → Δtotal = cum_rain), positivity (sub-mm, within tripwire), the inlet captures water (`cum_sinks["surface_inlet"] > 0`), consistent discharge → Q_eq.
```bash
git commit -am "P3-B1: the permanent PIDS-swale fixture (50x30m, 2-5% sides -> 1-2% swale, loam, inlet) + Tier-1 tests"
```

### Task B2: The Tier-2 storm matrix on the swale

**Files:** `tests/test_pids_swale.py` (or `_tier2`); `scratch/_p3_swale_storm_matrix.py`.

Run the standard storm matrix (typical + 100-yr extreme; dry/normal/wet antecedent — mirror `test_coupling_3d_tier2.py` / the B4/B5 matrices) on the swale fixture. **Tier-2 acceptance:** mass-conservative every scenario; physically plausible (inlet capture rises with ponding; recession re-infiltrates); the upwind tripwire never trips beyond the characterized sub-cm band; solver completes (record dt/iters/rejections).
```bash
git add scratch/_p3_swale_storm_matrix.py && git commit -am "P3-B2: PIDS-swale Tier-2 storm matrix (typical + extremes) -- conservation + plausibility"
```

### Task B3: Tier-3 HTML for the swale fixture

**Files:** `viz/make_pids_swale_html.py` (or reuse `make_coupling_3d_html.py`); data + HTML under `validation/sanity/`.

A self-contained offline HTML for Arik: surface depth map d(x,y) over the storm (the convergence to the swale), the swale hydrograph (consistent discharge vs Q_eq), the inlet capture, conservation. **Arik visual sign-off** (the permanent fixture's Tier-3).
```bash
git add viz/make_pids_swale_html.py && git commit -am "P3-B3: PIDS-swale Tier-3 HTML (depth map + swale hydrograph + inlet capture) for sign-off"
```

**>>> CHECKPOINT (Gate B): report + Arik Tier-3 sign-off on the swale fixture.**

---

# Part C — re-benchmark the tilted-V vs ParFlow (upwind)

### Task C1: Upwind tilted-V comparison harness + HTML

**Files:** `benchmarks/build_comparison_tiltedv.py`, `benchmarks/make_comparison_tiltedv_html.py` (extend for the upwind in-house run); the coupled upwind tilted-V npz from `scratch/_tiltedv_diag.py OVERLAND_SCHEME=upwind` (P2-E1).

Build the canonical + field tilted-V comparison: **in-house upwind vs galerkin vs ParFlow** (`~/parflow-runs/tilted_v/summaries/`), Q(t) vs Q_eq, cumulative-outflow mass check, dt distribution (pin lifted), runtime (~600× win), and the consistent-vs-lumped discharge (the kink-V artifact annotated). State each parent §6 bar pass/fail. **Honest note:** the kink-V consistent discharge stays ~0.85 (the artifact) — the point is the RESOLVED swale (Part A), not the idealized kink; say so.
```bash
git commit -am "P3-C1: tilted-V re-benchmark upwind-vs-galerkin-vs-ParFlow + HTML (kink artifact annotated; resolved swale = Part A)"
```

**>>> CHECKPOINT (Gate C): report.**

---

# Part D — the default-flip (DP-D) + workstream wrap-up

### Task D1: Flip the `CoupledProblem` default to upwind (Arik-gated)

**Files:** `pids_forward/physics/coupling.py`; the full test suite.

ONLY after Gate A/B confirm the resolved-swale accuracy + Arik signs off: change the `overland_scheme` default `"galerkin" → "upwind"`. Keep `"galerkin"` as a permanent, tested fallback mode. **This will change galerkin-default coupled tests** that exercise lateral overland — update them to assert the upwind behavior (or pin them to `overland_scheme="galerkin"` explicitly where they are galerkin-regression tests). Full suite green; the standalone galerkin `OverlandProblem` + MMS paths untouched.
```bash
git commit -am "P3-D1: flip CoupledProblem default to overland_scheme='upwind' (galerkin kept as fallback); suite green"
```

### Task D2: Wrap-up — benchmarks README, parent §8.9 verdict, memory

**Files:** `technologies/infiltration-runoff-model/benchmarks/README.md`; `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` (new §8.9 "P3 result / workstream close"); memory `pids-convergent-flow-priority.md`.

Write the P3 verdict against the parent §5 P3 + §6 acceptance bars (each pass/fail): resolved-swale absolute accuracy (consistent → Q_eq, mesh-convergent); the permanent PIDS-swale Tier-2 fixture (signed off); the tilted-V re-benchmark; the default-flip. Record carried items + the honest kink-V artifact. Update memory → the convergent-flow workstream is COMPLETE.
```bash
git commit -am "P3-D2: P3 verdict + workstream close (parent 8.9); benchmarks README + memory updated"
```

**>>> CHECKPOINT (Gate D / P3 complete):** use superpowers:finishing-a-development-branch.

---

## Acceptance bars (P3 definition of done)
1. **Consistent-discharge diagnostic** added to `CoupledProblem` (the accuracy metric vs the lumped identity); FD/analytic-free, read-only.
2. **Resolved-swale absolute accuracy:** coupled upwind consistent discharge → Q_eq within ≤3% at the product resolution, **converging upward** toward ~1% with refinement (vs the kink-V ~0.85 artifact); data-driven (Task A2), not assumed.
3. **The permanent PIDS-swale Tier-2 fixture** exists (product geometry + inlet), with Tier-1 (conservation, positivity, inlet capture, accuracy) + Tier-2 (storm matrix) green + a Tier-3 HTML signed off by Arik.
4. **Tilted-V re-benchmark** (upwind vs galerkin vs ParFlow) built + HTML; the dt-pin/runtime win + the kink-V artifact stated honestly.
5. **Default-flip** to upwind done (Arik-gated), galerkin kept as a tested fallback; full suite green; standalone galerkin/MMS untouched.
6. **Workstream closed:** parent §8.9 verdict + benchmarks README + memory updated.

## Risks / open questions
- **Consistent vs lumped discharge confusion** — the #1 trap of this workstream (the P1 over-claim). Always report the CONSISTENT ds-integral for accuracy; the lumped is conservation only. Re-read §8.7 if tempted.
- **Swale resolution** — the floor width W must span enough cells at the product mesh for the consistent discharge to heal (B5b: several cells). If the product mesh can't resolve W, that's a real meshing requirement to document, not a scheme failure.
- **The default-flip breaks galerkin-default coupled tests** — expected; update them deliberately (Task D1), distinguishing accuracy-tests (assert upwind) from galerkin-regression-tests (pin galerkin).
- **ParFlow-swale deck** is out of scope (DP-C); the swale is validated vs Q_eq + the consistent-integral convergence + the existing ParFlow hillslope. Add a ParFlow-swale deck later if a head-to-head is wanted.
- **Working on `main`** — keep every commit green (TDD); or use a short-lived p3 branch (confirm with Arik).
- **Coupled Newton health on stiff fronts** (the P2-E1 residual item) — if the swale storm matrix shows heavy rejections, the controller/linesearch tuning surfaces here; not a correctness blocker.

## Artifacts
- This plan; parent `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` (§8.9 to add); P1/P2 plans; `scratch/_b5b_valley_concentration.py` (the consistent-discharge methodology); `scratch/_tiltedv_diag.py` (the upwind tilted-V runner).
- New: `pids_forward/physics/coupling.py::consistent_outflow_rate`; `tests/test_pids_swale.py`; `viz/make_pids_swale_html.py`; `benchmarks/pids_swale_scope.md`; scratch `_p3_swale_accuracy.py`, `_p3_swale_storm_matrix.py`.
- Touched: `benchmarks/{build,make}_comparison_tiltedv*.py`, `benchmarks/README.md`, `coupling.py` (default-flip).
