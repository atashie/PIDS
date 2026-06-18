# Convergent-Flow P3 Implementation Plan — resolved-swale absolute accuracy + the permanent PIDS-swale fixture + the default-flip

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the convergent-flow workstream by proving the coupled upwind overland scheme is *absolutely accurate* on the geometry the product actually ships into (a RESOLVED finite-width swale, not the idealized kink-V), establishing that swale as a permanent Tier-2 benchmark fixture (with a surface inlet), re-benchmarking the tilted-V against ParFlow with upwind, and — once accuracy is confirmed — flipping the `CoupledProblem` default from galerkin to upwind.

**Architecture:** P3 is validation + productionization, not new physics. (A) Prove absolute accuracy via the DEPTH FIELD — the swale-floor depth → the analytic Manning normal-depth, mesh-convergent, on a resolved coupled swale, PLUS operator-equivalence to the validated standalone — NOT via the outlet discharge, which in the coupled engine is conservation-FORCED to Q_eq (it IS the sink; see Part A). This is the absolute-accuracy claim P1/P2 deferred (§8.7/§8.8). (B) Build the permanent PIDS-swale Tier-2 fixture (~50×30 m field, 2–5% side slopes → a 1–2% swale line, loam, `add_surface_inlet`, the storm matrix) with Tier-1/2 tests + a Tier-3 HTML. (C) Re-benchmark the tilted-V (canonical + field) upwind-vs-galerkin-vs-ParFlow. (D) Make upwind the default WHERE IT APPLIES via a dimension/comm-aware "auto" default (Arik-gated) + update benchmarks README / memory / a parent §8.9 verdict.

**Tech Stack:** Python, DOLFINx 0.10 + PETSc/petsc4py (WSL2 conda env `pids-fem`), pytest, Plotly (offline HTML). Run pattern (every command assumes this): from `technologies/infiltration-runoff-model/forward-model/`, with `PATH=/root/miniforge3/envs/pids-fem/bin:$PATH` (conda not on PATH non-interactively; `gcc` needed for FFCX JIT), `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, `PYTHONPATH=.`. The full suite's terminal summary line is suppressed by an M4 session-finish print — confirm via the process EXIT CODE (0 = green).

---

## Background & context (read before starting)

- **Parent plan:** `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` — §5 P3 bullet = the scope; §8.7 = the corrected P1 verdict (the accuracy framing); §8.8 = the P2 productionization verdict + the E1 disposition (the kink-V vs resolved-swale finding). **P2 plan:** `docs/plans/2026-06-16-overland-convergent-flow-P2.md`. **P1 plan:** `docs/plans/2026-06-14-overland-convergent-flow-P1.md`.
- **What P1/P2 established (so P3 inherits, not re-derives):**
  - The coupled upwind scheme (`CoupledProblem(overland_scheme="upwind")`) is SHIPPED, opt-in, galerkin-bit-identical, conserving, monotone, and ~600× faster than galerkin on the canonical tilted-V. Merged to `main` (`29cdbe6`); Tier-3 signed off (Arik 2026-06-18).
  - **The accuracy framing (CRITICAL — do NOT repeat the over-claim):** the LUMPED outlet discharge (`outflow_rate()`) is a CONSERVATION/EQUILIBRIUM identity (`Σ q·B = rain·area = Q_eq` for ANY converged steady field) — NOT discharge accuracy. **The genuine accuracy measure is the CONSISTENT ds-integral discharge** `∫ 86400·(1/n)·d^(5/3)·√S ds` over the outlet (a P1-interpolated functional of the depth field).
  - **The kink-V vs resolved-swale finding (B5b + P2-E1):** on the idealized tilted-V the valley is a measure-ZERO 1-cell kink → the consistent integral reads ~0.85 and DIVERGES under refinement, AND the coupled positivity undershoot is cm-scale (28.5 mm canonical) — a characterized ARTIFACT shared by galerkin, NOT a scheme defect. On a **RESOLVED finite-width swale** the consistent integral heals UPWARD to ~0.99 (≤~1%, B5b standalone: 0.971→0.982→0.990 over 48×30→96×60→192×120) and the undershoot drops to sub-mm (P2-E1: 0.5 mm). **P3's job is to confirm this in the COUPLED engine, on the product geometry, as a permanent fixture, and to make it the accuracy gate the default-flip rests on.**
  - The B5b lumped-vs-consistent distinction is a STANDALONE phenomenon (the standalone outlet sink is LUMPED node-weighted, so its `outflow_rate()` ≠ the free consistent integral, which reads ~0.85/0.99). **In the COUPLED engine that distinction does NOT transfer:** the coupled outlet sink IS the consistent ds-integral AND conservation-forced to Q_eq (Part A) — so there is no free consistent metric, and P3 does NOT add a consistent-discharge diagnostic; coupled accuracy is the DEPTH FIELD. The resolved-swale GEOMETRY methodology (`make_zb(W)` = flat-bottomed valley `z_b = SY·(LY−y) + SX·max(|x−XC|−W/2, 0)`) is reused from `scratch/_b5b_valley_concentration.py`.
- **The product geometry (parent §5 P3):** ~50×30 m field, 2–5% side slopes converging to a 1–2% swale line, realistic loam, the standard storm matrix, an `add_surface_inlet` in the swale — "the geometry the product actually ships into." This is SMALL (vs the 1.62 km tilted-V) → the swale is finite-width + naturally resolved.
- **The coupled APIs P3 uses (all on `main` post-merge):** `CoupledProblem(mesh, soil, overland_scheme="upwind", n_man=…)`; `set_topography`, `set_initial_condition`, `add_rain`, `add_outflow_bc`, `add_surface_inlet`, `add_drainage_bc`, `step`/`advance`; accounting `outflow_rate`/`sink_rates`/`cum_sinks`/`total_water`. The Tier-2 coupled pattern to mirror: `tests/test_coupling_3d_tier2.py` (storm phases on a 3-D hillslope) + the benchmark runners `viz/run_coupling_3d_sanity.py` + `benchmarks/{build,make}_comparison_*`.

**Decision points (resolve with Arik / refine during the work):**
- **(DP-A) The absolute-accuracy bar.** Parent §6 bar-3 is "±3% of Q_eq, error shrinking with h." The coupled accuracy is DEPTH-FIELD (the outlet is conservation-forced — Part A): propose the swale-floor depth within **≤~5% of the analytic Manning normal-depth at the product resolution, error SHRINKING with refinement** (the kink-V diverges). Task A2 measures it; do not hardcode the bar without the data.
- **(DP-B) The swale fixture specifics.** Parent gives ranges (~50×30 m, 2–5% sides, 1–2% swale, loam). Task B1 commits to concrete numbers (proposed: 50×30 m, SX=0.03 side, SY=0.015 valley, swale floor width W chosen so it spans **≥~6 cells across the floor** at the product mesh — state resolution in CELLS, not metres, Codex should-fix; loam Ks≈0.25; the `test_coupling_3d_tier2.py` storm matrix). Confirm with Arik — this becomes the PERMANENT fixture.
- **(DP-C) ParFlow-swale scope.** ParFlow has the tilted-V deck (`parflow/cases/tilted_v_catchment.py`) — Part C re-benchmarks THAT with upwind. A ParFlow *swale* deck is NEW work; propose deferring it (validate the swale via the depth-field normal-depth convergence + operator-equivalence to the standalone (Part A) + the existing 3-D coupled ParFlow hillslope `coupled_hillslope_3d.py`), with a ParFlow-swale deck as optional follow-up.
- **(DP-D) The default-flip + its BLAST RADIUS (Codex blocker).** Upwind is 3-D-only + serial-only (RAISES otherwise), so a blunt `galerkin → "upwind"` default breaks all 1-D/2-D/MPI callers. Options (Arik's call, only after Gate A/B + sign-off): **(a) an "auto" dimension/comm-aware default** = upwind in 3-D-serial else galerkin (RECOMMENDED — the fix by default where it works, zero breakage); (b) pin every non-3-D/non-serial caller to galerkin, then flip; (c) DON'T flip — keep galerkin default + upwind as the documented 3-D-serial opt-in, defer until upwind is 2-D/MPI-capable. Galerkin stays a permanent fallback in all cases.

**Working location.** Per Arik's "work from main": P3 executes on `main` in `C:\Users\arikt\Documents\GitHub\PIDS` (no worktree). ⚠️ **`main` is the pushed trunk** — committing WIP directly to it is unusual; keep every commit green (TDD), or (RECOMMENDED, confirm with Arik) use a short-lived `p3-…` branch merged when green. Either way, ONE folder. (Cleanup still pending from the P2 consolidation: the orphaned `PIDS-b6-docs` folder + the M4 untracked scratch + `M README.md` — unrelated to P3.)

**Baseline check (Task 0 — do this first):**
Run: `… python -m pytest tests/ -q --no-header -p no:cacheprovider` → expect **227 passed** (exit 0). If not, STOP and reconcile.

---

# Part A — the resolved-swale ABSOLUTE accuracy (DEPTH-FIELD; the coupled outlet is conservation-forced)

**Why first + THE KEY SUBTLETY (Codex review — do not repeat the §8.7 lumped-vs-consistent trap):** the
default-flip rests on the upwind scheme being *accurate* (not just conserving) on a resolved swale. **But
in the COUPLED engine the outlet `outflow_rate()` is the codim-2 consistent ds-integral AND it is
conservation-FORCED to ≈Q_eq** — it IS the outlet sink term, so summing the d-residual (the edge flux
telescopes to zero) gives `∫ q_out ds = rain·area − dStorage − ∫λ` at steady state. So
`outflow_rate() → Q_eq` proves CONSERVATION, not accuracy, and it CANNOT expose the kink-vs-swale
difference (which lives in the DEPTH FIELD — cf. the P2-E1 positivity: 28.5mm kink vs 0.5mm swale). The
standalone B5b distinction (lumped sink ≠ free consistent integral) does NOT transfer: the coupled sink
is already the consistent integral. **So Part A measures accuracy against EXTERNAL references of the
DEPTH FIELD: the analytic Manning normal-depth (A2) + operator-equivalence to the validated standalone
(A1).** All Part-A runs use **inlet OFF + near-impermeable bed + the surface export budget reported** (a
clean reference; raw rain·area is only valid when net soil exchange at plateau is negligible — Codex
should-fix). It's measurement + Tier-1 pins, no new physics.

### Task A1: Pin the conservation-forced outlet + operator-equivalence to the standalone

**Files:** `tests/test_coupling_upwind.py`; `scratch/_p3_swale_accuracy.py`.

**Why:** two guards that establish the accuracy footing — (i) DOCUMENT the trap so no one re-misuses the
outlet as accuracy; (ii) CARRY the validated standalone accuracy into the coupled engine.

**Step 1 — pin the trap (a documentation test):** the coupled `outflow_rate()` → ≈Q_eq at the storm
plateau REGARDLESS of swale resolution — run the kink-V (W=0) AND a resolved swale (W>0) on a
near-impermeable bed and assert BOTH outlet discharges ≈ Q_eq to ~1% (it is conservation-forced, NOT
accuracy; the kink artifact does NOT appear in the coupled outlet).
```python
def test_coupled_outlet_is_conservation_forced_not_accuracy():
    q_kink = _coupled_outlet_over_Qeq(W=0.0,   nx=..., ny=...)   # idealized kink valley
    q_swale = _coupled_outlet_over_Qeq(W=...,  nx=..., ny=...)   # resolved swale
    assert abs(q_kink - 1.0) < 0.02 and abs(q_swale - 1.0) < 0.02   # BOTH ~Q_eq -> conservation, not accuracy
```

**Step 2 — operator-equivalence (the INHERITED accuracy):** on the SAME resolved-swale surface +
near-impermeable bed (inlet off), the coupled-upwind surface depth field matches the STANDALONE
`UpwindOverlandProblem` depth field to tight tolerance (the coupling adds only the small λ
infiltration). The coupled path calls the SAME extracted edge kernel (P2-A1, bit-identical), so this
carries B5b's standalone resolved-swale accuracy (consistent integral ~0.99) into the coupled engine.
```python
def test_coupled_upwind_depth_matches_standalone_on_resolved_swale():
    d_coupled = _coupled_swale_plateau_depth(...)   # near-impermeable, inlet off
    d_standalone = _standalone_swale_plateau_depth(...)   # same surface geometry/forcing
    assert np.max(np.abs(d_coupled - d_standalone)) / d_standalone.max() < 0.05
```

**Step 3 — implement** the two tests + the `scratch/_p3_swale_accuracy.py` helpers (`_coupled_outlet_over_Qeq`, the coupled/standalone swale drivers).

**Step 4 — run, verify pass.**

**Step 5 — commit:** `git commit -am "P3-A1: pin the coupled outlet is conservation-forced (not accuracy) + coupled==standalone depth on the resolved swale"`

### Task A2: Depth-field absolute accuracy — swale-floor depth → analytic Manning normal-depth, mesh-convergent

**Files:** `scratch/_p3_swale_accuracy.py`; `tests/test_coupling_upwind.py`.

**Validation gate (the decisive depth-field accuracy result — the analytic, conservation-FREE reference):**
on a COUPLED resolved swale (flat valley floor of width W spanning **≥~6 CELLS across the floor** at the
product mesh — Codex should-fix: state resolution in cells, not metres; structured box top so the
M-matrix guard is satisfied), near-impermeable bed, **inlet OFF**, driven to the storm plateau, the
measured swale-floor depth — the **floor-MEAN / centerline plateau depth away from the swale edges**
(Codex: not a single extreme node, which carries geometric edge effects) — matches the analytic Manning
normal-depth `d_n = (q_w · n / √S_v)^(3/5)`, `q_w = Q_eq / W` (catchment discharge per unit swale width;
an asymptotic reference for a resolved flat floor), and the error **SHRINKS with mesh refinement** toward `d_n` (Codex should-fix: error-decreases, NOT "monotone
upward"). Contrast the **kink-V (W=1 cell): its floor depth GROWS with refinement** (the measure-zero
channel spike — the B5b/P2-E1 artifact). Report the surface export budget
(`cum_rain − cum_outflow − dStorage ≈ 0`) to confirm the reference is clean. Record the convergence
table; add one Tier-1 pin at the product resolution.
```python
def test_resolved_swale_floor_depth_matches_normal_depth_and_converges():
    # coupled upwind, resolved swale (>= ~6 cells across W), near-impermeable, inlet OFF, plateau:
    err_coarse, err_fine, dn = _swale_floor_depth_vs_normal(W=..., meshes=[(coarse), (fine)])
    assert err_fine < err_coarse              # error SHRINKS with refinement (mesh-convergent)
    assert err_fine < 0.05                    # floor depth within ~5% of analytic normal-depth d_n
```

**Step 3 — implement** `scratch/_p3_swale_accuracy.py` (the W / resolution sweep; the floor-depth-vs-`d_n`
convergence table; the kink-V divergence contrast; the export-budget check) + the Tier-1 pin.
```bash
git add scratch/_p3_swale_accuracy.py && git commit -am "P3-A2: coupled swale-floor depth -> analytic Manning normal-depth, mesh-convergent (depth-field absolute accuracy)"
```

**>>> CHECKPOINT (Gate A): report.** The depth-field absolute accuracy (floor depth → analytic normal-depth,
mesh-convergent; the coupled outlet is conservation-forced so it is NOT the metric) is what the
default-flip rests on — get Arik's read before building the permanent fixture.

---

# Part B — the permanent PIDS-swale Tier-2 fixture (the product geometry)

### Task B1: The swale fixture geometry + a reusable builder

**Files:** `pids_forward/physics/` or `viz/` helper `make_pids_swale(...)`; `tests/test_pids_swale.py` (new); `benchmarks/pids_swale_scope.md` (new, mirror `B5_coupled_3d_scope.md`).

Commit to the DP-B specifics (confirm with Arik): a ~50×30 m field, 2–5% side slopes converging to a
1–2% swale line of finite floor width W spanning **≥~6 cells** at the product mesh (Codex should-fix:
resolution in cells), loam, a host 3-D box (structured, so the top-facet M-matrix guard holds) meshed so
the top facet RESOLVES the swale floor. A builder returns a configured `CoupledProblem(overland_scheme="upwind")`
with `set_topography` (the swale `z_b`), the loam soil, an outlet at the swale's downstream end, and an
`add_surface_inlet` grate in the swale (the PRODUCT case — inlet ON; distinct from Part A's inlet-OFF
accuracy proof). **Tier-1 tests:** conservation with sinks (`Δtotal == cum_rain − cum_outflow − Σ cum_sinks`,
i.e. **outlet + inlet capture** vs the budget — Codex blocker: NOT outlet ≈ Q_eq alone, since the inlet
diverts flow), positivity (sub-mm, within the tripwire), the inlet captures water
(`cum_sinks["surface_inlet"] > 0` and rises with ponding). (Absolute discharge accuracy is Part A,
inlet off; here the swale carries Part A's validated depth field with the inlet diversion on top.)
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

A self-contained offline HTML for Arik: surface depth map d(x,y) over the storm (the convergence to the swale), the swale-floor depth vs the analytic normal-depth (the accuracy), the outlet hydrograph (conservation-forced) + the inlet capture, and conservation (`Δtotal = cum_rain − cum_outflow − Σ sinks`). **Arik visual sign-off** (the permanent fixture's Tier-3).
```bash
git add viz/make_pids_swale_html.py && git commit -am "P3-B3: PIDS-swale Tier-3 HTML (depth map + swale hydrograph + inlet capture) for sign-off"
```

**>>> CHECKPOINT (Gate B): report + Arik Tier-3 sign-off on the swale fixture.**

---

# Part C — re-benchmark the tilted-V vs ParFlow (upwind)

### Task C1: Upwind tilted-V comparison harness + HTML

**Files:** `benchmarks/build_comparison_tiltedv.py`, `benchmarks/make_comparison_tiltedv_html.py` (extend for the upwind in-house run); the coupled upwind tilted-V npz from `scratch/_tiltedv_diag.py OVERLAND_SCHEME=upwind` (P2-E1).

Build the canonical + field tilted-V comparison: **in-house upwind vs galerkin vs ParFlow** (`~/parflow-runs/tilted_v/summaries/`), outlet Q(t) vs Q_eq (note: conservation-forced), cumulative-outflow mass check, dt distribution (pin lifted), runtime (~600× win). State each parent §6 bar pass/fail. **Honest note:** the kink-V is the measure-zero-channel ARTIFACT geometry — the STANDALONE consistent ds-integral there reads ~0.85 (B5b), and the coupled outlet is conservation-forced; the accuracy point is the RESOLVED swale (Part A), not the idealized kink. Annotate that clearly.
```bash
git commit -am "P3-C1: tilted-V re-benchmark upwind-vs-galerkin-vs-ParFlow + HTML (kink artifact annotated; resolved swale = Part A)"
```

**>>> CHECKPOINT (Gate C): report.**

---

# Part D — the default-flip (DP-D) + workstream wrap-up

### Task D1: Make upwind the default WHERE IT APPLIES — a dimension/comm-aware "auto" default (Arik-gated)

**Files:** `pids_forward/physics/coupling.py`; the full test suite.

**Codex BLOCKER — the real blast radius.** Upwind RAISES for 2-D hosts (`NotImplementedError` at
construction) and for multi-rank (at solve). So a blunt default `"galerkin" → "upwind"` would BREAK
EVERY 1-D/2-D/MPI `CoupledProblem(...)` caller — `test_coupling_1d.py`, `test_coupling_2d.py`,
`test_engine_drains.py`, and any utility — not just "galerkin regression tests." Two safe paths:

**Recommended — an "auto" default (only after Gate A/B + Arik sign-off):** change the default to
`overland_scheme="auto"`, resolved in `__init__` to `"upwind"` iff (`mesh.topology.dim == 3` AND
`mesh.comm.size == 1`) else `"galerkin"`. 3-D-serial production gets the convergent-flow fix by default;
1-D/2-D/MPI transparently fall back to galerkin (no breakage). Keep `"galerkin"`/`"upwind"` as explicit
tested modes. **Codex note: keep the REQUESTED mode and the RESOLVED mode distinct** — store the input
(`self.overland_scheme = "auto"|"galerkin"|"upwind"`) AND the resolved `self._effective_overland_scheme
= "upwind"|"galerkin"`, and dispatch on the effective one — so tests/docs/diagnostics never conflate
"what was asked" with "what ran."
- Tests: a 3-D-serial default constructs upwind; a 2-D default constructs galerkin (no raise); the
  3-D coupled lateral-overland *accuracy* tests now assert upwind; the galerkin-*regression* tests are
  pinned explicitly `overland_scheme="galerkin"`.
- Full suite green (1-D/2-D/MPI callers NOT broken); standalone galerkin `OverlandProblem` + MMS untouched.

**Alternative (DP-D) if Arik prefers:** do NOT flip — keep `"galerkin"` default + `"upwind"` as the
documented 3-D-serial production opt-in — and defer any flip until upwind supports 2-D + MPI (the
scoping is lifted). Lowest risk; the convergent-flow fix is then opt-in, not on-by-default.
```bash
git commit -am "P3-D1: dimension/comm-aware 'auto' overland default (upwind in 3-D serial, galerkin elsewhere); suite green"
```

### Task D2: Wrap-up — benchmarks README, parent §8.9 verdict, memory

**Files:** `technologies/infiltration-runoff-model/benchmarks/README.md`; `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` (new §8.9 "P3 result / workstream close"); memory `pids-convergent-flow-priority.md`.

Write the P3 verdict against the parent §5 P3 + §6 acceptance bars (each pass/fail): resolved-swale absolute accuracy (swale-floor depth → analytic Manning normal-depth, mesh-convergent + operator-equivalence to the standalone); the permanent PIDS-swale Tier-2 fixture (signed off); the tilted-V re-benchmark; the default ("auto"). Record carried items + the honest kink-V artifact + that the coupled outlet is conservation-forced. Update memory → the convergent-flow workstream is COMPLETE.
```bash
git commit -am "P3-D2: P3 verdict + workstream close (parent 8.9); benchmarks README + memory updated"
```

**>>> CHECKPOINT (Gate D / P3 complete):** use superpowers:finishing-a-development-branch.

---

## Acceptance bars (P3 definition of done)
1. **The coupled outlet is documented + pinned as conservation-FORCED** (`outflow_rate() → ≈Q_eq` regardless of resolution) — NOT an accuracy metric. Accuracy is depth-field (bar 2). (Guards the §8.7 lumped-vs-consistent trap.)
2. **Resolved-swale absolute accuracy (depth-field):** coupled upwind swale-FLOOR depth → analytic Manning normal-depth `d_n`, error **SHRINKING with refinement** (≤~5% at the product resolution; kink-V diverges) — INLET OFF, near-impermeable, export budget reported; PLUS operator-equivalence to the standalone (inherits B5b ~0.99). Data-driven (Task A2), not assumed.
3. **The permanent PIDS-swale Tier-2 fixture** exists (product geometry + inlet ON — distinct from the inlet-OFF accuracy proof), with Tier-1 (conservation incl. `outlet + Σ sinks` vs the export budget, positivity, inlet capture) + Tier-2 (storm matrix) green + a Tier-3 HTML signed off by Arik.
4. **Tilted-V re-benchmark** (upwind vs galerkin vs ParFlow) built + HTML; the dt-pin/runtime win + the kink-V artifact stated honestly.
5. **Default handled (DP-D):** an "auto" dimension/comm-aware default (upwind in 3-D-serial else galerkin) — OR deferred; either way **no 1-D/2-D/MPI caller is broken**, galerkin kept as a tested fallback, full suite green, standalone galerkin/MMS untouched.
6. **Workstream closed:** parent §8.9 verdict + benchmarks README + memory updated.

## Risks / open questions
- **The coupled OUTLET is conservation-forced** (Codex blocker) — the #1 trap (the §8.7 over-claim's coupled form). `outflow_rate() → Q_eq` ALWAYS at steady state (it IS the sink); accuracy is the DEPTH FIELD (normal-depth + operator-equivalence), NOT any outlet integral. Re-read §8.7 + Part A's framing if tempted.
- **Swale resolution in CELLS** — the floor width W must span ≥~6 cells at the product mesh for the depth field to resolve (B5b). If the product mesh can't resolve W, that's a real meshing requirement to document, not a scheme failure. The M-matrix guard is on the surface triangulation (structured box top fine).
- **Inlet vs accuracy** (Codex blocker) — `add_surface_inlet` removes surface water (booked under sinks), so the inlet-ON fixture's outlet < Q_eq. Run the accuracy proof (Part A) with the inlet OFF; the fixture's conservation check (Part B) is `outlet + Σ sinks` vs the export budget, not outlet alone.
- **The default-flip blast radius** (Codex blocker) — upwind RAISES for 2-D + MPI, so flipping the default breaks ALL 1-D/2-D/MPI callers, not just galerkin-regression tests. Use the "auto" dimension/comm-aware default (or pin-all, or defer) — Task D1 / DP-D.
- **ParFlow-swale deck** is out of scope (DP-C); the swale is validated via the depth-field normal-depth convergence + operator-equivalence to the standalone + the existing ParFlow hillslope. Add a ParFlow-swale deck later if a cross-code head-to-head is wanted.
- **Working on `main`** — keep every commit green (TDD); or use a short-lived p3 branch (confirm with Arik).
- **Coupled Newton health on stiff fronts** (the P2-E1 residual item) — if the swale storm matrix shows heavy rejections, the controller/linesearch tuning surfaces here; not a correctness blocker.

## Artifacts
- This plan; parent `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` (§8.9 to add); P1/P2 plans; `scratch/_b5b_valley_concentration.py` (the resolved-swale geometry + depth-field methodology); `scratch/_tiltedv_diag.py` (the upwind tilted-V runner, `SWALE_W` knob).
- New: `tests/test_pids_swale.py`; `viz/make_pids_swale_html.py`; `benchmarks/pids_swale_scope.md`; scratch `_p3_swale_accuracy.py`, `_p3_swale_storm_matrix.py`; the swale-fixture builder + the depth-field accuracy / operator-equivalence tests in `tests/test_coupling_upwind.py`; the "auto" default in `coupling.py`.
- Touched: `benchmarks/{build,make}_comparison_tiltedv*.py`, `benchmarks/README.md`, `coupling.py` (default-flip).
