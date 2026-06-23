# Sequential (operator-split) overland flow — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: when executing this plan, use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement it task-by-task, and superpowers:test-driven-development for every code task in Part B.

**Goal:** Replace the overland↔subsurface coupling for PIDS with a sequential, time-lagged operator split — keep the implicit Richards solve, move the surface water with a separate explicit mass-conserving downslope routing sweep, and couple them through a *water-level* (not forced-rate) infiltration handoff — so the convergent-flow + stiff-clay case that broke both existing schemes runs without timestep collapse or sawtooth.

**Architecture:** Each timestep: (1) solve Richards implicitly with the surface ponded depth supplied as a self-limiting top boundary (the existing `add_ponding_bc`), so the soil computes its own infiltration (no overshoot); (2) read the leftover ponded depth at each surface node; (3) route that leftover downhill explicitly over a flow-direction graph **at a Manning rate limit** — so water advances a bounded, dt-controlled distance per step (toward its immediate downslope neighbours, not straight to the outlet), which is exactly what lets it infiltrate as run-on en route — conserving mass; (4) the redistributed depths become next step's run-on. The stiff subsurface never shares a solver Jacobian with the surface routing, and there is no surface PDE — so the two *original* failure modes are structurally removed; the split instead trades them for a new, bounded, dt-controlled coupling error that the spike must verify (not eliminate by fiat). The new scheme is a **selectable option** that coexists with the validated galerkin/upwind monolithic schemes (which stay as the fallback), exactly as upwind was added beside galerkin.

**Tech stack:** Python, DOLFINx 0.10 / PETSc (WSL2 conda env `pids-fem`), NumPy. Reuses `richards.py` (`RichardsProblem` + `add_ponding_bc`), `overland_edge_kernel.build_top_facet_edge_graph` (the surface connectivity graph), and the accounting patterns in `coupling.py`. New code: a routing-sweep kernel + a thin orchestrator that alternates the Richards solve and the sweep.

**Plain-language frame (for Arik):** We stop solving the surface and the underground as one giant linked calculation. Instead, each step the soil decides how much water soaks in (given how deep the standing water is), then we let whatever's left run downhill to the next parcel — a simple, conserve-the-water bookkeeping sweep. This is how CATHY and six other research models do it, and it sidesteps both numerical traps we hit.

---

## 0. Decisions locked before we start (and why)

These are settled by the decision record (`docs/plans/2026-06-22-overland-flow-sequential-coupling-decision.md`), the two surveys, the prior engine-evaluation memo (`docs/plans/2026-06-03-pids-forward-model-engine-evaluation.md` §7), and the code read below. They are **not** open for the spike to relitigate — the spike tests whether this design *works*, not which design.

1. **Coupling = sequential, time-lagged operator split (CATHY-style).** Richards stays implicit and is solved **alone** (just ψ). The surface update is separate. They are not co-solved in one Newton.
2. **Handoff = water-level, self-limiting — NOT a forced/lagged rate.** This is the crux that keeps us consistent with the §7 catch-valve probe (a lagged *flux* blew up to 50 m; a *level* handoff where the implicit soil computes the flux cannot). Concretely: the surface ponded depth enters Richards through `add_ponding_bc`'s `max(ψ,0)` storage term (when ponded the surface head is the water depth → soil self-limits; when dry, rain enters as a flux capped by capacity). The smooth `max(ψ,0)` *is* the Neumann↔Dirichlet switch, done implicitly.
3. **Lateral routing = explicit, mass-conserving downslope sweep, rate-limited by Manning.** MFD receiver proportions (`square_root_of_slope`) + topological (upslope-before-downslope) node order + a **per-node Manning rate limit** (the template's per-node depth law) + a depression-fill pre-pass. Built on the existing top-facet graph. The rate limit is **load-bearing, not optional** (Codex finding 2): without it the sweep drains a parcel's whole inventory to the outlet in one step (teleportation), and run-on infiltration — a *core* PIDS use case (the sand channel intercepting runoff) — could never form. No surface PDE ⇒ no sawtooth; convergence ("many cells into one") is the sweep's native operation.
4. **Coexistence, not rip-out.** The new scheme ships as a selectable option; the galerkin/upwind monolithic `CoupledProblem` stays the validated fallback until the new scheme is re-validated against PIDS's bar.
5. **Embedded features / catch-valve stay implicit.** The §7 finding still binds for *those* sub-grid exchanges (reverse catch-valve, `WellIndexExchange`): they remain in-solver (BC/VI/prescribed-rate inside the Richards Newton). The overland split does not touch them and must not be read as licence to split *them* out.
6. **Honest cost we accept:** a bounded, monitorable coupling mass-balance error from the time-lag (a knob: shrink the step; Fiorentini 2015), and loss of within-step flow timing/depth fidelity (PIDS's stated goal doesn't need it). Both are first-class outputs, not hidden.

## 0a. Foundational choices the spike must resolve FIRST (promoted out of "spike details" after the Codex review)

These are **design prerequisites**, not incidental tuning. The first draft mis-filed them as spike-side details; they gate the redesign and must be answered by the spike before any Part-B code.

- **F1 — Surface-state representation.** Is the ponded depth (a) *co-located* with `ψ_top` (reuse `add_ponding_bc` verbatim; routing overwrites `ψ_top`), or (b) a *separate* top-facet store `d`? Honest delta (Codex finding 1): option (b) is **not** "slightly more plumbing" — `add_ponding_bc` hard-codes ponding as `max(ψ,0)` on the Richards state (`richards.py:191-194`) with no hook for an external depth field, so (b) **requires a new/modified Richards top-BC** that reads `d`. The spike picks the winner on conservation evidence; either way this is the redesign's central data-model decision.
- **F2 — Sink evaluation order / state definition.** The split exposes ≥3 distinct surface states per step: pre-Richards carryover ponding, post-Richards leftover, post-routing redistributed. The monolith booked every sink at *one* solved state. The new scheme must **define** which state drives each sink (Codex finding 5). Working definition to validate in the spike: subsurface sinks (GHB, interior drain) are evaluated inside the Richards solve (unchanged); surface sinks (Manning outlet, grate inlet) are part of the routing sweep itself — boundary/inlet nodes are alternative receivers that remove water from the post-Richards ponded field during the sweep. This is a **model definition**, not an implementation detail; pin it before B6.
- **F3 — Positivity under state mutation.** "Mass-conserving and non-negative by construction" holds for the routing substep *only* when it operates on a valid standalone surface store. Under F1 option (a) (overwrite `ψ_top`), a non-negative routed depth does **not** by itself guarantee the coupled Richards state is mass-consistent (`ψ_top` is also subsurface storage). The conservation gate (Part A, gate #2) is the real check; do not conflate "surface depth ≥ 0" with "full coupled state consistent" (Codex finding 6).

---

## 0b. SPIKE OUTCOME — verified GREEN (2026-06-23)

The Part-A spike ran (record: `validation/sanity/overland_split_spike__2026-06-22.md`). Architecture validated — no dt-collapse, no sawtooth, routing exact + non-teleporting, interception forms. Conservation was initially AMBER; a focused extension closed it. **Independently re-verified by the parent:** sand-channel `|balance|/cum_rain = 5.3e-12` clean vs `2.13e-2` under a deliberate 10% falsification injection (the detector is real); both cases incl. recession close to ~1e-11.

**F1 RESOLVED — the winning design (SUPERSEDES the §0a (a)/(b) framing):** fully **DECOUPLED** (no vertical co-solve — the non-monolithic intent holds). The pond is carried **in ψ** via the existing `add_ponding_bc` self-limiting term (NOT a new separate-store BC); lateral routing enters the Richards solve as an under-relaxed **Neumann source**; **Picard** under-relaxation gives robustness. The prior "18% leak" was an **accounting bug, not physics**: the conserved ledger must be `∫θ dV + ∫max(ψ,0) ds_top` with the pond/rain/source terms on the **same lumped vertex quadrature** as the routing `Σd_iA_i`. Fixing the ledger (not the solve) → 18% → 5e-12. The vertical `[ψ,d]` co-solve fallback was NOT needed.

**Open Part-B concern is ACCURACY, not mass balance (solved):** the ω(=0.5) under-relaxation + one-step lag **throttles the lateral transport rate** — a large transient swale pond (~1 m) builds before draining, and the interception fraction (22.9%) depends on that rate. Part B must calibrate ω / Picard-iters against a **resolved upwind reference** (a transport criterion) as a first-class accuracy task, plus carry the 6 plumbing fixes + a load-bearing `∫max(ψ,0) ds_top == Σd_iA_i` quadrature-match test.

**This reframes Part B:** F1 = the matched-quadrature ledger over ψ's pond (NOT a new BC); the orchestrator wraps {`RichardsProblem` + `add_ponding_bc` + lateral Neumann source, routing sweep} in a Picard loop; after the routing kernel, the first functional task is the conserving ledger + the transport-rate calibration. The `win` mode in `scratch/overland_split_spike.py` (`win_sand_short|win_sand|win_v_storm|win_v`, `PIDS_WIN_LEAK=0.1` falsification) is the reference implementation to productionize.

---

## 1. Load-bearing physics vs. incidental numerical complexity

The kickoff asked us to separate these before designing. From reading `overland.py`, `overland_upwind.py`, `overland_edge_kernel.py`, `coupling.py`, `richards.py`:

**Load-bearing (must be preserved):**
- **Richards subsurface** (`richards.py:richards_bulk_residual`, van Genuchten, lumped storage, the quadrature-degree cap). Unchanged.
- **Self-limiting vertical infiltration** = `add_ponding_bc` (`richards.py:174-196`): `((max(ψ,0) − max(ψ_n,0))/dt)·v·ds − rain·v·ds`. This is the validated water-level handoff. **Keep verbatim.**
- **Mass conservation as a structural property** — every sink/source must telescope or be booked. The routing sweep must conserve `Σ d_i A_i` exactly; the handoff imbalance must be tracked.
- **The drainage / outlet / inlet / feature accounting** (GHB, interior drains, surface inlets, embedded features) in `coupling.py` — these are real PIDS use cases and must survive into the new orchestrator (they hang off the Richards solve and off surface-water bookkeeping, both of which we keep).
- **The top-surface control areas `A_i` and edge connectivity** (`build_top_facet_edge_graph`) — the conserved-quantity weights. Reuse.

**Incidental (the complexity we are deliberately shedding):**
- The **monolithic `[ψ, d, λ]` block solve** (`coupling.py:783-789`, `kind="mpi"`). This shared Jacobian is Pathology 2 (dt-collapse). Gone in the new path.
- The **λ (exchange-flux) field + Fischer-Burmeister NCP** (`coupling.py:50-56, 230-272`). A clever device to co-solve the supply-limited exchange smoothly — unnecessary once the handoff is the implicit ponding BC. Not carried into the new path (the monolithic path keeps it as fallback).
- The **lateral Manning overland term** — galerkin `overland_flux` (`coupling.py:246-247,259-262`, Pathology 1 sawtooth) and the upwind edge-flux callbacks (`coupling.py:794-857`). Replaced by the explicit routing sweep.
- The **positivity limiter / tripwire** (`coupling.py:860-903`) — the routing sweep itself only moves existing non-negative water downhill, so the *surface store* needs no clip/rescale. (But see F3: that does not by itself certify the full coupled state under F1 option (a) — the conservation gate is the real check.) Keep a cheap assert + the tracked imbalance.

**Net:** the new scheme = (implicit `RichardsProblem` + `add_ponding_bc` + drains/features) ⊕ (a new explicit *rate-limited* routing sweep) ⊕ (a step-loop that alternates them and books the handoff balance). **Caveat (Codex finding 4):** that step-loop is *not* "thin" — the non-overland responsibilities are entangled in `CoupledProblem` (outlet assembly via `_finalize_forms`/codim-2 measures `coupling.py:389`; `d`-dependent inlet UFL sinks on `_ds_top` `coupling.py:676`; embedded-feature `pre_step`/`post_step` hooks `coupling.py:717,906`; accept/reject bookkeeping `coupling.py:945`), and must be **extracted and re-expressed** against the new surface store (F1). Budget B4/B6 as a real extraction, not orchestration. Most of the *deleted* complexity is the monolith we are intentionally leaving.

---

## 2. The two *original* pathologies, and why the split removes them (a hypothesis the spike tests)

- **Sawtooth (Pathology 1):** lives in the galerkin surface PDE advection term. The routing sweep has *no* surface PDE — it is algebraic downhill bookkeeping with one stable mode. Structurally removed. (Corroborated by our own upwind result on the tilted-V.)
- **Coupled dt-collapse (Pathology 2):** lives in the one block Jacobian coupling fast surface + stiff clay. The routing sweep never enters the Richards Jacobian; the Richards solve sees only its own (validated, stiff-but-fine) physics plus a self-limiting ponding term it already handles. Structurally removed from the Richards Jacobian. **New in exchange (Codex finding 3):** a time-lag coupling error + the routing rate law's own dt-sensitivity — bounded and monitorable (gate #2), not eliminated by fiat. This is why the spike is a gate, not a formality.

---

## PART A — THE SPIKE (do this first; throwaway; the go/no-go gate)

**Purpose:** mirror the embedded-feature probe — cheaply prove the design works on the *exact* cases that broke before, *before* paying for the productionized TDD build. Scratch code, not tested-to-spec. One file.

**File:** `technologies/infiltration-runoff-model/forward-model/scratch/overland_split_spike.py` (new, scratch — not committed to the package).

### A.1 What the spike builds (minimal)
A small driver that, per step, on a 3-D box host:
1. Holds an implicit `RichardsProblem` (the host soil) with `add_ponding_bc` on the top, plus the same drains the demo uses.
2. Solves Richards for the step (implicit, alone).
3. Reads the surface ponded depth `d_i = max(ψ_top,i, 0)` at every top node.
4. Runs a **rate-limited routing sweep** over the top-facet graph: build downslope receiver weights from the surface head `H_i = z_b,i + d_i` (MFD ∝ √slope), order nodes high→low head, and push each node's excess to its receivers **capped by the per-node Manning discharge over `dt`** (so water advances a bounded distance per step, not straight to the outlet), conserving `Σ d_i A_i`. The rate cap is in from v1 — it is load-bearing (without it the channel-interception case can't form; Decision 3 / F2), not a "if needed".
5. Writes the redistributed depths back to the surface for the next step per **F1** (surface-state representation), evaluating the surface outlet/inlet sinks during the sweep per **F2**.
6. Books `Σ d_i A_i` before/after the sweep (must match) and the global balance `Δtotal = rain − outflow − drainage ± handoff_imbalance`.

**Reuse, do not rebuild:** `build_top_facet_edge_graph` (edges, `L_e`, `A_i`), `RichardsProblem`, `add_ponding_bc`, the WSL run pattern. The MFD receiver-weighting + topological order is the only genuinely new ~60 lines.

### A.2 The two failing cases to run (the gate)
1. **3-D sand-channel-in-clay storm** — port the geometry/soil/forcing from `scratch/m4_sand_channel_3d_demo.py` (the run that hit the wall). Same box, slope, swale, berm, sand band, clay, storm.
2. **Convergent tilted-V** — the canonical fixture (see `[[pids-convergent-flow-priority]]` / `[[pids-p3-convergent-progress]]`; `benchmarks/build_comparison_tiltedv.py` / the parflow case for parameters).

### A.3 Foundational decision F1 (surface-state) — co-located vs. separate store
The ponded depth and the soil-top pressure head are the **same** P1 dof in this realization (`d = max(ψ_top,0)`). Two ways to apply the routed depths:
- **(i) Co-located write-back:** overwrite `ψ_top,i ← d_i^new` where ponded. Simplest, reuses everything, but mutating ψ_top also nudges the top cell's soil moisture `θ(ψ_top)` — must verify this stays mass-clean.
- **(ii) Separate surface store `d`:** keep a distinct top-facet field `d` (like `CoupledProblem.d`); routing moves it, ψ is never overwritten. Cleaner conservation, but **requires a new/modified Richards top-BC** that reads an external `d` — today's `add_ponding_bc` hard-codes `max(ψ,0)` on the Richards state (`richards.py:191-194`), so this is a real code delta, not a footnote (F1).

Spike runs **(i) first** (cheapest); if the books don't close to ~1e-10·rain, switch to **(ii)**. The winner becomes the redesign's surface-state representation. **This is the main thing the spike decides.**

### A.4 Pass/fail gate (pre-registered — all four required)
1. **No timestep collapse** on either case (dt does not death-spiral; completes to `T_END`). The original demo collapsed/pinned — this is the headline.
2. **Mass conserved:** routing sweep `|Σd_i A_i after − before|/(Σd_i A_i) ≲ 1e-12`; global `|balance|/cum_rain ≲ 1e-3` with the handoff imbalance *explicitly tracked and reported* (not hidden in a residual).
3. **Runoff intercepted ⇒ no teleportation:** on the sand-channel case the channel captures a meaningful fraction via the channel outlet + sand conveyance. This *directly* tests the Manning rate cap: if water teleported to the toe outlet in one step, capture would be ≈0 — a near-zero capture is a RED gate (broken rate law), not a tuning issue.
4. **Convergence behaves on the V:** the tilted-V routes to the outlet with no sawtooth and a sane, non-pinned dt.

**Green on all four → proceed to Part B. Any red → stop, report, and we revisit §A.3 / the rate-cap / depression-fill before building.** Do **not** start Part B on an amber gate.

### A.5 Spike reporting
Short markdown note `technologies/infiltration-runoff-model/validation/sanity/overland_split_spike__2026-06-22.md`: the four gate results with numbers, the §A.3 verdict (i vs ii), wall-clock vs. the old collapse, and any surprises that reshape Part B. Adversarially sanity-check the conservation numbers (don't trust a "0.00" — print full precision).

---

## PART B — THE TDD REDESIGN (gated on a green spike)

Productionize the spiked design as a selectable scheme with tests. **Each code task is strict TDD** (write failing test → see it fail → minimal code → see it pass → commit). Granularity below is per-task; expand each into the red/green/commit micro-steps at execution time per superpowers:test-driven-development. The exact surface-state representation (A.3) and any rate-cap/depression-fill specifics come from the spike note — fill them in before starting.

**Branch discipline (load-bearing — the investor-branch gotcha):** verify `git branch` shows `main` (or a dedicated `overland-sequential` branch off main) before every commit. Confirm the branch decision with Arik at execution kickoff.

### Task B1: Routing-sweep kernel — receivers + topological order
**Files:** Create `pids_forward/physics/overland_routing.py`; Test `tests/test_overland_routing.py`.
- Pure functions on the existing graph arrays (`edges`, `L_e`, `A_i`, `z_b`): `build_receivers(H, edges, L_e)` → per-node downslope receiver list + MFD weights (∝√slope, normalized); `topo_order(receivers)` → upslope-before-downslope ordering (richdem-style O(N) dependency sweep).
- **Tests:** on a tiny hand-checkable graph — single steepest-descent line orders correctly; a "many into one" convergence node receives from all uphill neighbors (no oscillation, weights sum to 1); a flat patch yields no spurious direction.

### Task B2: Routing-sweep kernel — the conserving push
**Files:** Modify `overland_routing.py`; Test `tests/test_overland_routing.py`.
- `route_excess(d, A_i, receivers, order, manning_n, slope, dt)` → new depths after pushing each node's excess to receivers in topological order, **each transfer capped by the Manning discharge over `dt`** (the rate limit is core, not a keyword option).
- **Tests (the conservation heart):** `Σ d_i A_i` invariant to ≤1e-12 on a random non-negative field; output non-negative; **a single pulse on a uniform slope advances a bounded, Manning-rate distance per step (NOT to the outlet in one step) and reaches the outlet only after the expected number of steps, with each per-step transfer ≤ the Manning cap**; idempotent on an already-drained field. The un-capped "full drainage in one step" behaviour is an explicit NON-goal (it breaks run-on infiltration — Codex finding 2).

### Task B3: Depression-fill pre-pass
**Files:** Modify `overland_routing.py`; Test `tests/test_overland_routing.py`.
- `fill_depressions(H)` (PriorityFlood) so interior pits route out instead of trapping water (only if the spike showed pits matter on these geometries; otherwise stub + skip-test with a note).
- **Tests:** a single interior pit drains to the boundary; a monotone slope is unchanged.

### Task B4: The orchestrator — one step
**Files:** Create `pids_forward/physics/sequential_coupling.py` (a `SequentialCoupledProblem`, mirroring the public surface of `CoupledProblem`: `set_initial_condition`, `set_topography`, `add_rain`, `add_outflow_bc`, `add_drainage_bc`, `add_interior_drain`, `add_surface_inlet`, `step`, `advance`, `total_water`, `surface_water`, `soil_water`); Test `tests/test_sequential_coupling.py`.
- `step(dt)`: implicit Richards solve (reusing `RichardsProblem`/`add_ponding_bc`) → read ponded depths → routing sweep → write-back (per A.3) → book handoff balance + accept/reject (reuse the `stall_accept_fnorm` honest-rejection gate from `coupling.py`).
- **Tests:** flat lake-at-rest holds depth + ψ to machine precision (no spurious motion); a one-column no-lateral case reproduces the standalone `RichardsProblem` + `add_ponding_bc` result (the split reduces to the validated vertical physics when there's nowhere to route).

### Task B5: Conservation + handoff-balance as first-class outputs
**Files:** Modify `sequential_coupling.py`; Test `tests/test_sequential_coupling.py`.
- Expose `cum_rain`, `cum_outflow`, `cum_drainage`, `cum_handoff_imbalance`, and a `balance()` that closes `Δtotal = rain − outflow − drainage + handoff_imbalance` to a reported tolerance. Port the per-sink accounting (`last_sinks`/`cum_sinks`) from `coupling.py`.
- **Tests:** closed domain (no rain, no outlet) conserves to ~1e-10; a storm-then-recession run closes the books; the handoff imbalance shrinks as dt shrinks (the Fiorentini knob — pin the monotone trend, not a magic number).

### Task B6: Sinks & features — EXTRACTED and re-expressed (Codex findings 4 & 5)
**Files:** Modify `sequential_coupling.py`; Test `tests/test_sequential_coupling.py`.
- **First, pin F2** (the per-step evaluation state for each sink) and write it into the docstring as a model definition: subsurface sinks (GHB, interior drain) evaluated inside the Richards solve; surface sinks (Manning outlet, grate inlet) evaluated during the routing sweep on the post-Richards ponded field. No sink is left to "whatever state happens to be current".
- This is a real **extraction** of entangled `CoupledProblem` internals — outlet codim-2 assembly (`_finalize_forms`), `_ds_top` `d`-dependent inlet sinks, feature `pre_step`/`post_step` hooks, accept/reject bookkeeping — re-expressed against the F1 surface store, not a thin "wire-through". Consider lifting the shared per-sink accounting into a small helper both classes use.
- Outlets, GHB drains, interior drains, surface inlets, embedded features (`add_embedded_exchange`, prescribed-rate, stays implicit per Decision 5) all book at their F2-defined state.
- **Tests:** each sink reproduces its `CoupledProblem` counterpart on a non-convergent case where both schemes are valid (the new path must agree with the validated monolith where the monolith works); the evaluation-state choice is pinned by a test that would change the booked total if the wrong state were used.

### Task B7: Selectable wiring + the two failing cases as regression fixtures
**Files:** Modify `sequential_coupling.py` (and/or a factory/flag so callers choose sequential vs monolithic); Tests `tests/test_sequential_coupling.py`, plus promote the spike's two cases to Tier-2 fixtures.
- **Tests:** the 3-D sand-channel-in-clay and tilted-V cases run to completion without dt-collapse and pass the §A.4 gate as *automated* regressions (smaller mesh / shorter horizon as needed for suite speed).

### Task B8: Tier-3 visualization + sign-off package
**Files:** a `viz/` HTML generator (follow the existing sanity-viz pattern) + a sign-off note under `validation/sanity/`.
- Side-by-side: old monolithic (collapsed/pinned) vs. new sequential (completes), water-balance trace, the channel-interception story. For **Arik's visual sign-off** (the Tier-3 gate), mirroring prior module sign-offs.

### Task B9: Code review + finishing
- superpowers:requesting-code-review on the diff; a Codex pass (the project's adversarial-verification habit); then superpowers:finishing-a-development-branch (merge to main with the fallback intact, or PR — Arik's call).
- Update the memory (`pids-overland-flow-rethink` → DONE; note the new scheme + that galerkin/upwind/monolith remain the fallback) and the decision record status.

---

## 3. Risks & guardrails

- **The handoff still over/undershoots (the §7 ghost).** Mitigation: we use the *level* handoff (`add_ponding_bc`), not a lagged flux — structurally the safe variant. The spike's gate #2 (tracked imbalance) is the early-warning; if it's large or dt-sensitive beyond the knob, the upgrade path is sequential-*iterative* (one Picard loop around the same two solves), **not** a return to the monolith.
- **Co-located write-back leaks mass (A.3).** Mitigation: the spike decides (i) vs (ii) on evidence; gate #2 catches a leak before any production code.
- **Routing on unstructured/obtuse tops.** The cotangent graph already has an M-matrix guard (`overland_edge_kernel._check_m_matrix`); MFD on √slope is more forgiving, but keep the guard and the structured-box restriction until proven.
- **Serial-only,** like the upwind path (the top-facet graph is not ownership-aware). Guard loudly; MPI is deferred.
- **Scope creep into flow timing.** Resist. PIDS wants redistribution + local infiltration + run-on, not hydrographs. The rate-cap/Picard upgrades exist *only if* the gate demands them.

## 4. WSL run pattern (for the spike and the tests)
```
wsl bash -c 'cd /mnt/c/Users/arikt/Documents/GitHub/PIDS/technologies/infiltration-runoff-model/forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:/usr/local/bin:/usr/bin:/bin" && export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u <script-or-pytest>'
```

## 5. Sequencing
**Step 1 = this plan (done).** **Step 2 = Part A spike** → green gate. **Step 3 = Part B TDD redesign.** Do not begin Part B until the spike is green and the three foundational choices (F1 surface-state, F2 sink-evaluation-state, F3 positivity check) are resolved.

## 6. Review provenance
Adversarially reviewed by Codex (2026-06-22) after the first draft; verdict: **architecture directionally sound** — the six findings were precision/honesty fixes, not an architecture change. Folded in:
1. **Rate cap is load-bearing** (Decision 3, Architecture, A.1, B2): without it routing teleports water to the outlet in one step and run-on infiltration — the sand-channel interception case — can't form. The Manning per-node rate limit is core from v1, and the old "pulse ends fully at the outlet" test was removed as a NON-goal.
2. **Separate surface store needs a Richards BC change** (F1, A.3): `add_ponding_bc` hard-codes `max(ψ,0)`; option (b) is a real code delta, not "plumbing". Surface-state representation promoted to a foundational decision.
3. **No "cannot form" claims** (§2, Architecture): the two *original* pathologies are removed but the split trades them for a bounded, dt-controlled coupling error the spike must verify.
4. **`CoupledProblem` is not "thin orchestration"** (§1, B6): outlets/inlets/features/bookkeeping are entangled and require real extraction.
5. **Sink evaluation-state is a model definition** (F2, B6): pinned before B6, not hand-waved as "book correctly".
6. **Positivity ≠ coupled consistency** (F3, §1): the routed depth's non-negativity doesn't certify the full state under F1(a); the conservation gate is the check.
