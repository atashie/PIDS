# Iterated-Capped Split — Conservation Re-Architecture Plan

> **★★ OUTCOME (2026-06-26) — DONE; the partition bug is FIXED (validated spike).** Full record:
> `technologies/infiltration-runoff-model/validation/sanity/overland_partition_bug_investigation__2026-06-24.md`
> §12–§13. Arc vs this plan:
> - **Task 1–3 (conservation) — as planned, GREEN.** `CoCycledCappedSplit` (co-cycled sub-stepping, no
>   `I`-reconstruction) closes the ledger EXACTLY (`~1e-11`, falsification fires) for ANY `K`. The probe
>   confirmed the leak source (the `route(A−I_final)` reconstruction; routing telescoping was machine-exact).
> - **Task 4 (partition) — this plan's PRIMARY HYPOTHESIS was REFUTED.** Co-cycled does NOT converge to the
>   monolith as `K→∞`; it over-routes (`+31/+41 pp`) because each sub-step still routes-before-draws. The
>   film offered is a FREE knob for conservation, so a weighted film `w` was swept: `w=0.5` (midpoint) is
>   stable + slope-robust but `+9 pp`; `w≈0.7` is accurate (±2 pp) but FORCE-FEEDS infiltration →
>   dt-COLLAPSE. An **accuracy–stability tradeoff**, not the clean K-convergence this plan expected.
> - **Resolution — ARIK'S THIN-SKIN idea (not in this plan).** A z-graded mesh with a ~2 mm top skin
>   (`scratch/seq_cocycled_skin.py::make_graded_box`) saturates the surface immediately → infiltration
>   throttled by subsoil percolation, not the ponding head → the `w≈0.7` collapse is removed. Final:
>   exact + monolith-accurate (b1 base −0.3 / steep +2.7 pp @ `w=0.7`) + stable + slope-robust + SKIN-robust
>   (partition skin-invariant; `w` sets partition, skin only stabilizes) + clay-V robust.
> - **NEXT (superseding this plan's Task 6):** broader soil/storm/slope sweep (only b1 loam + clay-V tested)
>   + mesh-convergence, then a PRODUCTION design + TDD to promote `CoCycledCappedSplit` + `make_graded_box`
>   into `pids_forward/physics/`. Knobs: `w≈0.7` (film weight, the one real parameter), `K=6`, a thin skin.

> **For Claude:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` (or `superpowers:subagent-driven-development`) to implement this task-by-task. This is a SPIKE-level plan (scratch, no `pids_forward/` edits, no production commit until validated) — the "tests" are the conservation / partition / robustness GATES, run in the WSL `pids-fem` env. TDD spirit: **conservation gate FIRST, before any partition claim.**

**Goal:** Re-architect the iterated-capped overland split so its mass balance closes **EXACTLY** (`bal/rain → ~1e-11`, falsification-verified), while preserving the already-confirmed simultaneity fix (the steep partition is no longer structurally over-routed), then re-test partition accuracy + clay robustness.

**Architecture:** The current spike leaks (~1.7e-3) because it **reconstructs** the per-step infiltration `I_final = film − film_rem` (a vertex quantity) and routes `A − I_final` — but `I_final` does not exactly equal the actual `∫θ` soil gain, so the routing removes a slightly different amount than infiltrated. The fix is to **eliminate the reconstruction**: route only EXACT surface water and let the ledger close structurally the way route-first B does (`∫θ + ∫max(ψ,0)·ds + Σd_held·A` telescoping). Leading candidate = **co-cycled sub-stepping**: sub-cycle `(route → re-split → Richards-draw)` `K` times per global step. Each of the three operations is individually exact (matched-quadrature pond draw; telescoping routing; conservative ψ↔held re-split), so conservation is exact at **any** `K`, and the partition converges to the monolith as `K → ∞` (the soil gets a full `h_ref`-film draw every sub-step, *before* routing can thin the pond below `h_ref` — which is exactly what route-first failed to do on steep terrain).

**Tech stack:** DOLFINx 0.10 FEM (P1, lumped), WSL conda env `pids-fem`, serial, threads pinned. Reuses the committed scratch harness: `seq_href_iterated.py` (the current `IteratedCappedSplit`), `seq_href_cap_spike.py` (`HrefCappedPondInPsi` parent + its conservative source / held-store / matched-quadrature ledger), `seq_href_closure2.py` / `seq_iterative_prototype.py` (fixtures `B1`/`CV`, `_march`, monolith targets, `_top_area_ds`).

---

## Background — read these FIRST

- **Record `validation/sanity/overland_partition_bug_investigation__2026-06-24.md` §9–§11** — the full arc: off-the-shelf survey (CATHY/HYDRUS/ParFlow switching BC) → realization B (route-first) RETRACTED as structurally wrong → the iterated-capped split CONFIRMS simultaneity (steep +26 pp → +4–6 pp) but conservation is only ~2e-3.
- **`forward-model/scratch/seq_href_iterated.py`** — `IteratedCappedSplit.step()`. The leak is the reconstruction at the `# CONSERVATION re-route` block: `I_final = film − film_rem`, then `route(A − I_final)`. **This is the thing to replace.**
- **`forward-model/scratch/seq_href_cap_spike.py`** — `HrefCappedPondInPsi`: the conservative pieces to REUSE. Key invariants: (a) the ledger is `∫θ dV + ∫max(ψ,0)·ds_top + Σd_held·A` on **matched lumped vertex quadrature** (the routing's `Σd_i·A_i` is bit-consistent with `∫max(ψ,0)·ds_v`); (b) the pond is changed via the **lateral SOURCE** (`_lat_src`, no ψ writeback — a writeback resets θ off-ledger, see §9 dead end); (c) `_route(d_full, dt)` conserves `Σd·A + outflow` by telescoping; (d) the 10% `outflow_leak_frac` falsification hook.
- **`forward-model/pids_forward/physics/sequential_coupling.py`** — the parent design constraints (the module + class docstrings): non-iterative split, near-saturation no-Ss fragility (carry the pond as a comfortably-positive head), serial-only.

**Load-bearing lessons (do NOT relitigate):**
- **Never pipe a long run through `tail`** (it buffers until EOF → output lost on kill). Write live, un-`tail`ed; read the harness output file.
- **Always guard `_march` with `max_steps`** (a runaway tiny-dt grind cost 4 h once).
- **`dt_max ≲ 0.004` is required** for any sequential-split partition number on the b1 storm (the onset must be resolved; `dt_max=0.02` under-resolves → garbage). The dt-check proved B itself is dt-convergent at this level.
- **WSL run pattern:** `wsl bash -c 'cd /mnt/c/Users/arikt/Documents/GitHub/PIDS/technologies/infiltration-runoff-model/forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:/usr/local/bin:/usr/bin:/bin" && export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u scratch/<script>.py'`
- **Verify `git branch` = main before any commit** (investor-branch gotcha).

**Monolith targets (galerkin, dt-robust, cross-checked vs cached upwind 0.5466):** b1_base (S=0.03) **0.547**; b1_steep (S=0.10) **0.551**; b1_coarse (20×14×5) 0.6145. Hortonian intuition: the partition should be ≈ slope-INSENSITIVE (≈ (rain−Ks)/rain ≈ 0.5), which the monolith is and route-first was not.

---

## The re-architecture — candidates

**A. Co-cycled sub-stepping (PRIMARY — implement first).** Per global step `dt`, do `K` sub-steps of `dt/K`; in each: (1) route the total surface water `(max(ψ,0) + d_held)` over `dt/K` → outflow; (2) re-split via the source: `film = min(routed_pond, h_ref)`, `d_held = routed_pond − film`; (3) solve Richards over `dt/K` (ψ_n = the previous sub-step's solved ψ — a genuine sub-time-march, NOT re-solving from entry) so the soil draws the film. **Conservation is exact at any K (no reconstruction).** Accuracy converges to the monolith as `K→∞` (per-sub-step "route-then-draw" bias is O(dt/K)). Cost ≈ `K`× B's per-step cost. The accuracy knob is `K` (analogous to B's `route_substeps`, but with a Richards solve per sub-step — Codex's "route_substeps won't fix it" was about B's *single*-solve sub-stepping, which is a different scheme).

**B. Re-ordered Picard, no reconstruction (ALTERNATIVE if A's K-cost is too high).** Keep one Richards solve per outer iterate, but route the **actual remaining** surface water (`max(ψ,0) + d_held` AFTER the draw), never `A − reconstructed_I`. Iterate the midpoint film estimate to a fixed point. Exactly conservative (draw + route are each exact); the subtlety is the midpoint heuristic + Picard convergence.

**C. Fallbacks if A and B both stall:** (i) the implicit `[ψ,d,λ]` NCP vertical co-solve (realization A from §10) + explicit routing *source*, iterated to consistency — exactly conservative but more machinery; (ii) monolith hardening (Newton globalization / continuation) — co-solves so the partition is correct by construction, attack its dt-collapse directly (§10 path 2). Do NOT invent another explicit one-pass ordering variant.

---

## Tasks

### Task 1 — Reproduce + instrument the leak (confirm the diagnosis)

**Files:** new `forward-model/scratch/seq_href_cons_probe.py` (imports `IteratedCappedSplit`).

**Step 1.** Run b1_steep with the current `IteratedCappedSplit` (`h_ref=2e-3`, `dt_max=0.004`), and per step print: `Σ(film−film_rem)·A` (the routed-subtracted `I`), the actual `Δ∫θ` over the step (degree-8 + lumped), and `outflow`. Confirm the per-step residual `(Δ∫θ) − Σ(film−film_rem)·A` is O(1e-3·rain) and ACCUMULATES to the ~1.7e-3 balance gap. **Gate:** the leak is the reconstruction mismatch (not the routing telescoping, not the Picard count alone). Record the per-step residual magnitude.

**Step 2.** Commit the probe + a one-paragraph note in §11 confirming the leak source.

### Task 2 — Implement co-cycled sub-stepping (candidate A)

**Files:** add class `CoCycledCappedSplit(HrefCappedPondInPsi)` to `forward-model/scratch/seq_href_iterated.py` (NOT a new file — keep the iterated variants together). Param `K` (sub-steps/global-step, default 6).

**Step 1.** Write `step(dt)`:
```python
def step(self, dt):
    self._ensure_built(); rp = self._rp; td = self._top_dofs_arr
    psi_entry = rp.psi.x.array.copy()
    rain = float(self._rain_c.value)
    hsub = dt / self.K
    cum_of = 0.0
    rp.dt.value = hsub                                  # Richards sub-steps of dt/K
    for k in range(self.K):
        # 1) route the TOTAL surface water (film-in-psi + held) over dt/K
        d_full = np.zeros(self._n_dofs)
        d_full[td] = np.maximum(rp.psi.x.array[td], 0.0) + self.d_held[td] + rain * hsub
        d_routed, of = self._route(d_full, hsub); cum_of += of
        # 2) re-split via the SOURCE (no writeback): psi film := min(routed, h_ref), rest -> held
        film_prev = np.maximum(rp.psi.x.array[td], 0.0)
        film = np.minimum(d_routed[td], self.h_ref)
        self.d_held[td] = d_routed[td] - film
        lat = np.zeros(self._n_dofs); lat[td] = (film - film_prev) / hsub
        self._lat_src.x.array[:] = lat; self._lat_src.x.scatter_forward()
        # 3) Richards draws the film over dt/K (psi_n = prev solved psi -- genuine sub-march)
        rp.psi_n.x.array[:] = rp.psi.x.array; rp.psi_n.x.scatter_forward()
        rp._ensure_problem(); rp._problem.solve()
        snes = rp._problem.solver
        if not (int(snes.getConvergedReason()) > 0 and
                (int(snes.getConvergedReason()) != 4 or float(snes.getFunctionNorm()) <= self.stall_accept_fnorm)):
            rp.psi.x.array[:] = psi_entry; rp.psi.x.scatter_forward()      # retry-safe restore
            return False, int(snes.getIterationNumber())
    # accept: book outflow + rain + sinks (mirror HrefCappedPondInPsi.step's accounting block)
    rp.psi_n.x.array[:] = rp.psi.x.array; rp.psi_n.x.scatter_forward()
    self.cum_outflow += cum_of * (1.0 + self.outflow_leak_frac)
    self.last_outflow = cum_of
    # ... ghb/interior sink assembly (copy from HrefCappedPondInPsi.step) ...
    self.cum_rain += rain * self._top_area * dt
    self._t += dt
    return True, int(snes.getIterationNumber())
```
Reuse `surface_water()` / `balance()` from `HrefCappedPondInPsi` (they already include `d_held`). NB: the source `lat` is on the `dt/K` (`hsub`) scale, matched to the Richards `dt`.

**Step 2.** Sanity: a single tiny run (b1_base, `K=4`, `dt_max=0.01`, ~10 steps) just to confirm it RUNS (no crash, completes).

**Step 3.** Commit.

### Task 3 — CONSERVATION GATE (before any partition claim)

**Step 1.** Run b1_base + b1_steep with `CoCycledCappedSplit`, `K=6`, `h_ref=2e-3`, `dt_max=0.004`. **GATE: `bal/rain ≤ ~5e-11` on BOTH** (matched-quadrature exact, like B). If it is NOT ~1e-11, the re-split/source is mis-matched — debug before proceeding (most likely: the `lat`/`hsub` scaling, or a held-vs-film double count). Do not proceed to partition until this passes.

**Step 2.** Falsification: set `outflow_leak_frac=0.10` on b1_base; **GATE: the ledger breaks by ≈10% (|ratio|≈1.0)** — the detector still fires with the co-cycled outflow accumulation.

**Step 3.** Commit (conservation green).

### Task 4 — PARTITION RE-TEST (the simultaneity payoff, now conservative)

**Step 1.** K-convergence at b1_steep (the decisive case): `K ∈ {2, 4, 6, 10}` at `dt_max=0.004`, `h_ref=2e-3`. **GATE: `routed/R` converges (monotone, flattening) toward the monolith 0.551 as K grows** (expect it to APPROACH from slightly over-routed). Pick the smallest `K` that lands within ~2–3 pp.

**Step 2.** At the chosen `K`: b1_base (target 0.547) + b1_steep (0.551) + b1_coarse (0.6145). **GATE: all within ~±3 pp of their monolith targets, conservation ~1e-11.** This is the headline: a partition that is correct on BOTH mild AND steep (slope-robust), which route-first never achieved.

**Step 3.** `h_ref` sweep at the chosen `K` (b1_base + b1_steep, `h_ref ∈ {1, 2, 4} mm`): confirm a single `h_ref` works across slopes (it should, now that simultaneity is restored — if `h_ref` is still slope-sensitive, note it as a residual closure question). Commit.

### Task 5 — CLAY-V ROBUSTNESS (the regime the monolith dt-collapses on)

**Step 1.** Run the stiff convergent clay-V fixture (`CV` in `seq_iterative_prototype.py`) with `CoCycledCappedSplit` at the chosen `K`, `h_ref ∈ {2, 10} mm`. **GATE: completes (no dt-collapse), conserves ~1e-12.** (This is the whole reason for the sequential split — it must survive where the monolith fails.)

**Step 2.** Commit + write up the result in §12 of the investigation record (numbers, the chosen K, the conservation + partition + robustness verdict).

### Task 6 — Decision gate

- **If Tasks 3–5 all pass:** the co-cycled scheme is the validated fix → write a PRODUCTION design + TDD plan to promote it into `pids_forward/physics/` (a sibling/option to `SequentialCoupledProblem`, coexisting with the validated galerkin/upwind/monolith fallback), with `K` documented as the accuracy knob.
- **If conservation passes but partition stalls (K too large to be practical, or residual > 3 pp):** try candidate B (re-ordered Picard) — cheaper per step — reusing the conservation lessons; or escalate to candidate C.
- **If clay robustness fails:** fall back to monolith hardening (§10 path 2).

---

## Success criteria (the gates, in order)

1. **Conservation EXACT** — `bal/rain ≤ ~5e-11` on b1_base + b1_steep, falsification |ratio|≈1.0. *(Non-negotiable; the whole point of this plan.)*
2. **Partition slope-robust** — b1_base/steep/coarse all within ~±3 pp of their monolith targets at the chosen `K`. *(The simultaneity payoff.)*
3. **Robust** — clay-V completes + conserves.
4. **One `h_ref`** works across slopes (or the residual closure is characterized).

## Risks & fallback

- **K-cost:** co-cycled does `K` Richards solves/step. If the needed `K` is large (>~10), per-step cost may be impractical → candidate B (Picard, one solve/iterate).
- **Near-saturation fragility** (no-Ss clay): a sub-step that drives a ~zero pond on near-saturated clay can dt-collapse the standalone Richards — keep the pond a comfortably-positive head; if the clay-V collapses, that's the §10 fallback signal.
- **Ultimate fallback:** monolith hardening (Newton globalization / continuation). Do NOT add another explicit one-pass ordering variant (route-first, infiltrate-first, frozen-flux, writeback are all dead — §9–§10).

## After validation (out of scope here, next plan)

Promote the validated scheme to `pids_forward/physics/` with proper TDD (conservation + partition + robustness regression tests), coexisting with the galerkin/upwind/monolith schemes as a selectable option in `make_overland_coupled`.
