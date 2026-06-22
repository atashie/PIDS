# Wiring the embedded-feature subsystem into the coupled [ψ, d, λ] engine — design plan

**Date:** 2026-06-22 · **Status:** Codex 2-pass reviewed (all revisions incorporated 2026-06-22) — awaiting Arik approval before build
**Codex review (2026-06-22):** verdict **REVISE-THEN-SHIP**. Upheld: the A/B fork, A-before-B sequencing,
the acceptance-gate strategy (driver state is reject-safe iff `post_step` is accepted-only — verified
against the cut-and-retry path), and that the plan does not reintroduce the retracted host-controlled
sorptive-residual claim. Four required revisions, **all incorporated below**: (1) **catchment scope must be
an enforced API guard, not prose** — the WI host reads (`R_out/2` ring + volume-mean θ) span the whole
function space (`wi_exchange.py:293-304,370,378-382,414-421`), silently wrong on a host larger than one
catchment → §2 now requires an explicit `catchment_cells` selector threaded into the driver reads; (2)
**persistent absolute time** — `advance()` resets a local `t=0.0` every call and `step()` takes no `t`
(`coupling.py:829,906-926`), so a `self._t` must be specified before build (§A.1); (3) **`cum_feature` =
host-injected/extracted volume only** (= the driver's `inj`, `Σ rate·dt`), NOT `I_total` and NOT the
seed/reservoir (which stays a *scheme* ledger outside the host books); (4) corrected the
`validation/sanity` path citation. Also folded in: extending the `cum_sinks`/`last_sinks` dicts
(`coupling.py:321-322`) to carry a `'feature'` kind.
**Codex 2nd pass (confirming, 2026-06-22):** verified revisions 2/3/4 + the `'feature'` extension landed
cleanly; found **two more items, both now fixed** — (5) the §A.1 code sketch had the ridge-source sign as
`+` while the header/R-A4/the validated harness use `−rate/length·v·dΓ`, **corrected to `−`** (a `+` would
invert disperse↔drain source/sink); (6) masking the WI reads to a catchment needs **two post-mask guards**
(refuse zero effective catchment volume in `_set_volume_weights`; refuse an empty *masked* `R_out/2` ring
shell) — added to §2. The acceptance-gate reject-safety and the 4-block/upwind-offset analysis were
re-confirmed sound. **Plan is now build-ready pending Arik's go.**
**Goal:** make `EmbeddedFeature` + `WellIndexExchange` (the validated Module-4 sub-grid feature
subsystem) usable *inside* a `CoupledProblem` run, so an embedded PIDS feature (tunnel / french drain /
granular conveyor) participates in the same monolithic surface↔subsurface storm the rest of the engine
already solves — driven by, and accounted into, the coupled host.

---

## 0. The central fork (read first — it shapes everything)

The standalone subsystem contains **two physically and numerically distinct couplings**, and "wire it into
the coupled Newton" means different surgery for each:

| | **EmbeddedFeature residual block** | **WellIndexExchange prescribed-rate driver** |
|---|---|---|
| Feature head `H_f` | **SOLVED** (a field) | **HELD** (a fixed reference: disperse wall ψ=0; drain a prescribed channel head) |
| Physics | granular axial Darcy **conveyance** + storage + **potential-driven σ / K(ψ)-series exchange** | sub-grid **sorptive** wall exchange: disperse imbibition (cyl Green-Ampt clock → resolved-ring Kirchhoff) / drain PSS depletion |
| Coupling style | **fully implicit** residual block in the Newton | **prescribed rate** (explicit per step from a host read; host responds implicitly) |
| Validation | blocked `[ψ, H_f]` Newton, conserves machine-tight (`test_feature_embedding.py`, `test_feature_sorptive.py`) | prescribed-rate-into-a-resolved-host, 2.5–5.7% rel-L2 vs closed-domain refs (`test_wi_exchange.py`, `test_coupled_gate_refs.py`, `validation/sanity/m4_phase4_*`) |
| Mechanism in code | `feat.conveyance_form` / `storage_form` / `exchange_into_{feature,host}` / `hf_residual` | `driver.pre_step(feat,psi,t,dt)→rate`, host carries `−rate/length · w · dΓ`, `driver.post_step` books on accept |

These are **not** interchangeable. The sorptive closures are validated **only** as prescribed rates (the
imbibition/desorption front is sub-cell — the host mesh cannot resolve it, so it cannot be a residual
block); the conveyance + potential-driven exchange is validated **only** as a residual block. The
architecture's full vision — a feature that *both* conveys via a solved `H_f` *and* sorbs via the sub-grid
closure, with genuine host-controlled sorptive coupling — was **over-claimed twice and retracted**
(`technologies/infiltration-runoff-model/validation/sanity/m4_phase4_coupled_review__2026-06-11.md`; the
Phase-4 research track). This plan does
**not** re-open that; it ships the two *validated* couplings into the coupled host and keeps the
unification explicitly deferred.

So: **two capabilities + one deferred research track.**

- **Capability A — prescribed-rate `WellIndexExchange` on the coupled host** (the sorptive disperse/drain
  features). Highest value (the unique PIDS wall physics), lowest solver risk (no new block — a ridge
  *source*), most-validated closure. **Do this first.**
- **Capability B — `EmbeddedFeature` residual block in `[ψ, d, λ, H_f]`** (the granular conveyor / french
  drain with potential-driven exchange). The genuinely-monolithic 4-field coupling. Medium surgery,
  well-precedented by the existing blocked assembly.
- **Deferred:** unified solved-`H_f` + sub-grid sorptive; genuine host-controlled sorptive *residual*
  coupling (the retracted track); multi-feature catchment partitioning; surface-reaching gutters
  (σ_top↔d); MPI.

---

## 1. Decision points (for Codex / Arik)

- **DP-1 — sequence.** Recommend **A before B**: A productionizes the most-validated, highest-value, unique
  PIDS physics with no new block (lowest Newton risk); B then adds the conveyance field. *(Alt: B first if
  the immediate use case is a conveying french drain rather than a sorbing wall.)*
- **DP-2 — host = catchment scope (A) — ENFORCED, not documented (Codex blocker).** The `WellIndexExchange`
  host reads — the WI-era ring at `R_out/2` and the volume-mean θ (drain bulk drive + disperse capacity
  throttle) — currently span the **whole function space** (`wi_exchange.py:293-304,370,378-382,414-421`),
  which in the standalone harness *is* the single feature's catchment (radius `R_out`). On a coupled host
  larger than one catchment, or with ≥2 features, those whole-domain reads are **silently wrong**. v1 makes
  the catchment an **explicit, required** registration argument `catchment_cells` (the cell set the driver's
  reads are restricted to) — the *first supported selector is `"all"`* (whole domain, an explicit opt-in,
  not a silent default). The mask is threaded into the driver reads (a `wi_exchange.py` change: restrict the
  volume weights, the `R_out/2` ring mask, and the drain/disperse θ-means to the catchment dofs). Registering
  a 2nd feature requires its own `catchment_cells`. Per-feature *automatic* catchment partitioning stays
  deferred (R-A1), but the silent-misread is closed by the required selector.
- **DP-3 — `H_f` handling (A).** Keep `H_f` **held** (disperse wall ψ=0; drain a fixed prescribed channel
  head) exactly as the validated harness. The unified "solved-conveyance `H_f` + sub-grid sorptive wall" is
  deferred.
- **DP-4 — scoping.** **Serial + 3-D only**, consistent with the upwind path and the WI reads (which are
  rank-local by construction and already `raise` on `comm.size>1`). 2-D/MPI features are a guarded future
  item.
- **DP-5 — antecedent-state fidelity (A).** `configure_sorptive` fixes the closure scalars (Parlange `S`,
  Δθ, ΔΦ_ref) at a *single* configured pair `(ψ_i, ψ_wall)`. On a real coupled field ψ_i varies in space and
  time. First cut: configure at a representative antecedent ψ_i for the feature's neighbourhood and
  **document the caveat** (the gate evidence is uniform-ψ_i); a ψ_i-adaptive closure is future work.

---

## 2. Capability A — prescribed-rate `WellIndexExchange` on the coupled host

### A.0 What changes vs the standalone harness
The reference driver (`scratch/m4_phase4_embedded_harness.py:run_embedded`) runs a **bare Richards** host:
`F = richards_bulk(psi) − rel_c·w·dΓ + feat.sorptive_into_host(w,psi)` (the sorptive term inert, Ω≡0),
`feat.Hf` held, `pre_step→rel_c.value`, solve, `post_step` on accept. **The only change is swapping that
bare host for `CoupledProblem`** so ψ is now the coupled subsurface field (fed by surface ponding →
infiltration → the water table) and the injected/extracted water lands in the coupled books.

### A.1 The mechanics (code-grounded)
1. **A registration API on `CoupledProblem`:**
   ```python
   def add_embedded_exchange(self, feat, driver, catchment_cells):  # driver = a configured WellIndexExchange
       # catchment_cells: the cell set the driver's whole-field reads are restricted to (the feature's
       #   catchment); pass "all" to opt IN to whole-domain reads (NO silent default — DP-2/Codex blocker).
       # holds (feat, driver, rate_const, catchment); weaves the ridge source into F_psi; returns rate_const
   ```
   - Stores `(feat, driver, rate_const, catchment)` where `rate_const = fem.Constant(mesh, 0.0)`
     (= rate/length). The `catchment` dof mask is threaded into the driver so its volume weights
     (`_set_volume_weights`), the `R_out/2` ring mask, and the drain/disperse θ-means
     (`wi_exchange.py:293-304,272,378-382,414-421`) read **only the catchment dofs** — a `wi_exchange.py`
     change (a `catchment` kwarg on `setup`, defaulting to the full field for the standalone callers so the
     existing gates stay byte-identical).
   - **Two post-mask guards the masking makes necessary (Codex 2nd-pass):** masking can produce a degenerate
     read the current whole-mesh checks don't catch, so add (a) **refuse zero effective catchment volume**
     before normalizing the masked weights in `_set_volume_weights` (`wi_exchange.py:303`), and (b) **refuse
     an empty *masked* `R_out/2` ring shell** — extend the existing whole-mesh shell-exists check
     (`wi_exchange.py:271-277`) to the masked dofs (a subcatchment that excludes the annulus must raise, not
     silently read an empty ring). Both are loud `ValueError`s, consistent with the module's
     refuse-don't-degrade fence style.
   - **Extend the accounting dicts** (`coupling.py:321-322`): `last_sinks`/`cum_sinks` are currently fixed to
     `{ghb, interior_drain, surface_inlet}` — add a `'feature'` kind (list, add-order, like the others).
   - Adds the ridge **source** term to `F_psi` via `_build_F_psi` (parallel to the interior-drain weave at
     `coupling.py:563`): **`F_psi += −rate_const * self._vpsi * feat.dGamma`** — the SAME sign as the
     validated harness `−rel_c·w·dΓ` (`m4_phase4_embedded_harness.py:117`) and as R-A4. With
     `rate_const = rate/length` and `pre_step` returning `+` for disperse-into-soil / `−` for drain-out, the
     `−` makes disperse a positive source on ψ (water added) and drain a sink (water removed). *(The earlier
     draft wrote `+`, which would invert source/sink — Codex 2nd-pass blocker; corrected here. Pin the sign
     directly in the reduce-to-standalone test.)* `feat.dGamma` is an interior `ridge` measure on the *same
     host mesh* (realization
     A) — no mixed-dimensional assembly, FFCX-native, identical to how the standalone subsystem already
     runs. The off-Γ `H_f` pin is **not needed in A** (H_f is not solved).
   - `feat.Hf` is set once to the held wall/channel head (caller or a kwarg).
2. **`step()` hooks, inside the O5 acceptance gate (`coupling.py:829`):**
   - **Before** `self._problem.solve()` (after `self.dt.value = dt`): for each registered feature,
     `rate = driver.pre_step(feat, self.psi, self._t, dt)` then `rate_const.value = rate / feat.length`.
     (`self.psi` here is the previous accepted state `psi_n` — the same point the harness reads.)
   - **In the `converged:` branch only** (after the existing per-sink accounting, `coupling.py:856–884`):
     `driver.post_step(feat, self.psi, self._t, dt)` and book the signed volume into a **new sink kind
     `'feature'`**: `cum_sinks['feature'][i] += dt·rate` (sign per the host balance below).
   - **On rejection** (`else` branch, `:895`): do **nothing** for features — the retry re-enters `pre_step`
     and recomputes the rate from the restored `psi_n`. This is the load-bearing "accounting-inside-the-gate"
     invariant (the per-sink merge lesson, parent §8.8 / `8a891c2`): **a rejected reason-4 stall must never
     book a feature.**
   - **Persistent absolute time (Codex blocker).** Today `step()` takes no `t` and `advance()` resets a
     **local** `t=0.0` every call (`coupling.py:829,906-926`) — so the driver (seed age, handover timestamp)
     would see time restart on every `advance()` invocation. Specify a persistent `self._t` on
     `CoupledProblem` (init 0): `advance()` continues from and updates `self._t` (no per-call reset); add a
     `t` kwarg to `step()` defaulting to `self._t`. Pre-existing `advance()` callers are unaffected (single
     `advance()` from t=0 behaves as before); multi-`advance()` sessions now have monotone time.
3. **Conservation accounting.** The feature is a signed source/sink localized on Γ. The closed-system
   identity becomes:
   ```
   ΔW_host(soil+surface) = cum_rain − cum_outflow − cum_drainage + cum_feature
   ```
   where **`cum_feature` is the host-injected/extracted volume ONLY** = the driver's `inj` = `Σ rate·dt`
   (disperse `+` adds water to the soil; drain `−` removes it). **It is NOT `I_total` and NOT the
   seed/reservoir** (Codex revision 3): `I_total·perimeter·length == injected + seed` is a *scheme* ledger;
   the seed (~0.01% of capacity, the t0 contact water already in the sub-grid at seeding) stays **outside**
   the host books. So the host balance pairs only the water that actually crossed Γ into/out of the host
   field this run. Surface the feature in `sink_rates()` / `cum_sinks` under `'feature'` so existing tests'
   sink-sum invariants extend cleanly; keep the scheme's own `I_total == injected + seed` identity as a
   separate driver-level check (as the harness does).

### A.2 Tests (TDD, three-tier)
- **Tier-1 (`tests/test_coupling_embedded_exchange.py`, new):**
  - **reduce-to-standalone:** a disperse feature on a `CoupledProblem` with **rain off, flat bed, no surface
    activity** must reproduce the bare-Richards harness `I_total(t)` trajectory **within tol** (ideally
    byte-identical if the coupled `F_psi` bulk == the harness Richards bulk at Ω≡0 and no surface flux; if
    not byte-identical, pin a tight rel-L2 and explain the delta — e.g. the storage-quadrature/quad-cap).
  - **conservation:** `ΔW = cum_rain − cum_outflow − cum_drainage + cum_feature` to `<1e-6·scale`, with a
    feature active.
  - **rejection rollback:** force a reason-4 reject (tiny dt / stiff state) and assert `cum_feature` and
    `driver.inj` did **not** advance on the rejected step (mirror `test_step_acceptance.py`).
  - **fence + guards still fire:** the resolved-wall `_resolved_wall_fence` and the `comm.size>1` refusal
    raise unchanged through the coupled path.
- **Tier-2 (`tests/test_coupling_embedded_tier2.py`, new):** a coupled storm on a 3-D hillslope with (a) an
  embedded **disperse tunnel** above the water table and (b) a **drain** french-drain below it. Require:
  stable, conserves (full balance incl. `cum_feature`), θ bounded, **and the host-control property** — e.g.
  a surface storm that raises the water table measurably changes the drain extraction rate (the drain's
  live-θ-mean bulk drive responds; the standalone refD40-C recharge leg is the analogue). This is the gate
  that the coupling is *real* (host → feature), not a bolt-on clock.
- **Tier-3:** an HTML showing the coupled field (ponding + saturation) with the feature uptake/extraction
  overlaid over time (extend the convergent-dual-drain viz).

---

## 3. Capability B — `EmbeddedFeature` residual block in `[ψ, d, λ, H_f]`

### B.1 The mechanics
1. **API:** `def add_conveyance_feature(self, feat):` — registers a potential-driven embedded conveyor.
2. **Add `H_f` as a 4th block.** `feat.V` is already a **distinct** full-mesh P1 `FunctionSpace` object
   (`feature.py:86`), so it satisfies the DOLFINx block requirement (each block field on a distinct space —
   `[[pids-fem-block-distinct-spaces]]`); all four fields are equal-size (N dofs), so `d` stays block 1 at
   `[N:2N]` and the **upwind callback offsets are unchanged** (`coupling.py:744` reads block 1; verify with
   the no-malloc test). Assembly:
   ```
   NonlinearProblem([F_psi, F_d, F_lam, F_Hf], [psi, d, lam, Hf], bcs += [feat.pin_bc()])
   F_Hf  = feat.hf_residual(vHf) + feat.storage_form(vHf, dt) + feat.exchange_into_feature(vHf, psi)
   F_psi += feat.exchange_into_host(vpsi, psi)      # exact negative on the SAME dΓ → structural conservation
   ```
   `feat.hf_residual` already bundles the conveyance + the off-Γ pin diagonal allocation
   (`feature.py:201`, same device as the pinned d/λ rows).
3. **`step()` must advance/restore `Hf_n`** alongside `psi_n`/`d_n` (accept → `Hf_n ← Hf`; reject →
   `Hf ← Hf_n`). Add an `Hf` save (like `lam_save`).
4. **Conservation** is structural (sign-paired exchange telescopes); expose the net Γ flux in
   `cum_sinks['feature']` for diagnostics.
5. **Conditioning.** The conveyance block (`K_feat·area` along Γ, often K_feat ≫ K_soil) may stress the LU /
   any future fieldsplit (architecture §H.3 flags non-dimensionalizing the feature axial conductance).
   First cut: direct MUMPS LU (small systems) — measure Newton health; non-dimensionalization is a deferred
   perf item.

### B.2 Tests
- **Tier-1 (`tests/test_coupling_conveyance_feature.py`):** reduce-to-standalone (port the blocked `[ψ,H_f]`
  conservation test to `[ψ,d,λ,H_f]` with the surface inert → same `H_f`/ψ within tol); conveyance recovers
  1-D Darcy (`q = K_feat·A·ΔH/L`, the `test_feature_embedding.py` check on the coupled host); sign-paired
  conservation `<1e-12`; no-malloc on the 4-block Jacobian; both overland schemes still pass.
- **Tier-2:** a coupled storm where a sub-water-table french drain **intercepts groundwater and conveys it
  to an outlet** (`H_f` Dirichlet at the drain's outlet end) — conserves, drains the mound, the conveyed
  volume exits at the outlet.

---

## 4. Risks / open questions (Codex: please attack these)

- **R-A1 — catchment scoping of the WI reads (DP-2) — RESOLVED to an enforced guard (Codex blocker).** The
  `R_out/2` ring + volume-mean θ assume host = one catchment. v1 closes the silent-misread by making
  `catchment_cells` a **required** registration arg threaded into the driver reads (selector `"all"` is the
  explicit whole-domain opt-in). *Automatic* per-feature catchment partitioning (geometric Voronoi/flow
  basins) stays deferred — but no read is ever silently whole-domain.
- **R-A2 — antecedent ψ_i (DP-5).** The closure scalars are fixed at one `(ψ_i, ψ_wall)`; a coupled field
  evolves. How much fidelity is lost configuring at a representative ψ_i? Is a guard needed if the local ψ
  drifts far from the configured ψ_i?
- **R-A3 — accounting-inside-the-gate.** The single most likely bug: booking a feature on a rejected step,
  or failing to roll back `driver.inj`. The plan places `post_step` strictly inside the `converged:` branch
  and does nothing on reject — Codex, verify this is airtight against the adaptive cut-and-retry in
  `advance()`.
- **R-A4 — sign/units.** `pre_step` returns m³/day; the host carries `rate/length` per unit length on
  `dΓ`. The harness uses `−rel_c·w·dΓ` with `rel_c = rate/length`. The coupled `F_psi` weave must reproduce
  that exact sign so disperse adds and drain removes soil water — and the books term `cum_feature` must
  carry the matching sign. Easy to get backwards.
- **R-B1 — 4-block + upwind interaction.** Adding `H_f` shifts no offset (d stays block 1), but confirm the
  upwind residual/Jacobian callbacks (`_wire_upwind_callbacks`) and the no-malloc preallocation survive a
  4-field block matrix.
- **R-B2 — conveyance conditioning.** High-K_feat axial block in the monolithic LU — Newton health on a
  stiff storm; is non-dimensionalization needed in v1 or deferrable?
- **R-gen — do not re-claim the retracted coupling.** A is a *prescribed-rate* integration (honest: explicit
  rate, implicit host response); B is *potential-driven σ* (validated). Neither is the retracted
  "host-controlled sorptive residual" coupling. The Tier-2 host-control gate (A.2) must be framed as
  "the host drive (water table / θ-mean) moves the prescribed rate," **not** "the sorptive front is
  resolved in the Newton."

---

## 5. Acceptance bars (definition of done)

1. **Capability A shipped:** `add_embedded_exchange` (required `catchment_cells`, reads masked to it) +
   persistent `self._t` + the ridge source in `F_psi` + `step()` pre/post hooks inside the acceptance gate +
   `'feature'` accounting (`cum_feature` = injected/extracted volume only); reduce-to-standalone within tol;
   full balance (incl. `cum_feature`) `<1e-6`; rejection-rollback pinned; the resolved-wall fence + serial
   guard fire through the coupled path; Tier-2 coupled storm (disperse + drain) stable, conservative, with
   the host-control property demonstrated; Tier-3 signed off.
2. **Capability B shipped:** `add_conveyance_feature` + the 4-block `[ψ,d,λ,H_f]` assembly + `Hf_n`
   advance/restore; reduce-to-standalone; conveyance recovers 1-D Darcy; sign-paired conservation `<1e-12`;
   no-malloc 4-block Jacobian; a coupled french-drain Tier-2.
3. **No regressions:** 1-D/2-D/MPI callers unbroken; galerkin + upwind both still pass; full suite green;
   standalone subsystem + its gates untouched.
4. **Deferrals documented:** unified solved-`H_f`+sorptive; host-controlled sorptive residual; multi-feature
   catchment partitioning; surface-reaching gutters; MPI — each named with its gate.

---

## 6. Phasing (each phase = green TDD, committed)
- **A1** API (`add_embedded_exchange` with the **required `catchment_cells`** selector threaded into the
  driver reads) + persistent `self._t` time + `F_psi` ridge source + `step()` pre/post hooks inside the gate
  + `'feature'` accounting (extend `cum_sinks`/`last_sinks`) + Tier-1 (reduce-to-standalone byte-or-tight,
  conservation incl. `cum_feature`, rejection-rollback, the resolved-wall fence + serial guard, a
  catchment-restricted read sanity check).
- **A2** Tier-2 coupled storm (disperse + drain) + host-control gate + Tier-3 HTML. **Gate A (Arik sign-off).**
- **B1** 4th-block residual API + assembly + `Hf_n` lifecycle + Tier-1 (reduce-to-standalone, conveyance,
  conservation, no-malloc).
- **B2** Tier-2 coupled french-drain + Tier-3. **Gate B (Arik sign-off).**
- **Deferred track** (separate plan if pursued): host-controlled sorptive residual coupling; multi-feature
  catchments; gutters; MPI.

---

## 7. Artifacts / references
- Code: `pids_forward/physics/{coupling.py,feature.py,wi_exchange.py,sorptive_closure.py}`; reference driver
  `scratch/m4_phase4_embedded_harness.py`.
- Validation (under `technologies/infiltration-runoff-model/validation/sanity/`):
  `m4_phase4_coupled_review__2026-06-11.md` (the binding retraction wording),
  `pids_features_phase3__2026-06-10.md`, `m4_phase4_drain_endbias_attribution__2026-06-15.md`,
  `m4_itemC_resolved_wall__2026-06-16.md`.
- Architecture: `docs/plans/2026-06-04-pids-forward-model-architecture-design.md` §D.4 (composing the
  feature family) + §E + §H.3 (feature-conductance non-dimensionalization).
- Tests to mirror: `test_feature_embedding.py`, `test_feature_sorptive.py`, `test_wi_exchange.py`,
  `test_coupled_gate_refs.py`, `test_step_acceptance.py`, `test_engine_drains.py`.
