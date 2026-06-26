# Overland Run-on PARTITION Bug — Investigation & Session Record (2026-06-24)

> **Status:** FIX DIRECTION FOUND + SPIKE-VALIDATED (2026-06-25). The off-the-shelf survey (CATHY/HYDRUS/
> ParFlow switching infiltration BC) + a staged spike identified the fix: **cap the infiltration-driving
> head at a thin reference film `h_ref`, delivered via the sequential scheme's CONSERVATIVE, SELF-LIMITING
> pond-in-ψ source mechanism, with the excess held + routed (realization B).** It matches the
> ParFlow-validated monolith partition (b1 `routed/R` 0.546 vs 0.5466 at `h_ref`=2 mm, −0.1 pp), conserves
> to 1e-11 (falsification-verified), and stays robust on the stiff convergent clay-V (no dt-collapse,
> conserves 1e-12). The full `[ψ,d,λ]` NCP (realization A) is NOT needed. **OPEN:** the `h_ref` closure
> (generality across soils/slopes/storms/meshes — `routed/R` is sensitive to `h_ref`). See §9 (added
> 2026-06-25). Spike: `forward-model/scratch/seq_href_cap_spike.py`. Companion memory:
> `pids-overland-partition-bug.md`.

## TL;DR
Hardening the merged sequential overland scheme (`SequentialCoupledProblem`) uncovered a **structural
run-on infiltration/runoff PARTITION bug**: on permeable Hortonian cases it **over-infiltrates by
~24–40 pp** vs the ParFlow-validated monolith. Arik: *"failure to model infiltration correctly means our
model fails its purpose"* — regime-scoping REJECTED; the partition MUST be fixed. Root cause is the
**infiltration CLOSURE**, not the lateral lag. Three fixes refuted (capacity-cap-only; iteration-only;
iteration+cap [broken impl]). **Course-correction (Arik):** the fix does NOT require going monolithic —
CATHY / ParFlow-OverlandFlow / HydroGeoSphere all get infiltration right with NON-monolithic
(sequential, BC-switching) couplings. Next session revisits those pathways from fresh context.

---

## 1. The bug
- Hardening Test 2 (run-on accuracy) compared the sequential scheme vs the monolith on a **mild planar
  LOAM** Hortonian case (b1: 8×5×1 m, S0=0.03, LOAM Ks=0.25, RAIN=0.5 m/d > Ks, storm 0.08 d → T_END
  0.45 d, ψ_i=−0.4) where the monolith upwind scheme genuinely converges.
- **Partition metric:** of the rain `R_in`, the fractions `routed/R = cum_outflow/R_in` and
  `infil/R = (Δ∫θ + cum_drainage)/R_in` (degree-8 soil_gain). Each scheme's partition self-closes to ~1.
- At mesh 30×20×8: **monolith routed/R = 0.547; sequential routed/R = 0.17** (it infiltrates ~2× as much
  of the storm). Conservation is machine-tight in BOTH — this is a *partition* error, not a leak.

## 2. Ground truth (NO new ParFlow run needed — established from existing benchmarks)
The monolith ≈ ParFlow on exactly this regime:
- **B4** (done 2026-06-09): monolith vs ParFlow native coupled mode (`OverlandFlow` pond-as-pressure),
  LOAM ponding/infiltration partition, 6 storm×antecedent scenarios → **RMS Δθ 0.003–0.019**, peak
  ponding ~5–10 mm.
- **B5** (done 2026-06-10): monolith vs ParFlow, 3-D LOAM hillslope WITH lateral routing (≈ the b1
  regime) → **overland 0.479 vs 0.474 m³ (~1%), infiltration IDENTICAL, peak ponding 0.72 vs 0.73 mm**.
- ⟹ The monolith's runoff/infiltration partition is trustworthy; the sequential scheme is the outlier.
- ParFlow itself keeps a **thin 0.73 mm routing sheet**; the sequential scheme builds a **33 mm pond**
  and over-infiltrates it.

## 3. Diagnosis — it's the infiltration CLOSURE, not the lateral lag
Four scratch probes (`forward-model/scratch/`) eliminated every alternative cause:
- **NOT under-resolution** (`runon_partition_investigation.py`): mesh-refine *widens* the gap
  (−0.26 → −0.40 over 2.3k→60k cells). The two schemes converge to *different* continuous answers.
- **NOT CFL / route_substeps** (`cfl_substep_confirm.py`): SEQ plateaus by rs≈16–64, still ~24 pp short
  of the monolith at every mesh.
- **NOT O(dt) operator-splitting** (`dt_refine_split.py`): dt→0 (at CFL-resolved rs) plateaus at
  routed/R≈0.40, still ~24 pp below the monolith's dt-converged 0.642.
- **THE CLOSURE** (`seq_capped_infiltration_prototype.py` + the V1 result below): the sequential scheme
  infiltrates via `add_ponding_bc`'s pond-in-ψ **UNCAPPED** Richards-column uptake (a sorptive Darcy
  draw, >> Ks transiently). The monolith caps infiltration at `q_pot = kirchhoff(ψ_top, d)/ell_c`
  (Kirchhoff film conductance), via a Fischer-Burmeister NCP (`coupling.py:223-231`). **Different
  infiltration laws.** The monolith's q_pot is the ParFlow-validated one (B4/B5).
- **Circular coupling:** q_pot depends on pond depth d. The monolith keeps d thin (co-solved routing) →
  small q_pot → infiltration capped → excess routes → d stays thin. The sequential split lags routing →
  d builds deep → q_pot inflates → cap would be a no-op. The correct partition is a **simultaneous fixed
  point**: routing-keeps-d-thin ⟺ thin-d-keeps-capacity-low.

## 4. Refuted fix attempts (DO NOT blindly retry; understand why each failed)
1. **Capacity-cap-only** (`seq_capped_infiltration_prototype.py`): apply `inf = min(q_pot, supply)` as a
   Neumann influx. FAILED — went the *wrong* way (routed 0.19, WORSE) and **dt-collapsed on stiff clay**.
   Why: q_pot ≈ 1.7×Ks at a dry/deep-pond surface (no-op cap); a hard Neumann influx force-feeds the
   saturating no-Ss clay → singular.
2. **Iteration-only — "V1"** (`seq_iterative_prototype.py::IterativeSequentialV1`): wrap the
   route↔Richards step in an outer Picard loop to a fixed point on the pond (under-relaxed), keeping the
   existing pond-in-ψ closure. Robust (clay-V completes, bal/rain 1.5e-13) + falsification passes
   (−1.000), BUT the partition went **WORSE**: routed/R **0.141 vs parent 0.170 vs monolith 0.547
   (−40.6 pp)**. **★ DECISIVE:** iterating a *wrong closure* converges to that closure's natural answer
   (heavy infiltration). Proves the bug is the CLOSURE, not the lag — iteration cannot fix it.
3. **Iteration + q_pot cap — "V2"** (`...::IterativeSequentialV2`, separate pond array + capped influx):
   the prototype implementation **broke** (ledger bal/rain = 1.0, nothing moved) — INCONCLUSIVE as a
   test, and the cap-as-influx carries the same force-feed/no-op risks as attempt #1.

Earlier-refuted in the same session: the "iterated split via dt→0" framing (a mis-test of the
*non-iterated* split). §7's "operator-split blows up 50 m" was a **lagged stiff FLUX** (the catch-valve),
NOT the water-level handoff → it does NOT block an iterated/BC-switched water-level scheme.

## 5. ★ COURSE-CORRECTION (Arik, 2026-06-24) — the monolith is NOT the only path
After the V1/V2 refutations I framed the fix as "make the monolith's coupling robust" (block-iterative
solver, or harden the Newton). **Arik corrected this:** *"I strongly disagree that the monolith is the
only viable pathway forward — our research has shown that multiple other models address infiltration
correctly without the monolith."* This is right, and the doc review supports it:
- **CATHY** (Camporese 2010, Putti & Paniconi 2004): a **sequential, time-lagged** coupling with a
  **Neumann↔Dirichlet boundary-condition SWITCH** at the surface — flux-controlled (Neumann = rain) when
  the soil can accept the supply, head-controlled (Dirichlet = ponding/saturation) when it can't, with
  the un-accepted excess routed off. NON-monolithic, and it gets the partition right.
- **ParFlow `OverlandFlow`** (Kollet & Maxwell 2006): pond-as-pressure free-surface overland BC.
- **HydroGeoSphere / HYDRUS**: the textbook switching infiltration BC (our own monolith spec
  `docs/plans/2026-06-05-module3-landsurface-ncp-spec.md` §3 names these).
- The correct *closure* (capacity-limited / BC-switching infiltration) is **solver-agnostic**. The error
  was conflating "the monolith's q_pot closure" with "the monolith." A sequential scheme CAN get the
  partition right with a proper BC-switching / capacity-limited closure — the open question is the right
  formulation, not the solver architecture.

## 6. Validation gap that let the bug ship
**No gate ever pre-registered "partition vs monolith/ParFlow on permeable Hortonian run-on."** The
transport-calibration (`overland_transport_calibration__2026-06-23.md`) literally SAW it (loam: 87% vs
4.6% infiltration) and quarantined it as a "confounder." Every spike/sign-off/test cross-check vs the
monolith was on dry-surface, saturated-column, or low-K-clay geometry — none could exercise the
permeable Hortonian partition. The literature (Sulis 2010, decision record §9) even *predicted*
infiltration-excess + heterogeneous forcing as where sequential vs monolithic diverge most. → Any fix's
acceptance gate = hardening Test 2 Case A (partition vs the monolith on permeable Hortonian run-on).

## 7. State at session end
- **`main` is green and in sync** (HEAD has hardening Task 1 = field-scale robustness guard, `5a03569`).
- **Hardening Task 2** (`tests/test_sequential_hardening.py`): b2 (run-on signature) PASSES; b1
  (vs-upwind partition) + b3 (rs resolution) are the FAILING reproducers of this bug — marked `xfail`
  with a reason pointing here (preserved, suite stays green).
- **Scratch reproductions** (committed): `runon_partition_investigation.py`, `cfl_substep_confirm.py`,
  `dt_refine_split.py`, `seq_capped_infiltration_prototype.py`, `seq_iterative_prototype.py`. Regenerable
  outputs (`seq_iter_out.txt`, `seq_iter_mono_cache.npz`) gitignored.
- **Hardening plan/design** (`docs/plans/2026-06-24-sequential-overland-hardening-{design,plan}.md`):
  Tasks 1–2 partially done; Tasks 3–5 NOT started (paused by this bug).
- **Doc-review syntheses** (this session, in the transcript): CATHY/iterative coupling; monolith/ParFlow/
  §7; the redesign-rationale + validation gap. Key cited docs: the 2026-06-22 decision record §4–§9;
  the NCP spec `2026-06-05-module3-landsurface-ncp-spec.md` §3 (the switching infiltration BC);
  benchmarks README §5c/§5d (B4/B5 ground truth); engine-eval §7 (the catch-valve, NOT this bug).

## 8. Next-session direction (NOT decided — Arik to choose from fresh context)
Likely worth evaluating (revisiting the non-monolithic pathways Arik flagged):
1. A **proper CATHY-style Neumann↔Dirichlet BC-switching infiltration closure** in the sequential
   framework — the capacity-limited switch (flux-controlled vs ponded-head-controlled) that CATHY uses,
   replacing the uncapped pond-in-ψ draw. The crux: the ponded-regime infiltration must be limited to
   the soil's *acceptance* (≈ the q_pot film / saturated throughput), with the excess routing — done in
   a way that is conservative AND does not force-feed the no-Ss clay (the trap attempt #1 hit).
2. Re-examine WHY pond-in-ψ over-infiltrates vs the q_pot film at the continuum level (sorptive draw vs
   film-conductance; the mesh-dependence) — to pin the precise remedy.
3. Keep the monolith-side block-iterative / hardened-Newton ideas as ALTERNATIVES, not the only path.

Open question to settle early: what is the precise, ParFlow-cross-checked *acceptance* rule the sequential
closure must enforce, and can it be made robust on stiff convergent clay without a monolithic Newton.

## Reproductions (WSL pids-fem, serial, threads pinned)
```
cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
  export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
  python -u scratch/seq_iterative_prototype.py          # V1/V2 (monolith target cached)
# also: runon_partition_investigation.py, cfl_substep_confirm.py, dt_refine_split.py
```

---

## 9. Off-the-shelf survey + fix found + SPIKE-VALIDATED (2026-06-25)

**Survey (two parallel research passes — papers + open source — mutually corroborating).** The
infiltration/runoff partition is gotten right by a **switching infiltration BC** that keeps the
*infiltration-driving head at the saturated surface (`h_S ≈ 0` / a thin sheet)* and holds the queued
runoff *separately*. CATHY/HYDRUS/Scudeler-2017: per-node Neumann↔Dirichlet switch, re-evaluated each
Newton iteration; once ψ_surf hits the ponding threshold the node is head-controlled at `h_S` and the
infiltration is the *resulting* flux (≈ acceptance), never force-fed. ParFlow: pond-as-pressure
`d=max(p,0)` co-solved with lateral Manning routing keeps the sheet thin (~0.7 mm) so infiltration ≈ Ks
(this IS our monolith — why it matches B4/B5, and why it dt-collapses on convergent clay). Landlab
(operator-split, like us): `infiltration = min(capacity, ponded_supply)` as a sink on a stored depth.
**Diagnosis sharpened:** our sequential scheme conflates the infiltration-driving head with the queued
runoff — `add_ponding_bc` carries the FULL (lagged, deep) pond in ψ, so the deep head over-drives the
self-limiting draw; and `q_pot = ∫_{ψ_top}^{d}K/ℓ_c` inflates with `d` (the saturated range `Ks·d/ℓ_c`)
AND with the dry lower limit (sorptive). ParFlow gets away with pond-in-pressure only because co-solved
routing keeps `d` thin. In a SEQUENTIAL (lagged-routing) scheme you must pin the infiltration head.

**Staged spike (`forward-model/scratch/seq_href_cap_spike.py`), reusing the committed harness
(B1/CV fixtures, cached upwind-monolith target, `run_case1_iter`/`run_case2`/`run_falsification`):**

- **DEAD — hard-Neumann frozen-`q_pot` cap at `h_ref`** (`HrefCappedNeumann` = attempt #1 + the `h_ref`
  fix): `routed/R` stayed ~0.17 for every `h_ref` (−37 pp). A frozen entry-state `q_pot` delivered as a
  prescribed flux never lets the surface saturate self-consistently → the cap is moot. Confirms
  attempt #1's refutation is robust; the cap is worthless without a self-limiting delivery.
- **DEAD — pond-in-ψ WRITEBACK** (`HrefCappedPondInPsi`, first cut: set ψ_top := film each step): broke
  conservation hard (closure 3.23, `bal/rain` 2.2 — MASS CREATION; setting ψ_top resets the top cell's θ
  off-ledger) + 450 steps (the "bad Newton restart" dt-collapse the parent docstring warns of).
- **★ THE FIX — conservative realization B** (`HrefCappedPondInPsi`, corrected: cap ψ's film at
  `min(routed, h_ref)` via the parent's CONSERVATIVE lateral-SOURCE mechanism — NO writeback — with the
  excess in a held store that routes). Residual = bulk + pond-storage − lat (rain dropped, applied
  explicitly). Results:
  - **Partition (b1 loam, 30×20×8):** `routed/R` monotone in `h_ref` — 1 mm→0.401, **2 mm→0.546
    (−0.1 pp vs monolith 0.5466)**, 5 mm→0.677, 10 mm→0.677, 20 mm→0.673. The cap lands the partition.
  - **Conservation:** `bal/rain` 1.3e-11 at every `h_ref`; the 10% falsification mis-book breaks the
    ledger by exactly 10% (|ratio|=1.000) — genuine detector with the held store in the ledger.
  - **Clay-V robustness (stiff convergent, 24×16×5):** BOTH `h_ref` (2 mm, 10 mm) COMPLETE, no
    dt-collapse, conserve to 1e-12. The regime that collapsed the monolith.
  - Cost: 25–93 steps on b1 (the matching `h_ref`=2 mm = 62 steps/116 s), 78–103 on clay-V.

**Why B (the conservation math, closed-form):** Δ(soil)=+infiltrated; Δ(∫max(ψ,0)+Σheld) = rain·dt −
outflow − infiltrated ⇒ Δtotal = rain·dt − outflow. The lateral source is the parent's validated
conservative mechanism (ψ stays continuous → θ never reset). The film stays ≤ `h_ref` because the source
sets it to `film_target=min(routed,h_ref)` and infiltration only reduces it.

**VERDICT:** realization B is the fix; the `[ψ,d,λ]` NCP (A) is NOT needed (B reuses lighter, already
-validated machinery and hits every gate; A would face the same `h_ref` question). The cap MUST be
delivered conservatively + self-limitingly (pond-in-ψ source), NOT as a frozen flux or a writeback.

**OPEN (productionization crux): the `h_ref` closure.** `routed/R` is sensitive to `h_ref` (±factor-2 ≈
±14 pp), so `h_ref` is load-bearing, not cosmetic. `h_ref`=2 mm nails b1 (≈ 2–3× the monolith's 0.72 mm
sheet) — UNVERIFIED across soils/slopes/storms/meshes. Need a principled rule (tie to the local Manning
equilibrium sheet depth? to ℓ_c? to soil acceptance?) or a calibration procedure, validated against the
monolith on the broader regime, before a production design. (`HrefCappedNeumann`/writeback retained in
the spike file as the documented dead ends.)

---

## 10. h_ref CLOSURE STUDY — h_ref does NOT generalize; B over-routes on STEEP (2026-06-25)

Pinned the closure across slope/mesh. **Method fix first:** B's partition needs the storm onset
ADEQUATELY RESOLVED (`dt_max ≲ 0.004`); a first multi-case run at `dt_max=0.02` under-resolved → garbage
/non-monotone (a methodology bug, NOT a B flaw). The dt-check (`scratch/seq_href_dt_check.py`) proved B
is dt-CONVERGENT on b1_base: `1mm→0.520, 2mm→0.546, 4mm→0.664` at `dt_max=0.004`; 2 mm matches the
monolith 0.547 (and reproduces the original spike). So the earlier "fragility/maybe-pivot-to-A" alarm
was under-resolution — RETRACTED.

**At adequate resolution (`dt_max=0.004`, `scratch/seq_href_closure2.py`), the GENERALITY FAILS:**
- `b1_base` (S=0.03): `h_ref*` = 2.0 mm, matches monolith 0.547. ✓
- `b1_steep` (S=0.10): B routes **0.819 / 0.813 / 0.847** (h_ref 1/2/4 mm) vs monolith **0.551** —
  **+26 pp OVER-ROUTING, and FLAT in h_ref** (the cap is nearly inert here). No `h_ref` matches.
- `b1_coarse` (mesh 20×14×5): B routes ~0.68 vs monolith 0.6145 — **+7 pp**, also flat in `h_ref`.

**Diagnosis — the operator-split ORDER error.** B routes the WHOLE pond FIRST (books outflow), THEN
infiltrates the remaining film. On a steep slope Manning routing is fast (q∝√S), so route-first books
too much as runoff before infiltration claims its ~Ks share → over-routes. The monolith CO-SOLVES
(routing ⇄ infiltration simultaneous), so its partition is correctly slope-INSENSITIVE (0.547 base ≈
0.551 steep ≈ Hortonian (rain−Ks)/rain). `h_ref` caps the infiltration HEAD, not the routing SPEED, so
it cannot fix the slope error. **This matters: PIDS's core regime is convergent (steep) flow** — exactly
where B is worst. (Conservation stays machine-tight 1.2e-11 throughout — a true partition error.)

**OPEN (decisive, NOT yet run): is the steep over-routing O(dt) or FUNDAMENTAL?** The split order error is
formally O(dt) (route-first → simultaneous as dt→0), and steep routes ~1.8× faster than base (√(0.10/0.03)),
so steep may simply need finer dt to converge to ~0.55 (a documentable slope-dependent dt requirement).
Test = `b1_steep` @ `h_ref`=2 mm at `dt_max` 0.002 / 0.001 — does `routed/R` fall from 0.81 toward 0.55?
**★ VERDICT (2026-06-25, `scratch/seq_href_steep_dt.py`): FUNDAMENTAL — route-first B is structurally
wrong.** b1_steep @ h_ref=2 mm: `dt_max=0.004→0.8125, 0.002→0.8860, 0.001→0.9439` — `routed/R` climbs
toward **1.0** as dt→0 (linear extrapolation ~1.002), i.e. infiltration → 0. Finer dt makes it WORSE,
not better. **Mechanism:** route-first books the WHOLE pond as routing each step BEFORE infiltration
acts on the (post-routing) remainder; as dt→0 the routing claims everything and the infiltration share
→ 0. On a steep slope (fast routing) this is catastrophic; on a mild slope it's slow, so b1_base at a
FINITE dt landed a fair infiltration share that COINCIDENTALLY matched the monolith at h_ref=2 mm —
**that match is a finite-dt artifact, not a closure.** ⟹ **"realization B (route-first) is the fix" is
RETRACTED.** The h_ref cap + conservation machinery are still sound and reusable; the route-first
ORDER is the defect.

**Implication — the partition needs SIMULTANEITY (neither pure order works):** route-first → routed/R→1
(over-route); infiltrate-first → routed/R→0 (over-infiltrate = the original bug). The truth is the
monolith's simultaneous solve. Paths:
1. **Iterated-capped split (lead, untested):** wrap the CAPPED route⇄infiltrate in an outer Picard to a
   per-step simultaneous fixed point (order-independent at convergence → reproduces the monolith
   partition). NB the refuted V1 iterated the UNCAPPED closure (→ over-infiltration); a CAPPED iterate
   is the new, untested hypothesis. Cost: outer loop + conservation/clay-robustness gates + still pins
   h_ref. Risk: convergence/robustness on stiff convergent clay.
2. **Reconsider the monolith** (co-solves → correct partition by construction) and attack its
   dt-collapse another way (the problem the sequential pivot was avoiding) — e.g. better Newton
   globalization / continuation, NOT a return to the Manning-PDE overland.
3. The implicit `[ψ,d,λ]` NCP (A) does NOT obviously help: its LATERAL routing is still explicit →
   inherits the same order error (only the vertical infiltration co-solves).
Spikes: `seq_href_closure2.py` (corrected closure), `seq_href_dt_check.py` (b1_base dt-convergence),
`seq_href_steep_dt.py` (the steep verdict). The `dt_max ≲ 0.004` resolution requirement is load-bearing
for ANY sequential-split B result. **A checkpoint: route-first dead, iterated-capped split = the next
hypothesis to spike.**

**Codex adversarial review (2026-06-25) — CONFIRMS the verdict + SHARPENS the mechanism.** No spike/
harness bug invalidates it (the steep `routed/R→1` is B's behavior BY CONSTRUCTION). **Sharper
diagnosis than "route-first / O(dt)":** B drops rain from the Richards residual and couples infiltration
ONLY to the post-routing ENDPOINT film `film_target = min(routed, h_ref)`, so water gets an infiltration
opportunity *only if it survives routing into the end-of-step film*; on steep (fast routing) almost
nothing survives → infiltration→0. That is a DIFFERENT discrete model, not a Lie-splitting O(dt) error —
which is why refining dt cannot recover the monolith. **Caveat:** do NOT lean on the linear-extrapolation
-to-1.0 in `seq_href_steep_dt.py:66-70` (not evidence); the raw monotone trend `0.8125→0.8860→0.9439` is
the proof. **Endorses path 1 (iterated-capped split)** but ONLY as a TRUE per-step fixed point on the
conservative B state (film-in-ψ, `d_held`, routing/outflow); smaller global dt / more `route_substeps` /
another one-pass flux partition will NOT fix this class of error. **Fallback if the iterate still needs a
tuned `h_ref` or loses clay robustness: monolith hardening (path 2), NOT another explicit ordering
variant.** Retract "B is the fix", NOT the capped-film + conservation machinery (reusable).

---

## 11. ITERATED-CAPPED split — structural fix CONFIRMED (direction); refinement pending (2026-06-25)

Spiked the iterated-capped split (`scratch/seq_href_iterated.py`, class `IteratedCappedSplit`): an outer
Picard on the CAPPED route⇄infiltrate to a per-step SIMULTANEOUS fixed point. The soil sees the MIDPOINT
pond (avg of the pre- and post-routing un-infiltrated water, capped at `h_ref`), iterating the
infiltration depth `I` (under-relaxed). Per step: `route(A−I)→d_routed+outflow`; `film =
min(0.5·((A−I)+d_routed), h_ref)`; Richards draws `I_new = film − film_rem`; relax `I`; repeat. At
`h_ref`=2 mm, `dt_max`=0.004:

| case | route-first | iterated-capped (midpoint) | monolith |
|---|---|---|---|
| b1_steep (S=0.10) | 0.81 (+26 pp) | **0.611 (+6.0 pp)** | 0.551 |
| b1_base (S=0.03) | 0.546 (+0 pp) | 0.496 (−5.1 pp) | 0.547 |

**The structural steep failure is FIXED: +26 pp → +6 pp, and the error is now BALANCED (both within
±5–6 pp) instead of one catastrophic direction → SIMULTANEITY is the right idea, confirmed.** Picard avg
2.3–3.0 iters/step (8-iter cap hit on some steps). Runtime ~8–12 min/case (several Richards solves/step).

**Two issues pending — NOT a clean closure yet:**
1. **Conservation NOT clean (the harder problem).** Raw run: `bal/rain` ~0.5 % (the in-loop routing used
   the UNDER-RELAXED `I` but the soil drew `I_new`). The applied fix — a final re-route with `I_final =
   film − film_rem` — was **TESTED on b1_steep** (`seq_href_iter_verify.py`): it **reduced** the imbalance
   to **`bal/rain = 1.68e-3`** (and nudged the partition 0.611→**0.593**, +6→+4.2 pp) but did **NOT** close
   it to 1e-11. **Why:** `I_final = film − film_rem` (vertex reconstruction) does NOT exactly equal the
   actual ∫θ soil gain, and the Picard hit its 8-iter cap (the in-loop route↔draw never fully consistent).
   **⟹ the iterated split needs a CLEANER conservation STRUCTURE — close the ledger structurally like B
   (∫θ + ∫max(ψ,0) + Σd_held telescoping) rather than reconstructing `I` and subtracting it before
   routing. This is real next-session work, not a one-line patch.** (Route-first B was 1e-11 precisely
   because it never reconstructed `I`.)
2. **Residual ±4–6 pp:** the Picard hit its 8-iter cap (not fully converged), the MIDPOINT is an
   APPROXIMATION of true simultaneity, and `h_ref` is untuned. Next: tighten convergence (more iters /
   better relaxation); try a sharper simultaneity scheme (sub-cycled co-routing within the iterate);
   sweep `h_ref`; re-check across slope/mesh + clay-V robustness.

**VERDICT: the iterated-capped split is a VIABLE DIRECTION — it removes the route-first STRUCTURAL defect
(the headline win: steep +26 → +4–6 pp, error now balanced) and reuses B's machinery. But it is NOT yet a
clean closure: the partition residual is ±4–6 pp AND conservation is only ~2e-3 (the re-route fix helped
4.5e-3→1.68e-3 but the I-reconstruction is structurally lossy). Next session = (i) re-architect the
conservation to close structurally, (ii) tighten convergence + sweep h_ref to shrink the residual, then
(iii) clay-V + slope/mesh re-check. Fallback if it stalls: monolith hardening (§10 path 2).** Spike
`seq_href_iterated.py` (the re-route fix is in; conservation still ~2e-3 — see above). Verified figures
from `seq_href_iter_verify.py` (b1_steep 0.593 / bal 1.68e-3).

---

## 12. CONSERVATION RE-ARCHITECTURE — co-cycled sub-stepping (2026-06-25)

Plan: `docs/plans/2026-06-25-iterated-capped-split-conservation-rearchitecture.md`. Goal = close the
iterated split's `~1.7e-3` leak EXACTLY while keeping the confirmed simultaneity fix.

**Task 1 — leak source CONFIRMED (`scratch/seq_href_cons_probe.py`, b1_steep, 116 steps, 456 s).** An
instrumented copy of `IteratedCappedSplit.step()` faithfully reproduced the §11 figures (routed/R
**0.5933**, `bal/rain` **1.68e-3**) and isolated the leak:
- **Routing telescoping is machine-EXACT**: `Sum(route_resid) = −3.9e-17` (max `2.8e-17`) over the final
  re-route every step ⟹ the leak is **not** the routing.
- **The leak is the infiltration RECONSTRUCTION**: with routing exact, the ledger reduces analytically to
  `balance = (ledger soil θ-gain) − I_recon` (`I_recon = Σ(film−film_rem)·A` = the routed-subtracted
  infiltration; drainage = 0 here). i.e. the entire `1.68e-3` is the gap between the soil's *actual* θ
  gain and the reconstructed `I_final` that `route(A − I_final)` subtracts. The probe's `recon_gap` measured
  with degree-8 θ is `−4.08e-3` (same O(1e-3) magnitude; the deg-8-vs-ledger-quadrature offset ~6.8e-3
  flips the naive ratio to −1.5, not +1 — a *measurement* nuance, not a second leak). Per-step gap
  `max 2.1e-3·rain, mean 1.0e-4·rain`; **14/116 steps hit the Picard 8-iter cap** (un-converged
  `film`/`film_rem` feed the reconstruction).
- ⟹ The fix is to **eliminate the reconstruction**, not patch it.

**The re-architecture — co-cycled sub-stepping (`CoCycledCappedSplit`, candidate A).** Per global step
`dt`, run `K` sub-steps of `dt/K`: each = (1) route the TOTAL surface water (`max(ψ,0)+d_held+rain·dt/K`)
→ outflow; (2) re-split via the conservative `_lat_src` (`film=min(routed,h_ref)`, excess→`d_held`, NO ψ
writeback); (3) solve Richards over `dt/K` with `ψ_n` carried from the previous sub-step. **No `I`
reconstruction** — the pond IS ψ's pond, updated by the same solve that updates θ. Analytic conservation:
per sub-step `Δtotal = rain·(dt/K)·area − outflow_k − drain_k` EXACTLY for any `K` and any Newton state
(the infiltrated volume `dtheta_k + drain_k` cancels between the soil gain and the pond loss). `K` is the
accuracy knob; the partition is expected to approach the monolith as `K→∞` (each sub-step draws a full
`≤h_ref` film before routing can thin the pond — what route-first failed to do on steep terrain).
**Smoke (b1_base, K=4, short march): RUNS, `bal/rain = 1.25e-11`** — the leak is gone (vs `1.7e-3`).

**Task 3 — CONSERVATION GATE: PASS (K=6, h_ref=2 mm, dt_max=0.004).** The re-architecture closes the
ledger EXACTLY:

| case | `bal/rain` | routed/R | monolith | partition gap |
|---|---|---|---|---|
| b1_base (S=0.03) | **1.505e-11** ✅ | 0.8550 | 0.5470 | **+30.8 pp** |
| b1_steep (S=0.10) | **1.325e-11** ✅ | 0.9622 | 0.5508 | **+41.1 pp** |
| b1_base + 10% leak | **8.55e-02** ✅ (detector FIRES) | — | — | — |

⟹ **The conservation goal is MET** (`~1.3e-11` vs the iterated split's `1.7e-3`; the falsification detector
still fires at ~10%). This holds for ANY `K` (structural, no reconstruction). **The user's headline ask —
"re-architect the scheme's conservation to close exactly" — is DONE.**

**BUT Task 4 (partition) is REFUTED for candidate A.** Co-cycled `routed/R` **over-routes on BOTH** mild
(+30.8 pp) and steep (+41.1 pp) — *worse* than route-first (+0/+26 pp) and the WRONG direction (toward
`routed/R→1`). The plan's hypothesis ("partition converges to the monolith as `K→∞`") is **REFUTED**.

**Why (structural, not a bug):** each co-cycled sub-step still **routes-before-draws** — it routes the FULL
sheet `(film+d_held+rain·dt/K)`, then offers the soil only `film = min(d_routed, h_ref)` from the
*post-route* water. So sub-cycling just refines the effective routing timestep, which is exactly the §10
"finer dt → `routed/R→1`" route-first pathology. **The deeper constraint:** in the co-cycled structure the
soil can only draw from `d_routed` (post-route), because strict held-store conservation needs
`held = d_routed − film ≥ 0`, so the most infiltration-favoring *conservative* film is `min(d_routed,
h_ref)` = route-first. Genuine simultaneity (the iterated split's midpoint, which gave +4–6 pp) requires
offering the soil MORE than `d_routed` (the pre-route pond), i.e. a **signed/borrowed held** that repays
next sub-step. **Key realization: the film offered to the soil is a FREE knob for GLOBAL conservation** —
`Σ(film+held)·A = Σd_routed·A` holds for any `film` (the ledger is global; a transient local negative held
repays next sub-step). So the exact-conservation STRUCTURE and the simultaneity FILM-RULE are **separable**.

**⟹ Candidate B′ (next): keep the co-cycled exact-conservation structure, restore simultaneity via the
MIDPOINT film** `film = min(0.5·(d_full + d_routed), h_ref)` (or a capacity-based rule) — a one-line change
to `CoCycledCappedSplit.step`'s film computation, globally conservative by the above, expected to recover
the iterated split's +4–6 pp. Then re-run the Task-4 K-convergence + slope-robustness gate. Spikes:
`seq_href_cons_probe.py` (Task 1), `CoCycledCappedSplit` in `seq_href_iterated.py` + driver
`seq_cocycled_gate.py` (Tasks 2–3). Committed 2026-06-25.

**Task 4 — B′ PARTITION STUDY (co-cycled, weighted film `film = min((1−w)·d_routed + w·d_full, h_ref)`,
w∈[0,1]: w=0 route_first, 0.5 midpoint, 1 draw_first).** EVERY variant conserves `~1.3–1.9e-11` (the
free-knob result confirmed — film choice is conservation-independent). Partition `routed/R` (monolith
base 0.5470, steep 0.5508):

| w | b1_base | b1_steep | stable? |
|---|---|---|---|
| 0.0 (route_first) | 0.855 (+30.8) | 0.962 (+41.1) | ✅ |
| 0.5 (midpoint) | 0.619 (+7.2) | 0.653 (+10.2) | ✅ |
| 0.7 | 0.544 (**−0.3**) | 0.572 (**+2.1**) | ❌ dt-collapse ns≈22–23 (mid-storm) |
| 0.85 | — | 0.527 (−2.4) | ❌ dt-collapse ns=22 |
| 1.0 (draw_first) | collapse ns=6 | collapse ns=4 | ❌ |

Levers PROBED and ruled out: **K** (steep midpoint K=2/4/6/10 → 0.640/0.652/0.653/0.635, K-CONVERGENT,
~±1 pp band — NOT the lever, and the K→∞ limit is a STABLE +9 pp, not draw-first); **h_ref** (steep
midpoint 2/4/8 mm → all 0.6531 IDENTICAL — INERT, because routing keeps the ponds thinner than h_ref so
the cap never binds). The borrow (most-negative held) stays bounded ≈ −h_ref (deeper at higher w: −1.0 mm
@w=0.5 → −1.6/−1.9 mm @w=0.7).

**★ VERDICT — an ACCURACY–STABILITY TRADEOFF (confirmed on BOTH slopes).** The co-cycled scheme delivers
EXACT conservation (the headline ask, DONE) but the film weight that makes it accurate (w≈0.7: base −0.3,
steep +2.1 pp — within ±3 pp!) FORCE-FEEDS infiltration toward saturation and dt-collapses mid-storm (the
no-Ss saturation fragility, now triggered on LOAM by the thick film + the deeper borrow); the largest
STABLE weight (w=0.5 midpoint) is slope-robust + exact but +7–10 pp. The accurate region is structurally
unstable with the current (default-solver) Richards step. **So B′ is: exact + stable + slope-robust at
+9 pp, OR exact + accurate (±2 pp) but unstable — not all three yet.** Open options (Arik's call):
(1) accept the stable midpoint (+9 pp, a 3–4× improvement over the −24 pp bug, slope-robust, exact);
(2) HARDEN the Richards step to survive the w≈0.7 force-feed (saturation-aware linesearch/continuation,
cf. [[pids-fem-saturated-wall-linesearch]]) → exact + accurate; (3) test whether the COLLAPSE is the
negative-held BORROW (vs pure saturation) and limit it; (4) monolith hardening fallback (§10 path 2).
Driver `seq_cocycled_gate.py` (film_w knob); outputs `scratch/_cocy_*.txt`.

---

## 13. ★ THIN-SKIN FIX — the tradeoff RESOLVED (Arik's idea, 2026-06-26)

**Arik's diagnosis + fix (CORRECT).** The w≈0.7 collapse is a THICK-CELL artifact: the uniform mesh's
~125 mm top cell, under a ponded film over DRY soil (ψ≈−0.4 m), sees a gradient ~(film−ψ_cell)/half-cell
that drives a huge transient infiltration which fills the cell's large storage, then sharply saturates
mid-storm → Newton stiffness → dt-collapse (the monolith dodges this via its `q_pot=∫K dψ/ell_c` cap; the
co-cycled has none). **Fix: make the surface a THIN SKIN** so it saturates immediately and infiltration is
throttled by PERCOLATION into the (dry, low-K) subsoil, not the ponding head — the CATHY/HYDRUS
saturated-surface acceptance, realized through the MESH (physical, adaptive), not a frozen cap.

**Implementation:** `scratch/seq_cocycled_skin.py::make_graded_box` — a uniform tetra box with z warped
`z = Lz·(1−(1−s)^p)` to cluster nodes at the top (monotonic → no tet inversion). `p=2.5, nz=12` → a
**2.0 mm top skin** (vs 125 mm uniform), 9.3 mm second cell, grading to a coarse subsoil.

**Result — the FULL WIN (b1, `K=6`, `h_ref=2 mm`, `dt_max=0.004`).** On the 2 mm skin, EVERY run is STABLE
(ns≈116, the collapse is GONE across the whole w-range) and conserves `~1.0–1.3e-11`. The partition is a
smooth monotone function of w, and a SINGLE `w=0.7` lands BOTH slopes within ±3 pp:

| w (2 mm skin) | b1_base (0.5470) | b1_steep (0.5508) | stable |
|---|---|---|---|
| 0.5 | — | 0.6524 (+10.2) | ✅ |
| 0.6 | 0.5783 (+3.1) | 0.6121 (+6.1) | ✅ |
| **0.7** | **0.5437 (−0.3)** | **0.5776 (+2.7)** | ✅ |
| 0.8 | — | 0.5474 (−0.3) | ✅ |

Contrast: on the UNIFORM mesh w=0.7 COLLAPSED (ns=22); on the skin it runs to completion (ns=116) AND is
accurate. **⟹ EXACT conservation + monolith-accurate (±3 pp) + STABLE + SLOPE-ROBUST + a single film
weight — all at once.** The accuracy–stability tradeoff (§12) is RESOLVED by the thin skin.

**Caveat — the MONOLITH is NOT a valid same-mesh target on the skin:** galerkin monolith on the 2 mm skin
gives routed/R **0.254** (vs uniform 0.551) because its `q_pot=∫K dψ/ell_c` cap divides by the ~1 mm cell
half-height → the cap ~60× too large → stops biting → over-infiltrates. So the monolith's partition is
ell_c-(mesh-)dependent; the **co-cycled is NOT** (w=0.5 was 0.6524 on the skin vs 0.6531 uniform — the
skin changes STABILITY at high w, not the low-w partition). The true target stays the uniform 0.551.

**Mechanism confirmed:** the skin's effect is on STABILITY in the force-feeding (high-w) regime — at w=0.5
the thin offered film never stresses saturation so the skin is a no-op (partition unchanged); at w≥0.7 the
thick film WOULD force-feed, but the 2 mm skin saturates in ~one step (tiny storage) and the infiltration
becomes a smooth steady percolation into the subsoil instead of a sharp saturation front → Newton stays
healthy. Exactly Arik's "infiltration limited to subsoil percolation once the skin saturates."

**Validation COMPLETE — all gates PASS:**
- **Skin-thickness ROBUST (decisive):** steep w=0.7 across a 12× skin range — p=2.0/2.5/3.0 → top cell
  6.94/2.00/0.58 mm → routed/R **0.5774/0.5776/0.5776** (a 0.02 pp span), all stable, all `~1e-12`.
  **⟹ (w, skin) are NOT coupled: `w` sets the PARTITION; the skin only provides STABILITY.** Unlike the
  monolith (partition swung 0.55→0.25 with the skin via `ell_c`), the co-cycled is `ell_c`-free, so any
  sufficiently-thin skin works and the partition is mesh-independent. This was the key generalization risk
  (the §10 non-generalization) — and it is RESOLVED: the scheme has ONE partition knob (`w`), not two.
- **Clay-V ROBUST:** the stiff convergent clay-V (which the monolith dt-collapses on) COMPLETES at w=0.7
  on the 2 mm skin (`ns=23, t=0.400`), conserves **8.16e-13**; routed/R 0.923 (physically correct — clay
  `Ks=0.048` runs most off; no monolith target since it collapses, the gate is completes+conserves). The
  scheme RETAINS the sequential split's robustness raison d'être WHILE being accurate + exactly conservative.

**★★ FINAL VERDICT — the partition bug is FIXED.** `CoCycledCappedSplit` (co-cycled sub-stepping, no `I`
reconstruction, weighted film `w`) on a z-graded THIN-SKIN mesh delivers, all at once: **(1) EXACT
conservation** (`~1e-11`–`1e-13`, falsification-verified), **(2) monolith-accurate partition** (b1 base
−0.3 pp, steep +2.7 pp at `w=0.7`), **(3) STABLE** (no collapse), **(4) SLOPE-robust** (single `w`),
**(5) SKIN/mesh-robust** (partition skin-invariant), **(6) clay-V robust** (survives where the monolith
fails). Knobs: `w≈0.7` (the partition/film weight, the one real parameter), `K=6` (sub-steps), a thin top
skin (stability only). REMAINING scope before production: broader soil/storm/slope sweep (only b1 loam +
clay-V tested; `w` may need a wider check), `b1_coarse`/mesh-convergence, then promote
`CoCycledCappedSplit` + `make_graded_box` into `pids_forward/physics/` with TDD (a sibling option to the
galerkin/upwind/monolith/sequential schemes). Spike `seq_cocycled_skin.py` (`cocy`/`mono`/`clayv`);
outputs `scratch/_skin_*.txt`. **⚠ §14 TEMPERS this verdict: it holds for LOAM (the calibration soil) but
`w=0.7` is NOT soil-universal — see below.**

---

## 14. BROADER SOIL SWEEP — `w=0.7` is loam-CALIBRATED, not universal (2026-06-26)

The §13 win was confirmed only on b1 LOAM. The broader soil sweep (`scratch/seq_cocycled_sweep.py`:
loam/sand/silt at S=0.03, each vs its OWN uniform-mesh galerkin monolith target; co-cycled+skin at
`w=0.7`, 2 mm skin, K=6) tests universality:

| soil | Ks | rain | monolith target | co-cycled+skin w=0.7 | result |
|---|---|---|---|---|---|
| loam | 0.25 | 0.5 | 0.5470 | 0.5437 (−0.3 pp), `bal 1e-11` | ✅ accurate + stable |
| sand | 1.5 | 2.5 | 0.6939 | **COLLAPSED** (ns=50, t=0.079) | ❌ high-K stiffness |
| silt | 0.10 | 0.5 | 0.7674 | 0.6565 (**−11.1 pp**), `bal 1e-11` | ❌ over-infiltrates |

**Findings:**
1. **Conservation is soil-UNIVERSAL** — exact (`1e-11`–`6e-13`) on ALL soils, including the sand that
   collapsed. The conservation re-architecture (the headline ask) fully generalizes.
2. **`w=0.7` is loam-calibrated, NOT universal** — silt (lower Ks, target 0.767) over-infiltrates by
   −11 pp at `w=0.7` ⟹ low-Ks soils need a DIFFERENT (lower) `w`. So `w` is soil-DEPENDENT (it is NOT a
   pure numerical-simultaneity knob; it also compensates the scheme's soil-dependent infiltration-rate
   error). The §10 "the closure does not generalize" pattern, now on `w` instead of `h_ref`.
3. **Sand COLLAPSES** even with the 2 mm skin (the high-K near-saturation fragility §10 explicitly
   deferred; `alpha=14.5` sharp retention → the saturated skin→subsoil flux is stiff). The thin skin
   helped loam but is insufficient for sand.

**Lower-`w` probe — `w` does NOT generalize (decisive):**

| soil | target | w=0.7 | w=0.5 | `w` lever |
|---|---|---|---|---|
| loam | 0.5470 | 0.5437 (−0.3) | ~0.65 (+10) | STRONG (Δw 0.2 → ~10 pp) ✅ |
| silt | 0.7674 | 0.6565 (−11.1) | 0.6714 (−9.6) | **WEAK** (Δw 0.2 → 1.5 pp); stuck ~−10 pp |
| sand | 0.6939 | COLLAPSE | 0.5422 (−15.2), STABLE | lower-w STABILIZES but −15 pp |

**⟹ DECISIVE: the `w`-knob does NOT generalize.** Both non-loam soils OVER-INFILTRATE by 10–15 pp, and
`w` cannot recover them — silt is `w`-INSENSITIVE (the soil over-draws the film regardless), sand's
accurate-`w` COLLAPSES (lower-`w` is stable but −15 pp). `w=0.7` only worked because it was tuned to LOAM.
**Root cause (confirmed, the §9–§14 through-line): the co-cycled scheme has NO soil-aware infiltration
CAP** — it offers a film and lets Richards draw it at the soil's Darcy rate, which over-draws for any soil
whose acceptance differs from loam's. The MONOLITH is soil-accurate precisely because it caps at
`q_pot=∫K dψ/ell_c`. **Conservation stays EXACT + soil-universal throughout** (`1e-11`–`1e-13`, incl. the
collapsed sand).

**⟹ Corrected verdict: the §13 "fix" is REAL but LOAM-SPECIFIC. Soil-generality requires a SOIL-AWARE
INFILTRATION CAP, not a tuned `w`.** Leading direction = port the monolith's `q_pot` acceptance into the
co-cycled framework as an ADAPTIVE cap (evaluate `q_pot` at the SOLVED ψ each sub-step + deliver via the
conservative SOURCE — NOT the §9 frozen-`q_pot`-hard-Neumann that failed), on the UNIFORM mesh (where the
cap should ALSO prevent the §12 force-feed collapse → potentially retiring BOTH the `w`-knob and the thin
skin). This is the genuine infiltration-closure research §9–§14 kept circling — now attackable from a
much better base (exact conservation + a stable structure). Alternatives: scope to loam-like soils +
document the envelope; or harden the monolith's stiff-case dt-collapse (it is already soil-accurate, §10
path-2). Spikes `seq_cocycled_sweep.py`; outputs `scratch/_sw_*.txt`.

---

## 15. OPTION A — adaptive q_pot infiltration cap: prototyped + Codex-reviewed (2026-06-26)

Prototype `film_mode="qpot"` on `CoCycledCappedSplit` (offer `film = min(d_routed, q_pot·hsub)`,
`q_pot = max(kirchhoff(min(ψ,0), h_sat),0)/ell_c`, adaptive per sub-step, UNIFORM mesh, no `w`/skin).
**Full record + the verbatim Codex review: `validation/sanity/overland_qpot_codex_review__2026-06-26.md`.**

**Empirical (h_sat=2 mm):** loam 0.8744 (**+32.7 pp**), sand 0.6719 (**−2.2 pp** ✅ + STABLE, where `w`
collapsed), silt 0.9192 (**+15.2 pp**); all exact (`1e-11`–`1e-13`). The cap OVER-ROUTES loam/silt, NAILS
sand, and FIXES sand stability. `w` (over-infiltrate) and the cap (over-route) BRACKET the monolith on
every soil; a single `h_sat` does NOT generalize (errors scatter +33/−2/+15).

**Codex verdict:** conservation SOUND (no leak), force-feed STRUCTURALLY avoided (pond-in-ψ self-limiting,
not §9/V2 hard-Neumann), genuinely sub-step-adaptive — **but a SURROGATE, not the monolith's law:** it
uses a fixed `h_sat` instead of the actual co-solved sheet depth `d`, clamps ψ>0 to 0, caps offered-depth
not flux, is sub-step-FROZEN (not co-solved), and reintroduces the `ell_c` mesh knob. Codex rec: if it
misses soils, do NOT add another global knob — make the cap MONOLITH-FAITHFUL (actual `d` + a small inner
q_pot update); also tighten the `ell_c` autodetect to match the monolith + fix the `max_pond` diagnostic
(samples pre-solve only).

**⟹ Verdict: option A is structurally sound + fixes sand stability, but the fixed-`h_sat`/`ell_c` surrogate
does NOT replicate the monolith's co-solved acceptance → no single-knob soil-generality (the §9 circular
closure: the right kirchhoff sheet depth is the THIN co-solved `d` the sequential scheme lacks). FORK
(Arik): (A′) refine toward monolith-faithfulness (actual `d` + inner q_pot update — a step toward a surface
co-solve); or (C) harden the already-soil-accurate monolith. The bracketing + the sand-stability win say a
faithful cap is reachable, but it converges toward "a mini surface co-solve," narrowing the gap to option
C.** Conservation remains DONE + universal throughout.

---

## 16. A′ — both refinements REFUTED; the sequential-cap approach is EXHAUSTED (2026-06-26)

Per Arik ("finish A′", overland-flow regime only, lakes out of scope), the two A′ refinements were
implemented + tested:

**(1) Actual-`d` cap (`film_mode="qpot_d"`, Codex's lead rec):** `q_pot=kirchhoff(min(ψ,0), d_routed)/ell_c`
using the actual local pond depth (matches `coupling.py:230`). Result vs fixed-h_sat: **loam +33.1 (was
+32.7, UNCHANGED — pond thin so actual-`d`≈h_sat); silt +15.1 (UNCHANGED); sand −10.5 + ponds 3.8mm (WORSE
— deep pond inflates q_pot → over-infiltrate, the §9 effect).** ⟹ actual-`d` is NOT the lever; the
over-routing is `d`-INSENSITIVE.

**(2) Inner Picard (`picard_inner=4`, Arik's "delay/staleness" hypothesis + Codex's "small inner q_pot
update"):** re-evaluate q_pot at the POST-solve soil head + re-solve to a per-sub-step fixed point
(self-consistent q_pot, the closest the split gets to the monolith's per-Newton-iterate q_pot). Smoke
(loam, mid-storm, head-to-head): `qpot` routed **0.8744** → `qpot_pic` routed **0.8748** (inner_avg 3.26
iters) — **UNCHANGED.** ⟹ the q_pot staleness is NOT the cause; making it self-consistent does not move the
partition. (Minor: the convergence-break path leaks ~3e-6; moot, refuted.)

**★★ VERDICT — the SEQUENTIAL-CAP approach is EXHAUSTED.** Across every closure tried — `w` (offer film,
free draw: over-infiltrate/collapse), `h_ref`/`h_sat` (fixed thin cap: over-route), actual-`d` (scattered),
and self-consistent q_pot (unchanged) — NONE replicates the monolith's partition across soils. Each trades
one soil's error for another. The over-routing is structural: the cap is **route-first** (routes the full
sheet each sub-step before the cap acts) and the remaining gap to the monolith is the **routing⟷infiltration
co-solve** the monolith does simultaneously and the split cannot (a "surface co-solve" = rebuilding the
monolith's surface coupling). **Conservation re-architecture (the headline ask) remains DONE + soil-universal.
The loam-scoped thin-skin scheme (§13) remains exact+accurate+stable+robust ON LOAM.** ⟹ FORK collapses to:
**(B) ship loam-scoped** (if PIDS hosts are loam-ish) **or (C) harden the soil-accurate MONOLITH's stiff-case
dt-collapse** (the only path to soil-general accuracy). Spikes: `film_mode={qpot,qpot_d}` + `picard_inner`
in `seq_href_iterated.py`, driver modes `qpot`/`qpot_d`/`qpot_pic` in `seq_cocycled_sweep.py`.

---

## 17. ⚠ §16 RETRACTED — "monolith-only" was PREMATURE (critical review, 2026-06-26)

Arik was NOT convinced by §16 ("other models run independent sequential modules + get infiltration/runoff
correct — no need to reinvent the wheel"). A SKEPTICAL Codex review (full record
`validation/sanity/overland_sequential_critical_review__2026-06-26.md`) **confirms §16 over-generalized:**

- **The canonical sequential closure was NEVER TESTED.** All our closures are `pond-in-ψ` film-offers or
  explicit `q_pot=kirchhoff/ell_c` caps — NOT the standard **Neumann↔Dirichlet SWITCHING BC** (dry nodes →
  rain Neumann flux; ponded nodes → `ψ_top=d` Dirichlet; infiltration READ OFF the Richards solve as the
  boundary reaction flux). That is what CATHY (and the active-set partitioned-coupling literature) actually
  do. Production `sequential_coupling.py:18` even says "NO hard Dirichlet pin."
- **The `q_pot` cap structurally MISSES the gravity/Dirichlet limit:** as `ψ_top→0`, `q_pot≈Ks·d/ell_c`
  (tiny, a film-resistance), whereas a true ponded Dirichlet Richards solve carries the unit-gravity term
  and → ~Ks at steady ponded infiltration. So the `qpot` over-routing is a KNOWN artifact of a non-standard
  closure, NOT evidence that "sequential is exhausted."
- **The monolith is also NOT a switching-BC closure** (it is a co-solved Robin/NCP with our `q_pot` law) —
  so "monolith works ⟹ sequential needs co-solve" is an invalid inference.
- Example nuance: CATHY = right kind (sequential switching BC); HydroGeoSphere/ParFlow = actually co-solved;
  GSSHA/tRIBS = Green-Ampt/Smith-Parlange infiltration-CAPACITY models (a different untested closure family).

**⟹ CORRECTED VERDICT: the fork is NOT yet B-vs-C.** The honest scope of §13–§16: the `pond-in-ψ` +
`q_pot=kirchhoff/ell_c` closure FAMILY does not generalize. The CANONICAL switching-BC closure (the
standard CATHY/HGS/GSSHA approach) is UNTESTED and is the clear next step. **NEXT = implement a true
active-set Neumann↔Dirichlet switching BC** (separate surface store `d`; ponded→Dirichlet `ψ_top=d`; read
infiltration as the boundary reaction flux; conservative `d` update; Picard the active set within the
step). Conservation re-architecture (§12) stays DONE + universal. Sources: Schüller 2025 (arxiv 2408.12582),
Sochala 2009 (0809.1558), Berninger 2014 (1301.2488).

---

## 18. ★★ SWITCHING-BC SPIKE — works + faithful; the "0.547 target" is RESOLUTION-DEPENDENT (2026-06-26)

Built the canonical Neumann↔Dirichlet active-set switching BC (`scratch/seq_switching_bc.py`,
`SwitchingBCSplit` — the standard CATHY/HGS closure §17 said we'd never tested): separate surface store;
DRY→Neumann rain; PONDED→Dirichlet ψ=0 (per-node penalty `c_pen·ψ`, no form rebuild); SUPPLY→Neumann
avail/dt; 3-mode active set iterated; infiltration = the weak-form boundary flux (= Δθ exactly, gravity
INCLUDED). It RUNS + CONSERVES (1e-12 fine mesh, 9e-10 coarse) + active set converges (~1.3 iters) + the
penalty holds ψ≈0.

**Coarse-mesh partition (vs the monolith targets):** loam 0.254 (−29 pp), sand 0.077 (−62 pp, COLLAPSED),
silt 0.530 (−24 pp) — OVER-infiltrates all soils, scattered, sand collapses (ψ=0 over-saturates the 125 mm
top cell, worst for high-K). NOT a clean match to the coarse targets.

**★ THE REFRAME (decisive).** The mismatch is RESOLUTION, not a sequential limitation:

| loam | coarse (125 mm cell) | 2 mm thin-skin |
|---|---|---|
| MONOLITH | **0.547** (the "target") | 0.254 |
| SWITCHING BC | 0.254 | 0.210 |

On the SAME 2 mm-skin mesh, switching-BC 0.210 ≈ monolith 0.254 (~4 pp) — they AGREE at equal resolution.
**The 0.547 target is the COARSE-mesh value of the monolith's `q_pot=kirchhoff/ell_c` SUB-GRID FILM MODEL;
it is NOT mesh-converged.** Refine the surface and the film model vanishes (`ell_c→0 ⟹ q_pot→∞ ⟹ cap
inactive ⟹ raw Richards flux`), and BOTH schemes drop toward ~0.21–0.25 (the resolved sorptive front = MORE
infiltration). The §13 "monolith thin-skin 0.254" was this same effect (the cap going inactive), NOT a
"broken cap." **⟹ the §13–§16 "sequential can't match the monolith" framing was a TARGET CONFUSION:
variable-effective-resolution sequential schemes compared against ONE coarse `q_pot` value. The canonical
switching BC WORKS + is FAITHFUL to the co-solve at equal resolution (Arik was right).**

**⟹ THE REAL OPEN QUESTION (Arik's domain): what is the PHYSICALLY CORRECT partition?** The coarse `q_pot`
value (loam 0.547, ParFlow-validated AT ParFlow's resolution) or the mesh-converged value (~0.21, the
resolved sorptive front)? `q_pot=kirchhoff/ell_c` is a sub-grid film closure with a resolution-dependent
answer; the mm-scale physical infiltration interface argues for the fine value. NEXT = a MESH-CONVERGENCE
study (switching BC + monolith at coarse/medium/fine surface — where do they converge?) + reconcile with the
ParFlow benchmark resolution. Spike `seq_switching_bc.py`; commit `90116ab`.
