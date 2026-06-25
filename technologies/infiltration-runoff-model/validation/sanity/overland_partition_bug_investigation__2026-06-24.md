# Overland Run-on PARTITION Bug — Investigation & Session Record (2026-06-24)

> **Status:** OPEN. Diagnosis complete + ground truth established; THREE fix attempts refuted. A fix
> DIRECTION is NOT yet chosen — to be revisited next session with fresh context (Arik 2026-06-24:
> "multiple other models address infiltration correctly without the monolith"). This document is the
> handoff. Companion memory: `pids-overland-partition-bug.md`.

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
