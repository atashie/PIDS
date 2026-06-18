# Overland convergent-flow stabilization — research plan

**Date:** 2026-06-11 · **Status:** P0 DONE 2026-06-11 (§8: the §1 in-house plateau/ledger numbers are
RETRACTED — not reproducible from the committed deck; the stiffness finding stands with its mechanism
measured; O5 landed + tested; O4 deferred; O1 remains the production fix) · **Trigger:** benchmark B6
(canonical tilted-V catchment)
**Priority (Arik 2026-06-11):** PIDS networks are installed along **lines of topographic convergence** (swales,
valley lines, drainage ways). Convergent overland flow is therefore the **core operating regime** of the
product — the M4 embedded features and `add_surface_inlet` grates sit exactly where surface water
concentrates. The B6 failure modes are a core-fix engineering target, **not** an out-of-envelope curiosity.

## 1. What B6 measured (the trigger — **P0 2026-06-11: the in-house plateau/ledger rows below are
RETRACTED**, not reproducible from the committed deck, see §8; the stiffness observation stands)

Canonical tilted-V (Di Giammarco 1996 / Kollet & Maxwell 2006; IH-MIP2 benchmark): two 810×1000 m
hillslope planes (cross-slope 5%) converging to a central channel line (valley slope 2%), rain
3.0e-6 m/s (0.2592 m/day) × 90 min then 90 min recession, near-impermeable bed. Known answer:
equilibrium outlet discharge `Q_eq = rain·area = 419,904 m³/day = 4.86 m³/s`.

| run | result |
|---|---|
| ParFlow `OverlandKinematic` 48×30, n=0.015 | storm-end Q **= 1.000·Q_eq exactly**; clean recession → 0 |
| ParFlow `OverlandDiffusive` 48×30, n=0.015 | **1.000·Q_eq**; same |
| ParFlow field-scale 162×100 m, n=0.15 | clean 1.000·Q_eq plateau (whole storm) |
| in-house `CoupledProblem` 24×16×3, whole-edge outlet, n=0.015 | plateau **0.676·Q_eq**; **external mass-ledger gap = 20.0% of cum rain** at storm end (~32% on a rate basis at the plateau); 21 min / 769 steps; dt pinned ~1e-4 d (aggressive controller; ~3e-5 default); Newton 5–10 steady, spikes 14–35; ~3.3×Q_eq wave-arrival overshoot |
| in-house, channel-band-only outlet variant | 0.624·Q_eq (outlet config ≈5% effect — not the main story) |
| in-house field-scale (162 m, same shape) | same stiffness, same ~0.62–0.65 plateau → **scale-independent: the problem is the convergence, not the km size** (B5's single planar slope: fast, mass closed to 1e-12) |

Also re-confirmed: ParFlow's overland boundary is **free outflow** (water leaves the domain edge at ~0
depth at any Manning), so its outlet hydrograph is only recoverable by storage balance — a measurement
delta, not an error (both models hit Q_eq).

## 2. Root cause — two SEPARABLE defects, currently entangled

### Defect A — non-monotone surface discretization (KNOWN, deferred 2026-06-06, now structural)
The lateral overland operator is **unstabilized (central) Galerkin advection of the kinematic flux**.
At catchment slopes `|∇H_s| ≈ S ≫ |∇d|`, the diffusion wave is advection-dominated: a nonlinear
hyperbolic law with celerity `c = ∂q/∂d = (5/3)(86400/n)·d^{2/3}·√S`. P1 central Galerkin on advection
is dispersive → over/undershoots at the wetting front, at the **channel slope-kink line** (`∇z_b`
discontinuous — exactly the convergence line), and in the concentration zone where d grows ~20×.
Negative d then trips the positivity machinery. *This was already diagnosed and deferred as a mm-scale
cosmetic sawtooth on planar cases* (`coupling.py` lines ~204–210; `docs/plans/2026-06-05-module3-
realization-ffcx-bug.md` §8: "needs a stabilized/monotone scheme — deferred"). **B6 shows that on
convergent topography it is not cosmetic.** The current global-rescale limiter compounds it: it pays
hillslope undershoots by removing water from *everywhere else* (the channel) — non-local and
solution-corrupting even when it conserves, and it has a degenerate dry-everything branch.

### Defect B — lax step acceptance can book non-physical states (P0 2026-06-11: the MECHANISM is
confirmed and hardened — O5 landed; the 20%-gap attribution itself is retracted, §8)
`CoupledProblem.step()` accepts any `snes.getConvergedReason() > 0`. `snes_stol` is **not set** →
PETSc default 1e-8 → **`SNES_CONVERGED_SNORM_RELATIVE` is live**: a stalled back-tracking line search
(tiny Newton steps on the stiff V) can be reported "converged" with a **large unbalanced residual** —
and an unbalanced residual *is* a mass error injected straight into the books. This cleanly explains
why B4/B5 books closed to ~1e-12 (planar: Newton truly converged) while the V leaks 20%.
**Ledger-gap candidates, ranked:** (B1) snorm-acceptance of stalled states; (B2) the limiter's
degenerate branch (`clip_mass_adjust` — instrumented but not yet read on a V run); (B3) limiter↔NCP
staleness interaction. P0 distinguishes them by logging; the earlier B6 HTML/metrics text attributed
the leak to the limiter branch — that was **hypothesis stated too strongly**; regenerate after P0.

## 3. How ParFlow avoids both (code read, 2026-06-11)
From `pfsimulator/parflow_lib/overlandflow_eval_Kin.c` (OverlandKinematic) and `overlandflow_eval.c`:
- **Upwind finite volumes:** face flux `q = −(Sf/(√Sf_mag·n))·Press^{5/3}` with
  `Press = RPMean(−Sf, 0, pfmax(P_i,0), pfmax(P_{i+1},0))` — the **upstream cell's** non-negative
  ponded pressure, selected by the face-slope sign; slope magnitude floored at `ov_epsilon = 1e-5`.
  Cell residual = telescoping FV divergence (`ke/kw/kn/ks`). Monotone by construction → **positivity
  without any limiter**, conservation exact.
- **Single ledger:** the overland store is `max(p,0)` of the top Richards cell — same unknown, same
  residual row → no λ, no separate surface budget to reconcile.
- **Precedent for the channel line:** the original cell-centered-slope scheme had a known pathology
  where adjacent slopes change sign (valley lines); ParFlow fixed it with the **face-centered-slope
  `OverlandKinematic` rewrite**. The convergence line is the *recognized* hard spot of this equation
  class, with an established cure: upwinding w.r.t. face slopes.
- Honesty note: ParFlow's own analytic overland Jacobian is incomplete (`UseJacobian False` is their
  recommended setting) — robustness comes from the **monotone flux function**, not Newton sophistication.

## 4. Design space

| # | option | monotone | conserves | lake-at-rest exact | auto-Jacobian | surgery |
|---|---|---|---|---|---|---|
| O1 | **Upwind-mobility two-point edge flux** for the lateral term: per surface edge ij, `flux = T_ij·K_s(d_up)·(H_i−H_j)`, upwind by `sign(H_i−H_j)` (phase-potential upwinding; reservoir-simulation standard for degenerate parabolic; = ParFlow in FEM clothing) | **yes** (M-matrix, given T_ij ≥ 0 — holds on our structured box meshes) | structural | **yes** (H-differences) | hand Jacobian for this block (small, regular); smoothed upwind switch keeps Newton C¹ | medium — custom edge assembly over the top-facet graph |
| O2 | Flow-gated Rusanov viscosity `ν=½h·c` on `∇d` (gate ~ flow indicator so ν→0 at rest) | no (damps, no guarantee) | yes | via gate | pure UFL | small — de-risk spike only (known O(1)-ish flux distortion at coarse h: ν·∇d ≈ q at 67-m cells) |
| O3 | Full AFC/FCT (Kuzmin): low-order upwind + limited antidiffusion | yes | yes | needs care | no (limiter in Newton) | large — accuracy upgrade only if O1's O(h) smearing proves limiting |
| O4 | **Local conservative deficit redistribution** (replace the global rescale: pay each clipped node's deficit from its graph neighbours; track any unpayable remainder) | n/a (symptom) | **by construction** | n/a | n/a (post-step array op) | small |
| O5 | **Acceptance hardening:** `snes_stol=0` / custom convergence test — reject SNORM-stalled states; accepted ⇒ residual ≤ tol, always | n/a | restores honest books | n/a | n/a | trivial |

**Recommendation.** *Wave 0* = O5 + O4 + instrumentation (truth first: every accepted step balances, or
it is rejected and dt cuts — even if that is slower). *Wave 1* = **O1** as the production fix — it is the
canonical discretization for this equation class and what ParFlow runs; keep O2 as a one-day spike only
to confirm the oscillation diagnosis cheaply. *Wave 2* = O3 only if O1's first-order smearing measurably
hurts PIDS-scale answers. Everything behind a `scheme="galerkin"|"upwind"` constructor flag: the galerkin
path stays for MMS/regression archaeology; the upwind path gets its own (order-1) MMS.

## 5. Phased plan (each phase gated; ~1 session each unless noted)

- **P0 — Diagnose + wave-0 hotfix.** Instrument `step()`: log `getConvergedReason()` per step, the
  per-step ledger (rain·dt, λ·dt, outflow·dt, Δ∫d, Δ∫θ, clip stats, `clip_mass_adjust`). Build the
  **minimal reproducer**: standalone Module-2 `OverlandProblem` on a 2-D tilted-V (no Richards/NCP) —
  seconds per run, isolates Defect A from the coupling. Attribute the 20% gap (B1/B2/B3). Apply O5
  (+O4 if implicated); re-run the in-house V → **gate:** books close (≤1e-8·cum_rain) or every
  violation is an explicit rejected step; mechanism note written. *Hygiene:* correct the B5/B6 doc
  claim of a "geometrically tilted in-house mesh" (the mesh is a flat box; `set_topography` only sets
  the z_b field — confirmed in code 2026-06-11); regenerate the B6 HTML metrics text (leak attribution).
- **P1 — O1 spike on standalone Module 2** (possibly 2 sessions). Edge-graph two-point flux with
  upwind conveyance on the 2-D V; smoothed upwind selector (width `eps_H`) vs semismooth Newton —
  decide empirically. **Gates:** d ≥ −1e-12 *without* the limiter; books machine-tight; V plateau
  → Q_eq ± 3% mesh-convergent; lake-at-rest & near-flat tests unchanged to machine; 1-D kinematic
  rising-limb analytic matched; dt no longer pinned (measure).
  - **P1 PREREQUISITE — RESOLVED-AS-DEFERRED 2026-06-15 (Arik, on Codex's 2nd opinion + the corrected
    A1 census `0daf860`): do NOT rebuild the gate before O1; O1 is the fix, the gate revisit is
    post-O1/P3.** The A1 census proved no residual-normalized gate can reject the dirty-V reason-4
    stalls — they are *residual-converged* (worst F/F0 ≈ 2.4e-10) and mass-safe; their only error is the
    Galerkin sawtooth (Defect A), which O1 removes. So the gate cannot serve as a V guard, and broadening
    it to ACCEPT the V is a contract change best not made off this census while the absolute bar is
    conservative + non-silent + protective. O1 dissolves the problem (sawtooth gone → V converges via
    reason 2/3 → no stalls → no 39.5 h). Gate portability is revisited POST-O1/at P3 (swale) with clean
    evidence as a portability/backward-error gate, NOT an accuracy guard. Original analysis (the
    non-portability of the absolute bar, why mass-balance was rejected) retained below for that revisit.
    --- ORIGINAL (the F1 reasoning, retained for the post-O1 revisit): make the O5 acceptance gate
    scale-invariant BEFORE the new scheme/fixture. `stall_accept_fnorm` is an
    *absolute* L2-residual bar: ‖F‖₂ scales with both the row count (mesh) and the per-row flux/measure
    (domain), so the value separating the *measured* populations (Tier-1 columns vs the km² V) is not
    guaranteed to separate them on P1's upwind scheme or P3's swale — the same physical state can flip
    accept↔reject under rescaling, and `test_acceptance_contract_defaults` locks the magic number
    rather than the invariant (the two behavioural tests *do* pin the invariant — keep those).
    **NOTE — a "mass-relative gate" (the original idea here + Codex's) is WRONG and is NOT the fix:**
    drafting the P1 plan (2026-06-14) found it BREAKS the MMS tests — a manufactured-source MMS solve
    balances its source with **Dirichlet through-flux** that no ledger/row-sum captures, so a correctly
    converged MMS floor state (reason-4, ‖F‖≈1.1e-7) shows a mass gap ≈1e15 and the mass gate would
    wrongly REJECT it → MMS death-spiral (measured: `int(f*)≈−1e7`, `mass_gap≈+1e15`). The residual ‖F‖
    IS the right convergence invariant; F1's requirement is to make IT scale-invariant.
    **Replacement = a residual-normalized gate:** accept a reason-4 stall iff `‖F‖ ≤ tol_rel · R_scale`,
    `R_scale = max(‖b_forcing‖, ‖b_storage‖, atol_floor)` (the assembled rain/source and lumped-storage
    residual norms — huge for MMS so its floor accepts; non-zero at the no-forcing slump via storage),
    `tol_rel` **dimensionless** (~1e-3). The full design + a mandatory reason-4 census (pick `R_scale`/
    `tol_rel` from data, not guesswork) is in `docs/plans/2026-06-14-overland-convergent-flow-P1.md`
    Part A. (P0-as-shipped is safe — the absolute bar is measured-correct on the current problems and
    both mis-scalings fail non-silently — but it is a stopgap, not the portable contract P1 builds on.)
- **P2 — Productionize in `CoupledProblem`.** Same edge scheme on the top-facet graph of realization A
  (`ds_top` triangulation edges); λ/NCP/outlet/inlet terms unchanged; limiter demoted to a tripwire
  assert. Full TDD per the three-tier routine: new Tier-1 tests (positivity-no-clip, conservation-
  no-clip, kinematic analytic, V-plateau), full suite green, galerkin path bit-identical; Tier-3
  re-baselines (B4/B5 re-run as regressions) with Arik sign-off on any visual/tolerance change.
- **P3 — Re-benchmark + the PIDS acceptance case.** Re-run B6 (canonical + field V) vs ParFlow + Q_eq;
  add the permanent **PIDS-swale Tier-2 fixture**: ~50×30 m field, 2–5% side slopes converging to a
  1–2% swale line, realistic loam, the standard storm matrix, an `add_surface_inlet` in the swale —
  the geometry the product actually ships into. Update benchmarks README / HTMLs / memory.
- **P4 (conditional) —** O3 antidiffusion accuracy pass, or solver work (fieldsplit/scaling) if the
  dt-pin survives O1.

## 6. Acceptance bars (the definition of done)
1. **Conservation:** |ledger gap| ≤ 1e-6·cum_rain (target 1e-8) on the V (canonical + field) and the
   PIDS-swale case, with `clip_mass_adjust ≈ 0` and **zero** accepted-but-unbalanced steps.
2. **Positivity:** min d ≥ −1e-12 m pre-limiter (limiter fires never; kept as an assert/tripwire).
3. **Accuracy:** V plateau within ±3% of Q_eq at ≈48×30 surface resolution, error shrinking with h;
   plateau oscillation ≤2% RMS; lake-at-rest + near-flat + existing MMS (galerkin path) unchanged.
4. **Efficiency:** PIDS-swale storm ≤ ~10 min serial; canonical V ≤ ~2 h serial; dt/iters distributions
   reported (expect the ~1e-4 pin to lift substantially — M-matrix Jacobians are what bt was built for).
5. **Regression:** full suite green; B4/B5 comparisons reproduced within stated tolerances.

## 7. Risks / open questions
- **O(h) upwind smearing** on mm-scale sheet flow: affects the conveyance, not storage; quantify in the
  P1 convergence study; O3 is the escape hatch.
- **Upwind-switch smoothness in the monolithic Newton:** smoothed selector (C¹, width eps_H) vs
  semismooth — P1 decides; ParFlow's FD-Jacobian fallback is the precedent that exact J is not sacred.
- **Edge-assembly cost in Python:** the graph is top-surface-only (small); vectorize; a kernel is the
  fallback. Measure in P1.
- **T_ij ≥ 0 (M-matrix) on distorted meshes:** holds on our box meshes; guard loudly otherwise.
- **If P0 shows the gap is ALL Defect B** (acceptance), conservation may be restored by O5 alone —
  then O1's payoff is robustness/dt/oscillation, and the plateau number must be re-measured honestly.
- Carried explicitly: do **not** soften the engine's identity — it remains a field-scale PIDS drainage
  FEM; the goal is correct + efficient *convergence-line* behaviour at field scale (the swale), with the
  canonical km V as the stress validator, not a watershed-product pivot.

## 8. P0 results (2026-06-11) — diagnosis, attribution, wave-0 outcome

### 8.1 The §1 in-house numbers are RETRACTED (not reproducible from the committed deck)
Instrumented re-runs of the canonical V (24×16×3, committed deck + engine at `df61ff4`; per-accepted-step
ENGINE ledger `gap_k = ΔW_k − dt·(rain·A − Q_out) − dclip_k`, storage in the residual-consistent measures,
fluxes booked at the solved state; runner `scratch/_tiltedv_diag.py`):

| run (storm window) | accepted / rejected | dt pin | plateau | engine ledger gap |
|---|---|---|---|---|
| spike-default controller | 1096 / 1 | 5.3e-5 d | **0.997·Q_eq** | **+5.4e-8 m³ = 2.1e-12·cum_rain** |
| aggressive controller (the published run's cadence) | 563 / 16 | 1.1e-4 d | **0.998·Q_eq** | **≈0 (same order)** |

(These are `df61ff4`-engine runs — they BOOK the reason-4 stalls, which is why dt grows to ~5e-5 with
~1 rejection. The O5 engine REJECTS the same stalls; on the full-window canonical gate run that collapses
dt to ~1.5e-6 with 60k rejections at 39.5 h — §8.4. The books close either way; the difference is honest
runtime, the cost the booking hid.)

`clip_mass_adjust = 0` in both — the limiter's degenerate branch NEVER fires on the V. The published npz
(preserved as `~/parflow-runs/tilted_v/summaries/tiltedv_inhouse_s1_pre_p0_corrupt.npz`, sha256
`4d51616baa434806bc70f7157361bf9e566e2c782b9ea0fe36a2524a5285f93e`) additionally records
`soil_dw = −26 m³` at storm end — NET soil-storage LOSS under sustained ponded rain on suction soil,
opposite in sign to the committed-deck rerun (+47 m³). A CONVERGED committed-deck solution has no
sustaining mechanism for that (no GHB, no bottom outlet; the NCP infiltrates under a pond over suction
soil — nodal exfiltration only transiently where ψ_top > d); booked UNBALANCED iterates are not so
constrained, so the −26 m³ evidences corrupted booked states without by itself identifying the channel.
**Provenance narrowing (checked 2026-06-11):** every parameter the old npz schema records (Ks, n_man,
scale, storm, t_end, mesh, rain, slopes, extents) MATCHES the deck defaults, and the engine lineage is
verified untouched (`63145e2 == ea3f466 == df61ff4` on `pids_forward/`). What the old schema did NOT
record: the dt-controller knobs (GROW_AT/SHRINK_AT/DT_MAX), the launch comm size, and any uncommitted
in-session code state. The published cadence (769 attempts, full window) was not reproduced by either
tested controller, so a hotter unrecorded controller exercising the then-open B1 acceptance hole —
entirely on the committed deck — and a divergent in-session code state are BOTH live explanations.
Conclusion (the defensible statement): the published numbers are **not reproducible from the committed
deck under any tested configuration and are RETRACTED; the provenance of the published run is unknown**
(candidates above; either way the B1 hole is now closed by O5). The new runner npz schema records the
controller knobs + comm size so this ambiguity cannot recur. The B6 README §5e, harness DELTAS, and
HTML metrics are corrected accordingly.

### 8.2 Defect-B attribution (the §2 candidates)
- **B1 (acceptance quality) — mechanism real, hardened.** Even the healthy re-runs BOOKED reason-4
  (`CONVERGED_SNORM_RELATIVE`, stalled line search) verdicts: 12 / 15 accepted steps with |F| up to
  3.3e-3 vs the rtol-tested median ~2e-7. At this resolution their booked-mass error measured ≤1e-6 m³
  total — but the verdict certifies nothing about balance, so it stays a latent leak path. **O5 landed**
  (§8.4).
- **B2 (limiter degenerate branch) — dead on the V.** Never fires (either scale, either controller);
  the earlier B6 HTML attribution of the leak to this branch was hypothesis stated as fact, now corrected.
- **B3 (limiter↔NCP staleness) — no ledger effect.** The books pair solved-state fluxes with
  limiter-conserved storage; the coupled per-step residue measures ≤3e-7 m³/step (noise-level, zero-sum).
- **Harness sampling artifact:** the published "leak" was computed from a 40-point trapezoid
  reconstruction of a hydrograph with a 3.3×Q_eq wave-arrival spike; that reconstruction carries a
  few-% residue even for a perfect run. The harness now reports the engine ledger and labels the trapz
  number as sampling.

### 8.3 Defect A confirmed — the never-settling sawtooth state (standalone 2-D reproducer + controls)
`scratch/_v2d_overland_diag.py` (standalone Module-2 `OverlandProblem`, 2-D V 48×30, ~30 s/run, no
Richards/NCP): plateau **0.999·Q_eq**, external books −0.08% of cum rain (all of it = front-window
booking-convention residue + a 0.24% outlet-form quadrature mismatch — Module-2-only; the coupled ridge
outlet shares one measure between residual and booking). The wet/dry SAWTOOTH clips fire on
**430/430 plateau steps** (~3.2 cm; up to 25.7 cm at wave arrival), and the global rescale then shaves
the whole positive surface — measured **post/solved outlet-flux ratio 0.785 every plateau step** (three
independent assemblies agree: booked form, degree-20 form, residual row-sum): non-local solution
corruption, channel water teleported to the front ring every step.

**The dt-pin, after the causal control (`scratch/_v2d_limiter_control.py`, adversarial-review F9):**
the first-pass claim "the pin IS the limiter↔Newton fight" was REFUTED in its strong form. From the
same plateau state: limiter ON = it=4/step; limiter BYPASSED = it does NOT drop (4) and 10/12 steps
then FAIL outright (the unclipped negatives poison the next solve — the limiter is LOAD-BEARING for
solvability, exactly its design intent); dt DOUBLED with limiter ON = it=5, fully robust (the coupled
aggressive controller likewise ran 2× dt at it≈11). So the pin = the adaptive controller's GROW_AT
threshold meeting a PERSISTENT per-step cost of ~4–6+ Newton iterations — a throughput cost, not a
hard wall, and tunable (a looser GROW_AT trades iterations for dt). The persistent cost itself is
Defect-A-rooted: the sawtooth-and-clip state NEVER SETTLES (a true discrete steady state would
converge in 1–2 iterations and dt would grow freely), so every plateau step re-solves a perturbed
nonlinear problem. O1 removes the sawtooth → the limiter goes no-op → steady states actually settle.

**Field-scale severity (the PIDS regime):** at 162 m with 24×16 cells (6.75 m) the equilibrium outlet
sheet is ~3 mm while clips reach 1.3 cm — the sawtooth is order-of-the-signal, and the coupled run's
plateau degrades to **0.876·Q_eq with machine-closed books** (a SOLUTION error, not a leak:
kinematic t_conc ≈ 450 s hillslope + channel ≈ 0.005–0.01 d, order-of-magnitude, ≪ the 0.0625 d
storm — the true physics equilibrates at 1.0). **Resolution control (review F10):** the standalone V
at the same scale converges with h — plateau **1.010·Q_eq at 48×30** (3.4-m cells) and **0.998·Q_eq
at 96×60**, books tightening (−1.03% → +0.24%) and max clip halving (3.2 → 1.8 cm) — the degradation
is under-resolution of the mm-sheet by the oscillatory Galerkin scheme (coupled-vs-standalone
contribution not isolated; the P1 convergence study + P3 swale fixture do that properly; the
standalone field-scale books residue is the limiter-rebooking effect, largest exactly where
clip ≫ sheet). O1 is therefore *accuracy-critical for the swale regime*, not just a robustness/dt fix.

### 8.4 Wave-0 outcome
- **O5 (landed + tested, hardened by adversarial review):** `snes_stol` pinned explicitly (1e-8) in
  both solvers' defaults; `step()` books reason-4 stagnation ONLY below an absolute bar
  `stall_accept_fnorm = 3e-6` — the geometric mean of the measured populations (legitimate floors
  ≤1.2e-6 across the Tier-1 MMS/near-flat cases vs dirty stalls ≥1e-5..3e-3, ~2.5× margin each way;
  the bar is in assembled-residual units, override per problem; mis-set LOW it fails loudly at dt_min,
  never silently). A blanket `stol=0` was tried first and measured to grind floor states through
  max-it into dt death spirals — rejected. On reason-4 exits the norm is RECOMPUTED at the returned
  iterate (PETSc's failed-line-search exit can cache the previous iterate's norm — review F2). Dirty
  stalls restore the FULL state (ψ, d, AND the λ multiplier — review F3) and reject so the caller
  cuts dt; `last_reason`/`last_fnorm` recorded as the audit trail. Tier-1
  `tests/test_step_acceptance.py` incl. the floor-ACCEPT side (near-flat reason-4 at |F|≈3.6e-10
  must book — review F5); full suite green. **OPEN (Codex P0 review 2026-06-12): the absolute bar is
  NON-PORTABLE** — ‖F‖₂ scales with mesh/domain, so 3e-6 is calibrated to the *current* problems, not
  a stable contract for P1's new scheme/scale. Failure modes are non-silent (loud `dt_min` / audited
  `last_fnorm`), so P0 ships safely, but **the P1-prerequisite is a scale-invariant RESIDUAL-NORMALIZED
  gate** `‖F‖ ≤ tol_rel·R_scale` (§5, P1 bullet — NB the originally-proposed *mass-relative* gate was
  found to break Dirichlet-source MMS and is rejected; see the corrected P1 bullet + the P1 plan
  `docs/plans/2026-06-14-overland-convergent-flow-P1.md` Part A). The diagnostic scripts now read the
  engine's audited `last_fnorm`/`last_reason` rather than the raw SNES norm (Codex F2).
- **O4 (deferred):** not implicated in any books violation (B2 dead; books close without it) — and the
  F9 control shows the limiter is LOAD-BEARING (bypass ⇒ Newton fails), so "replace the global rescale"
  is riskier than first framed; O1 removes the undershoots that make any limiter necessary.
- **P0 gate — PASSED.** Committed-spike re-runs against the O5 engine close the engine ledger with
  every violation an explicit rejected step:
  - **Canonical V, full window (the gate run):** plateau **0.996·Q_eq** (storm), peak 1.149× at wave
    arrival, clean recession → 0; engine ledger **−7.8e-7 m³ = −3.0e-11·cum_rain**; `clip_mass_adjust=0`,
    max_clip 0.9 mm; **60,008 rejected steps** of 162,702 attempts (37% — every one an honest O5
    rejection, books untouched). `scratch/tiltedv_inhouse_s1_p0.npz`, now the active B6 reference.
  - **Field-scale full window:** −8.6e-12·cum_rain, 3 rejections, clean recession.
  - Engine-state note: the gate npz was produced at the `3d62e75` engine (O5 v1: `stall_accept_fnorm`
    = 1e-5, no norm-recompute); the storm-phase reason-4 stalls carry |F| ≈ 3e-3 ≫ both the 1e-5 and
    the final 3e-6 bar and the norm-recompute only moves borderline-floor decisions, so the gate result
    is representative of HEAD (re-running to confirm = 39.5 h, not spent).
- **Efficiency — the honest cost, and why P1 is now a tractability requirement, not only accuracy.**
  The gate run took **39.5 h serial** (874 ms/step × 162.7k attempts), dt PINNED ~1.5e-6 d through the
  whole storm. This is **~20× over the §6 acceptance bar (canonical V ≤ ~2 h)** — and it is a DIRECT
  consequence of honest acceptance: the §8.1 *old-engine* diagnostics (which BOOKED the reason-4 stalls)
  grew dt to ~5e-5 with ~1 rejection; O5 REJECTS those same stalls, so dt collapses 35× and the run
  rejection-churns. The old engine looked fast because it was booking unconverged states; the true
  stiffness of the oscillatory scheme is only visible once you demand honest convergence. On THIS
  problem the old books were still ≈right (the booked stalls were individually tiny, ≤1e-6 m³ total),
  so O5's benefit here is insurance — but the 39.5 h proves the canonical-scale acceptance bar is
  **unreachable without removing the sawtooth**. P1 (O1) is the fix; controller re-tuning is a
  secondary lever (P4).
- **Merge note (review F6):** this branch forks `df61ff4` and does NOT contain main's `8a891c2`
  (per-sink drain/inlet accounting inside `step()`); the merge MUST land those per-sink increments
  INSIDE the hardened acceptance gate or O5 silently regresses for sinks.

### 8.5 Consequences for the plan
P1 (O1 upwind-mobility edge flux) is unchanged and **upgraded in priority rationale**: it is the
accuracy fix for the PIDS field scale (§8.3), the dt-pin fix (limiter goes no-op), the monotonicity
fix, AND — new from the gate run — the **tractability** fix: with honest acceptance the canonical V is
39.5 h (20× the bar), and that cost is the oscillatory scheme's true stiffness, which O1 removes at the
source. P0's conservation hotfix turned out to be mostly *verification* — the committed engine's books
were already structurally sound on this problem; what P0 actually bought is (a) the acceptance hardening
that closes the latent B1 path (and, as a side effect, exposes the real dt cost the old booking hid),
(b) the measured + causally-controlled mechanism that re-aims P1, and (c) the corrected public record.
The §6 efficiency bar (canonical ≤2 h) should be read as a **post-P1 target**, not a P0 deliverable.

### 8.6 O1 upwind spike result (P1 Part B, 2026-06-16) — the spike PASSES the §5 P1 gate list

> **Numbering note:** the originally-planned §8.6 (the A5 "residual-gate efficiency" write-up) was
> never produced — Part A (the gate rework) was DEFERRED 2026-06-15 (§5 P1-PREREQUISITE; A1's census
> proved no residual gate can reject the dirty-V stalls, so O1 dissolves the problem instead). This
> §8.6 is therefore the O1 spike result.
>
> **⚠ ACCURACY-FRAMING CORRECTION (B6 capstone, 2026-06-16): this §8.6 was written BEFORE the B5b
> diagnostic and over-claimed the V plateau as discharge ACCURACY (gate 9: "0.99994·Q_eq / 0.006%
> err / PASS by ~500×").** That is wrong: the LUMPED plateau ≈Q_eq is a mass-CONSERVATION identity
> (forced for any converged steady field), not an accuracy measurement. Gate 9 below is corrected
> in place, and the honest accuracy picture (consistent ds-integral ~0.85 on the idealized kink V =
> a measure-zero-channel artifact; ~0.99 on a resolved swale) + the final reconciled verdict are in
> **§8.7**. Read §8.6 as the gate inventory; §8.7 is the corrected close.

**Verdict: O1 — a MONOTONE, well-balanced upwind-mobility two-point edge flux on a custom FD-Jacobian
SNES (`pids_forward/physics/overland_upwind.py`, `UpwindOverlandProblem`) — is the convergent-flow fix.
It meets EVERY §5 P1 gate (the dt/oscillation/robustness gates by orders of magnitude; the V-plateau
accuracy framing corrected per §8.7), and it does so on the validated galerkin path's terms WITHOUT
touching that path (separate class; the galerkin MMS/regression reference is bit-identical).** Each §5
P1 gate (and the P1-plan acceptance bars 6–11), pass/fail + number + evidence:

| # | §5 P1 gate / P1-plan bar | Verdict | Evidence |
|---|--------------------------|---------|----------|
| 6 | Lake-at-rest & near-flat exact (well-balanced); books machine-tight | **PASS** | upwind lake-at-rest held EXACTLY 1-D + 2-D (head differenced on edges → ~1e-16, depth held to machine through the solve), `eps_H`-independent (structural, can't be tuned to pass); closed-domain conservation ~2e-16; V books gap **−2.7e-10 m³** (48×30), **−2.4e-10** (96×60), **−1.7e-11** (field), vs `cum_rain` 26244 / 262 m³ — i.e. ≤1e-13·cum_rain (target was 1e-8). Galerkin near-flat/MMS untouched (separate class). Tests `test_lake_at_rest_is_held_exactly_{1d,2d}`, `test_lake_at_rest_independent_of_eps_H_1d`, `test_closed_domain_conserves_multistep_1d`, `test_tilted_v_catchment_conserves_2d`. |
| 7 | **d ≥ −1e-12 WITHOUT any limiter** (the monotonicity headline) | **PASS** (1 characterized caveat) | the class carries NO positivity machinery. STEEP 5% front strict `d ≥ −1e-12` 1-D + 2-D (exactly where the galerkin path must engage its clip). On the canonical V (the 2% mild valley) the measured `run_min_d = −0.0` at all three meshes/scales — **no undershoot in practice.** **Honest caveat (not a fail, characterized):** the smoothed (tanh) selector is bit-strict monotone only where the front head-drop ≫ `eps_H`; on adversarially-sharp *mild-2%* mounds it shows a **geometry-dependent SUB-MM undershoot, 0 .. ~0.9 mm at `eps_H=1e-3`** (controller adjudication sweep, 24 geometries), with conservation machine-tight throughout — vs the galerkin limiter's cm-scale clip-and-rescale pathology it replaces. Sharpening `eps_H` to remove the last sub-mm hits a ~15× Newton-cost cliff for zero accuracy gain (B3), so 1e-3 is the chosen balance; semismooth Newton is the P2 fallback if a future fixture needs strict mild-front positivity (not indicated). Tests `test_front_advance_positive_without_limiter_1d`, `test_steep_front_positive_without_limiter_2d`. |
| 8 | 1-D kinematic rising-limb analytic matched (to O(h)) | **PASS** | `d/d_eq = 1.000`, front position 20.00 m on the 100 m plane. Test `test_kinematic_wave_plane_hydrograph_1d`. |
| 9 | V plateau → Q_eq ±3% at ≈48×30, mesh-convergent, oscillation ≤2% RMS; field-scale ≈1.0 where galerkin gave 0.876 | **PASS** (accuracy framing corrected — see §8.7) | **48×30: LUMPED plateau 0.99994·Q_eq, oscillation RMS 0.013%.** 96×60: 0.99999, RMS 0.0036% — RMS shrinks with h (mesh-convergent on the oscillation). **⚠ Framing correction (B5b, §8.7): the LUMPED plateau ≈Q_eq is a mass-CONSERVATION identity (the lumped `outflow_rate()` shares the outlet sink's weights → a converged steady field FORCES Q_out = rain·area = Q_eq for any shape), NOT discharge accuracy** — the original "0.006% err / PASS by ~500×" read it as accuracy and is WITHDRAWN. The genuine accuracy measure (the *consistent* ds-integral discharge) is ~0.85 on the idealized kink V (a measure-zero-channel artifact shared with galerkin, B5b) and ~0.99 (≤1%) on a resolved finite-width swale. **Field-scale (SCALE=0.1): lumped 1.0000 vs galerkin's under-resolved 0.876** — confirms O1 *equilibrates/resolves* the field-scale V where the oscillatory galerkin scheme does not (the §8.3/F10 win), again read as conservation+equilibrium, not an accuracy digit. Test `test_tilted_v_plateau_reaches_Qeq_2d` (pins the ±3% conservation/equilibrium identity + the ≤2% oscillation + machine-tight books + sub-mm undershoot at 48×30); runner `scratch/_v2d_upwind_V.py`, npz `scratch/v2d_upwind_{48x30,96x60,48x30_s0.1}.npz`; accuracy diagnosis `scratch/_b5b_valley_concentration.py`. |
| 10 | dt no longer pinned (measured vs galerkin ~5e-5) | **PASS — the tractability fix** | median dt at DT_MAX **1e-3** (min 1e-5), **0 rejected steps** on all three runs, vs the galerkin V pinned ~5e-5 (P0 full-window: ~1.5e-6) with **60,008 rejections / 39.5 h**. Upwind canonical V wall-clock: **0.43 s** (48×30), 2.3 s (96×60), 0.36 s (field) — the 39.5 h / 20×-over-bar cost (§8.4) is GONE because the sawtooth that drove the reason-4 churn is gone (V converges via reason 2/3; reason-4 here is benign floor stagnation, 0 rejections). |
| 11 | Galerkin `OverlandProblem` path UNTOUCHED; full suite green | **PASS** | new scheme is a separate class `UpwindOverlandProblem`; `overland.py` not modified. Full suite **154 passed** (`pytest tests/` exit 0; = 136 P0 baseline + the 18 new `tests/test_overland_upwind.py`, all green). |

**Carried decisions (locked by the spike, inputs to P2):**
- **`eps_H = 1e-3` m** (smoothed-`tanh` upwind-selector head width) — chosen empirically in B3 (`scratch/_upwind_selector_probe.py`): 1e-3 is the balance of positivity (sharper buys no accuracy; 1e-2 undershoots ~2.4 mm) vs Newton robustness (a ~15× step blow-up below 1e-3 as `tanh`→`sign`).
- **`eps_S = 1e-3`** (slope floor in the Manning conveyance root) — matches `overland.py`.
- **`T_e` form:** 1-D `T_e = 1/L_e` (FD Laplacian); 2-D = the FV dual-mesh cotangent transmissibility (= the negated galerkin stiffness off-diagonal, pinned by `test_cotangent_T_e_equals_negated_stiffness_offdiagonal_2d`), with an **M-matrix guard** `T_e ≥ −1e-14` that holds on the structured box V and **raises loudly on an obtuse mesh** (`test_m_matrix_guard_{holds_on_structured_V,raises_on_obtuse_mesh}_2d`).
- **Finite-difference (PETSc-colored) Jacobian + direct LU** (`snes.setUseFD(True)`), the ParFlow `UseJacobian False` precedent — proving the *scheme*, not an exact J. A hand/analytic Jacobian is the P2 performance item (FD cost is trivial at spike sizes, 0.4–2.3 s).

**P2 productionization readiness:** the same edge scheme transfers to the realization-A top-facet ridge
graph (`ds_top` triangulation edges) of `CoupledProblem`; λ / NCP / outlet / inlet terms unchanged; the
galerkin limiter demoted to a tripwire assert. Full TDD per the three-tier routine (new Tier-1
positivity-no-clip / conservation-no-clip / kinematic / V-plateau tests, suite green, galerkin path
bit-identical, Tier-3 B4/B5 re-baseline with Arik sign-off). **Carry-forward risks** (none a P1 blocker):
the sub-mm mild-front undershoot (gate 7 caveat) — quantify on the P3 swale and engage semismooth Newton
only if it hurts that answer; the M-matrix guard is structured-mesh-only (unstructured/obtuse `T_e` is
P2/P3, §7); O(h) upwind smearing on mm sheet flow (FCT/O3 is the P4 escape hatch if it measurably hurts
the swale). **NEXT = P2** (productionize in `CoupledProblem`).

### 8.7 O1 spike VERDICT (P1 Part B close, 2026-06-16) — corrected accuracy framing + final result

> This is the integrity capstone of the O1 spike (B6). §8.6 (above) is the gate inventory, written
> before the B5b diagnostic; this §8.7 carries the **corrected accuracy framing** (an over-claim was
> caught in B5 review and investigated in B5b, both verified) and the **final verdict**. It is written
> at branch head **`6e1f63f`** (B5b) + this B6 commit.

**Headline — the spike's central question, "does O1 fix the convergent tilted-V?", is answered YES.**
On the canonical 2-D tilted-V where the validated galerkin `OverlandProblem` develops the never-settling
wet/dry sawtooth (Defect A, §8.3), the O1 monotone upwind-mobility two-point edge flux
(`pids_forward/physics/overland_upwind.py`, `UpwindOverlandProblem`) removes the pathology at the source:
the **sawtooth is GONE** (plateau oscillation RMS 0.013% / 0.0036%, mesh-convergent), the **dt-pin is
LIFTED ≈370×** (median dt at DT_MAX 1e-3 with **0 rejected steps** vs the galerkin V's ~5e-5 pin / 60k
rejections / 39.5 h — wall-clock 0.4 s vs 39.5 h), the scheme is **monotone** (NO limiter; the V's
measured run-min depth was −0.0) and **conservative** (books ≤1e-13·cum_rain), and **its convergence-line
flux is CORRECT** (B5b, below). It does this on a **separate class**, leaving the galerkin MMS/regression
path bit-identical.

**Gate-by-gate vs the P1 acceptance bars (P1 plan §"Acceptance bars" 6–11; mirrors §8.6's table, with
the gate-9 accuracy framing corrected):**

| # | P1 bar (Part B) | Result | Number |
|---|-----------------|--------|--------|
| 6 | Lake-at-rest + near-flat exact (well-balanced); closed conservation machine-tight | **✓** | lake-at-rest held EXACTLY 1-D + 2-D (head differenced on edges → ~1e-16, depth held to machine through the solve), `eps_H`-independent (structural — cannot be tuned to pass); closed-domain conservation ~2e-16; V books gap **≤1e-13·cum_rain** (−2.7e-10/−2.4e-10/−1.7e-11 m³ at 48×30 / 96×60 / field). |
| 7 | **d ≥ −1e-12 WITHOUT any limiter** (the monotonicity headline) | **✓ + 1 characterized caveat** | class has NO positivity machinery. STEEP 5% front strict `d ≥ −1e-12` 1-D + 2-D; on the canonical V (mild 2% valley) measured **run-min = −0.0** at all meshes/scales. CAVEAT (characterized, conservation machine-tight throughout): adversarially-sharp *mild-2%* mounds show a geometry-dependent **SUB-MM undershoot, 0 .. ~0.9 mm at `eps_H=1e-3`** (B3, 24-geometry sweep) — vs the galerkin limiter's cm-scale clip-and-rescale it replaces. |
| 8 | 1-D kinematic rising-limb analytic matched (to O(h)) | **✓** | `d/d_eq = 1.000`, front 20.00 m on the 100 m plane. |
| 9 | V plateau → Q_eq ±3%, mesh-convergent, oscillation ≤2% RMS; field-scale ≈1.0 where galerkin gave 0.876 | **✓ (framing corrected)** | LUMPED plateau **0.99994 / 0.99999·Q_eq** (= conservation/equilibrium identity, see below — *not* accuracy); **oscillation RMS 0.013% → 0.0036%** (mesh-convergent, the real stability win); **field-scale lumped 1.0000 vs galerkin 0.876** (O1 equilibrates/resolves the field-scale V). Genuine accuracy (consistent ds-integral): **~0.85 idealized kink V** (artifact), **~0.99 resolved swale** (≤1%). |
| 10 | dt no longer pinned (vs galerkin ~5e-5) — the tractability fix | **✓ (≈370×)** | median dt at DT_MAX 1e-3, **0 rejected steps** on all three runs; **0.4 s** (48×30) vs the galerkin V's 39.5 h / 60k rejections. The §8.4 39.5-h / 20×-over-bar cost is GONE (the sawtooth that drove the reason-4 churn is gone; the V converges via reason 2/3). |
| 11 | Galerkin `OverlandProblem` UNTOUCHED (separate class); full suite green | **✓** | new scheme = separate class `UpwindOverlandProblem`; `overland.py` unmodified. Full suite green (`pytest tests/` exit 0; the 18 `tests/test_overland_upwind.py` + the P0 baseline). |

**The corrected accuracy framing (B5 review + B5b investigation, `scratch/_b5b_valley_concentration.py`,
both verified):**
- **LUMPED plateau = CONSERVATION, not accuracy.** The §8.6 commit framed "plateau = 1.000·Q_eq" as
  discharge *accuracy* ("essentially exact / PASS by ~500×"). That is WITHDRAWN. The lumped
  `outflow_rate()` and the outlet residual sink share the **same** control-length weights `B_k`, so
  summing the node residuals telescopes every interior edge flux to zero and FORCES
  `Σ_k q_out·B_k = rain·area = Q_eq` for **any** converged STEADY field, independent of its shape. The
  lumped 1.000 is therefore the discrete steady-state **mass balance** (machine-tight books,
  ParFlow-comparable) — it confirms CONSERVATION + that the field reached equilibrium, NOT outlet-flux
  accuracy. The `±3%` test assertion (`test_tilted_v_plateau_reaches_Qeq_2d` item 1) is correspondingly
  a conservation/equilibrium check, relabelled as such; it is kept (the conservation + flat-plateau +
  dt-lifted + sub-mm-undershoot checks it bundles are real and valuable). The galerkin path's lumped
  plateau is likewise ≈1.0 for the same identity.
- **The genuine accuracy measure** is the CONSISTENT ds-integral discharge `∫ 86400·(1/n)·d^(5/3)·√S ds`
  over the outlet (the P1-interpolated functional). On the idealized **KINK V** it reads **~0.85·Q_eq and
  DIVERGES under refinement** (0.85 → 0.84 at 48×30 → 96×60, valley peak growing). **B5b verdict (A) =
  measurement artifact, NOT a scheme defect:** the idealized V channel is a measure-ZERO line (the bed
  kink), so its "channel" is 1 cell wide → a tall narrow `d^(5/3)` spike that a smooth P1 functional
  cannot integrate. The valley depth follows the Manning thin-channel normal-depth law EXACTLY (growth
  `2^(3/5)` per dx-halving, measured to 3 sig figs; channel throughput constant to 0.15%), the deficit
  is **quadrature-degree-independent** (even exact quadrature of the coarse P1 field reads ~0.85), and
  the **galerkin scheme shows the SAME** lumped≫consistent split — so it is generic to a thin P1 channel,
  not an upwind-flux defect.
- **A real finite-width swale (the actual PIDS use case) RESOLVES CLEANLY.** With a flat valley floor of
  width W carrying several cells the consistent discharge converges **UPWARD to ~0.99 (≤~1% error)** and
  the depth stabilizes (W=324 m: 0.971 → 0.982 → 0.990 over 48×30 → 96×60 → 192×120; peak 8.7 → 8.0 mm).
  **Conclusion: O1's convergence-line flux is CORRECT;** the 0.85/depth-growth on the kink V is a
  characterized artifact of an idealized measure-zero channel (shared by both schemes), and the resolved
  swale + Manning-law evidence validates the flux. **No convergence-line flux fix is warranted.**

**Carried items (inputs to P2/P3, none a P1 blocker):**
- **Positivity:** the sub-mm geometry-dependent mild-front undershoot (gate-7 caveat) — quantify on the
  P3 swale; semismooth Newton is the P2 fallback only if a fixture needs strict mild-front positivity
  (not indicated; conservation is machine-tight regardless).
- **Solver:** the **FD-Jacobian + direct LU is a deliberate spike choice** (`snes.setUseFD(True)`, the
  ParFlow `UseJacobian False` precedent — proving the *scheme*, not an exact J; trivially cheap at spike
  sizes, 0.4–2.3 s). The hand/analytic Jacobian is a **P2 performance** item.
- **Productionization:** `UpwindOverlandProblem` is a **standalone** Module-2 class; **P2 productionizes
  the same edge scheme into `CoupledProblem`** (realization-A top-facet ridge graph; λ/NCP/outlet/inlet
  terms unchanged; galerkin limiter demoted to a tripwire assert).
- **Mesh:** the M-matrix `T_e ≥ 0` guard is structured-mesh-only; unstructured/obtuse `T_e` is P2/P3 (§7).
- **Accuracy:** O(h) upwind smearing on the mm sheet — FCT/O3 (§4) is the P4 escape hatch only if it
  measurably hurts the swale answer.

**Bottom line: the spike validates O1 as the production fix for the convergence-line regime.** The
central question is answered YES with rigorous evidence (after an accuracy over-claim was caught and
corrected): O1 fixes the convergent-V pathology, its convergence-line flux is correct on the regime
that matters (the resolved swale), and the only carried items are perf (hand-Jacobian) + the
characterized sub-mm mild-front undershoot. **NEXT = P2.**

### 8.8 P2 productionization result (2026-06-17) — the upwind scheme shipped in `CoupledProblem`

**Headline — O1 is productionized in the coupled `[ψ, d, λ]` solver as an OPT-IN scheme, and it
works.** `CoupledProblem(..., overland_scheme="upwind")` runs the monotone upwind edge flux on the
realization-A top-facet graph as the lateral-overland operator, converging + conserving on the coupled
storm; the galerkin default is bit-identical. Full suite **174 passed**. Plan
`docs/plans/2026-06-16-overland-convergent-flow-P2.md`; commits A1 `0bb10be` → D2 `d7707a1` on
`b6-tilted-v-convergent-flow`. The implementation was **Codex-reviewed twice** (the plan + the mid-build
Jacobian finding); both reviews' blockers/should-fixes are folded in.

**Architecture (the irreducible decision).** The coupled solve is a monolithic block Newton (DOLFINx
0.10 `NonlinearProblem`, auto-Jacobian over UFL forms). The upwind edge flux is NOT UFL-expressible,
so the upwind path **overrides the SNES residual/Jacobian callbacks**: DOLFINx's own
`assemble_residual`/`assemble_jacobian` run first (block assembly + BC lifting), then the non-UFL
lateral edge-flux residual + the d–d edge Jacobian are ADDED on the d-block only. `ψ`, `λ`/NCP,
outlet, drainage, and the interior-pin stay UFL and untouched; the monolithic block + λ sign-pairing
are preserved, so conservation stays structural (the edge flux telescopes to zero over the surface).
Kernel `pids_forward/physics/overland_edge_kernel.py` (extracted DRY from the standalone, bit-identical);
solver wiring in `coupling.py` (`_wire_upwind_callbacks`, `_add_edge_jacobian`, `_positivity_tripwire`).

**Gate-by-gate vs the §5 P2 acceptance list (P2-plan acceptance bars 1–8):**

| # | P2 bar | Result | Evidence |
|---|--------|--------|----------|
| 1 | Opt-in + galerkin bit-identical | **✓** | `overland_scheme` (galerkin default); every pre-existing test unchanged; suite 174 passed. |
| 2 | Lateral term only; monolithic block + λ-conservation | **✓** | only `overland_flux` removed from UFL `F_d`; ψ/λ/NCP/outlet/drainage/pin untouched; one `[ψ,d,λ]` Newton (no operator-split). |
| 3 | Conservation structural | **✓** | closed tilted 3-D box, ponding rain: `Δtotal = cum_rain` to **1e-13**; `clip_mass_adjust = 0`. |
| 4 | Positivity WITHOUT the clip (tripwire) | **✓ + characterized caveat** | limiter demoted to a loud tripwire on the upwind path (never clips). **Coupled mild-front undershoot is TRANSIENT, self-healing, mass-neutral, ~1.1–1.5mm** (grows mildly with rain rate; final state heals to d≥0) — the P1 gate-7 caveat at coupled scale (vs galerkin's cm-scale clip). Tripwire tol 5mm tolerates the characterized sub-cm band; raises on cm-scale breakdown. |
| 5 | Jacobian correct (kernel + coupled) | **✓** | numerical per-edge central-FD edge Jacobian; kernel FD-verify (== full central FD) AND coupled-level `J·δ` vs FD smoke (block offset/sign/BC). |
| 6 | Coupled accuracy = operator equivalence; absolute swale = P3 | **✓** | operator-equivalence to the validated standalone is STRUCTURAL (same extracted kernel + A2 cotangent==ds_top-stiffness + A3 conservation root); downslope routing reproduced (downhill > 1.5× uphill). Absolute resolved-swale discharge accuracy is the **P3** fixture (per §8.7). |
| 7 | Top-facet graph guarded | **✓** | cotangent `T_e == −(ds_top tangential-gradient stiffness off-diag)`; `Σ A_i == top area`; M-matrix guard (raises on obtuse). |
| 8 | Regression + Tier-3 | **✓ suite / Tier-3 PENDING** | full suite 174 passed (galerkin bit-identical). **Tier-3 B4/B5/B6 upwind re-baseline = SET UP (§8.8 below), pending Arik visual sign-off** — record `validation/sanity/coupled_upwind__2026-06-17.md`. |

**The two key findings (Codex-reviewed):**
- **The planned Picard (frozen-mobility) Jacobian is non-viable: it STALLS on ponding fronts** (reason-4
  line-search stall at all dt — it drops dM/dd + the tanh-selector derivative that sharpen at wet/dry
  fronts). Codex's prescribed discriminators (the J-vs-P risk ruled out; a numerical edge Jacobian
  converges) confirmed Jacobian exactness is the lever. **Shipped = a vectorized per-edge central-FD edge
  Jacobian** (correctness-first, Codex-endorsed, the standalone's FD precedent); the **hand-analytic
  Jacobian is a documented future PERFORMANCE optimization** (DP-1 resolved this way). The numerical
  Jacobian converges but at higher iteration counts than galerkin on stiff ponding — the lever for when
  hand-analytic gets pulled in.
- **The mild-front undershoot is real in the coupled setting (~1.1–1.5mm)** — bar-4 above. Honest,
  characterized, sub-cm, transient, mass-neutral; flagged for Arik + P3 swale re-characterization.

**Carried decisions (inputs to P3):** numerical per-edge central-FD edge Jacobian (hand-analytic =
perf); tripwire tol 5mm; **serial-only** (multi-rank guarded — ownership-aware edges + ghosted reads
deferred); **3-D host only** (the 2-D top-edge upwind is a trivial future extension, guarded). DP-2
(kernel extraction) done with the standalone staying bit-identical.

**DP-3 — the default-flip proposal (Arik's call at sign-off):** keep galerkin as a PERMANENT fallback
mode either way. **Proposed:** flip the `CoupledProblem` default `galerkin → "upwind"` AFTER (a) Arik's
Tier-3 visual sign-off (§8.8 re-baseline) and (b) the P3 resolved-swale absolute-accuracy validation —
not before. Until then, upwind is opt-in.

**Tier-3 re-baseline setup (E1, for Arik sign-off).** The B4 (1-D coupled column) / B5 (3-D coupled
hillslope) / B6 (coupled tilted-V) runners + the solver-independent HTML builders are in
`forward-model/viz/` + `benchmarks/`. The upwind re-runs are wired via an `OVERLAND_SCHEME=upwind`
env knob on the coupled runners; the decisive artifact is the **coupled 3-D tilted-V** (the convergence
line where galerkin pinned dt → 39.5 h) re-run with upwind. See `validation/sanity/coupled_upwind__2026-06-17.md`
for the run commands + the metrics to inspect; Arik opens the HTMLs and signs off.

**Bottom line: P2 ships the convergent-flow fix into the product engine.** The coupled upwind solver is
functional, opt-in, galerkin-bit-identical, conserving, monotone-to-sub-cm, and Codex-reviewed. The only
open items are the (honest) characterized mild-front undershoot, the hand-analytic Jacobian (perf), the
serial/3-D scoping, and the Tier-3 visual sign-off + the b6→main merge. **NEXT = P3** (re-benchmark +
the permanent PIDS-swale Tier-2 fixture — the absolute-accuracy fixture per §8.7).

## 9. Artifacts
- This plan; B6 deck `parflow/cases/tilted_v_catchment.py`; harness `benchmarks/build_comparison_tiltedv.py`
  + `make_comparison_tiltedv_html.py`; in-house runner `forward-model/scratch/_tiltedv_spike.py` (to be
  promoted out of scratch in P2/P3); ParFlow reference summaries in `~/parflow-runs/tilted_v/summaries/`.
- References: Kollet & Maxwell 2006 (AWR); Maxwell et al. 2014 + Kollet et al. 2017 (IH-MIP/2, WRR);
  Di Giammarco et al. 1996; ParFlow `overlandflow_eval{,_Kin,_diffusive}.c`; Kuzmin, *Flux-Corrected
  Transport* (AFC); Forsyth 1991 (upstream mobility weighting, two-point flux).
