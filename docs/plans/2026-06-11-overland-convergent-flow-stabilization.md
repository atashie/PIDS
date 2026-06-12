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
  must book — review F5); full suite green.
- **O4 (deferred):** not implicated in any books violation (B2 dead; books close without it) — and the
  F9 control shows the limiter is LOAD-BEARING (bypass ⇒ Newton fails), so "replace the global rescale"
  is riskier than first framed; O1 removes the undershoots that make any limiter necessary.
- **P0 gate:** committed-spike re-runs against the fixed engine close the engine ledger with every
  violation an explicit rejected step (field-scale full window: −8.6e-12·cum_rain, 3 rejections, clean
  recession; canonical storm-window diagnostics: ≤2.1e-12 both controllers; canonical full-window
  spike-verbatim numbers in the regenerated B6 artifacts).
- **Merge note (review F6):** this branch forks `df61ff4` and does NOT contain main's `8a891c2`
  (per-sink drain/inlet accounting inside `step()`); the merge MUST land those per-sink increments
  INSIDE the hardened acceptance gate or O5 silently regresses for sinks.

### 8.5 Consequences for the plan
P1 (O1 upwind-mobility edge flux) is unchanged and **upgraded in priority rationale**: it is the
accuracy fix for the PIDS field scale (§8.3), the dt-pin fix (limiter goes no-op), and the monotonicity
fix. P0's conservation hotfix turned out to be mostly *verification* — the committed engine's books were
already structurally sound; what P0 actually bought is (a) the acceptance hardening that closes the
latent B1 path, (b) the measured mechanism that re-aims P1, and (c) the corrected public record.

## 9. Artifacts
- This plan; B6 deck `parflow/cases/tilted_v_catchment.py`; harness `benchmarks/build_comparison_tiltedv.py`
  + `make_comparison_tiltedv_html.py`; in-house runner `forward-model/scratch/_tiltedv_spike.py` (to be
  promoted out of scratch in P2/P3); ParFlow reference summaries in `~/parflow-runs/tilted_v/summaries/`.
- References: Kollet & Maxwell 2006 (AWR); Maxwell et al. 2014 + Kollet et al. 2017 (IH-MIP/2, WRR);
  Di Giammarco et al. 1996; ParFlow `overlandflow_eval{,_Kin,_diffusive}.c`; Kuzmin, *Flux-Corrected
  Transport* (AFC); Forsyth 1991 (upstream mobility weighting, two-point flux).
