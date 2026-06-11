# Overland convergent-flow stabilization — research plan

**Date:** 2026-06-11 · **Status:** PLAN (no code yet) · **Trigger:** benchmark B6 (canonical tilted-V catchment)
**Priority (Arik 2026-06-11):** PIDS networks are installed along **lines of topographic convergence** (swales,
valley lines, drainage ways). Convergent overland flow is therefore the **core operating regime** of the
product — the M4 embedded features and `add_surface_inlet` grates sit exactly where surface water
concentrates. The B6 failure modes are a core-fix engineering target, **not** an out-of-envelope curiosity.

## 1. What B6 measured (the trigger, all numbers reproduced this week)

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

### Defect B — lax step acceptance can book non-physical states (NEW finding, UNCONFIRMED — P0 confirms)
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

## 8. Artifacts
- This plan; B6 deck `parflow/cases/tilted_v_catchment.py`; harness `benchmarks/build_comparison_tiltedv.py`
  + `make_comparison_tiltedv_html.py`; in-house runner `forward-model/scratch/_tiltedv_spike.py` (to be
  promoted out of scratch in P2/P3); ParFlow reference summaries in `~/parflow-runs/tilted_v/summaries/`.
- References: Kollet & Maxwell 2006 (AWR); Maxwell et al. 2014 + Kollet et al. 2017 (IH-MIP/2, WRR);
  Di Giammarco et al. 1996; ParFlow `overlandflow_eval{,_Kin,_diffusive}.c`; Kuzmin, *Flux-Corrected
  Transport* (AFC); Forsyth 1991 (upstream mobility weighting, two-point flux).
