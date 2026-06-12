# Adversarial Review — M4 Phase-4 "rate-clock + WI v2" disperse coupled-embedding evidence — 2026-06-11

- **Protocol:** the mandatory Phase-4 review gate (multi-agent adversarial + Codex, per the working
  protocol that retracted the Phase-3 coupled claims twice). Five independent attack agents + one
  Codex external pass, each instructed to REFUTE; attack templates (a)–(f) pre-listed in the
  kickoff note before the review ran.
- **Evidence under review:** branch `m4-phase4-coupled-embedding` (2a6dd5a … 74d89b7), production
  `pids_forward/physics/wi_exchange.py`, prototype `scratch/m4_phase4_wi_probe.py`, gate
  `tests/test_coupled_gate_refs.py` + fixtures, plan
  `docs/plans/2026-06-10-m4-phase4-coupled-embedding-plan.md`.
- **Bottom line:** the GATE PASS IS REAL and pre-registration integrity is INTACT, but the claim
  "genuinely host-controlled coupling" is supported ONLY for the WI era, the ~5 % residual
  attribution to the offline-F large-ζ bias is REFUTED by measurement, and within the shipped
  disperse-only scope the capacity-clamped passive is killed by NO leg. Wording constraints below
  are binding; fixes applied in the review-fixes commit on this branch.

## Verdicts by attack

**(a) "The moving front-ring read is a shell-in-disguise" — UPHELD-WITH-QUALIFICATIONS
(wording-only).** The ring read is structurally the retracted shell read with a moving radius, and
it is causally near-decorative during the clock era: the Kirchhoff scale it feeds is ≈1 for any
plausible ring state in the committed scenario family (gross pre-wetting to ψ=−0.6 moves the rate
1.9 %; the Ref-B pulse band's own max ψ=−0.387 moves it 5.5 % — and the pulse never reaches the
clock era). Replacing the read with the constant ψ_i would reproduce every committed leg within
~1–2 %, far under tolerance: NO committed experiment discriminates the moving ring from a constant
drive. **The RefB-40 pulse (t=[7.0,10.5] d) lands AFTER handover at both n run (n=12: handover
2.5 d, margin 4.5 d; n=8: handover 6.78 d, margin 0.22 d — marginal), so RefB-40 validates WI-era
host control, not the ring.** What is genuinely different from the retracted shell: water goes
host-ward immediately (ledgered), and the WI era is a different coupling pathway entirely
(ψ_Γ live in the Newton solve). Host control is real — in the WI era.

**(b) "The rate-prescribed era is the retracted sin re-badged" — ATTACK FAILS on substance
(wording-only qualifications).** Measured era split (fixtures, offline reproduction exact: clock
26.6 %/34.1 %/1.01 % on RefA-40/RefB-40/control): the clock era is 8.7–34.8 % of cumulative I
(n=12…6) but only **0.4–7.5 % of the relL2 denominator and ~0 % of the clock's failure mass** —
100 % of the raw clock's failure accumulates after handover (>95 % at the final plateau sample).
A hybrid (raw clock in-era, exact after) scores 0.05–0.30 %: the clock era can neither pass nor
fail a leg. The pass is earned by the WI era, which carries every discriminating feature (bend,
plateau, pulse). In-era the clock matches the resolved refs to 0.77–1.11 % relL2 — prescribing the
rate there is correct physics, and the deposition footprint is the feature ridge itself (the host
operator distributes it; not the retracted fixed-shell + hidden-accumulator pattern; ledger-exact).

**(c) The non-degradation judgment call — ADJUDICATED: warning sign, not refutation
(wording-only).** The n=6 point (2.2 %) is clock-dominated (34.8 % of I; ~49 % of the control
window) and the clock is simply right there; the 2.2→5.1 jump measures the WI era's intrinsic
systematic appearing, not mesh degradation. Flat 5.1→5.7 % over the WI-dominated n=8..12 is the
honest non-degradation reading, the trend is capped by construction (n≥14 = the r_0=r_w degeneracy,
refused), and further n-refinement on this box is impossible — a larger-R box is the only way to
extend the sweep. BOTH readings must ship together wherever the sweep is cited. The control-leg
trend (2.1→6.0 %, n=8→12) is the same WI-era early/mid systematic being unmasked as the clock-era
share shrinks (49→12 % of the truncated window); its sign is over-DELIVERY — the opposite of the
over-throttling the control was registered to catch — but "no over-correction" is over-compressed
wording; the error sign must be recorded explicitly in the next run.

**(d) Two handover rules — NOT a fitted knob; disperse claim NOT contaminated (wording-only).**
front=2h derives from the Task-5 far-field probe (log fidelity <1.1 % beyond 2h), a pre-scheme
measurement recorded 2026-06-10 before any v2 run; it is NOT in the pre-registered plan (the plan's
explicit-reservoir handover was abandoned in the documented course-correction) and is validated as
a fidelity radius, not as a transient-handover optimum — say so. The drain rate-crossing rule
exists only on a leg that FAILED and is refused in production: a rule that never produced a pass
cannot have been fitted to pass.

**(e) Pre-registration integrity — INTACT.** EMBEDDED_TOL=0.10/BASELINE_KILL=0.20 byte-identical
from lock (954097a, 06-10 21:42) to HEAD; first scheme run 06-11 08:21. All 29 pre-registered
fixture arrays byte-stable; e442865 purely additive, and its R40 data is provably pre-scheme
(scratch npz mtime 06-10 21:55; the 05014ff commit message (08:01) quotes 26.6 %/7.471/4.3 %,
reproduced exactly from the committed fixture). Test-file changes after lock purely additive. The
`EMBEDDED_TOL + 0.02` at test_coupled_gate_refs.py:131 bounds a BASELINE's near-pass (design
documentation), not the scheme's bar (scheme bar = 0.10 exactly, test_wi_exchange.py). Production
port faithful (line-by-line: identical F closure, ring, handover, WI, ledger; three benign deltas).
Suite 18/18 via WSL. **Caveats:** RefB-40 is post-scheme supplementary evidence (generated 2 min
after 4d2ba31; parameters physically motivated and harder for passives, but weight it as
supplementary, not pre-registered); BASELINE_KILL=0.20 vs the plan's anticipated ≥0.30 (locked
pre-scheme, justified by worst measured 0.289 — but the separation factor is 2×, not 3×); the
dual-scale kill (13–39 %) is documented in commit messages/harness, never asserted in a committed
test (regression gap).

**(f) "~5 % residual = offline-F large-ζ bias; closure polish, not coupling failure" — REFUTED
(BLOCKING for the wording; the gate pass itself stands).** Measured offline (1-D radial open-domain
truth at 120 r_w, committed reference machinery, no FEM): F's actual extrapolation bias at
deployment ζ (70–800) is **−0.9…−1.5 % UNDER-prediction** (clock-vs-open relL2 1.45 % full-curve).
The v2 residual is **+3–5 % OVER** in the mid-curve and **−9 %** in the deep bend — wrong sign,
~4× wrong magnitude, and the wrong ERA (F acts only pre-handover at I/I_max ≤ 8.7–19.6 % for
n=8–12; the residual lives at 0.2–0.9). Carry-through via handover offset is bounded at ~−0.3 %.
"Capacity echo" is undefined anywhere in the repo. "Plateau exact" is vacuous (capacity-matched box
+ full-depletion window force it). Corroboration: the residual GROWS with the WI-era share (n=6
best at 2.2 %). **The residual is an UNATTRIBUTED WI-ERA COUPLING SYSTEMATIC** (candidates, all
untested: post-handover transient vs the steady-log r_0 assumption, coarse-cell ψ_Γ bias, deep-bend
bridge over-throttling). The planned "F large-ζ polish" roadmap item targets a component measured
at −1 % — dropped. Evidence: `scratch/_adv_f_attack_open120.npz` + `scratch/_adv_b_era_split.py`.

**(Codex external pass — claim as worded: REJECT; weaker wording supported, adopted below.)**
Additional production findings, all fixed in the review-fixes commit: `h_f_ref=0.0` hardcoded while
`configure_sorptive` accepts arbitrary `psi_wall` (now: the feature stores the configured wall head,
the exchange reads it and REFUSES non-zero — saturated-wall evidence only); the clock-era rate used
a magnitude-only Kirchhoff drop and would inject even with the ring overtopping the wall (now: rate
hard-zeros when ψ_ring ≥ wall head; new unit tests); the ρ construction is nearest-Γ-vertex
distance, not general perpendicular distance (comment honesty fix); dual-scale "13-16 %" in the
production docstring was inconsistent with the measured 13–39 % (fixed); harness docstring claimed
ledger tolerances 1e-8/1e-6 vs the asserted 1e-6/1e-4 (fixed, never loosened post-hoc).

## The scope finding (material to any "coupled" claim)

Within the shipped DISPERSE-ONLY scope, the **capacity-clamped offline clock passes every supported
leg** (RefA 0.6–2.7 %, RefA-40 4.3 %, RefB 8.5 %, RefB-40 9.8 %, control 1.0 %) — the clamp family
is killed ONLY by the drain legs, which production refuses. The gate genuinely discriminates v2
from the raw clock (27–34 %) and the retracted dual-scale (13–39 %), but the disperse evidence
alone does NOT rule out a passive capacity-aware alternative. (It also does not need to: the clamp
needs I_max = the closed-box capacity, an oracle quantity v2 never reads — but that argument is
design-level, not gate-measured. The honest statement is the scoped one.)

## The supported claim (binding wording)

> In the tested homogeneous-isotropic LOAM disperse deployment regime (R=40 r_w, positive-WI
> meshes n=6..12), the rate-clock + well-index scheme tracks independent closed-domain references
> at 2.2–5.7 % relL2 and tracks a host-history perturbation (landing on the pulse-shifted asymptote
> with no pulse knowledge), where the raw offline clock fails at 27–34 % and the retracted
> dual-scale at 13–39 %. **Host control is established for the WI era** (≥80–91 % of cumulative I
> at n=8–12, carrying all discriminating signal); the sub-grid era is a prescribed-rate closure
> whose host read is a second-order correction (≤~2 % here), present as mechanism but not exercised
> by any committed leg. The ~5 % residual is an unattributed WI-era systematic within the
> pre-registered tolerance. The evidence does not uniquely exclude passive capacity-aware schemes
> within the disperse-only scope (their killers are the drain legs, currently refused), and drain,
> resolved-wall, non-LOAM deployment, anisotropy, layering, and multi-feature regimes are out of
> scope.

MUST NOT claim (each was in pre-review wording, now fixed): "the clock is HOST-CONTROLLED through
its drive"; "a Ref-B pulse raises the ring and throttles the clock" (never fires in any committed
run); "closure polish, not coupling failure"; "no over-correction" unqualified; "first coupling to
PASS the discriminating gate" without the clamped-clock scope caveat; the n-sweep without both
degradation readings.

## Follow-ups (recorded, not blocking the disperse integration)

1. **The real residual diagnostic** (replaces the dropped F-polish item): one instrumented 3-D run
   logging the per-step WI-era exchange rate vs the resolved reference's instantaneous wall flux at
   matched I — localizes the +3–5 %/−9 % systematic to ψ_Γ read vs WI constant vs transient.
2. **Ref-C clock-era pulse** (pulse inside [0.5,1] d) + the scale≡1 ablation, pre-registered, if
   clock-era host control is ever to be claimed. Prediction on record: the ablation passes all
   current legs within ~1–2 % of v2.
3. Assert the dual-scale kill in a committed test (regression gap); record the control-leg error
   sign; SAND/SILT R40 disperse generality refs.
4. The retracted dual-scale shell machinery in `feature.py` (EXPERIMENTAL, measured failing
   13–39 %): **recommend excision** now that v2 supersedes it on every leg — deferred to Arik
   (working-style: confirm before large removals).

## Review provenance

Agents: attack-a (ring/shell, 18 tool uses), attack-b (clock re-badge + era split, evidence probe
committed), attack-c+d (killed by session limit; adjudicated from b/e/Codex evidence + the plan-doc
provenance check), attack-f (residual attribution, evidence npz committed), gate-integrity +
production-drift (fixture hashing, suite run), Codex external (numbered findings, REJECT-as-worded
verdict). All verdicts unanimous where they overlap.
