# Codex review — option-A adaptive q_pot infiltration cap (2026-06-26)

Independent review by Codex (gpt-5-codex, effort=high) of the `film_mode="qpot"` prototype
(`CoCycledCappedSplit`, `scratch/seq_href_iterated.py`), commissioned per Arik. Prompt:
`scratch/_codex_review_prompt.txt`. Read alongside §14–§15 of
`overland_partition_bug_investigation__2026-06-24.md`.

## Empirical results that accompany the review (qpot, h_sat=2mm, uniform mesh, K=6)

| soil | Ks | monolith target | qpot cap routed/R | gap | stable | bal/rain |
|---|---|---|---|---|---|---|
| loam | 0.25 | 0.5470 | 0.8744 | **+32.7 pp** | ✅ | 1.5e-11 |
| sand | 1.5 | 0.6939 | 0.6719 | **−2.2 pp** ✅ | ✅ (max_pond 0.25mm) | 1.0e-13 |
| silt | 0.10 | 0.7674 | 0.9192 | **+15.2 pp** | ✅ | 1.05e-11 |

The cap OVER-ROUTES (under-infiltrates) loam/silt and NAILS sand — the opposite error from `w`
(which over-infiltrated). `w` and the cap BRACKET the monolith on every soil. **The cap FIXES sand
stability** (the high-K case `w` collapsed on). But a single `h_sat` does not generalize (errors
scatter +33 / −2 / +15). Exactly Codex's prediction below.

## Codex verdict (verbatim)

> Conservation and the anti-force-feed structure look sound. I would not yet trust `film_mode="qpot"`
> as a soil-general replacement for the monolith, because the new cap is only a surrogate for the
> monolith's `q_pot`: it is substep-adaptive, but not co-solved; it uses a fixed wet-end `qpot_h_sat`
> instead of the actual surface state `d`; and it reintroduces an `ell_c` mesh knob.

**1. Conservation — SOUND, no leak path.** `_route` conserves routed sheet + outflow; `held = d_routed −
film` so `film+held = d_routed` exactly; the parent residual carries pond storage + the conservative
lateral source on the same vertex measure; rollback books `cum_*` only after all substeps succeed.
Remains conservative even if ponding occurs (rejected water stays in ψ, counted in `surface_water()` +
next substep's `d_full`).

**2. Adaptive vs frozen — genuinely adaptive ACROSS substeps** (`film_prev_psi` read inside the K-loop
after prior solves updated ψ), **but frozen WITHIN each substep solve** (q_pot computed once from
pre-solve ψ), unlike the monolith where `q_pot=kirchhoff(ψ,d)/ell_c` lives inside the nonlinear residual
and re-evaluates on the current Newton iterate. Fixes §9's "frozen for the whole step", NOT the deeper
"fully co-solved" gap.

**3. Force-feed — STRUCTURALLY avoided.** The dead §9/V2 prototypes used a prescribed Neumann influx
`_q_inf`; qpot mode offers water through the pond-in-ψ source and lets Richards draw from the
self-limiting film (infiltration ≤ offered film ≤ q_pot·hsub). The "deliver more than the soil accepts"
pathology is gone. ⚠ But `max_pond_seen=0` is sampled at substep ENTRY (pre-solve) → terminal-substep
ponding is invisible; "no ponding in smoke" is suggestive, not conclusive. (sand's run DID show
max_pond=0.25mm — minor rejection, conservation held at 1e-13.)

**4. Faithfulness — NOT the monolith's law.** Monolith: `kirchhoff(ψ, d)/ell_c` (actual ψ + actual
surface depth `d`) inside the NCP, caps the actual flux λ. Prototype: `kirchhoff(min(ψ,0), h_sat)/ell_c`,
caps the substep-integrated OFFERED depth. Deviations: (a) `d` replaced by fixed `h_sat`; (b) positive
ψ clamped to 0; (c) caps offered depth not instantaneous flux. Clamp-to-0 is a reasonable "saturated
interface" approx only while ponding is negligible.

**5. Hidden knobs — REAL, not plumbing.** `qpot_h_sat`: in the saturated regime adds ≈ `Ks·δ/ell_c` to
q_pot, so its sensitivity SCALES WITH Ks → **sand most sensitive**. `qpot_ell_c`: q_pot ∝ 1/ell_c — the
same mesh-sensitive quantity that made the monolith's cap vary with vertical discretization (§13);
reintroduces mesh dependence. The prototype's ell_c autodetect is weaker than the monolith's (no MPI MIN,
no fail-loud, silent 0.0625 fallback).

**6. Bugs / edge cases — units consistent** (kirchhoff m²/day ÷ ell_c → m/day × hsub → m); `film≤d_routed`
& `held≥0` hold by construction (signed-borrow not exercised); ψ_n carry + rollback correct; per-node
Python kirchhoff loop = perf only. **Code-quality:** `max_pond_seen`/`min_held_seen` not rolled back on
failed retries + `max_pond_seen` misses terminal-substep ponding.

**7. Generalization — better direction than `w`, but structural doubt remains** (not the monolith law,
ell_c-dependent, substep-adaptive not iterate-adaptive). Prediction: "loam may land; silt improves
materially over w; sand riskiest (high Ks → largest h_sat + ell_c sensitivity)." *(Empirically: loam
MISSED big +33, silt improved-but-off +15, sand NAILED −2 + stable — the scatter confirms the
no-single-knob diagnosis; the loam miss was unpredicted in sign.)*

**Prioritized recommendations:**
1. Don't call this "the monolith's q_pot" — it's a conservative, soil-aware SURROGATE.
2. Align `qpot_ell_c` with the monolith exactly (same autodetect, MPI reduction, fail-loud).
3. Record end-of-substep / end-of-run ponding, not only pre-solve.
4. **If the sweep misses sand/silt, the next fix is NOT another global knob — make the cap more
   monolith-faithful, especially the wet-end state `d`, and if needed a small inner update of q_pot
   within each substep instead of a frozen pre-solve value.**

## Synthesis (mine)

Codex + the empirics converge: **option A is structurally sound (exact + no force-feed) and FIXES sand
stability, but its fixed-`h_sat`/`ell_c` surrogate does NOT replicate the monolith's co-solved acceptance
→ scattered per-soil accuracy (+33/−2/+15), no single knob.** The root is the §9 circular closure: the
RIGHT kirchhoff upper limit is the THIN co-solved infiltrating sheet, which the sequential scheme lacks;
a fixed `h_sat` is too thin (over-route) and soil/mesh-dependent. `w` (over-infiltrate) and the cap
(over-route) bracket the truth → a properly co-solved acceptance should land between.

**Fork (Arik's call):** (A′) refine the cap toward monolith-faithfulness per Codex rec 4 — use the actual
(thin) surface depth `d` in the kirchhoff + a small inner q_pot update — which is a step toward
re-deriving the monolith's surface co-solve in the sequential frame; or (C) harden the soil-accurate
monolith's stiff-case dt-collapse directly (it already HAS the co-solved closure). The cap's sand-
stability win + the bracketing suggest a faithful cap is achievable, but it converges toward "a mini
co-solve at the surface," narrowing the gap to option C.
