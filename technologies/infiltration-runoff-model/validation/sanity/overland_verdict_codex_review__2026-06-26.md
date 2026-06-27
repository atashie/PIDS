# Codex review — the §21 mesh-convergence VERDICT (2026-06-26) — "still over-claimed (narrowly)"

4th Codex review (gpt-5-codex, high), checking whether the re-done clean test (`seq_sorptivity_meshconv.py`)
fixed the 3rd review's objections AND whether the §21 verdict is sound. Prompt:
`scratch/_codex_verdict_review_prompt.txt`. **Verdict: the re-do FIXED the core mis-framing; §21 is sound
on the 1-D claim but STILL OVER-CLAIMS the jump to the 3-D 0.547 partition.**

## Prior objections — now fixed? (Codex)
- **(1a) ell_c locked not swept — FIXED.** `run_monolith` does NOT pass ell_c; relies on auto = top-cell
  half-height (`seq_sorptivity_meshconv.py:110-119`, `coupling.py:191-209`).
- **(1b) same-mesh same-forcing — FIXED loam/clay; sand nz=40 only mostly** (its monolith ladder was paired
  with the converged nz=24 ponding reference, not a same-mesh nz=40 ref — minor, the ref was flat).
- **(1c) pond depth = conserved volume not node-max — FIXED** (`_pond_vol_form`).
- **(1d) solver health — PARTIAL:** `nbad` reported, but NOT total-water closure or SNES reason/fnorm.

## The valid remaining hits (Codex)
1. **§21 still over-claims "0.547 is confirmed under-resolved."** This is a 1-D no-routing column; 0.547 is
   a ROUTED 3-D partition. The 1-D test makes coarse q_pot a strong SUSPECT but does NOT quantify how much
   of the 3-D gap is vertical-under-capture vs the routing↔infiltration feedback that keeps real 3-D sheets
   thin. **The jump 1-D→3-D is the weakest link.**
2. **add_ponding_bc is the right ell_c→0 reference "in spirit", numerically supported, but not proven** —
   though Codex gives the algebra: as ell_c→0, `q_pot=∫K dψ/ell_c` forces `d−ψ_top=O(ell_c)` → `d≈ψ_top` →
   pond-in-ψ. So it IS well-justified (repo reduction test `test_coupling_1d.py:110-136`).
3. **The 0.92–0.98 plateau is NOT proven → 1.0.** loam/clay still 7–8% low at nz=40, increments shrinking
   but not decisively. Defensible wording: **"largely resolution-driven, with a possible small residual
   closure difference unresolved at finite nz"** — NOT "pure artifact → 1.0".
4. **Deep-pond confound (important):** rain=10·Ks + no routing → very deep ponds (sand ~920 mm). Large `d`
   directly boosts the monolith's `kirchhoff(ψ,d)/ell_c`, so deep ponding makes the monolith look CLOSER to
   the no-film reference than a thin-sheet ROUTED case would → **the ratio is likely OPTIMISTIC for the real
   storm partition.**

## Engineering conclusion (Codex refines)
"Mesh-objective subgrid closure, not brute refinement" = right direction. But NOT "pick a better constant
ell_c" (still a mesh knob in disguise). **Concrete: keep the NCP/surface ledger, replace `q_pot` with a
mesh-objective infiltration-CAPACITY law (Green-Ampt / Smith-Parlange / Philip), or equivalently an
adaptive `ell_eff(state,time,soil)` derived from that capacity.**

## ★ The single highest-value next test (Codex)
**Rerun the actual 3-D 0.547 case on a TOP-LAYER refinement ladder** — keep the lateral mesh + forcing
fixed, refine ONLY the top vertical resolution, `ell_c` auto-locked to the local top-cell half-height,
track `routed/R`, infiltration, peak pond depth. **If 0.547 collapses materially as only the top vertical
resolution refines → §21's modeling implication holds for the PARTITION. If it stays near 0.547 → the 1-D
result does not carry over.**

## Disposition (mine)
The 1-D finding STANDS (coarse q_pot under-captures sorptivity; resolution-driven; soil-agnostic). I
OVER-REACHED in tying it to 0.547 — TEMPER §21 to Codex's wording. NEXT = the 3-D top-layer refinement
ladder on the real tilted-plane partition case (the decisive link). Keep in view: the deep-pond ratio is
optimistic, so the 3-D move may be smaller than the 1-D column suggests.
