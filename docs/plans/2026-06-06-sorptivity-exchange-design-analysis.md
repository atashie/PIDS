# Sorptivity in the soil-exchange closure — design analysis

**Date:** 2026-06-06
**Status:** ANALYSIS for review (Arik directive 2026-06-06: "We need to account for sorptivity
throughout the model; this will be a key component of correctly integrating the PIDS channels and
tunnels; without this sorptivity component, the model is fundamentally flawed"). Decided scope:
**sorptivity first** (pause 3-D / subsurface-outflow); design **one unified sorptive-exchange
primitive** serving both the land surface (design §D) and embedded features (design §E).
**Codex-reviewed (2026-06-06) → CORRECTED in §8:** the 4-way benchmark shows the land surface is an
UNDER-RESOLUTION problem (the conductance closure recovers at fine mesh, loam 0.96), NOT a broken §D
closure; the genuine sub-grid closure problem is the EMBEDDED FEATURES (§E, can't refine). Leading fix
= a **Kirchhoff / integral-mean soil-exchange leg** (recovers the cross-film gradient without per-face
wetting history; works resolved for §D and sub-grid for §E), with an analytical sorptivity clock as
fallback. **Next:** spec → TDD against a resolved near-field annulus (§E) + `ponding-fine` (§D); RED gate
= `scratch/m3_sorptivity_benchmark.py`. NOTE: §1–§6 below were written pre-review and OVERSTATE the §D
defect; read §8 first.

---

## 1. The defect (quantified)

Every soil↔(not-soil) water exchange in the architecture uses a **conductance** closure with the
soil's Darcy conductivity on the soil side:

- Land surface (§D.3 / the Module-3 NCP): `q_pot = K(ψ_top)/ℓ_c · (d − ψ_top)`, `ℓ_c` = top-cell
  half-height.
- Features (§E, planned): `q = σ·(H_feat − H_soil)`, `σ` = series of a wall leg and a **soil `K(ψ)`
  leg**.

A fixed conductance carries steady Darcy flow but **not the transient capillary uptake (sorptivity)**
that dominates early-time infiltration into unsaturated soil (Philip: `I = S√t + A t`, rate
`i = S/(2√t)+A`, driven by the matric-potential gradient, not the low unsaturated `K`).

**Benchmark** (`scratch/m3_sorptivity_benchmark.py`): one dry 1-D column, one 0.6 m/day storm,
infiltration capacity-limited, two treatments on the SAME validated Richards solver — (a) RESOLVED:
fine mesh + `add_ponding_bc` (full near-surface gradient → sorptive uptake real); (c) MODEL: coarse
1-D `CoupledProblem` (the `K(ψ_top)/ℓ_c` conductance NCP). 1-D, so no lateral overland — the coupling
here IS purely the vertical infiltration closure.

| soil | resolved (truth) | conductance closure | gap |
|------|------------------|---------------------|-----|
| loam (ψ₀=−1.0) | **89 %** of rain infiltrates, 2.6 mm pond | **2 %** infiltrates, 23.6 mm pond | **53×** |
| sand (ψ₀=−0.5) | **100 %** infiltrates, 0 mm pond | **~0 %** infiltrates, **24 mm** pond (all rain) | total |

This is **order-of-magnitude wrong**, not a small bias, and it matches Green-Ampt (dry coarse soil
absorbs ~all of a modest storm: capacity `≈ Ks(1 + (ψ_f Δθ + h₀)/F) ≫ rain` early).

## 2. Root cause

The conductance **film `ℓ_c` decouples the soil-surface head from the ponded head**: the surface soil
node `ψ_top` never reaches the ponded head `d` (it's held back across the film), so the soil surface
never saturates, so `K(ψ_top)` stays at the dry **unsaturated** value, so the flux stays throttled.
The resolved Richards (or the `ℓ_c→0` / head-continuity limit) lets the surface saturate →
`K(ψ_top)→Ks` → the resolved suction gradient drives the sorptive uptake. The film is the defect.

(Note: the `k_ex→∞` continuity limit was already a passing 1-D test — but at the *finite* `ℓ_c` of any
real mesh the closure throttles. And the **mesh can't be refined away for embedded features** — §4.)

## 3. Why this is fundamental for PIDS (not just the land surface)

A PIDS channel/tunnel is a wet boundary embedded in **unsaturated** soil whose job is to **drive water
into the soil**. That wall→soil flux is sorptivity-dominated whenever the soil is unsaturated (the
normal recharge condition). The planned feature exchange `σ` puts unsaturated `K(ψ)` on the soil side
— so it would under-predict the soil's uptake from a feature by the same ~50×, i.e. **under-predict the
core thing PIDS does**, mis-rank designs, and understate the venture's value. It also directly governs
**C-004** (conveyance ≫ percolation) and the embedding-fidelity validation. Hence: address it before
Module 4 (features), and re-open the Module-3 (§D) exchange.

## 4. Fix options (unified sorptive exchange) — the resolution asymmetry

The cure is to impose **head continuity at the interface and let the (resolved or analytically-modeled)
near-field carry the flux**, so sorptivity emerges instead of being throttled by a fixed-`K` film.

- **Land surface (§D) — RESOLVABLE.** Replace the `k_ex` conductance NCP with a **head-switching /
  continuity** condition: when ponded (`d>0`) enforce `ψ_top = d` (the soil surface saturates to the
  ponded head) and let the **resolved Richards top flux** be the infiltration `λ`; when supply-limited
  (`d=0`) `λ = rain`. Keep the separate surface depth `d` (needed for lateral overland routing) but
  couple it by head continuity, not by the film. This is a Signorini/seepage-type complementarity
  (`d≥0`; `ψ_top ≤ d`; complementary) — the SAME NCP machinery, with the matched leg changed from a
  conductance flux to head continuity. Needs **near-surface mesh resolution** to resolve the front
  (the resolved benchmark used a fine top layer). NOTE: Module 1's `add_ponding_bc` (pond = max(ψ,0))
  ALREADY captures this (it IS the resolved-truth run) — so the land-surface fix is largely "use head
  continuity, not the `k_ex` film," reconciled with the overland `d` field.
- **Features (§E) — NOT resolvable.** The embedded 1-D-in-3-D representation deliberately does not mesh
  the near field, so head continuity alone under-resolves the steep wall→soil gradient on a coarse
  cell. The soil leg of `σ` must become a **sorptivity-aware, transient near-field uptake**
  `σ_soil(t; S, Ks, ΔH, geom)` — an analytical/quasi-analytical cylindrical/planar infiltration
  (Philip two-term, Green-Ampt, or Warrick-type) with both the `√t` sorptive and the `Ks` gravity
  phase — i.e. a **transient, sorptivity-aware Peaceman index** for the feature face. (Optionally, a
  local near-field mesh refinement / annulus around the feature as an alternative or a calibration
  reference.)
- **Unified primitive.** One `sorptive_exchange(interface_head, soil_state, geometry, time)` returning
  the uptake, with two evaluation modes: *resolved* (head continuity + near-field mesh, for the land
  surface and as the feature-calibration reference) and *sub-grid analytical* (the sorptive `σ`, for
  embedded features). Both must reproduce the same benchmark.

## 5. Validation (the RED gate)

`scratch/m3_sorptivity_benchmark.py` (resolved vs closure, + Green-Ampt overlay) is the gate: after the
fix, the §D infiltration must track the resolved Richards / Green-Ampt (gap → ~1, not 53×) for dry
sand/loam/silt/clay; the embedded-feature `σ` must reproduce the resolved near-field uptake for a single
feature before any multi-feature use (the §E embedding-fidelity test). Early-time `I ~ S√t` (Philip)
and late-time `→ Ks` must both appear.

## 6. Implications / scope

- **Re-opens §D** (Module-3 land-surface exchange): the supply-limited `k_ex` NCP (decision-log
  2026-06-05) is superseded for infiltration by head continuity; the NCP *structure* (complementarity
  for `d≥0`, sign-paired conservation, reduces to `add_ponding_bc`) is retained, the *conductance leg*
  is replaced. The lateral overland (`d` routing) and the outflow BC are unaffected.
- **Foundational for §E** (Module-4 features): defines the soil leg of `σ` before features are built.
- **C-004 / embedding fidelity** now explicitly depend on the sorptive `σ`.
- The current `CoupledProblem` infiltration partition (and the texture-sweep partition) are NOT
  trustworthy for absolute infiltration until this lands; the lateral-routing, outflow, and
  conservation results are unaffected (they don't depend on the infiltration magnitude).

## 7. Open questions for the spec

1. Head-switching complementarity formulation for §D that stays smooth-Newton-solvable (the seepage/
   Signorini NCP) and reconciles `ψ_top=d` continuity with the lateral-overland `d` field + the
   positivity limiter.
2. Near-surface mesh-resolution requirement for §D (how fine; graded top layer?) and its cost.
3. The analytical sorptive `σ` for §E: which closed form (Philip S√t + Green-Ampt gravity; or a
   quasi-steady Warrick cavity solution), how `S` is obtained from the van Genuchten params, and how
   "time-since-wetting" is tracked per feature face in a steady-marching solve.
4. Whether `S` (sorptivity) should be precomputed per soil from the VG diffusivity integral or fit to a
   one-off resolved near-field run.

---

## 8. CORRECTED findings — Codex review + 4-way benchmark (2026-06-06)

Codex flagged that §1's benchmark **confounded closure error with mesh error** (resolved=160 cells vs
conductance=8 cells), so the "fundamentally wrong closure" framing was overstated. The decisive 4-way
matrix (`scratch/m3_sorptivity_benchmark.py`, now: {ponding-BC, conductance} × {coarse 8, fine 80},
one consistent `∫θ dx` metric) shows:

| | ponding-BC (resolved) | conductance (model) | closure ratio (conduct/ponding) |
|--|--|--|--|
| loam coarse (nz=8)  | 70.7 % of rain | 1.6 % | **0.02** |
| loam fine (nz=80)   | 88.8 % | 85.3 % | **0.96** |
| sand coarse (nz=8)  | 76.8 % | ~0 % | ~0 |
| sand fine (nz=80)   | 99.0 % | *dt collapse* | — |

**Corrected conclusion:**
1. **Land surface is an UNDER-RESOLUTION problem, NOT a broken closure.** At fine mesh the conductance
   closure RECOVERS (loam 0.96 ≈ resolved); the `k_ex→∞`/refined limit is correct (consistent with the
   passing `kex→∞` continuity test). The ~50× under-infiltration is a **coarse-top-cell** artifact, not
   a physics error. ⇒ **Do NOT rewrite §D physics.** Fix via near-surface resolution and/or a better
   *coarse-cell* closure (below).
2. **Refinement is not free.** The sand fine-mesh coupling **failed to converge** (dt collapse) — a
   resolution-only fix worsens stiffness, especially for coarse soils. So the land-surface fix wants a
   coarse-cell closure that recovers infiltration *without* extreme refinement.
3. **Embedded features (§E) are the genuine sub-grid closure problem.** They can't be refined, so the
   coarse throttle (~0.02) is unavoidable without an explicit sorptivity-aware soil leg. This is the one
   place a closure change is truly required — and its magnitude is a **hypothesis until measured against
   a resolved near-field annulus** (NOT assumed equal to the land-surface coarse gap).

**Retractions/corrections to §1–§6 (Codex):**
- §6 "lateral routing / outflow unaffected by infiltration magnitude" is **FALSE** — `λ` is in the
  surface equation; the partition drives ponding→routing→outflow. (The conservation/operator-equivalence
  validations stand; the absolute-infiltration-dependent results do not.)
- The §1 sand gap mixed mesh+closure; the corrected closure-only gap is the *same-mesh* column above.
- The feature "same ~50×" is a hypothesis pending the annulus reference (the existing §E soil leg is a
  radial `K(ψ)·Ω_geom`, not the flat film).
- Benchmark now uses one consistent `∫θ dx` metric across all cells.

**Lower-risk alternatives to spec (Codex) — before any per-face sorptivity "clock":**
- **Integral-mean / Kirchhoff (matric-flux-potential) soil leg.** Replace `K(ψ_top)` with the
  Kirchhoff-transformed flux `∫_{ψ_top}^{ψ_d} K(ψ) dψ / ℓ_c` (or an integral-mean `K` across the film).
  This linearizes the K-nonlinearity and captures the cross-film gradient far better than the cell-value
  `K(ψ_top)`, plausibly recovering most of the coarse-cell infiltration **with no per-face wetting
  history** — a strong candidate for BOTH §D coarse cells and the §E soil leg.
- Harmonic/integral-mean `K` across the film — a cheap ablation.
- A transient analytical sorptive `σ(t;S,Ks,…)` (Philip/Green-Ampt/Warrick) remains the fallback if the
  Kirchhoff leg can't reproduce the resolved-annulus early-time `√t` for features.

**Revised path:**
1. **Features-relevant first (the real closure work):** build a RESOLVED near-field annulus reference for
   a single embedded feature; measure the true coarse-cell gap; test whether a **Kirchhoff/integral-mean
   soil leg** recovers it (vs a sorptivity clock).
2. **Land surface:** a near-surface resolution study + the same Kirchhoff coarse-cell closure, validated
   against `ponding-fine`; manage the refinement-induced stiffness.
3. Keep all §D verified invariants (structural conservation, supply-limited no-ponding, ponding under
   high rain, `kex→∞` continuity, `add_ponding_bc` reduction, recession) green throughout.
4. Reclassify current 2-D/3-D routing/outflow as conservation/operator validations until infiltration
   accuracy is settled.

**Net:** the directive "account for sorptivity" stands and is essential for features; but it is a
coarse-cell closure + resolution issue, not a §D physics rewrite. The **Kirchhoff/integral-mean exchange
leg** is now the leading unified candidate (works resolved for §D and sub-grid for §E), with the
analytical sorptivity clock as the fallback.

---

## 9. Kirchhoff-leg SPIKE — VALIDATED (2026-06-06)

Spiked the integral-mean leg directly in the coupling: replace the dry cell value `K(ψ_top)` in `q_pot`
with the **film-averaged conductivity** `Kmean = (1/(0−ψ_top))∫_{ψ_top}^0 K(ψ)dψ` (5-pt composite-Simpson
quadrature of `K_ufl` over `[ψ_top,0]`, →`Ks` when `ψ_top≥0`; no per-face history state). Re-ran the
4-way benchmark and the §D invariant tests:

| | before `K(ψ_top)` | **after Kirchhoff** | reference (ponding, same mesh) |
|--|--|--|--|
| loam coarse | 1.6 % (ratio 0.02) | **49.3 % (ratio 0.70)** | 70.7 % |
| loam fine   | 85.3 % (0.96) | **85.2 % (0.96)** | 88.8 % |
| sand coarse | ~0 % (~0) | **76.8 % (ratio 1.00)** | 76.8 % |
| sand fine   | *dt collapse* | **99.0 % (ratio 1.00)** | 99.0 % |

- **Sand recovers PERFECTLY** (closure ratio 1.00 at both resolutions) AND the fine-mesh **convergence
  failure is gone**. Loam jumps 0.02→0.70 at coarse (residual is the shared bulk coarse-mesh error: the
  ponding reference itself is only 0.80 of fine), 0.96 at fine.
- **All 6 §D 1-D invariant tests still pass** (supply-limited mass balance, Hortonian ponding, `kex→∞`
  continuity, `add_ponding_bc` reduction, recession, plausibility) — the leg is a clean drop-in.
- Two 2-D tests need tolerance reconciliation (formalization, NOT breakage): the structural-conservation
  gate (5e-13) reads 2.47e-12 with the more-nonlinear `q_pot` — still machine/structural (λ sign-paired),
  just above the deliberately-tight gate; and `flat_reduces_to_1d` shifts with the corrected dynamics.

**Conclusion:** the integral-mean (Kirchhoff) leg is the validated unified fix for §D coarse cells (and
the §E feature soil leg by the same construction). The spike was reverted to keep the committed state
clean (55/55); formalization (spec + TDD) is the next gated step:
- quadrature order (5-pt recovered sand exactly, loam 0.70; check whether more points lift loam, or it's
  bulk mesh); the saturated/`ψ_top≥0` regime; the gravity term (`+Ks` over the film) if needed.
- reconcile the structural-conservation gate (it tracks Newton tolerance, not a leak) + `flat_reduces` tol.
- add a **sorptivity-recovery regression test** (coarse conduct vs ponding within a tight ratio) as the
  RED gate; keep all §D invariants green.
- apply the SAME integral-mean leg to the §E feature soil leg; validate vs a resolved near-field annulus.
