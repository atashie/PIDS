# Field-scale feasibility probe (2 ha, heterogeneous layered soil) — 2026-06-08

**Status:** done · **Verdict: FEASIBLE.** A one-off probe (not a permanent feature) run BEFORE Module 4,
to check whether the whole coupled process is tractable at realistic field scale with depth-varying,
layered soil on a badly-anisotropic mesh. Script: `forward-model/scratch/feasibility_2ha_layered.py`.

## Scenario
- **Domain:** 210 × 95 m ≈ **2.0 ha** (30 m x-res → nx=7; 5 m y-res → ny=19); 2% bed slope toward the
  x=210 m outlet; 1 m deep. **Graded z-mesh** (nz=21): 0.05 m layers + a **0.01 m sand cell at
  z∈[0.50,0.51]**; impermeable base at z=0. **10,560 DOFs, 16,758 tets**, aspect ratio up to ~600:1
  (30 m × 5 m × 0.01–0.05 m).
- **Soil (heterogeneous):** surface **loam** (Ks 0.25) with **depth-decaying Ks (→0.048)** and
  **θ_s (→0.38)** toward clay-LIKE values at depth — **NOTE (Codex review):** only Ks/porosity decay;
  the van-Genuchten **retention SHAPE (α, n, θ_r) stays loam**, so this is a *clay APPROXIMATION* (per
  the spec "decay Ksat/porosity to approximate a clay"), **not a true clay retention curve**. A high-K
  **sand** layer (Ks 7.13, sand retention) at 0.5 m = a **capillary barrier**. Implemented as a
  duck-typed `LayeredSoil` (z-dependent van-Genuchten via a `SpatialCoordinate(z)` conditional + exp) —
  **NO changes to the validated module code**.
- **Forcing:** ψ₀ = −1.0 m; rain 0.3 m/day for 0.3 d (> loam Ks → infiltration-excess) then recession
  to 0.5 d; surface Manning edge outlet; impermeable base + no-flux sides (no subsurface GHB).

## Result
| metric | value |
|---|---|
| status | **COMPLETED** (full 0.5 d storm) |
| wall clock | **720.4 s (~12 min)**, 349 steps, **2.06 s/step** (compile+first 1.5 s, warm FFCX cache; ~5 s cold) |
| mass balance | discrete (vertex-lumped) balance **\|Δtotal − (cum_rain − cum_outflow + clip_adj)\|/cum_rain = 3.82e-12**; **clip_mass_adjust ≈ 1.6e-15** (limiter adjustment negligible) |
| rain / runoff / infiltration | 1795.5 m³ in (90 mm; cutoff now exact) · 305.3 m³ runoff (17.0%) · ≈1490 m³ infiltrated |
| Newton | **3–4 iters/step** except a ~10-step stiff patch at the rain cutoff (iters → 50) |
| peak surface depth | ~3.6 mm during the storm (thin overland sheet), 0 after recession |

## Findings (feasibility implications)
1. **The whole process scales to 2-ha heterogeneous field runs** — minutes, not hours; the discrete
   (vertex-lumped) mass balance closes to **~1e-12** (limiter adjustment ≈ 0); the direct (MUMPS LU)
   solver is **robust to the ~600:1 anisotropy** (no convergence breakdown — the main risk going in).
   Tractable for verification/assessment.
2. **Heterogeneous layered soil is achievable via a duck-typed shim** (z-dependent UFL closures), and
   **compiles fast** (~5 s cold) — the conditional sand layer + exp decay are FFCX-fine; the quadrature
   cap (default 8) keeps assembly cheap. The discrete lumped balance holds (~1e-12) even with the
   discontinuous (sand-layer) coefficients.
3. **Stiff patch at the rain on/off discontinuity** (Newton 25–47, dt → 8e-5 for ~10 steps, ~2 of the 11
   min) that **self-recovered**. A ramped hyetograph (or a smoothed rain cutoff) would remove it.
4. **Limits (for a future field-scale capability, not blockers for the probe):**
   - **Serial direct-solve only.** MPI + iterative (GMRES/fieldsplit) untested at this scale; needed for
     larger meshes / optimization loops (LU cost grows fast).
   - **The `LayeredSoil` shim assumes no subsurface GHB** — the kr-weighting in `add_drainage_bc` still
     uses scalar `soil.Ks`. A PROPER heterogeneous-soil feature (spatial Ks for kr; spatial ℓ_c) is needed
     for groundwater drainage / lateral GHB on layered soil. (Here the base is impermeable + outlet is the
     surface Manning, so it never bit.)
   - **Flat-mesh + slope-via-z_b convention** (the soil column is vertical, slope only in the overland
     z_b) carries over; a true slope-normal column is a later modeling choice.
   - Capillary-barrier/perching physics is present; the run now emits a NetCDF (ψ/θ cross-sections +
     top-down θ z-layer maps + the Ks(z) soil profile) so it can be inspected visually (Tier-3 HTML).

## Bottom line
The infiltration-runoff engine is **field-scale-feasible** for heterogeneous, layered, anisotropic 2-ha
runs on a stock serial toolchain. The next capability investments (when needed) are: a first-class
**heterogeneous-soil** module (spatial Ks/θ_s/ℓ_c, with kr fixed) and a **scalable solver path**
(iterative + MPI). Neither blocks Module 4 (§E embedded features).

## Codex review (2026-06-08) — applied
A static Codex audit returned FIX-FIRST; all findings addressed and the case re-run:
- **(blocker) rain cutoff** was applied to a step *straddling* `storm_dur`, treating it as fully dry →
  the step is now **clipped exactly at `storm_dur`** (cum_rain is now exactly 1795.5 m³; the bug had
  dropped ~1 m³). Passing finds (no change needed): scalar `soil.Ks` is not misapplied in this no-GHB
  config; the graded-mesh z-remap and sand-layer bounds are correct.
- **(should-fix) soil description** corrected: only Ks/θ_s decay; the retention shape stays loam (a clay
  *approximation*, above). The dead/wrong `se_ufl` was removed.
- **(should-fix) conservation wording** corrected: it is the **discrete vertex-lumped** balance, and the
  full balance now carries `clip_mass_adjust` (verified ≈ 1.6e-15 ⇒ negligible).

## Continuation to 0.8 d (+ checkpoint/resume)
Extended the run **0.5 → 0.8 d** (deterministic re-run reproduces 0→0.5 exactly, then continues): wall
~17.5 min / 512 steps, mass balance **3.09e-12**. After the storm ends the integrated budget is FROZEN
(`cum_outflow` 305.3 m³, `soil_water` 5885.55 m³ — impermeable base + no rain ⇒ no budget change); the
recession converges trivially (Newton = 3, dt at the 2e-3 cap). The 0.5→0.8 behavior is purely
**internal subsurface redistribution** (the front advancing/perching at the sand barrier + low-K base).
The probe now writes a **restartable checkpoint** (full ψ/d field + cumulative accounting), so further
continuations are a true instant resume (`python scratch/feasibility_2ha_layered.py resume`) rather than
a re-run. (HTML `validation/sanity/viz/feasibility_2ha_0p8d__2026-06-08.html`; gitignored.)
