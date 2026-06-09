# Benchmarks — In-House Forward Model vs ParFlow (side-by-side)

## 1. Purpose
Head-to-head comparison artifacts for the in-house [`../forward-model/`](../forward-model/)
against the off-the-shelf [`../parflow/`](../parflow/) benchmark, on PIDS-free bulk
hydrology. This folder holds the **comparison outputs only** — compact summaries plus
self-contained, side-by-side **HTMLs you can open by double-click** to view both models'
results together.

Distinct from its neighbors: the ParFlow *tool* (install / usage / case decks) lives in
[`../parflow/`](../parflow/); the in-house model's own single-engine sanity outputs live
in [`../validation/sanity/`](../validation/sanity/). **Here = the two models compared.**

## 2. Layout
| Path | Contents |
|---|---|
| `data/` | one compact **comparison summary** per case (`.nc`): both models' profiles on a common grid + matched times, per-model scalar diagnostics, and difference metrics. *Summaries only* — no raw `.pfb` / full-field dumps. |
| `html/` | one **self-contained interactive HTML** per case (Plotly inlined, offline) — the side-by-side viewer. |
| `make_comparison_html.py` | generator: reads `data/<case>.nc` → writes `html/<case>.html`. Extends the standardized Tier-3 viz ([`visualize-sanity-check-routine`](../../../governance/visualize-sanity-check-routine.md)) to two models. Imports neither solver. |

## 3. Data contract (combined comparison summary)
One NetCDF per case so the generator stays solver-free:
- dims `(time, z)`: `head_inhouse`, `head_parflow`, `theta_inhouse`, `theta_parflow` (+ their differences).
- per-model scalars: mass-balance error. *(Performance/timing deferred — accuracy-only — until a native ParFlow build enables a fair wall-clock comparison; decided 2026-06-08.)*
- global attrs: case, soil + vG params, both grids, and the active **formulation deltas** (air-entry, Ss, FV-vs-FEM) so the view is self-documenting.
- Both engines interpolated to a **common z grid** and **matched output times**.

> **Reading the mass-balance panel.** The plotted error is the water-content (θ) balance
> `|Δ∫θ − cumulative input| / cumulative input`, identical for both models so they're directly
> comparable. The in-house model has no specific storage, so `∫θ` is its exact conserved quantity
> (error ~1e-13, machine). ParFlow *requires* a small `Ss` and conserves the **full** storage
> `∫(θ + Ss·S·ψ)`; the θ-only metric omits that compressible term, so ParFlow's θ-error rises to
> ~3e-6 as the column wets (ψ → 0) — that residual is water stored **compressibly, not lost**.
> ParFlow's full-storage balance closes to ~2e-9, as tight as the in-house model. **Both conserve
> mass to tolerance; the higher ParFlow θ-error is the metric definition, not a model deficiency**
> (and `Ss` can't be set to 0 in ParFlow without the saturated zone going singular).

## 4. Conventions
- Self-contained offline HTML (Plotly inlined), per the standardized viz routine — opens by double-click on Windows.
- Naming: `data/<module>__<case>__<YYYY-MM-DD>.nc`, `html/<module>__<case>__<YYYY-MM-DD>.html`.
- Regenerate: `python make_comparison_html.py data/<case>.nc html/<case>.html`.

## 5. Cases
| Case | Status |
|---|---|
| subsurface — 1-D loam infiltration column | ✓ **done 2026-06-08** — RMS Δθ=9.1e-3 (max 0.050), RMS Δψ=0.054 m (max 0.34 m); differences front-localized. [`html/subsurface__column_1d__2026-06-08.html`](html/subsurface__column_1d__2026-06-08.html) |
| subsurface — non-ponding sweep (dry/mesic × small/typical) | ✓ **done 2026-06-09** — 4 matched columns; RMS Δθ 0.002–0.005, both mass-conservative; front-localized maxima (see §5b). `html/subsurface__{mesic,dry}_q0{02,10}__2026-06-09.html` |
| overland — standalone hillslope | ⚠ **deferred 2026-06-09** — ParFlow standalone-overland limitation (see §5a); benchmarked instead via the coupled comparison (§5c) |
| coupled — surface↔subsurface (ponding) | ✓ **done 2026-06-09** — 6 scenarios (normal/extreme rain × dry/normal/wet antecedent); RMS Δθ 0.003–0.019, both mass-conservative, peak ponding agrees to ~5–10 mm (see §5c). `html/coupled__{normal,extreme}_on_{dry,normal,wet}__2026-06-09.html` |

## 5a. Overland (B3) — deferred: ParFlow standalone-overland limitation
**Decision 2026-06-09:** the standalone overland comparison is **deferred**. Overland will be
benchmarked via the **coupled** surface–subsurface comparison instead — ParFlow's design domain.

**Why.** ParFlow is a coupled Richards + overland code and does **not** cleanly reproduce a
*standalone* overland sheet at the realistic small Manning the in-house pure diffusion-wave
solver targets (n≈0.05 ⇒ `n_PF = n_SI/86400 ≈ 5.8e-7` in day-units). Two attempts, both
diagnosed and Codex-reviewed:
1. **dry-IC slab** → the thin subsurface absorbed the storm (ParFlow ponded ~1 mm vs the analytic
   Manning-equilibrium 4.5 mm), and the surface-storage-balance outflow was contaminated by infiltration.
2. **saturated IC + exact ponding extraction** (`d = max(p_cell − DZ/2, 0)`, removing the
   saturated-cell hydrostatic offset) → ParFlow built **~0 mm** ponding: it evacuates water at ~0
   depth rather than forming the equilibrium sheet, consistent with the extreme conveyance
   (`1/n ≈ 1.7e6`) interacting with the `OverlandDiffusive` regularization.

ParFlow's own overland examples (`overland_slopingslab_*`, `overland_tiltedV_*`) are coarse
numerical **regression toys** (dry IC, 5×5 grids, undeclared Manning units) — not physically-
calibrated hydrograph benchmarks — so they offer no clean validated standalone path either.

**The implementation is retained as correct reference** — [`../parflow/cases/overland_hillslope.py`](../parflow/cases/overland_hillslope.py)
and [`build_comparison_overland.py`](build_comparison_overland.py). The Manning bridge, saturated-IC
runoff setup, and offset-corrected depth extraction are all sound; the blocker is ParFlow's
standalone-overland regime, not the scripts. They would serve the coupled comparison (B4) or a
rougher-Manning (n≈0.15) variant if a standalone overland number is later required.

## 5b. Subsurface sweep (non-ponding matrix) — done 2026-06-09
Four matched columns spanning antecedent wetness × storm intensity, constant **sub-Ks** rain
(storm-only infiltration; ponding / saturation-excess scenarios are deferred with the surface-water
work — see §5a). Agreement (in-house − ParFlow, common 101-pt grid; Carsel & Parrish loam):

| scenario | ψ₀ (m) | q (m/day) | RMS Δθ | max\|Δθ\| | RMS Δψ (m) | max\|Δψ\| (m) | MBE in-house | MBE ParFlow |
|---|---|---|---|---|---|---|---|---|
| mesic_q010 | −1.0 | 0.10 | 5.0e-3 | 0.028 | 0.029 | 0.20 | 1.5e-12 | 3.7e-6 |
| mesic_q002 | −1.0 | 0.02 | 2.0e-3 | 0.011 | 0.012 | 0.065 | 8.6e-12 | 6.9e-6 |
| dry_q010 | −5.0 | 0.10 | 4.4e-3 | 0.059 | 0.126 | 1.34 | 4.1e-12 | 1.6e-5 |
| dry_q002 | −5.0 | 0.02 | 2.4e-3 | 0.037 | 0.089 | 0.92 | 2.7e-11 | 2.6e-5 |

**Takeaways:** the two engines agree closely in the bulk (RMS Δθ ≤ 0.005 across all cases); the
largest differences are localized at the wetting front and grow with antecedent dryness (a sharper
front — where the FV-vs-FEM discretization + the air-entry-K delta bite hardest; max\|Δψ\| reaches
~1.3 m on the −5 m dry front). Both engines are mass-conservative — the ParFlow MBE (~1e-5) is the
θ-balance's omitted compressible `Ss` term (see the §3 note), not a conservation failure. Reproduce:
`../parflow/cases/column_sweep.py` (ParFlow, storm-only constant rain) → `build_comparison_sweep.py`
(in-house run + comparison `.nc` + HTML).

## 5c. Coupled surface↔subsurface (B4, ponding) — done 2026-06-09
The coupled comparison the standalone-overland case was deferred to (§5a), run on ParFlow's
**native coupled turf**: an **`OverlandFlow`** top BC (ParFlow's surface ponding store) on a flat
2 m / 80-cell loam column (Ks=0.25) with a **closed no-flux base**, matched to the in-house
Module-3 `CoupledProblem` sanity matrix (`forward-model/viz/run_coupling_sanity.py`): storm +
recession across rain {normal 0.30 m/d × 0.30 d; extreme 3.0 m/d × 0.05 d} × antecedent
{dry ψ₀=−3, normal −1, wet −0.15} = **6 scenarios**. Flat topography (`TopoSlopes=0`) ⇒ no lateral
routing ⇒ pure ponding / re-infiltration partition — the direct analog of the in-house surface
store. Recession via a **two-phase restart** (storm run → `PFBFile`-IC restart with rain off; the
storm/recession time cycle was unreliable in the sweep — §5b). Reproduce: `../parflow/cases/coupled_column.py`
→ `build_comparison_coupled.py` (in-house re-run + comparison `.nc` + coupled HTML via
`make_comparison_coupled_html.py`, which adds a surface-depth + infiltration-partition panel).

**Key methodological finding — the ParFlow ponding-store convention.** ParFlow's **mass-consistent**
`OverlandFlow` surface store is **`d = max(ψ_top, 0)`** (the positive part of the top-cell pressure),
**not** the `max(ψ_top − DZ/2, 0)` hydrostatic-offset extraction used for the *standalone* overland
case (§5a). With `max(ψ_top, 0)` the closed-column θ+pond balance `|Δ∫θ + d − cum_rain|` closes to
**≤2.6e-5** at every frame across the matrix; the DZ/2 offset breaks it by **exactly DZ/2 ≈ 12.5 mm**
the instant ponding starts (verified frame-by-frame). The `max(ψ_top, 0)` store carries a +DZ/2
cell-centred head offset vs the in-house surface-node depth — a documented FV-vs-FEM discretization
delta, not a leak.

Agreement (in-house − ParFlow, common 101-pt grid on [0,2] m; ParFlow interpolated to the in-house
output times; SE-Piedmont loam):

| scenario | ψ₀ (m) | rain (m/d × d) | RMS Δθ | max\|Δθ\| | max\|Δψ\| (m) | peak d ih/pf (mm) | MBE in-house | MBE ParFlow |
|---|---|---|---|---|---|---|---|---|
| normal_on_dry | −3.0 | 0.30 × 0.30 | 7.3e-3 | 0.078 | 0.82 | 2.9 / 4.5 | 5.8e-12 | 6.9e-6 |
| normal_on_normal | −1.0 | 0.30 × 0.30 | 9.4e-3 | 0.044 | 0.32 | 4.8 / 6.8 | 4.8e-12 | 3.1e-6 |
| normal_on_wet | −0.15 | 0.30 × 0.30 | 3.2e-3 | 0.032 | 1.76 | 15.5 / 13.0 | 8.2e-13 | 2.6e-5 |
| extreme_on_dry | −3.0 | 3.0 × 0.05 | 1.9e-2 | 0.187 | 2.49 | 112.4 / 103.3 | 4.1e-12 | 4.1e-6 |
| extreme_on_normal | −1.0 | 3.0 × 0.05 | 1.1e-2 | 0.116 | 0.60 | 114.6 / 110.7 | 4.9e-12 | 2.1e-6 |
| extreme_on_wet | −0.15 | 3.0 × 0.05 | 4.2e-3 | 0.030 | 1.96 | 123.8 / 128.8 | 3.2e-12 | 1.8e-5 |

**Takeaways:**
- **Bulk profiles agree** (RMS Δθ ≤ 0.019); the largest θ/ψ differences are front-localized and grow
  with antecedent dryness + intensity (extreme_on_dry: sharpest front, max\|Δψ\|≈2.5 m), where the
  FV-vs-FEM discretization and the **air-entry-K cap delta bite hardest — and that delta is *more*
  active here than in the sweep because the column SATURATES at the surface** (the cap modifies
  θ/K/tangent near saturation; ParFlow omits it).
- **The ponding partition agrees well** across two fundamentally different surface formulations
  (in-house land-surface exchange flux + ponding store vs ParFlow `OverlandFlow`): peak ponding
  depths match to **~5–10 mm**, comparable to (and sometimes opposite-signed from) the DZ/2 store
  offset, so neither the offset nor the formulation difference dominates.
- **Both engines are mass-conservative**: in-house machine-exact (≤6e-12, no Ss), ParFlow θ+pond
  ≤2.6e-5 (the residual is the omitted compressible Ss term, per the §3 note).
- **The partition physics is coherent across the matrix**: normal rain (just above Ks) infiltrates with
  only transient mm-scale ponding on dry/normal soil; extreme bursts (3 m/d ≫ Ks) pond hard (~0.1 m)
  then the recession re-infiltrates by soil capacity — **fully on dry soil (Hortonian flash-then-soak),
  nearly fully on normal, and only halfway on wet** (the closed column saturates and holds ~73 mm
  ponded with no outlet). This is the infiltration-excess / storage-excess regime the subsurface
  sweep deferred (§5a/§5b), now closed.

## 6. Change log
| Date | Change |
|---|---|
| 2026-06-08 | Folder created for side-by-side benchmark artifacts (summary data + interactive HTMLs). Generator + first case (subsurface column) to follow. |
| 2026-06-08 | HTML layout revised: θ(z) and ψ(z) now overlay both models on shared axes, each with a dedicated error panel Δ(z)=in-house−ParFlow (animated on the slider). |
| 2026-06-08 | First case built: subsurface 1-D column. `build_comparison_column.py` runs the in-house model on the matched setup + loads ParFlow profiles → `data/subsurface__column_1d__2026-06-08.nc`; `make_comparison_html.py` → `html/subsurface__column_1d__2026-06-08.html` (self-contained, offline-verified). Accuracy-only. Agreement: RMS Δθ=9.1e-3 / max 0.050; RMS Δψ=0.054 m / max 0.34 m; both engines mass-conservative; differences localized at the wetting front (FV-vs-FEM + air-entry-K deltas). |
| 2026-06-09 | Overland (B3) standalone comparison **deferred** (see §5a) — documented the ParFlow standalone-overland limitation; Codex-reviewed diagnosis. Overland deferred to the coupled comparison. Removed the preliminary flawed-extraction overland `.nc`. Proceeding to the subsurface scenario sweep. |
| 2026-06-09 | **Subsurface non-ponding sweep done** (4 scenarios: dry/mesic × small/typical sub-Ks storms; see §5b). `../parflow/cases/column_sweep.py` (ParFlow, storm-only constant rain) + `build_comparison_sweep.py` (in-house) → 4 `.nc` + 4 self-contained HTMLs. Agreement RMS Δθ 0.002–0.005, front-localized maxima; both mass-conservative. Diagnosed + fixed two ParFlow setup traps en route: FluxConst forcing q>Ks (or a wet IC) → surface-pressure blowup (ponding needs the surface store, deferred §5a); and a storm/recession time cycle that never switched off (→ storm-only constant rain). |
| 2026-06-09 | **Coupled comparison done (B4, ponding)** — 6 scenarios (normal/extreme rain × dry/normal/wet) matching the in-house Module-3 `CoupledProblem` against ParFlow's native coupled mode (`OverlandFlow` surface store on a flat closed-base 2 m loam column; two-phase storm+recession restart). New `../parflow/cases/coupled_column.py` + `build_comparison_coupled.py` + `make_comparison_coupled_html.py` (adds a surface-depth + infiltration-partition panel) → 6 `.nc` + 6 self-contained HTMLs. RMS Δθ 0.003–0.019 (front-localized; air-entry-K delta more active here as the column saturates), both mass-conservative, peak ponding agrees to ~5–10 mm (see §5c). Resolves the deferred ponding/overland regime (§5a). Key finding: ParFlow's mass-consistent OverlandFlow surface store = `max(ψ_top, 0)` (closes the closed-column θ+pond balance to ≤2.6e-5), NOT the `−DZ/2` extraction used for standalone overland (which breaks it by exactly DZ/2 once ponded). |
