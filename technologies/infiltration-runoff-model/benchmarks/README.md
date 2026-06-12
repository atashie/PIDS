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
| coupled 3-D — hillslope (lateral routing + GW seepage) | ✓ **done 2026-06-10** — loam (overland) matches to ~1% (overland 0.479 vs 0.474 m³, infiltration identical, peak ponding 0.72 vs 0.73 mm); sand (lateral GW) qualitatively matches with a documented GHB **BC delta** (constant-head face over-drains ~4.8×). Early-infiltration-transient delta characterized (see §5d). `html/coupled_3d__{loam_overland,sand_lateral_gw}__2026-06-10.html` |
| canonical tilted-V catchment (B6, convergent routing) | ✓ **done 2026-06-10/11, P0-corrected 2026-06-11** — ParFlow reproduces the known answer exactly (Q → **1.000·Q_eq** = 4.86 m³/s, kinematic AND diffusive); the in-house engine reaches **~1.0·Q_eq** with the engine per-step ledger closed *(the originally published "0.676·Q_eq / 20% mass-ledger gap" was NOT reproducible from the committed deck and is **RETRACTED** — see §5e)*; what remains real: scale-independent convergent-routing stiffness (mechanism measured) + a field-scale Defect-A accuracy limit → **core fix** workstream (PIDS installs along convergence lines), plan + P0 results in §5e. `html/tilted_v__canonical__2026-06-11.html` |

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

## 5d. Coupled 3-D hillslope (B5, lateral routing + GW seepage) — done 2026-06-10
The 3-D extension of B4: ParFlow's **native 3-D coupled** mode vs the in-house Module-3 `CoupledProblem`
on the in-house 3-D hillslope sanity case (`forward-model/viz/run_coupling_3d_sanity.py`) — a 5×1×1 m
box, mesh **16×6×8**, bed tilted `z_b=S0·(L−x)` with **S0=0.05**, hydrostatic antecedent **water table
at z=0.35**, Manning **n=0.05**, storm **0.5 m/day × 0.30 d** then recession to 0.50 d. **Two outlets at
x=L**: a surface **Manning overland edge** (the codim-2 ridge in-house; the downslope overland boundary in
ParFlow) and a **lateral groundwater outlet** on the x=L side face. **Texture contrast**: SAND (Ks=7.13 ≫
rain → infiltrate + lateral GW) vs LOAM (Ks=0.25 < rain → infiltration-excess overland). ParFlow deck:
[`../parflow/cases/coupled_hillslope_3d.py`](../parflow/cases/coupled_hillslope_3d.py) (flat grid +
`TopoSlopesX=−S0` routing, `OverlandFlow` top store, `DirEquilRefPatch` lateral-GW head face, two-phase
storm+recession restart). Harness: `build_comparison_coupled_3d.py` (loads the in-house `.nc` — no FEM
re-run — + the ParFlow `.npz`) + `make_comparison_coupled_3d_html.py`.

**Overland/GW separation method (validated).** ParFlow's `PrintOverlandSum` proved **unreliable** as an
overland *volume* on the kinematic `OverlandFlow` BC (read ~0.03 m³ against a true ~0.48 m³ — cross-checked
against the analytic Manning-equilibrium ponding depth). The robust split is the **difference of two runs**:
`overland = total_out(no-GW control, x=L face → no-flow)`; `lateral_GW = total_out(full) − total_out(control)`.
The 4-way partition (`cum_rain = infiltration + ponding + overland + lateral_GW`) then closes by construction.

**LOAM (overland-dominated) — strong quantitative match:**

| component | in-house | ParFlow | agreement |
|---|---|---|---|
| infiltration (Δ subsurface storage) | 0.2454 | 0.2454 m³ | identical |
| surface overland | 0.4741 | 0.4794 m³ | **1.01×** |
| lateral groundwater | 0.0282 | 0.0252 m³ | 0.89× |
| peak ponding depth | 0.73 | 0.72 mm | ~identical |

ParFlow's native coupled overland routing reproduces the in-house Manning-edge outlet + diffusion-wave to
**~1%**. Both pre-identified risks (§5a small-Manning conveyance; the GHB mapping) are **benign for loam**:
the thin Manning-equilibrium sheet forms correctly (peak ~0.7 mm = analytic `(q·n/(86400·√S))^0.6`; **the B3
standalone ~0-depth failure does NOT recur in the coupled 3-D setting**), and the lateral GW is small in both.

**SAND (lateral-GW-dominated) — qualitative match + documented GHB BC delta:**

| component | in-house | ParFlow | note |
|---|---|---|---|
| infiltration | 0.6943 | 0.5022 m³ | ParFlow stores less (over-drained) |
| lateral groundwater | 0.0514 | 0.2478 m³ | **4.82× over-drain** |
| surface overland | ~0 | ~0 | both: no ponding ✓ |

The in-house lateral-GW outlet is a **finite kr-weighted GHB** `q_n = C·kr(ψ)·(ψ+z−H_ext)`, C=0.5/day,
H_ext=0.20 — a Robin/Cauchy condition ParFlow has **no native analog** for. Its natural representation is a
**constant-head** face (`DirEquilRefPatch`, effectively C→∞), which over-drains the saturated toe ~4.8× when
Ks is large (7.13) and the water table climbs (so ParFlow's water table can't rise as far — it stores 0.502
vs the in-house 0.699 m³). The bulk physics is qualitatively right (infiltrates, no ponding, drains laterally),
but the lateral-GW **magnitude is BC-parameterization-dependent, not a model-physics discrepancy** (a finite-
conductance ParFlow drain-column was scoped and **declined** — Arik 2026-06-10: document as a BC delta).

**Early-infiltration-transient delta (loam) — characterized.** The two models reach the same partition end-state
but differ in the *timing* of the early infiltration→runoff transition. Putting the infiltration rate next to
the surface saturation S_surf is the diagnostic: at **equal S_surf≈0.985** (t≈0.045 d) the in-house rate has
already collapsed to its steady value (~0.85) while ParFlow is **still infiltrating at the full ~2.45 rate**.
The in-house **sorptive Kirchhoff surface closure** (`q_pot=∫K dψ/ℓ_c`, §D) — reinforced by the air-entry cap
and the node-at-surface FEM — decays infiltration capacity as the surface approaches saturation, so it reaches
the steady rate early and **generates runoff sooner**; ParFlow's **coarse 8-cell column buffers** rain at near-
full rate until the top cell hits S=1.0, then chokes — a longer, more curved early transient (would converge
under vertical refinement). Falsifiable corroboration, confirmed: **runoff onset in-house 0.020 d vs ParFlow
0.070 d** (ParFlow ~3.5× later, because it over-infiltrates early). Shown in the comparison HTML's infiltration-
rate + S_surf panel. A **timing** delta — the integrated partition and mass balance (machine-precision in both)
agree.

**Other formulation deltas in play:** both models use a FLAT grid — ParFlow `TopoSlopes` vs the in-house
`set_topography` z_b field (the bed tilt enters only the overland routing operator in both; *corrected
2026-06-11 — this line previously claimed a "geometrically tilted" in-house mesh, which was wrong: the
in-house mesh is a flat box, confirmed in `coupling.py`*); no Vogel/Ippisch air-entry cap; tiny
`SpecificStorage` vs no-Ss; cell-centred FV (16×6×8) vs P1 FEM (17×7×9 nodes); ParFlow `cum_rain`=0.750
vs in-house 0.748 (adaptive-step hyetograph integration).

## 5e. Canonical tilted-V catchment (B6, convergent routing) — done 2026-06-10/11: findings + core-fix plan
Both models run the **canonical known-answer watershed benchmark** (Di Giammarco 1996 / Kollet & Maxwell
2006; the IH-MIP2 test): two 810×1000 m hillslope planes (cross-slope **5%**) converging to a central
channel line (valley slope **2%**), Manning n=0.015, rain **3·10⁻⁶ m/s × 90 min** then 90 min recession,
near-impermeable bed → the analytic equilibrium outlet discharge is the known answer:
**Q_eq = rain·area = 419,904 m³/day = 4.86 m³/s**. ParFlow deck:
[`../parflow/cases/tilted_v_catchment.py`](../parflow/cases/tilted_v_catchment.py) (env-parameterized
scale / wave form / Manning; `PF_SKIPRUN=1` re-extracts from existing `.pfb`); in-house runner:
`../forward-model/scratch/_tiltedv_spike.py`; harness: `build_comparison_tiltedv.py` +
`make_comparison_tiltedv_html.py`.

| run | equilibrium plateau | recession | mass books (ENGINE per-step ledger) |
|---|---|---|---|
| ParFlow `OverlandKinematic` 48×30 | **1.000·Q_eq** | → 0 clean | closes (storage-balance extraction) |
| ParFlow `OverlandDiffusive` 48×30 | **1.000·Q_eq** | → 0 clean | closes |
| in-house `CoupledProblem` 24×16×3 *(P0-corrected 2026-06-11)* | **~1.0·Q_eq** (0.997–0.998 across controller settings) | → 0 | **closes to ≤2e-12·cum rain** (storm-window diagnostics; full-window spike runs ≤1e-11); `clip_mass_adjust=0`; stalls = explicit rejected steps |
| in-house field-scale 162 m, n=0.015 *(P0)* | **0.876·Q_eq** (coupled 24×16) — honest books, under-resolved *solution* (clip ~1.3 cm vs ~3 mm sheet; heals to 1.01 at 48×30 standalone; true physics equilibrates: t_conc ~0.005–0.01 d ≪ storm) | → 0 clean | closes to −8.6e-12·cum rain (full window) |

**Findings (P0-corrected 2026-06-11 — the original finding-2 numbers are RETRACTED, plan §8):**
1. **ParFlow reproduces the known answer exactly** (both wave formulations) — the off-the-shelf model
   validated on its core watershed turf. Its overland boundary is **free outflow** (the outlet evacuates
   at ~0 depth at ANY Manning — §5a re-confirmed at watershed scale), so its hydrograph is only
   recoverable by storage balance; a cold-start surge contaminates the rising limb (the plateau +
   recession are the clean comparison).
2. **The in-house engine routes the V to the known answer with closed books.** The originally published
   "0.676·Q_eq plateau / 20% mass-ledger gap" was **not reproducible from the committed deck** under any
   tested configuration and is **RETRACTED** (P0 re-ran it instrumented, both dt-controller settings:
   plateau 0.997–0.998·Q_eq, engine storm-window ledger ≤2e-12·cum rain — full-window spike runs close
   ≤1e-11 — the limiter's degenerate branch never fires; the published npz recorded NET soil *drying*
   (−26 m³) under sustained ponded rain, sign-opposite to the committed-deck rerun (+47 m³) = corrupted
   booked states). **Provenance of the published run: unknown** — all parameters its npz records match
   the deck defaults; the unrecorded variables are the dt-controller knobs, comm size, and any
   uncommitted in-session code (a hotter controller through the then-open acceptance hole, or divergent
   in-session code, are both live candidates; the hole is closed by O5 either way). Original preserved
   as `tiltedv_inhouse_s1_pre_p0_corrupt.npz`; the runner npz now records the controller knobs.
   What IS real at the convergence line: (a)
   **scale-independent stiffness** with its mechanism now **measured + causally controlled** — the
   wet/dry sawtooth (known-deferred unstabilized Galerkin advection) clips fire EVERY plateau step and
   the global-rescale limiter's non-local payback perturbs the whole surface (outlet flux shaved
   21.5%/step standalone); the state never settles, so each step costs ~4–6+ Newton iterations and the
   adaptive controller's growth threshold pins dt (~5e-5–1e-4 d). The bypass control shows the limiter
   is *load-bearing*, not the cost driver (bypassing it leaves iterations unchanged and Newton then
   fails outright), and dt can be doubled at +1 iteration — a throughput cost, not a hard wall; (b) at
   the **PIDS field scale** the sawtooth is order-of-the-signal (1.3 cm clips vs ~3 mm equilibrium
   sheet): the 24×16-cell coupled run plateaus at 0.876·Q_eq with closed books, and the resolution
   control (standalone, same scale) converges with h — 1.010·Q_eq at 48×30, 0.998 at 96×60 →
   under-resolution of the mm-sheet by the oscillatory scheme → the planned upwind flux is
   **accuracy-critical for the swale regime**, not just a robustness fix; (c) **step-acceptance hardening landed (P0/O5, adversarially reviewed)**:
   stagnation verdicts (`SNORM_RELATIVE`, observed booking |F| up to 3e-3) are bookable only at the
   residual floor (`stall_accept_fnorm=3e-6`, recomputed at the returned iterate), else honestly
   rejected with the FULL state (incl. λ) restored; `snes_stol` pinned, `last_reason`/`last_fnorm`
   audit trail, Tier-1 `test_step_acceptance.py`, full suite green.
3. **Reframe (Arik 2026-06-11): convergent flow is CORE, not edge-case** — PIDS networks are installed
   along lines of topographic convergence (swales/valley lines), where the M4 features and surface
   inlets sit. Fix prioritized: **`docs/plans/2026-06-11-overland-convergent-flow-stabilization.md`**
   (wave-0 acceptance hardening — DONE P0; wave-1 **upwind-mobility edge flux** — the same monotone
   scheme class ParFlow's `OverlandKinematic` uses, read from its source; O4 local limiter deferred —
   not implicated in any books violation, and O1 removes the undershoots at the source).

## 6. Change log
| Date | Change |
|---|---|
| 2026-06-11 | **B6 P0 (convergent-flow plan Phase 0) — the published in-house B6 numbers RETRACTED + corrected; adversarially reviewed.** Instrumented committed-deck re-runs (per-accepted-step engine ledger, SNES reason/`|F|` audit): canonical V plateau **0.997–0.998·Q_eq** with storm-window books ≤2e-12·cum rain under BOTH dt-controller settings (full-window spike runs ≤1e-11) — the published "0.676·Q_eq / 20% ledger gap" is **not reproducible from the committed deck** and its provenance is unknown (all npz-recorded params match deck defaults; unrecorded variables = controller knobs / comm size / in-session code; the npz's net soil drying under ponded rain = corrupted booked states; preserved as `tiltedv_inhouse_s1_pre_p0_corrupt.npz`; limiter degenerate-branch attribution dead — it never fires). Stiffness mechanism **measured + causally controlled**: the sawtooth-and-clip state never settles (clips 430/430 plateau steps; outlet shaved 21.5%/step standalone) → persistent ~4–6-iteration steps meet the controller's growth threshold → the dt-pin (a throughput cost, not a wall: 2×dt costs +1 iteration; the limiter is load-bearing — bypassing it fails Newton outright). Field scale (162 m): clips (1.3 cm) exceed the equilibrium sheet (~3 mm) → coupled 24×16 plateau 0.876·Q_eq with closed books, healing to 1.01 at 48×30 standalone → upwind flux is accuracy-critical for the PIDS swale regime. **O5 acceptance hardening landed** (`stall_accept_fnorm=3e-6` gate on `SNORM_RELATIVE` with the norm recomputed at the returned iterate, full-state restore incl. λ on rejection, `snes_stol` pinned, audit attrs, Tier-1 tests incl. the floor-accept side; full suite green; O4 deferred — books close without it and the limiter is load-bearing). Harness + HTML now report the ENGINE ledger (the 40-pt trapz number labeled as sampling); runner npz now records controller knobs + comm size. Plan §8 = the mechanism note. |
| 2026-06-08 | Folder created for side-by-side benchmark artifacts (summary data + interactive HTMLs). Generator + first case (subsurface column) to follow. |
| 2026-06-08 | HTML layout revised: θ(z) and ψ(z) now overlay both models on shared axes, each with a dedicated error panel Δ(z)=in-house−ParFlow (animated on the slider). |
| 2026-06-08 | First case built: subsurface 1-D column. `build_comparison_column.py` runs the in-house model on the matched setup + loads ParFlow profiles → `data/subsurface__column_1d__2026-06-08.nc`; `make_comparison_html.py` → `html/subsurface__column_1d__2026-06-08.html` (self-contained, offline-verified). Accuracy-only. Agreement: RMS Δθ=9.1e-3 / max 0.050; RMS Δψ=0.054 m / max 0.34 m; both engines mass-conservative; differences localized at the wetting front (FV-vs-FEM + air-entry-K deltas). |
| 2026-06-09 | Overland (B3) standalone comparison **deferred** (see §5a) — documented the ParFlow standalone-overland limitation; Codex-reviewed diagnosis. Overland deferred to the coupled comparison. Removed the preliminary flawed-extraction overland `.nc`. Proceeding to the subsurface scenario sweep. |
| 2026-06-09 | **Subsurface non-ponding sweep done** (4 scenarios: dry/mesic × small/typical sub-Ks storms; see §5b). `../parflow/cases/column_sweep.py` (ParFlow, storm-only constant rain) + `build_comparison_sweep.py` (in-house) → 4 `.nc` + 4 self-contained HTMLs. Agreement RMS Δθ 0.002–0.005, front-localized maxima; both mass-conservative. Diagnosed + fixed two ParFlow setup traps en route: FluxConst forcing q>Ks (or a wet IC) → surface-pressure blowup (ponding needs the surface store, deferred §5a); and a storm/recession time cycle that never switched off (→ storm-only constant rain). |
| 2026-06-09 | **Coupled comparison done (B4, ponding)** — 6 scenarios (normal/extreme rain × dry/normal/wet) matching the in-house Module-3 `CoupledProblem` against ParFlow's native coupled mode (`OverlandFlow` surface store on a flat closed-base 2 m loam column; two-phase storm+recession restart). New `../parflow/cases/coupled_column.py` + `build_comparison_coupled.py` + `make_comparison_coupled_html.py` (adds a surface-depth + infiltration-partition panel) → 6 `.nc` + 6 self-contained HTMLs. RMS Δθ 0.003–0.019 (front-localized; air-entry-K delta more active here as the column saturates), both mass-conservative, peak ponding agrees to ~5–10 mm (see §5c). Resolves the deferred ponding/overland regime (§5a). Key finding: ParFlow's mass-consistent OverlandFlow surface store = `max(ψ_top, 0)` (closes the closed-column θ+pond balance to ≤2.6e-5), NOT the `−DZ/2` extraction used for standalone overland (which breaks it by exactly DZ/2 once ponded). |
| 2026-06-09 | **B5 (3-D coupled hillslope) scoped** — `B5_coupled_3d_scope.md`: the next stage extends B4 from a flat 1-D ponding column to the in-house 3-D hillslope (5×1×1 m, 5% slope, water table z=0.35, sand vs loam) with lateral overland routing + a groundwater seepage outlet, vs ParFlow's native 3-D coupled mode. Dominant risks pre-identified: the small-Manning overland conveyance (§5a, now active with real lateral routing) and the GHB→ParFlow seepage-BC mapping. Not yet executed. |
| 2026-06-11 | *(in-house numbers in this row RETRACTED by P0 the same day — see the P0 row below)* **B6 (canonical tilted-V catchment) done — findings + core-fix plan** (see §5e). New `../parflow/cases/tilted_v_catchment.py` + `build_comparison_tiltedv.py` + `make_comparison_tiltedv_html.py` + the in-house V runner. ParFlow reproduces the analytic Q_eq=4.86 m³/s **exactly** (kinematic + diffusive); the in-house engine routes the V but exposes a **20% mass-ledger gap** + **scale-independent convergent-routing stiffness** (dt pin ~1e-4 at 1.6 km AND 162 m; plateau 0.676·Q_eq is gap-corrupted). Root-caused to two separable defects (known-deferred unstabilized Galerkin kinematic advection; lax SNES snorm step-acceptance candidate) after reading both our solver and ParFlow's `overlandflow_eval{,_Kin}.c` (upwind FV + positive-part depth + single ledger; ParFlow's own original scheme had a valley-line pathology fixed by the face-centered-slope rewrite — precedent). **Reframed per Arik: PIDS installs along convergence lines → core fix**, research plan `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` (wave-0 acceptance hardening + local conservative limiter; wave-1 upwind-mobility edge flux; acceptance bars incl. a permanent PIDS-swale fixture). Also corrected §5d's erroneous "geometrically tilted in-house mesh" claim (flat box + z_b field). |
| 2026-06-10 | **B5 (3-D coupled hillslope) done** (see §5d). New `../parflow/cases/coupled_hillslope_3d.py` (3-D tilted hillslope: `OverlandFlow` top + `DirEquilRefPatch` lateral-GW head face + two-phase restart) + `build_comparison_coupled_3d.py` + `make_comparison_coupled_3d_html.py` → 2 `.nc` + 2 self-contained HTMLs (loam, sand). **Loam (overland) matches to ~1%** (overland 0.479 vs 0.474 m³, infiltration identical, peak ponding 0.72 vs 0.73 mm; the small-Manning sheet forms correctly — the B3 standalone ~0-depth failure does NOT recur when coupled). **Sand (lateral GW)**: the in-house finite kr-weighted GHB has no native ParFlow analog; the constant-head `DirEquilRefPatch` face over-drains ~4.8× (0.248 vs 0.051 m³) → documented as a **BC delta**, a finite-conductance drain-column declined (Arik). Settled two risks: small-Manning conveyance is fine when coupled; the GHB maps to a constant-head face with a known over-drain. Method finding: ParFlow `PrintOverlandSum` is unreliable as an overland volume here → overland/GW split via a **no-GW control run** difference. Characterized an **early-infiltration-transient** delta (Arik-flagged): the in-house sorptive Kirchhoff surface closure (+ air-entry cap) reaches the steady infiltration rate early and runs off sooner, vs ParFlow's coarse-cell buffer-then-choke (runoff onset 0.020 vs 0.070 d); a timing delta — end states + mass balance converge. Added an infiltration-rate + S_surf panel to the comparison HTML. |
