# Sanity Check — subsurface (mixed-form Richards) — 2026-06-04

- **Module / version:** `forward-model/pids_forward/physics/{richards.py, constitutive.py}` — first complete build, `main` @ the Tier-3 commit of 2026-06-04.
- **Couplings exercised:** **alone** (subsurface solver standalone). Couplings with overland / pids-features / coupling modules are not yet built, so "in concert" is N/A for this module at this stage and will be re-run per `integration-protocol.md` when those land.
- **Run env:** WSL2 / Ubuntu 26.04, conda env `pids-fem` (DOLFINx 0.10.0, PETSc 3.25.1). **20 automated tests pass** (`python -m pytest -q`).

## Tier 1 — automated tests (16)
- **Analytical / MMS (with convergence order):**
  - **MMS spatial order ≈ 2** (P1 Lagrange, L2; consistent mass) — `test_subsurface_mms.py::test_mms_spatial_convergence_order_1d` (finest order > 1.85).
  - **MMS temporal order ≈ 1** (backward-Euler) — `::test_mms_temporal_convergence_order_1d` (finest order > 0.9).
  - **Hydrostatic equilibrium** held to **< 1e-9** (the decisive gravity-sign test; also pins gravity to the last axis in 2-D/3-D).
- **Conservation:**
  - Closed-system total water conserved to **< 1e-6** (routine criterion) in 1-D, 2-D, 3-D (actual closure ~1e-7 or tighter).
  - Open-system mass balance (Tier-2 constant-rain run): `|ΔStorage − net_flux|/scale ≈ 1.3e-14`.
- **Plausibility invariants:** `θ ∈ [θr, θs]`, no NaN/Inf, bounded heads — across all tests. **Wetting-front monotonicity** (no spurious ringing) confirmed with mass-lumped storage; consistent mass was shown to ring (−4.5e-3) → mass-lumping adopted (design B.4).
- **Solver / reproducibility:** convergence read from the PETSc SNES `getConvergedReason` (a prior false-green was found and fixed); **deterministic** (bit-identical re-run); adaptive backward-Euler cut-and-retry stepping; refinement orders confirmed above.

## Tier 2 — synthetic forcing (4)
- **Typical:** constant rainfall flux (0.05 m/day, below Ks) into a closed-bottom column — storage increase equals cumulative input to **rel < 1e-4**; PASS.
- **Extremes:**
  - **100-yr-style intense storm** (flux 1 m/day ≈ 4×Ks, above infiltration capacity) — the top **saturates / ponds**, mass conserved (~1e-14), `θ ∈ [θr, θs]`; PASS.
  - **Saturation-excess (intense rain on wet soil)** — the column saturates and the excess **ponds** (surface pressure head rises) via the vertical ponding store; converges; mass conserved across soil + pond to **rel < 1e-3**; PASS.
  - **Drought** (gentle sustainable evaporative outflux, no rain) — **monotone drying** + mass balance to rel < 1e-3; PASS.
- **Expected qualitative behaviours observed:** downward wetting-front advance under rainfall; surface saturation and **ponding (rising pressure head)** when rain exceeds infiltration capacity; monotone drying under evaporation; water redistribution under gravity with a closed bottom.

## Tier 3 — visual inspection
Four scenarios spanning antecedent wetness × event size, all in `validation/sanity/viz/` (self-contained, offline, ~4.9 MB; built by the independent viz subagent via `forward-model/viz/make_sanity_html.py`; data reproducible via `forward-model/viz/run_subsurface_sanity.py`). Each is animated `θ(z)`/`ψ(z)` profiles + a mass-balance diagnostic + a metrics panel.

| HTML (`subsurface__…__2026-06-04.html`) | Forcing × antecedent | Behaviour | Max pond | Max MB err |
|---|---|---|---|---|
| `typical-mesic` | typical storm (0.10 m/day) · mesic (ψ=−1) | infiltrates, no ponding | 0 | 1.3e-12 |
| `intense-dry` | extreme storm (2.0 m/day) · dry (ψ=−5) | sharp front, saturates, then ponds | 0.167 m | 1.5e-13 |
| `intense-wet` | extreme storm (2.0 m/day) · wet (ψ=−0.3) | **saturation-excess → ponds most** | 0.190 m | 3.8e-15 |
| `small-mesic` | small event (0.02 m/day) · mesic (ψ=−1) | shallow wetting | 0 | 6.0e-12 |

- Each run is a **storm followed by an equal-length rainless recession** (rain off in the second half). In the recession the wetting relaxes toward hydrostatic equilibrium (and the intense-dry pond drains into the still-unsaturated lower soil).
- **What to look for:** drag the time slider — for the small/typical events the column wets without ponding; for the intense events the top reaches **θ_s = 0.43** then **`ψ` climbs above 0 (ponding head)** as the surface store fills (most on wet soil), then recedes. With the impermeable (no-flux) base, the intense-wet column also saturates **from the bottom up** (water-table mounding) — see the Module-1 notes. The mass-balance error (soil + pond) stays at machine precision throughout.
- **Surface ponding** is a vertical-accumulation store only (raising the pressure head); lateral routing/runoff is the overland module.
- **Human sign-off:** **Arik Tashie — 2026-06-04 — PASS** ("this looks very reasonable"). Tier-3 visual gate cleared across all four scenarios (typical/small on mesic; intense on dry/wet), with soil-layer + field-capacity/saturation reference overlays and the storm-plus-recession timing.

## Residual concerns / waivers
1. **Saturation/ponding under realistic (rainfall) forcing now converges** via the surface ponding store (excess raises the pressure head, mass-conserving). The only remaining edge is an *instantaneous, prescribed-head* (Dirichlet ψ>0) jump onto bone-dry soil — a numerically violent idealization that lumped-mass + no-`Ss` cannot take in one step; it does not arise under flux/rainfall forcing. *Documented in the saturation test.*
2. **Supply-limited evaporation** (a fixed evaporative flux exceeding the soil's delivery capacity) diverges as the surface dries — needs a supply-limited atmospheric / Robin BC, deferred to the forcing/vegetation module.
3. **Cross-code validation (HydroGeoSphere / the `PIDS-MIN-1` benchmark)** is optional/deferred per the 2026-06-04 decision; this report establishes internal three-tier consistency, not external validation of headline numbers.
4. **MPI > 1 not yet exercised** — a ghost `scatter_forward` was added for parallel safety, but a 2-rank conservation/retry regression is still TODO.

## Verdict: **PASS** — all three tiers cleared (Tier-1 + Tier-2 automated; **Tier-3 human sign-off recorded 2026-06-04, Arik Tashie**). Module 1 (subsurface) is **DONE**; next per the build order is Module 2 (overland).
