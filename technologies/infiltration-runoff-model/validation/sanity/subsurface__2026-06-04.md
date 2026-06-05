# Sanity Check — subsurface (mixed-form Richards) — 2026-06-04

- **Module / version:** `forward-model/pids_forward/physics/{richards.py, constitutive.py}` — first complete build, `main` @ the Tier-3 commit of 2026-06-04.
- **Couplings exercised:** **alone** (subsurface solver standalone). Couplings with overland / pids-features / coupling modules are not yet built, so "in concert" is N/A for this module at this stage and will be re-run per `integration-protocol.md` when those land.
- **Run env:** WSL2 / Ubuntu 26.04, conda env `pids-fem` (DOLFINx 0.10.0, PETSc 3.25.1). 19 automated tests pass (`python -m pytest -q` → `...................`).

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

## Tier 2 — synthetic forcing (3)
- **Typical:** constant rainfall flux (0.05 m/day, below Ks) into a closed-bottom column — storage increase equals cumulative input to **rel < 1e-4**; PASS.
- **Extremes:**
  - **100-yr-style intense storm** (flux 1 m/day ≈ 4×Ks, above infiltration capacity) — the top **saturates / ponds**, mass conserved (~1e-14), `θ ∈ [θr, θs]`; PASS.
  - **Drought** (gentle sustainable evaporative outflux, no rain) — **monotone drying** + mass balance to rel < 1e-3; PASS.
- **Expected qualitative behaviours observed:** downward wetting-front advance under rainfall; surface saturation/ponding when rain exceeds infiltration capacity; monotone drying under evaporation; water redistribution under gravity with a closed bottom.

## Tier 3 — visual inspection
- **Result data:** `validation/sanity/data/subsurface__infiltration__2026-06-04.nc` (xarray/NetCDF; reproducible via `forward-model/viz/run_subsurface_sanity.py`).
- **HTML artifact:** `validation/sanity/viz/subsurface__infiltration__2026-06-04.html` (self-contained, offline, 4.95 MB; built by the independent viz subagent via `forward-model/viz/make_sanity_html.py`). Animated `θ(z)` / `ψ(z)` profiles with a time slider + a mass-balance-vs-time diagnostic + a metrics panel.
- **What to look for:** drag the time slider — the water-content profile rises to **θ_s = 0.43** at the top as the front advances over 0–0.2 day (ψ climbs toward ponding), the bottom stays near the dry initial state, and the mass-balance error stays **~1e-14** throughout.
- **Human sign-off:** _______________ (name / date / verdict — Arik)

## Residual concerns / waivers
1. **Instantaneous ponding of very dry soil** cannot be taken in one step by mass-lumped + no-`Ss` (a freshly-saturated node has a zero storage diagonal regardless of `dt`). Realistic gradual saturation works; fixes if ever needed: BC ramping, a tiny *numerical* `Ss`, or local consistent mass. *Documented in the saturation test.*
2. **Supply-limited evaporation** (a fixed evaporative flux exceeding the soil's delivery capacity) diverges as the surface dries — needs a supply-limited atmospheric / Robin BC, deferred to the forcing/vegetation module.
3. **Cross-code validation (HydroGeoSphere / the `PIDS-MIN-1` benchmark)** is optional/deferred per the 2026-06-04 decision; this report establishes internal three-tier consistency, not external validation of headline numbers.
4. **MPI > 1 not yet exercised** — a ghost `scatter_forward` was added for parallel safety, but a 2-rank conservation/retry regression is still TODO.

## Verdict: **PASS** (Tier-1 + Tier-2 automated; pending the Tier-3 human sign-off above)
