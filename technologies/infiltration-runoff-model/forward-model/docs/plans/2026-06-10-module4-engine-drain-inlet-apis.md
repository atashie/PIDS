# Module 4 follow-on: first-class INTERIOR-DRAIN + SURFACE-INLET APIs on `CoupledProblem` (2026-06-10)

**Goal.** Promote what the signed-off dual-drain illustration (`scratch/m4_hillslope_drain_dual.py`,
commit 63145e2) did with script-level form injection into validated, TDD'd engine APIs on
`pids_forward/physics/coupling.py`. Motivation: the engine's only drain is the exterior-facet GHB
(`add_drainage_bc`), which cannot represent (a) a tile drain at an interior horizon (e.g. on a clay
interface — the agronomically standard placement) or (b) a surface inlet mid-domain (the top is the
λ-coupled NCP boundary, GHB-forbidden). Both are core PIDS features (§C surface inlet, §E tile).

## New APIs (mirroring the existing add_* conventions)

### 1) `add_interior_drain(locator, conductance_density, drain_head, *, eps_act=1e-3)`
A MODFLOW-DRN-style **volumetric, outflow-only** sink over the cells matched by `locator`
(`dmesh.locate_entities(mesh, tdim, locator)` — interior allowed, unlike the GHB):

    q_vol = conductance_density · kr(ψ) · pos(ψ + z − drain_head)        [1/day]
    pos(u) = ½(u + √(u² + eps_act²))      (C^∞ smooth max — Newton-friendly)

- `conductance_density` [1/(day·m)]: band-integrated conductance = C·(band volume) per m of head.
- **Outflow-only** (DRN semantics): an air-filled pipe never injects — distinct from the
  bidirectional GHB. `pos` leak at activation is ~C·eps_act/2, documented; eps_act exposed.
- `kr = K(ψ)/Ks` self-limits unsaturated extraction (same physics rationale as the GHB, Codex
  2026-06-07). `drain_head` = pipe invert elevation for the standard "drain to atmosphere" tile.
- Enters `F_psi` as `+ q_vol·v_ψ·dx_drain`; recorded in the drainage accounting (below).

### 2) `add_surface_inlet(locator, intake_coeff)`
A grate/catch-basin intake on the **ponded depth** over the top-surface region matched by
`locator` (x–y footprint):

    q_in = intake_coeff · d        [m/day],  intake_coeff [1/day]

- Enters `F_d` as `+ q_in·v_d·ds_top`-restricted-to-footprint; removes only surface water (d ≥ 0 is
  maintained by the existing positivity limiter, so intake ≥ 0 up to the documented NCP-floor
  residue ~1e-4 m²/day, which the tests bound).
- Linear (supply-limited) intake first; a capacity cap `min(C·d, q_max)` is a documented future knob.

### 3) Per-drain accounting
`step()` currently lumps all `_drainage_forms` into one `cum_drainage`. Extend with parallel
per-sink bookkeeping: `drainage_rates() -> list[float]` and `cum_drainages -> list[float]`
(one entry per add_* call, GHB included), keeping `drainage_rate()`/`cum_drainage` totals unchanged
(backward compatible — the dual-drain script had to reconstruct the split by hand; the engine should
own it). Recorded at the SOLVED state pre-limiter, same convention as today.

## Design decisions + the ONE spike
- **Localization.** Preferred: located-cell **meshtags** + a tagged `dx` measure shared across all
  interior drains (mirror of `_build_F_psi`'s single shared facet meshtags). RISK: the
  one-subdomain_data-per-integral-type rule — `F_psi` already integrates PLAIN `dx` (Richards bulk)
  + the vertex-rule storage measure. **Spike first** (cheap, before design lock): plain-dx +
  tagged-dx coexistence in one form on DOLFINx 0.10. Fallback (proven in the dual run): a pure-UFL
  indicator `conditional` on `SpatialCoordinate` over plain `dx` — works today, exact when band
  edges align with cell boundaries, documented inexactness otherwise.
- **Surface inlet localization** stays **indicator-based** on the existing `_ds_top` measure: the
  top facets already carry tag 1 in the SHARED facet meshtags that also routes the λ-coupling and
  GHB terms; subdividing that meshtags would destabilize validated plumbing for zero benefit.
- **Guards** (mirror existing): empty locator raises; negative coefficients raise; interior-drain
  cells need no disjointness from facet drains (different integral types) but two interior drains
  sharing cells = raise (double-sink ambiguity); inlet footprint overlapping another inlet = raise.

## TDD plan (`tests/test_engine_drains.py`, written FIRST)
1. **Interior drain conserves**: closed box, Δtotal = −cum_drainage to machine precision.
2. **Outflow-only**: fully unsaturated band ⇒ rate bounded by the documented eps leak; NEVER
   negative (no injection), contrasted against a GHB which DOES inject when external_head is high.
3. **Reduces-to-GHB**: a thin interior band hugging the base ≈ the equivalent base GHB
   (saturated conditions, tolerance band).
4. **Drains-the-perch**: mini two-layer column/slab — perch forms without the drain; the interface
   band removes it (perched-thickness contrast) — the validated dual-run physics as a regression.
5. **Surface inlet conserves + ponded-only**: with ponding it captures; bone-dry it captures only
   the bounded NCP-floor residue.
6. **Coexistence**: rain + outlet + GHB + interior drain + inlet in one problem; balance closes;
   per-drain rates sum to the total (machine).
7. **Guards**: empty locator, negative C, overlapping interior drains / inlets raise.
8. **3-D smoke**: one tet-box step with both sinks (quadrature cap honored, finite, conservative).
9. **Dual-run regression**: re-express `scratch/m4_hillslope_drain_dual.py` via the new APIs and
   reproduce the committed run (tile 0.763 m², inlet 0.157 m², balance ≤1e-9), then simplify the
   script to use the APIs (the .nc contract unchanged).

## Non-goals (recorded)
- NOT the Module-4 §E sub-grid embedded feature (retracted; Phase-4 research) — these are RESOLVED
  drains on the validated engine.
- No drain-pipe hydraulics/capacity (free outfall assumed); no inlet capacity cap yet.
- MPI stays serial-only (existing engine waiver carries).

## Order of work
1. Spike: tagged-dx coexistence (decides meshtags vs indicator).  2. Tests 1–8 red→green on the
APIs.  3. Test 9 (dual-run regression) + script simplification.  4. Tier-2 forcing sanity reuse +
docstring/accounting docs.  5. Adversarial review pass, then commit; Tier-3 only if visuals change.

## AS-BUILT deviations (2026-06-10, recorded post-review)
- **Localization = DG-0 cell indicators** (a third option, adopted without the spike): cell-sharp,
  arbitrary-locator, and provably outside the one-subdomain_data-per-integral-type rule (no
  meshtags touched at all). Quadrature-exact for the selected cells; the sink geometry is
  quantized to whole cells. Equivalence to the signed-off conditional-indicator injection is
  pinned by ``test_api_matches_script_injection``; the full 5-day dual scenario re-run through the
  APIs reproduced the committed numbers exactly (0.76340/0.15746 m², balance 3.8e-13, 332 steps).
- **Per-sink accounting naming**: shipped as ``sink_rates() -> dict`` + ``last_sinks``/``cum_sinks``
  (kind-keyed: 'ghb'/'interior_drain'/'surface_inlet'), not the planned flat ``drainage_rates()``
  list — stable grouping when sinks of different kinds are added in any order. ``drainage_rate()``
  / ``cum_drainage`` totals unchanged. Raw script-appended ``_drainage_forms`` keep working
  (auto-extend under 'ghb'; INJECT-LAST discipline documented in the constructor).
- Review (adversarial, pre-commit): fix-then-ship, no correctness blockers; applied — stale .nc
  ``note`` claim, kr-normalization + post-limiter ``drainage_rate()`` docstrings, NaN-proof
  guards, MPI caveat on the inlet, indicator-wording, the inject-last comment, and a direct
  kr-weighting pin test (the saturated analytic test alone cannot catch a dropped kr).
