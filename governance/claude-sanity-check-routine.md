# Claude Sanity-Check Routine (Forward-Model Modules)

**Status:** standard, effective 2026-06-04. Applies to **every** module of the Pillar 2 forward model (`technologies/infiltration-runoff-model/forward-model/`), at **every** development step.

This extends [`verification-protocol.md`](verification-protocol.md) — which governs *facts and claims* — to govern **software / model correctness**. Both are mandatory. A module is **not "done"** until it has passed all three tiers below, **independently and in concert** with every module it interacts with, and a human has signed off on the visual inspection.

> **Core rule (evidence before assertion).** No module is reported "working" on the basis of "it ran." It is working only when the checks below have run and their *output* is on file. The reviewer's stance is **adversarial**: try to break the module, default to "not passing" when uncertain.

---

## When it runs
- After a module is first implemented (TDD — write the Tier-1 tests *before* the implementation).
- After **any** change to a module, or to a module it couples with (see the integration trigger below).
- Before a module is declared a dependency that another module builds on.

---

## The three tiers

### Tier 1 — Automated correctness tests (`pytest`)
Fast, deterministic, run on every change/commit. Four classes, all required where applicable:

1. **Analytical / benchmark agreement.** Compare against a closed-form solution, a manufactured solution (MMS), or a published benchmark. Verify the **error and its convergence order** under grid/time refinement, not just a single run. (Reference menu in the appendix.)
2. **Conservation.** Global **and** local mass balance closes: `|Δstorage − net_boundary_flux − sources| / scale < 1e-6` over the run. Energy/momentum where the module defines them.
3. **Physical-plausibility invariants** (property-based assertions): saturation ∈ [0, 1]; surface depth ≥ 0; heads/pressures bounded; **no NaN/Inf**; any **sealed/clay feature face passes zero normal flux**, and any **optional one-way device — when present** — never leaks the blocked direction (throughput ≥ −tol); monotonic responses where physics demands them.
4. **Solver & reproducibility.** Nonlinear solver converges within its iteration cap across the test matrix; results are **deterministic** with fixed seeds/inputs; refinement reduces error at the expected order.

Tier-1 lives in `forward-model/tests/` and must pass with **zero failures** before Tiers 2–3.

### Tier 2 — Synthetic-forcing sanity runs (typical **and** end-member extremes)
Force the module (and each assembly it participates in) with **synthetic** data under, at minimum:
- **Typical conditions** for the context (e.g., a moderate storm; seasonal ET).
- **End-member extremes appropriate to the module:** a **100-yr design storm** (SE-Piedmont, NOAA Atlas 14; include a **sub-hourly** intensity to exercise adaptive stepping); a **100-yr drought** (extended zero-rainfall + elevated PET); and antecedent end-members (fully saturated vs. bone-dry), steep vs. flat terrain, etc., **as the module demands**.

**Independently and in concert** is mandatory: test the module alone, then **each pair/assembly it couples to**, then the **full system**. Record which couplings were exercised.

**Acceptance (Tier 2):** under every scenario the run stays **mass-conservative and physically plausible** (Tier-1 invariants hold), the solver remains stable, and the **expected qualitative behaviour appears** — e.g., infiltration-excess vs. saturation-excess runoff partitioning, water-table mounding under recharge, plausible recession limbs, channel velocities that do/don't cross the damage threshold as forcing dictates. Surprises are written down as residual concerns, not waved away.

### Tier 3 — Visual inspection (human gate)
After Tiers 1–2 pass **and the agents are satisfied**, a **separate subagent** builds a self-contained interactive HTML per [`visualize-sanity-check-routine.md`](visualize-sanity-check-routine.md), using data **appropriate to the module/check** (hydrograph+hyetograph; saturation along a transect during a wetting front; surface-depth animation during a storm; etc.). **A human inspects and signs off.** No module clears Tier 3 on agent assertion alone.

---

## Roles (who does what)
- **Builder subagent** — implements the module test-first (Tier-1 tests precede code).
- **Skeptic/reviewer subagent** — runs the full battery **adversarially**: degenerate meshes, extreme parameters, tiny and huge `dt`, dry/saturated end-members; tries to violate conservation or the plausibility invariants. Default verdict "not passing" if anything is unexplained.
- **Visualization subagent** — builds the Tier-3 HTML (separate from the builder, to keep the visual check independent).
- **Human (Arik)** — signs off Tier 3 and any waiver of a residual concern.

---

## Standardized artifacts & naming
- **Tests:** `technologies/infiltration-runoff-model/forward-model/tests/test_<module>_*.py` (+ shared analytical solutions in `tests/analytical/`).
- **Sanity report:** `technologies/infiltration-runoff-model/validation/sanity/<module>__<YYYY-MM-DD>.md` using the template below.
- **Visualization:** `.../validation/sanity/viz/<module>__<check>__<YYYY-MM-DD>.html` (see the viz routine).
- **Result data** feeding the viz: emitted in the standardized format defined in the viz routine (decouples the visual layer from the solver).

### Sanity-report template
```
# Sanity Check — <module> — <YYYY-MM-DD>
- Module / version (commit): …
- Couplings exercised: <alone | +overland | +subsurface | full-system | …>
## Tier 1 — automated tests
- analytical/MMS: <which benchmarks; observed convergence order; pass/fail>
- conservation: <global & local mass-balance rel. error>
- plausibility invariants: <list checked; pass/fail>
- solver/reproducibility: <max iters; deterministic? refinement order>
## Tier 2 — synthetic forcing
- typical: <scenario; result; pass/fail>
- extremes: <100-yr storm / 100-yr drought / antecedent end-members; result; pass/fail>
- expected qualitative behaviours observed: <…>
## Tier 3 — visual inspection
- HTML artifact: <path> ; human sign-off: <name / date / verdict>
## Residual concerns / waivers
- <…>
## Verdict: PASS / FAIL
```

---

## Definition of done (a module)
1. Tier-1 suite green (analytical + conservation + plausibility + solver), with documented convergence order.
2. Tier-2 typical **and** extreme scenarios pass, alone and in concert with neighbours.
3. Tier-3 HTML built and **human-signed-off**.
4. Sanity report on file; residual concerns logged.
5. [`integration-protocol.md`](integration-protocol.md) run; any affected `claims-register` entries reset/re-verified.

## Integration trigger
Per [`integration-protocol.md`](integration-protocol.md): when a module changes, **re-run the sanity routine for that module and for every module it couples with** (not just the one edited). A coupling change invalidates the "in concert" evidence on both sides.

---

## Appendix — analytical / benchmark reference menu (by module)
Use the applicable rows; add as modules mature.

| Module | Analytical / benchmark checks |
|---|---|
| subsurface (Richards) | Philip infiltration (early-time series); Green–Ampt sharp front; steady hydrostatic equilibrium (zero-flux); Celia (1990) mass-balance benchmark; MMS for spatial/temporal order; Gardner-soil steady columns. |
| overland | Kinematic-wave analytical rising/falling hydrograph on a plane; diffusion-/inertial-wave against Stoker dam-break; tilted-V catchment benchmark. |
| coupling | Tilted-V and "superslab" coupled surface–subsurface benchmarks (vs. published HGS/ParFlow results); global mass balance across the interface. |
| pids-features (embedded 1-D vectors) | High-K Darcy **axial conveyance** vs. analytical (1-D Darcy / pipe); **bidirectional per-face exchange** vs. a fully-resolved 3-D reference — drains below the water table, disperses into unsaturated soil at the `ψ`-driven rate — validated with **one a-priori Peaceman connectivity factor across ≥2 geometries (horizontal french drain AND vertical tunnel) and both flow directions** (the C-004 falsifiability gate); **clay/sealed face passes zero normal flux** (structural `σ=0`, not tiny-K leakage); water-table interception; **storage** = `φ_eff·A·\|Γ\|`; **(optional, when present)** asymmetric one-way device never leaks the blocked direction. Gutters/pumps extend the same checks (detailed parametrization deferred). |
| domain / properties | Synthetic SE-Piedmont profile reproduces the specified K/porosity depth function; DEM round-trip; mesh refinement resolves a feature without global refinement. |
| forcing / vegetation | Design-storm depths match NOAA Atlas 14 targets; PET/root-uptake mass closure; drought = monotone drying. |

> Cross-code benchmarking (vs. HydroGeoSphere/HYDRUS) is **optional per the 2026-06-04 decision** (we chose the three-tier routine, not the "+cross-code" variant) — add it later for the headline coupled benchmarks if/when a licensed reference is available.
