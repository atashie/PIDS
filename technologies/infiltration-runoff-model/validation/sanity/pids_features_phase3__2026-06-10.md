# Sanity Check — pids-features Phase 3 (Module 4 §E, sorptive closure + offline fidelity gate) — 2026-06-10

- **Module / version (commit):** branch `m4-phase3-sorptive-closure` (da2421d gate+closures, 2052f39
  closure machinery in `EmbeddedFeature`, 5280616 dual-scale embedding, d42432c **retraction** +
  robustness fixes, + this commit). Closures: `pids_forward/physics/sorptive_closure.py`; machinery:
  `pids_forward/physics/feature.py`. Plan: `docs/plans/2026-06-08-module4-features-plan.md`.
- **The result being signed off:** the **OFFLINE full-curve fidelity gate (C-004)** for the sorptive
  wall-exchange closure — per-soil, per-geometry, direction-aware — against the resolved Phase-1
  references (committed as fixtures `tests/data/m4_phase1{b,c}_*_refs.npz`; metric = rel-L2 over the
  `I(t)` domain-integral arrays, the `*_pen` front diagnostics are never consumed).

## What PASSED (a-priori, no knob)
- **DISPERSE (wetting into the feature wall):** clock `dI/dt = (S²/2I)·F(ζ)`, `ζ = I/(Δθ·r_w)`, with
  `F = 2ζ/ln(1+2ζ)` **derived** (cylindrical Green-Ampt + steady radial Kirchhoff log-resistance; → 1
  in the planar limit) and `S` = **a-priori Parlange sorptivity** (ψ-space integral). **C=1, zero
  fitted parameters.** Rel-L2 vs the resolved references: SAND/LOAM/SILT tunnel 0.97/0.44/0.14 %,
  annulus 4.12/2.07/2.08 % (gate ≤ 5 %). The constant-κ PLANAR clock (F=1) FAILS the same harness
  (−29 … −56 %) — the gate discriminates.
- **Extended range (Arik/Codex review follow-up):** the Carsel-Parrish texture means span only ~2
  orders of Ks; real soils span ~10. New resolved references at the extremes — GRAVEL (Ks=500 m/day)
  and TIGHTCLAY (Ks=1e-3 m/day), ~5.7 orders — the same no-knob closure holds **1.0 % on both**
  (`scratch/m4_extended_range_refs.py` + committed `.npz`).

## What is FLAGGED (honest limits)
- **DRAIN (matrix desorption into the feature):** validated FORM `F = exp(−(ζ/z0)^k)` but
  **SEMI-EMPIRICAL** (3 fitted globals; no robust a-priori desorptivity — saturated ψ=0 is a
  D=K/C→∞ singularity). Advisory, non-gating (tunnel ≤ 12.9 %, annulus ≤ 14.7 %; SAND-annulus
  gravity exception documented). Drain is a CORE use case (Arik 2026-06-09: unsaturated soil
  gravity-draining its drainable porosity into the channel); an a-priori drain closure is real
  future work.
- **CLAY:** flag-and-exclude (Δθ=0.014 at ψ_i=−1, 96 % saturated — the sorptive leg is negligible
  and the cylindrical correction over-bends; clay-lined features are conveyance-only anyway).

## What was RETRACTED (adversarial reviews, d42432c)
The **coupled embedding** claim ("a sub-grid feature inside a coarse Richards host reproduces the
resolved reference") was over-claimed **twice** and retracted after two independent reviews
(multi-agent adversarial + Codex): both the dual-scale and the conservation-first v2 schemes are
"offline clock + deposition" — the host does not control the uptake, and the fixed-far-field Phase-1
reference **cannot** validate host-controlled coupling. The vacuous coupled test was deleted; the
shell/gate machinery in `EmbeddedFeature` is marked EXPERIMENTAL in-code. Genuine coupled embedding
= **Phase-4 research** (needs an evolving-far-field reference, a host-impedance-sensitivity test,
and a closed sub-grid mass balance). Robustness fixes from the reviews are in d42432c (NaN guards in
`F_throttle`/drain leg, `seed_clock`/`advance_clock` guards, `_setup_shell` O(N²) → cKDTree).

## Tier 1 — automated tests
`tests/test_sorptive_closure_gate.py` (24: PASS-required disperse ≤5 %, planar-fails proof,
CLAY+drain flagged), `tests/test_sorptive_closure_unit.py` (3), `tests/test_feature_sorptive.py` (8:
reduces-to-the-clock both directions, structural conservation, separate per-direction accumulators,
blocked-Newton conserves). **Full suite green** (see commit), prior modules unaffected.

## Tier 3 — visual inspection (human gate)
Four self-contained offline HTMLs in `validation/sanity/viz/` (data + HTML gitignored, regenerable;
built by a separate viz subagent reading only the standardized result files):
1. **`m4_phase3_gate__2026-06-09.html`** — the gate itself: closure-vs-reference I(t) per
   soil×geometry×direction (blue disperse model/ref, red dashed planar-fails, amber drain advisory)
   with the rel-L2 table. Generators: `viz/run_phase3_gate_data.py` + `viz/make_phase3_gate_html.py`.
2. **`m4_hillslope_drain__2026-06-09.html`** — loam hillslope, buried tile drain (RESOLVED Module-3
   drainage BC on the validated `CoupledProblem`): the drain pulls a clean ~1.0 m drawdown cone;
   mass balance ~3e-13. Run `scratch/m4_hillslope_drain.py`, viz `viz/make_hillslope_drain_html.py`.
3. **`m4_hillslope_drain_clay__2026-06-09.html`** — same slope, loam over TIGHT CLAY (genuine
   two-curve van Genuchten, interface z=1.0 m): the storm perches on the clay (band up to 0.40 m),
   the base drain is starved (0.054 m², drawdown 0.085 m vs the loam case's ~1.0 m). Run
   `scratch/m4_hillslope_drain_clay.py`, viz `viz/make_hillslope_drain_clay_html.py`.
4. **`m4_hillslope_drain_dual__2026-06-10.html`** (Arik-requested) — clay-subsoil slope with TWO
   drains sharing one x-footprint: a SURFACE grate inlet and an INTERFACE tile on the clay. The tile
   captures 0.763 m² and locally eliminates the perch (0.90 m far-field vs 0.00 m at the drain
   column; final water table 0.98 vs 1.81 m); the inlet captures 0.157 m² only during the
   saturation-excess window. Storm deliberately raised 0.085→0.12 m/day × 2 d so ponding occurs (at
   0.085 the loam absorbs everything and a surface drain captures exactly 0). The drains are
   SCRIPT-LEVEL sinks (the engine GHB cannot tag interior/top facets) wired into the engine's
   `_drainage_forms`, so the balance closes structurally (3.8e-13; per-drain split re-sum 7e-16).
   Run `scratch/m4_hillslope_drain_dual.py`, viz `viz/make_hillslope_drain_dual_html.py`.

**Framing (stated in every hillslope HTML):** the hillslope figures are ILLUSTRATIONS of drainage
physics on the validated Module-3 resolved engine — they are NOT the Module-4 sub-grid embedded
feature (that coupled claim is retracted) and not new validated modules.

- **Human sign-off:** **Arik — APPROVED 2026-06-10** ("The drainage patterns are reasonable; let's
  proceed"), after the walkthrough of the gate HTML + the two requested hillslope variants
  (clay-subsoil, then dual surface+interface drains).

## Residual concerns / waivers (carried forward)
- **Coupled embedding is NOT validated** — deferred Phase-4 research (evolving-far-field reference +
  host-impedance sensitivity + closed sub-grid mass balance + the drain leg).
- **Drain closure is semi-empirical**; a-priori desorptivity = future work (core use case).
- **CLAY excluded** from the gate (negligible sorptive leg; conveyance-only).
- **SAND-annulus gravity exception** on the drain leg (gravity add-back = noted future fix).
- The dual-drain illustration's sinks live in the run script, not the engine; a first-class
  interior-drain / surface-inlet API is Module-4 §C/Phase-4 scope.

## Verdict: **PASS — Module 4 §E Phase 3 (OFFLINE gate) COMPLETE** (Tier-3 signed off by Arik 2026-06-10)
The a-priori disperse closure is production-ready for the offline gate; drain/clay limits and the
coupled-embedding retraction are recorded honestly above. Merging `m4-phase3-sorptive-closure`.
