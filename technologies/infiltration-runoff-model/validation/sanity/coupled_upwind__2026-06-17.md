# Sanity Check — `CoupledProblem` upwind overland scheme (Convergent-flow P2) — 2026-06-17

- **Module / version:** `pids_forward/physics/coupling.py` (`overland_scheme="upwind"`) +
  `pids_forward/physics/overland_edge_kernel.py`. Branch `b6-tilted-v-convergent-flow`, commits
  `0bb10be`→`f57f1ed`. Plan `docs/plans/2026-06-16-overland-convergent-flow-P2.md`; verdict parent §8.8.
- **Couplings exercised:** full-system 3-D `[ψ, d, λ]` (Richards + NCP land-surface exchange + the
  monotone upwind lateral overland on the realization-A top-facet graph) + Manning outlet.
- **Status: SET UP + RUN; Arik Tier-3 visual sign-off PENDING, with two OPEN findings below.**

## Tier 1 — automated tests (PASS)
- Full suite **174 passed** (galerkin path bit-identical). New: `tests/test_coupling_upwind.py` (12) +
  `tests/test_overland_edge_kernel.py` (8). Conservation (closed tilted box Δtotal=cum_rain 1e-13);
  kernel + coupled-level Jacobian FD-verify; downslope routing; tripwire behaviour.

## Tier 2 — synthetic forcing (coupled 3-D tilted-V, the convergent-flow regime)
Runner `forward-model/scratch/_tiltedv_diag.py` (wired for upwind via `OVERLAND_SCHEME=upwind`;
`POS_TOL` relaxes the positivity tripwire for the diagnostic so the run completes + characterizes).

```
# field-scale (162 m) and canonical (1.62 km), coarse 24x16x3, storm window:
SCALE=0.1 OVERLAND_SCHEME=upwind POS_TOL=0.05 PYTHONPATH=. python scratch/_tiltedv_diag.py 24 16 0.0625
SCALE=1.0 OVERLAND_SCHEME=upwind POS_TOL=0.05 PYTHONPATH=. python scratch/_tiltedv_diag.py 24 16 0.0625
```

| case | accepted | rejected | runtime | dt (med / max) | conservation (ext_gap) | max kink undershoot |
|------|----------|----------|---------|----------------|------------------------|---------------------|
| field 162 m (upwind) | 52 | 27 | 3.2 min | 3e-4 / ~5e-4 | ~1e-9 m³ (machine-tight) | **7.7 mm** |
| canonical 1.62 km (upwind) | 175 | 98 | **3.8 min** | 1.1e-5 / **1e-3 (DT_MAX)** | **−1e-7 m³ = −0.0000%** | **28.5 mm** |
| canonical 1.62 km (galerkin, P0 ref) | — | **60,008** | **39.5 h** | pinned ~1.5e-6 | −3e-11·cum_rain | (cm-scale clip) |

**WINS (confirmed):** the **dt-pin is lifted** (upwind reaches DT_MAX 1e-3 vs galerkin's ~1.5e-6 pin)
and the canonical run completes in **3.8 min vs galerkin's 39.5 h (~600×)**; **conservation is
machine-tight** (−0.0000% of cum rain); the sawtooth is gone.

## OPEN FINDING 1 — kink-V positivity undershoot is CM-SCALE (not sub-cm), and scales with domain
On the **idealized tilted-V** the bed has a 1-cell-wide valley KINK — the **B5b measure-zero-channel
artifact** geometry. The wet/dry front is squeezed into that single cell, so the monotone scheme's
mild-front undershoot there is **not sub-cm**: **7.7 mm (field 162 m) → 28.5 mm (canonical 1.62 km)**,
growing with domain size. (Contrast the SMOOTH tilted box, P2-D1: ~1.1–1.5 mm.) This is the positivity
manifestation of the same artifact that makes the *consistent ds-integral* read ~0.85 on the kink V
(§8.7/B5b) — and, like that, it is expected to bound on a **RESOLVED finite-width swale** (the real PIDS
geometry, where B5b's standalone consistent discharge healed to ~0.99). **The tripwire (prod tol 5 mm)
correctly REJECTS the kink-V run** — that is the design (loud, not a silent clip). **Action:** the
absolute positivity bound is a **P3 resolved-swale characterization**; do NOT read "monotone, sub-cm"
as holding on the idealized kink V. The §8.8 "~1.1–1.5mm sub-cm" claim is corrected to this
geometry-dependent statement.

## OPEN FINDING 2 — coupled Newton health is rougher than the standalone
The standalone `UpwindOverlandProblem` had **0 rejections** on the V; the COUPLED upwind has many
(field 27, canonical 98 of ~273 attempts; + 14–22 reason-4 floor-accepts). The coupling (Richards +
NCP) plus the **numerical (per-edge FD) edge Jacobian** is stiffer. It still COMPLETES + conserves +
lifts the pin, but the iteration health is the lever for the **hand-analytic edge Jacobian** (the
deferred DP-1 perf item) — likely the right next step before a clean Tier-3 sign-off / the default flip.

## Tier 3 — visual inspection (human gate) — PENDING
**Not yet generated.** Once Findings 1–2 are dispositioned (resolved-swale characterization +/- the
analytic Jacobian), build the comparison HTML for Arik:
```
# in-house upwind vs galerkin vs ParFlow (needs ~/parflow-runs/tilted_v/summaries/):
cd benchmarks && python build_comparison_tiltedv.py
python make_comparison_tiltedv_html.py data/tilted_v__canonical__2026-06-17.nc \
  html/tilted_v__canonical__upwind__2026-06-17.html
```
Inspect: outlet hydrograph Q(t) vs Q_eq (cold-start shaded); cumulative-outflow mass check; dt
distribution (pin lifted); runtime; the kink-V undershoot annotation. **Arik signs off here.**

## DISPOSITION of findings 1 & 2 (2026-06-18) — RESOLVED on the real PIDS geometry

Both findings were dispositioned (Arik-chosen "disposition first"): (a) the **hand-analytic edge
Jacobian** (`edge_flux_jacobian_dd_analytic`, FD-verified) was implemented + shipped as the production
d-d block; (b) the coupled tilted-V was re-run on a **RESOLVED finite-width swale** (`SWALE_W=324`m
flat valley floor — the real PIDS geometry, vs the 1-cell KINK) at 48×16×3 canonical.

| geometry (canonical 1.62 km, upwind) | max undershoot | rejections | conservation | runtime |
|--------------------------------------|----------------|------------|--------------|---------|
| idealized KINK V (artifact) | **28.5 mm** | 98 | −0.0000% | 3.9 min |
| RESOLVED swale W=324 m (real PIDS) | **0.5 mm** | 60 | +0.0000% | 4.2 min |
| resolved swale at PRODUCTION tol (5mm, NO override) | 0.5 mm | 60 | +0.0000% | 4.2 min — **completes cleanly (tripwire does NOT fire)** |

- **Finding 1 (positivity) — RESOLVED.** The undershoot drops **28.5mm → 0.5mm (57×)** on the resolved
  swale — SUB-MM, exactly the B5b prediction (the cm-scale was the measure-zero-channel artifact of the
  1-cell kink, not the scheme). The **production config (tripwire tol 5mm) runs the realistic swale
  WITHOUT the tripwire firing.** The idealized kink V remains an acknowledged artifact geometry.
- **Finding 2 (Newton health) — characterized + improved.** The **analytic Jacobian gives IDENTICAL
  Newton health to the numerical FD** (kink V 98 rejections either way) — confirming the FD was already
  ~exact and the kink rejections are GENUINE STIFFNESS, not Jacobian error. The analytic is shipped for
  exactness/robustness (no FD step at the sharp front) + perf (DP-1), not as a Newton-health fix. On the
  RESOLVED swale the rejections drop to 60 (the realistic geometry is less stiff); still ~560× faster
  than galerkin's 39.5 h / 60k rejections. Full P3-swale Newton tuning (controller / linesearch) is the
  residual perf item, not a correctness blocker.

**Net:** on the geometry the product actually ships into (a resolved swale), the coupled upwind scheme
is sub-mm-positive, machine-tight conserving, dt-pin-lifted, and runs cleanly at the production tripwire
— the convergent-flow fix works in the engine. Tests: 174 + analytic-Jacobian = green.

## Tier 3 — visual artifact (built 2026-06-18)
Self-contained HTML `validation/sanity/viz/coupled_upwind_tiltedv__2026-06-18.html` (kink-V vs resolved
swale, upwind): outlet hydrograph q/Q_eq, dt distribution (pin lifted), max-undershoot-over-time
(28.5mm kink vs 0.5mm swale), conservation gap. **Arik opens it + signs off.**

## Verdict: P2 COMPLETE + both Tier-3 findings DISPOSITIONED on the real (resolved-swale) geometry —
sub-mm positive, conserving, dt-lifted, clean at production tol. **Tier-3 SIGNED OFF.**

- **Human sign-off:** Arik — **APPROVED 2026-06-18** (visual review of
  `validation/sanity/viz/coupled_upwind_tiltedv__2026-06-18.html`). The opt-in coupled upwind scheme
  is accepted; default stays galerkin (DP-3 default-flip remains gated on the P3 resolved-swale
  absolute-accuracy run).
