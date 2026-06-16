# M4 item (C): the resolved-wall regime for the embedded WellIndexExchange — binding record

**Date:** 2026-06-16. **Outcome:** DONE — the realistic resolved-wall band is validated and shipped as
an honest per-direction auto-allow fence; the unvalidated corners stay refused (opt-in). **Status:**
awaiting Arik sign-off. **Commits (main):** design `480e644`, plan `30ba5e2`, probe hook `13439cf`,
sweep+results `37e790a`, R-sweep `51ca2bb`, fence `92612a9`, docstring reconcile `72c1e90`.
**Design/plan:** `docs/plans/2026-06-15-m4-itemC-resolved-wall-{design,plan}.md`.

## The reframing (verified against code, not assumed)

The kickoff framed the core hazard as the negative-log on-ridge Peaceman bridge (`WI = 2π/ln(r_0/r_w)`,
repelling backward-Euler fixed point at handover). **Item A already retired that for the production
driver.** Verified: the disperse rate path is `_clock_rate` (front ring) + `_wi_ring_rate`/`_ring_bridge`
(the R_out/2 ring) — both positive-log prescribed rates; `self.WI`/`self.r0` feed nothing in the rate
path (grep: witness + the refusal guard only); `I_fill` is built from `2h`,`r_w`, not `r_0`. So the
production scheme is **r_0-independent**, and the old `r_0 ≤ 1.1 r_w` refusal was only a scope fence
around an *unvalidated* regime — not a guard against a live instability. Item C was therefore a
VALIDATION task, not a redesign (empirical-first, Arik-chosen).

## The discriminating gate (reused) + the sweep

Instrument: the existing closed-box harness `scratch/m4_phase4_embedded_harness.py::run_embedded` and the
mesh-independent 1-D radial references, run at FINE meshes via the default-off `allow_resolved_wall=True`
probe hook. Pre-registered (locked before the first run): `RW_EMBEDDED_TOL = 0.10`, discrimination
twin `≥ 0.20`. Sweep + results: `scratch/m4_phase4_resolved_wall.py` (+ `_results_*.npz`).

Realistic regime (large R_out catchment + fine mesh) — relL2 / twin / end·Imax⁻¹:

| leg | n | h/r_w | relL2 | twin | verdict |
|---|---|---|---|---|---|
| disperse RefA40 | 16/24/32 | 4.43/2.95/2.21 | 2.7/3.5/4.3% | 26.6% | PASS |
| history RefB40 | 16/24 | 4.43/2.95 | 2.5/3.5% | 34.1% | PASS |
| drain refD40 | 16/24 | 4.43/2.95 | 4.4/4.4% | 68.2% | PASS |
| drain R20 | 16 / **24** | 2.21/1.48 | 2.7% / **DT-COLLAPSE** | 125.6% | PASS / solver-fail |

Corner / R-sweep (small R_out) — characterization, not gate legs:

| leg | n | h/r_w | relL2 | twin | note |
|---|---|---|---|---|---|
| RefA3 | 8/16 | 0.63/0.31 | 8.4/6.5% | 34.3% | PASS past the analytic 2h≤r_w mode-2 boundary |
| RefA5 | 8/16 | 1.09/0.54 | 2.7/8.5% | 43.4% | degrades with refinement |
| RefA10 | 8/16/24/36 | 2.20/1.10/0.73/0.49 | **2.4/7.7/14.8/19.1%** | 0.8% | non-discriminating; degrades with refinement |
| killmap dual-scale | 16 | 4.43 | 12.6% (end 1.145) | — | fails the 10% gate; capacity-overshoot |

## What the data established (Codex-adjudicated, three passes)

1. **The realistic deployment regime passes cleanly at fine mesh** (2.5–4.4%, twins 26–126%). The
   item-A driver "just works" there — no new machinery needed.
2. **R_out-vs-h disambiguation (the decisive R-sweep):** the fixed-h pair R10 n8 and R40 n32 are both
   h = 2.2 r_w; R10 n8 tracks at **2.4%** (better than R40's 4.3%). So R10 is NOT intrinsically bad. The
   embedded relL2 grows with **REFINEMENT**, and the refinement budget **scales with R_out** (R40,
   R_out/2 = 20 r_w, robust to h = 2.2 r_w; R10, R_out/2 = 5 r_w, degrades 2.4→19.1% as h:2.2→0.49). It
   is an R_out × h interaction; large-R_out deployment (R_out ≫ 40 r_w) has a huge budget.
3. **The analytic mode-2 fence (2h ≤ r_w) is REFUTED** — R3 n16 passes past it. Not shipped as a refusal.
4. **The drain R20 n24 (h = 1.48) failure is a solver dt-collapse**, not a tracking failure — Codex:
   do NOT encode it as a physics fence (the completed drain legs track at 2.7–4.4%).
5. **Discrimination is ENSEMBLE** — the fine-mesh RefA40 alone does not kill the capacity-clamped passive
   at the 20% margin (12.6%); RefB40 + the drain fixed-drive twins (68–126%) carry it (the existing gate
   philosophy, confirmed at fine mesh).

## The shipped honest fence (production, `pids_forward/physics/wi_exchange.py`)

Replaces the blanket `r_0 ≤ 1.1 r_w` refusal. `_RW_*` constants + `_resolved_wall_fence`:
* DEPLOYMENT (h > 5.5 r_w): always allowed (unchanged).
* RESOLVED-WALL (h ≤ 5.5 r_w): AUTO-ALLOW the VALIDATED band — disperse `R_out ≥ 40 r_w` AND `h ≥ 2.2 r_w`
  (the WI-era ring read needs a deep R_out/2); drain `R_out ≥ 20 r_w` AND `h ≥ 2.2 r_w` (mass-based —
  no ring read — so R_out-robust, validated R20+R40). Else refuse with a precise message unless
  `allow_resolved_wall=True`.
* Wall-head refusal + R_out read moved early (refuse a misconfigured feature independent of ctx; the
  fence needs R_out). Drain now measures h (cKDTree nearest off-ridge vertex).

This unlocks ~2.5× finer meshes (h down to 2.2 r_w) for real catchments (R_out ≫ 40 r_w) — the deployment
use case — and keeps the unvalidated corners behind the opt-in. The fence auto-allow is conservative
(validated-only): larger R_out is monotonically safer; finer-than-2.2-r_w is unvalidated, not known to
fail.

## Verification

* Fast suite `tests/test_wi_exchange.py` 19/19 green (4 new fence tests; obsolete blanket-refusal test
  removed). FEM gate regression green: deployment disperse n8/12 + drain refD40 n8 UNCHANGED; the durable
  `test_resolved_wall_gate_refA40_auto_allowed` (n16, DEFAULT ctor) passes the auto-allow path end-to-end
  (2.7% vs clock 26.6%). Caller audit: `feature.py`/`test_feature_sorptive.py` reference the class in
  comments only; the only non-test instantiation `viz/run_phase4_battery_data.py` (NS=8,12) is all
  deployment or validated-band → unaffected.
* Codex reviews (3 passes, the codex:rescue thread): (1) pre-tail plan/code — flagged the missing RefB40
  leg (added) + the corner-fence caveat; (2) results — verdict "narrow empirical fence only", my broad
  "R_out/2 Thiem" interpretation refuted as unproven, requested the R-sweep; (3) final code — verdict
  **SHIP** (no behavioral defect), flagged stale docstrings (reconciled, `72c1e90`).

## Deferred (documented future work, NOT shipped)

* The ultra-fine / small-R_out resolved-wall corner (immersed footprint / capped bridge) — the fence
  refuses it (opt-in); only needed if a sub-deployment use case demands it.
* The exact R_out floor between R10 (degrades) and R40 (robust) for disperse — would broaden the
  auto-allow band toward smaller R_out (sub-deployment); the disperse R20 reference does not exist yet.
* The drain solver dt-collapse robustness at very fine mesh (R20 n24, h = 1.48) — a harness/solver
  hardening item, not a tracking issue.
* The R10 refinement-degradation MECHANISM (WI-era-share × ring-read bias hypothesis) — characterized,
  not fully isolated.
