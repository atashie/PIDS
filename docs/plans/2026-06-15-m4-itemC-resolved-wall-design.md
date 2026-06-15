# Module 4 item (C): the RESOLVED-WALL regime for the embedded WellIndexExchange (design)

> **Status:** validated brainstorming design (Arik, 2026-06-15). EMPIRICAL-FIRST + honest-fence scope.
> Process constraints (non-negotiable, from the Phase-3/4 retractions): build the discriminating gate
> FIRST; pre-register tolerances BEFORE any scheme run; TDD; adversarial review (+ Codex) before merge;
> honest failure recorded as failure. Working dir for all code paths:
> `technologies/infiltration-runoff-model/forward-model/`. Project commits DIRECTLY ON MAIN.

## Goal

Extend the embedded sub-grid coupling (`pids_forward/physics/wi_exchange.py::WellIndexExchange`) from
the validated coarse-around-the-feature DEPLOYMENT regime (`h > ~5.5 r_w`) into the RESOLVED-WALL
regime (`h < ~5.5 r_w`) — fine meshes around a 5 cm feature — for the part of that regime deployment
actually uses: **a large-R_out catchment domain meshed finely around the feature**. Replace the current
blanket `ValueError` refusal with an HONEST, narrow fence that fires only at the genuinely-degenerate
corner. NOT a deployment blocker (deployment grids are coarse around the feature); this unlocks fine
meshes.

## The reframing that changes the task (verified against code, 2026-06-15)

The kickoff (`pids-m4-itemC-resolved-wall-kickoff`) framed the core hazard as **failure mode 1**: the
on-ridge Peaceman bridge `WI = 2π/ln(r_0/r_w)` goes negative-log when `r_0 < r_w`, with a REPELLING
backward-Euler fixed point at handover → runaway backflow ("never ship naive", analyzed 2026-06-10).

**That hazard no longer applies to the production driver — item A retired it.** Verified:

* The disperse rate path is `_clock_rate` (front ring at `R_f`, positive-log) in the sub-grid era and
  `_wi_ring_rate` → `_ring_bridge` (the **R_out/2** ring, positive-log) in the WI era. Both are
  prescribed rates driven by the RESOLVED far field.
* `self.WI = 2π/ln(r_0/r_w)` (`wi_exchange.py:183`) and `self.r0` feed **nothing in the rate path** —
  grep confirms `self.WI` is read only by a test assertion + a diagnostic; `self.r0` only by the
  refusal guard (`:178`) and that unused witness. `I_fill` (`:194`) is built from `2h` and `r_w`, NOT
  `r_0`. The old implicit-Ω on-ridge bridge that drove on `WI` (`scratch/m4_phase4_wi_probe.py:108,235`)
  was replaced by the resolved-ring prescribed rate (item A).

**Consequence:** the production scheme is **r_0-independent**. The guard `r_0 ≤ 1.1 r_w` is now just a
proxy for "`h ≤ 5.5 r_w` = resolved-wall" — a scope fence around an *unvalidated* regime, not a guard
against a live instability. Item C is therefore mostly a VALIDATION task (does the item-A scheme just
work at fine meshes?) plus an honest relocation of the fence — not a from-scratch redesign.

## Regime map (the two sub-regimes have very different difficulty + relevance)

| Concern | Realistic regime: large R_out + fine mesh (R=40 r_w, n≤32) | Ultra-fine corner: small box, h<~r_w |
|---|---|---|
| Mode 1 (negative-log on-ridge bridge) | MOOT — not in the production driver (item A) | MOOT — not in the production driver |
| Mode 2: `I_fill = Δθ((2h)²−r_w²)/(2r_w)` handover | `I_fill > 0`, just *small* → clock era shortens (n=16/24/32: clock fraction 4.8/2.1/1.2% of I_max). `2h ≤ r_w` unreachable until n≈142. Plausibly benign. | `2h ≤ r_w` → `I_fill ≤ 0` → clock era SKIPPED from seed. Genuinely degenerate. |
| Mode 3: R_out/2 ring read | R_out/2 = 20 r_w (deep field) → Thiem log valid. Plausibly fine. | R_out/2 = 1.5 r_w (R3) → inside the non-Thiem near-wall floor → read invalid. |

**SCOPE DECISION (Arik 2026-06-15): validate + ship the realistic regime; fence the ultra-fine corner
honestly; defer ultra-fine machinery (immersed footprint / capped bridge) as documented future work
UNLESS the gate proves it's needed.**

Box geometry (the harness `L = sqrt(π(R_out²−r_w²))`, `h = L/n`), R_out = 40 r_w = 2.0 m → L = 3.5437 m:
n=16 → h = 4.43 r_w; n=24 → 2.95 r_w; n=32 → 2.21 r_w; n=48 → 1.48 r_w. All inside resolved-wall, all
with `2h > r_w` (mode-2-safe). R3 box (L = 0.2507 m): n=8 → h = 0.63 r_w (I_fill tiny, R_out/2 = 1.5 r_w);
n=16 → h = 0.31 r_w (I_fill < 0). R3 probes BOTH degeneracies at once — the corner-characterization leg.

## Pre-registered constants (LOCK before the first scheme run — no post-hoc tuning)

* `RW_EMBEDDED_TOL = 0.10` — the resolved-wall gate pass bar (the established gate constant; this is a
  NEW regime so the deployment 0.03 *polish* does not apply).
* discrimination `≥ 0.20` — the offline-clock / fixed-drive twin must fail by at least this (reuse the
  existing structural, mesh-independent twins).

## The gate (instrument = the existing harness; references already exist)

Reuse `scratch/m4_phase4_embedded_harness.py::run_embedded` UNCHANGED. References are mesh-independent
1-D radial truths — a resolved-wall gate = the existing item-A scheme run at FINE meshes against them.

Realistic-regime legs:
* **Disperse depleting reservoir:** LOAM `RefA40` (`tests/data/m4_phase4_refA_disperse.npz`, full
  depletion) at **n = 16/24/32**. Twin = offline `sorptive_clock` (already asserted `≥0.20`).
* **History / host-control:** `RefB40` (`scratch/m4_phase4_refB40_disperse.npz`) at n = 16/24.
* **Drain:** `refD40` (`scratch/m4_phase4_refD40_drain.npz`) + `LOAM_R20`
  (`scratch/m4_phase4_drain_fresh_refs.npz`) at n = 16/24. Twin = fixed-drive PSS. Drain has NO guard
  today (`wi_exchange.py:147-166`) — validate it here.

Corner-characterization legs (to PLACE the fence honestly, not to pass):
* LOAM `RefA3` at n = 8/16 — expected to break (mode 2 + mode 3). Records WHERE the scheme falls apart.

## Characterization probe + the honest fence

* **Probe hook:** add a default-off opt-in on the scheme instance —
  `WellIndexExchange(direction, allow_resolved_wall=False)`. Default `False` ⇒ production behavior and
  `test_resolved_wall_regime_is_refused` are byte-unchanged; the probe constructs with `True` to run
  below the current fence. Forward-compatible (the final fence keeps a guard, just relocated).
* **Sweep:** push n upward per realistic leg until the scheme breaks `RW_EMBEDDED_TOL` or reaches the
  analytic degeneracy; probe the R3 corner to confirm the breakdown boundary.
* **Fence = max(analytic degeneracy boundary + margin, empirical break point).** Analytic boundary =
  `2h ≤ r_w` (I_fill) AND an R_out/2-vs-wall Thiem floor (e.g. `R_out/2` must clear the near-wall
  read-fidelity floor by a measured margin). If the scheme degrades BEFORE the analytic boundary, the
  fence moves up to the empirical break.

## Production change (TDD, after the fence location is known)

* Replace the blanket `r_0 ≤ 1.1 r_w` refusal (`wi_exchange.py:178`) with the honest fence:
  **AUTO-ALLOW the validated realistic band** (no user flag needed in deployment — the validation IS the
  warrant), refuse ONLY the degenerate corner with a precise message. `allow_resolved_wall` remains as a
  probe/escape hatch.
* Split `test_resolved_wall_regime_is_refused` → *"degenerate corner still refused"* +
  *"realistic resolved-wall allowed + tracks the ref."* Add the resolved-wall gate legs as durable tests
  (small n in tests; the full sweep lives in a scratch acceptance script + the Tier-3 record).
* **Drain fence (on evidence):** drain is geometry/mass-based (R_out, θ-mean) and likely mesh-robust;
  the sweep decides whether it needs the same fence for symmetry/honesty or validates clean to the corner.

## Review + deliverables

* Adversarial review (multi-agent + Codex), the Phase-3/4 protocol. Attack templates MUST include: "is
  the resolved-wall gate actually discriminating at high n, or does the twin pass too?", "is the fence
  set on evidence or fitted?", "is `RW_EMBEDDED_TOL` pre-registered?", "does auto-allow silently admit a
  degrading regime?". Fix-then-ship; re-run the gate after fixes.
* Binding record in `validation/sanity/`. Update the `pids-m4-itemC-resolved-wall-kickoff` memory →
  outcome. Honest-failure (if the realistic regime does NOT track at 0.10) is an acceptable, recorded
  outcome — then the fence stays at the deployment boundary and the finding is documented.

## Risk register

* **The realistic regime may NOT just-work** (the WI-era systematic grows with WI-era share, which is
  *larger* at high n: clock fraction shrinks to ~1%). If so → diagnose mechanism FIRST
  (systematic-debugging), not parameter-fiddling; if it's a genuine high-n breakdown the fence stays put
  and we record the honest characterization.
* **R_out/2 plateau revalidation:** the [0.45,0.6]·R_out plateau was deployment-mesh-validated (n=6..12);
  re-confirm the ring read fidelity at high n before trusting the WI-era driver there.
* **FEM cost:** n=32 box (4×32×32) ≈ 10–20 min/leg; n=48 ≈ 40–60 min. Scope n=16/24/32 primary, n=48 as
  a stretch. Single-thread BLAS pins on every run ([[pids-fem-blas-threading]]); disperse=cp / drain=bt
  linesearch ([[pids-fem-saturated-wall-linesearch]]); quadrature-degree cap 8
  ([[pids-fem-quadrature-degree-cap]]).

## Relevant files

`pids_forward/physics/wi_exchange.py` (guard `:178`; `__init__` `:127`; `_wi_ring_rate`/`_ring_bridge`
`:285-325`; `I_fill` `:194`); `pids_forward/physics/sorptive_closure.py` (`R0_OVER_H_P1=0.1986`);
`tests/test_wi_exchange.py` (`test_resolved_wall_regime_is_refused` + the gate legs);
`scratch/m4_phase4_embedded_harness.py` (`run_embedded`); `scratch/m4_phase4_wi_ring_derivation.py`
(the R_out/2 plateau derivation); references as listed above; the prior regime analysis
`docs/plans/2026-06-10-m4-phase4-coupled-embedding-plan.md`.
