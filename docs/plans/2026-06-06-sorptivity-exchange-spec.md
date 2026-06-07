# Sorptive soil-exchange leg — implementation spec (Kirchhoff / integral-mean conductance)

**Date:** 2026-06-06
**Status:** SPEC — **Codex-reviewed 2026-06-06 → REVISED in §10 (read §10 first; it overrides §2–§9
where they conflict).** Supersedes the dry cell-value `K(ψ_top)` soil leg in the land-surface NCP
(design §D.3). Basis: design analysis `docs/plans/2026-06-06-sorptivity-exchange-design-analysis.md`
(Codex-reviewed §8; spike §9). **SCOPE NARROWED by the review:** this is a sound **quasi-steady
coarse-cell fix for §D (land surface)**; the **§E feature leg is a SEPARATE, harder gate** (unresolved
transient near-field) — NOT proven by the §D spike. No per-face wetting-history state in the §D leg.

---

## 1. Problem recap (one line)

The exchange conductance uses the **dry cell value `K(ψ_top)`**, which under-infiltrates dry soil by ~50×
at practical (coarse-cell / embedded-feature) resolution because it ignores that `K` rises steeply
(→`Ks`) across the film as the interface saturates. Fix: use the **film-integral (Kirchhoff) mean of
`K`**, which is exact for the capillary (matric) flux and carries no time/history.

## 2. The closure

**Matric flux potential (Kirchhoff):** `Φ(ψ) = ∫_{ψ_ref}^{ψ} K(s) ds`. The steady Darcy flux across a
film of thickness `ℓ_c` between the soil interface node (head `ψ_s`) and the contacting free water (head
`h_w` = ponded depth `d` for §D, or feature internal head `H_feat − z` for §E) is, exactly for the
capillary part and linear in `Φ`:

```
q_pot = [ Φ(h_w) − Φ(ψ_s) ] / ℓ_c                      (capillary / matric)   [+ K̄ · g_n  (gravity, §3.4)]
      = (1/ℓ_c) ∫_{ψ_s}^{h_w} K(ψ) dψ
```

with `K(ψ)=Ks` for `ψ ≥ ψ_aev` (saturated; Vogel/Ippisch air-entry already caps `K_ufl`). This REPLACES
the current `q_pot = K(ψ_s)/ℓ_c · (h_w − ψ_s)`. Everything else in the supply-limited NCP is unchanged:
`g = q_pot − λ`; Fischer-Burmeister `Φ_FB(d, τ_c·g)=0`; λ sign-paired into both blocks (conservation
stays structural); reduces to `add_ponding_bc`; `kex→∞` continuity (now `ℓ_c→0`) preserved.

Equivalent "integral-mean K" framing (what the spike used, validated): `q_pot = K̄/ℓ_c · (h_w − ψ_s)`,
`K̄ = (1/(h_w−ψ_s)) ∫_{ψ_s}^{h_w} K dψ`. The two agree exactly; spec the **integral form** (it is
unambiguous about the ponded `ψ>0 → Ks` part).

## 3. Evaluation (UFL, no precompute)

### 3.1 Quadrature
`∫_{ψ_s}^{0} K(ψ) dψ` by composite Simpson on `n` sub-intervals, points `ψ_j = ψ_s·(1 − j/n)`,
`j=0..n` (UFL-expressible: `K_ufl(ψ_s·(1−j/n))`). The integrand rises steeply near `ψ=0`, so:
- **DECIDE n by convergence** against the resolved reference (§5): the 5-pt spike gave sand ratio 1.00
  but loam only 0.70 — test whether `n=8/16` (or graded points clustered near 0, e.g. `ψ_j = ψ_s·(1−(j/n)^p)`,
  p>1) lifts loam, or the loam residual is bulk coarse-mesh error (the ponding reference itself is 0.80
  of fine). Pick the smallest `n` (or grading) that reaches a target ratio (§5) without undue cost.
- The ponded part (`h_w = d > 0`): add `Ks · max(d, 0)` (saturated above `ψ=0`).

So: `q_pot = ( S_n(ψ_s) + Ks·max(d,0) ) / ℓ_c [+ gravity]`, `S_n(ψ_s) = ∫_{ψ_s}^{0} K dψ` (Simpson, n).

### 3.2 Saturated interface (`ψ_s ≥ 0`)
`S_n` → 0 contribution from `[ψ_s,0]` collapses; use `Ks·(d − ψ_s)` (all-saturated film). The Simpson
points `ψ_s·(1−j/n) ≥ 0 → K=Ks` make `S_n(ψ_s) ≈ Ks·(0−ψ_s)` automatically; verify the limit is exact
and smooth (no kink at `ψ_s=0`) for Newton.

### 3.3 Smoothness / Jacobian
`K_ufl` is C¹ (Vogel/Ippisch); the Simpson sum is a smooth function of `ψ_s` → the auto-Jacobian is
fine. Confirm no `max`/`conditional` introduces a non-smooth kink that hurts Newton (use the smoothed
`max` already in the codebase if needed).

### 3.4 Gravity term (decision)
The capillary form omits the `+1` (gravity) in `dH/dz`. The spike (capillary-only) already recovered
sand 1.00, so gravity is secondary here. SPEC: implement capillary-only first; add `+ K̄·g_n` (with
`K̄` the same film-mean, `g_n` the unit gravity component on the interface normal) only if the resolved
benchmark shows a residual gap attributable to it. Keep it OFF by default unless validation demands it.

## 4. Application

### 4.1 §D land surface (drop-in)
Replace the `q_pot` line in `CoupledProblem.__init__` (`coupling.py`). `ψ_s = self.psi` (top soil node),
`h_w = self.d`, `ℓ_c = self.ell_c`. NCP structure, lateral overland, outflow BC, limiter: UNCHANGED.

### 4.2 §E embedded features (the unified point)
The feature soil leg (architecture §E: `σ_soil = K(ψ_soil)·Ω_geom`, radial Peaceman-like) uses the SAME
integral-mean: `σ_soil = K̄·Ω_geom`, `K̄` = film/annulus-integral mean of `K` over `[ψ_soil_farfield,
ψ_wall]`. `Ω_geom` (the radial geometric factor) is unchanged. This is the one closure change features
need; it must be validated against a resolved near-field annulus (§5) BEFORE multi-feature use, and its
true gap MEASURED (not assumed = the §D coarse gap). [Module 4 work; spec'd here so §D and §E share one
primitive.]

## 5. Validation (RED gates)

- **§D sorptivity-recovery test (new, RED→GREEN):** 1-D (or 2-D-flat) dry column, heavy rain, coarse
  mesh; assert coarse conductance cumulative infiltration ≥ `r·ratio` of the `ponding-fine` reference,
  ratio target e.g. **≥0.9 for sand, ≥0.5 for loam** (numbers from `scratch/m3_sorptivity_benchmark.py`;
  finalize after the §3.1 quadrature convergence). The current `K(ψ_top)` leg FAILS this (0.02) → it is
  the RED gate.
- **Reference fidelity:** the `ponding-fine` (`RichardsProblem + add_ponding_bc`, fine mesh) reference
  must itself track **Green-Ampt / Philip `I≈S√t`** early and `→Ks` late on a 1-D column (sanity that
  "truth" is truth). Add a light Green-Ampt overlay check.
- **§E annulus reference:** a resolved 2-D/3-D near-field mesh around ONE feature (fine radial cells) =
  the feature uptake truth; the `K̄·Ω_geom` leg must reproduce it on a coarse embedding cell.

## 6. Tolerance / test reconciliation (from the spike)

The more-nonlinear `q_pot` shifts two 2-D tests; both are tolerance, NOT structural breakage:
- `test_2d_closed_conservation_is_structural`: gate `5e-13` → reads `2.47e-12` (Newton-tolerance-limited,
  λ still sign-paired). RESOLUTION: (a) keep a structural argument (sum F_d against v=1 → λ cancels; the
  `dP_pin` allocation is unchanged and conservation-neutral), and (b) set the gate to a justified
  machine-relative value (e.g. `≤ 5e-12`, or tie it to `snes_atol`). Document that it tracks Newton
  tolerance, and keep a SEPARATE assertion that the `dP_pin` allocation adds zero to the free-row sum.
- `test_2d_flat_reduces_to_1d_column`: re-baseline the 2 %→ tolerance against the corrected dynamics
  (1-D and 2-D both use the new leg, so they still match; only the absolute values moved).

## 7. Invariants to KEEP green (regression set)

§D 1-D: supply-limited mass balance, Hortonian ponding, `kex→∞` continuity, `add_ponding_bc` reduction,
recession, plausibility/determinism (all 6 passed under the spike). 2-D: overland-operator equivalence,
flat closed conservation (structural), lateral redistribution + limiter bounds, outflow magnitude +
drain-and-conserve, empty-outlet/slope guards. Full suite must return to all-green.

## 8. Open questions / risks

1. Quadrature `n` / grading to lift loam coarse beyond 0.70 (vs accepting it as bulk mesh error). Decide
   via the §3.1 convergence study.
2. Whether the gravity term (§3.4) is needed (default off).
3. §E: the annulus reference geometry (cylindrical), the far-field `ψ_soil` definition for `K̄`, feature
   interaction/superposition, and the wetting-vs-exfiltration direction (the integral-mean is
   directionally symmetric in `Φ`, which is physically right for Darcy but check the dry-exfiltration
   edge case).
4. Cost: `n+1` `K_ufl` evaluations per surface node per Newton iteration (cheap; surface dofs only).
5. The structural-conservation gate philosophy (§6) — confirm the relaxed gate still catches a
   reintroduced whole-domain `eps_diag·dx` leak (it was 1.18e-12; a `5e-12` gate would NOT — so keep the
   separate dP-allocation structural assertion as the real guard).

## 9. TDD sequence

1. RED: add the §D sorptivity-recovery test (coarse vs ponding-fine ratio) — fails at 0.02.
2. Implement the integral-mean leg (§2-§3) in `coupling.py`; pick `n` via the §3.1 convergence check.
3. GREEN: sorptivity-recovery passes; reconcile the 2 tolerances (§6) with the documented justification;
   all §7 invariants green; full suite green.
4. Decision-log + design §D.3 update (closure superseded); design §E soil-leg note.
5. (Module 4 follow-on) §E annulus reference + `K̄·Ω_geom` validation.

---

## 10. Codex spec review (2026-06-06) — INCORPORATED revisions (override §2–§9 on conflict)

The review's verdict was **fix-then-implement**; all items below are accepted. Net reframing: the
**§D land-surface fix is sound** (quasi-steady coarse-cell film correction — the surrounding Richards
still carries the wetting-front dynamics); the **§E feature leg is a SEPARATE gate, not a proven unified
primitive** (its near field is truly unresolved + transient). "Validated unified fix" was too strong —
the §9 spike validates **quasi-steady cumulative-infiltration capacity recovery for §D only.**

1. **Smoothness (MUST):** drop the raw `Ks·max(d,0)` ponded term — it's non-differentiable at `d=0`
   and reintroduces a kink the smoothed Fischer-Burmeister NCP exists to avoid. Instead define a
   **constitutive matric-flux-potential primitive `Φ(ψ)`** with a SMOOTH saturated extension
   (`Φ(ψ) = Φ(0) + Ks·ψ` for `ψ>0`, C¹ at 0), and use `q_pot = [Φ(d) − Φ(ψ_s)]/ℓ_c` directly. Document
   the **exfiltration sign** (`ψ_s > h_w` ⇒ `q_pot < 0`, soil→surface): `Φ` differences are
   directionally correct for Darcy; confirm the dry-exfiltration edge case.
2. **`Φ` as a constitutive primitive, NOT in-residual Simpson (MUST):** add `VanGenuchten.kirchhoff_ufl`
   (`Φ(ψ)=∫_{ref}^ψ K`) — evaluated by a CONVERGED rule (graded/adaptive quadrature or a tabulated/
   closed-form primitive), with a standalone convergence test (`dΦ/dψ → K(ψ)` to tolerance; `n`/grading
   chosen there, not in the residual). Cleaner Jacobian + a real convergence target. (A Gardner-exponential
   surrogate Φ is a closed-form fallback if VG quadrature is troublesome — needs its own validation.)
3. **Conservation gate (MUST — order):** do NOT relax `test_2d_closed_conservation_is_structural` (5e-13)
   first. ADD the exact structural assertion BEFORE any relaxation: assemble `F_d` against `v=1` and show
   (a) the `dP_pin` allocation contributes **zero** on the free top rows and (b) the sign-paired `λ`
   terms cancel — a tolerance-independent guard. Only then, if the Newton-limited closure genuinely needs
   it, relax the numeric gate (and keep the structural assertion as the real tripwire that still catches
   a reintroduced whole-domain `eps_diag·dx` leak, which a loose numeric gate would miss).
4. **Validation split (MUST):** separate **(i) quasi-steady capacity recovery** (the §9 cumulative test)
   from **(ii) transient sorptivity** — add TIME-RESOLVED checks of the §D leg against `ponding-fine`
   AND Philip `I≈S√t` / Green-Ampt early-time, not just final `I[-1]`. The leg must be shown to track the
   transient, not only the end state, before "sorptivity-aware" is claimed.
5. **Gravity = unresolved, not default-off (MUST):** run a gravity on/off validation matrix (the §9
   benchmark can't isolate it) before freezing the default.
6. **RED thresholds (MUST):** tighten after the §3.1/(2) quadrature convergence is settled; `loam≥0.5`
   is too weak (spike got 0.70) and a cumulative ratio can pass while the time history is wrong. Set
   thresholds from the converged resolved reference, and include the time-resolved check from (4).
7. **§E reframed as a SEPARATE acceptance gate (MUST):** write the feature leg in the radial Kirchhoff
   form **`q = Ω_geom·[Φ(ψ_far) − Φ(ψ_wall)]`** (not a vague "annulus-mean K"); define the resolved
   annulus geometry, the exact `ψ_wall` and far-field `ψ` sampling, and pass/fail for BOTH wetting and
   exfiltration. Crucially: a STATIC `Φ` form gives only quasi-steady radial capacity — if the resolved
   annulus shows it misses early-time `√t`, fall back to a **transient local uptake clock** (Philip/
   Green-Ampt/Warrick reduced model with a `t_wet` state). Do NOT claim §E solved by the §D leg.

**Revised TDD scope (proceed with §D only):** implement the `Φ`-primitive Kirchhoff leg for §D (items
1–2), add the structural-conservation assertion (3) + the quasi-steady AND transient validation (4) +
gravity matrix (5) + tightened RED (6); keep all §7 invariants green. **§E (7) is deferred to a separate
Module-4 gate** with its own annulus reference.
