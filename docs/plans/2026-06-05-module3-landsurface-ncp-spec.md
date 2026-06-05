# Module 3 land-surface exchange — supply-limited complementarity spec

**Date:** 2026-06-05
**Status:** SPEC for review (build gated on Arik's approval) — refines design §D.3/§D.5
**Authors:** Claude, with Arik; formulation per the Codex adversarial review 2026-06-05

---

## 1. Why (the defect this fixes)

Design §D.3 specifies the land-surface exchange as the smooth Robin flux

```
q_pot = k_ex · (d − ψ_top),     k_ex = K_rel(ψ_top)·K_s / ℓ_c = K(ψ_top)/ℓ_c
```

entered sign-paired into the subsurface (ψ) and surface-store (d) blocks. **TDD + two adversarial
reviews proved this law is INCOMPLETE: it is not supply-limited in the dry / unponded regime.**
With `d = 0` and an unsaturated top (`ψ_top < 0`, draining downward under gravity), `q_pot > 0`, so
the soil draws water from an *empty* surface store and `d` goes negative (observed −0.05 m on a
closed 1-D column under `rain = 0.1 < K_s = 0.25` m/day; the physical answer is `d = 0`, all rain
infiltrates). The system still conserves total mass only because the soil's over-gain offsets the
negative `d` — i.e. the budget closes around an unphysical state.

Two rejected fixes (with reasons, so we don't relitigate):
- **Post-step conservative limiter (M2 style):** clips `d<0→0` and rescales. Correct for overland's
  *tiny* wet/dry-front undershoots, but here the over-draw is *systematic*, so clipping `d→0`
  **creates** the mass the soil already absorbed → fails the 1e-6 gate. Rejected.
- **SNES-VI bounding `d≥0` alone:** stops `d<0` but does **not** stop the soil block from still
  seeing `q_pot > supply`; the active-set reaction on `d` becomes an *unpaired* source/sink and
  breaks the budget a different way. Rejected (Codex's key catch).

**Root cause:** the constraint must be posed on the actual exchanged **flux regime** (ponded vs
supply-limited), not on the storage variable `d` alone. This is the textbook surface↔subsurface
"switching" infiltration boundary (HYDRUS / ParFlow / HydroGeoSphere). Module 1's `add_ponding_bc`
already gets the 1-D case right (`pond = max(ψ_top,0)`, `rain = infiltration + d(pond)/dt`); the
coupled law must **reduce to that** in the no-lateral-flow limit. It currently does not.

---

## 2. Requirements

R1. **Supply-limited:** when capacity ≥ supply, `d = 0` and infiltration = available supply
    (= rain − lateral divergence). When supply > capacity, the excess ponds (`d > 0`) and
    infiltration = `q_pot`.
R2. **Conservative:** total water balance closes to the 1e-6 gate, *structurally* (the exchanged
    flux is a single sign-paired quantity), independent of the active set.
R3. **`d ≥ 0`** (plausibility invariant; the Tier-1 test asserts it).
R4. **Plain Newton** (smooth formulation): avoid PETSc VI — both to sidestep the M2 stiff-overland
    VI failure and to keep one solver path. A *smooth NCP regularization* achieves this.
R5. **Reduces to `add_ponding_bc`** in the no-lateral-flow limit, and to the §D.3 Robin `q_pot` in
    the ponded regime; recovers head-continuity `ψ_top → d` as `k_ex → ∞` (the existing datum guard).
R6. **Realization-agnostic:** the same exchange composes with the co-located-`d` 1-D realization and
    the top-facet-submesh realization S for 2-D/3-D lateral overland.

---

## 3. Formulation — flux complementarity (NCP)

Introduce the **actual land-surface exchange flux** `λ` [m/day, surface→subsurface] as the single
exchanged quantity. Per surface location:

```
(a) surface store:   d(d)/dt + div(q_ovl) + λ − r = 0          (overland storage + lateral flux + infiltration = rain)
(b) subsurface top:  Richards top influx = λ                    (soil RECEIVES exactly λ, sign-paired with (a))
(c) complementarity:  d ≥ 0,   g ≥ 0,   d · g = 0,             where  g := q_pot − λ   ("unused capacity")
                      q_pot = k_ex·(d − ψ_top),  k_ex = K(ψ_top)/ℓ_c
```

- `q_ovl` is the diffusion-wave Manning flux (`overland_conveyance`/`overland_residual`); **zero in
  1-D** (the surface is a point — no lateral flow), so 1-D is `(a) d(d)/dt + λ = r`.
- `r` is rainfall (− PET) onto the surface.

**Regimes (what (c) selects):**
- **Ponded** (`d > 0`): forces `g = 0` ⇒ `λ = q_pot` (full §D.3 Robin infiltration). ✓ R5
- **Supply-limited** (`g > 0`): forces `d = 0` ⇒ from (a), `λ = r − div(q_ovl)` (all available water
  infiltrates), and `g = q_pot − λ ≥ 0` certifies capacity ≥ supply. ✓ R1

### 3.1 Weak form (block residual)

Blocks `[ψ, d, λ]` (λ a surface field, co-located in 1-D / on the submesh in S). On test functions
`(w_ψ, w_d, w_λ)`, with `dS` the surface measure and `ds_top` the host top facet:

```
R_ψ = richards_bulk_residual(ψ, ψ_n, w_ψ, …)  −  λ · w_ψ · ds_top              # soil influx = λ
R_d = ((d − d_n)/dt) · w_d · dS  +  overland_flux(d) terms  +  λ · w_d · dS  −  r · w_d · dS
R_λ = Φ( d ,  q_pot − λ ) · w_λ · dS                                            # smooth NCP, §3.3
```

`λ` enters `R_ψ` with `−` (influx) and `R_d` with `+` (loss) on the **same** `λ` — conservation is
structural (see §4).

### 3.2 Eliminated-λ variant (2 blocks, optional)
`λ` can be removed using (a): `λ = r − d(d)/dt − div(q_ovl)`, giving the soil influx
`= r − d(d)/dt − div(q_ovl)` and a single NCP equation for `d`:
`Φ(d, q_pot − r + d(d)/dt + div(q_ovl)) = 0`. Fewer DOFs but the blocks are more entangled and the
soil BC carries `d(d)/dt`. **Recommendation: implement the explicit-λ form (§3.1) first** (transparent
conservation, cleaner Jacobian, closer to Codex's statement); consider elimination later as an
optimization. *(Open choice O1 — see §7.)*

### 3.3 Smooth NCP function (plain Newton)
The complementarity `a ≥ 0, b ≥ 0, a·b = 0` is written as `Φ(a,b) = 0` with a smoothed
Fischer-Burmeister function

```
Φ(a, b) = a + b − sqrt( a² + b² + 2·ε² )
```

(`Φ→0` ⟺ the complementarity holds; `ε→0` recovers it exactly; `ε > 0` is C^∞ so the Jacobian is
analytic and Newton converges). **Units:** `d` is [m], `g` is [m/day], so they are rescaled to a
common unit via a coupling timescale `τ_c` (e.g. `ℓ_c / K_s`): apply `Φ(d, τ_c·g)`. `ε` is then a
small depth scale (`~µm–mm`). The smoothing perturbs only the *switching* (a band of width `~ε`
around `d=0`), never the conservation (λ stays sign-paired). A residual `d` undershoot of `O(ε)`
(far below the systematic over-draw) is the only positivity slack; clip-free at the gate, or a tiny
M2-style clip if ever needed. *(Open choice O2: Fischer-Burmeister vs smooth-min `½(a+b−√((a−b)²+ε²))`
— equivalent; FB is the default.)*

---

## 4. Conservation (structural, independent of the active set)

Total water `W = ∫_Ω θ(ψ) dx  +  ∫_Γ d dS`. With a no-flux base and no lateral outflow:

```
dW/dt = d/dt∫θ + d/dt∫d
      = (∫_Γ λ dS)                       # soil gains exactly the top influx λ  (from R_ψ)
      + (∫_Γ r dS − ∫_Γ λ dS − outflow)  # surface balance (from R_d)
      = ∫_Γ r dS − outflow
```

The `λ` terms **cancel exactly** because the identical `λ` enters both blocks with opposite sign —
so `dW/dt = (total rain) − (total lateral outflow)` regardless of whether each node is ponded or
supply-limited, and regardless of the NCP smoothing `ε`. R2 ✓.

---

## 5. Reduction checks (must hold)

- **1-D, dry, `r < capacity`:** `g > 0 ⇒ d = 0 ⇒ λ = r` (all rain infiltrates) — identical to
  `add_ponding_bc` with `pond = 0`. R5 ✓
- **1-D, `r > capacity`:** `g = 0 ⇒ λ = q_pot`; surface `d(d)/dt = r − q_pot > 0` ponds the excess —
  matches `add_ponding_bc` accumulating `pond = max(ψ_top,0)` (and `k_ex→∞ ⇒ d = ψ_top`, the pond
  head). R5 ✓
- **`k_ex → ∞`:** `q_pot` finite ⇒ `d → ψ_top` (head continuity) — the existing §D.3 datum guard. ✓

---

## 6. TDD plan (after approval)

Tests precede code; each is realization-agnostic (drives the `CoupledProblem` API).
1. **(already RED)** closed-column mass balance under rain `< K_s` ⇒ `Δtotal = ∫rain` **AND**
   `d = 0` (supply-limited, no spurious ponding), `surface_water ≥ 0`.
2. **infiltration-excess / Hortonian:** `rain > K_s` on wet soil ⇒ `d` builds (ponds), `λ ≈ q_pot`,
   balance closes.
3. **`k_ex→∞` head continuity:** large `k_ex` ⇒ `ψ_top → d` to tolerance (datum guard).
4. **reduction:** 1-D coupled vs Module 1 `add_ponding_bc` on the same forcing — same infiltration /
   storage trajectory to tolerance.
5. **recession:** rain off ⇒ ponded `d` drains into the soil, `d→0`, monotone redistribution.
6. plausibility: `d ≥ 0`, `0 ≤ S ≤ 1`, bounded heads, no NaN; determinism.

---

## 7. Open choices flagged for approval

- **O1 — explicit `λ` (3 blocks) vs eliminated `λ` (2 blocks).** Recommend explicit-`λ` first
  (clarity + structural conservation); revisit elimination for DOF efficiency.
- **O2 — NCP function:** smoothed Fischer-Burmeister (default) vs smooth-min; and the smoothing `ε`
  + scaling `τ_c`. Recommend FB with `ε` a small depth and `τ_c = ℓ_c/K_s`.
- **O3 — positivity slack:** rely on the `O(ε)` smoothing alone, or add a tiny M2-style clip as a
  belt-and-suspenders. Recommend ε-only first; add the clip only if a test needs it.

**This refines the design's §D.3 (Robin → flux-complementarity) and corrects §D.5 (the coupled
`d≥0` is enforced by the smooth NCP, not VI and not the M2 limiter).** On approval it gets a
decision-log row and a §D amendment note.
