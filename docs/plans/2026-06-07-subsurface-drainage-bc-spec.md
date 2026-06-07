# Subsurface Darcy/head drainage BC — spec

**Date:** 2026-06-07
**Status:** SPEC → TDD build. Adds a **Darcy/head subsurface outflow** boundary so the soil matrix can
exchange water with an external reservoir (lateral groundwater outflow, soil-moisture drainage, deep
percolation) — distinct from the surface Manning outlet. Closes the "soil matrix is a closed reservoir"
gap (the BC sweep / texture sweep had NO subsurface drainage: all sides + base were natural no-flux, so
infiltrated water could only redistribute internally). Arik 2026-06-07: "spec + build the subsurface
Darcy/head drainage BC."

---

## 1. The boundary condition — general-head / Cauchy (MODFLOW GHB)

> **Codex correction 2026-06-07 (implemented):** the flux is **relative-permeability-weighted** —
> `q_n = C·kr(ψ)·(H − H_ext)`, `kr = K(ψ)/Ks` — NOT constant-C. A constant-C GHB over-drains unsaturated
> soil (it imposes the flux regardless of saturation, so the solver drives the boundary head down through
> an unresolved layer to deliver it). The kr weight makes it the physical Darcy flux through a film of
> conductivity `K(ψ)`, so it self-limits as the boundary dries. Saturated boundary ⇒ `kr=1` ⇒ the
> standard GHB below. See §6. The global balance also carries `+ clip_mass_adjust` (the degenerate-limiter
> branch). Drain boundaries must be **disjoint** from the top surface and from each other.

A leaky boundary connecting the soil to an external reservoir at head `H_ext`, with conductance `C`:

```
outward Darcy flux   q_n = C · kr(ψ) · (H − H_ext),   kr = K(ψ)/Ks,   H = ψ + z   (z = elevation, last axis)
```

- `C` [1/day] = boundary conductance (inverse resistance to the reservoir: `K_interface/L_interface`,
  or an aquifer transmissivity factor). `H_ext` [m] = external reservoir head.
- **Bidirectional** (the physically-correct reservoir exchange): drains OUT when `H > H_ext` (`q_n>0`),
  draws IN when `H < H_ext`. Reduces to **no-flow** as `C→0` and to a **Dirichlet head `H=H_ext`** as
  `C→∞`. So it spans the regional-gradient / stream / drain / fixed-water-table cases by choice of
  `(C, H_ext)` and the boundary locator (a side = lateral groundwater exchange; the base = deep
  percolation; an elevation band = a drain).

This is Darcy/head physics — NOT the surface Manning normal-depth law (which is open-channel/sheet flow,
wrong for the porous matrix).

## 2. Weak form

The Richards bulk residual `K(ψ)(∇ψ + e_g)·∇v dx` (e_g = +ẑ) drops the boundary term `−∫_∂Ω (K∇H·n) v ds`
(natural zero-flux). Since `q = −K∇H`, `K∇H·n = −q_n`, so imposing an outward flux `q_n` adds
`+∫ q_n v ds` to `F_psi` (consistent with `add_flux_bc`'s `−∫ q_in v ds`, `q_in = −q_n`). Thus:

```
F_psi  +=  C · (ψ + z − H_ext) · v_ψ · ds(locator)
```

`z = ufl.SpatialCoordinate(mesh)[zaxis]`. This is a STANDARD codim-1 exterior-facet integral on the
domain boundary (sides/base) — no codim-2 / vertex-measure machinery (unlike the surface outlet). It is
LINEAR in ψ (for constant `C, H_ext`) → smooth, robust, exact auto-Jacobian.

## 3. API + accounting

- `RichardsProblem.add_drainage_bc(locator, conductance, external_head)` and
  `CoupledProblem.add_drainage_bc(locator, conductance, external_head)`: append the Robin term to `F_psi`
  (`self._problem = None`). Reject `conductance < 0` (caller error).
- `drainage_rate()` → `Σ ∫ C(ψ+z−H_ext) ds` over all drainage boundaries (NET outward flux, m^3/day per
  unit width in 2-D; sign: + = net out). Compiled `fem.form` per BC, summed + allreduced.
- `CoupledProblem`: `last_drainage` (solved-state rate) + `cum_drainage` (`+= dt·last_drainage` in `step`,
  recorded at the SOLVED state like `last_outflow`). Global balance becomes
  **`Δtotal = cum_rain − cum_outflow(surface) − cum_drainage`**.

## 4. Validation (TDD)

1. **Analytical steady Darcy (RichardsProblem, the decisive physics gate).** A SATURATED column
   (K=Ks), Dirichlet head at the top, GHB at the base. At steady state the flux is uniform and equals
   BOTH the Darcy column flux and the GHB flux:
   `q = Ks·(H_top − H_base)/L = C·(H_base − H_ext)`  ⇒ solve `H_base = (Ks/L·H_top + C·H_ext)/(Ks/L + C)`,
   `q = Ks·C/(Ks/L·... )`. Assert the model's `drainage_rate()` and interior head match the closed form
   to ~1%. RED until `add_drainage_bc` exists.
2. **Limits.** `C=0` ⇒ `drainage_rate==0` and the solution equals the no-flow case (no drainage).
   Large `C` ⇒ the boundary head `H_boundary → H_ext` (Dirichlet limit), `q → Ks(H_top−H_ext)/L`.
3. **Conservation (CoupledProblem).** A column under rain WITH a drainage BC: `Δtotal = cum_rain −
   cum_drainage` (no surface outlet) to ~solver precision; `cum_drainage > 0`; `soil_water` decreases
   relative to the no-drainage case (water genuinely leaves). With BOTH a surface outlet and a drainage
   BC: `Δtotal = cum_rain − cum_outflow − cum_drainage`.
4. **Direction (bidirectional).** `H > H_ext` ⇒ `drainage_rate > 0` (out); `H < H_ext` (dry soil, high
   `H_ext`) ⇒ `drainage_rate < 0` (in). Sign correct both ways.
5. Keep ALL existing invariants green (the drainage term is additive; off by default).

## 5. Scope / future

- **NOW:** the bidirectional GHB (general; covers lateral groundwater outflow + drains + the Dirichlet/
  no-flow limits). Applied to any boundary via the locator.
- **Simple follow-ons (noted, not built now):** (a) **unit-gradient free drainage** at the BASE
  (`q_n = K(ψ)`, gravity-only deep percolation) — a one-liner but base-orientation-specific; (b)
  **seepage face** (outflow ONLY where saturated: `ψ ≤ 0`, `q_n ≥ 0`, complementary) — needs the
  Fischer-Burmeister NCP, for a daylighting toe/drain; (c) a **one-way drain** (clamp the GHB to
  outflow) = a special case of (b).
- The GHB conductance `C` for a real site (aquifer T, drain geometry) is a parameterization question for
  the domain/forcing modules; here `C, H_ext` are caller inputs.

## 6. Codex adversarial review (2026-06-07) — corrections applied

Static review of the as-committed constant-C GHB (commit 61418c4). Verdict **fix-then-assess**. No
sign/convention bug (the `+q_n·v·ds`, `H=ψ+z`, `e_g=+ẑ` chain is correct). Three must-fixes, all applied
before the 2-D assessment:

1. **Relative-permeability weighting (the physics fix).** The claim "the Richards solve self-limits
   unsaturated drainage to the soil's K-capacity" was **wrong**: constant-C imposes `q_n=C(H−H_ext)`
   regardless of saturation, so the solver drives the boundary head down (gradients `q/K ≈ 12` at `ψ=−0.2`,
   `≈92` at `ψ=−0.5` for loam) to deliver an unphysical flux → over-drains. Fix: `q_n = C·kr(ψ)·(H−H_ext)`,
   `kr=K(ψ)/Ks`. This is the Darcy flux through a boundary film of conductivity `K(ψ)`; it self-limits as
   the boundary dries and reduces to the standard GHB when saturated (`kr=1`). Pinned by
   `test_drainage_relative_permeability_weighting` (unsaturated `drainage_rate == C·kr·ΔH`, kr≈0.017 for
   the loam at ψ=−0.5). The saturated analytical gate is unchanged (`kr=1`).
2. **Disjointness guard.** `_build_F_psi` concatenates top + drain facets into ONE meshtags; an
   overlapping drain locator would double-tag a facet (ambiguous). Both `add_drainage_bc` now reject a
   locator overlapping the top surface or a prior drain. Pinned by `test_drainage_bc_rejects_overlapping_facets`.
3. **Balance accounting.** The exact global balance is
   `Δtotal = cum_rain − cum_outflow − cum_drainage + clip_mass_adjust` (the degenerate-limiter branch's
   mass adjustment is part of the total change; it is 0 in the normal/non-clipping case). Folded into the
   conservation test.

**Bidirectional inflow** (a high `H_ext` makes a "drain" an injector) is correct physics for a reservoir
boundary; the one-way clamp stays in §5 future. **Assessment watch-items:** very negative ψ with large
`q_drain`; sensitivity to `C`/`H_ext`; negative `drainage_rate` (= inflow); `clip_mass_adjust ≠ 0`; the
surface-film/head caveat biasing the head field near a side drain (flux is right, the head field isn't).
