# Module 4 (§E) — embedded PIDS features: implementation plan

**Date:** 2026-06-08 · **Author:** Arik + Claude · **Status:** plan (Phase 0 DONE + Codex-vetted)
**Goal:** Add the **embedded 1-D-vector feature** layer (§E) — every PIDS feature (bare/clay channel,
tunnel, french drain, catch drain, pipe) as *one* parameterized 1-D vector Γ in the 3-D Richards host,
with its own head `H_f`, contributing **conveyance + per-face sorptive exchange + storage**, sign-paired
into the existing `CoupledProblem`. Build with the standard discipline: spike → TDD → Codex/adversarial
review → three-tier assessment → Arik sign-off → commit. Architecture: `2026-06-04-...-design.md` §E.

## Two design decisions (both DECIDED)
1. **Embedding representation = CO-LOCATED interior-ridge (the §E analog of realization A).** The
   design-intended 1-D-submesh + `entity_maps` coupling is **FFCX-blocked on stock 0.10** — conclusively
   (Phase-0 spike): BOTH the parent codim-2 `ridge` cross-mesh path AND the submesh-`dx_f` path fail
   ("Integral type not supported" / codim assert). So `H_f` is a **P1 field co-located on the host**,
   pinned to 0 off Γ (like M3's d,λ), and every feature term is assembled on an **interior `ridge` (dr)
   measure** over the tagged feature edges — single-mesh, stock toolchain, built on the M3 ridge
   primitive. (File the FFCX codim-2-cross-mesh limitation upstream; migrate to the DOF-efficient submesh
   when supported — physics/tests realization-agnostic, like S.)
2. **Exchange soil leg = annulus-reference-first + KIRCHHOFF.** A fixed `K(ψ)·Ω_geom` soil leg
   under-predicts dry-soil uptake ~50× (sorptivity), and the near field **can't be meshed away** (the
   genuine sub-grid problem). Use a **Kirchhoff matric-flux-potential** soil leg `σ_soil ~ Ω_geom·[Φ(ψ_far)
   − Φ(ψ_wall)]` (reuse `VanGenuchten.kirchhoff_ufl`), validated against a **RESOLVED near-field annulus**
   reference; analytical sorptivity-clock `σ(t;S,Ks)` as the fallback if the steady Kirchhoff leg can't
   reproduce the early-time `S√t`.

## Surface-inlet + multi-layer contact — design decisions (2026-06-08, Arik)
Verified against the code: **layered soil is ALREADY supported** — `CoupledProblem` only ever calls
`soil.theta_ufl/K_ufl/kirchhoff_ufl` + `soil.Ks`, so a duck-typed z-/x-dependent soil object
(`SpatialCoordinate`+`conditional`) drops in unchanged, proven at field scale (`feasibility_2ha_layered.py`,
conservation ~1e-12). The feature inherits **local-layer** exchange for free **iff** σ is built from the
soil OBJECT's UFL methods (not frozen scalars). The open-to-surface intake is NEW code but FFCX-clean in
realization A (`d` and `H_f` both co-located on the host → a single-mesh vertex/facet term, like the
existing λ/pin machinery).

1. **Surface inlet = large-σ conductance, behaving as CAPTURE-ALL.** `q_inlet = σ_surf·(H_s − H_f)`,
   `H_s = z_b + d`, **bidirectional** (overtopping: channel spills back when `H_f > H_s`), `σ_surf` LARGE.
   PIDS sits at the convergence low point, so the overland diffusion-wave already routes surface water to
   the channel node; large σ_surf then draws ponding down faster than the soil infiltrates → the channel
   gets FIRST CLAIM and the soil gets only the infiltration-excess: the **priority cascade EMERGES from the
   physics** (no mouth-width footprint, no hard priority constraint). When the channel fills/conveyance
   saturates (`H_f→H_s`) `q_inlet→0` and the remainder infiltrates/pools TOPOGRAPHICALLY. ("Excess spills to
   evenly cover the non-PIDS soils" is a PARCEL-scale conceptual outcome, NOT a FEM constraint — the FEM
   routes by topography; confirmed Arik 2026-06-08.) Sign-paired: sink on `d`, source on `H_f`. Surface
   budget: `∂d/∂t + div(q_ovl) + λ_soil + q_inlet = rain`.
2. **Wall exchange = ISOTROPIC σ over a configurable contact depth.** `q_wall = σ_wall·(H_f − ψ_soil)`
   (Kirchhoff form), isotropic (same exchange all directions), active from the surface down to a USER
   `D_contact` (→ resolve to N mesh layers), each layer's LOCAL properties. Bidirectional (disperse `H_f>ψ`
   / drain `ψ>H_f`). **NO per-face config**: bare-vs-clay is a per-FEATURE σ magnitude (clay-lined → σ≈0,
   conveyance-only; bare/open → σ>0), not per-face.
3. **The channel DISPLACES soil** over its footprint (footprint-weighted: subtract the channel cross-section
   from the soil leg) — a small/second-order correction for narrow channels.

## Phase-0 spike result (DONE, Codex-vetted; `scratch/m4_embedding_spike{,2}.py`)
- Co-located interior `ridge` integral works: `∫_Γ 1 ds = 1.0`, exchange residual+Jacobian, storage on Γ.
- **Conveyance is EXACT with the tangential projection** `∇H_f·t̂` (manufactured `H_f=b·s`: energy matches
  `K·A·b²·L` to 6e-16); the full `∇·∇` is provably wrong (~10× off — it's cell-trace-dependent). Use
  `t̂`-projection only (`t̂` = the known feature tangent).
- Sign-paired exchange is **structurally conservative** (`q_into_feature + q_into_host = 0`). Storage
  numerically couples only Γ dofs (off-Γ P1 basis vanish on Γ; off-Γ `H_f` is pinned).

## Phased plan
**Phase 1 — the RESOLVED near-field annulus reference (the falsifiability ground truth).** A
near-field-resolved single-feature Richards run (candidate: axisymmetric `(r,z)` via an `r`-weighted
measure — cheaper than a fine 3-D tube) giving the TRUE wall→soil uptake `q(t)` (early `S√t` Philip,
late gravity), for sand/loam/silt/clay and BOTH directions (disperse `H_f>soil`; drain `soil>H_f`), on
≥2 geometries (horizontal french drain + vertical tunnel). Tier-1: reproduces Philip `S√t`. This is the
ground truth the embedded σ must match.

**Phase 2 — the co-located feature primitive (TDD).** `H_f` co-located on the host (pinned off Γ); the
interior-ridge measure `dΓ`; **conveyance** `∫_Γ K_feat·A·(∇H_f·t̂)(∇v·t̂) dΓ`; **storage**
`∫_Γ φ·A·(H_f−H_f^n)/dt·v dΓ`; the **sign-paired exchange** skeleton (`σ·(H_f−H_soil)` into `H_f`,
`−` into the host ψ on the same `dΓ`). Initially a SIMPLE σ to get the machinery + the blocked Newton
working. RED→GREEN: conveyance vs analytical 1-D Darcy (`q = K_feat·A·ΔH/L`, convergence under
refinement); storage fills to `φ·A·|Γ|`; sign-paired exchange structural conservation (tolerance-free);
the pin is conservation-neutral; reduces-to-nothing when no feature.

**Phase 3 — the sorptive exchange CLOSURE (the make-or-break, claim C-004).** The Kirchhoff soil leg
`Ω_geom·[Φ(ψ_far)−Φ(ψ_wall)]`; **`Ω_geom`** = a Peaceman well-index (resolve `r_eq` + the near-field
`ψ`-sampling rule — cell value vs `r_eq`-sampled — jointly here). **FALSIFIABILITY GATE:** the embedded σ
reproduces the Phase-1 annulus uptake with a **SINGLE a-priori `C`** across the ≥2 geometries AND both
directions — else the embedding-fidelity claim is *unsupported* (a flagged outcome, not a silent pass).
If the steady Kirchhoff leg misses the early `√t` → the transient-clock fallback (per-face time-since-wetting).

**Phase 4 — the feature taxonomy + Tier-1 (E.6).** bare/clay channel, tunnel, french drain, catch
drain/pipe as parameter sets (`σ_feat^{face,dir}`, `K_feat`); **clay sealed face** (`σ_feat=0` → zero
normal flux, structural); **surface-reaching** top-face coupled to overland (§C, `H_s=z_surf+d`); the
optional one-way device DEFERRED (asymmetric σ → the VI path, insurance not norm). Tier-1: conveyance,
bidirectional exchange vs the resolved reference, clay zero-flux, water-table interception, storage,
global+local mass balance <1e-6, plausibility.

**Phase 5 — review + assessment + commit.** Adversarial multi-agent + Codex review; Tier-2 (features in
concert with §D under a 100-yr storm, dry vs saturated antecedent: bare-vs-clay channel, tunnel draining
a perched layer, french drain intercepting a rising table); Tier-3 viz (axial flow + per-face exchange
time series; a transect of a bare channel dispersing vs a clay channel conveying) + Arik sign-off; commit.

## Risks / watch-items
- **The falsifiability gate (Phase 3) is make-or-break.** If `C` must be re-tuned per geometry/direction,
  the embedding-fidelity claim fails — report honestly (a flagged M4 outcome), don't paper over it.
- **Conveyance needs `t̂`.** Validated exact for axis-aligned (the known feature tangent); a curved/non-
  axis-aligned feature needs the per-edge tangent field (a Phase-2 detail).
- **DOF cost of co-location** — `H_f` on the full host (O(N), mostly pinned) per feature; the realization-A
  tax. Tolerable for verification / a few features; many features → the submesh (when FFCX supports it).
- **The annulus reference** (Phase 1) — axisymmetric `r`-weighted Richards vs a resolved 3-D tube; decide
  in Phase 1. It is the ground truth, so it must itself reproduce Philip `S√t`.
- **MPI deferred** (serial verification, like M1–M3); **quadrature cap** carries over (van-Genuchten/Kirchhoff).
- File the **FFCX codim-2 cross-mesh** limitation upstream (with the realization-S codim-0 bug).

## Definition of done
A co-located embedded feature (`H_f` on host, interior ridge) with **tangential conveyance + sorptive
Kirchhoff exchange + storage**, sign-paired into `CoupledProblem`, **passing the falsifiability gate**
(single a-priori `C`, ≥2 geometries, both directions) + the Tier-1 suite + conservation; the feature
taxonomy (bare/clay/tunnel/french-drain/pipe) realized as parameter sets; Codex/adversarial-reviewed;
Tier-2 + Tier-3 with Arik sign-off; committed (suite green). Then: §F domain/embedding mechanics + the
DOF-efficient submesh migration when FFCX supports codim-2 cross-mesh.
