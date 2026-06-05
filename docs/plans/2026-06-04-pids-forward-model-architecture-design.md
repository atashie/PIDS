# PIDS Pillar-2 Forward Hydrologic Model — Architecture Design

**Date:** 2026-06-04
**Status:** DRAFT for review (build gated on Arik's approval)
**Author:** Drafted by Claude (Claude Code) with Arik Tashie

---

## Decisions locked (design to these; do not relitigate)

- **Engine.** Build a modular FEM model on **DOLFINx/FEniCSx + PETSc** — not extend an off-the-shelf code. Install spike **PASSED** (WSL2 + conda env `pids-fem`, Ubuntu 26.04, 24 cores/31 GB): DOLFINx 0.10.0 solves Poisson to machine precision; PETSc 3.25.1 `vinewtonrsls` VI solver converges with a non-trivial active set. *DOLFINx 0.10 API note:* `fem.petsc.LinearProblem` / `NonlinearProblem` require a **keyword-only `petsc_options_prefix`**.
- **Dimensionality.** Write all solver modules in **dimension-agnostic UFL**; validate **1-D first** (Philip, Green–Ampt, steady column, reusing the 1-D probe host), then run the **same code** on 2-D and 3-D meshes for conservation + plausibility. Full 3-D is the end target; only the gravity unit vector is dimension-aware.
- **Coupling.** **Monolithic two-way implicit, SMOOTH.** Subsurface Richards + overland + embedded-feature exchange assemble into **one residual + one Jacobian**, solved by a **single smooth Newton (PETSc SNES)** per timestep. **PETSc SNES-VI (`vinewtonrsls`) is reserved** for overland depth-positivity (`d ≥ 0`) and the **optional** hard one-way device — **never** the core.
- **Feature model (CORE).** The **1D-in-3D embedded-vector** representation is the core, not an optimization: **every** feature (bare/clay channel, tunnel, french drain, catch drain, standard pipe) is the **same parameterized 1-D linear vector** embedded in the 3-D Richards continuum, carrying **(1) axial Darcy conveyance**, **(2) per-face bidirectional potential-driven exchange** `q = σ·(H_feat − H_soil)`, and **(3) storage**.
- **Head datum (LOCKED, resolves former Q6).** The feature unknown `h_feat` is a **hydraulic head** `H_f = ψ_f + z`. All exchange comparisons are on **hydraulic heads**: lateral/bottom `q = σ·(H_f − (ψ_soil + z))`, top `q_top = σ_top·(H_f − (z_surf + d))`. The axial Darcy operator then carries the gravity drive with **no extra term** (`q_axial = −K_feat·A·∇_s H_f`), so a vertical tunnel drains downward natively. The subsurface primary variable stays the Richards **pressure head** `ψ`; the conversion `H = ψ + z` is applied wherever a feature or surface head meets the subsurface. This convention is uniform across §D/§E/§F — pressure-vs-hydraulic-head mixing is a defect, not a degree of freedom.
- **Clay = sealed/impermeable face.** Compacted bentonite is treated as effectively impermeable (`K ≈ 0`, `φ_eff ≈ 0`): a **sealed face `σ = 0`**, realized as **static material zoning**, not a runtime switch.
- **Conveyance physics.** Axial transport is **Darcy flow through coarse granular fill**, `K_feat` from **sediment grain size** (Kozeny–Carman / Hazen) — NOT open-channel Manning. Standard pipes use a pipe hydraulic conductance instead.
- **Minimal-viable solver path (LOCKED, resolves former Q17).** First cut ships **BE only** (no BDF2) + `newtonls`/`bt` line search + `dt`-cut-and-retry + `preonly`/LU for small problems, `gmres`+`fieldsplit` for 3-D. BDF2, `ptc`, `cp` line search, homotopy-on-`σ`, local-inertial overland, and SUPG/entropy-viscosity are **DEFERRED-UNTIL-TRIGGERED** escape hatches, each gated by a named Tier-1/Tier-2 failure (§H), not build items.
- **Config format (LOCKED, resolves former Q30).** **TOML** (`cases/*.toml`); per-feature `σ`-tables inline in the `[[feature]]` array. All §A YAML references are superseded.
- **Scope locks.** Erosion = a **boolean in-channel** (`depth·slope → velocity/shear` vs threshold) check, **not** sediment transport. Clay = static sealed faces / material zoning. **MPI/GPU kept as a HEDGE** (optimization-loop scale TBD). The **optimization/inverse loop is DEFERRED**.
- **Forcing.** Default = **NOAA Atlas 14 design storms** (SE Piedmont). Rigor target: **research-grade / publishable**.
- **Governance action (REQUIRED before the README is used as the build contract).** `forward-model/README.md` §3, §4.4, §7, §8 are **STALE**: they still name the **retired one-way catch-valve** as the decisive primitive and a divergent Module-4 scope ("channels, tunnels, barriers, catch-valve, pumps, gutters"). They **must be rewritten** to the corrected embedded-1D-vector framing (Module 4 = embedded 1-D-vector feature layer; catch-valve demoted to optional degenerate asymmetric-σ device; pumps/gutters retained as further embedded-vector parameterizations — gutter = high top-face influx + along-path sealed disperse + a single exfiltration outlet; pump = the lone active intake/redistribution member; detailed parametrization deferred). Likewise the sanity-routine appendix (`governance/claude-sanity-check-routine.md`) must re-scope the mandatory catch-valve invariant to "optional one-way device, when present" and replace the catch-valve/pump rows with the unified embedded-vector checks (see V&V notes). This document, not the stale README, is authoritative on Module 4's identity until that rewrite lands. *(Owner: governance; tracked as R17.)*

---

## Executive summary

PIDS (Passive Infiltration & Drainage System) is a network of engineered conductive features that **passively** move surface water into and through the soil. Pillar-2 needs a physics-based, high-resolution, ultimately 3-D variably-saturated (Richards) **forward** model to assess PIDS impact on drainage (effluence) and recharge (influence).

**The corrected core concept.** An earlier framing that treated a "reverse one-way catch-valve" as THE decisive primitive was **WRONG and is RETIRED**. The unifying model, confirmed with the inventor 2026-06-04, is:

> **Every** feature — bare channel, clay channel, tunnel, french drain, catch drain, standard pipe — is the **same object**: a **parameterized 1-D linear vector embedded in the 3-D variably-saturated continuum** (mixed-dimensional 1D-in-3D). The embedding is the **core representation**, not an optional optimization. Each vector carries three behaviors:
>
> 1. **Axial conveyance** — Darcy flow *along* the vector through coarse granular fill; `K_feat` from grain size (Kozeny–Carman/Hazen); high-K. (Pipes use a pipe conductance.)
> 2. **Per-face exchange** with the abutting soil/feature — a **smooth** potential-driven flux `q = σ·(H_feat − H_soil)` (compared on **hydraulic heads**, `H = ψ + z`), with `σ` set **per face** (top/bottom/lateral) and the **engineered feature-face leg `σ_feat` set independently for receiving vs dispersing**, each in `[0, ∞)`. `σ` is a **series** combination of the feature-face conductance and the soil's `ψ`-dependent (direction-symmetric) conductance: `σ = 0` → sealed; `σ → large` → **soil-limited**, governed by soil `K(ψ)` and matric potential `ψ`. Exchange is **bidirectional**: below the water table a feature **receives** groundwater and conveys it laterally (drain); over unsaturated soil a bare feature **disperses** at the soil-`ψ`-driven rate (dry clay imbibes fast despite low `K` via high matric potential — captured natively by unsaturated Richards / van Genuchten). The hydraulic-head comparison is what makes the drain-below-water-table direction correctly signed for deep/sloping features. These feedbacks are **central** to PIDS.
> 3. **Storage** — `φ_eff · cross-section (w·d) · length`.

Clay bottoms = compacted bentonite, **effectively impermeable** (sealed face, `σ = 0`). Tunnels = always-vertical high-K conduits draining to deeper soils. The feature taxonomy is just **parameterizations of the one vector**. One-way devices are the **degenerate asymmetric-σ limit** (receive>0, disperse=0, or vice-versa) — **supported but not the norm**.

**Pumps and gutters** (named in the README's Module 4) are **in scope as additional parameterizations of the same embedded-vector class** — not a separate mechanism. A **gutter** collects surface water (top-face receive `σ_top → ∞`) and conveys it axially with **no along-path effluent** (`σ_disperse = 0` along its length) to a single designated **exfiltration outlet node** (where it disperses to soil/surface). A **pump** is the one **active** member: it extracts at a point (intake — receive only, no effluence) and redistributes laterally to a point or series of points (outlets — effluence only, no influx), carrying a **prescribed/active flux (or head boost)** rather than a purely passive potential-driven exchange. Both remain 1-D vectors with per-node/per-face receive/disperse conductances (several set to 0) plus, for the pump, an active source/sink term — they fold into the abstraction without changing it. **Detailed parametrization is deferred** (not needed to build Modules 1–4); the taxonomy (E.3) extends to gutters and pumps as further parameter sets, and the governance appendix's pump/gutter checks are **re-scoped to the unified embedded-vector checks** rather than retired.

**Solver consequence.** With fixed/bidirectional `σ` the exchange is **smooth**, so the headline coupled solve is a **smooth monolithic Newton problem**. PETSc VI (`vinewtonrsls`) is reserved for **overland depth-positivity** and the **optional** one-way device, not the core. (The 1-D probe `scratch/catchvalve_probe.py` showed even a hard one-way exchange is tractable as an implicit hook — now *insurance* for the optional asymmetric case; non-iterated operator-splitting blew up and is disqualified for any in-solver exchange.)

### Module DAG (build/test order, each node gated by the three-tier sanity routine)

```
F domain/mesh ──┬─> B subsurface ──┬─> D coupling ──> E pids-features ──> I driver/IO
                │                   │                        ^
G forcing/veg ──┴─> C overland ────┘                        │
                                    (E consumes B host + D exchange pattern)

H cross-cutting-numerics : HORIZONTAL — provides solve/ (SNES, SNES-VI, fieldsplit)
                           + adaptive backward-Euler dt controller to B, C, D, E.
A architecture-skeleton  : HORIZONTAL — TermModule contract, blocked State,
                           monolithic assembly, result contract, repo layout.
```

Build **bottom-up**: **F + G first** (no solver — pure data + analytical property/forcing checks) → **B** (1-D Richards on the probe host; Philip/Green–Ampt/Celia) → **C** (overland; kinematic-wave/Stoker) → **D** (couples B + C; tilted-V/superslab, interface mass balance) → **E** (embedded features on the coupled host; Darcy conveyance, bidirectional exchange vs a locally-resolved 3-D reference, clay zero-flux) → **I** (driver/IO + result contract) wires the whole system. Each node clears Tier-1/2/3 before its dependents build on it; the **same** dimension-agnostic UFL is validated 1-D first, then run on 2-D/3-D meshes. **Module 1 (subsurface) is next** after the F+G data layer.

---

## A. Architecture skeleton: module/term contract, mixed-dimensional interfaces, monolithic assembly

This is the *frame* into which all physics drops. **Organizing principle: every physics and feature module contributes weak-form residual terms to one shared nonlinear problem.** The Jacobian is obtained by UFL automatic differentiation (`ufl.derivative`), and PETSc SNES solves the assembled residual + Jacobian **monolithically** each timestep. **No module owns a solve; modules own *terms* and *named fields*.** This is what makes the locked monolithic two-way-implicit coupling buildable bottom-up.

### A.1 The module/term contract

Every physics or feature module implements one `Protocol`. It is handed the shared `State` (current iterate as `ufl` coefficients) and the relevant test functions, and returns a UFL form that is *added* to the global residual. It does **not** see the solver, the timestepper, or other modules' internals — only named fields it declares as inputs.

```python
class TermModule(Protocol):
    name: str
    provides: tuple[str, ...]          # field names this module governs, e.g. ("h",)
    consumes: tuple[str, ...]          # fields it reads, e.g. ("theta_props", "h_feat")
    measures: tuple[str, ...]          # integration domains: "dx", "ds", "dS", "dx_feat"

    def register(self, reg: EntityRegistry) -> None:
        """Bind MeshTags / submeshes to this module's measures (materials, sealed faces, feature lines)."""

    def residual(self, s: State, w: TestFunctions) -> ufl.Form:
        """Return F_i(s; w) summed into the global residual. Jacobian via ufl.derivative(F, s.u)."""

    def fields(self) -> dict[str, FieldSpec]:
        """Named, unit-tagged accessors (subspace index, element, units) for IO/viz/coupling."""
```

Three module *kinds* satisfy this one contract:

- **(a) Volumetric host physics** — subsurface (3-D Richards, §B) integrates over `dx` on the host mesh; overland (§C) integrates over the surface measure `ds` (or a 2-D surface submesh). Both are dimension-agnostic UFL; only the gravity unit vector `e_g` is dimension-aware.
- **(b) Embedded 1-D features** (§E) integrate axial-conveyance + storage over a **1-D measure `dx_feat`** on a feature submesh, and contribute **per-face smooth exchange** `q = σ·(H_feat − H_soil)` (hydraulic heads) where the feature line meets the host. The exchange term writes into *both* the host residual (a line/tube source on `dx`) and the feature residual (`dx_feat`) — giving the bidirectional feedback natively.
- **(c) Coupling** (§D) is itself a `TermModule` whose `residual` is the surface↔subsurface exchange flux; it consumes `h` and the surface field and contributes **equal-and-opposite** terms to each, so global mass balance is **structural, not enforced post hoc**.

### A.2 Shared state and the monolithic function space

One **blocked** function space carries all unknowns for a single Newton solve:

| Block | Field | Space (typical) | Mesh |
|---|---|---|---|
| 0 | subsurface head `h` (Richards, mixed-form ψ) | CG-1 / CG-2 | host (1/2/3-D) |
| 1 | surface depth/head `d` (overland) | CG-1 | top boundary facets (working default; standalone 2-D submesh optional — see Q7) |
| 2..k | feature heads `h_feat^(j)` | CG-1 on feature submesh | 1-D submeshes |

**Recommendation: PETSc `nest` (block) layout, not a single `MixedElement`.** The blocks live on *different meshes* (3-D host, 2-D surface, 1-D feature lines) with different DOF counts, so a single `MixedElement` on one mesh cannot represent them; DOLFINx multi-mesh / submesh assembly produces a block system naturally. A nest also lets the overland/feature SNES-VI bounds (`d ≥ 0`; optional one-way device) apply to *only* their blocks, lets PETSc `fieldsplit` precondition the stiff Richards block separately, and keeps the MPI/GPU hedge tractable. The 1-D feature DOFs are exactly a CG function space on a DOLFINx **submesh** of embedded edges; the exchange term couples `h_feat` to host `h` via an interpolation/restriction operator evaluated at the feature entities (embedding mechanics finalized in §F).

> **Block-1 mesh realization is gated on Q7** (top boundary facets vs explicit 2-D submesh). The whole design is built on the **working default = top boundary facets** (used firmly in §C/§D), but the `TermModule`/`State` contract is written to tolerate **either** realization (a facet-restricted field or a co-dim-1 submesh) so the M0 contract freeze does not pre-bind it.

> **R1 conditionality (read before freezing any contract).** The blocked-nest layout here, the §F submesh embedding, and the §H three-field fieldsplit **all assume** cross-mesh `ufl.derivative` works in DOLFINx 0.10 — the dominant build risk (R1). The M0 **cross-mesh `ufl.derivative` spike is a STANDALONE blocking gate** that precedes the rest of M0's contract-freezing: **no contract is frozen until the spike passes or selects a fallback** — (a) host-mesh co-dim line restriction, or (b) immersed line-source-for-all. If the fallback is *immersed-for-all*, the conforming-submesh path in §F and the tunnel "prefer conforming" preference are dropped.

```python
@dataclass
class State:
    u: BlockFunction              # current iterate, blocks [h, d, h_feat...]
    u_n: BlockFunction            # previous timestep (backward-Euler)
    t: float; dt: Constant
    props: PropertyBundle         # van Genuchten fields, K_feat, sigma maps (read-only in residual)
    def field(self, name: str) -> ufl.Coefficient: ...
```

### A.3 Monolithic assembly + time loop (pseudocode)

```python
modules = [Subsurface(), Overland(), Coupling(), PidsFeatures(), ...]  # ordered by DAG
for m in modules: m.register(registry)            # bind MeshTags / submeshes / measures

def global_residual(s, w):
    F = 0
    for m in modules:
        F += m.residual(s, w)                     # host dx + surface ds + feature dx_feat + exchange
    return F                                       # ONE residual over the block space

F = global_residual(state, test)
J = ufl.derivative(F, state.u)                     # UFL auto-diff -> full coupled Jacobian (incl. exchange blocks)

solver = make_snes(F, J, bounds=collect_vi_bounds(modules))  # VI only where d>=0 / one-way device
while t < t_end:
    dt = controller.propose(t)                     # adaptive sub-hourly -> daily (§H)
    state.u_n.assign(state.u)
    ok, iters = solver.solve(state.u)              # single monolithic Newton/SNES-VI solve
    if not ok: dt = controller.shrink(); continue  # reject & retry; restore ALL blocks from u_n
    record_diagnostics(state, iters); io.maybe_write(state, t)
    t += dt; controller.accept(iters)
```

### A.4 Standardized in-code data interfaces

Inputs the skeleton standardizes (produced by §F/§G, consumed by §B–§E):

- **Mesh + embedded-entity sets:** host `dolfinx.Mesh` (1/2/3-D), surface entity set, a `feature_edges` set defining each 1-D feature submesh.
- **MeshTags:** `material_tags` (cell → soil zone), `feature_tags` (edge → feature id), `face_tags` (feature facet → top/bottom/lateral), `sealed_tags` (clay σ=0 faces), plus `inlet`/`outlet`, `is_vertical` (tunnel).
- **Property fields:** van Genuchten (`θ_s, θ_r, α, n, K_s`) per material; `K_feat` from grain-size via Kozeny–Carman/Hazen (§F); effective porosity.
- **Per-feature `FeatureRecord`** — the parameterization of the one vector:

```python
@dataclass
class FeatureRecord:
    fid: int; kind: str                  # bare|clay|tunnel|french|catch|pipe
    width: float; depth: float           # -> storage = phi_eff * width*depth * length
    orientation: str                     # axial direction / "vertical" for tunnels
    K_feat: float; phi_eff: float
    sigma_recv: dict[str, float]         # per-face {top,bottom,lateral} FEATURE-FACE receive cond. sigma_feat [0, inf)
    sigma_disp: dict[str, float]         # per-face FEATURE-FACE disperse cond.; asymmetry lives ONLY here, not in the soil leg
    # soil leg sigma_soil = K(psi)*Omega_geom is direction-symmetric (added in series by E/F, not stored here)
    # clay: sigma_recv[bottom]=sigma_disp[bottom]=0 (sealed); pipe: only inlet/outlet open
```

- **Forcing series** (§G): rainfall hyetograph + PET as time-interpolated `Constant`s.
- **BC specs:** Dirichlet/Neumann on host boundary; water-table / free-drainage bottom.

### A.5 Fixed field-name vocabulary (used everywhere downstream)

| Symbol | Meaning | Block |
|---|---|---|
| `h` / `ψ` | subsurface pressure head (Richards primary variable) | 0 |
| `H = ψ + z` | subsurface hydraulic head (derived from block 0) | derived |
| `d` | overland ponding depth (`≥ 0`); surface hydraulic head `H_s = z_b + d` | 1 |
| `h_feat` (`H_f`) | feature **hydraulic** head on a 1-D submesh, arc-length `s` (`H_f = ψ_f + z`) | 2..k |
| `q_ls` | land-surface exchange flux (overland ↔ subsurface) | coupling |
| `q_exch` / `q_feat` | per-face feature↔soil exchange flux `σ·(H_f − H_soil)` (hydraulic heads) | features |
| `q_top` | surface-reaching feature top-face flux `σ_top·(H_f − H_s)` | features↔overland |
| `axial_flux` / `q_axial` | along-vector Darcy/pipe conveyance `−K_feat·A·∇_s H_f` | features |
| `σ_face` (`σ_feat`) | per-face engineered feature-face conductance `[0, ∞)`; per-direction (receive/disperse) | features |
| `σ_soil` | direction-symmetric soil leg `K(ψ)·Ω_geom` | features/domain |
| `Ω_geom` | per-host-cell geometric connectivity factor (Peaceman well-index analogue) | domain |
| `e_g` | **global gravity unit vector** (sole dimension-aware export) | all |
| `t̂` | **feature tangent unit vector** (axial direction; `∇_s = t̂·∇`) | features |
| `K` (scalar) / `K̄` (tensor) | hydraulic conductivity; tensor `K̄ = diag(K_h,…,K_v)` path carried where anisotropy is needed (R12/Q19) | subsurface/domain |

> **Notation convention:** `h`, `ψ`, and `ψ_soil` all denote subsurface **pressure** head; `H_soil = ψ_soil + z` its hydraulic head; `h_feat`/`H_f` the feature **hydraulic** head; the two unknowns are kept as **distinct blocks** throughout (never folded together). **One gravity glyph `e_g`** for the global vertical (the former alias `e_z` is retired); the **distinct** feature-tangent glyph is `t̂`, so axial conveyance reads `∇_s = t̂·∇` without colliding with `e_g`.

---

## B. Module 1 — Subsurface (3-D Richards), dimension-agnostic

### B.1 Governing equation (mixed form)

The host continuum is variably-saturated single-phase flow governed by **Richards' equation in mixed (θ–h) form**:

```
d(theta(h))/dt + div(q) = -Q_ex        q = -K(h) ( grad h + e_g )
```

with `h` the pressure head [m] as **primary variable**, `θ` volumetric water content [-], `K(h)` unsaturated hydraulic conductivity [m/s], `e_g` the (global) gravity unit vector. Hydraulic head `H = h + z`; flux `q` positive along `−grad H`. `Q_ex` is the **per-face exchange sink/source** supplied by §E (pids-features) and the surface coupling term from §D — Module 1 only **exposes the slots** and the soil-side state (`h, θ, K(h)`) that make those exchanges soil-limited; it does **not** specify the feature side.

**Conductivity is scalar by default, tensor-ready.** `K(h)` above is scalar. SE-Piedmont layering may require a **diagonal tensor `K̄ = diag(K_h,…,K_v)`** (R12/Q19); the UFL is therefore written so `K(h)` can be a tensor field from the outset — the gravity term becomes `−K̄·(grad h + e_g)` and the Mualem relative-permeability scaling multiplies the saturated tensor `K̄_s·k_r(Se)`. The §F property generator and the MMS order/conservation gates both carry the tensor path so enabling anisotropy is a property-field swap, not a residual rewrite.

**Why mixed form (modified-Picard, Celia et al. 1990), not head-based.** The head-based form `C(h) dh/dt + div q = 0`, `C = dθ/dh`, conserves mass only as well as the chain-rule term reproduces the true `dθ/dt`; near sharp fronts `C(h)` is strongly nonlinear and head-based discretization leaks mass (documented O(10%) errors). The **modified-Picard/mixed** scheme linearizes the *storage* term about the current iterate while evaluating `θ` directly:

```
theta(h^{k+1}) ≈ theta(h^k) + C(h^k)(h^{k+1} - h^k)
```

so the time term is `(theta(h^k) - theta_n)/dt + C(h^k)(h^{k+1}-h^k)/dt`. At convergence the `C`-terms cancel and storage is governed by the *exact* `θ(h)` at `h_n, h_{n+1}` — **global mass balance holds to solver tolerance regardless of timestep**, exactly what the Tier-1 gate (`|ΔStorage − net_flux − sources|/scale < 1e-6`) requires. **The ENTIRE storage residual `(θ(h^{k+1}) − θ_n)/dt` is mass-lumped on one nodal quadrature rule — not only the `C(h^k)` increment.** Lumping a consistent-mass `θ` against a lumped `C` would reintroduce the O(10%) leak the mixed form exists to avoid; the conservation guarantee holds at convergence only when `θ`, `θ_n`, and `C` are evaluated on the **same** nodal quadrature. The 1-D probe `scratch/catchvalve_probe.py` already runs this mixed form (FV); Module 1 reuses its `theta − theta_n` storage residual and Newton loop, ported to FEM, **dropping the retired catch-valve term**.

### B.2 Constitutive closures — van Genuchten / Mualem

```
Se(h) = [ 1 + |alpha h|^n ]^{-m},   m = 1 - 1/n        (h<0)
theta(h) = theta_r + (theta_s-theta_r) Se,             (h>=0: theta=theta_s)
C(h)   = dtheta/dh                                     (analytic, h<0; 0 for h>=0)
K(h)   = Ks * Se^L * [ 1 - (1-Se^{1/m})^m ]^2, L=1/2   (h>=0: K=Ks)
```

These mirror the probe's `VG.theta`/`VG.K` (loam `θ_r=0.078, θ_s=0.43, α=3.6, n=1.56`; `Ks` from §F's SE-Piedmont generator). Parameters arrive as **spatially-varying DOLFINx `Function`s** (DG0/P1), enabling material zoning — clay zones with `Ks→0, (θ_s−θ_r)→0` realize §E's sealed faces as a *material* property, not a runtime switch (scope lock). `K(h)`, `C(h)` are UFL expressions of the primal `h`, so the Jacobian is assembled by symbolic/AD differentiation, replacing the probe's numerical Jacobian.

**Near-saturation regularization (NUMERICS DECISION — corrected 2026-06-04).** `dK/dSe` (hence `dK/dψ`, required for the **exact UFL Jacobian** and for the off-diagonal exchange linearization through `σ_soil = K(ψ)·Ω_geom`) is **singular as `Se → 1⁻` (approaching SATURATION)** — NOT at the dry end. The Mualem bracket-derivative contains `(1−Se^{1/m})^{m−1}` with `m−1 < 0`, which diverges as `Se → 1`; the dry end (`Se → 0`) is smooth with `dK/dψ → 0`. *An earlier draft had this inverted (a `Se`-floor at the dry end does not touch the real, wet-end singularity); a 2026-06-04 adversarial review verified numerically that `dK/dψ` blows up toward saturation and is benign when dry.* Whenever a node reaches saturation/ponding — routine in PIDS, since bare channels deliberately saturate the soil — `ufl.derivative` reproduces this divergence and stalls Newton. **Fix (Arik, 2026-06-04): the Vogel et al. (2001) air-entry modification** — a small air-entry head `h_s ≈ −2 cm`, so the soil reaches `Se = 1` (and `K = Ks`) at `h_s` rather than at `h = 0`, renormalizing the retention/Mualem curves so `K` and `dK/dψ` stay finite **through** saturation. Applied identically in the float and UFL closures (and consistently to `θ`, `K`, `C`). Standard and citable (HYDRUS-family).

### B.3 Primary variable & saturated/unsaturated transition

`h` is the single unknown everywhere (no regime switching), keeping saturated (`h≥0, K=Ks, C=0`) and unsaturated (`h<0`) zones in **one continuous formulation** — essential because PIDS features straddle the water table (below it a feature *receives* groundwater and drains laterally; over dry soil the same feature *disperses* at the matric-potential rate). The hazard is `C(h)→0` as `h→0+`: the saturated block degenerates toward Laplace/incompressibility. Mitigations: (a) the **Vogel air-entry** modification (B.2), which keeps `K` and `dK/dψ` smooth through saturation; (b) **mass-lumped** capacity (B.4); (c) line-search / pseudo-transient continuation. For an **unconfined** aquifer (the locked assumption) the incompressible saturated interior is well-posed when anchored by the capillary fringe (`C>0`) and the domain BCs. **SNES-VI is NOT used in Module 1** — reserved for overland depth-positivity (§C) and optional one-way devices (§E).

**No specific-storage term (DECISION — Arik, 2026-06-04, OVERRIDES the earlier provisional `Ss`).** The model **always assumes an UNCONFINED aquifer** (free water table, atmospheric at the phreatic surface) with a **permanent, non-deforming soil skeleton**, and **ignores storage due to compressibility** — so there is **no `Ss` term**. Saturated-zone transient storage is the rising/falling water table (specific yield), captured by the `θ(ψ)` capillary-fringe capacity (`C>0` through the fringe); the fully-saturated interior is incompressible / quasi-instantaneous, which is well-posed for an unconfined system anchored by that fringe and the domain BCs. Saturated-zone **lateral flow is driven by the water-table slope**, emergent from `H = ψ + z` (no separate Dupuit/depth-integrated solver needed). The Tier-1 conservation gate is therefore the **`θ`-only** balance `|Δ(∫θ) − net_flux − sources|/scale < 1e-6`. No solute transport.

### B.4 FEM discretization

- **Space:** continuous Galerkin **P1** default for `h` (robust near fronts, cheap, natural for lumping); **P2** available for MMS convergence-order studies (the spike confirmed P2 reproduces an exact quadratic to machine precision in DOLFINx 0.10). Dimension-agnostic UFL on `SpatialCoordinate`/`FacetNormal`; element degree is a config knob.
- **Mass lumping:** the capacity term is **lumped** (row-summed / nodal quadrature) — standard for Richards to suppress oscillations and non-monotone fronts and to keep `θ ∈ [θ_r, θ_s]`.
- **MMS order gate vs lumping (DECISION).** Mass-lumping degrades the P1 spatial L2 order below 2, so a lumped transient run generically will **not** show clean 2nd-order convergence — the order gate would then fail on the *correct production discretization*. Therefore: **MMS order verification runs the consistent-mass variant** (asserting P1≈2 / P2≈3 in L2), while **production and the conservation/monotonicity gates run lumped mass.** The order gate is thus falsifiable on the right discretization, and the lumped path is gated on conservation + plausibility, not order. This split is the authoritative reading of the Tier-1 order gate (see §I.1).
- **Front stabilization:** (i) lumping; (ii) inter-element **K weighting** option (probe arithmetic `Kf=0.5(K_i+K_{i+1})`; FEM adds upwind/geometric-mean since arithmetic over-predicts flux into dry cells); (iii) optional SUPG/entropy-viscosity only if gravity-dominated fronts under-resolve.
- **Time:** implicit **backward-Euler**, adaptive `dt` (sub-hourly storm peaks → daily inter-event) driven by Newton count + a front error estimate (**§H owns control**; Module 1 exposes residual/Jacobian + a step-acceptance hook).

**Residual (UFL pseudocode):**

```python
F = ((theta(h) - theta_n)/dt * v) * dx_lumped \   # FULL storage residual lumped (not just C-increment)
  + inner(K(h)*(grad(h) + e_g), grad(v)) * dx \    # K scalar or tensor K̄; e_g global gravity
  - q_neumann * v * ds(flux_marker) \
  + Q_exchange(h) * v * dx           # §D/§E hook; soil-side h,K(h) live here
J = derivative(F, h)                 # symbolic Jacobian -> PETSc SNES (Vogel air-entry keeps dK/dψ finite through saturation)
```

### B.5 Dimension-agnostic detail

The **only** dimension-aware quantity is gravity: `e_g = as_vector([0]*(gdim-1) + [1])` selects the **last spatial coordinate** as vertical (z up). Every other term — capacity, stiffness, BC integrals, exchange — uses `grad`/`div`/`dx`/`ds` and is identical for `gdim ∈ {1,2,3}`. The same module solves the 1-D verification column (Philip, Green–Ampt, Celia, Gardner), then 2-D/3-D meshes, with **zero code divergence**.

### B.6 Boundary & initial conditions

- **Dirichlet head** (`h=h_D`; water-table/fixed-head bottom, probe `H_bot=0`): `fem.dirichletbc`.
- **Neumann flux** (`q·n=q_N`; recharge / no-flow `q_N=0`): natural via `ds`.
- **Rainfall + surface ponding (IMPLEMENTED 2026-06-04, `add_ponding_bc`):** rainfall enters as a Neumann influx; the excess the soil cannot infiltrate accumulates as a **surface ponding store** of depth `d = max(h, 0)` that raises the surface head — `rain = infiltration + d(pond)/dt`, mass-conserving, **vertical accumulation only** (lateral routing is the §C/§D overland module). The pond store also gives a saturated surface node a nonzero storage diagonal, so **saturation-excess storms (intense rain on wet soil) converge** with no `Ss` term. *Deferred:* the **supply-limited evaporation** head-cap (a fixed evaporative flux exceeding the soil's delivery capacity diverges as the surface dries below what `K(ψ)` can supply) needs an atmospheric/Robin switch, owned with the forcing/vegetation module.
- **Seepage face** (complementary `h≤0, q·n≥0`): default smooth penalty, VI-exact optional (deferred).
- **Lower boundary (geologic setting):** the default **no-flux** base (impermeable bedrock / aquitard) makes the column fill from the **bottom up** under sustained recharge — *water-table mounding* against the base, meeting the top-down infiltration front in the middle last (an expected, verified behaviour). A **free-drainage / deep-water-table** lower BC (unit-gradient outflow `q = K(h)`) instead yields the classic top-down front; it is the alternative for soil over a deep water table (a small future addition).
- **Initial conditions:** hydrostatic `h=z_wt − z` (probe `psi=−z`, `H=0`, the exact zero-flux steady state = host sanity gate), or prescribed antecedent profiles (saturated / field-capacity / bone-dry) for Tier-2.

### B.7 Interface to pids-features (reference only)

Module 1 **exposes** soil-side state at any embedded location: `h(x), θ(x), K(h(x))`, and the per-cell `Q_exchange` slot. Because exchange `q = σ·(H_f − H_soil)` (hydraulic heads, `H_soil = ψ_soil + z`) is **series-limited by soil `K(ψ)` and matric potential `ψ = h`**, soil-limited behavior (dry clay imbibing fast via high suction despite low `K`) is captured natively. Module 1 guarantees `h, K(h)` are evaluable/**differentiable** at the embedding manifold (with the **Vogel air-entry** keeping `dK/dψ` finite *through saturation*, B.2) so §E/§D assemble a smooth monolithic exchange Jacobian. Module 1 does **not** define `σ`, faces, or geometry.

### B.8 Three-tier sanity plan

**Tier 1 (pytest, TDD — tests precede code):**

| Check | Reference | Pass criterion |
|---|---|---|
| Early-time infiltration | **Philip** series | profile L2 < tol; front-advance order |
| Sharp front | **Green–Ampt** | front position vs analytic |
| Zero-flux equilibrium | steady **hydrostatic** (`h=−z`) | residual ~0, ~1 Newton iter (host gate) |
| Mass conservation | **Celia 1990** column | `|ΔS − net_flux − src|/scale < 1e-6` |
| Steady columns | **Gardner** soil | L2 vs analytic |
| Convergence ORDER | **MMS** (consistent-mass variant) | spatial ≈ P1:2 / P2:3 (L2); temporal ≈ 1 (BE) under refinement |
| Plausibility | — (lumped production path) | `θ∈[θ_r,θ_s]`, bounded `h`, no NaN/Inf; conservation gated on lumped, order on consistent-mass |
| Solver | — | Newton convergence + determinism + refinement order |

Run **1-D first** (reuse probe host), then *same UFL* on 2-D/3-D for conservation + plausibility only.

**Tier 2 (1-D column → 2-D/3-D):** typical storm; 100-yr **sub-hourly** Atlas-14 storm; 100-yr drought (monotone drying); **saturated** & **bone-dry** antecedents. Expect ponding/runoff onset when supply>capacity; monotone inter-event redistribution; no mass leak across `dt` sub-hourly→daily.

**Tier 3 (viz subagent):** `h`/`θ`-vs-depth profile with **time slider**; **mass-balance vs time** diagnostic; metrics panel. NetCDF (dims `z,t`; vars `head, water_content, ponded_depth, mass_balance_error`; `metrics`/soil attrs); viz agent never imports the solver. *Implemented 2026-06-04:* `forward-model/viz/make_sanity_html.py` (self-contained offline Plotly HTML) over four scenarios (typical/small on mesic; intense on dry/wet), each a storm + equal rainless recession.

**Artifacts:** `forward-model/tests/test_subsurface_*.py` (+ `tests/analytical/`); `validation/sanity/subsurface__<YYYY-MM-DD>.md`; `validation/sanity/viz/subsurface__<check>__<YYYY-MM-DD>.html`. **Seed:** port `scratch/catchvalve_probe.py`'s mixed-form residual, VG/Mualem closures, hydrostatic gate, and Newton loop into dimension-agnostic UFL; **drop the retired catch-valve term**.

---

## C. Module 2 — Overland flow

### C.1 Formulation: diffusion-wave (default), local-inertial-ready

**2-D diffusion-wave** is the production default, written so a **local-inertial (Bates et al. 2010) momentum term is a one-line UFL addition behind a flag**. Full 2-D shallow-water (SWE) is **rejected** by YAGNI: PIDS questions are runoff partitioning, ponding depth/extent, where surface water reaches PIDS top faces, and a `depth·slope` erosion-threshold velocity — none need resolved shocks/hydraulic jumps that justify SWE's cost (Riemann fluxes, momentum positivity, far smaller CFL). Diffusion-wave drops local + convective inertia, leaving a gravity–friction balance: a **degenerate parabolic** nonlinear diffusion in surface head. It is the natural sibling of mixed-form Richards (same backward-Euler + Newton/SNES + monolithic assembly), reuses the probe-host time integration, and is the standard surface law in HGS/ParFlow (so coupled tilted-V/superslab benchmarks are apples-to-apples). Local-inertial is held in reserve only if very flat reaches show diffusion-wave time-step stiffness or backwater inaccuracy.

### C.2 Surface unknown and domain

Unknown: **ponding depth `d ≥ 0`** on the **top boundary facets** (codim-1 facets of the 3-D mesh; a standalone 2-D mesh for module tests). Surface head `H_s = z_b + d`, with `z_b` topography from §F. We carry `d` (not `H_s`) as the primary DOF because positivity is stated directly on `d`, and storage/forcing are linear in `d`. P1 continuous Lagrange on top facets by default; `z_b` enters as a known P1 field so bed slope `grad z_b` is exact per element.

### C.3 Governing PDE (dimension-agnostic UFL)

```
d(d)/dt + div(q) = r - i_inf - i_chan + s_pids
q = -K_s(d) grad(H_s),   K_s(d) = d^{5/3} / ( n_man * |grad H_s|^{1/2} + eps_S )
```

so `div(q) = −div( K_s(d) grad(z_b + d) )` is a **nonlinear degenerate diffusion** (vanishes as `d→0`). Sources (units m/s, all supplied by neighbors, never re-specified here): `r` rainfall/PET-evap from §G; `i_inf` net infiltration — **owned by §D coupling**, set from the monolithic surface↔subsurface flux so the interface mass closes; `s_pids`/`i_chan` inflow into surface-reaching PIDS feature **top faces** — **owned by §D/§E**, evaluated from the smooth `q = σ_top·(H_f − H_s)` exchange using the surface head this module exposes. Overland neither defines `σ` nor picks direction; it exposes `H_s`/`d` and accepts the flux. `eps_S` (slope floor) and the `d^{5/3}` conductance are the two nonlinearities; both have analytic UFL derivatives so the Jacobian is exact. **Only the bed-slope term `grad z_b` is dimension-aware** (§F supplies `z_b` for 1-D/2-D/3-D-top); the rest of the UFL is identical across dimensions.

**`eps_S` is a physical regularization, not only a Jacobian guard.** The **additive** `eps_S` in the denominator alters the constitutive flux on near-flat reaches: as `|grad H_s| → 0`, `K_s(d) → d^{5/3}/eps_S`, so `q → −(d^{5/3}/eps_S)·grad H_s` is **ordinary linear diffusion**, not the Manning `√slope` law — precisely the lake-at-rest / flat-terrain regime. `eps_S` is therefore documented with a **stated magnitude** (`eps_S ≈ 1e-3…1e-2 m^{1/2}`, smallest value that keeps the Jacobian non-singular) and the lake-at-rest Tier-1 test must confirm the residual spurious flux stays **below the 1e-6 gate** at that magnitude. For *very* flat reaches the preferred cure is the **local-inertial form** (C.1, flag-gated), not enlarging `eps_S`.

### C.4 Depth positivity

Two coordinated mechanisms, matching the locked "VI reserved for overland depth-positivity":

1. **Primary — PETSc SNES-VI (`vinewtonrsls`)** with **lower bound `d ≥ 0`**, upper `+∞`, the **same VI path proven in the install spike**. The active set is the dry region; the spike showed `vinewtonrsls` converges with a non-trivial active set — exactly the wet/dry front.
2. **Supporting — degenerate-diffusion regularization:** `K_s(d) → 0` as `d → 0`, so dry cells are naturally no-flux; a small `eps_d` floor inside `d^{5/3}` keeps the Jacobian non-singular, **not** to permit negative `d`. A well-balanced (lake-at-rest) discretization (C.6) gives zero spurious flux for a flat surface over uneven bed.

Clamping `d` post-solve would break mass conservation (1e-6 gate); the VI enforces the bound *inside* the converged solve so storage and flux stay consistent — the analogue of the probe's "implicit in-solver hook beats lagged operator-splitting", now applied to wet/dry complementarity.

### C.5 Velocity / direction field for the erosion threshold

Module 2 publishes a **cell/facet velocity vector** for §G's erosion check (locked: boolean `depth·slope → velocity/shear vs threshold`, NOT sediment transport):

```
u = q / max(d, d_min),  |u| = (d^{2/3}/n_man) * |grad H_s|^{1/2}
direction = -grad H_s / |grad H_s|
```

We expose the **vector `u`** (quiver + direction) and scalar **bed shear** `tau = rho g d S_f`, `S_f = |grad H_s|`, so the erosion check differences `tau` (or `|u|`) vs threshold without recomputing hydraulics. `d_min` guards the dry-cell division and is a reported diagnostic, not a physical depth.

### C.6 Discretization and stabilization

- **Space:** P1 CG for `d` on top facets; `z_b` P1.
- **Stabilization framing:** diffusion-wave is diffusion-dominated (no advective operator), so **classic SUPG-for-advection does not apply to the headline equation** — the need is **wet/dry-front robustness and well-balancedness**, not Petrov–Galerkin upwinding.
- **Well-balanced wet/dry:** hydrostatic-reconstruction-style treatment of `H_s = z_b + d` at facets so a still pond over sloping bed gives machine-zero flux (lake-at-rest); face `K_s` evaluated with upwind/harmonic depth so a dry downslope cell cannot draw water uphill.
- **If local-inertial enabled,** the momentum flux *is* advection-dominated on fast reaches; there apply **flux limiting / upwinded face discharge** (LISFLOOD-FP staggered-style) and SUPG becomes relevant — hence flag-gated, not baked into the default.
- **Time:** backward-Euler with the project adaptive controller (§H); flat-terrain parabolic stiffness is the main step driver and the documented trigger to consider local-inertial.

### C.7 Coupling exposure (mechanics deferred to §D/§E)

Overland **exposes** `d`, `H_s = z_b + d`, `u` on top facets; **consumes** `i_inf`, `i_chan/s_pids`, `r/PET` as residual sources. Per corrected PIDS physics a **surface-reaching channel's top face couples to overland**: §D/§E form `σ_top·(H_f − H_s)` (both hydraulic heads) and return withdrawal/return as `s_pids`; Module 2 supplies `H_s` and never owns `σ`, catch-drain logic, or the optional one-way (VI) device. All exchange enters the **single monolithic residual + Jacobian** each step — no operator-splitting of surface↔subsurface or surface↔feature flux.

### C.8 Three-tier sanity plan

**Tier-1 (`forward-model/tests/test_overland_*.py`, helpers in `tests/analytical/`):** (a) **kinematic-wave rising/falling hydrograph on a plane** (diffusion-wave → kinematic in the steep limit) — hydrograph shape + time-to-equilibrium; (b) **Stoker dam-break** front check (matches the slumping profile away from the shock; *documents* the expected SWE-vs-diffusion-wave discrepancy at the front rather than hiding it); (c) **tilted-V catchment** (standard surface benchmark + coupling on-ramp to §D); (d) **MMS** with manufactured `d(x,t)` for spatial/temporal order (≈2 space P1, 1 time BE). Conservation `|Δstorage − net_flux − sources|/scale < 1e-6`, storage `= ∫ d dA`. Plausibility: `d ≥ 0` (VI bound never violated), no NaN/Inf, bounded `H_s`, finite `u`, lake-at-rest zero flux. Solver/repro: SNES-VI converges within cap; deterministic; refinement reduces error at measured order.

**Tier-2 (alone + paired §D/§G + full system):** typical SE-Piedmont storm; **100-yr sub-hourly burst** (Atlas-14) exercising adaptive stepping + wet/dry front; **dry plane** (zero `d`, confirm no spurious wetting, VI active set stable); **steep vs flat** (flat = diffusion-wave stiffness stress probing the local-inertial trigger). Acceptance: mass-conservative + plausible + expected qualitative behavior (runoff partitioning, recession limb, velocities crossing/not crossing the erosion threshold as forcing dictates).

**Tier-3 (separate viz subagent; xarray-NetCDF: `surface_depth`, `velocity` on `(time,x,y)` + metrics attr):** surface-depth **animation with velocity quiver**; **hydrograph + hyetograph twin-axis** (inverted rain bars, outflow line, log-y recession); diagnostics (mass-balance error vs time, Newton iters vs time, MMS order log-log). Human sign-off.

---

## D. Module 3 — Monolithic coupling (land-surface AND embedded-feature exchanges)

### D.1 Role and scope

Module 3 is the **assembler**, not a holder of new state. It composes the residuals/Jacobians authored by §B (subsurface Richards), §C (overland), and the embedded-feature exchange terms authored by §E into **one** global residual `F(U)=0` and Jacobian `J = ∂F/∂U` over a **single blocked space**, solved by **one** Newton (PETSc SNES) iteration per timestep. Two families of *smooth, potential-driven* exchange live in that residual: **(i) land-surface** between ponded overland water and the top subsurface; **(ii) feature↔soil per-face** (physics owned by §E, injected into the same monolithic residual). Module 3 specifies how both compose, the global mass-balance accounting across **both** interface families, and the coupled preconditioner. Solver internals (SNES, SNES-VI, line search, adaptive `dt`) are referenced to §H, not re-specified.

### D.2 Global state and the one residual

Solution vector `U = (ψ, d, [h_feat])` over a blocked space `W = V_ψ × V_d × V_feat`:

- `ψ` — subsurface pressure head on the host mesh (§B; mixed-form Richards, van Genuchten/Mualem).
- `d` — overland ponding depth on the **top surface facets** (§C).
- `h_feat` — feature hydraulic head DOFs on the embedded 1-D vectors (§E; present only when features exist, so `V_feat` may be empty).

The monolithic residual is the sum of the three module residuals plus the two coupling families:

```
F(U) = R_sub(ψ; q_ls, q_feat) + R_ovl(d; q_ls, rain, ET) + R_feat(h_feat; q_feat)
     + C_ls(ψ, d) + C_feat(ψ, h_feat)
```

`C_ls` and `C_feat` are *the same fluxes* entered with opposite sign into the abutting blocks — that sign-paired entry makes the assembly **conservative by construction**. Module 3 owns `C_ls`; it *links* `C_feat` into the global system but the kernel is §E's.

### D.3 Land-surface exchange: first-order conductance vs common-node

**(A) Common-node / pressure-continuity.** Force `ψ_top = d` (head continuity) on the surface facet. This is **gravity-consistent** because both sides share the same surface elevation `z_surf` (surface hydraulic head `H_s = z_surf + d`; subsurface top hydraulic head `H_top = z_surf + ψ_top`; equating heads gives `ψ_top = d`). Exact at the interface, fewer parameters, but it *imposes* continuity (cannot represent a surface-seal/reduced-contact resistance) and the surface unknown collapses into the subsurface block (harder fieldsplit).

**(B) First-order (Robin) conductance — the default.** A smooth, **gravity-consistent** flux driven by the hydraulic-head drop across the thin coupling film (`H_s = z_surf + d`, `H_top = z_surf + ψ_top`, so `H_s − H_top = d − ψ_top` — the surface elevations cancel):

```
q_ls = k_ex · (d − ψ_top),   k_ex = K_rel(ψ_top) · K_s / ℓ_c
```

with `ℓ_c` a thin coupling length (~ top-cell half-height). **Do not** add `z_surf` to the driving potential: `z_surf` belongs to *both* hydraulic heads and cancels; a spurious `+z_surf` would over-drive infiltration by the full surface elevation (tens of metres of head error on a real DEM). Infiltration-capacity limiting and **ponding emerge** naturally: when `ψ_top` saturates, `K_rel→1` caps `q_ls` and excess rain accumulates in `d` (infiltration-excess / Hortonian runoff); when the water table reaches the surface, `ψ_top` rises to `d` and `q_ls→0` (saturation-excess runoff). **Recommendation: adopt (B) as default** — in the limit `k_ex→∞` it correctly forces `ψ_top = d`, **reproducing common-node (A)**; it keeps `d` and `ψ` as distinct blocks (preconditioner-friendly), and represents the reduced-contact/seal cases PIDS bare channels need. Retain (A) as a verification reference. A **Tier-1 assertion that the `k_ex→∞` limit of (B) reproduces (A) (`ψ_top = d`) to machine precision** guards against head-datum errors of exactly this class. `q_ls` is `C^1` in `(ψ,d)`, contributing analytic Jacobian entries to four blocks: `∂C_ls/∂ψ`, `∂C_ls/∂d`, and their transposes.

### D.4 Composing the feature family

The feature↔soil flux is the per-face form from §E, `q_feat = σ·(H_f − H_soil)` with `H_soil = ψ_soil + z` (hydraulic heads), `σ = (1/σ_feat + 1/(K(ψ)·Ω_geom))^{-1}` a series of the **engineered feature-face leg `σ_feat`** (set per face and **independently for receiving vs dispersing**) and the **direction-symmetric soil leg `K(ψ)·Ω_geom`**; `σ_feat ∈ [0,∞)`. Module 3's composition responsibilities:

1. **Restriction/prolongation operators** between the embedded 1-D DOFs and the 3-D `ψ` block (embedding mechanics finalized in §F; Module 3 consumes the resulting interpolation matrices `P`).
2. **Sign-paired assembly:** `+q_feat` into the `h_feat` block, `−P^T q_feat` into the `ψ` block, so feature storage gain equals soil loss to machine precision.
3. **Smoothness guarantee for the headline solve:** with fixed/bidirectional `σ`, `q_feat` is smooth → the coupled solve stays a **smooth monolithic Newton problem**. Bidirectionality is intrinsic and correctly signed *because the comparison is on hydraulic heads*: below the water table `H_soil > H_f` so the feature *receives* (drains laterally); over dry soil the large matric-potential magnitude in `ψ_soil` makes `H_f > H_soil` and drives *dispersion* out of a bare feature — both directions are the *same* smooth term, no switch. Clay faces enter as `σ_feat = 0` (identically zero rows). An **optional one-way device** is the only place an inequality enters; deferred to SNES-VI (§H) and never touching the default residual.

### D.5 Assembly → one Newton solve

Per step (backward-Euler, adaptive `dt` per §H):

```
assemble F(U_k):  R_sub + R_ovl + R_feat + C_ls + C_feat   (vectors, sign-paired)
assemble J(U_k):  block Jacobian incl. all 4 C_ls cross-blocks
                  and the C_feat ψ<->h_feat cross-blocks (via P)
SNES Newton solve J·δU = -F  ->  U_{k+1}
```

SNES-VI (`vinewtonrsls`, install-spike-proven) is engaged **only** when overland depth-positivity `d ≥ 0` is active or an optional one-way device imposes bounds; the default coupled solve is plain SNES Newton. **No operator-splitting** of the exchange terms (non-iterated splitting is disqualified upstream).

### D.6 Coupled preconditioning (high level)

PETSc **`fieldsplit`** with three fields `{ψ, d, h_feat}` (field indices `0/1/2`). Recommended **multiplicative/Schur** ordering with the **subsurface block as the heavy lift**: **GAMG/hypre AMG** on `J_ψψ` (the elliptic Richards block); the overland block is small and well-conditioned (point-block Jacobi / ILU); the feature block is 1-D and cheap (direct/ILU). **Implication of high-K embedded features:** features create order-of-magnitude conductivity contrast and a near-elliptic 1-D subnetwork inside the 3-D field, degrading plain AMG on `J_ψψ`. Mitigations: keep `h_feat` a *separate* field (do **not** fold it into `ψ`) so the contrast lives on the off-diagonal `C_feat` coupling, not inside the AMG operator; use the feature block as an exact inner solve in a Schur complement. **§H.3 is the single source for the concrete options string and field indices** (`petsc_options_prefix` keyword-only in DOLFINx 0.10); the field decomposition `{ψ=0, d=1, h_feat=2}` is settled here and in §H.3, and only the *multiplicative-vs-Schur robustness* choice is deferred to the R3 conditioning study (Q9) — not the field split itself.

### D.7 Three-tier sanity plan (independently AND in concert)

**Tier 1 (pytest/TDD).** (a) **MMS** on the coupled `(ψ,d[,h_feat])` system with manufactured `q_ls`/`q_feat`, confirming spatial+temporal **convergence order**. (b) **Tilted-V** + **superslab**; cross-code agreement vs published **HGS/ParFlow** is **OPTIONAL** (per 2026-06-04). (c) **Global mass balance across BOTH interface families:** `|Δstorage_sub + Δstorage_ovl + Δstorage_feat − net_boundary_flux − sources|/scale < 1e-6`, plus **per-interface** closure (land-surface `q_ls` and each feature face `q_feat`). (d) Plausibility: `0≤S≤1`, `d≥0`, bounded heads, no NaN/Inf; the `k_ex→∞` limit reproduces common-node (A); clay faces pass exactly zero flux; interface flux antisymmetry `q_surf→sub = −q_sub→surf`.

**Tier 2.** Infiltration-excess vs saturation-excess **partitioning** under typical + **100-yr** design storm (sub-hourly → adaptive stepping); water-table **mounding/recharge**; **recession** limb after rain ceases; then a **feature-laden domain in concert** (bare channel dispersing over dry soil; french/feature below the water table draining). Run **alone → pairwise (overland+subsurface; subsurface+features) → full system**, recording couplings exercised.

**Tier 3.** Separate viz subagent: coupled storm with **surface depth**, a **subsurface saturation transect**, and **feature exchange** on a **shared time slider**, plus **interface-flux** and **global-mass-balance** panels. Human signs off. Artifacts per the standardized paths; NetCDF data contract per the viz routine.

---

## E. Module 4 — pids-features (the embedded 1-D-vector feature layer)

The unifying PIDS abstraction. **Every** named feature (bare/clay channel, tunnel, french drain, catch drain, standard pipe) is *one* parameterized object: a **1-D linear vector embedded in the 3-D variably-saturated (Richards) continuum** (mixed-dimensional 1D-in-3D). The embedding is the **core representation**, not an optimization. Each vector carries an independent scalar DOF — the **feature hydraulic head `h_f(s,t)`** along arc-length `s` — and contributes three smooth residual terms (conveyance, per-face exchange, storage) into the **single monolithic residual** assembled by §D. The default solve is a **smooth monolithic Newton** problem; the PETSc VI path (`vinewtonrsls`, §H) is reserved for overland depth-positivity (§C) and the **optional** hard one-way device — never the default features.

### E.1 The one parameterized object

State: host soil head `h_s` (P1 Lagrange, §B) plus feature head `h_f` on a 1-D function space over the embedded edge set Γ (§F tags the polylines as `MeshTags` on a 1-D submesh; host cells pierced by each segment are tagged for the coupling quadrature). A feature is fully specified by:

| Field | Symbol | Meaning |
|---|---|---|
| centerline | Γ | tagged 1-D submesh edges |
| cross-section | `A = w·d` | width × depth (m²) |
| axial conductivity | `K_feat` | Kozeny–Carman / Hazen from grain size; granular Darcy (not Manning); pipes → pipe conductance `C_pipe` |
| effective porosity | `φ_eff` | storage; clay → ~0 |
| per-face feature-face conductance | `σ_feat^dir` | face ∈ {top,bottom,lateral}; dir ∈ {receive,disperse}; each ∈ [0,∞). **Asymmetry lives only here** |
| feature tangent | `t̂` | along-vector axis; `∇_s = t̂·∇` (tunnel ⇒ `t̂ ∥ e_g`) |
| gravity axis | `e_g` | global vertical unit vector (dimension-aware) |

### E.2 Residual contributions (dimension-agnostic UFL)

**(1) Axial conveyance** — 1-D Darcy along Γ through granular fill on the **feature hydraulic head `H_f`**: `q_axial = −K_feat·A·∂H_f/∂s`, with `∂/∂s = t̂·∇` the along-vector (tangential) gradient. Because `H_f = ψ_f + z` is a *hydraulic* head, the **gravity drive is carried natively — no extra `e_g` term** — so a vertical tunnel (`t̂ ∥ e_g`, F.3.3) drains downward without a special case. Weak form on test `v_f`: `∫_Γ K_feat·A·(grad_s H_f)·(grad_s v_f) ds`. Pipes substitute a hydraulic conductance (`C_pipe·A` lumped) for `K_feat·A`. High-K makes `H_f` nearly uniform along well-connected runs — the lateral-conveyance signature (patent claim **C-001**).

**(2) Per-face exchange** — smooth potential-driven flux per unit length on **hydraulic heads** (`H_soil = ψ_soil + z`), summed over faces:

```
q_exch,face = sigma_eff^dir * (H_f - H_soil)
sigma_eff^dir = ( 1/sigma_feat^dir + 1/(K(psi)*Omega_geom) )^(-1)   # series; per face; per direction
   # ASYMMETRY (receive vs disperse) lives ONLY in the engineered feature-face leg sigma_feat^dir.
   # The soil leg K(psi)*Omega_geom is direction-SYMMETRIC (soil K is not directional).
dir = receive if (H_soil > H_f) else disperse        # selects which sigma_feat leg; soil leg unchanged
```

`K(ψ)` is the van Genuchten/Mualem conductivity (§B), evaluated at the **same near-field `ψ` used to derive `Ω_geom`** (R2/Q4 — cell value vs `r_eq`-sampled, fixed jointly with §F so the dispersion rate is reproducible); `Ω_geom` is §F's radial well-index (below). The **flat-plate `P_face/ℓ` soil leg is retired** — it implies a resolved planar interface, the very near-field the 1D-in-3D embedding does *not* resolve, and mis-scales the dominant dry-soil dispersion rate; the **radial `K(ψ)·Ω_geom`** soil leg is authoritative. For the *default symmetric* feature `σ_feat^receive = σ_feat^disperse`, so `σ_eff` is constant across the switch and the term is **C¹ smooth** — no VI; and because the soil leg is direction-symmetric, the soil-limited limit (`σ_feat → ∞`) is direction-independent as physics requires. This term is **bidirectional and central:** below the water table `H_soil > H_f` ⇒ feature **receives** groundwater (drain); over unsaturated soil `H_f > H_soil` and `K(ψ)·Ω_geom` is set by dry-clay matric suction ⇒ **disperses** fast despite low `K` (native to Richards/van Genuchten). The exchange couples `H_f ↔ H_soil` symmetrically: **added** to the feature residual on `v_f` and **subtracted** from the host Richards residual on `v_s` at the same Γ-quadrature points (§D distributes `−q_exch` into pierced host cells), so the pairwise term is conservative by construction.

**(3) Storage** — `∫_Γ φ_eff·A·(∂H_f/∂t)·v_f ds` → backward-Euler `φ_eff·A·(H_f − H_f^n)/Δt` (§H adaptive Δt). Total stored `= φ_eff·A·|Γ|`.

**Assembled feature residual** (`v_f`):

```
R_f =  ∫_Γ phi_eff*A*(H_f - H_f^n)/dt * v_f ds
     + ∫_Γ K_feat*A*(grad_s H_f).(grad_s v_f) ds      # gravity carried in H_f; no extra e_g term
     + Σ_face ∫_Γ sigma_eff_face*(H_f - H_soil)*v_f ds  # H_soil = psi_soil + z
     - ∫_Γ q_source * v_f ds          # top-face inflow (E.4)
```

The host contributes `−Σ_face ∫_Γ sigma_eff_face·(H_f − H_soil)·v_s ds` to `R_s`. Both enter the one residual; the Jacobian (`∂R/∂(ψ_soil,H_f)`, UFL auto-diff) is solved by a single Newton/SNES step per timestep.

### E.3 Taxonomy = parameter sets of the one object

| Feature | `σ_top` | `σ_bottom` | `σ_lateral` | `K_feat` | Notes |
|---|---|---|---|---|---|
| bare channel | overland-coupled (E.4) | soil-limited | soil-limited | high | bidirectional, ψ-limited disperse |
| clay channel | overland-coupled | **0 (sealed)** | soil-limited | high | bentonite floor `φ_eff≈0`, `K≈0` |
| tunnel | sealed/lateral | soil-limited | soil-limited | high | **Γ vertical**, drains downward |
| french drain | n/a | soil-limited | soil-limited | high (open granular) | intercepts rising water table |
| catch drain / pipe | inlet only | **0** | **0** | `C_pipe` | sealed walls; conveys to distal outfall |

**Clay** = a **static sealed face:** `σ_feat = 0`, `φ_eff ≈ 0` (material zoning, not a runtime switch — §F sets it). With `σ_feat = 0` the series `σ_eff = 0` regardless of the soil leg, so a sealed face passes **zero normal flux** identically (term drops from `R`).

**Gutters and pumps** extend the same taxonomy (detailed parametrization deferred — not needed for Modules 1–4): a **gutter** = high top-face receive (`σ_top → ∞`) + along-path sealed disperse (`σ_disperse = 0`) + a single exfiltration outlet node; a **pump** is the lone **active** member — point intake (receive only) + an active prescribed flux/head + point or multi-point outlets (disperse only) — i.e. the one feature carrying a nonzero source/sink term beyond passive potential-driven exchange. Both remain the same 1-D embedded vector; only their per-face `σ` pattern (and, for the pump, an added active term) differs.

### E.4 Surface-reaching features

A feature whose top is open (bare/clay channel inlet) couples its **top face** to overland (§C) on **hydraulic heads** — consistent with the lateral/bottom faces (E.2) and the locked datum: `σ_top` is a series of the feature-top `σ_feat` and an overland film conductance, with surface hydraulic head `H_s = z_surf + d`. The flux `q_top = σ_top·(H_f − H_s)` is added to `R_f` and subtracted from the **overland residual** — the same conservative pairwise pattern. Within Module E, *all* exchange comparisons are thus hydraulic-head differences (top against `H_s`; lateral/bottom against `H_soil = ψ_soil + z`), datum-consistent throughout. Overland depth-positivity (`d ≥ 0`) is handled by the **overland** VI, not here.

### E.5 Optional one-way device (degenerate asymmetric σ)

A hard one-way device is the limit `σ_feat^disperse = 0` (or `σ_feat^receive = 0`) with **non-smooth** complementarity at `H_f = H_soil`. Per the de-risk probe (`scratch/catchvalve_probe.py`), this **must** enter as an implicit in-solver hook routed through the **PETSc VI path** (§H) — never operator-split (the probe's lagged variant diverged to a 50 m head excursion). Default features are smooth and need **no** VI. This is **insurance, not the norm.** Note: the direction-selector (`receive`/`disperse` branch of `σ_feat`) is `C⁰` (kinked in its derivative) at `H_f = H_soil`; for the *symmetric* default the two branches are numerically equal so the kink is inert, but **any asymmetric `σ_feat` must be explicitly routed to the VI path** rather than left in the default residual.

### E.6 Three-tier sanity plan

**Tier 1 (`tests/test_pids_features_*.py`):** (a) embedded high-K Darcy vector conveyance vs analytical 1-D Darcy/pipe (convergence order under refinement); (b) **bidirectional exchange** vs a locally-resolved 3-D reference — feature below water table drains groundwater at the right rate; bare feature over dry soil disperses at the `ψ`&`K`-driven rate. **Falsifiability gate (make-or-break for C-004):** the embedded solution must converge to the resolved reference in `h_cell` using a **single, a-priori-chosen Peaceman `C` and a single `h_soil` sampling rule across ≥2 independent geometries (a horizontal french drain AND a vertical tunnel) and BOTH flow directions (drain and disperse)**. If `C` must be re-tuned per geometry or per direction, the embedding-fidelity claim is **unsupported** and this is a flagged M1/M5 outcome, *not* a silent pass; (c) **clay face passes zero normal flux** (structural `σ_feat = 0`, not tiny-K leakage); (d) water-table interception (french drain); (e) storage fills to `φ_eff·A·|Γ|`; (f) global+local mass balance `< 1e-6`; (g) optional asymmetric device never leaks the blocked direction; plausibility (bounded `H_f`, no NaN/Inf).

**Tier 2:** bare vs clay channel under a 100-yr Atlas-14 influx over dry vs saturated antecedent; tunnel draining a perched layer; french drain intercepting a rising water table; assemblies in concert with §D — mass-conservative + expected qualitative behavior.

**Tier 3:** viz subagent HTML — axial flow + per-face exchange time series with a metrics panel; transect of a bare channel dispersing into unsaturated soil vs a clay channel conveying only; storage filling. Human sign-off.

> Embedding **mesh mechanics** (how Γ tags edges and pierces host cells, the 1D-in-3D quadrature coupling, the connectivity factor `Ω_geom`) are owned by **§F (domain)**; this module consumes them.

---

## F. Module 5 — domain (mesh, topography, properties) + 1D-in-3D embedding mechanics

### F.1 Scope and role

Module 5 (`domain`) builds the discrete world: the **mesh**, the **topographic surface**, the **SE-Piedmont subsurface property fields** (`K_s`, porosity, van Genuchten `α/n/θ_r/θ_s`), and — now the core deliverable — the **1D-in-3D embedding mechanics** that places every PIDS feature as a parameterized 1-D vector in the 3-D Richards continuum. It produces mesh + tags + property `Function`s plus the geometric/connectivity operators that §D and §E assemble into the monolithic residual. It does **not** assemble physics; it hands neighbors the discrete scaffolding plus the connectivity factor `Ω_geom`.

### F.2 Mesh generation and the 1D/2D/3D ladder

Two backends behind one dimension-agnostic interface `build_mesh(spec) -> (Mesh, MeshTags...)`:

- **DOLFINx built-ins** (`create_interval`, `create_rectangle`, `create_box`) for canonical analytical-test domains — the 1-D Philip/Green–Ampt column (reuse the probe host), 2-D tilted-V, 3-D superslab. Cheap, exactly reproducible, used for Tier-1 convergence studies.
- **gmsh** (via `dolfinx.io.gmshio.model_to_mesh`) for irregular topography and **conforming** feature meshes. The OCC kernel builds the box, imprints feature polylines as embedded 1-D entities (`gmsh.model.mesh.embed(1, line_tags, 3, volume_tag)`), and writes physical groups that become MeshTags.

The same UFL solver code runs on all three meshes; only `gdim` and `e_g` differ. Validate 1-D first, then re-mesh 2-D/3-D and re-run identical assembly for conservation + plausibility.

### F.3 The 1D-in-3D embedding mechanics (core)

Each feature is a **polyline Γ_f** (ordered points + radius `r_f`, per-face `σ` specs, fill grain-size → `K_feat`, effective porosity `θ_eff`). Two embedding realizations, both supported:

1. **Conforming (edge-aligned).** gmsh `embed` forces the polyline onto mesh edges → a 1-D `submesh` whose entities are exact mesh edges sharing DOFs with the 3-D host along Γ_f. Preferred for the verification reference and for tunnels. *(Dropped if the M0 R1 spike selects the immersed-line-source-for-all fallback.)*
2. **Non-conforming (immersed).** The polyline cuts arbitrarily through 3-D cells; coupling is a **distributed line source** along Γ_f, evaluated at line-quadrature points via `dolfinx.geometry` bounding-box-tree point location (`compute_collisions` + `compute_colliding_cells`). Preferred for dense networks where edge-conforming meshing is intractable, and the **fallback realization for all features** if the M0 cross-mesh spike rules out the conforming-submesh path.

The feature carries its own 1-D space `V_f` (P1 on the submesh / line quadrature) holding the **feature hydraulic head `H_f`**. Axial Darcy conveyance is the 1-D Laplacian on Γ_f in `H_f` (gravity carried natively) with `K_feat` and cross-section `A_f = w·d`; storage is `θ_eff·A_f` per unit length. Module 5 supplies geometry (`A_f`, length, tangent `t̂`); §E owns the conveyance/storage/exchange UFL.

#### F.3.1 Per-face exchange σ as a mesh connectivity factor Ω (Peaceman-style)

Exchange per unit feature length is `q = σ·(H_f − H_soil)` (hydraulic heads). Physics defines `σ` as a **series** of an **engineered feature-face conductance `σ_feat`** and a **soil-leg** conductance; **only the feature-face leg carries the receive/disperse asymmetry** (the soil leg is direction-symmetric, since soil `K(ψ)` is not directional). Module 5 supplies the **geometric connectivity factor** so the soil leg is a Peaceman-well-index analogue, not an arbitrary number:

```
sigma_eff(face, dir) = [ 1/sigma_feat(face,dir) + 1/(K_soil(psi)*Omega_geom) ]^(-1)   # series; soil leg dir-symmetric
Omega_geom = 2*pi / ln(r_eq / r_f)        # radial-inflow shape factor toward the line (3-D)
r_eq       = C * h_cell,  h_cell = cell_volume^(1/3)   # cube-root of host-cell volume on simplices; C≈0.2 (Peaceman 0.14–0.2)
```

`Ω_geom` is computed **per host cell** from cell size `h_cell` and feature radius `r_f`, so a coarse host cell still carries the *correct steady radial conductance* to a sub-cell-radius feature — exactly what lets ~0.08 m tunnels and ~0.025 m clay barriers live in a coarse 3-D mesh **without global sub-meter refinement**. **The `h_cell` measure on unstructured tets is fixed as the cube-root of cell volume** (`cell_volume^(1/3)`) so `r_eq` — the single largest fidelity lever — is **reproducible** rather than ambiguous (edge length vs inscribed-sphere etc.); a Peaceman-anisotropic generalization using transverse cell dimensions is the documented upgrade if the embedded-vs-resolved study (F.6) demands it. **`K_soil(ψ)` is read from the live Richards state each Newton iteration at the SAME near-field `ψ` used to derive `Ω_geom`** (cell value vs `r_eq`-sampled value, fixed jointly with §E per Q4) — so the dispersion rate, doubly sensitive to the mesh-dependent near-field head (R2), is evaluated consistently. Dry clay's low `K` *and* high matric potential `ψ` both enter natively; the bidirectional behavior is **emergent, not switched**. Per-face/per-direction `σ_feat` realizes the taxonomy: **clay bottom `σ_feat=0` → sealed face (exactly zero flux)**; bare-channel sides/bottom `σ_feat→∞` → soil-limited (direction-independent, as the symmetric soil leg requires); catch-drain/pipe walls `σ_feat=0` except tagged inlet/outlet; one-way device = asymmetric (`σ_feat^receive>0, σ_feat^disperse=0`).

#### F.3.2 Entity tagging

MeshTags carry: `feature_id` on the 1-D entity set; `face_role ∈ {top, bottom, lateral}` per feature face (and, for conforming meshes, the actual 3-D facets abutting Γ_f); a `sealed` flag for clay faces (forces σ=0 in assembly); `inlet/outlet` for catch-drain/pipe endpoints; `is_vertical` for tunnels. Tags are the contract §D/§E read to know *which* `σ` and *which* coupling each entity gets.

#### F.3.3 Vertical tunnels

A tunnel is a feature whose tangent `t̂ ∥ e_g`: axial Darcy `q_axial = −K_feat·A·∂H_f/∂s` is **gravity-driven downward natively** because `H_f` is a hydraulic head (the `z`-gradient along a vertical run supplies the drive — no extra gravity term, no special case). Lateral `σ` couples to surrounding soil along its length, and the deep endpoint tags into deeper (higher-K) soil zones to drain down. Same embedding machinery; only orientation and endpoint tag differ. *(Deep-endpoint boundary — free-drainage internal BC vs prescribed deep water table vs coupling into a deeper high-K zone — is Q20.)*

#### F.3.4 Fidelity trade-off vs C-001 / C-004

The embedding decouples conduit radius from cell size, so **C-001 (lateral conveyance)** is carried by `K_feat·A_f` axial transport regardless of mesh coarseness, and **C-004 (conveyance ≫ percolation)** is the ratio of axial flux to integrated `σ`-exchange. The fidelity cost: near-field radial gradients within `r_eq` are *not resolved* — they are represented by the steady `Ω_geom`. This is accurate when the feature is near radial-steady locally; it under-resolves sharp transient fronts at the feature wall. Tier-1 quantifies the error against a fully-resolved 3-D reference and reports convergence of the embedded solution to it.

### F.4 Topography

`build_surface(spec)`: synthetic analytic surfaces now (planar slope, tilted-V, Gaussian-hummock fields) returning a `z(x,y)` callable used to warp the box top and define overland elevation. A thin **DEM/LIDAR adapter** `dem_to_surface(geotiff) -> z(x,y)` (rasterio → interpolant → gmsh terrain surface → extruded/clipped volume) is stubbed now, contracted later from `aerial-mapping`. A **DEM round-trip** test (sample synthetic z → mesh → re-sample) guards the ingestion path.

### F.5 SE-Piedmont subsurface property generator

`generate_properties(mesh, profile_spec) -> dict[Function]` producing `K_s, θ_s, θ_r, α, n` as P0/P1 `Function`s. Properties are a **depth function** (Piedmont saprolite: organic/clay-rich A–B horizon → weathered saprolite → partially-weathered rock; K decreasing with depth, anisotropic) plus **material zoning via MeshTags** for layers and clay barriers. Parameters come from the `parameterization` neighbor (grain-size + sorting → Kozeny–Carman/Hazen + Rosetta pedotransfer). For granular features the **same grain-size → K** path yields `K_feat`, keeping feature and matrix on one parameter basis. Anisotropy/depth-decay are analytic functions at cell centroids; clay zones overwrite to `K≈0, θ_eff≈0`. **`K_s` is emitted as a scalar by default but as a diagonal tensor `K̄_s = diag(K_h,…,K_v)` when the profile spec sets anisotropy** (R12/Q19), matching §B's tensor-ready flux so enabling Kh/Kv is a generator-output swap. **Pending external parameterization (Q23/Q25):** until `parameterization` delivers the SE-Piedmont saprolite K-depth/van-Genuchten set, the generator ships a **named provisional default** (regional Rosetta saprolite class + a literature Kh/Kv ≈ 3 anisotropy) so M1's property and ET-sink gates can run; the default is flagged for replacement, not silently authoritative.

### F.6 Three-tier sanity plan

- **Tier 1 (pytest, TDD):** (a) generated profile reproduces the specified K/porosity/retention **depth function** at sampled depths (scalar and, when set, tensor `K̄_s`); (b) **DEM round-trip** error < tol; (c) **embedding resolves a feature without global refinement** — embedded-vector conveyance + bidirectional exchange vs a **fully-resolved 3-D reference** (feature meshed at true radius) agree within tolerance, **with a convergence study in `h_cell` (= cell_volume^{1/3})** showing convergence under a **single a-priori `C` and a single `h_soil` sampling rule across ≥2 geometries (horizontal french drain AND vertical tunnel) and both flow directions** — a per-geometry/per-direction refit is a flagged FAIL (the C-004 gate, shared with E.6); (d) **clay-sealed face passes exactly zero flux** (structural `σ_feat=0`); (e) storage = θ_eff·volume; plausibility (positive K/porosity, `θ_r<θ_s`, no NaN, tags partition the domain).
- **Tier 2:** steep vs flat surfaces; layered vs homogeneous profiles; sparse vs dense feature networks (conforming vs immersed embedding); saturated vs bone-dry antecedent to exercise the `K(ψ)`-dependent `σ` both as drain and as disperser; mass-conservative property/embedding behavior in concert with subsurface.
- **Tier 3 (viz subagent):** 3-D + transect property field (K/porosity); mesh with embedded feature vectors highlighted by `feature_id`/`face_role`; side-by-side embedded-vs-resolved comparison with the convergence metric. Emits NetCDF; human sign-off.

---

## G. Module 6 — Forcing / vegetation + boolean erosion threshold

Module 6 is a **pure forcing/boundary-data generator and a post-processor**: no PDE state of its own. It produces (a) the time-dependent atmospheric flux driving the overland surface (§C) and the Richards top BC (§B), (b) a depth-distributed ET sink for the subsurface residual, and (c) a post-solve boolean channel-damage flag. All outputs are pure functions of `(t, state queried read-only)`; Module 6 never assembles into the monolithic Jacobian except via the ET sink's linearization, exposed as a callable so **coupling** (§D) owns the actual residual entry.

### G.1 Synthetic rainfall — NOAA Atlas 14 design storms (SE Piedmont)

Parameterized by **depth–duration–frequency (DDF)** triples from NOAA Atlas 14 Volume 2 (SE Piedmont; representative grid point, e.g. Charlotte/Raleigh NC). Tables **vendored as static JSON** (`forcing/atlas14_se_piedmont.json`: `depth_mm[duration][return_period]` + 90% confidence bounds + source grid-cell/ARI/duration provenance) — no runtime network, deterministic. The JSON table stays general (durations 5 min … 24 h; return periods 1 … 1000 yr) but the **first cut implements only what the three scenarios need**.

A design event is built from a **total depth `D`** for a chosen `(duration, ARI)` and a **temporal pattern** mapping the dimensionless mass curve `P*(t/T) ∈ [0,1]`:

```
intensity i(t) = D * dP*/dt(t/T) / T          [mm/h]
```

**Implemented now: NRCS Type II + a nested 5-min peak + the drought factory.** The Piedmont uses Type II; **Type III and the Huff quartile menu are deferred** (added only when a sensitivity case actually needs them — YAGNI for a forcing module whose Tier-2 matrix is fixed), as are return periods beyond 100 yr.

The **100-yr design storm** is a 24-h NRCS-II nested storm whose embedded **5-min peak intensity equals the Atlas-14 5-min/100-yr depth**, guaranteeing a genuine **sub-hourly burst (i ~ 100–250 mm/h)** that exercises the adaptive `dt` controller (§H). **Nesting consistency (LOCKED, resolves Q22):** the storm is nested to a **single consistent ARI across all durations**, *not* max-per-duration alternating-block depths — otherwise the 24-h total can exceed the 24-h/100-yr depth (double-counting rarity) and a mis-nested storm would confound solver-failure blame (R5). This is the internally-consistent storm that drives every Tier-2 extreme and the adaptive-dt stress test. The generator emits a recommended `dt_max(t)` envelope (small near the burst) as a hint to the driver.

**Three canonical scenarios** (named factory functions):

| Scenario | Depth/forcing | Purpose |
|---|---|---|
| `typical_moderate` | ~2-yr, 6-h, NRCS-II | baseline infiltration/runoff partitioning |
| `extreme_100yr_storm` | 100-yr 24-h, NRCS-II, nested 5-min peak | adaptive stepping, infiltration-excess runoff, PIDS conveyance |
| `extreme_100yr_drought` | zero rain ≥ 90 d + PET at 90th-pct summer | monotone drying, ET/root closure |

### G.2 Coupling into the solve — atmospheric BC

Rainfall enters as a **time-dependent Neumann flux** on the overland surface (positive = into domain). Module 6 returns a callable `flux_atmos(t) → q_atm [m/s]`. Where ponding exists, the rain joins the overland depth source; over bare soil it is the Richards top flux, switching to a **head/seepage (Signorini-type) condition** when the surface saturates — that switch is owned by **§D**; Module 6 supplies only the *potential* (precip − evaporation) flux magnitude. PET evaporative demand is a **negative** component of the same flux, capped by available surface water and soil-limited supply (the soil-limit enforced by §D/§B, not here).

### G.3 PET and static-root ET sink

PET uses **Hargreaves** (temperature-based, fewest synthetic inputs) in the **first cut** — it satisfies the only ET-related Tier-1 gates (mass closure of the sink, monotone drought drying), so the full **Penman–Monteith (FAO-56)** input chain (synthetic `T_air, RH, u, Rs`) is **deferred until a scenario needs energy-balance fidelity**. Drought scenarios scale the Hargreaves temperature range / PET to a 90th-percentile summer envelope. Actual ET splits into **soil evaporation** (top atmospheric flux) and **root transpiration**, applied as a **depth-distributed volumetric sink `S(z)`** in the Richards residual for a single static PFT:

```
S(z,t) = T_pot(t) * beta(z) * alpha(psi(z))           [1/s, volumetric]
beta(z): normalized root density, integral(beta dz)=1  (exponential/Jackson, fixed depth d_root)
alpha(psi): Feddes stress reduction in [0,1]           (anaerobiosis + wilting limits)
```

Mass closure is exact by construction: `∫ S dV = T_pot·∫β·α dz·A`, and the **withdrawn transpiration is logged** so the Tier-1 mass-balance check sees ET as a quantified sink (`|ΔStorage − net_flux + ET_uptake| < tol`). `α(ψ)` makes the sink self-limit in dry soil (drought → monotone drying, no over-extraction). Module 6 exposes `S(ψ, z, t)` **and its derivative `∂S/∂ψ`** so §D can place it in the monolithic Jacobian (smooth → no VI). `α(ψ)` must be **C¹-smoothed** (no sharp kinks) to preserve Newton convergence near wilting.

### G.4 Boolean erosion / channel-damage threshold (post-processor)

A **post-solve, non-feedback** check — no sediment transport, no state coupling. **Two physically-distinct branches** (the open-channel shear law has no meaning for porous-fill flow — no free-surface boundary layer):

**Overland sheet-flow branch** (per surface cell):

```
v   = |overland velocity|                          [m/s]
tau = rho*g*R_h*S_f   ≈ rho*g*d*S_local            (wide-channel bed shear)
damaged = (v > v_crit_overland) OR (tau > tau_crit_overland)
```

**Embedded granular-fill branch** (per PIDS channel segment) — based on **seepage velocity / hydraulic gradient**, NOT a bed-shear law:

```
v_seep = Q_axial/(w*d)                  [m/s]   # Darcy-through-fill seepage velocity
i_hyd  = |dH_f/ds|                               # axial hydraulic gradient (pore-scale shear surrogate)
damaged = (v_seep > v_crit_fill) OR (i_hyd > i_crit_fill)   # granular-fill thresholds, sourced distinctly
```

`S_local` = local bed/topographic slope (from §F), `d` = flow depth. **Thresholds are sourced per branch and per material** — granular-fill `v_crit_fill/i_crit_fill` (internal-erosion / piping criteria) are distinct from open-channel `v_crit/τ_crit` and from bare-clay thresholds (Q24). Output: a boolean field + the scalar **fraction/length of channel flagged**, time-resolved, written to IO. It runs on cached fields, so it never affects convergence.

### G.5 Three-tier sanity plan

**Tier 1 (`tests/test_forcing_*.py`, TDD-first):** *Atlas-14 match* (generated cumulative depth reproduces the vendored target; peak 5-min intensity of `extreme_100yr_storm` ≥ Atlas-14 5-min/100-yr depth); *pattern integrity* (`∫ i(t) dt = D`, monotone cumulative for the NRCS-II pattern; same check applies to deferred patterns if/when added); *ET/root closure* (`∫S dV` = logged transpiration `<1e-6`; `α(ψ)→0` at wilting); *drought monotonicity* (zero-rain + PET ⇒ `dStorage/dt ≤ 0`, `θ` monotonically non-increasing); *threshold logic* (overland branch flags exactly where `v>v_crit ∨ τ>τ_crit`; granular-fill branch flags exactly where `v_seep>v_crit_fill ∨ i_hyd>i_crit_fill`; `d=0` ⇒ not damaged); *plausibility* (`i≥0, PET≥0, α∈[0,1]`, no NaN/Inf).

**Tier 2:** drive the assembled overland+subsurface(+features) system with `typical_moderate`, `extreme_100yr_storm` (burst must trigger adaptive sub-stepping + infiltration-excess runoff), and `extreme_100yr_drought` (water table recedes, ET sink self-limits). Run **alone** (forcing→BC only), **+overland**, **+subsurface (ET sink)**, **full-system**; record couplings exercised.

**Tier 3 (separate viz subagent):** hyetograph + cumulative-depth chart with Atlas-14 target band; PET/ET time series; an **in-channel velocity/shear map with damage-threshold exceedance highlighted** (time slider); drought soil-moisture decline transect. Emits NetCDF/`.npz`+JSON; human signs off.

---

## H. Cross-cutting numerics: time integration, smooth-monolithic Newton, VI scope, preconditioning, parallelism

This is the **single source of truth** for time integration, the nonlinear/linear solve, preconditioning, parallelism, and reproducibility. Every physics module references these contracts rather than re-specifying them. **Solver state is owned by the driver/IO module (§I);** physics modules contribute residual/Jacobian UFL forms only. All examples honor the locked smooth-bidirectional-headline / VI-reserved physics and the DOLFINx 0.10 `petsc_options_prefix` requirement.

> **Minimal-viable solver path vs deferred hedges (per the locked decision).** The **first cut ships exactly:** backward-Euler (BE) + `newtonls`/`bt` line search + `dt`-cut-and-retry + (`preonly`/LU small | `gmres`+`fieldsplit` 3-D). Everything below labelled **DEFERRED-UNTIL-TRIGGERED** is an escape hatch implemented only when its named Tier-1/Tier-2 trigger fires — it is documentation of insurance, **not** M0–M5 scope: **BDF2** (trigger: Tier-1 `dt`-error convergence demands >1st order), **pseudo-transient continuation `ptc`** (trigger: bone-dry / cold-start Newton divergence), **`cp` critical-point line search** (trigger: wetting-front stall `bt` cannot clear), **homotopy-on-`σ`** (trigger: a feature face is the divergence culprit), **local-inertial overland** (trigger: flat-reach diffusion-wave stiffness, §C), **SUPG/entropy-viscosity** (trigger: under-resolved gravity-dominated fronts, §B). None is built until triggered.

### H.1 Time integration

**Scheme.** Implicit backward-Euler (BE) is the **shipped default and the only time integrator in the first cut** (resolves Q17 in favour of BE-only-first). BE is L-stable, mandatory for the stiff Richards operator and the high-K embedded-feature axial term (conductivity contrast 4–6 orders). **BDF2 is DEFERRED-UNTIL-TRIGGERED** (trigger: Tier-1 `dt`-error convergence shows BE's 1st-order temporal error is the binding accuracy limit); when added it is started/restarted with one BE step and **auto-demoted to BE** on any `dt` change > 10% or after a Newton failure. Mixed-form Richards storage is differenced as `(theta(psi^{n+1}) − theta(psi^n))/dt` (Celia 1990) so the conservation test closes `<1e-6` independent of `dt`.

**Adaptive dt policy.** Two independent controllers run each step; take the **smaller** proposal:

1. *Newton-iteration target.* `it_target = 3..6`: `it≤3` → `dt ← min(dt·1.5, dt_max)`; `4≤it≤6` → hold; `it>6` or no convergence → `dt ← dt·0.5`, **retry the step** (do not advance).
2. *Local truncation error (LTE).* `e = ||y_BDF2 − y_BE|| / scale` (or step-doubling when BDF2 off); PI controller `dt_new = dt·(tol_lte/e)^0.5` clamped to `[0.5, 2.0]` per step, `tol_lte ~ 1e-3` on saturation-equivalent units.

Bounds: `dt_min = 1 s` (repeatedly hitting it is a **hard error**, not a silent clip), `dt_max = 86400 s`. A **forcing-aware cap** prevents averaging-out of a sub-hourly spike: `dt` never steps over a hyetograph breakpoint exposed by §G.

```
solve_step(dt): attempt BE Newton
  if not converged: dt *= 0.5; if dt < dt_min: raise; else retry
  else: it -> ctrl_iter; LTE -> ctrl_lte
        dt_next = min(ctrl_iter, ctrl_lte, breakpoint_cap, dt_max)
```

### H.2 Nonlinear solve (headline: smooth monolithic Newton)

Per the corrected physics, the per-face exchange `q = σ·(H_f − H_soil)` (hydraulic heads) with fixed/bidirectional `σ`, the axial Darcy conveyance, and the land-surface exchange `q_ls = k_ex·(d − ψ_top)` are all **smooth** potential-driven fluxes. The headline coupled solve is therefore a **single smooth PETSc SNES Newton + line search** over `F(U)=0`, `U = [ψ_soil, d, H_f]`, assembled by §D into one residual and one Jacobian per timestep.

- **SNES type:** `newtonls` with `bt` (back-tracking) line search (shipped). `cp` (critical-point) line search is a **DEFERRED-UNTIL-TRIGGERED** wetting-front-stiffness fallback.
- **Jacobian:** UFL automatic differentiation, `J = ufl.derivative(F, U)`. No hand-coded or matrix-free FD Jacobian on the headline path; the mixed-dimensional off-diagonal coupling blocks must be **exact** for Newton quadratic convergence.
- **Tolerances:** `snes_rtol=1e-8`, `snes_atol=1e-10`, `snes_stol=1e-8`, `snes_max_it=25`; residual scaled by storage capacity so atol stays meaningful across wet/dry.

```python
opts = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_rtol": 1e-8, "snes_atol": 1e-10, "snes_stol": 1e-8, "snes_max_it": 25,
    "snes_monitor": None, "snes_converged_reason": None,
    # ... linear-solve options from H.3 ...
}
# DOLFINx 0.10: NonlinearProblem REQUIRES keyword-only petsc_options_prefix
problem = NonlinearProblem(F, U, bcs=bcs, J=J,
                          petsc_options_prefix="pids_richards_",
                          petsc_options=opts)
```

**Divergence fallbacks, in order** (all but the first are DEFERRED-UNTIL-TRIGGERED): (1) **dt-cut-and-retry** (H.1) — shipped, the only first-cut divergence response; (2) **pseudo-transient continuation** (`snes_type ptc`, or relaxed BE with an inflated storage term ramped down) for bone-dry-antecedent / 100-yr-storm cold starts; (3) **homotopy on `σ`** (ramp exchange conductance from 0) only when a feature face is the culprit. Each fallback logs the converged-reason code for the determinism record.

**VI scope (reserved, NOT the core).** `snes_type vinewtonrsls` (spike-confirmed, PETSc 3.25.1, non-trivial active set) is used in exactly two cases, set via `setVariableBounds(lb, ub)`:

1. **Overland depth positivity** `d ≥ 0` (ponding depth bound on the surface block) — the dominant VI use.
2. **Optional hard one-way devices** — the degenerate asymmetric-`σ` limit, insurance per `catchvalve_probe` (implicit hook tractable; non-iterated operator-splitting disqualified).

The default **bidirectional** feature uses smooth `newtonls`, NOT VI. When VI is active the same UFL `F, J` are reused, only bounds vectors added; bounds default to `[−INFINITY, +INFINITY]` so one code path serves both regimes.

### H.3 Linear solve + preconditioning

Each Newton iteration solves `J δu = −F`, a 3-field block system (subsurface / surface / feature).

- **Small / 1-D / MMS / Tier-1 / determinism reference:** `ksp_type preonly`, `pc_type lu` (`pc_factor_mat_solver_type mumps`) — exact and reproducible, the spike's Poisson path.
- **Production 2-D/3-D:** `ksp_type gmres` (restart 100, `ksp_pc_side right` so the residual norm is preconditioner-independent for the tolerance test) with **PCFIELDSPLIT**.

**Fieldsplit + conditioning.** High-K embedded features create a large conductivity contrast, and the mixed-dimensional 1D-in-3D coupling injects off-diagonal blocks into rows otherwise the sparse Richards stencil — a badly-scaled, saddle-point-flavored system. **§H.3 is the single source for the fieldsplit recipe and field indices** (`{ψ=0, d=1, h_feat=2}`); the two-field `{ψ, d}` split is the only one that must work at **M4**, while the three-field `{ψ, d, h_feat}` tuning is an **M5 deliverable contingent on the R3 conditioning study**. A `multiplicative` (or `schur` for the surface–subsurface pair) fieldsplit isolates the well-behaved subsurface block for AMG and keeps the small feature block on a direct factor. **The options block below is an explicitly-labelled STARTING POINT for that conditioning study (Q9), not a tuned recommendation — the specific thresholds are inputs to the study, not its conclusion:**

```python
lin = {
    "ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_gmres_restart": 100, "ksp_pc_side": "right",
    "pc_type": "fieldsplit", "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_0_fields": "0",  # subsurface (Richards)
    "pc_fieldsplit_1_fields": "1",  # overland
    "pc_fieldsplit_2_fields": "2",  # embedded features
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "hypre", "fieldsplit_0_pc_hypre_type": "boomeramg",
    "fieldsplit_0_pc_hypre_boomeramg_strong_threshold": 0.7,  # 3-D anisotropy
    "fieldsplit_1_pc_type": "hypre", "fieldsplit_1_pc_hypre_type": "boomeramg",
    "fieldsplit_2_ksp_type": "preonly", "fieldsplit_2_pc_type": "lu",  # tiny feature block
}
```

Conditioning notes: (a) **non-dimensionalize / row-scale** the feature axial conductance against the soil block so AMG strong-threshold heuristics are not fooled by the K-contrast; (b) if BoomerAMG stalls on the strongly anisotropic near-saturation Richards operator, fall back to `pc_type gamg` or `pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi`; (c) the off-diagonal feature coupling stays in the *true* Jacobian but may be **dropped from the preconditioner** (`fieldsplit multiplicative` already approximates this) — Newton keeps the exact `J`, so convergence order is preserved while the preconditioner stays cheap. A linear-iteration blow-up is the first symptom of the K-contrast conditioning risk; track `ksp_converged_reason` and iteration counts in the sanity report.

### H.4 Parallelism hedge

MPI is the only parallelism wired now and is left **un-optimized**: meshes built distributed (`mesh.create_*(MPI.COMM_WORLD, ...)`), assembly/solve run rank-parallel under PETSc domain decomposition, reductions via `comm.allreduce`. No load-balancing of the embedded 1-D features across ranks (a feature vector may straddle partitions; **§D/§E must `scatter`/`gather` its DOFs — flagged, not solved**). GPU is **deferred** (`vec/mat_type cuda` + `pc_type gamg` is the eventual switch, not built). MPI *correctness* (mass balance identical at 1 vs N ranks) is asserted in Tier-1; MPI *performance* is out of scope.

### H.5 Reproducibility

Fixed inputs + fixed seeds yield identical results on a fixed rank count, asserted in Tier-1: (1) all RNG seeded from a single config seed; (2) the LU/`preonly` path is the determinism reference (iterative tolerances make GMRES reproducible only to `ksp_rtol`, so determinism tests pin `pc_type lu`); (3) PETSc options captured to the sanity report via `-options_left` / `PETSc.Options().view()`; (4) `petsc_options_prefix` is mandatory (DOLFINx 0.10) and **unique per solver instance** so concurrent solvers never collide in the global options DB; (5) MPI determinism asserted at fixed rank count only.

> **Conservation-gate parallel caveat (removes an implicit over-promise).** The `<1e-6` mass-balance gate is a **single-rank, LU/`preonly` deterministic reference**. At `N>1` MPI ranks, parallel reduction order perturbs the last bits, so the gate is asserted as "**identical mass balance to within reduction-order roundoff**", not bit-stable `<1e-6`. The single-rank LU path is the bit-stable reference against which the N-rank result is compared; do not read the document's `<1e-6` statements as a bit-stable *parallel* guarantee.

---

## I. V&V strategy, Module 7 (driver/IO), result contract, testing/CI

This is the verification-and-validation backbone, the orchestration/IO module, the solver→viz result contract, and the CI scaffolding that gates every other module. It assumes the smooth monolithic-Newton core (`q = σ·(H_f − H_soil)`, hydraulic heads); PETSc SNES-VI is invoked **only** for overland depth-positivity and optional asymmetric one-way devices, so the V&V harness must exercise **both** the smooth-Newton path (default) and the VI path.

### I.1 Per-module Tier-1 analytical/benchmark map

| Module | Spatial/temporal MMS | Closed-form / benchmark | Conservation & invariants |
|---|---|---|---|
| B subsurface | mixed-form Richards MMS (**consistent-mass variant** for order; lumped for production) → assert order | Philip (early-time series), Green–Ampt sharp front, Celia-1990, Gardner steady column, hydrostatic zero-flux | global+local (θ + `Ss`) mass; `S∈[0,1]`; bounded ψ |
| C overland | diffusion-wave MMS | kinematic-plane hydrograph, Stoker dam-break, tilted-V | depth `d≥0` (VI path), volume balance |
| D coupling | coupled MMS w/ exchange source | tilted-V + superslab (vs HGS/ParFlow, **optional**) | interface flux antisymmetry: `q_surf→sub = −q_sub→surf` exactly |
| E pids-features | embedded-Darcy MMS (1D-in-3D vector, hydraulic-head `H_f`) | axial high-K Darcy conveyance vs analytic; **bidirectional exchange vs locally-resolved 3-D reference** (drain below WT; bare feature dispersing into dry clay at the ψ-driven rate) — passes only with a **single a-priori `C` + `h_soil` rule across ≥2 geometries and both directions** (else flagged, not passed); **clay face → zero flux** (`σ_feat=0`, structural); water-table interception; storage `= φ·(w·d)·L`; optional asymmetric device never leaks blocked direction | per-segment storage + exchange closes |
| F domain | — | property-depth-function reproduction; DEM round-trip; **embedding resolves a feature without global refinement** | geometry/embedding fidelity |
| G forcing/veg | — | Atlas-14 design-storm depth match; PET/root-uptake closure; drought monotone drying | forcing-volume = ∫intensity·dt |

The embedded-feature checks (E) are the novel rows: the **bidirectional-exchange reference test** runs the same feature as (a) an embedded 1-D vector with `σ`-exchange and (b) a fully 3-D-resolved high-K inclusion on a refined mesh, and asserts cumulative exchange flux and storage agree within tolerance **across both flow directions** — this is the validation of the core PIDS representation, not the retired catch-valve.

> **Governance-menu reconciliation (REQUIRED — see R17).** The current `governance/claude-sanity-check-routine.md` appendix still lists the retired pids-features menu ("Catch-valve: never leaks outflow + self-limits to `H=h_surf`; … barrier blocks vertical flux; pump = prescribed sink + paired source") and a **mandatory-sounding Tier-1 plausibility invariant** ("the catch-valve never leaks outflow"). These must be updated to this document's menu: (1) replace the catch-valve/pump rows with the **bidirectional-exchange-vs-3D-reference** checks above; (2) **re-scope the mandatory catch-valve invariant to "optional one-way device, *when present*"** (a catch-valve is not always required); (3) **re-scope the catch-valve/pump rows to the unified embedded-vector checks** — pumps and gutters are in scope as vector-feature parameterizations (executive-summary taxonomy note), with their detailed benchmarks deferred until they are parametrized. Until that edit lands, this table — not the appendix — is the authoritative pids-features benchmark menu.

### I.2 Shared MMS convergence-order harness (`tests/analytical/mms.py`)

One reusable harness drives every module's order test. Given `u_exact(x,t)`, it derives the forcing `f = R[u_exact]` symbolically (UFL/`ufl.replace` or sympy→UFL), runs the **dimension-agnostic** residual on a refinement sequence, and regresses `log‖e‖` vs `log h` (and vs `Δt`):

```python
def assert_convergence_order(build_residual, u_exact, refinements,
                             norm="L2", expected_order=2.0, tol=0.3,
                             refine="space"):
    errs, hs = [], []
    for h in refinements:                       # same UFL on 1-D, 2-D, 3-D meshes
        uh = solve(build_residual, mesh(h), u_exact)   # f injected as MMS source
        errs.append(error_norm(uh, u_exact, norm)); hs.append(h)
    p = polyfit(log(hs), log(errs), 1).slope
    assert abs(p - expected_order) < tol, f"order {p:.2f} != {expected_order}"
    return p   # logged into the sanity report
```

`build_residual` is the module's own UFL form (dimension-agnostic; only gravity is dim-aware), so the harness is module-blind. The shared **analytical-solutions library** `tests/analytical/` provides callables: `philip(t)`, `green_ampt(t)`, `gardner_column()`, `kinematic_plane()`, `stoker_dambreak()`, `darcy_axial(L,K,Δh)`, `embedded_exchange_reference(...)`. Convergence orders and the mass-balance ratio `|ΔS − net_flux − sources|/scale` (target `< 1e-6`) are returned for the report.

### I.3 PIDS-MIN-1 cross-code benchmark (OPTIONAL / LATER)

Per the 2026-06-04 decision the three-tier routine is sufficient; cross-code (HGS) benchmarking is **deferred**, not part of the build gate. It slots in as a Tier-2-plus *scenario*, not a new tier: a fixed config `cases/pids_min_1.toml` (single bare/clay channel on an SE-Piedmont slab, design-storm forcing) the driver already runs. When a licensed HGS reference exists, a thin adapter `validation/crosscode/hgs_adapter.py` ingests HGS output into the **same NetCDF result contract** (I.5) and an opt-in test `test_crosscode_pids_min_1.py` asserts agreement on the two patent-bearing quantities — **C-001 lateral conveyance flux** and **C-004 conveyance ≫ percolation ratio**. Marked `@pytest.mark.crosscode @pytest.mark.skipif(no HGS)`; never blocks CI. No solver change is needed to add it later because the case and the contract already exist. **HGS = gold-standard reference, NOT the production engine.**

### I.4 Module 7 — driver/IO

Three responsibilities, no physics:

1. **Orchestration.** `assemble(config) → Problem`: select modules (B–G), instantiate the embedded-feature layer from per-feature records, build the **single monolithic residual + Jacobian**, then run the adaptive backward-Euler loop calling one SNES solve per step (VI variant auto-selected when overland or an asymmetric device is active). The time-integration contract is owned by §H; the driver executes it.
2. **Declarative config** (`cases/*.toml` — **TOML locked**, superseding any §A YAML reference): mesh/domain ref, material zones, forcing ref, solver options (incl. `petsc_options_prefix` per the 0.10 API), and a **`[[feature]]` array** — each record carries `kind` (bare|clay|tunnel|french|catch|pipe), polyline vertices, `K_feat`, cross-section `w×d`, `phi_eff`, and the **per-face `σ_feat` table** (top/bottom/lateral × receive/disperse), with clay encoded as `σ_feat=0` faces. Config is the single source of truth a scenario is reproduced from.
3. **Standardized result output** (I.5): the driver, not the solver kernels, writes NetCDF/npz — decoupling the solver from viz.

```python
problem = assemble(load_case("cases/pids_min_1.toml"))
for step in adaptive_time_loop(problem):
    problem.solve_step(step)          # one monolithic SNES solve
    writer.append(problem.state())     # state -> result contract
writer.close()                          # emits .nc + metrics
```

### I.5 Result contract (solver→viz decoupling)

Per the viz routine: **NetCDF via xarray** with named dims/coords/units (`time, z, x, y`; vars `saturation, head, surface_depth, velocity`; feature vars `h_feat`/`q_exchange`/`axial_flux` on a `feature`/`segment` dim, where `h_feat` is the feature **hydraulic** head `H_f`), embedded scalar diagnostics (`mass_balance_rel_err, newton_iters, dt, cum_recharge, cum_runoff`), and a `metrics` attr dict (mass-balance error, peak runoff, max Newton iters, MMS convergence order, per-feature cumulative exchange/conveyance, min throughput for optional one-way devices). **`.npz + sidecar JSON`** is the accepted small/1-D form (reused by the 1-D probe host). The reusable generator at `forward-model/viz/` reads **only** this contract and never imports the solver. *(Unstructured-mesh handling — sample-to-structured-grid/transect at write time vs UGRID NetCDF the viz generator triangulates — to be decided jointly with the viz routine; see open questions.)*

### I.6 Test / CI layout

```
technologies/infiltration-runoff-model/
  forward-model/
    pids_forward/
      mesh/        # host + surface + feature-submesh builders, MeshTags helpers   (F)
      physics/     # subsurface.py (B), overland.py (C), coupling.py (D); shared State, Protocol
      features/    # feature.py (E): axial Darcy + per-face sigma exchange + storage; FeatureRecord
      solve/       # block-space assembly, SNES/SNES-VI wrappers, adaptive dt controller   (H)
      io/          # config load, mesh/props IO, NetCDF result writer (the contract)        (I)
      viz/         # reusable generator (viz agent imports THIS, never the solver)
      config/      # declarative scenario + material + feature catalogs
    tests/
      analytical/  # mms.py + Philip, Green-Ampt, kinematic-wave, Stoker, embedded-exchange refs (shared)
      test_<module>_*.py                        # per-module Tier-1
      integration/ # test_coupling_in_concert.py, test_features_in_concert.py
      conftest.py  # markers: tier1, tier2, integration, crosscode, slow; module->couplings map
    driver/  cases/   # Module 7 + declarative case configs (cases/*.toml)
  validation/                       # SIBLING of forward-model/ (governance-mandated location)
    sanity/{*.md, viz/*.html}       # <module>__<date>.md ; viz/<module>__<check>__<date>.html
    crosscode/                      # hgs_adapter.py (optional, later)
```

> **Sanity artifacts live at `technologies/infiltration-runoff-model/validation/sanity/`** — a **sibling of `forward-model/`**, per `governance/claude-sanity-check-routine.md`, matching the existing `validation/` directory and the per-module paths used in §B.8/§C.8/§D.7/§E.6/§F.6 and the milestone governance row. They are **not** nested under `forward-model/`.

CI runs `tier1` (analytical + conservation + plausibility + solver, deterministic, fast) on **every change**. The **integration trigger** mirrors `integration-protocol.md`: a changed module re-runs its Tier-1 **and** the integration tests of every coupling it participates in — encoded as a module→couplings map in `conftest.py` so CI selects the right superset. Tier-2 synthetic-forcing runs (typical + 100-yr storm/drought + saturated/bone-dry × steep/flat) run on a nightly/`slow` lane and emit the result NetCDF + a **sanity report** (`validation/sanity/<module>__<date>.md`, governance template). Tier-3 is the separate viz subagent's HTML → human sign-off; **CI does not auto-pass it.**

### I.7 Per-module definition-of-done

- [ ] Tier-1 tests written **before** code (TDD); suite green, zero failures.
- [ ] Analytical/MMS agreement **with observed convergence order** logged (space + time).
- [ ] Global **and** local mass balance `< 1e-6`; plausibility invariants hold (no NaN/Inf, `S∈[0,1]`, `d≥0`, bounded heads; clay face zero-flux; optional one-way never leaks).
- [ ] Solver converges within iteration cap across the matrix; deterministic with fixed seed; refinement reduces error at expected order.
- [ ] Tier-2 typical **and** extremes pass, alone **and** in concert with neighbours; couplings exercised recorded.
- [ ] Result NetCDF/npz emitted to contract; viz HTML built by the **separate** subagent; **human sign-off** in the sanity report.
- [ ] Sanity report on file; residual concerns logged; integration trigger run; affected claims-register entries reset/re-verified.

---

## Viz routine (governance/visualize-sanity-check-routine.md)

Tier-3 for every module is a **separate viz subagent** that builds **ONE self-contained `.html`** (embedded JSON + **vendored Plotly.js**, fully offline, double-click on Windows), interactive (slider/hover/zoom), self-documenting (module–scenario–date, units, legends, embedded **metrics** panel), `<~10 MB`, deterministic. **Data contract:** sanity runs emit NetCDF via xarray (named dims/coords/units; vars `saturation/head/surface_depth/velocity` + feature vars + scalar diagnostics + `metrics` attr) — or `.npz`+JSON for small/1-D. The reusable generator lives at `forward-model/viz/`; **the viz agent NEVER imports the solver.** A **human signs off** before the next module builds on the gated one.

---

## Consolidated risk register

Merged and de-duplicated across all sections. **P** = probable severity if unmitigated; primary owner in brackets.

### R1 — Mixed-dimensional assembly maturity in DOLFINx 0.10 (HIGH) [A, E, F, H]

Assembling a **single residual + `ufl.derivative` Jacobian** that couples a 3-D host space, a 2-D surface space, and N 1-D feature submeshes into one PETSc nest — with **differentiable** mixed-dimensional coupling forms — is unproven; the spike confirmed `LinearProblem` + SNES-VI on a **single mesh only**. Cross-mesh forms and entity maps between Γ_f submesh and parent may hit API gaps in 0.10. If unsupported, the off-diagonal exchange Jacobian degrades to fixed-point (loses Newton quadratic convergence) or forces a fallback. **Mitigation/fallback:** (a) all unknowns on the host mesh with the feature as a co-dimension-2 line restriction; (b) the immersed line-source path for all features; (c) a focused spike on cross-mesh `ufl.derivative` before §D/§E build.

### R2 — Singular line source: near-field head sampling & the σ closure (HIGH) [E, F, H]

The 1D-in-3D coupling injects a **line source into a 3-D field**, producing a logarithmic head singularity at the centerline. The host head `H_soil` seen by `q = σ·(H_f − H_soil)` is **mesh-resolution-dependent** (it samples the regularized, not the true near-field, head), and `K(ψ)` in the soil leg inherits the same dependence — so the **dispersion rate of a bare feature over dry soil (the dominant PIDS influence mechanism) is doubly sensitive** to the near-field sampling. The `σ_soil = K(ψ)·Ω_geom` closure with `r_eq = C·h_cell` (`h_cell = cell_volume^{1/3}`, fixed in F.3.1) is derived for structured grids; on unstructured tets the Peaceman shape factor `C` may need an anisotropic generalization. **Mitigation:** the Tier-1 embedded-vs-fully-resolved-3-D **convergence study must confirm *consistency* (convergence in `h_cell`) with a SINGLE a-priori `C` and a single `h_soil` sampling rule across ≥2 geometries and BOTH flow directions** — *not* a per-mesh/per-geometry refit; `K(ψ)` must be evaluated at the **same** near-field `ψ` used to derive `Ω_geom` (Q4). If `C` must be re-tuned to match the reference, the embedding-fidelity claim for **C-004** is unsupported and that is a flagged outcome. This is the single largest fidelity lever (the gate for C-004).

### R3 — High-K contrast & saddle-point conditioning of the monolithic Jacobian (HIGH) [D, H]

High-K embedded features (4–6 orders K-contrast) plus the off-diagonal mixed-dimensional coupling ill-condition `J`, blowing up GMRES iteration counts; BoomerAMG can stall on the strongly anisotropic near-saturation Richards block. **Mitigation:** keep `h_feat` a **separate fieldsplit field** (contrast on the off-diagonal, not inside AMG); row-scale/non-dimensionalize the feature axial conductance; drop the off-diagonal coupling from the *preconditioner* while keeping it exact in `J`; `gamg`/`l1-Jacobi` fallback. Unproven on the real 3-D coupled operator — a conditioning study is required once §D exists.

### R4 — Richards numerical pathologies (HIGH) [B, H]

`C(h)→0` saturated-zone degeneracy (singular Laplace limit) without `Ss` regularization or lumping stalls Newton; **the Mualem `dK/dSe` is genuinely SINGULAR as `Se→0⁺`** (the bracket-derivative `(1−Se^{1/m})^{m−1}·Se^{1/m−1}` diverges for `n∈(1,2)`), so on a **bone-dry antecedent** `ufl.derivative` reproduces the divergence and the exact Jacobian — and the exchange-Jacobian `∂σ_soil/∂ψ` — blow up, stalling Newton (this is a derivative *singularity*, not just steepness); sharp wetting fronts are under-resolved on practical 3-D meshes (arithmetic inter-element K over-predicts flux into dry cells); small-`n` van Genuchten gives near-singular `C(h)`; mass-lumping (needed for monotonicity) **degrades MMS spatial convergence order** below 2 for P1, complicating the Tier-1 order gate. **Mitigation:** **`Se`-floor `Se∈[Se_min,1]` (B.2) — the specific fix for the dry-branch derivative singularity, bounding both `dK/dψ` and the exchange linearization** — optionally a Kirchhoff transform for the dry branch (Q16); `Ss·Se·dh/dt` regularization (physical, in the budget); lumped capacity; geometric-mean/upwind K; a consistent-mass variant kept **solely** for MMS order verification; pseudo-transient continuation for cold starts.

### R5 — Adaptive-dt vs VI active-set interaction & sub-hourly bursts (MEDIUM-HIGH) [C, H, G]

The two-controller dt policy (Newton-count + LTE) can fight/chatter near a 100-yr sub-hourly Atlas-14 spike, repeatedly hitting `dt_min` (hard error); the wet/dry SNES-VI active set can chatter as many cells flip per step, slowing/failing Newton; a **rejected step must cleanly restore `u` from `u_n` including all feature blocks** or state drifts. The `dt_max(t)` forcing hint is advisory — if the driver ignores it the 100-yr Tier-2 run may fail for solver, not forcing, reasons (confounding blame). **Mitigation:** forcing-aware breakpoint cap; airtight State/checkpoint discipline; well-balanced face fluxes; possible active-set damping; empirical tuning of `it_target` and LTE gain per scenario.

### R6 — Conservative local mass balance of the surface & coupling discretizations (MEDIUM) [C, D, E]

P1 CG diffusion-wave is **not locally conservative** in the FV sense; the strict local `<1e-6` gate may require a post-processed conservative flux or an RT/DG variant (a possible later formulation change). Mass conservation of the pairwise exchange depends on §D distributing `−q_exch` into **exactly** the pierced host cells with matching quadrature; a feature-side/host-side quadrature mismatch, or a `P^T` sign/indexing error, would still close **globally** while violating **local** closure without an obvious symptom. **Mitigation:** per-interface (not just global) closure tests at per-face / per-feature granularity; the EntityRegistry as the single source of truth for field→block index; hydrostatic reconstruction for lake-at-rest.

### R7 — Well-balancedness & dry-front velocity artifacts (MEDIUM) [C, G]

Naive `grad(z_b + d)` breaks lake-at-rest; sloping-bed ponds leak spurious flux failing the 1e-6 gate. `u = q/max(d, d_min)` near the dry front can produce large/noisy velocities feeding a **false erosion-threshold exceedance**; `d_min` couples numerics to the physical §G decision. **Mitigation:** hydrostatic-reconstruction face fluxes; smoothed/projected velocity for erosion if element-wise `u` is too noisy at the front.

### R8 — ET / evaporation partitioning split across modules (MEDIUM) [G, D, B]

PET partitioning between soil evaporation (top flux) and transpiration (volumetric sink) is split across §G and §D; an inconsistent cap (both limiting the same demand) double-counts or loses ET water, breaking global mass balance. Feddes `α(ψ)` introduces a steep nonlinearity; if piecewise-linear with sharp kinks it degrades Newton near wilting, and `∂S/∂ψ` must be exact. **Mitigation:** C¹-smoothed `α(ψ)`; a single explicit owner of the evaporation cap; the ET sink logged for the mass-balance audit.

### R9 — Smoothness leaks reintroducing non-smoothness on the headline path (MEDIUM) [B, C, E, D]

The atmospheric/seepage flux-cap and seepage complementarity (§B) are kinked; the smooth `conditional` default may chatter under intense sub-hourly forcing, possibly forcing the VI path back into §B against the smooth-monolithic intent. The feature direction-selector `conditional(h_f−h_s>0, …)` is `C⁰` and silently degrades Newton if asymmetric `σ` is ever set in the default residual. **Mitigation:** route any asymmetric `σ` / hard one-way device explicitly to the VI path; keep BC switching ownership with §D; smooth penalties by default.

### R10 — MPI correctness for features straddling rank partitions (MEDIUM, hedge) [H, D, E]

A feature vector may straddle MPI partitions; submesh ghosting + the host↔feature restriction operator under MPI is **unproven** and may constrain the block layout chosen now. An incorrect halo exchange silently breaks mass balance at N>1 ranks while passing at 1 rank; parallel reduction order can perturb the last bits of the 1e-6 ratio, making the determinism assertion flaky. **Mitigation:** a §D/§E `scatter`/`gather` design (flagged, not yet built); Tier-1 asserts MB identical at 1 vs N ranks; determinism pinned at a **fixed rank count** only.

### R11 — Result-file contract on unstructured meshes & viz budget (MEDIUM) [A, I, viz]

DOLFINx 1/2/3-D unstructured meshes do not map cleanly onto rectilinear xarray `(x,y,z)`; NetCDF emission needs either sampling to a structured grid/transect or a UGRID layout the viz heatmap/animation views can consume. Full 3-D transient runs may blow past the ~10 MB viz budget; downsampling on write must not lose the per-feature segment detail the PIDS checks depend on. **Mitigation:** decide sample-to-structured vs UGRID jointly with §I + the viz routine; driver strides/downsamples on write; schema/units validation on write so the viz agent fails fast on a malformed file.

### R12 — Anisotropic / depth-decaying Piedmont properties aliasing (LOW-MEDIUM) [F, B]

Anisotropic, depth-decaying Piedmont `K` as a P0 centroid field under-resolves sharp horizon contrasts on a coarse mesh, aliasing the layer interface and corrupting recharge partitioning independent of any feature. SE-Piedmont layering may require a full **K tensor** (Kh/Kv). **Mitigation (now reflected in the governing equation):** §B.1's flux is written **tensor-ready** (`−K̄·(grad h + e_g)`, Mualem scaling on `K̄_s·k_r`), §A.5 carries the `K`/`K̄` symbols, and §F.5 emits `K̄_s = diag(K_h,…,K_v)` when the profile spec sets anisotropy — so the MMS order and conservation gates cover the tensor path; refine near horizon contrasts; source the Kh/Kv ratio from `parameterization` (provisional default named in §F.5 until then).

### R13 — Clay "exactly zero flux" represented as tiny-K leakage (LOW) [E, F]

Clay zero-flux is exact **only** if `σ_feat = 0` is a structural omission of the exchange term (then the series `σ_eff = 0` regardless of the soil leg), not `K ≈ 1e-12` with a residual conductance that leaks a small flux. **Mitigation:** the zero-flux Tier-1 test must assert structural omission; enforce `σ_feat = 0` via the `sealed` tag in assembly.

### R14 — Atlas-14 storm internal consistency & provenance (MEDIUM) [G]

A single SE-Piedmont grid point may misrepresent site rainfall (LOW; vendored JSON must record source grid cell + ARI/duration provenance). The **nesting-consistency half is MEDIUM**: NRCS-II nesting that forces the 5-min peak to the 5-min/100-yr depth can produce a 24-h total **exceeding** the 24-h/100-yr depth (double-counting rarity across durations) — and this storm drives *every* Tier-2 extreme and the adaptive-dt stress test, so a mis-nested storm confounds solver-failure blame (R5). **Mitigation (locked in G.1, Q22):** **nest to a single consistent ARI across all durations**, not max-per-duration; carry the 90% confidence band; provenance fields in the vendored JSON.

### R15 — Erosion thresholds for Darcy-through-fill vs open-channel (LOW) [G]

The in-channel erosion velocity uses `Q_axial/(w·d)` through granular fill, which is **not** an open-channel velocity; reusing an open-channel `v_crit` mis-flags. **Mitigation:** distinct, sourced `v_crit/τ_crit` for granular fill vs bare clay vs vegetated overland; OR criterion on `v` and `τ`.

### R16 — Cross-code external validation deferred (LOW, accepted) [I]

PIDS-MIN-1 cross-code (HGS) is deferred, so the **C-001 / C-004** patent-bearing conveyance numbers stay unverified by an independent engine for the foreseeable build; the three-tier routine validates internal consistency, not external correctness of the headline numbers. **Accepted** per the 2026-06-04 decision; the case + contract are pre-built so the adapter can be added later without solver change.

### R17 — Retired-framing contamination of the probe seed AND the authoritative README/sanity-routine (MEDIUM) [B, E, governance]

Two coupled hazards. (1) `scratch/catchvalve_probe.py` still references the **disqualified one-way catch-valve** as the decisive primitive; porting must extract only the Richards host and **drop that term** to avoid re-importing the retired model. (2) The **"authoritative" `forward-model/README.md` is STALE** — §3, §4.4, §7, §8 still name the catch-valve as a primitive and a divergent Module-4 scope ("channels, tunnels, barriers, catch-valve, pumps, gutters"), and the **governance sanity-routine appendix** still mandates the catch-valve invariant ("never leaks outflow + self-limits to `H=h_surf`", a *Tier-1 plausibility* line implying a catch-valve is always present) and lists a "pump = prescribed sink + paired source" benchmark. README + draft therefore disagree on Module 4's identity. **Mitigation (REQUIRED before the README is the build contract):** (a) explicit port DoD item dropping the catch-valve term; (b) **rewrite `forward-model/README.md` §3/§4.4/§7/§8** to the embedded-1D-vector framing (Module 4 = embedded 1-D-vector feature layer); (c) **re-scope the sanity-routine's mandatory catch-valve invariant to "optional one-way device, when present"** and **re-scope the catch-valve/pump rows to the unified embedded-vector checks** (pumps/gutters retained as deferred vector-feature parameterizations); (d) update the routine's pids-features menu to the bidirectional-exchange-vs-3D-reference checks (E.6/I.1). Owner: governance. The catch-valve survives only as optional one-way-device insurance, not the core.

### R18 — Field-scale (100-ha, 3-D) mesh / DOF tractability (HIGH) [F, H, D]

The M2–M5 gates all run on **small** domains, so whether an **ultimately-3-D, variably-saturated, 100-ha PIDS-network** mesh with embedded sub-meter features is computationally tractable on the spike box (24 cores / 31 GB) is **unvalidated until a field-scale Tier-2 case at M6**. The embedding decouples feature *radius* from cell size (good), but the **host** mesh must still resolve topography, horizon contrasts (R12: coarse P0 K aliases interfaces), and wetting fronts over 100 ha — and "fine enough for horizons × large enough for 100 ha" is the classic DOF blow-up. A representative network at, say, ~10–50 m horizontal / sub-metre vertical near-surface resolution plausibly reaches **10⁷–10⁸ DOF**, which may **not fit single-node (31 GB)** and would force the MPI hedge (H.4) to become *real* before M6 — the very scale problem MPI is held against but which is otherwise unstated. **Mitigation:** compute a rough DOF budget for the headline 100-ha PIDS-network case early; decide single-node-vs-MPI for M6 from that budget, not from a late surprise; keep M2–M5 on small domains but flag tractability as an explicit, unresolved field-scale risk; the number of distinct embedded features in PIDS-MIN-1 and the field-scale case (Q5) must be quantified to anchor the conforming-vs-immersed default (R1 fallback b) and the per-feature fieldsplit/MPI-straddle cost (R10).

---

## Consolidated open questions

Merged and de-duplicated; grouped by theme. Each needs a decision before the dependent module's build gate.

### Mixed-dimensional / embedding mechanics

1. **DOLFINx 0.10 mixed-dimensional support (R1).** Does it support a single residual + `ufl.derivative` Jacobian coupling a 3-D host, a 2-D surface, and N 1-D feature submeshes into one PETSc nest with differentiable coupling forms? If not fully, what is the supported subset and the fallback? *(Owner: A/H spike; gates D, E.)*
2. **Where the 1D-in-3D restriction/interpolation operator lives** — §F (domain/embedding) or shared in `solve/` — and how its **UFL representation is made differentiable** (the off-diagonal Jacobian requirement). Must be agreed across A, E, F, H. *(R1, R2.)*
3. **`Ω_geom` shape factor** — the flat-plate `P_face/ℓ` soil leg is **retired**; the radial `K(ψ)·Ω_geom` is authoritative and **`h_cell` is fixed as `cell_volume^(1/3)`** on simplices (§F.3.1), so `r_eq = C·h_cell` is reproducible. *Remaining:* the **Peaceman `C` (`≈0.2`) must be a single a-priori value** that yields convergence across ≥2 geometries and both directions (the E.6/F.6 falsifiability gate), not a per-mesh fit — and whether an anisotropic-tetra well-index is needed. **The single largest fidelity lever; fix jointly §E/§F before the Tier-1 convergence study is a verification rather than a fit.** *(R2.)*
4. **Which `h_soil` does the exchange use** — host-cell value, `r_eq`-averaged value, or reconstructed near-field head? Must be **the same near-field `ψ` used to derive `Ω_geom`** (so `K(ψ)` in the soil leg is consistent and the dispersion rate is reproducible); owned jointly §E/§F. *(R2.)*
5. **Conforming vs non-conforming embedding default** — feature-count / mesh-density threshold at which conforming gmsh-embed becomes intractable; does PIDS-MIN-1 (one channel + tunnel + barrier) use conforming throughout for a clean benchmark? *(R1, R6.)*
6. **`h_feat` datum — RESOLVED/LOCKED: hydraulic head `H_f = ψ_f + z`** (Decisions-locked; all of §D/§E/§F write hydraulic-head exchange and the native gravity drive to this). *Remaining sub-question:* is one `H_f` variable sufficient across saturated conduits (pipe, french drain below WT) and partially-saturated bare channels, or do granular features need their own `(φ_eff, K_feat)` saturation relation? *(E.)*

### Coupling / solver / preconditioning

7. **Surface unknown topology** — **working default = host top boundary facets** (used in §A.2/§C.2/§D.2; the `TermModule`/`State` contract is written to tolerate either realization); open sub-question is only whether a module test needs a **standalone 2-D submesh** instead, and the M0 contract must not pre-bind it. Affects how §D's exchange term is written. *(A, C, D.)*
8. **Default land-surface coupling** — lock first-order Robin (recommended) as production with common-node only as a `k_ex→∞` verification reference, or keep both runtime-selectable? Principled rule for `ℓ_c` (top-cell half-height vs calibrated contact thickness)? *(D.)*
9. **Fieldsplit ordering for three fields** — multiplicative `{ψ,d,h_feat}` vs Schur with the feature block as an exact inner solve — which is robust under the strongest high-K contrast at full 3-D scale, and does it survive MPI partitioning of the embedded network? Non-dimensionalize feature axial conductance globally or per-feature, and how does that interact with AMG `strong_threshold`? *(D, H; needs a conditioning study once D exists.)*
10. **Feature-block tolerance** — share the subsurface block's timestep/Newton tolerance, or does fast axial Darcy warrant a tighter inner tolerance within the monolithic solve? *(D, H.)*
11. **VI localization** — does an optional asymmetric one-way device alongside overland depth-positivity force the **whole** monolithic solve onto the VI path (global cost), or can `vinewtonrsls` confine bounds to the device DOFs only (sparse active set)? Single combined VI bound set or two staged solves? *(E, H; the spike validated each in isolation only.)*
12. **Selective VI bounds in a nest** — does `vinewtonrsls` cleanly accept `±INF` bounds on unconstrained Richards/feature blocks while bounding only surface-depth / device DOFs, and how does that interact with fieldsplit? *(A, H.)*

### Numerics / discretization

13. **Specific storage `Ss` — PROVISIONALLY DECIDED: physical term included** (`Ss ≈ 1e-5…1e-4 m⁻¹`, in the mass-balance budget; §B.3). *Remaining:* confirm the SE-Piedmont saprolite/residuum value from `parameterization`; revisit only if a study shows it acts purely as numerical regularization. *(B.)*
14. **Mass-lumping scheme** (row-sum vs nodal quadrature) and whether to keep a consistent-mass variant solely for MMS order verification — accepted order tolerance given lumping? **Inter-element K weighting** default (arithmetic/geometric/upstream), same for verification vs storm runs? *(B, R4.)*
15. **Overland element choice for local conservation** — stay P1 CG with reconstructed fluxes, or move to lowest-order RT/DG to satisfy the local 1e-6 gate natively? Exact wet/dry treatment (hydrostatic reconstruction vs depth-regularized `K_s` floor vs both)? *(C, R6.)*
16. **Atmospheric/seepage BC under 100-yr sub-hourly intensity** — is the smooth `conditional` flux-cap sufficient, or must §B adopt SNES-VI for the surface BC (and would that conflict with §D's monolithic ownership)? The **dry-branch derivative singularity is already addressed by the `Se`-floor `Se∈[Se_min,1]` (B.2, DECIDED)**; remaining is whether a full **Kirchhoff transform** for the bone-dry branch is additionally needed for the cold-start path. *(B, D, R9.)*
17. **BDF2 — RESOLVED/LOCKED: ship BE-only first**, add BDF2 only if Tier-1 dt-error convergence shows BE's 1st-order temporal error is binding (Decisions-locked; §H.1). *Remaining sub-question:* **LTE norm/scale** across saturation/head/depth fields of very different magnitudes — per-field weighting or a single storage-equivalent scale? *(H.)*
18. **`preonly`+LU (MUMPS) viability threshold** — at what problem size must the `gmres`+fieldsplit path already be validated rather than deferred? **PTC parameterization** (initial pseudo-dt, ramp schedule) for a reliable automatic fallback? *(H.)*
19. **Tensor / anisotropic K** — does SE-Piedmont layering require a full `K` tensor (Kh/Kv) in §B's flux from the outset, and do the dimension-agnostic UFL and §F generator both support it? *(B, F, R12.)*

### Domain / forcing / vegetation

20. **`Ω_geom` authoritative source** (above, Q3) plus **deep tunnel boundary** — for vertical tunnels draining to "deeper soils", is the deep endpoint a free-drainage internal BC (owned §E), a prescribed deep water table, or coupling into a deeper high-K zone (owned §B/§F)? Domain tag or coupling BC? *(E, F.)*
21. **σ granularity per face** — three faces (top/bottom/lateral) sufficient in 3-D, or do lateral faces need azimuthal subdivision (upslope vs downslope) to capture directional drainage of a french drain intercepting an inclined water table? *(E, F.)*
22. **Nested-storm rarity rule — RESOLVED/LOCKED: pin a single ARI across all durations** (internally consistent; §G.1, R14). *Remaining:* which **Atlas-14 grid point(s)** define "SE Piedmont", and whether to carry the 90% confidence band as a target band or central estimate only — **inventor/hydrology call**. *(G, R14.)*
23. **Single static PFT** — which cover (turf, native grass, mixed) sets `d_root`, `β(z)`, Feddes `ψ_wilt`/anaerobiosis limits for the SE-Piedmont site? **A named provisional default is required so Module 6's ET sink (an M1 gate) can be parameterized** — first cut: a managed-turf/native-grass profile (e.g. `d_root ≈ 0.5 m`, exponential `β(z)`, literature Feddes limits) flagged for replacement from `parameterization`. Does soil evaporation get its own surface-resistance/cap or is all PET routed to transpiration in the first cut? Drought PET seasonality (not just elevated constant) and over what window (90 d? full summer)? *(G, R8.)*
24. **Erosion thresholds** — `v_crit/τ_crit` sourced per material (granular fill vs bare clay vs vegetated overland) from `parameterization`; shear or velocity (or OR) the governing criterion for granular fill; flag on instantaneous peak, duration-over-threshold, or both? *(G, R15.)*
25. **SE-Piedmont profile parameterization source** — which saprolite K-depth and van Genuchten dataset (regional pedotransfer vs site cores) seeds `generate_properties`, and the anisotropy ratio Kh/Kv? *(F; pending from `parameterization`.)*
26. **DEM→mesh fidelity** — preserve exact topographic gradients for overland routing (conforming surface triangulation) or is a warped structured box adequate for the synthetic-now phase? Affects when the `aerial-mapping` contract must harden. **Standalone 2-D overland test mesh** — literal extracted top-facet mesh of a 3-D domain (to keep `z_b`/tags consistent) or a separately generated 2-D mesh? *(C, F.)*

### Ownership / contracts / V&V

27. **EntityRegistry ownership** — is the canonical field→block-index registration mechanism owned by §A (skeleton) or §H (cross-cutting numerics)? The "skeleton wiring" vs "solver/assembly machinery" boundary needs one owner. *(A, H, R6.)*
28. **A/H seam** — does adaptive-dt control + VI-bounds collection live in `solve/` (§H) with §A only defining the loop skeleton, or does §A own the time-loop orchestration? *(A, H.)*
29. **Embedding-reference test tolerance & norm** — relative L2 on head, on integrated axial flux, or on net exchange? Target convergence order in `h_cell`? Different tolerance for the receiving (drain) vs dispersing (imbibition into dry clay) direction? **Must be set with §H so it is falsifiable.** *(E, F, I, H, R2.)*
30. **Config format — RESOLVED/LOCKED: TOML** (`cases/*.toml`; Decisions-locked; §I.4 authoritative, §A YAML references superseded). *Remaining editorial sub-question:* per-feature `σ_feat`-tables inline (default) or in a separate referenced feature library for large networks? *(A, I.)*
31. **Result contract on unstructured meshes (R11)** — standardize on (a) sampling to a structured grid/transect at write time, or (b) UGRID/unstructured NetCDF the viz generator triangulates? Provenance/units schema-check on write so the viz subagent fails fast? Decided jointly §I + viz routine. *(A, I, viz.)*
32. **Tier-2 extreme-scenario ownership** — fixed `cases/*.toml` fixtures owned by §I, or generated by §G and merely orchestrated by §I? Does §H own the step-acceptance criterion entirely, or does §I also need a determinism harness pinning `Δt` sequences for reproducibility tests? **Fixed MPI process count + seed/partition** for the deterministic Tier-1 reference run so CI can assert bit-stable mass balance `<1e-6`? *(G, H, I, R10.)*
33. **PIDS-MIN-1 cross-code mapping** — which published superslab/tilted-V config (HGS vs ParFlow) is the canonical Tier-1 cross-code reference if a licensed code is obtained; which HGS quantities map to C-001 (lateral conveyance flux) and C-004 (conveyance/percolation ratio); at what relative tolerance is "agreement" declared? *(D, I, R16.)*

---

## Milestone roadmap

Build bottom-up; **each module is gated by the full three-tier sanity routine** (Tier-1 pytest/TDD → Tier-2 synthetic forcing → Tier-3 viz + human sign-off) before its dependents build on it. The **same dimension-agnostic UFL is validated 1-D first, then 2-D/3-D.** Module 1 (subsurface) is next.

| # | Milestone | Modules | Key gates (analytical / benchmark) | Exit = three-tier pass |
|---|---|---|---|---|
| **M0** | **Skeleton + numerics contracts** | A, H | **STANDALONE BLOCKING GATE FIRST: mixed-dimensional cross-mesh `ufl.derivative` spike (R1)** — *no contract is frozen until it passes or selects a fallback* (a) host-mesh co-dim line restriction, (b) immersed line-source-for-all; *then* `TermModule` protocol, blocked `State`, monolithic assembly stub; SNES/SNES-VI wrappers (BE-only path); adaptive-dt controller; EntityRegistry owner fixed; result-contract writer skeleton | Spike passes/selects fallback (fallback's structural consequences recorded — if immersed-for-all, §F conforming path + tunnel preference drop); trivial Poisson + 1-D toy coupled solve assemble & solve monolithically; CI `tier1` lane green; contracts frozen |
| **M1** | **Data layer (no solver)** | F, G | Property depth-function reproduction (scalar + tensor-K path); DEM round-trip; **embedding-resolves-a-feature** convergence study vs fully-resolved 3-D (R2) — **passes ONLY with a single a-priori `C` + `h_soil` rule across ≥2 geometries (french drain AND tunnel) and both flow directions; a per-geometry refit is a flagged FAIL, not a pass**; Atlas-14 depth match (single-ARI nesting); PET/root closure (Hargreaves; provisional PFT default); drought monotone drying | F + G Tier-1/2/3 pass; `Ω_geom` (`h_cell=vol^{1/3}`, fixed `C`), MeshTags, `FeatureRecord`, forcing factories, NetCDF emission all gated |
| **M2** | **Subsurface (1-D first → 3-D) — NEXT** | B | Philip, Green–Ampt, Celia-1990, Gardner, hydrostatic zero-flux; MMS order (P1≈2/P2≈3 space, BE≈1 time); MB `<1e-6`; `θ∈[θ_r,θ_s]` | Port probe mixed-form Richards (drop catch-valve, R17); 1-D gate, then 2-D/3-D conservation+plausibility; B Tier-1/2/3 pass |
| **M3** | **Overland** | C | Kinematic-plane hydrograph, Stoker dam-break, tilted-V; diffusion-wave MMS; `d≥0` (SNES-VI); lake-at-rest zero flux; velocity/shear field for erosion | C Tier-1/2/3 pass; VI depth-positivity proven on the wet/dry front; local-inertial trigger documented |
| **M4** | **Coupling (B⊕C)** | D | Coupled MMS; tilted-V + superslab; **interface flux antisymmetry**; global **and per-interface** MB `<1e-6`; infiltration-excess vs saturation-excess partitioning; conditioning study (R3) | D Tier-1/2/3 pass; Robin land-surface coupling default proven; fieldsplit `{ψ,d}` validated; HGS cross-code remains optional |
| **M5** | **pids-features (embedded vectors on the coupled host)** | E (+ F embedding, D linkage) | Axial high-K Darcy conveyance vs analytic; **bidirectional exchange vs locally-resolved 3-D reference (both directions)**; clay face zero-flux; water-table interception; storage `=φ·A·L`; optional one-way never leaks | E Tier-1/2/3 pass; the **core PIDS representation validated**; `{ψ,d,h_feat}` fieldsplit + high-K conditioning proven |
| **M6** | **Driver/IO + full-system integration** | I (wires B–G) | Per-module DoD checklist; integration-trigger map; full-system Tier-2 matrix (typical + 100-yr storm/drought × saturated/bone-dry × steep/flat); **first FIELD-SCALE (~100-ha, 3-D) Tier-2 case — the only point R18 mesh/DOF tractability is validated** (DOF budget computed; single-node-vs-MPI decided from it); result contract + viz for the coupled system | I Tier-1/2/3 pass; declarative `cases/*.toml`; one monolithic SNES solve/step end-to-end; field-scale tractability resolved (single-node or MPI made real); sanity reports on file |
| **M7+** | **(Later, optional) PIDS-MIN-1 cross-code & scale-out** | I + crosscode; H (MPI/GPU) | HGS adapter → same NetCDF contract; C-001 / C-004 agreement at agreed tolerance; MPI correctness at 1 vs N ranks | Opt-in `@pytest.mark.crosscode`; never blocks CI; optimization-loop scale TBD |

**Governance per milestone:** tests precede code (TDD); artifacts at `forward-model/tests/test_<module>_*.py` (+ `tests/analytical/`), `validation/sanity/<module>__<YYYY-MM-DD>.md`, `validation/sanity/viz/<module>__<check>__<YYYY-MM-DD>.html`; a changed module re-runs its Tier-1 **and** the integration tests of every coupling it participates in; affected claims-register entries are reset and re-verified.