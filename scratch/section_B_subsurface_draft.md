# Module 1 — Subsurface (3-D Richards), Dimension-Agnostic

## 1. Governing equation (mixed form)

The host continuum is variably-saturated single-phase flow governed by **Richards' equation in mixed (theta-h) form**:

```
d(theta(h))/dt + div(q) = -Q_ex        q = -K(h) ( grad h + e_z )
```

with `h` the pressure head [m] as the **primary variable**, `theta` the volumetric water content [-], `K(h)` the unsaturated hydraulic conductivity tensor [m/s], and `e_z` the unit vector along the (dimension-dependent) gravity axis. Hydraulic head `H = h + z`; flux `q` [m/s] is positive in the direction of `-grad H`. `Q_ex` is the **per-face exchange sink/source** delivered by Module 4 (pids-features) and the surface coupling term from Module 3 — Module 1 only exposes the slots and the soil-side state (`h`, `theta`, `K(h)`) that make those exchanges soil-limited; it does not specify the feature side.

**Why mixed form (modified-Picard, Celia et al. 1990), not head-based.** The head-based form `C(h) dh/dt + div q = 0`, with specific moisture capacity `C = dtheta/dh`, conserves mass only to the accuracy with which the chain-rule term `C dh/dt` reproduces the true `dtheta/dt`. Near sharp wetting fronts `C(h)` is strongly nonlinear and the head-based discretization leaks mass (documented O(10%) balance errors). The **modified-Picard / mixed scheme** linearizes the *storage* term about the current iterate while evaluating `theta` directly:

```
theta(h^{k+1}) ≈ theta(h^k) + C(h^k)(h^{k+1} - h^k)
```

so the time term is `(theta(h^k) - theta_n)/dt + C(h^k)(h^{k+1}-h^k)/dt`. At convergence the `C`-terms cancel and storage is governed by the *exact* `theta(h)` evaluated at `h_n` and `h_{n+1}` — **global mass balance is satisfied to solver tolerance regardless of timestep**, which is what the Tier-1 conservation gate (`|dStorage - net_flux - sources|/scale < 1e-6`) requires. The 1-D probe (`scratch/catchvalve_probe.py`) already runs this mixed form in finite-volume; Module 1 reuses the residual structure verbatim in FEM (the probe's `theta - theta_n` storage term and Newton loop are the seed).

## 2. Constitutive closures — van Genuchten / Mualem

```
Se(h) = [ 1 + |alpha h|^n ]^{-m},     m = 1 - 1/n      (h < 0)
theta(h) = theta_r + (theta_s - theta_r) Se,            (h >= 0: theta = theta_s)
C(h) = dtheta/dh = (theta_s-theta_r) * (-alpha n m) |alpha h|^{n-1} sign(h) [1+|alpha h|^n]^{-m-1}
K(h) = Ks * Se^L * [ 1 - (1 - Se^{1/m})^m ]^2,   L = 1/2 (Mualem)      (h>=0: K = Ks)
```

These mirror the probe's `VG.theta`/`VG.K` (loam: `theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks` from the SE-Piedmont generator in Module 5). Parameters arrive as **spatially-varying DOLFINx `Function`s** (DG0 or P1 fields), enabling material zoning — including clay zones whose `Ks->0, theta_s-theta_r->0` realize Module 4's sealed faces as a *material* property, not a runtime switch (scope lock). `K(h)` and `C(h)` are wrapped as UFL expressions of the primal `h` so the Jacobian is assembled by symbolic/AD differentiation rather than the probe's numerical Jacobian.

## 3. Primary variable & the saturated/unsaturated transition

`h` is the single primary unknown everywhere (one field, no regime switching), which keeps the saturated zone (`h>=0`, `K=Ks`, `C=0`) and the unsaturated zone (`h<0`) in **one continuous formulation** — essential because PIDS features straddle the water table (a feature below it *receives* groundwater and drains laterally; the same feature over dry soil *disperses* at the matric-potential-driven rate). The numerical hazard is `C(h)->0` as `h->0+`: the storage term degenerates and the saturated block can become a pure Laplace/incompressibility constraint. Mitigations: (a) a small specific-storage term `Ss * Se * dh/dt` added in the saturated branch (compressible-storage regularization); (b) the **mass-lumped** capacity (below) which prevents the singular consistent-mass coupling; (c) line-search / pseudo-transient continuation in the Newton solve. The smooth bidirectional exchange (`sigma` fixed) keeps the *headline* solve a smooth monolithic Newton problem; **SNES-VI (`vinewtonrsls`) is NOT used here** — it is reserved for overland depth-positivity (Module 2/3) and optional hard one-way devices (Module 4).

## 4. FEM discretization

- **Space:** continuous Galerkin `P1` (Lagrange degree 1) is the default for `h` — robust near fronts, cheap, and the natural target for mass-lumping. `P2` is available for smooth verification problems and MMS convergence-order studies (the spike confirmed P2 reproduces an exact quadratic to machine precision in DOLFINx 0.10). The dimension-agnostic UFL is written once on `ufl.SpatialCoordinate`/`FacetNormal`; element degree is a config knob.
- **Mass lumping:** the capacity (time) term is **lumped** (row-summed / nodal-quadrature) — standard for Richards to suppress the spurious oscillations and non-monotone fronts that the consistent mass matrix produces in advancing-front problems, and to keep `theta` bounded in `[theta_r, theta_s]` (Tier-1 plausibility). Implemented via a vertex/`"Lagrange"`-quadrature scheme on the storage integral while the stiffness term keeps consistent integration.
- **Stabilization for sharp wetting fronts:** (i) mass-lumping as above; (ii) **upstream / arithmetic-vs-geometric inter-element K weighting** — the probe uses arithmetic face averaging `Kf=0.5(K_i+K_{i+1})`; FEM will offer upwinded/geometric-mean conductivity as an option since arithmetic averaging over-predicts flux into dry cells; (iii) optional SUPG-like or entropy-viscosity term only if gravity-dominated fronts under-resolve. Adaptive backward-Euler stepping (Module 8 cross-cutting) controls front CFL.
- **Time:** implicit **backward-Euler**, adaptive `dt` (sub-hourly storm peaks -> daily inter-event), driven by Newton-iteration count and a front-tracking error estimate (owned by Module 8; Module 1 exposes the residual/Jacobian and a step-acceptance hook).

**Residual (UFL pseudocode):**
```python
# h: Trial/Function, v: TestFunction; theta_n, h_n at previous step; e_z dim-aware
F = ( (theta(h) - theta_n)/dt * v ) * dx_lumped \
  + inner(K(h)*(grad(h) + e_z), grad(v)) * dx \
  - q_neumann * v * ds(flux_marker) \
  + Q_exchange(h) * v * dx   # Module 4 hook; soil-side h,K(h) live here
J = derivative(F, h)         # symbolic Jacobian -> PETSc SNES
```

## 5. Dimension-agnostic detail

The **only** dimension-aware quantity is the gravity unit vector. `e_z = as_vector([0]*(gdim-1) + [1])` selects the **last spatial coordinate** as the vertical (z up). Every other term — capacity, stiffness, BC integrals, exchange — is written on `grad`, `div`, `dx`, `ds` and is identical for `gdim ∈ {1,2,3}`. The same module therefore solves the 1-D verification column (Philip, Green-Ampt, Celia, Gardner), then 2-D and 3-D meshes for conservation+plausibility, with **zero code divergence** — satisfying the locked "validate-1-D-first, same code to 3-D" decision.

## 6. Boundary & initial conditions

- **Dirichlet head** (`h = h_D`, e.g. water-table / fixed-head bottom — the probe's `H_bot=0`): `fem.dirichletbc`.
- **Neumann flux** (`q.n = q_N`, e.g. prescribed recharge / no-flow `q_N=0`): natural in the weak form via `ds`.
- **Atmospheric BC** (infiltration/evaporation-limited): a **flux-with-head-cap switching** condition — apply forcing flux `q = -p(t)` while `h` stays in `[h_min, 0]`; when the cell would pond (`h>0`) or dry past `h_min`, switch to Dirichlet (`h=0` ponding head, or `h=h_min`). Implemented as a complementarity/`conditional` term; this is the canonical seepage-type switch and is the one place inside Module 1 that *may* later borrow the VI machinery, but the default is a smooth `conditional` cap.
- **Seepage face** (`h<=0` and `q.n>=0`, complementary): exit boundary where saturated soil meets atmosphere; default smooth penalty, VI-exact optional.
- **Initial conditions:** hydrostatic `h = z_wt - z` (the probe's `psi=-z` with `H=0` is the exact zero-flux steady state and serves as the host sanity gate), or a prescribed antecedent-moisture profile (saturated / field-capacity / bone-dry) for Tier-2 antecedent scenarios.

## 7. Interface to pids-features (reference only)

Module 1 **exposes** the soil-side state at any embedded location: `h(x)`, `theta(x)`, `K(h(x))`, and the per-cell residual slot `Q_exchange`. Because exchange `q = sigma*(h_feat - h_soil)` is **series-limited by the soil's `K(psi)` and matric potential `psi=h`**, the correct soil-limited behavior (dry clay imbibing fast via high matric suction despite low `K`) is captured natively by the unsaturated closures here — Module 1 guarantees `h` and `K(h)` are evaluable and differentiable at the embedding manifold so Module 4 can assemble its smooth monolithic exchange Jacobian. Module 1 does **not** define `sigma`, faces, or feature geometry.

## 8. Three-tier sanity plan

**Tier 1 (pytest, TDD — tests precede code):**
| Check | Reference | Pass criterion |
|---|---|---|
| Early-time infiltration | **Philip** series (sorptivity S, cumulative `I=S t^{1/2}+At`) | profile L2 < tol; front advance order |
| Sharp-front infiltration | **Green-Ampt** | front position vs analytic |
| Zero-flux equilibrium | steady **hydrostatic** (`h=-z`) | residual ~0, ~1 Newton iter (host gate) |
| Mass-conservation | **Celia 1990** benchmark column | global+local `|dS - net_flux - src|/scale < 1e-6` |
| Steady columns | **Gardner** exponential soil, analytic profile | L2 vs analytic |
| Convergence ORDER | **MMS** (manufactured `h(x,t)`, forcing `f`) | spatial order ≈ P1:2 / P2:3 in L2; temporal order ≈ 1 (BE) under refinement |
| Plausibility | — | `theta∈[theta_r,theta_s]`, bounded `h`, no NaN/Inf |
| Solver | — | Newton convergence + determinism + refinement order |

Run on a **1-D mesh first** (reuse probe host), then the *same UFL* on 2-D/3-D meshes for conservation+plausibility only.

**Tier 2 (synthetic forcing, 1-D column):** typical storm; 100-yr **sub-hourly** Atlas-14 design storm; 100-yr drought (monotone drying); **saturated** and **bone-dry** antecedents; then 2-D/3-D conservation+plausibility. Expect: ponding/runoff onset when supply > infiltration capacity; monotone redistribution between events; no mass leak across `dt` spanning sub-hourly->daily.

**Tier 3 (viz, separate subagent):** `h`/`theta`-vs-depth profile with **time slider**; wetting-front transect **heatmap**; **mass-balance + Newton-iteration** diagnostics panel. Emitted as NetCDF (xarray, named dims `z,t`; vars `head,saturation`; scalar `mass_balance_residual,newton_iters`; `metrics` attr) per the data contract; viz agent never imports the solver.

**Artifacts:** `forward-model/tests/test_subsurface_*.py` (+ `tests/analytical/`); `validation/sanity/subsurface__<YYYY-MM-DD>.md`; `validation/sanity/viz/subsurface__<check>__<YYYY-MM-DD>.html`. **Seed:** `scratch/catchvalve_probe.py` (1-D mixed-form Richards host — port its residual, VG/Mualem closures, hydrostatic gate, and Newton loop into dimension-agnostic UFL; drop the retired catch-valve term).
