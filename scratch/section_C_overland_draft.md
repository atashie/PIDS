## C. Module 2 - Overland Flow

### C.1 Formulation: diffusion-wave (default), local-inertial-ready

**Recommendation: 2-D diffusion-wave** as the production default, written so a **local-inertial (Bates et al. 2010 "LISFLOOD-FP") momentum term is a one-line UFL addition** behind a flag. Full 2-D shallow-water (SWE) is **rejected** by YAGNI: PIDS questions are runoff partitioning, ponding depth/extent, where surface water reaches PIDS top faces, and a depth*slope erosion-threshold velocity. None of these need resolved shocks/hydraulic jumps/supercritical transitions that justify SWE's cost (Riemann-aware fluxes, momentum positivity, far smaller CFL). Diffusion-wave drops local+convective inertia, leaving a gravity-friction balance that yields a **degenerate parabolic** equation - a nonlinear diffusion in surface head with a depth-and-slope-dependent diffusivity. That is the natural sibling of mixed-form Richards (same backward-Euler + Newton/SNES + monolithic assembly), reuses the probe-host time integration, and is the standard surface law in HGS/ParFlow (so the coupled Tier-1 tilted-V/superslab benchmarks are apples-to-apples). Local-inertial is held in reserve only if very flat reaches show the known diffusion-wave time-step stiffness or backwater inaccuracy.

### C.2 Surface unknown and domain

Unknown: **ponding depth `d >= 0`** on the **top boundary facets** (codimension-1 facets of the 3-D mesh; a standalone 2-D mesh for module tests). Surface (water-surface) head is `H_s = z_b + d`, with `z_b` the bed/topography elevation from Module 5 (domain). We carry `d` (not `H_s`) as the primary DOF because positivity is stated directly on `d` and storage/forcing are linear in `d`. A P1 continuous Lagrange space on the top facets is the default; `z_b` enters as a known P1 field so bed slope `grad z_b` is exact per element.

### C.3 Governing PDE (dimension-agnostic UFL)

Mass balance on the surface, with `d` the depth and `q` the depth-integrated discharge:

```
  d(d)/dt + div(q) = r - i_inf - i_chan + s_pids
```

Diffusion-wave momentum closure (Manning friction; SWE inertia dropped):

```
  q = -K_s(d) grad(H_s),   K_s(d) = d^{5/3} / ( n_man * |grad H_s|^{1/2} + eps_S )
```

so `div(q) = -div( K_s(d) grad(z_b + d) )` is a **nonlinear (degenerate, vanishes as d->0) diffusion**. Source/sink terms (units m/s, all supplied by neighbors, never re-specified here):
- `r` rainfall and PET-driven evaporation from Module 6 (forcing/veg);
- `i_inf` net infiltration to the subsurface top face - **owned by Module 4 (coupling)**, which sets it from the monolithic surface<->subsurface flux so global mass closes at the interface;
- `s_pids` / `i_chan` inflow withdrawn into surface-reaching PIDS feature **top faces** - **owned by Module 4**, evaluated from the smooth potential-driven exchange `q = sigma*(h_feat - h_soil)` of Module 4 using the surface head this module exposes (see C.7). Overland neither defines sigma nor decides direction; it only exposes `H_s`/`d` and accepts the resulting flux.

`eps_S` (small slope floor) and the `d^{5/3}` Manning conductance are the two nonlinearities Newton must linearize; both have analytic UFL derivatives so the Jacobian is assembled exactly (no numerical-Jacobian fallback like the 1-D probe). **Only the gravity/bed-slope term `grad z_b` is dimension-aware** (Module 5 supplies `z_b` for 1-D/2-D/3-D-top alike); the residual UFL is otherwise identical across dimensions per the locked dimension-agnostic decision.

### C.4 Depth positivity

Two coordinated mechanisms, matching the locked "VI reserved for overland depth-positivity" decision:

1. **Primary - PETSc SNES-VI (`vinewtonrsls`)** with **lower bound `d >= 0`**, upper bound `+inf`, the **same VI path proven in the install spike**. This is the headline enforcement for the standalone overland solve and is the overland contribution to the monolithic Newton solve. The active set is the dry region; the spike showed `vinewtonrsls` converges with a non-trivial active set, which is exactly the wet/dry front here.
2. **Supporting - degenerate-diffusion regularization for robustness:** `K_s(d)` already -> 0 as `d -> 0`, so dry cells are naturally no-flux; we add a small `eps_d` floor inside `d^{5/3}` only to keep the Jacobian non-singular, **not** to permit negative `d`. A well-balanced (lake-at-rest) discretization (C.6) ensures a flat water surface over uneven bed produces **zero** spurious flux.

Rationale for keeping VI rather than a pure penalty/clamp: clamping `d` post-solve breaks mass conservation (Tier-1 1e-6 gate); the VI enforces the bound *inside* the converged nonlinear solve so storage and flux stay consistent. This is the analogue of the probe's "implicit in-solver hook beats lagged operator-splitting" finding, now applied to the wet/dry complementarity instead of a one-way valve.

### C.5 Velocity / direction field for the erosion threshold

Module 2 must publish a **cell/facet velocity vector** because Module 7's erosion check (locked scope: boolean `depth*slope -> velocity/shear vs threshold`, NOT sediment transport) and Module 7/G's reporting consume it:

```
  u = q / max(d, d_min),     |u| = (d^{2/3}/n_man) * |grad H_s|^{1/2}
  direction = -grad H_s / |grad H_s|     (down the water-surface gradient)
```

We expose both the **vector field `u`** (for quiver viz + flow direction) and the scalar **bed shear** `tau = rho g d S_f` with `S_f = |grad H_s|` (friction slope), so the erosion module differences `tau` (or `|u|`) against its threshold without recomputing hydraulics. `d_min` guards the dry-cell division and is reported as a diagnostic, not a physical depth.

### C.6 Discretization and stabilization

- **Space:** P1 CG for `d` on top facets; bed `z_b` P1. Diffusion-wave is diffusion-dominated (no advective operator in the PDE), so the classic **SUPG-for-advection concern does not apply to the headline equation** - the stabilization need is **wet/dry front robustness and well-balancedness**, not Petrov-Galerkin upwinding.
- **Well-balanced wet/dry:** a hydrostatic-reconstruction-style treatment of `H_s = z_b + d` at facets so a still pond over sloping bed gives machine-zero flux (lake-at-rest); face conductance `K_s` evaluated with an upwind/harmonic depth so a dry downslope cell cannot draw water uphill.
- **If local-inertial is enabled,** the added momentum flux *is* advection-dominated on steep/fast reaches; there we apply **flux limiting / upwinded face discharge** (LISFLOOD-FP staggered-style), and SUPG becomes relevant - hence the flag-gated design rather than baking advection stabilization into the default.
- **Time:** backward-Euler with the project-wide adaptive controller (Module 8); diffusion-wave's parabolic stiffness on flat terrain is the main step-size driver and a documented trigger to consider local-inertial.

### C.7 Coupling exposure (defer mechanics to Module 4)

Overland **exposes** `d`, `H_s = z_b + d`, and `u` on top facets as fields Module 4 reads; it **consumes** `i_inf`, `i_chan/s_pids`, and `r/PET` as source terms in its residual. Per the corrected PIDS physics, a **surface-reaching channel's top face couples to overland**: Module 4 forms the smooth `sigma*(h_feat - H_s)` exchange and returns the withdrawal/return as `s_pids`; Module 2 supplies `H_s` and never owns sigma, the catch-drain logic, or the optional one-way (VI) device. All exchange enters the **single monolithic residual + Jacobian** assembled each step (locked decision) - no operator-splitting of the surface<->subsurface or surface<->feature flux.

### C.8 Three-tier sanity plan

**Tier-1 (pytest, TDD; tests precede code; lives in `forward-model/tests/test_overland_*.py`, analytical helpers in `tests/analytical/`):**
- *Analytical/MMS with convergence order:* (a) **kinematic-wave rising/falling hydrograph on a plane** (constant rain to equilibrium then recession; diffusion-wave -> kinematic in the steep limit) - check hydrograph shape and time-to-equilibrium; (b) **Stoker dam-break** as the inertial/front check (diffusion-wave matches the slumping profile away from the shock; documents the expected SWE-vs-diffusion-wave discrepancy at the front rather than hiding it); (c) **tilted-V catchment** (the standard surface benchmark, also the coupling on-ramp to Module 4); (d) **MMS** with a manufactured `d(x,t)` to measure spatial/temporal order (expect ~2 in space for P1 on smooth fields, 1 in time for BE).
- *Conservation:* global+local `|d(storage) - net_flux - sources|/scale < 1e-6`, storage `= integral d dA`.
- *Plausibility:* `d >= 0` everywhere (VI bound never violated), no NaN/Inf, bounded `H_s`, velocity finite; lake-at-rest gives zero flux.
- *Solver/repro:* SNES-VI converges within cap across the matrix; deterministic; refinement reduces error at the measured order.

**Tier-2 (synthetic forcing, alone + paired with Module 4/6 + full system):** typical SE-Piedmont storm hyetograph; **100-yr sub-hourly burst** (NOAA Atlas 14) to exercise adaptive stepping and the wet/dry front under intense `r`; **dry plane** (zero `d` everywhere, confirm no spurious wetting / VI active set stable); **steep vs flat** terrain (flat = the diffusion-wave stiffness stress that probes the local-inertial trigger). Acceptance: mass-conservative + plausible + expected qualitative behavior (runoff partitioning, recession limb, velocities that do/don't cross the erosion threshold as forcing dictates).

**Tier-3 (separate viz subagent; data via the xarray-NetCDF contract: `surface_depth`, `velocity` on `(time,x,y)` + metrics attr):** surface-depth **animation with velocity quiver**; **hydrograph+hyetograph twin-axis** (rain bars inverted top axis, outflow line, log-y recession option); diagnostics panel (mass-balance error vs time, Newton iters vs time, MMS order log-log). Human sign-off per the routine.
