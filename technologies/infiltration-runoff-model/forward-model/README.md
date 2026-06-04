# Forward Model

## 1. Purpose
The 3-D variably-saturated hydrologic engine that simulates infiltration, runoff, and recharge, with PIDS features represented directly. It answers: "for this site and this PIDS configuration, what happens to the water?" Per [`../DECISION-model-selection.md`](../DECISION-model-selection.md) (decided 2026-06-04), the engine is **built** as a modular FEM model on **DOLFINx / FEniCSx** (PETSc), not extended from an off-the-shelf code.

## 2. Status
- Maturity: **in-development** (engine selected 2026-06-04: build on DOLFINx/FEniCSx; install spike **PASSED 2026-06-04**; architecture design drafted — [`../../../docs/plans/2026-06-04-pids-forward-model-architecture-design.md`](../../../docs/plans/2026-06-04-pids-forward-model-architecture-design.md), awaiting review before module builds).
- Last reviewed: 2026-06-04

## 3. Scientific basis
- **Coupled physics:** 3-D variably-saturated subsurface flow (mixed-form Richards, van Genuchten/Mualem) coupled to overland surface routing (diffusion-/inertial-wave).
- **Numerics:** finite-element weak form on locally-refined unstructured meshes; implicit (backward-Euler) time stepping with adaptive step control spanning sub-hourly to daily; PETSc nonlinear (Newton/SNES) solvers.
- **Engineered features (embedded 1-D vectors — the core representation):** every PIDS feature — bare/clay channel, tunnel, french drain, catch drain, standard pipe, gutter, pump — is **one parameterized 1-D linear vector embedded in the 3-D Richards continuum** (mixed-dimensional 1D-in-3D), carrying **(1) axial Darcy conveyance** (high-K granular fill; `K_feat` from grain size), **(2) per-face bidirectional potential-driven exchange** with the soil (`q = σ·(H_feat − H_soil)` on hydraulic heads; `σ` a *series* of an engineered feature-face leg and a soil `K(ψ)` leg; **clay = a sealed face, `σ = 0`**), and **(3) storage**. The default solve is a **smooth monolithic Newton** problem; a PETSc variational-inequality path is reserved for overland depth-positivity and the **optional** hard one-way device — the degenerate asymmetric-`σ` limit that is the only role left for the former "catch-valve" (de-risked 2026-06-03, `scratch/catchvalve_probe.py`, as an implicit in-solver hook; **insurance, not the core**). Pumps add the lone *active* source/sink term; gutters are influx + along-path-sealed conveyance to an exfiltration outlet. Full spec: [`../../../docs/plans/2026-06-04-pids-forward-model-architecture-design.md`](../../../docs/plans/2026-06-04-pids-forward-model-architecture-design.md).

## 4. Architecture (modular, bottom-up)
Built and verified one module at a time (each gated by [`../../../governance/claude-sanity-check-routine.md`](../../../governance/claude-sanity-check-routine.md)):
1. **subsurface** — 3-D Richards solver (seeded from the de-risk probe).
2. **overland** — surface flow + velocity/direction field (also feeds the R5 channel-damage threshold).
3. **coupling** — mass-conservative surface↔subsurface exchange.
4. **pids-features** — the embedded 1-D-vector feature layer: channels (bare/clay), tunnels, french/catch drains, pipes, gutters, pumps — all parameterizations of one vector (axial conveyance + per-face bidirectional exchange + storage); clay = sealed face; the optional one-way device is a degenerate asymmetric-`σ` case.
5. **domain** — mesh + topography (synthetic now; DEM/LIDAR later) + SE-Piedmont subsurface-property generator.
6. **forcing / vegetation** — synthetic rainfall (typical + 100-yr extremes) + PET; simple PFT static-root ET.
7. **driver / IO** — orchestration + standardized result output for sanity checks and visualization.

## 5. Dependencies (outgoing)
- [`../parameterization/`](../parameterization/) — hydraulic properties (van Genuchten / pedotransfer).
- [`../pids-representation/`](../pids-representation/) — how features are encoded (informs the `pids-features` module).
- [`../../aerial-mapping/`](../../aerial-mapping/) — DEM/LIDAR terrain (operational; synthetic initially).

## 6. Interfaces
- **Inputs:** terrain/mesh, subsurface property fields, PIDS feature geometry + internal-BC specs, boundary conditions, forcing (rainfall hyetograph, PET).
- **Outputs:** state fields (saturation, head), fluxes (runoff, infiltration, recharge), ponding/surface depth + velocity, and a standardized result file consumed by the sanity-check + visualization routines.
- **Assumptions:** static clay-gating (fixed installed geometry); research-grade rigor; engine resolves PIDS features via local refinement / 1D-in-3D embedding rather than a global sub-meter mesh.

## 7. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Engine must reproduce C-001 (lateral conveyance) and C-004 (conveyance ≫ percolation) in validation | C-001, C-004 | unverified — gated on the M1/M5 embedding-fidelity test (one a-priori Peaceman connectivity factor across ≥2 geometries & both flow directions — the C-004 falsifiability gate) + optional `PIDS-MIN-1` cross-code |
| Embedded 1-D-vector features reproduce a fully-resolved 3-D reference (axial conveyance + bidirectional exchange) | — | unverified — the core Module-4 validation (design §E.6/§I.1) |
| Optional one-way device (degenerate asymmetric-σ) is tractable as an implicit in-solver hook — *insurance, not the core* | — | single-checked + adversarial probe (`scratch/catchvalve_probe.py`, 2026-06-03) |

## 8. Open questions / risks
- The coupled **monolithic mixed-dimensional** solver (surface ↔ subsurface ↔ embedded 1-D features) is the dominant build/validation risk; the 2026-06-03 probe validated only a 1-D Richards host, not the coupled 3-D solver.
- Sub-meter feature representation vs. tractable grid over ~100 ha (see [`../pids-representation/`](../pids-representation/)).
- DOLFINx + PETSc-VI capability **confirmed by the install spike (2026-06-04, PASS)**; the open numerical risks are now DOLFINx 0.10 **mixed-dimensional (1D-in-3D) assembly maturity**, the singular line-source `σ`/near-field closure, and high-K-contrast conditioning (design risks R1–R3).

## 9. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
| 2026-06-04 | Engine **decided** (DOLFINx/FEniCSx FEM build, Option B); modular architecture + bottom-up build order defined; status → in-development. | Reviewed related components: **parameterization** (no break — still supplies hydraulic properties; FEM consumes van Genuchten fields directly), **pids-representation** (no break — informs the `pids-features` module; 1D-in-3D embedding noted as the target pattern), **validation** (will host the `PIDS-MIN-1` benchmark + the promoted catch-valve probe), **aerial-mapping** (no break — DEM/LIDAR interface unchanged; synthetic terrain used initially). No `interfaces.md` change yet (engine-internal). Claims C-001/C-004 remain `unverified`. Logged in `decision-log.md` (Major). |
| 2026-06-04 | Install spike **PASSED** (DOLFINx 0.10.0 + PETSc VI). Architecture design drafted (`../../../docs/plans/2026-06-04-pids-forward-model-architecture-design.md`). **PIDS feature framing corrected:** the embedded 1-D-vector class (axial Darcy conveyance + per-face bidirectional exchange + storage; clay = sealed face; pumps/gutters as parameterizations; optional one-way device = degenerate asymmetric-σ) **replaces the retired "reverse catch-valve as decisive primitive."** §2/§3/§4.4/§7/§8 rewritten to match. | Engine-internal; no neighbor interface break. C-001/C-004 still `unverified` (now gated on the embedding-fidelity test + optional `PIDS-MIN-1`). |