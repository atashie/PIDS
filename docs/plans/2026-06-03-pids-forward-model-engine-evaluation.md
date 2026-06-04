# PIDS Forward-Model Engine — Evaluation & De-Risk Decision Memo

- **Date:** 2026-06-03
- **Status:** **DECISION OPEN** — this memo *supports* the choice; it does not make it. `technologies/infiltration-runoff-model/DECISION-model-selection.md` stays OPEN until you select an option in §9.
- **Author:** Drafted by Claude (Claude Code) with Arik Tashie.
- **Decision type:** Major (per `governance/principles.md` Guardrail #4 — requires positive confirmation + a `governance/decision-log.md` entry on resolution).
- **Scope:** Selecting (or deciding how to select) the **forward-model engine** for Pillar 2 — the 3-D hydrologic model that will assess PIDS impact on drainage (effluence) and recharge (influence). The `optimization/` inverse loop is **out of scope** (deferred).

---

## 1. What we are deciding

Pillar 2 needs a physics-based, high-resolution 3-D hydrologic **forward** model to assess PIDS impact across spatial/temporal scales. The repo's standing strategy (decision-log, 2026-06-01) is to **extend a pre-existing, validated model rather than build from scratch**. This memo tests that strategy against a refined requirement set and reports where the evidence actually points — including a genuine tension with the "extend, don't build" stance (§8).

---

## 2. Settled scope & the weighted rubric

Decisions locked during this evaluation (2026-06-03):

| Topic | Decision |
|---|---|
| Deliverable | **Forward model for impact assessment**; optimization/inverse loop deferred. |
| Rigor target | **Research-grade / publishable** (per the repo's credibility ethos). |
| Erosion | **Not** a sediment-transport model. A **boolean structural-damage threshold for PIDS conveyance channels** only: in-channel flow depth × slope → velocity/shear vs. a threshold. Trivial post-processing. |
| Clay-gated vertical exchange | **Static / fixed-geometry** (clay is where it is installed) → ordinary material zoning, **not** a runtime state-dependent switch. |
| Catch-drain | **Reverse one-way valve**: admits surface-runoff **influx**, blocks groundwater **outflow** — the *inverse* of a tile drain (confirmed from the PIDS spec). |
| Optimization-loop scale | **Unknown / decide later** → keep MPI/GPU parallelism as a hedge, don't let it dominate. |
| Engine strategy | **Open** — let the evaluation decide. |

**Rubric (0–5 per criterion; weights sum to 100):**

| Criterion (weight) | Criterion (weight) |
|---|---|
| engineered_bcs **16** (DECISIVE) | adaptive_time 6 |
| discrete_features 12 | boundary_flexibility 6 |
| physics (Richards+overland) 12 | subsurface_profiles 4 |
| flexible_grid (local refine) 11 | dem_ingestion 3 |
| performance 9 | vegetation 3 |
| scriptability 8 | surface_velocity 3 |
| community_maturity 7 | |

Full requirements R1–R11 are recorded in the evaluation scripts (see §11).

---

## 3. Method (and its rigor)

Three evidence passes, each with adversarial verification per Guardrail #1:

1. **16-engine evaluation** (multi-agent workflow, 68 agents): each engine web-researched and scored against the rubric, then its decision-critical claims adversarially verified, plus two cross-cutting "crux" analyses (engineered-BC paradigm; grid + time-stepping) and a completeness critic.
2. **PFLOTRAN + catch-valve de-risk** (multi-agent workflow, 11 agents): PFLOTRAN scored into the rubric (it was flagged as a decisive omission), its four flagged uncertainties adversarially verified, and the reverse-catch-valve injection feasibility probed across five engines.
3. **Standalone numerical probe** (run locally, Python/NumPy/SciPy): isolates and stress-tests the sole decisive primitive — the reverse catch-valve's numerical tractability — independent of any engine (§7).

---

## 4. The single most important finding

**The PIDS catch-drain is the *inverse* of a conventional drain.** A standard "one-way drain" (HydroGeoSphere `DRAIN-FLUX`, MODFLOW `DRN`) is **outflow-only**. The PIDS catch-drain is the opposite — **inflow-only, blocks groundwater outflow**. Every engine credited with "native one-way drains" was therefore **over-credited** for this feature; the catch-drain needs a programmable conditional internal flux, not a stock drain package. This reframing is what makes the engineered-BC criterion (weight 16) the true decision axis, and it was the most-conflated point across the entire candidate field.

---

## 5. Ranked evaluation (17 engines)

Weighted totals (0–100). OSS = open-source; Comm = usable by a private/commercial venture.

| Rank | Engine | Score | OSS | Comm | One-line verdict |
|---:|---|---:|:--:|:--:|---|
| 1 | **Custom — Python + native** (Firedrake/DOLFINx + Landlab + PETSc) | **91** | ✅ | ✅ | Highest R9 ceiling; valve is first-class & *exact*; **but** the coupled solver is greenfield (validation burden). |
| 2 | HydroGeoSphere (HGS) | 87 | ❌ | ✅ | Best off-the-shelf physics; **closed + lease + OpenMP-only**; one-way drain is wrong-direction. **Keep as validation benchmark only.** |
| 2 | Custom — Julia (SciML + Gridap + ClimaLand.jl) | 87 | ✅ | ✅ | Best dynamic-AMR/ensemble story; greenfield hydrology ecosystem → key-person risk. Documented fallback. |
| 4 | **PFLOTRAN** (+ inline overland) | **80** | ✅ | ✅ | Strong subsurface + best off-the-shelf valve template (Hammond sandbox); surface flow expert-only; valve needs a recompile. |
| 5 | MODFLOW 6 + FloPy | 77 | ✅ | ✅ | Unbeatable scripting + valve via API package; **released code lacks 3-D Richards + coupled overland (R1 unmet).** |
| 6 | ATS / Amanzi | 75 | ✅ | ✅ | Validated integrated core; **no AMR** (R2) + C++/XML extension cost. |
| 7 | ParFlow (+CLM) | 74 | ✅ | ✅ | Best physics/HPC; **uniform horizontal grid (R2 fail)**; symmetric flow barriers → valve needs kernel surgery. |
| 8 | DuMuX (mixed-dimensional / DFM) | 72 | ✅ | ❌ | Cleanest 1D-in-3D feature embedding; **GPL-3.0 blocks proprietary** use → keep as a design pattern. |
| 9 | OpenGeoSys 6 | 71 | ✅ | ✅ | Scriptable Python BC carries the valve **without recompile**; **no native overland flow.** |
| 10 | InHM | 67 | ❌ | ❌ | Near-ideal physics + native state-dependent BCs; **closed/unobtainable** → design reference only. |
| 11 | Landlab | 56 | ✅ | ✅ | Excellent surface/erosion/DEM toolkit; subsurface is Dupuit, not 3-D Richards → a building block, not the core. |
| 12 | HYDRUS 2D/3D | 55 | ❌ | ✅ | Great subsurface Richards; closed, no programmable internal BC, weak overland. |
| 13 | CATHY | 54 | ✅ | ❌ | Strong coupled physics; **research-only license**; structured grid. |
| 14 | GEOtop | 53 | ✅ | ❌ | Good physics; **no internal-BC framework**; fixed grid; GPLv3. |
| 14 | PIHM / Flux-PIHM | 53 | ✅ | ✅ | Vertically-lumped subsurface → can't represent infiltration tunnels/depth profile. |
| 16 | tRIBS | 43 | ✅ | ✅ | 1-D infiltration + routing paradigm; no internal-BC framework. |
| 17 | RHESSys | 31 | ✅ | ✅ | Conceptual daily-step ecohydrology; can't resolve sub-meter/sub-hourly PIDS. Vegetation donor only. |

**Pattern:** no single engine tops both halves of the problem. Integrated engines (HGS/ParFlow/ATS/PFLOTRAN) own the coupled physics but each fails one decisive axis (closed, uniform grid, no AMR, or weak surface). Custom builds own the engineered-BC + grid flexibility but are greenfield on the coupled solver.

---

## 6. Catch-valve feasibility — *how* the reverse valve goes in

This table reorders the "extend" candidates by the decisive criterion. ("Without kernel surgery" = no modification of the solver/Jacobian core.)

| Engine | Inject without kernel surgery? | Mechanism | Numerical character |
|---|:--:|---|---|
| **Firedrake / DOLFINx (FEM)** | **YES** | Conditional interior-facet term in the UFL residual (`max_value(h_surf−h,0)`), Jacobian auto-differentiated — **or exact** via PETSc variational-inequality solvers (`vinewtonrsls`). Model-layer Python, no recompile. | First-class; exact complementarity available. |
| OpenGeoSys 6 | partial (no recompile) | pybind11 `getFlux(t,coords,primary_vars)` reads current node head, returns switched flux **+ Jacobian**. | Implicit (Jacobian supplied). No native overland → `h_surf` must be coupled in. |
| MODFLOW 6 API | partial (no recompile) | `modflowapi` callback rewrites HCOF/RHS each outer iteration — exact inverse of DRN. | Implicit (Robin term). No released 3-D Richards/overland. |
| PFLOTRAN | partial (**recompile**) | New `SOURCE_SINK_SANDBOX` subclass of the existing `srcsink_sandbox_pressure.F90` (Hammond; Newton-consistent, analytic Jacobian). | Implicit. **"No-recompile" claim was refuted** — needs a Fortran+PETSc rebuild. Native SEEPAGE is wrong-direction. |
| ParFlow | **NO** | Flow Barriers are *symmetric* multipliers — physically can't rectify. Only an external operator-split workaround, or C-kernel surgery. | **Operator-split → see §7, it blows up.** |

LGPL note (verified): PFLOTRAN, Firedrake, and DOLFINx are LGPL; a **proprietary PIDS extension layer that dynamically links** them is permissible. Caveat: avoid/relicense **ParMETIS** (non-commercial license) in any distributed build.

---

## 7. The standalone probe — retiring the decisive risk

**Question:** is the reverse catch-valve — a non-smooth complementarity condition `q = C·max(0, h_surf − h_node)` — numerically tractable, and how must it be implemented? This is the universal hazard flagged for *every* engine.

**Design:** 1-D vertical mixed-form Richards column (van Genuchten loam, 40 cells, backward Euler, Newton with a numerical Jacobian), one interior node carrying the valve. Four implementations compared — `hard` (kink), `smoothed` (softplus), `active-set` (semismooth), `lagged` (operator-split). Built-in **host sanity gate**: the `steady_low` case (valve shut) must hold exact equilibrium. Artifact: `scratch/catchvalve_probe.py` (+ `.png`).

**Results:**

- **Host validated** — `steady_low` holds equilibrium in **1 iteration, zero mass-balance error**.
- **Normal regime** — all variants converge in **≤5 Newton iterations**, mass balance to **machine precision (~1e-15)**, and the clamp **never leaks outflow**.
- **Stress sweep** (conductance Cv ∈ {5,50,500} × dt ∈ {0.02,0.1} day, surface head pinned just above threshold):

| Implementation | Cv=500, dt=0.1 day | Verdict |
|---|---|---|
| **hard / active-set (implicit)** | **4 iterations**, MB ~1e-13, head self-limits to `H=h_surf` (0.050 m) — no overshoot | **Bulletproof** even at extreme stiffness. |
| smoothed (implicit) | 8–10 iterations, MB ~1e-12, head 0.062 m | Robust; marginally costlier; tiny softplus overshoot. |
| **lagged (operator-split)** | 21–23 iterations, **head excursion 50 m** | **Blows up** — physically absurd. |

**Three decisive findings:**

1. **The valve is numerically benign when solved *implicitly*** — even the naive `max(0,·)` needs no smoothing; it converges in 4 iterations and *self-limits*. The feared "active-set chatter" essentially did not materialize implicitly.
2. **Operator-split coupling is disqualified** (50 m head blow-up). The valve **must** enter as an implicit in-solver hook with a Jacobian — i.e. FEM residual+AD/VI, OGS6 `getFlux`, MODFLOW6 HCOF, or the PFLOTRAN sandbox. **ParFlow's only path (operator-split) is empirically eliminated.**
3. If smoothing is wanted, **ε ≈ 1e-3 m** gives a spurious leak < 1e-4 m even at the threshold — negligible.

**Caveat (important):** the probe validated the *valve primitive in a 1-D host I wrote* — **not** a full coupled surface–subsurface engine. The coupled solver remains the unretired risk (§8).

---

## 8. Where the evidence points — and the honest tension

**Retired by evidence:** the engineered-BC primitive everyone worried about (the reverse catch-valve) is tractable, *and* we know how it must be built (implicit hook, not operator-split). Static clay-gating, discrete features, depth profiles, vegetation, and the erosion-threshold are all low-risk.

**The remaining unknown is the coupled Richards ↔ overland solver itself**, and it sits on a real fault line:

> The engines with **mature coupled overland** (ParFlow, HGS) are exactly the ones that **cannot carry the valve cleanly** (ParFlow operator-split→blows up; HGS closed). The engines that **carry the valve cleanly** (FEM, OGS6, MODFLOW6) **lack mature coupled overland**. The only path that does **both** well — a **PETSc-FEM build (Firedrake/DOLFINx)** — is a *build*, which **revises the repo's 2026-06-01 "extend, don't build from scratch" decision.**

So the evidence's center of gravity is the **#1-ranked FEM build** — but choosing it is a strategic call to revise a prior governance decision, not a mechanical outcome. That is why this is yours to decide.

---

## 9. Options for decision

| # | Option | Upside | Risk / cost |
|---|---|---|---|
| A | **De-risk coupled flow next, then commit** | Resolves the last unknown with evidence (as the valve probe did): test PFLOTRAN's expert-only inline overland on a tilted-V benchmark, and/or scope how cheap a FEM coupled-overland really is. | One more iteration before deciding. |
| B | **Commit to the PETSc-FEM build (Firedrake/DOLFINx)** | Best fidelity; valve first-class & *exact*; owns IP; Python-scriptable for the future loop; one ecosystem. | Revises "extend"; **coupled-solver validation burden** (person-months + V&V vs. analytical solutions/HGS). |
| C | **Commit to extending PFLOTRAN** | Honors "extend"; mature 3-D Richards; valve via the Hammond sandbox; MPI/GPU. | Expert-only/exploratory surface flow — weakest exactly where PIDS rainfall-runoff + R5 velocity live; one-time Fortran rebuild. |
| D | **(this memo) Decide after review** | You are here. | — |

**Cross-cutting, regardless of choice:**
- **HydroGeoSphere** = the gold-standard **validation benchmark** (not the production engine).
- **DuMuX 1D-in-3D embedding** = the **design pattern** for sub-meter channels/tunnels without a global fine mesh.
- **InHM** = design reference for state-dependent internal BCs.
- Verify any build can avoid/relicense **ParMETIS** and honor LGPL relinking.

---

## 10. Residual open questions

1. **Coupled surface↔subsurface fidelity** — does PIDS need a fully self-consistent two-way coupling, or is a staggered coupler acceptable? (Dominant unresolved physics question; gates A vs. B/C.)
2. **PFLOTRAN inline-overland defensibility** — does the expert-only diffusion-wave module yield credible in-channel velocities for the R5 channel-damage threshold? (Gates C.)
3. **Sub-meter features vs. tractable grid** — can a ~0.08 m tunnel / ~0.025 m barrier be represented by 1D-in-3D embedding or upscaling without a global sub-meter mesh over ~100 ha, and at what fidelity cost to claims C-001/C-004? (Tracked in `pids-representation/`.)
4. **Optimization-loop scale** — order of magnitude of forward runs sets how hard MPI/GPU must weigh, and whether differentiable/adjoint capability matters.
5. **License cleanliness** — confirm ParMETIS-free, LGPL-compliant build for whichever path.

---

## 11. Artifacts & provenance

- **16-engine evaluation** — workflow script + full JSON result (ranked scorecards, verifications, crux analyses, critic, synthesis). Run 2026-06-03.
- **PFLOTRAN + catch-valve de-risk** — workflow script + full JSON result (PFLOTRAN scorecard, verifications, per-engine valve feasibility, synthesis). Run 2026-06-03.
- **Numerical probe** — `scratch/catchvalve_probe.py` (+ `scratch/catchvalve_probe.png`). Reproducible with Python 3 + NumPy + SciPy + Matplotlib.
- Transient workflow outputs live under the session temp/transcript dirs; promote the decisive numbers into `evidence/` and `technologies/infiltration-runoff-model/validation/` once the engine is chosen.

> Verification status: the catch-drain inversion (§4), PFLOTRAN's sandbox/license facts (§6), and the probe results (§7) are **single-checked + adversarially reviewed**. PFLOTRAN's inline-overland maturity and the ParMETIS/LGPL distribution conditions are flagged for a second confirmation before they are relied upon (see `evidence/claims-register.md` when this is promoted).

---

## 12. On selection

When you pick A/B/C: I will (1) record the decision in `governance/decision-log.md` and resolve `technologies/infiltration-runoff-model/DECISION-model-selection.md`, noting that it revises (or upholds) the 2026-06-01 "extend, don't build" stance; (2) run the integration protocol against `pids-representation/`, `forward-model/`, and `validation/`; and (3) proceed to the **forward-model architecture design** (the "Design" phase you queued after engine selection).
