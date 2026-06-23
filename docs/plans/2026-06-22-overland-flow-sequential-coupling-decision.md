# Decision record: move overland flow to sequential (operator-split) coupling

**Date:** 2026-06-22 · **Status:** DECIDED (Arik) — proceed with the sequential/operator-split design; validate by spike before the full rework. Citation wording under final verification (does not affect the decision).

## 1. Context — why we are revisiting overland flow

While building a 3-D demonstration of a coarse-sand conveyance channel intercepting storm runoff on a clay hillslope (`forward-model/scratch/m4_sand_channel_3d_demo.py`), the coupled solve **failed in both available overland schemes**:

- **Upwind** scheme: **time-step collapse** at t = 0.11 / 1.2 d — the monolithic Newton could not get through the clay stiffness + concentrated convergent inflow.
- **Galerkin** scheme: **time-step pinned at ~2×10⁻⁵ d** (the convergent-flow "sawtooth" oscillation), effectively non-terminating.

This is the known frontier-hard regime: **convergent overland flow coupled to a stiff (near-incompressible clay, van Genuchten n ≈ 1.09) subsurface**. Arik's standard: *if the model cannot represent convergent flow in a reasonably accurate way, the model is broken* — so we do not sidestep convergence; we fix the architecture.

## 2. What overland flow actually needs to do in PIDS (the goal)

Deliberately modest. The overland routine must determine:
1. **How much water overlies each surface parcel** right now → so local infiltration is computed there;
2. **Which direction the excess runs off**;
3. **Move that excess to the downslope parcel(s)** → so it becomes run-on that can infiltrate *there* the next step.

It is a water-**redistribution** job. It does **not** need accurate flood hydrographs, flow depths, or velocities. Almost all of the current architecture's complexity — and both failure modes above — come from solving the harder (momentum-hydraulics) problem.

## 3. The two pathologies and their root causes

- **Sawtooth (Pathology 1):** a documented artifact of discretizing the Manning kinematic/diffusive-wave PDE with a **consistent (non-lumped) Galerkin** finite-element scheme + central treatment of the advection term. It is *not* inherent to convergence. Standard fixes: mass-lumping and/or **upwind / finite-volume face fluxes** (ParFlow, GSSHA, PIHM/SHUD all do this; our own in-house upwind scheme already confirmed it on the tilted-V).
- **Coupled dt-collapse (Pathology 2):** from solving surface + subsurface in **one monolithic Newton** over a ponding on/off discontinuity against a near-incompressible subsurface. The disparate physics (fast hyperbolic surface vs stiff parabolic subsurface) condition the joint Jacobian badly.

## 4. Survey (two independent research passes, 2026-06-22)

A literature review and an open-source-code review were run in parallel; they **independently converged** on the same architecture.

### 4a. Literature
Recommends a **CATHY-style sequential, time-lagged operator split**: explicit mass-routing surface coupled to implicit 3-D Richards via Neumann↔Dirichlet **BC switching** (Putti & Paniconi). Key points:
- Maxwell et al. (2014) 7-model intercomparison: sequential (CATHY) ≈ monolithic (ParFlow / HydroGeoSphere) on the infiltration-/saturation-excess **runoff-generation regime PIDS targets** — coupling strategy drives *robustness and cost*, not correctness here.
- Making the monolithic integrated solve robust/scalable required substantial solver machinery — a terrain-following grid + Newton–Krylov with an **analytic-Jacobian preconditioner** (Kollet & Maxwell 2006; **Maxwell 2013**) — the engineering cost we avoid. (Maxwell 2013 frames this as scalability/performance, not explicitly the ponding discontinuity — see §9.)
- The only real downside of splitting — a **coupling mass-balance error** from the time-lag — is *bounded and monitorable* by the coupling step size (Fiorentini et al. 2015).
- Robustness/theory backing the split: Brandhorst et al. (2021) (iterative split more robust + far cheaper than fully-coupled); Schüller et al. (2024/25) (convergence factor + optimal relaxation).

### 4b. Code (other open-source models)
Concrete, adaptable template = **Landlab `KinwaveImplicitOverlandFlow`** (`landlab/src/landlab/components/overland_flow/generate_overland_flow_implicit_kinwave.py`): a single **upstream→downstream sweep** over an MFD flow-direction graph (`FlowDirectorMFD` proportions `square_root_of_slope` + `flow__upstream_node_order` topological order — the richdem-style O(N) dependency sweep), with a per-node scalar Newton for the Manning depth, plus a **depression-fill pre-pass** (PriorityFlood) for pits.
- **Avoid:** ParFlow `nl_function_eval.c` monolithic fold-in and MODFLOW 6's matrix-level GWF–SWF exchange — these *are* our current globally-implicit design (the dt-collapse architecture); and SHUD's all-in-one CVODE method-of-lines (elegant but a far bigger change than swapping the overland step).
- **Fallback if a surface PDE is retained:** Landlab `LinearDiffusionOverlandFlowRouter` (diffusion-wave, upwind, includes smooth infiltration; explicit-CFL-limited).

## 5. Trade-offs

| | Monolithic (current; ParFlow/MODFLOW-style) | Sequential non-iterative (CATHY-style; chosen) | Sequential iterative (Picard+relax; upgrade) |
|---|---|---|---|
| Robustness on stiff + convergent | Fragile (our dt-collapse); needs analytic Jac + MG | **Most robust** (no shared Jacobian) | Robust |
| Per-step cost | High | **Lowest** | Low–moderate |
| Mass conservation | Exact | Bounded coupling error (monitorable, step-controlled) | Exact at convergence |
| Tight-feedback accuracy | Best | Good; error bounded by step | Recovers monolithic accuracy |
| Engineering complexity | High | **Lowest** | Low + an outer loop |

## 6. Decision

**Adopt the sequential / operator-split overland coupling.** Keep the implicit 3-D Richards solve; replace the overland PDE with an **explicit, mass-conserving downslope routing sweep** on a flow-direction graph; couple the two by an **infiltration BC handoff** (Neumann flux when unsaturated/dry; Dirichlet ponded head when ponded/saturated).

Per timestep:
1. Solve Richards implicitly for infiltration given the current ponded depths;
2. Compute each parcel's excess = rain + run-on − infiltration − storage;
3. Route the excess explicitly downslope (MFD proportions, upstream-ordered sweep), conserving mass, optionally Manning-rate-capped per node;
4. New ponded depths → next step's surface BC (run-on).

This **structurally** eliminates both pathologies: a routing sweep has no surface PDE (no sawtooth; convergence is its native operation), and the surface update shares no Jacobian with the stiff-clay Richards Newton (no coupled-stiffness dt-collapse).

**Conditions/decision basis:** both surveys converged (the "unless the code review comes back with a wildly different interpretation" condition is not triggered).

## 7. Plan & guardrails

- **Validate by spike FIRST.** Build a minimal operator-split overland (MFD routing sweep + depression-fill + infiltration handoff to the existing Richards) and run on the **exact failing cases** — the 3-D sand-channel-in-clay storm and the tilted-V — to prove: no dt-collapse, mass-conserving (track the coupling mass-balance error), and runoff intercepted into the swale. Empirical green light before the full redesign (mirrors the embedded-feature probe).
- **Then** design doc + TDD build.
- **Coexistence, not rip-out.** The new scheme is added as a selectable option alongside the validated galerkin/upwind Manning schemes (exactly as upwind was added beside galerkin). The Manning stack remains the validated fallback until the sequential scheme is re-validated against PIDS's bar.
- **Before designing:** read `overland.py`, `overland_upwind.py`, and the coupling NCP in `coupling.py` to separate load-bearing physics from incidental numerical complexity.
- **Monitor** the coupling mass-balance error as a first-class output; if tight feedback under sustained ponding ever proves to need it, upgrade in place to sequential-iterative (Picard + relaxation) — same architecture, one outer loop — before ever returning to a monolithic Newton.

## 8. Relationship to prior work

- This **supersedes the Manning-PDE choice** of the P0–P3 convergent-flow workstream *for PIDS going forward* (`docs/plans/2026-06-18-overland-convergent-flow-P3.md` and the verdicts therein). That stack stays as a validated fallback; its in-house finding that upwinding removes the tilted-V sawtooth is consistent with the reference models and corroborates this direction.
- The embedded-feature redesign (`pids-unified-feature-redesign`: the local funnel-factor exchange probe + the 2-D disperse/convey demo, both validated) is independent and unaffected.

## 9. References (verified 2026-06-22; exact-quote status flagged)

Citations below are **confirmed** for authors/year/title/venue/volume/pages/DOI. Full-text was unavailable to the verifier, so only **one** verbatim quote could be confirmed (Maxwell et al. 2014); all other substance is a confirmed *paraphrase* — re-source exact sentences from the OA PDFs before printing quotation marks. Two corrections from the first draft are folded in: **Maxwell 2013** (not 2015); **Ebel et al. 2009** is in *Hydrological Processes* (and its slant is the opposite of what I first claimed — see item 9).

1. **Camporese, Paniconi, Putti & Orlandini (2010)**, *Water Resour. Res.* 46(2): W02512, DOI 10.1029/2008WR007536 — CATHY; the title itself reads "…path-based runoff routing, **boundary condition-based coupling**…". The sequential/path-based + BC-switching design whose spirit we adopt.
2. **Putti & Paniconi (2004)**, in *Computational Methods in Water Resources (CMWR XV)*, Developments in Water Science 55(2): 1391–1402, Elsevier (proceedings; no DOI) — the Neumann↔Dirichlet BC-switching algorithm.
3. **Furman (2008)**, *Vadose Zone J.* 7(2): 741–756, DOI 10.2136/vzj2007.0065 — the coupling-strategy review/taxonomy (global-implicit vs sequential-iterative vs non-iterative; interface conditions). *Exact taxonomy wording not verified — pull from the OA PDF before quoting.*
4. **Sulis, Meyerhoff, Paniconi, Maxwell, Putti & Kollet (2010)**, *Adv. Water Resour.* 33(4): 456–467, DOI 10.1016/j.advwatres.2010.01.010 — direct CATHY (sequential, time-LAGGED) vs ParFlow (monolithic, SAME-time-level). *[paraphrase]* the two perform very similarly under most scenarios; the largest differences appear under **infiltration-excess with heterogeneous precipitation**, attributed to the different coupling approaches + discretizations. ⇒ coupling strategy is a second-order discrepancy concentrated in the stiffest regime, not a correctness divide.
5. **Maxwell et al. (2014)**, *Water Resour. Res.* 50(2): 1531–1549, DOI 10.1002/2013WR013725 — 7-model intercomparison (CATHY, HGS, OGS, PIHM, PAWS, ParFlow, tRIBS-VEGGIE). *[verbatim — confirm punctuation against PDF]* "all the models demonstrate the same qualitative behavior, thus building confidence in their use for hydrologic applications." *[paraphrase]* agreement strongest on excess-infiltration/saturation runoff; differences emerge with heterogeneity + complex water-table dynamics. ⇒ strongest support that coupling strategy drives robustness/difficulty more than correctness on PIDS's regime.
6. **Brandhorst, Erdal & Neuweiler (2021)**, *Hydrol. Earth Syst. Sci.* 25(7): 4041–4059, DOI 10.5194/hess-25-4041-2021 (OA) — iterative vs non-iterative split vs fully-integrated 3-D. *Scope:* couples 2-D groundwater + multiple 1-D unsaturated columns (not overland); baseline = fully-integrated 3-D. *[paraphrase]* the non-iterative split is very fast; the iterative split is slower but more accurate, and both are very efficient vs fully-integrated 3-D.
7. **Kollet & Maxwell (2006)**, *Adv. Water Resour.* 29(7): 945–958, DOI 10.1016/j.advwatres.2005.08.006 — the ParFlow free-surface overland BC (= the monolithic coupling we move away from).
8. **Maxwell (2013)** *(corrected from "2015")*, *Adv. Water Resour.* 53: 109–117, DOI 10.1016/j.advwatres.2012.10.001 — terrain-following grid + Newton–Krylov with an analytic-Jacobian preconditioner; the solver machinery that makes the monolithic integrated solve robust/scalable. *Caveat:* framed as scalability/performance, NOT explicitly "the ponding discontinuity" — do not over-attribute.
9. **Ebel, Mirus, Heppner, VanderKwaak & Loague (2009)** *(corrected venue)*, *Hydrol. Process.* 23(13): 1949–1959, DOI 10.1002/hyp.7279 — first-order-exchange (surface-conductance) vs enforced pressure/flux continuity. ⚠️ *Correction:* this paper is actually **favorable** to first-order exchange (the coefficient can be tuned so the response is insensitive while cutting run time); the "nonphysical ℓₑ / common-node limit" critique I cited earlier is NOT this paper's wording (it belongs to the dual-node literature, e.g. Liggett/Delfs). Not load-bearing for our decision (we use BC-switching, not a dual-node conductance).
10. From the first literature survey (NOT re-verified here): Fiorentini et al. (2015), *Water Resour. Res.* — bounding the sequential coupling mass-balance error; Schüller, Birken & Dedner (2024/25), arXiv:2408.12582 — convergence factor + optimal relaxation for the iterative split; Jaber & Mohtar (2002), *Adv. Water Resour.* (mass-lumping/upwind cures KW Galerkin oscillation); de Almeida et al. (2012), *Water Resour. Res.* (stable semi-implicit inertial); ParFlow `OverlandKinematic` upwinding.

**Verification limitation:** the verifier could not open full PDFs (fetch disabled), so only the Maxwell 2014 line is confirmed verbatim. For publication-grade quotes, lift exact sentences from the OA sources (Brandhorst 2021 @ hess.copernicus.org; Furman 2008 @ acsess/ARS; Maxwell 2014 @ UMich DeepBlue; Camporese 2010 @ Wiley). The DECISION does not depend on exact wording.
