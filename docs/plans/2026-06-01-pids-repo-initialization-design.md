# PIDS Repository — Initialization Design & Plan

- **Date:** 2026-06-01
- **Status:** APPROVED 2026-06-01 — building chunk by chunk (see §14)
- **Author:** Drafted by Claude (Claude Code) with Arik Tashie
- **Scope of this document:** How we will initialize and organize this repository. This is a *planning* artifact (per the project's "planning phase" focus). It does **not** yet build the structure; the build is gated on your approval (Guardrail #4).

---

## 1. Purpose of the repository

This repository guides the development of the **scientific and technological underpinnings** of a future for-profit venture built around novel, low-cost, sustainable methods for improving drainage across four settings: suburban environments, sports fields, road/sidewalk networks, and agricultural systems.

The repository's job in the current phase is to establish **credibility of methods** — rigorous code and modeling, (eventually) physical models, patents, and peer-reviewed publications — *not* business administration (taxes, payroll, etc., are explicitly out of scope).

The flagship technology, **PIDS (Passive Infiltration Drainage System)**, is the subject of U.S. non-provisional application **18/632,186** (now in the process of allowance), which is one of three technology pillars described below.

---

## 2. Guiding principles (the four guardrails), operationalized

These are your four stated guardrails, each turned into a concrete mechanism in the repo. They will live in `governance/principles.md`.

| # | Principle | How the repo operationalizes it |
|---|-----------|---------------------------------|
| 1 | **Honesty, integrity, clarity of methods, transparency.** Third-party material must be double/triple-checked by adversarial review. | A central **`evidence/claims-register.md`** logs every third-party fact and every derived number with its source, derivation, and verification status. Nothing is treated as "true" until adversarially verified. Source provenance is archived in `evidence/sources/`. (See §7.) |
| 2 | **Cross-disciplinary integration.** Each component documented separately; updating one component triggers a holistic review of all related components. | A **two-axis architecture** (§3) keeps components independent, and an explicit **`integration/`** layer + **integration protocol** (§6) makes "review everything affected on every change" an enforced checklist, not an aspiration. |
| 3 | **Planning-phase focus on credibility, not business minutiae.** Rigorous code/modeling, physical models, patents, publications. | The structure centers on `technologies/`, `evidence/`, `patents/`, and (future) `publications/` and `physical-models/`. No payroll/tax/HR/finance folders. YAGNI applied throughout. |
| 4 | **Cautious building with strong positive confirmation before large additions/alterations.** | Every large change is gated. A dated **`governance/decision-log.md`** records confirmed decisions. The build plan (§14) proceeds in reviewable chunks, each with a confirmation gate. |

---

## 3. Architecture: two independent axes joined by an integration layer

The single most important design decision (confirmed 2026-06-01):

- **Axis 1 — `technologies/` (the science: "how it works").** Independent sibling building blocks, organized by scientific/engineering domain.
- **Axis 2 — `applications/` (the use cases: "where we deploy it").** Independent use cases, each of which *composes a different subset* of the technologies.
- **The two axes are organized independently of one another**, joined by an explicit **`integration/`** layer that records which applications depend on which technologies, the interfaces between components, and the cross-cutting open questions.

This separation is what lets a single technology improvement propagate cleanly to every use case that relies on it — and is what makes Guardrail #2's holistic review tractable.

---

## 4. Technologies axis — the three pillars

The three pillars form a **Sense → Solve → Build** pipeline:

- **Pillar 3 — Aerial mapping (Sense):** drone-borne LIDAR (or photogrammetry) produces the high-resolution topography + surface data.
- **Pillar 2 — Infiltration-runoff model (Solve):** consumes that data plus sediment characteristics (grain size, sorting), extends a *pre-existing* 3D hydrologic model to represent PIDS directly, and solves for the **optimal PIDS configuration**.
- **Pillar 1 — PIDS (Build):** the physical, patented system installed per that optimal configuration.

The loop closes: installed systems + field monitoring validate the model, and the model must reproduce the patent's quantitative claims (tying Pillar 2 to `evidence/claims-register.md`).

### Pillar 1 — `technologies/pids-physical-system/` (physical system; the patent)
Sub-components (each gets the standard component template, §8):
- `conveyance-channels/` — lateral high-conductivity coarse-sand network (sizing via Darcy: `V = A·K·(dz/dx)`)
- `infiltration-tunnels/` — vertical bores to bedrock/saprolite for deep percolation/recharge
- `impermeable-barriers/` — compacted bentonite clay (the "water-impermeable bottom layer")
- `sediment-filter/` — sand top + clay base
- `materials/` — sand grading/sorting, bentonite; natural & sustainable sourcing (the cost advantage)
- `installation-methods/` — trenching, augering

### Pillar 2 — `technologies/infiltration-runoff-model/` (3D model + optimizer; Solve)
- `forward-model/` — the pre-existing engine, extended to represent PIDS
- `pids-representation/` — how channels/tunnels/barriers are encoded in the model
- `parameterization/` — grain size & sorting → hydraulic conductivity, porosity, water retention (pedotransfer functions, e.g. Kozeny–Carman; van Genuchten/Rosetta)
- `optimization/` — solve for optimal configuration; objective functions + constraints (per-application objectives set in Axis 2)
- `validation/` — against physical models, field data, and the patent's claimed numbers
- `DECISION-model-selection.md` — see §11

### Pillar 3 — `technologies/aerial-mapping/` (drone/LIDAR acquisition; Sense)
- `platform-and-sensors/` — drone + LIDAR / RGB / multispectral payloads
- `acquisition/` — flight planning, ground control
- `processing/` — point cloud → DEM/DTM (or structure-from-motion photogrammetry)
- `derived-products/` — slope, flow accumulation, surface texture
- `model-data-interface/` — formats + resolution required by Pillar 2
- `DECISION-sensing-modality.md` — see §11

---

## 5. Applications axis — the four use cases

Each use case is an independent folder that **declares which technologies/sub-components it composes** and **sets the model's objective function**. Initial composition (to be refined in `integration/dependency-matrix.md`):

| Application | Primary objectives | Emphasized pillars/sub-components |
|-------------|--------------------|-----------------------------------|
| `suburban-drainage/` | Remove ponding; protect foundations; **eliminate off-site discharge** by routing excess water vertically to the deep subsurface; **ameliorate drought / reduce irrigation need** via increased local infiltration where desired | Infiltration tunnels (vertical disposal + recharge); conveyance channels; impermeable barriers near structures; model decides where to infiltrate vs. barrier |
| `sports-fields/` | Uniform, rapid surface drainage; **eliminate off-site discharge** | Conveyance channels (uniform grid) + infiltration tunnels (vertical disposal); model for uniformity |
| `roads-and-sidewalks/` | Vertical drainage of subsurface ponded water; protect subgrade. **Greatest benefit from eliminating off-site discharge, especially in urban environments** | Infiltration tunnels + impermeable barriers (protect road base); model for placement |
| `agriculture/` | Minimize surface runoff; **managed aquifer recharge (MAR)** using PIDS infrastructure | Infiltration tunnels (recharge); model objective = recharge vs. runoff trade-off |

**Cross-cutting value proposition (added 2026-06-01):** *Eliminating off-site discharge* by routing excess water vertically into the deep subsurface (via Pillar 1's infiltration tunnels) is a unifying benefit across all four applications — the patent's "optional water discharge outlet" is correspondingly de-emphasized. Where local infiltration is desirable, this yields a **drought-amelioration / reduced-irrigation co-benefit**; in agriculture it generalizes to **managed aquifer recharge (MAR)**. This theme is tracked as a first-class objective in `integration/dependency-matrix.md` and reflected in the model's objective functions (Pillar 2).

Each application README will also capture: site constraints, success criteria, and links to relevant evidence.

---

## 6. Integration & holistic-review mechanism (Guardrail #2)

The `integration/` layer is the cross-axis glue and the enforcement point for "review all related components on any change."

- **`integration/dependency-matrix.md`** — the single source of truth for relationships: an *application × technology/sub-component* composition matrix, plus a note on pillar-to-pillar interdependencies (e.g., Pillar 2 depends on Pillar 1 for what to represent and Pillar 3 for data; Pillar 1 depends on Pillar 2 for design). To avoid drift, each component README lists only its **outgoing** dependencies + interfaces; the **reverse** direction (dependents) is read from this matrix.
- **`integration/interfaces.md`** — the assumptions each component makes about its neighbors (inputs consumed, outputs produced, units, resolutions).
- **`integration/conflicts-and-open-questions.md`** — cross-cutting unknowns and unresolved tensions.

**The integration protocol (`governance/integration-protocol.md`)** — the checklist run on *every* component change:
1. Update the component's README (status + change-log entry).
2. Look the component up in `dependency-matrix.md` → enumerate dependents and dependencies.
3. For each related component, review whether the change breaks an assumption/interface; **record the review outcome (even "no impact") in the change-log entry.**
4. Update `interfaces.md` if any interface changed.
5. Re-verify any `claims-register` entries that depend on the change.
6. If it's a major change, log it in `governance/decision-log.md`.

A commit/PR checklist template will mirror these six steps so they are not skipped.

---

## 7. Verification & adversarial review (Guardrails #1 & #4)

Implemented as **both documentation discipline and agent-assisted checks** (your selection).

**Documentation discipline:**
- **`evidence/claims-register.md`** — every third-party fact and every derived/quantitative claim is one row: `ID · statement · value/units · source(s) · derivation · verification status · reviewer · date · notes`.
- **Verification status ladder:** `unverified → single-checked → adversarially-verified → disputed`. A claim counts as **double/triple-checked** only when confirmed by **≥2 independent methods** (e.g., independent re-derivation + a second data source) *or* **≥1 human + ≥1 adversarial agent run**, with explicit sign-off recorded.
- **`evidence/sources/`** — archived third-party materials, each with a provenance note (what, where obtained, when, citation).
- **`evidence/reviews/`** — outputs of adversarial reviews (human and agent).

**Agent-assisted (templates in `tools/review-agents/`):**
- Reusable prompts that instruct an AI agent to **refute** a given claim (skeptic stance, "default to disputed if uncertain"), optionally fanned out to multiple independent agents with a majority rule. Results are written to `evidence/reviews/` and update the claim's status.

**Seed claims (extracted from the PIDS patent for first-round verification):**

| ID | Claim to verify | Source | Initial status |
|----|-----------------|--------|----------------|
| C-001 | Lateral conveyance of **26.7–266.7 inches of water/day** through coarse sand at ~2° slope | Application p.7 | unverified — re-derive via Darcy; the unit "inches/day" for a *lateral* rate is dimensionally unusual and needs scrutiny |
| C-002 | Coarse-sand K exceeds surrounding soils by **6–9 orders of magnitude** | Application p.7 | unverified |
| C-003 | Compacted clay effective K **~10 orders of magnitude** below well-sorted sand | Application p.7 | unverified |
| C-004 | Conveyance exceeds natural downward percolation at unit gradient by **10⁴–10⁷×** | Application p.8 | unverified |
| C-005 | Urban compaction reduces porosity 2–5× and K by **10³–10⁹×** | Application p.8 | unverified |
| C-006 | Bedrock/saprolite more conductive than overlying soils by **3–9 orders of magnitude** | Application p.8 | unverified |
| C-007 | Saprolite shallowest depth **1–2 m (≈3–7 ft)** | Office Action Response p.11 | unverified — was a litigation assertion; verify geologically |

---

## 8. The standard component template

Every `technologies/` and `applications/` folder (and their sub-components) carries a `README.md` with the **same** sections, so holistic review is uniform. Defined once in `governance/component-template.md`:

1. **Purpose** — what it is and does.
2. **Status** — maturity (`concept / in-development / validated / deployed`) + last-reviewed date.
3. **Scientific basis** — governing physics, assumptions, key equations.
4. **Dependencies (outgoing)** — components this relies on (links). *(Reverse direction lives in the dependency matrix.)*
5. **Interfaces** — inputs consumed, outputs produced, units/resolutions, assumptions about neighbors.
6. **Key claims & evidence** — quantitative claims this component makes, each linked to a `claims-register` ID + status.
7. **Open questions / risks.**
8. **Change log** — dated entries; each entry notes the holistic review performed per the integration protocol.

---

## 9. Governance documents (`governance/`)

- `principles.md` — the four guardrails (§2).
- `integration-protocol.md` — the six-step holistic-review checklist (§6).
- `verification-protocol.md` — the claims-register workflow + adversarial review rules (§7).
- `component-template.md` — the standard README template (§8).
- `disclosure-policy.md` — **light** (per your "private, minimal process"): repo is private; brief rule that nothing patent-sensitive is made public before the relevant filing; flag IP review before any public release.
- `decision-log.md` — dated record of major decisions. **Seeded** with the decisions already confirmed:

| Date | Decision |
|------|----------|
| 2026-06-01 | Two-axis architecture (technologies × applications + integration layer) adopted. |
| 2026-06-01 | Three technology pillars: PIDS (physical), infiltration-runoff model, aerial mapping. Sense→Solve→Build pipeline. |
| 2026-06-01 | Repo visibility: private, minimal disclosure process. |
| 2026-06-01 | Adversarial review implemented as both discipline + agent-assisted. |
| 2026-06-01 | PIDS app. 18/632,186 in allowance; prosecution defense out of scope. Patent-strengthening science = lower priority, retained for credibility/future filings. |
| 2026-06-01 | Pillar 1 folder named `pids-physical-system/`. |
| 2026-06-01 | Build cadence: chunk by chunk, with a review gate after each chunk. |
| 2026-06-01 | Applications refined: eliminating off-site discharge (vertical infiltration) is a cross-cutting value proposition; agriculture targets managed aquifer recharge (MAR); suburban gains a drought-amelioration co-benefit. |

---

## 10. Patents component (`patents/`)

> **Filesystem note:** this is Windows (case-insensitive), so `patents/` and the existing `Patents/` directory are the *same* folder. We keep the existing `Patents/PIDS/` PDFs **in place** (not moved) and add the files below alongside them.

- `Patents/PIDS/` — the three existing source PDFs (application, office action, response). Left untouched.
- `patents/grant-record.md` — clean record of the **granted/allowed** PIDS claims and their actual scope; metadata (App 18/632,186, filed 2024-04-10; provisional 63/458,462, filed 2023-04-11; Examiner Lawson, Art Unit 3678).
- `patents/claim-vs-reality.md` — **one-time honesty note** (not a legal action): granted claim 1 requires the conveyance channels to "comprise a top layer and a water-impermeable bottom layer," whereas the specification describes the bentonite base as *optional/locational* and also describes an all-sand-filled-channel embodiment. We document this so our scientific descriptions of "how PIDS works" stay precise about what is actually claimed vs. how the system is typically deployed.
- `patents/forward-ip-strategy.md` — candidate future filings (continuations; the model and mapping pillars; the four application domains).
- `patents/science-to-claims.md` — maps `evidence/` results to claims (criticality/unexpected-results support for *future* filings).

---

## 11. Key decision records (deferred, evidence-based — not pre-decided)

- **`technologies/infiltration-runoff-model/DECISION-model-selection.md`** — which pre-existing 3D model to extend. Candidates to evaluate against requirements (variably-saturated 3D flow + overland coupling, ability to represent discrete high-K channels/tunnels + low-K barriers, performance at high resolution, open-source, scriptability): **ParFlow**, **HYDRUS (2D/3D)**, **MODFLOW 6 + FloPy** (with UZF/SFR), or an integrated surface–subsurface code (e.g., CATHY/GEOtop/tRIBS). *No target was pre-specified; first task is the documented selection.*
- **`technologies/aerial-mapping/DECISION-sensing-modality.md`** — LIDAR vs. structure-from-motion photogrammetry vs. hybrid (your "(LIDAR?)" flag), evaluated on vertical accuracy, vegetation penetration, cost, and operational throughput.

---

## 12. Tooling & computational stack (provisional; revisited after model selection)

- **Python-first** (3.11+). Rationale: the candidate hydrologic models (FloPy/MODFLOW 6, ParFlow pftools, phydrus/HYDRUS, Landlab) and the geospatial/LIDAR stack (PDAL, laspy, rasterio, GDAL, RichDEM, WhiteboxTools, GeoPandas) are Python-native.
- **Julia** reserved for performance-critical custom numerics/optimization if profiling demands it.
- **R** only for specific statistical needs.
- **Reproducibility:** pinned environments (per-pillar where stacks diverge), deterministic seeds, documented data provenance.
- This stack is *provisional* and explicitly tied to the model-selection decision (§11).

---

## 13. Proposed directory tree (full)

```
PIDS/
├── README.md
├── .gitignore
├── docs/
│   └── plans/2026-06-01-pids-repo-initialization-design.md   ← this file
├── governance/
│   ├── principles.md
│   ├── integration-protocol.md
│   ├── verification-protocol.md
│   ├── component-template.md
│   ├── disclosure-policy.md
│   └── decision-log.md
├── technologies/
│   ├── README.md
│   ├── pids-physical-system/
│   │   ├── README.md
│   │   ├── conveyance-channels/ · infiltration-tunnels/ · impermeable-barriers/
│   │   ├── sediment-filter/ · materials/ · installation-methods/
│   ├── infiltration-runoff-model/
│   │   ├── README.md · DECISION-model-selection.md
│   │   ├── forward-model/ · pids-representation/ · parameterization/
│   │   ├── optimization/ · validation/
│   └── aerial-mapping/
│       ├── README.md · DECISION-sensing-modality.md
│       ├── platform-and-sensors/ · acquisition/ · processing/
│       ├── derived-products/ · model-data-interface/
├── applications/
│   ├── README.md
│   ├── suburban-drainage/ · sports-fields/ · roads-and-sidewalks/ · agriculture/
├── integration/
│   ├── dependency-matrix.md · interfaces.md · conflicts-and-open-questions.md
├── evidence/
│   ├── claims-register.md
│   ├── sources/
│   └── reviews/
├── patents/   (== existing Patents/)
│   ├── PIDS/   ← existing PDFs, untouched
│   ├── grant-record.md · claim-vs-reality.md
│   ├── forward-ip-strategy.md · science-to-claims.md
└── tools/
    └── review-agents/
```

---

## 14. Build / execution plan (phased, gated)

Each chunk is low-risk Markdown scaffolding. Proposed cadence: build **Chunk A**, pause for your review, then proceed.

- **Chunk 0 — Foundation:** `git init`, `README.md`, `.gitignore`, commit this plan.
- **Chunk A — Governance + Evidence:** all of `governance/` + `evidence/` (incl. seeded claims-register and provenance notes for the 3 PDFs). *Highest-leverage; defines how we work.* → **review gate**
- **Chunk B — Technologies:** three pillar READMEs + sub-component stubs (PIDS seeded from the patent) + the two decision records. → **review gate**
- **Chunk C — Applications + Integration:** four use-case READMEs + the dependency matrix/interfaces/open-questions. → **review gate**
- **Chunk D — Patents + Tools:** patent records, claim-vs-reality note, forward IP strategy; agent-review templates. → **review gate**

(Commits happen per chunk; nothing is pushed anywhere — repo stays local/private.)

---

## 15. Out of scope (YAGNI / Guardrail #3)

Business administration (taxes, payroll, HR, finance, legal entity formation); marketing assets; CI/CD and heavy infrastructure; actual model code or LIDAR data (the build creates *structure and documentation*, not implementations, until each component's approach is confirmed).

---

## 16. Open questions for sign-off

1. **Approve the architecture and this plan?** (Two axes, three pillars, integration layer, verification system.)
2. **Build cadence:** stop for review after **each** chunk (A→B→C→D), or build A–D in one pass and review the whole skeleton?
3. **Pillar 2 model:** confirm "documented selection first" (no pre-chosen model), or name a target.
4. **Naming:** is `pids/` acceptable for Pillar 1 (it's the physical system, distinct from the repo umbrella), or prefer e.g. `pids-physical-system/`?
5. **Anything to add/cut** before I create files.
```