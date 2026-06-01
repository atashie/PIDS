# PIDS Physical System (Pillar 1)

## 1. Purpose
The Passive Infiltration Drainage System (PIDS) is the physical drainage system at the heart of the venture — the subject of U.S. non-provisional application 18/632,186 (in allowance). It drains surface water from land **without relying on slow permeation through compacted soil**: water moves *laterally* through high-conductivity coarse-sand conveyance channels and *vertically* through infiltration tunnels into the deep subsurface (bedrock/saprolite). This lets a site dispose of and recharge water locally, **eliminating the need for off-site discharge**.

## 2. Status
- Maturity: concept (patented; physical validation pending)
- Last reviewed: 2026-06-01

## 3. Scientific basis
- **Governing relation (channel sizing):** Darcy's law, `V = A·K·(dz/dx)` — `V` = volumetric flow rate, `A` = cross-sectional area, `K` = hydraulic conductivity, `dz/dx` = slope. Inverted to solve the channel cross-section required for a target flow.
- **Core principle:** exploit the large hydraulic-conductivity contrast between engineered coarse sand and compacted urban soil to convey water where the surrounding ground cannot (claims C-002, C-004 — `unverified`).
- **Vertical disposal / recharge:** infiltration tunnels bypass the compacted near-surface to reach more conductive deep soil and bedrock/saprolite (C-006 — `unverified`).

## 4. Dependencies (outgoing)
- [`../infiltration-runoff-model/`](../infiltration-runoff-model/) — designs the optimal configuration of this system.
- [`materials/`](materials/) — the sands and clays that make it work.

## 5. Interfaces
- **Inputs consumed:** optimal configuration (channel spacing/depth/width, tunnel placement/depth, barrier locations) from the infiltration-runoff model.
- **Outputs produced:** as-built geometry + field-monitoring data → feeds model validation.
- **Assumptions about neighbors:** the model can represent the as-built features; site characterization (depth-to-saprolite, soil K) is available.

## 6. Key claims & evidence
| Claim | claims-register ID | Status |
|-------|--------------------|--------|
| Lateral conveyance 26.7–266.7 in/day @ ~2° | C-001 | unverified |
| Coarse-sand K vs. soil: 6–9 OOM | C-002 | unverified |
| Compacted-clay K: ~10 OOM below sand | C-003 | unverified |
| Conveyance vs. vertical percolation: 10⁴–10⁷× | C-004 | unverified |
| Urban compaction: porosity 2–5×, K 10³–10⁹× | C-005 | unverified |
| Bedrock/saprolite K vs. soil: 3–9 OOM | C-006 | unverified |

## 7. Open questions / risks
- **Granted-claim vs. as-built scope (integrity note):** granted claim 1 requires the conveyance channels to "comprise a top layer and a water-impermeable bottom layer," but the specification treats the bentonite base as *optional/locational* and describes an all-sand-filled-channel embodiment. We must describe the system precisely on this point. Full reconciliation: `../../patents/claim-vs-reality.md` (added in Chunk D).
- C-001's unit ("inches of water/day" for a lateral rate) needs a clean re-derivation.

## 8. Change log
| Date | Change | Integration review (components checked, outcomes) |
|------|--------|---------------------------------------------------|
| 2026-06-01 | Created; seeded from patent S-001. | New component — no dependents yet; dependency matrix pending (Chunk C). |

---

## Sub-components
- [`conveyance-channels/`](conveyance-channels/) — lateral high-K coarse-sand network
- [`infiltration-tunnels/`](infiltration-tunnels/) — vertical bores to bedrock/saprolite (eliminates off-site discharge)
- [`impermeable-barriers/`](impermeable-barriers/) — compacted bentonite clay
- [`sediment-filter/`](sediment-filter/) — sand top + clay base
- [`materials/`](materials/) — sand grading/sorting, bentonite, sustainable sourcing
- [`installation-methods/`](installation-methods/) — trenching, augering
