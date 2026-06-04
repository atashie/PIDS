# Infiltration-Runoff Model (Pillar 2)

## 1. Purpose
A high-resolution 3D infiltration-runoff model that **represents PIDS directly** and **solves for the optimal PIDS configuration** at a site, given its topography (from aerial mapping) and sediment characteristics (grain size and sorting). Rather than build from scratch, we extend a **pre-existing, validated** hydrologic model and add PIDS features (channels, tunnels, barriers) plus an optimization layer. This is the "Solve" stage that turns sensed data into a build specification.

## 2. Status
- Maturity: in-development (forward-model engine **decided** 2026-06-04 — build on DOLFINx/FEniCSx; see [`DECISION-model-selection.md`](DECISION-model-selection.md))
- Last reviewed: 2026-06-04

## 3. Scientific basis
- Variably-saturated subsurface flow (Richards equation) coupled to overland/surface routing.
- PIDS features represented as high-K linear features (channels), high-K vertical columns (tunnels), and thin low-K layers (barriers).
- Sediment grain size & sorting → hydraulic properties via pedotransfer functions (parameterization).
- Optimization = an inverse problem: find the configuration that best meets a per-application objective (minimize ponding/runoff, maximize recharge, minimize cost) subject to site & cost constraints.

## 4. Dependencies (outgoing)
- [`../aerial-mapping/`](../aerial-mapping/) — supplies high-resolution topography + surface data.
- [`../pids-physical-system/`](../pids-physical-system/) — defines the features to represent and the design variables to optimize.

## 5. Interfaces
- **Inputs consumed:** high-res DEM + derived terrain (from aerial-mapping); sediment grain size/sorting (from materials / site characterization); per-application objective + constraints.
- **Outputs produced:** optimal PIDS configuration (channel spacing/depth/width, tunnel placement/depth, barrier placement) → pids-physical-system; predicted performance (ponding, runoff, recharge).
- **Assumptions about neighbors:** input DEM resolution/accuracy is sufficient to resolve PIDS features; sediment characterization is representative.

## 6. Key claims & evidence
The model is the primary instrument for *verifying* the patent's quantitative claims (C-001..C-006); it must reproduce them under stated conditions. See [`validation/`](validation/).

## 7. Open questions / risks
- **Model selection** is the gating decision ([`DECISION-model-selection.md`](DECISION-model-selection.md)).
- Resolving sub-meter PIDS features (3-inch tunnels, 1-inch barriers) in a 3D mesh is computationally demanding — may require sub-grid representation or local refinement.
- Optimization cost: each forward run may be expensive → may need surrogate models.

## 8. Change log
| Date | Change | Integration review (components checked, outcomes) |
|------|--------|---------------------------------------------------|
| 2026-06-01 | Created; pillar defined. | New component; dependencies on aerial-mapping & pids-physical-system noted; matrix pending (Chunk C). |
| 2026-06-04 | Forward-model engine **decided** (build modular FEM on DOLFINx/FEniCSx; revises "extend a pre-existing model"). See [`DECISION-model-selection.md`](DECISION-model-selection.md), [`forward-model/`](forward-model/), decision memo [2026-06-03](../../docs/plans/2026-06-03-pids-forward-model-engine-evaluation.md). | Sub-component `forward-model` moved to in-development; `parameterization`, `pids-representation`, `validation` reviewed (no interface break — they feed/consume the FEM build as before). Major decision logged. |

---

## Sub-components
- [`forward-model/`](forward-model/) — the pre-existing engine (to be selected), extended for PIDS
- [`pids-representation/`](pids-representation/) — how channels/tunnels/barriers are encoded
- [`parameterization/`](parameterization/) — grain size & sorting → K, porosity, retention
- [`optimization/`](optimization/) — solve the optimal configuration
- [`validation/`](validation/) — against physical models, field data, and patent claims

## Decision record
- [`DECISION-model-selection.md`](DECISION-model-selection.md) — which pre-existing model to extend
