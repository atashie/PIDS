# Model–Data Interface

## 1. Purpose
The contract between aerial mapping and the infiltration-runoff model: the formats, coordinate reference system (CRS), resolution, and vertical accuracy the model requires — and the spec the sensing/processing chain must deliver. This boundary keeps Pillars 3 and 2 consistent.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- The requirement flows *from* the model's sensitivity to terrain error *to* the sensor spec.

## 4. Dependencies (outgoing)
- [`../../infiltration-runoff-model/`](../../infiltration-runoff-model/) — defines the requirement.

## 5. Interfaces
- Inputs: model resolution/accuracy requirement.
- Outputs: data spec (formats, CRS, resolution, accuracy) → rest of Pillar 3; delivered products → the model.
- Assumptions: requirement is quantified (this links the two decision records).

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| — | — | — |

## 7. Open questions / risks
- Both decision records (model + sensing) hinge on this requirement being defined.

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
