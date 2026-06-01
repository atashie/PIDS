# Derived Products

## 1. Purpose
The terrain analytics the model actually consumes: slope, flow direction/accumulation, microtopography (ponding-prone low spots), and — optionally — surface texture/sediment classification.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Terrain derivatives from the DEM (slope, curvature, flow routing). Slope feeds Darcy `dz/dx` in the model.

## 4. Dependencies (outgoing)
- [`../processing/`](../processing/) — the DEM/DTM.

## 5. Interfaces
- Inputs: DEM/DTM.
- Outputs: terrain-derivative rasters → [`../model-data-interface/`](../model-data-interface/).
- Assumptions: DEM resolution/accuracy adequate for the derivatives.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| — | — | — |

## 7. Open questions / risks
- Error propagation from DEM to flow-accumulation.

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
