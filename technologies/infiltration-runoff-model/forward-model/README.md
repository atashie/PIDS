# Forward Model

## 1. Purpose
The pre-existing 3D hydrologic engine (to be selected) that simulates infiltration and runoff, extended to include PIDS features. It answers: "for this site and this PIDS configuration, what happens to the water?"

## 2. Status
- Maturity: concept (engine pending — see [`../DECISION-model-selection.md`](../DECISION-model-selection.md))
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Variably-saturated subsurface flow (Richards) + overland routing.

## 4. Dependencies (outgoing)
- [`../parameterization/`](../parameterization/) — hydraulic properties.
- [`../pids-representation/`](../pids-representation/) — feature encoding.

## 5. Interfaces
- Inputs: terrain, parameters, PIDS feature geometry, boundary/forcing (rainfall).
- Outputs: state fields (saturation, head), fluxes (runoff, infiltration, recharge), ponding.
- Assumptions: chosen engine can resolve/represent PIDS features adequately.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Engine must reproduce C-001, C-004 in validation | C-001, C-004 | unverified |

## 7. Open questions / risks
- Blocked on model selection.

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
