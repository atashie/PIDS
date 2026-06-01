# Validation

## 1. Purpose
Establish that the model is trustworthy: that the forward model + representation + parameterization reproduce (a) the patent's quantitative claims under stated conditions, (b) physical-model experiments, and (c) field monitoring from installed PIDS. Validation is what lets the optimizer's output be believed — and is a primary path to moving claims C-001..C-006 from `unverified` to `adversarially-verified`.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Benchmarks against analytical Darcy solutions; controlled experiments; field data.

## 4. Dependencies (outgoing)
- [`../forward-model/`](../forward-model/), [`../parameterization/`](../parameterization/), [`../pids-representation/`](../pids-representation/).
- [`../../pids-physical-system/`](../../pids-physical-system/) — as-built + field monitoring.

## 5. Interfaces
- Inputs: model predictions; experimental/field measurements.
- Outputs: validation reports; claim status updates → [`../../../evidence/`](../../../evidence/).
- Assumptions: experiments/field sites are representative.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| All patent quantitative claims | C-001..C-006 | unverified |

## 7. Open questions / risks
- Availability of physical-model / field data at initialization (none yet).

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
