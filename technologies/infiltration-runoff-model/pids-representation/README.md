# PIDS Representation

## 1. Purpose
How PIDS features are encoded in the forward model's domain: conveyance channels as high-K linear features, infiltration tunnels as high-K vertical columns, and impermeable barriers as thin low-K layers — at fidelity sufficient to capture their hydrologic effect.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Feature-vs-grid trade-offs: explicit mesh refinement vs. sub-grid / effective-property representation.

## 4. Dependencies (outgoing)
- [`../../pids-physical-system/`](../../pids-physical-system/) — the features to represent.
- [`../forward-model/`](../forward-model/) — the engine's representation capabilities.

## 5. Interfaces
- Inputs: PIDS geometry; engine discretization options.
- Outputs: a model-ready representation of a configuration.
- Assumptions: effective properties can stand in for sub-grid features where explicit resolution is infeasible.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Representation must not distort C-001 / C-004 | C-001, C-004 | unverified |

## 7. Open questions / risks
- Sub-meter features (3-inch tunnel, 1-inch barrier) vs. tractable grid size.

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
