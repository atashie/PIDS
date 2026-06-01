# Optimization

## 1. Purpose
Solve for the **optimal PIDS configuration** — channel spacing/depth/width, tunnel placement/depth, barrier placement — that best meets a per-application objective subject to site and cost constraints. This is the inverse problem wrapped around the forward model, and the output that becomes the build specification.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Objective functions (per application): minimize ponding/runoff; maximize recharge (MAR); minimize cost; or weighted combinations.
- Constraints: depth-to-saprolite, no-infiltration zones (foundations/roads), material/cost limits.
- Methods: derivative-free / surrogate-assisted / gradient-based optimization over expensive forward runs.

## 4. Dependencies (outgoing)
- [`../forward-model/`](../forward-model/) — evaluates candidate configurations.
- [`../../../applications/`](../../../applications/) — supplies objectives & constraints per use case.

## 5. Interfaces
- Inputs: site (terrain + sediment), objective, constraints, design-variable bounds.
- Outputs: optimal configuration + predicted performance → pids-physical-system.
- Assumptions: forward model is accurate enough that its optimum is meaningful.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Optima must respect verified physics (C-001..C-006) | C-001..C-006 | unverified |

## 7. Open questions / risks
- Forward-run cost → need for surrogates.
- Multi-objective trade-offs (drainage vs. recharge vs. cost).

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
