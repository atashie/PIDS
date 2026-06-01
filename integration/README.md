# Integration (cross-axis glue)

> **Status: intentionally under-specified (2026-06-01).** Folder scaffolded; the artifacts below are deferred and will be revisited.

This layer keeps the two axes — [`../technologies/`](../technologies/) and [`../applications/`](../applications/) — mutually consistent. It is the source of truth that the holistic-review protocol ([`../governance/integration-protocol.md`](../governance/integration-protocol.md)) points at.

## Planned artifacts
- **`dependency-matrix.md`** — application × technology/sub-component composition (plus pillar interdependencies). The authoritative "what depends on what."
- **`interfaces.md`** — the assumptions each component makes about its neighbors (inputs/outputs/units/resolutions).
- **`conflicts-and-open-questions.md`** — cross-cutting unknowns and unresolved tensions.

Until these exist, components record their *outgoing* dependencies in their own READMEs (per the [component template](../governance/component-template.md)).
