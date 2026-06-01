# Applications (Axis 2 — the use cases)

> **Status: intentionally under-specified (2026-06-01).** Folders are scaffolded; detailed per-application design is deferred and will be revisited.

Each application is a use case that **composes a different subset of the [`../technologies/`](../technologies/) pillars** and sets the model's objective function. The application ↔ technology mapping will live in [`../integration/dependency-matrix.md`](../integration/dependency-matrix.md).

## Use cases
- [`suburban-drainage/`](suburban-drainage/) — remove ponding; protect foundations; eliminate off-site discharge; drought-amelioration co-benefit.
- [`sports-fields/`](sports-fields/) — uniform, rapid surface drainage; eliminate off-site discharge.
- [`roads-and-sidewalks/`](roads-and-sidewalks/) — vertical drainage of subsurface ponded water; protect subgrade; greatest benefit from eliminating off-site discharge (esp. urban).
- [`agriculture/`](agriculture/) — minimize surface runoff; managed aquifer recharge (MAR).

## Cross-cutting theme
Eliminating off-site discharge by routing excess water vertically to the deep subsurface (via the infiltration tunnels) is a unifying value proposition across all four use cases. See [`../docs/plans/2026-06-01-pids-repo-initialization-design.md`](../docs/plans/2026-06-01-pids-repo-initialization-design.md) §5.

## When we flesh these out
Each use case will adopt the standard [`../governance/component-template.md`](../governance/component-template.md), declare its pillar composition + objective/constraints, and be wired into the dependency matrix (triggering the integration protocol).
