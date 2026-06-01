# PIDS

Scientific and technological foundations for a venture developing novel, low-cost, sustainable methods to improve drainage across **suburban environments, sports fields, road/sidewalk networks, and agricultural systems**.

> **Status:** Planning phase. Private repository. This repo exists to establish the *credibility of our methods* — rigorous modeling, physical models, patents, and publications — not business administration.

## What we're building

The flagship technology is **PIDS — the Passive Infiltration Drainage System** (U.S. non-provisional application 18/632,186, in allowance). PIDS moves surface water *laterally* through high-conductivity coarse-sand channels and *vertically* through infiltration tunnels into the deep subsurface — draining land **without relying on slow permeation through compacted soil**, and, critically, **eliminating the need for off-site discharge** by disposing of and recharging water locally.

### Three technology pillars (Sense → Solve → Build)

1. **Aerial mapping** (`technologies/aerial-mapping/`) — drone LIDAR / photogrammetry produces high-resolution topography and surface data. *(Sense)*
2. **Infiltration-runoff model** (`technologies/infiltration-runoff-model/`) — extends a pre-existing 3D hydrologic model to represent PIDS directly and solves for the optimal configuration from sediment characteristics and topography. *(Solve)*
3. **PIDS physical system** (`technologies/pids-physical-system/`) — the patented physical drainage system, installed per the optimal configuration. *(Build)*

### Four applications (each composes the pillars differently)
`suburban-drainage/` · `sports-fields/` · `roads-and-sidewalks/` · `agriculture/` — see `applications/`.

## How this repository is organized

Two **independent axes**, joined by an explicit integration layer:

| Path | What lives here |
|------|-----------------|
| `governance/` | How we work: principles, the integration (holistic-review) protocol, the verification protocol, the component template, disclosure policy, decision log. **Read this first.** |
| `technologies/` | **Axis 1** — the science: independent building blocks (the three pillars + sub-components). |
| `applications/` | **Axis 2** — the use cases, each composing a subset of technologies. |
| `integration/` | The cross-axis glue: dependency matrix, interfaces, conflicts & open questions. |
| `evidence/` | The citable knowledge base: claims register, archived sources, adversarial reviews. |
| `patents/` | IP record (granted PIDS claims) + forward IP strategy. |
| `tools/` | Shared tooling, incl. adversarial-review agent templates. |
| `docs/plans/` | Design & planning documents. |

## Working here — non-negotiables

From [`governance/principles.md`](governance/principles.md):

1. **Integrity first.** Every third-party fact and every derived number goes in [`evidence/claims-register.md`](evidence/claims-register.md) and is *adversarially verified* before we rely on it. Nothing is asserted as true on trust.
2. **Holistic integration.** When you change any component, run the **integration protocol** ([`governance/integration-protocol.md`](governance/integration-protocol.md)): review every component the change touches, and record the review.
3. **Credibility over minutiae.** We invest in rigorous methods, not business administration (out of scope this phase).
4. **Build cautiously.** Large additions require explicit confirmation and a [`governance/decision-log.md`](governance/decision-log.md) entry.

## Where to start
- New here? Read [`governance/principles.md`](governance/principles.md), then [`governance/component-template.md`](governance/component-template.md).
- The full initialization design is in [`docs/plans/2026-06-01-pids-repo-initialization-design.md`](docs/plans/2026-06-01-pids-repo-initialization-design.md).
