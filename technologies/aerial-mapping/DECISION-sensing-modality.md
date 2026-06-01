# Decision Record: Sensing Modality

- **Status:** OPEN (modality not yet chosen)
- **Owner:** TBD
- **Opened:** 2026-06-01
- **Decision type:** Major (requires confirmation + a `governance/decision-log.md` entry on resolution)

## Context
Pillar 3 must supply terrain at the resolution/accuracy the infiltration-runoff model needs. The acquisition modality determines accuracy, vegetation handling, cost, and throughput. You flagged this as open ("LIDAR?").

## The choice
| Modality | Strengths | Weaknesses |
|----------|-----------|------------|
| **LIDAR** | Penetrates vegetation (bare-earth returns under canopy); high vertical accuracy; direct 3D | Higher sensor cost; heavier payload; more processing |
| **Photogrammetry (SfM)** | Low cost; rich imagery (also yields surface texture/color); light payload | Cannot see ground under vegetation; accuracy depends on texture/control |
| **Hybrid** | Bare-earth (LIDAR) + texture/classification (imagery) | Most cost & complexity |

## Decision driver (must define first)
**The model's required terrain resolution & vertical accuracy.** PIDS features are sub-meter and slope matters for conveyance (Darcy `dz/dx`), so the accuracy requirement likely drives toward LIDAR or hybrid — but this must be *quantified* against the model's sensitivity, not assumed.

## Evaluation plan
1. From Pillar 2, derive the required DEM resolution + vertical accuracy (sensitivity of the optimal configuration to terrain error).
2. Map requirements → candidate sensors/platforms + cost.
3. Pilot on a representative site; compare against ground truth.
4. Record decision here + in `governance/decision-log.md`.

## Decision
*(pending)*
