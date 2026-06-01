# Conveyance Channels

## 1. Purpose
Shallow channels excavated parallel to the surface and backfilled with high-conductivity coarse sand, arranged in a network to receive surface water and convey it laterally — far faster than the surrounding compacted soil can.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Sized via Darcy `V = A·K·(dz/dx)` (invert for required cross-section).
- Patent geometry: ~1–2 ft deep, ~1 ft wide, spaced ~3–5 ft apart, parallel to surface, network configuration.
- Coarse (optionally rounded) sand resists compaction and keeps K high; K exceeds surrounding soil by 6–9 OOM (C-002).
- Lateral conveyance ~26.7–266.7 in/day at ~2° slope (C-001); exceeds vertical percolation by 10⁴–10⁷× (C-004).

## 4. Dependencies (outgoing)
- [`../materials/`](../materials/) — sand grading/sorting.
- [`../impermeable-barriers/`](../impermeable-barriers/) — optional/locational base layer.

## 5. Interfaces
- Inputs: optimal spacing/depth/width (from model); coarse sand (from materials).
- Outputs: conveyed lateral flow → infiltration tunnels and/or discharge point.
- Assumptions: surrounding soil K is far lower than channel fill.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Lateral conveyance 26.7–266.7 in/day @ ~2° | C-001 | unverified |
| Coarse-sand K vs. soil 6–9 OOM | C-002 | unverified |
| Conveyance vs. vertical percolation 10⁴–10⁷× | C-004 | unverified |

## 7. Open questions / risks
- Re-derive C-001 (unit scrutiny).
- Granted claim 1 requires a water-impermeable bottom layer in channels — see pillar README §7 and `../../../patents/claim-vs-reality.md`.

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub); seeded from S-001. | New stub; matrix pending. |
