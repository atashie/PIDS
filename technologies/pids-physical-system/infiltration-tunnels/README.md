# Infiltration Tunnels

## 1. Purpose
Vertical bores drilled down through the conveyance channels to bedrock or saprolite and backfilled with coarse, rounded sand. They route excess water *vertically* into the deep subsurface — the mechanism that **eliminates the need for off-site discharge**, recharges local aquifers, and bypasses the compacted near-surface.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Patent geometry: ~3-inch radius, ~6 ft deep (or to bedrock/saprolite if shallower); drilled gradually upslope near water-sensitive structures.
- Reaches non-compacted deep soil and bedrock/saprolite, more conductive than overlying soils by 3–9 OOM (C-006).
- Shallowest saprolite ~1–2 m / 3–7 ft (C-007) — governs achievable depth.

## 4. Dependencies (outgoing)
- [`../materials/`](../materials/) — coarse rounded sand backfill.
- [`../installation-methods/`](../installation-methods/) — augering.

## 5. Interfaces
- Inputs: tunnel placement & depth (from model); depth-to-saprolite (from site characterization / mapping).
- Outputs: vertical flux to the deep subsurface (recharge / disposal).
- Assumptions: a sufficiently conductive deep substrate is reachable.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Bedrock/saprolite K vs. soil 3–9 OOM | C-006 | unverified |
| Shallowest saprolite 1–2 m | C-007 | unverified |

## 7. Open questions / risks
- Verify reachable depth-to-saprolite regionally (C-007).
- **Regulatory:** vertical infiltration / managed aquifer recharge may be regulated (e.g., U.S. EPA Underground Injection Control — Class V wells). Flag for `applications/` (esp. roads/sidewalks, agriculture/MAR).

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub); seeded from S-001. | New stub; matrix pending. |
