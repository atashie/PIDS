# Claims Register

The single source of truth for every third-party fact and every derived/quantitative claim we rely on. See [`../governance/verification-protocol.md`](../governance/verification-protocol.md) for the workflow and the definition of "adversarially verified."

**Rule:** if we state a number anywhere in the repo (docs, models, papers, marketing), it has a row here, and we cite the claim ID.

## Status legend
`unverified` · `single-checked` · `adversarially-verified` · `disputed` — **do not rely on anything that is not `adversarially-verified`.**

## Register

| ID | Statement (claim) | Value / units | Source(s) | Derivation / how to check | Status | Reviewer | Date | Notes |
|----|-------------------|---------------|-----------|---------------------------|--------|----------|------|-------|
| C-001 | Lateral conveyance through coarse sand at ~2° slope | 26.7–266.7 inches of water/day | S-001 (application p.7) | Re-derive from Darcy `V = A·K·(dz/dx)`; determine what flux/area/units produce this. **"Inches of water/day" for a *lateral* rate is dimensionally unusual — scrutinize.** | unverified | — | 2026-06-01 | Seeded from patent. Pillar 2 must reproduce. |
| C-002 | Coarse-sand K exceeds surrounding soils | 6–9 orders of magnitude (10⁶–10⁹×) | S-001 (application p.7) | Compare literature K ranges: coarse sand vs. compacted urban soil. | unverified | — | 2026-06-01 | |
| C-003 | Compacted clay effective K below well-sorted sand | ~10 orders of magnitude (10¹⁰×) | S-001 (application p.7) | Compare K: compacted bentonite vs. well-sorted sand. | unverified | — | 2026-06-01 | |
| C-004 | Conveyance exceeds natural downward percolation at unit gradient | 10⁴–10⁷× | S-001 (application p.8) | Compare lateral channel flux vs. vertical percolation at slope = 1. | unverified | — | 2026-06-01 | |
| C-005 | Urban compaction reduces porosity and K | porosity 2–5×; K 10³–10⁹× | S-001 (application p.8) | Literature on compaction effects on porosity & K. | unverified | — | 2026-06-01 | |
| C-006 | Bedrock/saprolite more conductive than overlying soils | 3–9 orders of magnitude | S-001 (application p.8) | Literature on saprolite/weathered-bedrock K vs. soil. | unverified | — | 2026-06-01 | Plausibility check: saprolite K is highly variable. |
| C-007 | Shallowest depth to saprolite | 1–2 m (≈3–7 ft) | S-003 (response p.11) | Geological verification; regionally variable. | unverified | — | 2026-06-01 | Originated as a litigation assertion — verify independently. |

## How to add a claim
1. Add a row with status `unverified`.
2. Verify per [`../governance/verification-protocol.md`](../governance/verification-protocol.md); save evidence to [`reviews/`](reviews/).
3. Update status, reviewer, date, and notes.
