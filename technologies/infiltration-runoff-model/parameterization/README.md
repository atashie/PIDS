# Parameterization

## 1. Purpose
Translate **sediment characteristics — grain size and sorting** — into the hydraulic properties the model needs: saturated hydraulic conductivity (K), porosity, and water-retention parameters. This is where material choices become model inputs, and where several patent claims (C-002, C-003, C-005) are grounded in physics.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- K from grain size/porosity: e.g., Kozeny–Carman and related relations.
- Water retention: van Genuchten parameters; pedotransfer functions (e.g., Rosetta).
- **Sorting** (distribution width) strongly affects K and retention — not just the median grain size.

## 4. Dependencies (outgoing)
- [`../../pids-physical-system/materials/`](../../pids-physical-system/materials/) — characterized materials.

## 5. Interfaces
- Inputs: grain-size distribution + sorting (materials / site characterization).
- Outputs: K, porosity, retention parameters per material/zone → forward model.
- Assumptions: pedotransfer relations are valid for the engineered sands/clays used.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Coarse-sand K vs. soil 6–9 OOM | C-002 | unverified |
| Compacted-clay K ~10 OOM below sand | C-003 | unverified |
| Urban compaction porosity 2–5×, K 10³–10⁹× | C-005 | unverified |

## 7. Open questions / risks
- Validity of pedotransfer functions at the coarse/rounded end and for compacted bentonite.

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
