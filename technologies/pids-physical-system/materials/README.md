# Materials

## 1. Purpose
The natural, sustainable, low-cost materials that make PIDS work: graded coarse (and rounded) sands for the conveyance fill and tunnel backfill, and bentonite clay for impermeable barriers. Material selection (grain size and *sorting*) is also a primary input to the model's parameterization, and the source of the venture's cost/sustainability advantage over conventional drainage.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Sand: ~0.5–2 mm; coarse and rounded grains resist compaction and maintain high K.
- Grain size **and sorting** govern hydraulic conductivity and water retention (see [`../../infiltration-runoff-model/parameterization/`](../../infiltration-runoff-model/parameterization/)).
- Bentonite clay: ~0.8 µm–2000 µm; compacts to very low K.

## 4. Dependencies (outgoing)
- None (upstream building block). Feeds most PIDS sub-components and the model's parameterization.

## 5. Interfaces
- Inputs: source/supply specifications.
- Outputs: characterized materials (grain-size distribution, sorting, K, retention) → parameterization + physical components.
- Assumptions: local, low-cost sources are available at the required gradations.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Coarse-sand K vs. soil 6–9 OOM | C-002 | unverified |
| Compacted-clay K ~10 OOM below sand | C-003 | unverified |

## 7. Open questions / risks
- Define the cost & sustainability comparison vs. conventional methods (a key business claim — needs its own verified evidence in `evidence/`).
- Sourcing variability → property variability.

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub); seeded from S-001. | New stub; matrix pending. |
