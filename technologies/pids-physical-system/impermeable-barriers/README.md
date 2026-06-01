# Impermeable Barriers

## 1. Purpose
A thin, compacted bentonite-clay layer placed where downward percolation is *undesirable* — e.g., along channel sections near building foundations, roads, or sidewalks — acting as a near-impermeable barrier (functionally like a buried plastic sheet), so water is conveyed away rather than infiltrating in sensitive locations.

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Compacted bentonite, ~1 inch thick, sediment size ~0.8 µm–2000 µm.
- Effective K ~10 OOM below well-sorted sand (C-003) → effectively impermeable to downward flow over the relevant scales.

## 4. Dependencies (outgoing)
- [`../materials/`](../materials/) — bentonite clay.
- [`../installation-methods/`](../installation-methods/) — placement & compaction (tamping).

## 5. Interfaces
- Inputs: barrier locations (from model); bentonite (from materials).
- Outputs: a low-K boundary that suppresses vertical flux locally.
- Assumptions: compaction achieves and maintains the target low K.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Compacted-clay K ~10 OOM below sand | C-003 | unverified |

## 7. Open questions / risks
- **Granted-claim coupling (integrity note):** granted claim 1 makes a "water-impermeable bottom layer" a *required* feature of the conveyance channels, whereas here it is *optional/locational*. Reconcile in `../../../patents/claim-vs-reality.md`.
- Long-term integrity of bentonite (desiccation cracking, root intrusion).

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub); seeded from S-001. | New stub; matrix pending. |
