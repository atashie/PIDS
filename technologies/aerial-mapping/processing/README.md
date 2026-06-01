# Processing

## 1. Purpose
Turn raw captures into clean terrain: point-cloud processing and ground classification to produce a bare-earth DEM/DTM (LIDAR), or the structure-from-motion pipeline (photogrammetry).

## 2. Status
- Maturity: concept
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Point-cloud filtering/classification; ground extraction; rasterization to DEM/DTM. (Candidate tooling: PDAL, LAStools, SfM suites — confirm with the stack decision.)

## 4. Dependencies (outgoing)
- [`../acquisition/`](../acquisition/).

## 5. Interfaces
- Inputs: raw point clouds/imagery.
- Outputs: bare-earth DEM/DTM → derived-products.
- Assumptions: data quality sufficient for reliable ground classification.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| — | — | — |

## 7. Open questions / risks
- Vegetation handling (modality-dependent).

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
