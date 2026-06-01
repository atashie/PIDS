# Aerial Mapping (Pillar 3)

## 1. Purpose
The data-acquisition pillar: drone-borne sensing (LIDAR and/or photogrammetry) that produces the **high-resolution topography and surface data** the infiltration-runoff model needs to design PIDS in an operational context. This is the "Sense" stage — without it, the model cannot be run on a real site at the resolution PIDS requires.

## 2. Status
- Maturity: concept (sensing modality not yet chosen — see [`DECISION-sensing-modality.md`](DECISION-sensing-modality.md))
- Last reviewed: 2026-06-01

## 3. Scientific basis
- High-resolution terrain (DEM/DTM) from LIDAR point clouds or structure-from-motion photogrammetry.
- Derived terrain products (slope, flow accumulation/direction, microtopography) drive surface routing in the model.
- Optional surface characterization (texture/sediment) from imagery/multispectral.

## 4. Dependencies (outgoing)
- None upstream (it is the sensing front-end). Its **dependent** is the infiltration-runoff model (recorded in the dependency matrix).

## 5. Interfaces
- **Inputs consumed:** the physical site (flights), ground control.
- **Outputs produced:** high-res DEM/DTM + derived terrain + (optional) surface classification, at a resolution/accuracy specified by the model → [`../infiltration-runoff-model/`](../infiltration-runoff-model/).
- **Assumptions about neighbors:** the model's required resolution/accuracy is defined (this drives sensor choice).

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Achievable vertical accuracy/resolution — to be characterized | — | — |

## 7. Open questions / risks
- **Sensing modality** is the gating choice ([`DECISION-sensing-modality.md`](DECISION-sensing-modality.md)).
- The model's resolution/accuracy requirement must be defined first — it sets the bar for the sensor.

## 8. Change log
| Date | Change | Integration review (components checked, outcomes) |
|------|--------|---------------------------------------------------|
| 2026-06-01 | Created; pillar defined. | New component; dependent = infiltration-runoff-model; matrix pending (Chunk C). |

---

## Sub-components
- [`platform-and-sensors/`](platform-and-sensors/) — drone + LIDAR / RGB / multispectral payloads
- [`acquisition/`](acquisition/) — flight planning, ground control
- [`processing/`](processing/) — point cloud → DEM/DTM (or SfM)
- [`derived-products/`](derived-products/) — slope, flow accumulation, surface texture
- [`model-data-interface/`](model-data-interface/) — formats + resolution required by Pillar 2

## Decision record
- [`DECISION-sensing-modality.md`](DECISION-sensing-modality.md) — LIDAR vs photogrammetry vs hybrid
