# Platform & Sensors

## 1. Purpose
The drone (UAV) and its sensing payload(s) — LIDAR and/or RGB/multispectral cameras — selected to meet the model's terrain accuracy/resolution requirement.

## 2. Status
- Maturity: concept (pending [`../DECISION-sensing-modality.md`](../DECISION-sensing-modality.md))
- Last reviewed: 2026-06-01

## 3. Scientific basis
- Sensor accuracy/resolution vs. altitude / payload / endurance trade-offs.

## 4. Dependencies (outgoing)
- [`../model-data-interface/`](../model-data-interface/) — the requirement the sensor must meet.

## 5. Interfaces
- Inputs: accuracy/resolution requirement; site constraints.
- Outputs: raw sensor data (point clouds, imagery).
- Assumptions: a sensor/platform exists that meets requirements within budget.

## 6. Key claims & evidence
| Claim | ID | Status |
|-------|----|--------|
| Sensor accuracy specs — to characterize | — | — |

## 7. Open questions / risks
- Blocked on modality decision + model requirement.

## 8. Change log
| Date | Change | Integration review |
|------|--------|--------------------|
| 2026-06-01 | Created (stub). | New stub; matrix pending. |
