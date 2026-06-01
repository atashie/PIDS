# Technologies (Axis 1 — the science)

Independent scientific/engineering building blocks, organized by domain. Each is documented with the standard [`../governance/component-template.md`](../governance/component-template.md). Use cases in [`../applications/`](../applications/) compose these blocks; the authoritative mapping lives in [`../integration/dependency-matrix.md`](../integration/dependency-matrix.md).

## The three pillars (Sense → Solve → Build)

| Pillar | Role | Folder |
|--------|------|--------|
| **Aerial mapping** | *Sense* — high-resolution topography + surface data | [`aerial-mapping/`](aerial-mapping/) |
| **Infiltration-runoff model** | *Solve* — represent PIDS in a 3D hydrologic model; solve the optimal configuration | [`infiltration-runoff-model/`](infiltration-runoff-model/) |
| **PIDS physical system** | *Build* — the patented physical drainage system | [`pids-physical-system/`](pids-physical-system/) |

## Cross-pillar data / dependency flow

```
aerial-mapping ──(high-res DEM, surface data)──▶ infiltration-runoff-model
                                                          │
                              (optimal configuration: channel spacing/depth,
                               tunnel & barrier placement)
                                                          ▼
                                                pids-physical-system
                                                          │
                              (as-built + field monitoring) ───────────┐
                                                                        │ validation
                                                                        ▼
                                                    infiltration-runoff-model
```

- **infiltration-runoff-model** depends on **aerial-mapping** (input data) and on **pids-physical-system** (what it must represent and the design variables it optimizes).
- **pids-physical-system** depends on **infiltration-runoff-model** (it is *designed* by it).
- Field results from installed **pids-physical-system** feed back to validate the model.

(The machine-checkable version of these relationships lives in [`../integration/dependency-matrix.md`](../integration/dependency-matrix.md), added in the integration layer.)

## Status
All three pillars are at **concept** maturity at initialization. Two foundational choices are pending, captured as decision records:
- [`infiltration-runoff-model/DECISION-model-selection.md`](infiltration-runoff-model/DECISION-model-selection.md)
- [`aerial-mapping/DECISION-sensing-modality.md`](aerial-mapping/DECISION-sensing-modality.md)
