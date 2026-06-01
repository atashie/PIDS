# Component Template

Every `technologies/` and `applications/` folder (and each sub-component) has a `README.md` with these eight sections, in this order. Uniform structure is what makes the holistic review (Guardrail #2) tractable.

Copy the skeleton below into each new component.

---

````markdown
# <Component name>

## 1. Purpose
What this component is and does, in 2–4 sentences.

## 2. Status
- Maturity: concept | in-development | validated | deployed
- Last reviewed: YYYY-MM-DD

## 3. Scientific basis
Governing physics, key assumptions, and equations. Cite claims-register IDs for any quantitative basis.

## 4. Dependencies (outgoing)
Components this relies on (links). The reverse — what relies on this — is read from
`integration/dependency-matrix.md` (the single source of truth) to avoid drift.

## 5. Interfaces
- Inputs consumed: (data, units, resolution, source component)
- Outputs produced: (data, units, resolution, consuming component)
- Assumptions made about neighbors:

## 6. Key claims & evidence
| Claim | claims-register ID | Status |
|-------|--------------------|--------|
|       |                    |        |

## 7. Open questions / risks
-

## 8. Change log
| Date | Change | Integration review (components checked, outcomes) |
|------|--------|---------------------------------------------------|
| YYYY-MM-DD | Created. | — |
````

---

**Notes**
- Section 4 lists only *outgoing* dependencies; dependents live in the dependency matrix.
- Section 8's third column is mandatory and records the integration-protocol outcome (per [`integration-protocol.md`](integration-protocol.md)).
