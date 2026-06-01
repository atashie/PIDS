# Principles (Guardrails)

These four guardrails govern all work in this repository. They are not aspirational — each maps to a concrete mechanism, listed below. If a proposed action conflicts with a guardrail, stop and resolve the conflict first.

## 1. Honesty, integrity, clarity of methods, transparency

We do not assert anything as true on trust — especially numbers, material properties, or claims drawn from third parties.

- Every third-party fact and every derived/quantitative claim is recorded in [`../evidence/claims-register.md`](../evidence/claims-register.md) with its source, derivation, and verification status.
- Claims are **adversarially verified** (independent re-derivation and/or skeptic-stance agent review) before we rely on them. See [`verification-protocol.md`](verification-protocol.md).
- Source materials are archived with provenance in [`../evidence/sources/`](../evidence/sources/).
- We state uncertainty plainly. "Unverified" is an acceptable, honest status; "verified" requires evidence on file.

## 2. Cross-disciplinary integration with holistic review

Our work spans hydrology, soil physics, remote sensing, numerical modeling, materials science, and field engineering. Components are documented independently but **must remain mutually consistent**.

- Each component carries the standard [`component-template.md`](component-template.md) README.
- **When any component changes, run the [`integration-protocol.md`](integration-protocol.md)** — review every related component (per the dependency matrix) and record the outcome. Updating one thing requires checking everything it touches.

## 3. Credibility over business minutiae (planning phase)

We are establishing scientific and technical credibility: rigorous code & modeling, physical models, patents, and peer-reviewed publications.

- In scope: methods, models, data provenance, validation, IP, publications.
- Out of scope (for now): taxes, payroll, HR, entity formation, marketing. Do not add these.
- Apply YAGNI: build what the science needs, nothing speculative.

## 4. Cautious building with positive confirmation

Large additions or alterations require explicit confirmation before proceeding.

- Major decisions are recorded, dated, in [`decision-log.md`](decision-log.md).
- Prefer small, reviewable changes. Pause at natural checkpoints for review.
- When in doubt, ask before building.

---

*Changing these principles is itself a "large alteration" — it requires explicit confirmation and a decision-log entry.*
