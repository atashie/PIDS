# Integration Protocol (Holistic Review on Change)

This operationalizes **Guardrail #2**. It is the checklist run on *every* change to a component, so that integrating a new idea never silently breaks an established requirement elsewhere.

## When it applies
Any time you create or change a component in `technologies/`, `applications/`, or their sub-components — including changing an assumption, interface, parameter, equation, or claim.

## The six steps

1. **Update the component's own README** — set `Status` + `last-reviewed` date, and add a dated **Change log** entry describing what changed and why.
2. **Look up the component in [`../integration/dependency-matrix.md`](../integration/dependency-matrix.md)** — enumerate (a) its dependencies (what it relies on) and (b) its dependents (what relies on it).
3. **Review each related component** for impact: does the change break an assumption or interface? **Record the outcome for each — even "no impact" — in the change-log entry.**
4. **Update [`../integration/interfaces.md`](../integration/interfaces.md)** if any input/output/unit/resolution/assumption at a boundary changed.
5. **Re-verify affected claims** — any [`../evidence/claims-register.md`](../evidence/claims-register.md) entry that depends on the change moves back to `unverified` (or is re-checked) per [`verification-protocol.md`](verification-protocol.md).
6. **Log major changes** in [`decision-log.md`](decision-log.md).

## Commit / change checklist (copy into the commit body or PR description)

```
Integration review for: <component>
- [ ] Component README updated (status, last-reviewed, change-log entry)
- [ ] Dependencies & dependents enumerated from dependency-matrix.md
- [ ] Each related component reviewed for impact (outcomes recorded, incl. "no impact")
- [ ] interfaces.md updated (or: no interface change)
- [ ] Affected claims re-verified / reset to unverified (or: none affected)
- [ ] decision-log.md updated (or: not a major change)
```

## Why "no impact" must be recorded
A silent omission is indistinguishable from an overlooked dependency. Recording "reviewed X — no impact" is the evidence that the holistic review actually happened.
