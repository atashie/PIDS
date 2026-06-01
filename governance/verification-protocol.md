# Verification Protocol (Adversarial Review)

This operationalizes **Guardrail #1**. Implemented as **both** documentation discipline **and** agent-assisted adversarial checks.

## What must be verified
- Any **third-party** fact (material property, prior-art claim, geological/hydrological figure, vendor spec).
- Any **derived** or **quantitative** claim we make (model outputs, hand calculations, the patent's numbers).

## The claims register
Every such item is one row in [`../evidence/claims-register.md`](../evidence/claims-register.md):

`ID · statement · value/units · source(s) · derivation · status · reviewer · date · notes`

### Status ladder
| Status | Meaning |
|--------|---------|
| `unverified` | Recorded but not yet checked. **Do not rely on it.** |
| `single-checked` | Confirmed once, by one method/reviewer. |
| `adversarially-verified` | Survived a deliberate attempt to refute it. **Safe to rely on, with citation.** |
| `disputed` | A check found a problem; see notes. **Do not rely on it.** |

### What counts as "double / triple checked"
A claim reaches `adversarially-verified` only when **either**:
- confirmed by **≥2 independent methods** (e.g., independent re-derivation **and** a second, independent source), **or**
- confirmed by **≥1 human reviewer and ≥1 adversarial agent run**,

with the evidence (calculation, sources, or review output) on file in `../evidence/reviews/` and the sign-off recorded in the register.

## Source provenance
Third-party material is archived under [`../evidence/sources/`](../evidence/sources/) with a provenance note: *what it is, where/when obtained, citation, and any license/usage constraint.* We cite the archived copy, not a transient URL.

## Adversarial review — the skeptic stance
The reviewer's job is to **refute**, not confirm. Default to `disputed` when uncertain.

- **Human:** independently re-derive the number or find a contradicting source. Do not reuse the original author's reasoning.
- **Agent-assisted:** use a template in [`../tools/review-agents/`](../tools/review-agents/) that instructs the agent to attack the claim; for important claims, fan out to multiple independent agents and require a majority to clear it. Save the raw output to `../evidence/reviews/`.

## Output naming
Reviews are stored as `../evidence/reviews/<claim-id>__<method>__<YYYY-MM-DD>.md` (e.g., `C-001__darcy-rederivation__2026-06-15.md`).
