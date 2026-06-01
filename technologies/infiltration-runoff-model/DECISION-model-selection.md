# Decision Record: Forward Model Selection

- **Status:** OPEN (no model chosen yet)
- **Owner:** TBD
- **Opened:** 2026-06-01
- **Decision type:** Major (requires confirmation + a `governance/decision-log.md` entry on resolution)

## Context
Pillar 2 extends a **pre-existing** 3D hydrologic model rather than building from scratch. The choice constrains the whole pillar (representation approach, performance ceiling, tooling/language). This record frames the choice; it does **not** pre-decide it (we are in the planning phase).

## Requirements (draft — to refine)
1. **Physics:** variably-saturated 3D subsurface flow (Richards) coupled to overland/surface runoff.
2. **PIDS representability:** can encode discrete high-K linear channels, high-K vertical tunnels, and thin low-K barriers (via mesh refinement, sub-grid, or material zones).
3. **Resolution/performance:** usable at the high resolutions that PIDS features and drone-derived terrain imply.
4. **Open & scriptable:** open-source, automatable (for the optimization loop), active community.
5. **Licensing** compatible with a private/commercial venture.

## Candidates to evaluate
| Candidate | Notes |
|-----------|-------|
| **ParFlow** | Integrated parallel watershed model; 3D variably-saturated + overland flow; HPC-capable; open-source; Python tools (pftools). Strong fit for high-res 3D. |
| **HYDRUS (2D/3D)** | Mature variably-saturated flow; widely used/validated; 3D module is commercial; scripting more limited. |
| **MODFLOW 6 + FloPy** | Very widely used groundwater model; UZF (unsaturated) + SFR (streamflow) packages; strong Python (FloPy) scripting; surface-runoff coupling weaker. |
| **Integrated SSF codes** (CATHY, GEOtop, tRIBS, …) | Purpose-built surface–subsurface coupling; varying maturity/community/licensing. |

*(Candidates are starting points for evaluation, not endorsements. Verify each model's current capabilities and license before relying on the notes above.)*

## Evaluation plan
1. Finalize requirements + weighting (driven partly by what `aerial-mapping` can deliver and what PIDS features demand).
2. Shortlist 2.
3. Run a small benchmark: represent a single conveyance channel + infiltration tunnel + barrier on a synthetic slope; check it reproduces expected behavior and the relevant patent claims (C-001, C-004).
4. Record the decision here + in `governance/decision-log.md`; update `technologies/` status and the tooling stack in `docs/plans/`.

## Decision
*(pending)*
