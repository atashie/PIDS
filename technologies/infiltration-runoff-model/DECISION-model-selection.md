# Decision Record: Forward Model Selection

- **Status:** **DECIDED 2026-06-04** — BUILD a modular PETSc-FEM forward model on DOLFINx/FEniCSx (Option B), conditional on a DOLFINx + variational-inequality install spike. See the [decision memo](../../docs/plans/2026-06-03-pids-forward-model-engine-evaluation.md).
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
**DECIDED 2026-06-04 — Option B: BUILD a modular forward model on an open PETSc-FEM framework (DOLFINx / FEniCSx, Python), rather than extend an off-the-shelf engine.**

This **revises** the original framing (Pillar 2 "extends a pre-existing, validated model"). The full evaluation + de-risk — see the [decision memo](../../docs/plans/2026-06-03-pids-forward-model-engine-evaluation.md) — found that **no single existing engine** simultaneously meets R1 (coupled 3-D Richards + overland), R2 (local sub-meter refinement without a global fine mesh), and R9 (engineered internal BCs, dominated by the **reverse one-way catch-valve** — the *inverse* of a tile drain). A standalone numerical probe proved the catch-valve is tractable **only as an implicit in-solver hook** (operator-split coupling blows up), which an FEM weak form realizes natively (interior-facet residual term + PETSc variational-inequality solver).

- **Engine/stack:** DOLFINx (FEniCSx) over PETSc; Python; runtime on WSL2 + conda. Reverse catch-valve as an implicit VI / interior-facet residual term.
- **Validation benchmark:** HydroGeoSphere (cross-check only — not the production engine).
- **Discrete-feature pattern:** mixed-dimensional 1D-in-3D embedding for PIDS channels/tunnels (decouples sub-meter conduit geometry from a coarse 3-D matrix).
- **Eliminated as production engine:** ParFlow (symmetric flow barriers / operator-split only); MODFLOW 6 (no released 3-D Richards + coupled overland); closed/commercial (HGS, HYDRUS); copyleft/research-license (DuMuX, GEOtop, CATHY, InHM).
- **Conditional gate:** a DOLFINx install + variational-inequality smoke spike on the target machine before the architecture is committed (cautious-building guardrail).
- **Development methodology:** strictly modular, bottom-up, with the three-tier sanity-check routine ([`../../governance/claude-sanity-check-routine.md`](../../governance/claude-sanity-check-routine.md)) and standardized HTML visualization ([`../../governance/visualize-sanity-check-routine.md`](../../governance/visualize-sanity-check-routine.md)).

Recorded in [`../../governance/decision-log.md`](../../governance/decision-log.md) (2026-06-04). Benchmark `PIDS-MIN-1` (one channel + tunnel + barrier on a synthetic Piedmont slope, checking claims C-001/C-004) remains the production-commitment validation, per the [decision memo](../../docs/plans/2026-06-03-pids-forward-model-engine-evaluation.md).
