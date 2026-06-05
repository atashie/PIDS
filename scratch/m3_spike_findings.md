# Module 3 (coupling) — M0 cross-mesh spike findings (2026-06-05)

**Gate:** design R1 — can DOLFINx 0.10 assemble a blocked `[ψ(host volume), d(top-facet
submesh)]` residual with an EXACT auto-derived coupled Jacobian and one Newton solve, realizing
the land-surface exchange `q_ls = k_ex·(d − ψ_top)` sign-paired into both blocks?

Baseline confirmed first: full suite **39/39** in `pids-fem` (WSL2). DOLFINx **0.10.0**.

## VERDICT (corrected after Codex review): the design-intended SUBMESH realization IS buildable

> **Correction.** An earlier draft of this note concluded the submesh realization was "not
> buildable" — that was WRONG, caught by an independent Codex review. The error: I integrated the
> coupling on the *submesh* `dx` (probe2), which puts the volume field ψ at codim −1. The
> supported DOLFINx 0.10 idiom (per the official HDG demo) integrates the coupling on the
> **parent `ds_top`**, where ψ sits at its natural codim-1 facet trace and the submesh fields
> (d, v_d) at codim-0 — both legal. `probe4` confirms the full realization end to end.

What is actually true about the constraints (still useful):
1. `fem.form` is **single-integration-domain**, so the d-block residual (surface op on submesh
   `dx` + coupling on host `ds_top`) is assembled as the **sum of two single-domain forms**, and
   the high-level `NonlinearProblem([F0,F1])` convenience wrapper can't host it → use **low-level
   blocked assembly + a blocked Newton**.
2. The coupling must be integrated on the **parent `ds_top`**, NOT the submesh `dx` (the latter
   trips FFCX `codim = 1 − 2 = −1`).

**`probe4` (design-intended realization) PASSES:** ψ on the host volume, `d` on the codim-1
top-facet submesh, coupling on parent `ds_top`, cross-mesh relation via the `create_submesh`
`EntityMap`, blocked Newton. Results: exact cross-block Jacobian (1 Newton iter at moderate
k_ex), continuity guard `L2|ψ_top−d|` collapses 5.7e-2 → 5.9e-8 as k_ex→∞, **conservation exact**
(`q_into_ψ + q_out_of_d = 0.0` — EXACT Galerkin sign-pairing, not lumped), and `d` is thin
(13 DOFs vs ψ 169 — **DOF-efficient, no 2× penalty**). This is the design's intended architecture
("realization S"), and it dominates both fallbacks below.

## Fallback A (was recommended pre-Codex; now a fallback): facet-restricted `d` co-located on host (F4′)

`ψ` and `d` are two P1 functions on the SAME host mesh. The land-surface coupling
`q_ls = k_ex·(ψ_top − d)` is then a pure HOST `ds_top` facet integral — standard single-mesh
FEM, **no entity_maps, no cross-mesh tabulation** — so the blocked `NonlinearProblem`
(`kind="mpi"`, monolithic block AIJ) assembles the EXACT auto-derived coupled Jacobian. `d`'s
non-surface DOFs are Dirichlet-pinned to 0 (surface storage = a clean top-facet integral); a
tiny `eps·d·v_d·dx` term allocates interior diagonals so LU doesn't hit "missing diagonal".

Verified end-to-end (probe3 / F4′, 2-D, k_ex ∈ {1e0,1e2,1e4,1e6}):
- converges every k_ex (1–2 Newton iters ⇒ exact Jacobian);
- **continuity/datum guard**: `L2|ψ_top−d| = 0.3/k_ex` → 3e-1, 3e-3, 3e-5, 3e-7 — collapses to
  ~0 (a spurious `+z_surf` datum bug would plateau the gap near the top elevation 1.0);
- **structural conservation**: flux into ψ == −flux out of d to machine zero;
- **interior |d| = 0 exactly**.

Cost: `d` carries volume DOFs (~doubles unknowns; interior rows are trivial Dirichlet identity
rows, cheap for LU, eliminable later). The 2-D/3-D surface overland operator must be expressed
with **tangential gradients on `ds_top`** (`grad_T = grad − (grad·n)n`) — so M2's
`OverlandProblem` (full `grad` on its own mesh) is **not directly reusable** for the coupled
2-D/3-D surface; its diffusion-wave *physics* (Manning K_s(d), limiter, eps_S) is.

## Alternative (DOF-efficient, more complex): standalone surface mesh + P-operator (F3)

Keep `d` on a small standalone surface mesh (reuse M2's `OverlandProblem` verbatim), couple via
an explicit node-matching interpolation operator `P` and manual PETSc coupling-block assembly in
a custom SNES. DOF-efficient and reuses M2, but hand-rolled cross-Jacobian blocks (more code,
more risk). Better revisited as a perf optimization than adopted up front.

## Option B investigated (Arik's ask: is B worth it to avoid A's 2x?) — `m3_investigate_B.py`

**A's real cost (measured, NOT the ~10% I first guessed):** co-locating `d` on the host roughly
DOUBLES the system, and lumping the interior `eps` term does NOT reduce it (sparsity is allocated
from the form structure):

| mesh | variant | unknowns | matrix nnz | LU solve | nnz× / LU× |
|---|---|---|---|---|---|
| 2-D 96² | ψ-only | 9 409 | 65 089 | — | 1.00 / 1.00 |
| | A co-located | 18 818 | 131 716 | — | **2.02 / 1.5–1.8** |
| 3-D 20³ | ψ-only | 9 261 | 128 581 | — | 1.00 / 1.00 |
| | A co-located | 18 522 | 270 204 | — | **2.10 / 1.7–1.9** |

**B's coupling core works and is simple.** Standalone surface mesh (`create_submesh` of the top
facets ⇒ nodes coincide with host top-facet nodes), a coordinate node-match, and a **lumped,
diagonal** coupling (4 `setValues` per matched node) reproduce A's continuity EXACTLY
(`L2|ψ_top−d| = 0.3/k_ex` → 3e-1…3e-7). `d` lives only on the thin surface (17 DOFs vs doubling).

**Assessment.** B saves the full ~2× memory + ~1.5–1.9× solve cost and reuses M2's
`OverlandProblem` nearly verbatim (its own mesh ⇒ no tangential-gradient rewrite that A needs).
B's price is solver PLUMBING: a custom PETSc SNES with manual block residual/Jacobian assembly,
manual BC application on the block system, and manual fieldsplit — ~+150–200 lines, more bug
surface (but the coupling math is trivial/diagonal, and adversarial TDD covers the plumbing).
Net: the A/B choice is a real simplicity-vs-efficiency tradeoff; the 2× is genuine, so B's
efficiency motive is legitimate for 3-D-at-scale + optimization loops.

## Recommendation (post-Codex): realization **S** (design-intended submesh)

Adopt **S** for 2-D/3-D: `d` on the top-facet submesh, coupling on parent `ds_top`, low-level
blocked assembly + blocked Newton, exact Galerkin Robin, `EntityMap` from `create_submesh`. It
dominates the fallbacks: DOF-efficient like B (no 2× of A), exact coupling unlike B's lumped
diagonal, MPI-robust entity maps (not B's coordinate node-match), and it's the design's intended
architecture. Its cost is the same low-level blocked-Newton plumbing B needs (no high-level
wrapper) — but done on the proper submesh with exact coupling. Keep **A** as the simplest fallback.

Codex's other valid points to fold into the build:
- add a **sloped-top** datum test (the flat-top k_ex→∞ continuity check alone doesn't catch every
  trace/topography datum mistake);
- the blocked Newton should be a proper MPI-safe driver (ghost updates, BC lifting on the block
  system) — reference `scifem`'s BlockedNewtonSolver; budget more than a serial-prototype's lines;
- verify M2's overland runs on the submesh as a **manifold** (gdim>tdim; UFL `grad` becomes the
  tangential gradient there — should be fine, but test it).

## Build-sequence consequence

1-D first is trivially single-mesh: the "surface" is the top POINT, overland is degenerate (no
lateral flux), so coupling = a point exchange between ψ_top and a scalar `d` — validates q_ls,
the k_ex→∞ continuity, partitioning, mass balance, recession with no cross-mesh machinery. The
submesh realization (S) only matters for 2-D/3-D lateral overland.
