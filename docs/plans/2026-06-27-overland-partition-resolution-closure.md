# Overland Partition — Resolution Closure Plan (steps 1–3)

> **For Claude:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` to run this task-by-task. This is a
> SPIKE/research plan (scratch + benchmarks, no `pids_forward/` production edits until a closure is
> validated) — the "tests" are pre-registered GATES run in the WSL `pids-fem` env. Confirm each gate before
> moving on. **This plan is GATED on Arik's approval of the §22 finding + the 3-soil viz.**

**Goal:** Settle the rain/runoff partition for the PIDS overland model: (1) pin the mesh-converged value,
(2) reconcile it with the ParFlow benchmark, (3) build a mesh-objective subgrid infiltration-capacity
closure so the COARSE (field-scale-tractable) cell delivers the resolved sorptive partition.

**Architecture:** §22 (record `validation/sanity/overland_partition_bug_investigation__2026-06-24.md`)
established — across 5 Codex review rounds — that the "0.547 runoff" is a COARSE-SURFACE-RESOLUTION
artifact: the real routed b1 loam partition collapses 0.547 → ~0.27 as the top vertical cell refines
(uniform & graded agree; both slopes; clean), because the monolith's infiltration cap
`q_pot = kirchhoff(ψ,d)/ell_c` is a sub-grid FILM model that under-captures sorptivity on coarse cells.
Open: the exact converged value, the ParFlow reconciliation, and a mesh-objective closure.

**Tech Stack:** DOLFINx 0.10 FEM (`CoupledProblem`, `RichardsProblem`), WSL conda `pids-fem`, serial,
threads pinned. ParFlow toolchain in `technologies/infiltration-runoff-model/{parflow,benchmarks}/`.
Matplotlib for viz. Reuses `scratch/seq_partition_topref.py` (the 3-D top-refinement ladder),
`seq_sorptivity_meshconv.py` (1-D ladder), `seq_partition_viz.py` (the 3-soil viz).

---

## Load-bearing lessons (do NOT relitigate)
- **Never pipe a long run through `tail`** (buffers to EOF → lost on kill). Write live to a file; Read it.
- **Always guard `_march` with `max_steps`**; run independent cases as separate single-threaded processes
  (BLAS pinned: `OMP/OPENBLAS/MKL_NUM_THREADS=1`); parallelize via processes, not threads.
- **`ell_c` is auto = top-cell half-height** (`coupling.py:191-209`); do NOT sweep it free on a fixed mesh
  (that was the §20 mis-framing). Refine the MESH; let `ell_c` follow.
- **Verify `git branch` = main before every commit** (investor-branch gotcha).
- **WSL run pattern:** `wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u scratch/<script>.py ... > scratch/_out.txt 2>&1'`
- **Hedge claims to what the test shows** (5 Codex rounds caught over-reach each time). External-review each verdict.

**Established ladder (b1_base loam, routed/R, the quantity these tasks refine):**
uniform nz 8/16/32/64 → 0.547/0.419/0.319/0.275 (ell_c 62.5/31/15.6/7.8 mm);
graded p 1.0/1.5/2.0/2.5 → 0.547/0.364/0.287/0.266 (top 125/44/16/5.5 mm). Both still DECREASING.

---

## Task 1 — Pin the mesh-converged partition value

**Why:** §22 is hedged "at least halved, still decreasing, NOT converged at ~0.27." Need the asymptote (is
it ~0.27, or does it keep dropping toward the 1-D no-film ~0.21?). Graded refinement reaches a fine top
cell cheaply (keeps nz modest), so it is the affordable way to the asymptote.

**Files:** reuse `scratch/seq_partition_topref.py` (graded mode); new analysis in `scratch/_topref_conv.txt`.

**Step 1.** Run two finer graded rungs for b1_base AND b1_steep: `p=2.75` (top ~1.1 mm) and `p=3.0` (top
~0.58 mm). Command per run (background, live file):
`python -u scratch/seq_partition_topref.py b1_base 2.75 > scratch/_topref_base_p275.txt 2>&1` (and p3.0;
and b1_steep). 4 runs, parallel.

**Step 2. GATE (convergence):** is `routed/R(p=3.0) − routed/R(p=2.75)` ≤ ~1.5 pp on BOTH slopes? If yes →
report the converged value `routed/R* ≈ value(p=3.0)` + a Richardson estimate from the graded ladder. If
NO (still dropping >1.5 pp) → it is approaching the 1-D no-film ~0.21; report "≤ value, not yet converged"
and note one more rung needed. Either way: confirm base ≈ steep (slope-insensitive) holds at the fine end.
Watch solver health (a sub-mm top cell may stiffen the coupled Newton — record `ns`, any collapse).

**Step 3.** Record the converged value (or bound) in §23 of the investigation record; commit.

**Decision:** the validated partition target for b1 loam becomes `routed/R*` (replaces 0.547 as the
reference). If solver collapses at p=3.0, fall back to the uniform nz=96 rung (slower but robust).

---

## Task 2 — Reconcile with the ParFlow benchmark (is 0.55 universally under-resolved?)

**Why:** §22's "ParFlow ~0.55 was a code-to-code match at a shared coarse resolution, not continuum truth"
is the ONE inference Codex said is NOT yet earned — it needs ParFlow re-run at fine vertical resolution. If
ParFlow ALSO collapses toward ~0.27 → universal under-resolution (both codes), strong external
corroboration. If ParFlow STAYS ~0.55 at fine dz → the gap is specific to our `q_pot` and the in-house
collapse needs re-examination.

**Files:** `technologies/infiltration-runoff-model/{parflow,benchmarks}/` — locate the B5/B6 ParFlow
runner (grep `build_comparison_coupled_3d.py`, the benchmarks README §5; memory
[[pids-parflow-benchmark]]). New: `benchmarks/<name>_dz_refine.*` + record.

**Step 1.** FEASIBILITY FIRST: determine whether ParFlow is runnable in this environment (the memory notes
a "native ParFlow build" was deferred — pftools/python-pf may be present but the solver binary may not).
Grep the parflow silo for the runner + how it was invoked; check for a ParFlow binary / docker / pftools.
**GATE:** can a ParFlow case actually execute here? If NO → document the blocker, write the EXACT
fine-dz ParFlow input deck (a TCL/py-pf script with refined `ComputationalGrid.NZ` / `dz` near the
surface) for a later native-ParFlow run, and mark Task 2 BLOCKED-pending-ParFlow (do not fake it). If YES
→ proceed.

**Step 2.** Identify the ParFlow case whose in-house twin gave ~0.55 (the B5 loam hillslope or the closest
runoff-partition case). Re-run it at its baseline vertical resolution (reproduce ~0.55) then at 2×, 4×
near-surface vertical refinement, same forcing. Record routed/R (or the ParFlow runoff diagnostic) per dz.

**Step 3. GATE:** does ParFlow's runoff partition DROP materially with near-surface dz refinement (like the
in-house 0.547 → ~0.27), or hold ~0.55? Record the verdict in §24 + reconcile with the in-house ladder
(do they converge to the SAME value at matched resolution?). Commit.

**Decision:** ParFlow-collapses → "the coarse partition is universally under-resolved" is EARNED; ParFlow-
holds → re-open whether the in-house collapse is partly a scheme artifact (escalate). Either resolves the
last open inference.

---

## Task 3 — Mesh-objective subgrid infiltration-capacity closure (the production fix)

**Why:** mm-scale surface cells are NOT field-scale tractable (a 30×20×64 column is already minutes/run;
field domains are far larger). The fix (Codex-endorsed, GSSHA/tRIBS family) is a SUBGRID
infiltration-CAPACITY law so the COARSE cell delivers the resolved sorptive partition — NOT brute
refinement, NOT a tuned constant `ell_c`. Replace `q_pot = kirchhoff(ψ,d)/ell_c` with a capacity `f_c(t)`
from Green-Ampt / Smith-Parlange (or an adaptive `ell_eff(state,soil,t)` derived from sorptivity), driven
by the cumulative infiltration the cell has accepted.

**Files:** new `scratch/seq_subgrid_capacity.py` (a `CoupledProblem` or `SequentialCoupledProblem` subclass
overriding the surface acceptance); reuses `constitutive.py` (kirchhoff, K, sorptivity), the §21/§22
ladders as the resolved TRUTH targets.

**Step 1. Derive the capacity law.** Green-Ampt: `f_c = Ks·(1 + (ψ_f·Δθ)/F)`, with `F` = cumulative
infiltration, `ψ_f` = effective front suction = `|kirchhoff(ψ_i,0)|/Ks` (validated form, §19),
`Δθ = θ_s − θ(ψ_i)`. Smith-Parlange as the alt. Write it as a per-top-node capacity that caps the surface
infiltration each step; excess ponds + routes; track per-node cumulative `F` (a surface state array). Keep
the conservative ledger (the §12 machinery: matched-quadrature, no reconstruction).

**Step 2. 1-D GATE FIRST (cheap):** on the 1-D column (`seq_sorptivity_meshconv.py` fixtures), the GA
capacity closure on a COARSE nz=8 cell must reproduce the RESOLVED `I(t)` (the nz=240 / add_ponding_bc
reference) within ~5-10% across loam/sand/clay — i.e. mesh-OBJECTIVE (coarse == fine). This is the core
test: does the capacity law remove the `ell_c` dependence? Gate before any 3-D work.

**Step 3. 3-D GATE:** the closure on the COARSE (nz=8 uniform) b1 tilted plane must give `routed/R ≈
routed/R*` (Task 1's converged value), NOT 0.547 — on BOTH slopes, stable, conservative (`bal/rain`
~1e-11, falsification fires). Compare wall-clock: coarse+closure vs the resolved mesh (the whole point is
coarse-and-correct).

**Step 4.** If the GA capacity generalizes (1-D + 3-D gates pass): write up §25 + a PRODUCTION design/TDD
plan to promote it into `pids_forward/physics/` as the surface acceptance (a selectable option alongside
the existing q_pot). If it does not generalize: characterize where it fails (soil/storm), try
Smith-Parlange or the adaptive-`ell_eff` variant; fall back to "document the resolution requirement +
recommend graded surface meshes."

---

## Success criteria (the gates, in order)
1. **Converged value** — graded p=2.75/3.0 flatten (≤1.5 pp) → a validated `routed/R*` for b1 (Task 1).
2. **ParFlow reconciled** — ParFlow collapses with dz like the in-house (universal), or the blocker is
   documented + the deck written (Task 2).
3. **Mesh-objective closure** — a GA/Smith-Parlange capacity makes the COARSE cell reproduce the resolved
   `I(t)` (1-D) and `routed/R*` (3-D), conservative + stable + soil-general (Task 3).

## Risks & fallbacks
- **Task 1 solver stiffness** at sub-mm top cells → fall back to uniform nz=96/128 (robust, slower).
- **Task 2 ParFlow not runnable here** → write the deck, mark BLOCKED (do NOT fabricate ParFlow numbers).
- **Task 3 GA doesn't generalize** (heterogeneity, redistribution between storms, the no-Ss clay) → it is a
  genuine research closure; fall back to Smith-Parlange / adaptive `ell_eff`, or document the graded-mesh
  requirement as the interim production guidance.

## After validation (out of scope here)
Promote the validated closure to `pids_forward/physics/` with TDD (mesh-objectivity regression tests:
coarse == fine partition), coexisting with the existing schemes.
