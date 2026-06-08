# Module 3 → 3-D: implementation plan (next session)

**Date:** 2026-06-07 · **Author:** Arik + Claude · **Status:** plan, not yet started
**Goal:** Extend the coupled `CoupledProblem` (realization A, co-located `[ψ, d, λ]` on the host volume)
from 2-D to **3-D**, with the standard discipline: spike → TDD → Codex review → three-tier assessment →
commit. Most of the model is already **dimension-agnostic**; the one genuinely new piece is the **lateral
outflow outlet**, which in 3-D is a codim-2 *perimeter curve* (vs a single vertex in 2-D).

---

## 0. Load-context checklist (do first)
- **Env:** WSL2, conda `pids-fem` (DOLFINx 0.10.0, PETSc 3.25.1). **Always** `export OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before python — multithreaded BLAS oversubscribes the tiny
  systems ~30–50× (memory `pids-fem-blas-threading`). Run tests:
  `wsl -d Ubuntu -e bash -lc "source ~/miniforge3/etc/profile.d/conda.sh && conda activate pids-fem && cd /mnt/c/Users/arikt/Documents/GitHub/PIDS/technologies/infiltration-runoff-model/forward-model && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && python -m pytest -q"`
  (call `wsl` directly; a PowerShell `cd` first breaks the path. Scripts need `PYTHONPATH=.`.)
- **Read:** `pids_forward/physics/{coupling.py,richards.py,overland.py,constitutive.py}`;
  `tests/test_coupling_2d.py`, `tests/test_drainage_bc.py`; **`docs/plans/2026-06-05-module3-realization-ffcx-bug.md` §7**
  (3-D outlet notes + MPI tagging caveat); `governance/decision-log.md` (last ~5 rows).
- **Baseline:** suite **68/68** at commit `00c4ecf`. Realization **A** (S blocked by the FFCX 0.10 bug —
  do NOT revisit S here). Gravity is on the **last** coordinate axis (`_zaxis = gdim-1`).

## What already generalizes for free (verify, don't rebuild)
- Bulk Richards, the **sorptive Kirchhoff** land-surface NCP leg, lateral **overland** (tangential-gradient
  surface PDE on `ds_top`, `grad_T`), and the **subsurface drainage GHB** (`add_drainage_bc`, codim-1
  facet term on sides/base) are all dimension-agnostic — sides/base are codim-1 facets in 3-D too.
- The pinned-interior `d/λ` diagonal allocation uses a **vertex** `dP` integral → vertices are 0-dim in any
  dimension, so it carries over. Mesh: swap `create_rectangle` → `create_box`; `set_topography(z_b(x,y))`.

## The one hard piece: the outflow outlet in 3-D
2-D: outlet = downstream-TOP **corner vertex** (codim-2) → `add_outflow_bc` imposes the Manning
normal-depth discharge as a vertex `dP` integral, woven into `F_d` by `_finalize_forms` (shared vertex
meshtags: pin=tag1, outlets=tag2+). **3-D: the outlet is the downstream top EDGE — a codim-2 perimeter
curve (1-D ridges).** Two paths (spike the first, fall back to the second):
- **(R) ridge integral** — impose `∫_{outlet edge} q_out·v_d` as a codim-2 **`ridge`** measure (FFCX has a
  `ridge` entity type). Clean generalization of the vertex `dP`. **Risk:** unproven on stock FFCX 0.10 for a
  single mesh; the realization-S bug was a *mixed-dim codim-0* path, so a single-mesh ridge integral *may*
  be fine — **the spike decides.** Note: `ridge` and `vertex` (the pin) are different integral types, so
  both can live in one `F_d` with separate `subdomain_data` (no meshtags clash).
- **(B) normalized downstream band on `ds_top`** (fallback, dimension-agnostic) — a thin facet strip near
  the outlet, sink `q_out/band_width` as a standard codim-1 `ds` integral. Robust (no ridge codegen), but
  approximate (smears the line discharge over the band; document the width-convergence).

## Phased plan

**Phase 0 — Spike the codim-2 outlet (highest risk first).**
`scratch/m3_3d_outlet_spike.py`: tiny `create_box`, tag the downstream top edge, assemble
`∫ q(d)·v ds_ridge` + its Jacobian. If FFCX compiles + assembles cleanly → adopt **(R)**. If it hits a
codegen error → adopt **(B)** (band) and record the failure. **→ Codex review of the spike + the chosen
path before building** (this is the consequential design call).

**Phase 1 — 3-D smoke (verify the agnostic core).**
3-D closed box (no outlet), rain on, impermeable sides/base: assert conservation (structural balance) +
finite/stable. `tests/test_coupling_3d.py::test_3d_closed_conservation`. Confirms mesh/topography/pin/NCP
all work in 3-D. Keep meshes SMALL (e.g. 6×6×5) — 3-D dof count is the perf constraint.

**Phase 2 — Outflow BC in 3-D (TDD).**
Generalize `add_outflow_bc` to the edge outlet via the Phase-0 path (R or B). RED→GREEN:
`test_3d_outflow_conservation` (Δtotal = cum_rain − cum_outflow), `test_3d_downslope_routing_to_edge`
(a tilted box drains to the downstream edge; ponding piles upslope). Keep `outflow_rate`/`cum_outflow`.

**Phase 3 — Port the 2-D suite to 3-D (TDD).**
3-D analogs (small meshes): flat conservation (structural, tolerance-free `⟨F_d,1⟩` guard), drainage GHB
on a side face (codim-1, should pass), sorptivity recovery, and a **reduces-to-2-D** check (a 1-cell-thick
box ≈ the 2-D result). Reuse the 2-D assertions; only the mesh/locators change.

**Phase 4 — Standard review + three-tier assessment.**
- **Codex adversarial review** (code + physics + the ridge/band choice + conservation + MPI tagging).
- **Tier-2**: synthetic typical + 100-yr-extreme forcing on a 3-D hillslope; stability + conservation.
- **Tier-3 viz** (3-D is the viz challenge): a self-contained HTML with (a) surface ponding `d(x,y)` map
  (heatmap/surface), (b) 1–2 vertical ψ cross-section slices, (c) the partition / outlet hydrograph /
  conservation time-series (reuse the 2-D panels). Generator in `viz/`, data → `validation/sanity/data/`
  (gitignored), HTML → `validation/sanity/viz/` (gitignored). **Arik visual sign-off** gate.
- **Commit** (decision-log row + memory update + spec/plan).

## Risks / watch-items
- **Ridge codegen** (Phase 0) is the real unknown — spike before committing to (R); (B) is the safe net.
- **3-D dof count / perf** — keep verification meshes coarse; single-threaded BLAS pinned; direct solve is
  fine at small N, consider iterative only if needed.
- **Outlet tagging not MPI-safe** (ffcx-bug doc §7) — verification is serial; note it, defer the MPI fix.
- **Surface-film head caveat** persists (flux correct, near-surface ψ reads dry) — not a 3-D issue, just
  carry the caveat into the assessment framing.
- Honest framing in the assessment (per the last Codex FIX-FIRST): label storage vs fluxes correctly;
  report any numerical aids (ramps, band width) as such; trust conserved partition/flux over state proxies.

## Definition of done
3-D `CoupledProblem` with rain + sorptive infiltration NCP + lateral overland routing + outflow at the
codim-2 outlet + subsurface drainage GHB, passing 3-D Tier-1 tests and conserving; Codex-reviewed;
Tier-2 + Tier-3 assessment done with Arik sign-off; committed (suite green). Then: file the FFCX bug
upstream / migrate to realization S when fixed; Module 4 = §E embedded-feature sorptive leg.
