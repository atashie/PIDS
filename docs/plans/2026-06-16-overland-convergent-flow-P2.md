# Convergent-Flow P2 Implementation Plan — productionize the O1 upwind overland scheme in `CoupledProblem`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the validated monotone O1 upwind-mobility edge-flux overland scheme (P1, standalone `UpwindOverlandProblem`) the lateral-overland operator inside the *coupled* `[ψ, d, λ]` solver `CoupledProblem`, so the convergent-flow regime (the PIDS swale) gets the upwind scheme's positivity / accuracy / dt-tractability in the fully-coupled product engine — galerkin path preserved bit-identical, limiter demoted to a tripwire.

**Architecture:** The coupled solve is a monolithic block Newton (`dolfinx` high-level `NonlinearProblem`, exact UFL auto-Jacobian) over UFL residual forms `[F_psi, F_d, F_lam]`. The lateral overland is ONE UFL term in `F_d` (`overland_flux`, a tangential-gradient galerkin diffusion). P2 replaces *only that term* with the non-UFL edge-flux on the **top-facet edge graph**, supplied as a custom residual + analytic Jacobian on the **d-block alone**; ψ, λ/NCP, outlets, drainage, and the interior-pin stay UFL and untouched. Because the high-level `NonlinearProblem` cannot carry a non-UFL term, the upwind path runs a **custom block-SNES** (residual callback = block-assemble the reduced UFL forms + add the edge-flux node residual into the d-rows; Jacobian callback = block-assemble the reduced UFL auto-Jacobian + add the analytic edge-flux d–d block). It is **opt-in** (`overland_scheme="upwind"`); the galerkin default is unchanged until a final Arik-gated flip. Conservation stays structural: the edge flux telescopes to zero over the surface, and the λ sign-pairing across `F_d`/`F_psi` is untouched.

**Tech Stack:** Python, DOLFINx 0.10 + PETSc/petsc4py (WSL2 conda env `pids-fem`), pytest. Run pattern (every command below assumes this): from `technologies/infiltration-runoff-model/forward-model/`, with `PATH=/root/miniforge3/envs/pids-fem/bin:$PATH` (conda is NOT on PATH non-interactively; `gcc` from the env is needed for FFCX JIT), `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, and `PYTHONPATH=.`. The package is NOT pip-installed.

---

## Background & context (read before starting)

- **Parent plan:** `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` — §5 P2 bullet = the scope; **§8.7 = the CORRECTED P1/O1 spike VERDICT** (the §8.6 gate-inventory had a gate-9 accuracy OVER-CLAIM; it was caught in B5 review, investigated in B5b `6e1f63f`, and corrected in §8.7 at head `1e18920`). **Read §8.7 before this plan.** **P1 plan:** `docs/plans/2026-06-14-overland-convergent-flow-P1.md` (Part B = the standalone spike this productionizes). **Standalone scheme + carried decisions:** `pids_forward/physics/overland_upwind.py` (the module docstring is the authoritative spec: `eps_H=1e-3`, `eps_S=1e-3`, cotangent `T_e`, M-matrix guard, FD-vs-analytic Jacobian).
- **What P1 proved (so P2 inherits, not re-litigates) — per the CORRECTED §8.7:** the monotone upwind two-point edge flux on the surface edge graph (i) REMOVES the convergent-V sawtooth (oscillation RMS 0.013%→0.0036%, mesh-convergent — the real stability win), (ii) is MONOTONE without a limiter (V run-min `−0.0`; steep strict `≥−1e-12`; characterized sub-mm mild-2% caveat), (iii) is CONSERVATIVE (books `≤1e-13·cum_rain`), (iv) LIFTS the dt-pin ≈370× (0 rejections, 0.4 s vs galerkin 39.5 h / 60k rejections), and (v) has a CORRECT convergence-line flux. **Accuracy framing (CRITICAL, do not repeat the over-claim):** the lumped plateau `≈1.000·Q_eq` (and field-scale `1.0000` vs galerkin `0.876`) is a CONSERVATION/EQUILIBRIUM identity (the outlet sink and `outflow_rate()` share control-length weights `B_k`, so the residual telescoping FORCES `Σ q_out·B_k = rain·area = Q_eq` for ANY converged steady field — it certifies conservation + that equilibrium was reached, and that O1 EQUILIBRATES the field-scale V where galerkin does not; it is NOT outlet-flux accuracy). The GENUINE accuracy measure is the CONSISTENT ds-integral `∫86400·(1/n)·d^(5/3)·√S ds`: ~0.85 and refinement-DIVERGING on the idealized measure-zero KINK V (a B5b-characterized P1-thin-channel ARTIFACT, shared by galerkin, NOT a flux defect — the valley follows the Manning normal-depth law exactly) but converging UPWARD to ~0.99 (≤1%) on a RESOLVED finite-width swale (the actual PIDS use case). Conclusion: **O1's flux is correct; no flux fix is warranted; the resolved swale is the accuracy fixture (P3).** 18 TDD tests, suite 154 passed; P1 closed at head `1e18920`.
- **The coupled engine, exact seams (read these first):**
  - `pids_forward/physics/coupling.py:206-207` — the UFL lateral term to replace: `H_s, K_s = overland_conveyance(self.d, self.z_b, self.n_man, self.eps_S, grad=gT)`; `overland_flux = K_s * ufl.dot(gT(H_s), gT(vd)) * self._ds_top`.
  - `coupling.py:219-221` — `self._F_d_bulk = ((d-d_n)/dt)*vd*ds_top + overland_flux + lam*vd*ds_top`. **P2 builds an alternative `self._F_d_bulk` WITHOUT `overland_flux` when `overland_scheme="upwind"`.**
  - `coupling.py:515-519` — `self._problem = NonlinearProblem([F_psi, F_d, F_lam], [psi, d, lam], bcs=self._bcs, kind="mpi")`; `coupling.py:549-602` — `step()` calls `self._problem.solve()`, reads `snes.getConvergedReason()/getFunctionNorm()`, books reason-4 vs `stall_accept_fnorm`, records outflow/drainage, runs `_enforce_positivity`, advances `_n` shadows / restores on rejection. **The upwind path replaces the `_problem.solve()` line with a custom block-SNES; everything else in `step()` is reused verbatim.**
  - `coupling.py:522-546` — `_enforce_positivity` (the cm-scale clip-and-rescale to demote to a tripwire on the upwind path).
  - `coupling.py:147-149` (`_ds_top`), `:143-145` (`self._top_facets`), `:230-240` (the interior `below` pin + `_bcs` + `_eps_diag` diagonal allocation), `:171-174` (`_dx_storage` lumped). The surface fields `d/λ` are co-located on the host mesh and pinned to 0 below the top; the mesh top is a FLAT plane at `z=ztop`; topography is the `z_b` FIELD (head `H = z_b + d`), exactly like the standalone scheme (where `z_b` is also a field, not mesh tilt). So the top-facet triangulation is planar in (x,y) and the cotangent transmissibility is computed in that plane — the standalone 2-D math transfers directly.
  - The outlet (`add_outflow_bc`, codim-2 vertex/ridge UFL sink, `:364-423`), drainage GHB (`add_drainage_bc` on `F_psi`, `:439-476`), and NCP (`_F_lam_bulk` Fischer-Burmeister, `:222`) are UFL and STAY in the reduced forms — P2 does not touch them (`λ/NCP/outlet/inlet terms unchanged`, parent §5 P2).
- **The reusable standalone kernels (to extract, DRY):**
  - `overland_upwind.py:266-376` `_build_edge_graph_2d` — cotangent `T_e` (= negated assembled stiffness off-diagonal, pinned by `test_cotangent_T_e_equals_negated_stiffness_offdiagonal_2d`), M-matrix guard, `A_i = ∫φ_i dx` lumped.
  - `overland_upwind.py:423-471` `_assemble_residual` — the vectorized edge-flux node residual (`Q_e = T_e·M(d_up)·(H_i−H_j)`, smoothed-upwind `w=½(1+tanh(ΔH/eps_H))`, `np.add.at` telescoping).
  - `overland_upwind.py:379-421` `_setup_snes` — the standalone's own FD-Jacobian SNES (NOT reused in the coupled path; the coupled path drives its own block-SNES).
- **Conservation argument (state it in tests, do not re-derive in code):** summing the d-rows of the upwind residual, every interior edge contributes `+Q_e` to one row and `−Q_e` to the other → the lateral flux telescopes to **zero** over the closed surface; so `Σ_i R_d,i = Σ(storage) + Σ(λ) + Σ(rain) + Σ(outlet)`, identical to the galerkin balance. The λ term pairs with `F_psi`'s `−λ` (untouched). Therefore `Δtotal = cum_rain − cum_outflow − cum_drainage (+ clip_adjust→0 on the tripwire path)` holds structurally, as in galerkin.

**The committed P2 decisions (Codex review will stress-test these):**
1. **Opt-in, not replace.** Add `overland_scheme: str = "galerkin"` to `CoupledProblem.__init__`. `"galerkin"` = today's path, BIT-IDENTICAL (every existing coupled test passes unchanged). `"upwind"` = the new path. The default flip to `"upwind"` is a SEPARATE, final, Arik-gated task (E2) after Tier-3 sign-off — not assumed mid-plan.
2. **Replace only the lateral term; keep the monolithic block + λ-conservation.** Not an operator-split. The upwind flux is internal to the d-block; the `[ψ,d,λ]` Newton stays monolithic so the NCP active-set and exchange remain fully implicit.
3. **Analytic d–d edge-flux Jacobian (FD-verified), auto-Jacobian for the rest.** The flux is C¹ (tanh selector), so a local 2×2-per-edge analytic Jacobian is well-defined and assembles into the d–d block; the reduced UFL forms keep their exact auto-Jacobian. **Fallback (documented spike outcome, Task C2):** a frozen-mobility Picard d–d block (the cotangent-stiffness with `M` held at the iterate) if the analytic Newton is unhealthy — mirroring how P1 accepted an FD Jacobian for the standalone. Full-block FD is rejected up front (FD over the 3-D Richards block is prohibitively expensive).
4. **Top-facet cotangent edge graph, planar.** The surface edge graph is the top-facet sub-triangulation's edges; transmissibility from the (x,y) cotangents; `A_i = ∫φ_i ds_top` lumped (so `Σ A_i = top area` and `total_water/surface_water` stay consistent). Restriction (same as standalone): structured-box / Delaunay NON-OBTUSE top; M-matrix guard raises loudly otherwise.
5. **Limiter → tripwire on the upwind path only.** `_enforce_positivity` stays active for galerkin; on upwind it becomes an assert/log tripwire (loud if `min d < −tol`), never a silent clip-and-rescale. `clip_mass_adjust` stays 0 on the upwind path (asserted).

**DECISION POINTS for Codex / Arik (flagged, not blocking the draft):**
- **(DP-1) Jacobian strategy** — analytic vs frozen-mobility Picard as the SHIPPED default (decision 3). The plan codes analytic + FD-verify, with the Picard fallback measured in C2; Codex should sanity-check that the analytic derivation (incl. the tanh-selector and Manning-mobility chain terms) is worth the bug-surface vs starting with Picard.
- **(DP-2) Kernel extraction scope** — Part A extracts the standalone edge-graph/residual/Jacobian into shared pure functions reused by BOTH classes (DRY) and asserts the standalone tests stay bit-identical. Codex should check this refactor does not subtly perturb the validated standalone path. (If judged too risky, the fallback is to DUPLICATE the kernel into the coupled path and leave `overland_upwind.py` untouched — uglier but lower-risk; called out in Task A1.)
- **(DP-3) Default flip timing** — E2 proposes flipping the default to `"upwind"`; whether that ships in P2 or waits for P3 (the swale fixture) is Arik's call at the Gate-E checkpoint.

**Codex review (2026-06-16) — FOLDED IN below.** Two BLOCKERS now have explicit tasks: (i) the block-matrix **sparsity union** — removing `overland_flux` from UFL drops the surface-edge off-diagonals from the auto-Jacobian pattern, so the d–d edge nonzeros must be pre-seeded or MUMPS reallocates/fails (new Task B2; the standalone already preallocates its edge sparsity, `overland_upwind.py:398-412`); (ii) **MPI ownership/ghost** — the coupled target is `kind="mpi"`, so the upwind path needs an explicit owned-row / ghosted-read contract (Task B2; P2 is SERIAL-SCOPED with a loud multi-rank guard, mirroring the existing `add_outflow_bc` MPI deferral, `coupling.py:390-394`). Should-fixes folded in: BC handling simplified (the top-only graph never touches pinned dofs → standard constrained-row zeroing, NO re-lifting — Task B3); a **coupled-level** Jacobian FD smoke test (`J·δ` vs FD of the full assembled residual, pins active — Task C1, and acceptance bar 5); lifecycle invalidation of the cached upwind structures on `set_topography`/`_finalize_forms` (Task B3); an explicit `Σ R_edge = 0` conservation test (Task A3); and a residual-equality (not just graph-equality) extraction guard (Task A1).

**Scope honesty:** this is a multi-session plan (the custom block-SNES + analytic Jacobian + 3-D top-facet graph are each non-trivial). Each Part ends with a CHECKPOINT — report to the architect before the next Part. Parts A→D are TDD; Part C (Jacobian/Newton-health) and parts of D carry empirical degrees of freedom and use validation-gates (analytic/FD/reference comparisons) like the P1 Part-B spike. **Verification is SERIAL** (single-rank), as throughout this project; multi-rank MPI for the upwind path is an explicit deferred item with a guard (Task B2).

**Working location:** worktree `C:\Users\arikt\Documents\GitHub\PIDS-b6-docs` on branch `b6-tilted-v-convergent-flow` (HEAD = `1e18920`, P1 CLOSED: B5b diagnosis `6e1f63f` + corrected B6 verdict §8.7). ⚠️ **CONCURRENCY:** the B5b/B6-correction commits landed on this worktree concurrently with the P2 drafting — confirm no other session is committing here before executing. All commits land here. Do NOT touch the main checkout `C:\Users\arikt\Documents\GitHub\PIDS` (concurrent M4 session; it also holds `main` checked out, so the b6→main merge — 20↔37 diverged, with the `8a891c2` per-sink-accounting-inside-O5 reconciliation — is a SEPARATE deferred task, not part of P2).

**Baseline check (Task 0 — do this first):**
Run: `PATH=/root/miniforge3/envs/pids-fem/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/ -q --no-header`
Expected: green at head `1e18920` (≈154+ passed; the B5b/B6 correction `1e18920` edited `tests/test_overland_upwind.py`, so RE-CONFIRM the exact count). The full-suite terminal SUMMARY line is suppressed by an M4 session-finish print — confirm via the process EXIT CODE 0, and/or `python -m pytest tests/test_overland_upwind.py -v`. If not green, STOP and reconcile before starting.

---

# Part A — the top-facet edge graph + reusable edge-flux kernels (geometry; NO solver change)

**Why first:** the coupled path needs the edge graph on the 3-D mesh's top facet, plus the residual/Jacobian math, as building blocks BEFORE any solver surgery. Validate them in isolation (graph identities, residual reproduces the standalone V) so Part B integrates known-good pieces.

### Task A1: Extract the edge-flux kernels into shared pure functions (DRY; standalone bit-identical)

**Files:** Create `pids_forward/physics/overland_edge_kernel.py`; modify `pids_forward/physics/overland_upwind.py`; test `tests/test_overland_edge_kernel.py`.

**Step 1 — failing test (pin the kernel API + that it equals the standalone internals).**
```python
# extract: build_edge_graph_2d(V, mesh) -> (edges, L_e, T_e, A_i); edge_flux_residual(d, z_b, d_n, rain, dt, edges, L_e, T_e, A_i, n_man, eps_S, eps_H, outflows) -> R
def test_kernel_matches_standalone_internals_2d():
    msh = _unit_v_mesh(8, 5)
    prob = UpwindOverlandProblem(msh, n_man=0.03)        # builds its graph internally
    edges, L_e, T_e, A_i = build_edge_graph_2d(prob.V, msh)
    assert np.array_equal(np.sort(edges, axis=1), np.sort(prob.edges, axis=1))
    assert np.allclose(T_e_aligned(edges, T_e, prob), prob.T_e, atol=0, rtol=0)   # bit-identical
    assert np.allclose(A_i, prob.A_i, atol=0, rtol=0)

def test_kernel_residual_equals_standalone_at_nontrivial_state_2d():
    """Codex review: guard the EXTRACTION on the RESIDUAL, not just the graph. Drive a random
    non-flat positive state through both edge_flux_residual(...) and the standalone _assemble_residual
    path; assert bit-identical (the refactor must not perturb the validated numerics)."""
    ...
    assert np.allclose(R_kernel, R_standalone, atol=0, rtol=0)
```

**Step 2 — run, verify fail** (module missing).

**Step 3 — implement.** Move the body of `overland_upwind._build_edge_graph_2d` (`:266-376`) and the residual math of `_assemble_residual` (`:423-471`) into `overland_edge_kernel.py` as pure functions `build_edge_graph_2d(V, mesh)` and `edge_flux_residual(...)`. Re-point `UpwindOverlandProblem` to call them (its attributes/behaviour unchanged). **DP-2 fallback:** if the refactor risks the validated path, instead COPY the math into the kernel module and leave `overland_upwind.py` calling its own private methods — note which was chosen in the module docstring.

**Step 4 — run the FULL standalone suite, assert BIT-IDENTICAL.**
Run: `... pytest tests/test_overland_upwind.py -q` → `18 passed` (the standalone path must be unchanged — this is the DP-2 guard).
Run: `... pytest tests/test_overland_edge_kernel.py -q`.

**Step 5 — commit:**
```bash
git add pids_forward/physics/overland_edge_kernel.py tests/test_overland_edge_kernel.py pids_forward/physics/overland_upwind.py
git commit -m "P2-A1: extract edge-graph + edge-flux-residual kernels (DRY); standalone bit-identical"
```

### Task A2: Build the TOP-FACET edge graph on a 3-D host (the new geometry)

**Files:** `pids_forward/physics/overland_edge_kernel.py`; `tests/test_overland_edge_kernel.py`.

**Why:** in 3-D the surface is the top-facet sub-triangulation embedded in the host tet mesh (`overland_upwind.py:242` literally defers "the 3-D top-facet ridge graph is the P2 productionization"). The cotangents are planar (x,y) at `z=ztop`; `A_i = ∫φ_i ds_top` lumped.

**Step 1 — failing tests (graph identities, mirroring the standalone 2-D pins).**
```python
def test_top_facet_cotangent_equals_negated_ds_top_stiffness_3d():
    """T_e on the top facet == negated assembled  -∫ gT(phi_i)·gT(phi_j) ds_top  off-diagonal."""
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0,0,0],[L,1,1]], [8,4,4])
    V = fem.functionspace(msh, ("Lagrange", 1))
    top_facets = _locate_top(msh, V)
    edges, L_e, T_e, A_i = build_top_facet_edge_graph(V, msh, top_facets)
    # assemble the ds_top tangential-gradient stiffness; compare off-diagonals (FFCX-safe quad cap)
    assert np.allclose(T_e, _negated_ds_top_stiffness_offdiag(V, msh, top_facets, edges), atol=1e-12)

def test_top_facet_A_i_sums_to_top_area_3d():
    ...                                   # sum(A_i over top dofs) == top-facet area; interior dofs = 0
    assert abs(A_i[top_dofs].sum() - L*1.0) < 1e-12

def test_top_facet_m_matrix_guard_holds_structured_and_raises_obtuse_3d():
    ...                                   # structured box top: T_e >= -1e-14; an obtuse top remesh raises
```

**Step 2 — run, verify fail** (`build_top_facet_edge_graph` missing).

**Step 3 — implement `build_top_facet_edge_graph(V, mesh, top_facets)`.** From `top_facets` (the located codim-1 entities at `z=ztop`): `create_connectivity(fdim, 0)` (facet→vertices) and `create_connectivity(fdim, 1)`/`(1,0)` (facet→edges, edge→vertices). For each top facet (a triangle), accumulate the (x,y)-plane cotangent of the angle opposite each of its edges into `T_e[edge]` (reuse the `_cot_opposite` math from `overland_upwind.py:326-334` on the (x,y) coords); collect the unique surface edges as dof pairs (vertex→dof via the cell-local map, `overland_upwind.py:318-324`). `A_i` = assemble `∫φ_i ds_top` with the vertex quadrature rule (lumped) — reuse the `_ds_top` measure construction (`coupling.py:147-149`). Apply the SAME M-matrix guard (`overland_upwind.py:349-360`). Edges live in the host `Vd` dof index space; off-top dofs never appear.

**Step 4 — run, verify pass** + standalone suite still `18 passed`.

**Step 5 — commit:**
```bash
git commit -am "P2-A2: top-facet planar cotangent edge graph on the 3-D host (graph identities + M-matrix guard)"
```

### Task A3: Cross-check — the top-facet edge-flux residual reproduces the standalone V (no solver yet)

**Files:** `scratch/_p2_topfacet_residual_xcheck.py`; `tests/test_overland_edge_kernel.py`.

**Validation gate:** drive `edge_flux_residual` standalone-style on a 1-cell-thick 3-D slab whose top facet matches a 2-D V geometry, and confirm the assembled lateral flux equals the standalone `UpwindOverlandProblem` 2-D flux to ~1e-12 at a fixed non-flat state (the 3-D embedding must be a pure relabelling of the 2-D scheme). This is the coupled analogue of the galerkin `test_2d_overland_operator_matches_module2` (`test_coupling_2d.py:24-72`).

**Also (Codex review — pin the conservation root directly):** a unit test that the LATERAL part of the top-facet residual sums to exactly zero over the surface at an arbitrary non-flat state — `Σ_i (edge-flux contribution to R_i) == 0` to machine precision (the telescoping that makes coupled conservation structural; isolates it from storage/rain/outlet).
```python
def test_edge_flux_residual_sums_to_zero_3d():
    R_edge = edge_flux_residual(d_rand, z_b, d_n=d_rand, rain=0.0, dt=1.0, ...)  # storage cancels (d==d_n), no rain/outlet
    assert abs(R_edge.sum()) < 1e-12
```
```bash
git add scratch/_p2_topfacet_residual_xcheck.py && git commit -am "P2-A3: top-facet edge-flux residual == standalone 2-D flux (3-D embedding is a relabelling)"
```

**>>> CHECKPOINT (Gate A): report to the architect.** Graph identities + residual cross-check green; standalone path bit-identical. Confirm DP-2 (kernel extraction) before solver surgery.

---

# Part B — the opt-in `overland_scheme="upwind"` custom block-SNES (inexact Jacobian first)

**Nature:** integration + scaffolding. Get a CONVERGING coupled upwind solve with a SIMPLE (frozen-mobility Picard) d–d Jacobian first, proving residual assembly + BCs + conservation; the analytic Jacobian + Newton-health is Part C. Galerkin default stays bit-identical throughout.

### Task B1: Add `overland_scheme` + the reduced `F_d` (no solver change yet)

**Files:** `pids_forward/physics/coupling.py`; `tests/test_coupling_upwind.py` (new).

**Step 1 — failing test:** `CoupledProblem(msh, soil, overland_scheme="upwind")` constructs; its `F_d` UFL form has NO `overland_flux` term (the lateral conveyance is absent from UFL), while `"galerkin"` is unchanged.
```python
def test_upwind_scheme_omits_overland_flux_from_ufl_Fd():
    g = CoupledProblem(_box(), SOIL)                      # default galerkin
    u = CoupledProblem(_box(), SOIL, overland_scheme="upwind")
    assert g.overland_scheme == "galerkin" and u.overland_scheme == "upwind"
    # the reduced d-residual assembles to a DIFFERENT vector at a non-flat state (flux removed from UFL)
    ...
def test_galerkin_default_unchanged():
    assert CoupledProblem(_box(), SOIL).overland_scheme == "galerkin"
```

**Step 2 — run, verify fail.**

**Step 3 — implement.** Add the ctor param; store `self.overland_scheme`. When `"upwind"`, build `self._F_d_bulk` WITHOUT `overland_flux` (`coupling.py:219-221` → storage + `lam*vd*ds_top` only), and stash the top-facet edge graph via `build_top_facet_edge_graph` (Task A2). Validate `overland_scheme in {"galerkin","upwind"}` (raise otherwise). Reject unsupported combos early with clear errors (e.g. `degree != 1`).

**Step 4 — run; assert the FULL existing suite is bit-identical on galerkin.**
Run: `... pytest tests/test_coupling_1d.py tests/test_coupling_2d.py tests/test_coupling_3d.py tests/test_coupling_3d_tier2.py -q` (all pass; galerkin untouched).

**Step 5 — commit:** `git commit -am "P2-B1: opt-in overland_scheme='upwind'; reduced F_d omits the UFL lateral flux (galerkin bit-identical)"`

### Task B2: Block-matrix sparsity union + MPI ownership/ghost contract (the two Codex blockers)

**Files:** `pids_forward/physics/coupling.py`; `tests/test_coupling_upwind.py`.

**Why (before any solve):** removing `overland_flux` from UFL drops the surface-edge `d–d` off-diagonals from the auto-Jacobian pattern; inserting them later via `MatSetValues` into a block AIJ that did not preallocate them forces a reallocation (slow) or fails outright under MUMPS. And the coupled block is `kind="mpi"`, so reads of `(psi,d,lam)` from the SNES vector `X` and writes of residual/Jacobian must respect ownership/ghosting. Settle BOTH structurally before B3 wires the callbacks.

**Decisions this task locks:**
- **Sparsity union:** create the block Jacobian matrix from the union of (a) the reduced-UFL block pattern (via `dolfinx.fem.petsc.create_matrix_block` on the `J_ij` forms) and (b) the top-edge `d–d` structural nonzeros (diagonal per top dof + the two symmetric off-diagonals per surface edge — the exact pattern the standalone seeds at `overland_upwind.py:398-412`). Seed those `d–d` slots as explicit structural zeros up front. Pin it with a test that no MALLOC happens on the first Jacobian insert (`A.setOption(MAT_NEW_NONZERO_ALLOCATION_ERR, True)` then assemble once — raises if the pattern is short).
- **Ownership/ghost contract (SERIAL-SCOPED for P2):** build the top-facet edge graph on OWNED top dofs; read `(psi,d,lam)` from `X` with the block index-map offsets + a ghost scatter before evaluating the edge residual; insert residual/Jacobian only into owned rows. **Multi-rank is DEFERRED:** add a loud guard `if mesh.comm.size > 1 and overland_scheme=="upwind": raise NotImplementedError(...)` (mirroring the documented `add_outflow_bc` MPI caveat, `coupling.py:390-394`) so the serial-correct path can never silently mis-run in parallel. (Verification is serial throughout this project.)

**Step 1 — failing tests:**
```python
def test_upwind_block_jacobian_preallocates_edge_nonzeros():
    """The added d-d edge nonzeros must be IN the block pattern (no realloc on first insert)."""
    prob = CoupledProblem(_box(), SOIL, overland_scheme="upwind")
    A = prob._build_upwind_block_matrix()          # NEW (Task B2)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    prob._assemble_upwind_jacobian(A, prob._pack_state())   # must NOT raise (pattern is complete)

def test_upwind_rejects_multirank_until_supported():
    # comm.size>1 + overland_scheme="upwind" raises NotImplementedError (serial-scoped, guarded)
```

**Step 2 — run, verify fail.**

**Step 3 — implement** the block-matrix builder (union pattern), the state pack/unpack (block offsets + ghost scatter), the owned-row insertion helpers, and the multi-rank guard.

**Step 4 — run, verify pass** + galerkin suite bit-identical (this task adds upwind-only scaffolding; galerkin untouched).

**Step 5 — commit:** `git commit -am "P2-B2: upwind block-matrix sparsity union + serial ownership/ghost contract + multi-rank guard (Codex blockers)"`

### Task B3: The custom block-SNES — residual callback + Picard Jacobian + lifecycle

**Files:** `pids_forward/physics/coupling.py`; `tests/test_coupling_upwind.py`.

**Architecture:** when `overland_scheme=="upwind"`, `_ensure_problem` builds a `PETSc.SNES` over the monolithic block instead of the high-level `NonlinearProblem`, reusing the Task-B2 matrix + state helpers:
- Precompute the 3×3 Jacobian forms `J_ij = ufl.derivative(F_i, u_j)` for the REDUCED residuals `[F_psi, F_d_reduced, F_lam]` (what `NonlinearProblem` does internally); the residual block vector via `create_vector_block`.
- **Residual callback `F(snes, X, B)`:** unpack `X`→`(psi,d,lam)` with ghost scatter (Task B2); `assemble_vector_block(B, [F_psi,F_d_reduced,F_lam], Jforms, bcs)`; then ADD the edge-flux node residual (`edge_flux_residual`, top dofs only) into the d-block offset of `B` on OWNED rows.
- **BC handling (Codex should-fix — simplified):** the top-only edge graph never references the pinned off-top dofs, so the manual term needs NO separate lifting; rely on the standard block-assembly constrained-row treatment (`assemble_*_block` with `bcs` zeroes/sets the pinned rows). Do NOT re-lift after adding the edge term (that only invites a sign/order bug). Pin this with `test_upwind_pinned_interior_d_stays_zero`.
- **Jacobian callback `J(snes, X, A, P)`:** `assemble_matrix_block(A, Jforms, bcs)` into the Task-B2 preallocated matrix; then ADD the **Picard** d–d block (frozen-mobility cotangent stiffness: per edge `+T_e·M(d_up)` to `(i,i)/(j,j)`, `−` to `(i,j)/(j,i)`, `M` at the current `d`) into the d-block offset (pattern already preallocated); `A.assemble()`.
- **Lifecycle (Codex should-fix):** cache `J_ij`, the block vec/mat, and the edge graph; invalidate them (set the upwind problem to `None`, like `_problem=None`) on `set_topography`/`_finalize_forms`/`add_*` so a changed source/topography/outlet rebuilds them — matching the galerkin invalidation at `coupling.py:362,503,285`.
- Reuse `self._petsc_options` (MUMPS LU, monolithic AIJ). Expose `.solver`/reason/fnorm so `step()` (Task B4) is unchanged.

**Step 1 — failing tests (coupled upwind sanity, mirroring `test_coupling_2d/3d` + the standalone gates):**
```python
def test_upwind_coupled_lake_at_rest_held_2d():       # uniform H -> no spurious lateral flow; d held
def test_upwind_coupled_closed_conserves_2d():        # closed sloped 2-D, smooth rain<Ks: |Δtotal-cum_rain|<=1e-9
def test_upwind_coupled_closed_conserves_3d():        # 3-D box analogue (mirror test_coupling_3d:52)
def test_upwind_pinned_interior_d_stays_zero():       # off-top d-dofs remain exactly 0 after a solve
def test_upwind_topography_change_rebuilds_problem(): # set_topography invalidates the cached upwind structs
```

**Step 2 — run, verify fail** (upwind solve not wired).

**Step 3 — implement** the block-SNES as above; dispatch in `_ensure_problem` on `self.overland_scheme`.

**Step 4 — run, verify pass** + galerkin suite still bit-identical.

**Step 5 — commit:** `git commit -am "P2-B3: custom block-SNES for the upwind path (residual + Picard d-d Jacobian + lifecycle); coupled lake/conservation gates"`

### Task B4: Dispatch `step()` to the upwind solve; preserve booking + λ-restore

**Files:** `pids_forward/physics/coupling.py`; `tests/test_coupling_upwind.py`.

**Step 1 — failing test:** an upwind `step()` returns `(converged, iters)`, sets `last_reason/last_fnorm`, advances `_n` shadows on convergence and restores `(psi,d,lam)` on rejection — identical contract to galerkin (`coupling.py:549-602`).

**Step 3 — implement:** make the `self._problem.solve()` line (`:554`) dispatch (galerkin `NonlinearProblem` vs the upwind block-SNES); keep the reason-4 norm-recompute, `stall_accept_fnorm` gate, outflow/drainage recording, and `_n`/λ restore VERBATIM. (The limiter call `:587` stays for now — demoted in Part D.)

**Step 5 — commit:** `git commit -am "P2-B4: step() dispatch galerkin/upwind; identical booking + reason gate + λ-restore"`

**>>> CHECKPOINT (Gate B): report.** Coupled upwind solve converges with conservation; galerkin bit-identical; sparsity/ownership blockers resolved (B2). Jacobian is the inexact Picard placeholder (Part C makes it analytic).

---

# Part C — analytic d–d edge-flux Jacobian + Newton-health decision (DP-1)

### Task C1: Analytic edge-flux Jacobian, FD-verified

**Files:** `pids_forward/physics/overland_edge_kernel.py` (add `edge_flux_jacobian_dd`); `pids_forward/physics/coupling.py`; `tests/test_overland_edge_kernel.py`, `tests/test_coupling_upwind.py`.

**Why:** the Picard Jacobian omits `∂M/∂d` and `∂w/∂d` (the tanh selector) — exact Newton needs them. The local per-edge Jacobian is 2×2 (`∂(R_i,R_j)/∂(d_i,d_j)`); derive it through `Q_e = T_e·M(d_up)·ΔH` with `M = c·max(d_up,0)^{5/3}/(n·(S_f²+eps_S²)^{1/4})`, `d_up = w·d_i+(1−w)·d_j`, `w = ½(1+tanh(ΔH/eps_H))`, `ΔH = (z_b+d)_i−(z_b+d)_j`, `S_f = ΔH/L_e`.

**Step 1 — failing tests (TWO gates — Codex should-fix: kernel FD is necessary but NOT sufficient).**
```python
def test_edge_flux_jacobian_matches_central_fd_2d():        # KERNEL-level
    # assemble the analytic d-d edge-flux block at a random non-flat positive state; compare to a
    # central finite-difference of edge_flux_residual; assert max rel-err < 1e-6 per nonzero entry.

def test_upwind_coupled_jacobian_matches_fd_smoke():        # COUPLED-level (block offsets + sign + BC)
    # at a random plausible coupled state with the pins active, compare the assembled BLOCK Jacobian
    # action J*delta against a central finite-difference of the FULL assembled coupled residual along
    # a random direction delta; assert rel-err < 1e-6. Catches block-offset/sign/BC-overwrite bugs the
    # kernel FD cannot (the inserted d-d block lives inside the [psi,d,lam] block matrix).
```

**Step 2 — run, verify fail.**

**Step 3 — implement `edge_flux_jacobian_dd(...)`** (vectorized per-edge 2×2 contributions into the d–d block) and wire it into the Jacobian callback BEHIND a flag `jacobian="analytic"|"picard"` (default `"analytic"`, fallback retained).

**Step 4 — run, verify pass** (BOTH the kernel and the coupled-level FD gate) + the coupled conservation/lake tests still pass with the analytic Jacobian.

**Step 5 — commit:** `git commit -am "P2-C1: analytic d-d edge-flux Jacobian (FD-verified); picard kept as fallback"`

### Task C2: Newton-health probe + the shipped-Jacobian decision

**Files:** `scratch/_p2_coupled_newton_health.py`; module docstring; (DP-1 resolution).

**Validation gate (empirical, like P1-B3):** on a coupled 3-D hillslope storm (mirror `test_coupling_3d.py:232` / `test_coupling_3d_tier2.py`), record iteration counts, dt distribution, and convergence-reason histograms for galerkin vs upwind-analytic vs upwind-picard. **Decide** the shipped default Jacobian (analytic if it converges robustly with fewer iters; picard if analytic is fragile) and record the choice + the trade in the `coupling.py` upwind-path docstring. State the dt-pin behaviour vs galerkin.
```bash
git add scratch/_p2_coupled_newton_health.py && git commit -am "P2-C2: coupled Newton-health probe; shipped Jacobian chosen empirically (DP-1)"
```

**>>> CHECKPOINT (Gate C): report.** Analytic Jacobian FD-verified; Jacobian default decided with data.

---

# Part D — coupled accuracy + positivity + limiter demotion

### Task D1: Positivity WITHOUT the clip; demote the limiter to a tripwire (upwind path)

**Files:** `pids_forward/physics/coupling.py`; `tests/test_coupling_upwind.py`.

**Step 1 — failing tests:**
```python
def test_upwind_coupled_positive_without_clip_3d():
    # a coupled 3-D storm on the upwind path keeps min d >= -1e-12 across the run with NO clip firing:
    assert prob.max_clip_seen == 0.0 and prob.clip_mass_adjust == 0.0
def test_upwind_limiter_is_a_tripwire_not_a_clip():
    # on the upwind path _enforce_positivity must NOT rescale; a forced negative state RAISES/logs loudly
```

**Step 3 — implement:** on `overland_scheme=="upwind"`, replace the `_enforce_positivity` call (`coupling.py:587`) with a tripwire (assert `min d >= -tol` with a loud message + recorded diagnostic; NO clip/rescale; `clip_mass_adjust` stays 0). Galerkin keeps the existing clip. Honor the P1 mild-front caveat: `tol` is a small sub-mm bound (document it; the 2% valley measured `-0.0` on the standalone V but adversarial mild mounds reach ~0.9 mm — if the coupled swale trips it, that is a real finding for P3, not a silent clip).

**Step 5 — commit:** `git commit -am "P2-D1: upwind path positive without clip; limiter demoted to a loud tripwire"`

### Task D2: Coupled accuracy — operator-equivalence to the standalone, routing, conservation/equilibrium, dt-pin

**Files:** `scratch/_p2_coupled_v_plateau.py`; `tests/test_coupling_upwind.py`.

**Framing (per the CORRECTED §8.7 — do NOT repeat the over-claim):** the lumped plateau `≈Q_eq` is a CONSERVATION/EQUILIBRIUM identity, not discharge accuracy; the genuine accuracy measure is the CONSISTENT ds-integral, which is artifact-contaminated (~0.85, refinement-diverging) on the idealized measure-zero KINK V and only meaningful on a RESOLVED finite-width swale. **The absolute resolved-swale accuracy benchmark is P3** (the PIDS-swale Tier-2 fixture). P2's accuracy job is therefore OPERATOR EQUIVALENCE: prove the coupled upwind reproduces the *validated standalone* upwind scheme — so P2 inherits P1's correctness instead of re-deriving it.

**Validation gates (mirror the galerkin coupled tests + the standalone):**
- **Operator equivalence (the decisive P2 accuracy gate):** on the SAME planar surface geometry/forcing, the coupled-upwind CONSISTENT ds-integral discharge AND the depth field match the standalone `UpwindOverlandProblem` to tight tolerance (≤~1e-3 relative, allowing the coupling's λ/infiltration on a near-impermeable bed) at a fixed plateau state — the coupled analogue of `test_coupling_2d.py:24-72` (`test_2d_overland_operator_matches_module2`), but for the upwind operator. This carries P1's resolved-swale ~0.99 result into the coupled engine WITHOUT re-running the (P3) swale benchmark.
- **Lateral redistribution** (mirror `test_coupling_2d.py:210`): heavy rain on a sloped coupled surface routes downslope, downhill > 1.5× uphill, total conserves — on the upwind path.
- **3-D routing + conservation** (mirror `test_coupling_3d.py:232`): hillslope with a ridge outlet, `Δtotal = cum_rain − cum_outflow − cum_drainage`, `d ≥ −1e-12`.
- **Conservation/equilibrium plateau (relabelled, NOT accuracy):** the coupled plateau lumped discharge → `Q_eq` (the steady-state mass-balance identity, ±3% as an equilibrium-reached + flat-plateau check), oscillation ≤2% RMS — explicitly documented as conservation/equilibrium, per §8.7.
- **dt** distribution vs galerkin (expect the pin lifted, per §8.7); record.
```bash
git add scratch/_p2_coupled_v_plateau.py && git commit -am "P2-D2: coupled upwind == standalone (operator equiv) + routing + conservation/equilibrium plateau + dt-pin"
```

### Task D3: Full-suite regression

**Step 1 — full suite.** Run: `... pytest tests/ -q --no-header` → expect `>= 154 + new upwind tests passed` (galerkin bit-identical + the new `tests/test_coupling_upwind.py` + `tests/test_overland_edge_kernel.py`). Confirm via EXIT CODE 0 (the suite summary line is suppressed — see Task 0). A galerkin regression means the dispatch leaked into the default path — debug, do not loosen.

**Step 2 — commit** any fixes: `git commit -am "P2-D3: full-suite green (galerkin bit-identical + coupled-upwind tests)"`

**>>> CHECKPOINT (Gate D): report.** Coupled upwind meets the accuracy/positivity/dt gates; suite green.

---

# Part E — Tier-3 re-baseline, verdict, sign-off

### Task E1: Tier-3 benchmark re-baseline on the upwind path (B4/B5/B6 as regressions)

**Files:** the Tier-3 runners/HTML builders under `technologies/infiltration-runoff-model/{benchmarks,parflow}/` + `viz/`; new `validation/sanity/` record.

Per the three-tier routine (`governance/claude-sanity-check-routine.md`): re-run the coupled Tier-3 comparisons (B4 1-D ponding, B5 3-D hillslope, B6 tilted-V) with `overland_scheme="upwind"`, build/refresh the HTMLs, and capture any tolerance/visual change vs the galerkin baselines (esp. the field-scale accuracy gain and the dt/runtime). **Arik visual sign-off is required for any Tier-3 change** — do not self-approve. Write the sign-off package to `validation/sanity/pids_p2_coupled_upwind__2026-06-DD.md`.
```bash
git add -A && git commit -m "P2-E1: Tier-3 coupled-upwind re-baseline (B4/B5/B6 regressions + HTMLs) for Arik sign-off"
```

### Task E2: P2 verdict + the default-flip proposal (DP-3) + memory

**Files:** `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` (new **§8.8** "P2 productionization result" — §8.6/§8.7 are the P1 verdict); `coupling.py` + `overland_edge_kernel.py` docstrings; memory `pids-convergent-flow-priority.md`.

Write the P2 verdict against the parent §5 P2 acceptance list (each pass/fail + number); record carried decisions (DP-1 Jacobian, DP-2 extraction, edge-graph/`A_i` forms, the tripwire tol). **Propose** the default flip `galerkin → "upwind"` (DP-3) for Arik's decision at the checkpoint — keep galerkin as a permanent fallback mode either way. Update memory: NEXT = **P3** (re-benchmark + the permanent PIDS-swale Tier-2 fixture). Note the still-open b6→main merge (`8a891c2`-inside-O5).
```bash
git commit -am "P2-E2: P2 verdict vs the §5 P2 gate list + default-flip proposal; parent §8.8; memory NEXT=P3"
```

**>>> CHECKPOINT (Gate E / P2 complete):** use superpowers:finishing-a-development-branch.

---

## Acceptance bars (P2 definition of done)

1. **Opt-in + bit-identical galerkin:** `overland_scheme="galerkin"` (default) leaves EVERY existing test bit-identical; `"upwind"` is the new path; the default flip is a separate Arik-gated decision (E2/DP-3).
2. **Lateral term only:** only `overland_flux` is replaced; ψ, λ/NCP, outlet, drainage, and the interior-pin are untouched; the `[ψ,d,λ]` Newton stays monolithic (no operator-split).
3. **Conservation structural:** `Δtotal = cum_rain − cum_outflow − cum_drainage` to ≤1e-6·cum_rain (target ≤1e-9) on the upwind path (2-D + 3-D), with `clip_mass_adjust == 0` and the limiter never clipping.
4. **Positivity WITHOUT the clip:** coupled `min d ≥ −1e-12` (steep) on the upwind path; `_enforce_positivity` demoted to a loud tripwire (any residual mild-front undershoot is sub-mm + characterized, never silently rescaled).
5. **Jacobian correct (kernel AND coupled):** the analytic d–d edge-flux Jacobian matches central-FD to <1e-6 at the kernel level; AND a COUPLED-level check (the assembled block `J·δ` vs a finite-difference of the full assembled residual, pins active — Codex review) passes; the shipped default chosen from the Newton-health probe (DP-1).
6. **Coupled accuracy = operator equivalence (NOT a lumped-Q_eq accuracy claim, per §8.7):** the coupled upwind reproduces the validated STANDALONE upwind scheme (consistent ds-integral discharge + depth field) on the same geometry to ≤~1e-3; downslope routing reproduced; the lumped plateau → `Q_eq ± 3%` is recorded as a CONSERVATION/EQUILIBRIUM check (relabelled), oscillation ≤2% RMS; field-scale lumped ≈1.0 (equilibration, not discharge accuracy); dt distribution vs galerkin reported (pin expected lifted). The absolute resolved-swale accuracy benchmark is P3.
7. **Top-facet graph guarded:** cotangent `T_e == −(ds_top stiffness off-diagonal)`, `Σ A_i == top area`, M-matrix guard holds on the structured box / raises on obtuse.
8. **Regression:** full suite green (EXIT 0); Tier-3 B4/B5/B6 re-baselined on the upwind path with Arik sign-off on any change.

## Risks / open questions
- **(DP-1) Analytic Jacobian bug-surface.** The tanh-selector + Manning-mobility chain Jacobian is fiddly; the FD-verify gate (C1) and the Picard fallback (B2/C2) bound the risk. If analytic is fragile, ship Picard and make analytic a P-later optimization (the parent-plan "hand Jacobian is the P2 performance item" already frames it this way).
- **(DP-2) Kernel-extraction perturbing the validated standalone path.** Guarded by the `tests/test_overland_upwind.py` bit-identical assertion (A1 Step 4); duplicate-the-kernel is the fallback.
- **Custom block-SNES vs the high-level `NonlinearProblem`.** Replicating `NonlinearProblem`'s block assembly (Jacobian forms via `ufl.derivative`, `assemble_*_block`, BC lifting) is the largest single piece; B2 isolates it with conservation/lake gates before accuracy.
- **3-D top-facet non-obtuse restriction.** Same as the standalone (structured box / Delaunay); the M-matrix guard raises loudly otherwise. Unstructured/obtuse `T_e` (perpendicular-bisector / non-negative two-point) is P3.
- **Mild-front sub-mm positivity in the coupled swale.** If the tripwire trips on the real swale geometry, that is a P3 finding (semismooth Newton fallback), not a P2 blocker — surfaced honestly, never clipped.
- **The b6→main merge is NOT in scope** (20↔37 diverged + `8a891c2`-inside-O5); a separate task.

## Artifacts
- This plan; parent `docs/plans/2026-06-11-overland-convergent-flow-stabilization.md` (§5 P2; §8.6 gate inventory + §8.7 corrected P1 verdict; §8.8 to be added); P1 `docs/plans/2026-06-14-overland-convergent-flow-P1.md`.
- New: `pids_forward/physics/overland_edge_kernel.py`; `tests/test_overland_edge_kernel.py`, `tests/test_coupling_upwind.py`; scratch `_p2_topfacet_residual_xcheck.py`, `_p2_coupled_newton_health.py`, `_p2_coupled_v_plateau.py`.
- Touched: `pids_forward/physics/coupling.py` (opt-in scheme + block-SNES + tripwire), `pids_forward/physics/overland_upwind.py` (kernel re-point); Tier-3 runners/HTMLs under `benchmarks/`+`viz/`.
