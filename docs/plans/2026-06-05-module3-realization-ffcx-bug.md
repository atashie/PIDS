# Module 3 2-D/3-D realization decision + the FFCX mixed-dim codegen bug

**Date:** 2026-06-05
**Status:** DECIDED (Arik) — build 2-D/3-D coupling on realization **A** now; realization **S** deferred pending an upstream FFCX fix.
**Basis:** the M0 spikes (`scratch/m3_realizationS_spike.py`, `…_probe2/3/4.py`), two independent Codex adversarial reviews.

---

## 1. The two realizations

The monolithic surface↔subsurface coupling (Module 3, design §D) needs the surface fields
(`d` = ponding depth, `λ` = land-surface exchange flux) somewhere. Two realizations:

- **A — facet-restricted / co-located.** `d`,`λ` are P1 fields on the **host volume mesh**, pinned to
  0 below the surface; the Manning overland operator is a **tangential-gradient surface PDE on the
  host `ds_top`**. Single mesh — no submesh, no `entity_maps`, no mixed-dimensional assembly. This is
  exactly what the **working 1-D coupling already uses** (`pids_forward/physics/coupling.py`, suite
  45/45). Design §A.2 explicitly sanctions the facet-restricted realization.
- **S — top-facet submesh (design-intended).** `d`,`λ` live on a **codim-1 top-facet submesh**
  (`create_submesh`), coupling assembled on the parent `ds_top` via `entity_maps`. **DOF-efficient**
  (`d`,`λ` only on the surface, not the volume). This is the architecture §D/§A.2 prefers.

`probe4` proved S's cross-mesh assembly + exact coupled Jacobian works in principle (linear case),
and the explicit-`λ` formulation keeps every block single integration-domain. **But S is currently
unbuildable** — see §2.

## 2. The FFCX blocker (why S is deferred)

In DOLFINx **0.10.0** / FFCX, compiling S's coupled form set throws
`UnboundLocalError: cannot access local variable 't'` in `ffcx/ir/elementtables.py`
`build_optimized_tables` → `clamp_table_small_numbers(t["array"], …)`.

**Root cause (precise):** the element-table builder has

```
if (interior_facet or ridge or (is_mixed_dim and codim == 0)):
    if   entity_type == "facet": ...   # assigns t
    elif entity_type == "ridge": ...   # assigns t
    # ❌ NO else: entity_type == "cell" falls through, t never assigned
else:
    t = get_ffcx_table_values(...)     # the non-mixed-dim path
tbl = clamp_table_small_numbers(t["array"], …)   # crash: t unbound
```

The overland Manning operator on the submesh is a **mixed-dimensional, codim-0 CELL integral**
(`entity_type=="cell"`, `codim = tdim(1) − elem_cell_dim(1) = 0`), so when the coupled form set flags
the context `is_mixed_dim=True`, the pure-submesh self-Jacobian `∂F_d/∂d` hits the hole and crashes.
It compiles fine in isolation and only fails inside the full coupled form set. **Verified robust
against** per-block separate compilation, compiling the manifold block first, and clearing the FFCX
disk cache — none avoid it (the trigger is the *co-existence* of submesh + cross-mesh forms sharing
the coupled unknowns, which is unavoidable).

## 3. The fix (clean, but not for trusted results on a local patch)

The fix is a narrow missing branch — handle `entity_type=="cell"` (unpermuted table, identical to the
non-mixed-dim path), and **hard-error on any other unexpected entity type**:

```
    elif entity_type == "cell":
        t = get_ffcx_table_values(quadrature_rule.points, cell, integral_type, element,
                                  avg, entity_type, local_derivatives, flat_component, codim)
    else:
        raise RuntimeError(f"unsupported mixed-dim entity_type {entity_type!r}")
```

**Codex correctness review:** the unpermuted table is *probably correct* for this case — a cell
integral on the element's own cell is not a sub-entity trace, so the orientation-permutation logic
(which exists for facets/ridges) does not apply. **But** because the bug is in element-table codegen,
a subtly-wrong table = **silent Jacobian corruption** — the worst failure mode for a
research/publishable model. A broad `else` fallback is unsafe (silently mis-handles other entity
types); the patch must be **cell-only + hard-error**. And **"converges + matches A" is necessary but
not sufficient** general proof against silent corruption (it can miss other cell types, 3-D tets,
vector/tensor spaces, mesh orientations, MPI partitioning).

**Conclusion:** a cell-only patch is acceptable for *internal* S de-risking, **NOT** as the
production toolchain for trusted/published results.

## 4. Decision + path forward

- **Build the 2-D/3-D coupling on realization A now** — correct, publishable, on a stock FFCX, and a
  small extension of the already-approved 1-D code (overland becomes a tangential-gradient surface PDE
  on `ds_top`; the NCP/exchange physics and the Tier-1 tests are realization-agnostic).
- **File the FFCX bug upstream** with the narrow `entity_type=="cell"` fix + a minimal reproducer.
  It's an isolated, obvious one-branch change — the kind maintainers accept quickly.
- **Migrate to S when the upstream fix releases** — a drop-in DOF-efficiency win on a stock toolchain,
  no local patch carried. Because the physics/NCP/tests are realization-agnostic, the migration only
  swaps `d`,`λ` from co-located → submesh.

## 5. Cost of A (honest, Codex-corrected — do NOT oversell)

A puts 3 P1 fields on the host volume (`ψ`,`d`,`λ`); `d`,`λ` are physically meaningful only on the
~`N^(2/3)` surface, but **the algebra still pays O(N)** storage + assembly because they are
full-volume fields (interior pinned to 0 = trivial Dirichlet identity rows).

- **Measured (unoptimized direct LU, `d` co-located; `m3_investigate_B.py`):** ~2× unknowns, ~2× matrix
  nnz, ~1.5–1.9× LU time; with `λ` too, ~3× / ~2–2.5×.
- **Honest framing:** *tractable for verification-phase meshes now; probably acceptable with targeted
  optimization; NOT yet performance-cleared for production 3-D optimization loops.* Not "basically
  free after optimization" (an earlier overstatement, corrected).
- **Optimization levers (deferred, real work — not free):** diagonal interior allocation (cuts matrix
  nnz, not total solve cost); eliminate `λ` (3→2 fields, spec O1); iterative GMRES+`fieldsplit` (helps,
  but the surface-coupling off-diagonals + NCP block — not the identity rows — set the conditioning, so
  it's not automatic); and ultimately **realization S** once the FFCX fix lands (full DOF-efficiency).

> Bottom line: **S isn't dead — it's pending an upstream FFCX fix; A carries Module 3 correctly and
> publishably until then.**

## 6. 2-D shore-up (Codex review, 2026-06-06) + residual concerns

After the first 2-D pass, a Codex review flagged that the green suite was too weak to claim the 2-D
operator validated. Shore-up done (commit `7a26179`):

- **Operator equivalence (the key pin):** the co-located lateral overland (tangential gradient
  `grad_T` on host `ds_top`) is **machine-precision identical (~8e-15)** to standalone Module-2
  overland (`grad` on the surface mesh). `overland_conveyance` now takes a `grad=` arg so both paths
  SHARE one Manning formula. Test: `test_2d_overland_operator_matches_module2`.
- **Flat reduction:** a 2-D flat column matches the validated 1-D coupling quantitatively (`ψ_top`,
  infiltration, `d`) within ~1% (2-D-triangle vs 1-D-interval discretization). Test:
  `test_2d_flat_reduces_to_1d_column`.
- **Limiter hardening:** track `max_clip_seen` + `clip_mass_adjust`; degenerate `oldtotal≤0` branch
  records the unavoidable mass change (regression test); the lateral test asserts clips stay tiny
  (`max_clip_seen < 0.05`).
- **`ℓ_c`:** auto-detection now FAILS LOUDLY on degenerate z-levels (was a silent tiny `ℓ_c`).

**Residual concerns (carried forward):**
1. **Flat-top host only.** Top-facet detection (`z=ztop`), `ℓ_c` (z-level spacing), and the off-top
   pin all assume a flat, layered top. Non-flat/warped/sloped-geometry tops need a **per-facet local
   `ℓ_c`** and a different top detection — future work; guarded to fail loudly meanwhile.
2. **Post-step limiter ⇒ `λ` staleness.** The accepted `(ψ,d,λ)` after clipping `d` has `λ` solved
   against the un-clipped `d` (Codex). Bounded small (clips are mm/cm; `max_clip_seen` tracks it);
   a rigorous fix (re-solve, or in-solve complementarity for `d≥0`) is deferred.
3. **Surface storage is consistent (non-lumped)** in the coupling vs Module 2's lumped storage. The
   FLUX operator matches M2 to machine precision; only the storage scheme differs. Works with the
   limiter; lumping it (more oscillation suppression at wet/dry fronts) is a possible future change.
4. **MPI untested for the coupling.** MPI overhead dominates on the ~250-DOF verification meshes
   (slower than serial); serial covers correctness. A real MPI check belongs on a larger mesh.
5. **Test-suite speed.** The lateral-routing test is ~51 s (overland stiffness ⇒ ~1500 tiny adaptive
   steps; NOT a mesh-size issue — meshes are 50–200 cells). Use `pytest -k "not lateral"` for fast
   iteration; full suite for milestones.
