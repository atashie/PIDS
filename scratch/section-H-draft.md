# H. Cross-Cutting Numerics

This section is the single source of truth for time integration, the nonlinear/linear solve, preconditioning, parallelism, and reproducibility. Every module (B-G) references these contracts rather than re-specifying them. All solver state is owned by the **driver/IO** module (7); physics modules contribute residual/Jacobian UFL forms only.

## H.1 Time Integration

**Scheme.** Implicit backward-Euler (BE) is the default; BDF2 is an opt-in for smooth recharge/recession windows where it cuts dt-error at equal cost. BE is L-stable (mandatory for the stiff Richards operator and for the high-K embedded-feature axial term, whose conductivity contrast spans 4-6 orders of magnitude). BDF2 is started/restarted with one BE step and is automatically demoted to BE on any dt change > 10% or after a Newton failure, to avoid its variable-step ringing.

Mixed-form Richards storage is differenced as `(theta(psi^{n+1}) - theta(psi^n))/dt` (Celia 1990 mass-conservative form) so the conservation Tier-1 test closes to < 1e-6 independent of dt.

**Adaptive dt policy.** Two independent controllers, take the smaller proposal:

1. *Newton-iteration target.* Track iteration count `it`. Target band `it_target = 3..6`. Growth/cut:
   - `it <= 3`: `dt <- min(dt * 1.5, dt_max)`
   - `4 <= it <= 6`: hold
   - `it > 6` or no convergence: `dt <- dt * 0.5`, retry the step (do not advance).
2. *Local truncation error (LTE).* Estimate via BE-vs-BDF2 difference (or step-doubling when BDF2 off): `e = ||y_BDF2 - y_BE|| / scale`. PI controller `dt_new = dt * (tol_lte / e)^{0.5}` clamped to `[0.5, 2.0]` per step; `tol_lte ~ 1e-3` on saturation-equivalent units.

Bounds: `dt_min = 1 s` (a 100-yr Atlas-14 sub-hourly burst at 5-min resolution forces dt into the tens-of-seconds range; hitting `dt_min` repeatedly is a hard error, not a silent clip), `dt_max = 86400 s` (1 day, recession/drought). Forcing-aware cap: dt is additionally capped so it never steps over a hyetograph breakpoint (the forcing module (6) exposes the next breakpoint time); this prevents averaging-out of a sub-hourly intensity spike.

```
solve_step(dt):  attempt BE Newton
  if not converged: dt *= 0.5; if dt < dt_min: raise; else retry
  else: it -> dt controller; LTE -> dt controller; dt_next = min(both, breakpoint_cap, dt_max)
```

## H.2 Nonlinear Solve (the headline: smooth monolithic Newton)

Per the locked corrected physics, the embedded-feature **per-face exchange** `q = sigma*(h_feat - h_soil)` with fixed/bidirectional `sigma`, the **axial Darcy** term, and the **land-surface exchange** are all **smooth** potential-driven fluxes. Therefore the headline coupled solve is a **single smooth monolithic PETSc SNES Newton + line search** over the combined residual `F(U) = 0`, `U = [psi_soil, h_surf, h_feat]` assembled by the coupling module (3) into one residual + one Jacobian per timestep.

- **SNES type:** `newtonls` with `bt` (back-tracking) line search; `cp` (critical-point) line search as a fallback for the wetting-front stiffness.
- **Jacobian:** UFL automatic differentiation, `J = ufl.derivative(F, U)`. No hand-coded or finite-difference Jacobian on the headline path (the mixed-dimensional off-diagonal coupling blocks must be exact for Newton quadratic convergence).
- **Tolerances:** `snes_rtol=1e-8`, `snes_atol=1e-10`, `snes_stol=1e-8`, `snes_max_it=25`. Residual scaled by storage-capacity to keep atol meaningful across wet/dry.

```python
snes = problem.solver  # dolfinx.fem.petsc.NonlinearProblem -> PETSc SNES
opts = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_rtol": 1e-8, "snes_atol": 1e-10, "snes_stol": 1e-8, "snes_max_it": 25,
    "snes_monitor": None, "snes_converged_reason": None,
}
# DOLFINx 0.10: NonlinearProblem / LinearProblem REQUIRE keyword-only petsc_options_prefix
problem = NonlinearProblem(F, U, bcs=bcs, J=J,
                          petsc_options_prefix="pids_richards_",
                          petsc_options=opts)
```

**Divergence fallbacks, in order:** (1) dt-cut-and-retry (H.1); (2) **pseudo-transient continuation** (`snes_type ptc`, or a relaxed BE with an inflated storage term ramped down) for bone-dry antecedent / 100-yr-storm cold starts where the front is near-discontinuous; (3) homotopy on `sigma` (ramp exchange conductance from 0) only if a feature face is the culprit. Each fallback is logged with the converged-reason code for the Tier-1 determinism record.

**VI scope (reserved, NOT the core).** The variational-inequality solver `snes_type vinewtonrsls` (spike-confirmed, PETSc 3.25.1) is used in exactly two cases, set by `setVariableBounds(lb, ub)`:
1. **Overland depth positivity** `h_surf >= bed` (ponding depth >= 0) — the dominant VI use.
2. **Optional hard one-way devices** — the degenerate asymmetric-`sigma` limit (receive>0, disperse=0). Insurance per the `catchvalve_probe` (implicit hook tractable; operator-splitting disqualified). The default bidirectional feature uses **smooth** `newtonls`, NOT VI.

When VI is active the same UFL `F`, `J` are reused; only bounds vectors are added. Bounds default to `[-INFINITY, +INFINITY]` (i.e., plain Newton) so a single code path serves both.

## H.3 Linear Solve + Preconditioning

Each Newton iteration solves `J du = -F`. `J` is a 3-field block system (subsurface / surface / feature).

- **Small / 1-D / MMS / Tier-1:** `ksp_type preonly`, `pc_type lu` (`pc_factor_mat_solver_type mumps`) — exact, reproducible, the spike's Poisson path. This is the determinism reference.
- **Production 2-D/3-D:** `ksp_type gmres` (restart 100, `ksp_pc_side right` so the residual norm is preconditioner-independent for the tolerance test), with **PCFIELDSPLIT**.

**Fieldsplit + conditioning.** The high-K embedded features create a large conductivity contrast and the mixed-dimensional 1D-in-3D coupling injects off-diagonal blocks into rows that are otherwise the sparse Richards stencil — a classic saddle-point-flavored, badly-scaled system. Strategy: a `multiplicative` (or `schur` for the surface-subsurface pair) fieldsplit isolating the well-behaved subsurface block for AMG, and keeping the small dense-ish feature block on a direct factor.

```python
lin = {
    "ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_gmres_restart": 100, "ksp_pc_side": "right",
    "pc_type": "fieldsplit", "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_0_fields": "0",     # subsurface (Richards)
    "pc_fieldsplit_1_fields": "1",     # overland
    "pc_fieldsplit_2_fields": "2",     # embedded features
    # subsurface block: AMG
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "hypre", "fieldsplit_0_pc_hypre_type": "boomeramg",
    "fieldsplit_0_pc_hypre_boomeramg_strong_threshold": 0.7,  # 3-D anisotropy
    # surface block: small, AMG or ILU
    "fieldsplit_1_pc_type": "hypre", "fieldsplit_1_pc_hypre_type": "boomeramg",
    # feature block: tiny, direct
    "fieldsplit_2_ksp_type": "preonly", "fieldsplit_2_pc_type": "lu",
}
```

Conditioning notes: (a) **non-dimensionalize / row-scale** the feature axial conductance against the soil block so AMG strong-threshold heuristics are not fooled by the K-contrast; (b) if BoomerAMG stalls on the strongly anisotropic 3-D Richards operator near saturation, fall back to `gamg` or add `pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi`; (c) the off-diagonal feature coupling is kept in the *true* Jacobian but may be **dropped from the preconditioner** (`fieldsplit multiplicative` already approximates this) — Newton still uses the exact `J`, so convergence order is preserved while the preconditioner stays cheap. Track `ksp_converged_reason` and iteration counts in the sanity report; a linear-iteration blow-up is the first symptom of the K-contrast conditioning risk.

## H.4 Parallelism Hedge

MPI is the only parallelism wired now and is left **un-optimized**: meshes are built distributed (`mesh.create_*(MPI.COMM_WORLD, ...)`), all assembly/solve already run rank-parallel under PETSc domain decomposition, and reductions use `comm.allreduce`. No load-balancing of the embedded 1-D features across ranks is attempted (a feature vector may straddle partitions; the coupling module must `scatter`/`gather` its DOFs — flagged, not solved). GPU is **deferred** (PETSc `vec/mat_type cuda` + `pc_type gamg` is the eventual switch, not built). Optimization-loop scale is TBD, so MPI correctness (mass balance identical 1 vs N ranks) is asserted in Tier-1, but MPI *performance* is out of scope.

## H.5 Reproducibility

Fixed inputs + fixed seeds -> bitwise-identical results on a fixed rank count, asserted in Tier-1: (1) all RNG (synthetic property fields, MMS forcing) seeded from a single config seed; (2) the LU/`preonly` path is the determinism reference (iterative-solver tolerances make GMRES results reproducible only to `ksp_rtol`, so determinism tests pin `pc_type lu`); (3) PETSc options captured to the sanity report via `-options_left` / `PETSc.Options().view()`; (4) `petsc_options_prefix` is mandatory (DOLFINx 0.10) and **unique per solver instance** so concurrent solvers in one process never collide in the global options DB; (5) MPI determinism asserted at fixed rank count only (reduction order varies with rank count by design).
