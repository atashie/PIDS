# Next-session continuation prompt (PIDS Pillar-2 forward model — Module 3)

Paste the block below to resume in a clean context. Full state auto-loads from memory
(`MEMORY.md` → `pids-forward-model-engine.md`, `pids-forward-model-assumptions.md`,
`pids-feature-physics.md`) and the design dossier
`docs/plans/2026-06-04-pids-forward-model-architecture-design.md` (read the **§D header
NOTE 2026-06-05** and the **§C UPDATE 2026-06-05** first — they amend the spec).

---

```
Resume the PIDS Pillar-2 forward-model build. Module 1 (subsurface Richards) and Module 2
(overland diffusion-wave) are BOTH COMPLETE + Tier-3 signed off (M1 2026-06-04, M2
2026-06-05); full suite 39/39. Build Module 3 next: MONOLITHIC COUPLING (design §D) —
two-way implicit coupling of subsurface Richards (§B, pids_forward/physics/richards.py
RichardsProblem) + overland (§C, pids_forward/physics/overland.py OverlandProblem) via the
LAND-SURFACE exchange, assembled into ONE blocked residual [ψ, d] + one Newton (PETSc SNES)
solve per backward-Euler step. (Embedded-feature coupling is §E/Module 4, later.)

Work the same way M1/M2 were built:
- Strictly TDD (tests precede code) under the three-tier sanity routine
  (governance/claude-sanity-check-routine.md + visualize-sanity-check-routine.md): Tier-1
  analytical/conservation/plausibility pytest → Tier-2 typical + 100-yr-extreme forcing →
  Tier-3 interactive offline HTML (SEPARATE viz subagent) + Arik's human sign-off.
- Dimension-agnostic UFL; commit each green increment (Arik pushes via `! git push`, or add a
  Bash(git push:*) permission rule so Claude can push).
- Run a 4-lens adversarial review (it caught a real bug in M2) before declaring done.

Module 3 essentials (design §D, with the 2026-06-05 amendments):
- §D is the ASSEMBLER: composes R_sub(ψ) + R_ovl(d) + the coupling flux into one blocked
  system; no new physics state beyond the composition.
- Land-surface exchange = DEFAULT (B) first-order Robin (§D.3): q_ls = k_ex·(d − ψ_top),
  k_ex = K_rel(ψ_top)·K_s/ℓ_c, ℓ_c ≈ top-cell half-height. GRAVITY-CONSISTENT: z_surf is in
  BOTH hydraulic heads and cancels — do NOT add z_surf to the driving potential. Sign-paired
  into both blocks ⇒ conservation is structural. Infiltration-excess (Hortonian) AND
  saturation-excess runoff emerge naturally; k_ex→∞ reproduces common-node ψ_top=d (Tier-1
  assertion to machine precision). Retain common-node (A) as a verification reference.
- POSITIVITY (M2 pivot, important): overland d≥0 is NO LONGER a VI bound — it is smooth
  newtonls + a post-step CONSERVATIVE limiter. So the coupled solve is PLAIN SNES Newton (no
  VI block for d≥0). DECIDE where the limiter runs in the coupled loop (likely a post-step
  correction on the d DOFs, before acceptance, conserving each block's budget).
- KEY BUILD RISK (do this first): M2 was built standalone on its OWN surface mesh; coupling
  must realize d on the HOST TOP FACETS (codim-1) or a shared co-dim-1 submesh, and assemble
  the ψ↔d exchange + Jacobian across them. This is the first real exercise of the R1 cross-mesh
  `ufl.derivative` assumption (design §A.2/§F). Spike it: can DOLFINx 0.10 assemble a blocked
  [ψ(volume), d(top-facets)] residual + exact coupled Jacobian? If not, fall back to a
  host-mesh co-dim line/facet restriction. Freeze the coupling contract only after the spike.
- Tier-1 menu: (a) MMS on coupled (ψ,d) with manufactured q_ls → spatial+temporal order;
  (b) tilted-V + "superslab" coupled benchmarks (HGS/ParFlow cross-code OPTIONAL); (c) GLOBAL
  mass balance across the interface |Δstore_sub+Δstore_ovl − net_flux − src|/scale<1e-6 +
  per-interface closure + flux antisymmetry q_surf→sub = −q_sub→surf; (d) plausibility
  0≤S≤1, d≥0, bounded heads, no NaN; (e) k_ex→∞ ⇒ ψ_top=d to machine precision.
- Tier-2: infiltration-excess vs saturation-excess PARTITIONING under typical + 100-yr Atlas-14
  storm (sub-hourly → adaptive stepping); water-table mounding/recharge; recession limb after
  rain; run alone → pairwise (overland+subsurface) → full system, recording couplings exercised.
- M2 accessors to consume: OverlandProblem.add_rain / add_dirichlet / add_outflow_bc(loc,slope>0)
  / outflow_rate() / velocity()[SI m/s] / bed_shear()[Pa] / advance() / last_outflow /
  clip_mass_adjust. RichardsProblem exposes ψ, θ, K(ψ), and a Q_exchange volumetric slot (§B.7).

Env / how to run (WSL2 + conda 'pids-fem'; repo at /mnt/c/Users/arikt/Documents/GitHub/PIDS):
  wsl -d Ubuntu -u root -- bash -lc "source /root/miniforge3/etc/profile.d/conda.sh; conda activate pids-fem; cd /mnt/c/Users/arikt/Documents/GitHub/PIDS/technologies/infiltration-runoff-model/forward-model; python -m pytest -q"
Scripts (non-pytest) need PYTHONPATH=. . SHELL GOTCHA: in the PowerShell->wsl->bash chain a
$var gets eaten — use explicit paths, not bash variables, and prefer per-step commands over for-loops.
MPI: mpirun needs --allow-run-as-root in this env.

Locked modeling assumptions still hold (pids-forward-model-assumptions): UNCONFINED aquifer,
no Ss/compressibility, Vogel air-entry, permanent skeleton, no solutes. Confirm scope before
large/locked-decision changes (the M2 VI→limiter pivot was such a case — flagged + approved).
```

---

## State snapshot (2026-06-05)
- **Module 1 (subsurface) DONE + signed off 2026-06-04.** `pids_forward/physics/{constitutive.py, richards.py}` (mixed-form Celia, Vogel air-entry, mass-lumped, adaptive BE, Dirichlet/flux/ponding BCs; unconfined/no-Ss). 20 tests.
- **Module 2 (overland) DONE + signed off 2026-06-05.** `pids_forward/physics/overland.py` `OverlandProblem`: dimension-agnostic diffusion-wave, Manning conveyance (SECONDS_PER_DAY day-units; slope floor inside the root), adaptive `advance()`, **newtonls + conservative positivity limiter** (the locked SNES-VI failed the stiff problem — adversarially proven, Arik-approved pivot; decision-log 2026-06-05). 19 overland tests; full suite **39/39**. BCs/diagnostics: `add_rain`, `add_dirichlet`, `add_outflow_bc`+`outflow_rate`, `velocity`, `bed_shear`, `last_outflow`, `clip_mass_adjust`. Sanity report `validation/sanity/overland__2026-06-05.md` (incl. synoptic kinematic analysis: q=u·d ⇒ q∝u^{5/2}; outflow set by continuity = r·L, velocity by Manning/depth). Tier-3 HTML + 8-scenario assessment matrix (`viz/run_overland_sanity.py`, `build_all_overland_html.py`, `make_overland_html.py`); heavy HTML/NetCDF gitignored.
- **Residual concerns carried into M3** (from the overland report): limiter is a global (non-local) conservative rescale; Dirichlet-depth + limiter and all-dry+lumped=False+huge-dt are latent edge cases; outflow BC uses bed slope as friction slope (normal-depth surrogate); local-inertial flag not built (deferred-until-triggered). Design §C.4/§C.8/§H.2 SNES-VI text superseded by the §C UPDATE 2026-06-05 note.
- **Build order:** subsurface ✅ → overland ✅ → **coupling (NEXT, §D)** → pids-features (§E) → domain (§F) → forcing/veg (§G) → driver/IO (§I).
