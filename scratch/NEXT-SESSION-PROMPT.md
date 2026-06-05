# Next-session continuation prompt (PIDS Pillar-2 forward model — Module 2)

Paste the block below to resume in a clean context. Full state auto-loads from memory
(`MEMORY.md` → `pids-forward-model-engine.md`, `pids-forward-model-assumptions.md`,
`pids-feature-physics.md`) and the design dossier
`docs/plans/2026-06-04-pids-forward-model-architecture-design.md`.

---

```
Resume the PIDS Pillar-2 forward-model build. Module 1 (subsurface Richards) is COMPLETE and
Tier-3 signed off (2026-06-04) — see validation/sanity/subsurface__2026-06-04.md and
technologies/infiltration-runoff-model/forward-model/. Build Module 2 next: OVERLAND FLOW
(design §C of docs/plans/2026-06-04-pids-forward-model-architecture-design.md).

Work the same way Module 1 was built:
- Strictly TDD (tests precede code) under the three-tier sanity routine
  (governance/claude-sanity-check-routine.md + visualize-sanity-check-routine.md): Tier-1
  analytical/conservation/plausibility pytest → Tier-2 synthetic typical + 100-yr-extreme
  forcing → Tier-3 interactive offline HTML (separate viz step) + my human sign-off.
- Dimension-agnostic UFL; commit + push each green increment to main (github.com/atashie/PIDS).
- Per design §C: diffusion-wave default (justify vs local-inertial / full shallow-water, YAGNI);
  surface water depth d>=0 enforced via the PETSc VI path; expose the velocity/direction field
  the erosion threshold needs; well-balanced / stabilized wet-dry front. Tier-1 menu: kinematic-
  wave plane hydrograph, Stoker dam-break, tilted-V catchment, MMS order.
- Overland couples to the subsurface in Module 3; Module 1's add_ponding_bc is a vertical-only
  ponding store (precursor) — overland adds LATERAL routing/runoff.

Env / how to run (WSL2 + conda 'pids-fem'; repo at /mnt/c/Users/arikt/Documents/GitHub/PIDS):
  wsl -d Ubuntu -u root -- bash -lc "source /root/miniforge3/etc/profile.d/conda.sh; conda activate pids-fem; cd /mnt/c/Users/arikt/Documents/GitHub/PIDS/technologies/infiltration-runoff-model/forward-model; python -m pytest -q"
Scripts (non-pytest) need PYTHONPATH=. . SHELL GOTCHA: in the PowerShell->wsl->bash chain a
$var gets eaten — use explicit paths, not bash variables, and prefer per-step commands over for-loops.

Locked modeling assumptions still hold (see pids-forward-model-assumptions): unconfined aquifer,
no Ss/compressibility, no solute transport. Confirm scope/clarify as needed before large changes.
```

---

## State snapshot (2026-06-04)
- **Module 1 (subsurface) DONE + signed off.** `pids_forward/physics/{constitutive.py, richards.py}`:
  mixed-form Richards (Celia 1990), van Genuchten–Mualem + **Vogel/Ippisch air-entry**, **mass-lumped**
  storage, **adaptive backward-Euler** (`advance()`, cut-and-retry), PETSc SNES Newton, dimension-agnostic
  (gravity on the last coord). BCs: `add_dirichlet`, `add_flux_bc` (rain/evap), `add_ponding_bc`
  (surface store, vertical-only) + `ponded_depth`. **20 tests green** (Tier-1 + Tier-2).
- **Sanity artifacts:** report `validation/sanity/subsurface__2026-06-04.md`; viz scripts
  `forward-model/viz/{run_subsurface_sanity.py, make_sanity_html.py}` (4 scenarios, storm+recession,
  soil-layer + field-capacity/saturation overlays). Heavy HTML + NetCDF are gitignored (regenerable).
- **Residual concerns (Module 1, logged in the report):** supply-limited evaporation needs an
  atmospheric/Robin BC (forcing module); a free-drainage / deep-water-table lower BC is a small future
  add (the default no-flux base gives bottom-up water-table mounding); HGS cross-code deferred; MPI>1 untested.
- **Build order:** subsurface ✅ → **overland (next)** → coupling → pids-features → domain → forcing/veg → driver/IO.
