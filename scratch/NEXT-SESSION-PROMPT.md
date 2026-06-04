# Next-session continuation prompt (PIDS Pillar 2 forward-model build)

Paste the block below to resume. Full context auto-loads from memory (`MEMORY.md` → `pids-forward-model-engine.md`) and the dossier `docs/plans/2026-06-03-pids-forward-model-engine-evaluation.md`.

---

```
Resume the PIDS Pillar 2 forward-model build. Decision (recorded): Option B — build a modular
DOLFINx/FEniCSx (PETSc) FEM forward model. Standards already written:
governance/claude-sanity-check-routine.md and governance/visualize-sanity-check-routine.md.
The reverse catch-valve is de-risked (scratch/catchvalve_probe.py); the build runs on WSL2 + conda.

WSL2 was installed last session and the machine has now been rebooted. Pick up here:

1. Verify the environment: run `wsl -l -v` (expect Ubuntu, VERSION 2). If Ubuntu needs first-run
   user setup, walk me through it — I can run interactive commands myself with `!`.

2. Drive the staged DOLFINx install spike in scratch/spike/ (repo is at
   /mnt/c/Users/arikt/Documents/GitHub/PIDS inside WSL):
     - install Miniforge in WSL,
     - `conda env create -f environment.yml`  (creates `pids-fem`),
     - `conda activate pids-fem && python smoke_test.py`.
   The spike checks DOLFINx+PETSc (Poisson) and the PETSc variational-inequality solver
   (vinewtonrsls — the catch-valve's exact path). Expect possible minor DOLFINx/petsc4py API
   version tweaks in smoke_test.py; iterate until it passes or a real blocker emerges.

3. On spike PASS: write the forward-model ARCHITECTURE DESIGN (module DAG, interfaces, and the
   per-module sanity-check + visualization plan) to docs/plans/ and pause for my review BEFORE
   building. Then build module 1 — the subsurface Richards solver, seeded from
   scratch/catchvalve_probe.py — test-first under governance/claude-sanity-check-routine.md,
   ending with the interactive-HTML visual gate.

Work cautiously and modularly: confirm before large changes, build bottom-up, and sanity-check
every module independently AND in concert under typical + 100-yr-extreme synthetic forcing per the
routine. Design storms default to NOAA Atlas 14 (SE Piedmont) unless I say otherwise.
Ask clarifying questions as needed.
```

---

## State snapshot (2026-06-04)
- **Decision:** Option B (DOLFINx/FEniCSx FEM build). Recorded in `governance/decision-log.md`, `technologies/infiltration-runoff-model/DECISION-model-selection.md`; revises the 2026-06-01 "extend, don't build" framing.
- **Standards written:** `governance/claude-sanity-check-routine.md` (three-tier), `governance/visualize-sanity-check-routine.md` (offline interactive HTML).
- **De-risk artifact:** `scratch/catchvalve_probe.py` (+ `.png`) — reverse catch-valve tractable as an implicit hook; operator-split blows up.
- **Spike staged:** `scratch/spike/` (`environment.yml`, `smoke_test.py`, `README.md`).
- **Build order:** subsurface Richards → overland → coupling → PIDS internal-BC layer → domain/forcing/vegetation.
- **Blocker just cleared:** WSL2 installed (`wsl --install -d Ubuntu`), **reboot pending** → that's why this session is ending.
