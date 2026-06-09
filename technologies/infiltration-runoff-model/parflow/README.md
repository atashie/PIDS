# ParFlow — Off-the-Shelf Benchmark for the In-House Forward Model (Pillar 2)

> **Siloed track.** Everything about downloading, learning, and documenting ParFlow
> lives in this folder by request — nothing here leaks into `docs/plans/`, the
> in-house model's folders, or the global memory index. Running notes for this
> track go in [`NOTES.md`](NOTES.md), not the global memory.

## 1. Purpose
ParFlow is the open-source, community-standard integrated 3-D variably-saturated +
overland flow model we stand up **alongside** the in-house [`../forward-model/`](../forward-model/)
to independently cross-check its **accuracy and performance** on PIDS-free *bulk
hydrology* — Richards (subsurface), overland flow, and their coupling.

It is a **benchmark only**, not a candidate production engine. The production-engine
decision is settled: build on DOLFINx/FEniCSx (see [`../DECISION-model-selection.md`](../DECISION-model-selection.md)).
PIDS features (channels / tunnels / barriers) are **out of scope** here — representing
them is exactly what off-the-shelf engines can't do well, and the reason we build custom.

## 2. Status
- **Maturity:** in-setup (started 2026-06-08). **Phase A (install) complete; Phase B started — B1 (Python idiom) + B2 (1-D column) verified 2026-06-08.**
- **Decisions (2026-06-08):**
  - Engine = **ParFlow** (open-source, LGPL-2.1; current release 3.15.0, image `parflow/parflow:latest`).
  - Install = **Docker-first** in WSL2 Ubuntu (native build deferred to the performance-comparison stage).
  - Scope = **bulk hydrology only**.
  - This folder is the **single home** for all ParFlow docs + running notes.
- **Environment (verified 2026-06-08):** WSL2 Ubuntu 26.04 LTS, x86_64, systemd as PID 1,
  passwordless sudo, repo mounted at `/mnt/c/Users/arikt/Documents/GitHub/PIDS`.
  Docker Engine (`docker.io` 29.1.3) installed + smoke-tested 2026-06-08 (see [`INSTALL.md`](INSTALL.md)).

## 3. Why ParFlow, and why "benchmark ≠ production engine"
The factors that *disqualified* off-the-shelf engines as the **production** engine —
uniform grid / no local refinement, no engineered internal BCs (the reverse catch-valve),
closed-source — **do not apply to a benchmark**, whose only job is to independently
reproduce bulk physics on PIDS-free cases. On that role ParFlow is the natural pick:

- Purpose-built coupled **3-D mixed-form Richards ↔ 2-D overland** (Kollet & Maxwell 2006), globally implicit.
- Open, free, **Python-scriptable** (PFTools) → automatable comparison harness.
- **MPI / HPC-capable** → honest performance comparison.
- A **canonical published benchmark suite** (tilted-V catchment; Maxwell et al. 2014 / IH-MIP2) → inherited accuracy anchors.
- **Independent discretization** (cell-centered finite-difference/volume, structured grid) vs. our unstructured **FEM** → agreement is near-orthogonal evidence the physics is right.

## 4. Scientific basis & formulation deltas to track when comparing
| Aspect | ParFlow | In-house forward-model | Note when comparing |
|---|---|---|---|
| Subsurface | mixed-form Richards, van Genuchten–Mualem | same | verify vG parameter conventions match |
| Overland | **kinematic-wave** free-surface BC | **diffusion-wave** | expect small differences on mild slopes; larger where backwater matters |
| Discretization | cell-centered FD/FV, structured / terrain-following grid | unstructured FEM | "same resolution" is approximate — document cell size vs element size |
| Solver | globally implicit Newton–Krylov (KINSOL/PETSc) | PETSc SNES Newton | both fully implicit |
| Air-entry & K | standard van Genuchten–Mualem (no air-entry cap) | Vogel/Ippisch cap (h_s=−0.02 m) in θ, **K**, and capacity | differs in θ(ψ), K(ψ), and the near-saturation tangent → shifts front speed; matters near saturation |
| Storage | needs SpecificStorage (compressible S·Ss·ψ); set ~1e-6 | ignores Ss (unconfined) | tiny but nonzero; θ-balance omits it (≈1e-7 m — the 2.9e-6 column residual) |

## 5. Plan
- [x] **A — Install (Docker-first).** ✓ **verified 2026-06-08** — Docker Engine (`docker.io` 29.1.3) → pulled `parflow/parflow:latest` → smoke-tested `example_single.tcl` (`Problem solved`, 0.12 s) → recipe in [`INSTALL.md`](INSTALL.md).
- [ ] **B — Learn (in our bottom-up module order)** — *in progress.* ✓ **B1** (Python/pftools idiom) + ✓ **B2** (clean 1-D infiltration column → anchors *subsurface*; vG conversion verified vs analytic, mass balance 2.9e-6) done 2026-06-08 ([`cases/column_1d.py`](cases/column_1d.py), [`USAGE.md`](USAGE.md)). Remaining: (3) tilted-V overland → *overland*; (4) coupled surface–subsurface → *coupling*.
- [ ] **C — Document.** This README + `INSTALL.md` + `USAGE.md` + `cases/`; ParFlow runs emit the **same standardized `.nc` + HTML** as the in-house model (per [`../../../governance/visualize-sanity-check-routine.md`](../../../governance/visualize-sanity-check-routine.md)) so comparison is direct.
- [ ] **D — Side-by-side harness (follow-on).** One shared case definition → both models → side-by-side HTML (state, fluxes, mass-balance, runtime). Fairness controls: same hardware, pinned BLAS/OMP threads (project FEM-threading note), matched resolution, accuracy runs separated from performance runs.

## 6. Folder layout (files added as each step produces real content)
| Path | Contents |
|---|---|
| `README.md` | this file — purpose, plan, status, formulation deltas |
| `INSTALL.md` | reproducible Docker recipe (+ native-build notes) |
| `USAGE.md` | worked examples / learning log / gotchas |
| `NOTES.md` | siloed running memory for this track |
| `cases/` | per-case ParFlow run scripts + shared case definitions |

## 7. License
ParFlow is **GNU LGPL v2.1** (to be confirmed from the container's `LICENSE`). We only
*run* it as an external benchmark (no redistribution of a derivative), so copyleft
obligations do not attach to the in-house model.

## 8. References
[parflow.org](https://parflow.org) · [github.com/parflow/parflow](https://github.com/parflow/parflow) ·
[parflow/docker](https://github.com/parflow/docker) · [manual v3.15.0](https://parflow.readthedocs.io/en/latest/) ·
Kollet & Maxwell 2006, *Adv. Water Resour.* 29(7):945–958 · Maxwell et al. 2014 (IH-MIP) ·
[coupled ParFlow, GMD 2020](https://gmd.copernicus.org/articles/13/1373/2020/)

## 9. Change log
| Date | Change |
|---|---|
| 2026-06-08 | Folder created; ParFlow chosen as off-the-shelf benchmark (Docker-first, bulk-hydrology scope, siloed here). WSL2/Ubuntu 26.04 environment verified. Docker install pending confirmation. |
| 2026-06-08 | **Phase A complete.** Docker Engine (`docker.io` 29.1.3) installed in WSL; `parflow/parflow:latest` pulled (digest `sha256:ac194a1…`); smoke test `example_single.tcl` passed (`Problem solved`, no errors, 0.12 s). Recipe documented in [`INSTALL.md`](INSTALL.md). |
| 2026-06-08 | **Phase B started.** B1 Python/pftools idiom validated; B2 clean 1-D loam infiltration column ([`cases/column_1d.py`](cases/column_1d.py)) runs on ParFlow 3.15.0 and passes physical sanity — IC saturation matches analytic vG to 4 d.p., mass-balance error 2.9e-6. Worked up in [`USAGE.md`](USAGE.md). Pending Codex review of the track. |
| 2026-06-08 | **Codex review** (read-only) of the track: no correctness bug in the deck (grid/BCs/cycle/solver/vG mapping/flux-sign all validated). Fixed 3 doc-accuracy items: (1) readout now computes the **full** storage balance — ParFlow conserves to **2.1e-9**; the 2.9e-6 θ-residual is the compressible S·Ss·ψ term the in-house model lacks; (2) air-entry delta broadened (θ, K, near-sat tangent, front speed); (3) docstring run-cmd aligned to the explicit mount path. |
