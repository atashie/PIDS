# ParFlow — Usage & Learning Log

Worked examples, run idioms, and gotchas, built in the in-house model's bottom-up
module order. All runs use the Docker-first install ([`INSTALL.md`](INSTALL.md)).

## Conventions
- **Run on the native WSL fs** (`$HOME/parflow-runs/...`), not `/mnt/c` (9p is slow for ParFlow I/O). Canonical decks live here in `cases/`; copy to scratch to run.
- **Mount idiom:** use an **explicit absolute path** for `-v`, e.g. `-v $HOME/parflow-runs/<case>:/data`. In this PowerShell→wsl→docker setup the `-v $(pwd):/data` form (from the official docs) failed to resolve; the explicit path is reliable.
- **Python deck via `pfrun`:** `docker run --rm -v <dir>:/data parflow/parflow:latest <deck>.py 1 1 1`. The deck runs as plain Python (it calls `run.run()` itself) and may post-process in the same script.
- A `: Field _run_file is not part of the expected schema` line from `Run(name, __file__)` is a **benign** pftools notice, not an error.

## B1 — Python / pftools idiom (validated 2026-06-08)
Running a `.py` deck through `pfrun` exercised the full Python path: `from parflow import Run`, attribute-style key setting, `run.run()`, and reading `.pfb` outputs back to numpy via `from parflow.tools.io import read_pfb`. Confirmed working in-container (pftools + numpy present).

## B2 — Clean 1-D infiltration column → anchors the SUBSURFACE module
**Deck:** [`cases/column_1d.py`](cases/column_1d.py). Mirrors the in-house `typical-mesic`
constitutive setup: a uniform Carsel & Parrish (1988) loam column, constant rain flux
below Ks (pure flux-infiltration, never ponds).

**Run (verified 2026-06-08):**
```bash
mkdir -p $HOME/parflow-runs/column_1d
cp <repo>/technologies/infiltration-runoff-model/parflow/cases/column_1d.py $HOME/parflow-runs/column_1d/
sudo docker run --rm -v $HOME/parflow-runs/column_1d:/data parflow/parflow:latest column_1d.py 1 1 1
```

**In-house ↔ ParFlow parameter mapping** (the load-bearing conversion):

| in-house (`richards.py`) | value | ParFlow key | value |
|---|---|---|---|
| domain | [0,1] m, 60 cells, z up | `NX=NY=1, NZ=60, DZ=1/60` | cell-centred, k=0 = bottom |
| `theta_s` (porosity) | 0.43 | `Geom.domain.Porosity.Value` | 0.43 |
| `theta_r` | 0.078 | `Saturation.SRes = theta_r/theta_s` | 0.18140 |
| `alpha` | 3.6 /m | `Saturation/RelPerm.Alpha` | 3.6 |
| `n` | 1.56 | `Saturation/RelPerm.N` | 1.56 |
| `Ks` | 0.2496 m/day | `Perm.Value` (with ρ=μ=g=1) | 0.2496 |
| IC `psi0` | −1.0 m | `ICPressure.Type=Constant, Value` | −1.0 |
| top rain `q` | 0.10 m/day | `Patch.top FluxConst Value` | **−0.10** (neg = into domain) |
| bottom | no-flow | `Patch.bottom FluxConst Value` | 0.0 |

**Result (physical sanity — all PASS):**
- IC saturation: ParFlow mean **0.5631** = analytic van Genuchten S(−1 m) **0.5631** → the vG conversion is correct.
- Top saturation rises 0.5631 → 0.9819 under rain; never saturates (no ponding, q<Ks).
- Wetting/pressure signal propagates from the top downward through the column.
- **Mass balance:** cumulative infiltration 0.100000 m. The **full ParFlow storage** (incl. the compressible term S·Ss·ψ) is conserved to **2.1e-9** — the deck is mass-conservative to solver tolerance. The **water-content (θ-equivalent) balance** — the quantity comparable to the in-house model, which carries no compressible storage — closes to **2.9e-6**; that residual *is* the compressible term the θ-balance omits (≈1e-7 m). Flux BC injects exactly q·t; the no-flow bottom retains it.

**Formulation deltas in play (see [`README.md`](README.md) §4), reconfirmed here:**
- ParFlow van Genuchten **omits the Vogel/Ippisch air-entry cap** the in-house model applies (`h_s=-0.02 m`). The cap changes θ(ψ), **K(ψ)**, *and* the near-saturation tangent (applied in `effective_saturation`, `K`, and `capacity`) → it shifts infiltration-front speed and matters for any near-saturation comparison, not just θ. Small here (column stays unsaturated; top only reaches S≈0.98).
- ParFlow needs a `SpecificStorage` (set 1e-6); the in-house model ignores Ss (unconfined). **Tiny but not inactive** — ParFlow storage carries S·Ss·ψ alongside φ·S; here ≈1e-7 m, which is exactly the θ-balance residual above.
- ParFlow is **cell-centred FV** (60 centres); in-house is **P1 FEM** (61 nodes) → profile comparison needs interpolation.

**Next:** quantitative profile cross-check of this column against the in-house
`subsurface__typical-mesic` result (head/θ profiles at matched times), then B3 (overland).
