# Convergent dual-drain — coupled convergent-flow Tier-2 fixture (P3 Part B)

**Status:** in progress (P3, 2026-06-19). Builder `forward-model/scratch/_p3_convergent_fixture.py`;
Tier-1 tests `forward-model/tests/test_convergent_dualdrain.py`; Tier-2 storm matrix
`forward-model/scratch/_p3_convergent_storm_matrix.py`; Tier-3 HTML
`forward-model/viz/make_convergent_dualdrain_html.py`.

## Purpose

The permanent Tier-2 regression for the **convergent-flow overland fix** (the P0–P3 workstream): the
galerkin `OverlandProblem` develops a never-settling sawtooth on **convergence lines** (the topographic
low lines where overland flow concentrates); the monotone upwind scheme
(`CoupledProblem(overland_scheme="upwind")`) removes it (sawtooth gone, dt-pin lifted ~370×,
conservation machine-tight — parent plan §8.7/§8.8). Convergent overland flow is a **core PIDS regime**:
PIDS networks/inlets install along convergence lines, where surface water concentrates.

**Framing (Arik 2026-06-19).** A "swale" is **not a PIDS feature** — it is the convergent topographic
**setting** (a graded low line, standard site grading) with *variable* geometry. The PIDS element this
fixture embeds is the coupled-integrated **dual-drain** — the signed-off illustration
`scratch/m4_hillslope_drain_dual.py` (commit 63145e2: an interface tile drain + a surface grate on a
2-D galerkin hillslope) — **extended to this 3-D, upwind, convergent geometry**. (The full bidirectional
`WellIndexExchange` 1-D conveyance-with-storage feature is *not* wired into `CoupledProblem`; the
coupled-integrated `add_interior_drain` is the closest analogue of an embedded conveyance.)

## Configuration (all PARAMETERIZED in the builder; representative defaults below)

- **Host:** 3-D box, loam over tight clay (`TwoLayerSoil`, interface at `z_iface=1.0 m`); loam
  `Ks=0.25`, clay `Ks=0.005 m/day`. Default `48 × 30 × 2 m`, mesh `24 × 15 × 4`.
- **Convergent topography** (a field on the flat top facet, realization A):
  `z_b = SY·(Ly−y) + SX·max(|x−Xc|−W/2, 0)` — a flat-bottomed low line of floor width `W` at `x=Xc`
  running down `y`. Defaults `SX=3%` (sides), `SY=1.5%` (valley), `W=12 m` (spans **6 cells** across
  the floor — the resolution requirement; a coarser floor → the measure-zero kink artifact).
- **Overland:** `overland_scheme="upwind"` (the scheme under test); `n_man=0.05`.
- **Outlet:** the convergence floor at the downstream end (`y=Ly`, `|x−Xc|≤W/2`), `slope=SY` — the low
  line *is* the drainage path (sides drain into it and exit through it).
- **Embedded PIDS dual-drain** along the convergence line (shared footprint `|x−Xc| ≤ halfwidth`):
  - `add_surface_inlet` — grate intake on the ponded depth (`q = C_surf·d`, default `C_surf=500 /day`);
  - `add_interior_drain` — outflow-only tile drain on the clay interface (one loam cell-row,
    `q = C_vol·kr(ψ)·pos(ψ+z−z_iface)`, default `C_vol=300 /(day·m)`, `eps_act=1e-3`).
- **IC:** water table at `z_wt=0.7 m` (in the clay; loam above starts unsaturated → the storm perches a
  saturated mound on the clay that the interface tile taps).

## Storm matrix (Tier-2)

Mirrors `tests/test_coupling_3d_tier2.py`: a typical storm (rate `> Ks` → infiltration-excess runoff +
convergence ponding), a 100-yr-style burst, and a recession; dry / normal / wet antecedent (water-table
elevation). Every scenario must stay stable and mass-conservative.

## Acceptance

- **Tier-1** (in the suite, fast): structural conservation with ALL sinks
  `Δtotal == cum_rain − cum_outflow − cum_drainage (+ clip=0 on upwind)`; positivity within the upwind
  tripwire (sub-cm); the surface inlet AND the interior drain both capture water (`cum_sinks > 0`) and
  rise with ponding; finite ψ, d, θ.
- **Tier-2** (scratch run): mass-conservative + physically plausible across the storm matrix; the
  convergence concentrates overland flow; the inlet/drain division of labor; recession re-infiltrates;
  solver completes (dt/iters/rejections recorded).
- **Tier-3** (HTML, Arik visual sign-off = Gate B): the surface depth map `d(x,y)` (convergence to the
  line), the inlet + interior-drain capture hydrographs, the perched mound on the clay, and conservation.

## Provenance

Extends the signed-off resolved-drain dual-drain (`m4_hillslope_drain_dual.py`, 63145e2 — *resolved*
drains via the first-class `add_interior_drain`/`add_surface_inlet` engine APIs, NOT the retracted M4
sub-grid embedded feature) to the 3-D upwind convergent regime validated absolutely in P3 Part A
(`tests/test_coupling_upwind.py`, commit 4672eb5). See `pids-p3-convergent-progress` memory.
