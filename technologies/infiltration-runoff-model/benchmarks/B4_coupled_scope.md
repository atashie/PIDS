# B4 — Coupled Surface↔Subsurface Comparison (scope)

**Status:** scoped 2026-06-09, **EXECUTED 2026-06-09** — all 6 scenarios built and documented in
`README.md` §5c (RMS Δθ 0.003–0.019, both mass-conservative, peak ponding agrees ~5–10 mm). This
file is retained as the design scope; the running record + results live in `README.md` §5c. Brought
the ponding / infiltration-excess regime back on ParFlow's home turf (native coupled Richards +
overland). One scope assumption was corrected in execution: ParFlow's mass-consistent `OverlandFlow`
surface store is **`d = max(ψ_top, 0)`**, not the `max(p_top − DZ/2, 0)` extraction §4/§5 inherited
from the standalone-overland case (that offset breaks the closed-column balance by exactly DZ/2 once
ponded). Resolved via the two-phase restart (§5 option b).

## 1. Goal
Benchmark the in-house COUPLED solver (Module 3 §D, `pids_forward.physics.coupling.CoupledProblem`)
against ParFlow's **native coupled mode** (Richards + `OverlandFlow` BC), on the in-house coupling
scenario matrix. Covers the infiltration-excess / ponding regime the subsurface sweep deferred
(§5a/§5b), in the configuration ParFlow is designed and validated for.

## 2. Why this resolves the standalone-overland blocker
The standalone-overland deferral (§5a) and the sweep's FluxConst blow-up for q>Ks (§5b) both stem
from ParFlow having **no surface store under a FluxConst BC**, and not being a standalone overland
solver. ParFlow's **`OverlandFlow` BC *is* the surface store** (ponding depth = surface pressure)
coupled to Richards — the direct analog of the in-house surface-ponding store. On a **flat 1-D
column** (the in-house coupling geometry) OverlandFlow ponds and re-infiltrates with **no lateral
routing**, so the extreme-Manning / lateral issues never arise. B4 is both the right next comparison
and the clean way to exercise ponding.

## 3. The in-house coupled case to match (from `forward-model/viz/run_coupling_sanity.py`)
- Domain: **2 m loam column** (z up, surface at z=2), `ncell=80`, **closed no-flux base** → total
  water (soil ∫θ + surface store d) = cumulative rain (clean coupled mass-balance check).
- Soil: `VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)` (Carsel & Parrish / SE-Piedmont loam).
- Forcing matrix (storm + recession): rain {**normal** 0.30 m/day × 0.30 day; **extreme** 3.0 m/day × 0.05 day}
  × antecedent {**dry** ψ0=−3, **normal** −1, **wet** −0.15} → **6 scenarios**.
- Physics: land-surface exchange partitions rain into infiltration (λ into soil) + ponding (depth d);
  dry/normal + normal rain → infiltrates (d≈0); wet or extreme → ponds (infiltration-excess); recession drains the pond into the soil.
- In-house API: `CoupledProblem(msh, soil)`, `.set_initial_condition(psi0, d_value=0)`, `.add_rain(rate)`,
  `.step(dt)`, `.surface_depth()`, `.exchange_flux()`, `.soil_water()`, `.surface_water()`, `.total_water()`,
  `.psi.x.array`, `.Vpsi`. NetCDF contract: head/θ(t,z), d(t), λ(t), rain, soil_water, surface_water, cum_rain, mbe.

## 4. ParFlow coupled setup (to build — `../parflow/cases/coupled_column.py`)
- 1-D column: NX=NY=1, NZ≈80, DZ=2/80=0.025 over 2 m. Loam vG (match), **Perm=Ks=0.25 (PERMEABLE** — the point; the sweep used Perm=Ks too).
- **Top BC: `OverlandFlow`** (the surface ponding store). Flat (`TopoSlopes≈0` → no lateral routing → pure
  ponding/infiltration partition). Manning ~irrelevant (no lateral flux).
- Base: bottom `FluxConst=0` (closed). Sides no-flow.
- IC: uniform ψ0 per antecedent; dry surface (d=0).
- Surface-depth extraction: **`d = max(top-cell pressure − DZ/2, 0)`** (offset-corrected, §5a — exact for the
  saturated/ponded cell) or `parflow.tools.hydrology`. Infiltration λ ≈ d(soil θ-storage)/dt.

## 5. Forcing / recession — the dominant build risk
The recession (rain off → pond drains into soil) is physically essential here. **The ParFlow storm/recession
`rainrec` time cycle was unreliable in the sweep (rain never switched off — §5b).** Resolve up front by either:
(a) verifying/fixing the `rainrec` cycle for `OverlandFlow`, or (b) **two-phase run**: storm (constant rain,
StopTime=storm_dur) then **restart** with rain=0 from the storm's final pressure (`ICPressure.Type=PFBFile`),
concatenating outputs. (b) is the safe default.

## 6. Comparison quantities + harness
- ψ(z,t), θ(z,t) → **reuse `make_comparison_html.py`** (overlaid profiles + Δ error) directly.
- Surface depth d(t), infiltration λ(t), partition (cumulative infiltrated vs ponded), mass balance →
  extend the `.nc` contract (`surface_depth_*`, `lambda_*`) and add a surface/partition panel (a coupled
  HTML view, or a new `make_comparison_coupled_html.py`).
- New `build_comparison_coupled.py` (mirror `build_comparison_sweep.py`): loop the 6 scenarios → `.nc` + HTML each.
- Output: `data/coupling__{scenario}__<date>.nc`, `html/coupling__{scenario}__<date>.html`.

## 7. Risks / deltas to track
- ParFlow `OverlandFlow` surface-store physics vs the in-house §D exchange-flux λ formulation — the
  infiltration/ponding partition may differ near the threshold (the most interesting comparison).
- Recession handling (cycle bug) — resolve first (§5).
- Surface-depth extraction (offset) + the air-entry / Ss / FV-vs-FEM deltas (as in §5b).
- Extreme-rain stiffness (3 m/day on wet soil → rapid ponding) — watch ParFlow coupled convergence / dt.
- Accuracy-only (perf deferred until a native ParFlow build), per the standing decision.

## 8. Suggested order
1. ParFlow coupled deck, ONE scenario (e.g. `normal_on_normal`) → verify it ponds/infiltrates sensibly and the recession drains; settle the recession approach (§5).
2. In-house `CoupledProblem` run on the same scenario → first coupled comparison `.nc` + HTML.
3. Loop the 6-scenario matrix; add the surface-depth / partition panel.
4. Document in `README.md` (§5c) + change log; this closes the overland/ponding gap deferred in §5a.
