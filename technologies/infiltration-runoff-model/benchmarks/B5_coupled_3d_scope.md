# B5 — 3-D Coupled Hillslope Comparison (scope)

**Status:** scoped 2026-06-09, **not yet executed**. Next benchmark stage after B4. Extends the
coupled comparison from B4's flat 1-D ponding column to a full **3-D tilted hillslope** with
**lateral overland routing + a groundwater seepage outlet** — ParFlow's core design domain (coupled
3-D watershed Richards + OverlandFlow on real slopes). Builds directly on B4 (the OverlandFlow
surface-store convention + the two-phase recession restart carry over).

## 1. Goal
Benchmark the in-house **3-D** `CoupledProblem` (Module 3 §D, the 3-D hillslope) against ParFlow's
**native 3-D coupled mode** on the in-house 3-D coupling sanity case
(`forward-model/viz/run_coupling_3d_sanity.py`). Adds the **lateral overland routing** + **lateral
groundwater seepage** regime that B4 (flat column, no lateral flux) and the subsurface sweep deferred.

## 2. The in-house case to match (`run_coupling_3d_sanity.py` / `validation/sanity/coupling_3d__2026-06-08.md`)
- Domain: **5×1×1 m box**, mesh **16×6×8**, bed tilted `z_b = S0·(L−x)`, **S0=0.05** (downslope to x=L).
- Antecedent: **hydrostatic water table at z=0.35** (`ψ = 0.35 − z`; saturated below). Manning **n=0.05**.
- Storm: **0.5 m/day × 0.30 day** then recession to **0.50 day** (same persistent storm for both soils).
- **Texture contrast** (Carsel & Parrish): **SAND** (Ks=7.13 ≫ rain → infiltrate, water table rises
  0.35→0.75, **lateral groundwater outflow dominates**, ~0 overland) vs **LOAM** (Ks=0.25 < rain →
  infiltration-excess, **overland runoff dominates**: 0.474 vs 0.028 m³ lateral GW).
- **Two outlets, both at x=L:** (i) a surface **Manning overland edge outlet** (codim-2 ridge line
  discharge); (ii) a **lateral groundwater GHB** on the x=L side FACE (kr-limited → only the saturated
  zone seeps; external head **H_ext=0.20**, conductance **0.5**).
- In-house quantities (`coupling_3d__*.nc` contract): `surface_depth_map` d(x,y); `head_xsec`/`theta_xsec`
  on the y=0.5 section (water table = ψ=0 contour); top-down θ(x,y) at 3 z-layers; `outflow`/`cum_outflow`
  (overland), `drainage`/`cum_drainage` (lateral GW), `cum_rain`, `mass_balance_error`. Global balance:
  `Δtotal = cum_rain − cum_outflow − cum_drainage (+ clip_mass_adjust)`.

## 3. ParFlow 3-D coupled setup (to build — `../parflow/cases/coupled_hillslope_3d.py`)
- 3-D box NX×NY×NZ from 5×1×1 m (match 16×6×8, or refine). `TopoSlopesX = S0 = 0.05` toward x=L
  (verify sign from the depth profile, as in `overland_hillslope.py`); `TopoSlopesY = 0`.
- **Top BC: `OverlandFlow`** (the surface store; from B4 the mass-consistent store = `max(ψ_top, 0)`,
  NOT the −DZ/2 offset). `Mannings = n_SI/86400` (day-units).
- Downslope **x=L**: overland water routes off the edge via the slope (open overland boundary); the
  **lateral GW GHB** is the design risk (§4) — ParFlow has no direct GHB, approximate with a **seepage
  face** or a saturated-zone **Dirichlet head BC at H_ext** (e.g. `DirEquilRefPatch`/`OverlandKinematic`
  combos), or a thin high-K drain column. Decide in step-1.
- Base + non-outlet sides: no-flow (`FluxConst=0`). IC: `HydroStaticPatch` water table at z=0.35.
- Two soils (sand, loam); storm+recession via the **B4 two-phase restart** (`PFBFile` IC carries the pond).

## 4. Dominant risks (resolve up front)
- **Small-Manning overland conveyance (README §5a) is NOW ACTIVE.** n=0.05 ⇒ `n_PF≈5.8e-7`, `1/n≈1.7e6`.
  In the STANDALONE case ParFlow evacuated at ~0 depth (didn't form the Manning sheet). In a real 3-D
  coupled hillslope with a 5% slope this is ParFlow's *validated* regime — but **verify the LOAM-overland
  scenario routes + ponds + recedes sensibly FIRST** (§5 step 1). Fallback: a rougher Manning (n≈0.15).
- **GHB → ParFlow mapping** (lateral GW seepage on x=L): no direct GHB analog; the seepage-face / head-BC
  approximation may not match the in-house kr-limited conductance·(ψ−H_ext) exactly — a documented delta
  and the second thing to settle in step 1–2.
- Carried: OverlandFlow store = `max(ψ_top,0)` (B4); two-phase recession; air-entry-K cap (active — the
  surface saturates) + Ss + FV-vs-FEM deltas; 3-D coupled stiffness (sand Ks≫rain; loam excess) → adaptive dt.

## 5. Comparison quantities + harness
- **Partition is now 4-way:** `cum_rain = infiltration + ponding + overland outflow + lateral GW`. This
  partition + both hydrographs (overland, lateral GW) is the headline comparison.
- d(x,y) ponding map; ψ/θ cross-section (water-table ψ=0 contour); overland + lateral-GW hydrographs;
  mass balance. The **1-D coupled HTML generator does NOT carry over** (3-D needs map/section/hydrograph
  views) → new `make_comparison_coupled_3d_html.py` (mirror `forward-model/viz/make_coupling_3d_html.py`'s
  views, two-model overlay) + `build_comparison_coupled_3d.py` (re-run in-house + load ParFlow → `.nc` + HTML).
- Output: `data/coupled_3d__{sand,loam}__<date>.nc`, `html/coupled_3d__{sand,loam}__<date>.html`
  (both gitignored, regenerable — per the §6 README policy).

## 6. Suggested order
1. ParFlow 3-D deck, ONE scenario (**loam_overland** — overland-dominated) → verify it routes overland
   off the x=L edge, ponds, and the recession drains; **settle the small-Manning conveyance + the GHB
   mapping** (the two dominant risks). This is the make-or-break gate.
2. **sand_lateral_gw** → verify infiltration + the rising water table + lateral GW seepage; finalize the GHB.
3. In-house 3-D re-run + first comparison `.nc` + HTML (ponding map + cross-section + dual hydrographs + partition).
4. Both scenarios; document `README.md` §5d + change log; update memory. Closes the lateral-routing /
   groundwater-seepage regime — the 3-D extension of B4.
