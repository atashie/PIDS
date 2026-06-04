# F. Module 5 — domain (mesh, topography, properties) + 1D-in-3D embedding mechanics

## 1. Scope and role

Module 5 (`domain`) builds the discrete world the other modules solve on: the **mesh**, the **topographic surface**, the **SE-Piedmont subsurface property fields** (`K_s`, porosity, van Genuchten α/n/θ_r/θ_s), and — now the core deliverable — the **1D-in-3D embedding mechanics** that places every PIDS feature as a parameterized 1-D vector in the 3-D Richards continuum. It produces mesh + tags + property `Function`s and the geometric/connectivity operators that Module 4 (coupling) and the `pids-features` layer (Module 4 in the README, the embedded-vector physics) assemble into the monolithic residual. It does **not** assemble physics; it hands neighbors the discrete scaffolding plus the connectivity factor `Ω` (below).

## 2. Mesh generation and the 1D/2D/3D ladder

Two backends, one dimension-agnostic interface `build_mesh(spec) -> (Mesh, MeshTags...)`:

- **DOLFINx built-ins** (`create_interval`, `create_rectangle`, `create_box`) for canonical analytical-test domains — the 1-D Philip/Green-Ampt column (reuse the probe host), 2-D tilted-V, 3-D superslab. Cheap, exactly reproducible, used for Tier-1 convergence studies.
- **gmsh** (via `dolfinx.io.gmshio.model_to_mesh`) for irregular topography and for **conforming** feature meshes when a feature must be edge-aligned. gmsh OCC kernel builds the box, imprints feature polylines as embedded 1-D entities (`gmsh.model.mesh.embed(1, line_tags, 3, volume_tag)`), and writes physical groups that become MeshTags.

The same UFL solver code runs on all three; only `gdim` and the gravity unit vector `e_g` differ. The mesh module exposes `e_g` so subsurface/overland stay dimension-agnostic. We validate 1-D first, then re-mesh 2-D/3-D and re-run identical assembly for conservation + plausibility.

## 3. The 1D-in-3D embedding mechanics (core)

Each feature is a **polyline** Γ_f (ordered list of points + radius `r_f`, per-face σ specs, fill grain-size → `K_feat`, effective porosity). Two embedding realizations, both supported:

1. **Conforming (edge-aligned).** gmsh `embed` forces the polyline onto mesh edges → a 1-D `submesh` whose entities are exact mesh edges sharing DOFs with the 3-D host along Γ_f. Preferred for the verification reference and for tunnels.
2. **Non-conforming (immersed).** The polyline cuts arbitrarily through 3-D cells; coupling is realized by a **distributed Dirac source** along Γ_f, evaluated at quadrature points via `dolfinx.geometry` bounding-box-tree point location (`compute_colliding_cells`). Preferred for dense networks where edge-conforming meshing is intractable.

The feature carries its own 1-D function space `V_f` (P1 on the submesh / line quadrature) holding feature head `h_feat`. Axial Darcy conveyance is the standard 1-D Laplacian on Γ_f with `K_feat` and cross-section `A_f = w·d`; storage is `θ_eff · A_f` per unit length (Module 5 supplies the geometry `A_f`, length, tangent; the `pids-features` module owns the conveyance/storage UFL).

### 3.1 Per-face exchange σ as a mesh connectivity factor Ω (Peaceman-style)

The exchange flux per unit feature length is `q = σ·(h_feat − h_soil)`. The physics spec defines σ as a **series** combination of a feature-face conductance `σ_face` and the surrounding-soil conductance; the soil leg is *not* a constant — it is `K(ψ)`-dependent and geometry-dependent. Module 5 supplies the **geometric connectivity factor** so the soil leg is not an arbitrary number but a Peaceman-well-index analogue:

```
σ(face, dir) = [ 1/σ_face(face,dir) + 1/(K_soil(ψ) · Ω_geom) ]^(-1)     # series, per face, per direction
Ω_geom = 2π / ln(r_eq / r_f)          # radial-inflow shape factor toward the line (3-D)
r_eq   = C · h_cell                    # equivalent radius from local cell size h_cell; C≈0.2 (Peaceman 0.14–0.2)
```

`Ω_geom` is computed **per host cell** from the cell size `h_cell` and feature radius `r_f`, so a coarse host cell still gets the *correct steady radial conductance* to a sub-cell-radius feature — this is exactly what lets ~0.08 m tunnels and ~0.025 m clay barriers live in a coarse 3-D mesh without global sub-meter refinement. The `K_soil(ψ)` factor is read from the live Richards state each Newton iteration (so dry clay's low K *and* its high matric potential ψ both enter natively), making the bidirectional behavior (below water table → drains; over dry soil → disperses) emergent, not switched. Per-face/per-direction σ_face values realize the taxonomy: **clay bottom σ_face=0 → sealed face (zero flux exactly)**; bare-channel sides/bottom σ_face→∞ → soil-limited; catch-drain/standard-pipe walls σ_face=0 except tagged inlet/outlet; one-way device = asymmetric (`σ_receive>0, σ_disperse=0`).

### 3.2 Entity tagging

MeshTags carry: per-feature `feature_id` on the 1-D entity set; a `face_role` tag per feature face ∈ {top, bottom, lateral} (and for conforming meshes, the actual 3-D facets abutting Γ_f); a `sealed` flag for clay faces (forces σ=0 in assembly); `inlet/outlet` tags for catch-drain/pipe endpoints coupling to overland/outfall; and `is_vertical` for tunnels (gravity-aligned, lateral exchange only, drains downward). Tags are the contract Module 4 reads to know *which* σ and *which* coupling each entity gets.

### 3.3 Vertical tunnels

A tunnel is a feature whose tangent ‖ `e_g`: axial Darcy is gravity-driven downward, lateral σ couples to surrounding soil along its length, its deep endpoint tags into deeper (higher-K) soil zones. Same embedding machinery; only orientation and endpoint tag differ.

### 3.4 Fidelity trade-off vs C-001 / C-004

The embedding decouples conduit radius from cell size, so **C-001 (lateral conveyance)** is carried by `K_feat·A_f` axial transport regardless of mesh coarseness, and **C-004 (conveyance ≫ percolation)** is the ratio of axial flux to integrated σ-exchange. The fidelity cost is that near-field radial gradients within `r_eq` are *not resolved* — they are represented by the steady `Ω_geom`. This is accurate when the feature operates near radial-steady locally; it under-resolves sharp transient fronts at the feature wall. Tier-1 quantifies the error against a fully-resolved 3-D reference (below) and reports the convergence of the embedded solution to it.

## 4. Topography

`build_surface(spec)`: synthetic analytic surfaces now (planar slope, tilted-V, Gaussian-hummock fields) returning a z(x,y) callable used to warp the 3-D box top and define overland elevation. **DEM/LIDAR interface** is a thin adapter `dem_to_surface(geotiff) -> z(x,y)` (rasterio read → interpolant → gmsh terrain surface → extruded/clipped volume), stubbed now, contracted later from `aerial-mapping`. A **DEM round-trip** test (sample synthetic z → mesh → re-sample) guards the ingestion path.

## 5. SE-Piedmont subsurface property generator

`generate_properties(mesh, profile_spec) -> dict[Function]` producing `K_s`, `θ_s`, `θ_r`, `α`, `n` as P0/P1 `Function`s. Properties are a **depth function** (Piedmont saprolite: organic/clay-rich A–B horizon → weathered saprolite → partially-weathered rock; K generally decreasing with depth, anisotropic) plus **material zoning via MeshTags** for layers and clay barriers. Parameters come from Module F's parameterization neighbor (grain-size + sorting → Kozeny-Carman/Hazen + Rosetta pedotransfer). For granular features, the **same grain-size → K** path yields `K_feat` (Kozeny-Carman/Hazen on the fill), keeping feature and matrix on one parameter basis. Anisotropy and depth-decay are analytic functions evaluated at cell centroids; clay zones overwrite to `K≈0, θ_eff≈0`.

## 6. Three-tier sanity plan

- **Tier 1 (pytest, TDD):** (a) generated profile reproduces the specified K/porosity/retention **depth function** to tolerance at sampled depths; (b) **DEM round-trip** error < tol; (c) **embedding resolves a feature without global refinement** — embedded-vector conveyance + bidirectional exchange vs a **fully-resolved 3-D reference** (feature meshed at its true radius) agree within tolerance, **with a convergence study** in host cell size `h_cell` showing the embedded solution converging to the resolved one as `Ω_geom` is refined; (d) **clay-sealed face passes exactly zero flux**; (e) storage = θ_eff·volume; plausibility (positive K/porosity, θ_r<θ_s, no NaN, tags partition the domain).
- **Tier 2:** steep vs flat surfaces; layered vs homogeneous profiles; sparse vs dense feature networks (conforming vs immersed embedding); saturated vs bone-dry antecedent to exercise the `K(ψ)`-dependent σ both as drain and as disperser; check mass-conservative property/embedding behavior in concert with subsurface.
- **Tier 3 (viz subagent):** 3-D and transect property field (K/porosity color); mesh with embedded feature vectors highlighted by `feature_id`/`face_role`; side-by-side embedded-vs-resolved comparison panel with the convergence metric. Emits NetCDF per the viz data contract; human sign-off.
