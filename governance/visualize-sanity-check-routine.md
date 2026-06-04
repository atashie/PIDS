# Visualize Sanity-Check Routine (Interactive HTML)

**Status:** standard, effective 2026-06-04. Defines how the **Tier-3** visual inspection of [`claude-sanity-check-routine.md`](claude-sanity-check-routine.md) is produced: a **separate visualization subagent** turns a module's sanity-run output into a **single self-contained interactive HTML** for the human to inspect and sign off.

> **Why a separate subagent.** The visual check is independent evidence. Whoever built the module does not also build its visualization — this keeps the human gate from inheriting the builder's blind spots.

---

## Output standard (non-negotiable)
- **One self-contained `.html` file.** All data embedded (JSON) and all JS inlined or via a vendored library file — **no web server, no network/CDN, no build step.** It must open by **double-click on Windows**.
- **Interactive:** time slider / animation, hover read-outs, zoom/pan. Static-only is insufficient for time-dependent checks.
- **Library:** **Plotly.js** (vendored locally, offline) as the default. Matplotlib (`Agg`) is allowed for static fallbacks or for pre-rendered animation frames stacked behind a slider.
- **Self-documenting:** title states **module · scenario · date**; every axis labelled **with units**; legends present; an embedded **metrics panel** shows the numbers that decide the check (e.g., mass-balance error, peak runoff, min valve throughput, max iterations).
- **Lightweight & deterministic:** target < ~10 MB; regenerating from the same result data yields the same HTML (downsample large fields rather than embedding millions of points).

---

## Visualization catalog (pick what fits the check)
| Check / module | Standard view(s) |
|---|---|
| Runoff / forcing | **Hydrograph + hyetograph** — twin-axis time series (rainfall bars inverted top axis; discharge/runoff line); recession on log-y option. |
| Subsurface wetting front | **Transect heatmap** of saturation or pressure head along a y-transect (depth × distance) with a **time slider**; overlay the water table / front position. |
| 1-D Richards column | **Profile plot** of ψ and θ vs depth with a time slider; infiltration-front advance. |
| Overland / storm | **Surface-water-depth animation** over the 2-D domain (slider or autoplay); optional **velocity quiver** overlay. |
| Erosion threshold | **Map** of in-channel velocity / shear with **damage-threshold exceedance** highlighted. |
| Recharge / influence | **Longitudinal water-table profile** over time; cumulative recharge flux. |
| Diagnostics (any module) | **Mass-balance error vs time**; **Newton iterations vs time**; **error vs resolution** (MMS convergence order, log-log). |

A sanity HTML usually combines 2–4 of these (the physical view(s) + a diagnostics panel).

---

## Standardized data contract (decouples viz from solver)
Sanity runs **emit results in a documented, self-describing format** so the visualization generator never imports the solver:
- **Preferred:** NetCDF via `xarray` (named dims/coords/units as attributes) — e.g. `time`, `z`, `x`, `y`, variables `saturation`, `head`, `surface_depth`, `velocity`, plus scalar diagnostics and a `metrics` attr dict.
- **Acceptable for small/1-D:** `.npz` with named arrays + a sidecar JSON of metadata/units/metrics.
- Result files live beside the run; the generator reads only this contract.

---

## Build pattern (for the visualization subagent)
1. Read the standardized result file for the module/scenario (do **not** import the solver).
2. Select views from the catalog appropriate to the check; downsample fields for weight.
3. Generate **one** self-contained HTML (embedded JSON + vendored Plotly), including the **metrics panel** and full titling/units.
4. Save to `technologies/infiltration-runoff-model/validation/sanity/viz/<module>__<check>__<YYYY-MM-DD>.html`.
5. **Verify before reporting:** the file is self-contained (opens offline), renders the expected interactive views, and surfaces the deciding metrics. Report the path and a one-line "what to look for."

A reusable generator should live at `forward-model/viz/` so each subagent calls a shared, tested function rather than hand-rolling Plotly each time.

---

## Acceptance (Tier-3 gate)
The HTML clears Tier-3 only when it (a) opens **offline by double-click**, (b) shows the interactive view(s) the check requires, (c) displays the deciding metrics, and (d) the **human signs off** in the sanity report. Visual sign-off + the agents' Tier-1/2 evidence together close the module's [definition of done](claude-sanity-check-routine.md#definition-of-done-a-module).
