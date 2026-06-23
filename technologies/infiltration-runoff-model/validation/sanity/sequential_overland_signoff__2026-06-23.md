# Sanity Check (Tier-3 sign-off) — sequential operator-split overland coupling — 2026-06-23

- **Module / version:** `pids_forward/physics/sequential_coupling.py` (`SequentialCoupledProblem`;
  defaults ω=1.0, `route_substeps=4`). Branch `overland-sequential`. Decision record
  `docs/plans/2026-06-22-overland-flow-sequential-coupling-decision.md`; upstream spike + calibration
  `validation/sanity/overland_split_spike__2026-06-22.md`,
  `validation/sanity/overland_transport_calibration__2026-06-23.md`.
- **Couplings exercised:** implicit 3-D Richards (solved ALONE per step, pond carried in ψ as
  `max(ψ,0)`) + an explicit Manning rate-limited downslope routing sweep injected as an ω-relaxed
  Neumann source; sand-zone GHB subsurface conveyance + two Manning surface outlets.
- **Status: Part-B Tier-3 artifact BUILT + RUN. Arik visual sign-off PENDING.**

## The point of this artifact

This is the case that **triggered** the whole redesign: a coarse-**SAND** conveyance channel
intercepting convergent storm runoff on a low-K **CLAY** hillslope (`scratch/m4_sand_channel_3d_demo.py`
— convergent overland flow + a stiff, near-incompressible van Genuchten n≈1.09 clay). It is the
frontier-hard regime where **both** monolithic Manning schemes fail, and where the operator split is
supposed to win *structurally* (the stiff Richards Newton never shares a Jacobian with the surface
routing). The sign-off shows OLD-fails / NEW-succeeds side by side on the **same** mesh / soil / forcing.

## Headline result — sand channel in clay (native demo mesh 20×12×5, 4 914 DOFs)

| scheme | outcome | reached | dt behaviour | conservation |
|--------|---------|---------|--------------|--------------|
| **NEW `SequentialCoupledProblem`** | **COMPLETES to T_END** (84 steps, 56 s) | t = 1.200 / 1.2 d | climbs to the controller ceiling dt=3e-2 | **\|bal\|/cum_rain = 4.71e-12** (max 9.87e-10) |
| OLD `overland_scheme="upwind"` | **DT-COLLAPSE** (documented 2026-06-22; shown as a replay) | t ≈ 0.11 / 1.2 d | dt→8e-10 (documented; *this* deep-berm variant limps slowly rather than hard-collapsing) | — |
| OLD `overland_scheme="galerkin"` | **SAWTOOTH dt-PIN** (live capture) | t = 0.0065 / 1.2 d (76 steps, 180 s wall) | dt pinned ≈ 4.12e-5 (effectively non-terminating) | — |

**NEW interception story (where the routed/conveyed water went):** of **1.120 m³** that became
runoff/conveyance, the **channel intercepted 21 %** (subsurface Darcy conveyance via its sand-zone GHB
**0.077 m³** + dispersion into the surrounding clay **0.158 m³** = **0.235 m³** capture) and **79 %**
escaped at the surface (routed off-domain, dominated by the toe edge). Peak ponded depth ≈ 1.00 m (the
downslope berm pools a near-full-depth lake in the swale behind it — exactly the stiff deep-ponding
condition that breaks the monolith). cum rain = 1.800 m³.

**Reading of the panels:** the left (dt-vs-t) panel is the money plot — NEW dt rises and holds at the
ceiling while OLD upwind plunges into a dt-collapse and OLD galerkin pins at a tiny dt (the convergent
sawtooth). The middle panel shows the NEW run's mass-balance closure sitting far below the 10⁻³ coupling
bar at machine precision throughout. The right panel shows the channel capture (conveyance + dispersion)
vs surface escape accumulating over the storm.

## Corroborating case — convergent tilted-V (the original sawtooth pathology)

The tilted-V is the canonical convergent-flow geometry where galerkin sawtooths. In THIS sign-off the
loam (Ks=0.25) V case was **infiltration-dominated**: the NEW sequential run completed cleanly (40
steps, no dt-collapse, machine-tight balance) but routed ≈0 to the surface outlet (the rain infiltrated
rather than running off), so it does **not** illustrate the routing contrast and is **omitted from the
HTML panel** (kept honest — the panel would otherwise claim "routes to the outlet" while showing zero).
The convergent-flow cure on the tilted-V is instead validated by the **B7 automated regression**
`test_convergent_tilted_v_regression_no_collapse_conserves_routes`: under a runoff storm the NEW scheme
completes the convergent V with no dt-collapse, conserves `|bal|/cum_rain = 6.8e-12`, and routes ≈69% of
rain to the outlet — exactly where galerkin sawtooth-pins. The sand-channel panel above is the complete,
load-bearing sign-off story.

## Why it works (the structural argument, validated here)

The operator split removes both monolithic pathologies *by construction*: (1) the **sawtooth** is gone
because there is no surface PDE — the explicit descending-head routing sweep treats convergence as its
native operation; (2) the **coupled-stiffness dt-collapse** is gone because the fast hyperbolic surface
update shares no Jacobian with the stiff parabolic Richards Newton. Conservation is ω/substep-independent
to ~1e-12 (the spike's CONSERVATION PROOF; each Manning sub-sweep telescopes), which is exactly what the
machine-tight balance trace confirms on this case.

## route_substeps transport calibration (the one accuracy knob)

The lateral-transport **rate** is set by `route_substeps` (Manning sub-sweeps per Richards step): a single
sweep under-resolves intra-step travel and throttles transport ~40–50× vs the resolved
`CoupledProblem(overland_scheme="upwind")` reference; **`route_substeps=4` matches** the resolved
reference drain timing, and ≥8 overshoots (the fully sub-stepped explicit Manning becomes a kinematic wave
that outruns the diffusion-wave reference). ω is 1.0 by default (no standing under-relaxation — the right
transport setting); under-relaxation is the robustness fallback only. Conservation is substep-independent.
(Record: `overland_transport_calibration__2026-06-23.md`.)

## Known limitations / honest caveats

1. **Interception accounting is slightly conservative.** The routing books *all* surface outflow into one
   `cum_outflow` bucket (it does not split the toe edge from the small channel-mouth surface outlet), so
   "surface escaped" folds in the minor channel-surface discharge → the reported 21 % intercept fraction
   *understates* capture. "Channel captured" = subsurface conveyance (GHB) + dispersion into clay, each
   measured independently.
2. **Near-saturation no-Ss fragility (orthogonal to transport).** A near-saturated column carrying a
   ≈zero pond can dt-collapse the standalone-Richards path on the unconfined (no specific-storage)
   near-saturation singularity. Carry the pond as a comfortably-positive head (as the validated cases do);
   the deep berm pool here is well-positive, so this did not bite.
3. **Kinematic vs diffusion-wave depth fidelity.** The routing answers the PIDS questions — how much water
   overlies each parcel / which way the excess runs / move it downslope as run-on — *not* accurate flood
   hydrographs, depths, or velocities. The Manning galerkin/upwind stack remains the validated
   diffusion-wave fallback (coexistence, not rip-out; the `make_overland_coupled(scheme=...)` factory
   selects).
4. **Serial only** (the top-facet routing graph is not ownership-aware — guarded loudly).

## How to regenerate

```bash
# from forward-model/ in the WSL pids-fem env (serial, threads pinned):
export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH"
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u viz/run_sequential_overland_signoff.py          # -> scratch/sequential_overland_signoff.npz
python    viz/make_sequential_overland_signoff_html.py    # -> viz/sequential_overland_signoff__2026-06-23.html
```

The self-contained HTML (inline Plotly, opens offline) is the visual sign-off artifact:
`viz/sequential_overland_signoff__2026-06-23.html`. The committed source of truth is the two scripts;
the HTML + the npz are regenerable.

## Verdict

NEW sequential operator-split overland **completes the trigger storm that breaks both monolithic Manning
schemes**, conserves to machine precision (\|bal\|/cum_rain = 4.71e-12), and intercepts runoff into the
channel — the structural cure works in the engine. **Tier-3 visual sign-off: PENDING (Arik opens the
HTML).**

- **Human sign-off:** Arik — _pending_.
