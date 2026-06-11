# Module 4 Phase-4: Host-Controlled Coupled Embedding (discriminating gate + discrete well index)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Process constraints (non-negotiable, from the Phase-3 retractions): TDD; **no "coupled" claim without
> the discrimination assertion passing AND adversarial review**; honest-failure is an acceptable outcome.

**Goal:** Genuine host-controlled coupling for the sub-grid `EmbeddedFeature` sorptive exchange (the
deferred C-004 coupled leg): the HOST's ability to deliver/accept water must control the uptake I(t),
and a scheme that merely integrates the offline clock and deposits mass must FAIL the gate.

**Architecture:** (1) Build the DISCRIMINATING gate FIRST — evolving-far-field resolved references
(Ref A depleting reservoir, Ref B host-history) that the offline clock fails BY CONSTRUCTION, plus a
closed-box embedded harness with a machine-precision mass ledger. (2) Measure the discrete equivalent
radius r_0(h) of the P1 ridge source (Peaceman-for-FEM) via steady Laplace solves with analytic log
truth. (3) The exchange bridges wall→cell through WI = 2π/ln(r_0(h)/r_w) reading the ON-Γ discrete
ψ(Γ) (no shell heuristic); the I-clock supplies only the early sub-r_0 transient, with the uptake held
in an EXPLICIT sub-grid reservoir mass term until handover.

**Tech stack:** DOLFINx 0.10 (WSL2 conda `pids-fem`), petsc4py/SNES, numpy. All FEM runs:
`wsl bash -lc "source /root/miniforge3/etc/profile.d/conda.sh && conda activate pids-fem && cd <forward-model> && PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python ..."`

**Working dir for all paths below:** `technologies/infiltration-runoff-model/forward-model/`
(plan doc itself lives at repo-root `docs/plans/`). Branch: `m4-phase4-coupled-embedding` off
`main` = 8a891c2.

---

## Why the previous two schemes failed (the kill-map the gate must enforce)

| Retracted mode | Mechanism | Gate element that kills it |
|---|---|---|
| 1. Naive co-located read at Γ (`scratch/_zcoupled_embed_probe.py`) | injected flux saturates the near-Γ cell; closure reads saturated ψ(Γ) as far-field drive; clock stalls; **worsens with refinement** (55%→78%, n=8→24) | the **non-degrading-with-refinement** assertion across n=8..32 |
| 2. Dual-scale shell read + storage gate (`scratch/_zdualscale_probe.py`, retracted d42432c) | with a fixed far field, ψ_shell ≈ ψ_i ∀t ⇒ dI/dt = the offline clock exactly ⇒ host never controls anything; any passive accumulator passes a fixed-far-field reference | **Ref A/B's evolving far field** + the explicit **discrimination assertion** (the offline clock must FAIL the same harness by a wide margin) |

Key quantitative anchors (LOAM, r_w=0.05 m, ψ_i=−1, Δθ≈0.187): infinite-domain reference end
I≈4.1e-2 m at t=6e-2 d; Ref A capacity I_max = Δθ·(R²−r_w²)/(2r_w) ≈ {3r_w: 3.74e-2, 5r_w: 0.112,
10r_w: 0.463} m — so the 3r_w reservoir is nearly exhausted within the Phase-1 window already; the
offline clock sails past I_max unbounded ⇒ discrimination is structural.

---

## Task 0: Baseline — suite green on the new branch

**Files:** none (verification only).

**Step 1:** Confirm branch = `m4-phase4-coupled-embedding`, HEAD = 8a891c2.

**Step 2:** Run the full suite (from `forward-model/`, WSL preamble above):
`python -m pytest tests/ -x -q`
Expected: all pass (suite was green at 8a891c2). If not green → STOP, diagnose before any Phase-4 work.

---

## Task 1: Ref A — depleting-reservoir DISPERSE references

**Files:**
- Create: `scratch/m4_phase4_refA_disperse.py`
- Output: `scratch/m4_phase4_refA_disperse.npz`

Reuse `scratch/m4_phase1b_disperse_reference.py` by import (`SOILS`, `_solve_to`, `_vertex_dx`,
`parlange_sorptivity`, `_LU` — cp linesearch, the disperse gotcha) exactly as `m4_phase1c` does
(`import scratch.m4_phase1b_disperse_reference as dz`).

**Step 1: write the generator.** 1-D cylindrical radial solve on [r_w, R_out], wall Dirichlet ψ=0,
**NO outer Dirichlet** (no-flow natural BC — just omit the far BC from `bcs`). Gravity-free
(z-invariant tunnel argument, verified in Phase-1b). Soils LOAM/SAND/SILT (CLAY stays excluded,
Phase-3 decision). R_out ∈ {3, 5, 10}·r_w = {0.15, 0.25, 0.50} m. Mesh n=400 graded as in 1b
(uniform interval is what 1b used — keep it). I(t) = ∫(θ−θ_i)·r dx / r_w (domain gain = wall influx
exactly, closed domain). Save per (soil, R): `t`, `I`, plus scalars `I_max`, `S_parlange`, `dtheta`.

Per-(soil,R) time grids: geomspace from 1b's per-soil `t_start` formula to a `t_end` tuned so final
I/I_max reaches the **depletion targets ≥{80% @3r_w, 60% @5r_w, 30% @10r_w}**. First guesses (LOAM):
t_end = {0.12, 0.6, 3.0} d; scale other soils by their 1b `t_end` ratio (SAND ×3e-3/6e-2, SILT
×4e-1/6e-2). 32 samples per curve. Print final I/I_max; adjust t_end and re-run if targets missed.

**Step 2: in-script self-checks (assert, not just print):**
- I strictly monotone increasing; final I ≤ I_max·(1+1e-9).
- **Early-window cross-check vs the committed infinite-domain fixture** `tests/data/m4_phase1b_disperse_refs.npz`
  (`{SOIL}_tunnel_I` interpolated onto the shared early times where front ≪ R_out): rel dev < 1.5%.
  This validates the no-flow machinery against the signed-off reference before trusting the new curves.
- Print (not assert yet — Task 4 asserts with the measured numbers): offline-clock
  (`sorptive_clock(t, S_parlange, dtheta, r_w, F_cylindrical)`) rel-L2 vs each curve.

**Step 3: run** (WSL preamble), inspect the printout. Expected: early-window match <1.5%; offline-clock
rel-L2 LARGE for 3r_w/5r_w (anticipate ≳25–60%) and smaller for 10r_w (the 10r_w curve is the
"barely-depleting" control — discrimination is carried by 3r_w/5r_w).

**Step 4: commit** the script (npz committed as a fixture in Task 4).

---

## Task 2: Ref B — host-history DISPERSE reference

**Files:**
- Create: `scratch/m4_phase4_refB_disperse.py`
- Output: `scratch/m4_phase4_refB_disperse.npz`

**Design:** closed domain R_out=5r_w (as Ref A) + a **volumetric re-wetting pulse**: source density
s [1/day] uniform over the band r ∈ [3r_w, 4r_w], active t ∈ [0.15, 0.25]·t_end(5r_w), total volume =
30% of the band's remaining capacity Δθ·π(R_b2²−R_b1²) (per radian in the 1-D radial weak form: add
`− s_t·v·r·dx` over the band with `s_t` a `fem.Constant` toggled by time). Pulse must land while the
front is still inside ~3r_w (check; retune timing if not). Cap: if any ψ > −0.01 in the band during
the pulse, halve the volume (avoid ponding/saturation stiffness).

I_wall(t) = (∫(θ−θ_i)·r dx − cum_source)/r_w, cum_source tracked exactly (constant rate × elapsed
active time × band volume).

**Independent cross-check:** accumulate the discrete wall flux by the residual REACTION at the wall
dof each accepted step (assemble the residual form with the converged ψ, no BC zeroing; the wall-row
entry = the discrete influx). Cumulative reaction-I vs ledger-I: rel dev < 1%. (This is the
measurement machinery Ref-A doesn't need but proves the ledger.)

**Asserted checks:** pre-pulse match to Ref A(5r_w) < 0.5% (same machinery, same domain); post-pulse
I_wall(t) drops measurably below Ref A(5r_w) — print the end-of-window gap (expect several %·to·tens
of %; this gap is what the embedded scheme must track and the offline clock cannot).

Soil: LOAM only (Ref B is a discrimination instrument, not a generality sweep). Commit script.

---

## Task 3: DRAIN mirrors (Ref A-drain, Ref B-drain)

**Files:**
- Create: `scratch/m4_phase4_refAB_drain.py`
- Output: `scratch/m4_phase4_refAB_drain.npz`

Drain is a CORE use case (drainable-porosity depletion INTO the feature IS the depleting-reservoir
scenario). Mirror of Tasks 1–2 reusing `m4_phase1c_drain_reference.py` conventions: soil starts
ψ_i=0 (saturated), wall Dirichlet ψ=−1, **bt linesearch** (`_DRAIN_LU` — the saturated-start gotcha),
n=3200 interval (1c's converged resolution for the sharp desaturation front). No-flow outer at
R_out ∈ {3,5,10}·r_w; I_removed(t) = ∫(θ_i−θ)·r dx / r_w plateaus at the same I_max (Δθ identical,
|Δψ|=1 mirror). Ref B-drain: re-wetting pulse mid-drain in the [3r_w,4r_w] band → drainage
re-steepens/extends (the OPPOSITE-sign history response to Ref B-disperse — good lens diversity).
Soils: LOAM (+SAND for the drain generality check in Task 7's acceptance sweep — SAND was the
Phase-3 drain gravity exception; the 1-D radial mirror has no gravity so it's clean here).
Checks as Tasks 1–2 (early window vs `tests/data/m4_phase1c_drain_refs.npz` `{SOIL}_tunnel_I`;
offline THROTTLE clock rel-L2 printed). Commit script.

---

## Task 4: The discriminating gate (TDD) + retracted-scheme failure baselines

**Files:**
- Create: `tests/data/m4_phase4_refA_disperse.npz`, `tests/data/m4_phase4_refB_disperse.npz`,
  `tests/data/m4_phase4_refAB_drain.npz` (copied fixtures from Tasks 1–3)
- Create: `tests/test_coupled_gate_refs.py` (fast, numpy-only)
- Create: `scratch/m4_phase4_embedded_harness.py` (the reusable closed-box runner)

**Step 1 (TDD): write `tests/test_coupled_gate_refs.py` failing first** (fixtures not yet copied):
- fixtures load; every I monotone; Ref-A finals within [target_fraction, 1.0]·I_max;
- **DISCRIMINATION (the assertion without which the gate is vacuous):** offline disperse clock
  (`sorptive_clock`, F_cylindrical, Parlange S, C=1) rel-L2 vs Ref A(3r_w) and vs Ref B ≥
  `CLOCK_FAIL_MARGIN`; offline drain clock (F_throttle, semi-empirical S_des) vs Ref A-drain(3r_w)
  ≥ margin. Set `CLOCK_FAIL_MARGIN` from the Task-1/3 measured numbers, **pre-registered at ≥3×
  `EMBEDDED_TOL`** (gate constants module-level: `EMBEDDED_TOL = 0.10`, `CLOCK_FAIL_MARGIN ≥ 0.30`
  expected; lock the actual values BEFORE any WI implementation run — no post-hoc tuning).

**Step 2:** copy fixtures, tests pass. Commit (fixtures + test).

**Step 3: the embedded closed-box harness** (`scratch/m4_phase4_embedded_harness.py`). One function
`run_embedded(scheme, soil, R_out, n, t_grid, direction)` returning `(I(t), ledger)`:
- Box [0, Lx]×[0, L]² with **L = sqrt(π·(R_out²−r_w²))** — host water capacity per unit feature
  length exactly matches the reference annulus (this choice REQUIRES the sub-grid reservoir to drain
  into the host at handover, else the end-state double-counts — see Task 6). Lx short (e.g. 4 cells).
  n EVEN (feature line y=z=L/2 must lie on the vertex lattice). Gravity OFF (matches the radial refs;
  isolates coupling fidelity from the documented ~4% gravity asymmetry).
- ALL faces no-flow (closed). Disperse: host ψ_i=−1, feature H_f=0 fixed; drain: host ψ_i=0,
  H_f=−1 fixed, **bt linesearch** (disperse keeps cp).
- `scheme` = callbacks (pre-step: set Ω/sources; post-step: advance states) so the SAME harness runs
  the retracted dual-scale, the bare offline clock, and the Task-6 WI scheme.
- **Mass ledger every sample:** host θ-gain + sub-grid reservoir + feature storage − ∫q dt; assert
  |closure| ≤ 1e-10 relative. Ref-B variant: + the same volumetric pulse (band = the annular shell
  3r_w<ρ<4r_w around the feature line, same total volume as the reference pulse).

**Step 4: baseline the failures.** Run (a) the RETRACTED dual-scale scheme (port the logic from
`scratch/_zdualscale_probe.py` into a scheme callback) and (b) the bare offline clock against
Ref A(3r_w, 5r_w) at n ∈ {8, 16, 24}. Record rel-L2 and the refinement trend. Expected: both fail
(clock by construction; dual-scale via shell-read resolution-dependence and/or late-time tracking).
**If the dual-scale PASSES the closed-box gate → STOP and reassess the gate design** (that would mean
the gate does not discriminate host control — a finding, not a pass). Document numbers in the script
header. Commit harness + a short results note in the script docstring.

---

## Task 5: Measure r_0(h) — the P1 ridge discrete well index

**Files:**
- Create: `scratch/m4_phase4_well_index.py`
- Create: `tests/test_well_index_p1.py`
- Modify: `pids_forward/physics/sorptive_closure.py` (add the measured constant(s) with provenance)

**Math (exactly checkable):** steady Laplace −Δu = δ_Γ (unit line source per unit length: RHS
`v·dΓ` on the ridge measure) on the SAME structured tet lattice the harness uses (feature vertex-line
along x). Dirichlet on all boundary faces from the ANALYTIC log u_bc(x) = −(1/2π)·ln(ρ(x)) (ρ =
perpendicular distance to the line; valid for any boundary shape since the analytic solution is
global). Then the discrete well-block value defines r_0: **r_0 = exp(−2π·u_h(Γ))** (read u_h on
mid-domain Γ vertices, away from the x-ends).

**Steps (TDD):**
1. Failing test `tests/test_well_index_p1.py`: (a) far-field fidelity — u_h at probe vertices
   ρ ∈ [2h, 4h] matches the analytic log < 1%; (b) r_0/h from two resolutions (n=8, 16) agree < 3%;
   (c) r_0 > 0 and r_0/h ∈ (0.05, 2.0) (sanity bracket).
2. Implement the measurement script + a small importable helper (e.g.
   `measure_r0(n, L)` in the scratch script; the TEST builds its own tiny meshes inline or imports
   the helper via `scratch.` import, following the `m4_phase1c` pattern).
3. Sweep n ∈ {8, 12, 16, 24, 32} (3-D tets) + the 2-D triangle cross-section analog (cheap, literature-
   comparable). Report r_0/h mean ± spread; assert spread < 3%.
4. Record `R0_OVER_H_P1_TET` (and `_TRI`) in `sorptive_closure.py` with a provenance comment
   (measured, date, script). Commit.

**Known risk (flag, don't hide):** WI = 2π/ln(r_0(h)/r_w) **degenerates at r_0 ≈ r_w** (h ≈ r_w/(r_0/h):
e.g. if r_0/h≈0.6, that's h≈0.083, i.e. n≈12 on the L≈0.46 m 3r_w box — INSIDE our sweep range). The
formula stays well-posed on both sides (negative ln with Φ(r_0) inside the wall still gives positive
flux) but blows up AT equality, where the physical statement is "the cell resolves the wall: zero
bridge resistance". Task 6 must handle the crossover explicitly (e.g. conductance cap at the resolved
cell-scale Kirchhoff conductance, or a blended formulation) — design on evidence, document the choice.

---

## Task 6: WI prototype against the gate (the research core)

**Files:**
- Create: `scratch/m4_phase4_wi_probe.py` (scheme callbacks for the Task-4 harness)

**The scheme (leading hypothesis, kickoff-approved):**
- **Post-handover exchange:** per unit length q = WI_K·[Φ(H_f) − Φ(ψ_Γ)], WI_K = 2π/ln(r_0(h)/r_w),
  with ψ_Γ the ON-Γ discrete host value, in UFL via `soil.kirchhoff_ufl(psi, Hf)` (differentiable;
  Ω-equivalent coefficient = WI_K/perimeter to reuse `sorptive_into_host`/`sorptive_into_feature`).
  No shell, no lag beyond the handover gate.
- **Early sub-r_0 transient:** while I < I_fill = Δθ·(r_0²−r_w²)/(2r_w) (only if r_0 > r_w, else
  handover immediately): flux from the validated clock, **driven by ψ_Γ** (nothing has reached r_0
  yet, so Γ's discrete value IS the right drive and stays ≈ host-controlled), **all uptake into the
  EXPLICIT sub-grid reservoir** — the host receives NO source yet, so failure-mode-1 saturation
  cannot corrupt the drive.
- **Reservoir release at handover** (the capacity argument in Task 4 forces release): design probe,
  pick on evidence vs Ref A(3r_w): (i) release rate proportional to the ongoing WI flux with a
  relaxation time ~ the handover timescale; (ii) one-shot distributed deposition weighted by the
  measured discrete Green's profile u_h(ρ) from Task 5; (iii) ledger-only (never inject) — predicted
  to FAIL Ref A late-time by the capacity shortfall ≈ I_fill (keep as the control).
- Drain mirror: same structure, throttle clock + S_des, direction from sign(Φ(H_f)−Φ(ψ_Γ)), separate
  I accumulators (no reset on reversal — Arik 2026-06-09), bt linesearch.

**Iteration loop:** LOAM disperse Ref A(3r_w) first at n ∈ {8, 16}; then the full n ∈ {8,12,16,24,32}
× R_out ∈ {3,5,10}r_w; then Ref B; then drain. Target: rel-L2 ≤ `EMBEDDED_TOL` (0.10), and
**non-degrading**: rel-L2(n=32) ≤ rel-L2(n=8) + 0.02.

**Checkpoints / honest off-ramps (two retractions happened here — these are mandatory):**
- After the first n∈{8,16} pass: if tracking fails, diagnose mechanism FIRST (log ψ_Γ vs the
  reference Φ at r_0; WI conductance vs the resolved cell conductance; reservoir trajectory) — use
  superpowers:systematic-debugging, not parameter-fiddling. Documented fallbacks: the r_0≈r_w
  crossover handling (Task 5 risk); a transient correction to WI (front between r_0 and the cell
  scale); re-derived I_fill for the box-lattice geometry.
- If coarse-n genuine coupling proves impossible without enrichment: **SAY SO** — write the
  characterization (what tracks, what doesn't, why), commit the negative result + the gate (the gate
  is valuable regardless), and stop before Task 7. That is a publishable honest outcome.

---

## Task 7: Production integration (TDD) + acceptance

**Files:**
- Modify: `pids_forward/physics/feature.py` (new `coupling="well_index"` mode in
  `configure_sorptive`; explicit reservoir state + `sub_grid_reservoir()` accessor; handover +
  release logic; the EXPERIMENTAL shell path: replace or excise per what Task 6 learned — decide at
  review, do not silently keep dead code)
- Create: `tests/test_coupled_embedding_gate.py` (the discriminating gate as durable tests)
- Create: `scratch/m4_phase4_acceptance.py` (the full sweep: SAND/SILT generality, n→32, 10r_w,
  drain SAND)

**TDD order:** port the Task-6 scheme into `EmbeddedFeature` one behavior at a time, each behind a
failing test first: (1) WI_K coefficient from r_0(h) (unit, no solve); (2) reservoir
ledger + machine-precision closure on a 2-step toy box; (3) reduces-to-clock pre-handover; (4) the
gate itself — embedded vs Ref A(3r_w & 5r_w) LOAM n∈{8,16} ≤ EMBEDDED_TOL, vs Ref B, drain mirror,
**plus the discrimination twin** (bare offline clock through the same harness fails by
≥ CLOCK_FAIL_MARGIN) and the non-degradation pair. Keep test runtime sane (small n in tests; the
full sweep lives in the acceptance script and its numbers go in the script header + the Tier-3 HTML).
Full suite green. Commit per behavior.

---

## Task 8: Adversarial review (MANDATORY before any "coupled" claim) + Tier-3 + sign-off

1. **Adversarial review** (multi-agent + Codex, the Phase-3 protocol): the brief MUST include the two
   retracted failure modes as attack templates, plus: "is the gate actually discriminating, or could
   a passive accumulator pass Ref A/B?", "is the reservoir release a hidden offline clock?", "are
   EMBEDDED_TOL / CLOCK_FAIL_MARGIN pre-registered or post-hoc?", "does the r_0≈r_w crossover hide a
   fitted knob?". Fix-then-ship; re-run the gate after fixes.
2. **Tier-3 HTML** by a separate viz subagent (gate curves + n-sweep + mass ledger + r_0(h)
   measurement + the Ref-B history response), into `validation/sanity/viz/`.
3. Sign-off note `validation/sanity/pids_features_phase4__<date>.md`; **await Arik's explicit
   sign-off**; merge to main only after.

---

## Risk register

- **r_0 ≈ r_w degeneracy** inside the planned n-sweep (Task 5/6) — explicit crossover handling, on
  evidence.
- **Ref-B pulse saturating the band** (ψ>0 stiffness) — pulse volume cap + timing check (Task 2).
- **Drain closed-box stiffness** (all-saturated all-Neumann start, ψ=−1 wall) — bt linesearch
  ([[pids-fem-saturated-wall-linesearch]]); if dt collapses, soften to ψ_i=−1e-3 and document.
- **Offline-clock margin too small at 10r_w** — discrimination is asserted on 3r_w/5r_w + Ref B only;
  10r_w is the barely-depleting control.
- **Quadrature**: cap degree 8 on Darcy volume terms ([[pids-fem-quadrature-degree-cap]]); vertex
  rule on storage terms (the 1b/1c pattern).
- **Block spaces**: H_f and ψ co-located on ONE space here (the Phase-2/3 pattern, H_f fixed) — no
  block solve in the harness; if Task 7 wires into `CoupledProblem` blocks, DISTINCT spaces
  ([[pids-fem-block-distinct-spaces]]).
- **BLAS threading**: single-thread pins on every FEM run ([[pids-fem-blas-threading]]).
