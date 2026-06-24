# Spike: sequential operator-split overland flow — gate results (2026-06-22)

> THROWAWAY validation spike for Part A of `docs/plans/2026-06-22-overland-sequential-coupling-implementation-plan.md`.
> Script: `forward-model/scratch/overland_split_spike.py`. Two cases that broke both monolithic schemes:
> the 3-D sand-channel-in-clay storm and the convergent tilted-V. DOLFINx 0.10 / WSL `pids-fem`, serial.
> All conservation numbers printed full precision; adversarially self-checked. NOT committed to the package.

## VERDICT (one line)
**GREEN as of the 2026-06-23 extension (see the CONSERVATION PROOF section at the bottom)** — the one open gate
(global handoff conservation) is CLOSED: design #1 (fully decoupled) conserves to ~5e-12 on BOTH cases incl.
recession, adversarially verified (self-limit + soil-source + a falsification hook). The body below is the ORIGINAL
2026-06-22 AMBER spike (architecture + kernel + robustness proven; handoff accounting then-open), preserved for the
failure map; the bottom section resolves it.

**[original 2026-06-22 verdict] AMBER — architecture + kernel + ROBUSTNESS proven; conservation accounting of the
handoff is the one remaining Part-B task.** The split ARCHITECTURE removes the monolithic dt-collapse; the ROUTING KERNEL is exact
+ non-teleporting + makes interception form; and the **sequential-ITERATIVE (Picard, under-relaxed source)
handoff RUNS BOTH the sand storm to completion without collapse** (the plan's named upgrade — it fixes what the
three single-pass handoffs each broke on). The remaining gap is purely a **handoff mass-accounting leak (~18% in
the spike's first-cut omega-source bookkeeping)** — a conservation-bookkeeping fix, not an architecture problem.
Single-pass handoffs are dead ends; the iterative split is the way.

---

## What is SOLID (the architecture + kernel pass)
- **No monolithic dt-collapse (original Pathology 2 GONE).** Every variant that ran did so with the stiff
  Richards solve never sharing a Jacobian with the surface routing. The sand storm under option (ii) ran to
  t=0.5 in 73–95 steps at min dt 1e-3 (the original demo collapsed/pinned at ~2e-5).
- **Convergent V routes cleanly during the storm (original Pathology 1 / sawtooth GONE).** Tilted-V storm
  (t=0→0.0625) under option (ii): it=2–3 per step, dt GREW 2e-4→2.1e-3 monotonically, **no sawtooth, no dt-pin.**
- **Routing kernel is exact + positive + non-teleporting (PASS, isolation + in-situ):**
  - random-field 50-sweep `Σd_iA_i` invariant **resid 0.0** (full precision); min routed depth **0.0** (F3).
  - in every coupled run the per-step routing `|Σd_iA_i after−before|` held **≤4.4e-16**.
  - NO-TELEPORT: a 10 cm pulse moves only **2.77%** to the far outlet in ONE step; **12 steps** to drain 50%.
    (Caveat: a single descending-head sweep DOES cascade several cells per step — the 2.77% is that cascade
    reaching the edge — but it is Manning-cap-bounded, not a teleport.)
- **Interception forms (no teleportation, gate G3 PASS):** sand-channel option (ii) — of the runoff/conveyance
  the **channel captured 38–48%** (surface outlet + sand-GHB), ~52–61% escaped to the toe. Meaningful capture ⇒
  the rate cap let runoff pool in the swale and be intercepted.

## What is the WALL (F1 — the water-level handoff)
Three handoff mechanisms tried; each fails a DIFFERENT way (precise map for Part B):

| F1 option | mechanism | sand-channel | tilted-V | why it fails |
|---|---|---|---|---|
| **(i) co-located** | route `d=max(ψ,0)`, OVERWRITE `ψ_top←d_routed`, `ψ_n←ψ` | **RED collapse t≈0.029–0.033** (≈20 rejects, dt<1e-9; reproduced twice) | (not run) | the routing-perturbed top is a bad Newton restart for the next stiff solve — solver fragility (not a leak). The §7 ghost in BC form. |
| **(ii) separate store** | bounded Dirichlet ponded head `ψ_top=d_pre`; infiltration `=Δsoil+drainage` capped at store, remainder = tracked handoff | runs t=0→0.5, **books close (NET ~0)** but **handoff = ~54% of rain** | storm clean; **recession stalls t=0.0625** (SNES −5, fnorm grows as dt shrinks) | the hard Dirichlet pin **over-infiltrates the high-K sand** (K·dt > the thin store/step → real over-draw); at recession the un-pinned saturated nodes give `(θ−θ_n)/dt` stiffness |
| **(iii) hybrid** | `add_ponding_bc` (self-limiting, NO over-infiltration) + lateral routing injected as a per-node Neumann SOURCE | **RED collapse t=0.014** | **RED collapse t=0.039** | the lateral-source flux is too stiff once redistribution builds (convergence line / berm pile-up) |

**(iv) ITERATIVE (Picard + under-relaxed source) — the breakthrough on ROBUSTNESS.** Wrapping the option-(iii)
hybrid in a Picard loop that holds the lateral source fixed within each Richards solve and UNDER-RELAXES it
(omega=0.5, halve on a failed inner solve) **runs the sand storm to t=0.5 with NO collapse** (60 steps, 39 s, min
dt 1e-3; the single-pass hybrid died at t=0.014). Routing `Σd_iA_i` resid **0.0**. This is the plan's named
upgrade and it WORKS for robustness. **BUT** global `|bal|/cum_rain = 17.9%` — a conservation LEAK (≈ exactly
cum_outflow), diagnosed as: the lateral source injected on the ψ-carried pond can be satisfied by the soil
(infiltration reversing to feed the "outflow") rather than by removing SURFACE ponding, so the booked outflow is
partly sourced from soil storage. **This is fundamental to the HYBRID (ψ carries the pond → a source can't tell
surface from soil water), NOT to the iterative idea.**

**Net F1 read (the productionization design the spike pins down):** the winning design is **(a) a SEPARATE surface
store distinct from soil** (so outflow provably comes from surface water — option (ii)'s data model) **+ (b) a
SELF-LIMITING ponding BC that READS that store** (so no hard-pin over-infiltration — the F1(b) "new/modified
Richards top-BC" the plan flagged as real code, NOT plumbing) **+ (c) an ITERATIVE (Picard, under-relaxed)
coupling** (so it's robust — proven above). The three single-pass handoffs each break one of these legs:
overwrite is fragile, a flux source is stiff AND mis-attributes outflow, a hard pin over-infiltrates. NOT a
return to the monolith.

## Foundational items
- **F1 (surface state):** decided by elimination above — a SEPARATE store is required (option (i) co-located
  collapses), but the store→soil handoff must be SELF-LIMITING (option (ii)'s hard pin over-infiltrates) AND
  fed back conservatively without a stiff source (option (iii) is unstable). ⇒ **separate store + Picard handoff.**
- **F2 (sink eval state):** subsurface sinks (GHB) inside the Richards solve; surface sinks (Manning outlet)
  during the routing sweep on the post-Richards field. Books close under this split (NET balance machine-zero in
  option (ii)) — the choice itself is validated; it is the handoff MAGNITUDE that's the issue, not the sink wiring.
- **F3 (positivity):** routed depth ≥0 always (kernel min 0.0); confirmed the plan's warning that depth≥0 does
  NOT certify the coupled state — the global balance is the real check (and it exposed the (ii) over-infiltration).

## Routing rate law / W_i (the load-bearing cap)
Per top node i with `d_i>0`, surface head `H_i=z_b,i+d_i`: downslope receivers j (`H_j<H_i`), MFD weight
`w_j=√slope_ij`, `slope_ij=(H_i−H_j)/L_ij`, normalized; PLUS an off-domain outlet pseudo-receiver at boundary
outlets (weight `√outlet_slope`). Manning volumetric cap over dt:
`Vcap_i = (SECONDS_PER_DAY/n)·d_i^(5/3)·√S_i·W_i·dt`, `S_i`=max receiver slope; `V_out_i=min(d_i·A_i, Vcap_i)`,
distributed by MFD weight. **`W_i = Σ_{edges e∋i} T_e·L_e`** (cotangent transmissibility × edge length = the
dual-mesh face length the node carries — a geometry-derived width [m], NO tuning constant). Nodes processed in
descending head order (one richdem-style sweep). `route_excess` / `node_widths` in the script.

## Engineering findings that RESHAPE Part B (surprises)
1. **Quadrature cap is mandatory.** `RichardsProblem` defaults to FFCX auto quadrature; on the van Genuchten
   sand/clay `conditional` the first 3-D solve took **54.6 s** → cap to degree 8 (`richards_bulk_residual(...,
   quadrature_degree=8)`) → **~1 s/solve**. (memory: `pids-fem-quadrature-degree-cap`.)
2. **Pressure-head BC = ponded DEPTH, not z_b+depth.** A spike bug pinned `ψ_top=z_b+d` (z_b = routing topography,
   up to ~9 m) instead of `ψ_top=d`. The mesh top is FLAT at z=ztop, so elevation is uniform and must NOT enter
   the pressure-head Dirichlet (z_b is for the routing surface head H=z_b+d ONLY). The bug caused catastrophic
   over-infiltration + recession stalls. Part B's option-(ii)-style BC must use depth.
3. **Recovery-capable BAND dt-controller required.** A naive `it≤3→grow, it≥8→shrink` SOFT-COLLAPSES dt
   (5e-4→3e-8) during the ponding-onset transient because `basic` Newton legitimately needs ~8–10 iters there.
   A band controller (`it≤4→×1.4`, `it≥12→×0.7`, else hold) with `ctrl_high` ABOVE the transient spike keeps dt
   stable. (This was a controller bug masquerading as collapse — distinct from the real F1 collapses.)
4. **`basic` linesearch (full Newton) for the standalone Richards path.** `bt`/`cp` STALL at SNES −5 on the
   ponding-onset step; `basic` settles to it=3. (CoupledProblem wires a different solver; the split reuses the
   raw `RichardsProblem` path, which wants `basic`.)
5. **RichardsProblem multi-BC clash.** `add_ponding_bc` + `add_drainage_bc` on one `RichardsProblem` trips
   DOLFINx 0.10's one-`subdomain_data`-per-integral-type rule (AssertionError). The orchestrator must assemble
   surface + GHB facet terms on ONE shared meshtags (top=1, drains=2+). Part of the B6 extraction, not a footnote.
6. **The near-impermeable thin-slab V SINGULARIZES** in our unconfined (no-Ss) Richards once the slab fully
   saturates (NZ=2/LZ=0.5/ψ=−0.1 stalled). A deep unsaturated buffer (LZ=2, NZ=6, ψ=−1) regularizes it. The V
   fixture for Part B regression should keep an unsaturated buffer (or document the singularity).

## Gate scorecard
| Gate | sand-channel | tilted-V |
|---|---|---|
| G1 no dt-collapse | ✅ **iterative/Picard ran to t=0.5, 60 steps, 39 s** (also opt ii) | ✅ storm (opt ii, it 2–3, dt grows); recession ❌ on opt ii's hard-pin (a hard-pin artifact, not the iterative scheme) |
| G2 conserve (routing 1e-12; global) | routing ✅ **0.0**; global **❌ leak**: opt-ii hard-pin handoff ~54%, Picard hybrid ~18% (the ψ-pond outflow-attribution leak) — the one open task | routing ✅; storm-only books close (opt ii) |
| G3 interception, no teleport | ✅ **24–48% captured** across variants | n/a (single outlet; storm routes to it) |
| G4 convergence clean (no sawtooth, sane dt) | n/a | ✅ storm: it 2–3, dt grows 2e-4→2.1e-3, no sawtooth |
> G1 & G3 & G4 (and routing-G2) PASS. The ONLY red is the global-G2 handoff conservation, isolated to the
> surface↔soil water attribution — closed by the separate-store + self-limiting-BC design above.

## How to reproduce
```
wsl bash -c 'cd .../forward-model && export PATH=".../pids-fem/bin:..." && \
  export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
  python -u scratch/overland_split_spike.py [kernel|probe|sand_i_short|sand_ii_short|sand_iii_short|v_ii|v_ii_storm|v_iii]'
```

## Recommendation for Part B
Proceed with the sequential split — architecture + routing kernel + iterative ROBUSTNESS are validated. The
FIRST Part-B task is the handoff data-model + conservation (everything else is done):
1. **Separate surface store `d`** (numpy/Function on the top facet, distinct from ψ) — so outflow provably leaves
   the SURFACE, not soil storage (the root of the hybrid's 18% leak). This is F1 option (b).
2. **A self-limiting ponding BC that reads `d`** — a new/modified `RichardsProblem` top BC (F1(b), a real code
   delta per the plan): when `d>0` the soil sees the ponded head and self-limits infiltration (no hard-pin
   over-infiltration, which gave option (ii)'s 54%); when dry, rain flux.
3. **Iterative (Picard, under-relaxed) coupling** around {self-limiting vertical solve, routing sweep} — proven
   here to give robustness (the sand storm completed). Track the bounded handoff imbalance as a first-class output.
4. Re-gate on BOTH cases (sand-channel + a NON-SINGULAR tilted-V), carrying the spike's mandatory plumbing: the
   **depth (not z_b+depth) pressure-head BC**, the **band dt-controller**, the **quadrature cap (degree 8)**, the
   **unified surface+GHB meshtags**, and `basic` linesearch.
The overwrite / flux-source / hard-pin SINGLE-PASS handoffs are dead ends (each documented above); the iterative
split is the path. Do NOT return to the monolith.

---

## CONSERVATION PROOF (extension, 2026-06-23)

**VERDICT: GREEN.** The one open Part-B gate — GLOBAL MASS CONSERVATION of the surface↔soil handoff — is CLOSED.
Design **#1 (fully DECOUPLED / non-monolithic)** conserves to **machine precision on BOTH cases incl. recession**.
The vertical Richards solve runs **ALONE** each iterate (NO co-solved [ψ,d] unknown) — **design #2 was NOT needed.**

### Hard-gate results (full precision, mode `win`)
| case | t_end | steps / rejects | routing Σd·A resid | **GLOBAL \|bal\|/cum_rain** | self-limit | soil-source | interception |
|---|---|---|---|---|---|---|---|
| sand-channel (storm+recession) | 0.5 | 69 / 0 | **0.0** | **5.275656e-12** | 4.66e-12 (OK) | +5.20e-3 (OK) | 22.9% captured |
| sand-channel FULL (storm+recession) | 1.2 | 103 / 0 | **0.0** | **5.115414e-12** | — | — | 27.6% captured |
| tilted-V (storm only) | 0.0625 | 35 / 0 | **0.0** | **1.567428e-11** | 7.06e-9 (OK) | +0.197 (OK) | 100% to outlet |
| tilted-V FULL (storm+**recession**) | 0.125 | 40 / 1 | **0.0** | **1.011026e-11** | — | — | 100% to outlet |

- **The 54% (option ii) / 18% (hybrid) leaks are GONE** — down to ~5e-12 (≪ the 1e-3 gate). No dt-collapse on any
  case (min dt = dt0 = 1e-3 sand / 2e-4 V; ≤1 reject). Routing Σd·A conserves at **0.0**. Interception forms on sand.
- **The prior hybrid's recession STALL (tilted-V, option ii) is RESOLVED** — the V recession ran clean (1 reject).

### The exact self-limiting mechanism used — DESIGN #1 (decoupled), NOT #2 (vertical co-solve)
The surface store **IS the pond `max(ψ,0)` carried IN ψ continuously** (NOT a separate numpy store). The soil
draws the carried pond at its natural Darcy rate via the pond-storage term (`rain_eff = I + d(pond)/dt`) — genuinely
**self-limiting** (the soil takes only what it can absorb; the un-infiltrated remainder raises the pond). This is the
validated `add_ponding_bc` physics. Lateral routing enters as an **in-solve Neumann SOURCE** (run-on +, run-off −),
**under-relaxed (ω=0.5) + lagged** — the proven-robust hybrid handoff (Picard; halve ω before cutting dt).

**The breakthrough: the prior hybrid's "~18% leak" was almost entirely an ACCOUNTING bug, not a physics leak.** Two
fixes — and NOTHING in the robust vertical solve changed:
- **FIX 1 — the conserved ledger must INCLUDE the surface pond:** `total = ∫θ + ∫max(ψ,0) ds_top`. The prior hybrid
  measured `∫θ` only. The surface pond is a real stored quantity NOT in `∫θ` (θ is FLAT at θ_s for ψ≥h_s, so a ponded
  node's pond DEPTH never enters the volume integral — the boundary pond-storage term carries it). Omitting it made the
  residual pond look like a leak. (Adding only FIX 1 took the sand-short bal/rain 18% → **1.9e-3**.)
- **FIX 2 — quadrature match:** the pond-storage / rain / lateral-source terms MUST use the SAME lumped **vertex**
  quadrature as the ledger and the routing store's `Σd_i A_i`. A `degree-8` `ds` integrates the non-polynomial
  `max(ψ,0)` differently from the vertex-lumped ledger → an O(1e-3) gap between *what the solver conserves* and *what we
  measure*. (Adding FIX 2 took the sand-short bal/rain 1.9e-3 → **5.3e-12**.) Verified `∫max(ψ,0)ds_top` ≡ `Σd_i A_i`
  to ~1e-15.

**CONSERVATION IS ω-INDEPENDENT** (key structural fact): the converged residual gives, per step,
`d(∫θ + ∫pond) = rain·area·dt + (Σ lat_i A_i)·dt − drain·dt = rain − ω·outflow − drain`, and we book
`cum_outflow += ω·outflow` → the balance closes to solver tolerance for ANY ω. ω only sets the lateral transport RATE
(a physics/accuracy knob), never whether mass conserves. The un-applied (1−ω) routing is a bounded lag in ψ's pond
(in the ledger), NOT a leak.

**Did #1 (decoupled) work, or did ONLY #2 (vertical co-solve) conserve?** → **#1 (decoupled) WORKS.** The vertical
exchange did NOT need to be implicitly co-solved; only the lateral routing splits out (as the lagged source). This
resolves the brief's key architectural question in favour of the fully-decoupled design.

### Why the carried-pond-in-ψ is robust where the SEPARATE-STORE write-backs were NOT (dead ends found here)
Keeping the pond INSIDE the solve avoids both failure modes that the brief's literal "separate store + write-back"
phrasing hits (both were tried and FAILED before landing on the above):
- **A separate numpy store written BACK into ψ's positive part** (even θ-neutral, under-relaxed): the routed pond as a
  post-solve ψ_n edit is a bad Newton restart → the swale pond **ballooned to ~1 m and dt-COLLAPSED** (~7e-9) mid-storm.
- **Re-offering the carried pond as a `d_old/dt` FLUX** (so ψ never persistently carries it): numerically violent on
  clay (offered ≫ Ks → instant saturation / no-Ss elliptic stall) → **dt-collapse at t≈0.014.**
The fix is to let the pond live in ψ and evolve smoothly through the solve (the soil meters it out at its Darcy rate),
which is exactly the hybrid skeleton — so "separate store" is realized as **a separate-quadrature LEDGER over ψ's pond**,
not a separate state variable. (NB: the function/docstring `run_case_win` is named for design #1; its body is this
pond-in-ψ realization.)

### Adversarial — the balance GENUINELY closes (not absorbed, not a tautology)
- **No handoff bucket.** `bal = (∫θ+∫pond)_end − _start − (rain − outflow − drainage)`, every term measured
  independently. Nothing is credited/absorbed to make it look closed.
- **Self-limit (brief (b)):** per step, realized infiltration `Δθ+drainage ≤ available = (rain+run_on)·dt + entry_pond`.
  Max violation ≤ roundoff on both cases → the soil NEVER over-infiltrates (kills option (ii)'s 54%).
- **Soil-source:** the soil store **GAINS** water every storm step (min per-step `d(soil)` = +5.20e-3 sand / +0.197 V,
  both >0) → the booked outflow CANNOT have been sourced from soil storage (the brief's specific 18%-leak mechanism).
- **Falsification hook (`PIDS_WIN_LEAK`):** deliberately mis-booking outflow by 10% makes the balance report
  **bal/cum_rain = 2.132308e-2** = exactly 10% of `cum_outflow` (0.3838) over rain (1.8). The ~5e-12 close is therefore
  a genuine detector — a real leak shows up immediately at full magnitude, not hidden.

### Plumbing carried (all 6 from the note above)
depth-not-`z_b` pressure BC (pond is `max(ψ,0)`, no elevation) · band dt-controller (`ctrl_low=4`/`ctrl_high=12`) ·
`quadrature_degree=8` Darcy cap · unified surface(=1)+GHB(=2+) meshtags · `basic` linesearch · non-singular deep
unsaturated buffer on the V. PLUS the new load-bearing item: **the surface pond-storage/rain/source terms + the pond
ledger share VERTEX quadrature** (FIX 2).

### How to reproduce (new mode `win`)
```
wsl bash -c 'cd .../forward-model && export PATH=".../pids-fem/bin:..." && \
  export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
  python -u scratch/overland_split_spike.py [win_sand_short|win_sand|win_v_storm|win_v]'
# falsification (must report ~10% imbalance): prefix  PIDS_WIN_LEAK=0.1
```
`win_sand` = sand storm to t=1.2 (storm+recession); `win_v` = tilted-V to t=0.125 (storm+**recession**); the `_short`/
`_storm` variants are the faster legs. Driver = `run_case_win` (store="win" in `sand_channel_case`/`tilted_v_case`).

### Remaining risks for Part B (production extraction)
1. **Lateral transport ACCURACY vs the lag.** Conservation is ω-independent, but ω=0.5 + one-step lag throttles the
   lateral RATE. At the convergent swale a transient ~1 m pond builds before draining (sand t≈1.16) — robust + conserving,
   but the discharge *timing* is lag-dependent. Part B should pick ω / Picard-iters by a transport-accuracy criterion
   (vs a resolved upwind reference), NOT by conservation (which is already exact). Multiple Picard sweeps to a fixed
   point (here ω is applied per-step, not iterated to convergence) would tighten the lag.
2. **Outflow under-relaxation booking.** `cum_outflow += ω·outflow` is exact for the applied move, but the (1−ω) deferred
   routing means at any instant some outflow is "in-flight" in ψ's pond. Over a run it all exits (ledger conserves); for
   a *hydrograph* the booked outflow is slightly lagged vs a fully-applied routing — fine for water-balance, watch for
   timing-sensitive uses.
3. **Per-node self-limit is only checked globally.** `infil_step ≤ avail_step` is verified in aggregate; a node-level
   over-draw compensated elsewhere would pass. The global balance + soil-source check make this very unlikely, but a
   strict production version could enforce the cap per-column.
4. **Quadrature-match is load-bearing and easy to lose.** If the B6 extraction ever puts the pond-storage term on a
   different (e.g. degree-8) measure than the ledger/`A_i`, the O(1e-3) gap returns silently. Pin it with a test
   (`∫max(ψ,0)ds_top == Σd_i A_i`) in the package.
5. **Serial-only** (top-facet graph not ownership-aware) — same constraint as the rest of the overland work.
6. **`max_steps` / cost:** all gate legs ran in 18–58 s; no pathological step counts. Not a risk at these scales, but the
   convergent V at the canonical 1.62 km (stiffer) was not re-run here.
