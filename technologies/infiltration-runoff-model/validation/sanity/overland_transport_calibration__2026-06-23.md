# Sequential overland: lateral-TRANSPORT accuracy + omega / routing-substep calibration (2026-06-23)

> Focused ACCURACY VALIDATION (investigation) for `SequentialCoupledProblem`
> (`pids_forward/physics/sequential_coupling.py`). The scheme conserves mass exactly (~5e-12,
> omega-INDEPENDENT) but its LATERAL TRANSPORT RATE is APPROXIMATED: an under-relaxed (omega~=0.5)
> one-step-lagged Manning routing SWEEP (`overland_routing.route_excess`). Question: is the transport
> reasonably accurate vs the resolved reference, or artificially throttled — and what omega / routing
> sub-step count best matches the resolved transport?
> Reference = `CoupledProblem(overland_scheme="upwind")` (the validated monotone Manning diffusion-wave
> solver), run on a NON-stiff loam + resolved swale where BOTH schemes converge.
> Harness: `forward-model/scratch/overland_transport_calib.py`. DOLFINx 0.10 / WSL `pids-fem`, serial.
> NOT committed to the package. All numbers full precision; adversarially matched + confounder-checked.

## VERDICT (one line)
**The default omega=0.5 + single routing sweep per step ARTIFICIALLY THROTTLES lateral transport by
~40-50x vs the resolved upwind reference** (clean pond-release isolation). It still CONSERVES and water
eventually ends up in the right place, but it arrives FAR too slowly and piles up a large transient
swale pond (reproduces the spike's pathology, incl. a dt-collapse at omega=0.3 on loam run-on). **The
FIX is ROUTING SUB-STEPS, not omega**: omega alone (0.5->1.0) buys only ~2.4x; sub-stepping the explicit
Manning sweep `route_substeps` per Richards step buys ~4x per doubling and CLOSES the gap. **RECOMMENDED
DEFAULT: omega=1.0 + route_substeps=4** (matches the reference drain timing to ~1.0x in the clean
isolation). This is a transport-RATE knob only — conservation is omega/substep-independent at ~1e-12
throughout.

Against PIDS's MODEST bar ("water ends up in roughly the right places at roughly the right times for
run-on infiltration"): the default (omega=0.5, rs=1) is a **documented LIMITATION** — the >40x lag is too
slow for the run-on timing and it builds an unphysically large transient pond. With omega=1.0 +
route_substeps=4 the transport is ACCEPTABLE (≈ the resolved reference).

---

## The two confounders, isolated first (this is why the headline uses a POND-RELEASE)
A naive loam head-to-head is dominated by the **infiltration-handling difference**, NOT transport:
- The sequential scheme carries the pond IN psi (`max(psi,0)`) and infiltrates it with the validated
  self-limiting `add_ponding_bc` physics (Darcy at the matric gradient). On a dry/low-K loam this draws
  a LARGE sorptive flux (>> Ks transiently) — so it OVER-infiltrates and barely ponds.
- The upwind/monolithic scheme infiltrates via the Kirchhoff matric-flux-potential FILM `q_pot =
  kirchhoff(psi,d)/ell_c` — a film-throttled rate that ponds far more.
- **Measured** (loam Ks=0.25, rain=2.0, short storm): upwind ponds surf_peak 206 m^3 / 26mm and runs off
  90%; sequential ponds surf_peak 6.6 m^3 / 0.3mm and runs off 0.5%. That ~30x pond difference is
  INFILTRATION physics, not transport — it would swamp any transport read.
- A NEAR-IMPERMEABLE bed (Ks=1e-3) does NOT isolate transport either: from a dry start the sequential
  pond-in-psi still pours ALL the rain into the dry soil's sorptive capacity (388.8 m^3 dsoil, surf=0.0)
  while the upwind film throttles it to 99% runoff. (And a near-saturated near-impermeable start
  dt-collapses on the no-Ss singularity.)

=> To isolate LATERAL TRANSPORT cleanly we use a **POND RELEASE over an ALREADY-SATURATED column**
(hydrostatic, water table at the surface), NO rain: infiltration is ~shut off IDENTICALLY in both
schemes (the soil is full; a small Ks=0.05 leak is the same small leak both ways), both get the SAME
initial surface water (POND*top_area), and the pond just ROUTES downslope to the outlet — a pure
lateral-transport race. (A deep buffer dodges the no-Ss saturation singularity.)

## The CASE (reproducible)
Geometry (a RESOLVED tilted-V swale; W>0 avoids the measure-zero-kink artifact so upwind is clean):
- box `200 x 120 x 1.5 m`, mesh `20 x 16 x 3` (P1); `z_b = SY*(LY-y) + SX*max(|x-XC|-W/2, 0)`,
  XC=100, cross-slope SX=0.05, down-slope SY=0.02, resolved floor width **W=50 m** (~5 cells across).
- soil (transport-isolation): loam `VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56,
  Ks=0.05)`; n_man=0.05; outlet = the y=LY edge, slope=SY.
- **Regime P (pond release)**: IC = saturated hydrostatic column + uniform pond. upwind: psi=(H-z)
  (psi_top=0, saturated) + d_value=pond (pond in the SEPARATE store); sequential: psi=pond+(H-z)
  (pond carried in psi). pond=0.05 m, no rain, t_end=0.10/0.30. Both get surf0 = pond*area = 1200 m^3.
- Reproduce: `PYTHONPATH=. python scratch/overland_transport_calib.py P` (sweep) /
  `... Psub` (route_substeps convergence) / `... R` (loam run-on, the pathology demo) /
  `... R2` (steady-rain pile-up isolation).

## HEAD-TO-HEAD — Regime P pond release (the clean transport isolation)
Metric = `t_surf50` = time [day] for the surface store to half-drain (the lateral drain race). The
upwind reference drains a 50 mm pond on the SY=0.02 slope in **t_surf50 = 0.00156 day** (consistent with
the analytic Manning travel time ~LY/v, v=(1/n)d^(2/3)sqrt(S); the surf(t) trace falls smoothly over
~14 records, NOT a one-step jump — the read is real). All sequential variants conserve (|balance| ~1e-12)
and eventually export ~100% of the pond — purely a RATE/timing difference.

THROTTLE FACTOR = sequential t_surf50 / reference t_surf50  (1.0 = matches the resolved reference):

| route_substeps | omega=0.5 | omega=1.0 |
|---:|---:|---:|
| **1** (current default) | **46.7x** | 19.8x |
| 2 | 18.5x | 5.7x |
| 4 | 5.7x | **1.0x** |
| 8 | 1.0x | 0.5x (overshoots: drains FASTER than ref) |
| 16 | 0.7x | 0.3x (overshoots) |

Reading:
- **The transport IS throttled.** Default (omega=0.5, 1 sweep) is ~47x too slow.
- **omega is a WEAK lever**: 0.5 -> 1.0 at rs=1 only goes 46.7x -> 19.8x (~2.4x). It cannot reach 1.0x.
- **Routing sub-steps are the DOMINANT lever**: ~4x per doubling. omega=1.0 + rs=4 hits 1.0x (matches the
  resolved diffusion-wave drain rate). omega=0.5 needs rs=8 to reach 1.0x.
- **There is a tunable OPTIMUM, not "more is always better":** rs=8/16 OVERSHOOT (the pond drains FASTER
  than the reference). MECHANISM: the reference solves the Manning DIFFUSION-wave implicitly (some
  front attenuation); a fully sub-stepped explicit Manning sweep is a KINEMATIC wave that, once the
  intra-step travel is resolved, outruns the diffusion-wave. So sub-stepping past the match introduces a
  (mild) opposite error. omega=1.0 + rs=4 sits right at the match.

### Drain curves at MATCHED times (the single clearest picture; mode `Pcurve`)
Surface store REMAINING (% of the initial 50 mm pond), no rain, both schemes same release:

| t [day] | upwind ref | default omega=0.5 rs=1 | recommended omega=1.0 rs=4 |
|---:|---:|---:|---:|
| 0.0010 | 67.3 | 90.8 | 54.9 |
| 0.0020 | 40.3 | 88.6 | 40.0 |
| 0.0080 |  7.1 | 84.1 |  8.4 |
| 0.0160 |  2.4 | 79.7 |  0.2 |
| 0.0300 |  0.9 | 72.1 |  0.1 |
| 0.1200 |  0.1 | **24.2** |  0.0 |

The default leaves **24% of the pond still in the swale at t=0.12** (the reference is 99.9% drained by
t=0.03). The recommended omega=1.0 + rs=4 TRACKS the reference across the whole curve (40.0 vs 40.3 at
t=0.002; 8.4 vs 7.1 at t=0.008) — "right places at right times". This is the run-on-relevant statement.

MECHANISM of the throttle: a single `route_excess` sweep advances the descending-head cascade only a
bounded distance per Richards step (Manning-cap-limited; ~one-to-a-few cells), and omega then applies
only a FRACTION of even that. But the Richards dt is set by the ponding/Richards stiffness (small),
within which the surface water could physically travel several cells. One sweep UNDER-resolves that
intra-step travel; `route_substeps` sub-divides the sweep (nsub sweeps each over dt/nsub, receivers
recomputed from the live depth) so the cascade marches the full intra-step distance — a standard
explicit-routing temporal-resolution (CFL) fix. Conservation is untouched (each sub-sweep telescopes;
`cum_outflow += omega*outflow` still books the applied move exactly).

## REALISTIC pathology demo — Regime R loam run-on (with the infiltration confounder present)
loam Ks=0.25, rain=4.0 (>>Ks), storm 0.03 day, t_end=0.20; both schemes, same geometry/mesh/forcing.
This is NOT a clean transport isolation (the infiltration confounder is live) but it shows the
real-world consequence of the throttle:

| run | runoff % rain | surf_peak [m^3] | centerline peak depth | outcome |
|---|---:|---:|---:|---|
| upwind reference | 95.4% | 319.6 | 2.0 mm | clean, 265 steps, 0 rejects |
| seq omega=0.3 (rs=1) | 12.8% | **1585** (5.0x ref) | **354.5 mm** (~0.35 m) | **dt-COLLAPSE at t=0.055** |

The throttled routing cannot drain the swale as fast as rain fills it, so the pond PILES UP to ~0.35 m
(vs the reference's 2 mm) and the recession dt-collapsed — exactly the spike's "transient ~1 m swale pond
builds before draining" + collapse pathology. (Confounder caveat: the sequential scheme ALSO
over-infiltrates here — 87% vs 4.6% — so the runoff fractions are not a transport read; the surf_peak /
peak-DEPTH pile-up is the transport signal, and it is unambiguous.) Higher omega / sub-stepping relieve
the pile-up (it is the same lateral-rate knob); see Regime R2 below.

## ORTHOGONAL limitation surfaced (not the transport question): near-saturated IC singularity
A steady-rain pile-up attempt over a NEAR-SATURATED start (`psi_top0 = -0.05`, SAT soil, mode `R2`)
dt-COLLAPSED the sequential scheme at t~0.001 BEFORE any ponding (every omega/substep variant), while
the upwind reference ran clean (140 m^3 / 27.7 mm pond, 98.4% runoff, 0 rejects). This is the known
no-Ss near-saturation singularity of the standalone-Richards path (the pond must be carried as a
COMFORTABLY-POSITIVE head — as in the pond-release psi_top=+0.05 — not a near-zero one), NOT the
transport throttle. It is orthogonal to this study but worth flagging: the sequential scheme is fragile
when the column sits right at saturation with ~zero pond. (The clean Regime-P isolation sidesteps it by
releasing a positive pond.)

## RECOMMENDATION
- **Default: `relax`(omega) = 1.0 and route the sweep with `route_substeps = 4` per Richards step.**
  In the clean isolation this matches the resolved upwind transport timing (1.0x). omega=1.0 (no
  under-relaxation) is the right transport default; under-relaxation should be the ROBUSTNESS fallback
  the step() already does (halve omega on a failed inner solve), NOT the standing transport setting.
- `route_substeps` is NOT yet a class parameter — it was prototyped here by monkeypatching `_route` to
  run nsub Manning sweeps each over dt/nsub (faithful + conservative; the source still =
  omega*(d_routed-d_cur)/dt over the whole dt and outflow accumulates across sub-sweeps). Productionizing
  it = a small `route_substeps: int = 4` ctor arg threaded into `_route` (or `step`). LOW risk
  (conservation is structural; it only refines the intra-step routing travel).
- Do NOT push route_substeps high (>=8 here) as a blanket setting — it overshoots into faster-than-
  diffusion-wave kinematic transport (rs=8 -> 0.5x in both dt runs). ~4 is the match.
- **dt-robustness checked** (mode `Psub` with `DT_MAX=5e-4` vs default 2e-3, a 4x range): the SINGLE-sweep
  throttle scales with dt (omega=1.0 rs=1: 19.8x at dt_max=2e-3 -> 6.1x at dt_max=5e-4 — bigger steps need
  more sub-steps), BUT the omega=1.0 **rs=4 optimum held at 1.0x in BOTH** (and rs=8 overshot to 0.5x in
  both). So rs=4 is not a knife-edge tuned to one dt — it tracks the reference across the tested dt range.
  (The reason it is not a naive linear-Courant scaling: each sub-sweep's Manning cascade reach also
  depends on dt/nsub, so the per-step routing distance and the needed nsub move together.)

## Adversarial notes / confounders (honest)
- **Infiltration confounder controlled, not ignored.** The headline (Regime P) is a pond release over a
  saturated column with NO rain, so both schemes' infiltration is ~off and equal — the ONLY thing
  compared is lateral routing. The loam run-on (Regime R) is reported separately and explicitly flagged
  as confounded (the sequential scheme over-infiltrates there); its transport signal is the pile-up
  DEPTH, which the infiltration difference cannot fake (more infiltration would LOWER the pond, yet the
  throttled pond is 175x deeper than the reference).
- **The reference is the designated resolved transport oracle** (`overland_scheme="upwind"`, the
  validated monotone diffusion-wave). Its pond-drain rate matches the analytic Manning travel scale, so
  it is a sound ruler. It dt-collapses on stiff clay (the reason the sequential scheme exists) — hence
  the loam + resolved-swale matching, where it runs clean (0 rejects).
- **Conservation never moved**: |balance|/... ~ 1e-12 for every omega and every route_substeps (the
  omega-independent proof holds; sub-stepping telescopes the same way). This is purely an accuracy knob.
- **Diffusion vs kinematic.** The match at rs=4 is a kinematic-routing approximation to a diffusion-wave;
  it agrees on the integral drain TIMING here. For sharp-front DEPTH-profile fidelity the upwind scheme
  is still the better physics — but that exceeds PIDS's run-on bar (we want water in roughly the right
  place at roughly the right time, which rs=4 delivers).

## Reproduce
```
wsl bash -c 'cd .../forward-model && export PATH=".../pids-fem/bin:..." && \
  export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
  python -u scratch/overland_transport_calib.py [P|Psub|Pcurve|R|R2|smoke]'
```
`P` = pond-release omega sweep + route-substep variants; `Psub` = the route_substeps convergence table
(the throttle-factor headline); `Pcurve` = the matched-time drain curves (ref vs default vs recommended);
`R` = loam run-on pathology demo (pile-up + collapse); `R2` = near-saturated steady-rain (surfaces the
orthogonal singularity); `smoke` = tiny both-schemes sanity (exposes the infiltration confounder).
Each metric helper (`pondrelease_metrics`, `transport_metrics`) + driver is in the harness; the
`route_substeps` knob is prototyped by `_install_route_substeps` (monkeypatch of `_route`).
