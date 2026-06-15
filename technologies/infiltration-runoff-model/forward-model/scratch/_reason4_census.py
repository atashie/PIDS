"""P1-A1: reason-4 census -> evaluate candidate residual-scale (``R_scale``) forms + ``tol_rel``.

Plan: docs/plans/2026-06-14-overland-convergent-flow-P1.md Part A, Task A1 (parent
docs/plans/2026-06-11-overland-convergent-flow-stabilization.md §5 P1 prerequisite).

WHY THIS EXISTS
---------------
``OverlandProblem``/``CoupledProblem``.step() book a PETSc CONVERGED_SNORM_RELATIVE (reason 4 =
"the iterate stopped moving") step only if ``|F|2 <= stall_accept_fnorm`` (an ABSOLUTE 3e-6 bar).
That bar is NON-PORTABLE: ||F||2 scales with the mesh row-count AND the per-row flux/measure, so
the value separating *floor* reason-4 states (MMS, near-flat, lake -- legitimate stagnation at the
residual floor, must ACCEPT) from *dirty* reason-4 stalls (the stiff convergent V -- a stalled line
search, must REJECT) is calibrated to today's problems only. A2/A3 were to replace it with a
SCALE-INVARIANT residual gate ``accept reason-4 iff ||F|| <= tol_rel * R_scale``.

This census MEASURES ||F|| and the candidate normalizers across EVERY reason-4 state in the suite +
the V at two domain scales, so the R_scale form + tol_rel are picked from DATA, not guesswork.

WHAT WAS MEASURED (norms, all via assemble_vector(form(...)).norm())
-------------------------------------------------------------------
- ||F||         = the engine's recomputed residual norm at the returned reason-4 iterate (last_fnorm).
- ||b_forcing|| = || (source + rain) * v * dx ||  (MMS source f*; overland rain r). 0 for no-forcing.
- ||b_storage|| = || ((d - d_n)/dt) * v * dx_storage ||  at the candidate (post-solve) state.
- ||b_conv||    = || K_s(d) grad(H_s).grad(v) * dx ||  the assembled Manning conveyance-flux residual.
- ||F0||        = the residual norm at the step's START (d = d_n) -- the Newton's initial residual.
Four candidate R_scale forms are scored for floor/dirty separation:
  (1) max(||b_forcing||, ||b_storage||, atol)                      [the plan's leading candidate]
  (2) max(||b_forcing||, ||b_storage||, ||b_conv||, atol)          [the plan's conveyance escape hatch]
  (3) ||F0||                       -> the relative-reduction gate ||F||/||F0||
  (4) ||b_forcing|| + ||b_storage|| + ||b_conv||                   -> sum-of-term-magnitudes

REASON-4 SOURCING (per the task: drive each onto reason 4; forcing it where allowed is documented)
  - MMS spatial steady (nc=20,40,80): reason 4 NATURALLY (starts at the exact solution; step(1e8)).
  - MMS temporal (1 step): forced to reason 4 (FORCE4 = rtol/atol 1e-30 so only SNORM can exit).
  - near-flat 1-D / lake-at-rest: forced to reason 4 (FORCE4); both are reason 2/3 under defaults
    (they balance), so their reason-4 entry is ARTIFICIAL -- recorded to bound the floor population.
  - floor-stagnation (near-flat bed, uniform start, n=0.05): reason 4 NATURALLY under default options
    -- THE production floor case the gate must accept (cf. tests/test_step_acceptance.py).
  - 2-D convergent V at scale 1.0 AND 0.1: the WORST reason-4 stall over the storm window (highest
    ||F|| AND highest ||F||/R captured separately) -- the genuine dirty population. To keep prob.d at
    the stalled iterate for the storage/conv assembly (step() reverts d->d_n on a REJECTED dirty
    stall), the V probe sets stall_accept_fnorm=+inf so reason-4 is BOOKED -- this changes only which
    state we read, not the measured ||F|| (recomputed at the returned iterate either way).
  - coupled column under rain: a WORKING state (reason 2/3, not 4) -- recorded to document a healthy
    block step's ratio + verify the coupled forcing/storage assembly.

================================================================================
RESULT / CHOSEN CONSTANTS (read from the table below, 2026-06-14):

  ** NO tested R_scale form separates the floor and dirty reason-4 populations. **

  Across all FOUR candidate forms the populations INTERLEAVE -- the [separation] block prints the
  exact gaps; every form has min(dirty ratio) < max(floor ratio), broken by the SAME near-rest
  legitimate floor states (near-flat, floor-stagnation, lake). Representative (committed table):

    form                          max_floor (offender)        min_dirty           verdict
    R1 max(bforce,bstore,atol)    5.9e-5  (lake)              6.1e-10 (V s=1.0)   INTERLEAVED
    R2 + conveyance               5.9e-5  (lake)              1.2e-10 (V s=0.1)   INTERLEAVED
    F/F0 relative reduction       1.1e-9  (lake)/2.2e-10 (fs) 1.5e-10 (V s=0.1)   INTERLEAVED (closest)
    sum bforce+bstore+bconv       1.0e+0  (lake)              1.2e-10 (V s=0.1)   INTERLEAVED

  WHY: the near-rest floor states have normalizers (storage / conveyance / ||F0||) that COLLAPSE
  toward 0 as they approach rest -- near-flat ||bstore||~1, floor-stag ~4e-3, lake ~0 -- so dividing
  their tiny floor ||F|| (1e-10..1e-17) by a tiny normalizer gives a NOT-tiny ratio (1e-10..1e-5).
  The dirty-V stalls have HUGE normalizers (||F0||,||bconv||,wave-front ||bstore|| ~1e4-1e6) that
  DEFLATE their ||F||/R to look pristine (down to ~1e-10..1e-12). So a floor state lands ABOVE a
  dirty one on every normalized axis.

  Root cause (corroborates parent §8.4): the B6 dirty-V reason-4 stall is NOT a residual that failed
  to reduce -- at the settled plateau it reduces ~12 orders (||F||/||F0|| ~ 8e-12; even the WORST
  stall over the storm reduces to F/F0 ~ 1.5e-10..2e-10, see the V rows). The converged STATE is
  non-physical (the unstabilized-Galerkin sawtooth, outflow 0.785*Q_eq), but the RESIDUAL is genuinely
  small relative to every problem-natural scale. The V's error is a DISCRETIZATION error (Defect A),
  not an unbalanced residual -- so NO residual-normalized quantity flags it as "dirty": by the residual
  metric it is converged. A scale-invariant residual gate would ACCEPT these stalls, the opposite of
  the goal of rejecting them.

  => tol_rel: there is NO value that accepts all floors AND rejects both V scales (see [separation]).
     But that "reject the V" goal rests on the V being DIRTY -- and the data says it is NOT, by the
     residual/mass metric. TWO readings, both honest, the architect (Gate A) must pick:

     READING 1 -- goal = "reject the dirty V" (the task's literal floor-vs-dirty framing):
       FAILS. No residual-normalized gate can reject the V stalls -- they are residual-converged
       (F/F0 ~ 1e-10), so any ||F||<=tol_rel*R_scale that accepts the floors also accepts the V.
       Under this goal the floor-vs-dirty distinction is simply NOT in the (normalized) residual;
       it would need a different discriminant (e.g. an a-posteriori solution-quality / monotonicity
       check), which is out of scope for a step-acceptance gate -- O1 (Part B) is the real fix.

     READING 2 -- goal = "accept iff residual-converged + mass-safe" (what parent §8.1/§8.4 already
       established: the V's published 20% 'leak' was RETRACTED; booking the reason-4 V stalls injects
       <=1e-6 m^3 total, books close; the V's real error is the converged SOLUTION's accuracy = Defect
       A, fixed by O1, NOT the books): then ACCEPTING the V stalls is CORRECT and DESIRABLE -- it is
       exactly the A5 efficiency hypothesis (the absolute 3e-6 bar REJECTS the V's ||F||~7e-4..6e-6
       and costs 39.5 h / 60k rejections, §8.4; a residual gate accepts them and lifts the pin). The
       only stalls a residual gate must reject are GENUINELY large-residual ones -- and NONE of the
       census V steps qualify (worst F/R1 = 5.9e-9, F/F0 = 2e-10, all converged). So a residual gate
       is sound for portability + efficiency; it just does NOT also serve as a V-accuracy guard.

     What the census DID nail down for A2/A3 if Reading 2 is chosen:
       - R_scale = max(||b_forcing||, ||b_storage||, atol_floor) is the right forcing/storage scale
         (the conveyance term adds nothing and the sum/F0 variants do not help) -- BUT it must be
         combined with an ABSOLUTE floor for the truly-at-rest case: lake-at-rest has ||F||=5.9e-17
         with ||b_forcing||=||b_storage||=0, so R collapses to atol and F/R=5.9e-5. Either raise
         atol_floor so quiescent ||F|| passes absolutely, OR gate as
         ``||F|| <= max(tol_rel*R_scale, atol_abs)`` with atol_abs ~ the old 3e-6 (a hybrid:
         scale-relative OR a small absolute backstop). With atol_abs the floor states all accept and
         the V states accept too (Reading 2) -- consistent and portable.
       - tol_rel cannot be set from a floor/dirty GAP (there is none); set it loosely (~1e-3) as a
         relative sanity backstop above the genuine-divergence regime, NOT as a floor/dirty separator.

     The absolute bar as shipped stays measured-correct + fails non-silently (parent §8.4), so P0 is
     safe either way; this census's job was to tell A2/A3 which contract to build -- and the answer is
     "a residual gate is a PORTABILITY+EFFICIENCY tool (Reading 2), not the floor-vs-dirty separator
     the task hypothesized (Reading 1) -- because the V is not residual-dirty."

  atol_floor used below = 1e-12.
================================================================================

Run (pids-fem, from forward-model/):
  PATH=/root/miniforge3/envs/pids-fem/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONPATH=. python scratch/_reason4_census.py
"""
from __future__ import annotations

import math

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem import petsc as fpetsc

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.overland import OverlandProblem, overland_conveyance, SECONDS_PER_DAY

ATOL_FLOOR = 1e-12

# Acceptance-impossible rtol/atol + the default stol: PETSc can then ONLY exit via the SNORM
# stagnation test (reason 4). Used to drive the floor states that NATURALLY balance (reason 2/3)
# onto reason 4 so they appear in the census (the task: "drive each onto reason 4").
FORCE4 = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_rtol": 1e-30, "snes_atol": 1e-30, "snes_stol": 1e-8,
    "snes_max_it": 50, "ksp_type": "preonly", "pc_type": "lu",
}

N_MAN_MMS = 0.03
EPS_S = 1e-3


# ---------------------------------------------------------------------------
# norm helpers
# ---------------------------------------------------------------------------

def _vec_norm(form_ufl, V) -> float:
    """L2 norm of the assembled residual vector for a scalar linear form (test fn in V)."""
    b = fpetsc.create_vector(V)
    with b.localForm() as lf:
        lf.set(0.0)
    fpetsc.assemble_vector(b, fem.form(form_ufl))
    b.assemble()
    n = float(b.norm())
    b.destroy()
    return n


def _ovl_storage_norm(prob, d_prev) -> float:
    v = prob._v
    dn = fem.Function(prob.V)
    dn.x.array[:] = d_prev
    dn.x.scatter_forward()
    return _vec_norm(((prob.d - dn) / prob.dt) * v * prob._dx_storage, prob.V)


def _ovl_conv_norm(prob) -> float:
    v = prob._v
    H_s, K_s = overland_conveyance(prob.d, prob.z_b, prob.n_man, prob.eps_S)
    return _vec_norm(K_s * ufl.dot(ufl.grad(H_s), ufl.grad(v)) * ufl.dx, prob.V)


def _ovl_F0_norm(prob, d_prev) -> float:
    """||F|| at the step START (d = d_n): assemble the engine residual prob.F with d set to d_prev."""
    d_save = prob.d.x.array.copy()
    prob.d.x.array[:] = d_prev
    prob.d.x.scatter_forward()
    n = _vec_norm(prob.F, prob.V)
    prob.d.x.array[:] = d_save
    prob.d.x.scatter_forward()
    return n


def _ovl_forcing_norm(prob, source=None, rain=None) -> float:
    v = prob._v
    terms = []
    if source is not None:
        terms.append(source * v * ufl.dx)
    if rain is not None:
        terms.append(rain * v * ufl.dx)
    if not terms:
        return 0.0
    f = terms[0]
    for t in terms[1:]:
        f = f + t
    return _vec_norm(f, prob.V)


def _Ks_ufl(d_expr, n_man, eps_S):
    g = ufl.grad(d_expr)
    slope_sqrt = (ufl.dot(g, g) + eps_S**2) ** 0.25
    return SECONDS_PER_DAY * d_expr ** (5.0 / 3.0) / (n_man * slope_sqrt)


# ---------------------------------------------------------------------------
# row assembly
# ---------------------------------------------------------------------------

def _row(case, scale, reason, fnorm, bforce, bstore, bconv, f0, note=""):
    R1 = max(bforce, bstore, ATOL_FLOOR)                  # plan's leading candidate
    R2 = max(bforce, bstore, bconv, ATOL_FLOOR)           # + conveyance
    Rsum = bforce + bstore + bconv                        # sum-of-magnitudes
    return dict(
        case=case, scale=scale, reason=int(reason), fnorm=fnorm,
        bforce=bforce, bstore=bstore, bconv=bconv, f0=f0,
        ratio1=fnorm / R1 if R1 > 0 else math.inf,
        ratio2=fnorm / R2 if R2 > 0 else math.inf,
        ratio_f0=fnorm / f0 if f0 > 0 else math.inf,
        ratio_sum=fnorm / Rsum if Rsum > 0 else math.inf,
        note=note,
    )


# ---------------------------------------------------------------------------
# floor cases (overland, single field d)
# ---------------------------------------------------------------------------

def case_mms_spatial(rows):
    e = float(np.e)
    for nc in (20, 40, 80):
        msh = dmesh.create_unit_interval(MPI.COMM_WORLD, nc)
        x = ufl.SpatialCoordinate(msh)
        d_star = 1.0 + 0.5 * ufl.exp(x[0])
        f_star = -ufl.div(_Ks_ufl(d_star, N_MAN_MMS, EPS_S) * ufl.grad(d_star))
        prob = OverlandProblem(msh, n_man=N_MAN_MMS, eps_S=EPS_S, source=f_star, lumped=False)
        prob.stall_accept_fnorm = math.inf
        prob.set_initial_condition(lambda X: 1.0 + 0.5 * np.exp(X[0]))
        prob.add_dirichlet(lambda X: np.isclose(X[0], 0.0), 1.5)
        prob.add_dirichlet(lambda X: np.isclose(X[0], 1.0), 1.0 + 0.5 * e)
        d_prev = prob.d_n.x.array.copy()
        prob.step(dt=1.0e8)
        rows.append(_row(f"MMS-spatial nc={nc}", 1.0, prob.last_reason, prob.last_fnorm,
                         _ovl_forcing_norm(prob, source=f_star), _ovl_storage_norm(prob, d_prev),
                         _ovl_conv_norm(prob), _ovl_F0_norm(prob, d_prev), note="floor; natural r4"))


def case_mms_temporal(rows):
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 120)
    x = ufl.SpatialCoordinate(msh)
    time = fem.Constant(msh, 0.0)
    gt = 1.0 + 0.3 * ufl.sin(2.0 * ufl.pi * time)
    dgt = 0.3 * 2.0 * ufl.pi * ufl.cos(2.0 * ufl.pi * time)
    d_star = 1.0 + 0.5 * ufl.sin(ufl.pi * x[0]) * gt
    dd_dt = 0.5 * ufl.sin(ufl.pi * x[0]) * dgt
    f_star = dd_dt - ufl.div(_Ks_ufl(d_star, N_MAN_MMS, EPS_S) * ufl.grad(d_star))
    prob = OverlandProblem(msh, n_man=N_MAN_MMS, eps_S=EPS_S, source=f_star, lumped=False,
                           petsc_options=FORCE4)
    prob.stall_accept_fnorm = math.inf
    prob.set_initial_condition(lambda X: 1.0 + 0.5 * np.sin(np.pi * X[0]))
    on_ends = lambda X: np.isclose(X[0], 0.0) | np.isclose(X[0], 1.0)
    prob.add_dirichlet(on_ends, 1.0)
    dt = 0.5 / 5.0
    time.value = dt
    d_prev = prob.d_n.x.array.copy()
    prob.step(dt)
    rows.append(_row("MMS-temporal (1 step)", 1.0, prob.last_reason, prob.last_fnorm,
                     _ovl_forcing_norm(prob, source=f_star), _ovl_storage_norm(prob, d_prev),
                     _ovl_conv_norm(prob), _ovl_F0_norm(prob, d_prev), note="floor; forced r4"))


def case_near_flat(rows):
    L = 10.0
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, L])
    prob = OverlandProblem(msh, n_man=0.03, petsc_options=FORCE4)
    prob.stall_accept_fnorm = math.inf
    prob.set_initial_condition(lambda x: 0.1 + 1e-3 * np.cos(np.pi * x[0] / L))
    d_prev = prob.d_n.x.array.copy()
    prob.step(1e-3)
    rows.append(_row("near-flat 1-D", 1.0, prob.last_reason, prob.last_fnorm,
                     0.0, _ovl_storage_norm(prob, d_prev), _ovl_conv_norm(prob),
                     _ovl_F0_norm(prob, d_prev), note="floor; forced r4 (natural=3)"))


def case_floor_stagnation(rows):
    """Near-flat bed + uniform start, n=0.05: reason 4 NATURALLY -- the production floor case."""
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 10.0])
    prob = OverlandProblem(msh, n_man=0.05)
    prob.stall_accept_fnorm = math.inf
    prob.set_topography(lambda x: 1e-6 * x[0])
    prob.set_initial_condition(lambda x: 0.1 + 0.0 * x[0])
    d_prev = prob.d_n.x.array.copy()
    prob.step(1e-3)
    rows.append(_row("floor-stagnation", 1.0, prob.last_reason, prob.last_fnorm,
                     0.0, _ovl_storage_norm(prob, d_prev), _ovl_conv_norm(prob),
                     _ovl_F0_norm(prob, d_prev), note="floor; NATURAL r4 (production case)"))


def case_lake_at_rest(rows):
    msh = dmesh.create_unit_interval(MPI.COMM_WORLD, 20)
    prob = OverlandProblem(msh, n_man=0.03, petsc_options=FORCE4)
    prob.stall_accept_fnorm = math.inf
    prob.set_topography(lambda x: 0.5 - 0.3 * x[0])
    prob.set_initial_condition(lambda x: 0.2 + 0.3 * x[0])
    d_prev = prob.d_n.x.array.copy()
    prob.step(dt=0.1)
    rows.append(_row("lake-at-rest", 1.0, prob.last_reason, prob.last_fnorm,
                     0.0, _ovl_storage_norm(prob, d_prev), _ovl_conv_norm(prob),
                     _ovl_F0_norm(prob, d_prev), note="floor; forced r4 (natural=2); flux==0 struct"))


# --- the 2-D convergent V (DIRTY): worst reason-4 stall over the storm -----
V_SX, V_SY = 0.05, 0.02
V_NMAN = 0.015
V_RAIN = 0.2592
V_STORM = 0.0625
V_NX, V_NY = 48, 30


def case_v_dirty(rows, scale: float):
    LX, LY = 1620.0 * scale, 1000.0 * scale
    XC = LX / 2.0
    AREA = LX * LY
    Qeq = V_RAIN * AREA
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [LX, LY]], [V_NX, V_NY])
    prob = OverlandProblem(msh, V_NMAN)
    prob.stall_accept_fnorm = math.inf  # BOOK reason-4 stalls (probe-only; keeps d at the stall)
    prob.set_topography(lambda x: V_SY * (LY - x[1]) + V_SX * np.abs(x[0] - XC))
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    rain = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=V_SY)

    t, dt = 0.0, 1e-5
    n4 = 0
    worst_F = None    # the reason-4 step with the largest ||F||
    worst_r1 = None   # ... and the largest ||F||/R1 (candidate-1 ratio)
    while t < V_STORM - 1e-12:
        h = min(dt, V_STORM - t)
        rain.value = V_RAIN
        d_prev = prob.d_n.x.array.copy()
        converged, it = prob.step(h)
        if converged:
            if int(prob.last_reason) == 4:
                n4 += 1
                F = float(prob.last_fnorm)
                bf = _ovl_forcing_norm(prob, rain=rain)
                bs = _ovl_storage_norm(prob, d_prev)
                bc = _ovl_conv_norm(prob)
                f0 = _ovl_F0_norm(prob, d_prev)
                r1 = F / max(bf, bs, ATOL_FLOOR)
                qn = prob.outflow_rate() / Qeq
                cand = (F, bf, bs, bc, f0, t + h, h, qn)
                if worst_F is None or F > worst_F[0]:
                    worst_F = cand
                if worst_r1 is None or r1 > worst_r1[0]:
                    worst_r1 = (r1,) + cand
            t += h
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), 1e-3)
        else:
            dt = h * 0.5
            if dt < 1e-11:
                break
    if worst_F is None:
        rows.append(_row(f"2-D V (NO r4) scale={scale:g}", scale, -999,
                         float("nan"), float("nan"), float("nan"), float("nan"), float("nan")))
        print(f"  [warn] V scale={scale:g}: no reason-4 step in the storm window", flush=True)
        return
    F, bf, bs, bc, f0, tt, hh, qn = worst_F
    rows.append(_row(f"2-D V worst|F| s={scale:g}", scale, 4, F, bf, bs, bc, f0,
                     note=f"dirty; n_r4={n4} t={tt:.4f} q/Qeq={qn:.2f}"))
    r1, F, bf, bs, bc, f0, tt, hh, qn = worst_r1
    rows.append(_row(f"2-D V worst|F|/R s={scale:g}", scale, 4, F, bf, bs, bc, f0,
                     note=f"dirty; t={tt:.4f} q/Qeq={qn:.2f}"))


def case_coupled_column(rows):
    """Coupled 1-D column under rain: a WORKING state (reason 2/3). Documents a healthy block step."""
    SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 1.0])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    rain = prob.add_rain(0.1)
    prob.step(1e-3)
    # forcing = rain on the d block (ds_top); storage = max(surface, subsurface) residual norms.
    bforce = _vec_norm(rain * prob._vd * prob._ds_top, prob.Vd)
    surf = _vec_norm(((prob.d - prob.d_n) / prob.dt) * prob._vd * prob._ds_top, prob.Vd)
    theta = prob.soil.theta_ufl(prob.psi)
    theta_n = prob.soil.theta_ufl(prob.psi_n)
    soil = _vec_norm(((theta - theta_n) / prob.dt) * prob._vpsi * prob._dx_storage, prob.Vpsi)
    bstore = max(surf, soil)
    # conveyance / F0 not assembled for the block here (the gate question is settled on overland);
    # record them as 0 / nan so the coupled row documents forcing+storage only.
    rows.append(_row("coupled column (rain)", 1.0, prob.last_reason, prob.last_fnorm,
                     bforce, bstore, 0.0, prob.last_fnorm,
                     note=f"working r{prob.last_reason}; surf={surf:.1e} soil={soil:.1e}"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rows: list[dict] = []
    print("[census] driving reason-4 states ...", flush=True)
    case_mms_spatial(rows)
    case_mms_temporal(rows)
    case_near_flat(rows)
    case_floor_stagnation(rows)
    case_lake_at_rest(rows)
    case_v_dirty(rows, scale=1.0)
    case_v_dirty(rows, scale=0.1)
    case_coupled_column(rows)

    # ---- table ----
    hdr = (f"\n{'case':26s} {'scl':>5s} {'rsn':>4s} {'||F||':>10s} {'||bforce||':>11s} "
           f"{'||bstore||':>11s} {'||bconv||':>11s} {'||F0||':>10s} | "
           f"{'F/R1':>9s} {'F/R2':>9s} {'F/F0':>9s} {'F/sum':>9s}  note")
    print(hdr)
    print("-" * (len(hdr) + 20))
    for r in rows:
        print(f"{r['case']:26s} {r['scale']:5.2g} {r['reason']:4d} "
              f"{r['fnorm']:10.2e} {r['bforce']:11.2e} {r['bstore']:11.2e} "
              f"{r['bconv']:11.2e} {r['f0']:10.2e} | "
              f"{r['ratio1']:9.2e} {r['ratio2']:9.2e} {r['ratio_f0']:9.2e} "
              f"{r['ratio_sum']:9.2e}  {r['note']}", flush=True)
    print("\n  R1 = max(||bforce||,||bstore||,atol)            [plan candidate]")
    print("  R2 = max(||bforce||,||bstore||,||bconv||,atol)  [+ conveyance escape hatch]")
    print("  F0 = ||F at d=d_n||  -> F/F0 = relative residual reduction")
    print("  sum = ||bforce||+||bstore||+||bconv||           [sum-of-magnitudes]")

    # ---- separation verdict per candidate form ----
    floor = [r for r in rows if r["reason"] == 4 and r["case"].startswith(("MMS", "near", "floor", "lake"))]
    dirty = [r for r in rows if r["reason"] == 4 and r["case"].startswith("2-D V")]

    print("\n[separation]  (FLOOR must ACCEPT, DIRTY must REJECT; a form SEPARATES iff "
          "min(dirty ratio) > max(floor ratio))")
    for key, label in (("ratio1", "R1 = max(bforce,bstore,atol)"),
                       ("ratio2", "R2 = max(bforce,bstore,bconv,atol)"),
                       ("ratio_f0", "F0 (relative reduction F/F0)"),
                       ("ratio_sum", "sum (bforce+bstore+bconv)")):
        fr = [r[key] for r in floor if math.isfinite(r[key])]
        dr = [r[key] for r in dirty if math.isfinite(r[key])]
        if not fr or not dr:
            continue
        maxf, mindr = max(fr), min(dr)
        sep = mindr > maxf
        # who breaks it (the floor states above the dirty floor)?
        offenders = [r["case"] for r in floor if math.isfinite(r[key]) and r[key] >= mindr]
        verdict = "SEPARATES" if sep else "INTERLEAVED -> NO tol_rel works"
        line = (f"  {label:36s}: max_floor={maxf:.2e}  min_dirty={mindr:.2e}  "
                f"gap={mindr/maxf:.2e}x  -> {verdict}")
        print(line, flush=True)
        if sep:
            print(f"      recommended tol_rel = geomean = {math.sqrt(maxf*mindr):.2e}", flush=True)
        else:
            print(f"      floor states at/above the dirty min: {offenders}", flush=True)

    # ---- V scale-invariance of the candidate-1 ratio ----
    v1 = [r for r in dirty if r["scale"] == 1.0 and r["case"].startswith("2-D V worst|F| ")]
    v0 = [r for r in dirty if r["scale"] == 0.1 and r["case"].startswith("2-D V worst|F| ")]
    if v1 and v0:
        a, b = v1[0]["ratio1"], v0[0]["ratio1"]
        if math.isfinite(a) and math.isfinite(b) and a > 0:
            print(f"\n[scale-invariance] worst-|F| V F/R1: scale1.0={a:.2e} scale0.1={b:.2e}  "
                  f"differ {abs(a-b)/a*100:.0f}%", flush=True)
        f1, f0 = v1[0]["fnorm"], v0[0]["fnorm"]
        if math.isfinite(f1) and math.isfinite(f0):
            print(f"[old-bar check] worst-|F| V ||F||: scale1.0={f1:.2e} scale0.1={f0:.2e}  "
                  f"straddle absolute 3e-6? {(f1<=3e-6)!=(f0<=3e-6)}", flush=True)


if __name__ == "__main__":
    main()
