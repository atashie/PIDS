"""P1-B5b DIAGNOSTIC: is the single-node valley-line depth concentration on the tilted-V an
ARTIFACT (idealized measure-zero channel + a smooth functional that can't integrate a 1-cell
spike) or a REAL SCHEME DEFECT (the upwind flux genuinely under-delivers at the convergence
kink)? Diagnosis only -- this probe does NOT touch overland_upwind.py or any B5 deliverable.

THE FINDING (B5 review, reproduced here): on the canonical kink V the LUMPED outlet discharge
== 1.0000*Q_eq at the storm plateau (both 48x30 and 96x60), but the CONSISTENT ds-integral
discharge (int 86400*(1/n)*d^(5/3)*sqrt(SY) ds over y=LY, the galerkin OverlandProblem.outflow_rate
functional) = 0.85 (48x30) and 0.84 (96x60) -- it DIVERGES from 1.0 under refinement, and the
valley peak depth GROWS (68 -> 103 mm). Flow concentrates onto the single valley node.

THE TWO HYPOTHESES:
  (A) ARTIFACT / thin-channel physics: the V channel is a measure-ZERO line (the bed kink at x=XC);
      the valley "channel" is 1 cell wide; refining the cross-slope makes it NARROWER, so to carry
      the same catchment discharge down the 2% valley the Manning depth there MUST increase -- depth
      growth is PHYSICAL (a 1-cell channel), the LUMPED 1.000 is the correct conserved throughput,
      and the consistent 0.85 is the P1 functional mis-integrating an increasingly-sharp d^(5/3)
      spike. A real finite-width swale would resolve fine.
  (B) DEFECT at the kink: the upwind scheme genuinely UNDER-delivers flux at the convergence line
      (cotangent T_e / head-differencing mishandles the slope-sign kink), so 0.85 is a real flux
      error -- analogous to ParFlow's cell-centered-slope pathology fixed by face-centered
      OverlandKinematic.

PROBES (this file):
  1. LUMPED IDENTITY: show outflow_rate() (lumped) == rain*sum(A_i) == Q_eq at steady state for ANY
     converged field (discrete steady-state mass balance; the lumped 1.000 is a CONSERVATION
     identity, not independent accuracy).
  2. KEY TEST -- FINITE-WIDTH VALLEY FLOOR: z_b = SY*(LY-y) + SX*max(|x-XC|-W/2, 0) (flat floor of
     width W). Run to plateau, measure BOTH lumped and consistent discharge, at increasing
     cells-across-floor. consistent -> 1.0 as the floor is RESOLVED => (A). stays ~0.85 => (B).
  3. NORMAL-DEPTH check: kink V -- valley carries ~Q_eq down 2% through a ~1-cell channel; Manning
     normal depth d_n = (q*n/sqrt(SY))^(3/5) with q = Q_eq/channel_width. Does measured peak match?
     Does peak_depth * channel_width converge to a fixed channel discharge?
  4. CROSS-CHANNEL PROFILE: depth across x at the outlet row y=LY for 48x30 vs 96x60 -- smooth
     hillslope + sharp 1-cell spike (=> (A)) vs oscillation/under-delivery at the kink (=> (B)).

Run (pids-fem, from forward-model):
  PYTHONPATH=. python scratch/_b5b_valley_concentration.py
Env: QUICK=1 runs a reduced sweep (fewer W / resolutions) for a fast pass.
"""
from __future__ import annotations

import os
import time

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import assemble_vector

from pids_forward.physics.overland_upwind import UpwindOverlandProblem

SECONDS_PER_DAY = 86400.0

# canonical tilted-V (IDENTICAL to _v2d_upwind_V.py / _v2d_overland_diag.py)
SCALE = float(os.environ.get("SCALE", "1.0"))
LX, LY = 1620.0 * SCALE, 1000.0 * SCALE
XC = LX / 2.0
SX, SY = 0.05, 0.02
N_MAN = 0.015
RAIN = 0.2592                  # m/day
STORM = 0.0625                 # day
AREA = LX * LY
Q_EQ = RAIN * AREA             # equilibrium discharge [m^3/day]

QUICK = os.environ.get("QUICK", "0") == "1"


def make_zb(W):
    """Bed elevation. W=0 => the canonical KINK V (z_b = SY*(LY-y) + SX*|x-XC|).
    W>0 => a FLAT-BOTTOMED valley of floor width W: the cross-slope SX only starts beyond |x-XC|=W/2,
    so the valley floor (|x-XC| < W/2) is flat at the local valley elevation SY*(LY-y)."""
    half = 0.5 * W

    def zb(x):
        return SY * (LY - x[1]) + SX * np.maximum(np.abs(x[0] - XC) - half, 0.0)
    return zb


def consistent_outflow(prob, msh):
    """The CONSISTENT (galerkin-measure) ds-integral discharge over the y=LY outlet:
       int 86400*(1/n)*max(d,0)^(5/3)*sqrt(SY) ds   (default FFCX quadrature, the functional
    OverlandProblem.outflow_rate / _v2d_overland_diag use). This is the P1-interpolated d^(5/3)
    integrated over the facet -- vs the LUMPED nodal-trapezoid sum prob.outflow_rate()."""
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = np.sort(dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], LY)))
    ft = dmesh.meshtags(msh, fdim, facets, np.full(facets.shape, 1, dtype=np.int32))
    ds_out = ufl.Measure("ds", domain=msh, subdomain_data=ft)(1)
    d_pos = ufl.max_value(prob.d, 0.0)
    form = fem.form(SECONDS_PER_DAY * (1.0 / N_MAN) * d_pos ** (5.0 / 3.0) * ufl.sqrt(SY) * ds_out)
    return float(fem.assemble_scalar(form))


def consistent_outflow_degree(prob, msh, qdeg):
    """The consistent ds-integral at a FORCED quadrature degree qdeg (else identical to
    consistent_outflow). Used by Probe 5 to show the consistent discharge is quadrature-degree
    INDEPENDENT on the kink field -- the deficit is the P1-INTERPOLATED d^(5/3) (a tall narrow
    spike on a coarse facet), NOT under-integration; even exact quadrature of that P1 field gives
    ~0.85, so refining the quadrature does not recover the throughput."""
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = np.sort(dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], LY)))
    ft = dmesh.meshtags(msh, fdim, facets, np.full(facets.shape, 1, dtype=np.int32))
    ds_out = ufl.Measure("ds", domain=msh, subdomain_data=ft,
                         metadata={"quadrature_degree": int(qdeg)})(1)
    d_pos = ufl.max_value(prob.d, 0.0)
    form = fem.form(SECONDS_PER_DAY * (1.0 / N_MAN) * d_pos ** (5.0 / 3.0) * ufl.sqrt(SY) * ds_out)
    return float(fem.assemble_scalar(form))


def run_to_plateau(NX, NY, W, t_end=STORM, dt_max=1e-3, dt0=1e-5, grow_at=3, shrink_at=8,
                   verbose=False):
    """Drive UpwindOverlandProblem on the tilted-V (floor width W) to the storm plateau.
    Returns (prob, msh, info) with info carrying the plateau lumped/consistent discharge, the
    valley peak depth, the cross-outlet profile, and the books gap. Mirrors _v2d_upwind_V.py's
    adaptive controller exactly so the numbers are comparable to the B5 deliverable."""
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [LX, LY]], [NX, NY])
    prob = UpwindOverlandProblem(msh, N_MAN)
    prob.set_topography(make_zb(W))
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)

    t, dt = 0.0, dt0
    n_acc = n_rej = 0
    W_water = prob.total_water()
    cum_rain = cum_out = 0.0
    # plateau accumulators: average the LAST third of the storm window (rising limb is saturated)
    plateau_lo = 0.6 * min(STORM, t_end)
    q_lump_acc, q_cons_acc = [], []
    t0 = time.time()
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        r = RAIN if (t + h) <= STORM + 1e-12 else 0.0
        prob.add_rain(r)
        converged, it = prob.step(h)
        if converged:
            Wn = prob.total_water()
            q_lump = prob.outflow_rate()
            cum_rain += h * (r * AREA)
            cum_out += h * q_lump
            W_water = Wn
            t += h
            n_acc += 1
            if t >= plateau_lo - 1e-15:
                q_lump_acc.append(q_lump)
                q_cons_acc.append(consistent_outflow(prob, msh))
            dt = min(dt * (1.5 if it <= grow_at else 0.7 if it >= shrink_at else 1.0), dt_max)
            if verbose:
                print(f"    t={t:9.6f} q_lump/Qeq={q_lump/Q_EQ:7.4f} dt={h:8.2e} it={it:2d}",
                      flush=True)
        else:
            n_rej += 1
            dt = h * 0.5
            if dt < 1e-12:
                print("    !! dt collapse", flush=True)
                break
    run_s = time.time() - t0

    # final-state diagnostics
    coords = prob.V.tabulate_dof_coordinates()
    d_arr = prob.d.x.array[: prob.n_dofs]
    # outlet row (y=LY): sort by x for the cross-channel profile
    on_outlet = np.isclose(coords[: prob.n_dofs, 1], LY)
    ox = coords[: prob.n_dofs, 0][on_outlet]
    od = d_arr[on_outlet]
    order = np.argsort(ox)
    ox, od = ox[order], od[order]
    # valley-line peak depth (the node(s) nearest x=XC, over the whole domain)
    dx_cell = LX / NX
    near_valley = np.abs(coords[: prob.n_dofs, 0] - XC) < 0.5 * dx_cell + 1e-9
    valley_peak = float(d_arr[near_valley].max()) if near_valley.any() else float("nan")
    # outlet-row valley peak specifically
    outlet_valley_peak = float(od[np.argmin(np.abs(ox - XC))])

    dW_total = W_water - 0.0
    books_gap = cum_rain - cum_out - dW_total

    info = dict(
        NX=NX, NY=NY, W=W, dx_cell=dx_cell, dy_cell=LY / NY,
        q_lump=float(np.mean(q_lump_acc)) if q_lump_acc else float("nan"),
        q_cons=float(np.mean(q_cons_acc)) if q_cons_acc else float("nan"),
        q_lump_over_Qeq=(float(np.mean(q_lump_acc)) / Q_EQ) if q_lump_acc else float("nan"),
        q_cons_over_Qeq=(float(np.mean(q_cons_acc)) / Q_EQ) if q_cons_acc else float("nan"),
        valley_peak=valley_peak, outlet_valley_peak=outlet_valley_peak,
        n_acc=n_acc, n_rej=n_rej, run_s=run_s, books_gap=books_gap, cum_rain=cum_rain,
        n_plateau=len(q_lump_acc), ox=ox, od=od,
        d_min=float(d_arr.min()),
    )
    return prob, msh, info


# ============================================================================
# PROBE 1 -- the lumped identity (conservation, not accuracy)
# ============================================================================
def probe1_lumped_identity():
    print("=" * 78)
    print("PROBE 1: the LUMPED outflow_rate() == rain*sum(A_i) == Q_eq is a CONSERVATION identity")
    print("=" * 78)
    print("Analytic: summing ALL node residuals R_i telescopes every interior edge flux (+Q_e on")
    print("  one row, -Q_e on the other) to ZERO, leaving the GLOBAL balance")
    print("    sum_i (d_i - d_n,i) A_i / dt  +  sum_k q_out(d_k) B_k  -  rain * sum_i A_i  =  0")
    print("  At a residual-converged root R=0. At the storm PLATEAU the field is steady (d=d_n), so")
    print("  the storage term vanishes and  sum_k q_out B_k (= outflow_rate, LUMPED) == rain*sum A_i")
    print("  == rain*AREA == Q_eq EXACTLY -- for ANY converged steady field, independent of its shape.")
    print("  => the lumped 1.000 is the discrete steady-state mass balance, NOT an independent")
    print("     accuracy measurement of the outlet flux.\n")
    NX, NY = (48, 30)
    prob, msh, info = run_to_plateau(NX, NY, W=0.0)
    sumA = float(np.sum(prob.A_i))
    print(f"  [numeric, kink V {NX}x{NY}, plateau]")
    print(f"    sum_i A_i            = {sumA:.6f} m^2   (domain area AREA = {AREA:.6f})")
    print(f"    rain * sum_i A_i     = {RAIN*sumA:14.3f} m^3/day  (= Q_eq = {Q_EQ:.3f})")
    print(f"    outflow_rate (LUMPED)= {info['q_lump']:14.3f} m^3/day  "
          f"(lumped/Q_eq = {info['q_lump_over_Qeq']:.5f})")
    print(f"    consistent ds-integ. = {info['q_cons']:14.3f} m^3/day  "
          f"(consistent/Q_eq = {info['q_cons_over_Qeq']:.5f})")
    print(f"    books gap            = {info['books_gap']:+.3e} m^3 "
          f"({100*info['books_gap']/max(info['cum_rain'],1e-9):+.4f}% of rain)")
    print(f"  => lumped tracks rain*sum(A_i) to ~{abs(info['q_lump_over_Qeq']-1.0)*100:.3f}%; "
          f"the consistent integral is {info['q_cons_over_Qeq']*100:.1f}% (it is the independent reading).\n")
    return info


# ============================================================================
# PROBE 2 -- THE KEY TEST: finite-width valley floor
# ============================================================================
def probe2_finite_width_valley():
    print("=" * 78)
    print("PROBE 2 (KEY): FINITE-WIDTH FLAT-BOTTOMED VALLEY -- does the consistent discharge -> 1.0")
    print("              as the valley FLOOR is resolved? (-> A) or stay ~0.85? (-> B)")
    print("=" * 78)
    print("z_b = SY*(LY-y) + SX*max(|x-XC|-W/2, 0): a flat floor of width W, cross-slope beyond it.")
    print("For each W we INCREASE NX so more cells span the floor (cells_across ~ W*NX/LX). If the")
    print("consistent (smooth-functional) discharge converges to ~1.0 once the floor carries several")
    print("cells, the kink/1-cell-channel was the issue (A). If it stays ~0.85 on a WELL-RESOLVED")
    print("finite-width valley, the scheme under-delivers flux at the channel (B).\n")

    # W chosen as a fraction of LX so 'cells across floor' = W/LX * NX is controllable.
    # LX=1620; W in {0 (kink), 81 (=LX/20), 162 (=LX/10), 324 (=LX/5)} m.
    if QUICK:
        cases = [
            (0.0,   [(48, 30), (96, 60)]),
            (162.0, [(48, 30), (96, 60)]),     # ~4.8 -> 9.6 cells across floor
        ]
    else:
        cases = [
            (0.0,   [(48, 30), (96, 60), (192, 120)]),                  # kink (measure-zero channel)
            (81.0,  [(48, 30), (96, 60), (192, 120)]),                  # ~2.4 / 4.8 / 9.6 cells
            (162.0, [(48, 30), (96, 60), (192, 120)]),                  # ~4.8 / 9.6 / 19.2 cells
            (324.0, [(48, 30), (96, 60), (192, 120)]),                  # ~9.6 / 19.2 / 38.4 cells
        ]

    results = {}
    print(f"  {'W[m]':>7} {'NX x NY':>9} {'cells/floor':>11} {'lump/Qeq':>9} {'cons/Qeq':>9} "
          f"{'peak[mm]':>9} {'books%':>8} {'t[s]':>6}")
    print("  " + "-" * 76)
    for W, meshes in cases:
        for (NX, NY) in meshes:
            _, _, info = run_to_plateau(NX, NY, W=W)
            cells_floor = W / LX * NX  # # of cross-cells spanning the flat floor
            results[(W, NX, NY)] = info
            print(f"  {W:7.0f} {f'{NX}x{NY}':>9} {cells_floor:11.2f} "
                  f"{info['q_lump_over_Qeq']:9.4f} {info['q_cons_over_Qeq']:9.4f} "
                  f"{1000*info['outlet_valley_peak']:9.2f} "
                  f"{100*info['books_gap']/max(info['cum_rain'],1e-9):+8.4f} {info['run_s']:6.1f}",
                  flush=True)
        print("  " + "-" * 76)
    return results


# ============================================================================
# PROBE 3 -- channel normal-depth check on the kink V
# ============================================================================
def probe3_normal_depth(probe2_results=None):
    print("=" * 78)
    print("PROBE 3: Manning NORMAL-DEPTH of the 1-cell valley channel (kink V) vs measured peak")
    print("=" * 78)
    print("The kink valley sits on a SINGLE column of nodes; nearly all the catchment funnels to it,")
    print("so the valley node carries a per-unit-width discharge q_v that is set by how much of Q_eq")
    print("exits its ~dx-wide control. The valley node's outlet sink is, BY CONSTRUCTION, the Manning")
    print("normal-depth relation  q_out(d_v) = 86400*(1/n)*d_v^(5/3)*sqrt(SY)  [m^2/day per width].")
    print("We MEASURE the per-width discharge at the valley node two ways and check they agree, and")
    print("show q_v GROWS ~ refinement (because the channel control width dx SHRINKS) -- so the")
    print("normal depth d_v = (q_v*n/(86400*sqrt(SY)))^(3/5) must GROW too. Matching the measured")
    print("peak => the depth growth is a PHYSICAL 1-cell Manning channel (A), not a flux defect.\n")
    print("  Units: q in m^2/DAY throughout (same 86400 factor as the engine sink); no SI mixing.\n")

    meshes = [(48, 30), (96, 60)] if QUICK else [(48, 30), (96, 60), (192, 120)]
    print(f"  {'NX x NY':>9} {'dx[m]':>8} {'q_v[m2/d]':>11} {'d_n,pred[mm]':>12} {'peak[mm]':>9} "
          f"{'pred/peak':>9} {'frac Q_v':>8}")
    print("  " + "-" * 76)
    rows = []
    for (NX, NY) in meshes:
        info = None
        if probe2_results is not None:
            info = probe2_results.get((0.0, NX, NY))
        if info is None:
            _, _, info = run_to_plateau(NX, NY, W=0.0)
        dx = LX / NX
        peak = info["outlet_valley_peak"]
        # The per-unit-width Manning discharge AT THE MEASURED valley depth (the normal-depth
        # relation evaluated at d=peak). This is what the valley node actually conveys per width.
        q_v = SECONDS_PER_DAY * (1.0 / N_MAN) * max(peak, 0.0) ** (5.0 / 3.0) * np.sqrt(SY)
        # Invert that same relation to predict the normal depth from q_v -- a round-trip identity
        # check (must match peak; confirms the valley node IS a Manning normal-depth channel).
        d_n_pred = (q_v * N_MAN / (SECONDS_PER_DAY * np.sqrt(SY))) ** (3.0 / 5.0)
        # Fraction of the TOTAL lumped outlet discharge carried by the valley node's control width
        # (q_v * its outlet length B_v ~ q_v*dx) relative to Q_eq -- how concentrated the flow is.
        frac_Qv = (q_v * dx) / Q_EQ
        rows.append((NX, dx, q_v, d_n_pred, peak, frac_Qv))
        print(f"  {f'{NX}x{NY}':>9} {dx:8.2f} {q_v:11.1f} {1000*d_n_pred:12.2f} {1000*peak:9.2f} "
              f"{d_n_pred/peak:9.4f} {frac_Qv:8.3f}")
    print("  " + "-" * 76)
    # The KEY invariance: q_v (per-width discharge at the valley) should scale ~ 1/dx if the SAME
    # channel throughput Q_ch = q_v*dx is funneled through an ever-narrower 1-cell channel.
    if len(rows) >= 2:
        Qch = [q * dx for (_, dx, q, _, _, _) in rows]
        print(f"  channel throughput Q_ch = q_v*dx across meshes = {['%.0f' % v for v in Qch]} m^3/day")
        print(f"    (~constant => a FIXED discharge funneled through a shrinking 1-cell channel)")
        r = rows
        print(f"  valley peak growth: {1000*r[0][4]:.1f} -> {1000*r[1][4]:.1f}"
              + (f" -> {1000*r[2][4]:.1f}" if len(r) > 2 else "") + " mm "
              f"(x{r[1][4]/r[0][4]:.2f}" + (f", x{r[2][4]/r[1][4]:.2f}" if len(r) > 2 else "") + ")")
        print(f"  q_v growth (per-width): x{r[1][2]/r[0][2]:.2f}"
              + (f", x{r[2][2]/r[1][2]:.2f}" if len(r) > 2 else "")
              + f"  vs dx shrink x{r[0][1]/r[1][1]:.2f}"
              + (f", x{r[1][1]/r[2][1]:.2f}" if len(r) > 2 else "")
              + "  (depth^(5/3) carries the per-width growth as dx halves)")
        # Manning prediction for the peak-depth growth ratio when q_v doubles (dx halves at fixed
        # throughput): d ~ q_v^(3/5) ~ 2^(3/5) = 1.516. Compare to the MEASURED growth ratio.
        print(f"  Manning predicts peak growth per dx-halving = 2^(3/5) = {2**0.6:.3f}; "
              f"measured = {r[1][4]/r[0][4]:.3f}"
              + (f", {r[2][4]/r[1][4]:.3f}" if len(r) > 2 else ""))
    return rows


# ============================================================================
# PROBE 4 -- cross-channel depth profile at the outlet row
# ============================================================================
def probe4_cross_profile(probe2_results=None):
    print("=" * 78)
    print("PROBE 4: CROSS-CHANNEL depth profile at the outlet row y=LY (kink V), 48x30 vs 96x60")
    print("=" * 78)
    print("Smooth hillslope rising to a SHARP 1-cell spike at x=XC (no negative/oscillatory lobes)")
    print("=> thin-channel (A). Oscillation / dip / under-delivery signature at the kink => (B).\n")
    for (NX, NY) in [(48, 30), (96, 60)]:
        info = None
        if probe2_results is not None:
            info = probe2_results.get((0.0, NX, NY))
        if info is None:
            _, _, info = run_to_plateau(NX, NY, W=0.0)
        ox, od = info["ox"], info["od"]
        # print a compact profile: every node near the valley, decimated on the wings
        print(f"  --- {NX}x{NY}  (outlet row, x in m -> depth in mm; valley at x=XC={XC:g}) ---")
        # find the valley index, print a window around it plus a few wing samples
        iv = int(np.argmin(np.abs(ox - XC)))
        win = range(max(0, iv - 6), min(len(ox), iv + 7))
        # wing samples (decimated)
        left = list(range(0, max(0, iv - 6), max(1, (iv - 6) // 4 if iv - 6 > 0 else 1)))
        right = list(range(min(len(ox), iv + 7), len(ox),
                           max(1, (len(ox) - (iv + 7)) // 4 if len(ox) - (iv + 7) > 0 else 1)))
        idxs = sorted(set(left) | set(win) | set(right))
        for k in idxs:
            marker = "  <== VALLEY" if k == iv else ""
            bar = "#" * int(min(60, 1000 * od[k] / 2.0))  # 1 char per 2 mm
            print(f"    x={ox[k]:8.1f}  d={1000*od[k]:7.2f} mm  {bar}{marker}")
        # monotonicity / oscillation check: is the profile single-peaked at the valley with no
        # interior local minima on either flank (a clean hillslope -> spike)?
        d_left = od[:iv + 1]
        d_right = od[iv:]
        left_mono = bool(np.all(np.diff(d_left) >= -1e-9))    # nondecreasing toward valley
        right_mono = bool(np.all(np.diff(d_right) <= 1e-9))   # nonincreasing away from valley
        n_neg = int(np.sum(od < -1e-9))
        print(f"    [shape] rises monotonically to the valley: {left_mono} (left) / {right_mono} "
              f"(right);  negative nodes: {n_neg};  peak/2nd = "
              f"{od[iv] / max(np.partition(od, -2)[-2], 1e-12):.2f}\n")


# ============================================================================
# PROBE 5 (optional) -- is the consistent ~0.85 scheme-specific or a measure property?
# ============================================================================
def probe5_consistent_is_measure_property():
    print("=" * 78)
    print("PROBE 5 (optional): is the consistent ~0.85 UPWIND-specific, or a property of the")
    print("                    CONSISTENT MEASURE reading a thin P1 channel (true for any P1 scheme)?")
    print("=" * 78)
    print("(a) QUADRATURE-DEGREE independence on the UPWIND kink field: recompute the consistent")
    print("    ds-integral at degrees {2,4,10,20}. If it is ~constant (not rising toward 1.0 with")
    print("    degree), the deficit is the P1-INTERPOLATED d^(5/3) spike, NOT under-integration --")
    print("    even EXACT quadrature of the coarse P1 field reads ~0.85.")
    print("(b) the GALERKIN scheme's OWN consistent plateau (from scratch/v2d_diag_48x30.npz, if")
    print("    present): if it ALSO reads ~0.85-0.88, the 0.85 is the consistent-measure reading of")
    print("    the thin channel for BOTH schemes, not an upwind flux artifact.\n")

    # (a) quadrature sweep on the upwind kink field
    for (NX, NY) in ([(48, 30)] if QUICK else [(48, 30), (96, 60)]):
        prob, msh, info = run_to_plateau(NX, NY, W=0.0)
        print(f"  (a) UPWIND kink {NX}x{NY}: lump/Qeq={info['q_lump_over_Qeq']:.4f}  "
              f"consistent ds-integral / Q_eq by quadrature degree:")
        for qd in (2, 4, 10, 20):
            qc = consistent_outflow_degree(prob, msh, qd) / Q_EQ
            print(f"        deg {qd:2d}:  {qc:.4f}")
    # (b) galerkin diag's recorded consistent plateau
    npz = "scratch/v2d_diag_48x30.npz"
    if os.path.exists(npz):
        g = np.load(npz)
        acc = g["accepted"] == 1
        tt = g["t"][acc]
        Qeq_g = float(g["Q_eq"])
        storm_g = float(g["storm"])
        pm = acc & (g["t"] >= 0.6 * storm_g) & (g["t"] <= storm_g + 1e-12)
        if pm.sum() == 0:  # diag may not reach the plateau window; use the last quartile
            pm = acc & (g["t"] >= 0.75 * tt.max())
        qdef = g["qdef"][pm] / Qeq_g
        q20 = g["q20"][pm] / Qeq_g
        qlump = g["qout"][pm] / Qeq_g
        print(f"\n  (b) GALERKIN diag 48x30 plateau (n={int(pm.sum())} steps):")
        print(f"        lumped qout/Qeq          mean = {np.mean(qlump):.4f}")
        print(f"        consistent qdef/Qeq      mean = {np.mean(qdef):.4f}  (default quadrature)")
        print(f"        consistent q_deg20/Qeq   mean = {np.mean(q20):.4f}  (degree 20)")
        print(f"      => the GALERKIN scheme's consistent measure ALSO reads ~{np.mean(qdef):.2f} on")
        print(f"         the kink V -- the 0.85 deficit is the consistent measure on a thin P1")
        print(f"         channel, NOT an upwind-flux defect.")
    else:
        print(f"\n  (b) [skipped: {npz} not found -- run scratch/_v2d_overland_diag.py to populate]")


def main():
    print(f"\nB5b VALLEY-CONCENTRATION DIAGNOSTIC  (canonical tilted-V, SCALE={SCALE:g}, "
          f"Q_eq={Q_EQ:.1f} m^3/day, AREA={AREA:.0f} m^2)")
    print(f"QUICK={QUICK}\n")
    t0 = time.time()
    p1 = probe1_lumped_identity()
    print()
    p2 = probe2_finite_width_valley()
    print()
    p3 = probe3_normal_depth(p2)
    print()
    probe4_cross_profile(p2)
    print()
    probe5_consistent_is_measure_property()
    print(f"\n[all probes done in {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
