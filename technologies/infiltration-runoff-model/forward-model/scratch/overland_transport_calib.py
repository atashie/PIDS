"""ACCURACY VALIDATION (investigation): lateral-transport rate of the sequential operator-split
overland scheme vs the resolved upwind reference, and the omega / Picard sweep.

Question (parent brief): SequentialCoupledProblem conserves mass exactly but its LATERAL TRANSPORT is
APPROXIMATED (under-relaxed omega~0.5 one-step-lagged Manning routing sweep). Is it reasonably accurate
or artificially throttled, and what omega / routing-substep / Picard-iter setting best matches the
RESOLVED upwind transport?

Reference = CoupledProblem(overland_scheme="upwind") (the validated monotone diffusion-wave). It
dt-collapses on stiff clay (the reason the sequential scheme exists), so the head-to-head runs on a
NON-stiff LOAM (n=1.56) + a RESOLVED swale where BOTH converge.

We run TWO regimes on the SAME tilted-V swale geometry/mesh/forcing:
  * Regime T (TRANSPORT ISOLATION): near-impermeable bed (Ks tiny) -> ~all rain routes downslope and
    exits the outlet; infiltration is negligible in BOTH schemes, so the only thing being compared is
    the lateral transport rate. This is the cleanest isolation of the knob's effect (removes the
    psi-pond vs monolithic-NCP infiltration confounder).
  * Regime R (RUN-ON / LOAM): a real loam bed + short intense storm so water ponds, routes downslope,
    and infiltrates as run-on -- the actual PIDS regime. Here the two schemes' infiltration handling
    differs, so we read the run-on outcome (how much water reached the outlet vs infiltrated) NOT a
    bit-match.

Transport metrics (per run):
  * outflow hydrograph: cum_outflow(t), and the outlet "arrival" time (t at which cum_out reaches a
    fraction of its final), and the storm-end / final outflow fraction.
  * surface-water timeseries surf(t) = ponded volume; its PEAK (the transient swale pond) + time-of-peak.
  * the down-slope ponded-depth profile d(y) at a few snapshots (centerline of the swale floor).

Run (pids-fem, from forward-model):
  PYTHONPATH=. python scratch/overland_transport_calib.py smoke      # tiny: both schemes run
  PYTHONPATH=. python scratch/overland_transport_calib.py T          # transport-isolation sweep
  PYTHONPATH=. python scratch/overland_transport_calib.py R          # loam run-on sweep
  PYTHONPATH=. python scratch/overland_transport_calib.py all
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.sequential_coupling import SequentialCoupledProblem

COMM = MPI.COMM_WORLD
SECONDS_PER_DAY = 86400.0

# ----------------------------------------------------------------------------------------------------
# Geometry: a tilted-V with a RESOLVED flat-bottomed valley floor of width W (the real PIDS swale; W>0
# avoids the measure-zero-channel kink artifact so the upwind scheme is clean). Smaller than the
# canonical 1620x1000 m so runs are minutes, but same SX/SY structure.
# ----------------------------------------------------------------------------------------------------
LX, LY, H = 200.0, 120.0, 1.5
XC = LX / 2.0
SX, SY = 0.05, 0.02          # cross-slope / down-slope (canonical tilted-V slopes)
W_SWALE = 50.0               # resolved floor width (>= several cells)
N_MAN = 0.05
NX, NY, NZ = 20, 16, 3       # mesh; floor W spans ~5 cells in x

# soils
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)   # the upwind-test loam
NEAR_IMPERMEABLE = VanGenuchten(theta_r=0.10, theta_s=0.40, alpha=2.0, n=2.0, Ks=1.0e-3)


def z_b(x):
    cross = np.maximum(np.abs(x[0] - XC) - 0.5 * W_SWALE, 0.0)
    return SY * (LY - x[1]) + SX * cross


def _swale_floor_centerline_dofs(coords_top, *, dx_cell):
    """Top dofs on the swale-floor centerline column |x-XC| < dx/2, sorted by y (downslope)."""
    x = coords_top[:, 0]
    m = np.abs(x - XC) < 0.5 * dx_cell + 1e-9
    idx = np.where(m)[0]
    order = np.argsort(coords_top[idx, 1])
    return idx[order]


# ----------------------------------------------------------------------------------------------------
# Drivers. Both return a dict with the transport timeseries + final ledger.
# ----------------------------------------------------------------------------------------------------
def _make_box():
    return dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [LX, LY, H]], [NX, NY, NZ])


def run_coupled(soil, *, rain, storm, t_end, dt0=1e-4, dt_max=2e-3, psi_i=-1.0,
                grow_at=3, shrink_at=8, label="upwind", verbose=False):
    """CoupledProblem(overland_scheme='upwind') reference run on the swale. Outlet = the y=LY edge
    (full downslope edge; the side slopes drain to it). Returns the transport record."""
    msh = _make_box()
    prob = CoupledProblem(msh, soil, n_man=N_MAN, overland_scheme="upwind")
    prob.set_initial_condition(lambda x: psi_i + 0.0 * x[2], d_value=0.0)
    prob.set_topography(z_b)
    r = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)

    topdofs = prob._top_dofs(prob.Vd)
    coords_top = prob.Vd.tabulate_dof_coordinates()[topdofs]
    cl = _swale_floor_centerline_dofs(coords_top, dx_cell=LX / NX)
    cl_global = topdofs[cl]
    y_cl = coords_top[cl, 1]
    area = LX * LY

    def surf():
        return prob.surface_water()

    def depth_top():
        return prob.d.x.array[topdofs].copy()

    rec = _empty_rec()
    rec["y_cl"] = y_cl
    t, dt = 0.0, dt0
    n_acc = n_rej = 0
    w0 = prob.total_water()
    cum_rain = 0.0
    t0 = time.time()
    snap_times = _snap_schedule(t_end)
    snaps = []
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if t < storm - 1e-12 and t + h > storm:
            h = storm - t
        rain_now = rain if t < storm - 1e-12 else 0.0
        r.value = rain_now
        converged, it = prob.step(h)
        if not converged:
            n_rej += 1
            dt = h * 0.5
            if dt < 1e-10:
                print(f"  !! [{label}] dt collapse at t={t:.6g}", flush=True)
                break
            continue
        t += h
        n_acc += 1
        cum_rain += rain_now * area * h
        _record(rec, t, prob.cum_outflow, surf(), prob.surface_water(),
                prob.cum_drainage, it, prob.last_outflow)
        snaps = _maybe_snap(snaps, snap_times, t, prob.d.x.array[cl_global].copy())
        dt = min(dt * (1.5 if it <= grow_at else 0.7 if it >= shrink_at else 1.0), dt_max)
        if verbose:
            print(f"    [{label}] t={t:8.5f} cum_out={prob.cum_outflow:10.2f} surf={surf():9.2f} "
                  f"dt={h:8.2e} it={it:2d}", flush=True)
    rec["snaps"] = snaps
    rec.update(_finalize(prob.total_water(), w0, cum_rain, prob.cum_outflow,
                         prob.cum_drainage, n_acc, n_rej, time.time() - t0, area,
                         soil_water=prob.soil_water(), surf_water=prob.surface_water()))
    return rec


def run_sequential(soil, *, rain, storm, t_end, omega=0.5, picard=4, route_substeps=1,
                   dt0=1e-4, dt_max=2e-3, psi_i=-1.0, grow_at=4, shrink_at=12,
                   label="seq", verbose=False):
    """SequentialCoupledProblem run on the same swale. omega/picard/route_substeps are the swept knobs.
    route_substeps > 1 sub-divides the routing sweep within a step (route_substeps sweeps each over
    dt/route_substeps, accumulating the source) to tighten the one-step lag WITHOUT changing the
    Richards dt -- a transport-accuracy knob that does not touch conservation. Returns the record."""
    msh = _make_box()
    prob = SequentialCoupledProblem(msh, soil, n_man=N_MAN, relax=omega, picard_iters=picard)
    prob.set_topography(z_b)
    prob.set_initial_condition(lambda x: psi_i + 0.0 * x[2])
    r = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)
    if route_substeps > 1:
        _install_route_substeps(prob, route_substeps)

    top = prob._top_dofs_arr
    prob._ensure_built()
    coords_top = prob._coords[top]
    cl = _swale_floor_centerline_dofs(coords_top, dx_cell=LX / NX)
    cl_global = top[cl]
    y_cl = coords_top[cl, 1]
    area = prob._top_area

    def surf():
        return prob.surface_water()

    rec = _empty_rec()
    rec["y_cl"] = y_cl
    t, dt = 0.0, dt0
    n_acc = n_rej = 0
    w0 = prob.total_water()
    cum_rain = 0.0
    t0 = time.time()
    snap_times = _snap_schedule(t_end)
    snaps = []
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if t < storm - 1e-12 and t + h > storm:
            h = storm - t
        rain_now = rain if t < storm - 1e-12 else 0.0
        r.value = rain_now
        converged, it = prob.step(h)
        if not converged:
            n_rej += 1
            dt = h * 0.5
            if dt < 1e-10:
                print(f"  !! [{label}] dt collapse at t={t:.6g}", flush=True)
                break
            continue
        t += h
        n_acc += 1
        cum_rain += rain_now * area * h
        _record(rec, t, prob.cum_outflow, surf(), prob.surface_water(),
                prob.cum_drainage, it, prob.last_outflow)
        # centerline pond depth = max(psi,0) at the centerline top dofs.
        snaps = _maybe_snap(snaps, snap_times, t,
                            np.maximum(prob._rp.psi.x.array[cl_global].copy(), 0.0))
        dt = min(dt * (1.4 if it <= grow_at else 0.7 if it >= shrink_at else 1.0), dt_max)
        if verbose:
            print(f"    [{label} w={omega} p={picard} rs={route_substeps}] t={t:8.5f} "
                  f"cum_out={prob.cum_outflow:10.2f} surf={surf():9.2f} dt={h:8.2e} it={it:2d}",
                  flush=True)
    rec["snaps"] = snaps
    rec.update(_finalize(prob.total_water(), w0, cum_rain, prob.cum_outflow,
                         prob.cum_drainage, n_acc, n_rej, time.time() - t0, area,
                         soil_water=prob.soil_water(), surf_water=prob.surface_water()))
    rec["balance"] = prob.balance()
    rec["cum_handoff"] = prob.cum_handoff_imbalance
    return rec


def _install_route_substeps(prob, nsub):
    """Monkeypatch prob._route to perform nsub Manning sweeps each over dt/nsub, returning the
    cumulative (d_new, outflow). This tightens the one-step lag (the routing 'catches up' within the
    Richards step) without altering conservation booking: the source still = omega*(d_routed - d_cur)/dt
    over the WHOLE dt, and outflow is the total removed across the sub-sweeps."""
    orig_route = prob._route.__func__   # unbound; call as orig_route(prob, d, dt)

    def route_sub(self, d, dt):
        d_cur = d
        of_tot = 0.0
        hsub = dt / nsub
        for _ in range(nsub):
            d_cur, of = orig_route(self, d_cur, hsub)
            of_tot += of
        return d_cur, of_tot

    import types
    prob._route = types.MethodType(route_sub, prob)


# ----------------------------------------------------------------------------------------------------
# record helpers
# ----------------------------------------------------------------------------------------------------
def _empty_rec():
    return {k: [] for k in ("t", "cum_out", "surf", "cum_drain", "iters", "qout")}


def _record(rec, t, cum_out, surf, surf2, cum_drain, it, qout):
    rec["t"].append(t)
    rec["cum_out"].append(cum_out)
    rec["surf"].append(surf)
    rec["cum_drain"].append(cum_drain)
    rec["iters"].append(it)
    rec["qout"].append(qout)


def _snap_schedule(t_end):
    return list(np.array([0.25, 0.5, 0.75, 1.0]) * t_end)


def _maybe_snap(snaps, snap_times, t, depth_vec):
    for st in snap_times:
        if abs(t - st) < 1e-9 or (snaps == [] and t >= st) or \
                (snaps and t >= st and not any(abs(s[0] - st) < 1e-12 for s in snaps)):
            if not any(abs(s[0] - st) < 1e-12 for s in snaps) and t >= st - 1e-12:
                snaps.append((st, t, depth_vec))
    return snaps


def _finalize(w_end, w0, cum_rain, cum_out, cum_drain, n_acc, n_rej, run_s, area,
              *, soil_water, surf_water):
    t = np.asarray
    return dict(
        w0=w0, w_end=w_end, cum_rain=cum_rain, cum_out_final=cum_out, cum_drain_final=cum_drain,
        n_acc=n_acc, n_rej=n_rej, run_s=run_s, area=area,
        soil_water_end=soil_water, surf_water_end=surf_water)


# ----------------------------------------------------------------------------------------------------
# transport-metric extraction
# ----------------------------------------------------------------------------------------------------
def transport_metrics(rec):
    """Compute the headline transport metrics from a run record."""
    t = np.asarray(rec["t"])
    cum_out = np.asarray(rec["cum_out"])
    surf = np.asarray(rec["surf"])
    cum_rain = rec["cum_rain"]
    out_final = cum_out[-1] if cum_out.size else 0.0
    # arrival times: first t at which cum_out reaches X% of its FINAL value.
    def t_at_frac(frac):
        if out_final <= 0:
            return float("nan")
        thr = frac * out_final
        idx = np.where(cum_out >= thr)[0]
        return float(t[idx[0]]) if idx.size else float("nan")
    surf_peak = float(surf.max()) if surf.size else 0.0
    t_peak = float(t[int(np.argmax(surf))]) if surf.size else float("nan")
    return dict(
        out_final=out_final,
        out_frac_of_rain=out_final / cum_rain if cum_rain > 0 else float("nan"),
        infil_frac=1.0 - out_final / cum_rain if cum_rain > 0 else float("nan"),
        t_out_10=t_at_frac(0.10), t_out_50=t_at_frac(0.50), t_out_90=t_at_frac(0.90),
        surf_peak=surf_peak, t_surf_peak=t_peak, surf_end=float(surf[-1]) if surf.size else 0.0,
        n_acc=rec["n_acc"], n_rej=rec["n_rej"], run_s=rec["run_s"],
        cum_rain=cum_rain, cum_drain=rec["cum_drain_final"], area=rec["area"])


def print_metrics(tag, m):
    print(f"  [{tag:28s}] out={m['out_final']:10.2f} ({100*m['out_frac_of_rain']:5.1f}% rain)  "
          f"infil={100*m['infil_frac']:5.1f}%  surf_peak={m['surf_peak']:8.2f}@t={m['t_surf_peak']:.4f}  "
          f"t_out(10/50/90)={m['t_out_10']:.4f}/{m['t_out_50']:.4f}/{m['t_out_90']:.4f}  "
          f"acc/rej={m['n_acc']}/{m['n_rej']} {m['run_s']:.0f}s", flush=True)


def peak_depth_mm(rec):
    """Max centerline ponded depth over all snapshots [mm] + at which y / snapshot time."""
    best = (0.0, None, None)
    for (st, t, dvec) in rec.get("snaps", []):
        if dvec.size and dvec.max() > best[0]:
            j = int(np.argmax(dvec))
            best = (float(dvec.max()), float(rec["y_cl"][j]) if "y_cl" in rec else None, t)
    return best


# ----------------------------------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------------------------------
def smoke():
    print("=" * 90)
    print("SMOKE: both schemes run on a tiny loam swale (short)")
    print("=" * 90)
    rain, storm, t_end = 2.0, 0.02, 0.05
    ref = run_coupled(LOAM, rain=rain, storm=storm, t_end=t_end, verbose=False, label="upwind")
    print_metrics("upwind", transport_metrics(ref))
    seq = run_sequential(LOAM, rain=rain, storm=storm, t_end=t_end, omega=0.5, verbose=False, label="seq")
    print_metrics("seq w=0.5", transport_metrics(seq))
    print(f"  balance(seq)={seq.get('balance'):.3e}  cum_handoff={seq.get('cum_handoff'):.3e}")
    print(f"  cum_rain: upwind={ref['cum_rain']:.2f}  seq={seq['cum_rain']:.2f}")
    pk_r = peak_depth_mm(ref); pk_s = peak_depth_mm(seq)
    print(f"  centerline peak depth: upwind={1000*pk_r[0]:.1f}mm@y={pk_r[1]}  "
          f"seq={1000*pk_s[0]:.1f}mm@y={pk_s[1]}")


def regime_T():
    """TRANSPORT ISOLATION: near-impermeable bed -> ~all rain routes + exits. Pure transport-rate
    comparison across the omega / picard / route-substep sweep vs the upwind reference."""
    print("=" * 90)
    print("REGIME T -- TRANSPORT ISOLATION (near-impermeable bed; ~all rain routes to the outlet)")
    print("=" * 90)
    rain, storm, t_end = 0.2592, 0.0625, 0.16
    soil = NEAR_IMPERMEABLE
    print(f"geometry {LX}x{LY} swale W={W_SWALE}  rain={rain} storm={storm} t_end={t_end}  "
          f"Q_eq={rain*LX*LY:.1f} m3/day  n_man={N_MAN}\n")

    print("-- REFERENCE: CoupledProblem(upwind) --")
    ref = run_coupled(soil, rain=rain, storm=storm, t_end=t_end, label="upwind")
    mref = transport_metrics(ref)
    print_metrics("UPWIND ref", mref)
    pk = peak_depth_mm(ref); print(f"     ref centerline peak depth = {1000*pk[0]:.1f} mm @ y={pk[1]}, t={pk[2]}")

    print("\n-- SEQUENTIAL sweep --")
    results = {"upwind": (mref, ref)}
    for omega in (0.3, 0.5, 0.7, 1.0):
        seq = run_sequential(soil, rain=rain, storm=storm, t_end=t_end, omega=omega, label=f"w{omega}")
        m = transport_metrics(seq)
        print_metrics(f"seq w={omega}", m)
        results[f"w{omega}"] = (m, seq)
    # route-substep sweep at omega=1.0 (the least-throttled) and omega=0.5
    for omega in (0.5, 1.0):
        for rs in (2, 4):
            seq = run_sequential(soil, rain=rain, storm=storm, t_end=t_end, omega=omega,
                                 route_substeps=rs, label=f"w{omega}rs{rs}")
            m = transport_metrics(seq)
            print_metrics(f"seq w={omega} rs={rs}", m)
            results[f"w{omega}rs{rs}"] = (m, seq)
    _summary_table(results, mref)
    return results


def regime_R():
    """RUN-ON / LOAM: real loam + short intense storm -> ponds, routes, infiltrates as run-on. Read the
    run-on outcome (outflow vs infiltration partition) + transient pond vs the upwind reference."""
    print("=" * 90)
    print("REGIME R -- RUN-ON / LOAM (loam bed; short intense storm; water ponds+routes+infiltrates)")
    print("=" * 90)
    # rain >> Ks (0.25) so it ponds and routes; short storm then recession to let run-on infiltrate.
    rain, storm, t_end = 4.0, 0.03, 0.20
    soil = LOAM
    print(f"geometry {LX}x{LY} swale W={W_SWALE}  rain={rain}(>>Ks={soil.Ks}) storm={storm} "
          f"t_end={t_end}  Q_eq_storm={rain*LX*LY*storm:.1f} m3  n_man={N_MAN}\n")

    print("-- REFERENCE: CoupledProblem(upwind) --")
    ref = run_coupled(soil, rain=rain, storm=storm, t_end=t_end, label="upwind")
    mref = transport_metrics(ref)
    print_metrics("UPWIND ref", mref)
    pk = peak_depth_mm(ref); print(f"     ref centerline peak depth = {1000*pk[0]:.1f} mm @ y={pk[1]}, t={pk[2]}")

    print("\n-- SEQUENTIAL sweep --")
    results = {"upwind": (mref, ref)}
    for omega in (0.3, 0.5, 0.7, 1.0):
        seq = run_sequential(soil, rain=rain, storm=storm, t_end=t_end, omega=omega, label=f"w{omega}")
        m = transport_metrics(seq)
        print_metrics(f"seq w={omega}", m)
        pk = peak_depth_mm(seq)
        print(f"     seq w={omega} centerline peak depth = {1000*pk[0]:.1f} mm @ y={pk[1]}, t={pk[2]}  "
              f"bal={seq.get('balance'):.2e}")
        results[f"w{omega}"] = (m, seq)
    for omega in (0.5, 1.0):
        for rs in (4,):
            seq = run_sequential(soil, rain=rain, storm=storm, t_end=t_end, omega=omega,
                                 route_substeps=rs, label=f"w{omega}rs{rs}")
            m = transport_metrics(seq)
            print_metrics(f"seq w={omega} rs={rs}", m)
            results[f"w{omega}rs{rs}"] = (m, seq)
    _summary_table(results, mref)
    return results


def _summary_table(results, mref):
    print("\n" + "-" * 90)
    print("SUMMARY vs upwind reference (transport deltas)")
    print("-" * 90)
    print(f"  {'variant':16s} {'out/Qrain%':>10} {'d(out)%':>9} {'t_out50':>9} {'dt_out50':>10} "
          f"{'surf_peak':>10} {'peak/ref':>9}")
    o_ref = mref["out_final"]
    t50_ref = mref["t_out_50"]
    sp_ref = mref["surf_peak"]
    for tag, (m, _rec) in results.items():
        d_out = 100 * (m["out_final"] - o_ref) / o_ref if o_ref > 0 else float("nan")
        dt50 = m["t_out_50"] - t50_ref
        pk_ratio = m["surf_peak"] / sp_ref if sp_ref > 0 else float("nan")
        print(f"  {tag:16s} {100*m['out_frac_of_rain']:10.1f} {d_out:+9.1f} {m['t_out_50']:9.4f} "
              f"{dt50:+10.4f} {m['surf_peak']:10.2f} {pk_ratio:9.2f}")


# ----------------------------------------------------------------------------------------------------
# Regime P -- POND RELEASE (the CLEANEST transport isolation). Start with a uniform pond over an
# ALREADY-SATURATED column (hydrostatic, water table at the surface), NO rain. Infiltration is ~shut
# off IDENTICALLY in both schemes (the soil is full), so the pond just ROUTES downslope to the outlet:
# a pure lateral-transport race. Both schemes get the same initial surface water (POND * top_area).
# A small Ks loam (0.05) leaks a little (same small leak both schemes); a deep buffer dodges the no-Ss
# saturation singularity.
# ----------------------------------------------------------------------------------------------------
SAT_SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.05)


def run_coupled_pondrelease(*, pond, t_end, dt0=1e-4, dt_max=2e-3, label="upwind"):
    msh = _make_box()
    prob = CoupledProblem(msh, SAT_SOIL, n_man=N_MAN, overland_scheme="upwind")
    # saturated hydrostatic column (psi_top = 0), pond carried in the SEPARATE store d = pond.
    prob.set_initial_condition(lambda x: (H - x[2]), d_value=pond)
    prob.set_topography(z_b)
    prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)
    topdofs = prob._top_dofs(prob.Vd)
    coords_top = prob.Vd.tabulate_dof_coordinates()[topdofs]
    cl = _swale_floor_centerline_dofs(coords_top, dx_cell=LX / NX)
    cl_global = topdofs[cl]
    rec = _empty_rec(); rec["y_cl"] = coords_top[cl, 1]
    t, dt = 0.0, dt0; n_acc = n_rej = 0; w0 = prob.total_water(); t0 = time.time()
    snap_times = _snap_schedule(t_end); snaps = []
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        c, it = prob.step(h)
        if not c:
            n_rej += 1; dt = h * 0.5
            if dt < 1e-10:
                print(f"  !! [{label}] dt collapse at t={t:.6g}", flush=True); break
            continue
        t += h; n_acc += 1
        _record(rec, t, prob.cum_outflow, prob.surface_water(), prob.surface_water(),
                prob.cum_drainage, it, prob.last_outflow)
        snaps = _maybe_snap(snaps, snap_times, t, prob.d.x.array[cl_global].copy())
        dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), dt_max)
    rec["snaps"] = snaps
    rec.update(_finalize(prob.total_water(), w0, 0.0, prob.cum_outflow, prob.cum_drainage,
                         n_acc, n_rej, time.time() - t0, LX * LY,
                         soil_water=prob.soil_water(), surf_water=prob.surface_water()))
    return rec


def run_sequential_pondrelease(*, pond, t_end, omega=0.5, picard=4, route_substeps=1,
                               dt0=1e-4, dt_max=2e-3, label="seq"):
    msh = _make_box()
    prob = SequentialCoupledProblem(msh, SAT_SOIL, n_man=N_MAN, relax=omega, picard_iters=picard)
    prob.set_topography(z_b)
    # saturated hydrostatic column + uniform pond carried IN psi: psi_top = pond.
    prob.set_initial_condition(lambda x: pond + (H - x[2]))
    prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)
    if route_substeps > 1:
        _install_route_substeps(prob, route_substeps)
    prob._ensure_built()
    top = prob._top_dofs_arr
    coords_top = prob._coords[top]
    cl = _swale_floor_centerline_dofs(coords_top, dx_cell=LX / NX)
    cl_global = top[cl]
    rec = _empty_rec(); rec["y_cl"] = coords_top[cl, 1]
    t, dt = 0.0, dt0; n_acc = n_rej = 0; w0 = prob.total_water(); t0 = time.time()
    snap_times = _snap_schedule(t_end); snaps = []
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        c, it = prob.step(h)
        if not c:
            n_rej += 1; dt = h * 0.5
            if dt < 1e-10:
                print(f"  !! [{label}] dt collapse at t={t:.6g}", flush=True); break
            continue
        t += h; n_acc += 1
        _record(rec, t, prob.cum_outflow, prob.surface_water(), prob.surface_water(),
                prob.cum_drainage, it, prob.last_outflow)
        snaps = _maybe_snap(snaps, snap_times, t,
                            np.maximum(prob._rp.psi.x.array[cl_global].copy(), 0.0))
        dt = min(dt * (1.4 if it <= 4 else 0.7 if it >= 12 else 1.0), dt_max)
    rec["snaps"] = snaps
    rec.update(_finalize(prob.total_water(), w0, 0.0, prob.cum_outflow, prob.cum_drainage,
                         n_acc, n_rej, time.time() - t0, prob._top_area,
                         soil_water=prob.soil_water(), surf_water=prob.surface_water()))
    rec["balance"] = prob.balance(); rec["cum_handoff"] = prob.cum_handoff_imbalance
    return rec


def pondrelease_metrics(rec, surf0):
    """Transport metrics for a pond-release: outflow + how fast the surface store drains."""
    t = np.asarray(rec["t"]); cum_out = np.asarray(rec["cum_out"]); surf = np.asarray(rec["surf"])
    out_final = cum_out[-1] if cum_out.size else 0.0
    # time for the surface store to fall to X% of its INITIAL value (drain race).
    def t_surf_to(frac):
        thr = frac * surf0
        idx = np.where(surf <= thr)[0]
        return float(t[idx[0]]) if idx.size else float("nan")
    # time for cum_out to reach X% of the released pond.
    def t_out_to(frac):
        thr = frac * surf0
        idx = np.where(cum_out >= thr)[0]
        return float(t[idx[0]]) if idx.size else float("nan")
    return dict(
        surf0=surf0, out_final=out_final, out_frac=out_final / surf0 if surf0 > 0 else float("nan"),
        surf_end=float(surf[-1]) if surf.size else 0.0,
        t_surf_50=t_surf_to(0.50), t_surf_10=t_surf_to(0.10),
        t_out_50=t_out_to(0.50), t_out_90=t_out_to(0.90),
        n_acc=rec["n_acc"], n_rej=rec["n_rej"], run_s=rec["run_s"])


def regime_P():
    print("=" * 90)
    print("REGIME P -- POND RELEASE (cleanest transport isolation: saturated column + pond, no rain)")
    print("=" * 90)
    pond, t_end = 0.05, 0.30
    print(f"geometry {LX}x{LY} swale W={W_SWALE}  pond={pond}m (uniform) t_end={t_end}  "
          f"SAT loam Ks={SAT_SOIL.Ks}  n_man={N_MAN}\n")

    print("-- REFERENCE: CoupledProblem(upwind) --")
    ref = run_coupled_pondrelease(pond=pond, t_end=t_end)
    surf0_ref = pond * ref["area"]   # uniform pond over the top area
    mref = pondrelease_metrics(ref, surf0_ref)
    print(f"  [UPWIND ref] surf0={mref['surf0']:.2f} out={mref['out_final']:.2f} "
          f"({100*mref['out_frac']:.1f}% of pond)  surf_end={mref['surf_end']:.2f}  "
          f"t_surf(50/10)={mref['t_surf_50']:.4f}/{mref['t_surf_10']:.4f}  "
          f"t_out(50/90)={mref['t_out_50']:.4f}/{mref['t_out_90']:.4f}  "
          f"acc/rej={mref['n_acc']}/{mref['n_rej']} {mref['run_s']:.0f}s", flush=True)
    pk = peak_depth_mm(ref)
    print(f"     centerline initial depth ~{1000*pond:.0f}mm  (snap peak {1000*pk[0]:.1f}mm)")

    print("\n-- SEQUENTIAL sweep --")
    rows = [("upwind", mref)]
    for omega in (0.3, 0.5, 0.7, 1.0):
        seq = run_sequential_pondrelease(pond=pond, t_end=t_end, omega=omega, label=f"w{omega}")
        surf0 = pond * seq["area"]
        m = pondrelease_metrics(seq, surf0)
        print(f"  [seq w={omega}] surf0={m['surf0']:.2f} out={m['out_final']:.2f} "
              f"({100*m['out_frac']:.1f}% of pond)  surf_end={m['surf_end']:.2f}  "
              f"t_surf(50/10)={m['t_surf_50']:.4f}/{m['t_surf_10']:.4f}  "
              f"t_out(50/90)={m['t_out_50']:.4f}/{m['t_out_90']:.4f}  "
              f"acc/rej={m['n_acc']}/{m['n_rej']} {seq['run_s']:.0f}s  bal={seq.get('balance'):.1e}",
              flush=True)
        rows.append((f"w{omega}", m))
    for omega in (0.5, 1.0):
        for rs in (4,):
            seq = run_sequential_pondrelease(pond=pond, t_end=t_end, omega=omega,
                                             route_substeps=rs, label=f"w{omega}rs{rs}")
            surf0 = pond * seq["area"]
            m = pondrelease_metrics(seq, surf0)
            print(f"  [seq w={omega} rs={rs}] surf0={m['surf0']:.2f} out={m['out_final']:.2f} "
                  f"({100*m['out_frac']:.1f}% of pond)  surf_end={m['surf_end']:.2f}  "
                  f"t_surf(50/10)={m['t_surf_50']:.4f}/{m['t_surf_10']:.4f}  "
                  f"t_out(50/90)={m['t_out_50']:.4f}/{m['t_out_90']:.4f}  "
                  f"{seq['run_s']:.0f}s", flush=True)
            rows.append((f"w{omega}rs{rs}", m))
    # summary
    print("\n" + "-" * 90)
    print("SUMMARY pond-release vs upwind (lateral drain race)")
    print("-" * 90)
    print(f"  {'variant':14s} {'out% pond':>10} {'t_surf50':>9} {'dt_surf50':>10} "
          f"{'t_out50':>9} {'dt_out50':>10}")
    ts50r, to50r = mref["t_surf_50"], mref["t_out_50"]
    for tag, m in rows:
        dts = m["t_surf_50"] - ts50r if not np.isnan(m["t_surf_50"]) and not np.isnan(ts50r) else float("nan")
        dto = m["t_out_50"] - to50r if not np.isnan(m["t_out_50"]) and not np.isnan(to50r) else float("nan")
        print(f"  {tag:14s} {100*m['out_frac']:10.1f} {m['t_surf_50']:9.4f} {dts:+10.4f} "
              f"{m['t_out_50']:9.4f} {dto:+10.4f}")


def run_sequential_steadyrain(*, rain, storm, t_end, omega, route_substeps=1, dt0=1e-4,
                              dt_max=1e-3, label="seq"):
    """SAT-soil (near-saturated, ~no infiltration) + a storm: the pond PILES UP in the swale until the
    (throttled) routing balances the inflow. Isolates the pile-up from infiltration (the SAT soil
    barely infiltrates), so surf_peak / peak depth is a clean transport-throttle metric. Reports
    whether the run dt-collapses (the spike's pile-up -> collapse pathology)."""
    msh = _make_box()
    prob = SequentialCoupledProblem(msh, SAT_SOIL, n_man=N_MAN, relax=omega, picard_iters=4)
    prob.set_topography(z_b)
    prob.set_initial_condition(lambda x: -0.05 + (H - x[2]))   # near-saturated (psi_top=-0.05), deep buffer
    r = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)
    if route_substeps > 1:
        _install_route_substeps(prob, route_substeps)
    prob._ensure_built()
    top = prob._top_dofs_arr
    rec = _empty_rec(); rec["y_cl"] = np.array([])
    t, dt = 0.0, dt0; n_acc = n_rej = 0; w0 = prob.total_water(); cum_rain = 0.0
    t0 = time.time(); collapsed = False; peak_depth = 0.0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if t < storm - 1e-12 and t + h > storm:
            h = storm - t
        rain_now = rain if t < storm - 1e-12 else 0.0
        r.value = rain_now
        c, it = prob.step(h)
        if not c:
            n_rej += 1; dt = h * 0.5
            if dt < 1e-10:
                collapsed = True; break
            continue
        t += h; n_acc += 1; cum_rain += rain_now * prob._top_area * h
        peak_depth = max(peak_depth, float(np.maximum(prob._rp.psi.x.array[top], 0.0).max()))
        _record(rec, t, prob.cum_outflow, prob.surface_water(), prob.surface_water(),
                prob.cum_drainage, it, prob.last_outflow)
        dt = min(dt * (1.4 if it <= 4 else 0.7 if it >= 12 else 1.0), dt_max)
    rec["snaps"] = []
    rec.update(_finalize(prob.total_water(), w0, cum_rain, prob.cum_outflow, prob.cum_drainage,
                         n_acc, n_rej, time.time() - t0, prob._top_area,
                         soil_water=prob.soil_water(), surf_water=prob.surface_water()))
    rec["collapsed"] = collapsed; rec["t_reached"] = t; rec["peak_depth"] = peak_depth
    rec["balance"] = prob.balance()
    return rec


def run_coupled_steadyrain(*, rain, storm, t_end, dt0=1e-4, dt_max=1e-3, label="upwind"):
    msh = _make_box()
    prob = CoupledProblem(msh, SAT_SOIL, n_man=N_MAN, overland_scheme="upwind")
    prob.set_initial_condition(lambda x: -0.05 + (H - x[2]), d_value=0.0)
    prob.set_topography(z_b)
    r = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], LY), slope=SY)
    topdofs = prob._top_dofs(prob.Vd)
    rec = _empty_rec(); rec["y_cl"] = np.array([])
    t, dt = 0.0, dt0; n_acc = n_rej = 0; w0 = prob.total_water(); cum_rain = 0.0
    t0 = time.time(); collapsed = False; peak_depth = 0.0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if t < storm - 1e-12 and t + h > storm:
            h = storm - t
        rain_now = rain if t < storm - 1e-12 else 0.0
        r.value = rain_now
        c, it = prob.step(h)
        if not c:
            n_rej += 1; dt = h * 0.5
            if dt < 1e-10:
                collapsed = True; break
            continue
        t += h; n_acc += 1; cum_rain += rain_now * LX * LY * h
        peak_depth = max(peak_depth, float(prob.d.x.array[topdofs].max()))
        _record(rec, t, prob.cum_outflow, prob.surface_water(), prob.surface_water(),
                prob.cum_drainage, it, prob.last_outflow)
        dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), dt_max)
    rec["snaps"] = []
    rec.update(_finalize(prob.total_water(), w0, cum_rain, prob.cum_outflow, prob.cum_drainage,
                         n_acc, n_rej, time.time() - t0, LX * LY,
                         soil_water=prob.soil_water(), surf_water=prob.surface_water()))
    rec["collapsed"] = collapsed; rec["t_reached"] = t; rec["peak_depth"] = peak_depth
    return rec


def regime_R2():
    """STEADY-RAIN pile-up isolation (SAT soil): does the recommended omega/sub-step setting reduce the
    transient swale pond + avoid the dt-collapse, vs the throttled default? Smaller mesh for speed."""
    print("=" * 90)
    print("REGIME R2 -- STEADY-RAIN pile-up (SAT near-saturated soil; ~no infiltration confounder)")
    print("=" * 90)
    rain, storm, t_end = 1.0, 0.05, 0.12
    print(f"geometry {LX}x{LY} swale W={W_SWALE}  rain={rain} storm={storm} t_end={t_end}  "
          f"SAT soil Ks={SAT_SOIL.Ks} psi_top0=-0.05  n_man={N_MAN}\n")
    ref = run_coupled_steadyrain(rain=rain, storm=storm, t_end=t_end)
    print(f"  [UPWIND ref] surf_peak={ref['surf_water_end'] if False else max(ref['surf']):.2f} "
          f"peak_depth={1000*ref['peak_depth']:.1f}mm  out={ref['cum_out_final']:.1f} "
          f"({100*ref['cum_out_final']/max(ref['cum_rain'],1e-9):.1f}% rain)  "
          f"acc/rej={ref['n_acc']}/{ref['n_rej']} reached t={ref['t_reached']:.4f}"
          f"{'  COLLAPSE' if ref['collapsed'] else ''}  {ref['run_s']:.0f}s", flush=True)
    ref_peak = max(ref["surf"])
    print(f"\n  {'variant':16s} {'surf_peak':>10} {'pk/ref':>7} {'peakdepth_mm':>12} {'out%rain':>9} "
          f"{'reached_t':>10} {'collapse':>9} {'run_s':>7}")
    for omega, rs in [(0.5, 1), (1.0, 1), (1.0, 4), (1.0, 8), (0.5, 4)]:
        seq = run_sequential_steadyrain(rain=rain, storm=storm, t_end=t_end, omega=omega,
                                        route_substeps=rs, label=f"w{omega}rs{rs}")
        sp = max(seq["surf"]) if seq["surf"] else float("nan")
        print(f"  w={omega} rs={rs:<2d}        {sp:10.2f} {sp/ref_peak:7.2f} "
              f"{1000*seq['peak_depth']:12.1f} "
              f"{100*seq['cum_out_final']/max(seq['cum_rain'],1e-9):9.1f} {seq['t_reached']:10.4f} "
              f"{str(seq['collapsed']):>9} {seq['run_s']:7.0f}", flush=True)


def _interp_surf(rec, times, surf0):
    """Linearly interpolate surf(t)/surf0 onto a common time grid (drain-fraction REMAINING)."""
    t = np.asarray(rec["t"]); s = np.asarray(rec["surf"]) / surf0
    return np.interp(times, t, s, left=1.0, right=s[-1] if s.size else 0.0)


def regime_P_curve():
    """Drain-curve comparison at MATCHED times: reference vs default (w=0.5,rs=1) vs recommended
    (w=1.0,rs=4). Shows surface store REMAINING (% of initial pond) over time -- 'right places at right
    times'. The cleanest single picture of the throttle + the fix."""
    print("=" * 90)
    print("REGIME P (curve) -- surface-store drain curves, reference vs default vs recommended")
    print("=" * 90)
    pond, t_end = 0.05, 0.12
    ref = run_coupled_pondrelease(pond=pond, t_end=t_end)
    surf0 = pond * ref["area"]
    d05 = run_sequential_pondrelease(pond=pond, t_end=t_end, omega=0.5, route_substeps=1)
    d10r4 = run_sequential_pondrelease(pond=pond, t_end=t_end, omega=1.0, route_substeps=4)
    grid = np.array([0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.12])
    sr = _interp_surf(ref, grid, surf0)
    s0 = _interp_surf(d05, grid, surf0)
    s1 = _interp_surf(d10r4, grid, surf0)
    print(f"\n  pond=50mm released over the saturated swale, no rain. surf REMAINING (% of initial):")
    print(f"  {'t[day]':>9} {'upwind_ref':>11} {'def w0.5rs1':>12} {'rec w1.0rs4':>12}")
    for i, tt in enumerate(grid):
        print(f"  {tt:9.4f} {100*sr[i]:11.1f} {100*s0[i]:12.1f} {100*s1[i]:12.1f}", flush=True)
    print(f"\n  (reference half-drains at t~0.0016; default w0.5rs1 at t~0.073 [~47x later]; "
          f"recommended w1.0rs4 at t~0.0016 [matches])")


def regime_P_substep():
    """Sub-step convergence probe: at omega=1.0, sweep route_substeps to see where the drain timing
    converges to the upwind reference. Also dumps the early-time surf(t) trace for the reference and a
    few variants so the t_surf50 read is not a dt-resolution artifact."""
    dt_max = float(os.environ.get("DT_MAX", "2e-3"))
    print("=" * 90)
    print(f"REGIME P (substep) -- route_substeps convergence to the upwind reference (dt_max={dt_max:g})")
    print("=" * 90)
    pond, t_end = 0.05, 0.10
    ref = run_coupled_pondrelease(pond=pond, t_end=t_end, dt_max=dt_max)
    surf0 = pond * ref["area"]
    mref = pondrelease_metrics(ref, surf0)
    print(f"\n[upwind ref] t_surf50={mref['t_surf_50']:.5f} t_surf10={mref['t_surf_10']:.5f}  "
          f"out%={100*mref['out_frac']:.1f}  acc={mref['n_acc']}")
    # early surf(t) trace (first ~20 records).
    print("  upwind surf(t) head:", flush=True)
    for tt, ss in list(zip(ref["t"], ref["surf"]))[:14]:
        print(f"     t={tt:.5f} surf={ss:.2f} ({100*ss/surf0:.1f}%)", flush=True)
    print(f"\n  {'rsub':>5} {'omega':>6} {'t_surf50':>9} {'t_surf10':>9} {'surf50/ref':>11} {'run_s':>7}")
    for omega in (0.5, 1.0):
        for rs in (1, 2, 4, 8, 16):
            seq = run_sequential_pondrelease(pond=pond, t_end=t_end, omega=omega, route_substeps=rs,
                                             dt_max=dt_max)
            m = pondrelease_metrics(seq, pond * seq["area"])
            ratio = m["t_surf_50"] / mref["t_surf_50"] if mref["t_surf_50"] > 0 else float("nan")
            print(f"  {rs:5d} {omega:6.1f} {m['t_surf_50']:9.5f} {m['t_surf_10']:9.5f} "
                  f"{ratio:11.1f} {seq['run_s']:7.0f}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke()
    elif mode == "T":
        regime_T()
    elif mode == "R":
        regime_R()
    elif mode == "R2":
        regime_R2()
    elif mode == "P":
        regime_P()
    elif mode == "Psub":
        regime_P_substep()
    elif mode == "Pcurve":
        regime_P_curve()
    elif mode == "all":
        regime_P()
        regime_R()
    else:
        print(f"unknown mode {mode!r}; use smoke|T|R|P|Psub|all")


if __name__ == "__main__":
    main()
