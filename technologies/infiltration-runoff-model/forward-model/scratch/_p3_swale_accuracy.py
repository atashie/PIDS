"""P3 Part A -- resolved-swale ABSOLUTE accuracy of the COUPLED upwind overland scheme (depth field).

THE KEY SUBTLETY (parent plan 8.7 / P3 Part A; Codex-reviewed -- do NOT re-misuse the outlet):
in the COUPLED [psi,d,lam] engine the outlet `outflow_rate()` is the consistent codim-2 ds-integral
AND it is conservation-FORCED to ~Q_eq at the storm plateau (it IS the outlet sink; summing the
d-residual telescopes the edge flux to zero, leaving int q_out ds = rain*area - dStorage - int lam).
So `outflow_rate() -> Q_eq` proves CONSERVATION, not accuracy, and it CANNOT expose the
kink-vs-swale difference (which lives in the DEPTH FIELD). Therefore Part A measures accuracy against
EXTERNAL references of the DEPTH FIELD:

  (A1.1) the conservation-forced trap: outflow_rate()/Q_eq ~ 1 for BOTH the kink-V (W=0) and a
         resolved swale (W>0) on a near-impermeable bed -- identical, regardless of resolution.
  (A1.2) operator-equivalence: the coupled-upwind surface depth field == the STANDALONE
         UpwindOverlandProblem depth field on the SAME resolved swale (near-impermeable, inlet OFF)
         -- the coupling adds only the small lambda infiltration -> carries B5b's standalone
         resolved-swale accuracy (consistent integral ~0.99) into the coupled engine.
  (A2)   depth-field absolute accuracy: the swale-FLOOR depth -> the analytic Manning normal-depth
         d_n = (q_w*n/(86400*sqrt(S_v)))^(3/5), q_w = Q_eq/W, error SHRINKING with refinement; the
         kink-V (W=1 cell) floor depth GROWS with refinement (the B5b measure-zero-channel artifact).

GEOMETRY (reused from scratch/_b5b_valley_concentration.py + scratch/_tiltedv_diag.py): a tilted-V
with a flat-bottomed valley floor of width W -- z_b = SY*(LY-y) + SX*max(|x-XC|-W/2, 0). W=0 is the
canonical KINK V (the measure-zero channel). The COUPLED engine puts this z_b on the top facet of a
3-D box host (realization A); the STANDALONE solver puts it on a 2-D rectangle (the surface IS the
domain). All Part-A runs: near-impermeable bed (Ks=1e-3 -> outflow ~ rain*area, a clean reference) +
inlet OFF + the surface export budget reported (cum_rain - cum_outflow - dStorage ~ 0).

Run (pids-fem, from forward-model):
  PYTHONPATH=. python scratch/_p3_swale_accuracy.py          # authoritative sweep (slow, tilted-V)
  FAST=1 PYTHONPATH=. python scratch/_p3_swale_accuracy.py    # quick small-swale validation
"""
from __future__ import annotations

import os
import time

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.overland_upwind import UpwindOverlandProblem

SECONDS_PER_DAY = 86400.0

# near-impermeable bed (matches scratch/_tiltedv_diag.py): Ks=1e-3 m/day << rain so outflow ~ rain*area
NEAR_IMPERMEABLE = VanGenuchten(theta_r=0.10, theta_s=0.40, alpha=2.0, n=2.0, Ks=1.0e-3)

RAIN = 0.2592      # m/day (canonical tilted-V storm)
STORM = 0.0625     # day


def make_zb(W, LX, LY, XC, SX, SY):
    """Bed elevation z_b(x,y). W=0 -> the canonical KINK V (z_b = SY*(LY-y) + SX*|x-XC|); W>0 -> a
    FLAT-bottomed valley of floor width W (cross-slope SX only beyond |x-XC|=W/2)."""
    half = 0.5 * W

    def zb(x):
        return SY * (LY - x[1]) + SX * np.maximum(np.abs(x[0] - XC) - half, 0.0)

    return zb


def normal_depth(Q_eq, W, n_man, S_v):
    """Analytic Manning normal-depth of a flat floor of width W carrying the catchment discharge:
       q_w = Q_eq / W  [m^2/day per unit floor width];  d_n = (q_w*n/(86400*sqrt(S_v)))^(3/5).
    (The 86400 converts the SI Manning conveyance to the engine's m^2/day, exactly as the outlet
    sink q_out = 86400*(1/n)*d^(5/3)*sqrt(S) does -- d_n is its steady inverse at q=q_w.)"""
    q_w = Q_eq / W
    return (q_w * n_man / (SECONDS_PER_DAY * np.sqrt(S_v))) ** (3.0 / 5.0)


def plane_normal_depth(rain, LY, n_man, S_v):
    """Analytic Manning normal depth at the outlet of a tilted PLANE (no cross-slope / no convergence)
    under uniform rain: per-width outlet discharge q = rain*LY, d_n = (q*n/(86400*sqrt(S_v)))^(3/5).
    On a plane the steady depth IS uniform across the slope, so this is the clean DIRECT absolute
    reference the convergent tilted-V floor (edge-heavy, not uniform) is not."""
    q = rain * LY
    return (q * n_man / (SECONDS_PER_DAY * np.sqrt(S_v))) ** (3.0 / 5.0)


def outlet_row_mean_depth(info, nx, ny):
    """Mean top-surface depth over the outlet row (y=LY), INTERIOR in x (exclude the x=0/x=LX no-flux
    wall columns). On a tilted plane this is the uniform Manning normal depth."""
    LX, LY = info["LX"], info["LY"]
    dx, dy = LX / nx, LY / ny
    x, y, d = info["top_x"], info["top_y"], info["d_top"]
    m = (y > LY - 0.5 * dy - 1e-9) & (x > dx - 1e-9) & (x < LX - dx + 1e-9)
    return float(np.mean(d[m])) if m.any() else float("nan")


def consistent_outflow(prob, msh, n_man, S_v, LY):
    """B5b's FREE consistent ds-integral discharge over y=LY: int 86400*(1/n)*max(d,0)^(5/3)*sqrt(S_v) ds
    (the P1-interpolated functional). On the STANDALONE solver the resolved swale reads ~0.99*Q_eq -- the
    GENUINE discharge-accuracy measure (the lumped nodal sink is forced to Q_eq; this free integral is the
    independent reading, B5b/parent 8.7). Operator-equivalence carries it into the coupled engine, whose
    OUTLET SINK already IS this consistent integral (-> conservation-forced, so no free coupled reading)."""
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = np.sort(dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], LY)))
    ft = dmesh.meshtags(msh, fdim, facets, np.full(facets.shape, 1, dtype=np.int32))
    ds_out = ufl.Measure("ds", domain=msh, subdomain_data=ft)(1)
    d_pos = ufl.max_value(prob.d, 0.0)
    form = fem.form(SECONDS_PER_DAY * (1.0 / n_man) * d_pos ** (5.0 / 3.0) * ufl.sqrt(S_v) * ds_out)
    return float(fem.assemble_scalar(form))


def measure_floor_depth(top_x, top_y, d_top, XC, W, LX, LY, nx, ny):
    """Swale-FLOOR depth at the plateau: the centerline / floor-mean depth at the OUTLET ROW (y=LY,
    where the floor carries ~Q_eq), AWAY from the cross-slope floor edges (Codex: not a single
    extreme node). For W=0 (kink) the 'floor' is the single valley column |x-XC|<dx/2."""
    dx, dy = LX / nx, LY / ny
    outlet = top_y > LY - 0.5 * dy - 1e-9                    # most-downstream node row
    centerline = np.abs(top_x - XC) < 0.5 * dx + 1e-9        # the valley column
    if W <= 0.0:
        floor = centerline                                  # kink: the measure-zero valley line
    else:
        # interior of the flat floor: drop the outermost ~1 cell near the slope break |x-XC|=W/2
        floor = np.abs(top_x - XC) <= max(0.5 * W - dx, 0.5 * dx) + 1e-9
    m_out = outlet & floor
    m_cl = outlet & centerline
    return dict(
        floor_mean=float(np.mean(d_top[m_out])) if m_out.any() else float("nan"),
        floor_centerline=float(np.mean(d_top[m_cl])) if m_cl.any() else float("nan"),
        n_floor=int(m_out.sum()),
        cells_across=W / LX * nx,
    )


def _run_controller(step_fn, outflow_fn, water_fn, set_rain, area, rain, t_end,
                    dt0, dt_max, grow_at, shrink_at, plateau_frac, verbose, label):
    """Shared adaptive backward-Euler driver (mirrors scratch/_tiltedv_diag.py's controller). Rain
    is ON throughout (drive to + hold the storm plateau); averages outflow over the last (1-frac)."""
    t, dt = 0.0, dt0
    n_acc = n_rej = 0
    w0 = water_fn()
    cum_rain = cum_out = 0.0
    plateau_lo = plateau_frac * t_end
    q_acc = []
    t0 = time.time()
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        set_rain(rain)
        converged, it = step_fn(h)
        if converged:
            cum_rain += h * rain * area
            q = outflow_fn()
            cum_out += h * q
            t += h
            n_acc += 1
            if t >= plateau_lo - 1e-15:
                q_acc.append(q)
            dt = min(dt * (1.5 if it <= grow_at else 0.7 if it >= shrink_at else 1.0), dt_max)
            if verbose:
                print(f"    [{label}] t={t:9.6f} q/Qeq={q/(rain*area):7.4f} dt={h:8.2e} it={it:2d}",
                      flush=True)
        else:
            n_rej += 1
            dt = h * 0.5
            if dt < 1e-12:
                print(f"    !! [{label}] dt collapse at t={t:.6g}", flush=True)
                break
    return dict(w0=w0, wend=water_fn(), cum_rain=cum_rain, cum_out=cum_out,
                q_plateau=float(np.mean(q_acc)) if q_acc else float("nan"),
                n_acc=n_acc, n_rej=n_rej, run_s=time.time() - t0, n_plateau=len(q_acc))


def outlet_locator(outlet, LX, LY, XC, W, nx):
    """The downstream (y=LY) outlet locator. outlet='floor' (DEFAULT, the PIDS swale): the outlet is
    the swale FLOOR only -- the side slopes drain INTO the swale and exit through it, so the floor
    carries the FULL catchment discharge Q_eq and q_w=Q_eq/W is the correct normal-depth reference
    (this is the product picture: the swale IS the drainage line). outlet='edge' (the canonical
    tilted-V benchmark, Part C vs ParFlow): the full y=LY edge, so the side slopes discharge directly
    -- then the floor carries < Q_eq and q_w=Q_eq/W over-predicts. For W=0 (kink) the floor is the
    single valley column (>= 1 cell)."""
    if outlet == "edge":
        return lambda x: np.isclose(x[1], LY)
    half = max(0.5 * W, LX / nx)   # >= 1 cell so the band captures a 3-D ridge EDGE (both endpoints in)
    return lambda x: np.isclose(x[1], LY) & (np.abs(x[0] - XC) <= half + 1e-9)


def drive_coupled_swale(W, nx, ny, nz=3, *, LX=1620.0, LY=1000.0, H=2.0, SX=0.05, SY=0.02,
                        n_man=0.015, soil=None, rain=RAIN, t_end=STORM, dt0=1e-5, dt_max=1e-3,
                        grow_at=3, shrink_at=8, scheme="upwind", pos_tol=None, plateau_frac=0.6,
                        outlet="floor", verbose=False):
    """Drive the COUPLED [psi,d,lam] solver (overland `scheme`) on a tilted-V swale of floor width W
    to the storm plateau. near-impermeable bed + inlet OFF. Returns (prob, msh, info) with the
    plateau outflow, Q_eq, the top-surface depth field, the export budget, timing, and min depth."""
    if soil is None:
        soil = NEAR_IMPERMEABLE
    XC = LX / 2.0
    area = LX * LY
    Q_eq = rain * area
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [LX, LY, H]], [nx, ny, nz])
    prob = CoupledProblem(msh, soil, n_man=n_man, overland_scheme=scheme)
    if pos_tol is not None:
        prob._upwind_pos_tol = pos_tol
    prob.set_initial_condition(lambda x: -1.0 + 0.0 * x[2], d_value=0.0)
    prob.set_topography(make_zb(W, LX, LY, XC, SX, SY))
    r = prob.add_rain(0.0)
    prob.add_outflow_bc(outlet_locator(outlet, LX, LY, XC, W, nx), slope=SY)

    res = _run_controller(
        step_fn=prob.step,
        outflow_fn=lambda: prob.last_outflow,         # SOLVED-state pre-limiter outflow (books-consistent)
        water_fn=prob.total_water,
        set_rain=lambda val: setattr(r, "value", val),
        area=area, rain=rain, t_end=t_end, dt0=dt0, dt_max=dt_max, grow_at=grow_at,
        shrink_at=shrink_at, plateau_frac=plateau_frac, verbose=verbose, label="coupled")

    topdofs = prob._top_dofs(prob.Vd)
    coords = prob.Vd.tabulate_dof_coordinates()[topdofs]
    d_top = prob.d.x.array[topdofs].copy()
    dW = res["wend"] - res["w0"]
    info = dict(
        W=W, nx=nx, ny=ny, nz=nz, LX=LX, LY=LY, XC=XC, SX=SX, SY=SY, n_man=n_man, area=area,
        Q_eq=Q_eq, q_plateau=res["q_plateau"], q_over_Qeq=res["q_plateau"] / Q_eq,
        top_x=coords[:, 0].copy(), top_y=coords[:, 1].copy(), d_top=d_top,
        d_min=float(prob.d.x.array.min()), max_clip=prob.max_clip_seen,
        export_gap=res["cum_rain"] - res["cum_out"] - dW, cum_rain=res["cum_rain"],
        cum_out=res["cum_out"], dW=dW, n_acc=res["n_acc"], n_rej=res["n_rej"],
        run_s=res["run_s"], n_plateau=res["n_plateau"])
    return prob, msh, info


def drive_standalone_swale(W, nx, ny, *, LX=1620.0, LY=1000.0, SX=0.05, SY=0.02, n_man=0.015,
                           rain=RAIN, t_end=STORM, dt0=1e-5, dt_max=1e-3, grow_at=3, shrink_at=8,
                           plateau_frac=0.6, outlet="floor", verbose=False):
    """Drive the STANDALONE UpwindOverlandProblem (2-D surface, no soil) on the SAME swale surface
    geometry/forcing -- the operator-equivalence reference for A1.2. Returns (prob, msh, info)."""
    XC = LX / 2.0
    area = LX * LY
    Q_eq = rain * area
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [LX, LY]], [nx, ny])
    prob = UpwindOverlandProblem(msh, n_man)
    prob.set_topography(make_zb(W, LX, LY, XC, SX, SY))
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    prob.add_rain(0.0)
    prob.add_outflow_bc(outlet_locator(outlet, LX, LY, XC, W, nx), slope=SY)

    res = _run_controller(
        step_fn=prob.step,
        outflow_fn=prob.outflow_rate,
        water_fn=prob.total_water,
        set_rain=prob.add_rain,
        area=area, rain=rain, t_end=t_end, dt0=dt0, dt_max=dt_max, grow_at=grow_at,
        shrink_at=shrink_at, plateau_frac=plateau_frac, verbose=verbose, label="standalone")

    coords = prob.V.tabulate_dof_coordinates()[: prob.n_dofs]
    d = prob.d.x.array[: prob.n_dofs].copy()
    info = dict(
        W=W, nx=nx, ny=ny, LX=LX, LY=LY, XC=XC, SX=SX, SY=SY, n_man=n_man, area=area, Q_eq=Q_eq,
        q_plateau=res["q_plateau"], q_over_Qeq=res["q_plateau"] / Q_eq,
        top_x=coords[:, 0].copy(), top_y=coords[:, 1].copy(), d_top=d,
        d_min=float(d.min()), export_gap=res["cum_rain"] - res["cum_out"] - (res["wend"] - res["w0"]),
        n_acc=res["n_acc"], n_rej=res["n_rej"], run_s=res["run_s"])
    return prob, msh, info


def coupled_outlet_over_Qeq(W, nx, ny, **kw):
    """A1.1 helper: the COUPLED plateau outlet discharge as a fraction of Q_eq (= rain*area). This
    is conservation-FORCED to ~1 for ANY swale shape/resolution -- it is NOT an accuracy measure."""
    _, _, info = drive_coupled_swale(W, nx, ny, **kw)
    return info["q_over_Qeq"]


def depth_field_reldiff(info_coupled, info_standalone, *, interior_only=False):
    """A1.2 helper: |d_coupled - d_standalone| / max(d_standalone) over the SHARED top-surface (x,y)
    nodes (both meshes have P1 vertices on the same [0,LX]x[0,LY] grid -> match by coordinate).
    Returns dict(max, mean, median, n, x_at_max, y_at_max). interior_only excludes nodes within ~1
    cell of any domain edge (the outlet/corner treatment differs slightly: coupled codim-2 ridge vs
    standalone nodal length-weighting) so the BULK-field agreement is isolated."""
    def key(x, y):
        return (round(float(x), 6), round(float(y), 6))

    LX, LY = info_standalone["LX"], info_standalone["LY"]
    nx, ny = info_standalone["nx"], info_standalone["ny"]
    dx, dy = LX / nx, LY / ny
    cmap = {key(x, y): d for x, y, d in
            zip(info_coupled["top_x"], info_coupled["top_y"], info_coupled["d_top"])}
    dref = max(float(np.max(info_standalone["d_top"])), 1e-30)
    diffs, xs, ys = [], [], []
    for x, y, d in zip(info_standalone["top_x"], info_standalone["top_y"], info_standalone["d_top"]):
        if interior_only and (x < dx - 1e-9 or x > LX - dx + 1e-9
                              or y < dy - 1e-9 or y > LY - dy + 1e-9):
            continue
        c = cmap.get(key(x, y))
        if c is not None:
            diffs.append(abs(c - d) / dref)
            xs.append(x)
            ys.append(y)
    if not diffs:
        return dict(max=float("nan"), mean=float("nan"), median=float("nan"), n=0)
    diffs = np.asarray(diffs)
    imax = int(np.argmax(diffs))
    return dict(max=float(diffs.max()), mean=float(diffs.mean()), median=float(np.median(diffs)),
                n=len(diffs), x_at_max=xs[imax], y_at_max=ys[imax])


# ============================================================================
# main(): the authoritative Part-A measurement (Gate A evidence)
# ============================================================================
def main():
    FAST = os.environ.get("FAST", "0") == "1"
    print("=" * 80)
    print(f"P3 Part-A resolved-swale absolute accuracy  (FAST={FAST})")
    print("=" * 80)

    if FAST:
        # small + COARSE + large-dt: the fast pin-sizing config (~10-20 s/run). floor W spans >=6 cells.
        cfg = dict(LX=120.0, LY=80.0, H=1.0, SX=0.04, SY=0.02, n_man=0.015,
                   dt0=1e-4, dt_max=1e-2, t_end=0.04)
        W = 48.0
        meshes = [(16, 12), (24, 16)]
        nz = 2
    else:
        # the established tilted-V resolved swale (B5b SWALE_W=324 / P2-E1); SX,SY canonical
        cfg = dict(LX=1620.0, LY=1000.0, H=2.0, SX=0.05, SY=0.02, n_man=0.015)
        W = 324.0
        meshes = [(24, 16), (48, 32)]
        nz = 3

    XC = cfg["LX"] / 2.0
    S_v = cfg["SY"]
    Q_eq = RAIN * cfg["LX"] * cfg["LY"]
    d_n = normal_depth(Q_eq, W, cfg["n_man"], S_v)
    print(f"\ngeometry: {cfg['LX']:g}x{cfg['LY']:g} m  swale W={W:g}  SX={cfg['SX']}  SY={cfg['SY']}  "
          f"Q_eq={Q_eq:.1f} m^3/day  analytic d_n={1000*d_n:.2f} mm\n")

    skip_kink = os.environ.get("SKIP_KINK", "0") == "1"
    nx0, ny0 = meshes[-1]
    storm_kink = 0.3 * STORM   # the kink dt-collapses; a short window grabs its (growing) depth cheaply

    # --- A1.1 conservation-forced outlet: the swale outlet -> ~Q_eq, independent of resolution -----
    print("-" * 80)
    print("A1.1  conservation-forced outlet (NOT accuracy): outflow/Q_eq, resolution-INDEPENDENT")
    print("-" * 80)
    a11 = {}
    for (nx, ny) in meshes:
        _, _, info = drive_coupled_swale(W, nx, ny, nz, outlet="edge", **cfg)
        a11[(nx, ny)] = info
        print(f"  swale W={W:g} {nx}x{ny}: outflow/Q_eq={info['q_over_Qeq']:.4f}  export_gap="
              f"{info['export_gap']:+.2e} ({100*info['export_gap']/max(info['cum_rain'],1e-9):+.4f}% rain)"
              f"  d_min={1000*info['d_min']:+.3f}mm  run={info['run_s']:.1f}s  acc/rej={info['n_acc']}/{info['n_rej']}")
    isw = a11[(nx0, ny0)]

    # --- A1.2 operator-equivalence: coupled depth == standalone depth on the resolved swale -------
    print("-" * 80)
    print("A1.2  operator-equivalence: coupled-upwind depth vs standalone UpwindOverlandProblem")
    print("-" * 80)
    scfg = {k: v for k, v in cfg.items() if k != "H"}
    pstd, mstd, istd = drive_standalone_swale(W, nx0, ny0, outlet="edge", **scfg)
    full = depth_field_reldiff(isw, istd)
    int_ = depth_field_reldiff(isw, istd, interior_only=True)
    print(f"  swale W={W:g} {nx0}x{ny0}  (standalone d_max={1000*np.max(istd['d_top']):.2f}mm, "
          f"coupled d_max={1000*np.max(isw['d_top']):.2f}mm)")
    print(f"    ALL nodes (n={full['n']}):      max={full['max']:.4f}  mean={full['mean']:.4f}  "
          f"median={full['median']:.4f}   (max @ x={full.get('x_at_max',0):.1f}, y={full.get('y_at_max',0):.1f})")
    print(f"    INTERIOR (n={int_['n']}):  max={int_['max']:.4f}  mean={int_['mean']:.4f}  "
          f"median={int_['median']:.4f}")
    # the standalone FREE consistent ds-integral discharge -> ~0.99*Q_eq (B5b genuine accuracy);
    # operator-equivalence (interior ~0) carries this resolved-swale accuracy into the coupled engine.
    qc = consistent_outflow(pstd, mstd, cfg["n_man"], S_v, cfg["LY"]) / istd["Q_eq"]
    print(f"    standalone consistent ds-integral discharge / Q_eq = {qc:.4f}  (B5b resolved-swale ~0.99)")

    # --- A1.3 DIRECT coupled absolute accuracy on a clean tilted PLANE (no convergence) -----------
    print("-" * 80)
    print("A1.3  coupled tilted-PLANE outlet depth -> analytic normal depth (direct absolute check)")
    print("-" * 80)
    d_plane = plane_normal_depth(RAIN, cfg["LY"], cfg["n_man"], S_v)
    pcfg = dict(cfg)
    pcfg["SX"] = 0.0   # no cross-slope -> a pure tilted plane (uniform normal depth holds)
    for (nx, ny) in meshes:
        _, _, ip = drive_coupled_swale(0.0, nx, ny, nz, outlet="edge", **pcfg)
        d_out = outlet_row_mean_depth(ip, nx, ny)
        print(f"  plane {nx}x{ny}: outlet depth={1000*d_out:.3f}mm  vs analytic d_n={1000*d_plane:.3f}mm"
              f"   err={abs(d_out-d_plane)/d_plane:.4f}  outflow/Q_eq={ip['q_over_Qeq']:.4f}  "
              f"run={ip['run_s']:.1f}s")

    # --- A2 depth-field accuracy: swale floor -> d_n (converges) vs kink (diverges) ---------------
    print("-" * 80)
    print("A2  swale-FLOOR depth -> analytic normal-depth d_n, mesh-convergent  (kink diverges)")
    print("-" * 80)
    print(f"  analytic d_n = {1000*d_n:.3f} mm   (q_w = Q_eq/W = {Q_eq/W:.2f} m^2/day)")
    print(f"  {'W[m]':>7} {'NXxNY':>9} {'cells/flr':>9} {'floor_mean[mm]':>14} {'cl[mm]':>9} "
          f"{'err vs d_n':>10} {'run[s]':>7}")
    te_swale = cfg.get("t_end", STORM)
    series = [(W, "swale", "edge", None, te_swale)]
    if not skip_kink:
        series.append((0.0, "kink ", "edge", 1.0, min(storm_kink, te_swale)))   # relaxed tripwire
    for Wc, label, outl, ptol, te in series:
        prev_err = None
        for (nx, ny) in meshes:
            if Wc == W and (nx, ny) in a11:
                info = a11[(nx, ny)]                                # reuse the A1.1 swale runs
            else:
                _, _, info = drive_coupled_swale(Wc, nx, ny, nz, outlet=outl, pos_tol=ptol,
                                                 **{**cfg, "t_end": te})
            fl = measure_floor_depth(info["top_x"], info["top_y"], info["d_top"], XC, Wc,
                                     cfg["LX"], cfg["LY"], nx, ny)
            err = abs(fl["floor_mean"] - d_n) / d_n
            trend = "" if prev_err is None else ("  SHRINKS" if err < prev_err else "  GROWS")
            prev_err = err
            print(f"  {Wc:7.0f} {f'{nx}x{ny}':>9} {fl['cells_across']:9.2f} "
                  f"{1000*fl['floor_mean']:14.3f} {1000*fl['floor_centerline']:9.3f} "
                  f"{err:10.4f} {info['run_s']:7.1f}{trend}")
        print()


if __name__ == "__main__":
    main()
