"""SCRATCH STUDY (no pids_forward/ edits, no commit) -- pin the h_ref CLOSURE for realization B.

Realization B (seq_href_cap_spike.HrefCappedPondInPsi) matches the monolith partition at a CALIBRATED
h_ref (b1 loam: h_ref*=2 mm). routed/R is SENSITIVE to h_ref (and counter-intuitively, larger h_ref ->
MORE runoff). This study asks: is h_ref* a UNIVERSAL constant, DERIVABLE from a physical scale (the
local Manning equilibrium sheet depth d_M, or the monolith's own sheet), or MESH-dependent (an artifact)?

For each case we (a) run the galerkin monolith (target routed/R + its peak mean sheet depth), (b) sweep
B over h_ref and interpolate h_ref* where routed/R == target, (c) compare h_ref* to:
  * d_M   = Manning normal depth at the outlet for the Hortonian excess  (n q / sqrt(S))^(3/5), q=(rain-Ks)*L
  * d_mono= the monolith's PEAK mean surface-sheet depth (surface_water/top_area), the actual sheet
  * ell_c = top-cell half-height (Lz/nz/2), the mesh scale

DISCRIMINATING CONTRASTS:
  * b1_base  (loam, S=0.03, 30x20x8)  -- the anchor (h_ref* ~ 2 mm known)
  * b1_steep (loam, S=0.10, 30x20x8)  -- steeper => THINNER Manning sheet (d_M ~ S^-0.3). If h_ref* drops
    with it, h_ref ~ d_M; if h_ref* is unchanged, h_ref is constant.
  * b1_coarse(loam, S=0.03, 20x14x5)  -- coarser mesh (LARGER ell_c). If h_ref* shifts vs b1_base, it is
    MESH-dependent (an artifact needing ell_c in the rule); if stable, it is physical.
  * sand     (Ks=1.5, S=0.03, 30x20x8)-- high-K => different acceptance + sheet.

Run (WSL pids-fem, threads pinned) -- LONG (each case = 1 monolith + ~4 B runs):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_href_closure_study.py 2>&1 | tail -80'
"""
from __future__ import annotations

import time

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from scratch.seq_href_cap_spike import HrefCappedPondInPsi
from scratch.seq_iterative_prototype import _top_area_ds, _soil_water_deg8

COMM = MPI.COMM_WORLD
SECONDS_PER_DAY = 86400.0


def make_box(nx, ny, nz, Lx, Ly, Lz):
    return dmesh.create_box(
        COMM, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz], cell_type=dmesh.CellType.tetrahedron)


def manning_normal_depth(rain, Ks, n_man, S, L):
    """Manning normal depth [m] at the outlet for the Hortonian excess (rain-Ks) over slope length L.
    q = (rain-Ks)*L  [m^2/s per width];  d = (n q / sqrt(S))^(3/5)."""
    excess = max(rain - Ks, 0.0) / SECONDS_PER_DAY      # m/s
    q = excess * L                                       # m^2/s per unit width
    if q <= 0.0:
        return 0.0
    return (n_man * q / np.sqrt(S)) ** 0.6


def _march(prob, rain_c, *, storm_dur, storm_rain, t_end, dt0=2e-3, dt_max=0.02,
           ctrl_low=4, ctrl_high=12, track_sheet=False, top_area=None, max_steps=400):
    """March a storm-then-recession with the band controller. Optionally track the PEAK mean sheet
    depth (surface_water/top_area). Returns (nstep, collapsed, t_reached, peak_sheet). A ``max_steps``
    guard trips ``collapsed`` (a runaway tiny-dt grind that never crosses the 1e-9 collapse floor --
    the bug that wasted 4 h on the first run)."""
    t, nstep, dt = 0.0, 0, dt0
    collapsed = False
    peak_sheet = 0.0
    while t < t_end - 1e-12:
        if nstep >= max_steps:
            collapsed = True        # runaway: too many steps -> treat as a (slow) failure, bail
            break
        h = min(dt, t_end - t)
        if t < storm_dur - 1e-12 and t + h > storm_dur:
            h = storm_dur - t
        rain_c.value = storm_rain if t < storm_dur - 1e-12 else 0.0
        conv, it = prob.step(h)
        if not conv:
            dt *= 0.5
            if dt < 1e-9:
                collapsed = True
                break
            continue
        t += h
        nstep += 1
        if track_sheet and top_area:
            peak_sheet = max(peak_sheet, prob.surface_water() / top_area)
        if it <= ctrl_low:
            dt = min(dt * 1.4, dt_max)
        elif it >= ctrl_high:
            dt = dt * 0.7
    return nstep, collapsed, t, peak_sheet


def run_mono(case, mesh_dims):
    """Galerkin monolith target: routed/R + peak mean sheet depth."""
    f = case
    soil = VanGenuchten(**f["soil"])
    msh = make_box(*mesh_dims, f["Lx"], f["Ly"], f["Lz"])
    mono = CoupledProblem(msh, soil, n_man=f["n"], overland_scheme="galerkin")
    mono.set_initial_condition(lambda x: f["psi_i"] + 0.0 * x[0], d_value=0.0)
    mono.set_topography(lambda x: f["S0"] * x[1])
    rain_c = mono.add_rain(0.0)
    mono.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=f["S0"])
    th0 = _soil_water_deg8(mono, soil)
    top_area = _top_area_ds(msh, f["Lz"])
    R_in = f["RAIN"] * top_area * f["STORM"]
    t0 = time.perf_counter()
    nstep, coll, tend, peak_sheet = _march(
        mono, rain_c, storm_dur=f["STORM"], storm_rain=f["RAIN"], t_end=f["TEND"],
        ctrl_low=3, ctrl_high=8, track_sheet=True, top_area=top_area)
    routed = mono.cum_outflow / R_in
    return dict(routed=routed, peak_sheet=peak_sheet, coll=coll, tend=tend, nstep=nstep,
                wall=time.perf_counter() - t0, ok=(not coll and tend >= f["TEND"] - 1e-9))


def run_B(case, mesh_dims, h_ref, rs=4):
    """Realization B at a given h_ref: routed/R + ledger residual."""
    f = case
    soil = VanGenuchten(**f["soil"])
    msh = make_box(*mesh_dims, f["Lx"], f["Ly"], f["Lz"])
    prob = HrefCappedPondInPsi(msh, soil, n_man=f["n"], route_substeps=rs, h_ref=h_ref)
    prob.set_initial_condition(lambda x: f["psi_i"] + 0.0 * x[0])
    prob.set_topography(lambda x: f["S0"] * x[1])
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=f["S0"])
    top_area = _top_area_ds(msh, f["Lz"])
    R_in = f["RAIN"] * top_area * f["STORM"]
    t0 = time.perf_counter()
    nstep, coll, tend, _ = _march(prob, rain_c, storm_dur=f["STORM"], storm_rain=f["RAIN"],
                                  t_end=f["TEND"])
    routed = prob.cum_outflow / R_in
    bal = prob.balance()
    return dict(routed=routed, bal_frac=abs(bal) / prob.cum_rain if prob.cum_rain > 0 else np.nan,
                coll=coll, tend=tend, nstep=nstep, wall=time.perf_counter() - t0,
                ok=(not coll and tend >= f["TEND"] - 1e-9))


def find_href_star(case, mesh_dims, target, h_grid):
    """Sweep B over h_grid; linearly interpolate h_ref* where routed/R == target (routed/R is
    monotone increasing in h_ref). Returns (h_star, sweep_rows)."""
    rows = []
    for h in h_grid:
        r = run_B(case, mesh_dims, h)
        rows.append((h, r["routed"], r["bal_frac"], r["ok"]))
        print(f"      B h_ref={h*1000:6.2f}mm -> routed/R={r['routed']:.4f} "
              f"bal/rain={r['bal_frac']:.1e} ok={r['ok']} [{r['nstep']} steps {r['wall']:.0f}s]",
              flush=True)
    hs = np.array([r[0] for r in rows]); rr = np.array([r[1] for r in rows])
    h_star = np.nan
    for i in range(len(hs) - 1):
        if (rr[i] - target) * (rr[i + 1] - target) <= 0 and rr[i + 1] != rr[i]:
            w = (target - rr[i]) / (rr[i + 1] - rr[i])
            h_star = hs[i] + w * (hs[i + 1] - hs[i])
            break
    return h_star, rows


# --- cases (mild planar Hortonian; the monolith genuinely converges) ---------------------------------
BASE_SOIL = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)   # LOAM (b1)
SAND_SOIL = dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=1.5)   # coarse sand-ish
CASES = {
    "b1_base":  dict(soil=BASE_SOIL, Lx=8.0, Ly=5.0, Lz=1.0, S0=0.03, psi_i=-0.4,
                     RAIN=0.5, STORM=0.08, TEND=0.45, n=0.05, mesh=(30, 20, 8)),
    "b1_steep": dict(soil=BASE_SOIL, Lx=8.0, Ly=5.0, Lz=1.0, S0=0.10, psi_i=-0.4,
                     RAIN=0.5, STORM=0.08, TEND=0.45, n=0.05, mesh=(30, 20, 8)),
    "b1_coarse": dict(soil=BASE_SOIL, Lx=8.0, Ly=5.0, Lz=1.0, S0=0.03, psi_i=-0.4,
                      RAIN=0.5, STORM=0.08, TEND=0.45, n=0.05, mesh=(20, 14, 5)),
    # "sand" DEFERRED to a 2nd pass (high-K near-saturation fragility = the runaway risk; the loam
    # contrasts below answer the core closure question -- physical scale vs mesh-dependence -- first):
    # "sand": dict(soil=SAND_SOIL, ..., RAIN=2.5, ...),
}
# h_ref sweep grid per case (mm). 3 values bracketing the expected h_ref* (monotone routed/R in h_ref).
H_GRID = {
    "b1_base":   [0.001, 0.002, 0.004],
    "b1_steep":  [0.0005, 0.001, 0.002],
    "b1_coarse": [0.001, 0.002, 0.004],
}


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("#" * 92)
    print("h_ref CLOSURE STUDY -- is h_ref* universal / Manning-derivable / mesh-dependent?")
    print("#" * 92, flush=True)
    results = {}
    for name, case in CASES.items():
        mesh_dims = case["mesh"]
        ell_c = 0.5 * case["Lz"] / mesh_dims[2]
        d_M = manning_normal_depth(case["RAIN"], case["soil"]["Ks"], case["n"], case["S0"], case["Ly"])
        print(f"\n=== {name}: Ks={case['soil']['Ks']} S={case['S0']} mesh={mesh_dims} "
              f"ell_c={ell_c*1000:.1f}mm d_M(Manning)={d_M*1000:.2f}mm ===", flush=True)
        mono = run_mono(case, mesh_dims)
        print(f"    MONOLITH(galerkin): routed/R={mono['routed']:.4f} peak_sheet={mono['peak_sheet']*1000:.3f}mm "
              f"[{mono['nstep']} steps {mono['wall']:.0f}s ok={mono['ok']}]", flush=True)
        if not mono["ok"]:
            print("    !! monolith did not complete -- skipping h_ref* for this case", flush=True)
            results[name] = dict(target=mono["routed"], h_star=np.nan, d_M=d_M,
                                 d_mono=mono["peak_sheet"], ell_c=ell_c, mono_ok=False)
            continue
        h_star, rows = find_href_star(case, mesh_dims, mono["routed"], H_GRID[name])
        print(f"    => h_ref* = {h_star*1000:.3f} mm  (matches monolith routed/R={mono['routed']:.4f})",
              flush=True)
        results[name] = dict(target=mono["routed"], h_star=h_star, d_M=d_M,
                             d_mono=mono["peak_sheet"], ell_c=ell_c, mono_ok=True)

    # ---- the closure table -------------------------------------------------------------------------
    print("\n" + "#" * 92)
    print("CLOSURE TABLE  (all depths in mm)")
    print("#" * 92)
    print(f"{'case':>10} | {'target':>7} | {'h_ref*':>8} | {'d_M':>7} | {'d_mono':>7} | {'ell_c':>7} "
          f"| {'h*/d_M':>7} | {'h*/d_mono':>9}")
    for name, r in results.items():
        hs, dM, dm, ec = r["h_star"], r["d_M"], r["d_mono"], r["ell_c"]
        print(f"{name:>10} | {r['target']:>7.4f} | {hs*1000:>8.3f} | {dM*1000:>7.3f} | {dm*1000:>7.3f} "
              f"| {ec*1000:>7.1f} | {hs/dM if dM>0 else np.nan:>7.2f} | "
              f"{hs/dm if dm>0 else np.nan:>9.2f}")
    print("\nREAD:")
    print("  * h*/d_mono ~ CONSTANT across cases  => h_ref = C * (monolith sheet)  [derivable, knob-free-ish]")
    print("  * h*/d_M    ~ CONSTANT across cases  => h_ref = C * Manning normal depth  [knob-free, a-priori]")
    print("  * h_ref*    ~ CONSTANT (mm)          => a single universal constant suffices")
    print("  * h_ref* moves with ell_c (b1_base vs b1_coarse) => MESH-dependent (artifact; needs ell_c)")
    print("#" * 92, flush=True)


if __name__ == "__main__":
    main()
