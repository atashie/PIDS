"""SCRATCH SPIKE -- ARIK'S THIN-SKIN idea for the co-cycled split's force-feed / accuracy-stability
tradeoff (Task-4 verdict, SS12).

DIAGNOSIS: the co-cycled scheme force-feeds infiltration because the THICK ~6 cm top cell, under a
ponded film over DRY soil (psi~-0.4 m), sees a gradient ~(film - psi_cell)/(half-cell) ~ 13 -> a huge
transient infiltration that fills the cell's large storage before it saturates; at the accurate weight
w~0.7 this drives a saturation front -> Newton stiffness -> dt-collapse. The monolith dodges this by
capping infiltration at q_pot = INT K dpsi / ell_c (a cell-scale acceptance); the co-cycled has no cap.

ARIK'S FIX (this spike): make the surface a THIN SKIN (z-graded mesh, top cell ~2 mm) so the skin
saturates immediately and infiltration is throttled by PERCOLATION into the (dry, low-K) subsoil -- NOT
the ponding head. Realizes the CATHY/HYDRUS saturated-surface acceptance through the MESH (physical,
adaptive), unlike the SS9 frozen-q_pot cap (which never let the surface saturate).

HYPOTHESIS: with the skin, the partition becomes ~w-INSENSITIVE (film head decoupled) AND accurate AND
stable -> exact + accurate + stable, the full win.

GATE: on the z-graded mesh, b1_steep -- (a) does co-cycled w=0.7 STOP collapsing? (b) is the partition
near the monolith 0.551? (c) is it ~w-insensitive (w=0.5 ~ w=0.7)? (d) conservation still ~1e-11?

Run (WSL pids-fem, threads pinned) -- LIVE to a file (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_cocycled_skin.py <mode> <w_or_scheme> <p> <nz> > scratch/_skin_<tag>.txt 2>&1'
    mode = cocy  : co-cycled at film_w=<w>     (e.g. cocy 0.7 2.5 12)
    mode = mono  : galerkin monolith control   (e.g. mono galerkin 2.5 12)
"""
from __future__ import annotations

import sys
import time

import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from scratch.seq_href_iterated import CoCycledCappedSplit, COMMON, CASES, LOAM
from scratch.seq_href_closure_study import _march
from scratch.seq_iterative_prototype import _top_area_ds, _soil_water_deg8

COMM = MPI.COMM_WORLD


def make_graded_box(nx, ny, nz, Lx, Ly, Lz, p=2.5):
    """Uniform tetra box, then warp the z-coordinates to cluster near the TOP (z=Lz): a thin skin.
    z = Lz*(1 - (1 - s)^p), s = z_uniform/Lz; monotonic -> no tet inversion. Top cell ~ Lz*(1/nz)^p."""
    msh = dmesh.create_box(
        COMM, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz], cell_type=dmesh.CellType.tetrahedron)
    x = msh.geometry.x
    s = np.clip(x[:, 2] / Lz, 0.0, 1.0)
    x[:, 2] = Lz * (1.0 - (1.0 - s) ** p)
    return msh


def _top_cell_thickness(Lz, nz, p):
    return Lz * (1.0 / nz) ** p   # top cell = Lz*(1 - (1-1/nz)^p) - ... = Lz*(1/nz)^p exactly


def run_cocy(case, film_w, p, nz, dt_max=0.004, h_ref=2e-3, K=6):
    c = COMMON
    soil = VanGenuchten(**LOAM)
    msh = make_graded_box(c["MESH"][0], c["MESH"][1], nz, c["Lx"], c["Ly"], c["Lz"], p=p)
    prob = CoCycledCappedSplit(msh, soil, n_man=c["NMAN"], route_substeps=4, h_ref=h_ref, K=K,
                               film_w=film_w)
    prob.set_initial_condition(lambda x: c["PSI_I"] + 0.0 * x[0])
    prob.set_topography(lambda x: case["S0"] * x[1])
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=case["S0"])
    top_area = _top_area_ds(msh, c["Lz"])
    R_in = c["RAIN"] * top_area * c["STORM"]
    t0 = time.perf_counter()
    ns, coll, tend, _ = _march(prob, rain_c, storm_dur=c["STORM"], storm_rain=c["RAIN"],
                               t_end=c["TEND"], dt0=dt_max / 4.0, dt_max=dt_max, max_steps=600)
    ok = (not coll) and tend >= c["TEND"] - 1e-9
    return dict(routed=prob.cum_outflow / R_in, bal=abs(prob.balance()) / prob.cum_rain,
                ns=ns, ok=ok, wall=time.perf_counter() - t0, min_held=prob.min_held_seen)


CV = dict(Lx=10.0, Ly=6.0, Lz=1.5, psi_i=-0.30, RAIN=1.0, STORM=0.08, TEND=0.40,
          soil=dict(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.048))   # stiff convergent CLAY-V


def run_cocy_clayv(film_w, p, nz, dt_max=0.02, h_ref=2e-3, K=6, nx=24, ny=16):
    """Stiff convergent CLAY-V on a z-graded (thin-skin) mesh -- the robustness case the monolith
    dt-collapses on (Task 5). GATE: completes (no collapse) + conserves ~1e-12."""
    f = CV
    soil = VanGenuchten(**f["soil"])
    msh = make_graded_box(nx, ny, nz, f["Lx"], f["Ly"], f["Lz"], p=p)
    prob = CoCycledCappedSplit(msh, soil, n_man=0.05, route_substeps=4, h_ref=h_ref, K=K, film_w=film_w)
    prob.set_initial_condition(lambda x: f["psi_i"] + 0.0 * x[0])
    prob.set_topography(lambda x: 0.05 * x[1] + 0.08 * np.abs(x[0] - 5.0))
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=0.05)
    top_area = _top_area_ds(msh, f["Lz"])
    R_in = f["RAIN"] * top_area * f["STORM"]
    t0 = time.perf_counter()
    ns, coll, tend, _ = _march(prob, rain_c, storm_dur=f["STORM"], storm_rain=f["RAIN"],
                               t_end=f["TEND"], dt0=dt_max / 4.0, dt_max=dt_max, max_steps=600)
    ok = (not coll) and tend >= f["TEND"] - 1e-9
    return dict(routed=prob.cum_outflow / R_in, bal=abs(prob.balance()) / prob.cum_rain,
                ns=ns, ok=ok, wall=time.perf_counter() - t0, min_held=prob.min_held_seen, tend=tend)


def run_mono(case, scheme, p, nz, dt_max=0.02):
    c = COMMON
    soil = VanGenuchten(**LOAM)
    msh = make_graded_box(c["MESH"][0], c["MESH"][1], nz, c["Lx"], c["Ly"], c["Lz"], p=p)
    mono = CoupledProblem(msh, soil, n_man=c["NMAN"], overland_scheme=scheme)
    mono.set_initial_condition(lambda x: c["PSI_I"] + 0.0 * x[0], d_value=0.0)
    mono.set_topography(lambda x: case["S0"] * x[1])
    rain_c = mono.add_rain(0.0)
    mono.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=case["S0"])
    top_area = _top_area_ds(msh, c["Lz"])
    R_in = c["RAIN"] * top_area * c["STORM"]
    t0 = time.perf_counter()
    ns, coll, tend, _ = _march(mono, rain_c, storm_dur=c["STORM"], storm_rain=c["RAIN"],
                               t_end=c["TEND"], dt0=dt_max / 4.0, dt_max=dt_max,
                               ctrl_low=3, ctrl_high=8, max_steps=600)
    ok = (not coll) and tend >= c["TEND"] - 1e-9
    return dict(routed=mono.cum_outflow / R_in, ns=ns, ok=ok, wall=time.perf_counter() - t0,
                eff=getattr(mono, "_effective_overland_scheme", scheme))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "cocy"
    arg2 = sys.argv[2] if len(sys.argv) > 2 else "0.7"
    p = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    nz = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    case_name = sys.argv[5] if len(sys.argv) > 5 else "b1_steep"
    case = CASES[case_name]
    tc = _top_cell_thickness(COMMON["Lz"], nz, p)
    print("#" * 92)
    print(f"THIN-SKIN test ({mode}) -- {case_name}  z-graded nz={nz} p={p} -> top cell ~ {tc*1000:.2f} mm "
          f"(uniform was {COMMON['Lz']/COMMON['MESH'][2]*1000:.1f} mm)")
    print("#" * 92, flush=True)
    if mode == "mono":
        r = run_mono(case, arg2, p, nz)
        print(f"  MONOLITH({arg2}, eff={r['eff']}): routed/R={r['routed']:.4f} vs uniform-mono "
              f"{case['target']:.4f}  ok={r['ok']} ns={r['ns']} wall={r['wall']:.0f}s", flush=True)
    elif mode == "clayv":
        w = float(arg2)
        r = run_cocy_clayv(w, p, nz)
        print(f"  CLAY-V CO-CYCLED w={w}: completed={r['ok']} routed/R={r['routed']:.4f} "
              f"bal/rain={r['bal']:.2e} min_held={r['min_held']*1000:.2f}mm  "
              f"[ns={r['ns']} t={r['tend']:.3f} wall={r['wall']:.0f}s]  "
              f"=> {'ROBUST (no collapse)' if r['ok'] else 'COLLAPSED'}", flush=True)
    else:
        w = float(arg2)
        r = run_cocy(case, w, p, nz)
        gp = r["routed"] - case["target"]
        print(f"  CO-CYCLED w={w}: routed/R={r['routed']:.4f} vs monolith {case['target']:.4f} "
              f"(gap {gp*100:+.1f}pp)  bal/rain={r['bal']:.2e}  min_held={r['min_held']*1000:.2f}mm")
        print(f"     ok={r['ok']} ns={r['ns']} wall={r['wall']:.0f}s  "
              f"=> {'STABLE+ran' if r['ok'] else 'COLLAPSED'}", flush=True)
    print("#" * 92, flush=True)
