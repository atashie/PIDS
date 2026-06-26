"""SCRATCH BENCHMARK -- 1-D vertical SORPTIVITY: which surface-infiltration closure captures the true
sorptive uptake of a DRY soil column? (Arik: sorptivity is critical for water transfer into the PIDS
granular infrastructure; getting it right decides the overland scheme + retires the coarse-mesh "0.547".)

A homogeneous soil COLUMN, uniformly DRY (psi_i), surface held PONDED (saturated) from t=0; measure the
cumulative infiltration I(t) = [INT theta(t) - INT theta(0)] / top_area  [m]. Compare:
  * REFINED Richards (psi_top=0 Dirichlet, nz=240)  -- the converged "truth" of the van Genuchten model.
  * Dirichlet psi=0 at COARSE nz=8 and a 2mm THIN-SKIN  -- exactly what the SWITCHING BC does (resolution
    sweep => does it converge to the refined?).
  * q_pot = max(kirchhoff(psi_top, h_ref),0)/ell_c as a Neumann influx at COARSE nz=8  -- the MONOLITH's
    sub-grid film model, evaluated adaptively at the current psi_top.
  * Analytical GREEN-AMPT (sharp front) + PHILIP (S*sqrt(t)) as physical cross-checks.

The question: does the coarse q_pot match the refined I(t) (good sorptivity model) or UNDER-infiltrate it
(under-corrects the sorptive limb)? And does psi=0 Dirichlet (switching BC) converge to the refined?

Run (WSL pids-fem, threads pinned) -- LIVE to a file (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_sorptivity_benchmark.py'
"""
from __future__ import annotations

import time

import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import RichardsProblem
from scratch.seq_cocycled_skin import make_graded_box

COMM = MPI.COMM_WORLD
SOILS = {
    "loam": dict(theta_r=0.078, theta_s=0.43, alpha=3.6,  n=1.56, Ks=0.25),
    "sand": dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=1.5),
    "silt": dict(theta_r=0.067, theta_s=0.45, alpha=2.0,  n=1.41, Ks=0.10),
    "clay": dict(theta_r=0.068, theta_s=0.38, alpha=0.8,  n=1.09, Ks=0.048),  # clay-V soil
}
LZ = 1.0
PSI_I = -0.4
TIMES = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]   # day -- log I(t) here (storm scale ~0.08)


def _column(nz, p=None):
    """A thin vertical column (2x2 horizontal), nz cells deep; graded toward the top if p is given."""
    if p is not None:
        return make_graded_box(2, 2, nz, 0.1, 0.1, LZ, p=p)
    return dmesh.create_box(COMM, [np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, LZ])],
                            [2, 2, nz], cell_type=dmesh.CellType.tetrahedron)


def _top_dofs(rp):
    zc = rp.V.tabulate_dof_coordinates()[:, 2]
    return np.where(np.isclose(zc, LZ))[0]


def _march_log(rp, soil, qpot=False, h_ref=1e-3, ell_c=None, dt0=2e-5, dt_max=2e-3, t_end=None):
    """March with an adaptive band controller; return I at each TIMES entry. For qpot mode, set the top
    Neumann flux = max(kirchhoff(psi_top, h_ref),0)/ell_c each step (adaptive)."""
    t_end = t_end or TIMES[-1]
    th0 = rp.total_water()
    top = _top_dofs(rp)
    q_const = None
    if qpot:
        q_const = rp.add_flux_bc(lambda x: np.isclose(x[2], LZ), 0.0)
    targets = list(TIMES)
    out = {}
    t, dt = 0.0, dt0
    nstep = 0
    while t < t_end - 1e-12 and nstep < 20000:
        h = min(dt, t_end - t)
        if targets and t + h > targets[0]:
            h = targets[0] - t
        if qpot:
            psi_top = float(np.mean(rp.psi.x.array[top]))
            q_const.value = max(soil.kirchhoff(psi_top, h_ref), 0.0) / ell_c
        conv, it = rp.step(h)
        if not conv:
            dt *= 0.5
            if dt < 1e-9:
                break
            continue
        t += h; nstep += 1
        if it <= 3:
            dt = min(dt * 1.4, dt_max)
        elif it >= 8:
            dt *= 0.7
        while targets and t >= targets[0] - 1e-12:
            out[targets.pop(0)] = (rp.total_water() - th0) / 0.01   # / top_area (0.1*0.1)
    for tt in targets:
        out[tt] = (rp.total_water() - th0) / 0.01
    return out, nstep


def run_dirichlet(soil_name, nz, p=None):
    soil = VanGenuchten(**SOILS[soil_name])
    msh = _column(nz, p=p)
    rp = RichardsProblem(msh, soil)
    rp.set_initial_condition(lambda x: PSI_I + 0.0 * x[0])
    rp.add_dirichlet(lambda x: np.isclose(x[2], LZ), 0.0)        # ponded saturated surface psi=0
    t0 = time.perf_counter()
    I, ns = _march_log(rp, soil)
    return I, ns, time.perf_counter() - t0


def run_qpot(soil_name, nz, h_ref=1e-3):
    soil = VanGenuchten(**SOILS[soil_name])
    msh = _column(nz)
    rp = RichardsProblem(msh, soil)
    rp.set_initial_condition(lambda x: PSI_I + 0.0 * x[0])
    ell_c = 0.5 * LZ / nz
    t0 = time.perf_counter()
    I, ns = _march_log(rp, soil, qpot=True, h_ref=h_ref, ell_c=ell_c)
    return I, ns, time.perf_counter() - t0


def green_ampt(soil_name):
    """Green-Ampt cumulative infiltration I(t) under a ponded (psi_s=0) surface, dry psi_i.
    Effective front suction psi_f = |kirchhoff(psi_i, 0)| / Ks (Kirchhoff matric drive over Ks);
    dtheta = theta_s - theta(psi_i). Implicit: I - psi_f*dtheta*ln(1 + I/(psi_f*dtheta)) = Ks*t."""
    s = SOILS[soil_name]
    soil = VanGenuchten(**s)
    Ks = s["Ks"]
    dth = s["theta_s"] - float(soil.theta(np.array([PSI_I]))[0])
    psi_f = abs(soil.kirchhoff(PSI_I, 0.0)) / Ks
    B = psi_f * dth
    out = {}
    for t in TIMES:
        I = max(Ks * t + B, 1e-9)                       # initial guess
        for _ in range(80):                              # Newton on GA implicit
            f = I - B * np.log1p(I / B) - Ks * t
            fp = 1.0 - B / (B + I)
            I -= f / fp
        out[t] = I
    return out, dict(psi_f=psi_f, dtheta=dth, S=float(np.sqrt(2 * Ks * B)))


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("#" * 100)
    print("1-D SORPTIVITY benchmark -- cumulative infiltration I(t) [mm], ponded dry column, psi_i=-0.4 m")
    print("#" * 100, flush=True)
    for soil in ("loam", "clay"):
        print(f"\n=== {soil} (Ks={SOILS[soil]['Ks']}) ===", flush=True)
        ga, gap = green_ampt(soil)
        print(f"  Green-Ampt params: psi_f={gap['psi_f']*1000:.1f}mm dtheta={gap['dtheta']:.3f} "
              f"S={gap['S']*1000:.2f} mm/sqrt(day)", flush=True)
        ref, nref, wref = run_dirichlet(soil, 240)
        cdir, ncd, wcd = run_dirichlet(soil, 8)
        skin, nsk, wsk = run_dirichlet(soil, 12, p=2.5)
        qp, nq, wq = run_qpot(soil, 8)
        hdr = "  t[day] |" + "".join(f"{t:>9.3f}" for t in TIMES)
        print(hdr)
        print(f"  REFINED(nz240) | " + "".join(f"{ref[t]*1000:>8.2f} " for t in TIMES) +
              f"  [{nref} steps {wref:.0f}s]")
        print(f"  switch coarse8 | " + "".join(f"{cdir[t]*1000:>8.2f} " for t in TIMES) +
              f"  [{ncd} steps {wcd:.0f}s]")
        print(f"  switch 2mm-skin| " + "".join(f"{skin[t]*1000:>8.2f} " for t in TIMES) +
              f"  [{nsk} steps {wsk:.0f}s]")
        print(f"  q_pot coarse8  | " + "".join(f"{qp[t]*1000:>8.2f} " for t in TIMES) +
              f"  [{nq} steps {wq:.0f}s]")
        print(f"  Green-Ampt     | " + "".join(f"{ga[t]*1000:>8.2f} " for t in TIMES))
        # ratios to the refined truth (the key numbers)
        print(f"  --- ratio to REFINED (1.00 = matches the converged sorptive uptake) ---")
        print(f"  switch coarse8 | " + "".join(f"{cdir[t]/ref[t]:>8.2f} " for t in TIMES))
        print(f"  switch 2mm-skin| " + "".join(f"{skin[t]/ref[t]:>8.2f} " for t in TIMES))
        print(f"  q_pot coarse8  | " + "".join(f"{qp[t]/ref[t]:>8.2f} " for t in TIMES), flush=True)
    print("\n" + "#" * 100, flush=True)


if __name__ == "__main__":
    main()
