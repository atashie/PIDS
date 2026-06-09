"""Module 4 (§E) Phase-1a: the sorptivity GROUND TRUTH (the falsifiability ground truth, step 1).

The Phase-3 embedding-fidelity gate (claim C-004) compares the EMBEDDED sorptive leg sigma against a
RESOLVED near-field reference. Before trusting that reference we must show the resolved Richards solve
itself reproduces Philip's absorption law. This spike establishes two things, incrementally:

  (1) PLANAR (Cartesian, gravity-off) Richards absorption into dry soil reproduces Philip S*sqrt(t):
        - cumulative uptake I(t) ~ t^p with p ~ 0.5 (log-log slope) -- the rigorous sorption signature,
        - the measured S_num agrees (~<15%) with an INDEPENDENT analytical Parlange (1975) sorptivity
          S_an = sqrt( int_{psi_i}^{0} (theta0 + theta(psi) - 2 theta_i) K(psi) dpsi )   [D dtheta = K dpsi].
      (Critical-review point: I(t)~sqrt(t) alone is self-confirming; the magnitude S is the physics.)

  (2) CYLINDRICAL (r-weighted axisymmetric, gravity-off) absorption from a wall at r=r_w MATCHES the
      planar uptake-per-wall-area at early time (curvature negligible when penetration << r_w), and
      diverges (more soil volume -> slightly more uptake) only later. This (a) validates the r-weighted
      measure machinery the Phase-1b near-field annulus reference (axisym tunnel + drain cross-section)
      will use, and (b) confirms the cylindrical reference is the right early-time ground truth.

This is the DISPERSE direction (wall saturated, soil dry) -- the one Philip's law governs. The DRAIN
direction (soil wet, wall a sink) is gravity/desorption, validated differently in Phase 1b.

Run from forward-model/:
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scratch/m4_phase1_sorptivity_reference.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten

COMM = MPI.COMM_WORLD

# Carsel & Parrish (1988) van Genuchten textures (alpha 1/m, Ks m/day). LOAM matches the §D test soil.
SAND = VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13)
LOAM = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)

_LU = {
    # cp (critical-point) linesearch, NOT bt: the hard saturated-wall Dirichlet (psi=0 against dry
    # soil) is a stiff step-change that stalls bt's backtracking (residual flat -> DIVERGED_LINESEARCH);
    # cp converges robustly in 3-5 Newton iters across dt and wall head (probed 2026-06-08).
    "snes_type": "newtonls", "snes_linesearch_type": "cp",
    "snes_rtol": 1e-10, "snes_atol": 1e-12, "snes_max_it": 50,
    "ksp_type": "preonly", "pc_type": "lu",
}


def parlange_sorptivity(soil: VanGenuchten, psi_i: float, psi_w: float = 0.0, npts: int = 4001) -> float:
    """Parlange (1975) sorptivity S [m / day^0.5].

    S^2 = int_{theta_i}^{theta_0} (theta_0 + theta - 2 theta_i) D(theta) dtheta, and with D = K/C,
    dtheta = C dpsi this collapses to the kink-free 1-D integral over pressure head:
        S^2 = int_{psi_i}^{0} (theta_0 + theta(psi) - 2 theta_i) K(psi) dpsi.

    WARNING (adversarial review 2026-06-08): the theta-space and psi-space forms are NOT interchangeable
    when a SATURATED air-entry plateau exists. The change of variables D dtheta = K dpsi is exact, but a
    NAIVE theta-space quadrature collapses the [h_s, 0] plateau (where theta=theta_s is constant, C=0) to a
    zero-width interval and DROPS the near-saturation contribution -> e.g. SAND gives 0.364 instead of the
    correct 0.491 (~25% low). The psi-form here is the correct one (the plateau is a genuine growing
    saturated zone scaling as sqrt(t); confirmed by independent Boltzmann + finite-volume solvers). Always
    integrate in psi, not theta, for soils with an air-entry cutoff.
    """
    psi = np.linspace(psi_i, psi_w, npts)
    th = soil.theta(psi)
    th_i = float(soil.theta(psi_i))
    th0 = float(soil.theta(psi_w))
    integrand = (th0 + th - 2.0 * th_i) * soil.K(psi)
    s2 = float(np.trapezoid(integrand, psi))
    return float(np.sqrt(max(s2, 0.0)))


def _vertex_dx():
    return ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})


def _solve_to(problem, psi, psi_n, dt_c, t0, t_target, dt):
    """Adaptive backward-Euler march from t0 to t_target; returns the dt to resume with."""
    t = t0
    while t < t_target - 1e-15:
        h = min(dt, t_target - t)
        dt_c.value = h
        problem.solve()
        snes = problem.solver
        if snes.getConvergedReason() > 0:
            psi_n.x.array[:] = psi.x.array
            psi_n.x.scatter_forward()
            t += h
            it = int(snes.getIterationNumber())
            dt = dt * 1.5 if it <= 4 else (dt * 0.7 if it >= 8 else dt)
        else:
            psi.x.array[:] = psi_n.x.array
            psi.x.scatter_forward()
            dt *= 0.5
            assert dt > 1e-13, f"dt collapse near t={t:.3e}"
    return dt


def _uptake_run(soil, *, cylindrical, r_w, psi_i, L, N, samples):
    """Fine 1-D Richards absorption (wall psi=0 at the near edge, dry psi_i far). Returns I(t) at
    `samples` times. PLANAR: I = int (theta-theta_i) dx [m]. CYLINDRICAL: per-radian uptake
    int (theta-theta_i) r dr, reported as uptake-per-wall-area (/r_w) so it is directly comparable
    to the planar per-area uptake."""
    x0 = r_w if cylindrical else 0.0
    msh = dmesh.create_interval(COMM, N, [x0, x0 + L])
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi = fem.Function(V)
    psi_n = fem.Function(V)
    psi.x.array[:] = psi_i
    psi_n.x.array[:] = psi_i
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))

    w = ufl.SpatialCoordinate(msh)[0] if cylindrical else fem.Constant(msh, PETSc.ScalarType(1.0))
    dxs = _vertex_dx()
    theta = soil.theta_ufl(psi)
    theta_n = soil.theta_ufl(psi_n)
    K = soil.K_ufl(psi)
    # gravity OFF (horizontal absorption): pure capillary sorption -> clean Philip S*sqrt(t).
    F = ((theta - theta_n) / dt_c) * v * w * dxs + K * ufl.dot(ufl.grad(psi), ufl.grad(v)) * w * ufl.dx

    near = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x0))
    far = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x0 + L))
    bcs = [
        fem.dirichletbc(PETSc.ScalarType(0.0), near, V),       # saturated wall (disperse)
        fem.dirichletbc(PETSc.ScalarType(psi_i), far, V),       # dry far field
    ]
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4p1_", petsc_options=_LU)

    th_i = float(soil.theta(psi_i))
    stored = fem.form((theta - th_i) * w * dxs)
    front = fem.form(ufl.conditional(ufl.gt(psi, psi_i + 0.1 * abs(psi_i)), 1.0, 0.0) * ufl.dx)

    out = []
    dt = 1e-8
    t_prev = 0.0
    for t_s in samples:
        dt = _solve_to(problem, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        I_raw = COMM.allreduce(fem.assemble_scalar(stored), op=MPI.SUM)
        I = I_raw / r_w if cylindrical else I_raw          # -> uptake per wall area for both
        pen = COMM.allreduce(fem.assemble_scalar(front), op=MPI.SUM)
        out.append((t_s, float(I), float(pen)))
    return out


def _fit_exponent(ts, Is):
    """Log-log slope of I vs t over the window (the Philip exponent; expect ~0.5)."""
    lt = np.log(np.asarray(ts))
    li = np.log(np.asarray(Is))
    p, _ = np.polyfit(lt, li, 1)
    return float(p)


def run_soil(name, soil, *, psi_i, t_end, L, N, r_w):
    print(f"\n{'='*78}\n{name}  (psi_i={psi_i} m,  t_end={t_end} d,  L={L} m, N={N},  r_w={r_w} m)\n{'='*78}")
    samples = list(np.geomspace(t_end / 60.0, t_end, 14))
    S_an = parlange_sorptivity(soil, psi_i)
    print(f"  analytical Parlange sorptivity S_an = {S_an:.5f} m/day^0.5")

    pl = _uptake_run(soil, cylindrical=False, r_w=r_w, psi_i=psi_i, L=L, N=N, samples=samples)
    # fit window: drop the first few (initial backward-Euler transient) and any sample whose front
    # has reached > 0.6 L (boundary contamination); keep the clean middle.
    ts = [t for (t, _, pen) in pl]
    Is = [I for (_, I, _) in pl]
    pens = [pen for (_, _, pen) in pl]
    keep = [k for k in range(len(ts)) if k >= 3 and pens[k] < 0.6 * L]
    p = _fit_exponent([ts[k] for k in keep], [Is[k] for k in keep])
    S_num = float(np.mean([Is[k] / np.sqrt(ts[k]) for k in keep]))
    S_spread = float(np.std([Is[k] / np.sqrt(ts[k]) for k in keep]))
    print(f"  PLANAR I(t):    (kept {len(keep)}/{len(ts)} samples for the fit)")
    for (t, I, pen) in pl:
        print(f"     t={t:10.3e} d   I={I:11.5e} m   I/sqrt(t)={I/np.sqrt(t):8.5f}   front~{pen*100:5.2f} cm")
    print(f"  --> Philip exponent p = {p:.4f}   (target ~0.50)")
    print(f"  --> S_num = {S_num:.5f} +/- {S_spread:.5f} m/day^0.5    "
          f"vs S_an {S_an:.5f}   rel-err {abs(S_num-S_an)/S_an*100:5.1f}%")

    cy = _uptake_run(soil, cylindrical=True, r_w=r_w, psi_i=psi_i, L=L, N=N, samples=samples)
    # The cyl/planar uptake-per-wall-area ratio is ~1 + c*(penetration/r_w): an OUTWARD-diverging
    # annulus exposes more soil volume than a planar wall, so per-area uptake grows with penetration.
    # Correct r-weighting <=> the ratio EXTRAPOLATES to 1 at zero penetration. Fit ratio vs pen/r_w
    # over the small-penetration samples and check the intercept (NOT a fixed-index ratio, which for a
    # fast/medium soil samples a front already a sizeable fraction of r_w -- the loam "fail" before).
    print(f"  CYLINDRICAL vs PLANAR (uptake per wall area; ratio -> 1 as pen -> 0 validates r-weighting):")
    xs, ys = [], []
    for (t, Ic, _), (_, Ip, penp) in zip(cy, pl):
        ratio = Ic / Ip if Ip > 0 else float("nan")
        x = penp / r_w
        if x < 1.0:
            xs.append(x); ys.append(ratio)
        print(f"     t={t:10.3e} d   cyl/planar={ratio:7.4f}   pen/r_w={x:5.3f}")
    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"  --> ratio ~ {intercept:.4f} + {slope:.3f}*(pen/r_w)   (intercept -> 1.0 validates r-weighting)")

    p_ok = abs(p - 0.5) < 0.05
    s_ok = abs(S_num - S_an) / S_an < 0.15
    cyl_ok = abs(intercept - 1.0) < 0.03
    print(f"  GATE: exponent~0.5 [{ '_PASS_' if p_ok else '_FAIL_' }]  "
          f"S match<15% [{ '_PASS_' if s_ok else '_FAIL_' }]  "
          f"cyl r-weight (intercept {intercept:.4f}) [{ '_PASS_' if cyl_ok else '_FAIL_' }]")
    return p_ok and s_ok and cyl_ok


if __name__ == "__main__":
    ok = []
    ok.append(run_soil("LOAM", LOAM, psi_i=-1.0, t_end=4e-2, L=0.5, N=600, r_w=0.05))
    ok.append(run_soil("SAND", SAND, psi_i=-1.0, t_end=2e-3, L=0.5, N=800, r_w=0.05))
    print(f"\n{'='*78}\nPHASE-1a SUMMARY: {sum(ok)}/{len(ok)} soils pass the sorptivity ground-truth gate")
    print("=" * 78)
