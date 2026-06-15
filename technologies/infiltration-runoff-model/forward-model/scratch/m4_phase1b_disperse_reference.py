"""Module 4 (§E) Phase-1b: the DISPERSE near-field q(t) ground-truth tables (all 4 soils, 2 geometries).

The Phase-3 fidelity gate (claim C-004) requires the embedded sorptive leg sigma to reproduce a
RESOLVED single-feature reference q(t) with a SINGLE a-priori well-index C across >=2 geometries. This
builds those reference tables for the DISPERSE direction (saturated wall psi=0 dispersing into dry
soil), for SAND/LOAM/SILT/CLAY, on the two PIDS geometries:

  TUNNEL (vertical feature): the radial wetting from a uniformly-saturated vertical wall is z-INVARIANT,
    and a z-invariant field exactly satisfies the gravity term (K*d_z(psi+z): d_z(z)=1 -> d_z K = 0 when
    psi is z-invariant). So gravity is IRRELEVANT to the tunnel's radial uptake -> it reduces to a 1-D
    CYLINDRICAL RADIAL absorption (validated against Philip S*sqrt(t) in Phase-1a). We VERIFY the
    z-invariance numerically: a full 2-D axisymmetric (r,z) solve WITH gravity must match the 1-D radial.

  DRAIN (horizontal feature): the line lies along x, so its cross-section is the (y,z) plane and gravity
    (along z) sits IN that plane -> the circular wall's top and bottom wet asymmetrically. This is a
    genuine 2-D problem on a body-fitted ANNULUS mesh (built directly in physical (y,z) coords; no gmsh).
    Early time -> radial sorption (matches the tunnel); late time -> gravity elongates the plume down.

The two geometries genuinely differ (drain adds the gravity asymmetry), which is exactly what makes the
Phase-3 "single C across geometries" gate a real test. Uptake is reported as I(t) = cumulative uptake
per unit WALL AREA [m], directly comparable across geometries. Tables are saved to
scratch/m4_phase1b_disperse_refs.npz for Phase-3 to load without re-running.

Run from forward-model/:
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scratch/m4_phase1b_disperse_reference.py
"""
from __future__ import annotations

import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten

COMM = MPI.COMM_WORLD

# Carsel & Parrish (1988) van Genuchten textures (alpha 1/m, Ks m/day).
SOILS = {
    "SAND": VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13),
    "LOAM": VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25),
    "SILT": VanGenuchten(theta_r=0.034, theta_s=0.46, alpha=1.6, n=1.37, Ks=0.06),
    "CLAY": VanGenuchten(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.048),
}
# per-soil window (psi_i dry head [m], t_end [day]); psi_i=-1 throughout (matches the §D test). CLAY is
# near-saturated at -1 m (flat n=1.09 retention) so its disperse signal is inherently small -- reported.
CONF = {
    "SAND": dict(psi_i=-1.0, t_end=3e-3),
    "LOAM": dict(psi_i=-1.0, t_end=6e-2),
    "SILT": dict(psi_i=-1.0, t_end=4e-1),
    "CLAY": dict(psi_i=-1.0, t_end=4e-1),
}

R_W = 0.05      # feature radius [m]
R_OUT = 0.55    # outer (far-field) radius [m]; R = 0.5 m radial soil extent
LZ = 0.40       # axisym (r,z) height for the z-invariance check [m]

_LU = {  # cp (NOT bt) linesearch: bt stalls on the hard saturated-wall psi=0 step (memory 2026-06-08).
    "snes_type": "newtonls", "snes_linesearch_type": "cp",
    "snes_rtol": 1e-10, "snes_atol": 1e-12, "snes_max_it": 50,
    "ksp_type": "preonly", "pc_type": "lu",
}


def _vertex_dx():
    return ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})


def parlange_sorptivity(soil, psi_i, psi_w=0.0, npts=4001):
    """Parlange (1975) sorptivity S [m/day^0.5]:  S^2 = int_{psi_i}^{0} (theta0+theta-2 theta_i) K dpsi."""
    psi = np.linspace(psi_i, psi_w, npts)
    th = soil.theta(psi)
    th_i, th0 = float(soil.theta(psi_i)), float(soil.theta(psi_w))
    return float(np.sqrt(max(float(np.trapezoid((th0 + th - 2.0 * th_i) * soil.K(psi), psi)), 0.0)))


def _solve_to(problem, psi, psi_n, dt_c, t0, t_target, dt, dt_max=np.inf):
    """Adaptive backward-Euler march t0->t_target. dt_max caps every BE step (the adaptive
    controller still drops BELOW it on Newton stiffness, never rises above it). Default inf =
    the original uncapped behavior (disperse refs unchanged); the drain refs pass window/2048 to
    remove the ~4% first-order BE temporal under-count documented in item (B), 2026-06-15
    (validation/sanity/m4_phase4_drain_endbias_attribution__2026-06-15.md)."""
    t = t0
    while t < t_target - 1e-15:
        h = min(dt, t_target - t, dt_max)
        dt_c.value = h
        problem.solve()
        snes = problem.solver
        if snes.getConvergedReason() > 0:
            psi_n.x.array[:] = psi.x.array
            psi_n.x.scatter_forward()
            t += h
            it = int(snes.getIterationNumber())
            dt = dt * 1.5 if it <= 4 else (dt * 0.7 if it >= 8 else dt)
            dt = min(dt, dt_max)
        else:
            psi.x.array[:] = psi_n.x.array
            psi.x.scatter_forward()
            dt *= 0.5
            assert dt > 1e-13, f"dt collapse near t={t:.3e}"
    return dt


def _annulus_mesh(r_w, r_out, n_r, n_phi, grade=1.5):
    """Body-fitted annulus in physical (y,z): inner ring r=r_w (wall), outer r=r_out (far field)."""
    s = np.linspace(0.0, 1.0, n_r + 1) ** grade
    rs = r_w + (r_out - r_w) * s
    phis = 2.0 * np.pi * np.arange(n_phi) / n_phi
    Rg, Pg = np.meshgrid(rs, phis, indexing="ij")
    pts = np.column_stack([(Rg * np.cos(Pg)).ravel(), (Rg * np.sin(Pg)).ravel()]).astype(np.float64)
    idx = lambda j, k: j * n_phi + (k % n_phi)
    cells = []
    for j in range(n_r):
        for k in range(n_phi):
            a, b, c, d = idx(j, k), idx(j, k + 1), idx(j + 1, k + 1), idx(j + 1, k)
            cells.append((a, b, c)); cells.append((a, c, d))
    cells = np.array(cells, dtype=np.int64)
    el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    return dmesh.create_mesh(COMM, cells, ufl.Mesh(el), pts)


def _run(msh, soil, psi_i, samples, *, weight, gravity, bc_locs, wall_area, front_coord, label=""):
    """Generic resolved disperse run. weight: UFL volume weight (r for axisym, 1 for Cartesian).
    bc_locs=(wall_locator, far_locator). Returns list of (t, I_per_wall_area, penetration)."""
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi = fem.Function(V); psi_n = fem.Function(V)
    psi.x.array[:] = psi_i; psi_n.x.array[:] = psi_i
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    dxs = _vertex_dx()
    # Cap the Darcy-volume quadrature at degree 8: van Genuchten K/theta fractional powers blow FFCX's
    # auto-estimate to ~26, which on 2-D triangles is a big assembly cost (memory: quadrature-degree-cap).
    # NB (review 2026-06-08): 1b/1c cap at 8 here; the 1a sorptivity ground truth uses FFCX AUTO degree.
    # The two differ ~0.28% on early-time S near the psi=h_s air-entry kink (the documented cap deviation),
    # so 1b/1c's "vs Parlange" % already folds in that ~0.3% cap bias. No correctness impact (a 15% gate).
    dxq = ufl.dx(metadata={"quadrature_degree": 8})
    theta, theta_n, K = soil.theta_ufl(psi), soil.theta_ufl(psi_n), soil.K_ufl(psi)
    darcy = ufl.grad(psi)
    if gravity:
        gd = msh.geometry.dim
        eg = np.zeros(gd, dtype=PETSc.ScalarType); eg[-1] = 1.0
        darcy = darcy + fem.Constant(msh, eg)
    F = ((theta - theta_n) / dt_c) * v * weight * dxs + K * ufl.dot(darcy, ufl.grad(v)) * weight * dxq
    wall_dofs = fem.locate_dofs_geometrical(V, bc_locs[0])
    far_dofs = fem.locate_dofs_geometrical(V, bc_locs[1])
    bcs = [fem.dirichletbc(PETSc.ScalarType(0.0), wall_dofs, V),
           fem.dirichletbc(PETSc.ScalarType(psi_i), far_dofs, V)]
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4p1b_", petsc_options=_LU)

    th_i = float(soil.theta(psi_i))
    stored = fem.form((theta - th_i) * weight * dxs)
    xc = V.tabulate_dof_coordinates()
    fc = front_coord(xc)                       # radius (or coord) used to measure penetration
    out, dt, t_prev = [], 1e-8, 0.0
    for i, t_s in enumerate(samples):
        dt = _solve_to(problem, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        I = COMM.allreduce(fem.assemble_scalar(stored), op=MPI.SUM) / wall_area
        wet = psi.x.array > psi_i + 0.05 * abs(psi_i)
        pen = (COMM.allreduce(float(fc[wet].max()) if np.any(wet) else 0.0, op=MPI.MAX)) - R_W
        out.append((t_s, float(I), float(pen)))
        if label:
            print(f"      [{label}] {i+1}/{len(samples)}  t={t_s:.3e}  I={I:.4e}  pen={pen*100:.2f}cm", flush=True)
    return out


def radial_tunnel(soil, psi_i, samples, n=400, label=""):
    msh = dmesh.create_interval(COMM, n, [R_W, R_OUT])
    r = ufl.SpatialCoordinate(msh)[0]
    return _run(msh, soil, psi_i, samples, weight=r, gravity=False,
                bc_locs=(lambda x: np.isclose(x[0], R_W), lambda x: np.isclose(x[0], R_OUT)),
                wall_area=R_W, front_coord=lambda xc: xc[:, 0], label=label)


def axisym_rz(soil, psi_i, samples, n_r=160, n_z=64, label=""):
    msh = dmesh.create_rectangle(COMM, [[R_W, 0.0], [R_OUT, LZ]], [n_r, n_z], dmesh.CellType.triangle)
    r = ufl.SpatialCoordinate(msh)[0]
    return _run(msh, soil, psi_i, samples, weight=r, gravity=True,
                bc_locs=(lambda x: np.isclose(x[0], R_W), lambda x: np.isclose(x[0], R_OUT)),
                wall_area=R_W * LZ, front_coord=lambda xc: xc[:, 0], label=label)


def drain_annulus(soil, psi_i, samples, n_r=44, n_phi=128, label=""):
    msh = _annulus_mesh(R_W, R_OUT, n_r, n_phi)
    p_wall = n_phi * 2.0 * R_W * np.sin(np.pi / n_phi)   # faceted wall perimeter (per unit axial length)
    rr = lambda xc: np.hypot(xc[:, 0], xc[:, 1])
    return _run(msh, soil, psi_i, samples, weight=fem.Constant(msh, PETSc.ScalarType(1.0)), gravity=True,
                bc_locs=(lambda x: np.isclose(np.hypot(x[0], x[1]), R_W, atol=1e-7),
                         lambda x: np.isclose(np.hypot(x[0], x[1]), R_OUT, atol=1e-7)),
                wall_area=p_wall, front_coord=rr, label=label)


def _sqrt_fit(rows, S_an):
    """Early-time radial sorption check. I/sqrt(t) ~ S0*(1 + c*pen/r_w): the cylindrical curvature
    inflates it with penetration, so we EXTRAPOLATE to zero penetration (intercept = the planar-
    equivalent sorptivity) and compare that to analytical Parlange S_an -- the apples-to-apples cross-
    check (the raw radial value carries the ~10% curvature, as Phase-1a quantified).

    PRECISION CAVEAT (adversarial review 2026-06-08): `pen` is a NODE-quantized front (the outermost node
    above the wetting threshold), so it under-estimates the true sub-cell front by ~1 cell, biasing the
    extrapolated intercept ~+0.6% HIGH -- i.e. the reported "0.5% vs Parlange" already includes a ~+0.5%
    detector bias (true geometric agreement is ~0.1% after a sub-cell correction). This is a DIAGNOSTIC-only
    band: the actual Phase-3 reference is the I(t) DOMAIN INTEGRAL (`(theta-th_i)*weight*dxs`), which is
    independent of the front detector and is mesh/dt/quadrature-converged to ~0.2%. The saved `*_pen` arrays
    are diagnostic only -- Phase 3 must NOT consume them as load-bearing."""
    pts = [(pen / R_W, I / np.sqrt(t)) for (t, I, pen) in rows if t > 0 and 0 < pen < 0.6 * R_W]
    if len(pts) < 3:
        return None, None, len(pts)
    x = np.array([p for p, _ in pts]); y = np.array([s for _, s in pts])
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), abs(intercept - S_an) / S_an, len(pts)


if __name__ == "__main__":
    saved = {}
    print("=" * 84)
    print("PHASE-1b DISPERSE near-field references  (r_w=%.2f m, R_out=%.2f m)" % (R_W, R_OUT))
    print("=" * 84)
    for name, soil in SOILS.items():
        c = CONF[name]
        psi_i, t_end = c["psi_i"], c["t_end"]
        Se_i = float(soil.effective_saturation(psi_i))
        S_an = parlange_sorptivity(soil, psi_i)
        # per-soil early start so the FIRST sample's (planar) penetration ~0.1 r_w -> the radial-sorption
        # S*sqrt(t) regime is captured for every soil (a fixed t_end/80 start is far too late for the slow
        # fine soils -- silt/clay -- whose front is already past 0.6 r_w by then).
        dtheta = max(float(soil.theta(0.0)) - float(soil.theta(psi_i)), 1e-4)
        t_start = (dtheta * 0.1 * R_W / S_an) ** 2
        samples = list(np.geomspace(t_start, t_end, 24))
        print(f"\n{'-'*84}\n{name}: psi_i={psi_i} m (Se_i={Se_i:.3f}),  S_an={S_an:.5f} m/day^0.5,  t_end={t_end} d")

        tun = radial_tunnel(soil, psi_i, samples, label=f"{name} tunnel")
        dr = drain_annulus(soil, psi_i, samples, label=f"{name} drain")
        saved[f"{name}_t"] = np.array([r[0] for r in tun])
        saved[f"{name}_tunnel_I"] = np.array([r[1] for r in tun])
        saved[f"{name}_drain_I"] = np.array([r[1] for r in dr])
        saved[f"{name}_tunnel_pen"] = np.array([r[2] for r in tun])
        saved[f"{name}_drain_pen"] = np.array([r[2] for r in dr])

        print("   t [day]      tunnel I    drain I    drain/tun   tun pen   drain pen   (I per wall area [m])")
        for (t, It, pent), (_, Id, pend) in zip(tun, dr):
            ratio = Id / It if It > 0 else float("nan")
            print(f"   {t:10.3e}  {It:10.4e}  {Id:10.4e}   {ratio:7.4f}   {pent*100:6.2f}cm  {pend*100:6.2f}cm")

        St, et, nt = _sqrt_fit(tun, S_an)
        Sd, ed, nd = _sqrt_fit(dr, S_an)
        msg_t = f"S0(extrap)={St:.5f} ({et*100:.1f}% vs Parlange, n={nt})" if St else f"too few early samples (n={nt})"
        msg_d = f"S0(extrap)={Sd:.5f} ({ed*100:.1f}% vs Parlange, n={nd})" if Sd else f"too few early samples (n={nd})"
        late = dr[-1][1] / tun[-1][1] if tun[-1][1] > 0 else float("nan")
        print(f"   early radial S sqrt(t) (pen->0 extrap):  tunnel {msg_t}")
        print(f"                                            drain  {msg_d}")
        print(f"   late drain/tunnel ratio (gravity asymmetry) = {late:.4f}  "
              f"(>1 expected: gravity elongates the plume)")

    # z-invariance verification (gravity irrelevant for the tunnel) on LOAM: 2-D axisym(r,z)+gravity vs 1-D radial
    print(f"\n{'-'*84}\nZ-INVARIANCE CHECK (LOAM): 2-D axisym(r,z) WITH gravity  vs  1-D radial (gravity-free)")
    loam, ci = SOILS["LOAM"], CONF["LOAM"]
    samp = list(np.geomspace(ci["t_end"] / 80.0, ci["t_end"], 8))
    r1 = radial_tunnel(loam, ci["psi_i"], samp)
    r2 = axisym_rz(loam, ci["psi_i"], samp)
    maxdev = max(abs(a[1] - b[1]) / a[1] for a, b in zip(r1, r2) if a[1] > 0)
    for (t, I1, _), (_, I2, _) in zip(r1, r2):
        print(f"   t={t:10.3e}  radial I={I1:10.4e}  axisym(r,z)+g I={I2:10.4e}  rel-dev={abs(I1-I2)/I1:.2e}")
    print(f"   --> max rel deviation = {maxdev:.2e}   "
          f"[{'_PASS_ gravity irrelevant (z-invariant)' if maxdev < 0.02 else '_FAIL_'}]")

    out_path = "scratch/m4_phase1b_disperse_refs.npz"
    np.savez(out_path, **saved)
    print(f"\nSaved disperse ground-truth tables -> {out_path}")
    print("=" * 84)
