"""Module 4 (§E) Phase-1c: the DRAIN near-field q(t) ground-truth tables (the exfiltration mirror).

The DRAIN direction (soil wetter than the feature -> water leaves the soil INTO the feature, which acts
as a sink) is the second half of the Phase-1 reference set and the second axis of the Phase-3 gate
("single a-priori C across geometries AND both directions"). This builds the resolved drain reference
for all 4 soils on the 2 geometries, as the DIRECT MIRROR of the disperse run (Phase-1b):

  disperse:  wall psi=0  (high),  soil psi_i=-1 (low)  -> water OUT of the wall (capillary sorption, S*sqrt(t))
  DRAIN:     wall psi=-1 (low),   soil psi_i=0  (wet)  -> water INTO the wall  (matrix DESORPTION)

Same head drop |dH|=1 m, reversed -> the Kirchhoff potential difference Phi(psi_far)-Phi(psi_wall) =
int_{-1}^{0} K dpsi is IDENTICAL to disperse. So the sign-symmetric embedded sigma predicts the SAME
flux magnitude both ways; the resolved drain reference tests whether real Richards agrees (the crux of
the "both directions, one C" claim). This is matrix DESORPTION (the nonlinear regime, the stringent
test); saturated perched-table GRAVITY drainage (psi_i>0 -> wall 0, the K=Ks linear regime) is the
easier complementary case, deferred.

Drain is NOT governed by Philip sorption, so the checks differ from S*sqrt(t):
  (1) MASS BALANCE: water removed from the soil = the integral of theta loss (structural in I(t)).
  (2) MONOTONE DRAWDOWN: near-wall theta decreases monotonically (no spurious oscillation).
  (3) SELF-LIMITING FLUX: dI/dt decreases as the near-wall soil desaturates and K(psi) throttles.
  (4) DESORPTIVITY: early-time I(t) ~ S_des*sqrt(t) (Philip desorptivity); report S_des and compare to
      the disperse SORPTIVITY S (loaded from the Phase-1b npz) -- the resolved sign-(a)symmetry.
Tunnel = 1-D radial (gravity z-invariant, same argument as disperse, re-verified); drain = 2-D annulus.

Run from forward-model/:
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scratch/m4_phase1c_drain_reference.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

import scratch.m4_phase1b_disperse_reference as dz   # reuse soils, mesh builders, marcher, constants

COMM = MPI.COMM_WORLD
SOILS, R_W, R_OUT, LZ = dz.SOILS, dz.R_W, dz.R_OUT, dz.LZ

# DRAIN is the OPPOSITE stiff case to disperse: the soil starts SATURATED (psi_i=0 -> C=0 in the whole
# bulk), and cp linesearch STALLS there at small dt (DIVERGED_MAX_IT), while bt converges robustly --
# the mirror image of the disperse saturated-WALL case (where bt stalled and cp worked). So the drain
# uses bt. (We keep psi_i=0: for drainage the near-saturated range is where K is largest and the fast
# initial drainage / Kirchhoff driving potential lives -- a slightly-unsaturated start would gut it.)
_DRAIN_LU = dict(dz._LU, snes_linesearch_type="bt")

# DRAIN config: wet soil psi_i (saturated, 0), low wall head psi_wall (-1) = the direct disperse mirror.
PSI_I, PSI_WALL = 0.0, -1.0
CONF = {  # t_end per soil (drainage timescale ~ disperse; monitored/windowed by the drained front)
    "SAND": 3e-3, "LOAM": 6e-2, "SILT": 4e-1, "CLAY": 4e-1,
}


def _run_drain(msh, soil, samples, *, weight, gravity, bc_locs, wall_area, front_coord, label=""):
    """Resolved DRAIN run: soil starts wet (PSI_I), wall pulled to PSI_WALL -> water drains into the wall.
    Returns (t, I_removed_per_wall_area, drained_penetration). I = int(theta_i - theta)/wall_area >= 0."""
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi = fem.Function(V); psi_n = fem.Function(V)
    psi.x.array[:] = PSI_I; psi_n.x.array[:] = PSI_I
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    dxs = dz._vertex_dx()
    dxq = ufl.dx(metadata={"quadrature_degree": 8})
    theta, theta_n, K = soil.theta_ufl(psi), soil.theta_ufl(psi_n), soil.K_ufl(psi)
    darcy = ufl.grad(psi)
    if gravity:
        eg = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType); eg[-1] = 1.0
        darcy = darcy + fem.Constant(msh, eg)
    F = ((theta - theta_n) / dt_c) * v * weight * dxs + K * ufl.dot(darcy, ufl.grad(v)) * weight * dxq
    wall_dofs = fem.locate_dofs_geometrical(V, bc_locs[0])
    far_dofs = fem.locate_dofs_geometrical(V, bc_locs[1])
    bcs = [fem.dirichletbc(PETSc.ScalarType(PSI_WALL), wall_dofs, V),   # the drain sink (low head)
           fem.dirichletbc(PETSc.ScalarType(PSI_I), far_dofs, V)]       # wet far reservoir
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4p1c_", petsc_options=_DRAIN_LU)

    th_i = float(soil.theta(PSI_I))
    removed = fem.form((th_i - theta) * weight * dxs)     # water LOST by the soil (>=0), per wall area below
    theta_min_form = fem.form(theta * dxs)                 # for monotone-drawdown (total theta only goes down)
    xc = V.tabulate_dof_coordinates(); fc = front_coord(xc)
    dH = abs(PSI_I - PSI_WALL)
    out, dt, t_prev = [], 1e-8, 0.0
    for i, t_s in enumerate(samples):
        dt = dz._solve_to(problem, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        I = COMM.allreduce(fem.assemble_scalar(removed), op=MPI.SUM) / wall_area
        drained = psi.x.array < PSI_I - 0.05 * dH
        pen = (COMM.allreduce(float(fc[drained].max()) if np.any(drained) else 0.0, op=MPI.MAX)) - R_W
        out.append((t_s, float(I), float(pen)))
        if label:
            print(f"      [{label}] {i+1}/{len(samples)}  t={t_s:.3e}  I={I:.4e}  pen={pen*100:.2f}cm", flush=True)
    return out


def radial_tunnel(soil, samples, n=3200, label=""):
    # n=3200 (not 400): the saturated-start desaturation front at the wall is SHARP (C=0 in the saturated
    # bulk -> a near-discontinuity), so the early-window S_des fit is mesh-sensitive. Convergence study
    # (LOAM ratio): n=400->0.506, 800->0.474, 1600->0.468, 3200->0.465 (adversarial review 2026-06-08;
    # the disperse wetting-into-dry front is much softer and was already converged at n=400).
    msh = dmesh.create_interval(COMM, n, [R_W, R_OUT])
    r = ufl.SpatialCoordinate(msh)[0]
    return _run_drain(msh, soil, samples, weight=r, gravity=False,
                      bc_locs=(lambda x: np.isclose(x[0], R_W), lambda x: np.isclose(x[0], R_OUT)),
                      wall_area=R_W, front_coord=lambda xc: xc[:, 0], label=label)


def axisym_rz(soil, samples, n_r=160, n_z=64, label=""):
    """2-D axisym (r,z) DRAIN with gravity, for the z-invariance check ONLY. Top/bottom z-boundaries get
    FREE-DRAINAGE (unit-gradient throughflow): we subtract the gravity flux K(eg.n) so the effective BC is
    grad(psi).n=0 -- which is what represents an INFINITE vertical tunnel. (NO-FLOW there is WRONG for a
    saturated drain: it dams the gravity throughflow -> hydrostatic pile-up at the bottom + over-desaturation
    at the top -> a spurious ~179% over-drainage. Probed 2026-06-08.) With free drainage the column-integrated
    uptake/wall-area CONVERGES to the 1-D radial value at LATE time (<0.3%), confirming gravity is z-invariant
    for the tunnel's RADIAL drainage, so the 1-D radial reference is the exact asymptotic target.

    The remaining EARLY-time deviation (~22% at the first sample) is NOT an edge or gravity effect (an earlier
    comment mislabeled it a "finite-column edge transient" -- corrected per the adversarial review 2026-06-08,
    which proved it is LZ-INDEPENDENT, bit-identical across LZ 0.4/0.8/1.6, AND identical with gravity OFF). It
    is a near-wall DISCRETIZATION transient: the 1-D interval and the 2-D rectangle resolve the SHARP saturated-
    start desaturation front differently at early time. It shrinks as both meshes are refined; the late-time
    convergence (not the early window) is the z-invariance proof."""
    msh = dmesh.create_rectangle(COMM, [[R_W, 0.0], [R_OUT, LZ]], [n_r, n_z], dmesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", 1))
    r = ufl.SpatialCoordinate(msh)[0]; nrm = ufl.FacetNormal(msh)
    psi = fem.Function(V); psi_n = fem.Function(V); psi.x.array[:] = PSI_I; psi_n.x.array[:] = PSI_I
    v = ufl.TestFunction(V); dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    dxs = dz._vertex_dx(); dxq = ufl.dx(metadata={"quadrature_degree": 8})
    th = soil.theta_ufl(psi); thn = soil.theta_ufl(psi_n); K = soil.K_ufl(psi)
    eg = fem.Constant(msh, PETSc.ScalarType([0.0, 1.0]))
    fdim = msh.topology.dim - 1; msh.topology.create_connectivity(fdim, msh.topology.dim)
    tb = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], LZ))
    ft = dmesh.meshtags(msh, fdim, np.sort(tb), np.ones(tb.size, dtype=np.int32))
    ds_tb = ufl.Measure("ds", domain=msh, subdomain_data=ft, metadata={"quadrature_degree": 8})(1)
    F = ((th - thn) / dt_c) * v * r * dxs + K * ufl.dot(ufl.grad(psi) + eg, ufl.grad(v)) * r * dxq \
        - K * ufl.dot(eg, nrm) * v * r * ds_tb                       # free-drainage throughflow top/bottom
    wall = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], R_W))
    far = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], R_OUT))
    bcs = [fem.dirichletbc(PETSc.ScalarType(PSI_WALL), wall, V),
           fem.dirichletbc(PETSc.ScalarType(PSI_I), far, V)]
    problem = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4p1c_az_", petsc_options=_DRAIN_LU)
    thi = float(soil.theta(PSI_I)); rem = fem.form((thi - th) * r * dxs)
    out = []; dt = 1e-8; tp = 0.0
    for ts in samples:
        dt = dz._solve_to(problem, psi, psi_n, dt_c, tp, ts, dt); tp = ts
        out.append((ts, COMM.allreduce(fem.assemble_scalar(rem), op=MPI.SUM) / (R_W * LZ), 0.0))
    return out


def drain_annulus(soil, samples, n_r=160, n_phi=128, grade=2.5, label=""):
    # n_r=160 + steeper near-wall grading (grade 2.5 vs the disperse default 1.5): same sharp-desaturation
    # convergence need as the tunnel. Study (LOAM ratio): Nr=44->0.559, 88->0.486, 132->0.473;
    # grade2.5 Nr160->0.462 (converged, matches tunnel n=3200's 0.465). Adversarial review 2026-06-08.
    msh = dz._annulus_mesh(R_W, R_OUT, n_r, n_phi, grade)
    p_wall = n_phi * 2.0 * R_W * np.sin(np.pi / n_phi)
    return _run_drain(msh, soil, samples, weight=fem.Constant(msh, PETSc.ScalarType(1.0)), gravity=True,
                      bc_locs=(lambda x: np.isclose(np.hypot(x[0], x[1]), R_W, atol=1e-7),
                               lambda x: np.isclose(np.hypot(x[0], x[1]), R_OUT, atol=1e-7)),
                      wall_area=p_wall, front_coord=lambda xc: np.hypot(xc[:, 0], xc[:, 1]), label=label)


def planar_desorptivity(soil, samples, n=3200, L=0.5, label=""):
    """Fine 1-D PLANAR (Cartesian, gravity-off) desorption -> the planar-equivalent DESORPTIVITY anchor.
    This is the robust in-suite cross-check the adversarial review asked for: the radial tunnel S_des,
    extrapolated to zero penetration, must MATCH this planar desorptivity (within the cylindrical-curvature
    tolerance) -- the desorption mirror of the disperse 1a planar/cylindrical validation. (We use this FEM
    planar anchor rather than a Bruce-Klute solve_bvp: the similarity ODE has D=K/C -> infinity at the
    saturated far boundary, which is numerically singular -- the review's analytical attempt blew up on SAND.
    The review DID independently confirm the asymmetry by three methods: a flux-concentration iteration, a
    regularized solve_bvp (LOAM 0.53/SILT 0.61/CLAY 0.70), and a constant-D control returning EXACTLY 1.000,
    proving the asymmetry comes solely from the nonlinear D(theta) skew -- the classic Philip result.)"""
    msh = dmesh.create_interval(COMM, n, [0.0, L])
    return _run_drain(msh, soil, samples, weight=fem.Constant(msh, PETSc.ScalarType(1.0)), gravity=False,
                      bc_locs=(lambda x: np.isclose(x[0], 0.0), lambda x: np.isclose(x[0], L)),
                      wall_area=1.0, front_coord=lambda xc: xc[:, 0] + R_W, label=label)


def _desorptivity(rows):
    """Early-time desorptivity: I/sqrt(t) ~ S_des*(1 + c*pen/r_w); extrapolate pen->0 (the planar-equiv
    desorptivity). Also returns whether dI/dt is monotone-decreasing (self-limiting) over the run."""
    pts = [(pen / R_W, I / np.sqrt(t)) for (t, I, pen) in rows if t > 0 and 0 < pen < 0.6 * R_W]
    S_des = None
    if len(pts) >= 3:
        x = np.array([p for p, _ in pts]); y = np.array([s for _, s in pts])
        S_des = float(np.polyfit(x, y, 1)[1])
    ts = np.array([t for (t, _, _) in rows]); Is = np.array([I for (_, I, _) in rows])
    dIdt = np.diff(Is) / np.diff(ts)
    self_limiting = bool(np.all(np.diff(dIdt) <= 1e-9 * max(abs(dIdt).max(), 1e-30)))  # dI/dt non-increasing
    return S_des, self_limiting, len(pts)


if __name__ == "__main__":
    try:
        ref = np.load("scratch/m4_phase1b_disperse_refs.npz")   # disperse sorptivity reference
    except FileNotFoundError:
        ref = None
    saved = {}
    print("=" * 88)
    print(f"PHASE-1c DRAIN near-field references  (soil psi_i={PSI_I}, wall psi_wall={PSI_WALL}; "
          f"r_w={R_W} m, R_out={R_OUT} m)")
    print("=" * 88)
    for name, soil in SOILS.items():
        t_end = CONF[name]
        S_sorp = dz.parlange_sorptivity(soil, PSI_WALL)   # disperse Parlange S over the SAME [-1,0] range
        Se_i = float(soil.effective_saturation(PSI_I))
        # early start so the desaturation front is captured small (mirror of the disperse fix)
        dtheta = max(float(soil.theta(PSI_I)) - float(soil.theta(PSI_WALL)), 1e-4)
        t_start = (dtheta * 0.1 * R_W / max(S_sorp, 1e-6)) ** 2
        samples = list(np.geomspace(t_start, t_end, 24))
        print(f"\n{'-'*88}\n{name}: Se_i={Se_i:.3f} (wet),  Parlange S(disperse,[-1,0])={S_sorp:.5f},  t_end={t_end} d")

        tun = radial_tunnel(soil, samples, label=f"{name} drain-tunnel")
        dr = drain_annulus(soil, samples, label=f"{name} drain-annulus")
        saved[f"{name}_t"] = np.array([r[0] for r in tun])
        saved[f"{name}_tunnel_I"] = np.array([r[1] for r in tun])
        saved[f"{name}_drain_I"] = np.array([r[1] for r in dr])
        saved[f"{name}_tunnel_pen"] = np.array([r[2] for r in tun])
        saved[f"{name}_drain_pen"] = np.array([r[2] for r in dr])

        print("   t [day]      tun I(rem)  drain I    drain/tun   tun pen   drain pen   disp/drain (sign-sym)")
        disp_t = ref[f"{name}_t"] if ref is not None else None
        disp_I = ref[f"{name}_tunnel_I"] if ref is not None else None
        for (t, It, pent), (_, Id, pend) in zip(tun, dr):
            ratio = Id / It if It > 0 else float("nan")
            ds = (np.interp(t, disp_t, disp_I) / It) if (disp_t is not None and It > 0) else float("nan")
            print(f"   {t:10.3e}  {It:10.4e}  {Id:10.4e}   {ratio:7.4f}   {pent*100:6.2f}cm  {pend*100:6.2f}cm   {ds:7.4f}")

        St, slim_t, nt = _desorptivity(tun)
        Sd, slim_d, nd = _desorptivity(dr)
        pl = planar_desorptivity(soil, samples, label=f"{name} drain-planar")
        Sp, slim_p, npn = _desorptivity(pl)
        saved[f"{name}_planar_I"] = np.array([r[1] for r in pl])
        msg_t = f"S_des={St:.5f} (desorptivity/sorptivity = {St/S_sorp:.3f})" if St else f"n<3 ({nt})"
        print(f"   tunnel desorptivity:  {msg_t};  self-limiting dI/dt: {slim_t}")
        if Sd:
            print(f"   drain  desorptivity:  S_des={Sd:.5f} (ratio {Sd/S_sorp:.3f});  self-limiting dI/dt: {slim_d}")
        # PLANAR anchor (the review's requested independent falsifiability): the radial tunnel S_des
        # extrapolated to pen->0 must MATCH the planar desorptivity (geometry consistency); the planar
        # desorptivity/sorptivity is the geometry-INDEPENDENT asymmetry the Phase-3 closure must capture.
        if Sp and St:
            print(f"   PLANAR anchor:  S_des_planar={Sp:.5f} (asymmetry {Sp/S_sorp:.3f});  "
                  f"radial/planar = {St/Sp:.3f} (->1.0 validates the r-weighting)")
        # monotone drawdown: theta total is non-increasing over the run (no spurious uptake)
        Is = [r[1] for r in tun]
        mono = all(Is[k+1] >= Is[k] - 1e-12 for k in range(len(Is)-1))   # cumulative removal non-decreasing
        print(f"   monotone drawdown (cum removal non-decreasing): {mono}")

    # z-invariance (gravity irrelevant for the tunnel's RADIAL drainage) -- re-verify for DRAIN on LOAM.
    # The infinite vertical tunnel needs THROUGHFLOW (free-drainage) top/bottom (see axisym_rz): with it,
    # the axisym(r,z)+gravity uptake CONVERGES to the 1-D radial at LATE time, so we judge by the late-time
    # deviation, NOT the max (the early deviation is a near-wall 1-D-vs-2-D discretization transient -- LZ-
    # and gravity-independent, adversarial review 2026-06-08 -- not a real z-dependence; see axisym_rz docstring).
    print(f"\n{'-'*88}\nZ-INVARIANCE CHECK (LOAM drain): 2-D axisym(r,z)+gravity (free-drainage) vs 1-D radial")
    loam = SOILS["LOAM"]
    S_sorp = dz.parlange_sorptivity(loam, PSI_WALL)
    dth = float(loam.theta(PSI_I)) - float(loam.theta(PSI_WALL))
    samp = list(np.geomspace((dth*0.1*R_W/S_sorp)**2, CONF["LOAM"], 8))
    r1 = radial_tunnel(loam, samp)
    r2 = axisym_rz(loam, samp)
    devs = [abs(a[1]-b[1])/a[1] for a, b in zip(r1, r2) if a[1] > 0]
    for (t, I1, _), (_, I2, _) in zip(r1, r2):
        print(f"   t={t:10.3e}  radial I={I1:10.4e}  axisym(r,z)+g I={I2:10.4e}  rel-dev={abs(I1-I2)/I1:.2e}")
    late = float(np.mean(devs[-3:]))   # converged (late-time) deviation = the z-invariance test
    print(f"   --> late-time (converged) deviation = {late:.2e}  "
          f"[{'_PASS_ gravity z-invariant (radial drainage)' if late < 0.02 else '_FAIL_'}]")
    print(f"       early(max) deviation = {max(devs):.2e}  (a near-wall 1-D-vs-2-D DISCRETIZATION transient, "
          f"LZ- & gravity-independent; NOT an edge effect; shrinks with refinement)")

    np.savez("scratch/m4_phase1c_drain_refs.npz", **saved)
    print(f"\nSaved DRAIN ground-truth tables -> scratch/m4_phase1c_drain_refs.npz")
    print("=" * 88)
