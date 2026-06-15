"""Item (A) offline derivation: PIN the resolved-ring radius for the disperse WI-era bridge.

The 2026-06-12 diagnostic (m4_phase4_wi_residual_diag.py) localized the 5.7% disperse WI-era
residual to the ON-RIDGE psi read sitting WETTER than the continuum at r_0 (dPhi -7..-18%), the
bridge FORM exonerated. Arik 2026-06-13 chose the RESOLVED-RING read: drive the steady cylindrical
Kirchhoff bridge q = 2*pi*[Phi(H_f)-Phi(psi_ring)]/ln(r_ring/r_w) off the resolved field at a far
ring instead of the on-ridge node.

A first pass over c*h rings (since superseded) established: (1) the single-ring full-arm Thiem
beats the on-ridge read; (2) a LOCAL log-slope multi-ring fit is far worse (the depleting cone is
not a clean log between adjacent rings); (3) the matched-I rate dev RATEe tracks the ABSOLUTE
radius (r_w units), NOT c*h -- so the read radius is a fixed PHYSICAL radius in the resolved steady
annulus, mesh-independent. This probe (the decisive one) samples fixed absolute radii across the
deployment mesh range n=6,8,10,12: RATEe is flat across meshes at fixed absolute radius, with a
broad sweet spot ~[0.45,0.6]*R_out. r_ring = R_out/2 -> worst |rate dev| 2.0% across n=6..12, the
production choice (the coupled run then needs a Heun lag correction + a live capacity throttle --
see pids_forward/physics/wi_exchange.py; the matched-I dev here is the lag-free lower bound).

The resolved ref field at a physical radius is MESH-INDEPENDENT -> saved once. Run (WSL, from
forward-model/): PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scratch/m4_phase4_wi_ring_derivation.py {ref|emb|analyze}
"""
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

HERE = pathlib.Path(__file__).parent
R_W = 0.05
R_OUT = 40 * R_W
L_BOX = float(np.sqrt(np.pi * (R_OUT ** 2 - R_W ** 2)))
N_SWEEP = (6, 8, 10, 12)
N_DENSE = 256
RS = (12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0)      # candidate radii in units of r_w


def part1_ref():
    import ufl
    from mpi4py import MPI
    from dolfinx import mesh as dmesh, fem
    from dolfinx.fem.petsc import NonlinearProblem
    from petsc4py import PETSc
    import scratch.m4_phase1b_disperse_reference as dz

    soil = dz.SOILS["LOAM"]
    fix = np.load("tests/data/m4_phase4_refA_disperse.npz")
    t32 = fix["LOAM_R40_t"]
    t = np.geomspace(t32[0], t32[-1], N_DENSE)
    cell = 0.5 / 400.0
    n = max(int(round((R_OUT - R_W) / cell)), 40)
    msh = dmesh.create_interval(MPI.COMM_WORLD, n, [R_W, R_OUT])
    r = ufl.SpatialCoordinate(msh)[0]
    V = fem.functionspace(msh, ("Lagrange", 1))
    psi, psi_n = fem.Function(V), fem.Function(V)
    psi.x.array[:] = -1.0; psi_n.x.array[:] = -1.0
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    dxs, dxq = dz._vertex_dx(), ufl.dx(metadata={"quadrature_degree": 8})
    theta, theta_n, K = soil.theta_ufl(psi), soil.theta_ufl(psi_n), soil.K_ufl(psi)
    F = ((theta - theta_n) / dt_c) * v * r * dxs + K * ufl.dot(ufl.grad(psi), ufl.grad(v)) * r * dxq
    wall = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], R_W))
    bcs = [fem.dirichletbc(PETSc.ScalarType(0.0), wall, V)]
    prob = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="zwiabs_", petsc_options=dz._LU)
    th_i = float(soil.theta(-1.0))
    stored = fem.form((theta - th_i) * r * dxs)
    xc = V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(xc)
    I_out = []
    psi_ring = {rs: [] for rs in RS}
    dt, t_prev = 1e-8, 0.0
    for k, t_s in enumerate(t):
        dt = dz._solve_to(prob, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        I_out.append(float(fem.assemble_scalar(stored)) / R_W)
        for rs in RS:
            psi_ring[rs].append(float(np.interp(rs * R_W, xc[order], psi.x.array[order])))
        if k % 32 == 0:
            print(f"  [abs ref] {k+1}/{t.size} t={t_s:.3e} I={I_out[-1]:.4e}", flush=True)
    out = {"t": t, "I": np.array(I_out)}
    for rs in RS:
        out[f"psi_ring_{rs}"] = np.array(psi_ring[rs])
    np.savez(HERE / "m4_phase4_wi_ring_derivation_ref.npz", **out)
    print("Saved -> scratch/m4_phase4_wi_ring_derivation_ref.npz")


def part2_emb(n):
    import ufl
    from mpi4py import MPI
    from dolfinx import mesh as dmesh, fem
    from dolfinx.fem.petsc import NonlinearProblem
    from petsc4py import PETSc
    from pids_forward.physics.feature import EmbeddedFeature
    from pids_forward.physics.wi_exchange import WellIndexExchange
    from scratch.m4_phase4_embedded_harness import SOILS, _LU_BASE, _vertex_dx

    soil = SOILS["LOAM"]
    fix = np.load("tests/data/m4_phase4_refA_disperse.npz")
    t_grid = fix["LOAM_R40_t"]
    L = L_BOX
    h = L / n
    Lx = 4 * h
    msh = dmesh.create_box(MPI.COMM_WORLD, [[0, 0, 0], [Lx, L, L]], [4, n, n])
    feat = EmbeddedFeature(msh, lambda x: np.isclose(x[1], L / 2) & np.isclose(x[2], L / 2),
                           tangent=(1.0, 0.0, 0.0), K_feat=1.0, area=np.pi * R_W ** 2, porosity=0.4)
    feat.configure_sorptive(soil, psi_i=-1.0, psi_wall=0.0)
    V = feat.V
    psi, psi_n = fem.Function(V), fem.Function(V)
    psi.x.array[:] = -1.0; psi_n.x.array[:] = -1.0
    feat.Hf.x.array[:] = 0.0; feat.Hf.x.scatter_forward()
    w = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    dxs, dxq = _vertex_dx(), ufl.dx(metadata={"quadrature_degree": 8})
    th, thn = soil.theta_ufl(psi), soil.theta_ufl(psi_n)
    K = soil.K_ufl(psi)
    rel_c = fem.Constant(msh, PETSc.ScalarType(0.0))
    F = (((th - thn) / dt_c) * w * dxs + K * ufl.dot(ufl.grad(psi), ufl.grad(w)) * dxq
         - rel_c * w * feat.dGamma + feat.sorptive_into_host(w, psi))
    prob = NonlinearProblem(F, psi, bcs=[], petsc_options_prefix="zwiabsge_",
                            petsc_options=dict(_LU_BASE, snes_type="newtonls",
                                               snes_linesearch_type="cp"))
    sch = WellIndexExchange().setup(feat, soil, dict(h=h, t0=float(t_grid[0]), R_out=R_OUT))
    rho = sch._rho
    bands, mean_rho = {}, {}
    for rs in RS:
        m = (np.abs(rho - rs * R_W) <= 0.6 * h) & (rho > 1e-12)
        bands[rs] = m
        mean_rho[rs] = float(rho[m].mean()) if np.any(m) else np.nan
        print(f"  [abs emb n={n}] r={rs} r_w: {int(m.sum())} dofs, mean rho={mean_rho[rs]/R_W:.1f} r_w "
              f"(h={h/R_W:.1f} r_w)", flush=True)
    g = feat._gamma_dofs
    keys = ("t", "I", "era") + tuple(f"psi_ring_{rs}" for rs in RS)
    log = {k: [] for k in keys}
    dt, t_prev = 1e-7, float(t_grid[0])
    for t_s in t_grid[1:]:
        t = t_prev
        while t < t_s - 1e-15:
            hstep = min(dt, t_s - t)
            dt_c.value = hstep
            rel = sch.pre_step(feat, psi, t) or 0.0
            rel_c.value = rel / feat.length
            prob.solve()
            if prob.solver.getConvergedReason() > 0:
                sch.post_step(feat, psi, t, hstep)
                psi_n.x.array[:] = psi.x.array; psi_n.x.scatter_forward()
                t += hstep
                log["t"].append(t); log["I"].append(sch.I_total(feat))
                log["era"].append(0 if sch.in_subgrid_era else 1)
                for rs in RS:
                    log[f"psi_ring_{rs}"].append(
                        float(psi.x.array[bands[rs]].mean()) if np.any(bands[rs]) else np.nan)
                it = int(prob.solver.getIterationNumber())
                dt = dt * 1.5 if it <= 4 else (dt * 0.6 if it >= 9 else dt)
            else:
                psi.x.array[:] = psi_n.x.array; psi.x.scatter_forward()
                dt *= 0.5
                assert dt > 1e-13, "dt collapse"
        t_prev = t_s
    out = {k: np.array(v) for k, v in log.items()}
    out["t_handover"] = np.array(sch.t_handover if sch.t_handover is not None else np.nan)
    out["h"] = np.array(h)
    for rs in RS:
        out[f"mean_rho_{rs}"] = np.array(mean_rho[rs])
    np.savez(HERE / f"m4_phase4_wi_ring_derivation_emb_n{n}.npz", **out)
    print(f"  Saved -> scratch/m4_phase4_wi_ring_derivation_emb_n{n}.npz (handover I/Imax="
          f"{np.interp(float(out['t_handover']), out['t'], out['I'])/out['I'][-1]:.2f})")


def part3_analyze():
    from pids_forward.physics.constitutive import VanGenuchten
    soil = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
    Phi = lambda p: float(soil.kirchhoff(p, 0.0))
    ref = np.load(HERE / "m4_phase4_wi_ring_derivation_ref.npz")
    t_r, I_r = ref["t"], ref["I"]
    q_ref = np.gradient(I_r, t_r) * 2.0 * np.pi * R_W
    bands = ((0.2, 0.5, "mid "), (0.5, 0.8, "bend"), (0.8, 0.97, "deep"))
    # worst |RATEe| over bands per (n, radius) -> the selection table
    worst = {rs: [] for rs in RS}
    for n in N_SWEEP:
        emb = np.load(HERE / f"m4_phase4_wi_ring_derivation_emb_n{n}.npz")
        wi = emb["era"] == 1
        I_e = emb["I"][wi]
        i_frac = I_e / I_r[-1]
        q_truth = np.interp(I_e, I_r, q_ref)
        print(f"\n===== n={n} (h={float(emb['h'])/R_W:.1f} r_w, WI-era share={1-I_e[0]/I_e[-1]:.0%}) "
              f"RATEe = bridge(emb ring) vs truth =====")
        for rs in RS:
            mrho = float(emb[f"mean_rho_{rs}"])
            if not np.isfinite(mrho):
                print(f"  r={rs} r_w: (no resolved ring at this mesh)")
                worst[rs].append(np.nan)
                continue
            lnf = np.log(mrho / R_W)
            psi_e = emb[f"psi_ring_{rs}"][wi]
            dphi_e = np.array([Phi(p) for p in psi_e])
            q_e = 2 * np.pi / lnf * dphi_e
            devs, line = [], f"  r={rs:4.0f} r_w (got {mrho/R_W:4.1f}, ln={lnf:.2f}): "
            for lo, hi, tag in bands:
                m = (i_frac >= lo) & (i_frac < hi) & (q_truth > 0)
                if np.any(m):
                    d = float(np.median(q_e[m] / q_truth[m] - 1.0))
                    devs.append(abs(d))
                    line += f"| {tag}{d:+6.1%} "
            print(line)
            worst[rs].append(max(devs) if devs else np.nan)
    print("\n=== worst |RATEe| over all bands, per radius (rows) x mesh (cols n=6,8,10,12) ===")
    for rs in RS:
        cells = " ".join(f"{w:5.1%}" if np.isfinite(w) else "  -- " for w in worst[rs])
        wn = [w for w in worst[rs] if np.isfinite(w)]
        print(f"  r={rs:4.0f} r_w:  {cells}   | max over meshes = "
              f"{max(wn):.1%}" if wn else "  --")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if mode == "ref":
        part1_ref()
    elif mode == "emb":
        for n in N_SWEEP:
            part2_emb(n)
    else:
        part3_analyze()
