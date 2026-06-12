"""Follow-up #3: the INSTRUMENTED disperse WI-era residual diagnostic (LOAM R40).

The ~5% disperse deployment residual (mid-curve +3-5% over-delivery, deep-bend -9%) is an
UNATTRIBUTED WI-ERA systematic -- the F-large-zeta attribution was REFUTED by measurement
(2026-06-11 review attack f; F's real bias is -0.9..-1.5%, wrong sign and era -- the F-polish
roadmap item is DROPPED, do not resurrect). This diagnostic localizes the systematic among the
three candidate carriers by comparing, AT MATCHED CUMULATIVE I (not matched t):

  (a) BRIDGE FORM: the resolved reference's instantaneous wall rate  q_ref(I)  vs the constant-WI
      Kirchhoff bridge evaluated on the RESOLVED field at the measured equivalent radius,
      q_bridge(I) = WI_n * [Phi(0) - Phi(psi_ref(r_0(h_n)))](I).  Deviation here = the bridge/WI
      CONSTANT is the carrier (e.g. the effective resistance drifts as the reservoir depletes).
  (b) RIDGE READ: the embedded lattice's on-ridge value vs the resolved field at r_0:
      Phi(psi_Gamma_emb)(I) vs Phi(psi_ref(r_0))(I).  Deviation here = the READ is the carrier.
  (c) POST-HANDOVER TRANSIENT: deviation localized right after t_handover vs persistent.

Part 1 (this script, mode "ref"): dense resolved 1-D radial regen of the LOAM R40 depleting leg
(the committed fixture's window, 256 log-spaced samples) saving I(t) and psi(r_0(h_n)) for
n=8,12 -> scratch/m4_phase4_wi_diag_ref.npz. q_ref = dI/dt by central differences on the dense
log grid (the rates of interest vary on decade scales; the grid resolves them to << 1%).

Part 2 (mode "emb"): the embedded harness loop (verbatim copy of run_embedded's marching,
instrumented -- the harness itself stays untouched: it is the gate's instrument) logging per
accepted step: t, I, the assembled WI-era exchange rate, the clock-era prescribed rate, the
ridge-mean psi, handover time -> scratch/m4_phase4_wi_diag_emb_n{8,12}.npz.

Part 3 (mode "analyze"): the matched-I comparison + era attribution printout.

Run (WSL, from forward-model/): PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scratch/m4_phase4_wi_residual_diag.py {ref|emb|analyze}
"""
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

HERE = pathlib.Path(__file__).parent
R_W = 0.05
R_OUT = 40 * R_W
L_BOX = float(np.sqrt(np.pi * (R_OUT ** 2 - R_W ** 2)))      # the capacity-matched box edge
N_SWEEP = (8, 12)
N_DENSE = 256


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
    prob = NonlinearProblem(F, psi, bcs=bcs, petsc_options_prefix="m4diag_", petsc_options=dz._LU)
    th_i = float(soil.theta(-1.0))
    stored = fem.form((theta - th_i) * r * dxs)
    xc = V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(xc)
    r0s = {nn: 0.1986 * (L_BOX / nn) for nn in N_SWEEP}
    I_out, psi_r0 = [], {nn: [] for nn in N_SWEEP}
    dt, t_prev = 1e-8, 0.0
    for k, t_s in enumerate(t):
        dt = dz._solve_to(prob, psi, psi_n, dt_c, t_prev, t_s, dt)
        t_prev = t_s
        I_out.append(float(fem.assemble_scalar(stored)) / R_W)
        for nn in N_SWEEP:
            psi_r0[nn].append(float(np.interp(r0s[nn], xc[order], psi.x.array[order])))
        if k % 16 == 0:
            print(f"  [diag ref] {k+1}/{t.size} t={t_s:.3e} I={I_out[-1]:.4e}", flush=True)
    out = {"t": t, "I": np.array(I_out)}
    for nn in N_SWEEP:
        out[f"psi_r0_n{nn}"] = np.array(psi_r0[nn])
        out[f"r0_n{nn}"] = np.array(r0s[nn])
    np.savez(HERE / "m4_phase4_wi_diag_ref.npz", **out)
    print("Saved -> scratch/m4_phase4_wi_diag_ref.npz")


def part2_emb(n):
    """run_embedded's marching loop, instrumented (the harness module is the gate's instrument
    and stays untouched; this copy logs per-step internals)."""
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
    prob = NonlinearProblem(F, psi, bcs=[], petsc_options_prefix="m4dge_",
                            petsc_options=dict(_LU_BASE, snes_type="newtonls",
                                               snes_linesearch_type="cp"))
    exch_f = fem.form(feat._perimeter * feat.Omega * soil.kirchhoff_ufl(psi, feat.Hf) * feat.dGamma)
    sch = WellIndexExchange().setup(feat, soil, dict(h=h, t0=float(t_grid[0])))
    g = feat._gamma_dofs
    log = {k: [] for k in ("t", "I", "q_exch", "q_clock", "psi_g", "era")}
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
                q = float(fem.assemble_scalar(exch_f))
                sch.post_step(feat, psi, t, hstep)
                psi_n.x.array[:] = psi.x.array; psi_n.x.scatter_forward()
                t += hstep
                log["t"].append(t); log["I"].append(sch.I_total(feat))
                log["q_exch"].append(q); log["q_clock"].append(rel)
                log["psi_g"].append(float(psi.x.array[g].mean()))
                log["era"].append(0 if sch.in_subgrid_era else 1)
                it = int(prob.solver.getIterationNumber())
                dt = dt * 1.5 if it <= 4 else (dt * 0.6 if it >= 9 else dt)
            else:
                psi.x.array[:] = psi_n.x.array; psi.x.scatter_forward()
                dt *= 0.5
                assert dt > 1e-13, "dt collapse"
        t_prev = t_s
        print(f"  [diag emb n={n}] t={t_s:.3e} I={log['I'][-1]:.4e} era={log['era'][-1]}", flush=True)
    out = {k: np.array(v) for k, v in log.items()}
    out["t_handover"] = np.array(sch.t_handover if sch.t_handover is not None else np.nan)
    out["WI"] = np.array(sch.WI)
    out["h"] = np.array(h)
    out["length"] = np.array(feat.length)
    np.savez(HERE / f"m4_phase4_wi_diag_emb_n{n}.npz", **out)
    print(f"Saved -> scratch/m4_phase4_wi_diag_emb_n{n}.npz")


def part3_analyze():
    from pids_forward.physics.constitutive import VanGenuchten
    soil = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
    ref = np.load(HERE / "m4_phase4_wi_diag_ref.npz")
    t_r, I_r = ref["t"], ref["I"]
    # resolved instantaneous wall rate per unit feature length [m^2/day]
    q_ref = np.gradient(I_r, t_r) * 2.0 * np.pi * R_W
    for n in N_SWEEP:
        emb = np.load(HERE / f"m4_phase4_wi_diag_emb_n{n}.npz")
        WI, t_h = float(emb["WI"]), float(emb["t_handover"])
        wi_era = emb["era"] == 1
        I_e, q_e = emb["I"][wi_era], emb["q_exch"][wi_era] / float(emb["length"])
        psi_g = emb["psi_g"][wi_era]
        # (a) bridge form on the RESOLVED field at matched I
        psi_r0 = np.interp(I_e, I_r, ref[f"psi_r0_n{n}"])
        q_bridge = WI * np.array([float(soil.kirchhoff(p, 0.0)) for p in psi_r0])
        q_truth = np.interp(I_e, I_r, q_ref)
        # (b) the embedded ridge read vs the resolved field at r_0 (Kirchhoff scale)
        dphi_emb = np.array([float(soil.kirchhoff(p, 0.0)) for p in psi_g])
        dphi_r0 = np.array([float(soil.kirchhoff(p, 0.0)) for p in psi_r0])
        i_frac = I_e / I_r[-1]
        print(f"\n===== n={n}  (WI={WI:.3f}, handover t={t_h:.3e} d, "
              f"WI-era I share={1 - I_e[0]/I_e[-1]:.0%}) =====")
        for lo, hi, tag in ((0.2, 0.5, "mid-curve"), (0.5, 0.8, "bend"), (0.8, 0.97, "deep-bend")):
            m = (i_frac >= lo) & (i_frac < hi) & (q_truth > 0)
            if not np.any(m):
                continue
            dev_tot = np.median(q_e[m] / q_truth[m] - 1.0)
            dev_a = np.median(q_bridge[m] / q_truth[m] - 1.0)
            dev_b = np.median(dphi_emb[m] / dphi_r0[m] - 1.0)
            print(f"  I/Imax {lo:.1f}-{hi:.1f} ({tag:9s}): TOTAL rate dev {dev_tot:+7.1%}   "
                  f"(a) bridge-form {dev_a:+7.1%}   (b) ridge-read(dPhi) {dev_b:+7.1%}")
        # (c) transient: the first decade of WI-era time after handover
        tt = emb["t"][wi_era]
        early = tt <= t_h * 2.0
        if np.any(early) and np.any(~early):
            d_early = np.median(q_e[early] / np.interp(I_e[early], I_r, q_ref) - 1.0)
            print(f"  (c) post-handover transient (t <= 2*t_h): rate dev {d_early:+.1%}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if mode == "ref":
        part1_ref()
    elif mode == "emb":
        for n in N_SWEEP:
            part2_emb(n)
    else:
        part3_analyze()
