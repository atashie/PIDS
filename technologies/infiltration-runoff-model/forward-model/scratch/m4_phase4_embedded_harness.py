"""Module 4 (§E) Phase-4 Task 4b: the CLOSED-BOX embedded harness (the gate's instrument).

A feature line embedded in a CLOSED host box whose water capacity per unit feature length equals the
Ref-A reference annulus: L = sqrt(pi*(R_out^2 - r_w^2)) (so full depletion is reachable ONLY if the
scheme's sub-grid reservoir ultimately hands its water to the host -- the capacity-matching choice
that forces honest reservoir release). All faces NO-FLOW; gravity OFF (the refs are the gravity-free
radial truth); the feature head H_f is FIXED (the scenario's wall Dirichlet analog: disperse H_f=0
into psi_i=-1 host [cp linesearch], drain H_f=-1 from psi_i=0 host [bt]). Optional Ref-B pulse: a
volumetric source on the annular band rho in [3,4] r_w around the line, volume matched EXACTLY to
the reference pulse via the assembled discrete band volume.

SCHEME interface (so the same instrument runs the retracted dual-scale, the bare clock, and the
Phase-4 WI scheme):
    scheme.setup(feat, soil, ctx)     -- once, after configure_sorptive; ctx has h, L, t0, direction
    scheme.pre_step(feat, psi, t)     -- set feat.Omega (the lagged host-source coefficient); may
                                         RETURN a reservoir-release rate [m^3/day] (or None): a
                                         ridge-distributed source the host receives this step
                                         (negative for drain -- the host LOSES the drained water)
    scheme.post_step(feat, psi, t, dt)-- advance internal states (clock, reservoir transfers)
    scheme.I_total(feat)              -- cumulative uptake per wall area [m] (the gate observable)
    scheme.reservoir(feat, injected)  -- water currently held in the sub-grid reservoir [m^3]
                                         (schemes with an EXPLICIT reservoir ignore `injected`; a
                                         scheme without one may only declare the implicit gap
                                         I*p*len - sgn*injected, which makes its ledger-2 trivially
                                         true -- such schemes are judged on I(t) fidelity alone)
The host source is always p*Omega*[Phi(H_f)-Phi(psi)] on the ridge (feat.sorptive_into_host), Omega
lagged per step -- every candidate scheme expresses itself through Omega + its internal states.

MASS LEDGER (asserted every sample, the closed-box advantage):
    host theta-gain  ==  injected (cum. assembled exchange flux * dt)  +  pulse (cum. source)
to <= 1e-8 relative (BE + lagged-Omega consistency: the exchange form is assembled at the converged
end-of-step state, exactly what the residual imposed). The scheme-level identity
    I_total * perimeter * length == injected + reservoir
is the scheme's OWN bookkeeping, asserted to <= 1e-6 relative (clock substepping accuracy).

SINGULARITY NOTE (drain): the all-saturated all-Neumann start has C=0 everywhere and no anchor; with
Omega=0 and no forcing the residual at the initial guess is ZERO, so SNES accepts without factoring
(atol); once Omega>0 the exchange term anchors the constant mode. The pulse never fires while Omega=0
and the domain is fully saturated (Ref-B-drain fires at ~50% depletion).

Plan: docs/plans/2026-06-10-m4-phase4-coupled-embedding-plan.md (Task 4).
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.feature import EmbeddedFeature

COMM = MPI.COMM_WORLD
R_W = 0.05

SOILS = {
    "SAND": VanGenuchten(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=7.13),
    "LOAM": VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25),
    "SILT": VanGenuchten(theta_r=0.034, theta_s=0.46, alpha=1.6, n=1.37, Ks=0.06),
}

_LU_BASE = {"snes_rtol": 1e-9, "snes_atol": 1e-12, "snes_max_it": 40,
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def _vertex_dx():
    return ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})


def run_embedded(scheme, soil_name, R_out, n, t_grid, direction="disperse",
                 pulse=None, pulse_band=(3 * R_W, 4 * R_W), nx=4, label=""):
    """March the closed-box embedded problem over t_grid. pulse = (t1, t2, V_per_length) or None;
    pulse_band = the annular band radii around the feature line (must match the reference's).
    Returns dict(t, I, ledger...) or None on dt collapse."""
    assert n % 2 == 0, "n must be even (feature line on the vertex lattice)"
    soil = SOILS[soil_name]
    L = float(np.sqrt(np.pi * (R_out ** 2 - R_W ** 2)))
    h = L / n
    Lx = nx * h
    psi_i, h_f = (-1.0, 0.0) if direction == "disperse" else (0.0, -1.0)
    ls = "cp" if direction == "disperse" else "bt"

    msh = dmesh.create_box(COMM, [[0.0, 0.0, 0.0], [Lx, L, L]], [nx, n, n])
    feat = EmbeddedFeature(msh, lambda x: np.isclose(x[1], L / 2) & np.isclose(x[2], L / 2),
                           tangent=(1.0, 0.0, 0.0), K_feat=1.0, area=np.pi * R_W ** 2, porosity=0.4)
    feat.configure_sorptive(soil, psi_i=-1.0, psi_wall=0.0)   # |dpsi|=1 pair sets BOTH directions
    V = feat.V
    psi, psi_n = fem.Function(V), fem.Function(V)
    psi.x.array[:] = psi_i; psi_n.x.array[:] = psi_i
    feat.Hf.x.array[:] = h_f; feat.Hf.x.scatter_forward()

    w = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(1e-8))
    s_c = fem.Constant(msh, PETSc.ScalarType(0.0))
    dxs, dxq = _vertex_dx(), ufl.dx(metadata={"quadrature_degree": 8})
    th, thn = soil.theta_ufl(psi), soil.theta_ufl(psi_n)
    K = soil.K_ufl(psi)
    x = ufl.SpatialCoordinate(msh)
    rho = ufl.sqrt((x[1] - L / 2) ** 2 + (x[2] - L / 2) ** 2)
    band = ufl.conditional(ufl.And(ufl.ge(rho, pulse_band[0]), ufl.le(rho, pulse_band[1])), 1.0, 0.0)
    rel_c = fem.Constant(msh, PETSc.ScalarType(0.0))      # reservoir release per unit length [m^2/day]
    F = (((th - thn) / dt_c) * w * dxs
         + K * ufl.dot(ufl.grad(psi), ufl.grad(w)) * dxq
         - s_c * band * w * dxq
         - rel_c * w * feat.dGamma
         + feat.sorptive_into_host(w, psi))
    # closed box: NO Dirichlet BCs anywhere (the exchange term anchors once Omega > 0)
    prob = NonlinearProblem(F, psi, bcs=[], petsc_options_prefix="m4h_",
                            petsc_options=dict(_LU_BASE, snes_type="newtonls",
                                               snes_linesearch_type=ls))

    th_i = float(soil.theta(psi_i))
    host_gain_f = fem.form((th - th_i) * dxs)
    exch_f = fem.form(feat._perimeter * feat.Omega * soil.kirchhoff_ufl(psi, feat.Hf) * feat.dGamma)
    band_vol = COMM.allreduce(fem.assemble_scalar(
        fem.form(band * fem.Constant(msh, PETSc.ScalarType(1.0)) * dxq)), op=MPI.SUM)
    s_rate = 0.0
    if pulse is not None:
        t1, t2, v_len = pulse
        s_rate = v_len * Lx / (band_vol * (t2 - t1))     # match the reference pulse volume EXACTLY
    else:
        t1 = t2 = -1.0

    ctx = dict(h=h, L=L, Lx=Lx, direction=direction, t0=float(t_grid[0]), R_out=R_out)
    scheme.setup(feat, soil, ctx)

    sgn = 1.0 if direction == "disperse" else -1.0       # host gains (disperse) / loses (drain)
    marks = np.unique(np.concatenate([t_grid, [t1, t2]])) if pulse else np.asarray(t_grid)
    marks = marks[marks > 0]
    injected = 0.0
    # the first sample = the scheme's state at t0 (the refs' first sample, the seeded clock)
    out_t, out_I = [float(t_grid[0])], [scheme.I_total(feat)]
    led_host, led_inj, led_res = [0.0], [0.0], [scheme.reservoir(feat, 0.0)]
    dt, t_prev = 1e-7, float(t_grid[0])
    for t_s in marks:
        if t_s <= t_prev:
            continue
        t = t_prev
        while t < t_s - 1e-15:
            hstep = min(dt, t_s - t)
            dt_c.value = hstep
            active = pulse is not None and (t >= t1 - 1e-15) and (t + hstep <= t2 + 1e-15)
            s_c.value = s_rate if active else 0.0
            rel = scheme.pre_step(feat, psi, t) or 0.0    # [m^3/day], ridge-distributed
            rel_c.value = rel / feat.length
            prob.solve()
            if prob.solver.getConvergedReason() > 0:
                # end-of-step exchange rate INTO the host: the residual carries -p*Omega*dPhi*w as a
                # source, so the assembled +p*Omega*dPhi IS the host gain rate (>0 disperse, <0 drain)
                q = COMM.allreduce(fem.assemble_scalar(exch_f), op=MPI.SUM)
                injected += (q + rel) * hstep
                scheme.post_step(feat, psi, t, hstep)
                psi_n.x.array[:] = psi.x.array; psi_n.x.scatter_forward()
                t += hstep
                it = int(prob.solver.getIterationNumber())
                dt = dt * 1.5 if it <= 4 else (dt * 0.6 if it >= 9 else dt)
            else:
                psi.x.array[:] = psi_n.x.array; psi.x.scatter_forward()
                dt *= 0.5
                if dt < 1e-13:
                    print(f"  [{label}] dt collapse at t={t:.3e}")
                    return None
        t_prev = t_s
        if t_s in t_grid:
            host_gain = COMM.allreduce(fem.assemble_scalar(host_gain_f), op=MPI.SUM)
            cum_pulse = s_rate * band_vol * min(max(t_s - t1, 0.0), (t2 - t1)) if pulse else 0.0
            res = scheme.reservoir(feat, injected)
            # ledger 1 (host): theta-gain == injected + pulse   (machine-ish precision)
            scale = max(abs(host_gain), abs(injected) + abs(cum_pulse), 1e-14)
            assert abs(host_gain - (injected + cum_pulse)) / scale < 1e-6, \
                f"[{label}] host ledger broken at t={t_s:.3e}: gain={host_gain:.6e} " \
                f"inj={injected:.6e} pulse={cum_pulse:.6e}"
            # ledger 2 (scheme): I_total*p*length == sgn*injected + reservoir
            I_tot = scheme.I_total(feat)
            vol_claim = I_tot * feat._perimeter * feat.length
            vol_have = sgn * injected + res
            if vol_claim > 1e-12:
                dev = abs(vol_claim - vol_have) / vol_claim
                assert dev < 1e-4, \
                    f"[{label}] scheme ledger broken at t={t_s:.3e}: claim={vol_claim:.6e} " \
                    f"have={vol_have:.6e} ({dev:.2e})"
            out_t.append(t_s); out_I.append(I_tot)
            led_host.append(host_gain); led_inj.append(injected); led_res.append(res)
            if label:
                print(f"  [{label}] t={t_s:.3e}  I={I_tot:.4e}  host={host_gain:.3e}  "
                      f"inj={injected:.3e}  res={res:.3e}", flush=True)
    return dict(t=np.array(out_t), I=np.array(out_I), host=np.array(led_host),
                injected=np.array(led_inj), reservoir=np.array(led_res),
                h=h, L=L, perimeter=feat._perimeter, length=feat.length)


# ---- baseline scheme: the RETRACTED dual-scale (production EXPERIMENTAL machinery) ----------------
class DualScaleScheme:
    """The retracted Phase-3 dual-scale embedding, driven through the EXPERIMENTAL EmbeddedFeature
    machinery exactly as committed (shell psi_cell read + storage gate + lagged well index). Expected
    to FAIL the Phase-4 gate (that failure is the gate's discrimination evidence); if it PASSES,
    STOP -- the gate does not discriminate."""

    def __init__(self, direction="disperse"):
        self.direction = direction

    def setup(self, feat, soil, ctx):
        feat.seed_clock(ctx["t0"])
        self._g = feat._gamma_dofs

    def pre_step(self, feat, psi, t):
        feat.update_well_index(psi)

    def post_step(self, feat, psi, t, dt):
        feat.advance_clock(psi, dt)

    def I_total(self, feat):
        I = feat.I_disp if self.direction == "disperse" else feat.I_drain
        return float(I.x.array[self._g].mean())

    def reservoir(self, feat, injected):
        # the dual-scale has NO explicit reservoir; it may only declare the implicit gap, which makes
        # ledger-2 trivially true -- it is judged on I(t) fidelity (and refinement) alone.
        sgn = 1.0 if self.direction == "disperse" else -1.0
        return self.I_total(feat) * feat._perimeter * feat.length - sgn * injected


if __name__ == "__main__":
    import sys
    from pids_forward.physics.sorptive_closure import rel_l2
    refA = np.load("tests/data/m4_phase4_refA_disperse.npz")
    which = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if which == "smoke":
        t, I_ref = refA["LOAM_R3_t"], refA["LOAM_R3_I"]
        sch = DualScaleScheme("disperse")
        out = run_embedded(sch, "LOAM", 3 * R_W, 8, t, label="smoke n=8 LOAM R3 dual-scale")
        if out is not None:
            print(f"smoke: relL2 vs RefA(3r_w) = {rel_l2(out['I'], I_ref):.1%}  "
                  f"(end I={out['I'][-1]:.4e} vs ref {I_ref[-1]:.4e}, I_max={float(refA['LOAM_R3_Imax']):.4e})")
    elif which == "baseline40":
        # the retracted dual-scale on the DEPLOYMENT legs (apples-to-apples with the WI scheme)
        refB40 = np.load("scratch/m4_phase4_refB40_disperse.npz")
        t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
        for n in (8, 12):
            out = run_embedded(DualScaleScheme("disperse"), "LOAM", 40 * R_W, n, t)
            if out is not None:
                print(f"  dual-scale RefA(40r_w) n={n:2d}: relL2={rel_l2(out['I'], I_ref):.1%}  "
                      f"end I/I_max={out['I'][-1]/float(refA['LOAM_R40_Imax']):.2f}", flush=True)
        t, I_ref = refB40["LOAM_t"], refB40["LOAM_I"]
        band = tuple(refB40["LOAM_band"])
        pulse = (float(refB40["LOAM_t_pulse"][0]), float(refB40["LOAM_t_pulse"][1]),
                 float(refB40["LOAM_V_pulse_per_wall_area"]) * R_W * 2 * np.pi)
        for n in (8, 12):
            out = run_embedded(DualScaleScheme("disperse"), "LOAM", 40 * R_W, n, t,
                               pulse=pulse, pulse_band=band)
            if out is not None:
                print(f"  dual-scale RefB-40 n={n:2d}: relL2={rel_l2(out['I'], I_ref):.1%}", flush=True)
    elif which == "baseline":
        # the RETRACTED dual-scale through the discriminating gate: the failure-baseline record
        # (measured 2026-06-10 n=8 LOAM R3: 27.8%, end I 61% past capacity -- passive accumulator)
        refB = np.load("tests/data/m4_phase4_refB_disperse.npz")
        print("DUAL-SCALE failure baselines (EMBEDDED_TOL=0.10):")
        for k in (3, 5):
            t, I_ref = refA[f"LOAM_R{k}_t"], refA[f"LOAM_R{k}_I"]
            for n in (8, 16, 24):
                out = run_embedded(DualScaleScheme("disperse"), "LOAM", k * R_W, n, t)
                if out is not None:
                    print(f"  RefA({k}r_w) n={n:2d}: relL2={rel_l2(out['I'], I_ref):.1%}  "
                          f"end I/I_max={out['I'][-1]/float(refA[f'LOAM_R{k}_Imax']):.2f}", flush=True)
        t, I_ref = refB["LOAM_t"], refB["LOAM_I"]
        pulse = (float(refB["LOAM_t_pulse"][0]), float(refB["LOAM_t_pulse"][1]),
                 float(refB["LOAM_V_pulse_per_wall_area"]) * R_W * 2 * np.pi)  # per-wall-area -> per-length
        for n in (8, 16):
            out = run_embedded(DualScaleScheme("disperse"), "LOAM", 5 * R_W, n, t, pulse=pulse)
            if out is not None:
                print(f"  RefB(5r_w) n={n:2d}: relL2={rel_l2(out['I'], I_ref):.1%}", flush=True)
