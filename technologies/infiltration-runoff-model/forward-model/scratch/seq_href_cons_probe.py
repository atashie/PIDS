"""SCRATCH PROBE -- confirm the iterated-capped split's ~1.7e-3 conservation LEAK = the per-step
infiltration RECONSTRUCTION (route(A - I_final), I_final = film - film_rem), NOT the routing
telescoping and NOT the Picard count alone.

Task 1 of docs/plans/2026-06-25-iterated-capped-split-conservation-rearchitecture.md.

Method: subclass IteratedCappedSplit with a VERBATIM-copied step() plus 4 instrumentation captures.
Per accepted step we record:
  * I_recon  = Sum( (film - film_rem) * A )   -- the reconstructed infiltration VOLUME that the final
               re-route subtracts (route(A - I_final)); this is what the SURFACE ledger removes.
  * dtheta   = Delta(Int theta dV)  over the step (degree-8, the ACTUAL soil gain the bulk Richards did).
  * route_resid = Sum((A - I_final)*A) - (Sum(d_routed*A) + outflow)   -- the routing telescoping error.
  * of, npic.

The DECISIVE triple:
  (1) Sum(route_resid)         ~ 0          => routing telescoping is exact (NOT the leak).
  (2) recon_gap = Sum(dtheta) - Sum(I_recon)               => the reconstruction mismatch.
  (3) recon_gap  ~= balance()  (both ~1.7e-3)              => the leak IS the reconstruction.

Run (WSL pids-fem, threads pinned) -- LIVE to a file (NO tail; the run is ~8-12 min):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_href_cons_probe.py > scratch/_cons_probe_out.txt 2>&1'
"""
from __future__ import annotations

import time

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from scratch.seq_href_iterated import IteratedCappedSplit, COMMON, CASES, LOAM
from scratch.seq_href_closure_study import make_box, _march
from scratch.seq_iterative_prototype import _top_area_ds


class ConsProbe(IteratedCappedSplit):
    """IteratedCappedSplit with an instrumented step() (verbatim body + per-step captures)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._theta_form = None
        # per-step history: (dtheta, I_recon, of, route_resid, npic)
        self.hist_dtheta = []
        self.hist_I_recon = []
        self.hist_of = []
        self.hist_route_resid = []
        self.hist_npic = []

    def _theta(self) -> float:
        if self._theta_form is None:
            dxq = ufl.dx(metadata={"quadrature_degree": self._quad_degree})
            self._theta_form = fem.form(self.soil.theta_ufl(self._rp.psi) * dxq)
        return self.mesh.comm.allreduce(fem.assemble_scalar(self._theta_form), op=MPI.SUM)

    def step(self, dt: float):
        self._ensure_built()
        rp = self._rp
        rp.dt.value = dt
        td = self._top_dofs_arr
        A_area = self._A_i
        psi_entry = rp.psi.x.array.copy()
        rain = float(self._rain_c.value)
        film_entry = np.maximum(psi_entry[td], 0.0)
        A = film_entry + self.d_held[td] + rain * dt      # available surface DEPTH per top node
        theta_before = self._theta()                       # <-- INSTRUMENT

        I = np.zeros_like(A)
        film = None
        film_rem = None
        d_routed_top = None
        of = 0.0
        ok = False
        it_last = 0
        npic = 0
        reason = 0
        for m in range(self.picard_inf):
            npic = m + 1
            to_route = np.zeros(self._n_dofs)
            to_route[td] = np.maximum(A - I, 0.0)
            d_routed_full, of = self._route(to_route, dt)
            d_routed_top = d_routed_full[td]
            pond_mid = 0.5 * (np.maximum(A - I, 0.0) + d_routed_top)
            film = np.minimum(pond_mid, self.h_ref)
            lat = np.zeros(self._n_dofs)
            lat[td] = (film - film_entry) / dt
            self._lat_src.x.array[:] = lat
            self._lat_src.x.scatter_forward()
            rp.psi.x.array[:] = psi_entry
            rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry
            rp.psi_n.x.scatter_forward()
            rp._ensure_problem()
            rp._problem.solve()
            snes = rp._problem.solver
            reason = int(snes.getConvergedReason()); it_last = int(snes.getIterationNumber())
            fnorm = float(snes.getFunctionNorm())
            if not (reason > 0 and (reason != 4 or fnorm <= self.stall_accept_fnorm)):
                break
            film_rem = np.maximum(rp.psi.x.array[td], 0.0)
            I_new = np.maximum(film - film_rem, 0.0)
            resid = float(np.max(np.abs(I_new - I))) if I.size else 0.0
            I = (1.0 - self.picard_omega) * I + self.picard_omega * I_new
            if resid < self.picard_tol:
                ok = True
                break
        else:
            ok = film_rem is not None
        self.last_picard_inf = npic
        self.last_reason = reason
        if not ok:
            rp.psi.x.array[:] = psi_entry; rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry; rp.psi_n.x.scatter_forward()
            self._pic_hist.append(npic)
            return False, it_last

        # CONSERVATION re-route (the suspected leak source).
        I_final = np.maximum(film - film_rem, 0.0)
        to_route[td] = np.maximum(A - I_final, 0.0)
        d_routed_full, of = self._route(to_route, dt)
        d_routed_top = d_routed_full[td]
        self.d_held[td] = np.maximum(d_routed_top - film_rem, 0.0)
        rp.psi_n.x.array[:] = rp.psi.x.array
        rp.psi_n.x.scatter_forward()
        of_booked = of * (1.0 + self.outflow_leak_frac)
        self.last_outflow = of
        self.cum_outflow += of_booked
        ghb = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM) for f in self._drain_forms]
        inter = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                 for f in self._interior_forms]
        self.last_drainage = float(sum(ghb) + sum(inter))
        self.cum_drainage += dt * self.last_drainage
        self.cum_rain += rain * self._top_area * dt
        self._t += dt
        self._pic_hist.append(npic)

        # ---- INSTRUMENTATION (post-accept) -------------------------------------------------------
        theta_after = self._theta()
        dtheta = theta_after - theta_before                                    # actual soil gain
        I_recon = float(np.sum(I_final * A_area[td]))                          # routed-subtracted I
        # routing telescoping residual on the FINAL re-route: in - (retained + out).
        in_vol = float(np.sum(np.maximum(A - I_final, 0.0) * A_area[td]))
        retained_vol = float(np.sum(d_routed_top * A_area[td]))
        route_resid = in_vol - (retained_vol + of)
        self.hist_dtheta.append(dtheta)
        self.hist_I_recon.append(I_recon)
        self.hist_of.append(of)
        self.hist_route_resid.append(route_resid)
        self.hist_npic.append(npic)
        return True, it_last


def run_probe(case_name="b1_steep", h_ref=2e-3, dt_max=0.004):
    c = COMMON
    soil = VanGenuchten(**LOAM)
    case = CASES[case_name]
    msh = make_box(*c["MESH"], c["Lx"], c["Ly"], c["Lz"])
    prob = ConsProbe(msh, soil, n_man=c["NMAN"], route_substeps=4, h_ref=h_ref)
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

    dtheta = np.array(prob.hist_dtheta)
    I_recon = np.array(prob.hist_I_recon)
    of = np.array(prob.hist_of)
    rroute = np.array(prob.hist_route_resid)
    per_step_gap = dtheta - I_recon                       # (actual soil gain) - (routed-subtracted I)
    recon_gap = float(per_step_gap.sum())
    bal = prob.balance()
    routed = prob.cum_outflow / R_in
    print("#" * 92)
    print(f"CONS PROBE -- {case_name}  h_ref={h_ref*1000:.1f}mm dt_max={dt_max}  "
          f"[{ns} steps {time.perf_counter()-t0:.0f}s ok={ok}]")
    print("#" * 92)
    print(f"  routed/R            = {routed:.4f}  (monolith {case['target']:.4f}, "
          f"gap {(routed-case['target'])*100:+.1f}pp)")
    print(f"  cum_rain            = {prob.cum_rain:.6e}")
    print(f"  balance()           = {bal:.6e}   bal/rain = {abs(bal)/prob.cum_rain:.3e}")
    print("  --- the decisive triple ---")
    print(f"  (1) Sum(route_resid)= {rroute.sum():.3e}   (max|.|={np.abs(rroute).max():.2e})  "
          f"=> ~0 means routing telescoping is EXACT (not the leak)")
    print(f"  (2) Sum(dtheta)     = {dtheta.sum():.6e}   (actual soil gain, deg-8)")
    print(f"      Sum(I_recon)    = {I_recon.sum():.6e}   (routed-subtracted infiltration)")
    print(f"      recon_gap (2)   = {recon_gap:.6e}   = Sum(dtheta) - Sum(I_recon)")
    print(f"  (3) recon_gap vs balance(): {recon_gap:.3e} vs {bal:.3e}  "
          f"ratio={recon_gap/bal if bal != 0 else float('nan'):.3f}  "
          f"(want ~1 => the leak IS the reconstruction)")
    print("  --- per-step residual magnitude (relative to rain) ---")
    print(f"  max |per-step gap|/rain = {np.abs(per_step_gap).max()/prob.cum_rain:.2e}   "
          f"mean = {np.abs(per_step_gap).mean()/prob.cum_rain:.2e}")
    print(f"  picard npic: avg={np.mean(prob.hist_npic):.2f} max={int(np.max(prob.hist_npic))} "
          f"(cap={prob.picard_inf}); steps hitting cap = "
          f"{int(np.sum(np.array(prob.hist_npic) >= prob.picard_inf))}/{len(prob.hist_npic)}")
    print("#" * 92, flush=True)
    return dict(routed=routed, bal=bal, recon_gap=recon_gap, route_resid_sum=float(rroute.sum()))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    run_probe("b1_steep", h_ref=2e-3, dt_max=0.004)
