"""SCRATCH SPIKE -- ITERATED-CAPPED split (the route-first fix).

Route-first realization B (HrefCappedPondInPsi) over-routes on steep terrain (b1_steep +26pp, WORSE at
finer dt -> routed/R 1) because infiltration only ever sees the POST-routing endpoint film (Codex's
sharpened diagnosis: rain that routes away never infiltrates). The fix = SIMULTANEITY: an outer Picard
on the CAPPED route<->infiltrate to a per-step fixed point, where the soil sees the MIDPOINT pond
(avg of pre- and post-routing, capped at h_ref) so neither routing nor infiltration has priority.

Per step (per top node), available pond A = pond_entry + rain*dt; iterate the infiltration depth I:
  1. route (A - I) -> d_routed + outflow                 (route only the UN-infiltrated water)
  2. film = min( 0.5*((A - I) + d_routed), h_ref )        (MIDPOINT pond, capped)
     offer film to psi (conservative source), solve Richards -> I_new = film - max(psi_top,0)
  3. under-relax I, repeat to |I_new - I| < tol.
Accept: psi carries the un-drawn film_remaining; d_held = d_routed - film_remaining; book I + outflow.
Conservation: A = I (soil) + d_routed (retained) + outflow  =>  d(total) = rain*dt - outflow (verified
by the inherited balance() + the 10% falsification gate).

GATE: does it match the monolith on BOTH mild (b1_base 0.547) AND steep (b1_steep 0.551), conserve
(~1e-11, falsification |ratio|~1), at adequate dt (dt_max=0.004)? Route-first matched only mild.

Run (WSL pids-fem) -- guarded, live (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_href_iterated.py'
"""
from __future__ import annotations

import time

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from scratch.seq_href_cap_spike import HrefCappedPondInPsi
from scratch.seq_iterative_prototype import _top_area_ds
from scratch.seq_href_closure_study import make_box, _march


class IteratedCappedSplit(HrefCappedPondInPsi):
    """Iterated-capped split: outer Picard on the capped route<->infiltrate to a per-step SIMULTANEOUS
    fixed point (the soil sees the midpoint pond, capped at h_ref). Reuses B's conservative source +
    held-store machinery; overrides step()."""

    def __init__(self, mesh, soil, *, picard_inf=8, picard_tol=1e-4, picard_omega=0.7, **kwargs):
        super().__init__(mesh, soil, **kwargs)
        self.picard_inf = int(picard_inf)        # max infiltration<->routing iterations per step
        self.picard_tol = float(picard_tol)      # depth convergence tol [m]
        self.picard_omega = float(picard_omega)  # under-relaxation on I
        self.last_picard_inf = 0
        self._pic_hist: list[int] = []

    def picard_iter_stats(self):
        h = np.asarray(self._pic_hist, dtype=float)
        if h.size == 0:
            return dict(n=0, avg=float("nan"), mx=float("nan"))
        return dict(n=int(h.size), avg=float(h.mean()), mx=int(h.max()))

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

        I = np.zeros_like(A)            # infiltration depth this step (Picard variable)
        film_rem = None
        d_routed_top = None
        of = 0.0
        ok = False
        it_last = 0
        npic = 0
        for m in range(self.picard_inf):
            npic = m + 1
            to_route = np.zeros(self._n_dofs)
            to_route[td] = np.maximum(A - I, 0.0)
            d_routed_full, of = self._route(to_route, dt)
            d_routed_top = d_routed_full[td]
            pond_mid = 0.5 * (np.maximum(A - I, 0.0) + d_routed_top)   # MIDPOINT pond
            film = np.minimum(pond_mid, self.h_ref)
            # offer the film to psi via the CONSERVATIVE source (no writeback); solve from psi_entry.
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
            I_new = np.maximum(film - film_rem, 0.0)           # per-node infiltration from the film
            resid = float(np.max(np.abs(I_new - I))) if I.size else 0.0
            I = (1.0 - self.picard_omega) * I + self.picard_omega * I_new
            if resid < self.picard_tol:
                ok = True
                break
        else:
            ok = film_rem is not None
        self.last_picard_inf = npic
        self.last_reason = reason if 'reason' in dir() else 0
        if not ok:
            rp.psi.x.array[:] = psi_entry; rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry; rp.psi_n.x.scatter_forward()
            self._pic_hist.append(npic)
            return False, it_last

        # CONSERVATION re-route (fixes the ~0.5% imbalance): the in-loop routing used the UNDER-RELAXED
        # I, but the soil actually drew I_final = film - film_rem. Re-route (A - I_final) so the retained
        # pond + outflow are consistent with the soil gain. Hand-verified: A = I_final(soil) +
        # d_routed(retained) + outflow  =>  d(total) = rain*dt - outflow. (No extra Richards solve --
        # just one routing sweep.)
        I_final = np.maximum(film - film_rem, 0.0)
        to_route[td] = np.maximum(A - I_final, 0.0)
        d_routed_full, of = self._route(to_route, dt)
        d_routed_top = d_routed_full[td]
        # accept: psi carries the un-drawn film_rem (from the last solve); the rest of the routed pond is
        # held. retained total pond per node = d_routed_top = film_rem + d_held_new.
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
        return True, it_last


# ---- fixtures ---------------------------------------------------------------------------------------
LOAM = dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)
COMMON = dict(Lx=8.0, Ly=5.0, Lz=1.0, PSI_I=-0.4, RAIN=0.5, STORM=0.08, TEND=0.45, NMAN=0.05,
              MESH=(30, 20, 8))
CASES = {
    "b1_base":  dict(S0=0.03, target=0.5470),
    "b1_steep": dict(S0=0.10, target=0.5508),
}


def run(cls, case, h_ref, dt_max=0.004, leak=0.0, **kw):
    c = COMMON
    soil = VanGenuchten(**LOAM)
    msh = make_box(*c["MESH"], c["Lx"], c["Ly"], c["Lz"])
    prob = cls(msh, soil, n_man=c["NMAN"], route_substeps=4, h_ref=h_ref, **kw)
    prob.outflow_leak_frac = leak
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
    st = prob.picard_iter_stats() if hasattr(prob, "picard_iter_stats") else dict(avg=1.0, mx=1)
    return dict(routed=prob.cum_outflow / R_in, bal=abs(prob.balance()) / prob.cum_rain,
                ns=ns, ok=ok, wall=time.perf_counter() - t0, pic_avg=st["avg"], pic_max=st["mx"])


def main():
    np.set_printoptions(precision=4, suppress=True)
    HREF = 0.002
    print("#" * 90)
    print(f"ITERATED-CAPPED split @ h_ref={HREF*1000:.0f}mm, dt_max=0.004 -- does it match the monolith")
    print(f"on BOTH mild AND steep? (route-first: mild 0.546 OK, steep 0.81 +26pp FAIL)")
    print("#" * 90, flush=True)
    # decisive case FIRST: steep.
    for name in ("b1_steep", "b1_base"):
        case = CASES[name]
        r = run(IteratedCappedSplit, case, HREF)
        gap = r["routed"] - case["target"]
        print(f"\n  {name}: routed/R={r['routed']:.4f} vs monolith {case['target']:.4f}  "
              f"(gap {gap:+.3f} = {gap*100:+.1f}pp)  bal={r['bal']:.1e} ok={r['ok']} "
              f"[picard avg={r['pic_avg']:.1f}/max={r['pic_max']} | {r['ns']} steps {r['wall']:.0f}s]",
              flush=True)
    # falsification on b1_base (the ledger detector with the iterated split).
    print("\n  --- falsification (10% outflow mis-book) on b1_base ---", flush=True)
    rf = run(IteratedCappedSplit, CASES["b1_base"], HREF, leak=0.10)
    # |bal|/rain should be ~the leak fraction*outflow share, NOT ~1e-11.
    print(f"  leak=10%: |bal|/rain={rf['bal']:.2e}  (clean was ~1e-11; a real break ~ 0.05-0.10)",
          flush=True)
    print("\n" + "#" * 90)
    print("READ: BOTH cases near their monolith targets + clean ledger => the iterated split fixes the")
    print("route-first slope error. If steep still over-routes, the midpoint/iteration is insufficient.")
    print("#" * 90, flush=True)


if __name__ == "__main__":
    main()
