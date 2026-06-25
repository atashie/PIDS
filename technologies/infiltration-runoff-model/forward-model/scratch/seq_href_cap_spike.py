"""SCRATCH SPIKE (no pids_forward/ edits, no commit) -- h_ref-capped infiltration for the sequential
operator-split overland coupling.

THE HYPOTHESIS (from the 2026-06-24 partition-bug investigation + the off-the-shelf survey):
the sequential scheme over-infiltrates because the infiltration capacity inflates with the accumulated
(lagged, deep) pond ``d``. Both refuted prototypes evaluated q_pot = kirchhoff(psi_top, d)/ell_c at the
ACCUMULATED d (so the cap was a no-op) AND delivered a hard Neumann influx (which force-feeds the no-Ss
clay). The off-the-shelf fix (CATHY/HYDRUS/ParFlow): cap infiltration at the SATURATED-SURFACE
acceptance -- i.e. evaluate the capacity at a THIN reference film h_ref (NOT the accumulated d) -- and
hold the un-accepted excess as routable surface water.

STAGE 0 (this file, first cut): isolate the CAP. ``HrefCappedNeumann`` = the refuted attempt-#1
structure (separate pond array + hard Neumann influx) with ONE change: q_pot is evaluated at
``min(d, h_ref)`` instead of d. Sweep h_ref on the b1 LOAM partition (Case 1) and ask: does the cap
move routed/R from ~0.17 toward the monolith's ~0.55? (Case 2 stiff clay will likely still force-feed
via the Neumann influx -- that is the EXPECTED contrast that motivates the self-limiting delivery,
realization B, added next.)

Reuses the committed harness in ``seq_iterative_prototype`` (B1/CV fixtures, _march_storm,
run_case1_iter / run_case2 / run_falsification) and the cached upwind-monolith partition target
(seq_iter_mono_cache.npz).

Run (WSL pids-fem, threads pinned):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_href_cap_spike.py 2>&1 | tail -60'
"""
from __future__ import annotations

import os

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import richards_bulk_residual
from pids_forward.physics.overland_edge_kernel import build_top_facet_edge_graph
from pids_forward.physics.overland_routing import build_adjacency, node_widths
from pids_forward.physics.sequential_coupling import SequentialCoupledProblem
from scratch.seq_capped_infiltration_prototype import CappedSequentialCoupledProblem
from scratch.seq_iterative_prototype import (
    B1, make_box, run_case1_iter, run_case2, run_falsification,
)

COMM = MPI.COMM_WORLD


# =====================================================================================================
# STAGE 0 -- the h_ref CAP on attempt #1's (separate-pond-array + hard-Neumann-influx) structure.
# The ONLY change vs CappedSequentialCoupledProblem: q_pot's Kirchhoff upper limit is min(d, h_ref),
# so the capacity is the THIN-film saturated acceptance, NOT the inflated deep-pond value.
# =====================================================================================================
class HrefCappedNeumann(CappedSequentialCoupledProblem):
    """attempt-#1 capped-Neumann scheme, but q_pot = kirchhoff(psi_top, min(d, h_ref))/ell_c.

    Isolates whether capping the capacity at a thin reference film h_ref (decoupled from the accumulated
    pond) fixes the b1 partition. Still a HARD Neumann influx (so the no-Ss clay can force-feed -- the
    expected contrast)."""

    def __init__(self, mesh, soil, *, h_ref=1e-3, **kwargs):
        super().__init__(mesh, soil, **kwargs)
        self.h_ref = float(h_ref)

    def step(self, dt: float):
        # faithful copy of CappedSequentialCoupledProblem.step with the q_pot upper limit capped at
        # min(d_top, h_ref). Supply is still the FULL available column (rain*dt + d_top).
        self._ensure_built_capped()
        rp = self._rp
        rp.dt.value = dt
        td = self._top_dofs_arr
        A_i = self._A_i

        psi_entry = rp.psi.x.array.copy()
        d_entry = self.d_surf.copy()

        rain = float(self._rain_c.value)
        psi_top = psi_entry[td]
        d_top = d_entry[td]
        supply = rain * dt + d_top
        # *** THE FIX: capacity evaluated at the THIN reference film, not the accumulated pond. ***
        b_cap = np.minimum(d_top, self.h_ref)
        q_pot = np.array([self.soil.kirchhoff(float(a), float(b)) for a, b in zip(psi_top, b_cap)])
        q_pot = np.maximum(q_pot, 0.0) / self.ell_c
        inf_depth = np.minimum(q_pot * dt, supply)
        inf_depth = np.maximum(inf_depth, 0.0)
        active = supply > 1e-15
        capped = active & (q_pot * dt < supply - 1e-15)
        self.last_cap_frac = float(np.sum(capped)) / max(int(np.sum(active)), 1)

        q_inf_arr = np.zeros(self._n_dofs, dtype=np.float64)
        q_inf_arr[td] = inf_depth / dt
        self._q_inf.x.array[:] = q_inf_arr
        self._q_inf.x.scatter_forward()

        rp.psi.x.array[:] = psi_entry
        rp.psi_n.x.array[:] = psi_entry
        rp.psi.x.scatter_forward()
        rp.psi_n.x.scatter_forward()
        rp._ensure_problem()
        rp._problem.solve()
        snes = rp._problem.solver
        reason = int(snes.getConvergedReason())
        it = int(snes.getIterationNumber())
        fnorm = float(snes.getFunctionNorm())
        conv = reason > 0 and (reason != 4 or fnorm <= self.stall_accept_fnorm)
        self.last_reason = reason
        self.last_fnorm = fnorm
        if not conv:
            rp.psi.x.array[:] = psi_entry
            rp.psi_n.x.array[:] = psi_entry
            rp.psi.x.scatter_forward()
            rp.psi_n.x.scatter_forward()
            return False, it

        rp.psi_n.x.array[:] = rp.psi.x.array
        rp.psi_n.x.scatter_forward()

        d_post = supply - inf_depth
        d_new_arr = d_entry.copy()
        d_new_arr[td] = d_post
        d_routed, of = self._route(d_new_arr, dt)
        self.d_surf = d_routed

        of_booked = of * (1.0 + self.outflow_leak_frac)
        self.last_outflow = of
        self.cum_outflow += of_booked
        self.last_inf_total = float(np.sum(inf_depth * A_i[td])) / dt if dt > 0 else 0.0

        ghb_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                     for f in self._drain_forms]
        interior_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                          for f in self._interior_forms]
        total_rate = float(sum(ghb_rates) + sum(interior_rates))
        self.last_drainage = total_rate
        self.cum_drainage += dt * total_rate

        self.cum_rain += rain * self._top_area * dt
        self._t += dt
        return True, it


# =====================================================================================================
# REALIZATION B -- the self-limiting pond-in-psi draw on a CAPPED film (<= h_ref); excess held + routed.
# The soil draws the carried film at its natural Darcy rate (self-limiting, no force-feed), but the film
# never exceeds h_ref, so the deep lagged pond cannot over-drive infiltration. Rain + routing are handled
# EXPLICITLY (in d_held / the route sweep); the residual is bulk + pond-storage ONLY (rain/lat dropped).
# =====================================================================================================
class HrefCappedPondInPsi(SequentialCoupledProblem):
    """Realization B: pond-in-psi self-limiting draw, film capped at h_ref, excess held + routed.

    Reuses ALL the parent machinery (routing graph, areas, ledger, GHB/inlet sinks) but:
      * the residual is bulk Richards + pond-STORAGE only (the rain and lateral-source terms are dropped
        -- rain and routing are applied EXPLICITLY here, so the residual never double-counts them);
      * each step offers the soil a film = min(routed_pond, h_ref) carried in psi (psi_top := film), and
        holds the excess (routed_pond - film) in ``d_held`` (a separate surface array) which routes next.
    The infiltration is the soil's self-limiting Darcy response to the <=h_ref ponded head -- capped at
    the saturated-surface acceptance, not inflated by the lagged deep pond."""

    def __init__(self, mesh, soil, *, h_ref=1e-3, **kwargs):
        super().__init__(mesh, soil, **kwargs)
        self.h_ref = float(h_ref)
        self.d_held = np.zeros(self._n_dofs, dtype=np.float64)  # excess pond (> h_ref) held out of psi
        self.last_inf_total = 0.0
        self.max_film_seen = 0.0

    # -- residual: bulk + pond-storage ONLY (drop rain + lateral source; handled explicitly in step) --
    def _finalize_forms(self) -> None:
        rp = self._rp
        msh = self.mesh
        fdim = msh.topology.dim - 1
        msh.topology.create_connectivity(fdim, msh.topology.dim)
        rp.F = richards_bulk_residual(rp.psi, rp.psi_n, rp._v, self.soil, rp.dt, rp.e_g,
                                      dx_storage=rp._dx_storage, quadrature_degree=self._quad_degree)
        top_facets = np.sort(dmesh.locate_entities_boundary(
            msh, fdim, lambda x: np.isclose(x[self._zaxis], self._ztop))).astype(np.int32)
        all_f = [top_facets]
        all_t = [np.full(top_facets.size, 1, dtype=np.int32)]
        drain_specs = []
        for k, (loc, C, H) in enumerate(self._drains, start=2):
            df = np.sort(dmesh.locate_entities_boundary(msh, fdim, loc)).astype(np.int32)
            all_f.append(df)
            all_t.append(np.full(df.size, k, dtype=np.int32))
            drain_specs.append((k, C, H))
        ents = np.concatenate(all_f)
        tags = np.concatenate(all_t)
        order = np.argsort(ents)
        ft = dmesh.meshtags(msh, fdim, ents[order], tags[order])
        ds = ufl.Measure("ds", domain=msh, subdomain_data=ft,
                         metadata={"quadrature_degree": self._quad_degree})
        ds_top_v = ufl.Measure("ds", domain=msh, subdomain_data=ft,
                               metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})(1)
        z = ufl.SpatialCoordinate(msh)[self._zaxis]
        pond = ufl.max_value(rp.psi, 0.0)
        pond_n = ufl.max_value(rp.psi_n, 0.0)
        # rain DROPPED (handled explicitly in step). The lateral SOURCE is KEPT -- it carries the
        # conservative film change (psi's pond -> min(routed, h_ref)) via the parent's source mechanism,
        # WITHOUT a non-conservative psi writeback (writing psi_top resets theta -> mass creation).
        rp.F = rp.F + ((pond - pond_n) / rp.dt) * rp._v * ds_top_v \
            - self._lat_src * rp._v * ds_top_v
        self._pond_ledger = fem.form(ufl.max_value(rp.psi, 0.0) * ds_top_v)
        self._lat_src_ledger = fem.form(self._lat_src * ds_top_v)
        self._drain_forms = []
        for (k, C, H) in drain_specs:
            kr = self.soil.K_ufl(rp.psi) / self.soil.Ks
            q_n = C * kr * (rp.psi + z - H)
            rp.F = rp.F + q_n * rp._v * ds(k)
            self._drain_forms.append(fem.form(q_n * ds(k)))
        dxq = ufl.dx(metadata={"quadrature_degree": self._quad_degree})
        self._interior_forms = []
        for j, (cells, C, H, eps_act) in enumerate(self._interior_drains):
            chi = self._dg0_indicator(cells, f"chi_drain{j}")
            u = rp.psi + z - H
            pos = 0.5 * (u + ufl.sqrt(u * u + eps_act * eps_act))
            kr = self.soil.K_ufl(rp.psi) / self.soil.Ks
            q_vol = C * kr * pos * chi
            rp.F = rp.F + q_vol * rp._v * dxq
            self._interior_forms.append(fem.form(q_vol * dxq))
        rp._problem = None
        edges, L_e, T_e, A_i = build_top_facet_edge_graph(rp.V, msh, top_facets)
        self._A_i = A_i
        self._adj = build_adjacency(edges, L_e, self._n_dofs)
        self._W = node_widths(edges, L_e, T_e, self._n_dofs)
        self._top_area = float(A_i.sum())
        self._outlet_mask = np.zeros(self._n_dofs, dtype=bool)
        self._outlet_slope_node = np.zeros(self._n_dofs, dtype=np.float64)
        td = self._top_dofs_arr
        for (loc, slope) in self._outlets:
            sel = td[loc(self._coords[td].T)]
            self._outlet_mask[sel] = True
            self._outlet_slope_node[sel] = np.maximum(self._outlet_slope_node[sel], slope)
        if self._w0 is None:
            self._w0 = rp.total_water()
            self._surf0 = self._surf_pond()
        self._built = True

    def surface_water(self) -> float:
        self._ensure_built()
        td = self._top_dofs_arr
        return self._surf_pond() + float(np.sum(self.d_held[td] * self._A_i[td]))

    def balance(self) -> float:
        self._ensure_built()
        dtotal = self.total_water() - (self._w0 + self._surf0)
        return dtotal - (self.cum_rain - self.cum_outflow - self.cum_drainage)

    def step(self, dt: float):
        self._ensure_built()
        rp = self._rp
        rp.dt.value = dt
        td = self._top_dofs_arr
        A_i = self._A_i
        psi_entry = rp.psi.x.array.copy()
        rain = float(self._rain_c.value)

        film_entry = np.maximum(psi_entry[td], 0.0)     # pond carried in psi (<= h_ref)
        # total surface water (film + held excess) + rain; route the WHOLE sheet downslope.
        d_full = np.zeros(self._n_dofs, dtype=np.float64)
        d_full[td] = film_entry + self.d_held[td] + rain * dt
        d_routed, of = self._route(d_full, dt)
        film_target = np.minimum(d_routed[td], self.h_ref)   # what stays in psi (<= h_ref)
        held_new = d_routed[td] - film_target                # excess held out of psi (routes next step)
        self.max_film_seen = max(self.max_film_seen,
                                 float(film_target.max()) if film_target.size else 0.0)

        # change psi's film from film_entry to film_target via the lateral SOURCE (the parent's
        # CONSERVATIVE mechanism -- NO writeback of psi, so theta is never reset). The soil then draws the
        # <=h_ref film at its self-limiting Darcy rate (converged film = film_target - infiltrated).
        lat = np.zeros(self._n_dofs, dtype=np.float64)
        lat[td] = (film_target - film_entry) / dt
        self._lat_src.x.array[:] = lat
        self._lat_src.x.scatter_forward()
        rp.psi.x.array[:] = psi_entry
        rp.psi.x.scatter_forward()
        rp.psi_n.x.array[:] = psi_entry
        rp.psi_n.x.scatter_forward()
        rp._ensure_problem()
        rp._problem.solve()
        snes = rp._problem.solver
        reason = int(snes.getConvergedReason()); it = int(snes.getIterationNumber())
        fnorm = float(snes.getFunctionNorm())
        conv = reason > 0 and (reason != 4 or fnorm <= self.stall_accept_fnorm)
        self.last_reason, self.last_fnorm = reason, fnorm
        if not conv:
            rp.psi.x.array[:] = psi_entry; rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry; rp.psi_n.x.scatter_forward()
            return False, it

        # accept: the un-infiltrated film stays in psi; commit the held excess; book sinks.
        self.d_held[td] = held_new
        rp.psi_n.x.array[:] = rp.psi.x.array
        rp.psi_n.x.scatter_forward()

        of_booked = of * (1.0 + self.outflow_leak_frac)
        self.last_outflow = of
        self.cum_outflow += of_booked
        ghb_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                     for f in self._drain_forms]
        interior_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                          for f in self._interior_forms]
        total_rate = float(sum(ghb_rates) + sum(interior_rates))
        self.last_drainage = total_rate
        self.cum_drainage += dt * total_rate
        self.cum_rain += rain * self._top_area * dt
        self._t += dt
        return True, it


# =====================================================================================================
def _load_mono_target():
    cache = os.path.join(os.path.dirname(__file__), "seq_iter_mono_cache.npz")
    if not os.path.exists(cache):
        return None
    z = np.load(cache)
    return float(z["routed_R"])


def main():
    np.set_printoptions(precision=4, suppress=True)
    MESH1 = (30, 20, 8)
    mono_routed = _load_mono_target()
    print("#" * 92)
    print("REALIZATION B -- self-limiting pond-in-psi draw on a CAPPED film (h_ref); excess held+routed")
    print(f"  monolith (upwind, cached) target routed/R = "
          f"{mono_routed if mono_routed is not None else 'NO CACHE -- run seq_iterative_prototype first'}")
    print(f"  parent (uncapped) sequential ~0.17;  STAGE-0 hard-Neumann h_ref cap ALSO ~0.17 (frozen "
          f"q_pot never lets the surface saturate -> dead). B uses the SOLVED head -> should saturate.")
    print("#" * 92, flush=True)

    # b1 top-cell half-height ~ Lz/nz/2 = 0.0625 m; sweep h_ref from sub-mm to cm (the thin-film
    # saturated-acceptance head). Goal: routed/R -> the monolith's 0.547 at a thin film.
    print("\n=== B Case 1 (b1 LOAM) -- h_ref sweep ===", flush=True)
    print(f"{'h_ref [m]':>10} | {'routed/R':>9} {'infil/R':>9} {'closure':>9} "
          f"{'bal/rain':>10} {'resid pp':>9} | {'maxfilm':>8} steps wall ok", flush=True)
    for h_ref in (0.001, 0.002, 0.005, 0.01, 0.02):
        r = run_case1_iter(HrefCappedPondInPsi, *MESH1, route_substeps=4,
                           label=f"href={h_ref}", h_ref=h_ref)
        resid_pp = (r["routed_R"] - mono_routed) * 100 if mono_routed is not None else float("nan")
        print(f"{h_ref:>10.4f} | {r['routed_R']:>9.4f} {r['infil_R']:>9.4f} {r['closure']:>9.5f} "
              f"{r['bal_frac']:>10.1e} {resid_pp:>+9.1f} | {'-':>8} {r['nstep']:>5d} {r['wall']:>5.1f} "
              f"{r['ok']}", flush=True)

    # falsification (the conservation detector must still fire with the held store in the ledger).
    print("\n=== B falsification (10% outflow mis-book must break the ledger ~10%) ===", flush=True)
    f = run_falsification(HrefCappedPondInPsi, h_ref=0.002)
    print(f"  B falsify: bal={f['bal']:.3e} expect={f['expect']:.3e} ratio={f['ratio']:.3f} "
          f"(want ~1.0) |bal|/rain={f['rel']:.2e} (clean ~1e-9)", flush=True)

    # Case 2 -- stiff convergent clay-V robustness at a representative thin film.
    print("\n=== B Case 2 (stiff convergent CLAY-V) -- robustness + conservation ===", flush=True)
    for h_ref in (0.002, 0.01):
        c2 = run_case2(HrefCappedPondInPsi, label=f"B href={h_ref}", h_ref=h_ref)
        print(f"  h_ref={h_ref}: completed={c2['completed']} routed/R={c2['routed_R']:.4f} "
              f"infil/R={c2['infil_R']:.4f} bal/rain={c2['bal_frac']:.2e} "
              f"[nstep={c2['nstep']} coll={c2['coll']} t={c2['tend']:.3f} wall={c2['wall']:.1f}s]",
              flush=True)

    print("\n" + "#" * 92)
    print("READ: B Case-1 routed/R near the monolith 0.547 (thin h_ref) + clean ledger + clay-V "
          "completes => realization B is the fix; the cap needs the SELF-LIMITING (pond-in-psi) "
          "delivery, NOT the hard Neumann.")
    print("#" * 92, flush=True)


if __name__ == "__main__":
    main()
