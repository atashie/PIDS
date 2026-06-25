"""SCRATCH PROTOTYPE (no pids_forward/ edits, no commit) -- SEQUENTIAL-ITERATIVE (Picard) coupling
for the operator-split overland scheme.

THE PROBLEM
-----------
The merged ``SequentialCoupledProblem`` (sequential operator-split overland) OVER-INFILTRATES the
Hortonian run-on partition (~24 pp vs the ParFlow-validated monolith ``CoupledProblem``). ROOT CAUSE
(pinned, circular): the scheme does routing + infiltration as ONE pass per step, so the pond builds
DEEP before routing removes it; the pond-in-psi infiltration draws at the deep-head rate -> the soil
over-infiltrates. The monolith CO-SOLVES, keeping the pond THIN (routed off each Newton), so the
capacity stays low and the excess runs off. The correct partition is a SIMULTANEOUS fixed point:
routing keeps the pond thin <=> a thin pond keeps the infiltration low.

THE FIX PROTOTYPED HERE
-----------------------
Wrap the scheme's existing once-per-step coupling in an OUTER PICARD LOOP that iterates to a fixed
point on the surface pond each step (under-relaxed), so routing and infiltration compete WITHIN the
step and the pond settles to the thin-sheet equilibrium.

  VARIANT 1 (iteration only -- the EXISTING pond-in-psi closure):
    iterate m: route from the CURRENT pond d^m -> lateral source -> solve Richards-alone (psi_n FROZEN
    at the step-entry state, NOT re-snapshot per iterate) -> read d_solved = max(psi_solved,0) ->
    under-relax d^{m+1} = d^m + omega_p*(d_solved - d^m); stop when ||d^{m+1}-d^m||/||d|| < tol.

  VARIANT 2 (iteration + q_pot capacity closure -- only if V1 falls short):
    same outer Picard loop, but ALSO inject the monolith's capacity cap q_pot = kirchhoff(psi_top,d)/
    ell_c as a HEAD-LIMITING Neumann influx (the pond is tracked as a SEPARATE array and infiltration
    is min(q_pot, ...)*dt), so infiltration cannot exceed the thin-film capacity at the iterated pond.

The conservation ledger is the parent's: total = int theta dV + int max(psi,0) ds_top (V1) /
int theta dV + sum d_i A_i (V2, separate pond array). The outer Picard must keep it closing (|balance|/
cum_rain ~1e-9). A deliberate 10% outflow mis-book must break it by ~10% (the falsification gate).

Run (WSL pids-fem, threads pinned):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_iterative_prototype.py 2>&1 | tail -70'
"""
from __future__ import annotations

import time

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.sequential_coupling import SequentialCoupledProblem

COMM = MPI.COMM_WORLD


# =====================================================================================================
# VARIANT 1 -- the outer Picard loop on the EXISTING pond-in-psi closure.
# =====================================================================================================
class IterativeSequentialV1(SequentialCoupledProblem):
    """SequentialCoupledProblem with an OUTER PICARD LOOP on the surface pond each step (Variant 1).

    Reuses ALL of the parent's machinery (routing graph, areas, top dofs, outlet masks, the pond-in-psi
    Richards form built by _finalize_forms, the per-sink accounting, the ledger). Only ``step`` is
    overridden: instead of routing ONCE from the entry pond, it iterates route<->solve to a fixed point
    on the pond (under-relaxed), holding psi_n FROZEN at the step-entry state across the inner iterates.

    The fixed-point variable is the pond field ``d`` we route from. Each outer iterate m:
      1. route d^m over dt -> (d_routed, of); lateral source lat = omega*(d_routed - d^m)/dt on the top.
      2. reset psi := psi_entry, psi_n := psi_entry (psi_n is the BACKWARD-EULER baseline, the SAME for
         every iterate -- the iterate re-solves the SAME step toward the fixed point), solve Richards.
      3. d_solved = max(psi_solved, 0); under-relax d^{m+1} = d^m + omega_p*(d_solved - d^m).
      4. stop when ||d^{m+1}-d^m|| / max(||d||,floor) < picard_tol, else continue (cap picard_outer).
    On accept, BOOK the FINAL iterate's (of, lat_src): cum_outflow += omega*of, and the handoff
    consistency residual uses the SAME of -> the leak-detector still works. psi_n := psi_solved (post
    surface-inlet), exactly as the parent.

    The parent's omega-halving robustness retry is NESTED inside each outer iterate: if an inner solve
    fails, halve omega and re-route+re-solve THAT iterate; a shared budget caps the total attempts.
    """

    def __init__(self, mesh, soil, *, picard_omega=0.5, picard_tol=1e-3, picard_outer=20,
                 picard_floor=1e-4, **kwargs):
        super().__init__(mesh, soil, **kwargs)
        self.picard_omega = float(picard_omega)   # pond under-relaxation factor omega_p
        self.picard_tol = float(picard_tol)       # ||dd||/||d|| convergence tol
        self.picard_outer = int(picard_outer)     # outer-iteration cap (stiff-case fallback)
        self.picard_floor = float(picard_floor)   # ||d|| floor in the relative test [m] (avoid /0)
        # diagnostics
        self.last_picard_iters = 0
        self._picard_iter_hist: list[int] = []
        self.last_picard_resid = np.nan

    # ---- the iterated step ----------------------------------------------------------------------
    def step(self, dt: float):
        self._ensure_built()
        rp = self._rp
        rp.dt.value = dt
        top_dofs, A_i = self._top_dofs_arr, self._A_i

        psi_entry = rp.psi.x.array.copy()
        d_entry = np.maximum(psi_entry, 0.0)        # the step-entry pond d^0

        d_m = d_entry.copy()                        # the pond we route from (the fixed-point variable)
        attempts_total = 0                          # shared Newton-solve budget across outer iters
        budget = max(self.picard_outer, 1) + 4      # a few extra for omega-halving retries
        ok = False
        # accepted-iterate carry (booked only after the loop settles / hits the cap)
        of_acc = 0.0
        lat_acc = None
        it_acc = 0
        reason_acc = 0
        fnorm_acc = np.nan
        omega_acc = self.relax
        picard_iters = 0
        last_resid = np.nan

        for _outer in range(self.picard_outer):
            picard_iters += 1
            # --- route from the current pond estimate d^m ---------------------------------------
            d_routed, of = self._route(d_m, dt)
            # --- inner Richards solve with omega-halving robustness retry -----------------------
            omega = self.relax
            solved = False
            d_solved = None
            while True:
                attempts_total += 1
                lat = np.zeros(self._n_dofs, dtype=np.float64)
                lat[top_dofs] = omega * (d_routed[top_dofs] - d_m[top_dofs]) / dt
                self._lat_src.x.array[:] = lat
                self._lat_src.x.scatter_forward()
                # psi_n FROZEN at the step-entry state (NOT re-snapshot per iterate); psi reset to the
                # same Newton initial guess each iterate.
                rp.psi.x.array[:] = psi_entry
                rp.psi.x.scatter_forward()
                rp.psi_n.x.array[:] = psi_entry
                rp.psi_n.x.scatter_forward()
                rp._ensure_problem()
                rp._problem.solve()
                snes = rp._problem.solver
                reason = int(snes.getConvergedReason())
                it = int(snes.getIterationNumber())
                fnorm = float(snes.getFunctionNorm())
                conv = reason > 0 and (reason != 4 or fnorm <= self.stall_accept_fnorm)
                if conv:
                    d_solved = np.maximum(rp.psi.x.array.copy(), 0.0)
                    solved = True
                    break
                omega *= 0.5
                if omega < self.picard_floor or attempts_total >= budget:
                    break
            if not solved:
                break   # this outer iterate could not solve even at the omega floor -> fail the step

            # --- under-relaxed pond update + convergence test -----------------------------------
            d_next = d_m.copy()
            d_next[top_dofs] = (d_m[top_dofs]
                                + self.picard_omega * (d_solved[top_dofs] - d_m[top_dofs]))
            dd = float(np.linalg.norm((d_next - d_m)[top_dofs]))
            nrm = max(float(np.linalg.norm(d_next[top_dofs])), self.picard_floor)
            last_resid = dd / nrm

            # carry THIS iterate as the (current) accepted state -- the books use the FINAL iterate.
            of_acc, lat_acc = of, lat
            it_acc, reason_acc, fnorm_acc, omega_acc = it, reason, fnorm, omega

            d_m = d_next
            if last_resid < self.picard_tol:
                ok = True
                break
            if attempts_total >= budget:
                # hit the budget cap (stiff fallback): accept the last iterate as-is (still solved +
                # conserving; the partition may carry a small residual but the step COMPLETES).
                ok = True
                break
        else:
            # exhausted the outer cap without meeting tol: accept the last (solved) iterate (fallback).
            ok = solved

        self.last_picard_iters = picard_iters
        self.last_picard_resid = last_resid
        self.last_reason = reason_acc
        self.last_fnorm = fnorm_acc
        self.last_omega = omega_acc

        if not ok:
            # restore entry state -> caller cuts dt and retries (retry-safe).
            rp.psi.x.array[:] = psi_entry
            rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry
            rp.psi_n.x.scatter_forward()
            self._picard_iter_hist.append(picard_iters)
            return False, it_acc

        # RE-SOLVE the FINAL accepted source so psi sits at the booked state (the loop may have ended
        # on a convergence-test break whose last solve corresponds to d_m's PREVIOUS source; re-applying
        # the carried lat_acc and re-solving guarantees psi == the state lat_acc/of_acc describe, so the
        # ledger handoff residual is exact). One extra solve per step.
        self._lat_src.x.array[:] = lat_acc
        self._lat_src.x.scatter_forward()
        rp.psi.x.array[:] = psi_entry
        rp.psi.x.scatter_forward()
        rp.psi_n.x.array[:] = psi_entry
        rp.psi_n.x.scatter_forward()
        rp._ensure_problem()
        rp._problem.solve()

        # ---- accept: book exactly as the parent does (subsurface sinks at solved psi, then surface
        # inlet on the post-Richards pond, then psi_n := psi). --------------------------------------
        ghb_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                     for f in self._drain_forms]
        interior_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                          for f in self._interior_forms]

        inlet_rates = []
        psi_arr = rp.psi.x.array
        for (dofs, C_const) in self._inlets:
            d_post = np.maximum(psi_arr[dofs], 0.0)
            coeff = float(C_const.value)
            remove_depth = np.minimum(coeff * d_post * dt, d_post)
            psi_arr[dofs] -= remove_depth
            inlet_rates.append(float(np.sum(remove_depth * A_i[dofs])) / dt if dt > 0 else 0.0)
        if self._inlets:
            rp.psi.x.scatter_forward()

        rp.psi_n.x.array[:] = rp.psi.x.array
        rp.psi_n.x.scatter_forward()

        of_applied = omega_acc * of_acc
        of_booked = of_applied * (1.0 + self.outflow_leak_frac)
        self.last_outflow = of_applied
        self.cum_outflow += of_booked

        lat_int = self.mesh.comm.allreduce(fem.assemble_scalar(self._lat_src_ledger), op=MPI.SUM)
        self.last_handoff_resid = lat_int * dt + of_applied
        self.cum_handoff_imbalance += self.last_handoff_resid

        total_rate = 0.0
        sink_sources = (("ghb", ghb_rates),
                        ("interior_drain", interior_rates),
                        ("surface_inlet", inlet_rates))
        for kind, rates in sink_sources:
            while len(self.last_sinks[kind]) < len(rates):
                self.last_sinks[kind].append(0.0)
                self.cum_sinks[kind].append(0.0)
            for i, r in enumerate(rates):
                self.last_sinks[kind][i] = r
                self.cum_sinks[kind][i] += dt * r
                total_rate += r
        self.last_drainage = total_rate
        self.cum_drainage += dt * total_rate

        self.cum_rain += float(self._rain_c.value) * self._top_area * dt

        # routing-store residual (use the FINAL accepted route: m_pre from d at that iterate). Recompute
        # from the carried lat to stay consistent: the booked of_acc came from routing d_m_prev; report
        # the parent-style residual on the final route.
        self._t += dt
        self._picard_iter_hist.append(picard_iters)
        return True, it_acc

    def picard_iter_stats(self):
        h = np.asarray(self._picard_iter_hist, dtype=float)
        if h.size == 0:
            return dict(n=0, avg=float("nan"), mx=float("nan"))
        return dict(n=int(h.size), avg=float(h.mean()), mx=int(h.max()))


# =====================================================================================================
# VARIANT 2 -- the outer Picard loop + q_pot capacity closure (separate pond array, capped influx).
# =====================================================================================================
class IterativeSequentialV2(SequentialCoupledProblem):
    """Variant 2: the outer Picard loop, but infiltration is CAPPED at the monolith's capacity
    q_pot = kirchhoff(psi_top, d)/ell_c, evaluated at the ITERATED (thin) pond.

    Unlike V1 (pond carried in psi), here the pond ``d`` is a SEPARATE surface array and the Richards
    solve sees a per-step Neumann INFLUX BC equal to the capped infiltration rate (psi_top a free
    Richards variable). The outer Picard iterates the pond: a thinner pond lowers q_pot -> less
    infiltration -> a deeper pond, and vice-versa; the fixed point is the thin-film equilibrium with
    the capacity consistently evaluated at it.

    Each outer iterate m (pond d^m over the top, post-rain):
      supply = d^m                                   (water at the surface this iterate; rain added once)
      q_pot  = kirchhoff(psi_top_entry, d^m)/ell_c   (capacity at the iterated pond, entry psi_top)
      inf    = min(q_pot*dt, supply)                 (capacity- or supply-limited infiltration depth)
      solve Richards with influx inf/dt; d_after_inf = supply - inf
      route d_after_inf -> (d_routed, of); d^{m+1} = d^m + omega_p*(d_routed - d^m)
    Converged when ||d^{m+1}-d^m||/||d|| < tol. The ledger is total = int theta + sum d_i A_i.
    """

    def __init__(self, mesh, soil, *, ell_c=None, picard_omega=0.5, picard_tol=1e-3, picard_outer=20,
                 picard_floor=1e-4, **kwargs):
        super().__init__(mesh, soil, **kwargs)
        if ell_c is None:
            zc = self._rp.V.tabulate_dof_coordinates()[:, self._zaxis]
            zu = np.unique(np.round(zc, 9))
            if zu.size < 2:
                raise ValueError("ell_c auto-detect: <2 z-levels.")
            ell_c = 0.5 * float(zu[-1] - zu[-2])
        self.ell_c = float(ell_c)
        self.picard_omega = float(picard_omega)
        self.picard_tol = float(picard_tol)
        self.picard_outer = int(picard_outer)
        self.picard_floor = float(picard_floor)
        self.d_surf = np.zeros(self._n_dofs, dtype=np.float64)   # separate pond array (NOT in psi)
        self._q_inf = fem.Function(self._rp.V, name="q_inf")     # per-step Neumann influx [m/day]
        self._built_v2 = False
        self.last_picard_iters = 0
        self._picard_iter_hist: list[int] = []
        self.last_picard_resid = np.nan
        self.last_cap_frac = 0.0
        # V2 ledger (separate pond): reset and re-baselined in _finalize_v2.
        self.cum_rain = 0.0
        self.cum_outflow = 0.0
        self.cum_drainage = 0.0
        self._w0 = None
        self._surf0 = 0.0

    def _finalize_v2(self):
        """Build the Richards form with a Neumann INFLUX (no pond-in-psi storage/rain) + the routing
        graph; baseline the separate-pond ledger."""
        from pids_forward.physics.richards import richards_bulk_residual
        from pids_forward.physics.overland_edge_kernel import build_top_facet_edge_graph
        from pids_forward.physics.overland_routing import build_adjacency, node_widths
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
        # Neumann influx into the soil top (matched lumped vertex measure, so the residual removes
        # exactly the volume we book infiltrated). - q_inf*v*ds (positive INTO the domain).
        rp.F = rp.F - self._q_inf * rp._v * ds_top_v
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
            self._surf0 = float(np.sum(self.d_surf[td] * A_i[td]))
        self._built_v2 = True

    def _ensure_v2(self):
        if not self._built_v2:
            self._finalize_v2()

    def step(self, dt: float):
        self._ensure_v2()
        rp = self._rp
        rp.dt.value = dt
        td = self._top_dofs_arr
        A_i = self._A_i
        psi_entry = rp.psi.x.array.copy()
        psi_top = psi_entry[td]
        rain = float(self._rain_c.value)

        # add rain ONCE to the entry pond; the outer loop iterates the pond from there.
        d0 = self.d_surf.copy()
        d0[td] += rain * dt
        d_m = d0[td].copy()                  # the iterated pond over the top (post-rain)

        picard_iters = 0
        last_resid = np.nan
        ok = False
        # carried accepted state
        inf_acc = None
        of_acc = 0.0
        d_route_acc = None
        it_acc = reason_acc = 0
        fnorm_acc = np.nan
        attempts = 0
        budget = self.picard_outer + 4

        for _outer in range(self.picard_outer):
            picard_iters += 1
            supply = d_m.copy()
            # capacity q_pot = kirchhoff(psi_top, d)/ell_c at the iterated pond (entry psi_top).
            q_pot = np.array([self.soil.kirchhoff(float(a), float(b))
                              for a, b in zip(psi_top, d_m)])
            q_pot = np.maximum(q_pot, 0.0) / self.ell_c
            inf = np.minimum(q_pot * dt, supply)
            inf = np.maximum(inf, 0.0)
            active = supply > 1e-15
            self.last_cap_frac = (float(np.sum(active & (q_pot * dt < supply - 1e-15)))
                                  / max(int(np.sum(active)), 1))
            # solve Richards with the capped influx (omega-halving not needed for an influx BC, but we
            # keep a single attempt + a fail-out on a non-converged solve).
            qa = np.zeros(self._n_dofs)
            qa[td] = inf / dt
            self._q_inf.x.array[:] = qa
            self._q_inf.x.scatter_forward()
            rp.psi.x.array[:] = psi_entry
            rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry
            rp.psi_n.x.scatter_forward()
            rp._ensure_problem()
            attempts += 1
            rp._problem.solve()
            snes = rp._problem.solver
            reason, it, fn = (int(snes.getConvergedReason()), int(snes.getIterationNumber()),
                              float(snes.getFunctionNorm()))
            if not (reason > 0 and (reason != 4 or fn <= self.stall_accept_fnorm)):
                break    # inner solve failed -> fail the step (caller cuts dt)

            # the un-infiltrated remainder, then route it.
            d_after = d0.copy()
            d_after[td] = supply - inf
            d_routed, of = self._route(d_after, dt)
            d_next = d_m + self.picard_omega * (d_routed[td] - d_m)
            dd = float(np.linalg.norm(d_next - d_m))
            nrm = max(float(np.linalg.norm(d_next)), self.picard_floor)
            last_resid = dd / nrm

            inf_acc, of_acc, d_route_acc = inf.copy(), of, d_routed.copy()
            it_acc, reason_acc, fnorm_acc = it, reason, fn
            d_m = d_next
            if last_resid < self.picard_tol or attempts >= budget:
                ok = True
                break
        else:
            ok = inf_acc is not None

        self.last_picard_iters = picard_iters
        self.last_picard_resid = last_resid
        self.last_reason = reason_acc
        self.last_fnorm = fnorm_acc

        if not ok:
            rp.psi.x.array[:] = psi_entry
            rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry
            rp.psi_n.x.scatter_forward()
            self._picard_iter_hist.append(picard_iters)
            return False, it_acc

        # RE-SOLVE the FINAL accepted influx so psi sits at the booked infiltration state (consistency).
        qa = np.zeros(self._n_dofs)
        qa[td] = inf_acc / dt
        self._q_inf.x.array[:] = qa
        self._q_inf.x.scatter_forward()
        rp.psi.x.array[:] = psi_entry
        rp.psi.x.scatter_forward()
        rp.psi_n.x.array[:] = psi_entry
        rp.psi_n.x.scatter_forward()
        rp._ensure_problem()
        rp._problem.solve()
        rp.psi_n.x.array[:] = rp.psi.x.array
        rp.psi_n.x.scatter_forward()

        # commit the routed pond (the FINAL accepted route).
        self.d_surf = d_route_acc

        of_booked = of_acc * (1.0 + self.outflow_leak_frac)
        self.last_outflow = of_acc
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
        self._picard_iter_hist.append(picard_iters)
        return True, it_acc

    # V2 ledger over the separate pond array.
    def surface_water(self) -> float:
        self._ensure_v2()
        td = self._top_dofs_arr
        return float(np.sum(self.d_surf[td] * self._A_i[td]))

    def soil_water(self) -> float:
        return self._rp.total_water()

    def total_water(self) -> float:
        return self.soil_water() + self.surface_water()

    def balance(self) -> float:
        self._ensure_v2()
        dtotal = self.total_water() - (self._w0 + self._surf0)
        return dtotal - (self.cum_rain - self.cum_outflow - self.cum_drainage)

    def picard_iter_stats(self):
        h = np.asarray(self._picard_iter_hist, dtype=float)
        if h.size == 0:
            return dict(n=0, avg=float("nan"), mx=float("nan"))
        return dict(n=int(h.size), avg=float(h.mean()), mx=int(h.max()))


# =====================================================================================================
# Partition-measurement helpers (lifted verbatim from runon_partition_investigation.py).
# =====================================================================================================
def _top_area_ds(mesh, ztop):
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    tf = np.sort(dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[mesh.geometry.dim - 1], ztop))).astype(np.int32)
    ft = dmesh.meshtags(mesh, fdim, tf, np.ones(tf.size, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=mesh, subdomain_data=ft,
                         metadata={"quadrature_degree": 8})(1)
    return mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ds_top)), op=MPI.SUM)


def _soil_water_deg8(prob, soil):
    dxq = ufl.dx(metadata={"quadrature_degree": prob._quad_degree})
    return prob.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(soil.theta_ufl(prob.psi) * dxq)), op=MPI.SUM)


def _mono_surface_store(prob):
    return prob.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(prob.d * prob._ds_top)), op=MPI.SUM)


def _march_storm(prob, rain, *, storm_dur, storm_rain, t_end, dt0=1e-3, dt_max=0.03,
                 ctrl_low=4, ctrl_high=12):
    """March a storm-then-recession with the production BAND dt-controller. Works for all schemes
    (same step(dt)->(conv,it)). Returns (nstep, collapsed, t_reached, wall)."""
    t, nstep, dt = 0.0, 0, dt0
    collapsed = False
    t0 = time.perf_counter()
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if t < storm_dur - 1e-12 and t + h > storm_dur:
            h = storm_dur - t
        rain.value = storm_rain if t < storm_dur - 1e-12 else 0.0
        conv, it = prob.step(h)
        if not conv:
            dt *= 0.5
            if dt < 1e-9:
                collapsed = True
                break
            continue
        t += h
        nstep += 1
        if it <= ctrl_low:
            dt = min(dt * 1.4, dt_max)
        elif it >= ctrl_high:
            dt = dt * 0.7
    return nstep, collapsed, t, time.perf_counter() - t0


# ---- the b1 fixture (Case 1) and the clay-V fixture (Case 2) -----------------------------------------
B1 = dict(Lx=8.0, Ly=5.0, Lz=1.0, S0=0.03, psi_i=-0.4, RAIN=0.5, STORM=0.08, TEND=0.45,
          soil=dict(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25))
CV = dict(Lx=10.0, Ly=6.0, Lz=1.5, psi_i=-0.30, RAIN=1.0, STORM=0.08, TEND=0.40,
          soil=dict(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09, Ks=0.048))


def make_box(nx, ny, nz, Lx, Ly, Lz):
    return dmesh.create_box(
        COMM, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz], cell_type=dmesh.CellType.tetrahedron)


# ---- Case 1: the monolith target (upwind) at the SAME mesh ------------------------------------------
def run_case1_monolith(nx, ny, nz, scheme="upwind"):
    f = B1
    soil = VanGenuchten(**f["soil"])
    msh = make_box(nx, ny, nz, f["Lx"], f["Ly"], f["Lz"])
    mono = CoupledProblem(msh, soil, n_man=0.05, overland_scheme=scheme)
    mono.set_initial_condition(lambda x: f["psi_i"] + 0.0 * x[0], d_value=0.0)
    mono.set_topography(lambda x: f["S0"] * x[1])
    mono.add_rain(0.0)
    mono.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=f["S0"])
    th0 = _soil_water_deg8(mono, soil)
    rain = mono._rain
    nstep, coll, tend, wall = _march_storm(
        mono, rain, storm_dur=f["STORM"], storm_rain=f["RAIN"], t_end=f["TEND"],
        ctrl_low=3, ctrl_high=8)
    top_area = _top_area_ds(msh, f["Lz"])
    R_in = f["RAIN"] * top_area * f["STORM"]
    soil_gain = _soil_water_deg8(mono, soil) - th0
    routed, drained = mono.cum_outflow, mono.cum_drainage
    surf = _mono_surface_store(mono)
    eff = mono._effective_overland_scheme
    return dict(routed_R=routed / R_in, infil_R=(soil_gain + drained) / R_in,
                surf_R=surf / R_in, closure=(routed + soil_gain + drained + surf) / R_in,
                nstep=nstep, wall=wall, coll=coll, tend=tend, eff=eff, R_in=R_in,
                ok=(eff == scheme and not coll and tend >= f["TEND"] - 1e-9))


# ---- Case 1: an iterative prototype on b1 ----------------------------------------------------------
def run_case1_iter(cls, nx, ny, nz, *, route_substeps=8, label="", **kw):
    f = B1
    soil = VanGenuchten(**f["soil"])
    msh = make_box(nx, ny, nz, f["Lx"], f["Ly"], f["Lz"])
    prob = cls(msh, soil, n_man=0.05, route_substeps=route_substeps, **kw)
    prob.set_initial_condition(lambda x: f["psi_i"] + 0.0 * x[0])
    prob.set_topography(lambda x: f["S0"] * x[1])
    prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=f["S0"])
    th0 = _soil_water_deg8(prob, soil)
    rain = prob._rain_c
    nstep, coll, tend, wall = _march_storm(
        prob, rain, storm_dur=f["STORM"], storm_rain=f["RAIN"], t_end=f["TEND"])
    top_area = _top_area_ds(msh, f["Lz"])
    R_in = f["RAIN"] * top_area * f["STORM"]
    soil_gain = _soil_water_deg8(prob, soil) - th0
    routed, drained = prob.cum_outflow, prob.cum_drainage
    surf = prob.surface_water()
    bal = prob.balance()
    bal_frac = abs(bal) / prob.cum_rain if prob.cum_rain > 0 else float("nan")
    st = prob.picard_iter_stats() if hasattr(prob, "picard_iter_stats") \
        else dict(avg=1.0, mx=1)   # the parent runs exactly one pass per step
    return dict(routed_R=routed / R_in, infil_R=(soil_gain + drained) / R_in,
                surf_R=surf / R_in, closure=(routed + soil_gain + drained + surf) / R_in,
                nstep=nstep, wall=wall, coll=coll, tend=tend, R_in=R_in, bal=bal, bal_frac=bal_frac,
                pic_avg=st["avg"], pic_max=st["mx"], label=label,
                ok=(not coll and tend >= f["TEND"] - 1e-9))


# ---- Case 2: stiff clay-V robustness ---------------------------------------------------------------
def run_case2(cls, nx=24, ny=16, nz=5, *, route_substeps=4, label="", **kw):
    f = CV
    soil = VanGenuchten(**f["soil"])
    msh = make_box(nx, ny, nz, f["Lx"], f["Ly"], f["Lz"])
    prob = cls(msh, soil, n_man=0.05, route_substeps=route_substeps, **kw)
    prob.set_initial_condition(lambda x: f["psi_i"] + 0.0 * x[0])
    prob.set_topography(lambda x: 0.05 * x[1] + 0.08 * np.abs(x[0] - 5.0))
    prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=0.05)
    th0 = _soil_water_deg8(prob, soil)
    rain = prob._rain_c
    nstep, coll, tend, wall = _march_storm(
        prob, rain, storm_dur=f["STORM"], storm_rain=f["RAIN"], t_end=f["TEND"], dt0=2e-3, dt_max=0.02)
    completed = (not coll) and tend >= f["TEND"] - 1e-9
    top_area = _top_area_ds(msh, f["Lz"])
    R_in = f["RAIN"] * top_area * f["STORM"]
    soil_gain = _soil_water_deg8(prob, soil) - th0
    bal = prob.balance()
    bal_frac = abs(bal) / prob.cum_rain if prob.cum_rain > 0 else float("nan")
    st = prob.picard_iter_stats() if hasattr(prob, "picard_iter_stats") \
        else dict(avg=1.0, mx=1)
    return dict(completed=completed, coll=coll, tend=tend, nstep=nstep, wall=wall,
                routed_R=prob.cum_outflow / R_in,
                infil_R=(soil_gain + prob.cum_drainage) / R_in,
                bal=bal, bal_frac=bal_frac, pic_avg=st["avg"], pic_max=st["mx"], label=label)


# ---- Falsification: a 10% outflow mis-book must break the ledger ~10% -------------------------------
def run_falsification(cls, nx=20, ny=14, nz=6, **kw):
    f = B1
    soil = VanGenuchten(**f["soil"])
    msh = make_box(nx, ny, nz, f["Lx"], f["Ly"], f["Lz"])
    prob = cls(msh, soil, n_man=0.05, route_substeps=4, **kw)
    prob.outflow_leak_frac = 0.10
    prob.set_initial_condition(lambda x: f["psi_i"] + 0.0 * x[0])
    prob.set_topography(lambda x: f["S0"] * x[1])
    prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=f["S0"])
    rain = prob._rain_c
    _march_storm(prob, rain, storm_dur=f["STORM"], storm_rain=f["RAIN"], t_end=f["TEND"])
    bal = prob.balance()
    true_outflow = prob.cum_outflow / 1.10
    expect = -0.10 * true_outflow
    ratio = bal / expect if expect != 0 else float("nan")
    return dict(bal=bal, expect=expect, ratio=ratio,
                rel=abs(bal) / prob.cum_rain if prob.cum_rain > 0 else float("nan"))


# ====================================================================================================
def main():
    np.set_printoptions(precision=4, suppress=True)
    MESH1 = (30, 20, 8)
    print("#" * 90)
    print("SEQUENTIAL-ITERATIVE (Picard) overland coupling -- prototype validation")
    print(f"  Case 1 = b1 mild planar LOAM, mesh {MESH1[0]}x{MESH1[1]}x{MESH1[2]} (partition closure)")
    print("  Case 2 = stiff convergent CLAY-V, mesh 24x16x5 (robustness + conservation)")
    print("#" * 90, flush=True)

    # ---- the apples-to-apples MONOLITH target at the SAME mesh (run FIRST to fix the target). The
    # upwind monolith costs ~280s here; CACHE it to an npz so a re-run (after a fixable crash in a
    # later stage) skips it. Delete scratch/seq_iter_mono_cache.npz to force a fresh monolith.
    import os
    cache = os.path.join(os.path.dirname(__file__), "seq_iter_mono_cache.npz")
    print("\n=== MONOLITH target (upwind) @ %dx%dx%d ===" % MESH1, flush=True)
    if os.path.exists(cache):
        z = np.load(cache)
        mono = {k: float(z[k]) for k in z.files}
        mono["ok"] = bool(mono["ok"])
        mono["eff"] = "upwind"
        print(f"  MONO(upwind) [CACHED]: routed/R={mono['routed_R']:.4f} "
              f"infil/R={mono['infil_R']:.4f} surf/R={mono['surf_R']:.4f} "
              f"closure={mono['closure']:.5f} ok={mono['ok']}", flush=True)
    else:
        mono = run_case1_monolith(*MESH1, scheme="upwind")
        np.savez(cache, routed_R=mono["routed_R"], infil_R=mono["infil_R"], surf_R=mono["surf_R"],
                 closure=mono["closure"], ok=float(mono["ok"]), R_in=mono["R_in"])
        print(f"  MONO(upwind): routed/R={mono['routed_R']:.4f} infil/R={mono['infil_R']:.4f} "
              f"surf/R={mono['surf_R']:.4f} closure={mono['closure']:.5f}  "
              f"[eff={mono['eff']} nstep={mono['nstep']} wall={mono['wall']:.1f}s ok={mono['ok']}]",
              flush=True)
    MONO_ROUTED = mono["routed_R"]

    # ---- baseline: the CURRENT once-per-step sequential (picard disabled, 1 outer iter) at rs8. -----
    print("\n=== BASELINE once-per-step sequential (parent, rs8) @ %dx%dx%d ===" % MESH1, flush=True)
    base = run_case1_iter(SequentialCoupledProblem, *MESH1, route_substeps=8, label="parent")
    print(f"  PARENT(rs8): routed/R={base['routed_R']:.4f} infil/R={base['infil_R']:.4f} "
          f"surf/R={base['surf_R']:.4f} closure={base['closure']:.5f} bal/rain={base['bal_frac']:.1e}  "
          f"[nstep={base['nstep']} wall={base['wall']:.1f}s ok={base['ok']}]", flush=True)

    # =================================================================================================
    # VARIANT 1 -- iteration only.
    # =================================================================================================
    print("\n" + "=" * 90)
    print("VARIANT 1 -- outer Picard on the pond-in-psi closure")
    print("=" * 90, flush=True)
    v1 = run_case1_iter(IterativeSequentialV1, *MESH1, route_substeps=8, label="V1",
                        picard_omega=0.5, picard_tol=1e-3, picard_outer=20)
    print(f"  V1 Case1: routed/R={v1['routed_R']:.4f} infil/R={v1['infil_R']:.4f} "
          f"surf/R={v1['surf_R']:.4f} closure={v1['closure']:.5f}", flush=True)
    print(f"           bal={v1['bal']:.2e} bal/rain={v1['bal_frac']:.2e}  "
          f"picard_iters avg={v1['pic_avg']:.2f} max={v1['pic_max']}  "
          f"[nstep={v1['nstep']} wall={v1['wall']:.1f}s ok={v1['ok']}]", flush=True)
    print(f"  V1 vs MONO: routed/R {v1['routed_R']:.4f} vs {MONO_ROUTED:.4f}  "
          f"residual = {v1['routed_R'] - MONO_ROUTED:+.4f} ({(v1['routed_R']-MONO_ROUTED)*100:+.1f} pp)",
          flush=True)

    print("\n--- V1 Case 2 (stiff clay-V robustness) ---", flush=True)
    v1c2 = run_case2(IterativeSequentialV1, label="V1", picard_omega=0.5, picard_tol=1e-3,
                     picard_outer=20)
    print(f"  V1 Case2: completed={v1c2['completed']} routed/R={v1c2['routed_R']:.4f} "
          f"infil/R={v1c2['infil_R']:.4f}  bal/rain={v1c2['bal_frac']:.2e}  "
          f"picard avg={v1c2['pic_avg']:.2f} max={v1c2['pic_max']}  "
          f"[nstep={v1c2['nstep']} wall={v1c2['wall']:.1f}s coll={v1c2['coll']} t={v1c2['tend']:.3f}]",
          flush=True)

    print("\n--- V1 falsification (10% outflow mis-book) ---", flush=True)
    v1f = run_falsification(IterativeSequentialV1, picard_omega=0.5)
    print(f"  V1 falsify: bal={v1f['bal']:.3e} expect={v1f['expect']:.3e} ratio={v1f['ratio']:.3f} "
          f"(want ~1.0) |bal|/rain={v1f['rel']:.2e} (clean was ~1e-9)", flush=True)

    # decide whether V2 is needed.
    v1_resid_pp = abs(v1["routed_R"] - MONO_ROUTED) * 100.0
    v1_ok = v1["ok"] and np.isfinite(v1["bal_frac"]) and v1["bal_frac"] < 1e-6
    run_v2 = (v1_resid_pp > 5.0) or (not v1_ok)
    print(f"\n>>> V1 residual {v1_resid_pp:.1f} pp, ledger clean={v1_ok}; "
          f"{'RUNNING V2 (capacity closure)' if run_v2 else 'V2 not needed (V1 closes it)'}",
          flush=True)

    v2 = v2c2 = v2f = None
    if run_v2:
        print("\n" + "=" * 90)
        print("VARIANT 2 -- outer Picard + q_pot capacity closure (separate pond array, capped influx)")
        print("=" * 90, flush=True)
        v2 = run_case1_iter(IterativeSequentialV2, *MESH1, route_substeps=8, label="V2",
                            picard_omega=0.5, picard_tol=1e-3, picard_outer=20)
        print(f"  V2 Case1: routed/R={v2['routed_R']:.4f} infil/R={v2['infil_R']:.4f} "
              f"surf/R={v2['surf_R']:.4f} closure={v2['closure']:.5f}", flush=True)
        print(f"           bal={v2['bal']:.2e} bal/rain={v2['bal_frac']:.2e}  "
              f"picard avg={v2['pic_avg']:.2f} max={v2['pic_max']}  "
              f"[nstep={v2['nstep']} wall={v2['wall']:.1f}s ok={v2['ok']}]", flush=True)
        print(f"  V2 vs MONO: routed/R {v2['routed_R']:.4f} vs {MONO_ROUTED:.4f}  "
              f"residual = {v2['routed_R'] - MONO_ROUTED:+.4f} "
              f"({(v2['routed_R']-MONO_ROUTED)*100:+.1f} pp)", flush=True)

        print("\n--- V2 Case 2 (stiff clay-V robustness) ---", flush=True)
        v2c2 = run_case2(IterativeSequentialV2, label="V2", picard_omega=0.5, picard_tol=1e-3,
                         picard_outer=20)
        print(f"  V2 Case2: completed={v2c2['completed']} routed/R={v2c2['routed_R']:.4f} "
              f"infil/R={v2c2['infil_R']:.4f}  bal/rain={v2c2['bal_frac']:.2e}  "
              f"picard avg={v2c2['pic_avg']:.2f} max={v2c2['pic_max']}  "
              f"[nstep={v2c2['nstep']} coll={v2c2['coll']} t={v2c2['tend']:.3f}]", flush=True)

        print("\n--- V2 falsification (10% outflow mis-book) ---", flush=True)
        v2f = run_falsification(IterativeSequentialV2, picard_omega=0.5)
        print(f"  V2 falsify: bal={v2f['bal']:.3e} expect={v2f['expect']:.3e} ratio={v2f['ratio']:.3f} "
              f"(want ~1.0) |bal|/rain={v2f['rel']:.2e}", flush=True)

    # =================================================================================================
    # SUMMARY
    # =================================================================================================
    print("\n" + "#" * 90)
    print("SUMMARY")
    print("#" * 90)
    print(f"Case-1 partition routed/R @ {MESH1}:")
    print(f"   MONOLITH (upwind, the ParFlow-validated target): {MONO_ROUTED:.4f}")
    print(f"   PARENT once-per-step sequential (rs8):           {base['routed_R']:.4f}  "
          f"(residual {(base['routed_R']-MONO_ROUTED)*100:+.1f} pp)")
    print(f"   VARIANT 1 (iteration only):                      {v1['routed_R']:.4f}  "
          f"(residual {(v1['routed_R']-MONO_ROUTED)*100:+.1f} pp)  "
          f"bal/rain={v1['bal_frac']:.1e}  picard avg={v1['pic_avg']:.2f}/max={v1['pic_max']}")
    if v2 is not None:
        print(f"   VARIANT 2 (iteration + q_pot cap):               {v2['routed_R']:.4f}  "
              f"(residual {(v2['routed_R']-MONO_ROUTED)*100:+.1f} pp)  "
              f"bal/rain={v2['bal_frac']:.1e}  picard avg={v2['pic_avg']:.2f}/max={v2['pic_max']}")
    print()
    print(f"Case-2 (clay-V robustness): V1 completed={v1c2['completed']} bal/rain={v1c2['bal_frac']:.1e}"
          + (f" | V2 completed={v2c2['completed']} bal/rain={v2c2['bal_frac']:.1e}"
             if v2c2 is not None else ""))
    print(f"Falsification (want |ratio|~1): V1={v1f['ratio']:.3f}"
          + (f" | V2={v2f['ratio']:.3f}" if v2f is not None else ""))
    print("#" * 90, flush=True)


if __name__ == "__main__":
    main()
