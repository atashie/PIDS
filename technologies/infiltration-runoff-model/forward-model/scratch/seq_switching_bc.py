"""SCRATCH SPIKE -- the CANONICAL sequential infiltration closure: a Neumann<->Dirichlet SWITCHING
boundary condition (CATHY / HydroGeoSphere / GSSHA style), the standard approach §17 found we had
NEVER tested (all prior closures were pond-in-psi film-offers or q_pot=kirchhoff/ell_c caps).

THE IDEA (per the Codex critical review, validation/sanity/overland_sequential_critical_review__2026-06-26):
  * a SEPARATE surface store d (NOT pond-in-psi);
  * DRY top nodes -> NEUMANN (rain influx) -- the soil takes the rain if it can;
  * PONDED/saturated nodes -> DIRICHLET psi_top = 0 (thin saturated sheet; lakes out of scope) imposed as
    a strong per-node PENALTY (Robin) c_pen*(psi-0); the realized infiltration = the BOUNDARY REACTION
    FLUX, which the Richards solve computes INCLUDING gravity (-> ~Ks at steady ponded infiltration) --
    exactly the term the q_pot=kirchhoff/ell_c cap structurally MISSED;
  * ACTIVE-SET iteration: a dry node that saturates (psi_top>0) switches to ponded; a ponded node that
    cannot sustain psi=0 with the available water switches to supply-limited Neumann.
Conservation is EXACT by the weak-form mass balance: with v=1, INT(theta-theta_n)/dt = INT(q_top - c_pen*psi)
ds = SUM infil_i*A_i, so Delta(INT theta) = SUM infil_i*A_i*dt; rain in, infil to soil, excess ponds+routes.

GATE: does it hit loam 0.547 / sand 0.694 / silt 0.767 (uniform mesh, NO knob, stable, conservative)?

Run (WSL pids-fem, threads pinned) -- LIVE to a file (NO tail):
  wsl bash -c 'cd .../forward-model && export PATH="/root/miniforge3/envs/pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_switching_bc.py'
"""
from __future__ import annotations

import time

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
from scratch.seq_href_closure_study import make_box, _march
from scratch.seq_cocycled_skin import make_graded_box
from scratch.seq_iterative_prototype import _top_area_ds


class SwitchingBCSplit(SequentialCoupledProblem):
    """Sequential overland with a Neumann<->Dirichlet active-set SWITCHING top BC + separate surface store.

    Reuses the parent's routing graph / areas / outlet machinery (built in _finalize_forms) but replaces
    the pond-in-psi residual with: bulk Richards + a per-node surface coupling (c_pen*psi - q_top) on the
    matched vertex measure, where c_pen (penalty toward psi=0) and q_top (Neumann influx) are Functions
    toggled per node by the active set each step. NO form rebuild (only the Function values change)."""

    def __init__(self, mesh, soil, *, c_pen_big=2e4, pond_eps=1e-9, sat_eps=0.0,
                 active_set_max=8, **kwargs):
        super().__init__(mesh, soil, **kwargs)
        self.c_pen_big = float(c_pen_big)      # penalty magnitude [1/day] -> forces psi~0 on ponded nodes
        self.pond_eps = float(pond_eps)        # surface-water threshold [m] to be "ponded"
        self.sat_eps = float(sat_eps)          # psi_top threshold [m] to declare a dry node saturated
        self.active_set_max = int(active_set_max)
        self.d_surf = np.zeros(self._n_dofs, dtype=np.float64)   # separate surface store (NOT in psi)
        self.clip_adjust = 0.0                 # cumulative surface-mass clipped (should stay ~0)
        self.last_active_iters = 0
        self.max_pond_psi = 0.0                # largest psi_top on a ponded node (should be ~0)
        self._as_hist: list[int] = []

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
        ents = np.concatenate(all_f); tags = np.concatenate(all_t)
        order = np.argsort(ents)
        ft = dmesh.meshtags(msh, fdim, ents[order], tags[order])
        ds = ufl.Measure("ds", domain=msh, subdomain_data=ft,
                         metadata={"quadrature_degree": self._quad_degree})
        ds_top_v = ufl.Measure("ds", domain=msh, subdomain_data=ft,
                               metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})(1)
        z = ufl.SpatialCoordinate(msh)[self._zaxis]
        # ★ the SWITCHING-BC surface coupling: (c_pen*psi - q_top) on the matched vertex measure.
        #   ponded node: c_pen=big, q_top=0 -> penalty forces psi->0; infiltration = -c_pen*psi (reaction).
        #   dry node:    c_pen=0, q_top=rain -> Neumann influx; infiltration = q_top (=rain) if psi stays<0.
        self._c_pen = fem.Function(rp.V, name="c_pen")
        self._q_top = fem.Function(rp.V, name="q_top")
        rp.F = rp.F + (self._c_pen * rp.psi - self._q_top) * rp._v * ds_top_v
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
            self._surf0 = 0.0
        self._built = True

    def surface_water(self) -> float:
        self._ensure_built()
        td = self._top_dofs_arr
        return float(np.sum(self.d_surf[td] * self._A_i[td]))

    def balance(self) -> float:
        self._ensure_built()
        dtotal = self.total_water() - (self._w0 + self._surf0)
        return dtotal - (self.cum_rain - self.cum_outflow - self.cum_drainage + self.clip_adjust)

    def _solve_with_modes(self, q_td, c_td, psi_entry):
        """Set per-node q_top/c_pen + solve. Returns (converged, it, infil[td], psi_top[td])."""
        rp = self._rp
        td = self._top_dofs_arr
        q = np.zeros(self._n_dofs); q[td] = q_td
        c = np.zeros(self._n_dofs); c[td] = c_td
        self._c_pen.x.array[:] = c; self._c_pen.x.scatter_forward()
        self._q_top.x.array[:] = q; self._q_top.x.scatter_forward()
        rp.psi.x.array[:] = psi_entry; rp.psi.x.scatter_forward()
        rp.psi_n.x.array[:] = psi_entry; rp.psi_n.x.scatter_forward()
        rp._ensure_problem(); rp._problem.solve()
        snes = rp._problem.solver
        reason = int(snes.getConvergedReason()); it = int(snes.getIterationNumber())
        fnorm = float(snes.getFunctionNorm())
        ok = reason > 0 and (reason != 4 or fnorm <= self.stall_accept_fnorm)
        psi_top = rp.psi.x.array[td].copy()
        infil = q_td - c_td * psi_top           # boundary flux per node [m/day] (matched vertex measure)
        return ok, it, infil, psi_top

    def step(self, dt: float):
        self._ensure_built()
        rp = self._rp
        rp.dt.value = dt
        td = self._top_dofs_arr
        A = self._A_i
        psi_entry = rp.psi.x.array.copy()
        d_entry = self.d_surf.copy()
        rain = float(self._rain_c.value)
        avail = self.d_surf[td] + rain * dt          # available surface water depth this step [m]

        # ACTIVE-SET iteration over 3 node modes (CATHY-style):
        #   0 DRY    : Neumann rain flux       (q_top=rain, c_pen=0)  -- unsaturated, takes the rain.
        #   1 PONDED : Dirichlet psi=0 penalty (q_top=0, c_pen=big)   -- saturated, infil = reaction flux.
        #   2 SUPPLY : Neumann avail/dt        (q_top=avail/dt)       -- ponded-but-thin, takes ALL it has.
        # Switches: DRY saturates (psi>0)->PONDED; PONDED over-draws (infil*dt>avail)->SUPPLY; SUPPLY
        # saturates->PONDED. With SUPPLY, d_new = avail - infil*dt >= 0 by construction (no over-draw).
        mode = np.where(self.d_surf[td] > self.pond_eps, 1, 0)
        ok = False; it = 0; infil = None; psi_top = None
        n_as = 0
        for _ in range(self.active_set_max):
            n_as += 1
            q_td = np.where(mode == 0, rain, np.where(mode == 2, avail / dt, 0.0))
            c_td = np.where(mode == 1, self.c_pen_big, 0.0)
            ok, it, infil, psi_top = self._solve_with_modes(q_td, c_td, psi_entry)
            if not ok:
                break
            new_mode = mode.copy()
            new_mode[(mode == 0) & (psi_top > self.sat_eps)] = 1            # dry saturated -> ponded
            new_mode[(mode == 1) & (infil * dt > avail + 1e-15)] = 2        # ponded over-draw -> supply
            new_mode[(mode == 2) & (psi_top > self.sat_eps)] = 1           # supply saturated -> ponded
            if np.array_equal(new_mode, mode):
                break
            mode = new_mode
        self._as_hist.append(n_as)
        self.last_active_iters = n_as
        if not ok:
            rp.psi.x.array[:] = psi_entry; rp.psi.x.scatter_forward()
            rp.psi_n.x.array[:] = psi_entry; rp.psi_n.x.scatter_forward()
            self.d_surf = d_entry
            return False, it
        pm = mode == 1
        if pm.any():
            self.max_pond_psi = max(self.max_pond_psi, float(np.max(psi_top[pm])))

        # Book the RAW boundary flux `infil` (= the weak-form surface flux = Delta(INT theta)/dt/A, so the
        # soil gain is EXACTLY SUM infil*A*dt -- do NOT clamp it, that would desync from Delta-theta). The
        # surface store then absorbs (rain - infil); a node whose ponded reaction over-draws (infil*dt >
        # avail, a thin film over hungry soil) would drive d<0 -> CLIP to 0 and book the clip into the
        # ledger (clip_adjust), so conservation stays exact. A large clip => a SUPPLY-limited active-set
        # mode is needed (v2); monitor it.
        d_new = self.d_surf.copy()
        d_new[td] = self.d_surf[td] + (rain - infil) * dt
        neg = d_new[td] < 0.0
        if neg.any():
            self.clip_adjust += float(np.sum(-d_new[td][neg] * A[td][neg]))
            d_new[td] = np.maximum(d_new[td], 0.0)
        # route the surface store downslope.
        d_routed, of = self._route(d_new, dt)
        self.d_surf = d_routed
        rp.psi_n.x.array[:] = rp.psi.x.array; rp.psi_n.x.scatter_forward()

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
        return True, it

    def active_iters_stats(self):
        h = np.asarray(self._as_hist, dtype=float)
        return dict(avg=float(h.mean()) if h.size else float("nan"),
                    mx=int(h.max()) if h.size else 0)


# ---- fixtures + runner ------------------------------------------------------------------------------
SOILS = {
    "loam": dict(theta_r=0.078, theta_s=0.43, alpha=3.6,  n=1.56, Ks=0.25),
    "sand": dict(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=1.5),
    "silt": dict(theta_r=0.067, theta_s=0.45, alpha=2.0,  n=1.41, Ks=0.10),
}
TARGET = {"loam": 0.5470, "sand": 0.6939, "silt": 0.7674}
GEOM = dict(Lx=8.0, Ly=5.0, Lz=1.0, PSI_I=-0.4, STORM=0.08, TEND=0.45, NMAN=0.05, nx=30, ny=20, nz=8)


def run(soil_name, S0=0.03, rain=0.5, dt_max=0.004, c_pen_big=2e4, t_end=None, p=None, nz=None):
    g = GEOM
    soil = VanGenuchten(**SOILS[soil_name])
    if p is not None:          # z-graded thin-skin mesh (CATHY-style surface refinement)
        msh = make_graded_box(g["nx"], g["ny"], nz or 12, g["Lx"], g["Ly"], g["Lz"], p=p)
    else:
        msh = make_box(g["nx"], g["ny"], g["nz"], g["Lx"], g["Ly"], g["Lz"])
    prob = SwitchingBCSplit(msh, soil, n_man=g["NMAN"], route_substeps=4, c_pen_big=c_pen_big)
    prob.set_initial_condition(lambda x: g["PSI_I"] + 0.0 * x[0])
    prob.set_topography(lambda x: S0 * x[1])
    rain_c = prob.add_rain(0.0)
    prob.add_outflow_bc(lambda x: np.isclose(x[1], 0.0), slope=S0)
    top_area = _top_area_ds(msh, g["Lz"])
    R_in = rain * top_area * g["STORM"]
    te = t_end if t_end is not None else g["TEND"]
    t0 = time.perf_counter()
    ns, coll, tend, _ = _march(prob, rain_c, storm_dur=g["STORM"], storm_rain=rain, t_end=te,
                               dt0=dt_max / 4.0, dt_max=dt_max, max_steps=900)
    ok = (not coll) and tend >= te - 1e-9
    st = prob.active_iters_stats()
    return dict(routed=prob.cum_outflow / R_in, bal=abs(prob.balance()) / prob.cum_rain,
                ns=ns, ok=ok, wall=time.perf_counter() - t0, as_avg=st["avg"], as_mx=st["mx"],
                max_pond_psi=prob.max_pond_psi, clip=prob.clip_adjust)


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("#" * 92)
    print("SWITCHING-BC split -- canonical Neumann<->Dirichlet active-set closure (loam, S=0.03)")
    print("#" * 92, flush=True)
    r = run("loam")
    gp = r["routed"] - TARGET["loam"]
    print(f"  loam: routed/R={r['routed']:.4f} vs monolith {TARGET['loam']:.4f} (gap {gp*100:+.1f}pp)  "
          f"bal/rain={r['bal']:.2e}  max_pond_psi={r['max_pond_psi']*1000:.3f}mm clip={r['clip']:.1e}")
    print(f"     ok={r['ok']} ns={r['ns']} active-set avg={r['as_avg']:.1f}/max={r['as_mx']} "
          f"wall={r['wall']:.0f}s  => {'STABLE' if r['ok'] else 'COLLAPSED'}", flush=True)
    print("#" * 92, flush=True)


if __name__ == "__main__":
    main()
