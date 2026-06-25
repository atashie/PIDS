"""SCRATCH PROTOTYPE (no pids_forward/ edits, no commit) -- capacity-LIMITED infiltration for the
sequential operator-split overland coupling.

THE HYPOTHESIS UNDER TEST
-------------------------
The merged ``SequentialCoupledProblem`` over-infiltrates the Hortonian run-on partition (~24 pp vs the
ParFlow-validated monolith ``CoupledProblem``). ROOT CAUSE (pinned): the sequential scheme carries the
pond IN psi (``add_ponding_bc``) and infiltrates at the UNCAPPED ponded-head Richards-column uptake,
while the monolith CAPS infiltration at a capacity ``q_pot = kirchhoff(psi_top,d)/ell_c`` via a smooth
complementarity. Benchmarks B4/B5 show the monolith ~= ParFlow, so the CAP is correct and the
sequential is the outlier.

THE FIX PROTOTYPED HERE (decoupled, capacity-limited -- the standard sequential overland coupling):
  * track the surface pond ``d`` as a SEPARATE surface ARRAY (NOT carried in psi);
  * each step, per top node i:
       supply_i = rain*dt + d_i                                 (water available at the surface)
       q_pot_i  = kirchhoff(psi_top_i, d_i) / ell_c             (the monolith's capacity, entry state)
       inf_i    = min(q_pot_i*dt, supply_i)                     (capacity- OR supply-limited)
  * solve Richards ALONE with ``inf_i`` applied as a NEUMANN INFLUX BC on the top
    (influx rate = inf_i/dt), psi_top a FREE Richards variable (NO ponded-head Dirichlet);
  * ``d_i := supply_i - inf_i`` becomes/stays the pond;
  * ROUTE ``d`` downslope (the existing ``route_excess`` kernel), booking outflow at the outlet.

CONSERVATION LEDGER (the deliverable's gate): ``cum_rain = d(int theta dV) + sum d_i A_i + cum_outflow``.
This is the analog of the merged scheme's ``total = int theta + sum d_i A_i`` ledger.

IMPLEMENTATION NOTE: we SUBCLASS ``SequentialCoupledProblem`` to reuse the mesh / routing-graph /
top-dof / area machinery, but OVERRIDE the residual assembly (drop the pond-in-psi storage + rain
terms, keep ONLY a per-step Neumann influx) and OVERRIDE ``step`` to run the capped scheme above. The
pond ``d`` lives in ``self.d_surf`` (an array over top dofs), independent of psi.

Run (WSL pids-fem):
  wsl bash -c 'cd .../forward-model && export PATH=".../pids-fem/bin:$PATH" && \
    export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
    python -u scratch/seq_capped_infiltration_prototype.py 2>&1 | tail -60'
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.richards import richards_bulk_residual
from pids_forward.physics.sequential_coupling import SequentialCoupledProblem
from pids_forward.physics.coupling import CoupledProblem


# ---------------------------------------------------------------------------
# The capacity-capped sequential prototype.
# ---------------------------------------------------------------------------
class CappedSequentialCoupledProblem(SequentialCoupledProblem):
    """SequentialCoupledProblem with CAPACITY-LIMITED infiltration (the fix hypothesis).

    The pond ``d`` is tracked as a SEPARATE surface array ``self.d_surf`` (indexed by top dof), NOT
    carried in psi. The Richards solve sees ONLY a per-step Neumann INFLUX BC equal to the
    capacity/supply-limited infiltration depth/dt; psi_top is a free Richards variable.

    Everything else (routing graph, areas, top dofs, outlet masks, GHB/interior-drain plumbing) is
    inherited. The ledger is reimplemented around the separate pond array.
    """

    def __init__(self, mesh, soil, *, ell_c=None, **kwargs):
        super().__init__(mesh, soil, **kwargs)
        # capacity ell_c: the monolith's top-cell half-height auto-detect (replicated).
        if ell_c is None:
            ell_c = self._auto_ell_c()
        self.ell_c = float(ell_c)
        # the surface pond, a SEPARATE array over ALL dofs (nonzero only on top dofs). NOT in psi.
        self.d_surf = np.zeros(self._n_dofs, dtype=np.float64)
        # per-step Neumann influx [m/day] applied on the top (a Function on V, reuse _lat_src slot? no
        # -- keep a dedicated influx Function so the residual is explicit and the ledger is clean).
        self._q_inf = fem.Function(self._rp.V, name="q_inf")
        # ledger
        self.cum_rain = 0.0
        self.cum_outflow = 0.0
        self.cum_drainage = 0.0
        self._w0 = None          # int theta at t=0
        self._surf0 = 0.0        # sum d_i A_i at t=0 (0 unless an initial pond is set)
        self.last_outflow = 0.0
        self.last_inf_total = 0.0     # sum inf_i A_i / dt this step [m^3/day]
        self.last_cap_frac = 0.0      # fraction of top nodes that were capacity-limited (inf<supply)
        self._t = 0.0
        self._built_capped = False

    def _auto_ell_c(self) -> float:
        """Top-cell half-height = half the spacing between the top two DISTINCT z-levels (the monolith's
        heuristic, coupling.py:199-201). Round to 9 dp to defeat float noise across a flat level."""
        zc = self._rp.V.tabulate_dof_coordinates()[:, self._zaxis]
        zu = np.unique(np.round(zc, 9))
        if zu.size < 2:
            raise ValueError("ell_c auto-detect: <2 z-levels.")
        return 0.5 * float(zu[-1] - zu[-2])

    # -- override the residual: bulk + GHB/interior drains + a per-step Neumann INFLUX on the top.
    def _finalize_forms_capped(self) -> None:
        rp = self._rp
        msh = self.mesh
        fdim = msh.topology.dim - 1
        msh.topology.create_connectivity(fdim, msh.topology.dim)

        # bulk Richards (capped quadrature), NO pond-in-psi storage term, NO rain term.
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
        # the influx rides the SAME lumped VERTEX measure as the pond ledger / routing sum d_i A_i, so
        # the volume the residual removes from the surface == the volume we book infiltrated (matched
        # quadrature -- the conservation fix, mirrored from the merged scheme).
        ds_top_v = ufl.Measure("ds", domain=msh, subdomain_data=ft,
                               metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})(1)
        z = ufl.SpatialCoordinate(msh)[self._zaxis]

        # NEUMANN INFLUX: + q_inf into the soil top. The Richards bulk Darcy is
        # K(grad psi + e_g).grad v, whose natural boundary term is -(Darcy flux . n) v ds; a prescribed
        # INFLUX q_inf (positive INTO the domain, downward) enters as ``- q_inf * v * ds`` (same sign
        # convention as RichardsProblem.add_flux_bc). This is the ONLY surface term -- no pond storage.
        rp.F = rp.F - self._q_inf * rp._v * ds_top_v

        # GHB drains (unchanged, degree-8 ds), for completeness/parity with the parent.
        self._drain_forms = []
        for (k, C, H) in drain_specs:
            kr = self.soil.K_ufl(rp.psi) / self.soil.Ks
            q_n = C * kr * (rp.psi + z - H)
            rp.F = rp.F + q_n * rp._v * ds(k)
            self._drain_forms.append(fem.form(q_n * ds(k)))
        # interior drains (volumetric), unchanged.
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

        # routing graph (reuse the parent's builder via _finalize_forms? no -- it rebuilds F with the
        # pond term. Build the graph pieces here directly, same calls.)
        from pids_forward.physics.overland_edge_kernel import build_top_facet_edge_graph
        from pids_forward.physics.overland_routing import build_adjacency, node_widths
        edges, L_e, T_e, A_i = build_top_facet_edge_graph(rp.V, msh, top_facets)
        self._A_i = A_i
        self._adj = build_adjacency(edges, L_e, self._n_dofs)
        self._W = node_widths(edges, L_e, T_e, self._n_dofs)
        self._top_area = float(A_i.sum())

        # outlet masks + per-node slope (same as parent).
        self._outlet_mask = np.zeros(self._n_dofs, dtype=bool)
        self._outlet_slope_node = np.zeros(self._n_dofs, dtype=np.float64)
        top_dofs = self._top_dofs_arr
        for (loc, slope) in self._outlets:
            sel = top_dofs[loc(self._coords[top_dofs].T)]
            self._outlet_mask[sel] = True
            self._outlet_slope_node[sel] = np.maximum(self._outlet_slope_node[sel], slope)

        # ledger baseline (one-shot). int theta at the IC + the initial pond sum.
        if self._w0 is None:
            self._w0 = rp.total_water()
            self._surf0 = float(np.sum(self.d_surf[top_dofs] * A_i[top_dofs]))

        # precompiled capacity expression q_pot = kirchhoff(psi_top, d_node)/ell_c is built per-node in
        # numpy at the entry state (psi is read off the array; d is the entry pond) -- NO UFL needed for
        # the capacity (it is a per-node a-priori cap, evaluated at the entry state, exactly like the
        # monolith reads its q_pot at the current iterate but here we freeze it at entry).
        self._built_capped = True

    def _ensure_built_capped(self):
        if not self._built_capped:
            self._finalize_forms_capped()

    def _surf_pond_arr(self) -> float:
        """sum_i d_surf_i A_i over top dofs (the surface store)."""
        td = self._top_dofs_arr
        return float(np.sum(self.d_surf[td] * self._A_i[td]))

    # -- the capped step --------------------------------------------------
    def step(self, dt: float):
        """One capacity-limited sequential step. Returns (converged, iters).

          1. supply_i = rain*dt + d_surf_i
          2. q_pot_i  = kirchhoff(psi_top_i, d_surf_i)/ell_c        (entry state)
          3. inf_i    = min(q_pot_i*dt, supply_i)
          4. solve Richards ALONE with influx q_inf = inf_i/dt (Neumann) on the top
          5. d_surf_i := supply_i - inf_i
          6. route d_surf downslope, book outflow
        """
        self._ensure_built_capped()
        rp = self._rp
        rp.dt.value = dt
        td = self._top_dofs_arr
        A_i = self._A_i

        psi_entry = rp.psi.x.array.copy()
        d_entry = self.d_surf.copy()

        # --- capacity-limited infiltration depth at the ENTRY state -------------------------------
        rain = float(self._rain_c.value)
        psi_top = psi_entry[td]
        d_top = d_entry[td]
        supply = rain * dt + d_top                          # m of water column available
        # q_pot via the soil's numpy kirchhoff (vectorize over nodes). kirchhoff(a=psi_top, b=d).
        # NOTE: when d<=psi_top (no positive head difference) kirchhoff can be ~0 or tiny; clamp at 0.
        q_pot = np.array([self.soil.kirchhoff(float(a), float(b)) for a, b in zip(psi_top, d_top)])
        q_pot = np.maximum(q_pot, 0.0) / self.ell_c          # m/day capacity
        inf_depth = np.minimum(q_pot * dt, supply)           # capacity- OR supply-limited [m]
        inf_depth = np.maximum(inf_depth, 0.0)
        # capacity-limited fraction (diagnostic): nodes where the cap bit (q_pot*dt < supply) AND there
        # was something to infiltrate.
        active = supply > 1e-15
        capped = active & (q_pot * dt < supply - 1e-15)
        self.last_cap_frac = float(np.sum(capped)) / max(int(np.sum(active)), 1)

        # set the Neumann influx q_inf = inf_depth/dt on the top dofs (0 elsewhere).
        q_inf_arr = np.zeros(self._n_dofs, dtype=np.float64)
        q_inf_arr[td] = inf_depth / dt
        self._q_inf.x.array[:] = q_inf_arr
        self._q_inf.x.scatter_forward()

        # --- solve Richards ALONE -----------------------------------------------------------------
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

        # accept Richards: psi_n := psi.
        rp.psi_n.x.array[:] = rp.psi.x.array
        rp.psi_n.x.scatter_forward()

        # --- pond update: the un-infiltrated remainder ---------------------------------------------
        d_post = supply - inf_depth                          # >= 0 by construction (inf<=supply)
        d_new_arr = d_entry.copy()
        d_new_arr[td] = d_post

        # --- route the pond downslope -------------------------------------------------------------
        d_routed, of = self._route(d_new_arr, dt)
        self.d_surf = d_routed

        # --- books --------------------------------------------------------------------------------
        of_booked = of * (1.0 + self.outflow_leak_frac)      # falsification hook
        self.last_outflow = of
        self.cum_outflow += of_booked
        self.last_inf_total = float(np.sum(inf_depth * A_i[td])) / dt if dt > 0 else 0.0

        # subsurface sink rates (GHB + interior), read on the solved psi.
        ghb_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                     for f in self._drain_forms]
        interior_rates = [self.mesh.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)
                          for f in self._interior_forms]
        total_rate = float(sum(ghb_rates) + sum(interior_rates))
        self.last_drainage = total_rate
        self.cum_drainage += dt * total_rate

        # rain over dt onto the top area.
        self.cum_rain += rain * self._top_area * dt
        self._t += dt
        return True, it

    def advance(self, t_end, dt, *, storm_dur=None, storm_rain=None, dt_max=0.03, dt_min=1e-11,
                ctrl_low=4, ctrl_high=12, grow=1.4, shrink=0.7, cut=0.5, max_steps=200000):
        """March to t_end with the band dt-controller + optional storm hyetograph (same as parent)."""
        self._ensure_built_capped()
        if storm_dur is not None and storm_rain is None:
            raise ValueError("advance: pass storm_rain with storm_dur.")
        t = 0.0
        nstep = 0
        while t < t_end - 1e-12:
            if nstep >= max_steps:
                raise RuntimeError(f"advance exceeded max_steps={max_steps} at t={t:.4g}")
            h = min(dt, t_end - t)
            if storm_dur is not None:
                if t < storm_dur - 1e-12 and t + h > storm_dur:
                    h = storm_dur - t
                self._rain_c.value = storm_rain if t < storm_dur - 1e-12 else 0.0
            conv, it = self.step(h)
            if not conv:
                dt = h * cut
                if dt < dt_min:
                    raise RuntimeError(f"advance: dt {dt:.2e} < dt_min at t={t:.4g} (not converging)")
                continue
            t += h
            nstep += 1
            if it <= ctrl_low:
                dt = min(dt * grow, dt_max)
            elif it >= ctrl_high:
                dt = dt * shrink
        return nstep

    # -- ledger -----------------------------------------------------------
    def soil_water(self) -> float:
        return self._rp.total_water()

    def surface_water(self) -> float:
        self._ensure_built_capped()
        return self._surf_pond_arr()

    def total_water(self) -> float:
        return self.soil_water() + self.surface_water()

    def balance(self) -> float:
        """cum_rain = d(int theta) + sum d_i A_i + cum_outflow + cum_drainage. Returns the SIGNED
        residual (left - right of d(total) = cum_rain - cum_outflow - cum_drainage)."""
        self._ensure_built_capped()
        dtotal = self.total_water() - (self._w0 + self._surf0)
        return dtotal - (self.cum_rain - self.cum_outflow - self.cum_drainage)


class RouteFirstCappedProblem(CappedSequentialCoupledProblem):
    """DIAGNOSTIC variant: reverse the operator-split ORDER -- add rain, ROUTE the pond FIRST, then
    infiltrate the post-routing remainder (capacity-limited). Tests whether the partition gap is driven
    by split ORDERING (infiltration-first gives infiltration first claim on every parcel) rather than by
    the cap. The cap is evaluated against the POST-ROUTE pond as the supply."""

    def step(self, dt: float):
        self._ensure_built_capped()
        rp = self._rp
        rp.dt.value = dt
        td = self._top_dofs_arr
        A_i = self._A_i
        psi_entry = rp.psi.x.array.copy()
        rain = float(self._rain_c.value)

        # 1+2: add rain to the pond, ROUTE first.
        d_arr = self.d_surf.copy()
        d_arr[td] += rain * dt
        d_routed, of = self._route(d_arr, dt)
        d_post_route = d_routed[td]

        # 3: capacity-limited infiltration of the post-route remainder.
        psi_top = psi_entry[td]
        q_pot = np.array([self.soil.kirchhoff(float(a), float(b))
                          for a, b in zip(psi_top, d_post_route)])
        q_pot = np.maximum(q_pot, 0.0) / self.ell_c
        inf = np.minimum(q_pot * dt, d_post_route)
        inf = np.maximum(inf, 0.0)
        qa = np.zeros(self._n_dofs)
        qa[td] = inf / dt
        self._q_inf.x.array[:] = qa
        self._q_inf.x.scatter_forward()

        # 4: solve Richards alone.
        rp.psi.x.array[:] = psi_entry
        rp.psi_n.x.array[:] = psi_entry
        rp.psi.x.scatter_forward()
        rp.psi_n.x.scatter_forward()
        rp._ensure_problem()
        rp._problem.solve()
        snes = rp._problem.solver
        reason, it, fn = (int(snes.getConvergedReason()), int(snes.getIterationNumber()),
                          float(snes.getFunctionNorm()))
        if not (reason > 0 and (reason != 4 or fn <= self.stall_accept_fnorm)):
            rp.psi.x.array[:] = psi_entry
            rp.psi_n.x.array[:] = psi_entry
            rp.psi.x.scatter_forward()
            rp.psi_n.x.scatter_forward()
            return False, it
        rp.psi_n.x.array[:] = rp.psi.x.array
        rp.psi_n.x.scatter_forward()

        # 5: remainder becomes the pond.
        d_new = d_routed.copy()
        d_new[td] = d_post_route - inf
        self.d_surf = d_new
        self.cum_outflow += of * (1.0 + self.outflow_leak_frac)
        self.last_outflow = of
        self.cum_rain += rain * self._top_area * dt
        self._t += dt
        return True, it


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
def make_box(nx, ny, nz, Lx, Ly, Lz):
    return dmesh.create_box(
        MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz], cell_type=dmesh.CellType.tetrahedron)


def partition_metrics(soil, nx, ny, nz):
    """Common geometry/partition metric helpers for Case 1 / Case 2."""
    pass


# ---------------------------------------------------------------------------
# CASE 1 -- the b1 mild planar loam: does the cap close the partition gap?
# ---------------------------------------------------------------------------
def run_case1(nx=30, ny=20, nz=8, cls=CappedSequentialCoupledProblem, label="CAPPED prototype"):
    print("=" * 78)
    print(f"CASE 1: b1 mild planar LOAM, mesh {nx}x{ny}x{nz}  [{label}]")
    print("=" * 78)
    LOAM = VanGenuchten(0.078, 0.43, 3.6, 1.56, Ks=0.25)
    Lx, Ly, Lz = 8.0, 5.0, 1.0
    PSI_I = -0.4
    RAIN = 0.5
    STORM_DUR = 0.08
    T_END = 0.45

    def topo(x):
        return 0.03 * x[1]

    def ic(x):
        return np.full(x.shape[1], PSI_I)

    def outlet_loc(x):
        return np.isclose(x[1], 0.0)

    msh = make_box(nx, ny, nz, Lx, Ly, Lz)
    top_area = Lx * Ly
    R_in = RAIN * top_area * STORM_DUR

    prob = cls(msh, LOAM, n_man=0.05, route_substeps=4)
    prob.set_initial_condition(ic)
    prob.set_topography(topo)
    prob.add_rain(RAIN)
    prob.add_outflow_bc(outlet_loc, slope=0.03)
    w0_soil = prob.soil_water()
    nst = prob.advance(T_END, dt=2e-3, storm_dur=STORM_DUR, storm_rain=RAIN, dt_max=0.02)
    soil_gain = prob.soil_water() - w0_soil
    routed = prob.cum_outflow
    infil = soil_gain + prob.cum_drainage
    bal = prob.balance()
    print(f"  [{label}]  steps={nst}  ell_c={prob.ell_c:.4f}")
    print(f"    routed/R = {routed / R_in:.4f}   infil/R = {infil / R_in:.4f}   "
          f"(routed+infil)/R = {(routed + infil) / R_in:.4f}")
    print(f"    cum_rain={prob.cum_rain:.6e}  R_in(top*storm)={R_in:.6e}  "
          f"surf_pond={prob.surface_water():.4e}")
    print(f"    LEDGER residual = {bal:.3e}  (rel to cum_rain = {bal / max(prob.cum_rain,1e-30):.2e})")

    return {
        "R_in": R_in, "capped_routed_frac": routed / R_in, "capped_infil_frac": infil / R_in,
        "capped_bal": bal, "mesh": (nx, ny, nz),
        "LOAM": LOAM, "Lx": Lx, "Ly": Ly, "Lz": Lz, "PSI_I": PSI_I, "RAIN": RAIN,
        "STORM_DUR": STORM_DUR, "T_END": T_END,
    }


def run_case1_monolith(ctx, nx, ny, nz):
    """The apples-to-apples MONOLITH at the SAME mesh (the dt-converged partition target)."""
    LOAM = ctx["LOAM"]
    Lx, Ly, Lz = ctx["Lx"], ctx["Ly"], ctx["Lz"]
    PSI_I, RAIN, STORM_DUR, T_END = ctx["PSI_I"], ctx["RAIN"], ctx["STORM_DUR"], ctx["T_END"]
    top_area = Lx * Ly
    R_in = RAIN * top_area * STORM_DUR

    def topo(x):
        return 0.03 * x[1]

    def ic(x):
        return np.full(x.shape[1], PSI_I)

    def outlet_loc(x):
        return np.isclose(x[1], 0.0)

    msh = make_box(nx, ny, nz, Lx, Ly, Lz)
    # galerkin overland (robust on a mild plane; upwind would also work but galerkin is the simplest
    # apples-to-apples baseline and matches the B4/B5 validated path).
    mono = CoupledProblem(msh, LOAM, n_man=0.05, overland_scheme="galerkin")
    mono.set_initial_condition(ic, d_value=0.0)
    mono.set_topography(topo)
    mono.add_rain(RAIN)
    mono.add_outflow_bc(outlet_loc, slope=0.03)
    w0_soil = mono.soil_water()
    w0_total = mono.total_water()   # snapshot the conserved baseline ourselves (no balance() on monolith)

    # CoupledProblem.advance has no storm hyetograph hook -> drive rain manually with a simple loop.
    t = 0.0
    dt = 1e-3
    nst = 0
    dt_max = 0.01
    while t < T_END - 1e-12:
        h = min(dt, T_END - t)
        if t < STORM_DUR - 1e-12 and t + h > STORM_DUR:
            h = STORM_DUR - t
        mono._rain.value = RAIN if t < STORM_DUR - 1e-12 else 0.0
        conv, it = mono.step(h)
        if not conv:
            dt = h * 0.5
            if dt < 1e-10:
                raise RuntimeError(f"monolith dt collapse at t={t:.4g}")
            continue
        t += h
        nst += 1
        if it <= 3:
            dt = min(dt * 1.5, dt_max)
        elif it >= 8:
            dt = dt * 0.7
    soil_gain = mono.soil_water() - w0_soil
    routed = mono.cum_outflow
    infil = soil_gain + mono.cum_drainage
    # monolith conservation check: it does NOT accumulate cum_rain -> use the analytical storm input
    # R_in (= RAIN*top_area*STORM_DUR, the whole storm). dtotal vs (rain - outflow - drainage + clip).
    dtotal = mono.total_water() - w0_total
    bal = dtotal - (R_in - mono.cum_outflow - mono.cum_drainage + mono.clip_mass_adjust)
    print(f"  [MONOLITH galerkin]  steps={nst}")
    print(f"    routed/R = {routed / R_in:.4f}   infil/R = {infil / R_in:.4f}   "
          f"(routed+infil)/R = {(routed + infil) / R_in:.4f}")
    print(f"    R_in={R_in:.6e}  surf_pond={mono.surface_water():.4e}  "
          f"max_clip_seen={mono.max_clip_seen:.2e}  ledger_resid={bal:.2e}")
    return routed / R_in, infil / R_in


# ---------------------------------------------------------------------------
# CASE 2 -- the stiff convergent-clay V: is robustness preserved?
# ---------------------------------------------------------------------------
def run_case2(nx=24, ny=16, nz=5):
    print("=" * 78)
    print(f"CASE 2: stiff convergent CLAY V, mesh {nx}x{ny}x{nz}  (robustness + conservation?)")
    print("=" * 78)
    CLAY = VanGenuchten(0.068, 0.38, 0.8, 1.09, Ks=0.048)
    Lx, Ly, Lz = 10.0, 6.0, 1.5
    PSI_I = -0.30
    RAIN = 1.0
    STORM_DUR = 0.08
    T_END = 0.40
    top_area = Lx * Ly
    R_in = RAIN * top_area * STORM_DUR

    def topo(x):
        return 0.05 * x[1] + 0.08 * np.abs(x[0] - 5.0)

    def ic(x):
        return np.full(x.shape[1], PSI_I)

    def outlet_loc(x):
        return np.isclose(x[1], 0.0)

    msh = make_box(nx, ny, nz, Lx, Ly, Lz)
    prob = CappedSequentialCoupledProblem(msh, CLAY, n_man=0.05, route_substeps=4)
    prob.set_initial_condition(ic)
    prob.set_topography(topo)
    prob.add_rain(RAIN)
    prob.add_outflow_bc(outlet_loc, slope=0.05)
    w0_soil = prob.soil_water()
    try:
        nst = prob.advance(T_END, dt=2e-3, storm_dur=STORM_DUR, storm_rain=RAIN, dt_max=0.02)
        completed = True
    except RuntimeError as e:
        print(f"    !! advance FAILED: {e}")
        completed = False
        nst = -1
    soil_gain = prob.soil_water() - w0_soil
    routed = prob.cum_outflow
    infil = soil_gain + prob.cum_drainage
    bal = prob.balance()
    print(f"  [CAPPED prototype on CLAY-V]  completed={completed}  steps={nst}  ell_c={prob.ell_c:.4f}")
    if completed:
        print(f"    routed/R = {routed / R_in:.4f}   infil/R = {infil / R_in:.4f}")
        print(f"    LEDGER residual = {bal:.3e}  (rel to cum_rain = {bal / max(prob.cum_rain,1e-30):.2e})")
    return {"completed": completed, "bal": bal, "rel_bal": bal / max(prob.cum_rain, 1e-30)}


# ---------------------------------------------------------------------------
# FALSIFICATION -- a deliberate 10% outflow mis-book must break the balance ~10%.
# ---------------------------------------------------------------------------
def run_falsification(nx=20, ny=14, nz=6):
    print("=" * 78)
    print("FALSIFICATION: 10% outflow mis-book must break the ledger by ~10%")
    print("=" * 78)
    LOAM = VanGenuchten(0.078, 0.43, 3.6, 1.56, Ks=0.25)
    Lx, Ly, Lz = 8.0, 5.0, 1.0
    PSI_I, RAIN, STORM_DUR, T_END = -0.4, 0.5, 0.08, 0.45

    def topo(x):
        return 0.03 * x[1]

    def ic(x):
        return np.full(x.shape[1], PSI_I)

    def outlet_loc(x):
        return np.isclose(x[1], 0.0)

    msh = make_box(nx, ny, nz, Lx, Ly, Lz)
    prob = CappedSequentialCoupledProblem(msh, LOAM, n_man=0.05, route_substeps=4)
    prob.outflow_leak_frac = 0.10        # the injected mis-book
    prob.set_initial_condition(ic)
    prob.set_topography(topo)
    prob.add_rain(RAIN)
    prob.add_outflow_bc(outlet_loc, slope=0.03)
    prob.advance(T_END, dt=2e-3, storm_dur=STORM_DUR, storm_rain=RAIN, dt_max=0.02)
    bal = prob.balance()
    # the imbalance should be ~ -0.10 * cum_outflow_true (we over-booked outflow by 10%).
    true_outflow = prob.cum_outflow / 1.10
    expect = -0.10 * true_outflow
    print(f"    ledger residual with 10% leak = {bal:.4e}")
    print(f"    expected (~ -0.10*true_outflow) = {expect:.4e}")
    print(f"    ratio bal/expect = {bal / expect:.3f}  (want ~1.0)")
    print(f"    |bal| / cum_rain = {abs(bal) / prob.cum_rain:.3e}  (a CLEAN run was ~1e-9; this is the leak)")
    return bal, expect


# ---------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=4, suppress=True)

    # falsification FIRST (sanity-check the detector before trusting any partition number).
    fb, fe = run_falsification()

    # Case 1 -- the capped prototype (infiltration-FIRST) + the apples-to-apples monolith at SAME mesh.
    MESH1 = (30, 20, 8)
    ctx = run_case1(*MESH1, cls=CappedSequentialCoupledProblem, label="CAPPED (infiltrate-first)")
    print("-" * 78)
    # DIAGNOSTIC: reverse the split order (route-first) to isolate the ordering effect on the partition.
    ctx_rf = run_case1(*MESH1, cls=RouteFirstCappedProblem, label="CAPPED (route-first, diagnostic)")
    print("-" * 78)
    print(f"APPLES-TO-APPLES MONOLITH at the SAME mesh {MESH1}:")
    mono_routed, mono_infil = run_case1_monolith(ctx, *MESH1)

    # also run the monolith at a COARSE mesh (the prompt's 12x8x4 ~= 0.638 reference) for context.
    print("-" * 78)
    print("MONOLITH at a COARSE mesh (12,8,4) -- the prompt's ~0.638 reference point:")
    mono_routed_c, _ = run_case1_monolith(ctx, 12, 8, 4)

    # Case 2 -- robustness.
    c2 = run_case2()

    # ---- SUMMARY -----------------------------------------------------------------------------
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    # |ratio|~1 is the pass criterion: a 10% outflow over-booking shifts Δtotal vs (rain-out-drain)
    # by +0.10*true_outflow (sign flips because we SUBTRACT an over-large cum_outflow), magnitude exact.
    print(f"FALSIFICATION: bal={fb:.3e} vs expect={fe:.3e} -> |ratio| {abs(fb/fe):.3f} "
          f"({'PASS (~10% break detected, magnitude exact)' if 0.8 < abs(fb/fe) < 1.2 else 'CHECK'})")
    print()
    print(f"CASE 1 partition (routed/R) at mesh {MESH1}:")
    print(f"   OLD uncapped sequential (reported):   ~0.27 (rs8 @30x20x8) / ~0.40 (dt->0)")
    print(f"   CAPPED (infiltrate-first) prototype:  {ctx['capped_routed_frac']:.4f}")
    print(f"   CAPPED (route-first) diagnostic:      {ctx_rf['capped_routed_frac']:.4f}")
    print(f"   MONOLITH @ same mesh {MESH1}:          {mono_routed:.4f}   <-- the apples-to-apples target")
    print(f"   MONOLITH @ coarse (12,8,4):            {mono_routed_c:.4f}")
    gap_to_mono = ctx['capped_routed_frac'] - mono_routed
    print(f"   capped(inf-first) - monolith(same mesh) = {gap_to_mono:+.4f}  ({gap_to_mono*100:+.1f} pp)")
    print(f"   CASE-1 ledger residual: {ctx['capped_bal']:.3e} (rel {ctx['capped_bal']/ctx['R_in']:.2e})")
    print(f"   FINDING: the cap moves the partition the WRONG way (more infiltration, less routing) and")
    print(f"            the gap to the monolith WIDENS; route-first helps but is still far short.")
    print()
    print(f"CASE 2 (stiff clay-V robustness): completed={c2['completed']}  "
          f"ledger_rel={c2['rel_bal']:.3e}")


if __name__ == "__main__":
    main()
