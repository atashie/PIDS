"""Embedded 1-D-in-3-D PIDS feature primitive (Module 4 §E), realization A -- CO-LOCATED.

A PIDS feature is a 1-D vector Γ (a tagged set of host EDGES) embedded in the 3-D Richards host. It
carries its own head ``H_f`` as a P1 field on the HOST function space, pinned to 0 off Γ, and every
feature term is an integral on the interior ``ridge`` (codim-2 edge) measure ``dΓ`` over the tagged
edges -- single-mesh, FFCX-native on stock DOLFINx 0.10. The Phase-0 spike (scratch/m4_embedding_spike
{,2}.py) conclusively showed the design-intended 1-D-submesh + ``entity_maps`` coupling is FFCX-blocked
(both the codim-2 cross-mesh ridge path and the submesh ``dx_f`` path fail), so the co-located ridge is
the working route; physics/tests are realization-agnostic (migrate to the DOF-efficient submesh when
FFCX supports codim-2 cross-mesh).

CONVEYANCE is the 1-D Darcy current along the centerline using the TANGENTIAL projection ``∇H_f·t̂``
(``t̂`` = the known feature tangent). The full ``∇·∇`` is cell-trace-dependent and wrong (Phase-0 spike:
~10x off; tangential is exact to 6e-16). Storage and the sign-paired sorptive exchange are added on top.

Design: docs/plans/2026-06-04-pids-forward-model-architecture-design.md §E +
docs/plans/2026-06-08-module4-features-plan.md.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
import dolfinx.fem.petsc as fp
from mpi4py import MPI
from petsc4py import PETSc

from .sorptive_closure import (
    F_cylindrical,
    F_throttle,
    des_sorp_ratio,
    parlange_sorptivity,
    throttle_params,
    R_W_DEFAULT,
)


class EmbeddedFeature:
    """A single embedded feature Γ on a host mesh: ``H_f`` co-located on the host, pinned off Γ.

    ``locator`` selects the feature edges (codim-2 in 3-D, codim-1 in 2-D); ``tangent`` is the unit
    feature direction ``t̂`` for the conveyance projection. ``K_feat`` (granular conductivity), ``area``
    (cross-section), ``porosity``, ``sigma`` (soil-exchange coefficient) are the feature parameters.
    """

    def __init__(self, mesh, locator, tangent, *, K_feat, area, porosity, sigma=0.0, degree: int = 1):
        self.mesh = mesh
        self.K_feat = float(K_feat)
        self.area = float(area)
        self.porosity = float(porosity)
        self.sigma = float(sigma)

        # P1-only: the vertex dP diagonal allocation (below) covers only vertex dofs; degree>1 would pin
        # off-Γ edge/face dofs without a structural diagonal (Codex review 2026-06-08).
        if int(degree) != 1:
            raise ValueError(f"EmbeddedFeature is P1-only (the dP pin allocates vertex dofs); got degree={degree}.")
        # the conveyance form scales as |t̂|² -> a non-unit tangent silently distorts the current. Validate
        # the shape against gdim and NORMALIZE (Codex review 2026-06-08).
        gdim = mesh.geometry.dim
        t = np.asarray([float(c) for c in tangent], dtype=float)
        tnorm = float(np.linalg.norm(t))
        if t.shape != (gdim,) or tnorm < 1e-14:
            raise ValueError(f"tangent must be a non-zero length-{gdim} vector; got {tangent!r}.")
        t = t / tnorm
        self.t_hat = ufl.as_vector([float(c) for c in t])

        tdim = mesh.topology.dim
        self._edim = tdim - 2  # the feature lives on codim-2 entities (edges in 3-D)
        for d in (0, self._edim):
            mesh.topology.create_connectivity(d, tdim)
            mesh.topology.create_connectivity(tdim, d)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        self.feat_edges = np.sort(dmesh.locate_entities(mesh, self._edim, locator)).astype(np.int32)
        if mesh.comm.allreduce(int(self.feat_edges.size), op=MPI.SUM) == 0:
            raise ValueError("EmbeddedFeature: the locator selected no feature edges (empty Γ).")
        # reject feature edges on the domain boundary: the INTERIOR ridge measure silently omits / zero-
        # measures them, corrupting length/storage/exchange/conveyance without an error (Codex review).
        bdry = dmesh.locate_entities_boundary(
            mesh, self._edim, lambda x: np.ones(x.shape[1], dtype=bool))
        if mesh.comm.allreduce(int(np.intersect1d(self.feat_edges, bdry).size), op=MPI.SUM) > 0:
            raise NotImplementedError("EmbeddedFeature: feature edges on the domain boundary are not "
                                      "supported (the interior ridge measure mis-measures them).")
        ft = dmesh.meshtags(mesh, self._edim, self.feat_edges,
                            np.ones(self.feat_edges.size, dtype=np.int32))
        # interior codim-2 ridge measure over the tagged feature edges
        self.dGamma = ufl.Measure("ridge", domain=mesh, subdomain_data=ft)(1)

        self.V = fem.functionspace(mesh, ("Lagrange", 1))
        self.Hf = fem.Function(self.V, name="H_f")
        self.Hf_n = fem.Function(self.V, name="H_f_n")

        # Γ vertices TOPOLOGICALLY from the feature edges (robust -- independent of the locator's geometric
        # form, which on a midpoint/segment locator could misclassify endpoint vertices). The Γ dofs are
        # the P1 dofs at those vertices; the off-Γ dofs (everything else) are pinned to 0 -- the ridge
        # integral only couples Γ dofs (off-Γ vertex bases vanish on every Γ edge), so the off-Γ rows are
        # otherwise unconstrained (singular) and get the trivial equation H_f=0.
        mesh.topology.create_connectivity(self._edim, 0)
        e2v = mesh.topology.connectivity(self._edim, 0)
        gverts = (np.unique(np.concatenate([e2v.links(e) for e in self.feat_edges])).astype(np.int32)
                  if self.feat_edges.size else np.empty(0, dtype=np.int32))
        nv = mesh.topology.index_map(0).size_local
        off_verts = np.setdiff1d(np.arange(nv, dtype=np.int32), gverts).astype(np.int32)
        self._gamma_dofs = fem.locate_dofs_topological(self.V, 0, gverts)
        self._off_dofs = fem.locate_dofs_topological(self.V, 0, off_verts)
        # Diagonal allocation for the pin: the ridge integral touches ONLY Γ dofs, so the off-Γ rows are
        # absent from the Jacobian sparsity and the Dirichlet pin has no diagonal slot to overwrite (LU:
        # "missing diagonal entries"). A tiny VERTEX (dP) integral on exactly the off-Γ vertices allocates
        # those diagonals; it is conservation-neutral (only the pinned, overwritten rows are touched) --
        # the same device CoupledProblem uses for its pinned d/λ rows.
        vt = dmesh.meshtags(mesh, 0, off_verts, np.ones(off_verts.size, dtype=np.int32))
        self._dPoff = ufl.Measure("vertex", domain=mesh, subdomain_data=vt)(1)

        self.length = mesh.comm.allreduce(
            fem.assemble_scalar(fem.form(fem.Constant(mesh, PETSc.ScalarType(1.0)) * self.dGamma)),
            op=MPI.SUM)
        # the two feature ENDS (extreme projection of the Γ nodes onto t̂), for the conveyance current
        s = self.V.tabulate_dof_coordinates()[self._gamma_dofs] @ t
        self._end_lo = int(self._gamma_dofs[np.argmin(s)])
        self._end_hi = int(self._gamma_dofs[np.argmax(s)])

    # -- boundary conditions --------------------------------------------------
    def pin_bc(self):
        """Dirichlet ``H_f = 0`` on all off-Γ dofs (co-location pin; conservation-neutral)."""
        return fem.dirichletbc(PETSc.ScalarType(0.0), self._off_dofs, self.V)

    # -- residual term builders ----------------------------------------------
    def conveyance_form(self, v):
        """1-D Darcy conveyance along Γ: ``∫_Γ K_feat·A·(∇H_f·t̂)(∇v·t̂) dΓ`` (tangential projection)."""
        gt = lambda f: ufl.dot(ufl.grad(f), self.t_hat)
        return self.K_feat * self.area * gt(self.Hf) * gt(v) * self.dGamma

    def storage_form(self, v, dt):
        """Feature storage rate: ``∫_Γ φ·A·(H_f − H_f^n)/dt · v dΓ`` (linear fill; reduces to dStored/dt)."""
        return self.porosity * self.area * (self.Hf - self.Hf_n) / dt * v * self.dGamma

    def exchange_into_feature(self, v, psi):
        """Sign-paired soil exchange, FEATURE block: ``∫_Γ σ·(H_f − ψ)·v dΓ`` (feature loses when H_f>ψ
        -- disperse; gains when ψ>H_f -- drain). Phase 3 replaces the constant σ with the Kirchhoff leg."""
        return self.sigma * (self.Hf - psi) * v * self.dGamma

    def exchange_into_host(self, w, psi):
        """Sign-paired soil exchange, HOST block: ``−∫_Γ σ·(H_f − ψ)·w dΓ`` -- the exact negative of the
        feature term on the SAME dΓ, so the coupling conserves water to machine precision (structural)."""
        return -self.sigma * (self.Hf - psi) * w * self.dGamma

    # -- Phase-3 sorptive-exchange CLOSURE (the gate-validated sorptivity clock) -----------------------
    # Upgrades the constant-σ wall exchange to q = Ω(I)·[Φ(H_f)−Φ(ψ)] (the Kirchhoff form): a per-face
    # cumulative-uptake state advances a sorptivity clock, with a DIRECTION SWITCH -- DISPERSE (H_f>ψ) uses
    # the a-priori cylindrical Green-Ampt branch (F_cylindrical + Parlange S), DRAIN (ψ>H_f) the semi-
    # empirical sub-√t throttle branch (F_throttle + a desorptivity). SEPARATE I_disp/I_drain accumulators
    # (Arik 2026-06-09: no reset on reversal). Validated by tests/test_sorptive_closure_gate.py (the full-
    # curve C-004 gate) + tests/test_feature_sorptive.py (the machinery reduces to the validated clock).
    def configure_sorptive(self, soil, *, psi_i=-1.0, psi_wall=0.0, des_sorp=None,
                           r_w=R_W_DEFAULT, perimeter=None, r_eq=None, C_disp=1.0, C_drain=1.0):
        """Switch the wall exchange to the Phase-3 sorptivity clock. ``soil`` is a VanGenuchten-like object
        (provides ``theta``/``K``/``kirchhoff``/``kirchhoff_ufl``). Reference (de)sorptivity, storable
        contrast Δθ and Kirchhoff drop ΔΦ are evaluated at the antecedent soil head ``psi_i`` and the wall
        head ``psi_wall`` (gate scenario: disperse dry soil psi_i=−1, saturated wall psi_wall=0; the drain
        mirror uses the same |ΔΦ| magnitude). DISPERSE = a-priori (cyl Green-Ampt + Parlange S). DRAIN =
        semi-empirical (throttle + a ``des_sorp``·S desorptivity; pass ``des_sorp`` to override the ratio
        default). The closure flux is per unit WALL AREA, so the per-length ridge exchange carries the wall
        ``perimeter`` (default circular ``2π·r_w``). Creates the SEPARATE per-face states
        ``I_disp``/``I_drain`` and the lagged ``Omega``."""
        self._soil = soil
        self._r_w = float(r_w)
        self._perimeter = 2.0 * np.pi * self._r_w if perimeter is None else float(perimeter)
        self._C_disp, self._C_drain = float(C_disp), float(C_drain)
        self.S_disp = parlange_sorptivity(soil, psi_i, psi_wall)
        self.dth_disp = abs(float(soil.theta(psi_wall) - soil.theta(psi_i)))
        self.dPhi_ref_disp = float(soil.kirchhoff(psi_i, psi_wall))
        self._des_ratio = des_sorp_ratio(self.dth_disp) if des_sorp is None else float(des_sorp)
        self.S_drain = self._des_ratio * self.S_disp
        self.dth_drain = self.dth_disp
        self.dPhi_ref_drain = self.dPhi_ref_disp
        self.I_disp = fem.Function(self.V, name="I_disp")
        self.I_drain = fem.Function(self.V, name="I_drain")
        self.Omega = fem.Function(self.V, name="Omega")
        self._setup_shell(r_eq)
        return self

    # -- dual-scale embedding: the off-Γ shell (far-field ψ_cell) + the sub-grid fill capacity -----------
    # The naive co-located coupling UNDER-predicts ~3x: injecting the wall flux into the near-Γ cell
    # saturates it, so ψ(Γ) -> 0 and the clock drive collapses (worsens with refinement). The fix
    # (validated 2026-06-09, ≤2% across host resolutions): (1) read the clock DRIVE from an off-Γ SHELL
    # ψ_cell (~1.5 host cells out -- the true far field), NOT ψ(Γ); (2) GATE the host source OFF until the
    # sub-grid annulus [r_w, r_eq] fills (I >= I_fill = Δθ(r_eq²−r_w²)/2r_w), so during sorption the host
    # stays at ψ_i, ψ_cell ≈ ψ_i, and the clock reproduces the resolved reference. The I-clock is the total
    # uptake; the sub-grid reservoir (I·perimeter − host_gain) is held internally (an explicit mass term
    # for a fully-coupled sim). See [[pids-module4-starting-note]], scratch/_zdualscale_probe.py.
    def _setup_shell(self, r_eq):
        xc = self.V.tabulate_dof_coordinates()
        gco = xc[self._gamma_dofs]
        off = np.setdiff1d(np.arange(xc.shape[0], dtype=np.int32), self._gamma_dofs).astype(np.int32)
        # distance of each off-Γ dof to the feature (nearest Γ dof) ~ perpendicular radial distance
        d = (np.min(np.linalg.norm(xc[off][:, None, :] - gco[None, :, :], axis=2), axis=1)
             if off.size and gco.size else np.empty(0))
        h = float(np.min(d)) if d.size else self._r_w        # local host cell size ~ nearest off-Γ dof
        if r_eq is None:
            sel = (d > 0.9 * h) & (d < 2.1 * h)
            if not np.any(sel):
                sel = d < 2.5 * h                            # widen if the first ring is empty
            self._r_eq = float(d[sel].mean()) if np.any(sel) else float(2.0 * self._r_w)
        else:
            self._r_eq = float(r_eq)
            sel = np.abs(d - self._r_eq) < h
            if not np.any(sel) and d.size:
                sel = np.zeros(d.size, dtype=bool); sel[np.argsort(np.abs(d - self._r_eq))[:8]] = True
        self._shell_dofs = off[sel] if d.size else np.empty(0, dtype=np.int32)
        self._I_fill = max(self.dth_disp * (self._r_eq**2 - self._r_w**2) / (2.0 * self._r_w), 0.0)

    def _psi_cell(self, psi):
        """Far-field ψ on the off-Γ shell (the clock drive reference); global host mean if the shell is empty."""
        return float(psi.x.array[self._shell_dofs].mean() if self._shell_dofs.size else psi.x.array.mean())

    def seed_clock(self, t_seed):
        """Seed both per-face accumulators to the planar early limb ``S·sqrt(t_seed)`` (the antecedent
        contact age) -- avoids the 1/I singularity at I=0; off-Γ stays 0."""
        self._t_seed = float(t_seed)
        for Ifun, S in ((self.I_disp, self.S_disp), (self.I_drain, self.S_drain)):
            Ifun.x.array[:] = 0.0
            Ifun.x.array[self._gamma_dofs] = S * np.sqrt(t_seed)
            Ifun.x.scatter_forward()

    def _omega_branch(self, I, S, dPhi_ref, dth, F, C):
        Ipos = np.maximum(I, 1e-30)
        return C * S * S * F(Ipos / (dth * self._r_w)) / (2.0 * Ipos * dPhi_ref)

    def update_well_index(self, psi):
        """Refresh the LAGGED well-index coefficient ``Omega`` on Γ. Direction is set by the far-field
        ψ_cell (H_f>ψ_cell → cylindrical disperse branch, else throttle drain branch); ``Omega`` is GATED
        to 0 until that direction's sub-grid annulus fills (``I >= I_fill``) -- during the sub-grid sorption
        phase the host receives NO source (the uptake is held in the I-clock reservoir). Call once at the
        start of a step; the residual then treats ``Omega`` as a frozen coefficient (the on-Γ Kirchhoff
        difference still self-limits as ψ(Γ) saturates)."""
        g = self._gamma_dofs
        hfg, psi_cell = self.Hf.x.array[g], self._psi_cell(psi)
        z0, k = throttle_params(self.dth_drain)
        om_d = self._omega_branch(self.I_disp.x.array[g], self.S_disp, self.dPhi_ref_disp,
                                  self.dth_disp, F_cylindrical, self._C_disp)
        om_r = self._omega_branch(self.I_drain.x.array[g], self.S_drain, self.dPhi_ref_drain,
                                  self.dth_drain, lambda z: F_throttle(z, z0, k), self._C_drain)
        om_d = np.where(self.I_disp.x.array[g] >= self._I_fill, om_d, 0.0)   # storage gate
        om_r = np.where(self.I_drain.x.array[g] >= self._I_fill, om_r, 0.0)
        self.Omega.x.array[:] = 0.0
        self.Omega.x.array[g] = np.where(hfg > psi_cell, om_d, om_r)
        self.Omega.x.scatter_forward()

    def sorptive_into_feature(self, v, psi):
        """Sorptive wall exchange, FEATURE block: ``∫_Γ p·Ω·[Φ(H_f)−Φ(ψ)]·v dΓ`` (the Kirchhoff flux q per
        wall area times the wall perimeter p; a sink on H_f when disperse). ``Ω`` is the lagged coefficient;
        the Kirchhoff difference is the differentiable ``kirchhoff_ufl`` (so the Newton Jacobian is clean)."""
        return self._perimeter * self.Omega * self._soil.kirchhoff_ufl(psi, self.Hf) * v * self.dGamma

    def sorptive_into_host(self, w, psi):
        """Sorptive wall exchange, HOST block: ``−∫_Γ p·Ω·[Φ(H_f)−Φ(ψ)]·w dΓ`` -- the exact negative on the
        SAME dΓ (structurally conservative)."""
        return -self._perimeter * self.Omega * self._soil.kirchhoff_ufl(psi, self.Hf) * w * self.dGamma

    def host_sorptive_flux(self, psi):
        """Net sorptive exchange rate across Γ ``∫_Γ p·Ω·[Φ(H_f)−Φ(ψ)] dΓ`` (+ into the soil = disperse;
        − = drain), using the current lagged ``Omega`` (volumetric: includes the wall perimeter p)."""
        return self.mesh.comm.allreduce(fem.assemble_scalar(
            fem.form(self._perimeter * self.Omega * self._soil.kirchhoff_ufl(psi, self.Hf) * self.dGamma)),
            op=MPI.SUM)

    def advance_clock(self, psi, dt, nsub=400):
        """Advance the per-face state(s) over ``dt`` (post-step), integrating the clock
        ``dI/dt = (S²/2I)·F(ζ)·(|ΔΦ_live|/ΔΦ_ref)`` with the END-of-step driving potential
        ``ΔΦ_live = Φ(H_f)−Φ(ψ)`` (a partial/evolving head drop scales the flux). Each face feeds only its
        ACTIVE accumulator (disperse: H_f>ψ → I_disp; drain → I_drain); the inactive one is frozen.
        ``nsub`` sub-steps the 1/I stiffness over the step."""
        g = self._gamma_dofs
        hfg, psi_cell = self.Hf.x.array[g], self._psi_cell(psi)   # drive read from the off-Γ far field
        disperse = hfg > psi_cell
        # the driving-potential MAGNITUDE, evaluated low->high so the K-graded Kirchhoff quadrature clusters
        # at the wet end both ways (kirchhoff is graded toward its upper limit; kirchhoff(0,-1) would under-
        # resolve the large-K wet end and break the drain<->disperse symmetry of |ΔΦ|).
        dPhi_mag = self._soil.kirchhoff(np.minimum(psi_cell, hfg), np.maximum(psi_cell, hfg))
        scale_d = np.where(disperse, dPhi_mag / self.dPhi_ref_disp, 0.0)
        scale_r = np.where(~disperse, dPhi_mag / self.dPhi_ref_drain, 0.0)
        z0, k = throttle_params(self.dth_drain)
        Id, Ir = self.I_disp.x.array[g].copy(), self.I_drain.x.array[g].copy()
        h = dt / nsub
        for _ in range(nsub):
            Id = Id + h * (self._C_disp * self.S_disp ** 2 / (2.0 * Id)
                           * F_cylindrical(Id / (self.dth_disp * self._r_w)) * scale_d)
            Ir = Ir + h * (self._C_drain * self.S_drain ** 2 / (2.0 * Ir)
                           * F_throttle(Ir / (self.dth_drain * self._r_w), z0, k) * scale_r)
        self.I_disp.x.array[g], self.I_drain.x.array[g] = Id, Ir
        self.I_disp.x.scatter_forward()
        self.I_drain.x.scatter_forward()

    def hf_residual(self, v):
        """The feature's contribution to the ``H_f`` block: conveyance + the off-Γ pin diagonal
        allocation (storage and the sign-paired exchange are layered on by the caller / integration)."""
        return self.conveyance_form(v) + self.Hf * v * self._dPoff

    # -- diagnostics ----------------------------------------------------------
    def stored_water(self) -> float:
        """Water stored in the feature: ``∫_Γ φ·A·H_f dΓ`` (= φ·A·|Γ| at unit fill H_f≡1)."""
        return self.mesh.comm.allreduce(
            fem.assemble_scalar(fem.form(self.porosity * self.area * self.Hf * self.dGamma)), op=MPI.SUM)

    def host_exchange_flux(self, psi) -> float:
        """Net soil↔feature exchange across Γ: ``∫_Γ σ·(H_f − ψ) dΓ`` (+ = into the host). Returns 0 for
        a sealed σ=0 feature without assembling the (UFL-empty) zero form."""
        if self.sigma == 0.0:
            return 0.0
        return self.mesh.comm.allreduce(
            fem.assemble_scalar(fem.form(self.sigma * (self.Hf - psi) * self.dGamma)), op=MPI.SUM)

    # -- diagnostics ----------------------------------------------------------
    def conveyance_current(self) -> float:
        """The 1-D Darcy current through the feature: the reaction of the conveyance operator at the
        low end (= ``K_feat·A·∂H_f/∂s``, constant along Γ at steady state). Serial-only diagnostic: it
        reads a LOCAL vector entry, so the endpoint dof must be locally owned (MPI deferred, like M1-M3)."""
        if self.mesh.comm.size > 1:
            raise NotImplementedError("conveyance_current() is serial-only (reads a local vector entry).")
        v = ufl.TestFunction(self.V)
        R = fp.assemble_vector(fem.form(self.conveyance_form(v)))
        R.assemble()
        return float(R.array[self._end_lo])
