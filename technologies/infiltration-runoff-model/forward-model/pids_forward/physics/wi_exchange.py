"""Sub-grid wall exchange via the measured discrete well index (Module 4 §E, Phase 4).

PRODUCTION form of the Phase-4 "rate-clock + WI" scheme. ADVERSARIALLY REVIEWED 2026-06-11
(validation/sanity/m4_phase4_coupled_review__2026-06-11.md -- the binding wording): in the tested
homogeneous-isotropic LOAM disperse deployment regime (R=40 r_w, positive-WI meshes n=6..12) the
scheme tracks independent closed-domain references at 2.2-5.7% rel-L2 (offline clock 26.6%, the
retracted dual-scale 13-39% across its measured legs) and tracks a host-history perturbation
(RefB-40 = 5.0-5.4%, landing on the pulse-shifted asymptote with no pulse knowledge; clock 34.1%);
over-correction control = 2.1-6.0% (clock rightly 1.0% there; the error sign is the mid-curve
over-delivery, not over-throttling). HOST CONTROL IS ESTABLISHED FOR THE WI ERA (>=80-91% of
cumulative I at n=8-12, carrying ALL discriminating signal -- the RefB-40 pulse lands after
handover); the sub-grid era is a PRESCRIBED-RATE CLOSURE whose host read is a second-order
correction (<=~2% on the committed legs, mechanism-present but not gate-exercised). The disperse
WI-era residual (5.7% worst, the LOAM n=12 leg, on the original ON-RIDGE read) was RESOLVED by item
A (2026-06-14): the 2026-06-12 diagnostic (scratch/m4_phase4_wi_residual_diag.py) exonerated the
bridge FORM (evaluated on the RESOLVED field at r_0 it reproduces the resolved wall rate to
+0.3..0.6% through the bend) and localized the carrier to the embedded ON-RIDGE psi sitting WETTER
than the continuum at r_0 (dPhi -7..-18%) under transient nonlinear depletion -- NOT the offline F
closure's large-zeta bias (that is -0.9..-1.5%, wrong sign and era; review attack f). THE FIX (see
THE SCHEME below): the WI-era bridge reads the RESOLVED far-field annulus at r_ring = R_out/2
instead of the on-ridge node (scratch/m4_phase4_wi_ring_derivation.py: worst rate dev 2.0% across n=6..12), as a
Heun-corrected prescribed rate with a live recharge-aware capacity throttle -- the disperse worst
dropped to 2.6% (every leg <=3% across n=8/12 + the soil triad + the RefB40 history leg). The
disperse evidence does not by itself exclude passive capacity-aware schemes (their killers are the
drain legs this class refuses). Prototype + evidence: scratch/m4_phase4_wi_probe.py,
scratch/m4_phase4_embedded_harness.py, tests/test_coupled_gate_refs.py, tests/test_wi_exchange.py.

THE SCHEME (no tuned knobs -- every number is measured, derived, or, for r_ring, SELECTED from a
broad measured plateau):
* Post-handover (the WI era) the wall exchange is a PRESCRIBED ridge rate set by the steady
  cylindrical Kirchhoff bridge driven by the RESOLVED host field at r_ring = R_out/2 (the catchment-
  radius midpoint):
      q_per_length = 2*pi * [Phi(H_f) - Phi(psi_ring)] / ln(r_ring/r_w),
  psi_ring = the mean head over the vertex band |rho - R_out/2| <= 0.6 h (a ~+-0.6 h annulus, several
  shells on coarse meshes, NOT a single thin shell -- which is why the broad plateau matters), and
  r_ring is its captured mean radius. R_out/2 is not a first-principles constant: it was SELECTED
  from the resolved-truth derivation (scratch/m4_phase4_wi_ring_derivation.py, LOAM R40) as the
  centre of a flat [0.45,0.6]*R_out plateau (worst matched-I rate dev 2.0% across n=6..12 -- above
  the near-wall read-fidelity floor, below the outer no-flow boundary that breaks the Thiem log);
  the plateau's breadth makes the exact pick insensitive. Reading the resolved far field instead of
  the on-ridge node (the original Peaceman bridge WI=2*pi/ln(r_0/r_w) on psi_Gamma, r_0=
  R0_OVER_H_P1*h, Peaceman-for-FEM; tests/test_well_index_p1.py) removes the -7..-18% wet-read bias
  (item A). Two DERIVED corrections, no knobs: the rate is HEUN-averaged (the start-state rate held
  over the big WI-era steps is an explicit lag that over-injects) and THROTTLED by the live remaining
  capacity (theta_s - theta_bulk)*V_box/dt (the infinite-domain bridge has no outer-boundary
  knowledge; the live volume-mean theta read makes the cap mass-exact AND recharge-aware). r_0(h)
  still sets the resolved-wall scope guard and the handover fill I_fill.
* While the Green-Ampt front R_f(I) = sqrt(r_w^2 + 2 r_w I/dtheta) is inside the lattice's measured
  log-fidelity radius 2h, the host cannot represent the sorption front: the validated clock
  prescribes the flux as a RATE source on the ridge (a prescribed rate has no dPhi feedback to
  collapse -- the v1 prototype's potential-driven handover stalled at emb/ref = 0.556), the lattice
  builds its own consistent response, and handover at I_fill = dtheta*((2h)^2 - r_w^2)/(2 r_w) is
  seamless. (front=2h is the Task-5 far-field fidelity radius, a pre-scheme measurement; it is
  validated as a fidelity radius, not as a transient-handover optimum.) The clock-era rate carries
  a live Kirchhoff scale read from the FRONT RING (off-ridge vertices one cell ahead of R_f; it
  moves with the front, hard-zeros if the ring reaches the wall head) -- a saturation safeguard and
  second-order correction; no committed leg discriminates it from a constant drive (review attack
  a), so no clock-era host-control claim rests on it.
* All uptake is host-ward immediately -- no sub-grid reservoir; the only ledger remainder is the
  t0 seed S*sqrt(t0) (water already in the ground at seeding, ~0.01% of a deployment capacity).

THE DRAIN DIRECTION (closed/deployment domains, 2026-06-12): the PSS DEPLETION closure -- the
front-similarity machinery does NOT apply (the desat front stays sub-cell and the closed
near-saturated bulk leaves the similarity regime within ~a day; prior failures: open-fitted
throttle 98%, cyl-GA family 35-81%, the WI bridge 10.9/16.8% degrading). The validated mechanism
(3.3-3.9% on refD40 + two PRE-REGISTERED fresh refs, SAND R40 + LOAM R20 deep-depletion, zero
knobs -- scratch/m4_phase4_drain_desorptivity.py + m4_phase4_drain_fresh_refs.py) is quasi-steady
closed-reservoir Kirchhoff depletion. Embedded form: a prescribed-rate ridge SINK
      rate = 2*pi*[Phi(psi_bulk) - Phi(mean H_f)] / (ln(R_out/r_w) - 3/4)  per unit length,
      psi_bulk = retention^-1( lumped volume-mean theta of the live host field ),
i.e. the LIVE discrete closed-box WATER BALANCE -- the bulk state the validated PSS law is
defined on, mass-exact however coarsely the drawdown cone is resolved, and recharge-aware (an
injected source raises the theta field and the drive responds; the refD40-C property) -- and the
live wall head. (The original plain dof-mean-psi read was measured +2.3-2.7% hot in dPhi -- the
follow-up-4 +8-10% end bias, boundary-vertex over-weighting of the wetter far field;
scratch/m4_phase4_drain_drive_diag.py. The remaining end excess decomposed EXACTLY as a +2.0%
law-vs-ref factor (then read as the law's own bias; item B reattributed it to the refs -- see
below) x the +4.7% first-order explicit lag of the start-state rate held over the step; the
Heun trapezoid rate in the scheme's own mass variable -- pre_step's optional dt -- removes the
lag.) With the theta-mean + Heun drive the legs score 4.4/3.6/2.9% rel-L2 at end emb/ref
0.979/0.986/0.983 (refD40/SAND-R40/refD40-C, n=8 vs the CONVERGED refs), n-INDEPENDENT at
n=8/12 -- the mass-exact read removed the resolution dependence. The small remaining UNDER
(~1-3%) is the offline PSS law's own Jensen volume-average gap (it drives on Phi(theta_mean) <=
<Phi>): the law is accurate, slightly conservative. (The previously-recorded "+2-4% OVER end
bias = the law's own model-form" was REFUTED by item B, 2026-06-15: it was a REFERENCE
first-order backward-Euler UNDER-count -- the closed-drain refs read ~4% low; regenerated at a
converged dt cap (window/2048) the over-bias flips to this small Jensen under. Binding record:
validation/sanity/m4_phase4_drain_endbias_attribution__2026-06-15.md.)
The host read is LOAD-BEARING: the depletion bend exists only because the
host bulk falls (fixed-drive twin fails refD40 at 74%; the original dof-mean read scored
7.4/6.9% at end 1.085). R_out = the closed/catchment equivalent radius, a physical geometry input
(ctx["R_out"], required). Omega stays 0 in drain mode (pure prescription; capacity-safe: the rate
hard-zeros as the bulk mean reaches the wall head).

SCOPE GUARDS (refusals, not silent degradation):
* DISPERSE positive-WI deployment regime only (r_0 = R0_OVER_H_P1*h > 1.1 r_w, i.e. h > ~5.5 r_w --
  field grids around a 5 cm feature) AND ctx["R_out"] required (the WI-era ring read needs R_out/2).
  This is the validated regime; finer (resolved-wall) meshes are a separate design task, refused with
  ValueError (historically the on-ridge Peaceman bridge there was negative-log with a repelling BE
  fixed point at handover, analyzed 2026-06-10; the item-A ring read is positive-log but is unvalidated
  in the resolved-wall regime too).
* DRAIN requires ctx["R_out"] (the PSS geometry); refused with ValueError without it.

USAGE (the step contract, harness-compatible -- scratch/m4_phase4_embedded_harness.py is the
reference driver): the host residual carries the prescribed ridge rate from pre_step's return as a
ridge source (per unit length = rate/feat.length, e.g. `-rate_c * w * feat.dGamma`).
feat.sorptive_into_host(w, psi) / feat.Omega is retained for interface compatibility but is now
INERT (Omega == 0 in ALL eras -- the WI-era implicit-Omega bridge was retired by item A's
resolved-ring prescribed rate; the harness may keep sorptive_into_host in the residual, it
contributes 0). Per step: rate = pre_step(feat, psi, t, dt) BEFORE the solve -- PASS dt: the
disperse WI era uses it for the Heun rate + the live capacity throttle, the drain for the Heun
depletion rate (omitting it falls back to the lagged start-state rate); post_step(feat, psi, t, dt)
AFTER it. Mass identity: I_total*perimeter*length == injected + seed*perimeter*length.
"""
from __future__ import annotations

import numpy as np

from .sorptive_closure import F_cylindrical, R0_OVER_H_P1, R_W_DEFAULT


class WellIndexExchange:
    """Embedded wall exchange. DISPERSE: sub-grid rate-clock era + resolved-ring Kirchhoff era (both
    prescribed ridge rates; host-controlled in the WI era via the resolved R_out/2 read). DRAIN: the
    PSS depletion closure as a live-host-driven prescribed-rate ridge sink -- see the module docstring."""

    def __init__(self, direction: str = "disperse"):
        if direction not in ("disperse", "drain"):
            raise ValueError(f"direction must be 'disperse' or 'drain' (got {direction!r})")
        self.direction = direction

    # -- lifecycle -------------------------------------------------------------
    def setup(self, feat, soil, ctx):
        """Bind to an EmbeddedFeature (after configure_sorptive). DISPERSE ctx: t0 (required, the
        seeding contact age, > 0); h (optional -- auto-measured as the nearest off-ridge vertex
        distance). DRAIN ctx: R_out (required, the closed/catchment equivalent radius)."""
        comm_size = getattr(getattr(getattr(feat.V, "mesh", None), "comm", None), "size", 1)
        if comm_size > 1:
            raise ValueError(
                f"WellIndexExchange: parallel mesh refused (comm size {comm_size}); every host "
                f"read here (dof/theta means, the front-ring mean, the WI-era R_out/2 ring mean, "
                f"the lumped volume weights) is rank-local and would be silently partition-dependent. "
                f"The gate evidence is "
                f"serial-only (2026-06-12 Codex review finding 1).")
        from scipy.spatial import cKDTree
        self.soil, self.feat = soil, feat
        if self.direction == "drain":
            R_out = (ctx or {}).get("R_out") if isinstance(ctx, dict) else None
            if not R_out:
                raise ValueError(
                    "WellIndexExchange(drain): ctx['R_out'] is required -- the PSS depletion "
                    "closure needs the closed/catchment equivalent radius (ln(R_out/r_w) - 3/4).")
            self._r_w = feat._r_w
            self.geo = np.log(float(R_out) / self._r_w) - 0.75
            self._g = feat._gamma_dofs
            self._p_len = feat._perimeter * feat.length
            # lumped vertex volume weights: the drive is psi(volume-mean theta) -- the discrete
            # closed-box water balance (mass-exact however coarsely the drawdown cone is
            # resolved, and recharge-aware), which is the bulk state the PSS law is defined on.
            # The plain dof-mean psi read was measured +2.3-2.7% hot in dPhi (-> the +8-10% end
            # bias): boundary vertices (over half the dofs at n=8, carrying 1/2-1/8 cell
            # volumes) over-weight the wetter far field (scratch/m4_phase4_drain_drive_diag.py).
            self._set_volume_weights(feat)             # self._vol, self._wvol (the Heun mass step)
            self.inj = 0.0                              # cumulative EXTRACTED volume [m^3], >= 0
            self._last_rate = 0.0
            return self
        self._g = feat._gamma_dofs
        self._r_w = feat._r_w
        xc = feat.V.tabulate_dof_coordinates()
        gco = xc[self._g]
        # nearest-Gamma-VERTEX distance of every dof (exact perpendicular distance for the straight,
        # vertex-resolved ridges validated so far; an approximation for curved/coarse Gamma)
        self._rho = cKDTree(gco).query(xc, k=1)[0]
        off = self._rho > 1e-12
        h = ctx.get("h") if isinstance(ctx, dict) else None
        self.h = float(h) if h else float(self._rho[off].min())
        self.r0 = R0_OVER_H_P1 * self.h
        if self.r0 <= 1.1 * self._r_w:
            raise ValueError(
                f"WellIndexExchange: resolved-wall regime refused (r_0={self.r0:.4f} m <= 1.1*r_w; "
                f"h={self.h:.3f} m = {self.h/self._r_w:.1f} r_w; need h > ~5.5 r_w) -- outside the "
                f"validated deployment regime; a resolved-wall coupling is a separate design task.")
        self.WI = 2.0 * np.pi / np.log(self.r0 / self._r_w)   # regime witness (the ring read uses r_ring)
        self.S, self.dth = feat.S_disp, feat.dth_disp
        self.dPhi_ref = feat.dPhi_ref_disp
        wall = float(getattr(feat, "psi_wall_sorp", 0.0))
        if wall != 0.0:
            raise ValueError(
                f"WellIndexExchange: non-zero sorptive wall head refused (psi_wall={wall}); the "
                f"gate evidence is saturated-wall (psi_wall=0) only -- the clock-era Kirchhoff "
                f"scaling is unvalidated elsewhere (2026-06-11 review).")
        self.h_f_ref = wall                                 # the configure_sorptive wall head
        two_h = 2.0 * self.h
        self.I_fill = self.dth * (two_h ** 2 - self._r_w ** 2) / (2.0 * self._r_w)
        self.seed_I = 0.0
        self.inj = 0.0                                      # cumulative host-ward volume [m^3]
        self.in_subgrid_era = True
        self.t_handover = None
        self._last_rate = 0.0
        self._p_len = feat._perimeter * feat.length
        # The WI-era drive reads the RESOLVED host field at the catchment-radius midpoint R_out/2,
        # NOT the on-ridge node: the discrete on-ridge value drifts WETTER than the continuum at r_0
        # under transient nonlinear depletion (dPhi -7..-18%, the localized WI-era residual), while
        # the far-field annulus is faithfully resolved. r_ring = R_out/2 sits in the steady cyl
        # annulus (above the near-wall read-fidelity floor, below the outer no-flow boundary that
        # breaks the Thiem log); the derivation (scratch/m4_phase4_wi_ring_derivation.py) measured
        # worst |rate dev| 2.0% across the mesh range n=6..12 (broad plateau over [0.45,0.6]*R_out).
        R_out = ctx.get("R_out") if isinstance(ctx, dict) else None
        if not R_out:
            raise ValueError(
                "WellIndexExchange(disperse): ctx['R_out'] is required -- the WI-era bridge reads "
                "the resolved host field at the catchment-radius midpoint R_out/2 (the on-ridge read "
                "was -7..-18% wet; derivation scratch/m4_phase4_wi_ring_derivation.py).")
        self.r_ring_target = 0.5 * float(R_out)
        ring = (np.abs(self._rho - self.r_ring_target) <= 0.6 * self.h) & (self._rho > 1e-12)
        if not np.any(ring):
            raise ValueError(
                f"WellIndexExchange(disperse): no resolved vertex shell at R_out/2="
                f"{self.r_ring_target:.4f} m on this mesh (h={self.h:.4f} m); the WI-era ring read "
                f"needs one. Refine the mesh or check R_out.")
        self._ring_mask = ring
        self.r_ring = float(self._rho[ring].mean())
        self.ring_lnf = float(np.log(self.r_ring / self._r_w))
        # the WI-era rate is an EXPLICIT prescribed rate -> Heun-corrected (the start-state rate held
        # over the big WI-era steps over-injects -- the drain follow-up-4 lag) and throttled by the
        # LIVE remaining capacity: the infinite-domain ring bridge has no outer-no-flow-boundary
        # knowledge and would over-inject past saturation (+17% end overshoot + ledger break,
        # 2026-06-13 debug), so the rate is capped at (theta_s - theta_bulk_live)*V_box/dt. Reading
        # the LIVE volume-mean theta makes the cap mass-exact AND recharge-aware (a pulse pre-fills
        # the box and the wall backs off -- the RefB40 property). Same volume machinery as the drain.
        self._set_volume_weights(feat)
        t0 = ctx["t0"] if isinstance(ctx, dict) else float(ctx)
        self.seed(t0)
        return self

    def seed(self, t0: float):
        """Seed the clock at the contact age t0 > 0 (the 1/I singularity guard; matches the
        references' first sample). The seed water is ledgered, never injected (~0.01% of capacity)."""
        if float(t0) <= 0.0:
            raise ValueError(f"seed needs t0 > 0 (got {t0}); t0=0 -> I=0 -> 1/I blow-up.")
        self.seed_I = self.S * np.sqrt(float(t0))
        self.in_subgrid_era = self.seed_I < self.I_fill
        return self

    # -- helpers ----------------------------------------------------------------
    def _set_volume_weights(self, feat):
        """Lumped vertex volume weights w_i (self._wvol normalized, self._vol the total): the
        discrete closed-box water balance, mass-exact however coarsely the field is resolved and
        recharge-aware. Used by the drain bulk drive AND the disperse Heun mass step + live capacity
        throttle."""
        import ufl
        from dolfinx import fem as _fem
        v = ufl.TestFunction(feat.V)
        w = _fem.assemble_vector(_fem.form(v * ufl.dx(
            metadata={"quadrature_rule": "vertex", "quadrature_degree": 1}))).array.copy()
        self._vol = float(w.sum())
        self._wvol = w / self._vol

    def _psi_of_theta(self, th) -> float:
        """Closed-form inverse van-Genuchten retention psi(theta) (the bulk drive's mass variable)."""
        se = (th - self.soil.theta_r) / (self.soil.theta_s - self.soil.theta_r) * self.soil.Sc
        if se >= 1.0:
            return self.soil.h_s
        return -((max(se, 1e-12) ** (-1.0 / self.soil.m) - 1.0) ** (1.0 / self.soil.n)) \
            / self.soil.alpha

    # -- the per-step contract --------------------------------------------------
    def _I(self) -> float:
        return self.seed_I + self.inj / self._p_len

    def _clock_rate(self, psi) -> float:
        """Sub-grid-era prescribed ridge rate [m^3/day]: the validated cylindrical clock at the
        current cumulative state, scaled by the live Kirchhoff drop to the FRONT RING."""
        I = max(self._I(), 1e-12)
        R_f = np.sqrt(self._r_w ** 2 + 2.0 * self._r_w * I / self.dth)
        ring = (self._rho >= R_f + 0.5 * self.h) & (self._rho <= R_f + 1.5 * self.h)
        psi_far = float(psi.x.array[ring].mean()) if np.any(ring) else \
            float(psi.x.array[self._rho > 1e-12].mean())
        if psi_far >= self.h_f_ref:
            return 0.0                                      # ring at/above the wall head: no drive
        scale = self.soil.kirchhoff(psi_far, self.h_f_ref) / self.dPhi_ref
        dIdt = self.S ** 2 / (2.0 * I) * F_cylindrical(I / (self.dth * self._r_w)) * scale
        return dIdt * self._p_len

    def _ring_bridge(self, psi_ring) -> float:
        """The bare steady cylindrical Kirchhoff bridge rate [m^3/day] at a ring head psi_ring:
            q = 2*pi*[Phi(H_f) - Phi(psi_ring)] / ln(r_ring/r_w) * length
              = dPhi / (r_w * ln(r_ring/r_w)) * p_len   (the drain PSS rate shape, no -3/4)."""
        if psi_ring >= self.h_f_ref:
            return 0.0                                      # ring at/above the wall head: no drive
        dphi = float(self.soil.kirchhoff(psi_ring, self.h_f_ref))
        return dphi / (self._r_w * self.ring_lnf) * self._p_len

    def _wi_ring_rate(self, psi, dt=0.0) -> float:
        """WI-era prescribed ridge rate [m^3/day]: the steady cylindrical Kirchhoff bridge driven by
        the RESOLVED host field at r_ring = R_out/2 (the catchment-radius midpoint), read as the mean
        psi over the vertex shell there. Reading the far-field annulus instead of the on-ridge node
        removes the -7..-18% wet-read residual (the node drifts wetter than the continuum at r_0 under
        transient nonlinear depletion; the bridge FORM was already exonerated). Derivation +
        the resolved-ring radius: scratch/m4_phase4_wi_ring_derivation.py.

        With dt > 0 (production) two coupled corrections, both DERIVED (no knobs):
        * HEUN: the start-state rate held over the big WI-era steps is a first-order explicit lag
          (over-injection; the drain follow-up-4 lag). rate = (r0+r1)/2, r1 = the bridge at the ring
          head PREDICTED after the step's mean theta rise r0*dt/V_box (the scheme's own mass step).
          APPROXIMATION: the ring's LOCAL theta rise is taken equal to the box-MEAN rise (the injected
          water is wall-concentrated, not uniform) -- a second-order correction to a prescribed rate;
          the measured 2.0-2.6% disperse result validates it, and it mirrors the drain Heun's step.
        * LIVE CAPACITY THROTTLE: the infinite-domain bridge has no outer-no-flow-boundary knowledge
          and would over-inject past saturation. Cap the rate at the LIVE remaining capacity
          (theta_s - theta_bulk)*V_box/dt with theta_bulk the volume-mean of the current field --
          mass-exact and recharge-aware (a pulse raises theta_bulk and the wall backs off)."""
        psi_ring = float(psi.x.array[self._ring_mask].mean())
        r0 = self._ring_bridge(psi_ring)
        if dt <= 0.0 or r0 == 0.0:
            return r0
        th_pred = min(float(self.soil.theta(psi_ring)) + r0 * dt / self._vol,
                      self.soil.theta_s)
        r1 = self._ring_bridge(self._psi_of_theta(th_pred))
        rate = 0.5 * (r0 + r1)
        th_bulk = float((self._wvol * self.soil.theta(psi.x.array)).sum())
        rem = self.soil.theta_s - th_bulk                   # live remaining capacity (theta units)
        if rem <= 1e-12 * (self.soil.theta_s - self.soil.theta_r):
            return 0.0                                      # box full (pad absorbs weight-sum rounding)
        return min(rate, rem * self._vol / dt)

    def _drain_rate_at(self, th_bar, hf_bar) -> float:
        """The PSS depletion rate [m^3/day, >= 0] at a given bulk water content th_bar."""
        # capacity safety in the mass variable (the law's state): bulk at/below the wall-head
        # water content -> no drive (the 1e-12-porosity pad absorbs the weight-sum rounding)
        if th_bar <= float(self.soil.theta(hf_bar)) \
                + 1e-12 * (self.soil.theta_s - self.soil.theta_r):
            return 0.0
        se_vg = (th_bar - self.soil.theta_r) / (self.soil.theta_s - self.soil.theta_r) \
            * self.soil.Sc
        psi_bar = self.soil.h_s if se_vg >= 1.0 else \
            -((max(se_vg, 1e-12) ** (-1.0 / self.soil.m) - 1.0) ** (1.0 / self.soil.n)) \
            / self.soil.alpha                              # closed-form inverse retention
        if psi_bar <= hf_bar:
            return 0.0                                     # residual psi-space safety
        dPhi = float(self.soil.kirchhoff(hf_bar, psi_bar))
        return dPhi / (self._r_w * self.geo) * self._p_len

    def _drain_rate(self, feat, psi, dt=0.0) -> float:
        """Drain prescribed extraction [m^3/day, >= 0]: the PSS depletion law with the LIVE
        water-balance bulk drive psi(volume-mean theta) and the LIVE wall head. With dt > 0 the
        rate is the HEUN (trapezoid) value -- the start-state rate is a first-order explicit lag
        (the rate is held over the step while the bulk depletes; measured +4.7% end excess on
        the refD40 mark grid): rate = (r0 + r1)/2, r1 evaluated at th_bar - r0*dt/V_box, the
        scheme's own mass prediction. Second-order in dt W.R.T. THE SCHEME'S OWN EXTRACTION
        ONLY (2026-06-12 Codex review finding 2): external host sources (recharge, rain) enter
        the predictor only through the live theta read at step start, so the rate stays
        first-order during active external forcing -- by design: the scheme cannot enumerate
        arbitrary host sources; the live field read is the general mechanism (measured on the
        refD40-C recharge leg: end 0.996, relL2 2.7%, better than the 1.025 pre-Heun end).
        Zero knobs; dt=0 degenerates to r0."""
        th_bar = float((self._wvol * self.soil.theta(psi.x.array)).sum())
        hf_bar = float(feat.Hf.x.array[self._g].mean())
        r0 = self._drain_rate_at(th_bar, hf_bar)
        if dt <= 0.0 or r0 == 0.0:
            return r0
        r1 = self._drain_rate_at(th_bar - r0 * dt / self._vol, hf_bar)
        return 0.5 * (r0 + r1)

    def pre_step(self, feat, psi, t, dt=0.0) -> float:
        """Call BEFORE the solve. Sets feat.Omega (the WI-era coefficient; 0 in drain mode) and
        returns the prescribed ridge rate [m^3/day] (negative = host sink in drain mode; 0 after
        the disperse handover) -- the caller carries it as a ridge source in the host residual.
        Pass the step's dt (known before the solve) to get the lag-free Heun drain rate; dt is
        ignored by the disperse direction (its clock-era lag is inside the validated clock)."""
        feat.Omega.x.array[:] = 0.0
        self._last_rate = 0.0
        if self.direction == "drain":
            feat.Omega.x.scatter_forward()
            self._last_rate = self._drain_rate(feat, psi, dt)
            return -self._last_rate
        if self.in_subgrid_era:
            feat.Omega.x.scatter_forward()
            self._last_rate = self._clock_rate(psi)
            return self._last_rate
        feat.Omega.x.scatter_forward()                      # WI era: Omega stays 0 (prescribed rate)
        self._last_rate = self._wi_ring_rate(psi, dt)
        return self._last_rate

    def post_step(self, feat, psi, t, dt):
        """Call AFTER an accepted solve: account the step's prescribed exchange and handle handover.
        Both disperse eras (clock + WI-ring) and drain are now prescribed ridge rates -- the WI-era
        implicit-Omega bridge was retired by the item-A resolved-ring read (2026-06-13)."""
        self.inj += self._last_rate * dt                   # exactly what the caller injected/extracted
        if self.direction == "drain":
            return
        if self.in_subgrid_era and self._I() >= self.I_fill:
            self.in_subgrid_era = False
            self.t_handover = t + dt

    # -- observables -------------------------------------------------------------
    def I_total(self, feat=None) -> float:
        """Cumulative uptake per unit wall area [m] (the gate observable)."""
        if self.direction == "drain":
            return self.inj / self._p_len
        return self._I()

    def reservoir(self, feat=None, injected=None) -> float:
        """Sub-grid-held water [m^3]: the t0 seed for disperse (everything else goes host-ward
        live); 0 for drain (pure live prescription, no seed)."""
        if self.direction == "drain":
            return 0.0
        return self.seed_I * self._p_len
