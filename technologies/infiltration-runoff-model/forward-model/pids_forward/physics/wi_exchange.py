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
correction (<=~2% on the committed legs, mechanism-present but not gate-exercised). The ~5%
residual is a WI-ERA SYSTEMATIC within the pre-registered tolerance, LOCALIZED by the 2026-06-12
instrumented diagnostic (scratch/m4_phase4_wi_residual_diag.py + the _wi_diag npzs): the
constant-WI bridge FORM is exonerated (evaluated on the RESOLVED field at r_0 it reproduces the
resolved wall rate to +0.3..0.6% through the bend, +3.4% deep-bend); the carrier is the LATTICE
RIDGE STATE sitting WETTER than the resolved field at r_0 (dPhi read -10..-18%, shrinking with n)
plus a short post-handover spin-up transient (rate -59% for t <= 2*t_h, self-healing -- the v1
crash mechanism, now bounded); the cumulative mid-curve +3-5% over-delivery is CLOCK-ERA SURPLUS
carried over handover, not WI-era over-delivery. It is NOT the offline F closure's large-zeta
bias (measured -0.9..-1.5%, wrong sign and era; review attack f). The disperse evidence does not by itself exclude passive
capacity-aware schemes (their killers are the drain legs this class refuses). Prototype + evidence:
scratch/m4_phase4_wi_probe.py, scratch/m4_phase4_embedded_harness.py,
tests/test_coupled_gate_refs.py.

THE SCHEME (no free constants -- every number is measured or derived):
* Post-handover the wall exchange is the constant-coefficient Kirchhoff bridge
      q_per_length = WI * [Phi(H_f) - Phi(psi_Gamma)],   WI = 2*pi / ln(r_0(h)/r_w),
  with r_0(h) = R0_OVER_H_P1 * h the MEASURED equivalent radius of the P1 ridge source
  (Peaceman-for-FEM; tests/test_well_index_p1.py) and psi_Gamma the ON-ridge discrete value -- no
  shell read, no lag (the coefficient is constant; the nonlinearity is the differentiable
  kirchhoff_ufl), which is what makes it refinement-robust (r_0 scales with h).
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
scratch/m4_phase4_drain_drive_diag.py. The remaining end excess decomposed EXACTLY as the
law's own +2.0% x the +4.7% first-order explicit lag of the start-state rate held over the
step; the Heun trapezoid rate in the scheme's own mass variable -- pre_step's optional dt --
removes the lag.) With the theta-mean + Heun drive the legs score 3.4/3.5/2.7% rel-L2 at end
bias 1.019/1.030/0.996 (refD40/SAND-R40/refD40-C), n-INDEPENDENT at n=8/12 -- the mass-exact
read removed the resolution dependence; what remains is the offline PSS closure's own
documented +2-4%. The host read is LOAD-BEARING: the depletion bend exists only because the
host bulk falls (fixed-drive twin fails refD40 at 74%; the original dof-mean read scored
7.4/6.9% at end 1.085). R_out = the closed/catchment equivalent radius, a physical geometry input
(ctx["R_out"], required). Omega stays 0 in drain mode (pure prescription; capacity-safe: the rate
hard-zeros as the bulk mean reaches the wall head).

SCOPE GUARDS (refusals, not silent degradation):
* POSITIVE-WI regime only for DISPERSE (r_0 > 1.1 r_w, i.e. h > ~5.5 r_w -- field grids around a
  5 cm feature). For finer meshes the bridge is negative-log and the BE transient has a REPELLING
  fixed point at handover (runaway backflow, analyzed 2026-06-10) -- a resolved-wall coupling is a
  separate design task. Refused with ValueError.
* DRAIN requires ctx["R_out"] (the PSS geometry); refused with ValueError without it.

USAGE (the step contract, harness-compatible -- scratch/m4_phase4_embedded_harness.py is the
reference driver): the host residual must include BOTH feat.sorptive_into_host(w, psi) (the WI-era
exchange; this class drives feat.Omega) AND a ridge source carrying the sub-grid-era prescribed
rate (per unit length = rate/feat.length, e.g. `-rate_c * w * feat.dGamma` with rate_c set from
pre_step's return). Per step: rate = pre_step(feat, psi, t, dt) BEFORE the solve (dt optional;
passing it gets the lag-free Heun drain rate); post_step(feat, psi, t, dt) AFTER it. Mass
identity: I_total*perimeter*length == injected + seed*perimeter*length.
"""
from __future__ import annotations

import numpy as np

from .sorptive_closure import F_cylindrical, R0_OVER_H_P1, R_W_DEFAULT


class WellIndexExchange:
    """Embedded wall exchange. DISPERSE: sub-grid rate-clock era + constant-WI Kirchhoff era
    (host-controlled in the WI era; the clock era is a prescribed-rate closure). DRAIN: the PSS
    depletion closure as a live-host-driven prescribed-rate ridge sink -- see the module docstring."""

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
                f"read here (dof/theta means, the front-ring mean, the lumped volume weights) "
                f"is rank-local and would be silently partition-dependent. The gate evidence is "
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
            import ufl
            from dolfinx import fem as _fem
            v = ufl.TestFunction(feat.V)
            w = _fem.assemble_vector(_fem.form(v * ufl.dx(
                metadata={"quadrature_rule": "vertex", "quadrature_degree": 1}))).array.copy()
            self._vol = float(w.sum())                  # the host volume (the Heun mass step)
            self._wvol = w / self._vol
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
                f"h={self.h:.3f} m = {self.h/self._r_w:.1f} r_w; need h > ~5.5 r_w). The negative-log "
                f"bridge is transiently unstable (repelling BE fixed point at handover).")
        self.WI = 2.0 * np.pi / np.log(self.r0 / self._r_w)
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
        feat.Omega.x.array[self._g] = self.WI / feat._perimeter
        feat.Omega.x.scatter_forward()
        return 0.0

    def post_step(self, feat, psi, t, dt):
        """Call AFTER an accepted solve: account the step's exchange and handle handover."""
        if self.direction == "drain":
            self.inj += self._last_rate * dt               # extracted magnitude, >= 0
            return
        if self.in_subgrid_era:
            self.inj += self._last_rate * dt               # exactly what the caller injected
            if self._I() >= self.I_fill:
                self.in_subgrid_era = False
                self.t_handover = t + dt
        else:
            self.inj += feat.host_sorptive_flux(psi) * dt

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
