"""Host-controlled sub-grid wall exchange via the measured discrete well index (Module 4 §E, Phase 4).

PRODUCTION form of the Phase-4 "rate-clock + WI" scheme, the first coupling to PASS the
discriminating coupled-embedding gate (evolving-far-field references the offline clock fails by
construction): LOAM disperse deployment legs RefA-40 = 2.2-5.7% rel-L2 across n (offline clock
26.6%, the retracted dual-scale 13-16%), RefB-40 host-history = 5.0-5.4% landing EXACTLY on the
pulse-shifted asymptote with no knowledge of the pulse (clock 34.1%), over-correction control =
2.1-6.0% (clock right there by design). Prototype + evidence: scratch/m4_phase4_wi_probe.py,
scratch/m4_phase4_embedded_harness.py, tests/test_coupled_gate_refs.py. NOTE: "coupled" status is
prototype-validated; the Phase-4 protocol's adversarial review is the remaining gate before the
claim is final.

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
  seamless. The clock is HOST-CONTROLLED through its drive: the Kirchhoff drop to the FRONT RING
  (off-ridge vertices one cell ahead of R_f -- it moves with the front; it is NOT the retracted
  fixed shell, which reduced to the offline clock and failed the gate at 13-39%).
* All uptake is host-ward immediately -- no sub-grid reservoir; the only ledger remainder is the
  t0 seed S*sqrt(t0) (water already in the ground at seeding, ~0.01% of a deployment capacity).

SCOPE GUARDS (refusals, not silent degradation):
* DISPERSE ONLY. The drain direction's sub-grid closure for closed/deployment domains is OPEN
  research: the desaturation front stays sub-cell for whole deployment windows and NO a-priori form
  fits the resolved closed drain (open-fitted throttle 98% rel-L2, cyl+S_des 42% under, cyl+S_sorp
  109% over vs the refD40 reference) -- see scratch/m4_phase4_wi_probe.py (drain status) and the
  pids-drain-usecase memory. Refused with NotImplementedError.
* POSITIVE-WI regime only (r_0 > 1.1 r_w, i.e. h > ~5.5 r_w -- field grids around a 5 cm feature).
  For finer meshes the bridge is negative-log and the BE transient has a REPELLING fixed point at
  handover (runaway backflow, analyzed 2026-06-10) -- a resolved-wall coupling is a separate design
  task. Refused with ValueError.

USAGE (the step contract, harness-compatible -- scratch/m4_phase4_embedded_harness.py is the
reference driver): the host residual must include BOTH feat.sorptive_into_host(w, psi) (the WI-era
exchange; this class drives feat.Omega) AND a ridge source carrying the sub-grid-era prescribed
rate (per unit length = rate/feat.length, e.g. `-rate_c * w * feat.dGamma` with rate_c set from
pre_step's return). Per step: rate = pre_step(feat, psi, t) BEFORE the solve; post_step(feat, psi,
t, dt) AFTER it. Mass identity: I_total*perimeter*length == injected + seed*perimeter*length.
"""
from __future__ import annotations

import numpy as np

from .sorptive_closure import F_cylindrical, R0_OVER_H_P1, R_W_DEFAULT


class WellIndexExchange:
    """Disperse host-controlled wall exchange: sub-grid rate-clock era + constant-WI Kirchhoff era."""

    def __init__(self, direction: str = "disperse"):
        if direction != "disperse":
            raise NotImplementedError(
                "WellIndexExchange is DISPERSE-only: the drain sub-grid closure for closed domains "
                "is open research (no a-priori form fits the resolved closed drain; throttle 98%, "
                "cyl 42-109% off -- see scratch/m4_phase4_wi_probe.py and pids-drain-usecase).")
        self.direction = direction

    # -- lifecycle -------------------------------------------------------------
    def setup(self, feat, soil, ctx):
        """Bind to an EmbeddedFeature (after configure_sorptive). ctx: t0 (required, the seeding
        contact age, > 0); h (optional -- auto-measured as the nearest off-ridge vertex distance)."""
        from scipy.spatial import cKDTree
        self.soil, self.feat = soil, feat
        self._g = feat._gamma_dofs
        self._r_w = feat._r_w
        xc = feat.V.tabulate_dof_coordinates()
        gco = xc[self._g]
        # perpendicular distance of every dof to the feature line (general -- no box assumption)
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
        self.h_f_ref = 0.0                                  # the configure_sorptive wall head
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
        lo, hi = min(psi_far, self.h_f_ref), max(psi_far, self.h_f_ref)
        scale = self.soil.kirchhoff(lo, hi) / self.dPhi_ref
        dIdt = self.S ** 2 / (2.0 * I) * F_cylindrical(I / (self.dth * self._r_w)) * scale
        return dIdt * self._p_len

    def pre_step(self, feat, psi, t) -> float:
        """Call BEFORE the solve. Sets feat.Omega (the WI-era coefficient) and returns the sub-grid
        era's prescribed ridge rate [m^3/day] (0 after handover) -- the caller carries it as a ridge
        source in the host residual."""
        feat.Omega.x.array[:] = 0.0
        self._last_rate = 0.0
        if self.in_subgrid_era:
            feat.Omega.x.scatter_forward()
            self._last_rate = self._clock_rate(psi)
            return self._last_rate
        feat.Omega.x.array[self._g] = self.WI / feat._perimeter
        feat.Omega.x.scatter_forward()
        return 0.0

    def post_step(self, feat, psi, t, dt):
        """Call AFTER an accepted solve: account the step's exchange and handle handover."""
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
        return self._I()

    def reservoir(self, feat=None, injected=None) -> float:
        """Sub-grid-held water [m^3]: only the t0 seed (everything else goes host-ward live)."""
        return self.seed_I * self._p_len
