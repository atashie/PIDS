"""Module 4 (§E) Phase-4 Task 6: the WELL-INDEX coupled-embedding scheme (deployment regime).

The exchange bridges wall->cell through the MEASURED discrete well index of the P1 ridge:

    q_per_length = WI * [Phi(H_f) - Phi(psi_Gamma)],     WI = 2*pi / ln(r_0(h) / r_w),
    r_0(h) = R0_OVER_H_P1 * h = 0.1986 h   (measured, scratch/m4_phase4_well_index.py)

with psi_Gamma the ON-Gamma discrete host value -- no shell heuristic; WI is a constant coefficient
(the nonlinearity lives in the differentiable kirchhoff_ufl), so there is nothing to lag and no
far-field read for failure-mode-1 to corrupt; refinement-robustness comes from r_0 scaling with h.

REGIME GUARD (analyzed 2026-06-10, plan course-correction): this scheme is the POSITIVE-WI
deployment regime ONLY (h > ~5 r_w, i.e. field grids around a 5 cm feature). For r_0 <= ~1.1 r_w
the bridge is negative-log and the BE transient has a REPELLING fixed point at handover (runaway
backflow -- predicted analytically, not run); that resolved-wall regime needs a different coupling
(immersed footprint / capped bridge) and is a separate, explicitly-flagged design task. The
constructor REFUSES the wrong regime rather than silently degrading.

EARLY SUB-GRID TRANSIENT: while the Green-Ampt front R(I) = sqrt(r_w^2 + 2 r_w I/dtheta) is inside
the lattice's measured log-fidelity radius 2h, the validated clock supplies the flux (drive =
psi_Gamma, which IS the host state) and ALL uptake goes to an EXPLICIT reservoir (host receives
nothing). Handover at I >= I_fill = dtheta*((2h)^2 - r_w^2)/(2 r_w). NO free constant: 2h is the
measured fidelity radius (Task-5: far field = analytic log beyond ~2h, <1.1%).

RESERVOIR RELEASE after handover ("tau" variant): rate = sgn * res/tau with tau = t_handover - t0
(the time the sub-grid took to fill is the scale on which its content relaxes into the host;
derived, no free constant). "none" variant = ledger-only control (expected to undershoot late by
~I_fill). Ledger identity (asserted by the harness): I_total*p*len == sgn*injected + reservoir.

SWEEP GEOMETRY (R=40 r_w, L/r_w = sqrt(pi*1599) = 70.9): h/r_w = 70.9/n -> positive-WI needs n <= 12
(n=14 lands ON the r_0 = r_w degeneracy; n >= 16 is the negative regime); the clock fraction
I_fill/I_max = ((2h)^2 - r_w^2)/(R^2 - r_w^2) = {35, 20, 12.5, 8.6}% at n = {6, 8, 10, 12} caps the
coarse end. The deployment sweep is therefore n in {6, 8, 10, 12} (h/r_w 11.8 -> 5.9).

DRAIN LEG STATUS (measured 2026-06-11): HONEST FAIL of the gate as-is -- relL2 10.9% (n=8) /
16.8% (n=12), DEGRADING, early-mid over-delivery up to +120%. Mechanism: the desaturation front
stays SUB-CELL for the whole 20-d window (R_f ~ 1 cell at the end), so psi_Gamma reads the wet bulk
and the WI bridge misses the growing sub-cell desaturated-annulus resistance; the crossing handover
fires immediately (the throttle clock plateaus 80x low for closed domains -- it was fitted on OPEN
0.5-m refs). NO existing a-priori sub-grid form fits the closed deployment drain: throttle 98%,
cyl+S_des 42% under, cyl+S_sorp 109% over (the implied desorptivity ~0.74*S_sorp would be a knob
fitted to the gate -- forbidden by pre-registration). The missing piece is the KNOWN Phase-3 gap
(no a-priori desorptivity; [[pids-drain-usecase]]) now extended to closed-domain conditions. The
coupling ARCHITECTURE is sound even here (the host-side bulk response is why a wrong sub-grid model
still lands within 11-17%); the drain sub-grid closure is open research -- characterized, not
claimed, not fitted.

Run (deployment sweep, LOAM disperse, R=40 r_w):
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scratch/m4_phase4_wi_probe.py sweep
Plan: docs/plans/2026-06-10-m4-phase4-coupled-embedding-plan.md (Task 6).
"""
from __future__ import annotations

import numpy as np

from pids_forward.physics.sorptive_closure import (
    F_cylindrical, F_throttle, throttle_params, rel_l2, R0_OVER_H_P1)

R_W = 0.05


class WellIndexScheme:
    """Host-controlled exchange: constant positive-WI bridge + early clock-to-reservoir phase."""

    def __init__(self, direction="disperse", release="tau"):
        assert release in ("tau", "none")
        self.direction = direction
        self.release = release
        self._sgn = 1.0 if direction == "disperse" else -1.0

    def setup(self, feat, soil, ctx):
        self.soil, self.feat = soil, feat
        h = ctx["h"]
        self.r0 = R0_OVER_H_P1 * h
        if self.r0 <= 1.1 * R_W:
            raise ValueError(
                f"WellIndexScheme: r_0={self.r0:.4f} <= 1.1*r_w -- the negative-log/degenerate "
                f"bridge regime (h={h:.3f} = {h/R_W:.1f} r_w). The transient is unstable here by "
                f"construction; use the (future) resolved-wall coupling instead.")
        self.WI = 2.0 * np.pi / np.log(self.r0 / R_W)     # > 0 in this regime
        self.h_f = 0.0 if self.direction == "disperse" else -1.0
        self._g = feat._gamma_dofs
        if self.direction == "disperse":
            self.S, self.dth = feat.S_disp, feat.dth_disp
            self.dPhi_ref = feat.dPhi_ref_disp
            self.F = F_cylindrical
        else:
            self.S, self.dth = feat.S_drain, feat.dth_drain
            self.dPhi_ref = feat.dPhi_ref_drain
            z0, k = throttle_params(self.dth)
            self.F = lambda z: F_throttle(z, z0, k)
        two_h = 2.0 * h
        self.I_fill = self.dth * (two_h ** 2 - R_W ** 2) / (2.0 * R_W)
        self.I_clock = self.S * np.sqrt(ctx["t0"])        # seeded contact age (the refs' I[0])
        self.res_vol = self.I_clock * feat._perimeter * feat.length   # sub-grid water (>=0 both dirs)
        self.inj = 0.0                                     # cumulative host exchange (signed)
        self._in_clock = self.I_clock < self.I_fill
        self._t0 = ctx["t0"]
        self.t_handover, self.tau, self._rel_rate = None, None, 0.0

    def pre_step(self, feat, psi, t):
        feat.Omega.x.array[:] = 0.0
        self._rel_rate = 0.0
        if self._in_clock:
            feat.Omega.x.scatter_forward()
            return 0.0
        feat.Omega.x.array[self._g] = self.WI / feat._perimeter
        feat.Omega.x.scatter_forward()
        if self.release == "tau" and self.tau and self.res_vol > 1e-15:
            self._rel_rate = self._sgn * self.res_vol / self.tau   # into the host (signed)
        return self._rel_rate

    def post_step(self, feat, psi, t, dt):
        if not self._in_clock:
            self.inj += feat.host_sorptive_flux(psi) * dt
            if self._rel_rate != 0.0:
                self.inj += self._rel_rate * dt
                self.res_vol -= abs(self._rel_rate) * dt
                if self.res_vol < 1e-15:
                    self.res_vol = 0.0
            return
        # clock phase: advance I from the ON-GAMMA drive; all uptake -> the explicit reservoir
        psi_g = float(psi.x.array[self._g].mean())
        lo, hi = min(psi_g, self.h_f), max(psi_g, self.h_f)
        scale = self.soil.kirchhoff(lo, hi) / self.dPhi_ref
        I, hh = self.I_clock, dt / 200
        for _ in range(200):
            Ic = max(I, 1e-300)
            I = I + hh * (self.S ** 2 / (2.0 * Ic) * self.F(Ic / (self.dth * R_W)) * scale)
        self.res_vol += (I - self.I_clock) * feat._perimeter * feat.length
        self.I_clock = I
        if I >= self.I_fill:
            self._in_clock = False
            self.t_handover = t + dt
            self.tau = max(self.t_handover - self._t0, 1e-12)

    def I_total(self, feat):
        return (self._sgn * self.inj + self.res_vol) / (feat._perimeter * feat.length)

    def reservoir(self, feat, injected):
        return self.res_vol


class RateClockWIScheme:
    """v2 (designed from the v1 diagnostic, 2026-06-10): v1 held the sub-grid era's water in an
    off-host reservoir, so at handover the lattice sat at psi_i -- OFF the consistent manifold --
    and the switched-on potential-driven exchange stalled while the lattice spun up (measured:
    emb/ref crashes 0.99 -> 0.556 right after handover, recovering only by the window end).

    v2 fix (structural, no knob): during the sub-grid era the clock flux is injected as a
    RATE-PRESCRIBED ridge source (a prescribed rate cannot collapse -- there is no Omega*dPhi
    feedback to corrupt), so the lattice continuously builds its OWN consistent response to the flux
    history; at handover (front = 2h, the measured fidelity radius) the discrete field is already
    consistent and the WI exchange continues seamlessly. NO reservoir: every drop goes host-ward
    immediately (the harness ledger sees reservoir = the tiny t0 seed only, S*sqrt(t0) ~ 0.01% of
    I_max, water that was already in the ground at t0).

    The clock era's FRONT-RING read (vertices one cell ahead of the Green-Ampt front R_f(I)): the
    rate is scaled by the live Kirchhoff drop between H_f and the ring -- a saturation safeguard
    and second-order correction (<=~2% on the committed legs; 2026-06-11 review attack a measured
    that NO committed leg discriminates it from a constant drive -- the RefB pulse lands after
    handover, so no clock-era host-control claim rests on this read). The ring MOVES with the front
    (unlike the retracted fixed shell) and exists only during the sub-grid era; host control is
    established for the WI era, which carries all discriminating signal."""

    def __init__(self, direction="disperse"):
        self.direction = direction
        self._sgn = 1.0 if direction == "disperse" else -1.0

    def setup(self, feat, soil, ctx):
        self.soil, self.feat = soil, feat
        h = ctx["h"]
        self.h = h
        self.r0 = R0_OVER_H_P1 * h
        if self.r0 <= 1.1 * R_W:
            raise ValueError(f"RateClockWIScheme: negative/degenerate bridge regime "
                             f"(r_0={self.r0:.4f}, h={h/R_W:.1f} r_w) -- refused.")
        self.WI = 2.0 * np.pi / np.log(self.r0 / R_W)
        self.h_f = 0.0 if self.direction == "disperse" else -1.0
        self._g = feat._gamma_dofs
        if self.direction == "disperse":
            self.S, self.dth = feat.S_disp, feat.dth_disp
            self.dPhi_ref, self.F = feat.dPhi_ref_disp, F_cylindrical
        else:
            self.S, self.dth = feat.S_drain, feat.dth_drain
            self.dPhi_ref = feat.dPhi_ref_drain
            z0, k = throttle_params(self.dth)
            self.F = lambda z: F_throttle(z, z0, k)
        self.I_fill = self.dth * ((2 * h) ** 2 - R_W ** 2) / (2.0 * R_W)
        self.seed = self.S * np.sqrt(ctx["t0"])
        self.inj = 0.0
        self._in_clock = self.seed < self.I_fill
        self.t_handover = None
        # vertex radii around the line (for the moving front ring)
        xc = feat.V.tabulate_dof_coordinates()
        Lc = ctx["L"] / 2.0
        self._rho = np.hypot(xc[:, 1] - Lc, xc[:, 2] - Lc)
        self._p_len = feat._perimeter * feat.length

    def _I(self):
        return self.seed + self._sgn * self.inj / self._p_len

    def _clock_rate(self, psi):
        """[m^3/day] prescribed ridge source from the clock at the current state, front-ring drive."""
        I = max(self._I(), 1e-12)
        R_f = np.sqrt(R_W ** 2 + 2.0 * R_W * I / self.dth)
        ring = (self._rho >= R_f + 0.5 * self.h) & (self._rho <= R_f + 1.5 * self.h)
        psi_far = float(psi.x.array[ring].mean()) if np.any(ring) else float(psi.x.array.mean())
        lo, hi = min(psi_far, self.h_f), max(psi_far, self.h_f)
        scale = self.soil.kirchhoff(lo, hi) / self.dPhi_ref
        dIdt = self.S ** 2 / (2.0 * I) * self.F(I / (self.dth * R_W)) * scale
        return self._sgn * dIdt * self._p_len

    def pre_step(self, feat, psi, t):
        feat.Omega.x.array[:] = 0.0
        self._last_rate = 0.0
        if self._in_clock:
            rate = self._clock_rate(psi)
            if self.direction == "drain":
                # DRAIN handover = the CROSSING: the throttle clock self-terminates (its plateau
                # ~0.026 m is far below any I_fill at deployment scale, the throttle being an
                # OPEN-domain near-field fit), so the sub-grid era ends when its rate decays to the
                # quasi-steady WI rate at the current ridge state -- the natural intersection, no
                # free constant. (Disperse keeps the front=2h geometric handover, validated.)
                psi_g = float(psi.x.array[self._g].mean())
                qWI_len = self.WI * self.soil.kirchhoff(min(psi_g, self.h_f), max(psi_g, self.h_f))
                if abs(rate) / feat.length <= abs(qWI_len):
                    self._in_clock = False
                    self.t_handover = t
            if self._in_clock:
                feat.Omega.x.scatter_forward()
                self._last_rate = rate
                return rate
        feat.Omega.x.array[self._g] = self.WI / feat._perimeter
        feat.Omega.x.scatter_forward()
        return 0.0

    def post_step(self, feat, psi, t, dt):
        if self._in_clock:
            self.inj += self._last_rate * dt    # exactly what the harness injected this step
            if self._I() >= self.I_fill:
                self._in_clock = False
                self.t_handover = t + dt
        else:
            self.inj += feat.host_sorptive_flux(psi) * dt

    def I_total(self, feat):
        return self._I()

    def reservoir(self, feat, injected):
        return self.seed * self._p_len           # the t0 seed water (never injected; ~0.01% I_max)


if __name__ == "__main__":
    import sys
    from scratch.m4_phase4_embedded_harness import run_embedded
    refA = np.load("scratch/m4_phase4_refA_disperse.npz")   # the regenerated (R40-bearing) tables
    which = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if which == "v2":
        t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
        i_max = float(refA["LOAM_R40_Imax"])
        print(f"RATE-CLOCK+WI v2 vs Ref A(40 r_w) LOAM disperse (I_max={i_max:.4e}):")
        for n in (6, 8, 10, 12):
            sch = RateClockWIScheme("disperse")
            out = run_embedded(sch, "LOAM", 40 * R_W, n, t)
            if out is None:
                print(f"  n={n}: DT COLLAPSE", flush=True)
                continue
            print(f"  n={n:2d} (h={out['h']/R_W:.1f}r_w, handover t={sch.t_handover}): "
                  f"relL2={rel_l2(out['I'], I_ref):.1%}  end I/ref={out['I'][-1]/I_ref[-1]:.3f}",
                  flush=True)
            prof = " ".join(f"{a/b:.3f}" for a, b in zip(out["I"][::4], I_ref[::4]))
            print(f"        emb/ref profile (every 4th): {prof}", flush=True)
    elif which == "v2b":
        # the HISTORY leg at deployment scale: v2 vs Ref B-40 (the host-control make-or-break)
        refB = np.load("scratch/m4_phase4_refB40_disperse.npz")
        t, I_ref = refB["LOAM_t"], refB["LOAM_I"]
        band = tuple(refB["LOAM_band"])
        pulse = (float(refB["LOAM_t_pulse"][0]), float(refB["LOAM_t_pulse"][1]),
                 float(refB["LOAM_V_pulse_per_wall_area"]) * R_W * 2 * np.pi)
        print(f"RATE-CLOCK+WI v2 vs Ref B-40 (pulse {pulse[0]}-{pulse[1]} d, band {band}):")
        for n in (8, 12):
            out = run_embedded(RateClockWIScheme("disperse"), "LOAM", 40 * R_W, n, t,
                               pulse=pulse, pulse_band=band)
            if out is None:
                print(f"  n={n}: DT COLLAPSE", flush=True)
                continue
            print(f"  n={n:2d}: relL2={rel_l2(out['I'], I_ref):.1%}  "
                  f"end I/ref={out['I'][-1]/I_ref[-1]:.3f}", flush=True)
            prof = " ".join(f"{a/b:.3f}" for a, b in zip(out["I"][::4], I_ref[::4]))
            print(f"        emb/ref (every 4th): {prof}", flush=True)
    elif which == "v2ctl":
        # the OVER-CORRECTION control: the early-truncated R40 window, where the offline clock is
        # nearly right -- an embedded scheme that over-corrects (over-throttles) fails here
        t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
        i_max = float(refA["LOAM_R40_Imax"])
        K = int(np.searchsorted(I_ref, 0.3 * i_max))
        tc, Ic = t[:K], I_ref[:K]
        from pids_forward.physics.sorptive_closure import sorptive_clock
        S, dth = float(refA["LOAM_S"]), float(refA["LOAM_dtheta"])
        clk = sorptive_clock(tc, S, dth, R_W, F_cylindrical)
        print(f"CONTROL (R40 truncated to I<0.3 I_max, {K} samples, t_end={tc[-1]:.2f} d): "
              f"clock relL2={rel_l2(clk, Ic):.1%} (the control property)")
        for n in (8, 12):
            out = run_embedded(RateClockWIScheme("disperse"), "LOAM", 40 * R_W, n, tc)
            if out is not None:
                print(f"  v2 n={n:2d}: relL2={rel_l2(out['I'], Ic):.1%}", flush=True)
    elif which == "v2d":
        # the DRAIN leg at deployment scale (crossing handover)
        refD = np.load("scratch/m4_phase4_refD40_drain.npz")
        t, I_ref = refD["LOAM_t"], refD["LOAM_I"]
        print(f"RATE-CLOCK+WI v2 DRAIN vs refD40 (I_end={I_ref[-1]:.4e} m):")
        for n in (8, 12):
            sch = RateClockWIScheme("drain")
            out = run_embedded(sch, "LOAM", 40 * R_W, n, t, direction="drain")
            if out is None:
                print(f"  n={n}: DT COLLAPSE", flush=True)
                continue
            print(f"  n={n:2d} (handover t={sch.t_handover}): relL2={rel_l2(out['I'], I_ref):.1%}  "
                  f"end I/ref={out['I'][-1]/I_ref[-1]:.3f}", flush=True)
            prof = " ".join(f"{a/b:.3f}" for a, b in zip(out["I"][::4], I_ref[::4]))
            print(f"        emb/ref (every 4th): {prof}", flush=True)
    elif which == "v2diag":
        # the deep-bend dip: full per-sample profile at n=12
        t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
        sch = RateClockWIScheme("disperse")
        out = run_embedded(sch, "LOAM", 40 * R_W, 12, t)
        print(f"handover t={sch.t_handover:.3f} d")
        print("    t [d]      emb/ref")
        for i in range(16, len(t)):
            print(f"  {t[i]:9.3e}   {out['I'][i]/I_ref[i]:6.3f}")
    elif which == "diag":
        # deviation PROFILE at n=12, tau release: where does the 17% live?
        t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
        sch = WellIndexScheme("disperse", release="tau")
        out = run_embedded(sch, "LOAM", 40 * R_W, 12, t)
        print(f"handover t={sch.t_handover}, I_fill={sch.I_fill:.4f} "
              f"({sch.I_fill/float(refA['LOAM_R40_Imax']):.1%} of I_max)")
        print("    t [d]      I_emb       I_ref      emb/ref   reservoir")
        for i in range(0, len(t), 2):
            print(f"  {t[i]:9.3e}  {out['I'][i]:.4e}  {I_ref[i]:.4e}   {out['I'][i]/I_ref[i]:6.3f}"
                  f"   {out['reservoir'][i]:.2e}")
    elif which == "sweep":
        t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
        i_max = float(refA["LOAM_R40_Imax"])
        print(f"WELL-INDEX scheme vs Ref A(40 r_w) LOAM disperse  (I_max={i_max:.4e}, "
              f"EMBEDDED_TOL=0.10):")
        for rel in ("tau", "none"):
            for n in (6, 8, 10, 12):
                out = run_embedded(WellIndexScheme("disperse", release=rel), "LOAM",
                                   40 * R_W, n, t)
                if out is None:
                    print(f"  rel={rel} n={n}: DT COLLAPSE", flush=True)
                    continue
                print(f"  rel={rel} n={n:2d} (h={out['h']/R_W:.1f}r_w): "
                      f"relL2={rel_l2(out['I'], I_ref):.1%}  end I/ref={out['I'][-1]/I_ref[-1]:.3f}  "
                      f"res_end={out['reservoir'][-1]:.2e}", flush=True)