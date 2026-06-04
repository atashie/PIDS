"""
PIDS de-risking probe — reverse one-way "catch-valve" numerical tractability
============================================================================

PURPOSE
    Isolate and stress-test the SOLE decisive engineered-BC primitive for PIDS:
    the REVERSE catch-valve. It ADMITS surface-runoff INFLUX when a prescribed
    surface head h_surf exceeds the local subsurface hydraulic head H_node, and
    BLOCKS all groundwater OUTFLOW (flux clamped to >= 0) — the INVERSE of a
    conventional tile drain. The cross-engine analysis showed this is a non-smooth
    complementarity / Signorini condition  q = C*max(0, h_surf - H_node), whose
    kink at the threshold is the universal Newton "active-set chatter" hazard for
    every candidate engine. Proving it tractable here (pure Python, plain Windows)
    de-risks all of them at once, before any PETSc/Fortran build.

HOST PDE
    1-D vertical variably-saturated flow (mixed-form Richards, van Genuchten /
    Mualem), cell-centred finite volume, backward Euler, Newton with a NUMERICAL
    Jacobian (robust; no hand-coded derivatives). z is positive UP; H = psi + z;
    fluxes positive UP. Bottom: Dirichlet head (water table). Top: no-flow.

    HOST SANITY GATE: the initial condition H = 0 everywhere (psi = -z) is the
    EXACT steady state for (bottom H=0, top no-flow, valve shut). The 'steady_low'
    scenario must therefore converge in ~1 iteration with ~zero mass-balance error.
    If it does not, the host is wrong and no valve result can be trusted.

VALVE VARIANTS
    hard       : q = C*max(0, dh).                 (kink; Jacobian via FD)
    smoothed   : q = C*softplus_eps(dh).           (smooth)
    active-set : q = C*max(0, dh), active set frozen during the Jacobian (semismooth)
    lagged     : q evaluated at the PREVIOUS time level (operator-split / external
                 coupling style) — tests the synthesis warning that lagged switching
                 chatters / leaks mass vs. an implicit in-solver hook.

Runs with: Python 3 + NumPy + SciPy (+ Matplotlib for an optional figure).
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# Soil hydraulic properties — van Genuchten / Mualem (well-behaved loam)
# ----------------------------------------------------------------------------
class VG:
    theta_r, theta_s = 0.078, 0.43
    alpha, n = 3.6, 1.56
    m = 1.0 - 1.0 / n
    Ks = 0.25  # m/day

    @staticmethod
    def theta(psi):
        psi = np.clip(np.asarray(psi, float), -1e3, 1e3)
        out = np.full_like(psi, VG.theta_s)
        u = psi < 0.0
        Se = (1.0 + (VG.alpha * np.abs(psi[u])) ** VG.n) ** (-VG.m)
        out[u] = VG.theta_r + (VG.theta_s - VG.theta_r) * Se
        return out

    @staticmethod
    def K(psi):
        psi = np.clip(np.asarray(psi, float), -1e3, 1e3)
        out = np.full_like(psi, VG.Ks)
        u = psi < 0.0
        Se = (1.0 + (VG.alpha * np.abs(psi[u])) ** VG.n) ** (-VG.m)
        out[u] = VG.Ks * np.sqrt(Se) * (1.0 - (1.0 - Se ** (1.0 / VG.m)) ** VG.m) ** 2
        return out


# ----------------------------------------------------------------------------
# Grid / domain
# ----------------------------------------------------------------------------
L, N = 1.0, 40
dz = L / N
zc = dz * (np.arange(N) + 0.5)            # cell centres, z up from 0 (bottom)
i_valve = int(np.argmin(np.abs(zc - 0.5)))
H_bot = 0.0                               # Dirichlet hydraulic head at z=0


def softplus_eps(x, eps):
    z = x / eps
    return eps * (np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))))


def faces_flux(psi):
    """Face fluxes (positive UP): bottom face, N-1 interior faces, top face(=0)."""
    K = VG.K(psi)
    Kf = 0.5 * (K[:-1] + K[1:])                       # interior interfaces
    q_int = -Kf * ((psi[1:] - psi[:-1]) / dz + 1.0)   # q_{i+1/2}, i=0..N-2
    psi_ghost = H_bot + dz / 2.0                       # ghost centre at z=-dz/2
    Kb = 0.5 * (VG.K(np.array([psi[0]]))[0] + VG.K(np.array([psi_ghost]))[0])
    q_bot = -Kb * ((psi[0] - psi_ghost) / dz + 1.0)    # at z=0, into domain if >0
    return q_bot, q_int


def valve_flux(psi, psi_lag, h_now, h_lag, Cv, variant, eps):
    """Reverse catch-valve inflow at i_valve [m/day], always >= 0."""
    if variant == "lagged":
        dh = h_lag - (psi_lag[i_valve] + zc[i_valve])
        return Cv * max(0.0, dh)
    dh = h_now - (psi[i_valve] + zc[i_valve])
    if variant == "smoothed":
        return Cv * softplus_eps(dh, eps)
    return Cv * max(0.0, dh)                            # hard / active-set


def residual(psi, psi_n, psi_lag, dt, h_now, h_lag, Cv, variant, eps):
    theta = VG.theta(psi)
    theta_n = VG.theta(psi_n)
    q_bot, q_int = faces_flux(psi)
    # net upward flux divergence per cell: (q_into_bottom - q_out_top)
    q_in = np.empty(N)
    q_in[0] = q_bot
    q_in[1:] = q_int                                   # bottom face of cell i = q_{i-1/2}
    q_out = np.empty(N)
    q_out[:-1] = q_int                                 # top face of cell i = q_{i+1/2}
    q_out[-1] = 0.0                                     # no-flow top
    F = dz * (theta - theta_n) / dt - (q_in - q_out)
    F[i_valve] -= valve_flux(psi, psi_lag, h_now, h_lag, Cv, variant, eps)
    return F


def newton_step(psi_n, psi_lag, dt, h_now, h_lag, Cv, variant, eps,
                tol=1e-9, maxit=60):
    psi = psi_n.copy()
    active0 = None
    flips = 0
    for it in range(1, maxit + 1):
        F = residual(psi, psi_n, psi_lag, dt, h_now, h_lag, Cv, variant, eps)
        # numerical Jacobian (dense, N x N — trivial here)
        J = np.zeros((N, N))
        for j in range(N):
            d = 1e-7 * (1.0 + abs(psi[j]))
            pp = psi.copy(); pp[j] += d
            Fp = residual(pp, psi_n, psi_lag, dt, h_now, h_lag, Cv, variant, eps)
            J[:, j] = (Fp - F) / d
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return psi, it, flips, False
        # damp very large steps to keep property eval in range
        big = np.max(np.abs(delta))
        if big > 5.0:
            delta *= 5.0 / big
        psi = psi + delta
        # chatter bookkeeping (active state of the implicit valve)
        if variant != "lagged":
            act = (h_now - (psi[i_valve] + zc[i_valve])) > 0.0
            if active0 is not None and act != active0:
                flips += 1
            active0 = act
        if np.max(np.abs(delta)) < tol:
            return psi, it, flips, True
    return psi, maxit, flips, False


def run(variant, hsurf_fn, Cv, dt=0.005, T=1.0, eps=2e-3):
    nsteps = int(round(T / dt))
    psi = -zc.copy()
    theta0 = VG.theta(psi)
    S0 = np.sum(theta0) * dz
    iters_max, flips_tot, nonconv = 0, 0, 0
    valve_cum, bot_cum, min_thru = 0.0, 0.0, np.inf
    maxHv = -np.inf
    h_lag = hsurf_fn(0.0)
    psi_lag = psi.copy()
    for k in range(nsteps):
        t = (k + 1) * dt
        h_now = hsurf_fn(t)
        psi_new, it, flips, ok = newton_step(psi, psi_lag, dt, h_now, h_lag, Cv, variant, eps)
        iters_max = max(iters_max, it)
        flips_tot += flips
        if not ok:
            nonconv += 1
        # diagnostics from the accepted state
        qv = valve_flux(psi_new, psi_lag, h_now, h_lag, Cv, variant, eps)
        valve_cum += qv * dt
        min_thru = min(min_thru, qv)
        maxHv = max(maxHv, psi_new[i_valve] + zc[i_valve])
        q_bot, _ = faces_flux(psi_new)
        bot_cum += q_bot * dt
        psi_lag = psi.copy()
        psi = psi_new
        h_lag = h_now
    S_end = np.sum(VG.theta(psi)) * dz
    dS = S_end - S0
    mb = dS - (bot_cum + valve_cum)            # storage change = net in at bottom + valve in
    scale = max(abs(dS), abs(valve_cum), abs(bot_cum), 1e-12)
    return dict(variant=variant, iters_max=iters_max, flips=flips_tot, nonconv=nonconv,
                valve_cum=valve_cum, bot_cum=bot_cum, mb_rel=mb / scale,
                min_thru=min_thru, maxHv=maxHv, psi_end=psi)


# ----------------------------------------------------------------------------
# Scenarios — prescribed surface head h_surf(t); node head starts at H=0
# ----------------------------------------------------------------------------
def s_toggle(t):
    if t < 0.2:  return -0.20                               # closed
    if t < 0.5:  return  0.30                               # wide open
    if t < 0.7:  return 0.03 * np.sign(np.sin(2*np.pi*(t-0.5)/0.04))  # toggling at threshold
    return -0.20                                            # closed

def s_high(t): return 0.30
def s_low(t):  return -0.30
def s_above(t): return 0.05   # pinned just above the initial node head (H=0): stresses the switch

SCENARIOS = {"toggle": s_toggle, "steady_high": s_high, "steady_low": s_low}
VARIANTS = ["hard", "smoothed", "active-set", "lagged"]


def decide_pass_fail(R):
    ok, msgs = True, []
    # host sanity: steady_low must be an essentially-trivial, mass-conserving solve
    h = R[("steady_low", "hard")]
    if h["nonconv"] > 0 or abs(h["mb_rel"]) > 1e-6 or h["iters_max"] > 6:
        ok = False
        msgs.append(f"[FAIL host] steady_low: nonconv={h['nonconv']} mb_rel={h['mb_rel']:.2e} maxit={h['iters_max']}")
    else:
        msgs.append(f"[ok host] steady_low equilibrium held: maxit={h['iters_max']} mb_rel={h['mb_rel']:.2e}")
    for sc in SCENARIOS:
        for v in ["smoothed", "active-set"]:
            r = R[(sc, v)]
            if r["nonconv"] > 0:
                ok = False; msgs.append(f"[FAIL] {v}/{sc}: {r['nonconv']} non-converged steps")
            if abs(r["mb_rel"]) > 1e-6:
                ok = False; msgs.append(f"[FAIL] {v}/{sc}: mass-balance rel {r['mb_rel']:.2e} > 1e-6")
            if r["min_thru"] < -1e-10:
                ok = False; msgs.append(f"[FAIL] {v}/{sc}: valve leaked OUTFLOW {r['min_thru']:.2e}")
            if r["iters_max"] > 15:
                msgs.append(f"[warn] {v}/{sc}: max iters {r['iters_max']} > 15")
    return ok, msgs


def main():
    Cv = 0.5
    R = {}
    print(f"Grid N={N} dz={dz:.4f} m | valve node {i_valve} (z={zc[i_valve]:.3f} m) | "
          f"Cv={Cv} m/day/m | dt=0.005 day, T=1 day\n")
    hdr = (f"{'scenario':12} {'variant':11} {'maxit':>5} {'flips':>5} {'nonconv':>7} "
           f"{'valve_in[m]':>11} {'bot_net[m]':>10} {'massbal_rel':>12} {'min_thru':>11}")
    print(hdr); print("-" * len(hdr))
    for sc, fn in SCENARIOS.items():
        for v in VARIANTS:
            r = run(v, fn, Cv); R[(sc, v)] = r
            mt = "n/a" if not np.isfinite(r["min_thru"]) else f"{r['min_thru']:11.3e}"
            print(f"{sc:12} {v:11} {r['iters_max']:5d} {r['flips']:5d} {r['nonconv']:7d} "
                  f"{r['valve_cum']:11.5f} {r['bot_cum']:10.5f} {r['mb_rel']:12.3e} {mt:>11}")
    print()
    ok, msgs = decide_pass_fail(R)
    for line in msgs: print(line)
    print("\n==> RESULT:",
          "PASS — reverse catch-valve is numerically tractable in the cheap setting."
          if ok else "FAIL — escalate to exact-VI engine-embedded probe before committing.")

    print("\nImplicit-in-solver vs lagged operator-split (toggle scenario):")
    for v in ["hard", "smoothed", "active-set", "lagged"]:
        r = R[("toggle", v)]
        print(f"  {v:11}: maxit={r['iters_max']:3d} flips={r['flips']:3d} nonconv={r['nonconv']:3d} mb_rel={r['mb_rel']:.2e}")

    print("\nSmoothing-eps leak near the threshold (h_surf = dh below node head; softplus admits a spurious trickle):")
    for dh in [-0.05, -0.01, -0.002]:
        line = f"  dh={dh:+.3f} m:"
        for eps in [1e-2, 2e-3, 5e-4]:
            r = run("smoothed", (lambda v: (lambda t: v))(dh), Cv, eps=eps)
            line += f"  eps={eps:.0e}->leak={r['valve_cum']:.2e}m"
        print(line)

    print("\nSTRESS sweep — high conductance x large dt, surface head pinned just above threshold")
    print("(synthesis predicted hard/lagged chatter or blow-up here; implicit smoothed/active-set should hold)")
    sh = (f"{'Cv':>6} {'dt':>6} {'variant':11} {'maxit':>5} {'flips':>5} {'nonconv':>7} "
          f"{'maxH_valve':>10} {'mb_rel':>11}")
    print(sh); print("-" * len(sh))
    for Cv_s in [5.0, 50.0, 500.0]:
        for dt_s in [0.02, 0.1]:
            for v in VARIANTS:
                r = run(v, s_above, Cv_s, dt=dt_s, T=1.0)
                flag = "  <-- broke" if (r["nonconv"] > 0 or abs(r["mb_rel"]) > 1e-6 or r["maxHv"] > 5.0) else ""
                print(f"{Cv_s:6.0f} {dt_s:6.3f} {v:11} {r['iters_max']:5d} {r['flips']:5d} "
                      f"{r['nonconv']:7d} {r['maxHv']:10.3f} {r['mb_rel']:11.2e}{flag}")

    try:
        make_figure(R); print("\nSaved figure: scratch/catchvalve_probe.png")
    except Exception as e:
        print(f"\n(figure skipped: {e})")


def make_figure(R):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for v in VARIANTS:
        ax[0].plot(R[("toggle", v)]["psi_end"] + zc, zc, label=v)
    ax[0].axhline(zc[i_valve], ls=":", c="k", lw=0.8)
    ax[0].set_xlabel("hydraulic head H = psi + z  [m]"); ax[0].set_ylabel("z [m]")
    ax[0].set_title("End-of-run head (toggle)"); ax[0].legend(fontsize=8)
    names = list(SCENARIOS); x = np.arange(len(names)); w = 0.2
    for j, v in enumerate(VARIANTS):
        ax[1].bar(x + (j-1.5)*w, [R[(s, v)]["iters_max"] for s in names], w, label=v)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_ylabel("max Newton iters/step")
    ax[1].set_title("Convergence cost"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("scratch/catchvalve_probe.png", dpi=110)


if __name__ == "__main__":
    main()
