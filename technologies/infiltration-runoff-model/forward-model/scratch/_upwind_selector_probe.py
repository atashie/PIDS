"""P1-B3 probe: sweep the smoothed-upwind selector width ``eps_H`` and record the trade.

Empirical spike question (plan docs/plans/2026-06-14-overland-convergent-flow-P1.md, Part B, B3:
"smoothed selector vs semismooth -- decide empirically"). The smoothed C1 selector
``w = 1/2(1 + tanh((H_i-H_j)/eps_H))`` is only CONDITIONALLY monotone: B2 found the default
``eps_H = 1e-3`` holds strict ``d >= 0`` on STEEP fronts (drop >> eps_H) but admits a small
centered flux -> ~1.5 mm front undershoot on MILD ~2% slopes (drop comparable to eps_H). The B5
tilted-V has BOTH a 5% headwall and a 2% valley, so the eps_H chosen here decides whether O1 is
positive on the V.

This probe sweeps ``eps_H in {1e-2, 1e-3, 1e-4, 1e-5, 1e-6}`` over three adversarial 1-D
scenarios and prints ONE table per scenario plus a combined verdict:

  (1) STEEP front  -- 5% slope, sharp Gaussian peak 0.25, 20 m / 60 cells, t_end 0.03 day
                      (the exact B2 ``test_front_advance_positive_without_limiter_1d`` scenario).
  (2) MILD front   -- 2% slope, sharp mound peak 0.25, 20 m / 60 cells, t_end 0.03 day
                      (the regime B2 characterized at ~1.5-1.6 mm undershoot; == the B5 valley).
  (3) KINEMATIC    -- tilted plane + steady rain + normal-depth outlet (the B2
                      ``test_kinematic_wave_plane_hydrograph_1d`` reference): equilibrium accuracy.

Metrics, per eps_H:
  * Newton health   -- total SNES iters over the march, mean iters/step, and the converged-reason
                       HISTOGRAM (reason 2/3 = residual-tested clean; reason 4 = SNORM stagnation
                       floor; any negative reason = a DIVERGED step). "Robust Newton" = the march
                       completes (no step fails -> dt-collapse) AND no blow-up (all-finite, no
                       negative reasons). Too-sharp a tanh approaches the non-smooth ``sign`` and
                       can hurt the FD-Jacobian Newton -- that is what we are watching for.
  * Positivity      -- run-minimum ``d`` over the WHOLE march (the headline). >= -1e-12 == strict
                       monotone (no limiter on this class). The undershoot magnitude (-min, in mm)
                       is reported so the trade is quantitative.
  * Accuracy/smear  -- (fronts) the wetted-front POSITION at t_end (x where d crosses 2% of peak)
                       and the post-run peak depth, to see if a sharper selector moves the front
                       or sharpens/diffuses the profile. (kinematic) the outlet-depth ratio
                       d_outlet/d_eq and the mass-balance ratio outflow/(r*L) at equilibrium.

The probe is read-only (no engine edit); the chosen eps_H + the trade table are recorded in the
module docstring of overland_upwind.py by the B3 task. Run (from forward-model/, WSL pids-fem):
  PATH=.../pids-fem/bin:$PATH OMP_NUM_THREADS=1 ... PYTHONPATH=. python scratch/_upwind_selector_probe.py
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.overland_upwind import UpwindOverlandProblem

N_MAN = 0.03  # Manning roughness (SI s.m^{-1/3}); same smooth-ish plane as the B2 suite.
EPS_H_SWEEP = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def _march(prob, t_end, dt0=1e-4, dt_max=5e-3, grow=1.5, cut=0.5, dt_min=1e-9,
           target_low=3, target_high=8, shrink=0.7, max_steps=200000):
    """Adaptive backward-Euler march (mirrors tests/test_overland_upwind.py::_march).

    Returns (t_reached, nsteps, min_d_over_run, reasons, total_iters, n_failed_steps).
    ``n_failed_steps`` counts step() calls that did NOT converge (dt is then cut + retried);
    a nonzero count or a dt that collapses below dt_min flags a Newton-robustness problem.
    """
    t, dt, nsteps = 0.0, dt0, 0
    min_d = float(prob.d.x.array.min())
    reasons, total_iters, n_failed = [], 0, 0
    while t < t_end - 1e-12:
        if nsteps >= max_steps:
            raise RuntimeError(f"_march exceeded max_steps at t={t:.4g} (eps_H robustness)")
        h = min(dt, t_end - t)
        converged, iters = prob.step(h)
        reasons.append(prob.last_reason)
        total_iters += int(iters)
        if converged:
            t += h
            nsteps += 1
            min_d = min(min_d, float(prob.d.x.array.min()))
            if iters <= target_low:
                dt = min(dt * grow, dt_max)
            elif iters >= target_high:
                dt *= shrink
        else:
            n_failed += 1
            dt = h * cut
            if dt < dt_min:
                raise RuntimeError(
                    f"_march: dt {dt:.2e} < dt_min at t={t:.4g} (eps_H Newton NOT robust)")
    return t, nsteps, min_d, reasons, total_iters, n_failed


def _reason_hist(reasons):
    """Compact {reason: count} string, e.g. '2:3 4:51'."""
    vals, counts = np.unique(np.array(reasons, dtype=int), return_counts=True)
    return " ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))


def _front_scenario(eps_H, slope, label):
    """Run a wet/dry-front advance at the given bed slope; return a metrics dict.

    Same 20 m / 60 cell / sharp-Gaussian (peak 0.25, sigma 1.5) / t_end 0.03 day setup as the B2
    front test, differing only in the bed slope (5% = steep, 2% = mild). The front advances
    DOWNSLOPE into dry ground -- the regime that drives undershoot.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 60, [0.0, 20.0])
    prob = UpwindOverlandProblem(msh, n_man=N_MAN, eps_H=eps_H)
    prob.set_topography(lambda x: slope * (20.0 - x[0]))
    prob.set_initial_condition(lambda x: 0.25 * np.exp(-((x[0] - 7.0) / 1.5) ** 2))
    w0 = prob.total_water()

    coords = prob.V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(coords)
    xs = coords[order]
    d_init = prob.d.x.array[order].copy()
    wet = 0.02 * 0.25  # 2%-of-peak wetted threshold
    front_pre = float(xs[d_init > wet].max())

    failed = False
    try:
        t, nsteps, min_d, reasons, tot_it, n_fail = _march(prob, t_end=0.03)
    except RuntimeError as exc:
        failed = True
        return dict(label=label, eps_H=eps_H, failed=True, why=str(exc))

    ds = prob.d.x.array[order]
    front_post = float(xs[ds > wet].max()) if np.any(ds > wet) else float("nan")
    peak_post = float(ds.max())
    mass_drift = abs(prob.total_water() - w0) / max(1.0, abs(w0))
    neg_reason = any(r < 0 for r in reasons)
    return dict(
        label=label, eps_H=eps_H, failed=False,
        nsteps=nsteps, tot_it=tot_it, mean_it=tot_it / max(1, len(reasons)),
        hist=_reason_hist(reasons), neg_reason=neg_reason, n_fail=n_fail,
        min_d=min_d, undershoot_mm=max(0.0, -min_d) * 1e3,
        front_pre=front_pre, front_post=front_post, advance=front_post - front_pre,
        peak_post=peak_post, mass_drift=mass_drift,
        finite=bool(np.all(np.isfinite(prob.d.x.array))),
    )


def _kinematic_scenario(eps_H):
    """Steady rain on a tilted plane + normal-depth outlet -> kinematic equilibrium accuracy.

    The exact B2 ``test_kinematic_wave_plane_hydrograph_1d`` setup (L=100, S0=0.01, n=0.10,
    r=0.2, nc=50, t_end 0.25 day). Returns the outlet-depth ratio d_outlet/d_eq, the mass-balance
    ratio outflow/(r*L), Newton health, and whether the rising limb stayed monotone -- so a sharper
    selector that smears/oscillates the limb shows up.
    """
    L, S0, n, r = 100.0, 0.01, 0.10, 0.2
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, L])
    prob = UpwindOverlandProblem(msh, n_man=n, eps_H=eps_H)
    prob.set_topography(lambda x: S0 * (L - x[0]))
    prob.set_initial_condition(lambda x: 0.0 * x[0])
    prob.add_rain(r)
    prob.add_outflow_bc(lambda x: np.isclose(x[0], L), slope=S0)

    hydro, reasons, tot_it = [], [], 0
    t, dt, t_end, next_rec, n_fail = 0.0, 1e-4, 0.25, 0.025, 0
    failed = False
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        converged, iters = prob.step(h)
        reasons.append(prob.last_reason)
        tot_it += int(iters)
        if not converged:
            n_fail += 1
            dt = h * 0.5
            if dt < 1e-9:
                failed = True
                break
            continue
        t += h
        if t >= next_rec - 1e-12:
            hydro.append(prob.outflow_rate())
            next_rec += 0.025
        if iters <= 3:
            dt = min(dt * 1.5, 5e-3)
        elif iters >= 8:
            dt *= 0.7
    if failed:
        return dict(eps_H=eps_H, failed=True)

    hydro = np.array(hydro)
    r_si = r / 86400.0
    d_eq = (r_si * L * n / np.sqrt(S0)) ** (3.0 / 5.0)
    coords = prob.V.tabulate_dof_coordinates()[:, 0]
    outlet = int(np.argmin(np.abs(coords - L)))
    d_out = float(prob.d.x.array[outlet])
    monotone = bool(np.all(np.diff(hydro) >= -1e-2 * (r * L))) if hydro.size > 1 else True
    return dict(
        eps_H=eps_H, failed=False,
        nsteps=len(reasons) - n_fail, tot_it=tot_it, mean_it=tot_it / max(1, len(reasons)),
        hist=_reason_hist(reasons), neg_reason=any(rr < 0 for rr in reasons), n_fail=n_fail,
        depth_ratio=d_out / d_eq, mass_ratio=float(prob.outflow_rate()) / (r * L),
        d_out=d_out, d_eq=d_eq, monotone=monotone, min_d=float(prob.d.x.array.min()),
    )


def _print_front_table(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'eps_H':>8} | {'nstep':>5} {'totit':>6} {'mit':>5} {'reason-hist':>14} "
          f"{'nfail':>5} | {'min_d':>11} {'undsh_mm':>8} | {'front_x':>8} {'adv':>6} {'peak':>7}")
    print("-" * 116)
    for r in rows:
        if r.get("failed"):
            print(f"{r['eps_H']:>8.0e} | *** Newton NOT robust: {r.get('why','')[:80]}")
            continue
        flag = "  <-- DIVERGED reason!" if r["neg_reason"] else ("  (nonfinite!)" if not r["finite"] else "")
        print(f"{r['eps_H']:>8.0e} | {r['nsteps']:>5d} {r['tot_it']:>6d} {r['mean_it']:>5.1f} "
              f"{r['hist']:>14} {r['n_fail']:>5d} | {r['min_d']:>11.3e} {r['undershoot_mm']:>8.3f} | "
              f"{r['front_post']:>8.2f} {r['advance']:>6.2f} {r['peak_post']:>7.4f}{flag}")
    print(f"  (mass_drift across the sweep, max): {max(r.get('mass_drift', 0.0) for r in rows if not r.get('failed')):.2e}")


def _print_kinematic_table(rows):
    print(f"\n=== KINEMATIC plane (L=100, S0=0.01, n=0.10, r=0.2): equilibrium accuracy ===")
    print(f"{'eps_H':>8} | {'nstep':>5} {'totit':>6} {'mit':>5} {'reason-hist':>14} {'nfail':>5} | "
          f"{'d/d_eq':>7} {'Qout/rL':>8} {'mono':>5} {'min_d':>11}")
    print("-" * 104)
    for r in rows:
        if r.get("failed"):
            print(f"{r['eps_H']:>8.0e} | *** Newton NOT robust (dt collapsed)")
            continue
        flag = "  <-- DIVERGED reason!" if r["neg_reason"] else ""
        print(f"{r['eps_H']:>8.0e} | {r['nsteps']:>5d} {r['tot_it']:>6d} {r['mean_it']:>5.1f} "
              f"{r['hist']:>14} {r['n_fail']:>5d} | {r['depth_ratio']:>7.3f} {r['mass_ratio']:>8.3f} "
              f"{str(r['monotone']):>5} {r['min_d']:>11.3e}{flag}")


def main():
    print("P1-B3 smoothed-upwind selector-width (eps_H) sweep")
    print("=" * 70)
    print(f"sweep: eps_H in {EPS_H_SWEEP}  (m)   n_man={N_MAN}")
    print("Newton-robust := march completes (nfail tolerated via dt-cut) AND all reasons > 0 AND finite.")
    print("Positivity headline := run-min d over the whole march (>= -1e-12 == strict monotone).")

    steep = [_front_scenario(e, slope=0.05, label="steep5%") for e in EPS_H_SWEEP]
    mild = [_front_scenario(e, slope=0.02, label="mild2%") for e in EPS_H_SWEEP]
    kin = [_kinematic_scenario(e) for e in EPS_H_SWEEP]

    _print_front_table("STEEP front (5% slope) -- B2 positivity scenario", steep)
    _print_front_table("MILD front (2% slope) -- B2 undershoot regime == B5 valley slope", mild)
    _print_kinematic_table(kin)

    # ---- combined verdict --------------------------------------------------
    print("\n" + "=" * 70)
    print("COMBINED VERDICT (per eps_H): steep min_d | mild min_d (mm) | newton ok | kin d/d_eq")
    print("-" * 70)
    for e in EPS_H_SWEEP:
        s = next(r for r in steep if r["eps_H"] == e)
        m = next(r for r in mild if r["eps_H"] == e)
        k = next(r for r in kin if r["eps_H"] == e)
        s_pos = "n/a" if s.get("failed") else f"{s['min_d']:.2e}"
        m_pos = "n/a" if m.get("failed") else f"{m['undershoot_mm']:.3f}mm"
        m_strict = (not m.get("failed")) and m["min_d"] >= -1e-12
        nok = all(
            (not r.get("failed")) and (not r.get("neg_reason", False))
            and r.get("finite", True) for r in (s, m, k))
        kdr = "n/a" if k.get("failed") else f"{k['depth_ratio']:.3f}"
        print(f"  eps_H={e:.0e}: steep min_d={s_pos:>11} | mild undsh={m_pos:>9} "
              f"(strict={m_strict!s:>5}) | newton_ok={nok!s:>5} | kin d/d_eq={kdr}")
    print("\n(read the trade: smallest eps_H that keeps newton_ok AND mild strict, w/o moving the")
    print(" kinematic d/d_eq or the front position -- that is the B3 choice.)")


if __name__ == "__main__":
    main()
