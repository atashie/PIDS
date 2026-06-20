"""P3 Part B (B2+B3 data-gen, streamlined to Gate B): drive the convergent dual-drain fixture through a
SHOWCASE storm (time snapshots for the Tier-3 HTML) + a brief ROBUSTNESS check (a 100-yr burst and a wet
antecedent -- conservation + capture), saving an npz for viz/make_convergent_dualdrain_html.py.

Run (pids-fem, from forward-model):  PYTHONPATH=. python -u scratch/_p3_convergent_storm_matrix.py
"""
from __future__ import annotations

import os
import time

import numpy as np

from scratch._p3_convergent_fixture import make_convergent_dualdrain, balance_residual

DATE = os.environ.get("DATE", "2026-06-20")
OUT = f"../validation/sanity/data/p3_convergent_dualdrain__{DATE}.npz"


def run_scenario(name, *, rate, z_wt, t_storm, t_recess, snapshot=False, n_out=30,
                 nx=24, ny=15, nz=4, dt_max=1e-2):
    """Drive one storm+recession scenario on the fixture; return time series (+ d(x,y) snapshots if
    snapshot). Adaptive backward-Euler, snapshots at evenly spaced out_times."""
    prob, info, _drain, _inlet = make_convergent_dualdrain(nx, ny, nz, z_wt=z_wt)
    rain = prob.add_rain(0.0)
    w0 = prob.total_water()
    topdofs = prob._top_dofs(prob.Vd)
    tc = prob.Vd.tabulate_dof_coordinates()[topdofs]
    xu = np.unique(np.round(tc[:, 0], 6))
    yu = np.unique(np.round(tc[:, 1], 6))
    ixg = np.searchsorted(xu, np.round(tc[:, 0], 6))
    iyg = np.searchsorted(yu, np.round(tc[:, 1], 6))

    t_end = t_storm + t_recess
    out_times = np.linspace(0.0, t_end, n_out)
    rec = {k: [] for k in ("t", "cum_rain", "cum_out", "cum_inlet", "cum_drain",
                           "q_inlet", "q_drain", "bal", "d_max")}
    d_grids = []
    cum_rain = [0.0]

    def snap(t):
        rec["t"].append(t)
        rec["cum_rain"].append(cum_rain[0])
        rec["cum_out"].append(prob.cum_outflow)
        rec["cum_inlet"].append(prob.cum_sinks["surface_inlet"][0])
        rec["cum_drain"].append(prob.cum_sinks["interior_drain"][0])
        rec["q_inlet"].append(prob.last_sinks["surface_inlet"][0])
        rec["q_drain"].append(prob.last_sinks["interior_drain"][0])
        rec["bal"].append(balance_residual(prob, w0, cum_rain[0]))
        rec["d_max"].append(prob.surface_depth())
        if snapshot:
            g = np.full((yu.size, xu.size), np.nan)
            g[iyg, ixg] = prob.d.x.array[topdofs]
            d_grids.append(g.copy())

    t0 = time.time()
    snap(0.0)
    k_out = 1
    t, dt, n_acc, n_rej, max_it = 0.0, 5e-4, 0, 0, 0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if t < t_storm - 1e-12 and t + h > t_storm:
            h = t_storm - t                                 # land exactly on the storm->recession edge
        if k_out < n_out and t + h > out_times[k_out]:
            h = out_times[k_out] - t
        r = rate if t < t_storm - 1e-12 else 0.0
        rain.value = r
        conv, it = prob.step(h)
        if conv:
            cum_rain[0] += r * info["top_area"] * h
            t += h
            n_acc += 1
            max_it = max(max_it, it)
            dt = min(dt * (1.5 if it <= 3 else 0.7 if it >= 8 else 1.0), dt_max)
            while k_out < n_out and t >= out_times[k_out] - 1e-12:
                snap(t)
                k_out += 1
        else:
            n_rej += 1
            dt *= 0.5
            if dt < 1e-9:
                raise RuntimeError(f"{name}: dt collapse at t={t:.5f}")
    while k_out < n_out:
        snap(t)
        k_out += 1
    run_s = time.time() - t0

    a = {k: np.asarray(v) for k, v in rec.items()}
    bal_rel = float(a["bal"].max() / max(cum_rain[0], 1e-30))
    print(f"  [{name}] rate={rate} z_wt={z_wt}: {n_acc} acc / {n_rej} rej (max_it {max_it}) in {run_s:.0f}s"
          f"  cum_rain={cum_rain[0]:.3f} out={prob.cum_outflow:.3f} inlet={a['cum_inlet'][-1]:.4f} "
          f"drain={a['cum_drain'][-1]:.4f}  bal_rel={bal_rel:.1e}  max_pond={1e3*a['d_max'].max():.1f}mm "
          f"d_min={1e3*prob.d.x.array.min():+.2f}mm", flush=True)
    return dict(name=name, rate=float(rate), z_wt=float(z_wt), t_storm=float(t_storm),
                xu=xu, yu=yu, Xc=info["Xc"], W=info["W"], Lx=info["Lx"], Ly=info["Ly"],
                d_grids=np.asarray(d_grids) if snapshot else np.zeros((0,)),
                bal_rel=bal_rel, max_pond=float(a["d_max"].max()),
                d_min=float(prob.d.x.array.min()), n_acc=n_acc, n_rej=n_rej, run_s=run_s, **a)


def main():
    print("P3 convergent dual-drain: showcase storm + robustness check", flush=True)
    t0 = time.time()
    show = run_scenario("typical/normal", rate=0.35, z_wt=1.05, t_storm=0.05, t_recess=0.05,
                        snapshot=True, n_out=30)
    burst = run_scenario("100yr-burst/normal", rate=1.2, z_wt=1.05, t_storm=0.03, t_recess=0.02)
    wet = run_scenario("typical/wet", rate=0.35, z_wt=1.20, t_storm=0.05, t_recess=0.0)

    out = {f"show_{k}": v for k, v in show.items() if not isinstance(v, str)}
    out["show_name"] = show["name"]
    robust = [burst, wet]
    out["robust_names"] = np.array([r["name"] for r in robust])
    for key in ("rate", "z_wt", "bal_rel", "max_pond", "d_min", "n_acc", "n_rej", "run_s"):
        out[f"robust_{key}"] = np.array([r[key] for r in robust])
    out["robust_cum_rain"] = np.array([r["cum_rain"][-1] for r in robust])
    out["robust_cum_out"] = np.array([r["cum_out"][-1] for r in robust])
    out["robust_cum_inlet"] = np.array([r["cum_inlet"][-1] for r in robust])
    out["robust_cum_drain"] = np.array([r["cum_drain"][-1] for r in robust])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print(f"[done in {(time.time()-t0)/60:.1f} min] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
