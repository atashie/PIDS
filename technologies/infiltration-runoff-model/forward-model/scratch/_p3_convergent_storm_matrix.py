"""P3 Part B (B2+B3 data-gen, Gate-B v2): drive the convergent dual-drain fixture through three storm
scenarios (typical / 100-yr burst / wet antecedent), capturing -- over TIME, for EACH scenario -- the
surface ponded depth d(x,y) AND a subsurface saturation cross-section theta(x,z) at mid-valley, plus the
water-budget / capture / conservation time series. Saves an npz for viz/make_convergent_dualdrain_html.py.

Gate-B feedback (Arik 2026-06-20): show the saturation profile + ponding depth CHANGE WITH TIME across
scenarios. Run (pids-fem, from forward-model):  PYTHONPATH=. python -u scratch/_p3_convergent_storm_matrix.py
"""
from __future__ import annotations

import os
import time

import numpy as np

from scratch._p3_convergent_fixture import make_convergent_dualdrain, balance_residual

DATE = os.environ.get("DATE", "2026-06-20")
OUT = f"../validation/sanity/data/p3_convergent_dualdrain__{DATE}.npz"

SCENARIOS = [
    dict(key="typical", name="typical storm / normal antecedent", rate=0.35, z_wt=1.05,
         t_storm=0.05, t_recess=0.05),
    dict(key="burst", name="100-yr burst / normal antecedent", rate=1.2, z_wt=1.05,
         t_storm=0.03, t_recess=0.03),
    dict(key="wet", name="typical storm / wet antecedent", rate=0.35, z_wt=1.20,
         t_storm=0.05, t_recess=0.05),
]


def run_scenario(sc, *, n_out=26, nx=24, ny=15, nz=4, dt_max=1e-2):
    """Drive one storm+recession scenario; capture surface d(x,y) + subsurface theta(x,z) at mid-valley
    over time, plus the budget/capture/conservation series."""
    name, rate, z_wt = sc["name"], sc["rate"], sc["z_wt"]
    t_storm, t_recess = sc["t_storm"], sc["t_recess"]
    prob, info, _drain, _inlet = make_convergent_dualdrain(nx, ny, nz, z_wt=z_wt)
    soil = prob.soil
    rain = prob.add_rain(0.0)
    w0 = prob.total_water()

    # surface (x,y) grid from the top Vd dofs
    topdofs = prob._top_dofs(prob.Vd)
    tc = prob.Vd.tabulate_dof_coordinates()[topdofs]
    sxu = np.unique(np.round(tc[:, 0], 6))
    syu = np.unique(np.round(tc[:, 1], 6))
    six = np.searchsorted(sxu, np.round(tc[:, 0], 6))
    siy = np.searchsorted(syu, np.round(tc[:, 1], 6))
    # subsurface (x,z) cross-section at the mid-valley node plane y=y_mid (Vpsi dofs)
    cp = prob.Vpsi.tabulate_dof_coordinates()
    yvals = np.unique(np.round(cp[:, 1], 6))
    y_mid = float(yvals[np.argmin(np.abs(yvals - 0.5 * info["Ly"]))])
    msk = np.abs(cp[:, 1] - y_mid) < 1e-6
    cx = np.round(cp[msk, 0], 6)
    cz = np.round(cp[msk, 2], 6)
    pxu = np.unique(cx)
    pzu = np.unique(cz)
    pix = np.searchsorted(pxu, cx)
    piz = np.searchsorted(pzu, cz)

    t_end = t_storm + t_recess
    out_times = np.linspace(0.0, t_end, n_out)
    rec = {k: [] for k in ("t", "cum_rain", "cum_out", "cum_inlet", "cum_drain",
                           "q_inlet", "q_drain", "bal", "d_max")}
    d_grids, th_grids = [], []
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
        dg = np.full((syu.size, sxu.size), np.nan)
        dg[siy, six] = prob.d.x.array[topdofs]
        d_grids.append(dg)
        tg = np.full((pzu.size, pxu.size), np.nan)
        tg[piz, pix] = soil.theta_of(prob.psi.x.array[msk], cz)
        th_grids.append(tg)

    t0 = time.time()
    snap(0.0)
    k_out, t, dt, n_acc, n_rej, max_it = 1, 0.0, 5e-4, 0, 0, 0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        if t < t_storm - 1e-12 and t + h > t_storm:
            h = t_storm - t
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
    print(f"  [{name}] {n_acc} acc / {n_rej} rej (max_it {max_it}) in {run_s:.0f}s  cum_rain={cum_rain[0]:.2f} "
          f"inlet={a['cum_inlet'][-1]:.3f} drain={a['cum_drain'][-1]:.3f} bal={bal_rel:.1e} "
          f"pond={1e3*a['d_max'].max():.1f}mm dmin={1e3*prob.d.x.array.min():+.2f}mm", flush=True)
    out = dict(name=name, rate=float(rate), z_wt=float(z_wt), t_storm=float(t_storm),
               sxu=sxu, syu=syu, pxu=pxu, pzu=pzu, y_mid=y_mid, z_iface=float(info["z_iface"]) if
               "z_iface" in info else 1.0, Xc=float(info["Xc"]), W=float(info["W"]),
               theta_s=float(soil.theta_s), theta_r=float(soil.theta_r),
               d_grids=np.asarray(d_grids), th_grids=np.asarray(th_grids),
               bal_rel=bal_rel, max_pond=float(a["d_max"].max()), d_min=float(prob.d.x.array.min()),
               n_acc=n_acc, n_rej=n_rej, run_s=run_s, **a)
    return out


def main():
    print("P3 convergent dual-drain: 3 scenarios with surface+subsurface time snapshots", flush=True)
    t0 = time.time()
    results = [run_scenario(sc) for sc in SCENARIOS]
    out = {"keys": np.array([sc["key"] for sc in SCENARIOS]),
           "names": np.array([sc["name"] for sc in SCENARIOS])}
    for sc, r in zip(SCENARIOS, results):
        for k, v in r.items():
            if isinstance(v, str):
                out[f"{sc['key']}__{k}"] = np.array(v)
            else:
                out[f"{sc['key']}__{k}"] = v
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print(f"[done in {(time.time()-t0)/60:.1f} min] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
