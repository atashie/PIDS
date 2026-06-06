"""Quick inspection of the BC-sweep NetCDFs: do the open-outlet SLOPE variants discriminate at all
(hydrograph timing / recession / ponding profile), or only closed-vs-open?"""
import numpy as np, xarray as xr

DATA = "technologies/infiltration-runoff-model/validation/sanity/data"
keys = ["closed", "open_matched", "open_steep", "open_shallow"]
ds = {k: xr.open_dataset(f"{DATA}/coupling_bc__{k}__2026-06-06.nc") for k in keys}

t = ds["closed"]["time"].values
print(f"n_times={t.size}  t_end={t[-1]:.3f}  storm_dur=0.03")
print("\n=== outlet hydrograph q(t) [m2/day] at sampled times ===")
print("  t/day  " + "".join(f"{k:>14}" for k in keys))
for i in range(0, t.size, max(1, t.size // 12)):
    row = "".join(f"{float(ds[k]['outflow'].values[i]):>14.4e}" for k in keys)
    print(f"  {t[i]:.4f}{row}")

print("\n=== cumulative drained [m2] (final) + partition ===")
for k in keys:
    a = ds[k].attrs
    print(f"  {k:>13}: drained={a['final_drained_m2']:.5f}  ponded={a['final_ponded_m2']:.5f}  "
          f"infil={a['final_infiltrated_m2']:.5f}  peak_q={a['peak_outflow_m2_per_day']:.4f}  "
          f"peak_d={a['peak_surface_depth_m']:.5f}  mbe={a['mass_balance_error_max']:.2e}")

print("\n=== final ponding profile d(x) [m] (downhill = high x; outlet at x=L) ===")
x = ds["closed"]["x"].values
print("  x/m   " + "".join(f"{k:>14}" for k in keys))
for j in range(x.size):
    row = "".join(f"{float(ds[k]['ponding_depth'].values[-1, j]):>14.4e}" for k in keys)
    print(f"  {x[j]:.3f}{row}")

# discrimination metric among the 3 open variants
print("\n=== discrimination among the 3 OPEN variants ===")
opens = ["open_matched", "open_steep", "open_shallow"]
q = np.stack([ds[k]["outflow"].values for k in opens])  # (3, time)
qspread = np.ptp(q, axis=0)  # peak-to-peak across variants at each time
qmax = q.max()
print(f"  max |q_variant spread| / max q = {qspread.max()/ (qmax+1e-30):.3%}  (small => slope barely matters)")
dr = np.array([ds[k].attrs['final_drained_m2'] for k in opens])
print(f"  drained range across open variants = [{dr.min():.5f}, {dr.max():.5f}]  "
      f"spread {(dr.max()-dr.min())/dr.mean():.2%}")
