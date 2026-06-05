"""Build offline HTMLs for every overland sanity NetCDF via the (solver-free) generator.

Run from forward-model/ with PYTHONPATH=. :  python viz/build_all_overland_html.py
"""
import glob
import os

from viz.make_overland_html import build_overland_html  # reads only the NetCDF; no solver

for nc in sorted(glob.glob("../validation/sanity/data/overland__*.nc")):
    html = nc.replace("/data/", "/viz/").replace(".nc", ".html")
    build_overland_html(nc, html)
    print(f"WROTE {html}  ({os.path.getsize(html) / 1e6:.2f} MB)", flush=True)
