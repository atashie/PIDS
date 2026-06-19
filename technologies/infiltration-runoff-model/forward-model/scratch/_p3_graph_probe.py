"""P3-A diagnostic: does the COUPLED top-facet edge graph (3-D box) == the STANDALONE 2-D edge graph
(create_rectangle)? If the triangulations differ (diagonal direction), the upwind edge fluxes differ
-> operator-equivalence (A1.2) can never be bit-tight. Build both for the SAME nx,ny,domain and
compare the edge sets (by endpoint coordinates) + the nodal areas A_i."""
from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.overland_upwind import UpwindOverlandProblem

SOIL = VanGenuchten(theta_r=0.10, theta_s=0.40, alpha=2.0, n=2.0, Ks=1.0e-3)
LX, LY, H = 120.0, 80.0, 1.0
nx, ny, nz = 8, 6, 3


def edge_coord_set(edges, coords):
    s = set()
    for a, b in edges:
        pa = (round(float(coords[a, 0]), 6), round(float(coords[a, 1]), 6))
        pb = (round(float(coords[b, 0]), 6), round(float(coords[b, 1]), 6))
        s.add(frozenset((pa, pb)))
    return s


# coupled: top-facet edge graph in Vd dof space
mc = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [LX, LY, H]], [nx, ny, nz])
pc = CoupledProblem(mc, SOIL, overland_scheme="upwind")
cc = pc.Vd.tabulate_dof_coordinates()
ec = edge_coord_set(pc._upwind_edges, cc)

# standalone: 2-D triangle edge graph
ms = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [LX, LY]], [nx, ny])
ps = UpwindOverlandProblem(ms, 0.015)
cs = ps.V.tabulate_dof_coordinates()
es = edge_coord_set(ps.edges, cs)

print(f"coupled top-facet edges = {len(ec)}   standalone 2-D edges = {len(es)}")
print(f"  common = {len(ec & es)}   only-coupled = {len(ec - es)}   only-standalone = {len(es - ec)}")
if ec == es:
    print("  => IDENTICAL triangulation -> operator-equivalence CAN be tight (diff = lambda only)")
else:
    print("  => DIFFERENT triangulation (diagonal direction) -> operator-equivalence is NOT bit-tight")
    ex = sorted(ec - es)[:3]
    print(f"     example only-coupled edges: {ex}")

# nodal areas A_i by coordinate (sum should both equal the domain area)
print(f"\nsum A_i: coupled (top) = {float(np.sum(pc._upwind_A_i)):.3f}   "
      f"standalone = {float(np.sum(ps.A_i)):.3f}   domain area = {LX*LY:.3f}")
