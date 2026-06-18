"""B5 probe: confirm the length-weighting extension is BACKWARD-COMPATIBLE with the 1-D point outlet.

The 1-D outlet is a single node; its boundary facet is a POINT. The galerkin path integrates
q_out*v*ds over that point-facet. We confirm the lumped boundary "control length" B_i = assemble(v*ds)
at the 1-D outlet node equals 1.0 (a point measure carries unit weight), so weighting the upwind
1-D outlet by B_i leaves outflow_rate() == the existing un-weighted nodal value -> the B2 1-D tests
(test_outflow_discharge_absolute_magnitude_1d pins the un-weighted value) stay green.

Run (pids-fem): PYTHONPATH=. python scratch/_b5_outlet_1d_probe.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import assemble_vector

L, S0 = 50.0, 0.01


def main():
    msh = dmesh.create_interval(MPI.COMM_WORLD, 20, [0.0, L])
    V = fem.functionspace(msh, ("Lagrange", 1))
    fdim = msh.topology.dim - 1  # = 0 (points)
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = np.sort(dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], L)))
    ft = dmesh.meshtags(msh, fdim, facets, np.full(facets.shape, 1, dtype=np.int32))
    ds_out = ufl.Measure("ds", domain=msh, subdomain_data=ft)(1)
    v = ufl.TestFunction(V)
    bvec = assemble_vector(fem.form(v * ds_out))
    bvec.assemble()
    B_i = bvec.getArray().copy()
    bvec.destroy()
    odofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], L))
    print(f"[1-D] outlet dofs = {odofs}  B_i at outlet = {B_i[odofs]}  (expect 1.0 -> point measure)")
    print(f"[1-D] sum B_i over whole boundary = {B_i.sum()}  (two endpoints, each weight 1.0 => 2.0)")
    assert np.allclose(B_i[odofs], 1.0), "1-D point outlet weight != 1.0 -> would break B2 1-D tests"
    print("[1-D] OK: point-facet control length == 1.0 -> length-weighting is a no-op in 1-D.")


if __name__ == "__main__":
    main()
