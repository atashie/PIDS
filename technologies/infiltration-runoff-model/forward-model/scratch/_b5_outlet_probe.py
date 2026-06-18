"""B5 outlet-subtlety probe: the 2-D LINE outlet needs per-node boundary-edge LENGTH weighting.

B2's UpwindOverlandProblem.outflow_rate()/residual sink were written for a 1-D POINT outlet (a
single node, no length weighting). The V outlet is a LINE of nodes along y=LY; the total discharge
is the INTEGRAL of the per-unit-width Manning flux q_out [m^2/day] over the outlet edge length, so
each node must carry its share of the outlet-edge control LENGTH B_i [m] -> a volumetric sink
[m^3/day] that telescopes with the storage (d*A_i/dt) and edge fluxes (T_e*M*dH), both volumetric.

This probe verifies, at a KNOWN uniform depth on the LY edge of the canonical V geometry:
  (a) the galerkin ds-integral outflow == q_out_per_width * LX  (the analytic line discharge);
  (b) the NAIVE nodal sum (B2's outflow_rate) == q_out_per_width * (n_outlet_nodes)  -- WRONG,
      off by ~LX/n_nodes (a per-node spacing factor), i.e. it does NOT equal the line discharge;
  (c) sum of the per-node boundary control lengths B_i == LX (so length-weighting recovers (a)).

Run (pids-fem): PYTHONPATH=. python scratch/_b5_outlet_probe.py
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

SECONDS_PER_DAY = 86400.0
SCALE = 1.0
LX, LY = 1620.0 * SCALE, 1000.0 * SCALE
XC = LX / 2.0
SX, SY = 0.05, 0.02
N_MAN = 0.015
NX, NY = 48, 30
D0 = 0.01  # uniform test depth [m]


def main():
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [LX, LY]], [NX, NY])
    V = fem.functionspace(msh, ("Lagrange", 1))
    d = fem.Function(V)
    d.interpolate(lambda x: D0 + 0.0 * x[0])

    # (a) galerkin ds-integral over the LY outlet edge.
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = np.sort(dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], LY)))
    ft = dmesh.meshtags(msh, fdim, facets, np.full(facets.shape, 1, dtype=np.int32))
    ds_out = ufl.Measure("ds", domain=msh, subdomain_data=ft)(1)
    d_pos = ufl.max_value(d, 0.0)
    q_form = fem.form(SECONDS_PER_DAY * (1.0 / N_MAN) * d_pos ** (5.0 / 3.0) * ufl.sqrt(SY) * ds_out)
    Q_galerkin = float(fem.assemble_scalar(q_form))

    q_per_width = SECONDS_PER_DAY * (1.0 / N_MAN) * D0 ** (5.0 / 3.0) * np.sqrt(SY)  # m^2/day
    Q_analytic = q_per_width * LX  # the true line discharge [m^3/day]

    # (b) NAIVE nodal sum (B2 outflow_rate): outlet dofs, sum of nodal q_out, no length weight.
    odofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], LY))
    Q_naive = float(np.sum(q_per_width * np.ones_like(odofs, dtype=float)))  # = q_per_width * n_nodes

    # (c) per-node boundary control length B_i: assemble v*ds over the outlet edge (lumped).
    v = ufl.TestFunction(V)
    Bform = fem.form(v * ds_out)
    from dolfinx.fem.petsc import assemble_vector
    bvec = assemble_vector(Bform)
    bvec.assemble()
    B_i = bvec.getArray().copy()
    bvec.destroy()
    B_outlet = B_i[odofs]
    Q_lenweighted = float(np.sum(q_per_width * B_outlet))

    print(f"[geom] V {NX}x{NY} on {LX:g}x{LY:g} m  outlet nodes = {odofs.size}")
    print(f"[a] galerkin ds-integral   Q = {Q_galerkin:14.3f} m^3/day")
    print(f"[a] analytic line q*LX     Q = {Q_analytic:14.3f} m^3/day  "
          f"(match: {Q_galerkin / Q_analytic:.6f})")
    print(f"[b] NAIVE nodal sum (B2)   Q = {Q_naive:14.3f} m^3/day  "
          f"(ratio to analytic {Q_naive / Q_analytic:.4f}  -- WRONG, ~n_nodes/LX scaling)")
    print(f"[c] sum B_i (control len)    = {B_outlet.sum():14.3f} m   (should == LX = {LX:g})")
    print(f"[c] length-weighted sum    Q = {Q_lenweighted:14.3f} m^3/day  "
          f"(match to galerkin: {Q_lenweighted / Q_galerkin:.6f})")
    print(f"[interior B_i] {np.sort(B_outlet)}  (corners = half-spacing, interior = full)")


if __name__ == "__main__":
    main()
