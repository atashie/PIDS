"""Phase-4 Task 5: the P1-ridge discrete well index r_0(h) (Peaceman-for-FEM).

A line (ridge) source on a P1 host produces a DISCRETE log field; the wall->cell exchange must bridge
through WI = 2*pi/ln(r_0(h)/r_w) where r_0(h) is the MEASURED equivalent radius of the discrete ridge
source: solve steady Laplace -div(grad u) = delta_Gamma (unit strength per unit length) with the
ANALYTIC log u = -(1/2pi)*ln(rho) imposed on the lateral boundary, and read the well-block value
u_h(Gamma) -> r_0 = exp(-2*pi*u_h(Gamma)). Exactly checkable: away from the ridge the discrete field
must BE the analytic log (that is what makes r_0 well-defined); r_0/h must be a lattice constant
(h-independent), which is what makes the WI refinement-robust by construction.

Measurement helper: scratch/m4_phase4_well_index.py (the 1c-style scratch import pattern).
Plan: docs/plans/2026-06-10-m4-phase4-coupled-embedding-plan.md (Task 5).
"""
import numpy as np
import pytest

from scratch.m4_phase4_well_index import measure_r0_tet, measure_r0_tri


@pytest.mark.parametrize("measure,kind", [(measure_r0_tet, "tet"), (measure_r0_tri, "tri")])
def test_far_field_is_the_analytic_log(measure, kind):
    """The discrete field away from the ridge must carry the analytic log with the right source
    strength: the slope of u_h against ln(rho) over the probe band equals -1/(2*pi) within 1%."""
    res = measure(n=16)
    assert abs(res["log_slope"] - (-1.0 / (2.0 * np.pi))) / (1.0 / (2.0 * np.pi)) < 0.01, \
        f"{kind}: far-field log slope {res['log_slope']:.5f} vs analytic {-1/(2*np.pi):.5f}"


@pytest.mark.parametrize("measure,kind", [(measure_r0_tet, "tet"), (measure_r0_tri, "tri")])
def test_r0_over_h_is_a_lattice_constant(measure, kind):
    """r_0/h must be h-independent (the refinement-robustness of the WI rests on this)."""
    a, b = measure(n=8), measure(n=16)
    ra, rb = a["r0"] / a["h"], b["r0"] / b["h"]
    assert abs(ra - rb) / rb < 0.03, f"{kind}: r0/h drifts with h ({ra:.4f} vs {rb:.4f})"
    assert 0.05 < rb < 2.0, f"{kind}: r0/h={rb:.4f} outside the sanity bracket"
