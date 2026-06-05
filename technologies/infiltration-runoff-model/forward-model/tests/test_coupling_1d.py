"""Tier-1 sanity (Module 3 coupling, 1-D): land-surface exchange ψ↔d.

Validates the monolithic two-way coupling of subsurface Richards (ψ) and a surface water store
(d) via the land-surface exchange q_ls = k_ex·(d − ψ_top) (design §D.3, Robin form B), in the
1-D limit where the surface is the top point (no lateral overland). The coupling is built test-
first per governance/claude-sanity-check-routine.md: conservation, then continuity/datum, then
partitioning/recession.

1-D is the realization-agnostic validation of the COUPLING PHYSICS (the cross-mesh submesh
machinery of realization S only bites in 2-D/3-D lateral overland).
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from pids_forward.physics.constitutive import VanGenuchten
from pids_forward.physics.coupling import CoupledProblem
from pids_forward.physics.richards import RichardsProblem


def _psi_top(prob):
    zc = prob.Vpsi.tabulate_dof_coordinates()[:, 0]
    return float(prob.psi.x.array[int(np.argmax(zc))])

# A loam (van Genuchten/Mualem); Ks in m/day.
SOIL = VanGenuchten(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=0.25)


def test_closed_column_mass_balance_under_rain_1d():
    """Closed 1-D column under rain: Δ(soil storage)+Δ(surface store) = ∫rain, to the 1e-6 gate.

    The decisive coupling invariant. The land-surface exchange is sign-paired (water leaving the
    surface store enters the soil top), so with a no-flux base and no outflow the total water
    (∫θ over the column + the surface depth) changes by EXACTLY the cumulative rainfall. This
    forces the full coupled solve (Richards block + surface store + ψ↔d exchange, one monolithic
    Newton per step) to exist and close the books regardless of how the rain partitions between
    infiltration and surface storage.
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 1.0])  # 1 m column; z = x[0] in [0,1]
    prob = CoupledProblem(msh, SOIL)
    # uniformly unsaturated soil, dry surface; closed base (default no-flux), no outflow.
    prob.set_initial_condition(lambda x: -2.0 + 0.0 * x[0], d_value=0.0)
    w0 = prob.total_water()

    rate = 0.1  # m/day rainfall onto the surface store
    rain = prob.add_rain(rate)
    t_end = 0.5
    prob.advance(t_end=t_end, dt=1e-3, dt_max=0.05)

    cum_rain = rate * t_end  # closed column: all rain stays in the system (soil + surface)
    assert abs((prob.total_water() - w0) - cum_rain) / cum_rain < 1e-6
    # SUPPLY-LIMITED: rain (0.1) < Ks (0.25) on unsaturated soil -> all rain infiltrates, NO ponding
    # (the NCP must keep d ~ 0, NOT let the dry soil over-draw an empty store into negative depth).
    assert prob.surface_depth() < 1e-3, f"spurious ponding: d={prob.surface_depth():.3e} (should be ~0)"
    assert prob.surface_water() >= -1e-12  # plausibility: depth non-negative
    assert np.all(np.isfinite(prob.psi.x.array))


def test_infiltration_excess_ponds_1d():
    """Rain >> Ks on a closed column PONDS the surface (infiltration-excess / Hortonian), conserves.

    The complementarity's ponded branch: when supply exceeds what the soil can take, the NCP picks
    g=0 ⇒ λ=q_pot (full Robin capacity) and the excess accumulates in d>0. On a fairly-wet closed
    column the limited storage fills, then the surface ponds; total water still equals the rainfall
    input (closed, no outflow). This is the opposite regime from the supply-limited test above, so it
    exercises the OTHER active set of the NCP -- d must go POSITIVE here (and never negative).
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 1.0])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -0.3 + 0.0 * x[0], d_value=0.0)  # fairly wet, small storage
    w0 = prob.total_water()

    rate = 1.0  # m/day >> Ks=0.25: the soil cannot take it all -> ponding
    prob.add_rain(rate)
    t_end = 0.3
    prob.advance(t_end=t_end, dt=5e-4, dt_max=0.02)

    cum_rain = rate * t_end
    assert abs((prob.total_water() - w0) - cum_rain) / cum_rain < 1e-6  # closed: conserves
    assert prob.surface_depth() > 0.05, (
        f"no Hortonian ponding: d={prob.surface_depth():.3e} (rain {rate} >> Ks {SOIL.Ks})")
    assert prob.surface_water() >= -1e-12
    assert np.all(np.isfinite(prob.psi.x.array))
    # soil also wetted (took some water before saturating), so it is genuine partitioning.
    assert prob.soil_water() > w0


def test_kex_to_infinity_head_continuity_1d():
    """As k_ex→∞ (thin film ℓ_c) the ponded surface head equals the soil surface head: ψ_top → d.

    Design §D.3 datum guard. In the ponded regime λ=q_pot=k_ex·(d−ψ_top) is finite (≈ the infiltration
    rate), so d−ψ_top = λ/k_ex → 0 as k_ex grows: the surface depth and the soil surface pressure head
    coincide (head continuity). A spurious +z_surf in the driving potential would instead leave an
    O(1) gap. We pond the surface (rain>Ks) with a thin film and assert |ψ_top−d| is O(ℓ_c).
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 1.0])
    ell_c = 1e-3
    prob = CoupledProblem(msh, SOIL, ell_c=ell_c)
    prob.set_initial_condition(lambda x: -0.2 + 0.0 * x[0], d_value=0.0)
    prob.add_rain(1.5)  # >> Ks -> ponds
    prob.advance(t_end=0.2, dt=5e-4, dt_max=0.01)

    d = prob.surface_depth()
    assert d > 0.01, f"not ponded (d={d:.3e}); continuity would be trivial"
    gap = abs(_psi_top(prob) - d)
    assert gap < 30 * ell_c, f"head continuity broken: |psi_top-d|={gap:.3e} vs ell_c={ell_c}"


def test_reduces_to_add_ponding_bc_1d():
    """The 1-D coupled limit reproduces Module 1's add_ponding_bc (pond = max(ψ_top,0)).

    Codex's required reduction check: with a thin film (small ℓ_c → head continuity) the coupled
    [ψ,d,λ] model must give the SAME infiltration/storage and ψ profile as the independently-validated
    standalone ponding BC on identical forcing. Run both on a partly-ponding storm and compare soil
    storage and the ψ profile.
    """
    rate, t_end = 0.4, 0.25  # 0.4 > Ks=0.25 -> partial ponding
    msh_c = dmesh.create_interval(MPI.COMM_WORLD, 40, [0.0, 1.0])
    cpl = CoupledProblem(msh_c, SOIL, ell_c=1e-3)
    cpl.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.0)
    cpl.add_rain(rate)
    cpl.advance(t_end=t_end, dt=5e-4, dt_max=0.01)

    msh_r = dmesh.create_interval(MPI.COMM_WORLD, 40, [0.0, 1.0])
    ref = RichardsProblem(msh_r, SOIL)
    ref.set_initial_condition(lambda x: -0.5 + 0.0 * x[0])
    ref.add_ponding_bc(lambda x: np.isclose(x[0], 1.0), rate)
    ref.advance(t_end=t_end, dt=5e-4, dt_max=0.01)

    # soil storage matches the independently-validated ponding BC within a small tolerance.
    assert abs(cpl.soil_water() - ref.total_water()) / ref.total_water() < 0.02
    # ψ profiles agree (atol comparable to the film drop ~ℓ_c and discretization).
    assert np.allclose(cpl.psi.x.array, ref.psi.x.array, atol=0.03)
    # both ponded comparably (coupled d vs reference max(ψ_top,0)).
    assert abs(cpl.surface_depth() - ref.ponded_depth()) < 0.02


def test_recession_pond_drains_into_soil_1d():
    """Rain off: the ponded surface store drains into the soil (d→0), soil wets, monotone, conserves.

    After ponding, set rain=0; while d>0 the exchange λ=q_pot>0 keeps infiltrating, so the pond depth
    falls monotonically toward 0 and the soil gains the drained water. A closed column conserves total
    water throughout the recession (no rain, no outflow).
    """
    msh = dmesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 1.0])
    prob = CoupledProblem(msh, SOIL)
    prob.set_initial_condition(lambda x: -0.3 + 0.0 * x[0], d_value=0.0)
    rain = prob.add_rain(1.0)
    prob.advance(t_end=0.2, dt=5e-4, dt_max=0.02)  # pond it
    d_ponded = prob.surface_depth()
    soil_ponded = prob.soil_water()
    w_mid = prob.total_water()
    assert d_ponded > 0.05  # genuinely ponded before recession

    rain.value = 0.0
    prob.advance(t_end=0.6, dt=5e-4, dt_max=0.02)  # recession
    assert prob.surface_depth() < d_ponded            # pond drained down
    assert prob.soil_water() > soil_ponded            # soil gained the drained water
    assert prob.surface_depth() >= -1e-12             # never negative
    assert abs(prob.total_water() - w_mid) / w_mid < 1e-6  # closed: conserves through recession


def test_plausibility_and_determinism_1d():
    """Coupled invariants (0≤Se≤1, bounded ψ, d≥0, no NaN) and bit-identical reruns (determinism)."""
    def run():
        msh = dmesh.create_interval(MPI.COMM_WORLD, 40, [0.0, 1.0])
        prob = CoupledProblem(msh, SOIL)
        prob.set_initial_condition(lambda x: -0.5 + 0.0 * x[0], d_value=0.0)
        prob.add_rain(0.5)
        prob.advance(t_end=0.2, dt=5e-4, dt_max=0.02)
        return prob.psi.x.array.copy(), prob.d.x.array.copy(), prob.total_water()

    psi_a, d_a, w_a = run()
    psi_b, d_b, w_b = run()
    assert np.array_equal(psi_a, psi_b) and np.array_equal(d_a, d_b) and w_a == w_b  # deterministic
    Se = SOIL.effective_saturation(psi_a)
    assert Se.min() >= -1e-9 and Se.max() <= 1.0 + 1e-9   # saturation bounded
    assert np.all(np.isfinite(psi_a)) and np.all(np.isfinite(d_a))
    assert d_a.min() >= -1e-9                              # non-negative depth everywhere
    assert np.abs(psi_a).max() < 1e3                       # bounded heads
