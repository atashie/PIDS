"""M4 item (C): the RESOLVED-WALL sweep -- the item-A driver at FINE meshes (h < 5.5 r_w) through the
closed-box harness, against the mesh-independent 1-D radial references, with the existing discrimination
twins. Empirical-first: does item A just-work in the realistic regime (large R_out + fine mesh)? Where
does it break? Pre-registered RW_EMBEDDED_TOL = 0.10, discrimination >= 0.20 (LOCKED before running).

The production driver is r_0-independent (item A retired the negative-log on-ridge bridge); the
resolved-wall scheme runs via the allow_resolved_wall=True opt-in (default-off in production). The
references and discrimination twins are reused verbatim from the deployment gate (tests/test_wi_exchange.py).

Usage (WSL pids-fem, PYTHONPATH=.):  python scratch/m4_phase4_resolved_wall.py {smoke|disperse|drain|corner|all}
Design/plan: docs/plans/2026-06-15-m4-itemC-resolved-wall-{design,plan}.md.
"""
import sys
import numpy as np

from scratch.m4_phase4_embedded_harness import run_embedded, SOILS, R_W
from pids_forward.physics.wi_exchange import WellIndexExchange
from pids_forward.physics.sorptive_closure import sorptive_clock, F_cylindrical, rel_l2

RW_EMBEDDED_TOL = 0.10     # PRE-REGISTERED 2026-06-15 (locked before any run -- no post-hoc tuning)
DISCRIM = 0.20
A = "tests/data/m4_phase4_refA_disperse.npz"
rows = []                  # (tag, n, h/r_w, relL2|None, twin|None, endfrac|None)


def _hbar(R_out, n):
    L = float(np.sqrt(np.pi * (R_out ** 2 - R_W ** 2)))
    return (L / n) / R_W


def disperse_leg(R_out, n, tkey, ikey, imaxkey, skey, dkey, tag):
    ref = np.load(A)
    t, I_ref = ref[tkey], ref[ikey]
    out = run_embedded(WellIndexExchange(allow_resolved_wall=True), "LOAM", R_out, n, t, label=tag)
    if out is None:
        rows.append((tag, n, _hbar(R_out, n), None, None, None)); return
    e = rel_l2(out["I"], I_ref)
    clk = sorptive_clock(t, float(ref[skey]), float(ref[dkey]), R_W, F_cylindrical)
    d = rel_l2(clk, I_ref)
    rows.append((tag, n, _hbar(R_out, n), e, d, out["I"][-1] / float(ref[imaxkey])))


def disperse_pulse_leg(R_out, n, tag):
    """The RefB40 host-HISTORY leg (Codex 2026-06-15: RefA40 alone does not kill the capacity-clamped
    passive; the pulse-shifted asymptote is the host-control discriminator). Twin = the offline clock,
    which has no host knowledge and cannot track the pulse (deployment: clock ~34% vs embedded 5%)."""
    ref = np.load("scratch/m4_phase4_refB40_disperse.npz")
    t, I_ref = ref["LOAM_t"], ref["LOAM_I"]
    band = tuple(float(x) for x in ref["LOAM_band"])
    pulse = (float(ref["LOAM_t_pulse"][0]), float(ref["LOAM_t_pulse"][1]),
             float(ref["LOAM_V_pulse_per_wall_area"]) * R_W * 2 * np.pi)
    out = run_embedded(WellIndexExchange(allow_resolved_wall=True), "LOAM", R_out, n, t,
                       pulse=pulse, pulse_band=band, label=tag)
    if out is None:
        rows.append((tag, n, _hbar(R_out, n), None, None, None)); return
    e = rel_l2(out["I"], I_ref)
    clk = sorptive_clock(t, float(ref["LOAM_S"]), float(ref["LOAM_dtheta"]), R_W, F_cylindrical)
    d = rel_l2(clk, I_ref)
    rows.append((tag, n, _hbar(R_out, n), e, d, out["I"][-1] / float(ref["LOAM_Imax"])))


def drain_leg(R_out, n, refnpz, tkey, ikey, imaxkey, soil, tag):
    ref = np.load(refnpz)
    t, I_ref = ref[tkey], ref[ikey]
    out = run_embedded(WellIndexExchange(direction="drain", allow_resolved_wall=True),
                       soil, R_out, n, t, direction="drain", label=tag)
    if out is None:
        rows.append((tag, n, _hbar(R_out, n), None, None, None)); return
    e = rel_l2(out["I"], I_ref)
    geo = np.log(R_out / R_W) - 0.75
    rate_fixed = float(SOILS[soil].kirchhoff(-1.0, -0.03)) / (R_W * geo)
    I_twin = np.minimum(rate_fixed * (t - t[0]), float(ref[imaxkey]))
    d = rel_l2(I_twin, I_ref)
    rows.append((tag, n, _hbar(R_out, n), e, d, out["I"][-1] / float(ref[imaxkey])))


def _report(which):
    print("\n  tag                      n   h/r_w   relL2    twin   endI/Imax  pass", flush=True)
    for tag, n, hb, e, d, ef in rows:
        if e is None:
            print(f"  {tag:24s} {n:3d} {hb:6.2f}   DT-COLLAPSE", flush=True); continue
        ok = (e <= RW_EMBEDDED_TOL) and (d >= DISCRIM)
        print(f"  {tag:24s} {n:3d} {hb:6.2f}  {e:6.1%}  {d:6.1%}  {ef:7.3f}   {'PASS' if ok else 'FAIL'}",
              flush=True)
    tags = np.array([r[0] for r in rows])
    mat = np.array([[r[1], r[2],
                     -1.0 if r[3] is None else r[3],
                     -1.0 if r[4] is None else r[4],
                     -1.0 if r[5] is None else r[5]] for r in rows], dtype=float)
    np.savez(f"scratch/m4_phase4_resolved_wall_results_{which}.npz", tags=tags, mat=mat,
             cols=np.array(["n", "h_over_rw", "relL2", "twin", "endfrac"]))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if which in ("smoke", "all"):
        disperse_leg(40 * R_W, 16, "LOAM_R40_t", "LOAM_R40_I", "LOAM_R40_Imax",
                     "LOAM_S", "LOAM_dtheta", "RefA40 n16")
    if which in ("disperse", "all"):
        for n in (24, 32):
            disperse_leg(40 * R_W, n, "LOAM_R40_t", "LOAM_R40_I", "LOAM_R40_Imax",
                         "LOAM_S", "LOAM_dtheta", f"RefA40 n{n}")
    if which in ("history", "all"):                         # Codex blocking fix: the host-history leg
        for n in (16, 24):
            disperse_pulse_leg(40 * R_W, n, f"RefB40 n{n}")
    if which in ("drain", "all"):
        for n in (16, 24):
            drain_leg(40 * R_W, n, "scratch/m4_phase4_refD40_drain.npz",
                      "LOAM_t", "LOAM_I", "LOAM_Imax", "LOAM", f"refD40 n{n}")
            drain_leg(20 * R_W, n, "scratch/m4_phase4_drain_fresh_refs.npz",
                      "LOAM_R20_t", "LOAM_R20_I", "LOAM_R20_Imax", "LOAM", f"R20 n{n}")
    if which in ("corner", "all"):
        for n in (8, 16):                                  # R3: mode-2 (I_fill<=0) + mode-3 conflated
            disperse_leg(3 * R_W, n, "LOAM_R3_t", "LOAM_R3_I", "LOAM_R3_Imax",
                         "LOAM_S", "LOAM_dtheta", f"RefA3 n{n} corner")
        for n in (24, 36):                                  # R10: mode-2 BRACKET at R_out/2=5 r_w (cleaner
            disperse_leg(10 * R_W, n, "LOAM_R10_t", "LOAM_R10_I", "LOAM_R10_Imax",  # than R3 for locating
                         "LOAM_S", "LOAM_dtheta", f"RefA10 n{n} m2")               # the degeneracy; Codex #2)
    if which in ("killmap", "all"):                         # Codex (b): the passive must still DIE at fine
        from scratch.m4_phase4_embedded_harness import DualScaleScheme  # mesh (the gate still discriminates)
        ref = np.load(A)
        t, I_ref = ref["LOAM_R40_t"], ref["LOAM_R40_I"]
        out = run_embedded(DualScaleScheme("disperse"), "LOAM", 40 * R_W, 16, t, label="killmap n16")
        if out is not None:
            e = rel_l2(out["I"], I_ref)
            print(f"\n  KILLMAP dual-scale passive RefA40 n16: relL2={e:.1%}  "
                  f"{'KILLED (>=20%, gate discriminates)' if e >= DISCRIM else 'NOT KILLED -- STOP'}",
                  flush=True)
    _report(which)
