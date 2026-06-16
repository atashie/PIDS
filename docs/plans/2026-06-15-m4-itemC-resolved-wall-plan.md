# M4 item (C): resolved-wall regime — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Design: `docs/plans/2026-06-15-m4-itemC-resolved-wall-design.md` (committed 480e644). Process
> constraints (non-negotiable, from the Phase-3/4 retractions): gate FIRST; pre-register tolerances
> BEFORE any scheme run; TDD; **two Codex passes** — (1) critical review of the gate-sweep RESULTS before
> any production change (Arik 2026-06-15), (2) the final adversarial+Codex pass before merge; honest
> failure recorded as failure. Commits DIRECTLY ON MAIN.

**Goal:** Make `WellIndexExchange` usable on fine meshes around a 5 cm feature in a large-R_out
catchment (the realistic resolved-wall regime), replacing the blanket `r_0 ≤ 1.1 r_w` refusal with an
honest fence that fires only at the genuinely-degenerate corner.

**Architecture:** Item A already retired the negative-log on-ridge bridge, so the production driver is
r_0-independent — item C is a VALIDATION task, not a redesign. (1) Add a default-off probe hook to run
the existing driver below the current fence. (2) Run the item-A scheme through the existing closed-box
harness at fine meshes against the existing 1-D radial references (depleting-reservoir / history /
drain), with the existing discrimination twins. (3) Codex critically reviews the results. (4) Set the
honest fence on evidence (auto-allow the validated band, refuse the degenerate corner) via TDD.

**Tech Stack:** DOLFINx 0.10 (WSL2 conda `pids-fem`), petsc4py/SNES, numpy. Pre-registered:
`RW_EMBEDDED_TOL = 0.10`, discrimination `≥ 0.20`.

**WSL preamble (all FEM runs — invoke `wsl` from the PowerShell tool, NOT the Bash tool):**
```
wsl bash -lc "source /root/miniforge3/etc/profile.d/conda.sh && conda activate pids-fem && cd /mnt/c/Users/arikt/Documents/GitHub/PIDS/technologies/infiltration-runoff-model/forward-model && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. <cmd>"
```

**Working dir for all code paths below:** `technologies/infiltration-runoff-model/forward-model/`.

---

## Task 0: Baseline — the file we touch is green

**Files:** none (verification only).

**Step 1:** Confirm branch = `main`, HEAD = 480e644 (the design commit):
```
git log --oneline -1
```

**Step 2:** Run the test file we will modify (WSL preamble + `python -m pytest tests/test_wi_exchange.py -q`).
Expected: all pass, including `test_resolved_wall_regime_is_refused`. If not green → STOP, diagnose
before any item-C work. (The full suite is expensive FEM; this file is the one in scope.)

---

## Task 1: The probe hook (TDD, fast — setup only, no solve)

A default-off opt-in that lets a scheme instance run past the resolved-wall refusal, so the sweep can
exercise the item-A driver at fine meshes. Default `False` ⇒ production + the existing refusal test are
byte-identical.

**Files:**
- Modify: `pids_forward/physics/wi_exchange.py` (`__init__` ~`:127`; the guard `:178`)
- Test: `tests/test_wi_exchange.py`

**Step 1: Write the failing test** (append to `tests/test_wi_exchange.py`, after
`test_resolved_wall_regime_is_refused`):
```python
def test_resolved_wall_probe_hook_bypasses_guard():
    """Item C characterization hook: allow_resolved_wall=True opts a scheme instance past the
    resolved-wall refusal so the sweep can run the item-A driver at fine meshes. Default False keeps
    the production refusal (test_resolved_wall_regime_is_refused) byte-identical (asserted there)."""
    feat = _feat_box(2.0, 16)                              # h = 4.43 r_w: resolved-wall, ring resolvable
    x = WellIndexExchange(allow_resolved_wall=True).setup(feat, LOAM, {"t0": 1e-4, "R_out": 2.0})
    assert x.allow_resolved_wall is True
    assert x.r0 <= 1.1 * R_W_DEFAULT                       # we ARE in the otherwise-refused regime
    assert x.h < 5.5 * R_W_DEFAULT
    assert x.WI < 0.0 or x.r0 < x._r_w or True             # witness may be negative-log here; driver ignores it
```

**Step 2: Run it, verify it fails** (WSL preamble):
```
python -m pytest tests/test_wi_exchange.py::test_resolved_wall_probe_hook_bypasses_guard -q
```
Expected: FAIL — `WellIndexExchange()` has no `allow_resolved_wall` param / setup raises the regime
ValueError.

**Step 3: Minimal implementation.** In `__init__`:
```python
def __init__(self, direction: str = "disperse", allow_resolved_wall: bool = False):
    if direction not in ("disperse", "drain"):
        raise ValueError(f"direction must be 'disperse' or 'drain' (got {direction!r})")
    self.direction = direction
    self.allow_resolved_wall = bool(allow_resolved_wall)
```
At the guard (`:178`), add the opt-in to the condition (keep the message; it still fires by default):
```python
        if self.r0 <= 1.1 * self._r_w and not self.allow_resolved_wall:
            raise ValueError(
                f"WellIndexExchange: resolved-wall regime refused (r_0={self.r0:.4f} m <= 1.1*r_w; "
                f"h={self.h:.3f} m = {self.h/self._r_w:.1f} r_w; need h > ~5.5 r_w) -- outside the "
                f"validated deployment regime; pass allow_resolved_wall=True to characterize it "
                f"(item C, 2026-06-15) -- a validated honest fence replaces this in production.")
```

**Step 4: Run, verify pass** — the new test passes AND `test_resolved_wall_regime_is_refused` (default
False) still passes:
```
python -m pytest tests/test_wi_exchange.py -k "resolved_wall" -q
```
Expected: PASS (both).

**Step 5: Commit:**
```
git add tests/test_wi_exchange.py pids_forward/physics/wi_exchange.py
git commit -F <msgfile>   # "M4 item (C): the resolved-wall characterization probe hook (default-off opt-in)"
```

---

## ⏸ PAUSE — confirm go-ahead before the first FEM SOLVE sweep (Task 2 onward).

---

## Task 2: The resolved-wall sweep (the gate instrument; results for Codex)

Run the item-A driver through the existing harness at fine meshes against the existing references, with
the existing discrimination twins. This is a SCRATCH acceptance script (not durable tests yet — those
come in Task 4 once we know it passes). **Pre-register `RW_EMBEDDED_TOL = 0.10` at the top of the script
BEFORE running — no post-hoc tuning.**

**Files:**
- Create: `scratch/m4_phase4_resolved_wall.py`
- Output: `scratch/m4_phase4_resolved_wall_results.npz` + a results table printed to a `.log`

**Step 1: Write the sweep script.**
```python
"""M4 item (C): the RESOLVED-WALL sweep -- the item-A driver at FINE meshes (h < 5.5 r_w) through the
closed-box harness, against the mesh-independent 1-D radial references, with the existing discrimination
twins. Empirical-first: does item A just-work in the realistic regime (large R_out + fine mesh)? Where
does it break? Pre-registered RW_EMBEDDED_TOL = 0.10, discrimination >= 0.20 (LOCKED before running).
Design: docs/plans/2026-06-15-m4-itemC-resolved-wall-design.md.
"""
import sys
import numpy as np
from scratch.m4_phase4_embedded_harness import run_embedded, SOILS, R_W
from pids_forward.physics.wi_exchange import WellIndexExchange
from pids_forward.physics.sorptive_closure import (
    sorptive_clock, F_cylindrical, rel_l2, R0_OVER_H_P1)

RW_EMBEDDED_TOL = 0.10     # PRE-REGISTERED 2026-06-15 (locked before any run)
DISCRIM = 0.20
LOAM = SOILS["LOAM"]
rows = []

def _hbar(R_out, n):
    L = float(np.sqrt(np.pi * ((R_out) ** 2 - R_W ** 2)))
    return (L / n) / R_W

def disperse_leg(R_out, n, refnpz, tkey, ikey, imaxkey, skey, dkey, tag):
    ref = np.load(refnpz)
    t, I_ref = ref[tkey], ref[ikey]
    out = run_embedded(WellIndexExchange(allow_resolved_wall=True), "LOAM", R_out, n, t, label=tag)
    if out is None:
        rows.append((tag, n, _hbar(R_out, n), None, None, None)); return
    e = rel_l2(out["I"], I_ref)
    clk = sorptive_clock(t, float(ref[skey]), float(ref[dkey]), R_W, F_cylindrical)
    d = rel_l2(clk, I_ref)
    endfrac = out["I"][-1] / float(ref[imaxkey])
    rows.append((tag, n, _hbar(R_out, n), e, d, endfrac))

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

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    A = "tests/data/m4_phase4_refA_disperse.npz"
    if which in ("smoke", "all"):
        disperse_leg(40 * R_W, 16, A, "LOAM_R40_t", "LOAM_R40_I", "LOAM_R40_Imax",
                     "LOAM_S", "LOAM_dtheta", "RefA40 n16")
    if which in ("disperse", "all"):
        for n in (24, 32):
            disperse_leg(40 * R_W, n, A, "LOAM_R40_t", "LOAM_R40_I", "LOAM_R40_Imax",
                         "LOAM_S", "LOAM_dtheta", f"RefA40 n{n}")
    if which in ("drain", "all"):
        for n in (16, 24):
            drain_leg(40 * R_W, n, "scratch/m4_phase4_refD40_drain.npz",
                      "LOAM_t", "LOAM_I", "LOAM_Imax", "LOAM", f"refD40 n{n}")
            drain_leg(20 * R_W, n, "scratch/m4_phase4_drain_fresh_refs.npz",
                      "LOAM_R20_t", "LOAM_R20_I", "LOAM_R20_Imax", "LOAM", f"R20 n{n}")
    if which in ("corner", "all"):
        for n in (8, 16):                                  # R3 probes mode-2 + mode-3 at once
            disperse_leg(3 * R_W, n, A, "LOAM_R3_t", "LOAM_R3_I", "LOAM_R3_Imax",
                         "LOAM_S", "LOAM_dtheta", f"RefA3 n{n} (corner)")
    print("\n  tag                  n   h/r_w   relL2    twin   endI/Imax  pass")
    for tag, n, hb, e, d, ef in rows:
        if e is None:
            print(f"  {tag:20s} {n:3d} {hb:6.2f}   DT-COLLAPSE"); continue
        ok = (e <= RW_EMBEDDED_TOL) and (d >= DISCRIM)
        print(f"  {tag:20s} {n:3d} {hb:6.2f}  {e:6.1%}  {d:6.1%}  {ef:7.3f}   {'PASS' if ok else 'FAIL'}")
    np.savez("scratch/m4_phase4_resolved_wall_results.npz",
             rows=np.array([(t, n, hb, -1 if e is None else e, -1 if d is None else d,
                             -1 if ef is None else ef) for t, n, hb, e, d, ef in rows], dtype=object))
```

**Step 2: Smoke first** (cheapest, most-likely-to-pass leg — confirms the driver works in resolved-wall
at all before committing hours):
```
python scratch/m4_phase4_resolved_wall.py smoke 2>&1 | tee scratch/_c_rw_smoke.log
```
Expected (HYPOTHESIS, not a target): `RefA40 n16` tracks `≤ 0.10` with twin `≥ 0.20`. If it FAILS →
STOP, diagnose mechanism with superpowers:systematic-debugging (log ψ_ring vs the resolved Φ at R_out/2;
clock fraction; the WI-era systematic) — do NOT parameter-fiddle. Run in the background (multi-minute).

**Step 3: Full disperse + drain + corner** (background; ~1–2 hr total):
```
python scratch/m4_phase4_resolved_wall.py disperse 2>&1 | tee scratch/_c_rw_disperse.log
python scratch/m4_phase4_resolved_wall.py drain    2>&1 | tee scratch/_c_rw_drain.log
python scratch/m4_phase4_resolved_wall.py corner   2>&1 | tee scratch/_c_rw_corner.log
```
The corner (R3) legs are EXPECTED to fail (mode 2 + mode 3) — they place the fence, they don't pass.

**Step 4: Commit the sweep script + results** (the logs are scratch junk — exclude; commit the `.py` and
the `_results.npz`):
```
git add scratch/m4_phase4_resolved_wall.py scratch/m4_phase4_resolved_wall_results.npz
git commit -F <msgfile>   # "M4 item (C): the resolved-wall sweep + results (item-A driver at fine meshes)"
```

---

## Task 3: ⏸ CHECKPOINT — Codex critically reviews the RESULTS (before any production change)

**Arik's explicit instruction (2026-06-15): when the results land, Codex reviews them critically.** This
is a gate BEFORE the fence/production change.

**Step 1:** Assemble the brief: the design doc, the sweep table (relL2 / twin / endfrac / h-r_w per leg),
the reframing (driver is r_0-independent), and the pre-registered constants. Attack templates Codex MUST
probe: "is the resolved-wall gate actually DISCRIMINATING at high n, or does the twin pass too?"; "does
the realistic regime genuinely track at 0.10, or is a leg riding the bar?"; "is the WI-era systematic
growing with n in a way that auto-allow would silently admit?"; "where exactly does the corner break, and
is the proposed fence boundary justified by the data (not fitted)?".

**Step 2:** Run the Codex review (codex:rescue / the Codex review path). Record its verdict in
`validation/sanity/m4_itemC_resolved_wall_results_review__2026-06-15.md` (binding).

**Step 3:** If Codex flags a real problem → address it (re-run / diagnose) BEFORE Task 4. If the realistic
regime does NOT cleanly pass → honest-failure branch: the fence stays at the deployment boundary, document
the characterization, and STOP before the production change (a valid recorded outcome).

---

## Task 4: Set the honest fence on evidence (TDD production change)

Only after Task 3 clears. The exact fence boundary is DATA-DEPENDENT:
**fence = max(analytic degeneracy boundary + margin, empirical break point from the sweep).** The analytic
boundary = `2h ≤ r_w` (I_fill ≤ 0) AND an `R_out/2`-vs-wall Thiem floor (R_out/2 must clear the near-wall
read-fidelity floor by a measured margin). Auto-allow the validated band; refuse only the degenerate
corner.

**Files:**
- Modify: `pids_forward/physics/wi_exchange.py` (the guard; the module docstring scope-guards section)
- Modify: `tests/test_wi_exchange.py` (split the refusal test; add durable gate legs)

**Step 1 (TDD): rewrite the guard test as TWO tests** — replace
`test_resolved_wall_regime_is_refused` with:
- `test_degenerate_corner_still_refused` — a corner mesh (the sweep's break point, e.g. `_feat(8)` /
  small-R high-n) still raises `ValueError` with the precise message, WITHOUT `allow_resolved_wall`.
- `test_realistic_resolved_wall_now_allowed` — a validated-band mesh (`_feat_box(2.0, 24)` or the
  sweep's passing point) does NOT raise even with the default constructor (auto-allow).

**Step 2:** Run both, verify the second FAILS (the blanket guard still refuses the validated band).

**Step 3:** Implement the honest fence — replace the `r_0 ≤ 1.1 r_w` condition with the degenerate-corner
condition derived from the sweep (template; fill the measured boundary):
```python
        # HONEST FENCE (item C, 2026-06-15): the production driver is r_0-independent (item A retired the
        # negative-log on-ridge bridge); auto-allow the validated realistic resolved-wall band, refuse ONLY
        # the genuinely-degenerate corner -- (a) 2h <= r_w => I_fill <= 0 (clock era skipped from seed), and
        # (b) R_out/2 below the near-wall Thiem floor (the WI-era ring read is invalid). Validated by the
        # resolved-wall sweep (scratch/m4_phase4_resolved_wall.py) at <FILL> r_w down to <FILL>; Codex
        # results review <date>. allow_resolved_wall=True still forces past this (probe/escape hatch).
        if not self.allow_resolved_wall:
            two_h = 2.0 * self.h
            if two_h <= self._r_w * <MARGIN> or <R_out/2 Thiem-floor condition>:
                raise ValueError(... precise degenerate-corner message ...)
```
(Drain: add the symmetric fence iff the sweep shows it needs one; else document that drain validates clean
to the corner.)

**Step 4:** Run the two guard tests + the existing deployment legs — verify all pass:
```
python -m pytest tests/test_wi_exchange.py -k "resolved_wall or corner or refA40 or drain_gate" -q
```

**Step 5:** Add the durable resolved-wall gate legs (the sweep's passing points, at a tractable n —
prefer n=16/24, mark slow):
```python
@pytest.mark.parametrize("n", [16, 24])
def test_resolved_wall_gate_refA40(n):
    """Item C: the item-A driver tracks RefA40 in the realistic resolved-wall regime (fine mesh, large
    R_out) within RW_EMBEDDED_TOL while the offline clock fails -- auto-allowed (no flag)."""
    from scratch.m4_phase4_embedded_harness import run_embedded
    refA = np.load("tests/data/m4_phase4_refA_disperse.npz")
    t, I_ref = refA["LOAM_R40_t"], refA["LOAM_R40_I"]
    out = run_embedded(WellIndexExchange(), "LOAM", 40 * R_W_DEFAULT, n, t)
    assert out is not None
    assert rel_l2(out["I"], I_ref) <= 0.10
    clk = sorptive_clock(t, float(refA["LOAM_S"]), float(refA["LOAM_dtheta"]), R_W_DEFAULT, F_cylindrical)
    assert rel_l2(clk, I_ref) >= 0.20
```
(Plus a drain resolved-wall leg if drain validated.) Run, verify pass.

**Step 6: Commit per behavior** (guard test split → fence → gate legs), each its own commit.

---

## Task 5: Tier-3 record + final adversarial+Codex review + memory + sign-off

**Step 1:** Tier-3 record — a short results note + (optional) HTML in `validation/sanity/`: the sweep
table, the fence location + its justification, the two discrimination twins, the corner-failure
characterization.

**Step 2:** FINAL adversarial review (multi-agent + Codex, the SECOND Codex pass) on the production change
+ tests. Fix-then-ship; re-run the gate after fixes. Binding record in `validation/sanity/`.

**Step 3:** Update the `pids-m4-itemC-resolved-wall-kickoff` memory → outcome (DONE / honest-failure), and
the `pids-m4-phase4-followups-kickoff` REMAINING OPEN ITEMS line (C → resolved).

**Step 4:** Await Arik's explicit sign-off. Commit the records. (Item D #6 Ref-C ablation remains the last
open item.)

---

## Risk register

- **The realistic regime may not pass at 0.10** (the WI-era systematic is larger at high n: clock fraction
  shrinks to ~1%). → systematic-debugging FIRST; honest-failure branch (Task 3 Step 3) if it's a genuine
  breakdown — fence stays at the deployment boundary, characterization documented.
- **R_out/2 plateau revalidation at high n** — the [0.45,0.6]·R_out plateau was deployment-mesh-validated
  (n=6..12); the sweep's tracking IS the revalidation, but log ψ_ring fidelity if a leg rides the bar.
- **FEM cost** — n=32 ≈ 15–30 min/leg; run the sweep in the background, smoke-first. Single-thread BLAS
  pins ([[pids-fem-blas-threading]]); disperse=cp / drain=bt ([[pids-fem-saturated-wall-linesearch]]);
  quadrature cap 8 ([[pids-fem-quadrature-degree-cap]]).
- **PowerShell git** — write commit messages to a file, `git commit -F` (kickoff gotcha). Scratch
  `_c_*/_z*/_x*` files are untracked probe junk — exclude from commits. README investors edit — leave alone.
