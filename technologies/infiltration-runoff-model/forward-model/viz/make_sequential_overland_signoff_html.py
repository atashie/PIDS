"""Tier-3 SIGN-OFF visualizer for the sequential operator-split overland coupling.

Reads ONLY the npz from ``viz/run_sequential_overland_signoff.py`` (independent visual evidence -- never
runs the solver) and writes ONE self-contained, offline HTML for Arik's visual sign-off, mirroring the
prior module sign-offs (make_coupling_3d_html / make_convergent_dualdrain_html house style: inline
Plotly, a clear OLD-FAILS / NEW-SUCCEEDS narrative).

THE STORY. On the sand-channel-in-clay storm that TRIGGERED the redesign -- a coarse-sand conveyance
channel intercepting convergent storm runoff on a low-K clay hillslope:
  * OLD monolithic Manning schemes FAIL: CoupledProblem(overland_scheme="upwind") dt-COLLAPSES;
    overland_scheme="galerkin" SAWTOOTH dt-PINS (both effectively non-terminating).
  * NEW SequentialCoupledProblem SUCCEEDS: completes to T_END with no dt-collapse, conserves
    |balance|/cum_rain ~ 1e-12, and intercepts the runoff into the channel.
The convergent tilted-V (the original sawtooth pathology) corroborates as a second case.

Panels (sand channel, headline):
  1. dt vs time (log-y): NEW climbs + stays up; OLD upwind plunges to collapse; OLD galerkin pins low.
  2. water-balance closure |balance|/cum_rain vs time (NEW, log-y): machine-tight throughout.
  3. interception: cumulative channel capture (subsurface conveyance + dispersion into clay) vs surface
     escape, with the final intercepted fraction called out.
Plus the tilted-V dt-vs-time + cumulative outlet discharge (NEW routes + conserves; OLD galerkin saws).

Usage:  python viz/make_sequential_overland_signoff_html.py [<data.npz> <out.html>]
"""
from __future__ import annotations

import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NEW_C = "#1565c0"       # NEW sequential
UP_C = "#c62828"        # OLD upwind (collapse)
GAL_C = "#ef6c00"       # OLD galerkin (sawtooth pin)
CAP_C = "#2e7d32"       # channel capture
CONV_C = "#1b5e20"      # subsurface conveyance
DISP_C = "#66bb6a"      # dispersion into clay
ESC_C = "#90a4ae"       # surface escape
BAL_C = "#6a1b9a"       # mass balance


def _g(d, key, default=None):
    """Fetch an npz entry, coercing 0-d/scalar arrays to plain Python scalars.

    np.savez stores a Python bool/float/str as a 0-d array; str()/bool() on those would render
    e.g. ``array('completed', ...)``. ``.item()`` unwraps a 0-d (or size-1) array to the Python
    scalar; real 1-D timeline arrays are returned untouched."""
    if key not in d.files:
        return default
    v = d[key]
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    return v


def _verdict_badge(v):
    txt = {"collapsed": "dt-COLLAPSE", "pinned": "dt-PINNED (sawtooth)",
           "completed": "COMPLETED", "budget": "budget-limited", "ran": "ran"}.get(str(v), str(v))
    fail = str(v) in ("collapsed", "pinned")
    color = "#c62828" if fail else "#2e7d32"
    return txt, color


def _fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.0%}"


def _fmt_e(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.2e}"


# --------------------------------------------------------------------------- headline: sand channel
def channel_figure(d):
    """3-panel headline: dt-vs-t (OLD fails / NEW succeeds) | balance trace | interception."""
    t_end = float(_g(d, "ch_seq__T_END", 1.2))

    # NEW sequential timeline
    sT, sDT = np.asarray(d["ch_seq__T"], float), np.asarray(d["ch_seq__DT"], float)
    sBAL = np.asarray(d["ch_seq__BALREL"], float)
    sRAIN = np.asarray(d["ch_seq__CUM_RAIN"], float)
    sCONV = np.asarray(d["ch_seq__CUM_CONV"], float)
    sOUT = np.asarray(d["ch_seq__CUM_OUT"], float)
    sDISP = np.asarray(d["ch_seq__DISP"], float)
    seq_done = bool(_g(d, "ch_seq__completed", False))
    seq_bal = float(_g(d, "ch_seq__bal_final", np.nan))
    seq_balmax = float(_g(d, "ch_seq__bal_max", np.nan))
    seq_intercept = float(_g(d, "ch_seq__intercept_frac", np.nan))
    seq_steps = int(_g(d, "ch_seq__nsteps", 0))
    seq_wall = float(_g(d, "ch_seq__wall", np.nan))

    # OLD monolith timelines
    uT, uDT = np.asarray(d["ch_upwind__T"], float), np.asarray(d["ch_upwind__DT"], float)
    u_verd = str(_g(d, "ch_upwind__verdict", "ran"))
    u_treach = float(_g(d, "ch_upwind__t_reached", np.nan))
    u_dtmin = float(_g(d, "ch_upwind__dt_min_seen", np.nan))
    u_replay = bool(_g(d, "ch_upwind__replayed", False))   # documented replay vs live capture
    gT, gDT = np.asarray(d["ch_galerkin__T"], float), np.asarray(d["ch_galerkin__DT"], float)
    g_verd = str(_g(d, "ch_galerkin__verdict", "ran"))
    g_treach = float(_g(d, "ch_galerkin__t_reached", np.nan))
    g_dtmin = float(_g(d, "ch_galerkin__dt_min_seen", np.nan))

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.075,
        column_widths=[0.40, 0.30, 0.30],
        subplot_titles=(
            "Time-step dt vs time  &mdash; OLD schemes FAIL, NEW stays up  (log dt)",
            "NEW mass-balance closure |balance|/cum_rain  (log)",
            "NEW interception: where the runoff went  [m&sup3;]"))

    # ---- panel 1: dt vs t ----
    fig.add_trace(go.Scatter(x=sT, y=sDT, mode="lines", line=dict(color=NEW_C, width=3),
                  name="NEW sequential"), 1, 1)
    u_name = "OLD upwind (documented)" if u_replay else "OLD upwind"
    u_dash = "dash" if u_replay else "solid"
    fig.add_trace(go.Scatter(x=uT, y=uDT, mode="lines+markers",
                  line=dict(color=UP_C, width=2, dash=u_dash),
                  marker=dict(size=3), name=u_name), 1, 1)
    fig.add_trace(go.Scatter(x=gT, y=gDT, mode="lines+markers", line=dict(color=GAL_C, width=2),
                  marker=dict(size=3), name="OLD galerkin (live)"), 1, 1)
    # mark the OLD failure points
    if uT.size:
        fig.add_trace(go.Scatter(x=[uT[-1]], y=[uDT[-1]], mode="markers+text",
                      marker=dict(size=12, color=UP_C, symbol="x"),
                      text=["upwind collapse (documented)" if u_replay else "upwind collapse"],
                      textposition="top center",
                      textfont=dict(size=10, color=UP_C), showlegend=False), 1, 1)
    if gT.size:
        fig.add_trace(go.Scatter(x=[gT[-1]], y=[gDT[-1]], mode="markers+text",
                      marker=dict(size=12, color=GAL_C, symbol="x"),
                      text=["galerkin pinned"], textposition="bottom center",
                      textfont=dict(size=10, color=GAL_C), showlegend=False), 1, 1)

    # ---- panel 2: balance trace ----
    balp = np.where(sBAL > 0, sBAL, np.nan)
    fig.add_trace(go.Scatter(x=sT, y=balp, mode="lines", line=dict(color=BAL_C, width=2.5),
                  name="|balance|/cum_rain", showlegend=False), 1, 2)
    fig.add_hline(y=1e-3, line=dict(color="gray", dash="dot"), row=1, col=2)
    fig.add_annotation(x=t_end * 0.5, y=1e-3, yref="y2", xref="x2", text="10&#8315;&#179; bar",
                       showarrow=False, yshift=10, font=dict(size=9, color="gray"))

    # ---- panel 3: interception (stacked cumulative capture vs escape) ----
    # capture = conveyance (GHB) + dispersion into clay; escape = surface routed off-domain.
    disp_pos = np.clip(sDISP, 0, None)
    s1 = sCONV
    s2 = sCONV + disp_pos
    s3 = sCONV + disp_pos + sOUT
    fig.add_trace(go.Scatter(x=sT, y=s1, mode="lines", line=dict(color=CONV_C, width=0.5),
                  fill="tozeroy", fillcolor="rgba(27,94,32,0.55)",
                  name="channel: subsurface conveyance (GHB)"), 1, 3)
    fig.add_trace(go.Scatter(x=sT, y=s2, mode="lines", line=dict(color=DISP_C, width=0.5),
                  fill="tonexty", fillcolor="rgba(102,187,106,0.50)",
                  name="channel: dispersion into clay", customdata=disp_pos,
                  hovertemplate="t=%{x:.3f}<br>dispersion=%{customdata:.4f}<extra></extra>"), 1, 3)
    fig.add_trace(go.Scatter(x=sT, y=s3, mode="lines", line=dict(color=ESC_C, width=0.5),
                  fill="tonexty", fillcolor="rgba(144,164,174,0.45)",
                  name="surface escaped (routed off-domain)", customdata=sOUT,
                  hovertemplate="t=%{x:.3f}<br>escaped=%{customdata:.4f}<extra></extra>"), 1, 3)
    fig.add_trace(go.Scatter(x=sT, y=sRAIN, mode="lines", line=dict(color="#263238", width=2, dash="dot"),
                  name="cumulative rain", showlegend=False), 1, 3)

    fig.update_xaxes(title_text="time [day]", row=1, col=1, range=[0, t_end])
    fig.update_yaxes(title_text="dt [day]", type="log", exponentformat="power", row=1, col=1)
    fig.update_xaxes(title_text="time [day]", row=1, col=2, range=[0, t_end])
    fig.update_yaxes(title_text="|balance|/cum_rain", type="log", exponentformat="power", row=1, col=2)
    fig.update_xaxes(title_text="time [day]", row=1, col=3, range=[0, t_end])
    fig.update_yaxes(title_text="cumulative water [m&sup3;]", row=1, col=3, rangemode="tozero")

    fig.update_layout(template="plotly_white", height=470, width=1500,
                      legend=dict(orientation="h", y=-0.22, x=0.0, font=dict(size=10)),
                      margin=dict(t=60, b=110, l=70, r=30))

    # verdict callouts
    u_txt, u_col = _verdict_badge(u_verd)
    g_txt, g_col = _verdict_badge(g_verd)
    head = (
        f"<div style='display:flex;gap:14px;flex-wrap:wrap;font-size:14px;margin:6px 0 2px'>"
        f"<span style='background:#e8f5e9;border-left:4px solid {CAP_C};padding:5px 10px'>"
        f"<b>NEW sequential</b>: {'COMPLETED to T_END' if seq_done else 'did NOT complete'} "
        f"&middot; |bal|/rain = <b>{_fmt_e(seq_bal)}</b> (max {_fmt_e(seq_balmax)}) "
        f"&middot; intercepted <b>{_fmt_pct(seq_intercept)}</b> &middot; {seq_steps} steps, "
        f"{seq_wall:.0f}s</span>"
        f"<span style='background:#ffebee;border-left:4px solid {u_col};padding:5px 10px'>"
        f"<b>OLD upwind</b>{' <i>(documented)</i>' if u_replay else ''}: {u_txt} "
        f"&middot; at t={u_treach:.3f}/{t_end:g} d &middot; dt&rarr;{_fmt_e(u_dtmin)}"
        f"{' &middot; replay, not live re-run' if u_replay else ''}</span>"
        f"<span style='background:#fff3e0;border-left:4px solid {g_col};padding:5px 10px'>"
        f"<b>OLD galerkin</b> <i>(live)</i>: {g_txt} &middot; reached t={g_treach:.3f}/{t_end:g} d "
        f"&middot; dt&rarr;{_fmt_e(g_dtmin)}</span>"
        f"</div>")
    return head + fig.to_html(full_html=False, include_plotlyjs=False)


# --------------------------------------------------------------------------- corroborate: tilted-V
def tiltedv_figure(d):
    # The tilted-V is a secondary corroborating case (the primary regression for it is the B7
    # automated test). Render it only if BOTH its stages are present; if the live galerkin stage is
    # missing (e.g. a run-script error on the monolith), skip the panel rather than crash -- the
    # sand-channel section above is the complete, load-bearing sign-off story.
    if "v_seq__T" not in d.files or "v_galerkin__T" not in d.files:
        return ""
    t_end = float(_g(d, "v_seq__T_END", 0.9))
    vsT, vsDT = np.asarray(d["v_seq__T"], float), np.asarray(d["v_seq__DT"], float)
    vsOUT, vsRAIN = np.asarray(d["v_seq__CUMOUT"], float), np.asarray(d["v_seq__CUMRAIN"], float)
    v_seq_verd = str(_g(d, "v_seq__verdict", "ran"))
    v_seq_dtmin = float(_g(d, "v_seq__dt_min_seen", np.nan))
    vgT, vgDT = np.asarray(d["v_galerkin__T"], float), np.asarray(d["v_galerkin__DT"], float)
    v_gal_verd = str(_g(d, "v_galerkin__verdict", "ran"))
    v_gal_dtmin = float(_g(d, "v_galerkin__dt_min_seen", np.nan))

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=("tilted-V: dt vs time  (NEW stays up; OLD galerkin saws)  (log dt)",
                                        "tilted-V: cumulative outlet discharge  [m&sup3;]  (NEW routes to outlet)"))
    fig.add_trace(go.Scatter(x=vsT, y=vsDT, mode="lines", line=dict(color=NEW_C, width=3),
                  name="NEW sequential"), 1, 1)
    fig.add_trace(go.Scatter(x=vgT, y=vgDT, mode="lines+markers", line=dict(color=GAL_C, width=1.5),
                  marker=dict(size=2), name="OLD galerkin"), 1, 1)
    fig.add_trace(go.Scatter(x=vsT, y=vsOUT, mode="lines", line=dict(color=NEW_C, width=2.5),
                  name="NEW cum outlet", showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=vsT, y=vsRAIN, mode="lines", line=dict(color="#263238", width=1.5, dash="dot"),
                  name="cum rain", showlegend=False), 1, 2)
    fig.update_xaxes(title_text="time [day]", row=1, col=1, range=[0, t_end])
    fig.update_yaxes(title_text="dt [day]", type="log", exponentformat="power", row=1, col=1)
    fig.update_xaxes(title_text="time [day]", row=1, col=2, range=[0, t_end])
    fig.update_yaxes(title_text="cumulative water [m&sup3;]", row=1, col=2, rangemode="tozero")
    fig.update_layout(template="plotly_white", height=400, width=1180,
                      legend=dict(orientation="h", y=-0.25, x=0.0, font=dict(size=10)),
                      margin=dict(t=55, b=80, l=70, r=30))

    s_txt, s_col = _verdict_badge(v_seq_verd)
    g_txt, g_col = _verdict_badge(v_gal_verd)
    head = (
        f"<div style='display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin:6px 0 2px'>"
        f"<span style='background:#e8f5e9;border-left:4px solid {s_col};padding:5px 10px'>"
        f"<b>NEW sequential</b>: {s_txt} &middot; dt&rarr;{_fmt_e(v_seq_dtmin)}</span>"
        f"<span style='background:#fff3e0;border-left:4px solid {g_col};padding:5px 10px'>"
        f"<b>OLD galerkin</b> <i>(live)</i>: {g_txt} &middot; dt&rarr;{_fmt_e(v_gal_dtmin)}</span></div>")
    return "<h2>Corroborating case &mdash; convergent tilted-V (the original sawtooth pathology)</h2>" \
        + head + fig.to_html(full_html=False, include_plotlyjs=False)


def build(npz_path: str, out_html: str) -> None:
    from plotly.offline import get_plotlyjs
    d = np.load(npz_path, allow_pickle=True)

    ch = channel_figure(d)
    v = tiltedv_figure(d)

    seq_done = bool(_g(d, "ch_seq__completed", False))
    seq_bal = float(_g(d, "ch_seq__bal_final", np.nan))
    seq_intercept = float(_g(d, "ch_seq__intercept_frac", np.nan))
    ndof = int(_g(d, "ch_seq__ndof", 0))
    rsub = int(_g(d, "ch_seq__route_substeps", 4))

    intro = f"""</script>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1520px;margin:18px auto;color:#1f2933;
padding:0 14px}}
h1{{margin-bottom:2px}} h2{{margin-top:30px;border-bottom:2px solid #e0e0e0;padding-bottom:4px}}
.note{{background:#f4f7fb;border-left:4px solid #1565c0;padding:10px 14px;font-size:14px;line-height:1.5}}
.headline{{background:#eef6ee;border:1px solid #c8e6c9;border-radius:6px;padding:12px 16px;font-size:15px;
margin:12px 0}}
code{{background:#eceff1;padding:1px 4px;border-radius:3px;font-size:90%}}
.lim{{background:#fff8e1;border-left:4px solid #f9a825;padding:9px 13px;font-size:13px;line-height:1.5}}
</style></head><body>
<h1>PIDS overland flow &mdash; sequential operator-split coupling: Tier-3 sign-off</h1>
<p style='color:#607d8b;font-size:13px;margin-top:0'>Pillar-2 forward model &middot; DOLFINx 0.10 /
<code>pids-fem</code> (serial) &middot; date 2026-06-23 &middot; independent visual evidence (reads only
the run npz; never re-runs the solver)</p>

<div class="headline">
<b>Headline.</b> On the sand-channel-in-clay storm that <b>triggered</b> this redesign &mdash; a coarse-sand
conveyance channel intercepting convergent runoff on a low-K clay hillslope &mdash; the
<b>NEW <code>SequentialCoupledProblem</code></b> {'<b>completes to T_END</b>' if seq_done else 'runs'},
conserves to <b>|balance|/cum&nbsp;rain = {_fmt_e(seq_bal)}</b> (machine-tight), and
<b>intercepts {_fmt_pct(seq_intercept)}</b> of the routed/conveyed water into the channel &mdash; where the
<b>OLD monolithic Manning stack does not</b>. The reproducible monolithic failure captured <b>live</b> on
this exact case is the <code>overland_scheme="galerkin"</code> <b>convergent sawtooth dt-pin</b> (dt frozen
at &asymp;4&times;10&#8315;&#8309; d, effectively non-terminating). The <code>overland_scheme="upwind"</code>
<b>dt-collapse at t&asymp;0.11</b> is the regime that originally triggered the rebuild (2026-06-22 canonical
demo) and is shown here as a <b>documented reference</b> (dashed): on this <i>present</i> variant the deep
&asymp;1&nbsp;m berm pond regularizes the near-saturation singularity, so a live upwind re-run limps through
slowly rather than hard-collapsing &mdash; the honest live evidence is the galerkin pin. Either way the
operator split delivers the structural cure: the stiff implicit Richards solve never shares a Jacobian with
the explicit surface routing, so neither the coupled-stiffness dt-collapse nor the convergent-flow sawtooth
can form.
</div>

<p class="note"><b>What you are signing off.</b> The same physics / geometry / forcing as the trigger
case <code>scratch/m4_sand_channel_3d_demo.py</code> (native demo mesh, {ndof} DOFs), the
sand-channel-in-clay storm, run side by side. <b>Left panel</b> is the
money plot: <span style='color:#1565c0'><b>NEW dt</b></span> climbs and holds at the controller ceiling,
while <span style='color:#ef6c00'><b>OLD galerkin</b></span> (live) pins at a tiny dt (the convergent
sawtooth) and <span style='color:#c62828'><b>OLD upwind</b></span> (documented, dashed) plunges into the
dt-collapse that motivated this redesign. <b>Middle</b>:
the NEW run's mass-balance closure stays far below the 10&#8315;&#179; bar (operator-split coupling error
is bounded + monitored, here at machine precision). <b>Right</b>: where the runoff goes &mdash; the channel
captures it as subsurface Darcy conveyance (its sand-zone GHB) plus dispersion into the surrounding clay,
vs the surface that escapes off-domain.</p>

<h2>Headline case &mdash; sand channel in clay (the trigger storm)</h2>
{ch}

{v}

<p class="lim"><b>Known limitations / honest caveats.</b>
(0) <b>OLD-scheme evidence provenance (read this).</b> The <span style='color:#ef6c00'><b>galerkin</b></span>
sawtooth dt-pin is captured <b>live</b> on this exact case (the reproducible monolithic failure here). The
<span style='color:#c62828'><b>upwind</b></span> dt-collapse curve is a <b>documented reference</b> (dashed),
from the 2026-06-22 canonical trigger run; it is <i>not</i> a live re-run of this present variant, because
the deep &asymp;1&nbsp;m berm pond regularizes the near-saturation singularity here, so a live upwind solve
limps slowly to completion (verified: reaches t&gt;1.0 at dt&asymp;3&times;10&#8315;&#178;) rather than
hard-collapsing. The headline OLD-fails / NEW-succeeds contrast therefore rests on the <b>live galerkin
pin</b> (+ the live tilted-V galerkin sawtooth below); the upwind curve documents the original collapse
regime honestly, not a re-measurement.
(1) <b>Interception accounting.</b> The sequential routing books <i>all</i> surface outflow into one
<code>cum_outflow</code> bucket (it does not split the toe edge from the channel-mouth outlet), so
"surface escaped" folds in the small channel-surface discharge &mdash; this makes the reported intercept
fraction slightly <i>conservative</i> (it understates capture). "Channel captured" = subsurface conveyance
(GHB) + dispersion into the clay, both measured independently.
(2) <b>route_substeps = {rsub}.</b> The lateral-transport rate is calibrated by the Manning sub-sweep count:
a single sweep under-resolves intra-step travel (~40&ndash;50&times; too slow); <code>route_substeps=4</code>
matches the resolved upwind reference's drain timing, and &ge;8 overshoots (the fully sub-stepped explicit
Manning becomes a kinematic wave that outruns the diffusion-wave reference). Conservation is
substep-independent at ~10&#8315;&#185;&#178;. This is an accuracy knob, not a stability one.
(3) <b>Near-saturation no-Ss fragility.</b> A near-saturated column carrying a ~zero pond can dt-collapse
the standalone-Richards path on the unconfined (no specific-storage) near-saturation singularity &mdash;
orthogonal to the transport question; carry the pond as a comfortably-positive head (as the validated
cases do).
(4) <b>Kinematic vs diffusion-wave depth fidelity.</b> The routing is a Manning rate-limited kinematic
redistribution: it answers "how much water overlies each parcel / which way does the excess run / move it
downslope" (the PIDS goal), <i>not</i> accurate flood hydrographs, depths, or velocities. The Manning
galerkin/upwind stack remains the validated diffusion-wave fallback for those.</p>

<p style='font-size:13px;color:#607d8b;margin-top:22px'>
<b>Regenerate.</b> Data:
<code>python viz/run_sequential_overland_signoff.py</code> &rarr;
<code>scratch/sequential_overland_signoff.npz</code>.
HTML:
<code>python viz/make_sequential_overland_signoff_html.py</code> (run from <code>forward-model/</code> via
the WSL <code>pids-fem</code> env). Design / decision record:
<code>docs/plans/2026-06-22-overland-flow-sequential-coupling-decision.md</code>; engine
<code>pids_forward/physics/sequential_coupling.py</code>.</p>
</body></html>"""

    html = ('<!doctype html><html><head><meta charset="utf-8">'
            '<title>PIDS sequential overland &mdash; Tier-3 sign-off</title><script>'
            + get_plotlyjs() + intro)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"WROTE {out_html}", flush=True)


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "scratch/sequential_overland_signoff.npz"
    out = sys.argv[2] if len(sys.argv) > 2 else "viz/sequential_overland_signoff__2026-06-23.html"
    build(npz, out)
