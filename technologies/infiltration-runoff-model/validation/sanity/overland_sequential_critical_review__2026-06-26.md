# Critical review — is the sequential approach really "exhausted"? (Codex, 2026-06-26)

Commissioned by Arik (who was NOT convinced by §16's "monolith is the only path"): independent,
SKEPTICAL review by Codex (gpt-5-codex, effort=high) challenging the §16 conclusion. Prompt:
`scratch/_codex_critical_review_prompt.txt`. **Outcome: §16 is PREMATURE — Arik was right.**

## Verdict (Codex, verbatim key points)

**1. §16 over-claims — we never tested the canonical sequential closure.** The production sequential
path explicitly says "NO hard Dirichlet pin" (`sequential_coupling.py:18`) and assembles only pond
storage + rain + lateral source (`:522`). Every scratch variant is either an explicit Neumann-influx cap
(`seq_capped_infiltration_prototype.py:17`, `seq_href_cap_spike.py:56`) or an "offer film in ψ via source,
let Richards draw it" scheme (`seq_href_cap_spike.py:151`, `seq_href_iterated.py:146`). **None is a
Neumann↔Dirichlet switching top BC.**

**2. The monolith is NOT a switching-Dirichlet closure either** — it is a co-solved Robin/NCP exchange law
(`q_pot=kirchhoff(ψ,d)/ell_c`, `coupling.py:230`; λ injected as the soil-top flux `:584`). So "the monolith
works ⟹ sequential needs co-solve" is NOT a clean inference about standard switching-BC sequential coupling;
it is only an inference about OUR specific `q_pot` film law.

**3. The gravity criticism is right (with a nuance).** `kirchhoff(a,b)` DOES include the saturated segment
(`K=Ks` for ψ≥h_s, `constitutive.py:55,147`) so it is not purely sorptive — but it behaves like a
FILM-RESISTANCE law, not a Dirichlet Richards boundary. As `ψ_top→0` and `d≪ell_c`, `q_pot≈Ks·d/ell_c`
(small), whereas a true ponded DIRICHLET Richards solve carries the unit-gravity term `K(∇ψ+e_g)`
(`richards.py:31`) and tends toward **~Ks at steady ponded infiltration**. **⟹ the `qpot/qpot_d` family is
structurally missing the Dirichlet/gravity limit — a STRONG reason not to generalize from its failure to
"sequential exhausted."**

**4. An older spike got closer but does not count.** `scratch/overland_split_spike.py:647` has a
separate-store + ponded-node Dirichlet head + reaction-flux idea, but it is outside the §16 evidence base
and was not a clean canonical active-set switch.

**VERDICT: §16 is PREMATURE.** The evidence supports only the NARROW claim: *our* `pond-in-ψ` and
`q_pot=kirchhoff/ell_c` closures do not generalize across soils. It does NOT support "soil-general accuracy
requires the monolith" — the standard sequential closure was never tested.

**Nuance on the off-the-shelf examples:** CATHY is the right kind (sequential + switching BC). But
HydroGeoSphere and ParFlow are actually INTEGRATED/co-solved (not evidence for independent sequential
modules). GSSHA and tRIBS get the partition right with **infiltration-capacity** models (Green-Ampt / GAR /
Smith-Parlange) — a DIFFERENT closure family from our `q_pot` cap, also untested here.

## Recommended next implementation — a TRUE active-set switching BC (separate surface store)

NOT another `q_pot` surrogate. NOT pond-in-ψ. The canonical CATHY-style switch:
1. Separate surface store `d` (not in ψ).
2. Non-ponded top nodes → atmospheric/rain **NEUMANN** flux.
3. Ponded nodes → `ψ_top = d` (or `ψ_top = 0`, thin-sheet first pass) as a true **DIRICHLET** BC
   (`richards.py:131` `add_dirichlet`).
4. Solve Richards.
5. **Read the realized top infiltration on the Dirichlet nodes as the BOUNDARY REACTION FLUX** (the FEM
   residual at the constrained DOFs) — prescribed Neumann on dry nodes. *(This is where gravity enters for
   free.)*
6. Conservative surface update: `d^{n+1} = d^n + (rain + runon − infil − routed_out)·dt`.
7. Recompute the ponded set + Picard-iterate the active-set switch within the step until the ponded mask
   and `d` stabilize.
Shortest path: resurrect the old separate-store spike as scaffolding, but make it a REAL active-set
Neumann↔Dirichlet switch, not a hard-pinned heuristic.

## Sources (partitioned Richards/surface-water coupling via BCs is standard + iterative)
- Schüller, Birken, Dedner (2025), arxiv 2408.12582 — partitioned Richards/shallow-water via BCs, iterative.
- Sochala, Ern, Piperno (2009), arxiv 0809.1558 — mass-conservative coupling, continuity of pressure + normal flux.
- Berninger et al. (2014), arxiv 1301.2488 — coupled Richards/surface-water via nonlinear interface conditions.

## My (Claude's) correction
§16's "sequential-cap EXHAUSTED / monolith-only" was an OVER-REACH. The honest scope: the `pond-in-ψ` +
`q_pot=kirchhoff/ell_c` closure FAMILY does not generalize. The CANONICAL switching-BC closure (reading
infiltration off a Dirichlet Richards solve — the standard CATHY/HydroGeoSphere/GSSHA approach) was never
tested and is the clear next step. The fork is NOT yet B-vs-C.
