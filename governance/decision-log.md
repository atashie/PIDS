# Decision Log

Dated record of major decisions (Guardrail #4). Newest at the bottom. Each large addition/alteration to the repo should add a row.

| Date | Decision | Context / rationale |
|------|----------|---------------------|
| 2026-06-01 | Adopt a **two-axis architecture** — `technologies/` (science) and `applications/` (use cases), joined by an explicit `integration/` layer. | Technologies are reusable building blocks; use cases compose them differently. Keeps concerns independent and makes holistic review tractable. |
| 2026-06-01 | Define **three technology pillars**: PIDS physical system, infiltration-runoff model, aerial mapping — as a **Sense → Solve → Build** pipeline. | Mapping (sense) feeds the model (solve), which designs the physical system (build); field results validate the model. |
| 2026-06-01 | Repository is **private, minimal disclosure process**. | Planning phase; revisit before any public release. |
| 2026-06-01 | Adversarial review implemented as **both documentation discipline and agent-assisted** checks. | Strongest integrity guarantee for third-party and derived numbers. |
| 2026-06-01 | PIDS app. **18/632,186 is in allowance**; prosecution defense is **out of scope**. Patent-strengthening science is **lower priority**, retained for credibility & future filings. | The examiner's rejections were overcome; no live prosecution fight. |
| 2026-06-01 | Pillar 1 folder named **`pids-physical-system/`**. | Disambiguates the physical-system pillar from the repo/company umbrella (also "PIDS"). |
| 2026-06-01 | Build cadence: **chunk by chunk**, with a review gate after each chunk. | Matches the cautious-building guardrail. |
| 2026-06-01 | Applications refined: **eliminating off-site discharge** (vertical infiltration to the deep subsurface) is a **cross-cutting value proposition**; agriculture targets **managed aquifer recharge (MAR)**; suburban gains a **drought-amelioration** co-benefit. | Sharpens the value proposition and the model's objective functions. |
| 2026-06-01 | **Chunk C left intentionally under-specified** — `applications/` (4 use-case folders) and `integration/` scaffolded with high-level READMEs only; detailed design (per-application specs, dependency matrix, interfaces) deferred. | Per user direction; revisit later. |
