# Adversarial verification — NC / UNC faculty-gated research-commercialization money

**Claims under test:** C-016, C-017 (NCInnovation), C-018 (UNC KickStart), C-019 (UNC OTC) — the faculty-gated NC awards from `investors/research/2026-06-08-unc-triangle-ecosystem.md` (whose original figures were entirely snippet-grade; the source agent could not deep-fetch primary pages).
**Stance:** *Attempt to refute.* Default to UNVERIFIED unless stated verbatim on an official primary page. The decision-affecting question: **what does a UNC faculty partnership unlock that an alum (Arik) cannot reach alone?**
**Method:** Agent-assisted adversarial run, 2026-06-22; live WebFetch + curl of innovate.unc.edu (loads directly); NCInnovation site is behind a JavaScript WAF (Bunny Shield, 403s all non-browser fetches) so its **official pages were recovered via Wayback Machine raw captures** (snapshots Dec 2025 / Feb 2026).
**Sources:** S-010 (NCInnovation), S-011 (Innovate Carolina / UNC OTC).

---

## C-016 — NCInnovation keystone (private companies ineligible) → RESULT: NOT REFUTED (confirmed verbatim)
Claims: (a) ~$500M state endowment; (b) funds NC **public-university** applied research at TRL 3–6; (d) **private for-profit companies are NOT eligible** — reachable only through a university researcher.
- (a) **Confirmed** — "Just the Facts": *"NCInnovation uses the interest and income from a $500 million State-funded endowment to provide non-dilutive grant funding…"* + news: *"NCInnovation endowment now has full $500 million prescribed in 2023-25 state budget."*
- (b) **Confirmed** — grants-overview: *"NCI focuses on the middle phase – TRLs 3-6"*; *"Provides grant funding for university applied researchers to mature a proof of concept…"*
- (d) **Confirmed (THE keystone), two independent primary statements** — "Just the Facts": *"We do not fund or invest in private companies, take equity positions of any kind, or benefit from the future financial success of grant recipients."* + *"NCInnovation provides grant funding to North Carolina public university applied researchers."*; pilot-grant announcement: *"Private companies are not eligible."*
- Refutation attempt: searched the live site for any private-company-eligible path → **none found.**
- **Recommended status: `single-checked`** (single issuer, two pages; recovered via Wayback). The faculty-partnership thesis's foundation is solid.

## C-017 — NCInnovation per-grant award range "$300K–$1.1M" → RESULT: CANNOT CONFIRM
- The figure appears on **no reachable primary page** (grants-overview, /grants/, "Just the Facts", award announcements publish only aggregate totals + project topics, no per-grant range). The "$300K–$1.1M" (and variant "$200K–$1M anticipated") lives only in WebSearch summaries and the official **Policies & Procedures PDF / "applying-for-a-grant" page, both WAF-blocked with no Wayback capture.**
- **Recommended status: `unverified`.** Do not cite the per-grant range until the primary policy doc is obtained (request from NCInnovation directly).

## C-018 — UNC KickStart (amounts + the alum question) → RESULT: NOT REFUTED (faculty/IP gate confirmed)
Claims: (a) $5K–$50K; (b) requires a UNC faculty founder + a startup based on UNC IP.
- (a) **Confirmed** — KickStart Grant Awards: *"The awards are in the $5,000 to $50,000 range, depending the on the company's needs."*; Venture Services Fund: *"Awards range from $5,000 to $50,000…"*
- (b) **Confirmed** — Venture Services Fund: *"Startup must be based on UNC IP (does not need to be licensed) or a UNC research innovation"* and *"Requests for applications (RFAs) are solicited on a quarterly basis from faculty founders."*; Get Funding: *"…funding of up to $50,000 to UNC-Chapel Hill startups that are founded on University intellectual property… If you are a faculty founder who would like to be considered, you may participate in the quarterly request-for-applications process."*
- **ALUM QUESTION (decision-critical):** no primary-source path exists for a non-faculty alum with no current UNC affiliation and no UNC IP. Entry is solicited *"from faculty founders"* with a startup *"based on UNC IP."* → **a UNC faculty partner + UNC-IP basis is required.** Refutation failed.
- Notes: exact "faculty/staff/student" enumeration + "non-dilutive" label were NOT verbatim on the primary page (deferred to a linked eligibility doc); the site states RFA cadence inconsistently ("annual" vs "quarterly").
- **Recommended status: `single-checked`.**

## C-019 — UNC OTC Technology Development (Translational) Grants → RESULT: NOT REFUTED (confirmed; amount discrepancy noted)
Claim: up to $50K, tied to UNC inventors / UNC IP.
- **Confirmed** — OTC page: *"Funding levels up to $50,000 are available and respective eligibility and milestones are determined in partnership with OTC staff."*; eligibility (Get Funding): *"If you are a full- or part-time UNC-Chapel Hill employee (faculty, trainees or staff), you are eligible to apply. You must be seeking to develop or commercialize a technology or invention created at the University…"*
- Discrepancy: Get Funding also says *"Most grants range from $5,000 to $25,000"* → $50K is the ceiling, typical awards lower.
- Alum answer: requires a **current full/part-time UNC employee** developing a University-created invention → a non-affiliated alum is not eligible.
- **Recommended status: `single-checked`.**

---

### Net for the decision
**The UNC keystone holds and is now primary-confirmed:** NCInnovation bars private companies (money flows only through a UNC-System university researcher), and KickStart + OTC both gate on a current UNC faculty/employee + UNC IP. **Arik-as-alum has no confirmed direct path into any of the three** — a UNC faculty research partnership is precisely what unlocks them, exactly as the keystone thesis claimed. Two figures need follow-up: the NCInnovation per-grant range ($300K–$1.1M, unconfirmed) and the exact KickStart eligibility enumeration (on a linked doc not yet fetched).
