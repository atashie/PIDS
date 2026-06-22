# Adversarial verification — Alabama (UA) funding claims

**Claims under test:** C-014 (ARDEF), C-015 (Innovate Alabama Supplemental Grant) — decision-affecting Alabama figures from `investors/research/2026-06-22-multi-university-academic-funding.md`.
**Stance:** *Attempt to refute.* Default to DISPUTED unless stated verbatim on an official primary page/PDF.
**Method:** Agent-assisted adversarial run, 2026-06-22; live WebFetch of ADECA + Innovate Alabama pages, direct text-extraction of the ADECA ARDEF Program Guide PDF (PyMuPDF), and Innovate Alabama official press releases.
**Sources:** S-008, S-009.

---

## C-014 — ARDEF (the NCInnovation analog) → RESULT: NOT REFUTED (confirmed verbatim; two corrections)
Claims: (a) university-research-entity + private-sector-applicant pairing; (b) 50% match required, state funds can't count; (c) private partner may be HQ'd outside AL; (d) FY27 deadline 2026-07-29; (e) max award.
- Fetched: https://adeca.alabama.gov/ardef/ (loaded) + ARDEF Program Guide PDF ("Revised February 2024", text-extracted) + ADECA FAQ.
- (a) **Confirmed** — Program Guide "Eligibility": *"Eligible research entities must include one or more of the following: ▪ A public or private university in the state in partnership with a private sector applicant; ▪ A university research foundation… in partnership with a private sector applicant; ▪ A public two-year college… in partnership with a private sector applicant; ▪ A publicly-owned hospital… in partnership with a private sector applicant; ▪ An entity duly formed, domiciled or qualified to do business in the state in partnership with a private sector applicant…"*
  - **Correction:** "eligible research entity = a university" is **too narrow** — a university is the lead example, but two-year colleges, publicly-owned hospitals, and qualified in-state research entities also qualify. **A private-sector partner is required in every case** (the partnership structure — the faculty-gated thesis — holds).
- (b) **Confirmed** — FAQ: *"The applicant is required to provide a match equal to fifty percent (50%) of the total project cost."*; *"State funds cannot be used as match."* Program Guide "Match": *"Ineligible sources of match include state funds and in-kind cost share."*
- (c) **Confirmed** — FAQ verbatim: *"Can the private sector industry partner be headquartered outside of Alabama? Yes."* (The *research entity* must be in-state: HQ + ≥75% property/payroll in Alabama.)
- (d) **Confirmed** — ADECA references the FY27 ARDEF application with deadline *"July 29, 2026 at 11:59 p.m. CST."*
- (e) **Max award = no fixed dollar ceiling.** Program Guide "Funding": *"Individual grants awarded by ADECA may not exceed the lesser of: 1. Twenty percent (20%) of the total grant funds awarded in a single fiscal year, or 2. Fifty percent (50%) of the budgeted project costs."* (Indirect/admin ≤10%.) The binding cap is **relative + appropriation-dependent**, not a published dollar figure (the research doc's "no stated max" was correct).
- **Recommended status: `single-checked`** (one adversarial agent pass against ADECA primary sources incl. the Program Guide PDF).

## C-015 — Innovate Alabama Supplemental Grant (the One-NC analog) → RESULT: caps NOT REFUTED; "no residency requirement" REFUTED
Claims: (a) state match to a federal SBIR/STTR award; (b) Ph I 50%/$100K, Ph II 50%/$250K, non-dilutive; (c) requires an active federal award; + the open question of an Alabama residency requirement.
- Fetched: https://innovatealabama.org/programs/supplemental-grant-program/ (loaded) + the "updated" program page + two official press releases (third-round SBIR/STTR funding; "nearly $4M awarded").
- (a) **Confirmed** — *"Small businesses who have an active Phase I or Phase II Federal Small Business Innovation Research (SBIR) or Small Business Technology Transfer (STTR) grant are eligible for non-dilutive funding…"*
- (b) **Confirmed** — Phase I: *"eligible to apply for 50% of your SBIR/STTR grant up to $100,000"*; Phase II: *"…up to $250,000."* Non-dilutive: *"recipients who do not need to exchange shares of their company for the grant."*
- (c) **Confirmed** — applicants without an active SBIR/STTR award are *"not eligible."*
- **Residency requirement — EXISTS (refutes the "no AL requirement" reading).** The live program page is silent, but the official press release states verbatim: *"Before, businesses interested in applying for SBIR grants had to have at least one of their top executives and 75% of their employees residing in Alabama when applying. This new legislation has loosened the restriction, allowing applicants a 12-month window after receiving grant funds to relocate to Alabama and fulfill the other residency requirements."* The "$4M" release confirms recipients agreed to *"relocate their headquarters to Alabama… within the next 12 months."*
  - **Net:** an Alabama HQ/residency condition applies, but post-2025 ("The Game Plan" legislation) it is a **post-award condition** — apply from out of state, then relocate HQ + majority of team within **12 months**.
- **Recommended status: `single-checked`** for the caps/match/non-dilutive/active-award facts; the residency correction is primary-confirmed via press release (live program-page eligibility text does not state it — verify the exact current statutory wording before relying).

---

### Net for the decision
Alabama is confirmed as the **strongest analog**: ARDEF (faculty-gated collaborative R&D, 50% match) + the Supplemental Grant (a real **state 50% SBIR/STTR match up to $250K**). Two decision-affecting nuances now resolved: ARDEF has **no fixed dollar max** (relative/appropriation-bound), and the Supplemental Grant **does carry an Alabama residency condition**, now satisfiable within **12 months post-award** — so PIDS can apply from out of state but must plan to relocate HQ + majority of team to Alabama. This makes **state of domicile** a live strategic decision (mirrors the Virginia residency tests).
