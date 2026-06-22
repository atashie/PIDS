# Adversarial verification — Texas (SMU) funding claims

**Claims under test:** C-012, C-013 (decision-affecting Texas figures — both NEGATIVE claims — from `investors/research/2026-06-22-multi-university-academic-funding.md`).
**Stance:** *Attempt to refute.* For a negative claim, refutation = actively hunting for a counterexample that would make it false.
**Method:** Agent-assisted adversarial run, 2026-06-22; live WebFetch of SSTI, Texas Legislature bill records, THECB, the Governor's office, and the Texas Education Code.
**Sources:** S-006, S-007.

---

## C-012 — No statewide Texas SBIR/STTR cash match → RESULT: NOT REFUTED (negative survives)
Claim: Texas has no statewide SBIR/STTR matching grant; the only Texas cash match is the City of San Antonio (Ph 0 ≤$2K / Ph I ≤$50K / Ph II ≤$75K, SA-HQ only).
- Fetched (loaded): https://ssti.org/state-sbirsttr-resource-guide ; https://greatersatx.com/site-selectors/incentives/sbir-sttr-matching-grants ; Texas Legislature bill text + history for SB 209 / HB 1268 (89R) and HB 2466 (88R) at capitol.texas.gov ; the LRL list of 89R bills that became effective ; https://gov.texas.gov/business/page/incentives
- SSTI (Texas entry): lists only *"Texas SBIR/STTR Assistance Program (Pathways)"* = *"No-cost or discounted proposal development assistance"* — **no match line.** SSTI's matching-fund states are NJ/IA/KY/MD/MA/SC, **not Texas.**
- San Antonio (verbatim): *"Phase 0: up to $2,000"*; *"Phase I … up to $50,000"*; *"Phase II … up to $75,000"*; *"Be headquartered in the City of San Antonio."*
- **Strongest counterexample investigated + defeated:** secondary pages assert a live "Texas Technology Innovation Program" matching up to $50,000 effective 2025-09-01. This program exists **only in failed legislation** — SB 209 (last action *"Not again placed on intent calendar,"* 2025-05-08) and HB 1268 (89R), plus HB 2466 (88R); **none passed** or appear in the LRL effective-bills list.
- **Recommended status: `adversarially-verified`** — the negative survived a deliberate refutation across ≥2 independent methods (SSTI third-party guide + primary legislative records + Governor's incentives page).
- **Caveat for the register:** a statewide match is a *recurring legislative proposal* and could be enacted in a future session — "no state match" is true **as of 2026-06**, not a permanent fact. Re-check each session.

## C-013 — TUF / NRUF / GURI are public-university-only → SMU (private) ineligible → RESULT: NOT REFUTED
Claim: Texas's flagship state research funds (Texas University Fund, National Research University Fund, Governor's University Research Initiative) are restricted to public universities; SMU is ineligible and none reach a startup.
- Fetched: https://gov.texas.gov/business/page/guri ; https://www.highered.texas.gov/research-funding-in-texas/research-funding-and-programs/ ; Tex. Educ. Code § 61.003 (statutory enumeration).
- GURI (verbatim): *"Eligible Texas public institutions of higher education attempting to recruit distinguished researchers"*; applicant *"cannot recruit a distinguished researcher from… a private or independent institution of higher education."*
- TUF (THECB, verbatim): *"TUF-eligible general academic institutions must expend an average of $20 million in federal and private research annually… award an average of 45 research doctoral degrees annually, and not be eligible for the Permanent University Fund (PUF)."* Eligible set = UH, UNT, TTU, TSU (all public).
- Tex. Educ. Code § 61.003 "general academic teaching institution" = an explicit enumeration of **public, state-supported** universities; SMU is not in the list.
- **No counterexample found.** All three funds key on statutory public/PUF-eligible categories that exclude private SMU.
- **Recommended status: `adversarially-verified`** (statute + two official pages; survived refutation).

---

### Net for the decision
Both negative findings hold: the **One-NC analog (state SBIR/STTR match) is absent in Texas**, and SMU cannot reach the state research endowments. This confirms reframing the SMU relationship around **federal STTR + a named PI + Dallas/TWDB municipal recharge pilots**, and considering a **Texas public co-PI** if state/federal research dollars are the goal. Watch for a future-session statewide-match enactment.
