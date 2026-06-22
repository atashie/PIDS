# Adversarial verification — Virginia (VIPC) funding claims

**Claims under test:** C-008, C-009, C-010, C-011 (decision-affecting Virginia figures from `investors/research/2026-06-22-multi-university-academic-funding.md`).
**Stance:** *Attempt to refute.* Default to DISPUTED unless the figure is stated verbatim on an official VIPC primary page. Third-party aggregator snippets do **not** count.
**Method:** Agent-assisted adversarial run (general-purpose web agent), 2026-06-22; live WebFetch of VIPC pages + direct text-extraction of the HE-POC Program Guidelines PDF and the Lab-to-Launch Terms & Conditions PDF.
**Sources:** S-004, S-005.

---

## C-008 — HE POC excludes for-profit applicants → RESULT: NOT REFUTED (confirmed verbatim)
Claim: VIPC Commonwealth Commercialization Fund (CCF) **Higher-Education Proof-of-Concept (HE POC)** requires the applicant to be a VA university / its IP foundation / a VA federal research facility / a university research consortium / a VA nonprofit research institute — **a for-profit company cannot apply** to this track.

- Fetched: https://vipc.org/funding/he-poc/ (HTML loaded) + https://vipc.org/wp-content/uploads/HE-POC-Program-Guidelines.pdf (text extracted).
- Verbatim (eligible applicants): *"Virginia public or private institution of higher education or its associated intellectual property foundation"*; *"Federal research facility located in Virginia"*; *"University research consortium that includes Virginia college and university member institutions"*; *"Other nonprofit research institution located in Virginia whose purpose includes performing basic and/or applied scientific research."*
- Refutation attempt: searched the live page + guidelines for any statement that a for-profit company may apply to HE POC → **none found.** All five enumerated applicant types are research/academic entities.
- **Recommended status: `single-checked`** (one adversarial agent pass against primary source; elevate to `adversarially-verified` on human sign-off). This is the core "faculty-gated" fact and is solid.

## C-009 — HE POC award amounts → RESULT: NOT REFUTED (confirmed verbatim)
Claim: Track 1 ≤$75K, Track 2 ≤$150K, discretionary Advanced Commercialization ≤$300K, 1:1 match.
- Verbatim: *"Applicants may request $75,000 or $150,000, depending on the designated track… Applicants with technologies at a more advanced technology readiness level (TRL)… may be invited to seek up to $300,000. Each grant requires a minimum one-to-one match."*
- **Recommended status: `single-checked`.**

## C-010 — VIPC SBIR/STTR "Phase 0 ~$3K / ~$100K post-award match" → RESULT: REFUTED / UNSUBSTANTIATED
Claim (from the research doc, flagged snippet-only): VIPC provides up to ~$3,000 Phase-0 prep + up to ~$100,000 post-award SBIR/STTR match.
- Fetched (all loaded): https://vipc.org/initiatives/sbir-sttr/ , https://vipc.org/funding/startups/ , https://vipc.org/resources/sbir-sttr/
- Findings: the Federal Funding Assistance Program (FFAP) pages describe **services only**. The startups page: *"funding assistance for select companies seeking their first Phase I or II award (to help pay for some of the above services)"* and quotes only the **federal** range (*"$150K to $1.5M"*), not a VIPC match. The resources page: VIPC *"facilitate[s] a Phase 0 grant award program to help phase I and phase II applicants hire professional consultants"* — **no dollar figure.**
- **No VIPC primary page states any $3,000 Phase-0 amount or any $100,000 post-award match.** Those numbers are aggregator artifacts (Eva Garland / SBIR.org), not primary.
- **Recommended status: `disputed`.** The *program exists* (FFAP + a Phase-0 consultant-grant); the *dollar figures we cited do not*. Do not use $3K/$100K. To resolve: contact VIPC's federal-funding office for the actual amounts.

## C-011 — VA for-profit residency/HQ eligibility tests → RESULT: PARTIALLY REFUTED (tests confirmed; framing + grant-cap wrong)
Claim: the CCF private-sector track requires VA HQ, ≥50% senior management VA-resident, ≥50% founder ownership VA-resident, and ≤1 prior VIPC grant.
- The main CCF landing page (https://vipc.org/funding/commonwealth-commercialization-fund/) shows only the three **university** tracks and contains **none** of the residency language.
- The residency/HQ tests were found verbatim in the **Lab-to-Launch (L2L)** for-profit CCF Terms & Conditions PDF (https://vipc.org/wp-content/uploads/L2L-TCs.pdf, extracted locally): *"Be a for-profit Virginia-based company, defined as: Headquartered in… Virginia; … Virginia is the primary state of residence for 50% or more of senior management, and at least 50% of founders' ownership is held by Virginia residents."*
- Corrections: (1) the tests belong to the **L2L track**, which *additionally* requires the company to be a **spinout licensing IP from one of Virginia's six R-1 universities** — i.e. this for-profit money is itself **university-IP-gated**, not a clean "founder-direct" path. (2) The **"≤1 prior VIPC grant" cap is REFUTED** — the doc only says prior-grant *performance* is weighed: *"award decisions shall take into consideration the Applicant's performance on and compliance with their prior VIPC grant award."*
- **Recommended status: `single-checked`** for the residency/HQ tests (confirmed verbatim) **with a `disputed` note** on the one-grant cap and a correction that the for-profit CCF path (L2L) is university-spinout-gated.

---

### Net for the decision
The **faculty-gated thesis strengthens**: even Virginia's main *for-profit* CCF money (L2L) requires a VA-R1-university IP license, so a VT/UVA faculty/IP tie is central there too — not just for HE POC. The **VIPC SBIR/STTR match dollar figure must be dropped** until VIPC confirms it. Virginia residency/HQ tests are real and decision-affecting for PIDS's state of domicile.
