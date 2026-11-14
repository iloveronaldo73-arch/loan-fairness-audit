# Loan Fairness Audit — one page

**What it is.** A single-page audit of a mortgage-approval model on **HMDA 2025**
public loan-level data. It compares four decisions — the ones actually recorded,
a model that sees demographics, the same model blinded to them, and that blind
model with a separate cutoff per group — and probes whether the remaining columns
can rebuild the trait they were meant to hide.

**Why the rule matters.** A gap is a measurement; legality depends on the rule.
On **April 22, 2026** the CFPB issued a final rule amending Regulation B, which
implements ECOA — the statute behind fair-lending review of this data (91 FR
21620, document **2026-07804**). Effective **July 21, 2026**, it removes the
"effects test" and states that ECOA does not authorize disparate-impact
liability. Disparate treatment and intentional proxy use stay prohibited; the
Fair Housing Act and state law did not change, and the rule is being challenged
(*Nat'l Fair Housing Alliance v. CFPB*, No. 1:26-cv-01820, D.D.C.). The audit
still measures gaps; they now sit against a narrower liability theory. Not legal
advice.

**Data.** HMDA 2025, Georgia, Maryland, Michigan and Arizona, home-purchase and
refinance applications marked originated or denied: 614,687 filings, of which
445,112 are from White or Black applicants who reported an age. Those are cut to a
fixed **100,000-application sample** (17,772 Black, 5,636 under 25) so 50 splits
stay tractable; **84.5% were approved**. Only application-time fields are used —
pricing, purchaser, HOEPA and denial reasons would leak the label. HMDA has no
credit score.

**Method.** Numeric fields imputed and scaled, categoricals one-hot encoded,
logistic regression per strategy. Each cutoff is calibrated on validation to
reproduce the recorded 84.5% approval rate; at this base an uncalibrated 0.5 cut
would approve too many and hide gaps. The per-group variant approves each group
at the overall rate. Every number averages **50 random 60/20/20 splits**, scored
on the test part. The proxy check fits the unprotected columns to the trait and
reports AUC (0.5 = chance).

**Results** (approval rate on test data, averaged over 50 splits):

| trait | strategy | under 25 / Black | 25+ / White | accuracy | gap (95% range) |
|---|---|---|---|---|---|
| age | observed in data | 85.1 | 84.6 | — | +0.5 (−0.9 to +2.5) |
| age | full model | 85.3 | 84.5 | 88.0 | +0.8 (−0.9 to +2.1) |
| age | blind model | 84.4 | 84.5 | 87.8 | −0.1 (−2.0 to +1.2) |
| age | blind + group cutoffs | 84.6 | 84.5 | 87.8 | +0.1 (−3.5 to +2.4) |
| race | observed in data | 74.6 | 86.8 | — | −12.2 (−13.3 to −11.1) |
| race | full model | 68.3 | 88.0 | 88.0 | −19.7 (−21.4 to −17.7) |
| race | blind model | 75.3 | 86.5 | 87.8 | −11.2 (−12.8 to −9.8) |
| race | blind + group cutoffs | 84.6 | 84.5 | 87.4 | +0.1 (−1.4 to +1.5) |

**Reading it.**

1. *The race gap is large and precisely measured.* Black applicants were approved
   74.6% of the time against 86.8% for White applicants — 12.2 points, and the
   band across splits now spans only 2.2 points.
2. *The model that sees race makes it worse.* At the same overall approval rate it
   cuts Black approvals to 68.3% (−19.7) for 0.2 points of accuracy. Its rankings
   separate the groups more sharply than lenders' own decisions did.
3. *Blinding barely helps.* Removing every demographic column leaves −11.2, not
   zero. The proxy check says why: tract demographics, loan size, DTI and program
   flags rebuild Black status at AUC **0.84**, so the pattern survives the column
   being dropped.
4. *Only the forbidden strategy closes it.* Per-group cutoffs reach +0.1 at a cost
   of 0.6 accuracy points — but they read the protected trait at decision time,
   the disparate treatment that survived the 2026 amendment. Parity and legality
   remain different tests.
5. *Age shows no gap here.* Under-25 applicants are approved at 85.1% against
   84.6% (+0.5, −0.9 to +2.5), and the models keep it near zero. The earlier
   Oregon purchase-only run read −3.2, which is a warning about conclusions drawn
   from one state and one product.

**Limits.** Four states, two loan purposes, one year; self-reported demographics
with large "not available" shares; no credit score, so this is neither a lender's
model nor a lender's denial pattern. An approved loan is not a fair loan —
pricing and terms are outside this page. Tight bands reflect a large sample, not
certainty about other states, products or years. Hackathon prototype; not
financial or legal advice.

**Reproduce.**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py          # audit + comparison -> results/
.venv/bin/python app.py serve    # http://127.0.0.1:4173
```

The first run downloads a 224 MB HMDA extract (about 45 MB compressed) and
caches it in `data/`. `audit/dataset.py` holds the query, filters and sample
seed, `audit/audit.py` every number above, and `audit/build_site.py` injects them
into `audit/template.html`, the whole page; `results/index.html` is that page.
