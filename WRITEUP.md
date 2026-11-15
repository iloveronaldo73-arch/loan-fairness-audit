# Loan fairness audit

**Track:** AI + Finance (starting point 4.2.3, fairness audit of a loan model)
**Team:** _names, emails and schools go here_
**Code:** `fairness_audit.ipynb` (this repository), executed with its outputs saved

## The problem

A housing counselor at a nonprofit sits with an applicant whose mortgage was
declined. The lender's notice gives a reason code, but not the answer the
applicant needs: was this about my finances, or about who I am? HMDA 2025, the
public file US regulators use for fair-lending review, says the question is live.
In Georgia, Maryland, Michigan and Arizona, **86.7% of White applicants were
approved and 74.3% of Black applicants were not** — a gap of 12 points that no
individual applicant can see.

## What we built

A one-page audit of a mortgage-approval model, plus a notebook that explains any
single decision. **In:** the fields present on an application — loan amount,
loan-to-value, income, debt-to-income, loan type, underwriting system, property
and census tract. **Out:** an approval score, the decision at a calibrated
cutoff, and the three columns that moved the score most.

The counselor uses it two ways. They paste an application and read the reasons,
which are columns the applicant can check on their own form, so a real problem
can be fixed. And they read the page, which compares four decision strategies
and shows how much of the 12-point gap survives each one.

## What we added

The brief's version uses German Credit, a 1994 dataset of 1,000 applicants with
no modern demographics. We replaced it with **HMDA 2025** — 100,000 real
applications across four states and two loan purposes — because that is the file
the law actually applies to, and because a 12-point gap on it is worth measuring.

We also asked a question the brief does not: **does removing race fix it, and
what does closing the gap cost?** We test blinding, a proxy probe, and five
cutoff rules, and we report the accuracy price of each.

## How it works

**Data.** 614,687 filings for 2025, narrowed to 445,112 applications from White
or Black applicants who reported an age, then to a fixed random sample of
**100,000** (17,772 Black, 5,636 under 25) so 50 resamples stay tractable. 84.5%
were approved. Only application-time fields are modelled: interest rate, points,
fees, purchaser type and the lender's denial reasons are reported after the
decision and would leak the label. HMDA has no credit score.

**Model.** Median imputation and scaling for numeric columns, one-hot encoding
for categoricals, logistic regression. Each cutoff is calibrated on a validation
slice so the model approves the same share of applications the lenders did — at
an 84.5% base rate an uncalibrated 0.5 cut approves almost everyone and hides
every gap. Every number averages **50 random 60/20/20 splits**, scored on the
held-out test part.

## Results

Approval rate on test data; the baseline approves everyone.

| strategy | Black | White | accuracy | gap (95% range) |
|---|---|---|---|---|
| Baseline: approve everyone | 100.0 | 100.0 | 84.5 | +0.0 |
| Observed in the data | 74.6 | 86.8 | — | −12.2 (−13.3 to −11.1) |
| Model that sees race | 68.3 | 88.0 | 88.0 | −19.7 (−21.4 to −17.7) |
| Blind model (race removed) | 75.3 | 86.5 | 87.8 | −11.2 (−12.8 to −9.8) |
| Blind + group cutoffs | 84.6 | 84.5 | 87.4 | +0.1 (−1.4 to +1.5) |

Blinding is not a fix. A model trained on the remaining columns still rebuilds
Black status at **AUC 0.84**, driven by census-tract minority share, loan size,
debt-to-income and underwriting system: dropping the column removes the evidence,
not the pattern. Per-group cutoffs do close the gap (+0.1) for about 0.6 points
of accuracy — but they read race at decision time, which is the disparate
treatment that survived the 2026 amendment to Regulation B. **Parity and legality
are different tests.** Matching error rates instead of approval rates only gets
the gap to −3.1. Age shows no meaningful gap here (+0.5, −0.9 to +2.5).

**Where it fails.** In one split we found the closest Black-denied and
White-approved pair in the test set: identical loan amount ($305,000), identical
loan-to-value (96), income ($69k), debt-to-income bucket (50%–60%), term and age
group. The model scores them **0.716 and 0.874** and sends them opposite ways.
The failure is not a stray example: it is the gap, expressed as one decision. It
also fails in the other direction — some applicants the lender denied score
comfortably above the cutoff, and the tool would have approved them.

## What we'd do next

1. **Fix it without reading race.** Drop the census-tract columns that do the
   proxy work and measure both sides of the trade: how far proxy AUC falls, and
   what accuracy it costs. If the tract columns carry real risk information we
   will see the cost in the accuracy column instead of guessing at it.
2. **Test whether the gap is stable.** Run 2023 and 2024 the same way. Our own
   earlier run on Oregon purchase-only loans showed a −3.2 age gap that vanished
   once the sample widened, which is a warning about one-slice conclusions.

## Sources

- HMDA loan-level data, CFPB Data Browser API: <https://ffiec.cfpb.gov/data-publication/>
- Regulation B final rule, 91 FR 21620 (FR doc. 2026-07804), effective 21 July 2026: <https://www.federalregister.gov/d/2026-07804>
- *National Fair Housing Alliance v. CFPB*, No. 1:26-cv-01820 (D.D.C.), the pending challenge to that rule
- scikit-learn, pandas, NumPy — modelling and data handling
- AI tools: an AI coding assistant (Codebuff) helped write and debug the pipeline, the page and the notebook. Every number is produced by the code in this repository; the recorded approval rates were checked against the raw HMDA file.

**Reproduce.** `pip install -r requirements.txt`, then `python app.py` (audit +
comparison, about two minutes) and `python app.py serve`. Or open
`fairness_audit.ipynb` and Run all: it recomputes the audit on 20 splits and
prints the same table in about a minute.

*This project is a prototype and is not financial advice.*
