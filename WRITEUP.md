<!-- Two placeholders are left: the country comparison and the AI tools line. Fill them, then delete this comment. -->

# Loan fairness audit

**Track:** Finance
**Team:** names are on the submission form

# The problem

A credit officer at a small lender uses a statistical model to approve or decline loans, and applicants get no explanation. A model can approve one age group less often even when age is removed, because other columns stand in for it. We test whether a loan model gives applicants under 25 and women different outcomes, and what it costs to narrow the gap. The rules matter too. A CFPB final rule effective July 21, 2026 states that ECOA does not authorize disparate-impact liability, while law-firm summaries say disparate treatment and intentional use of proxies stay prohibited. [[Add one or two sentences on how lending and the rules differ in your own country.]]

# What we built

Our project helps a credit officer by taking an applicant's details and giving back an approval estimate from two models, plus a report on whether approval rates differ by age and sex. A web page lets you enter an applicant, change only their age or sex and watch the estimate move. An eligibility calculator estimates the largest loan an applicant's income can carry. An assistant in the page answers questions about the results.

# What we added

We started from the suggested fairness-audit project and added five things:

1. A proxy check. With age and personal status removed, the other columns still predict who is under 25 (AUC 0.79) and who is female (AUC 0.71).
2. A fix with a price. A separate approval cutoff for each group moves the age gap from -14.1 to +0.3 points on average and costs 0.4 points of accuracy.
3. The dataset's 5:1 cost as our measure and as a cutoff. It lowers cost per applicant to 0.54 but approves far fewer people.
4. The features behind each gap, and the columns that stand in for age.
5. Today's rules for US lenders.

# How it works

**Data.** Statlog German Credit (UCI id 144): 1,000 applicants, 20 attributes, 700 good and 300 bad risks. We take age group from the age column and sex from personal_status. The data holds 149 applicants under 25 and 310 women.

**Model.** Logistic regression on scaled numeric and one-hot categorical columns. We fit it four ways: with age and sex (full), without them (blind), blind with a cutoff per group that equalizes the true positive rate, and blind with the cutoff set by the 5:1 cost (approve if the chance of repayment is at least 5/6).

**Evaluation.** 50 random splits, each 60% train, 20% validation, 20% test. Every result comes from test data the model did not see. Ranges are the 2.5th to 97.5th percentile across splits.

# Results

Approving everyone is the baseline: 70.0% accurate, cost 1.50 per applicant. The blind model is 74.6% accurate with cost 0.89 and AUC 0.78. Gaps are the first group minus the reference group, in percentage points.

| Strategy | Accuracy | Cost | Under 25 approved | 25 and over approved | Age gap |
|---|---|---|---|---|---|
| Approve everyone | 70.0% | 1.50 | 100.0% | 100.0% | 0.0 |
| Full model | 74.7% | 0.88 | 63.9% | 78.0% | -14.1 (-27.0 to -1.1) |
| Blind model | 74.6% | 0.89 | 67.6% | 77.8% | -10.2 (-22.5 to +3.0) |
| Blind + group cutoffs | 74.2% | 0.89 | 76.1% | 75.8% | +0.3 (-26.7 to +23.5) |
| Blind + cost cutoff | 62.5% | 0.54 | 26.9% | 43.2% | -16.4 (-32.2 to -0.3) |

For sex, the full model approved women 10.5 points less often than men. The blind model cut that to 3.6 points (-13.2 to +9.9), and group cutoffs to 1.1 points.

In the blind model, the largest contributors to the age gap are the checking account (26% of under-25 applicants have none, against 42% of older ones), housing (46% rent, against 13%) and credit history.

The cheapest policy for the lender, the cost cutoff, is the least fair for young applicants. It also scores below the baseline on accuracy, though not on cost.

**Where it fails.** The blind model approved a 28-year-old at 98% who was a bad risk: a 12-month loan of 939 DM with a checking balance of 200 DM or more. The data holds no repayment history after approval, and two applicants with the same recorded details can have different outcomes. The ranges show variation across splits and understate uncertainty about new applicants, because the splits overlap and each test set holds about 30 under-25 applicants. Group cutoffs use a protected trait at decision time, which some jurisdictions restrict. The data is from 1994.

# What we'd do next

1. Run the audit on a recent lending dataset. Our code already supports HMDA 2025 mortgage records, but we have not reported results from it.
2. Test the group-cutoff fix on more under-25 applicants, since 149 is too few to tell a real gain from noise.

# Sources

- Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- FFIEC. HMDA Data Browser, summary of originated loans by purpose. https://ffiec.cfpb.gov/data-browser/
- CFPB. Final rule amending Regulation B, Federal Register 2026-07804, and law-firm summaries of it.
- scikit-learn, pandas, NumPy, Matplotlib.
- AI tools: Claude (Anthropic) drafted the code and this text. [[Say what the team changed and checked.]]

This project is a prototype and is not financial advice.
