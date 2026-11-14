<!-- One placeholder is left: the AI tools line under Sources. Edit it, then delete this comment. -->

# Loan model fairness audit

**Track:** Finance
**Team:** names are on the submission form

## The problem

A lender's model decides who gets a loan, and it can give two groups different outcomes without anyone telling it to. Housing, job tenure and savings can stand in for age, so dropping the age column may leave the effect in place. We measure how a credit model treats applicants under 25 and women compared with applicants 25 and over and men, and we test two fixes.

## What we built

Our project helps a loan officer or a regulator by taking a trained credit model and a dataset, and giving back approval rates and error rates for each group. The code is a Python audit pipeline plus a Gradio web app. In the app you enter an applicant and it rescores the same person with only their age or sex changed.

## How it works

**Data.** UCI Statlog German Credit: 1,000 applications, 20 features, 700 good and 300 bad borrowers. We derive the age group from the age column and sex from personal_status. The data holds 149 applicants under 25 and 310 women.

**Models.** Logistic regression on scaled numeric and one-hot categorical columns, fit three ways: with age and sex (full), with both removed (blind), and blind with a separate approval cutoff per group. We choose each group's cutoff on a validation set so its true positive rate matches the overall rate.

**Baseline.** Approving every applicant scores 70.0% accuracy.

**Evaluation.** 50 random splits, each 60% train, 20% validation, 20% test. Every result comes from test data the model did not see during fitting or cutoff selection. Ranges are the 2.5th to 97.5th percentile across splits.

## Results

Gaps are the first group minus the reference group, in percentage points. TPR is the share of good borrowers approved.

| Strategy | Accuracy | Age: approval gap | Age: TPR gap | Sex: approval gap | Sex: TPR gap |
|---|---|---|---|---|---|
| Approve everyone | 70.0% | 0.0 | 0.0 | 0.0 | 0.0 |
| Full model | 74.7% | -14.1 (-27.0 to -1.1) | -13.8 (-31.3 to -2.1) | -10.5 (-22.6 to +3.2) | -5.9 (-16.2 to +3.7) |
| Blind model | 74.6% | -10.2 (-22.5 to +3.0) | -10.5 (-29.8 to +4.9) | -3.6 (-13.2 to +9.9) | -1.5 (-10.6 to +10.4) |
| Blind + group cutoffs | 74.0% age, 74.3% sex | -0.2 (-28.5 to +22.6) | -2.7 (-29.4 to +18.7) | -1.3 (-15.3 to +14.9) | +0.1 (-13.2 to +15.5) |

With age in the model, applicants under 25 were approved 63.9% of the time and applicants 25 and over 78.0%. Removing the age column moved the average gap from -14.1 to -10.2 points. The blind model's gap was closer to zero in 33 of the 50 splits, but the move is smaller than the spread across splits. A logistic regression trained on the remaining columns predicts under-25 status with AUC 0.79, where 0.5 is chance.

Group cutoffs brought the average age gap to -0.2 points and cost 0.6 points of accuracy against the blind model. The under-25 cutoff averaged 0.43 against 0.53 for everyone else. The false positive rate gap rose to +14.2 points (-19.4 to +48.1), so more bad borrowers under 25 got approved.

For sex, the full model approved women 10.5 points less often than men. The blind model cut that to 3.6 points, with a range that includes zero. The remaining columns still predict sex with AUC 0.71, although personal_status is gone.

**Weak spots.** The ranges show how results vary with the split. They understate the uncertainty about new applicants because the splits overlap. Each test set holds about 30 applicants under 25 and 61 women, so the group-cutoff results range from large gaps in one direction to large gaps in the other. The age cutoff of 25 is a choice, and 35 is also common in the literature. Group cutoffs use age or sex at decision time, which some jurisdictions restrict. The dataset is old and German, so the numbers say little about any bank today.

## What we'd do next

1. Repeat the audit on a larger lending dataset such as HMDA mortgage records, where group sizes support tighter estimates.
2. Choose cutoffs with the dataset's own cost rule, which prices a bad loan at five times a missed good one, and measure the gaps again.

## Sources

- Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data. Loaded through OpenML as `credit-g`.
- scikit-learn, pandas, NumPy, Matplotlib, Gradio.
- AI tools: Claude (Anthropic) drafted the first version of the code and this text. [[Say what the team changed and checked.]]
- Hackathon prototype. Not financial advice.
