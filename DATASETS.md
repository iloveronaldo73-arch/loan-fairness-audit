# Dataset justification for the loan audit

This project is a prototype and is not financial advice.

## What a dataset must have to support a loan fairness audit

1. **A real repayment outcome**, so "accuracy" means something beyond predicting a label we invented.
2. **Economic features a lender would actually use** (savings, income stability, debt burden, repayment history), so we can ask which economic factors drive the model and the gap.
3. **Age and sex available without collecting anything new.** The rules forbid collecting personal information, so protected attributes must already be in a public dataset.
4. **Public, documented and licensed**, so judges can check our sources.
5. **Enough people in each group.** This is the weak point of small credit datasets, which is why we use two datasets of very different size.

## Summary

| Dataset | Rows | Outcome | Protected attributes | Role in this project | Main limitation |
|---|---|---|---|---|---|
| UCI Statlog German Credit | 1,000 | good / bad credit risk (70% / 30%) | age; sex inferred from personal-status code | **Primary**: all headline results | Tiny groups (about 30 under-25 applicants per test set) |
| UCI Default of Credit Card Clients (Taiwan) | 30,000 | default next month (about 22%, verify on the dataset page) | SEX, AGE | **Replication** on a second source | Credit cards in 2005; outcome is for existing cardholders |
| Synthetic data | 1,000 | invented | invented | **Code tests only** | Not real; never used for a reported result |

## 1. Statlog German Credit (primary)

**Why we chose it.** It is the dataset the brief suggests for this project, and it fits the criteria above:
- It records 1,000 loan applications with 20 attributes and a good/bad label (700 good, 300 bad).
- It includes `age` and a combined personal-status field from which sex can be inferred, so we can audit age and sex without collecting any personal data.
- Its documentation states a **cost matrix**: approving a bad applicant is five times as costly as rejecting a good one. That gives us a principled way to evaluate decisions beyond plain accuracy.
- It is small enough to retrain hundreds of times in under a minute, which is what makes permutation tests and bootstraps practical.
- It has been studied widely, so our numbers can be compared with published work.

**Why its economic factors are meaningful.** We group the columns by the economic idea a lender is testing:

| Factor group | Columns | What a lender is checking |
|---|---|---|
| Liquidity and savings | checking account status, savings | Can the borrower absorb a shock without missing payments? |
| Loan size and term | credit amount, duration, installment rate | How heavy is the repayment burden? |
| Employment stability | employment duration, job type | Is income likely to continue? |
| Assets and housing | property, housing | Is there collateral or accumulated wealth? |
| Credit history | credit history, other payment plans, other debtors, existing credits | Has the borrower repaid before? |
| Loan purpose | purpose | Is the loan for something that builds or consumes value? |
| Household context | residence length, dependents, telephone, foreign worker | Stability and obligations outside the loan |
| Engineered ratios | monthly payment (amount / duration), log of amount | Does a simple affordability ratio add information beyond the raw columns? |

This grouping lets us test whether the age gap comes from legitimate economic differences (for example, younger people have less savings) or from age itself.

**Limitations we state openly.**
- **Small groups.** There are only 149 applicants under 25 and 310 women, so each 20% test set has about 30 and 61. Our intervals are wide for this reason, and we report them rather than hide them.
- **Old and German.** The dataset was donated in 1994 and the loans are older still. Lending rules and borrowers have changed, so the numbers say little about any bank today.
- **Sex is inferred.** There is no sex column, only a personal-status code that mixes sex and marital status, and one of its categories (female, single) does not appear in the data.
- **Selection bias.** In credit data, repayment outcomes are normally only observed for people who were approved, so the model learns from a filtered group. This applies to any credit dataset.

## 2. UCI Default of Credit Card Clients, Taiwan (replication)

**Why we chose it.** A fairness finding on 1,000 rows could be noise or specific to one dataset. This dataset lets us ask whether the pattern holds elsewhere:
- It has **30,000 clients**, so the under-25 and female groups contain thousands of people and the intervals become much tighter.
- It includes SEX, AGE, EDUCATION and MARRIAGE.
- It records six months of **repayment status, bill amounts and payment amounts**, which are real behavioural economic signals. From them we build economic ratios: credit utilization (bills / limit), payment ratio (paid / billed) and worst delinquency.
- The label is an observed outcome (default in the following month).
- It is public. The UCI page lists its licence and the source paper (Yeh & Lien, 2009); check both and cite them.

**Why its economic factors are meaningful.**

| Factor group | Columns | What a lender is checking |
|---|---|---|
| Credit limit | LIMIT_BAL | How much trust the lender has already extended |
| Repayment history | PAY_0 to PAY_6 | Delinquency, the strongest signal of future default |
| Bill balances | BILL_AMT1 to 6 | How much debt is outstanding |
| Payments made | PAY_AMT1 to 6 | Whether the borrower is actually paying it down |
| Education and marital status | EDUCATION, MARRIAGE | Weak socioeconomic proxies; also sensitive, so we test whether they leak protected information |
| Engineered ratios | utilization, payment ratio, max delay | Whether simple ratios beat the raw columns |

**Limitations.**
- **Different decision.** These are existing credit-card holders in Taiwan in 2005, and the outcome is default after approval. The audit therefore applies to monitoring or credit-limit decisions, not to application approval, and we do not pool it with the German data.
- **No cost rule.** The dataset documents no cost ratio, so we assume 1:1 and say so.
- **Unusual period.** It covers a time of heavy card lending in Taiwan, so default rates may not be typical.
- **Messy columns.** EDUCATION and MARRIAGE contain undocumented codes, and SEX is binary.
- **Check before use.** Confirm the column names, licence and default rate on the UCI page; we could not download it in our test environment.

## 3. Synthetic data (tests only)

`--dataset synthetic` generates fake applicants with the same column format. It exists so the code and tests run offline in seconds. Every report made from it carries a warning banner, and **no number from synthetic data appears in our results**.

## Datasets considered but not used
- **HMDA mortgage records.** Very large and well suited to fairness work, but the processing (and its race and ethnicity fields) would not fit in a weekend. It is a good next step.
- **Credit datasets from Kaggle competitions.** Their terms vary, and some are anonymised in ways that hide the economic meaning of columns, which would weaken our economic-factor analysis.

## Ethics and limits
- We use only public datasets and collect no personal information.
- Group-specific cutoffs use age or sex at decision time, which many jurisdictions restrict. We show them as an analysis, not as a deployable fix.
- A gap in approval rates shows that decisions differ by group, not that the model is unfair. Part 2 of the uncertainty study separates economic explanations from group effects, and even then the result is evidence, not a verdict.

## Sources
- Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data (also on OpenML as `credit-g`).
- Yeh, I-C. & Lien, C-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2). UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Nadeau, C. & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52, 239-281 (the corrected interval used in the uncertainty study).
