# Audit results

Dataset: German Credit (1994), demo data only. 1,000 applications. Label: good credit risk.
Model: logreg. Splits: 50 random 60/20/20 train/validation/test splits. Base cutoff: 0.5.

All numbers come from the test part of each split. Gaps are the first group minus the reference group, in percentage points, with the 2.5th to 97.5th percentile across splits in brackets. A negative gap means the first group does worse. TPR is the share of good borrowers approved; FPR is the share of bad borrowers approved.

## age: under 25 vs 25 and over

Applicants in the data: under 25 149, 25 and over 851.

| Strategy | AUC | Accuracy | Approval, under 25 | Approval, 25 and over | Approval gap | TPR gap | FPR gap | Cost |
|---|---|---|---|---|---|---|---|---|
| approve everyone | n/a | 70.0% | 100.0% | 100.0% | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) | 1.50 |
| full model | 0.780 | 74.7% | 63.9% | 78.0% | -14.1 (-27.0 to -1.1) | -13.8 (-31.3 to -2.1) | -4.2 (-28.7 to +24.1) | 0.88 |
| blind model | 0.780 | 74.6% | 67.6% | 77.8% | -10.2 (-22.5 to +3.0) | -10.5 (-29.8 to +4.9) | +0.3 (-26.6 to +26.1) | 0.89 |
| blind + group cutoffs | 0.780 | 74.2% | 76.1% | 75.8% | +0.3 (-26.7 to +23.5) | -2.3 (-24.8 to +18.7) | +14.7 (-19.0 to +51.4) | 0.89 |
| blind + cost cutoff | n/a | 62.5% | 26.9% | 43.2% | -16.4 (-32.2 to -0.3) | -13.9 (-34.3 to +13.5) | -8.8 (-25.0 to +6.2) | 0.54 |

## sex: female vs male

Applicants in the data: female 310, male 690.

| Strategy | AUC | Accuracy | Approval, female | Approval, male | Approval gap | TPR gap | FPR gap | Cost |
|---|---|---|---|---|---|---|---|---|
| approve everyone | n/a | 70.0% | 100.0% | 100.0% | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) | 1.50 |
| full model | 0.780 | 74.7% | 68.5% | 79.0% | -10.5 (-22.6 to +3.2) | -5.9 (-16.2 to +3.7) | -12.8 (-35.9 to +19.6) | 0.88 |
| blind model | 0.780 | 74.6% | 73.7% | 77.4% | -3.6 (-13.2 to +9.9) | -1.5 (-10.6 to +10.4) | -1.0 (-22.9 to +30.2) | 0.89 |
| blind + group cutoffs | 0.780 | 74.3% | 75.3% | 76.5% | -1.1 (-15.3 to +13.5) | +0.2 (-13.2 to +15.5) | +3.2 (-18.9 to +35.2) | 0.89 |
| blind + cost cutoff | n/a | 62.5% | 39.4% | 41.3% | -1.9 (-11.4 to +6.8) | +0.5 (-10.8 to +13.0) | +1.6 (-12.2 to +24.1) | 0.54 |

## Proxy check

AUC of a logistic regression that predicts group membership from the columns the blind model uses (5-fold cross-validation). 0.5 is chance.

| Attribute | Group | AUC |
|---|---|---|
| age | under 25 | 0.79 (std 0.03) |
| sex | female | 0.71 (std 0.01) |
