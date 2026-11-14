# Audit results

Model: logreg. Splits: 50 random 60/20/20 train/validation/test splits. Base cutoff: 0.5.

All numbers come from the test part of each split. Gaps are the first group minus the reference group, in percentage points, with the 2.5th to 97.5th percentile across splits in brackets. A negative gap means the first group does worse. TPR is the share of good borrowers approved, FPR the share of bad borrowers approved.

## age: under 25 vs 25 and over

Applicants in the data: under 25 149, 25 and over 851.

| Strategy | Accuracy | Approval, under 25 | Approval, 25 and over | Approval gap | TPR gap | FPR gap |
|---|---|---|---|---|---|---|
| approve everyone | 70.0% | 100.0% | 100.0% | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) |
| full model | 74.7% | 63.9% | 78.0% | -14.1 (-27.0 to -1.1) | -13.8 (-31.3 to -2.1) | -4.2 (-28.7 to +24.1) |
| blind model | 74.6% | 67.6% | 77.8% | -10.2 (-22.5 to +3.0) | -10.5 (-29.8 to +4.9) | +0.3 (-26.6 to +26.1) |
| blind + group cutoffs | 74.0% | 75.5% | 75.7% | -0.2 (-28.5 to +22.6) | -2.7 (-29.4 to +18.7) | +14.2 (-19.4 to +48.1) |

## sex: female vs male

Applicants in the data: female 310, male 690.

| Strategy | Accuracy | Approval, female | Approval, male | Approval gap | TPR gap | FPR gap |
|---|---|---|---|---|---|---|
| approve everyone | 70.0% | 100.0% | 100.0% | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) |
| full model | 74.7% | 68.5% | 79.0% | -10.5 (-22.6 to +3.2) | -5.9 (-16.2 to +3.7) | -12.8 (-35.9 to +19.6) |
| blind model | 74.6% | 73.7% | 77.4% | -3.6 (-13.2 to +9.9) | -1.5 (-10.6 to +10.4) | -1.0 (-22.9 to +30.2) |
| blind + group cutoffs | 74.3% | 75.2% | 76.4% | -1.3 (-15.3 to +14.9) | +0.1 (-13.2 to +15.5) | +3.1 (-20.7 to +36.0) |

## Proxy check

AUC of a logistic regression that predicts group membership from the columns the blind model uses (5-fold cross-validation). 0.5 is chance.

| Attribute | Group | AUC |
|---|---|---|
| age | under 25 | 0.79 (std 0.03) |
| sex | female | 0.71 (std 0.01) |
