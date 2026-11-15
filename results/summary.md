# HMDA 2025, four states, purchase and refinance

100,000 applications · 84.5% approved (originated) · 50 random 60/20/20 splits · model logreg.

## age: under 25 vs 25 and over

| strategy | under 25 | 25 and over | accuracy | gap | 95% range |
|---|---|---|---|---|---|
| approve everyone | 100.0 | 100.0 | 84.5 | +0.0 | +0.0 to +0.0 |
| observed in data | 85.1 | 84.6 | — | +0.5 | -0.9 to +2.5 |
| full model | 85.3 | 84.5 | 88.0 | +0.8 | -0.9 to +2.1 |
| blind model | 84.4 | 84.5 | 87.8 | -0.1 | -2.0 to +1.2 |
| blind + group cutoffs | 84.6 | 84.5 | 87.8 | +0.1 | -3.5 to +2.4 |

## race: Black vs White

| strategy | Black | White | accuracy | gap | 95% range |
|---|---|---|---|---|---|
| approve everyone | 100.0 | 100.0 | 84.5 | +0.0 | +0.0 to +0.0 |
| observed in data | 74.6 | 86.8 | — | -12.2 | -13.3 to -11.1 |
| full model | 68.3 | 88.0 | 88.0 | -19.7 | -21.4 to -17.7 |
| blind model | 75.3 | 86.5 | 87.8 | -11.2 | -12.8 to -9.8 |
| blind + group cutoffs | 84.6 | 84.5 | 87.4 | +0.1 | -1.4 to +1.5 |

## Cutoff rules on the blind model

| trait | rule | approval gap | 95% range | TPR gap | FPR gap | accuracy |
|---|---|---|---|---|---|---|
| age | none | -0.1 | -2.0 to +1.2 | +0.6 | -6.7 | 87.8 |
| age | equal approval | +0.1 | -3.5 to +2.4 | +0.8 | -6.3 | 87.8 |
| age | equal TPR | -1.0 | -3.6 to +1.0 | -0.2 | -7.6 | 87.8 |
| age | equal FPR | +3.8 | +0.3 to +7.0 | +4.0 | +0.1 | 87.7 |
| age | equal error | +1.1 | -15.4 to +13.7 | +0.5 | +2.9 | 87.9 |
| race | none | -11.2 | -12.8 to -9.8 | -5.3 | -3.0 | 87.8 |
| race | equal approval | +0.1 | -1.4 to +1.5 | +5.4 | +10.4 | 87.4 |
| race | equal TPR | -5.8 | -7.5 to -4.3 | +0.0 | +2.7 | 87.7 |
| race | equal FPR | -8.1 | -11.0 to -5.1 | -2.2 | +0.2 | 87.6 |
| race | equal error | -3.1 | -15.2 to +1.5 | +3.5 | -2.0 | 87.4 |
