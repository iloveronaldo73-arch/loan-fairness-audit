# Loan model fairness audit

A credit model can give two groups of applicants different outcomes without anyone telling it to. This repo trains a loan approval model on the German Credit dataset and measures how approval rates and error rates differ by age and by sex. It then tests two common fixes: dropping the protected columns, and setting a separate approval cutoff for each group.

Built for the Veritas AI International High School Hackathon, November 2026. The one-page writeup is in [WRITEUP.md](WRITEUP.md).

## Run it

```
pip install -r requirements.txt
python -m audit.run_audit      # trains the models, writes tables and a figure to results/
python app.py                  # opens the demo in your browser
```

The first run downloads the dataset from OpenML to `data/credit_g.csv` and needs an internet connection that one time. Commit that file and the `results/` folder so the demo runs offline.

Logistic regression takes a few seconds for the full audit. Add `--model gbm` for gradient boosting, which takes under a minute.

In Google Colab:

```
!git clone https://github.com/YOUR-USERNAME/loan-fairness-audit.git
%cd loan-fairness-audit
!pip install -q -r requirements.txt
!python -m audit.run_audit
```

## What is in here

```
audit/
  data.py         loads the data, builds the age and sex groups
  models.py       preprocessing and the two model types
  metrics.py      group rates, gaps, and the per-group cutoff search
  run_audit.py    runs 50 random splits and writes results/
  scoring.py      helpers for the what-if demo
app.py            Gradio demo
tests/            checks for the metrics, with numbers worked out by hand
results/          summary.md, summary.csv, runs.csv, proxy_check.csv, gaps.png
data/             cached copy of the dataset
```

## What it measures

For each group, on test data the model did not see:

- Approval rate, and the gap between the two groups
- True positive rate: the share of good borrowers who get approved
- False positive rate: the share of bad borrowers who get approved
- Proxy check: how well the columns left in the blind model predict group membership

Four strategies are compared: approving everyone (the baseline), a model with age and sex, a blind model without them, and the blind model with a separate cutoff per group.

## Tests

```
python tests/test_audit.py
```

The pipeline test uses synthetic data with the same columns as the real set. It checks that the code runs and that the outputs have the right shape. No reported number comes from it.

## Data

Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

## Status

A hackathon prototype on a 1,000-row dataset from 1994. It is not financial advice and no lender should use it.
