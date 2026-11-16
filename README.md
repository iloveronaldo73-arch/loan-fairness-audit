# Loan fairness audit

Veritas AI International High School Hackathon 2026, Finance track, project 4.2.3: fairness audit of a loan model.

We train a loan approval model on the Statlog German Credit data, compare it with the "approve everyone" baseline, and test whether applicants of different ages and sexes get different outcomes. The write-up is [WRITEUP.md](WRITEUP.md). The app is [docs/index.html](docs/index.html).

## Run it

```
pip install -r requirements.txt
python -m audit.run_audit      # trains the models, writes tables, figure and extra.json to results/
python -m audit.build_site     # puts the numbers into the app's data block
```

The first run downloads the dataset from OpenML to `data/credit_g.csv` (internet needed once). Commit that file and `results/`. Then open `docs/index.html` in a browser, or turn on GitHub Pages (Settings, then Pages, deploy from the `/docs` folder).

In Google Colab:

```
!git clone https://github.com/YOUR-USERNAME/loan-fairness-audit.git
%cd loan-fairness-audit
!pip install -q -r requirements.txt
!python -m audit.run_audit
!python -m audit.build_site
```

Options: `--model gbm` for gradient boosting, `--splits 10` for a faster run.

## The app

One page, no server, no account. Open it from GitHub Pages and everything runs in the visitor's browser. Its sections:

- Data: a summary of HMDA originated loans by purpose (6.83 million loans, $2.41 trillion), from the FFIEC Data Browser export the team supplied. The numbers are typed into the page. The fairness audit itself still runs on German Credit, because the HMDA records have not been downloaded.
- Check: an eligibility calculator. It takes monthly income, monthly savings, loan tenure and a requested amount, and estimates the largest loan the income can carry from a payment share of income and an interest rate, both editable assumptions. Credit history is a simulated lookup, and an adverse history sends the file to a credit officer. Name, address and telephone are demo fields that no model uses and that never leave the browser. Years of credit history, open credit lines and credit utilization are not inputs.
- Test: scores an applicant with the same logistic regression weights scikit-learn learned, and shows what changes when only age or sex changes.
- Results, Why and Failures: the baseline comparison, the approval gaps with their ranges, the features behind each gap, and the model's confident mistakes.
- Added: what we added to the brief, and the 2026 rule change.

The assistant in the page answers questions about the results. It works at three levels:

- Built-in (default). It answers from the audit numbers with rules in the page. It needs no internet or key, so it works for anyone who opens the link. It is not a language model.
- Your own key. Click "Add a Gemini key" in the assistant panel and paste a free key from Google AI Studio. The page then asks Gemini directly. The key stays in the open tab. Do not commit a key to the repo. If the model name has changed, edit `GM` near the end of `docs/index.html`.
- Live for every visitor. Deploy `worker/agent.js` as a Cloudflare Worker, add your Gemini key as the secret `GEMINI_API_KEY` and set `ALLOW_ORIGIN` to `https://YOUR-USERNAME.github.io`. Then paste the Worker's address into `AGENT_URL` at the top of the script in `docs/index.html`. The key stays on the Worker, so visitors get generative answers without one. We have not tested this path against a live deployment.

## What is in here

```
audit/
  data.py         loads German Credit (and optional HMDA), builds the age and sex groups
  models.py       preprocessing and the two model types
  metrics.py      group rates, gaps, and the per-group cutoff search
  run_audit.py    runs random splits and writes results/
  explain.py      feature drivers, stand-in columns, example mistakes, browser model weights
  build_site.py   writes the results into docs/index.html
  fetch_hmda.py   optional: downloads HMDA data for one state
docs/index.html   the app
worker/agent.js   optional Worker for live generative answers
tests/            metrics, loaders, pipeline, and a check that the browser scorer matches scikit-learn
results/          summary.md, summary.csv, runs.csv, proxy_check.csv, extra.json, meta.json, gaps.png
```

## What it measures

For each group, on test data the model did not see: accuracy, approval rate, the gap between groups, true and false positive rates, and cost per applicant. Cost uses the dataset documentation's weights: approving a bad risk costs 5, declining a good one costs 1. Strategies compared: approve everyone, a model with age and sex, a blind model without them, the blind model with a separate cutoff per group, and the blind model with a cutoff set by the 5:1 cost.

## Optional: HMDA 2025

The code also runs on US mortgage records (`python -m audit.fetch_hmda --state WA`, then `python -m audit.run_audit --dataset hmda`). We have not reported results from it. Its label is a past lender decision, not repayment, so it answers a different question.

## Tests

```
python tests/test_audit.py
```

The pipeline tests use synthetic data with the same columns as the real files. No reported number comes from them.

## Data

Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

## Status

This project is a prototype and is not financial advice.
