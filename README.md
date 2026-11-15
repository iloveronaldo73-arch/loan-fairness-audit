# Fair-lending audit

A one-page audit of a mortgage-approval model on HMDA 2025 public loan-level
data. The page compares the decisions actually recorded with a demographics-aware
model, a blinded model, and that blind model with per-group cutoffs, then probes
how well the remaining columns rebuild the protected trait. `WRITEUP.md` has the
findings.

## Layout

```
fairness_audit.ipynb  the whole audit as one executed notebook (hackathon submission)
app.py            build the page and result tables, or serve them locally
audit/            dataset query, audit pipeline, site builder, cutoff comparison
data/             cached HMDA extract (downloaded on the first run)
results/          generated artifacts: audit.json, audit.csv, cutoffs.csv,
                  summary.md and the built page, index.html
tests/            unittest suite, no network needed
requirements.txt  numpy, pandas, scikit-learn
WRITEUP.md        one-page write-up
```

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py         # ~80s: audit + comparison, writes results/
.venv/bin/python app.py serve   # http://127.0.0.1:4173
```

The first build downloads a 224 MB HMDA extract (about 45 MB compressed) and
caches it in `data/`; later runs are offline.

## Commands

| command | what it does |
|---|---|
| `python app.py` | audit and comparison, writes all five files in `results/` |
| `python app.py build` | audit only: `audit.json`, `audit.csv`, `index.html` |
| `python app.py compare` | cutoff rules only: `cutoffs.csv` |
| `python app.py serve` | serves `results/` at `http://127.0.0.1:4173` |

`--splits` and `--seed` control the resampling (default 50 and 0).

## Notebook

`fairness_audit.ipynb` is the submission version: load, baseline, model, the
failure pair, the blind model, the proxy probe, group cutoffs, a per-application
explanation, and the results table. It uses the cached extract in `data/` when
present and otherwise downloads it. Run all takes about a minute.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

## Data

HMDA 2025, Georgia, Maryland, Michigan and Arizona, home-purchase and refinance
applications marked originated or denied, from the CFPB Data Browser API
(`audit/dataset.py` holds the exact query). Filings are narrowed to White and
Black applicants who reported an age, then to a fixed 100,000-application sample
(seed 0) so 50 splits stay tractable. Only application-time fields are modelled;
pricing, purchaser, HOEPA and denial reasons are excluded because they are
reported after the decision.

## Deploying

`results/index.html` is self-contained — one HTML file with its CSS, SVG and
JavaScript inline. Upload it as the site root.
