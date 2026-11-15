"""HMDA 2025 public loan-level data: four states, purchase and refinance.

Source: the CFPB HMDA Data Browser API (public LAR, modified to protect privacy).
Query: activity year 2025, Georgia, Maryland, Michigan and Arizona, loan purpose
1 (home purchase) or 31 (refinancing), action taken 1 (originated) or 3 (denied).

Only fields that exist when the application is filed are kept. Pricing
(interest_rate, rate_spread, points, fees), purchaser type, HOEPA status and the
denial reasons are reported after the decision and would leak the label.

The analysis set is restricted to White and Black or African American applicants
who reported an age, then cut to a fixed random sample: the pool is large enough
that 50 splits would otherwise dominate the build time.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

YEAR = 2025
STATES = "GA,MD,MI,AZ"
LOAN_PURPOSES = "1,31"  # home purchase, refinancing
API_QUERY = (
    f"years={YEAR}&states={STATES}&actions_taken=1,3&loan_purposes={LOAN_PURPOSES}"
)
API_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?" + API_QUERY

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / f"hmda_{YEAR}_{STATES.replace(',', '_').lower()}.csv"

TITLE = f"HMDA {YEAR}, four states, purchase and refinance"
SOURCE = "https://ffiec.cfpb.gov/data-publication/"
LABEL = "approved (originated)"
CAVEAT = (
    "Four states, two loan purposes, self-reported demographics and no credit "
    "score: a public sample, not a lender's model."
)

SAMPLE = 100_000
SAMPLE_SEED = 0

NUMERIC = [
    "loan_amount", "loan_to_value_ratio", "loan_term", "property_value", "income",
    "tract_population", "tract_minority_population_percent",
    "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
]

CATEGORICAL = [
    "loan_type", "lien_status", "total_units", "construction_method", "occupancy_type",
    "conforming_loan_limit", "derived_loan_product_type", "derived_dwelling_category",
    "reverse_mortgage", "open-end_line_of_credit", "business_or_commercial_purpose",
    "debt_to_income_ratio", "aus-1", "submission_of_application",
    "initially_payable_to_institution", "preapproval", "derived_sex",
    "derived_ethnicity", "derived_race", "applicant_age",
]

# Dropped by the blind model: every demographic field, including the two traits.
PROTECTED = ["derived_race", "derived_ethnicity", "derived_sex", "applicant_age"]

GROUPS = {
    "age": {"other": "under 25", "ref": "25 and over", "column": "age_group"},
    "race": {"other": "Black", "ref": "White", "column": "race"},
}

AGE_BUCKETS = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
RACE_LABELS = {"Black or African American": "Black", "White": "White"}


# The API's CDN rejects several custom user agents (403), but accepts urllib's
# default one, so no User-Agent header is set here.
def _download() -> None:
    with urllib.request.urlopen(API_URL, timeout=600) as response:
        payload = response.read()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(payload)


def load(refresh: bool = False) -> pd.DataFrame:
    """Return the filtered analysis table: features, label and both group columns."""
    if refresh or not CACHE.exists():
        _download()

    wanted = set(NUMERIC + CATEGORICAL + ["action_taken"])
    raw = pd.read_csv(
        CACHE, usecols=lambda c: c in wanted, dtype=str, low_memory=False
    )

    frame = raw[raw["action_taken"].isin(["1", "3"])].copy()
    frame["target"] = (frame["action_taken"] == "1").astype(int)
    frame["race"] = frame["derived_race"].map(RACE_LABELS)
    frame["age_group"] = np.where(frame["applicant_age"] == "<25", "under 25", "25 and over")

    frame = frame[frame["race"].notna() & frame["applicant_age"].isin(AGE_BUCKETS)]
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if len(frame) > SAMPLE:
        frame = frame.sample(n=SAMPLE, random_state=SAMPLE_SEED)
    return frame.reset_index(drop=True)
