"""Datasets and group definitions.

hmda   : US mortgage applications (Home Mortgage Disclosure Act), CFPB/FFIEC.
german : UCI German Credit (1994), kept as a small demo set.

Each loader returns features X, labels y (1 = approved / good risk), a frame of
group labels, and a spec dict that tells the audit what the columns mean.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "credit_g.csv"  # German Credit cache

HMDA_NUMERIC = [
    "loan_amount", "income", "property_value", "combined_loan_to_value_ratio",
    "loan_term", "tract_to_msa_income_percentage",
]
HMDA_CATEGORICAL = [
    "debt_to_income_ratio", "loan_type", "derived_loan_product_type",
    "conforming_loan_limit", "preapproval",
]
HMDA_PROTECTED = ["derived_race", "derived_ethnicity", "derived_sex", "applicant_age"]
# Keep first-lien, owner-occupied home purchases that are not business loans,
# reverse mortgages or lines of credit. Codes follow the HMDA data dictionary.
HMDA_FILTERS = {
    "loan_purpose": 1, "occupancy_type": 1, "lien_status": 1, "reverse_mortgage": 2,
    "open-end_line_of_credit": 2, "business_or_commercial_purpose": 2,
}
HMDA_NEEDED = set(HMDA_NUMERIC + HMDA_CATEGORICAL + HMDA_PROTECTED + ["action_taken"] + list(HMDA_FILTERS))

HMDA = {
    "name": "HMDA 2025 mortgage applications (CFPB/FFIEC)",
    "source": "https://ffiec.cfpb.gov/data-browser/",
    "attributes": {
        "race": {"reference": "White", "other": "Black"},
        "sex": {"reference": "Male", "other": "Female"},
        "age": {"reference": "25 to 64", "other": "65 and over"},
    },
    "protected": HMDA_PROTECTED,
    "hero": "Would this mortgage model treat every applicant the same?",
    "p_label": "chance of approval",
    "fail_note": "The label is a past lender decision, so a model can copy a lender’s choices without explaining them.",
    "show": ["loan_amount", "income", "combined_loan_to_value_ratio", "debt_to_income_ratio", "loan_type"],
    "mistake_kinds": ["Model approves, record shows denied", "Model denies, record shows approved"],
    "form": [
        {"id": "race", "label": "Race", "col": "derived_race", "kind": "select"},
        {"id": "sex", "label": "Sex", "col": "derived_sex", "kind": "select"},
        {"id": "age", "label": "Age band", "col": "applicant_age", "kind": "select",
         "choices": ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]},
        {"id": "loan_amount", "label": "Loan amount ($)", "col": "loan_amount", "kind": "range"},
        {"id": "income", "label": "Income ($ thousands)", "col": "income", "kind": "range"},
        {"id": "ltv", "label": "Loan-to-value (%)", "col": "combined_loan_to_value_ratio", "kind": "range"},
        {"id": "dti", "label": "Debt-to-income", "col": "debt_to_income_ratio", "kind": "select"},
    ],
    "cases": [
        {"label": "Race set to White", "set": {"race": "White"}},
        {"label": "Race set to Black", "set": {"race": "Black or African American"}},
        {"label": "Sex set to Female", "set": {"sex": "Female"}},
        {"label": "Sex set to Male", "set": {"sex": "Male"}},
        {"label": "Age band 25-34", "set": {"age": "25-34"}},
        {"label": "Age band 65-74", "set": {"age": "65-74"}},
    ],
    "label": "approved in the historical record (originated, or approved but not accepted)",
    "decision_label": True,
    "cost_fp": None,
    "scorer": True,
    "match_rate": True,  # cutoff chosen so the model approves as many as the training data did
    "tpr_text": "TPR is the share of applicants approved in the record that the model also approves; FPR is the share denied in the record that the model approves.",
    "caveat": "Public HMDA records do not include credit scores, so a gap does not prove discrimination and no gap does not prove fairness.",
    "default_splits": 20,
}
GERMAN = {
    "name": "German Credit (1994), demo data only",
    "source": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
    "attributes": {
        "age": {"reference": "25 and over", "other": "under 25"},
        "sex": {"reference": "male", "other": "female"},
    },
    "protected": ["age", "personal_status"],
    "hero": "Would this loan model treat you the same at 22 and at 42?",
    "p_label": "likely to repay",
    "fail_note": "Two applicants with the same recorded details can have different outcomes, and the data holds no repayment history after approval.",
    "show": ["age", "duration", "credit_amount", "checking_status", "savings_status", "employment", "housing", "purpose"],
    "mistake_kinds": ["Approved, but a bad risk", "Declined, but a good risk"],
    "form": [
        {"id": "age", "label": "Age", "col": "age", "kind": "range"},
        {"id": "sex", "label": "Sex", "col": "personal_status", "kind": "sex", "choices": ["female", "male"]},
        {"id": "duration", "label": "Duration (months)", "col": "duration", "kind": "range"},
        {"id": "credit_amount", "label": "Loan amount (DM)", "col": "credit_amount", "kind": "range"},
        {"id": "checking_status", "label": "Checking account", "col": "checking_status", "kind": "select"},
        {"id": "savings_status", "label": "Savings", "col": "savings_status", "kind": "select"},
        {"id": "employment", "label": "Time in current job", "col": "employment", "kind": "select"},
        {"id": "housing", "label": "Housing", "col": "housing", "kind": "select"},
        {"id": "purpose", "label": "Purpose", "col": "purpose", "kind": "select"},
    ],
    "cases": [
        {"label": "Age set to 24", "set": {"age": 24}}, {"label": "Age set to 45", "set": {"age": 45}},
        {"label": "Sex set to female", "set": {"sex": "female"}}, {"label": "Sex set to male", "set": {"sex": "male"}},
    ],
    "label": "good credit risk",
    "decision_label": False,
    "cost_fp": 5,  # the dataset documentation prices approving a bad risk at 5 and declining a good one at 1
    "scorer": True,
    "match_rate": False,
    "tpr_text": "TPR is the share of good borrowers approved; FPR is the share of bad borrowers approved.",
    "caveat": "A 1994 German dataset says little about any lender today.",
    "default_splits": 50,
}
SPECS = {"hmda": HMDA, "german": GERMAN}


def _german():
    if not CACHE.exists():
        from sklearn.datasets import fetch_openml

        CACHE.parent.mkdir(exist_ok=True)
        fetch_openml("credit-g", version=1, as_frame=True).frame.to_csv(CACHE, index=False)
    df = pd.read_csv(CACHE)
    y = (df.pop("class") == "good").astype(int)
    groups = pd.DataFrame(
        {
            "age": np.where(df["age"] < 25, "under 25", "25 and over"),
            "sex": np.where(df["personal_status"].str.startswith("female"), "female", "male"),
        },
        index=df.index,
    )
    return df, y, groups


def _hmda(path, max_rows, seed):
    if path is None:
        raw = DATA_DIR / "hmda_raw.csv"
        path = raw if raw.exists() else DATA_DIR / "hmda_sample.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m audit.fetch_hmda --state WA, or pass --csv."
        )
    df = pd.read_csv(path, usecols=lambda c: c in HMDA_NEEDED, low_memory=False)
    df["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce")
    df = df[df["action_taken"].isin([1, 2, 3])]
    for col, val in HMDA_FILTERS.items():
        if col in df.columns:
            df = df[pd.to_numeric(df[col], errors="coerce") == val]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    df = df.reset_index(drop=True)
    if path.name != "hmda_sample.csv":
        df.to_csv(DATA_DIR / "hmda_sample.csv", index=False)

    y = df["action_taken"].isin([1, 2]).astype(int)
    numeric = [c for c in HMDA_NUMERIC if c in df.columns]
    categorical = [c for c in HMDA_CATEGORICAL if c in df.columns]
    X = df[numeric + categorical + HMDA_PROTECTED].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical + HMDA_PROTECTED:
        X[c] = X[c].fillna("missing").astype(str)

    race = X["derived_race"]
    age = X["applicant_age"]
    groups = pd.DataFrame(
        {
            "race": np.where(race == "Black or African American", "Black",
                             np.where(race == "White", "White", "excluded")),
            "sex": np.where(X["derived_sex"].isin(["Male", "Female"]), X["derived_sex"], "excluded"),
            "age": np.where(age.isin(["65-74", ">74"]), "65 and over",
                            np.where(age.isin(["25-34", "35-44", "45-54", "55-64"]), "25 to 64", "excluded")),
        },
        index=X.index,
    )
    return X, y, groups


def load(name="german", path=None, max_rows=40000, seed=0):
    spec = SPECS[name]
    X, y, groups = _german() if name == "german" else _hmda(path, max_rows, seed)
    return X, y, groups, spec
