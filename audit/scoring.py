"""Helpers for the what-if demo in app.py."""
import pandas as pd

from .data import PROTECTED_COLUMNS


def typical_applicant(X):
    """Median for numeric columns, most common value for the rest."""
    row = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            row[col] = X[col].median()
        else:
            row[col] = X[col].mode()[0]
    return row


def what_if(full, blind, base_row, status_by_sex, age, sex, overrides, cutoff=0.5):
    """Score one applicant, then change only age or sex and score again."""
    other_sex = "male" if sex == "female" else "female"
    cases = [
        ("As entered", age, sex),
        ("Age set to 24", 24, sex),
        ("Age set to 45", 45, sex),
        (f"Sex changed to {other_sex}", age, other_sex),
    ]
    rows = []
    for label, a, s in cases:
        applicant = {**base_row, **overrides, "age": a, "personal_status": status_by_sex[s]}
        df = pd.DataFrame([applicant])
        p_full = full.predict_proba(df)[0, 1]
        p_blind = blind.predict_proba(df.drop(columns=PROTECTED_COLUMNS))[0, 1]
        rows.append(
            {
                "Case": label,
                "Full model": f"{p_full:.0%}",
                "Full model decision": "approve" if p_full >= cutoff else "decline",
                "Blind model": f"{p_blind:.0%}",
                "Blind model decision": "approve" if p_blind >= cutoff else "decline",
            }
        )
    return pd.DataFrame(rows)
