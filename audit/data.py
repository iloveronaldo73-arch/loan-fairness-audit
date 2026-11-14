"""Load the German Credit data and build the group columns.

Source: UCI Statlog (German Credit Data), fetched through OpenML as "credit-g".
The first run downloads it to data/credit_g.csv; after that everything reads
the local copy, so the demo works without a network connection.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data" / "credit_g.csv"

AGE_CUTOFF = 25

# The two groups for each audited attribute. "other" is the group we check for
# worse treatment, "reference" is the group we compare it against.
ATTRIBUTES = {
    "age": {"reference": "25 and over", "other": "under 25"},
    "sex": {"reference": "male", "other": "female"},
}

# Columns that carry the protected attributes. personal_status mixes sex and
# marital status in one field, so the whole column is dropped for the blind model.
PROTECTED_COLUMNS = ["age", "personal_status"]


def _download():
    from sklearn.datasets import fetch_openml

    frame = fetch_openml("credit-g", version=1, as_frame=True).frame
    CACHE.parent.mkdir(exist_ok=True)
    frame.to_csv(CACHE, index=False)


def load():
    """Return features X, labels y (1 = good credit risk) and a frame of groups."""
    if not CACHE.exists():
        _download()
    df = pd.read_csv(CACHE)
    y = (df.pop("class") == "good").astype(int)
    groups = pd.DataFrame(
        {
            "age": np.where(df["age"] < AGE_CUTOFF, "under 25", "25 and over"),
            "sex": np.where(
                df["personal_status"].str.startswith("female"), "female", "male"
            ),
        },
        index=df.index,
    )
    return df, y, groups
