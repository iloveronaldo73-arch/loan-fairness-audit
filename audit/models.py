"""Model construction. Preprocessing is the same for every model, so the only
thing that changes between runs is which columns go in."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build(kind, X, seed=0):
    numeric = X.select_dtypes("number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_steps = Pipeline(
        [("fill", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
    )
    prep = ColumnTransformer(
        [
            ("num", num_steps, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif kind == "gbm":
        clf = GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([("prep", prep), ("clf", clf)])
