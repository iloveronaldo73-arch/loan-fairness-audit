"""The audit pipeline.

For each protected trait (age, race) three decision strategies are scored on
held-out applications over many random 60/20/20 splits:

    observed in data     the decisions actually recorded, as a benchmark
    full model           logistic regression that sees the demographic columns
    blind model          the same model with every demographic column removed
    blind + group cutoffs blind scores, with a separate cutoff per group chosen
                         so that each group is approved at the overall rate

Every model cutoff is calibrated on the validation part so the predicted
approval rate matches the rate recorded in the data; without that, an
uncalibrated 0.5 cut at a 90% base rate approves almost everyone and hides real
gaps. The proxy check asks how well the columns that remain can rebuild the
protected trait on their own (AUC, 0.5 = chance).

`python -m audit.build_site` calls `run()` and writes the numbers into the page.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import dataset

STRATEGIES = ["full model", "blind model", "blind + group cutoffs"]

# A group cutoff is only tuned when the group has at least this many rows.
MIN_GROUP = 100


def make_model(features: Sequence[str]) -> Pipeline:
    """Impute and scale the numeric columns, one-hot the rest, then fit a logistic model."""
    numeric = [c for c in features if c in dataset.NUMERIC]
    categorical = [c for c in features if c in dataset.CATEGORICAL]
    transform = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [("transform", transform), ("model", LogisticRegression(max_iter=2000))]
    )


def calibrated_cut(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Score cutoff whose predicted approval rate equals the observed base rate."""
    return float(np.quantile(p_val, 1 - y_val.mean()))


def group_cutoffs(
    p_val: np.ndarray, groups: np.ndarray, target_rate: float
) -> Dict[str, float]:
    """Per-group cutoffs that approve each group at `target_rate` on validation."""
    cuts: Dict[str, float] = {}
    for group in sorted(set(groups.tolist())):
        scores = p_val[groups == group]
        cuts[group] = (
            float(np.quantile(scores, 1 - target_rate))
            if len(scores) >= MIN_GROUP
            else float(np.median(p_val))
        )
    return cuts


def _split_indices(n: int, rng, test_size: float, val_size: float):
    order = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    return order[:n_test], order[n_test : n_test + n_val], order[n_test + n_val :]


def run(
    splits: int = 50,
    seed: int = 0,
    test_size: float = 0.2,
    val_size: float = 0.2,
    refresh: bool = False,
) -> dict:
    """Score every strategy and return the dict the page renders."""
    df = dataset.load(refresh=refresh)
    y = df["target"].to_numpy()

    features = dataset.NUMERIC + dataset.CATEGORICAL
    blind_features = [c for c in features if c not in dataset.PROTECTED]
    X = df[features]

    observations = {
        trait: {
            strategy: {"other": [], "ref": [], "accuracy": [], "gap": []}
            for strategy in STRATEGIES
        }
        for trait in dataset.GROUPS
    }
    recorded = {
        trait: {"other": [], "ref": [], "gap": []} for trait in dataset.GROUPS
    }
    proxy_auc = {trait: [] for trait in dataset.GROUPS}

    rng = np.random.default_rng(seed)
    for _ in range(splits):
        test, val, train = _split_indices(len(df), rng, test_size, val_size)
        y_train, y_val, y_test = y[train], y[val], y[test]

        full = make_model(features).fit(X.iloc[train], y_train)
        p_full_val = full.predict_proba(X.iloc[val])[:, 1]
        p_full_test = full.predict_proba(X.iloc[test])[:, 1]
        cut_full = calibrated_cut(y_val, p_full_val)

        blind = make_model(blind_features).fit(X.iloc[train], y_train)
        p_blind_val = blind.predict_proba(X.iloc[val])[:, 1]
        p_blind_test = blind.predict_proba(X.iloc[test])[:, 1]
        cut_blind = calibrated_cut(y_val, p_blind_val)
        overall_rate = float((p_blind_val >= cut_blind).mean())

        for trait, meta in dataset.GROUPS.items():
            g_val = df[meta["column"]].to_numpy()[val]
            g_test = df[meta["column"]].to_numpy()[test]

            cuts = group_cutoffs(p_blind_val, g_val, overall_rate)
            decisions = {
                "full model": p_full_test >= cut_full,
                "blind model": p_blind_test >= cut_blind,
                "blind + group cutoffs": p_blind_test
                >= np.array([cuts[g] for g in g_test]),
            }

            for strategy, approved in decisions.items():
                record = observations[trait][strategy]
                rate_other = float(approved[g_test == meta["other"]].mean() * 100)
                rate_ref = float(approved[g_test == meta["ref"]].mean() * 100)
                record["other"].append(rate_other)
                record["ref"].append(rate_ref)
                record["accuracy"].append(float((approved == y_test).mean() * 100))
                record["gap"].append(rate_other - rate_ref)

            approved_actual = y_test == 1
            rate_other = float(approved_actual[g_test == meta["other"]].mean() * 100)
            rate_ref = float(approved_actual[g_test == meta["ref"]].mean() * 100)
            recorded[trait]["other"].append(rate_other)
            recorded[trait]["ref"].append(rate_ref)
            recorded[trait]["gap"].append(rate_other - rate_ref)

            # Proxy check: can the remaining columns rebuild the trait?
            is_other = (df[meta["column"]] == meta["other"]).to_numpy()
            probe = make_model(blind_features).fit(X.iloc[train], is_other[train])
            proxy_auc[trait].append(
                roc_auc_score(is_other[test], probe.predict_proba(X.iloc[test])[:, 1])
            )

    base = round(float(y.mean() * 100), 1)
    res = {}
    for trait in dataset.GROUPS:
        other = np.array(recorded[trait]["other"])
        ref = np.array(recorded[trait]["ref"])
        gap = other - ref
        res[trait] = {
            "approve everyone": [100.0, 100.0, base, 0.0, 0.0],
            "observed in data": [
                round(float(other.mean()), 1),
                round(float(ref.mean()), 1),
                None,
                round(float(np.percentile(gap, 2.5)), 1),
                round(float(np.percentile(gap, 97.5)), 1),
            ],
        }
        for strategy in STRATEGIES:
            record = observations[trait][strategy]
            other = np.array(record["other"])
            ref = np.array(record["ref"])
            gap = other - ref
            res[trait][strategy] = [
                round(float(other.mean()), 1),
                round(float(ref.mean()), 1),
                round(float(np.mean(record["accuracy"])), 1),
                round(float(np.percentile(gap, 2.5)), 1),
                round(float(np.percentile(gap, 97.5)), 1),
            ]

    return {
        "dataset": dataset.TITLE,
        "source": dataset.SOURCE,
        "label": dataset.LABEL,
        "caveat": dataset.CAVEAT,
        "rows": int(len(df)),
        "base": base,
        "model": "logreg",
        "splits": splits,
        "groups": {
            trait: {
                "other": meta["other"],
                "ref": meta["ref"],
                "n": [
                    int((df[meta["column"]] == meta["other"]).sum()),
                    int((df[meta["column"]] == meta["ref"]).sum()),
                ],
            }
            for trait, meta in dataset.GROUPS.items()
        },
        "res": res,
        "proxy": {trait: round(float(np.mean(scores)), 2) for trait, scores in proxy_auc.items()},
    }
