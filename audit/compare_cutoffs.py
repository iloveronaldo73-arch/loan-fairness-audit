"""Compare per-group cutoff rules for the blind model.

Every rule sees the same scores; only the decision thresholds differ. Each rule
is fitted on the validation part and scored on the held-out test part over the
same random splits the page uses, so the numbers are directly comparable.

    none             one calibrated cutoff for everyone
    equal approval   each group approved at the overall rate (demographic parity)
    equal TPR        each group's true positive rate matches (equal opportunity)
    equal FPR        each group's false positive rate matches (predictive equality)
    equal error      each group's misclassification rate matches (accuracy parity)

    python -m audit.compare_cutoffs [--splits 50] [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np

from . import dataset
from .audit import _split_indices, calibrated_cut, make_model

RULES = ["none", "equal approval", "equal TPR", "equal FPR", "equal error"]
MIN_GROUP = 100


def _equal_error_cut(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """Cutoff whose misclassification rate is closest to `target` within a group."""
    order = np.argsort(p)
    scores, labels = p[order], y[order]
    n = len(scores)
    positives = int(labels.sum())
    # suffix[i] = positives at or after position i
    suffix = np.concatenate([np.cumsum(labels[::-1])[::-1], np.array([0])])
    candidates = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 200)))
    idx = np.searchsorted(scores, candidates, side="left")
    approved = n - idx
    pos_above = suffix[idx]
    error = ((approved - pos_above) + (positives - pos_above)) / n
    return float(candidates[int(np.argmin(np.abs(error - target)))])


def _rates(y: np.ndarray, approved: np.ndarray) -> dict:
    return {
        "approval": float(approved.mean()),
        "tpr": float(approved[y == 1].mean()),
        "fpr": float(approved[y == 0].mean()),
        "error": float((approved != y).mean()),
    }


def _group_cuts(rule: str, p_val: np.ndarray, y_val: np.ndarray, g_val: np.ndarray, overall: dict):
    cuts = {}
    for group in sorted(set(g_val.tolist())):
        mask = g_val == group
        p_g, y_g = p_val[mask], y_val[mask]
        if len(p_g) < MIN_GROUP:
            cuts[group] = float(np.median(p_val))
            continue
        if rule == "equal approval":
            cuts[group] = float(np.quantile(p_g, 1 - np.clip(overall["approval"], 0.001, 0.999)))
        elif rule == "equal TPR":
            scores = p_g[y_g == 1]
            cuts[group] = float(np.quantile(scores, 1 - np.clip(overall["tpr"], 0.001, 0.999)))
        elif rule == "equal FPR":
            scores = p_g[y_g == 0]
            cuts[group] = float(np.quantile(scores, 1 - np.clip(overall["fpr"], 0.001, 0.999)))
        elif rule == "equal error":
            cuts[group] = _equal_error_cut(p_g, y_g, overall["error"])
        else:
            raise ValueError(rule)
    return cuts


def run(splits: int = 50, seed: int = 0) -> dict:
    df = dataset.load()
    y = df["target"].to_numpy()
    features = dataset.NUMERIC + dataset.CATEGORICAL
    blind_features = [c for c in features if c not in dataset.PROTECTED]
    X = df[features]

    results = {
        trait: {rule: {"other": [], "ref": [], "tpr": [], "fpr": [], "accuracy": [], "gap": []}
                for rule in RULES}
        for trait in dataset.GROUPS
    }

    rng = np.random.default_rng(seed)
    for _ in range(splits):
        test, val, train = _split_indices(len(df), rng, 0.2, 0.2)
        y_train, y_val, y_test = y[train], y[val], y[test]

        model = make_model(blind_features).fit(X.iloc[train], y_train)
        p_val = model.predict_proba(X.iloc[val])[:, 1]
        p_test = model.predict_proba(X.iloc[test])[:, 1]

        cut = calibrated_cut(y_val, p_val)
        overall = _rates(y_val, p_val >= cut)

        for trait, meta in dataset.GROUPS.items():
            g_val = df[meta["column"]].to_numpy()[val]
            g_test = df[meta["column"]].to_numpy()[test]

            for rule in RULES:
                if rule == "none":
                    approved = p_test >= cut
                else:
                    cuts = _group_cuts(rule, p_val, y_val, g_val, overall)
                    approved = p_test >= np.array([cuts[g] for g in g_test])

                record = results[trait][rule]
                other = g_test == meta["other"]
                ref = g_test == meta["ref"]
                rate_other = float(approved[other].mean() * 100)
                rate_ref = float(approved[ref].mean() * 100)
                record["other"].append(rate_other)
                record["ref"].append(rate_ref)
                record["gap"].append(rate_other - rate_ref)
                record["tpr"].append(
                    float(approved[other & (y_test == 1)].mean() * 100)
                    - float(approved[ref & (y_test == 1)].mean() * 100)
                )
                record["fpr"].append(
                    float(approved[other & (y_test == 0)].mean() * 100)
                    - float(approved[ref & (y_test == 0)].mean() * 100)
                )
                record["accuracy"].append(float((approved == y_test).mean() * 100))

    summary = {}
    for trait in dataset.GROUPS:
        summary[trait] = {}
        for rule in RULES:
            record = results[trait][rule]
            gap = np.array(record["gap"])
            # Round the rates the way the page displays them, so the gap subtracts
            # exactly the numbers a reader sees.
            other = round(float(np.mean(record["other"])), 1)
            ref = round(float(np.mean(record["ref"])), 1)
            summary[trait][rule] = {
                "other": other,
                "ref": ref,
                # Same definition as the page: gap between averaged rates, with the
                # band taken from the per-split gaps.
                "gap": other - ref,
                "gap_low": float(np.percentile(gap, 2.5)),
                "gap_high": float(np.percentile(gap, 97.5)),
                "tpr_gap": float(np.mean(record["tpr"])),
                "fpr_gap": float(np.mean(record["fpr"])),
                "accuracy": float(np.mean(record["accuracy"])),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run(splits=args.splits, seed=args.seed)
    for trait, rules in summary.items():
        meta = dataset.GROUPS[trait]
        print(f"\n{trait}: {meta['other']} vs {meta['ref']}")
        header = (f"  {'rule':15} {meta['other']:>7} {meta['ref']:>7}  "
                  f"{'approval gap [95%]':>22}  {'TPR gap':>8} {'FPR gap':>8} {'acc':>6}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for rule, v in rules.items():
            print(f"  {rule:15} {v['other']:7.1f} {v['ref']:7.1f}  "
                  f"{v['gap']:+7.1f} [{v['gap_low']:+6.1f},{v['gap_high']:+6.1f}]  "
                  f"{v['tpr_gap']:+8.1f} {v['fpr_gap']:+8.1f} {v['accuracy']:6.1f}")


if __name__ == "__main__":
    main()
