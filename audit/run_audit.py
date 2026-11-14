"""Run the fairness audit and write tables and a figure to results/.

    python -m audit.run_audit                 # logistic regression, 50 splits
    python -m audit.run_audit --model gbm     # gradient boosting instead
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import data
from .metrics import apply_cutoffs, compare, pick_cutoffs
from .models import build

BASE_CUTOFF = 0.5
STRATEGIES = ["approve everyone", "full model", "blind model", "blind + group cutoffs"]
METRICS = [
    "accuracy",
    "approval_other",
    "approval_ref",
    "approval_gap",
    "impact_ratio",
    "tpr_gap",
    "fpr_gap",
]


def make_split(y, groups, seed):
    """60% train, 20% validation, 20% test, stratified on label and age group."""
    strata = y.astype(str) + "_" + groups["age"]
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=0.2, stratify=strata, random_state=seed)
    train, val = train_test_split(
        rest, test_size=0.25, stratify=strata.iloc[rest], random_state=seed
    )
    return train, val, test


def evaluate(X, y, groups, kind="logreg", n_splits=50):
    X_blind = X.drop(columns=data.PROTECTED_COLUMNS)
    rows = []
    for seed in range(n_splits):
        train, val, test = make_split(y, groups, seed)
        full = build(kind, X, seed).fit(X.iloc[train], y.iloc[train])
        blind = build(kind, X_blind, seed).fit(X_blind.iloc[train], y.iloc[train])

        s_full = full.predict_proba(X.iloc[test])[:, 1]
        s_blind = blind.predict_proba(X_blind.iloc[test])[:, 1]
        s_blind_val = blind.predict_proba(X_blind.iloc[val])[:, 1]
        y_test = y.iloc[test].to_numpy()
        y_val = y.iloc[val].to_numpy()

        for attr, info in data.ATTRIBUTES.items():
            levels = [info["reference"], info["other"]]
            g_test = groups[attr].iloc[test].to_numpy()
            g_val = groups[attr].iloc[val].to_numpy()
            cuts = pick_cutoffs(s_blind_val, y_val, g_val, levels, BASE_CUTOFF)

            decisions = {
                "approve everyone": np.ones(len(test), dtype=int),
                "full model": (s_full >= BASE_CUTOFF).astype(int),
                "blind model": (s_blind >= BASE_CUTOFF).astype(int),
                "blind + group cutoffs": apply_cutoffs(s_blind, g_test, cuts),
            }
            for name, pred in decisions.items():
                row = {
                    "attribute": attr,
                    "strategy": name,
                    "seed": seed,
                    "accuracy": (pred == y_test).mean(),
                    "approval": pred.mean(),
                }
                row.update(compare(y_test, pred, g_test, info["other"], info["reference"]))
                if name == "blind + group cutoffs":
                    row["cutoff_other"] = cuts[info["other"]]
                    row["cutoff_ref"] = cuts[info["reference"]]
                rows.append(row)
    return pd.DataFrame(rows)


def summarise(runs):
    grouped = runs.groupby(["attribute", "strategy"], sort=False)[METRICS]
    out = grouped.mean().add_suffix("_mean")
    out = out.join(grouped.quantile(0.025).add_suffix("_lo"))
    out = out.join(grouped.quantile(0.975).add_suffix("_hi"))
    return out.reset_index()


def proxy_check(X, groups):
    """Can the columns left in the blind model predict the protected attribute?"""
    X_blind = X.drop(columns=data.PROTECTED_COLUMNS)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for attr, info in data.ATTRIBUTES.items():
        target = (groups[attr] == info["other"]).astype(int)
        auc = cross_val_score(
            build("logreg", X_blind), X_blind, target, cv=cv, scoring="roc_auc"
        )
        rows.append(
            {"attribute": attr, "group": info["other"], "auc": auc.mean(), "auc_std": auc.std()}
        )
    return pd.DataFrame(rows)


def plot_gaps(summary, path):
    strategies = STRATEGIES[1:]
    panels = [("approval_gap", "Approval rate gap"), ("tpr_gap", "True positive rate gap")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharey=True)
    for j, (attr, info) in enumerate(data.ATTRIBUTES.items()):
        sub = summary[(summary["attribute"] == attr) & summary["strategy"].isin(strategies)]
        sub = sub.set_index("strategy").loc[strategies]
        ypos = np.arange(len(strategies))[::-1]
        for i, (metric, label) in enumerate(panels):
            ax = axes[i, j]
            mean = sub[f"{metric}_mean"].to_numpy() * 100
            lo = sub[f"{metric}_lo"].to_numpy() * 100
            hi = sub[f"{metric}_hi"].to_numpy() * 100
            ax.errorbar(mean, ypos, xerr=[mean - lo, hi - mean], fmt="o", capsize=3, color="#222")
            ax.axvline(0, color="#999", linewidth=1)
            ax.set_yticks(ypos)
            ax.set_yticklabels(strategies)
            ax.set_xlabel(f"{label} (percentage points)")
            if i == 0:
                ax.set_title(f"{info['other']} vs {info['reference']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pp(row, metric):
    """Mean and 95% range across splits, in percentage points."""
    m, lo, hi = (row[f"{metric}_{s}"] * 100 for s in ("mean", "lo", "hi"))
    return f"{m:+.1f} ({lo:+.1f} to {hi:+.1f})"


def to_markdown(summary, proxy, groups, kind, n_splits):
    lines = [
        "# Audit results",
        "",
        f"Model: {kind}. Splits: {n_splits} random 60/20/20 train/validation/test splits. "
        f"Base cutoff: {BASE_CUTOFF}.",
        "",
        "All numbers come from the test part of each split. Gaps are the first group minus "
        "the reference group, in percentage points, with the 2.5th to 97.5th percentile "
        "across splits in brackets. A negative gap means the first group does worse. "
        "TPR is the share of good borrowers approved, FPR the share of bad borrowers approved.",
        "",
    ]
    for attr, info in data.ATTRIBUTES.items():
        counts = groups[attr].value_counts()
        lines += [
            f"## {attr}: {info['other']} vs {info['reference']}",
            "",
            f"Applicants in the data: {info['other']} {counts[info['other']]}, "
            f"{info['reference']} {counts[info['reference']]}.",
            "",
            f"| Strategy | Accuracy | Approval, {info['other']} | Approval, {info['reference']} "
            "| Approval gap | TPR gap | FPR gap |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in summary[summary["attribute"] == attr].iterrows():
            lines.append(
                f"| {r['strategy']} | {r['accuracy_mean']:.1%} | {r['approval_other_mean']:.1%} "
                f"| {r['approval_ref_mean']:.1%} | {pp(r, 'approval_gap')} "
                f"| {pp(r, 'tpr_gap')} | {pp(r, 'fpr_gap')} |"
            )
        lines.append("")
    lines += [
        "## Proxy check",
        "",
        "AUC of a logistic regression that predicts group membership from the columns the "
        "blind model uses (5-fold cross-validation). 0.5 is chance.",
        "",
        "| Attribute | Group | AUC |",
        "|---|---|---|",
    ]
    for _, r in proxy.iterrows():
        lines.append(f"| {r['attribute']} | {r['group']} | {r['auc']:.2f} (std {r['auc_std']:.2f}) |")
    lines.append("")
    return "\n".join(lines)


def run(X, y, groups, kind="logreg", n_splits=50, out="results"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = evaluate(X, y, groups, kind, n_splits)
    runs.to_csv(out / "runs.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(out / "summary.csv", index=False)
    proxy = proxy_check(X, groups)
    proxy.to_csv(out / "proxy_check.csv", index=False)
    plot_gaps(summary, out / "gaps.png")
    (out / "summary.md").write_text(to_markdown(summary, proxy, groups, kind, n_splits))
    return summary, proxy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="logreg", choices=["logreg", "gbm"])
    p.add_argument("--splits", type=int, default=50)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    X, y, groups = data.load()
    run(X, y, groups, args.model, args.splits, args.out)
    print(Path(args.out, "summary.md").read_text())


if __name__ == "__main__":
    main()
