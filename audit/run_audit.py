"""Run the fairness audit and write tables, a figure and meta.json to results/.

    python -m audit.run_audit                       # German Credit, the brief dataset
    python -m audit.run_audit --dataset hmda        # optional: HMDA 2025 sample in data/
    python -m audit.run_audit --model gbm --splits 10
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import data
from .metrics import apply_cutoffs, compare, pick_cutoffs
from .models import build

BASE_CUTOFF = 0.5
METRICS = ["auc", "accuracy", "cost", "approval_other", "approval_ref", "approval_gap", "impact_ratio", "tpr_gap", "fpr_gap"]
NOT_FAIR_SCORES = ["observed approvals", "approve everyone"]  # context rows, left out of the figure


def make_split(y, groups, seed):
    """60% train, 20% validation, 20% test, stratified on label and the first group column."""
    strata = y.astype(str) + "_" + groups.iloc[:, 0].astype(str)
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=0.2, stratify=strata, random_state=seed)
    train, val = train_test_split(rest, test_size=0.25, stratify=strata.iloc[rest], random_state=seed)
    return train, val, test


def evaluate(X, y, groups, spec, kind="logreg", n_splits=50):
    X_blind = X.drop(columns=spec["protected"])
    rows = []
    for seed in range(n_splits):
        train, val, test = make_split(y, groups, seed)
        full = build(kind, X, seed).fit(X.iloc[train], y.iloc[train])
        blind = build(kind, X_blind, seed).fit(X_blind.iloc[train], y.iloc[train])

        s_full = full.predict_proba(X.iloc[test])[:, 1]
        s_full_val = full.predict_proba(X.iloc[val])[:, 1]
        s_blind = blind.predict_proba(X_blind.iloc[test])[:, 1]
        s_blind_val = blind.predict_proba(X_blind.iloc[val])[:, 1]
        y_test = y.iloc[test].to_numpy()
        y_val = y.iloc[val].to_numpy()

        # Base cutoff: 0.5, or (HMDA) the score that approves the same share as the training data did.
        cut_full = cut_blind = BASE_CUTOFF
        if spec["match_rate"]:
            keep = 1 - y.iloc[train].mean()
            cut_full, cut_blind = np.quantile(s_full_val, keep), np.quantile(s_blind_val, keep)

        for attr, info in spec["attributes"].items():
            levels = [info["reference"], info["other"]]
            g_test = groups[attr].iloc[test].to_numpy()
            g_val = groups[attr].iloc[val].to_numpy()
            cuts = pick_cutoffs(s_blind_val, y_val, g_val, levels, cut_blind)

            decisions = {}
            if spec["decision_label"]:
                decisions["observed approvals"] = y_test
            decisions.update(
                {
                    "approve everyone": np.ones(len(test), dtype=int),
                    "full model": (s_full >= cut_full).astype(int),
                    "blind model": (s_blind >= cut_blind).astype(int),
                    "blind + group cutoffs": apply_cutoffs(s_blind, g_test, cuts, cut_blind),
                }
            )
            if spec.get("cost_fp"):  # approving a bad risk costs cost_fp times a missed good one
                decisions["blind + cost cutoff"] = (s_blind >= spec["cost_fp"] / (spec["cost_fp"] + 1)).astype(int)
            for name, pred in decisions.items():
                row = {"attribute": attr, "strategy": name, "seed": seed,
                       "accuracy": (pred == y_test).mean(), "approval": pred.mean()}
                c = spec.get("cost_fp")
                row["cost"] = ((((pred == 1) & (y_test == 0)).sum() * c + ((pred == 0) & (y_test == 1)).sum()) / len(pred)
                               if c else np.nan)
                row.update(compare(y_test, pred, g_test, info["other"], info["reference"]))
                scores = {"full model": s_full, "blind model": s_blind, "blind + group cutoffs": s_blind}.get(name)
                row["auc"] = roc_auc_score(y_test, scores) if scores is not None else np.nan
                if name == "observed approvals":  # this row is the label itself, so error rates mean nothing
                    for k in ["accuracy", "tpr_other", "tpr_ref", "tpr_gap", "fpr_other", "fpr_ref",
                              "fpr_gap", "accuracy_other", "accuracy_ref", "accuracy_gap"]:
                        row[k] = np.nan
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


def proxy_check(X, groups, spec):
    """Can the columns left in the blind model predict the protected attribute?"""
    X_blind = X.drop(columns=spec["protected"])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for attr, info in spec["attributes"].items():
        keep = groups[attr].isin([info["other"], info["reference"]]).to_numpy()
        target = (groups[attr][keep] == info["other"]).astype(int)
        auc = cross_val_score(build("logreg", X_blind[keep]), X_blind[keep], target, cv=cv, scoring="roc_auc")
        rows.append({"attribute": attr, "group": info["other"], "auc": auc.mean(), "auc_std": auc.std()})
    return pd.DataFrame(rows)


def plot_gaps(summary, spec, path):
    panels = [("approval_gap", "Approval rate gap"), ("tpr_gap", "True positive rate gap")]
    attrs = list(spec["attributes"].items())
    fig, axes = plt.subplots(2, len(attrs), figsize=(4.6 * len(attrs), 5.5), sharey=True, squeeze=False)
    for j, (attr, info) in enumerate(attrs):
        sub = summary[(summary["attribute"] == attr) & ~summary["strategy"].isin(NOT_FAIR_SCORES)]
        sub = sub.set_index("strategy")
        ypos = np.arange(len(sub))[::-1]
        for i, (metric, label) in enumerate(panels):
            ax = axes[i, j]
            mean, lo, hi = (sub[f"{metric}_{s}"].to_numpy() * 100 for s in ("mean", "lo", "hi"))
            ax.errorbar(mean, ypos, xerr=[mean - lo, hi - mean], fmt="o", color="#4f5bff", ecolor="#9aa0b5",
                        elinewidth=2.5, markersize=8, capsize=0)
            ax.axvline(0, color="#5d6072", linestyle=":", linewidth=1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.2)
            ax.set_yticks(ypos)
            ax.set_yticklabels(sub.index)
            ax.set_xlabel(f"{label} (percentage points)")
            if i == 0:
                ax.set_title(f"{info['other']} vs {info['reference']}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pp(row, metric):
    vals = [row[f"{metric}_{s}"] * 100 for s in ("mean", "lo", "hi")]
    if np.isnan(vals[0]):
        return "n/a"
    return f"{vals[0]:+.1f} ({vals[1]:+.1f} to {vals[2]:+.1f})"


def pct(x):
    return "n/a" if np.isnan(x) else f"{x:.1%}"


def to_markdown(summary, proxy, groups, spec, kind, n_splits, n_rows):
    cut_text = ("Cutoff: chosen per model so it approves the same share as the training data."
                if spec["match_rate"] else f"Base cutoff: {BASE_CUTOFF}.")
    lines = [
        "# Audit results", "",
        f"Dataset: {spec['name']}. {n_rows:,} applications. Label: {spec['label']}.",
        f"Model: {kind}. Splits: {n_splits} random 60/20/20 train/validation/test splits. {cut_text}", "",
        "All numbers come from the test part of each split. Gaps are the first group minus the reference "
        "group, in percentage points, with the 2.5th to 97.5th percentile across splits in brackets. "
        "A negative gap means the first group does worse. " + spec["tpr_text"], "",
    ]
    for attr, info in spec["attributes"].items():
        counts = groups[attr].value_counts()
        lines += [
            f"## {attr}: {info['other']} vs {info['reference']}", "",
            f"Applicants in the data: {info['other']} {counts.get(info['other'], 0):,}, "
            f"{info['reference']} {counts.get(info['reference'], 0):,}.", "",
            f"| Strategy | AUC | Accuracy | Approval, {info['other']} | Approval, {info['reference']} | Approval gap | TPR gap | FPR gap | Cost |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in summary[summary["attribute"] == attr].iterrows():
            lines.append(
                f"| {r['strategy']} | {'n/a' if np.isnan(r['auc_mean']) else format(r['auc_mean'], '.3f')} | {pct(r['accuracy_mean'])} | {pct(r['approval_other_mean'])} "
                f"| {pct(r['approval_ref_mean'])} | {pp(r, 'approval_gap')} | {pp(r, 'tpr_gap')} | {pp(r, 'fpr_gap')} "
                f"| {'n/a' if np.isnan(r['cost_mean']) else format(r['cost_mean'], '.2f')} |"
            )
        lines.append("")
    lines += ["## Proxy check", "",
              "AUC of a logistic regression that predicts group membership from the columns the blind "
              "model uses (5-fold cross-validation). 0.5 is chance.", "",
              "| Attribute | Group | AUC |", "|---|---|---|"]
    for _, r in proxy.iterrows():
        lines.append(f"| {r['attribute']} | {r['group']} | {r['auc']:.2f} (std {r['auc_std']:.2f}) |")
    lines.append("")
    return "\n".join(lines)


def run(X, y, groups, spec, kind="logreg", n_splits=None, out="results"):
    n_splits = n_splits or spec["default_splits"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = evaluate(X, y, groups, spec, kind, n_splits)
    runs.to_csv(out / "runs.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(out / "summary.csv", index=False)
    proxy = proxy_check(X, groups, spec)
    proxy.to_csv(out / "proxy_check.csv", index=False)
    plot_gaps(summary, spec, out / "gaps.png")
    (out / "summary.md").write_text(to_markdown(summary, proxy, groups, spec, kind, n_splits, len(X)))
    meta = {
        "dataset": spec["name"], "source": spec["source"], "label": spec["label"], "caveat": spec["caveat"],
        "cost_fp": spec.get("cost_fp"), "hero": spec["hero"], "p_label": spec["p_label"], "fail_note": spec["fail_note"], "rows": int(len(X)), "base": round(float(y.mean()) * 100, 1), "model": kind, "splits": n_splits,
        "groups": {
            a: {"other": i["other"], "ref": i["reference"],
                "n": [int((groups[a] == i["other"]).sum()), int((groups[a] == i["reference"]).sum())]}
            for a, i in spec["attributes"].items()
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    from . import explain

    explain.run_extras(X, y, groups, spec, out)
    return summary, proxy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="german", choices=list(data.SPECS))
    p.add_argument("--csv", default=None, help="HMDA csv; default is data/hmda_raw.csv or data/hmda_sample.csv")
    p.add_argument("--max-rows", type=int, default=40000)
    p.add_argument("--model", default="logreg", choices=["logreg", "gbm"])
    p.add_argument("--splits", type=int, default=None)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    X, y, groups, spec = data.load(args.dataset, args.csv, args.max_rows)
    run(X, y, groups, spec, args.model, args.splits, args.out)
    print(Path(args.out, "summary.md").read_text())


if __name__ == "__main__":
    main()
