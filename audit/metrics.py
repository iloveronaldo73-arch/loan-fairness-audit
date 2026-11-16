"""Group metrics.

Throughout: y_true = 1 means the applicant would have repaid (a good borrower),
y_pred = 1 means the model approved the loan.
"""
import numpy as np


def rates(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    good = y_true == 1
    bad = y_true == 0
    return {
        "n": len(y_true),
        "approval": y_pred.mean() if len(y_pred) else np.nan,
        # share of good borrowers who got approved
        "tpr": y_pred[good].mean() if good.any() else np.nan,
        # share of bad borrowers who got approved
        "fpr": y_pred[bad].mean() if bad.any() else np.nan,
        "accuracy": (y_pred == y_true).mean() if len(y_true) else np.nan,
    }


def compare(y_true, y_pred, group, other, reference):
    """Rates for both groups and the gaps between them (other minus reference)."""
    y_true, y_pred, group = map(np.asarray, (y_true, y_pred, group))
    a = rates(y_true[group == other], y_pred[group == other])
    b = rates(y_true[group == reference], y_pred[group == reference])

    out = {"n_other": a["n"], "n_ref": b["n"]}
    for key in ["approval", "tpr", "fpr", "accuracy"]:
        out[f"{key}_other"] = a[key]
        out[f"{key}_ref"] = b[key]
        out[f"{key}_gap"] = a[key] - b[key]
    out["impact_ratio"] = a["approval"] / b["approval"] if b["approval"] > 0 else np.nan
    return out


def pick_cutoffs(scores, y_true, group, levels, base=0.5):
    """One approval cutoff per group.

    The cutoff for each group is the one whose true positive rate on this data
    comes closest to the overall true positive rate at the base cutoff.
    """
    scores, y_true, group = map(np.asarray, (scores, y_true, group))
    target = (scores[y_true == 1] >= base).mean()
    grid = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))  # follows the score range

    cuts = {}
    for g in levels:
        good = (group == g) & (y_true == 1)
        if not good.any():
            cuts[g] = base
            continue
        tpr = np.array([(scores[good] >= t).mean() for t in grid])
        # smallest distance to the target TPR; ties go to the cutoff nearest base
        best = np.lexsort((np.abs(grid - base), np.abs(tpr - target)))[0]
        cuts[g] = float(grid[best])
    return cuts


def apply_cutoffs(scores, group, cuts, default=0.5):
    """Groups without their own cutoff (for example "excluded") use the default."""
    cut = np.array([cuts.get(g, default) for g in group])
    return (np.asarray(scores) >= cut).astype(int)
