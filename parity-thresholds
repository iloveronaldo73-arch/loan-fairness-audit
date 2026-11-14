"""Three mitigation strategies and their fairness/accuracy trade-off."""
import numpy as np


def reweighing_weights(y, priv):
    """Kamiran & Calders reweighing: w(g, y) = P(g) P(y) / P(g, y).
    After weighting, outcome is statistically independent of group in the training data."""
    y, priv = np.asarray(y), np.asarray(priv, bool)
    w = np.ones(len(y))
    for g in (True, False):
        for label in (0, 1):
            m = (priv == g) & (y == label)
            if m.any():
                w[m] = (priv == g).mean() * (y == label).mean() / m.mean()
    return w


def parity_thresholds(oof, thr, priv_tr):
    """Per-group thresholds that give both groups the same approval rate."""
    target = (oof >= thr).mean()
    return {g: float(np.quantile(oof[priv_tr == g], 1 - target)) for g in (True, False)}


def equal_opportunity_thresholds(oof, ytr, thr, priv_tr):
    """Per-group thresholds that approve the same share of GOOD applicants in each group."""
    good = np.asarray(ytr) == 1
    target = (oof[good] >= thr).mean()
    return {g: float(np.quantile(oof[(priv_tr == g) & good], 1 - target)) for g in (True, False)}


def apply_thresholds(p, priv, thresholds):
    priv = np.asarray(priv, bool)
    return np.where(priv, p >= thresholds[True], p >= thresholds[False])
