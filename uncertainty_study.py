#!/usr/bin/env python3
"""Uncertainty and economic-factor study for the loan fairness audit.

Part 1  Accuracy / fairness-gap curves over every decision cutoff, with four ways of
        expressing uncertainty side by side:
          * naive CI       mean over repeated splits +/- 1.96*s/sqrt(J)  (treats overlapping splits as independent: too narrow)
          * corrected CI   Nadeau & Bengio (2003) correction for overlapping splits
          * bootstrap CI   resample applicants
          * permutation    what a model with NO real signal / NO group effect would produce (null band + p-value)
Part 2  Economic factors: group permutation importance, leave-one-group-out, greedy forward
        addition, and a test of whether engineered economic ratios really improve accuracy.

Usage:
    python uncertainty_study.py --dataset german            # uses data/credit_g.csv, or downloads from OpenML
    python uncertainty_study.py --dataset taiwan            # UCI id 350 via ucimlrepo, or --csv UCI_Credit_Card.csv
    python uncertainty_study.py --dataset synthetic --quick # fake data, only tests the code

This project is a prototype and is not financial advice.
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTS = np.linspace(0.05, 0.95, 91)

# ------------------------------------------------------------------ datasets
GERMAN_GROUPS = {
    "liquidity_savings": ["checking_status", "savings_status"],
    "loan_size_and_term": ["credit_amount", "duration", "installment_commitment"],
    "employment_stability": ["employment", "job"],
    "assets_and_housing": ["property_magnitude", "housing"],
    "credit_history": ["credit_history", "other_payment_plans", "other_parties", "existing_credits"],
    "loan_purpose": ["purpose"],
    "household_context": ["residence_since", "num_dependents", "own_telephone", "foreign_worker"],
    "engineered_economic": ["monthly_payment", "log_credit_amount"],
}
TAIWAN_NAMES = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
                 "PAY_5", "PAY_6"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)])
TAIWAN_GROUPS = {
    "credit_limit": ["LIMIT_BAL"],
    "repayment_history": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    "bill_balances": [f"BILL_AMT{i}" for i in range(1, 7)],
    "payments_made": [f"PAY_AMT{i}" for i in range(1, 7)],
    "education_marital": ["EDUCATION", "MARRIAGE"],
    "engineered_economic": ["utilization", "payment_ratio", "max_delay"],
}


@dataclass
class DS:
    name: str
    X: pd.DataFrame            # non-protected features only (what the blind model sees)
    y: np.ndarray              # 1 = good outcome (repaid)
    age: np.ndarray
    male: np.ndarray           # True = privileged for sex
    groups: dict               # economic factor groups present in X
    cost_bad: float
    note: str


def _present(groups, X):
    return {g: [c for c in cols if c in X.columns] for g, cols in groups.items()
            if any(c in X.columns for c in cols)}


def load_german(path):
    if Path(path).exists():
        df = pd.read_csv(path)
    else:
        from sklearn.datasets import fetch_openml
        df = fetch_openml("credit-g", version=1, as_frame=True).frame
    ycol = "class" if "class" in df.columns else df.columns[-1]
    y = df[ycol].astype(str).str.lower().isin(["good", "1"]).astype(int).values
    X = df.drop(columns=[ycol]).copy()
    for c in X.columns:
        if str(X[c].dtype) in ("category", "object", "str") or X[c].dtype == object:
            X[c] = X[c].astype(str)
    male = ~X["personal_status"].str.lower().str.startswith("female").values
    age = X["age"].astype(float).values
    if {"credit_amount", "duration"} <= set(X.columns):
        X["monthly_payment"] = X["credit_amount"] / X["duration"].clip(lower=1)
        X["log_credit_amount"] = np.log1p(X["credit_amount"])
    X = X.drop(columns=["age", "personal_status"])
    return DS("german", X, y, age, male, _present(GERMAN_GROUPS, X), 5.0,
              "UCI Statlog German Credit (1,000 applications); cost rule 5:1 from the dataset documentation")


def load_taiwan(csv):
    if csv:
        df = pd.read_csv(csv)
    else:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=350)
        df = pd.concat([d.data.features, d.data.targets], axis=1)
    if "X1" in df.columns:
        df = df.rename(columns=dict(zip([f"X{i}" for i in range(1, 24)], TAIWAN_NAMES)))
    tcol = next(c for c in df.columns if c == "Y" or c.lower().replace(".", "_").startswith("default"))
    y = (1 - df[tcol].astype(int)).values
    X = df[TAIWAN_NAMES].copy()
    bills = X[[f"BILL_AMT{i}" for i in range(1, 7)]]
    pays = X[[f"PAY_AMT{i}" for i in range(1, 7)]]
    X["utilization"] = bills.mean(axis=1) / X["LIMIT_BAL"].clip(lower=1)
    X["payment_ratio"] = (pays.sum(axis=1) / bills.sum(axis=1).clip(lower=1)).clip(0, 5)
    X["max_delay"] = X[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].max(axis=1)
    X["EDUCATION"], X["MARRIAGE"] = X["EDUCATION"].astype(str), X["MARRIAGE"].astype(str)
    male, age = (X["SEX"] == 1).values, X["AGE"].astype(float).values
    X = X.drop(columns=["SEX", "AGE"])
    return DS("taiwan", X, y, age, male, _present(TAIWAN_GROUPS, X), 1.0,
              "UCI Default of Credit Card Clients (30,000 clients, Taiwan 2005); no published cost rule, so 1:1 is assumed")


def synthetic(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(35, 11, n), 19, 75).round()
    pick = lambda opts, p=None: rng.choice(opts, n, p=p)
    sav = pick(["<100", "100<=X<500", "500<=X<1000", ">=1000", "unknown"])
    chk = pick(["<0", "0<=X<200", ">=200", "no checking"], [.27, .27, .06, .40])
    df = pd.DataFrame({
        "checking_status": chk, "duration": rng.lognormal(3, .5, n).round(),
        "credit_history": pick(["critical", "existing paid", "delayed", "all paid"]),
        "purpose": pick(["radio/tv", "car", "furniture", "education", "business"]),
        "credit_amount": rng.lognormal(7.8, .7, n).round(), "savings_status": sav,
        "employment": pick(["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]),
        "installment_commitment": rng.integers(1, 5, n),
        "personal_status": pick(["male single", "female div/dep/mar", "male mar/wid", "male div/sep"],
                                [.55, .31, .09, .05]),
        "other_parties": pick(["none", "guarantor", "co applicant"]),
        "residence_since": rng.integers(1, 5, n), "property_magnitude": pick(["real estate", "car", "none"]),
        "age": age, "other_payment_plans": pick(["none", "bank", "stores"]),
        "housing": np.where(rng.random(n) < np.where(age < 25, .6, .15), "rent", "own"),
        "existing_credits": rng.integers(1, 4, n), "job": pick(["skilled", "unskilled", "management"]),
        "num_dependents": rng.integers(1, 3, n), "own_telephone": pick(["yes", "none"]),
        "foreign_worker": pick(["yes", "no"], [.96, .04])})
    z = (1.6 - .04 * df.duration - .00012 * df.credit_amount + .03 * (age - 35)
         + 0.9 * np.isin(sav, [">=1000", "500<=X<1000"]) + 0.9 * (chk == ">=200") + rng.normal(0, .8, n))
    df["class"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "good", "bad")
    path = "/tmp/_synthetic_credit.csv"
    df.to_csv(path, index=False)
    return load_german(path)


# ------------------------------------------------------------------- models
def build_pipe(cols, X):
    num = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in cols if c not in num]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, C=0.5))])


def oof(X, y, cols, seed):
    """One out-of-fold probability of 'good' for every applicant (5-fold)."""
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = build_pipe(cols, X).fit(X.iloc[tr][cols], y[tr])
        p[va] = m.predict_proba(X.iloc[va][cols])[:, 1]
    return p


def cv_probs(X, y, cols, seed, repeats):
    return np.mean([oof(X, y, cols, seed + 100 * r) for r in range(repeats)], axis=0)


# ------------------------------------------------------------------ curves
def curves(p, y, pa, ps):
    """All metrics at every cutoff. y: 1 = good; pa/ps: privileged masks (age 25+, male)."""
    A = p[:, None] >= CUTS[None, :]
    good = (y == 1)
    out = {"accuracy": (A == good[:, None]).mean(0)}
    out["accuracy_vs_baseline"] = out["accuracy"] - good.mean()
    for tag, m in (("age", pa), ("sex", ps)):
        out[f"gap_{tag}"] = A[m].mean(0) - A[~m].mean(0)
        out[f"tpr_gap_{tag}"] = A[m & good].mean(0) - A[~m & good].mean(0)
    return out


def op_cutoff(p, y, cost_bad):
    A = p[:, None] >= CUTS[None, :]
    cost = (cost_bad * (A & (y[:, None] == 0)).sum(0) + (~A & (y[:, None] == 1)).sum(0)) / len(y)
    return int(np.argmin(cost))


def stack(list_of_dicts):
    return {k: np.array([d[k] for d in list_of_dicts]) for k in list_of_dicts[0]}


def split_curves(ds, cols, pa, ps, J, seed):
    out, n = [], len(ds.y)
    strat = ds.y * 2 + (~pa)
    for j in range(J):
        tr, te = train_test_split(np.arange(n), test_size=0.2, stratify=strat, random_state=seed + j)
        m = build_pipe(cols, ds.X).fit(ds.X.iloc[tr][cols], ds.y[tr])
        out.append(curves(m.predict_proba(ds.X.iloc[te][cols])[:, 1], ds.y[te], pa[te], ps[te]))
    return stack(out), len(te) / len(tr)


def boot_curves(p, y, pa, ps, B, rng):
    n, out = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        out.append(curves(p[i], y[i], pa[i], ps[i]))
    return stack(out)


def perm_curves(ds, cols, p, pa, ps, n_perm, n_gap, seed, rng):
    """Null distributions. Accuracy: refit on shuffled outcomes. Gaps: shuffle group labels.
    TPR gaps: shuffle group labels among good applicants only."""
    y, null = ds.y, {}
    acc = []
    for k in range(n_perm):
        yp = rng.permutation(y)
        pk = oof(ds.X, yp, cols, seed)
        A = pk[:, None] >= CUTS[None, :]
        acc.append((A == (yp == 1)[:, None]).mean(0))
    null["accuracy"] = np.array(acc)
    null["accuracy_vs_baseline"] = null["accuracy"] - (y == 1).mean()
    A = p[:, None] >= CUTS[None, :]
    good = y == 1
    gi = np.where(good)[0]
    for tag, m in (("age", pa), ("sex", ps)):
        g, t = [], []
        for _ in range(n_gap):
            mp = rng.permutation(m)
            g.append(A[mp].mean(0) - A[~mp].mean(0))
            mg = m.copy(); mg[gi] = rng.permutation(m[gi])
            t.append(A[mg & good].mean(0) - A[~mg & good].mean(0))
        null[f"gap_{tag}"], null[f"tpr_gap_{tag}"] = np.array(g), np.array(t)
    return null


def intervals(obs, S, ratio, bt, null):
    J = S[next(iter(S))].shape[0]
    tcrit, res = t_dist.ppf(.975, J - 1), {}
    for k in obs:
        m, s = S[k].mean(0), S[k].std(0, ddof=1)
        nb_half = tcrit * s * np.sqrt(1 / J + ratio)
        res[k] = {"obs": obs[k], "spread": np.percentile(S[k], [2.5, 97.5], axis=0),
                  "naive": np.array([m - 1.96 * s / np.sqrt(J), m + 1.96 * s / np.sqrt(J)]),
                  "corrected": np.array([m - nb_half, m + nb_half]),
                  "boot": np.percentile(bt[k], [2.5, 97.5], axis=0),
                  "null": np.percentile(null[k], [2.5, 97.5], axis=0)}
    return res


def perm_p(obs_val, null_vals, k):
    if k.startswith("accuracy"):
        return (1 + (null_vals >= obs_val).sum()) / (1 + len(null_vals))
    return (1 + (np.abs(null_vals) >= abs(obs_val)).sum()) / (1 + len(null_vals))


def plot_curves(R, ci, base, path, title):
    panels = [("accuracy", "Accuracy vs cutoff"), ("gap_age", "Approval-rate gap, age 25+ minus under 25"),
              ("gap_sex", "Approval-rate gap, male minus female")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (k, ttl) in zip(axes, panels):
        r = R[k]
        ax.fill_between(CUTS, *r["null"], color="0.8", label="permutation null (95%)")
        ax.fill_between(CUTS, *r["boot"], color="C0", alpha=.30, label="bootstrap 95% CI")
        ax.fill_between(CUTS, *r["corrected"], color="C1", alpha=.45, label="corrected CI (Nadeau-Bengio)")
        ax.fill_between(CUTS, *r["naive"], color="C3", alpha=.9, label="naive CI (understated)")
        ax.plot(CUTS, r["obs"], "k", lw=1.2, label="observed (out-of-fold)")
        if k == "accuracy":
            ax.axhline(base, c="g", ls="--", lw=1, label="approve-everyone baseline")
        else:
            ax.axhline(0, c="k", lw=.6)
        ax.axvline(CUTS[ci], c="purple", ls=":", lw=1, label="operating cutoff")
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("approval cutoff (approve if P(good) >= cutoff)")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------- economic factors
def paired_boot(p_new, p_old, y, pa, cut, B, rng):
    def m(p, i):
        a = p[i] >= cut
        return (roc_auc_score(y[i], p[i]), (a == (y[i] == 1)).mean(), a[pa[i]].mean() - a[~pa[i]].mean())
    n = len(y); allidx = np.arange(n)
    est = np.array(m(p_new, allidx)) - np.array(m(p_old, allidx))
    bs = np.array([np.array(m(p_new, i)) - np.array(m(p_old, i))
                   for i in (rng.integers(0, n, n) for _ in range(B))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {k: (est[j], lo[j], hi[j]) for j, k in enumerate(["d_auc", "d_acc", "d_gap_age"])}


def verdict(lo, hi):
    return "detectable" if (lo > 0 or hi < 0) else "within noise"


def group_perm_importance(ds, cols, seed, rng, n_rep=5):
    rows = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(ds.X, ds.y):
        m = build_pipe(cols, ds.X).fit(ds.X.iloc[tr][cols], ds.y[tr])
        Xv, yv = ds.X.iloc[va][cols].copy(), ds.y[va]
        base = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        for g, gc in ds.groups.items():
            for _ in range(n_rep):
                Xp, perm = Xv.copy(), rng.permutation(len(Xv))
                for c in gc:
                    Xp[c] = Xv[c].values[perm]
                rows.append((g, base - roc_auc_score(yv, m.predict_proba(Xp[cols])[:, 1])))
    return (pd.DataFrame(rows, columns=["group", "auc_drop"]).groupby("group").auc_drop
            .agg(mean_drop="mean", std_drop="std").reset_index().sort_values("mean_drop", ascending=False))


def forward_addition(ds, seed, repeats, B, rng):
    remaining, cols, rows = dict(ds.groups), [], []
    while remaining:
        best = None
        for g, c in remaining.items():
            p = cv_probs(ds.X, ds.y, cols + c, seed, repeats)
            a = roc_auc_score(ds.y, p)
            if best is None or a > best[0]:
                best = (a, g, c, p)
        a, g, c, p = best
        cols, _ = cols + c, remaining.pop(g)
        n = len(ds.y)
        bs = [roc_auc_score(ds.y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
        rows.append({"step": len(rows) + 1, "added": g, "auc": a, "auc_lo": np.percentile(bs, 2.5),
                     "auc_hi": np.percentile(bs, 97.5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- main
def fmt(v, pct=True):
    return f"{v * 100:.1f}" if pct else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["german", "taiwan", "synthetic"], default="german")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--outdir", default="results/uncertainty")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--age-cutoff", type=float, default=25)
    ap.add_argument("--cost-bad", type=float, default=None)
    ap.add_argument("--cutoff", default="0.5",
                    help="operating cutoff: a number (default 0.5, matches a standard accuracy-style audit) or 'cost' to minimise cost")
    ap.add_argument("--quick", action="store_true", help="tiny settings, for testing")
    ap.add_argument("--splits", type=int); ap.add_argument("--boot", type=int)
    ap.add_argument("--perm", type=int); ap.add_argument("--perm-gap", type=int)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    ds = (load_german(a.csv or "data/credit_g.csv") if a.dataset == "german"
          else load_taiwan(a.csv) if a.dataset == "taiwan" else synthetic())
    big = len(ds.y) > 5000
    d = {"splits": 50, "boot": 1000, "perm": 200, "perm_gap": 1000}
    if big:
        d = {"splits": 20, "boot": 200, "perm": 30, "perm_gap": 200}
    if a.quick:
        d = {"splits": 8, "boot": 60, "perm": 10, "perm_gap": 60}
    J, B, NP, NG = (a.splits or d["splits"], a.boot or d["boot"], a.perm or d["perm"], a.perm_gap or d["perm_gap"])
    cost_bad = a.cost_bad or ds.cost_bad
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.outdir, exist_ok=True)

    pa, ps = ds.age >= a.age_cutoff, ds.male
    cols = [c for g in ds.groups.values() for c in g]
    raw_cols = [c for g, cs in ds.groups.items() if g != "engineered_economic" for c in cs]
    print(f"[{ds.name}] n={len(ds.y)}, good={ds.y.mean():.1%}, under-cutoff={(~pa).sum()}, women={(~ps).sum()}")

    # ---- Part 1: curves
    print("part 1: split curves, bootstrap, permutation tests ...")
    p = oof(ds.X, ds.y, cols, a.seed)
    use_cost = a.cutoff == "cost"
    ci = op_cutoff(p, ds.y, cost_bad) if use_cost else int(np.argmin(np.abs(CUTS - float(a.cutoff))))
    obs = curves(p, ds.y, pa, ps)
    S, ratio = split_curves(ds, cols, pa, ps, J, a.seed)
    bt = boot_curves(p, ds.y, pa, ps, B, rng)
    null = perm_curves(ds, cols, p, pa, ps, NP, NG, a.seed, rng)
    R = intervals(obs, S, ratio, bt, null)
    plot_curves(R, ci, ds.y.mean(), f"{a.outdir}/uncertainty_curves.png",
                f"{ds.name}: honest vs understated uncertainty (n={len(ds.y)}, {J} splits, {B} bootstraps, {NP} label permutations)")

    rows = []
    for k, r in R.items():
        rows.append({"metric": k, "estimate": r["obs"][ci],
                     "split_spread_lo": r["spread"][0][ci], "split_spread_hi": r["spread"][1][ci],
                     "naive_lo": r["naive"][0][ci], "naive_hi": r["naive"][1][ci],
                     "corrected_lo": r["corrected"][0][ci], "corrected_hi": r["corrected"][1][ci],
                     "boot_lo": r["boot"][0][ci], "boot_hi": r["boot"][1][ci],
                     "null_lo": r["null"][0][ci], "null_hi": r["null"][1][ci],
                     "perm_p": perm_p(r["obs"][ci], null[k][:, ci], k),
                     "boot_width_vs_naive": (r["boot"][1][ci] - r["boot"][0][ci]) /
                                            max(r["naive"][1][ci] - r["naive"][0][ci], 1e-12)})
    T1 = pd.DataFrame(rows); T1.to_csv(f"{a.outdir}/interval_comparison.csv", index=False)

    # ---- Part 2: economic factors
    print("part 2: economic factors ...")
    P_full = cv_probs(ds.X, ds.y, cols, a.seed, a.repeats)
    cut = CUTS[op_cutoff(P_full, ds.y, cost_bad)] if use_cost else CUTS[ci]
    BE = max(B // 3, 50)
    imp = group_perm_importance(ds, cols, a.seed, rng)
    logo = []
    for g, gc in ds.groups.items():
        keep = [c for c in cols if c not in gc]
        if not keep:
            continue
        r = paired_boot(cv_probs(ds.X, ds.y, keep, a.seed, a.repeats), P_full, ds.y, pa, cut, BE, rng)
        logo.append({"dropped_group": g, "columns": ", ".join(gc),
                     **{f"{m}": r[m][0] for m in r}, **{f"{m}_lo": r[m][1] for m in r},
                     **{f"{m}_hi": r[m][2] for m in r}, "auc_effect": verdict(r["d_auc"][1], r["d_auc"][2])})
    LOGO = pd.DataFrame(logo).sort_values("d_auc")
    P_raw = cv_probs(ds.X, ds.y, raw_cols, a.seed, a.repeats)
    eng = paired_boot(P_full, P_raw, ds.y, pa, cut, BE, rng)
    FWD = forward_addition(ds, a.seed, max(a.repeats - 1, 1), BE, rng)
    imp.to_csv(f"{a.outdir}/economic_permutation_importance.csv", index=False)
    LOGO.to_csv(f"{a.outdir}/economic_leave_one_group_out.csv", index=False)
    FWD.to_csv(f"{a.outdir}/economic_forward_addition.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].barh(imp.group, imp.mean_drop, xerr=imp.std_drop, color="C0"); ax[0, 0].invert_yaxis()
    ax[0, 0].set_title("(a) Permutation importance: AUC lost when a factor group is shuffled")
    L = LOGO.iloc[::-1]
    ax[0, 1].barh(L.dropped_group, L.d_auc, xerr=[L.d_auc - L.d_auc_lo, L.d_auc_hi - L.d_auc], color="C1")
    ax[0, 1].axvline(0, c="k", lw=.6); ax[0, 1].set_title("(b) Change in AUC if group is dropped (95% bootstrap CI)")
    ax[1, 0].barh(L.dropped_group, L.d_gap_age, xerr=[L.d_gap_age - L.d_gap_age_lo, L.d_gap_age_hi - L.d_gap_age], color="C2")
    ax[1, 0].axvline(0, c="k", lw=.6); ax[1, 0].set_title("(c) Change in age approval gap if group is dropped")
    ax[1, 1].plot(FWD.step, FWD.auc, "o-"); ax[1, 1].fill_between(FWD.step, FWD.auc_lo, FWD.auc_hi, alpha=.25)
    ax[1, 1].set_xticks(FWD.step); ax[1, 1].set_xticklabels(FWD.added, rotation=40, ha="right", fontsize=7)
    ax[1, 1].set_title("(d) Greedy forward addition of factor groups (AUC, 95% CI)")
    fig.suptitle(f"{ds.name}: which economic factors matter, and for accuracy or for the age gap?")
    fig.tight_layout(); fig.savefig(f"{a.outdir}/economic_factors.png", dpi=150); plt.close(fig)

    # ---- report
    md = [f"# Uncertainty and economic-factor study: {ds.name}\n",
          f"{ds.note}.\n\nn = {len(ds.y)}; good outcomes {ds.y.mean():.1%} (the approve-everyone baseline accuracy); "
          f"age cutoff {a.age_cutoff:g} ({(~pa).sum()} below it); {(~ps).sum()} women. "
          f"**Operating cutoff {CUTS[ci]:.2f}** "
          + (f"(minimises cost when approving a bad loan costs {cost_bad:g}x rejecting a good one). " if use_cost
             else "(fixed; rerun with `--cutoff cost` for the cost-minimising cutoff, which approves far fewer applicants "
                  "and can score below the approve-everyone baseline on accuracy). ")
          + "Model: blind logistic regression (no age or sex). All numbers are percentage points unless stated.\n"]
    if a.dataset == "synthetic":
        md.append("> **WARNING: synthetic data. Numbers are meaningless; this run only tests the code.**\n")
    md.append("\n## 1. Interval comparison at the operating cutoff\n\n"
              "| metric | estimate | spread across splits | naive CI | corrected CI | bootstrap CI | permutation null 95% | perm. p |\n|---|---|---|---|---|---|---|---|\n")
    for r in T1.itertuples():
        md.append(f"| {r.metric} | {fmt(r.estimate)} | {fmt(r.split_spread_lo)} to {fmt(r.split_spread_hi)} | "
                  f"{fmt(r.naive_lo)} to {fmt(r.naive_hi)} | {fmt(r.corrected_lo)} to {fmt(r.corrected_hi)} | "
                  f"{fmt(r.boot_lo)} to {fmt(r.boot_hi)} | {fmt(r.null_lo)} to {fmt(r.null_hi)} | {r.perm_p:.3f} |\n")
    md.append("\n**How to read it.**\n")
    for r in T1.itertuples():
        w = r.boot_width_vs_naive
        flip = ((r.naive_lo > 0 or r.naive_hi < 0) and not (r.boot_lo > 0 or r.boot_hi < 0)) \
            if not r.metric.startswith("accuracy") else False
        md.append(f"- `{r.metric}`: the bootstrap interval is {w:.1f}x as wide as the naive one"
                  + ("; the naive interval would have wrongly shown a detectable effect." if flip else ".") + "\n")
    md.append("\nThe naive and corrected CIs describe the *average over repeated splits*; the naive one treats the "
              "overlapping splits as independent, so it shrinks as you add splits even though no new applicants were seen. "
              "The permutation p-values ask a different question: could a model with no real signal (accuracy) or a group "
              "label unrelated to decisions (gaps) produce a result this large? A small p for a gap shows approval "
              "differs by group, **not** that the model is unfair: the gap may come from legitimate economic differences, "
              "which Part 2 examines. The TPR gaps (good applicants only) are the stronger fairness signal.\n")

    md.append("\n## 2. Economic factors\n\n### Permutation importance (AUC drop)\n\n" +
              "| group | mean drop | std |\n|---|---|---|\n" +
              "\n".join(f"| {r.group} | {r.mean_drop:.4f} | {r.std_drop:.4f} |" for r in imp.itertuples()) + "\n")
    md.append("\n### Leave-one-group-out (change when the group is removed; negative AUC/accuracy = the group helped)\n\n"
              "| dropped group | d AUC (95% CI) | d accuracy (pp) | d age gap (pp, 95% CI) | AUC effect |\n|---|---|---|---|---|\n" +
              "\n".join(f"| {r.dropped_group} | {r.d_auc:+.4f} ({r.d_auc_lo:+.4f}, {r.d_auc_hi:+.4f}) | {r.d_acc*100:+.1f} | "
                        f"{r.d_gap_age*100:+.1f} ({r.d_gap_age_lo*100:+.1f}, {r.d_gap_age_hi*100:+.1f}) | {r.auc_effect} |"
                        for r in LOGO.itertuples()) + "\n")
    e = eng
    md.append(f"\n### Do engineered economic ratios improve accuracy?\n\nAdding the engineered group to the raw columns changes "
              f"AUC by {e['d_auc'][0]:+.4f} (95% CI {e['d_auc'][1]:+.4f} to {e['d_auc'][2]:+.4f}), accuracy by "
              f"{e['d_acc'][0]*100:+.1f} pp ({e['d_acc'][1]*100:+.1f} to {e['d_acc'][2]*100:+.1f}) and the age gap by "
              f"{e['d_gap_age'][0]*100:+.1f} pp. AUC effect: **{verdict(e['d_auc'][1], e['d_auc'][2])}**; "
              f"accuracy effect: **{verdict(e['d_acc'][1], e['d_acc'][2])}**.\n")
    md.append("\n### Forward addition\n\n| step | group added | AUC (95% CI) |\n|---|---|---|\n" +
              "\n".join(f"| {r.step} | {r.added} | {r.auc:.3f} ({r.auc_lo:.3f}, {r.auc_hi:.3f}) |" for r in FWD.itertuples()) + "\n")
    md.append("\nIf neighbouring CIs overlap, the extra group did not demonstrably improve accuracy; adding factors "
              "does not automatically help on a dataset this small.\n\n"
              "## Limitations\n\n- Bootstrap CIs ignore uncertainty from model fitting; permutation tests assume exchangeability.\n"
              "- Repeated CV averages are not independent either; use CIs with the sample size in mind.\n"
              "- Association with economic factors is not causation. Age and sex gaps can reflect real economic differences.\n"
              "- This project is a prototype and is not financial advice.\n")
    Path(f"{a.outdir}/uncertainty_report.md").write_text("".join(md))
    print(f"Done. See {a.outdir}/uncertainty_report.md")


if __name__ == "__main__":
    main()
