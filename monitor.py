#!/usr/bin/env python3
"""Batch + global monitoring for a deployed loan model.

A one-time audit says nothing about next month. After a model goes live, every new batch of
applications is checked for (1) drift in who applies and how the model scores them,
(2) unfair treatment of age / sex groups, and (3) when outcomes arrive, whether the model's
predictions are still calibrated.

  BatchMonitor   scores ONE model on each incoming batch and raises alerts.
  GlobalMonitor  combines every batch and every model: health status, alert log,
                 pooled fairness audit with bootstrap CIs, first-half vs second-half
                 trend, Simpson's-paradox check and a dashboard.

Checks per batch
  * PSI (population stability index) on the score and every input column.
    Rule of thumb from credit-risk practice: <0.10 stable, 0.10-0.25 moderate, >0.25 major.
  * Approval-rate shift (z-test against the reference approval rate).
  * Disparate impact by age and sex, ONLY alarmed when a permutation test says the gap is
    unlikely to be chance and both groups have enough people (small batches are noisy).
  * CUSUM (Page, 1954) on the approval gap: adds up small shifts that no single batch
    can prove, so slow drift is still caught.
  * When outcomes are supplied: accuracy, AUC, calibration drift, good-applicant approval gap.

Usage
  python monitor.py simulate --data data/credit_g.csv --kind fairness   # inject a drift, watch it get caught
  python monitor.py simulate --kind none                                # control run: should stay quiet
  python monitor.py evaluate --runs 20                                  # the monitor's own false-alarm rate and delay
  python monitor.py batch --data data/credit_g.csv --batches "new/*.csv"  # monitor real incoming files

Alerts mean "investigate", not "the model is biased": a growing gap can come from a real change
in the applicants' economics. This project is a prototype and is not financial advice.
"""
import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROTECTED = ["age", "personal_status"]
EPS = 1e-4
SEVERITY = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Thresholds:
    psi_warn: float = 0.10
    psi_crit: float = 0.25
    di_floor: float = 0.80        # four-fifths rule
    alpha: float = 0.05           # permutation-test level for fairness alerts
    min_group: int = 30           # smaller groups are reported but never alarmed
    z_crit: float = 3.0           # approval-rate shift
    cusum_k: float = 0.5          # CUSUM slack, in standard errors
    cusum_h: float = 5.0          # CUSUM alarm level, in standard errors
    calib_warn: float = 0.10      # |mean predicted - observed| good-rate
    n_perm: int = 500
    age_cut: float = 25


# ------------------------------------------------------------------- helpers
def psi(expected, actual):
    """Population stability index between two binned distributions."""
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def group_masks(Xf, age_cut=25):
    """True = privileged group (age >= cutoff, male)."""
    age = Xf["age"].astype(float).values
    male = ~Xf["personal_status"].astype(str).str.lower().str.startswith("female").values
    return {"age": age >= age_cut, "sex": male}


def _frac(x, edges):
    return np.bincount(np.digitize(x, edges), minlength=len(edges) + 1) / max(len(x), 1)


def build_pipe(cols, X):
    num = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in cols if c not in num]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, C=0.5))])


class Reference:
    """What 'normal' looks like: score and feature distributions, approval rate, gaps."""

    def __init__(self, pipe, cols, Xref, cutoff, th):
        p = pipe.predict_proba(Xref[cols])[:, 1]
        a = p >= cutoff
        inner = np.linspace(0, 1, 11)[1:-1]
        self.cutoff, self.n, self.approval = cutoff, len(p), a.mean()
        self.score_edges = np.unique(np.quantile(p, inner))
        self.score_frac = _frac(p, self.score_edges)
        self.feat = {}
        for c in cols:
            if pd.api.types.is_numeric_dtype(Xref[c]):
                e = np.unique(np.quantile(Xref[c], inner))
                self.feat[c] = ("num", e, _frac(Xref[c].values, e))
            else:
                f = Xref[c].astype(str).value_counts(normalize=True)
                self.feat[c] = ("cat", list(f.index), np.append(f.values, 0.0))
        self.gap = {k: a[m].mean() - a[~m].mean() for k, m in group_masks(Xref, th.age_cut).items()}

    def feature_psi(self, X):
        out = {}
        for c, (kind, edges, frac) in self.feat.items():
            if kind == "num":
                out[c] = psi(frac, _frac(X[c].values, edges))
            else:
                s = X[c].astype(str).where(X[c].astype(str).isin(edges), "__other__")
                act = np.array([(s == k).mean() for k in edges] + [(s == "__other__").mean()])
                out[c] = psi(frac, act)
        return out


# -------------------------------------------------------------- batch monitor
class BatchMonitor:
    def __init__(self, name, pipe, cols, ref, th=None, seed=0):
        self.name, self.pipe, self.cols, self.ref = name, pipe, cols, ref
        self.th, self.rng = th or Thresholds(), np.random.default_rng(seed)
        self.records, self.alerts = [], []
        self.cusum = {"age": [0.0, 0.0], "sex": [0.0, 0.0]}
        self.pool_a, self.pool_g = [], {"age": [], "sex": []}
        self.recent = {}          # breach history for the 2-of-3 persistence rule

    def _persistent(self, key, breached, immediate=False):
        """True if `immediate`, or the check breached in at least 2 of the last 3 batches."""
        h = self.recent.setdefault(key, [])
        h.append(bool(breached))
        del h[:-3]
        return bool(immediate or sum(h) >= 2)

    def _alert(self, batch, sev, typ, msg):
        self.alerts.append({"model": self.name, "batch_index": len(self.records), "batch": batch,
                            "severity": sev, "type": typ, "message": msg})

    def process(self, batch_id, Xf, y=None):
        th, ref = self.th, self.ref
        p = self.pipe.predict_proba(Xf[self.cols])[:, 1]
        a, n = p >= ref.cutoff, len(p)
        masks = group_masks(Xf, th.age_cut)
        rec = {"model": self.name, "batch": batch_id, "batch_index": len(self.records) + 1, "n": n,
               "approval_rate": a.mean(), "mean_score": p.mean()}
        # approval-rate shift
        z = (a.mean() - ref.approval) / np.sqrt(max(ref.approval * (1 - ref.approval), 1e-9) / n)
        rec["approval_z"] = z
        breach = abs(z) > th.z_crit
        if self._persistent("approval", breach, immediate=abs(z) > th.z_crit * 1.67):
            self._alert(batch_id, "critical", "approval_shift",
                        f"approval rate {a.mean():.1%} vs reference {ref.approval:.1%} (z={z:+.1f}), persistent")
        elif breach:
            self._alert(batch_id, "warning", "approval_shift_single",
                        f"approval rate {a.mean():.1%} vs reference {ref.approval:.1%} (z={z:+.1f}), one batch so far")
        # drift
        rec["score_psi"] = psi(ref.score_frac, _frac(p, ref.score_edges))
        fp = ref.feature_psi(Xf[self.cols])
        worst = max(fp, key=fp.get)
        rec["max_feature_psi"], rec["max_feature"] = fp[worst], worst
        for label, v, what in (("score_drift", rec["score_psi"], "score distribution"),
                               ("feature_drift", fp[worst], f"input `{worst}`")):
            if v >= th.psi_crit:
                self._alert(batch_id, "critical", label, f"{what} PSI {v:.2f} (major shift)")
            elif v >= th.psi_warn:
                self._alert(batch_id, "warning", label, f"{what} PSI {v:.2f} (moderate shift)")
        # fairness per attribute
        good = None if y is None else (np.asarray(y) == 1)
        for attr, m in masks.items():
            nP, nU = int(m.sum()), int((~m).sum())
            under = min(nP, nU) < th.min_group
            ap = a[m].mean() if nP else np.nan
            au = a[~m].mean() if nU else np.nan
            gap = ap - au
            di = au / ap if ap and ap > 0 else np.nan
            pval = np.nan
            if not under:
                null = [a[mp].mean() - a[~mp].mean() for mp in (self.rng.permutation(m) for _ in range(th.n_perm))]
                pval = (1 + (np.abs(null) >= abs(gap)).sum()) / (1 + th.n_perm)
                ph = a.mean()
                se = np.sqrt(max(ph * (1 - ph), 1e-9) * (1 / nP + 1 / nU))
                zg = (gap - ref.gap[attr]) / se
                s = self.cusum[attr]
                s[0], s[1] = max(0.0, s[0] + zg - th.cusum_k), max(0.0, s[1] - zg - th.cusum_k)
            else:
                zg = 0.0
            rec.update({f"{attr}_n_priv": nP, f"{attr}_n_unpriv": nU, f"{attr}_approval_priv": ap,
                        f"{attr}_approval_unpriv": au, f"{attr}_gap": gap, f"{attr}_di": di,
                        f"{attr}_perm_p": pval, f"{attr}_gap_z": zg,
                        f"{attr}_cusum_hi": self.cusum[attr][0], f"{attr}_cusum_lo": self.cusum[attr][1],
                        f"{attr}_underpowered": under})
            if good is not None:
                try:
                    rec[f"{attr}_good_approved_gap"] = a[m & good].mean() - a[~m & good].mean()
                except Exception:
                    rec[f"{attr}_good_approved_gap"] = np.nan
            if under:
                self._alert(batch_id, "info", "underpowered",
                            f"{attr}: smaller group has {min(nP, nU)} people (< {th.min_group}); fairness not tested")
            else:
                breach = di < th.di_floor and pval < th.alpha
                persistent = self._persistent(f"fair_{attr}", breach)
                if breach and persistent:
                    self._alert(batch_id, "critical", "fairness",
                                f"{attr}: disparate impact {di:.2f} < {th.di_floor:.2f}, gap {gap:+.1%}, p={pval:.3f}, in 2 of last 3 batches")
                elif breach:
                    self._alert(batch_id, "warning", "fairness_single",
                                f"{attr}: disparate impact {di:.2f} < {th.di_floor:.2f}, p={pval:.3f}, first breach")
                elif di < th.di_floor:
                    self._alert(batch_id, "info", "fairness_unproven",
                                f"{attr}: DI {di:.2f} below {th.di_floor:.2f} but p={pval:.2f}, could be chance in a batch this size")
            for j, direction in ((0, "widening"), (1, "narrowing")):
                if self.cusum[attr][j] > th.cusum_h:
                    self._alert(batch_id, "critical", "cusum",
                                f"{attr}: cumulative evidence of a {direction} approval gap (CUSUM {self.cusum[attr][j]:.1f} > {th.cusum_h:g})")
                    self.cusum[attr][j] = 0.0
        # outcomes (optional, usually arrive late)
        if good is not None:
            rec["accuracy"] = float((a == good).mean())
            rec["calibration_gap"] = float(p.mean() - good.mean())
            rec["auc"] = float(roc_auc_score(good, p)) if 0 < good.sum() < n else np.nan
            if abs(rec["calibration_gap"]) > th.calib_warn:
                self._alert(batch_id, "warning", "calibration",
                            f"mean predicted good-rate {p.mean():.1%} vs observed {good.mean():.1%}")
        self.records.append(rec)
        self.pool_a.append(a)
        for k in masks:
            self.pool_g[k].append(masks[k])
        return rec


# ------------------------------------------------------------- global monitor
class GlobalMonitor:
    def __init__(self):
        self.monitors = {}

    def add(self, mon):
        self.monitors[mon.name] = mon

    def table(self):
        return pd.DataFrame([r for m in self.monitors.values() for r in m.records])

    def alert_log(self):
        return pd.DataFrame([a for m in self.monitors.values() for a in m.alerts])

    def health(self, last=3):
        """RED = critical alert in the last `last` batches, AMBER = warning, else GREEN."""
        out = {}
        for name, m in self.monitors.items():
            cur = len(m.records)
            recent = [a for a in m.alerts if a["batch_index"] > cur - last]
            top = max([SEVERITY[a["severity"]] for a in recent], default=0)
            out[name] = {"status": ["GREEN", "AMBER", "RED"][top], "batches": cur,
                         "alerts_total": len(m.alerts),
                         "critical_total": sum(a["severity"] == "critical" for a in m.alerts)}
        return out

    def pooled_audit(self, B=1000, seed=0):
        """Pool every decision ever made. Reports the overall gap with a bootstrap CI,
        first-half vs second-half trend, and a Simpson's-paradox check against per-batch gaps."""
        rng, rows = np.random.default_rng(seed), []

        def gap(a, g):
            return a[g].mean() - a[~g].mean()

        for name, m in self.monitors.items():
            a = np.concatenate(m.pool_a)
            for attr in ("age", "sex"):
                g = np.concatenate(m.pool_g[attr])
                bs = []
                for _ in range(B):
                    i = rng.integers(0, len(a), len(a))
                    bs.append(gap(a[i], g[i]))
                half = len(m.pool_a) // 2
                h1a, h2a = np.concatenate(m.pool_a[:half]), np.concatenate(m.pool_a[half:])
                h1g, h2g = np.concatenate(m.pool_g[attr][:half]), np.concatenate(m.pool_g[attr][half:])
                diffs = []
                for _ in range(B):
                    i, j = rng.integers(0, len(h1a), len(h1a)), rng.integers(0, len(h2a), len(h2a))
                    diffs.append(gap(h2a[j], h2g[j]) - gap(h1a[i], h1g[i]))
                bg = pd.DataFrame(m.records)[f"{attr}_gap"].dropna()
                pooled = gap(a, g)
                rows.append({"model": name, "attribute": attr, "pooled_gap": pooled,
                             "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                             "first_half_gap": gap(h1a, h1g), "second_half_gap": gap(h2a, h2g),
                             "trend_ci_lo": np.percentile(diffs, 2.5), "trend_ci_hi": np.percentile(diffs, 97.5),
                             "mean_batch_gap": bg.mean(),
                             "simpson_flag": bool(np.sign(pooled) != np.sign(bg.mean()) and abs(pooled) > 1e-9)})
        return pd.DataFrame(rows)

    def dashboard(self, path, drift_start=None):
        T = self.table()
        fig, ax = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
        for name, d in T.groupby("model"):
            x = d.batch_index
            ax[0, 0].plot(x, d.approval_rate, "o-", label=name)
            ax[0, 1].plot(x, d.score_psi, "o-", label=f"{name} score")
            ax[1, 0].plot(x, d.age_di, "o-", label=name)
            sig = d.age_perm_p < 0.05
            ax[1, 0].plot(x[sig], d.age_di[sig], "rx", ms=9)
            ax[1, 1].plot(x, d.age_cusum_hi, "o-", label=f"{name} widening")
            ax[1, 1].plot(x, d.age_cusum_lo, "s--", alpha=.5, label=f"{name} narrowing")
            ax[0, 2].plot(x, d.sex_di, "o-", label=name)
            sig = d.sex_perm_p < 0.05
            ax[0, 2].plot(x[sig], d.sex_di[sig], "rx", ms=9)
        first = next(iter(self.monitors.values()))
        ax[0, 0].axhline(first.ref.approval, c="k", ls=":", lw=1, label="reference")
        d0 = T[T.model == first.name]
        ax[0, 1].plot(d0.batch_index, d0.max_feature_psi, "k--", label="worst input column")
        th = first.th
        ax[0, 1].axhline(th.psi_warn, c="orange", ls=":"); ax[0, 1].axhline(th.psi_crit, c="r", ls=":")
        for k in (ax[1, 0], ax[0, 2]):
            k.axhline(th.di_floor, c="r", ls=":", label="0.8 rule")
        ax[1, 1].axhline(th.cusum_h, c="r", ls=":", label="alarm level")
        titles = ["Approval rate per batch", "Drift (PSI): orange .10, red .25", "Sex disparate impact (x = p<.05)",
                  "Age disparate impact (x = p<.05)", "CUSUM on age approval gap", "Global health"]
        for a_, t in zip(ax.ravel(), titles):
            a_.set_title(t, fontsize=10); a_.set_xlabel("batch")
            if a_ is not ax[1, 2]:
                if drift_start:
                    a_.axvline(drift_start + .5, c="purple", ls="--", lw=1)
                a_.legend(fontsize=6)
        ax[1, 2].axis("off")
        h = self.health()
        txt = "\n".join(f"{n}: {v['status']}   ({v['critical_total']} critical / {v['alerts_total']} alerts "
                        f"over {v['batches']} batches)" for n, v in h.items())
        if drift_start:
            txt += f"\n\n(purple line: simulated drift starts after batch {drift_start})"
        ax[1, 2].text(0, .95, txt, fontsize=9, va="top", family="monospace", transform=ax[1, 2].transAxes)
        fig.suptitle("Loan model monitor: batch and global view", fontsize=12)
        fig.savefig(path, dpi=140); plt.close(fig)

    def report(self, path, first_alert_note=""):
        h, A, P = self.health(), self.alert_log(), self.pooled_audit()
        pct = lambda v: f"{v * 100:+.1f}"
        md = ["# Monitoring report\n", first_alert_note,
              "\n## Global health\n\n| model | status | batches | alerts | critical |\n|---|---|---|---|---|\n"]
        md += [f"| {n} | **{v['status']}** | {v['batches']} | {v['alerts_total']} | {v['critical_total']} |\n"
               for n, v in h.items()]
        md.append("\nGREEN/AMBER/RED reflects the last 3 batches. RED = a critical alert; AMBER = a warning.\n")
        md.append("\n## Pooled fairness audit across all batches (gaps in percentage points, priv minus unpriv)\n\n"
                  "| model | attribute | pooled gap (95% CI) | first half | second half | change (95% CI) | Simpson flag |\n|---|---|---|---|---|---|---|\n")
        md += [f"| {r.model} | {r.attribute} | {pct(r.pooled_gap)} ({pct(r.ci_lo)}, {pct(r.ci_hi)}) | "
               f"{pct(r.first_half_gap)} | {pct(r.second_half_gap)} | {pct(r.second_half_gap - r.first_half_gap)} "
               f"({pct(r.trend_ci_lo)}, {pct(r.trend_ci_hi)}) | {'YES' if r.simpson_flag else 'no'} |\n" for r in P.itertuples()]
        md.append("\n*Simpson flag: the pooled gap has the opposite sign to the average per-batch gap, so the "
                  "headline number hides what happens inside batches.*\n\n## Alert log\n\n")
        if len(A):
            crit = A[A.severity != "info"]
            md.append("| model | batch | severity | type | message |\n|---|---|---|---|---|\n")
            md += [f"| {r.model} | {r.batch} | {r.severity} | {r.type} | {r.message} |\n" for r in crit.head(60).itertuples()]
            md.append(f"\n{len(A) - len(crit)} informational notes (small groups, unproven gaps) are in `alerts.csv`.\n")
        else:
            md.append("No alerts.\n")
        md.append("\n## How to read this\n\n"
                  "- An alert means *investigate*, not *the model is biased*. A growing gap can reflect a real change in "
                  "applicants' economics; the next step is to look at which input columns drifted (the PSI columns in `batches.csv`).\n"
                  "- With many batches and checks, a few false alarms are expected. Critical fairness alerts need p < 0.05 "
                  "AND DI < 0.8 AND at least 30 people in each group, but they are not corrected for multiple testing.\n"
                  "- PSI cutoffs (0.10 / 0.25) are industry rules of thumb, not statistical tests.\n"
                  "- This project is a prototype and is not financial advice.\n")
        Path(path).write_text("".join(md))


# ------------------------------------------------------------------ data + run
def load_df(path):
    if path == "synthetic":
        return _synthetic()
    if Path(path).exists():
        df = pd.read_csv(path)
    else:
        from sklearn.datasets import fetch_openml
        df = fetch_openml("credit-g", version=1, as_frame=True).frame
    return df


def prepare(df):
    ycol = "class" if "class" in df.columns else None
    y = None if ycol is None else df[ycol].astype(str).str.lower().isin(["good", "1"]).astype(int).values
    X = df.drop(columns=[ycol]) if ycol else df.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("category", "str"):
            X[c] = X[c].astype(str)
    return X.reset_index(drop=True), y


def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(35, 11, n), 19, 75).round()
    pick = lambda o, p=None: rng.choice(o, n, p=p)
    sav, chk = pick(["<100", "100<=X<500", "500<=X<1000", ">=1000", "unknown"]), pick(["<0", "0<=X<200", ">=200", "no checking"])
    df = pd.DataFrame({"checking_status": chk, "duration": rng.lognormal(3, .5, n).round(),
                       "credit_amount": rng.lognormal(7.8, .7, n).round(), "savings_status": sav,
                       "employment": pick(["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]),
                       "personal_status": pick(["male single", "female div/dep/mar", "male mar/wid"], [.55, .31, .14]),
                       "age": age, "housing": np.where(rng.random(n) < np.where(age < 25, .6, .15), "rent", "own"),
                       "purpose": pick(["radio/tv", "car", "furniture", "education"])})
    z = (1.6 - .04 * df.duration - .00012 * df.credit_amount + .03 * (age - 35)
         + .9 * np.isin(sav, [">=1000", "500<=X<1000"]) + .9 * (chk == ">=200") + rng.normal(0, .8, n))
    df["class"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "good", "bad")
    return df


def fit_models(Xf, y, th, seed, cutoff=0.5):
    """Train on 70%, keep the other 30% as the reference ('what normal looks like')."""
    strat = y * 2 + (Xf["age"].values < th.age_cut)
    tr, ho = train_test_split(np.arange(len(y)), test_size=0.3, stratify=strat, random_state=seed)
    blind = [c for c in Xf.columns if c not in PROTECTED]
    mons = {}
    for name, cols in (("blind_model", blind), ("aware_model", list(Xf.columns))):
        pipe = build_pipe(cols, Xf).fit(Xf.iloc[tr][cols], y[tr])
        mons[name] = BatchMonitor(name, pipe, cols, Reference(pipe, cols, Xf.iloc[ho], cutoff, th), th, seed)
    return mons, ho, blind


def simulate(Xf, y, kind, n_batches, batch_size, drift_start, strength, seed, th):
    """Resample the hold-out pool into batches; from `drift_start` on, a growing share of rows
    (all applicants for 'covariate', only under-25s for 'fairness') get the economic profile of a
    random low-scoring applicant. Their outcomes follow the donor, so the shift is a realistic one."""
    rng = np.random.default_rng(seed)
    mons, ho, blind = fit_models(Xf, y, th, seed)
    pool, yp = Xf.iloc[ho].reset_index(drop=True), y[ho]
    m0 = mons["blind_model"]
    score = m0.pipe.predict_proba(pool[blind])[:, 1]
    donors = np.where(score <= np.quantile(score, .25))[0]
    gm = GlobalMonitor()
    for m in mons.values():
        gm.add(m)
    for b in range(n_batches):
        i = rng.integers(0, len(pool), batch_size)
        B, yb = pool.iloc[i].reset_index(drop=True), yp[i].copy()
        if kind != "none" and b >= drift_start:
            q = strength * (b - drift_start + 1) / (n_batches - drift_start)
            mask = rng.random(batch_size) < q
            if kind == "fairness":
                mask &= ~group_masks(B, th.age_cut)["age"]
            d = rng.choice(donors, int(mask.sum()))
            for c in blind:
                B.loc[mask, c] = pool[c].values[d]
            yb[mask] = yp[d]
        for m in mons.values():
            m.process(f"B{b + 1:02d}", B, yb)
    return gm


def evaluate(Xf, y, th, runs, n_batches, batch_size, drift_start, strength, seed):
    """Operating characteristics of the monitor itself: how often it cries wolf on clean data and
    how many batches it needs to notice a drift. Run before trusting any alert."""
    rows = []
    for kind in ("none", "covariate", "fairness"):
        stats = {}
        for r in range(runs):
            gm = simulate(Xf, y, kind, n_batches, batch_size, drift_start, strength, seed + r, th)
            A = gm.alert_log()
            for name in gm.monitors:
                c = A[(A.model == name) & (A.severity == "critical")] if len(A) else A
                idx = c.batch_index.values if len(c) else np.array([], int)
                pre = idx[idx <= drift_start] if kind != "none" else idx
                post = idx[idx > drift_start]
                st = stats.setdefault(name, {"fa": [], "det": [], "delay": []})
                st["fa"].append(len(pre) > 0)
                if kind != "none":
                    st["det"].append(len(post) > 0)
                    if len(post):
                        st["delay"].append(int(post.min()) - drift_start)
        for name, st in stats.items():
            rows.append({"scenario": kind, "model": name, "runs": runs,
                         "false_alarm_rate": float(np.mean(st["fa"])),
                         "detection_rate": float(np.mean(st["det"])) if st["det"] else np.nan,
                         "median_delay_batches": float(np.median(st["delay"])) if st["delay"] else np.nan})
    return pd.DataFrame(rows)


def monitor_files(files, Xf, y, th, seed):
    mons, _, _ = fit_models(Xf, y, th, seed)
    gm = GlobalMonitor()
    for m in mons.values():
        gm.add(m)
    for f in sorted(files):
        B, yb = prepare(pd.read_csv(f))
        for m in mons.values():
            m.process(Path(f).stem, B, yb)
    return gm


def save(gm, out, drift_start=None, note=""):
    os.makedirs(out, exist_ok=True)
    gm.table().to_csv(f"{out}/batches.csv", index=False)
    gm.alert_log().to_csv(f"{out}/alerts.csv", index=False)
    gm.pooled_audit().to_csv(f"{out}/pooled_audit.csv", index=False)
    gm.dashboard(f"{out}/dashboard.png", drift_start)
    gm.report(f"{out}/monitor_report.md", note)
    for n, v in gm.health().items():
        print(f"{n}: {v['status']}  ({v['critical_total']} critical / {v['alerts_total']} alerts, {v['batches']} batches)")
    print(f"Report: {out}/monitor_report.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="measure the monitor's own false-alarm rate and detection delay")
    ev.add_argument("--data", default="data/credit_g.csv"); ev.add_argument("--out", default="results/monitor")
    ev.add_argument("--seed", type=int, default=42); ev.add_argument("--runs", type=int, default=20)
    ev.add_argument("--batches", type=int, default=16); ev.add_argument("--batch-size", type=int, default=400)
    ev.add_argument("--drift-start", type=int, default=8); ev.add_argument("--strength", type=float, default=0.7)
    for name in ("simulate", "batch"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="data/credit_g.csv", help="reference data (or 'synthetic')")
        p.add_argument("--out", default="results/monitor")
        p.add_argument("--seed", type=int, default=42)
        if name == "simulate":
            p.add_argument("--kind", choices=["none", "covariate", "fairness"], default="fairness")
            p.add_argument("--batches", type=int, default=16)
            p.add_argument("--batch-size", type=int, default=400)
            p.add_argument("--drift-start", type=int, default=8, help="number of clean batches first")
            p.add_argument("--strength", type=float, default=0.7)
        else:
            p.add_argument("--batches", required=True, help='glob of batch CSVs, e.g. "new/*.csv"')
    a = ap.parse_args()
    th = Thresholds()
    Xf, y = prepare(load_df(a.data))
    if y is None:
        raise SystemExit("reference data needs the 'class' column")
    if a.cmd == "evaluate":
        T = evaluate(Xf, y, th, a.runs, a.batches, a.batch_size, a.drift_start, a.strength, a.seed)
        os.makedirs(a.out, exist_ok=True)
        T.to_csv(f"{a.out}/monitor_operating_characteristics.csv", index=False)
        print(T.round(2).to_string(index=False))
        return
    if a.cmd == "simulate":
        gm = simulate(Xf, y, a.kind, a.batches, a.batch_size, a.drift_start, a.strength, a.seed, th)
        note = (f"> **Simulation**: scenario `{a.kind}`, drift injected after batch {a.drift_start}. "
                "Batches are resampled from one held-out pool, so they share that pool's chance gaps and are not independent; "
                "this demonstrates the monitor, not real-world drift.\n")
        if a.data == "synthetic":
            note += "> Synthetic data: numbers are meaningless.\n"
        save(gm, a.out, a.drift_start if a.kind != "none" else None, note)
    else:
        files = glob.glob(a.batches)
        if not files:
            raise SystemExit(f"no files match {a.batches}")
        save(monitor_files(files, Xf, y, th, a.seed), a.out)


if __name__ == "__main__":
    main()
