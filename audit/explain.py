"""Extra analysis for the site: which features drive each gap, which columns stand in
for the protected trait, example mistakes, and the model weights the page uses to
score an applicant in the browser (German Credit only)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import build



def _params(pipe, X):
    """Logistic regression weights as plain numbers, so a browser can reproduce predict_proba."""
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    steps = prep.named_transformers_["num"].named_steps
    ind = getattr(steps["fill"], "indicator_", None)
    skip = len(ind.features_) if ind is not None else 0  # missing-value flags sit after the numeric columns; the page never sends a missing value
    num, cat = prep.transformers_[0][2], prep.transformers_[1][2]
    coef = clf.coef_[0]
    mean, scale, n = steps["scale"].mean_, steps["scale"].scale_, len(num)
    # the scaler also standardizes the 0/1 missing flags; for a complete applicant they add a constant
    b0 = float(clf.intercept_[0]) + sum(float(coef[n + k]) * (0 - mean[n + k]) / scale[n + k] for k in range(skip))
    out = {"b0": b0, "num": {}, "cat": {}}
    for j, c in enumerate(num):
        out["num"][c] = [float(mean[j]), float(scale[j]), float(coef[j])]
    pos = len(num) + skip
    for c, cats in zip(cat, prep.named_transformers_["cat"].categories_):
        out["cat"][c] = {str(v): float(coef[pos + i]) for i, v in enumerate(cats)}
        pos += len(cats)
    return out


def score(p, row):
    """Same arithmetic as the page's JavaScript."""
    z = p["b0"]
    for c, (m, s, w) in p["num"].items():
        z += w * (row[c] - m) / s
    for c, w in p["cat"].items():
        z += w.get(str(row[c]), 0.0)
    return 1 / (1 + np.exp(-z))


def drivers(X, groups, spec, pb):
    """Share of each group's difference in model score (log-odds) that each column accounts for."""
    Xb = X.drop(columns=spec["protected"])
    out = {}
    for attr, info in spec["attributes"].items():
        o = (groups[attr] == info["other"]).to_numpy()
        r = (groups[attr] == info["reference"]).to_numpy()
        rows = []
        for c, (m, s, w) in pb["num"].items():
            a, b = Xb.loc[o, c].mean(), Xb.loc[r, c].mean()
            rows.append((c, w * (a - b) / s, f"{a:,.0f} vs {b:,.0f}"))
        for c, ws in pb["cat"].items():
            parts = {v: ((Xb.loc[o, c] == v).mean(), (Xb.loc[r, c] == v).mean()) for v in ws}
            rows.append((c, sum(ws[v] * (p[0] - p[1]) for v, p in parts.items()), ""))
            v = max(parts, key=lambda k: abs(ws[k] * (parts[k][0] - parts[k][1])))
            rows[-1] = (c, rows[-1][1], f"{v}: {parts[v][0]:.0%} vs {parts[v][1]:.0%}")
        rows.sort(key=lambda t: -abs(t[1]))
        out[attr] = [{"feature": c, "effect": round(float(d), 2), "note": n} for c, d, n in rows[:5]]
    return out


def standins(X, groups, spec):
    """Columns the blind model could use to rebuild the protected trait."""
    Xb = X.drop(columns=spec["protected"])
    out = {}
    for attr, info in spec["attributes"].items():
        keep = groups[attr].isin([info["other"], info["reference"]]).to_numpy()
        target = (groups[attr][keep] == info["other"]).astype(int)
        pipe = build("logreg", Xb[keep]).fit(Xb[keep], target)
        names = pipe.named_steps["prep"].get_feature_names_out()
        coef = pipe.named_steps["clf"].coef_[0]
        rows = []
        for i in np.argsort(-np.abs(coef))[:5]:
            nm = names[i].split("__", 1)[1]
            for col in Xb.columns:
                if nm.startswith(col + "_"):
                    nm = f"{col} = {nm[len(col) + 1:]}"
            rows.append({"feature": nm, "toward": info["other"] if coef[i] > 0 else info["reference"]})
        out[attr] = rows
    return out


def mistakes(X, y, groups, spec):
    """The most confident wrong calls of the blind model on one test split."""
    from .run_audit import make_split

    Xb = X.drop(columns=spec["protected"])
    train, _, test = make_split(y, groups, 0)
    pipe = build("logreg", Xb).fit(Xb.iloc[train], y.iloc[train])
    p = pipe.predict_proba(Xb.iloc[test])[:, 1]
    yt = y.iloc[test].to_numpy()
    show = [c for c in spec["show"] if c in X.columns]
    bad = np.where((p >= 0.5) & (yt == 0))[0]
    miss = np.where((p < 0.5) & (yt == 1))[0]
    rows = []
    for kind, idx in [(spec["mistake_kinds"][0], bad[np.argsort(-p[bad])][:3]), (spec["mistake_kinds"][1], miss[np.argsort(p[miss])][:2])]:
        for i in idx:
            rec = X.iloc[test].iloc[i][show].to_dict()
            rows.append({"kind": kind, "p": round(float(p[i]), 2), "applicant": {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}})
    return rows


def _py(v):
    v = v.item() if hasattr(v, "item") else v
    return None if isinstance(v, float) and np.isnan(v) else v


def _form(X, groups, spec):
    """Describe the page's input form: range sliders and dropdowns with defaults."""
    out, big = [], len(X) > 5000
    for f in spec["form"]:
        if f["col"] not in X.columns:
            continue  # a file without this column just gets a shorter form
        col = X[f["col"]]
        d = {k: f[k] for k in ("id", "label", "col", "kind")}
        if f["kind"] == "range":
            lo, hi = (col.quantile(0.005), col.quantile(0.995)) if big else (col.min(), col.max())
            d.update(min=_py(round(lo)), max=_py(round(hi)), default=_py(round(col.median())))
        elif f["kind"] == "sex":
            d.update(choices=f["choices"], default=f["choices"][0],
                     map={s: X.loc[groups["sex"] == s, f["col"]].mode()[0] for s in f["choices"]})
        else:
            d["choices"] = f.get("choices") or col.value_counts().index[:12].tolist()
            d["default"] = col.mode()[0] if col.mode()[0] in d["choices"] else d["choices"][0]
        out.append(d)
    return out


def run_extras(X, y, groups, spec, out):
    if not spec.get("scorer"):
        return
    Xb = X.drop(columns=spec["protected"])
    full, blind = build("logreg", X).fit(X, y), build("logreg", Xb).fit(Xb, y)
    pf, pb = _params(full, X), _params(blind, Xb)
    cut = {"full": 0.5, "blind": 0.5}
    if spec["match_rate"]:  # approve the same share as the data did
        cut = {"full": float(np.quantile(full.predict_proba(X)[:, 1], 1 - y.mean())),
               "blind": float(np.quantile(blind.predict_proba(Xb)[:, 1], 1 - y.mean()))}
    defaults = {c: _py(X[c].median()) if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode()[0] for c in X.columns}
    extra = {
        "model": {"full": pf, "blind": pb},
        "cut": cut,
        "defaults": defaults,
        "form": _form(X, groups, spec),
        "cases": spec["cases"],
        "drivers": drivers(X, groups, spec, pb),
        "standins": standins(X, groups, spec),
        "mistakes": mistakes(X, y, groups, spec),
    }
    Path(out, "extra.json").write_text(json.dumps(extra))
