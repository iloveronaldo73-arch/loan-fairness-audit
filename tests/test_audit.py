import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_data
from audit import build_site, data
from audit.explain import _params, score
from audit.models import build
from audit.metrics import apply_cutoffs, compare, pick_cutoffs, rates
from audit.run_audit import run


def test_rates_by_hand():
    r = rates([1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0])
    assert r["approval"] == 0.5
    assert abs(r["tpr"] - 2 / 3) < 1e-9
    assert abs(r["fpr"] - 1 / 3) < 1e-9
    assert abs(r["accuracy"] - 4 / 6) < 1e-9


def test_gaps_are_other_minus_reference():
    c = compare([1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 1], ["a", "a", "a", "b", "b", "b"], other="a", reference="b")
    assert abs(c["approval_gap"] - (1 / 3 - 1)) < 1e-9
    assert abs(c["tpr_gap"] - (0.5 - 1)) < 1e-9
    assert abs(c["fpr_gap"] - (0 - 1)) < 1e-9
    assert abs(c["impact_ratio"] - 1 / 3) < 1e-9


def test_cutoffs_equalise_tpr_and_ignore_other_groups():
    rng = np.random.default_rng(1)
    n = 6000
    group = rng.choice(["a", "b"], n)
    y = (rng.random(n) < 0.7).astype(int)
    scores = np.clip(0.35 + 0.3 * y + rng.normal(0, 0.12, n) - 0.08 * (group == "a"), 0, 1)
    cuts = pick_cutoffs(scores, y, group, ["a", "b"])
    pred = apply_cutoffs(scores, group, cuts)
    target = (scores[y == 1] >= 0.5).mean()
    for g in ["a", "b"]:
        assert abs(pred[(group == g) & (y == 1)].mean() - target) < 0.02
    assert cuts["a"] < cuts["b"]
    # a group with no cutoff of its own falls back to the default
    assert list(apply_cutoffs([0.6, 0.4], ["excluded", "excluded"], cuts)) == [1, 0]


def test_german_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake.csv"
        fake_data.make_german().to_csv(fake, index=False)
        data.CACHE = fake
        X, y, groups, spec = data.load("german")
        assert set(groups["sex"]) == {"female", "male"} and "class" not in X.columns
        summary, _ = run(X, y, groups, spec, "logreg", 3, Path(tmp) / "out")
        assert len(summary) == 2 * 5 and "blind + cost cutoff" in set(summary["strategy"])
        assert (abs(summary[summary["strategy"] == "approve everyone"]["cost_mean"] - 5 * (1 - y.mean())) < 0.4).all()
        extra = json.loads((Path(tmp) / "out" / "extra.json").read_text())
        assert set(extra["drivers"]) == {"age", "sex"} and extra["mistakes"] and extra["form"][0]["id"] == "age"
        # the browser scorer must agree with scikit-learn
        pipe = build("logreg", X).fit(X, y)
        p = _params(pipe, X)
        mine = [score(p, X.iloc[i]) for i in range(40)]
        assert np.allclose(pipe.predict_proba(X.iloc[:40])[:, 1], mine)
        assert (summary[summary["strategy"] == "approve everyone"]["approval_gap_mean"] == 0).all()


def test_hmda_loader_and_pipeline_and_site():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data.DATA_DIR = tmp
        raw = tmp / "hmda_raw.csv"
        fake_data.make_hmda().to_csv(raw, index=False)
        X, y, groups, spec = data.load("hmda", max_rows=4000)
        assert len(X) == 4000 and (tmp / "hmda_sample.csv").exists()
        # columns that exist only for approved loans must never reach the model
        assert "interest_rate" not in X.columns and "denial_reason-1" not in X.columns
        assert not X.select_dtypes(exclude="number").isna().any().any()
        assert set(groups["race"]) == {"Black", "White", "excluded"}
        assert y.isin([0, 1]).all()

        # second load reads the saved sample on its own
        raw.unlink()
        X2, _, _, _ = data.load("hmda")
        assert len(X2) == 4000

        out = tmp / "out"
        summary, proxy = run(X, y, groups, spec, "logreg", 2, out)
        assert len(summary) == 3 * 5
        obs = summary[summary["strategy"] == "observed approvals"]
        assert obs["accuracy_mean"].isna().all() and obs["approval_gap_mean"].notna().all()
        assert json.loads((out / "meta.json").read_text())["rows"] == 4000
        # the browser scorer must agree with scikit-learn, including with missing-value flags in the model
        pipe = build("logreg", X).fit(X, y)
        p = _params(pipe, X)
        ok = X.index[X["income"].notna()][:40]
        assert np.allclose(pipe.predict_proba(X.loc[ok])[:, 1], [score(p, X.loc[i]) for i in ok])
        assert 0.5 < json.loads((out / "extra.json").read_text())["cut"]["full"] < 1

        page = tmp / "index.html"
        shutil.copy(ROOT / "docs" / "index.html", page)
        build_site.build(out, page)
        html = page.read_text()
        block = html.split("/*DATA*/")[1].split("/*END*/")[0]
        site = json.loads(block)
        assert set(site["res"]) == {"race", "sex", "age"} and "proxy" in site
        assert len(site["res"]["race"]["full model"]) == 6  # other %, ref %, accuracy, gap low, gap high, cost
        build_site.build(out, page)  # running twice must still work


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
