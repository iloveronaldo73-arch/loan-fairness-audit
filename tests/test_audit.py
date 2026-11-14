import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_data
from audit import data
from audit.metrics import apply_cutoffs, compare, pick_cutoffs, rates
from audit.run_audit import run


def test_rates_by_hand():
    r = rates([1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0])
    assert r["approval"] == 0.5
    assert abs(r["tpr"] - 2 / 3) < 1e-9
    assert abs(r["fpr"] - 1 / 3) < 1e-9
    assert abs(r["accuracy"] - 4 / 6) < 1e-9


def test_gaps_are_other_minus_reference():
    y_true = [1, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 1, 1]
    group = ["a", "a", "a", "b", "b", "b"]
    c = compare(y_true, y_pred, group, other="a", reference="b")
    assert abs(c["approval_gap"] - (1 / 3 - 1)) < 1e-9
    assert abs(c["tpr_gap"] - (0.5 - 1)) < 1e-9
    assert abs(c["fpr_gap"] - (0 - 1)) < 1e-9
    assert abs(c["impact_ratio"] - 1 / 3) < 1e-9


def test_cutoffs_equalise_tpr():
    rng = np.random.default_rng(1)
    n = 6000
    group = rng.choice(["a", "b"], n)
    y = (rng.random(n) < 0.7).astype(int)
    scores = np.clip(0.35 + 0.3 * y + rng.normal(0, 0.12, n) - 0.08 * (group == "a"), 0, 1)
    cuts = pick_cutoffs(scores, y, group, ["a", "b"])
    pred = apply_cutoffs(scores, group, cuts)
    target = (scores[y == 1] >= 0.5).mean()
    for g in ["a", "b"]:
        tpr = pred[(group == g) & (y == 1)].mean()
        assert abs(tpr - target) < 0.02
    assert cuts["a"] < cuts["b"]


def test_pipeline_runs_on_fake_data():
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake.csv"
        fake_data.make().to_csv(fake, index=False)
        data.CACHE = fake
        X, y, groups = data.load()
        assert set(groups["sex"]) == {"female", "male"}
        assert set(groups["age"]) == {"under 25", "25 and over"}
        assert "class" not in X.columns

        summary, proxy = run(X, y, groups, "logreg", n_splits=3, out=Path(tmp) / "out")
        assert len(summary) == 2 * 4
        assert (Path(tmp) / "out" / "gaps.png").exists()
        assert (Path(tmp) / "out" / "summary.md").exists()
        everyone = summary[summary["strategy"] == "approve everyone"]
        assert (everyone["approval_gap_mean"] == 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
