import unittest

import numpy as np
import pandas as pd

from audit import dataset
from audit.audit import _split_indices, calibrated_cut, group_cutoffs, make_model


class SplitTest(unittest.TestCase):
    def test_splits_are_disjoint_and_complete(self):
        test, val, train = _split_indices(1000, np.random.default_rng(0), 0.2, 0.2)
        self.assertEqual((len(test), len(val), len(train)), (200, 200, 600))
        self.assertEqual(len(set(test) | set(val) | set(train)), 1000)
        self.assertEqual(len(set(test) & set(val)), 0)


class CalibrationTest(unittest.TestCase):
    def test_cut_reproduces_the_base_rate(self):
        rng = np.random.default_rng(0)
        y = (rng.random(20000) < 0.85).astype(int)
        p = np.clip(0.5 + 0.4 * y + rng.normal(0, 0.15, 20000), 0.01, 0.99)
        cut = calibrated_cut(y, p)
        self.assertAlmostEqual(float((p >= cut).mean()), float(y.mean()), delta=0.01)


class GroupCutoffTest(unittest.TestCase):
    def test_each_group_is_approved_at_the_target_rate(self):
        rng = np.random.default_rng(1)
        groups = np.where(rng.random(10000) < 0.2, "other", "ref")
        p = np.clip(rng.normal(0.7, 0.2, 10000) - (groups == "other") * 0.08, 0.01, 0.99)
        cuts = group_cutoffs(p, groups, 0.75)
        for name in ("other", "ref"):
            rate = float((p[groups == name] >= cuts[name]).mean())
            self.assertAlmostEqual(rate, 0.75, delta=0.02)


class ModelTest(unittest.TestCase):
    def test_pipeline_survives_missing_values(self):
        rng = np.random.default_rng(2)
        n = 500
        frame = pd.DataFrame({
            "loan_amount": rng.normal(200_000, 50_000, n),
            "income": rng.normal(80_000, 20_000, n),
            "loan_type": rng.choice(["1", "2", "3"], n),
            "applicant_age": rng.choice(["<25", "25-34", "35-44"], n),
        })
        frame.loc[::7, "loan_amount"] = np.nan
        frame.loc[::11, "loan_type"] = None
        features = ["loan_amount", "income", "loan_type", "applicant_age"]
        y = (rng.random(n) < 0.8).astype(int)

        model = make_model(features).fit(frame, y)
        proba = model.predict_proba(frame)

        self.assertEqual(proba.shape, (n, 2))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))


class DatasetTest(unittest.TestCase):
    @unittest.skipUnless(dataset.CACHE.exists(), "HMDA extract is not cached yet")
    def test_analysis_sample_has_both_traits(self):
        frame = dataset.load()
        self.assertEqual(len(frame), dataset.SAMPLE)
        self.assertEqual(set(frame["target"].unique()), {0, 1})
        self.assertLessEqual(set(frame["race"].unique()), {"Black", "White"})
        self.assertLessEqual(set(frame["age_group"].unique()), {"under 25", "25 and over"})
        self.assertGreater(int((frame["age_group"] == "under 25").sum()), 0)


if __name__ == "__main__":
    unittest.main()
