"""Data pipeline: cleaning, leakage, windows. Uses the real cached data when present."""
import unittest

import numpy as np

from fedtraffic import config as cfg
from fedtraffic import data as ds


def real_data():
    try:
        return ds.load(cfg.load())
    except Exception as error:   # no network / data not downloaded yet
        raise unittest.SkipTest(f"METR-LA not available: {error}")


class CleaningTests(unittest.TestCase):
    def test_only_short_interior_gaps_are_filled(self):
        x = np.array([[1.], [0.], [0.], [4.], [0.], [0.], [0.], [0.], [9.], [0.]], np.float32)
        mask = x[:, 0] > 0
        filled, usable = ds.fill_short_gaps(x, mask[:, None], max_gap=3)
        np.testing.assert_allclose(filled[1:3, 0], [2, 3])      # 2-step gap: interpolated
        self.assertFalse(usable[4:8, 0].any())                   # 4-step gap: left missing
        self.assertFalse(usable[9, 0])                           # trailing gap: no right neighbour

    def test_kernel_adjacency_thresholds_and_ignores_missing_paths(self):
        dist = np.array([[0, 100, np.inf], [100, 0, 5000], [np.inf, 5000, 0]], float)
        w = ds.kernel_adjacency(dist, threshold=.1)
        np.testing.assert_allclose(np.diag(w), 1)
        self.assertEqual(w[0, 2], 0)
        self.assertGreater(w[0, 1], w[1, 2])

    def test_week_slot_monday_midnight_is_zero(self):
        stamps = np.array(["2012-03-05T00:00", "2012-03-05T00:05", "2012-03-11T23:55"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(ds.week_slot(stamps), [0, 1, ds.STEPS_PER_WEEK - 1])

    def test_historical_average_ignores_the_future(self):
        rng = np.random.default_rng(0)
        speeds = rng.uniform(20, 70, (ds.STEPS_PER_WEEK * 3, 2)).astype(np.float32)
        mask = np.ones_like(speeds, bool)
        slots = np.arange(len(speeds)) % ds.STEPS_PER_WEEK
        train_end = ds.STEPS_PER_WEEK * 2
        a = ds.historical_average(speeds, mask, slots, train_end)
        speeds[train_end:] = 0
        b = ds.historical_average(speeds, mask, slots, train_end)
        np.testing.assert_array_equal(a, b)


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = real_data()

    def test_shape_and_known_graph_size(self):
        self.assertEqual(self.d["speeds"].shape, (34272, 207))
        self.assertEqual(int((self.d["adjacency"] > 0).sum()), 1722)   # matches Li et al. (2018)

    def test_normalisation_uses_training_period_only(self):
        train_end = self.d["split"][0]
        s, o = self.d["speeds"][:train_end], self.d["observed"][:train_end]
        np.testing.assert_allclose(self.d["norm"], [s[o].mean(), s[o].std()], rtol=1e-5)

    def test_windows_never_cross_split_boundaries(self):
        train_end, val_end = self.d["split"]
        for part, lo, hi in (("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, 34272)):
            w = ds.Windows(self.d, part, 12, 12, stride=1)
            self.assertGreaterEqual(w.starts.min(), lo)
            self.assertLessEqual(w.starts.max() + 24, hi)

    def test_targets_are_raw_and_inputs_follow_them(self):
        w = ds.Windows(self.d, "test", 12, 12, stride=1)
        x, y, m, future = w.batch(np.array([0, 100]))
        self.assertEqual(x.shape, (2, 12, 207, 4))
        np.testing.assert_array_equal(future[:, 0] - 12, w.starts[[0, 100]])
        self.assertTrue((y[m] > 0).all())                # every scored target is a real reading
        self.assertTrue((y[~m] == 0).all())

    def test_cache_is_reused(self):
        again = ds.load(cfg.load())
        self.assertEqual(again["cache_key"], self.d["cache_key"])


if __name__ == "__main__":
    unittest.main()
