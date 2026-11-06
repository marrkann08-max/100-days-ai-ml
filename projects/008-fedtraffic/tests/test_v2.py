"""Round 2: seasonal inputs and targets, the seasonal-residual baseline, per-day errors, paired tests."""
import unittest

import numpy as np
import torch

from fedtraffic import config as cfg
from fedtraffic import data as ds
from fedtraffic import models, train
from fedtraffic.metrics import Accumulator
from fedtraffic.seasonal import SeasonalResidual
from fedtraffic.stats import paired_bootstrap


def toy_data(t=ds.STEPS_PER_WEEK * 2, n=3, seed=0):
    """Two weeks of a clean weekly pattern plus noise, in the cached-data layout."""
    rng = np.random.default_rng(seed)
    slots = (np.arange(t) % ds.STEPS_PER_WEEK).astype(np.int32)
    profile = 50 + 10 * np.sin(2 * np.pi * np.arange(ds.STEPS_PER_WEEK) / ds.STEPS_PER_DAY)[:, None] + np.arange(n)
    speeds = (profile[slots] + rng.normal(0, 1, (t, n))).astype(np.float32)
    mean, std = float(speeds.mean()), float(speeds.std())
    return {"speeds": speeds, "observed": np.ones((t, n), bool), "inputs": ((speeds - mean) / std).astype(np.float32),
            "time": np.zeros((t, 2), np.float32), "slots": slots, "historical": profile.astype(np.float32),
            "split": np.array([int(t * .7), int(t * .8)]), "norm": np.array([mean, std], np.float32)}


class SeasonalWindowTests(unittest.TestCase):
    def test_plain_windows_are_unchanged(self):
        w = ds.Windows(toy_data(), "train", 12, 12, 5)
        x, *_ , future = w.batch(np.arange(4))
        self.assertEqual(x.shape[-1], 4)
        self.assertEqual(w.offset(future), w.mean)

    def test_seasonal_channel_is_deviation_from_profile(self):
        data = toy_data()
        w = ds.Windows(data, "train", 12, 12, 5, seasonal=True)
        x, _, _, future = w.batch(np.arange(4))
        self.assertEqual(x.shape[-1], 5)
        past = w.starts[:4, None] + np.arange(12)
        expected = (data["speeds"][past] - data["historical"][data["slots"][past]]) / w.std
        np.testing.assert_allclose(x[..., 4], expected, atol=1e-4)
        np.testing.assert_allclose(w.offset(future), data["historical"][data["slots"][future]])

    def test_seasonal_model_predicts_profile_plus_residual(self):
        data = toy_data()
        config = cfg.load(overrides={"model.name": "gconv_lstm", "model.seasonal": True})
        model = models.build(config)
        self.assertEqual(model.cell.gates.linear.in_features, (5 + 32) * 5)
        for p in model.head.parameters():   # zero output: the forecast is exactly the profile
            torch.nn.init.zeros_(p)
        w = ds.Windows(data, "test", 12, 12, 7, seasonal=True)
        x, _, _, future = w.batch(np.arange(3))
        graph = models.Graph(np.eye(3, dtype=np.float32), 2)
        with torch.no_grad():
            pred = train.predict(model, graph, w, x, future, w.mean, w.std)
        np.testing.assert_allclose(pred, data["historical"][data["slots"][future]], atol=1e-4)


class SeasonalResidualTests(unittest.TestCase):
    def test_learns_to_use_a_persistent_deviation(self):
        data = toy_data()
        data["speeds"][:, 0] += 5   # sensor 0 runs 5 mph above its profile the whole time
        w = ds.Windows(data, "val", 12, 12, 3)
        model = SeasonalResidual().fit(w)
        self.assertTrue(all(weight >= .9 for weight in model.weight))
        plain = train.evaluate("historical_average", None, w, data["norm"]).result()["mae"]
        tuned = train.evaluate(model, None, w, data["norm"]).result()["mae"]
        self.assertLess(tuned, plain)

    def test_matches_the_standalone_analysis_on_metr_la(self):
        try:
            data = ds.load(cfg.load())
        except Exception as error:
            raise unittest.SkipTest(f"METR-LA not available: {error}")
        config = cfg.load()
        model = SeasonalResidual().fit(train.windows_for(data, config, "val"))
        self.assertEqual((model.lookback[11], model.weight[11]), (3, 0.45))   # results/tables/day008_seasonal_results.csv
        test = train.evaluate(model, None, train.windows_for(data, config, "test"), data["norm"]).result()
        self.assertAlmostEqual(test["mae@60min"], 4.103859, places=5)


class DailyTests(unittest.TestCase):
    def test_daily_sums_equal_totals_and_survive_merge(self):
        rng = np.random.default_rng(0)
        a, b = Accumulator(2), Accumulator(2)
        for acc in (a, b):
            pred, target = rng.normal(50, 5, (6, 2, 3)), rng.normal(50, 5, (6, 2, 3))
            acc.add(pred, target, rng.random((6, 2, 3)) > .2, days=np.array([0, 0, 1, 1, 1, 2]))
        a.merge(b)
        daily = a.daily()
        self.assertEqual(daily["day"], [0, 1, 2])
        np.testing.assert_allclose(np.sum(daily["abs"], axis=0), a.abs)
        np.testing.assert_allclose(np.sum(daily["count"], axis=0), a.count)


class BootstrapTests(unittest.TestCase):
    def daily(self, offset):
        rng = np.random.default_rng(1)
        abs_ = rng.uniform(3, 5, (20, 2)) + offset
        return {"day": list(range(20)), "abs": (abs_ * 100).tolist(), "count": np.full((20, 2), 100.).tolist()}

    def test_identical_methods_differ_by_zero(self):
        r = paired_bootstrap([self.daily(0)], [self.daily(0)])
        self.assertEqual((r["diff"], r["low"], r["high"]), (0, 0, 0))

    def test_consistent_improvement_excludes_zero(self):
        r = paired_bootstrap([self.daily(-.2)], [self.daily(0)])
        self.assertAlmostEqual(r["diff"], -.2)
        self.assertLess(r["high"], 0)
        self.assertEqual(r["a_better_days"], 20)

    def test_refuses_different_readings(self):
        other = self.daily(0)
        other["count"][0][0] = 50
        with self.assertRaises(ValueError):
            paired_bootstrap([self.daily(0)], [other])


if __name__ == "__main__":
    unittest.main()
