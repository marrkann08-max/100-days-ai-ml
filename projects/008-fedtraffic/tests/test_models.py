import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from fedtraffic import config as cfg
from fedtraffic import models, tracking
from fedtraffic.federated import compress
from fedtraffic.metrics import Accumulator, masked_mae_loss
from fedtraffic.partition import cut_edges, partition
from fedtraffic.privacy import epsilon


def ring(n):
    w = np.eye(n, dtype=np.float32)
    for i in range(n):
        w[i, (i + 1) % n] = .5
    return w


class GraphTests(unittest.TestCase):
    def test_hops_are_powers_of_the_random_walk(self):
        w = ring(5)
        g = models.Graph(w, steps=2)
        p = models.Graph.random_walk(w)
        hops = g.hops.reshape(4, 5, 5).numpy()
        np.testing.assert_allclose(hops[0], p, atol=1e-6)
        np.testing.assert_allclose(hops[1], p @ p, atol=1e-6)
        np.testing.assert_allclose(hops[2], models.Graph.random_walk(w.T), atol=1e-6)
        np.testing.assert_allclose(hops[0].sum(axis=1), 1, atol=1e-6)

    def test_diffusion_conv_respects_sensor_order(self):
        # Permuting the sensors (and the graph with them) must permute the output the same way.
        torch.manual_seed(0)
        w = np.random.default_rng(1).random((6, 6)).astype(np.float32) * (np.random.default_rng(2).random((6, 6)) > .5)
        conv = models.DiffusionConv(3, 4, 2)
        x = torch.randn(2, 6, 3)
        perm = np.random.default_rng(3).permutation(6)
        out = conv(x, models.Graph(w, 2))
        out_perm = conv(x[:, perm], models.Graph(w[np.ix_(perm, perm)], 2))
        torch.testing.assert_close(out[:, perm], out_perm, atol=1e-5, rtol=1e-5)

    def test_every_model_shape_and_size_independent_of_sensor_count(self):
        config = cfg.load()
        for name in models.LEARNED:
            config["model"]["name"] = name
            model = models.build(config)
            for n in (5, 9):
                out = model(torch.randn(2, 12, n, 4), models.Graph(ring(n), 2))
                self.assertEqual(out.shape, (2, 12, n))
            self.assertGreater(models.parameters(model), 0)

    def test_graph_actually_matters_for_graph_models(self):
        config = cfg.load()
        torch.manual_seed(0)
        model = models.build(config)   # gconv_gru
        x = torch.randn(1, 12, 5, 4)
        a = model(x, models.Graph(np.eye(5, dtype=np.float32), 2))
        b = model(x, models.Graph(ring(5), 2))
        self.assertGreater((a - b).abs().max().item(), 1e-4)


class MetricTests(unittest.TestCase):
    def test_masked_values_are_ignored(self):
        pred = np.array([[[50., 10.]]])
        target = np.array([[[55., 0.]]])
        mask = np.array([[[True, False]]])
        acc = Accumulator(1)
        acc.add(pred, target, mask)
        self.assertEqual(acc.count.sum(), 1)
        self.assertAlmostEqual(acc.abs.sum(), 5)
        loss = masked_mae_loss(torch.tensor(pred), torch.tensor(target), torch.tensor(mask))
        self.assertAlmostEqual(loss.item(), 5)

    def test_pooled_metrics_equal_one_big_batch(self):
        rng = np.random.default_rng(0)
        p, t = rng.uniform(20, 70, (2, 4, 12, 6)), rng.uniform(20, 70, (2, 4, 12, 6))
        m = rng.random((2, 4, 12, 6)) > .2
        a, b, whole = Accumulator(), Accumulator(), Accumulator()
        a.add(p[0], t[0], m[0])
        b.add(p[1], t[1], m[1])
        whole.add(np.concatenate(p), np.concatenate(t), np.concatenate(m))
        merged = a.merge(b).result()
        for k, v in whole.result().items():
            self.assertAlmostEqual(merged[k], v, places=9)


class FederatedTests(unittest.TestCase):
    def test_partitions_cover_every_sensor_once(self):
        locations = np.random.default_rng(0).random((60, 2))
        for method in ("spatial", "random"):
            groups = partition(locations, 4, method)
            self.assertEqual(sorted(np.concatenate(groups).tolist()), list(range(60)))

    def test_spatial_partition_cuts_fewer_edges_than_random(self):
        rng = np.random.default_rng(1)
        locations = rng.random((80, 2))
        d = np.hypot(*(locations[:, None] - locations[None]).transpose(2, 0, 1))
        adj = (d < .15).astype(float)
        spatial = cut_edges(adj, partition(locations, 4, "spatial"))
        random = cut_edges(adj, partition(locations, 4, "random"))
        self.assertLess(spatial, random / 2)

    def test_update_compression(self):
        torch.manual_seed(0)
        old = {"w": torch.randn(100), "b": torch.randn(10)}
        new = {k: v + .01 * torch.randn_like(v) for k, v in old.items()}
        same, size32 = compress(new, old, 32)
        self.assertEqual(size32, 110 * 4)
        half, size16 = compress(new, old, 16)
        self.assertEqual(size16, 110 * 2)
        byte, size8 = compress(new, old, 8)
        self.assertEqual(size8, 110 + 16)
        for k in old:
            torch.testing.assert_close(same[k], new[k])
            delta = new[k] - old[k]
            step = (delta.max() - delta.min()) / 255
            self.assertLessEqual((byte[k] - new[k]).abs().max().item(), step.item() / 2 + 1e-6)
            self.assertLess((half[k] - new[k]).abs().max().item(), 1e-4)


class PrivacyTests(unittest.TestCase):
    def test_epsilon_behaves(self):
        self.assertEqual(epsilon(0, 12), float("inf"))
        self.assertGreater(epsilon(1.0, 12), epsilon(2.0, 12))      # more noise, more privacy
        self.assertGreater(epsilon(1.0, 24), epsilon(1.0, 12))      # more rounds, less privacy

    def test_epsilon_matches_a_hand_computed_case(self):
        # One round, sigma = 1, delta = 1e-5: minimise a/2 + ln(1e5)/(a - 1) by hand.
        a = 1 + np.sqrt(2 * np.log(1e5))
        expected = a / 2 + np.log(1e5) / (a - 1)
        self.assertAlmostEqual(epsilon(1.0, 1), expected, delta=.02)


class TrackingTests(unittest.TestCase):
    def test_run_records_config_git_and_results(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(tracking, "RUNS", Path(folder)):
            run = tracking.Run("demo", {"seed": 3, "x": 1}, group="test")
            run.log(epoch=1, val_mae=4.0)
            run.finish(test={"mae": 3.5})
            meta = json.loads((run.dir / "run.json").read_text())
            self.assertEqual(meta["seed"], 3)
            self.assertIn("commit", meta["git"])
            self.assertIn("torch", meta["versions"])
            loaded = tracking.load_runs("test")
            self.assertEqual(loaded[0]["log"][0]["val_mae"], 4.0)
            self.assertEqual(tracking.load_runs("other"), [])


if __name__ == "__main__":
    unittest.main()
