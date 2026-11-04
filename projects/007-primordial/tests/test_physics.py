import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from primordial import physics
from primordial.lifeness import clusters, measure, score
from primordial.rules import Rules, crossover, mutate, random_rules, species_name


class ForceTests(unittest.TestCase):
    def test_close_particles_always_repel(self):
        for a in (-1, 0, 1):
            self.assertAlmostEqual(float(physics.force(0, a, .3)), -1)
            self.assertLess(float(physics.force(.15, a, .3)), 0)

    def test_attraction_peaks_mid_range_and_vanishes_past_radius(self):
        beta = .3
        peak = (1 + beta) / 2
        self.assertAlmostEqual(float(physics.force(peak, .8, beta)), .8)
        self.assertAlmostEqual(float(physics.force(1.0, .8, beta)), 0)
        self.assertEqual(float(physics.force(1.5, .8, beta)), 0)
        self.assertAlmostEqual(float(physics.force(peak, -.5, beta)), -.5)

    def test_wrap_takes_the_short_way_round(self):
        size = np.array([1.0, .5])
        np.testing.assert_allclose(physics.wrap(np.array([.9, .4]), size), [-.1, -.1])

    def test_world_size_keeps_density(self):
        a, b = physics.world_size(8000), physics.world_size(2000)
        self.assertAlmostEqual(8000 / np.prod(a), 2000 / np.prod(b))
        self.assertAlmostEqual(a[0] / a[1], 16 / 9)


class StepTests(unittest.TestCase):
    def test_two_attracting_particles_approach(self):
        rules = Rules(np.array([[1.0]]))
        pos = np.array([[.5, .5], [.58, .5]])
        out, vel = physics.step(pos, np.zeros_like(pos), np.zeros(2, int), rules, (1, 1))
        self.assertLess(out[1, 0] - out[0, 0], .08)
        self.assertGreater(vel[0, 0], 0)

    def test_neutral_far_particles_stay_put(self):
        rules = Rules(np.zeros((1, 1)))
        pos = np.array([[.1, .1], [.6, .6]])
        out, _ = physics.step(pos, np.zeros_like(pos), np.zeros(2, int), rules, (1, 1))
        np.testing.assert_allclose(out, pos)

    def test_forces_reach_across_the_edge(self):
        rules = Rules(np.array([[1.0]]))
        pos = np.array([[.01, .5], [.95, .5]])
        _, vel = physics.step(pos, np.zeros_like(pos), np.zeros(2, int), rules, (1, 1))
        self.assertLess(vel[0, 0], 0)   # pulled left, through the wrap
        self.assertGreater(vel[1, 0], 0)

    def test_positions_stay_inside_the_world(self):
        rng = np.random.default_rng(0)
        rules = random_rules(rng, 4)
        size = np.array([1.2, .7])
        pos = rng.random((200, 2)) * size
        vel = rng.normal(0, 2, (200, 2))
        for _ in range(5):
            pos, vel = physics.step(pos, vel, rng.integers(0, 4, 200), rules, size)
        self.assertTrue(((pos >= 0) & (pos < size)).all())


class RulesTests(unittest.TestCase):
    def test_json_round_trip(self):
        rules = random_rules(np.random.default_rng(1), 5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.json"
            rules.save(path)
            loaded = Rules.load(path)
        np.testing.assert_allclose(loaded.matrix, rules.matrix, atol=1e-4)
        self.assertEqual(loaded.name, rules.name)

    def test_names_are_repeatable(self):
        m = np.random.default_rng(2).uniform(-1, 1, (6, 6))
        self.assertEqual(species_name(m), species_name(m.copy()))
        self.assertTrue(species_name(m).isalpha())

    def test_mutation_stays_in_range_and_renames(self):
        rng = np.random.default_rng(3)
        rules = random_rules(rng)
        child = mutate(rules, rng, size=3, rate=1)
        self.assertTrue((np.abs(child.matrix) <= 1).all())
        self.assertNotEqual(child.fingerprint(), rules.fingerprint())

    def test_crossover_takes_whole_rows(self):
        rng = np.random.default_rng(4)
        a, b = Rules(np.ones((6, 6))), Rules(-np.ones((6, 6)))
        child = crossover(a, b, rng)
        for row in child.matrix:
            self.assertEqual(len(set(row)), 1)

    def test_bundled_species_load(self):
        folder = Path(__file__).resolve().parent.parent / "species"
        files = sorted(folder.glob("*.json"))
        self.assertGreaterEqual(len(files), 8)
        for path in files:
            rules = Rules.load(path)
            self.assertEqual(rules.matrix.shape, (rules.types, rules.types))
            self.assertGreater(rules.score, 0)
            json.loads(path.read_text())


class LifenessTests(unittest.TestCase):
    size = np.array([16 / 9, 1.0])

    def blobs(self, rng, centres, per=400, spread=.015):
        pos = np.concatenate([c + rng.normal(0, spread, (per, 2)) for c in centres]) % self.size
        return pos

    def test_clusters_wrap_around_edges(self):
        dense = np.zeros((6, 4), bool)
        dense[0, 1] = dense[5, 1] = True   # neighbours across the edge
        dense[3, 3] = True
        self.assertEqual(clusters(dense.ravel(), (6, 4)), 2)

    def test_gas_scores_near_zero(self):
        rng = np.random.default_rng(5)
        a = rng.random((3000, 2)) * self.size
        b = rng.random((3000, 2)) * self.size
        m = measure(a, b, rng.integers(0, 6, 3000), self.size, .1, 6, 200)
        self.assertLess(m["clumping"], 1.5)
        self.assertLess(score(m), .05)

    def test_moving_mixed_organisms_beat_frozen_ones(self):
        rng = np.random.default_rng(6)
        centres = rng.random((6, 2)) * self.size
        before = self.blobs(rng, centres)
        moved = (before + (.15, .05)) % self.size
        kinds = rng.integers(0, 6, len(before))
        alive = score(measure(before, moved, kinds, self.size, .1, 6, 200))
        frozen = score(measure(before, before, kinds, self.size, .1, 6, 200))
        single = score(measure(before, moved, np.zeros_like(kinds), self.size, .1, 6, 200))
        self.assertGreater(alive, .4)
        self.assertGreater(alive, frozen * 1.5)
        self.assertGreater(alive, single)

    def test_one_giant_blob_is_penalised(self):
        rng = np.random.default_rng(7)
        one = self.blobs(rng, [self.size / 2], per=2400, spread=.04)
        many = self.blobs(rng, rng.random((6, 2)) * self.size)
        kinds = rng.integers(0, 6, 2400)
        moved = lambda p: (p + .1) % self.size
        self.assertLess(score(measure(one, moved(one), kinds, self.size, .1, 6, 200)),
                        score(measure(many, moved(many), kinds, self.size, .1, 6, 200)))


if __name__ == "__main__":
    unittest.main()
