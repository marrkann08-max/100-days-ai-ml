import unittest

import numpy as np

from descent.brain import GENOME_SIZE
from descent.evolution import Run, breed
from descent.tower import Config


class BreedTests(unittest.TestCase):
    def test_elite_survive_unchanged(self):
        rng = np.random.default_rng(0)
        genomes = rng.normal(size=(20, GENOME_SIZE))
        children = breed(genomes, np.arange(20.0), rng, Config(elite=3))
        np.testing.assert_array_equal(children[:3], genomes[[19, 18, 17]])
        self.assertEqual(children.shape, genomes.shape)

    def test_children_mix_parents_genes(self):
        rng = np.random.default_rng(0)
        genomes = np.vstack((np.zeros((10, GENOME_SIZE)), np.ones((10, GENOME_SIZE))))
        children = breed(genomes, np.ones(20), rng, Config(elite=0, mutation_rate=0))
        self.assertTrue(np.isin(children, (0, 1)).all())

    def test_full_chance_ignores_fitness(self):
        rng = np.random.default_rng(0)
        genomes = np.arange(40.0).repeat(GENOME_SIZE).reshape(40, GENOME_SIZE)
        fitness = np.arange(40.0)
        config = Config(mutation_rate=0)
        picky = breed(genomes, fitness, rng, config, chance=0).mean()
        random = breed(genomes, fitness, rng, config, chance=1).mean()
        self.assertGreater(picky, random + 5)


class RunTests(unittest.TestCase):
    small = Config(towers=4, levels=6, days=20, month=10)

    def test_same_seed_repeats_the_run(self):
        a, b = Run(7, self.small), Run(7, self.small)
        for run in (a, b):
            run.run_generation()
            run.run_generation()
        self.assertEqual(a.history, b.history)
        np.testing.assert_array_equal(a.genomes, b.genomes)

    def test_solidarity_copies_best_towers_over_worst(self):
        run = Run(3, self.small, solidarity=1.0)
        self.assertEqual(run.replaced(), 2)
        self.assertEqual(Run(3, self.small).replaced(), 0)
        run.run_generation()
        self.assertEqual(run.generation, 2)
        self.assertEqual(run.history[0]["solidarity"], 1.0)

    def test_without_solidarity_greed_evolves(self):
        run = Run(1)
        for _ in range(25):
            run.run_generation()
        greed = [h["greed_top"] for h in run.history]
        self.assertGreater(np.mean(greed[-5:]), np.mean(greed[:5]) + .2)
        self.assertGreater(np.mean(greed[-5:]), 2.2)


if __name__ == "__main__":
    unittest.main()
