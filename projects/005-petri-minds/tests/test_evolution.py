import unittest

import numpy as np

from petri_minds.brain import GENOME_SIZE
from petri_minds.evolution import Run, breed
from petri_minds.world import Config


class BreedTests(unittest.TestCase):
    def test_elite_survive_unchanged(self):
        rng = np.random.default_rng(0)
        genomes = rng.normal(size=(20, GENOME_SIZE))
        fitness = np.arange(20.0)
        children, hues = breed(genomes, fitness, np.zeros(20), rng, Config(elite=3))
        np.testing.assert_array_equal(children[:3], genomes[[19, 18, 17]])
        self.assertEqual(children.shape, genomes.shape)
        self.assertEqual(hues.shape, (20,))

    def test_children_mix_parents_genes(self):
        rng = np.random.default_rng(0)
        genomes = np.vstack((np.zeros((10, GENOME_SIZE)), np.ones((10, GENOME_SIZE))))
        config = Config(elite=0, mutation_rate=0)
        children, _ = breed(genomes, np.ones(20), np.zeros(20), rng, config)
        self.assertTrue(np.isin(children, (0, 1)).all())
        self.assertTrue(((children > 0) & (children < 1)).sum() == 0)


class RunTests(unittest.TestCase):
    small = Config(population=24, steps=300)

    def test_same_seed_repeats_the_run(self):
        a, b = Run(7, self.small), Run(7, self.small)
        for run in (a, b):
            run.run_generation()
            run.run_generation()
        self.assertEqual(a.history, b.history)
        np.testing.assert_array_equal(a.champion, b.champion)

    def test_generation_records_history_and_champion(self):
        run = Run(3, self.small)
        run.run_generation()
        self.assertEqual(run.generation, 2)
        self.assertEqual(run.history[0]["generation"], 1)
        self.assertEqual(run.world.time, 0)
        self.assertEqual(run.champion.shape, (GENOME_SIZE,))

    def test_evolution_improves_mean_fitness(self):
        run = Run(5, Config(population=40, steps=600))
        for _ in range(20):
            run.run_generation()
        means = [h["mean"] for h in run.history]
        self.assertGreater(np.mean(means[-5:]), 2 * np.mean(means[:5]))


if __name__ == "__main__":
    unittest.main()
