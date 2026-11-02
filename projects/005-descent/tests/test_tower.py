import unittest

import numpy as np

from descent.brain import GENOME_SIZE, N_INPUTS, random_genomes, think, unpack
from descent.tower import Config, Towers

SMALL = Config(towers=2, levels=4, days=6, month=3)


def towers_with(genomes=None, config=SMALL):
    rng = np.random.default_rng(1)
    if genomes is None:
        genomes = random_genomes(rng, config.towers, config.inmates)
    return Towers(genomes, rng, config)


def constant_appetite(portion, config=SMALL):
    """Genomes whose brains ask for the same portion whatever they see."""
    genomes = np.zeros((config.towers, config.inmates, GENOME_SIZE))
    genomes[..., -1] = np.arctanh(portion / config.max_portion * 2 - 1)
    return genomes


class BrainTests(unittest.TestCase):
    def test_genome_unpacks_into_layers(self):
        w1, b1, w2, b2 = unpack(np.zeros((3, GENOME_SIZE)))
        self.assertEqual(w1.shape, (3, N_INPUTS, 6))
        self.assertEqual(b2.shape, (3, 1))

    def test_each_inmate_uses_its_own_brain(self):
        genomes = np.zeros((2, GENOME_SIZE))
        genomes[1, -1] = 5
        _, outputs = think(genomes, np.zeros((2, N_INPUTS)))
        self.assertEqual(outputs[0, 0], 0)
        self.assertGreater(outputs[1, 0], .99)


class TowerTests(unittest.TestCase):
    def test_two_inmates_per_level(self):
        towers = towers_with()
        for t in range(SMALL.towers):
            self.assertEqual(sorted(towers.level[t]), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_platform_visits_every_level_then_reloads(self):
        towers = towers_with(constant_appetite(1.0))
        for _ in range(SMALL.levels):
            towers.tick()
        self.assertEqual(towers.day, 1)
        self.assertEqual(towers.stop, 0)
        self.assertTrue(np.allclose(towers.food, SMALL.feast))
        # Arrivals fall by two rations per level.
        np.testing.assert_allclose(towers.arrived[0], SMALL.feast - 2 * np.arange(SMALL.levels))

    def test_greedy_top_starves_the_bottom(self):
        towers = towers_with(constant_appetite(3.0))
        towers.run_day()
        np.testing.assert_allclose(towers.arrived[0], [SMALL.feast, SMALL.feast - 6, 0, 0], atol=1e-9)
        bottom = towers.level[0] == SMALL.levels - 1
        self.assertTrue((towers.energy[0, bottom] == SMALL.start_energy - 1).all())

    def test_overdemand_is_split_in_proportion(self):
        towers = towers_with(constant_appetite(3.0))
        towers.food[:] = 3.0
        towers.tick()
        top = towers.level[0] == 0
        np.testing.assert_allclose(towers.energy[0, top], SMALL.start_energy + 1.5)
        self.assertEqual(towers.food[0], 0)

    def test_starving_inmates_die_and_stop_ageing(self):
        towers = towers_with(constant_appetite(0.0))
        while not towers.done:
            towers.run_day()
        self.assertFalse(towers.alive.any())
        self.assertTrue((towers.age == SMALL.start_energy - 1).all())

    def test_levels_reshuffle_each_month(self):
        config = Config(towers=1, levels=10, days=6, month=3)
        towers = towers_with(constant_appetite(1.0, config), config)
        before = towers.level.copy()
        for _ in range(3):
            towers.run_day()
        self.assertFalse(np.array_equal(before, towers.level))

    def test_fair_brains_keep_everyone_alive(self):
        towers = towers_with(constant_appetite(1.0))
        while not towers.done:
            towers.run_day()
        self.assertTrue(towers.alive.all())
        np.testing.assert_allclose(towers.tower_scores(), 1)

    def test_sensors_are_bounded(self):
        config = Config(towers=3)
        towers = towers_with(config=config)
        for _ in range(200):
            towers.tick()
        seen = towers.inputs[towers.portion > 0]
        self.assertTrue(len(seen))
        self.assertTrue(((seen >= 0) & (seen <= 1)).all())


if __name__ == "__main__":
    unittest.main()
