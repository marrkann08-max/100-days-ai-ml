import unittest

import numpy as np

from petri_minds.brain import GENOME_SIZE, N_INPUTS, N_OUTPUTS, random_genomes, think, unpack
from petri_minds.world import Config, World, scatter, wall_distance


def world_with(population=2, **changes):
    config = Config(population=population, **changes)
    rng = np.random.default_rng(1)
    return World(random_genomes(rng, population), rng, config)


class BrainTests(unittest.TestCase):
    def test_genome_unpacks_into_layers(self):
        genomes = np.arange(3 * GENOME_SIZE, dtype=float).reshape(3, GENOME_SIZE)
        w1, b1, w2, b2 = unpack(genomes)
        self.assertEqual(w1.shape, (3, N_INPUTS, 8))
        self.assertEqual(b2.shape, (3, N_OUTPUTS))
        self.assertEqual(b2[0, -1], GENOME_SIZE - 1)

    def test_each_creature_uses_its_own_brain(self):
        genomes = np.zeros((2, GENOME_SIZE))
        genomes[1, -2] = 5  # turn bias of the second brain only
        _, outputs = think(genomes, np.zeros((2, N_INPUTS)))
        self.assertEqual(outputs[0, 0], 0)
        self.assertGreater(outputs[1, 0], .99)


class WorldTests(unittest.TestCase):
    def test_scatter_stays_inside_disc(self):
        points = scatter(np.random.default_rng(0), 500, 50)
        self.assertTrue((np.hypot(points[:, 0], points[:, 1]) <= 50).all())

    def test_wall_distance_from_center_and_edge(self):
        pos = np.array([[0.0, 0.0], [90.0, 0.0], [90.0, 0.0]])
        angle = np.array([1.0, 0.0, np.pi])
        np.testing.assert_allclose(wall_distance(pos, angle, 100), [100, 10, 190])

    def test_sensors_are_bounded(self):
        world = world_with(population=40)
        for _ in range(50):
            world.step()
        inputs = world.sense()
        self.assertEqual(inputs.shape, (40, N_INPUTS))
        self.assertTrue((np.abs(inputs) <= 2).all())

    def test_food_points_toward_nearest_pellet(self):
        world = world_with(population=1)
        world.pos[:] = 0
        world.heading[:] = 0
        world.food = np.array([[0.0, 50.0], [200.0, 0.0]])
        sin, cos, near = world.sense()[0, :3]
        self.assertAlmostEqual(sin, 1)
        self.assertAlmostEqual(cos, 0, places=6)
        self.assertGreater(near, 0)

    def test_pellet_goes_to_closest_creature(self):
        world = world_with(population=2)
        world.pos = np.array([[0.0, 0.0], [5.0, 0.0]])
        world.food = np.array([[4.0, 0.0], [100.0, 100.0]])
        gained = world.consume(world.food, world.eaten)
        self.assertEqual(list(gained), [0, 1])
        self.assertFalse(np.allclose(world.food[0], [4, 0]))  # respawned

    def test_touching_the_wall_kills(self):
        world = world_with(population=1)
        world.pos[:] = [world.config.radius - 1, 0]
        world.step()
        self.assertFalse(world.alive[0])
        self.assertTrue(world.done)

    def test_poison_costs_energy_and_fitness(self):
        world = world_with(population=1, food=1, poison=1)
        world.pos[:] = 0
        world.heading[:] = 0
        world.genomes[:] = 0  # straight ahead at half speed
        world.food[:] = [200, 0]
        world.poison[:] = [3, 0]
        world.step()
        self.assertEqual(world.poisoned[0], 1)
        self.assertLess(world.energy[0], world.config.start_energy - 40)
        self.assertLess(world.fitness()[0], 0)


if __name__ == "__main__":
    unittest.main()
