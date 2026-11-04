import unittest

import numpy as np

from primordial import physics
from primordial.discover import Discovery
from primordial.rules import random_rules


class GpuTests(unittest.TestCase):
    def setUp(self):
        try:
            import moderngl
            self.ctx = moderngl.create_standalone_context(require=430)
        except Exception as error:
            self.skipTest(f"no OpenGL 4.3: {error}")
        from primordial.world import World
        self.World = World

    def tearDown(self):
        self.ctx.release()

    def test_gpu_matches_numpy_reference(self):
        rules = random_rules(np.random.default_rng(3))
        world = self.World(self.ctx, rules, particles=700, seed=1)
        pos, vel, kinds = world.positions(), world.velocities(), world.kinds()
        for _ in range(5):
            pos, vel = physics.step(pos, vel, kinds, rules, world.size)
        world.step(5)
        np.testing.assert_allclose(world.positions(), pos, atol=1e-5)
        np.testing.assert_allclose(world.velocities(), vel, atol=1e-4)

    def test_stirring_pushes_particles_away(self):
        rules = random_rules(np.random.default_rng(4))
        rules.matrix[:] = 0
        world = self.World(self.ctx, rules, particles=1000, seed=2)
        centre = world.size / 2
        world.stir(*centre, 6.0, .2)
        before = np.hypot(*(world.positions() - centre).T)
        world.step(10)
        after = np.hypot(*(world.positions() - centre).T)
        near = before < .15
        self.assertGreater((after[near] - before[near]).mean(), 0)

    def test_changing_type_count_rescatters(self):
        rng = np.random.default_rng(5)
        world = self.World(self.ctx, random_rules(rng, 4), particles=500, seed=3)
        world.set_rules(random_rules(rng, 7))
        self.assertLess(world.kinds().max(), 7)
        self.assertGreaterEqual(world.kinds().max(), 4)

    def test_discovery_ranks_and_improves_on_random(self):
        d = Discovery(self.ctx, seed=7, population=8, generations=3, elite=2, fresh=2,
                      particles=1500, warm=300, watch=100)
        for _ in d.run():
            pass
        best = d.best(3)
        self.assertEqual(len(d.history), 3)
        self.assertEqual(d.evaluated, 8 + 2 * 6)
        self.assertGreaterEqual(best[0].score, best[-1].score)
        self.assertGreaterEqual(d.history[-1]["best"], d.history[0]["best"])
        self.assertIn("clusters", best[0].notes)
        d.release()


if __name__ == "__main__":
    unittest.main()
