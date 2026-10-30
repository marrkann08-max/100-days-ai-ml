import unittest
import numpy as np
from ki_forge.effects import Renderer
from ki_forge.gestures import Energy
from ki_forge.audio import burst_samples, Sound


class ParticleTests(unittest.TestCase):
    def setUp(self):
        self.renderer = Renderer(seed=1)
        self.frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        self.energy = Energy()
        self.energy.center, self.energy.charge, self.energy.radius = (320, 240), 1, 65
        for _ in range(30):
            self.renderer.draw(self.frame, self.energy, 1/30)

    def draw(self, count):
        for _ in range(count):
            self.renderer.draw(self.frame, self.energy, 1/60)

    def test_same_particles_compress_then_expand(self):
        positions = self.renderer.positions
        before = np.linalg.norm(positions, axis=1).mean()
        self.renderer.blast((320, 240), 65)
        self.energy.charge = 0
        self.draw(10)
        compressed = np.linalg.norm(positions, axis=1).mean()
        self.assertLess(compressed, before*.4)
        self.draw(20)
        self.assertEqual(self.renderer.phase, 'burst')
        self.assertIs(self.renderer.positions, positions)
        self.assertGreater(np.linalg.norm(positions, axis=1).mean(), before)
        np.testing.assert_array_equal(self.renderer.origin, [320, 240])
        self.assertTrue(self.renderer.consume_sound())
        self.assertFalse(self.renderer.consume_sound())

    def test_burst_expires_and_reset_clears_it(self):
        self.renderer.blast((320, 240), 65)
        self.energy.charge = 0
        self.draw(110)
        self.assertEqual(self.renderer.phase, 'idle')
        self.assertEqual(self.renderer.visibility, 0)
        self.renderer.blast((320, 240), 65)
        self.renderer.reset()
        self.assertIsNone(self.renderer.label)
        self.assertFalse(self.renderer.consume_sound())

    def test_charge_does_not_replace_particles_during_explosion(self):
        self.renderer.blast((320, 240), 65)
        self.energy.center = (500, 100)
        self.draw(30)
        self.assertEqual(self.renderer.phase, 'burst')
        np.testing.assert_array_equal(self.renderer.origin, [320, 240])

    def test_bright_camera_detail_is_retained(self):
        result = self.renderer.draw(np.full_like(self.frame, 200), self.energy, 1/30)
        self.assertLess(int(result.max()), 255)

    def test_sound_is_finite_bounded_and_muting_skips_device(self):
        samples = burst_samples()
        self.assertEqual(samples.dtype, np.float32)
        self.assertTrue(np.isfinite(samples).all())
        self.assertLessEqual(float(np.abs(samples).max()), .66)
        sound = Sound(enabled=False)
        sound.play()
        self.assertIsNone(sound.device)


if __name__ == '__main__':
    unittest.main()
