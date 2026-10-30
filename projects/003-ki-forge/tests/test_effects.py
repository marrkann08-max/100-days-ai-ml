import unittest
import numpy as np
from ki_forge.effects import Renderer
from ki_forge.gestures import Energy


class BlastTests(unittest.TestCase):
    def test_burst_stays_centered_and_expires(self):
        renderer = Renderer(seed=1)
        renderer.blast((320, 240), 40)
        background = np.zeros((480, 640, 3), dtype=np.uint8)
        energy = Energy()
        for _ in range(9):
            frame = renderer.draw(background, energy, 1/30)
        self.assertEqual(renderer.blasts[0][:2], [320, 240])
        self.assertGreater(int(frame.sum()), 0)
        # After the central flash, the center clears instead of moving sideways.
        self.assertLess(int(frame[240, 320].max()), 10)
        for _ in range(40):
            renderer.draw(background, energy, 1/30)
        self.assertEqual(renderer.blasts, [])
        self.assertEqual(renderer.particles, [])

    def test_glow_keeps_detail_on_a_bright_camera_frame(self):
        renderer = Renderer(seed=1)
        energy = Energy()
        energy.center, energy.charge = (320, 240), 1
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        result = renderer.draw(frame, energy, 1/30)
        self.assertLess(int(result.max()), 255)


if __name__ == '__main__':
    unittest.main()
