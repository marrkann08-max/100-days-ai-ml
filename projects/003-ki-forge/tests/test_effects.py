import unittest
import numpy as np
from ki_forge.effects import Renderer
from ki_forge.gestures import Energy


class BlastTests(unittest.TestCase):
    def test_beam_connects_origin_to_head_and_expires(self):
        renderer = Renderer(seed=1)
        renderer.blast((200, 240), (1, 0), 40)
        background = np.zeros((480, 640, 3), dtype=np.uint8)
        energy = Energy()
        for _ in range(9):
            frame = renderer.draw(background, energy, 1/30)
        self.assertGreater(int(frame[240, 300].min()), 180)
        for _ in range(40):
            renderer.draw(background, energy, 1/30)
        self.assertEqual(renderer.blasts, [])
        self.assertEqual(renderer.particles, [])


if __name__ == '__main__':
    unittest.main()
