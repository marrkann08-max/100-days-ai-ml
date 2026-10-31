from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from opcode_canvas.presets import PRESETS
from opcode_canvas.visuals import rgb, export_gif, PALETTE


class ExportTests(unittest.TestCase):
    def test_palette_preserves_pixel_positions(self):
        pixels = bytes([0, 255, 31, 72])
        image = rgb(pixels, 2, 2)
        np.testing.assert_array_equal(image[1, 0], PALETTE[31])

    def test_gif_contains_animated_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "test.gif"
            export_gif(PRESETS[0].source, target, frames=4)
            with Image.open(target) as gif:
                self.assertEqual(gif.n_frames, 4)
                self.assertEqual(gif.size, (512, 320))
                first = gif.convert("RGB").tobytes()
                gif.seek(1)
                self.assertNotEqual(first, gif.convert("RGB").tobytes())


if __name__ == "__main__":
    unittest.main()
