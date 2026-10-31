from pathlib import Path

import numpy as np
from PIL import Image

from .engine import PixelProgram, WIDTH, HEIGHT


def palette():
    anchors = np.array([[13, 21, 28], [27, 69, 82], [71, 139, 137],
                        [194, 235, 180], [250, 196, 115], [235, 94, 66],
                        [76, 37, 62], [13, 21, 28]], dtype=float)
    x = np.linspace(0, len(anchors) - 1, 256)
    return np.stack([np.interp(x, np.arange(len(anchors)), anchors[:, c])
                     for c in range(3)], axis=1).astype(np.uint8)


PALETTE = palette()


def rgb(pixels: bytes, width=WIDTH, height=HEIGHT):
    return PALETTE[np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)]


def export_gif(source: str, path: Path, start_tick=0, frames=48):
    program = PixelProgram(source)
    images = []
    for i in range(frames):
        frame = program.render(start_tick + i * 2)
        image = Image.fromarray(rgb(frame.pixels)).resize((WIDTH * 4, HEIGHT * 4),
                                                        Image.Resampling.NEAREST)
        images.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=70,
                   loop=0, optimize=False)
    return path
