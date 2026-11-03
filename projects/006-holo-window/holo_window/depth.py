import os

import numpy as np

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


class DepthEstimator:
    """Depth Anything V2 Small through ONNX Runtime on the CPU."""

    def __init__(self, model_path, long_side=518):
        import onnxruntime as ort
        options = ort.SessionOptions()
        # Leave half the cores for rendering and face tracking so the window never stutters.
        options.intra_op_num_threads = max(2, (os.cpu_count() or 4) // 2)
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.long_side = long_side

    def input_size(self, width, height):
        """Keep the photo's aspect; the model needs multiples of 14."""
        scale = self.long_side / max(width, height)
        return max(14, round(width * scale / 14) * 14), max(14, round(height * scale / 14) * 14)

    def __call__(self, rgb):
        import cv2
        w, h = self.input_size(rgb.shape[1], rgb.shape[0])
        x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        disparity = self.session.run(None, {self.input: x})[0][0]
        return normalise(disparity)


def normalise(disparity):
    """Relative disparity to 0 (far) .. 1 (near), ignoring the extreme 2% at each end."""
    lo, hi = np.percentile(disparity, (2, 98))
    return np.clip((disparity - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def photo_rect(aspect, width, height, overscan=1.12):
    """Width and height of the photo on the glass: covers the window with a margin to look around."""
    if aspect > width / height:
        h = height * overscan
        return h * aspect, h
    w = width * overscan
    return w, w / aspect


def photo_mesh(disparity, aspect, settings, view=60.0, columns=256, tear=1.3, backdrop=False):
    """Displaced grid for a photo: (vertices[x, y, z, u, v], triangle indices).

    Each pixel is pushed back along the ray from a centred viewer, so from
    the middle the photo looks flat and untouched; move your head and the
    near parts slide over the far parts. Triangles spanning a big depth jump
    are dropped instead of being stretched into rubber sheets.

    With backdrop=True every pixel goes just past the far distance instead,
    giving a flat copy that fills those holes with plausible background.
    """
    import cv2
    rows = max(2, round(columns / aspect))
    d = cv2.resize(disparity, (columns, rows), interpolation=cv2.INTER_AREA)
    near, far = settings.photo_near, settings.photo_far
    depth = np.full_like(d, far * 1.05) if backdrop else 1 / (d / near + (1 - d) / far)

    # The backdrop reaches well past the photo; the texture clamps, smearing the edge pixels outwards.
    span = (-1.0, 2.0) if backdrop else (0.0, 1.0)
    u, v = np.meshgrid(np.linspace(*span, columns), np.linspace(*span, rows))
    rw, rh = photo_rect(aspect, settings.screen_width, settings.screen_height)
    px, py = (u - .5) * rw, (.5 - v) * rh
    t = (view + depth) / view
    vertices = np.stack((px * t, py * t, -depth, u, v), axis=-1).reshape(-1, 5).astype(np.float32)

    index = np.arange(rows * columns).reshape(rows, columns)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, e = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    flat = depth.ravel()
    corners = np.stack((flat[a], flat[b], flat[c], flat[e]))
    keep = backdrop | (corners.max(axis=0) / corners.min(axis=0) < tear)
    triangles = np.concatenate((np.stack((a, c, b), 1)[keep], np.stack((b, c, e), 1)[keep]))
    return vertices, triangles.astype(np.uint32)
