import tempfile
import unittest
from pathlib import Path

import numpy as np

from holo_window.config import Settings
from holo_window.depth import normalise, photo_mesh, photo_rect
from holo_window.geometry import off_axis
from holo_window.scenes import BACKGROUND, room_mesh
from holo_window.tracker import DEFAULT_EYE, MouseTracker, ease_home

SETTINGS = Settings(screen_width=34, screen_height=19)


def project(matrix, points):
    clip = np.c_[points, np.ones(len(points))] @ matrix.T
    return clip[:, :2] / clip[:, 3:]


class DepthTests(unittest.TestCase):
    def test_normalise_spans_zero_to_one(self):
        d = normalise(np.linspace(-5, 20, 1000).reshape(20, 50))
        self.assertAlmostEqual(float(d.min()), 0)
        self.assertAlmostEqual(float(d.max()), 1)

    def test_photo_rect_covers_the_window(self):
        for aspect in (.5, 1.0, 16 / 9, 3.0):
            w, h = photo_rect(aspect, 34, 19)
            self.assertGreaterEqual(w, 34)
            self.assertGreaterEqual(h, 19)
            self.assertAlmostEqual(w / h, aspect)

    def test_near_pixels_are_near_and_far_pixels_far(self):
        d = np.zeros((40, 64), np.float32)
        d[:, 32:] = 1
        vertices, _ = photo_mesh(d, 1.6, SETTINGS, columns=64)
        z = vertices[:, 2].reshape(-1, 64)
        self.assertAlmostEqual(float(z[:, 0].mean()), -SETTINGS.photo_far, places=3)
        self.assertAlmostEqual(float(z[:, -1].mean()), -SETTINGS.photo_near, places=3)

    def test_photo_looks_flat_from_the_centre(self):
        # From the centred viewer every vertex lands where the flat photo would be.
        rng = np.random.default_rng(1)
        d = rng.random((40, 64)).astype(np.float32)
        vertices, _ = photo_mesh(d, 1.6, SETTINGS, columns=64)
        flat = d.copy()
        flat[:] = 1
        reference, _ = photo_mesh(flat, 1.6, SETTINGS, columns=64)
        m = off_axis(DEFAULT_EYE, 34, 19)
        np.testing.assert_allclose(project(m, vertices[:, :3]), project(m, reference[:, :3]), atol=1e-4)

    def test_depth_jumps_tear_and_backdrop_fills(self):
        d = np.zeros((40, 64), np.float32)
        d[:, 32:] = 1
        _, torn = photo_mesh(d, 1.6, SETTINGS, columns=64)
        smooth = np.full_like(d, .5)
        _, whole = photo_mesh(smooth, 1.6, SETTINGS, columns=64)
        self.assertLess(len(torn), len(whole))
        backdrop, filled = photo_mesh(d, 1.6, SETTINGS, columns=64, backdrop=True)
        self.assertEqual(len(filled), len(whole))
        self.assertTrue((backdrop[:, 2] < -SETTINGS.photo_far).all())


class TrackerTests(unittest.TestCase):
    def test_ease_home_waits_then_returns(self):
        eye = np.array([10.0, 5.0, 40.0])
        np.testing.assert_allclose(ease_home(eye, .2), eye)
        np.testing.assert_allclose(ease_home(eye, 5), DEFAULT_EYE)

    def test_mouse_tracker_maps_window_to_head_position(self):
        tracker = MouseTracker(SETTINGS, (800, 600))
        tracker.move((400, 300))
        np.testing.assert_allclose(tracker.eye(), DEFAULT_EYE)
        tracker.move((800, 0))
        x, y, _ = tracker.eye()
        self.assertGreater(x, 0)
        self.assertGreater(y, 0)
        tracker.zoom(100)
        self.assertEqual(tracker.eye()[2], 20)


class SettingsTests(unittest.TestCase):
    def test_round_trip_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            self.assertIsNone(Settings.load(path))
            Settings(fov=48.5, screen_width=30).save(path)
            path.write_text(path.read_text().replace("{", '{"old": 1,', 1))
            loaded = Settings.load(path)
            self.assertEqual(loaded.fov, 48.5)
            self.assertEqual(loaded.screen_width, 30)
            self.assertEqual(loaded.camera_y, loaded.screen_height / 2 + loaded.camera_offset)


class RenderTests(unittest.TestCase):
    def setUp(self):
        try:
            import moderngl
            self.ctx = moderngl.create_standalone_context(require=330)
        except Exception as error:
            self.skipTest(f"no OpenGL: {error}")
        from holo_window.__main__ import Renderer
        size = (160, 90)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(size, 3)],
                                        depth_attachment=self.ctx.depth_renderbuffer(size))
        self.renderer = Renderer(self.ctx, SETTINGS)

    def tearDown(self):
        self.ctx.release()

    def frame(self, scene, eye):
        self.fbo.use()
        self.renderer.render(scene, eye, (34, 19))
        return np.frombuffer(self.fbo.read(components=3), np.uint8).reshape(90, 160, 3)

    def test_room_mesh_is_triangles(self):
        data = room_mesh(34, 19)
        self.assertEqual(data.shape[1], 6)
        self.assertEqual(len(data) % 3, 0)

    def test_room_renders_and_changes_with_the_eye(self):
        a = self.frame(0, DEFAULT_EYE)
        b = self.frame(0, DEFAULT_EYE + (12, 0, 0))
        background = np.round(np.array(BACKGROUND) * 255)
        self.assertGreater(np.abs(a.astype(int) - background).sum(axis=2).astype(bool).mean(), .5)
        self.assertGreater(np.abs(a.astype(int) - b).mean(), 2)

    def test_photo_scene_fills_the_window(self):
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)
        index = self.renderer.add_photo("noise", rgb, rng.random((90, 160)).astype(np.float32))
        for x in (-14, 0, 14):
            image = self.frame(index, DEFAULT_EYE + (x, 0, 0))
            background = np.round(np.array(BACKGROUND) * 255)
            empty = (np.abs(image.astype(int) - background).sum(axis=2) < 3).mean()
            self.assertLess(empty, .02)
        self.renderer.rebuild(index)
        self.frame(index, DEFAULT_EYE)


if __name__ == "__main__":
    unittest.main()
