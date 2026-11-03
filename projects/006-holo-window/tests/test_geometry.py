import math
import unittest

import numpy as np

from holo_window.config import Settings
from holo_window.geometry import (LEFT_IRIS, RIGHT_IRIS, OneEuro, eye_from_landmarks, focal_length,
                                  fov_for, iris_gap, off_axis)


def project(matrix, point):
    clip = matrix @ np.append(point, 1.0)
    return clip[:3] / clip[3]


def face(u, v, gap_px, size=(640, 480)):
    """478 landmarks with only the iris centres set: midpoint (u, v) in pixels, gap_px apart."""
    w, h = size
    points = np.zeros((478, 3))
    points[LEFT_IRIS] = ((u - gap_px / 2) / w, v / h, 0)
    points[RIGHT_IRIS] = ((u + gap_px / 2) / w, v / h, 0)
    return points


class OffAxisTests(unittest.TestCase):
    width, height = 34.0, 19.0

    def test_screen_corners_stay_on_the_window_edges(self):
        # The defining property: wherever the eye is, the physical screen edges map to the image edges.
        for eye in ((0, 0, 60), (-15, 8, 45), (20, -10, 90), (3, 12, 25)):
            m = off_axis(eye, self.width, self.height)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    ndc = project(m, (sx * self.width / 2, sy * self.height / 2, 0))
                    np.testing.assert_allclose(ndc[:2], (sx, sy), atol=1e-9)

    def test_centred_eye_gives_a_symmetric_frustum(self):
        m = off_axis((0, 0, 60), self.width, self.height)
        self.assertAlmostEqual(m[0, 2], 0)
        self.assertAlmostEqual(m[1, 2], 0)

    def test_parallax_like_a_real_window(self):
        # Like the moon following you: things behind the glass move with your head,
        # things floating in front of it move the other way.
        left, right = off_axis((-10, 0, 60), self.width, self.height), off_axis((10, 0, 60), self.width, self.height)
        behind, front = np.array([0, 0, -40.0]), np.array([0, 0, 15.0])
        self.assertLess(project(left, behind)[0], 0)
        self.assertGreater(project(right, behind)[0], 0)
        self.assertGreater(project(left, front)[0], 0)
        self.assertLess(project(right, front)[0], 0)

    def test_points_on_the_line_of_sight_stay_put(self):
        eye = np.array([-8.0, 5.0, 50.0])
        target = np.array([4.0, -3.0, 0.0])
        m = off_axis(eye, self.width, self.height)
        a = project(m, target)
        b = project(m, eye + (target - eye) * 2.5)
        np.testing.assert_allclose(a[:2], b[:2], atol=1e-9)

    def test_eye_never_reaches_the_glass(self):
        m = off_axis((0, 0, 0), self.width, self.height)
        self.assertTrue(np.isfinite(m).all())


class EyeTests(unittest.TestCase):
    settings = Settings(fov=60, ipd=6.3)

    def test_centred_face_distance_from_iris_gap(self):
        f = focal_length(640, 60)
        eye = eye_from_landmarks(face(320, 240, 100), (640, 480), self.settings)
        self.assertAlmostEqual(eye[0], 0)
        self.assertAlmostEqual(eye[1], self.settings.camera_y)
        self.assertAlmostEqual(eye[2], f * 6.3 / 100)

    def test_smaller_gap_means_further_away(self):
        near = eye_from_landmarks(face(320, 240, 120), (640, 480), self.settings)[2]
        far = eye_from_landmarks(face(320, 240, 60), (640, 480), self.settings)[2]
        self.assertAlmostEqual(far, near * 2)

    def test_camera_image_is_mirrored(self):
        # Moving to your right moves your face towards the left of the camera image.
        eye = eye_from_landmarks(face(200, 240, 100), (640, 480), self.settings)
        self.assertGreater(eye[0], 0)

    def test_lower_in_image_is_lower_in_world(self):
        high = eye_from_landmarks(face(320, 100, 100), (640, 480), self.settings)[1]
        low = eye_from_landmarks(face(320, 400, 100), (640, 480), self.settings)[1]
        self.assertGreater(high, low)

    def test_head_turn_does_not_change_gap(self):
        points = face(320, 240, 100)
        flat = iris_gap(points, (640, 480))
        points[RIGHT_IRIS, 0] -= .03
        points[RIGHT_IRIS, 2] = math.sqrt((100 / 640) ** 2 - (100 / 640 - .03) ** 2)
        self.assertAlmostEqual(iris_gap(points, (640, 480)), flat, places=6)

    def test_fov_round_trip(self):
        self.assertAlmostEqual(fov_for(640, focal_length(640, 57.5)), 57.5)


class OneEuroTests(unittest.TestCase):
    def test_constant_input_passes_through(self):
        f = OneEuro()
        for i in range(30):
            out = f([1.0, 2.0, 3.0], i / 30)
        np.testing.assert_allclose(out, (1, 2, 3))

    def test_reduces_jitter_when_still(self):
        rng = np.random.default_rng(0)
        f = OneEuro()
        noisy = 60 + rng.normal(0, .5, (300, 3))
        smooth = np.array([f(x, i / 30) for i, x in enumerate(noisy)])
        self.assertLess(smooth[100:].std(), noisy[100:].std() / 2)

    def test_keeps_up_with_fast_motion(self):
        f = OneEuro()
        for i in range(60):
            out = f([i * 1.0, 0, 0], i / 30)  # 30 cm/s
        self.assertGreater(out[0], 55)
        self.assertGreater(f.velocity[0], 20)


if __name__ == "__main__":
    unittest.main()
