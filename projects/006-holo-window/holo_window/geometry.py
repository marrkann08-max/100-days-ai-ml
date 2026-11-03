"""Head-coupled perspective maths.

World units are centimetres. The origin is the centre of the screen, x points
right, y points up and z points out of the screen towards the viewer, so
anything "inside" the window has a negative z.
"""
import math

import numpy as np

LEFT_IRIS, RIGHT_IRIS = 468, 473


def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = x, y, z
    return m


def frustum(left, right, bottom, top, near, far):
    return np.array([
        [2 * near / (right - left), 0, (right + left) / (right - left), 0],
        [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ])


def off_axis(eye, width, height, near=1.0, far=1000.0):
    """View-projection matrix for an eye looking through a width × height window at z = 0.

    The window's edges stay fixed in the world while the eye moves, so the
    frustum becomes asymmetric. That asymmetry is the whole illusion.
    """
    ex, ey, ez = eye
    ez = max(ez, near * 2)
    s = near / ez
    projection = frustum((-width / 2 - ex) * s, (width / 2 - ex) * s,
                         (-height / 2 - ey) * s, (height / 2 - ey) * s, near, far)
    return projection @ translate(-ex, -ey, -ez)


def focal_length(frame_width, fov_degrees):
    return frame_width / 2 / math.tan(math.radians(fov_degrees) / 2)


def fov_for(frame_width, focal):
    return math.degrees(2 * math.atan(frame_width / 2 / focal))


def iris_gap(points, frame_size):
    """Pixel distance between the iris centres, including MediaPipe's depth axis.

    Using the 3D distance means turning your head doesn't shrink the gap and
    make you appear further away.
    """
    w, h = frame_size
    scale = np.array([w, h, w])
    return float(np.linalg.norm((points[LEFT_IRIS] - points[RIGHT_IRIS]) * scale))


def eye_from_landmarks(points, frame_size, settings):
    """Midpoint between the eyes, in world centimetres, from 478 normalised face landmarks."""
    w, h = frame_size
    f = focal_length(w, settings.fov)
    distance = f * settings.ipd / iris_gap(points, frame_size)
    u, v = (points[LEFT_IRIS, :2] + points[RIGHT_IRIS, :2]) / 2 * (w, h)
    x_cam = (u - w / 2) / f * distance
    y_cam = (v - h / 2) / f * distance
    # The camera faces the viewer, so its image is a mirror of the viewer's left and right.
    return np.array([-x_cam, settings.camera_y - y_cam, distance])


class OneEuro:
    """One Euro filter (Casiez et al., 2012): smooth when still, responsive when moving."""

    def __init__(self, min_cutoff=.8, beta=.08, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.value = self.velocity = self.time = None

    @staticmethod
    def alpha(cutoff, dt):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, value, time):
        value = np.asarray(value, float)
        if self.value is None or time <= self.time:
            self.value, self.velocity, self.time = value, np.zeros_like(value), time
            return value
        dt = time - self.time
        raw_velocity = (value - self.value) / dt
        a = self.alpha(self.d_cutoff, dt)
        self.velocity = a * raw_velocity + (1 - a) * self.velocity
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.velocity))
        a = self.alpha(cutoff, dt)
        self.value = a * value + (1 - a) * self.value
        self.time = time
        return self.value
