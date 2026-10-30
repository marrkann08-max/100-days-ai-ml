"""Geometry and gesture state, independent of the camera and renderer."""
from dataclasses import dataclass
from math import dist


@dataclass
class Hand:
    center: tuple[float, float]
    size: float
    opened: bool


def describe(landmarks, width, height):
    points = [(p.x * width, p.y * height) for p in landmarks]
    center = tuple(sum(points[i][axis] for i in (0, 5, 9, 13, 17)) / 5
                   for axis in (0, 1))
    size = max(dist(points[0], points[9]), 1)
    # Wrist-relative distances make this work with rotated hands too.
    extended = sum(dist(points[tip], points[0]) >
                   dist(points[pip], points[0]) * 1.18
                   for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
    return Hand(center, size, extended >= 3)


class Energy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.charge = 0.0
        self.center = None
        self.radius = 30.0
        self.state = 'SEEKING'
        self.cooldown = 0.0
        self.release_time = 0.0
        self.direction = (1.0, 0.0)

    def update(self, hands, dt, width):
        dt = max(0.0, min(dt, 0.1))
        self.cooldown = max(0.0, self.cooldown - dt)
        if not hands:
            self.charge = max(0, self.charge - dt * 0.7)
            self.state = 'SEEKING'
            self.center = None
            self.release_time = 0
            return False
        opened = [h for h in hands if h.opened]
        target = hands[0].center
        if len(opened) >= 2:
            a, b = opened[:2]
            target = ((a.center[0] + b.center[0]) / 2,
                      (a.center[1] + b.center[1]) / 2)
            gap = dist(a.center, b.center)
            charging = gap < (a.size + b.size) * 3.5
            desired_radius = min(110, max(25, gap * 0.3))
        else:
            charging = bool(opened)
            if opened:
                target = opened[0].center
            desired_radius = min(90, max(25, hands[0].size * 0.65))
        alpha = 1 - 0.001 ** dt
        if self.center is None:
            self.center = target
        self.center = tuple(a + (b-a) * alpha for a, b in zip(self.center, target))
        self.radius += (desired_radius - self.radius) * alpha
        if charging:
            self.release_time = 0
            self.charge = min(1, self.charge + dt * 0.38)
            self.state = 'READY' if self.charge >= 0.65 else 'CHARGING'
        else:
            self.release_time = self.release_time + dt if not opened else 0
            if self.charge >= 0.65 and self.cooldown == 0 and self.release_time >= 0.12:
                self.direction = (1.0 if self.center[0] >= width/2 else -1.0, -0.10)
                self.charge = 0
                self.cooldown = 0.65
                self.release_time = 0
                self.state = 'BLAST'
                return True
            self.state = 'COOLDOWN' if self.cooldown else 'HOLD'
            self.charge = max(0, self.charge - dt * 0.08)
        return False
