"""Persistent 3D particles: form, compress, explode, then fade."""
import math
import cv2
import numpy as np

PALETTES = [(255, 190, 50), (25, 165, 255), (255, 65, 195)]


class Renderer:
    COUNT = 850
    COMPRESSION = .22
    LIFETIME = 1.35

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.palette = 0
        self.time = 0.0
        # Fibonacci sphere, with several shells rather than a solid white disk.
        i = np.arange(self.COUNT)
        y = 1 - 2*(i+.5)/self.COUNT
        angle = i * math.pi * (3-math.sqrt(5))
        ring = np.sqrt(1-y*y)
        self.offsets = np.column_stack((np.cos(angle)*ring, y, np.sin(angle)*ring))
        self.offsets *= self.rng.uniform(.55, 1, (self.COUNT, 1))
        self.speeds = self.rng.uniform(220, 570, (self.COUNT, 1))
        self.sizes = self.rng.choice([1, 1, 1, 2], self.COUNT)
        self.positions = np.zeros((self.COUNT, 3))
        self.previous = self.positions.copy()
        self.velocity = self.positions.copy()
        self.origin = np.zeros(2)
        self.radius = 60.0
        self.phase = 'idle'
        self.age = 0.0
        self.visibility = 0.0
        self.sound_due = False
        self.burst_palette = 0

    def reset(self):
        self.positions.fill(0)
        self.previous.fill(0)
        self.velocity.fill(0)
        self.phase = 'idle'
        self.age = self.visibility = 0.0
        self.sound_due = False

    def targets(self, radius):
        angle = self.time*1.4
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, 0, s], [s*.3, .954, -c*.3], [-s*.954, .3, c*.954]])
        return self.offsets @ rotation.T * radius

    def blast(self, center, radius):
        if self.phase in ('compress', 'burst'):
            return
        # Freeze the current cloud. No replacement explosion particles are spawned.
        if self.phase == 'idle':
            self.origin = np.array(center, dtype=float)
            self.positions[:] = self.targets(radius*.8)
        self.radius = max(20, radius)
        self.previous[:] = self.positions
        self.compression_start = self.positions.copy()
        self.phase = 'compress'
        self.age = 0.0
        self.visibility = 1.0
        self.burst_palette = self.palette

    def consume_sound(self):
        due = self.sound_due
        self.sound_due = False
        return due

    @property
    def label(self):
        return {'compress': 'COMPRESSING', 'burst': 'BURST'}.get(self.phase)

    def _step(self, energy, dt):
        self.time += dt
        self.previous[:] = self.positions
        if self.phase == 'compress':
            self.age += dt
            progress = min(1, self.age/self.COMPRESSION)
            ease = progress*progress*(3-2*progress)
            self.positions[:] = self.compression_start * (1-.90*ease)
            if progress >= 1:
                direction = self.compression_start.copy()
                direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), .001)
                self.velocity[:] = direction*self.speeds
                self.phase = 'burst'
                self.age = 0.0
                self.sound_due = True
        elif self.phase == 'burst':
            self.age += dt
            drag = 1.1
            damping = math.exp(-drag*dt)
            self.positions += self.velocity*(1-damping)/drag
            self.velocity *= damping
            self.visibility = max(0, 1-self.age/self.LIFETIME)**.65
            if self.age >= self.LIFETIME:
                self.reset()
        elif energy.center is not None and energy.charge > .01:
            center = np.array(energy.center)
            radius = energy.radius * (.35+.55*energy.charge)
            target = self.targets(radius)
            if self.phase == 'idle':
                self.origin[:] = center
                self.positions[:] = target*1.7
            self.phase = 'forming'
            self.origin += (center-self.origin)*(1-math.exp(-18*dt))
            self.positions += (target-self.positions)*(1-math.exp(-10*dt))
            self.radius = radius
            self.visibility = min(1, energy.charge*2)
        else:
            self.visibility *= math.exp(-12*dt)
            if self.visibility < .01:
                self.reset()

    def draw(self, frame, energy, dt):
        dt = max(0, min(dt, .1))
        # Fixed small steps keep compression and damping stable at webcam FPS.
        remaining = dt
        trail_start = self.positions.copy()
        while remaining > 1e-8:
            step = min(remaining, 1/60)
            self._step(energy, step)
            remaining -= step
        layer = np.zeros_like(frame)
        if self.visibility > .01:
            color_index = self.burst_palette if self.phase in ('compress', 'burst') else self.palette
            base = np.array(PALETTES[color_index], dtype=float)
            depth = np.clip(1+self.positions[:, 2]/350, .65, 1.5)
            xy = self.positions[:, :2]*depth[:, None] + self.origin
            old_xy = trail_start[:, :2]*depth[:, None] + self.origin
            order = np.argsort(self.positions[:, 2])
            h, w = frame.shape[:2]
            for i in order:
                x, y = map(int, xy[i])
                if not (3 <= x < w-3 and 3 <= y < h-3):
                    continue
                brightness = (.40+.60*(depth[i]-.65)/.85)*self.visibility
                color = tuple(map(int, np.minimum(255, base*brightness*1.5)))
                size = int(self.sizes[i]) + int(depth[i] > 1.15)
                if self.phase == 'burst':
                    tail = tuple(map(int, old_xy[i]))
                    cv2.line(layer, tail, (x, y), tuple(int(c*.65) for c in color), 1, cv2.LINE_AA)
                cv2.circle(layer, (x, y), size, color, -1, cv2.LINE_AA)
                if size >= 2:
                    cv2.circle(layer, (x, y), 1,
                               tuple(min(245, int(c*.7+75*self.visibility)) for c in color), -1)
            if self.phase == 'burst' and self.age < .65:
                progress = self.age/.65
                color = tuple(map(int, base*(1-progress)*.7))
                ring = int(20+240*(1-(1-progress)**2))
                cv2.circle(layer, tuple(map(int, self.origin)), ring, color, 2, cv2.LINE_AA)
        # Small bloom buffer keeps the renderer affordable on laptop CPUs.
        small = cv2.resize(layer, (max(1, frame.shape[1]//3), max(1, frame.shape[0]//3)))
        glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5), (frame.shape[1], frame.shape[0]))
        light = cv2.addWeighted(layer, .78, glow, .65, 0)
        np.minimum(light, 225, out=light)
        return cv2.add(frame, cv2.multiply(255-frame, light, scale=1/255))
