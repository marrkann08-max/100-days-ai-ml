"""Glowing energy orbs and bursts that expand from the palm."""
import math
import random
import cv2
import numpy as np

PALETTES = [(255, 190, 50), (25, 165, 255), (255, 65, 195)]


class Renderer:
    def __init__(self, seed=None):
        self.random = random.Random(seed)
        self.particles = []
        self.blasts = []
        self.palette = 0
        self.time = 0

    def blast(self, center, radius):
        self.blasts.append([*center, radius, 0.0])
        for _ in range(120):
            angle = self.random.uniform(0, math.tau)
            speed = self.random.uniform(120, 460)
            self.particles.append([*center, math.cos(angle)*speed,
                                   math.sin(angle)*speed, self.random.uniform(.3, .9)])

    def orb(self, layer, center, radius, power):
        target = layer
        layer = np.zeros_like(target)
        x, y = map(int, center)
        radius = max(4, int(radius))
        color = PALETTES[self.palette]
        for scale, intensity in ((2.0, .03), (1.6, .06), (1.25, .12), (1, .35), (.7, .5)):
            cv2.circle(layer, (x, y), int(radius*scale),
                       tuple(int(c*intensity) for c in color), -1, cv2.LINE_AA)
        cv2.circle(layer, (x, y), max(2, int(radius*.20)), (190, 220, 235), -1, cv2.LINE_AA)
        for i in range(3):
            angle = self.time * (1.5+i*.3) + i*2.1
            cv2.ellipse(layer, (x, y), (int(radius*1.35), int(radius*.45)),
                        math.degrees(angle), 0, 320, color, 2, cv2.LINE_AA)
        for i in range(10 + int(power*18)):
            angle = i*2.399 + self.time*(.7+i%3*.2)
            distance = radius*(1.4+.35*math.sin(self.time*3+i))
            point = (int(x+math.cos(angle)*distance), int(y+math.sin(angle)*distance))
            cv2.circle(layer, point, 2+i%2, color, -1, cv2.LINE_AA)
        cv2.add(target, layer, dst=target)

    def draw(self, frame, energy, dt):
        self.time += dt
        layer = np.zeros_like(frame)
        if energy.center is not None and energy.charge > .01:
            radius = energy.radius * (.28 + .62*energy.charge)
            self.orb(layer, energy.center, radius*(1+.035*math.sin(self.time*18)), energy.charge)
        alive = []
        for blast in self.blasts:
            blast[3] += dt
            x, y, radius, age = blast
            if age < 1.0:
                color = PALETTES[self.palette]
                origin = (int(x), int(y))
                # The center stays at the palm; only the shockwaves expand.
                for delay in (0, .10, .20):
                    progress = (age-delay)/.8
                    if 0 < progress < 1:
                        ring = max(2, int(radius*.4 + 240*progress))
                        intensity = (1-progress)**1.4
                        ring_color = tuple(int(c*intensity) for c in color)
                        cv2.circle(layer, origin, ring, ring_color,
                                   max(1, int(7*(1-progress))), cv2.LINE_AA)
                if age < .18:
                    flash_radius = max(2, int(radius*.65*(1-age/.18)))
                    cv2.circle(layer, origin, flash_radius, (180, 215, 235), -1, cv2.LINE_AA)
                alive.append(blast)
        self.blasts = alive
        alive = []
        for p in self.particles:
            p[0] += p[2]*dt
            p[1] += p[3]*dt
            p[4] -= dt
            if p[4] > 0:
                color = tuple(int(c*min(1, p[4]*3)) for c in PALETTES[self.palette])
                point = (int(p[0]), int(p[1]))
                tail = (int(p[0]-p[2]*.025), int(p[1]-p[3]*.025))
                cv2.line(layer, tail, point, color, 2, cv2.LINE_AA)
                alive.append(p)
        self.particles = alive
        glow = cv2.GaussianBlur(layer, (0, 0), 15)
        light = cv2.addWeighted(layer, .55, glow, .30, 0)
        # Screen blending retains camera detail instead of clipping to white.
        return cv2.add(frame, cv2.multiply(255-frame, light, scale=1/255))


def hud(frame, energy, fps, demo=False):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 115), (19, 15, 12), -1)
    cv2.rectangle(panel, (0, h-65), (w, h), (19, 15, 12), -1)
    cv2.addWeighted(panel, .78, frame, .22, 0, dst=frame)
    def label(text, pos, size=.5, color=(180, 185, 185)):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    label('KI / FORGE', (28, 42), .95, (240, 245, 250))
    label('HAND-CONTROLLED ENERGY', (30, 66), .4)
    label(f'{"MOUSE LAB" if demo else "LIVE CAMERA"}  /  {fps:.0f} FPS', (w-255, 37), .45)
    label(energy.state, (w-255, 66), .55, (255, 205, 90))
    cv2.rectangle(frame, (30, 85), (w-30, 91), (55, 50, 45), -1)
    cv2.rectangle(frame, (30, 85), (30+int((w-60)*energy.charge), 91), (255, 200, 70), -1)
    instruction = ('MOVE MOUSE: charge   HOLD CLICK: burst' if demo else
                   'OPEN PALM: charge   TWO PALMS: grow   CLOSE FIST: burst')
    label(instruction, (28, h-40), .48, (225, 230, 235))
    label('C  color     R  reset     S  screenshot     Q / ESC  quit', (28, h-17), .42)
    return frame
