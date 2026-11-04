"""Particle life physics. The GPU shader in world.py implements exactly this; the
NumPy version here is the readable reference the tests compare it against."""
import numpy as np

DT = .02


def force(r, a, beta):
    """Force profile at distance r (in units of the interaction radius).

    Inside beta everything repels, regardless of type, so particles never
    collapse onto each other. Between beta and 1 the attraction a (-1..1) rises
    and falls as a triangle. Beyond 1 there's nothing.
    """
    r = np.asarray(r, float)
    inner = r / beta - 1
    outer = a * (1 - np.abs(2 * r - 1 - beta) / (1 - beta))
    return np.where(r < beta, inner, np.where(r < 1, outer, 0.0))


def wrap(d, size):
    """Shortest offset on a torus: the world's edges join up."""
    return d - np.round(d / size) * size


def step(pos, vel, types, rules, size, dt=DT):
    """One O(n²) step. Returns new positions and velocities."""
    size = np.asarray(size, float)
    d = wrap(pos[None, :, :] - pos[:, None, :], size)      # d[i, j] = from i to j
    r = np.hypot(d[..., 0], d[..., 1])
    near = (r > 0) & (r < rules.radius)
    a = rules.matrix[types[:, None], types[None, :]]
    f = np.where(near, force(r / rules.radius, a, rules.beta), 0.0)
    safe = np.where(r > 0, r, 1)
    total = (d / safe[..., None] * f[..., None]).sum(axis=1)
    vel = vel * rules.friction(dt) + total * rules.radius * rules.force * dt
    pos = (pos + vel * dt) % size
    return pos, vel


def world_size(particles, aspect=16 / 9, density=4500):
    """World dimensions (height 1 for the reference density) so any particle count
    behaves the same: fewer particles get a proportionally smaller world."""
    height = np.sqrt(particles / (density * aspect))
    return np.array([height * aspect, height])
