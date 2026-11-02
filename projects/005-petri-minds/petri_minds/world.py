from dataclasses import dataclass

import numpy as np

from .brain import N_HIDDEN, N_INPUTS, N_OUTPUTS, think


@dataclass(frozen=True)
class Config:
    population: int = 60
    food: int = 70
    poison: int = 22
    steps: int = 900
    radius: float = 280.0
    body: float = 6.0
    pellet: float = 4.0
    max_speed: float = 3.0
    max_turn: float = .16
    sight: float = 160.0
    start_energy: float = 100.0
    food_energy: float = 35.0
    poison_energy: float = 45.0
    elite: int = 4
    tournament: int = 3
    mutation_rate: float = .12
    mutation_size: float = .35


def scatter(rng, count, radius):
    """Uniform points inside a disc."""
    r = radius * np.sqrt(rng.random(count))
    a = rng.random(count) * 2 * np.pi
    return np.column_stack((r * np.cos(a), r * np.sin(a)))


def wall_distance(pos, angle, radius):
    """Distance from points inside the dish to its edge along each angle."""
    u = np.column_stack((np.cos(angle), np.sin(angle)))
    b = np.sum(pos * u, axis=1)
    c = np.sum(pos * pos, axis=1) - radius * radius
    return -b + np.sqrt(np.maximum(b * b - c, 0))


class World:
    """One generation: a population of brains sharing one dish of pellets."""

    def __init__(self, genomes, rng, config=Config()):
        self.config, self.rng, self.genomes = config, rng, genomes
        n = len(genomes)
        self.pos = scatter(rng, n, config.radius * .6)
        self.heading = rng.random(n) * 2 * np.pi
        self.speed = np.zeros(n)
        self.energy = np.full(n, config.start_energy)
        self.alive = np.ones(n, bool)
        self.age = np.zeros(n, int)
        self.eaten = np.zeros(n, int)
        self.poisoned = np.zeros(n, int)
        self.food = scatter(rng, config.food, config.radius - 12)
        self.poison = scatter(rng, config.poison, config.radius - 12)
        self.inputs = np.zeros((n, N_INPUTS))
        self.hidden = np.zeros((n, N_HIDDEN))
        self.outputs = np.zeros((n, N_OUTPUTS))
        self.time = 0

    @property
    def done(self):
        return self.time >= self.config.steps or not self.alive.any()

    def fitness(self):
        survival = self.age / self.config.steps
        return self.eaten - 1.5 * self.poisoned + .5 * survival

    def nearest(self, pellets):
        delta = pellets[None, :, :] - self.pos[:, None, :]
        dist = np.hypot(delta[..., 0], delta[..., 1])
        index = dist.argmin(axis=1)
        rows = np.arange(len(self.pos))
        d = delta[rows, index]
        relative = np.arctan2(d[:, 1], d[:, 0]) - self.heading
        near = np.clip(1 - dist[rows, index] / self.config.sight, 0, 1)
        return np.column_stack((np.sin(relative), np.cos(relative), near))

    def sense(self):
        c = self.config
        food = self.nearest(self.food)
        poison = self.nearest(self.poison)
        walls = [
            np.clip(1 - wall_distance(self.pos, self.heading + offset, c.radius) / c.sight, 0, 1)
            for offset in (-.6, 0, .6)
        ]
        energy = np.clip(self.energy / c.start_energy, 0, 2)[:, None]
        return np.column_stack((food, poison, *walls, energy))

    def consume(self, pellets, counts):
        """Each pellet in reach goes to its closest living creature, then respawns."""
        c = self.config
        dist = np.linalg.norm(pellets[None, :, :] - self.pos[:, None, :], axis=2)
        dist = np.where(self.alive[:, None], dist, np.inf)
        reach = dist < c.body + c.pellet
        taken = np.flatnonzero(reach.any(axis=0))
        if not len(taken):
            return np.zeros(len(self.pos), int)
        eaters = dist[:, taken].argmin(axis=0)
        gained = np.bincount(eaters, minlength=len(self.pos))
        counts += gained
        pellets[taken] = scatter(self.rng, len(taken), c.radius - 12)
        return gained

    def step(self):
        if self.done:
            return
        c = self.config
        inputs = self.sense()
        hidden, outputs = think(self.genomes, inputs)
        live = self.alive
        self.inputs[live], self.hidden[live], self.outputs[live] = inputs[live], hidden[live], outputs[live]

        self.heading[live] += outputs[live, 0] * c.max_turn
        self.speed = np.where(live, (outputs[:, 1] + 1) / 2 * c.max_speed, 0)
        self.pos += self.speed[:, None] * np.column_stack((np.cos(self.heading), np.sin(self.heading)))

        self.energy -= live * (.07 + .035 * self.speed)
        self.energy += self.consume(self.food, self.eaten) * c.food_energy
        self.energy -= self.consume(self.poison, self.poisoned) * c.poison_energy
        np.minimum(self.energy, 2 * c.start_energy, out=self.energy)

        hit_wall = np.hypot(self.pos[:, 0], self.pos[:, 1]) > c.radius - c.body
        self.alive &= ~hit_wall & (self.energy > 0)
        self.age += self.alive
        self.time += 1
