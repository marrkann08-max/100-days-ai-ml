from dataclasses import dataclass

import numpy as np

from .brain import N_HIDDEN, N_INPUTS, think


@dataclass(frozen=True)
class Config:
    towers: int = 8
    levels: int = 20
    per_level: int = 2
    days: int = 60
    month: int = 30
    surplus: float = 1.1
    max_portion: float = 3.0
    start_energy: float = 4.0
    max_energy: float = 8.0
    elite: int = 2
    tournament: int = 3
    mutation_rate: float = .12
    mutation_size: float = .35

    @property
    def inmates(self):
        return self.levels * self.per_level

    @property
    def feast(self):
        """Food loaded at the top each day: everyone's ration plus a small surplus."""
        return self.surplus * self.inmates


class Towers:
    """One generation: every tower runs the same days in lockstep.

    A tick is one stop of the platform. It visits levels 0..levels-1 in order;
    after the last level the day ends, everyone burns one ration of energy,
    and the platform is reloaded at the top.
    """

    def __init__(self, genomes, rng, config=Config()):
        self.config, self.rng, self.genomes = config, rng, genomes
        t, k = config.towers, config.inmates
        self.home = np.repeat(np.arange(config.levels), config.per_level)
        self.level = np.stack([rng.permutation(self.home) for _ in range(t)])
        self.energy = np.full((t, k), config.start_energy)
        self.alive = np.ones((t, k), bool)
        self.age = np.zeros((t, k), int)
        self.portion = np.zeros((t, k))
        self.hidden = np.zeros((t, k, N_HIDDEN))
        self.inputs = np.zeros((t, k, N_INPUTS))
        self.food = np.full(t, config.feast)
        self.arrived = np.zeros((t, config.levels))
        self.asked = np.zeros((t, config.levels))
        self.asks = np.zeros((t, config.levels))
        self.stop = 0
        self.day = 0

    @property
    def done(self):
        return self.day >= self.config.days or not self.alive.any()

    def fitness(self):
        """Individual score: days survived, plus a little for food in reserve."""
        return self.age / self.config.days + .05 * self.energy * self.alive

    def tower_scores(self):
        return self.age.mean(axis=1) / self.config.days

    def tick(self):
        if self.done:
            return
        c = self.config
        here = self.alive & (self.level == self.stop)
        below = (self.alive & (self.level >= self.stop)).sum(axis=1)
        self.arrived[:, self.stop] = self.food

        t, i = np.nonzero(here)
        if len(t):
            fair = self.food[t] / np.maximum(below[t], 1)
            inputs = np.column_stack((
                np.full(len(t), self.stop / (c.levels - 1)),
                np.minimum(fair, c.max_portion) / c.max_portion,
                self.food[t] / c.feast,
                self.energy[t, i] / c.max_energy,
                np.full(len(t), 1 - (self.day % c.month) / c.month),
            ))
            hidden, outputs = think(self.genomes[t, i], inputs)
            want = (outputs[:, 0] + 1) / 2 * c.max_portion
            demand = np.bincount(t, weights=want, minlength=c.towers)
            share = np.minimum(1, self.food / np.maximum(demand, 1e-9))
            eaten = want * share[t]
            self.food -= np.bincount(t, weights=eaten, minlength=c.towers)
            np.maximum(self.food, 0, out=self.food)
            self.energy[t, i] = np.minimum(self.energy[t, i] + eaten, c.max_energy)
            self.portion[t, i], self.inputs[t, i], self.hidden[t, i] = want, inputs, hidden
            np.add.at(self.asked, (t, self.stop), want)
            np.add.at(self.asks, (t, self.stop), 1)

        self.stop += 1
        if self.stop == c.levels:
            self.end_day()

    def end_day(self):
        c = self.config
        self.energy -= self.alive
        self.alive &= self.energy > 0
        self.age += self.alive
        self.day += 1
        self.stop = 0
        self.food[:] = c.feast
        if self.day % c.month == 0:
            self.level = np.stack([self.rng.permutation(self.home) for _ in range(c.towers)])

    def run_day(self):
        day = self.day
        while self.day == day and not self.done:
            self.tick()

    def greed(self, levels=slice(0, 5)):
        """Average portion asked for on the given levels, across all towers."""
        asks = self.asks[:, levels].sum()
        return float(self.asked[:, levels].sum() / asks) if asks else 0.0
