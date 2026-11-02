import numpy as np

from .brain import random_genomes
from .tower import Config, Towers

SOLIDARITY = (0.0, .5, 1.0)


def breed(genomes, fitness, rng, config=Config(), chance=0.0):
    """Keep the elite, then fill the rest with tournament winners' mutated children.

    `chance` is the fraction of parents picked at random instead of by fitness.
    """
    n = len(genomes)
    order = np.argsort(fitness)[::-1]
    elite = genomes[order[:round(config.elite * (1 - chance))]]

    def pick(count):
        entrants = rng.integers(0, n, (count, config.tournament))
        winners = entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]
        return np.where(rng.random(count) < chance, entrants[:, 0], winners)

    count = n - len(elite)
    mums, dads = pick(count), pick(count)
    mask = rng.random((count, genomes.shape[1])) < .5
    kids = np.where(mask, genomes[mums], genomes[dads])
    kids = mutate(kids, rng, config)
    return np.vstack((elite, kids))


def mutate(genomes, rng, config=Config()):
    hit = rng.random(genomes.shape) < config.mutation_rate
    return genomes + hit * rng.normal(0, config.mutation_size, genomes.shape)


class Run:
    """Evolution inside each tower, plus optional competition between towers.

    Inside a tower, inmates breed according to their own survival. Solidarity
    shifts selection from the individual to the tower: that fraction of parents
    is picked regardless of personal score, and the worst towers are replaced by
    mutated copies of the best ones (none at 0, half at 1).
    """

    def __init__(self, seed=None, config=Config(), solidarity=0.0):
        self.seed = seed
        self.config = config
        self.solidarity = solidarity
        self.rng = np.random.default_rng(seed)
        self.genomes = random_genomes(self.rng, config.towers, config.inmates)
        self.generation = 1
        self.history = []
        self.towers = Towers(self.genomes, self.rng, config)

    def step(self):
        self.towers.tick()
        if self.towers.done:
            self.next_generation()

    def run_generation(self):
        generation = self.generation
        while self.generation == generation:
            self.step()

    def replaced(self):
        return round(self.solidarity * self.config.towers / 2)

    def next_generation(self):
        towers, c = self.towers, self.config
        fitness = towers.fitness()
        scores = towers.tower_scores()
        self.history.append({
            "generation": self.generation,
            "solidarity": self.solidarity,
            "survival": float(scores.mean()),
            "alive": float(towers.alive.mean()),
            "greed_top": towers.greed(slice(0, c.levels // 4)),
            "greed_bottom": towers.greed(slice(-c.levels // 4, None)),
            "best_tower": float(scores.max()),
        })
        genomes = np.stack([breed(self.genomes[t], fitness[t], self.rng, c, self.solidarity) for t in range(c.towers)])
        order = np.argsort(scores)[::-1]
        n = self.replaced()
        for winner, loser in zip(order[:n], order[::-1][:n]):
            genomes[loser] = mutate(genomes[winner], self.rng, c)
        self.genomes = genomes
        self.generation += 1
        self.towers = Towers(self.genomes, self.rng, c)
