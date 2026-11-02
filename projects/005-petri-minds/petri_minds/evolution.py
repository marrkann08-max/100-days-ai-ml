import numpy as np

from .brain import random_genomes
from .world import Config, World


def breed(genomes, fitness, hues, rng, config=Config()):
    """Keep the elite, then fill the rest with tournament winners' mutated children."""
    n = len(genomes)
    order = np.argsort(fitness)[::-1]
    children = [genomes[order[:config.elite]]]
    child_hues = [hues[order[:config.elite]]]

    def pick():
        entrants = rng.integers(0, n, config.tournament)
        return entrants[np.argmax(fitness[entrants])]

    count = n - config.elite
    mums = np.array([pick() for _ in range(count)])
    dads = np.array([pick() for _ in range(count)])
    mask = rng.random((count, genomes.shape[1])) < .5
    kids = np.where(mask, genomes[mums], genomes[dads])
    mutate = rng.random(kids.shape) < config.mutation_rate
    kids = kids + mutate * rng.normal(0, config.mutation_size, kids.shape)
    children.append(kids)
    # Children inherit their mother's colour with a little drift, so lineages stay visible.
    child_hues.append((hues[mums] + rng.normal(0, 4, count)) % 360)
    return np.vstack(children), np.concatenate(child_hues)


class Run:
    """A whole evolutionary run: one World per generation plus its history."""

    def __init__(self, seed=None, config=Config()):
        self.seed = seed
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.genomes = random_genomes(self.rng, config.population)
        self.hues = self.rng.random(config.population) * 360
        self.generation = 1
        self.history = []
        self.champion = self.genomes[0].copy()
        self.champion_hue = self.hues[0]
        self.world = World(self.genomes, self.rng, config)

    def step(self):
        self.world.step()
        if self.world.done:
            self.next_generation()

    def run_generation(self):
        generation = self.generation
        while self.generation == generation:
            self.step()

    def next_generation(self):
        world = self.world
        fitness = world.fitness()
        best = int(np.argmax(fitness))
        self.champion = self.genomes[best].copy()
        self.champion_hue = self.hues[best]
        self.history.append({
            "generation": self.generation,
            "best": float(fitness[best]),
            "mean": float(fitness.mean()),
            "food": int(world.eaten.sum()),
            "poison": int(world.poisoned.sum()),
            "survivors": int(world.alive.sum()),
        })
        self.genomes, self.hues = breed(self.genomes, fitness, self.hues, self.rng, self.config)
        self.generation += 1
        self.world = World(self.genomes, self.rng, self.config)
