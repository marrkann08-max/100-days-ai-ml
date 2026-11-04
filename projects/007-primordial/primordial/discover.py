from dataclasses import replace

import numpy as np

from .lifeness import measure, score
from .rules import crossover, mutate, random_rules
from .world import World


class Discovery:
    """Evolutionary search for life-like rule sets.

    Every candidate gets a fresh small world: it runs `warm` steps to settle,
    then `watch` more while we measure what changed. The best few survive
    unchanged, the rest of the next generation are their children (row
    crossover plus mutation) and some brand-new random species to keep
    exploring. run() is a generator that yields after every `chunk` steps, so
    the app can keep drawing while it searches.
    """

    def __init__(self, ctx, seed=None, population=16, generations=8, elite=4, fresh=4,
                 particles=2000, warm=500, watch=200, chunk=50, start=()):
        self.rng = np.random.default_rng(seed)
        self.population, self.generations = population, generations
        self.elite, self.fresh = elite, fresh
        self.warm, self.watch, self.chunk = warm, watch, chunk
        pool = list(start)[:population]
        while len(pool) < population:
            pool.append(self.random())
        self.pool = pool
        self.world = World(ctx, pool[0], particles, seed=int(self.rng.integers(1 << 30)))
        self.found = {}
        self.generation = 0
        self.evaluated = 0
        self.history = []

    def random(self):
        return random_rules(self.rng, types=int(self.rng.choice((4, 5, 6, 6, 7))))

    def evaluate(self, rules):
        """Generator: simulate one candidate, then return it with its score."""
        world = self.world
        world.set_rules(rules, reshuffle=True)
        for _ in range(self.warm // self.chunk):
            world.step(self.chunk)
            yield
        before = world.positions()
        for _ in range(self.watch // self.chunk):
            world.step(self.chunk)
            yield
        m = measure(before, world.positions(), world.kinds(), world.size, rules.radius, rules.types, self.watch)
        self.evaluated += 1
        return replace(rules, score=score(m), notes={k: round(v, 3) for k, v in m.items()})

    def breed(self, ranked):
        parents = ranked[:self.elite]
        children = list(parents)
        while len(children) < self.population - self.fresh:
            a, b = (parents[i] for i in self.rng.choice(len(parents), 2, replace=False))
            child = crossover(a, b, self.rng) if a.types == b.types and self.rng.random() < .5 else a
            children.append(mutate(child, self.rng))
        children += [self.random() for _ in range(self.fresh)]
        return children

    def run(self):
        for self.generation in range(1, self.generations + 1):
            scored = []
            for self.index, rules in enumerate(self.pool):
                if rules.score and rules.notes:  # survivors keep their score
                    scored.append(rules)
                    continue
                result = yield from self.evaluate(rules)
                scored.append(result)
                self.found.setdefault(result.fingerprint(), result)
                yield
            ranked = sorted(scored, key=lambda r: -r.score)
            self.history.append({"generation": self.generation, "best": ranked[0].score,
                                 "mean": float(np.mean([r.score for r in ranked]))})
            self.pool = self.breed(ranked)

    def best(self, count=8):
        return sorted(self.found.values(), key=lambda r: -r.score)[:count]

    def progress(self):
        total = self.population * self.generations
        done = (self.generation - 1) * self.population + getattr(self, "index", 0)
        return min(1.0, done / total)

    def release(self):
        self.world.release()
