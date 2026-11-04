from dataclasses import dataclass, field, replace
import hashlib
import json

import numpy as np

MAX_TYPES = 8
SYLLABLES = ("ka", "lo", "vi", "re", "su", "mon", "tha", "el", "zi", "or", "qua", "ny",
             "dra", "pe", "lu", "xo", "ri", "sa", "ven", "ith", "ul", "mo", "ae", "ko")


@dataclass
class Rules:
    """Everything that defines a species: an attraction matrix plus a few physical constants.

    matrix[a, b] is how strongly a particle of type a is pulled towards (positive)
    or pushed from (negative) a particle of type b. It needn't be symmetric, which is
    what lets chasing, orbiting and swimming structures appear.
    """
    matrix: np.ndarray
    radius: float = .1        # interaction range, in world heights
    beta: float = .3          # inner zone, as a fraction of radius, where everything repels
    half_life: float = .04    # seconds for velocity to halve from friction
    force: float = 10.0
    name: str = ""
    score: float = 0.0
    notes: dict = field(default_factory=dict)

    @property
    def types(self):
        return len(self.matrix)

    def friction(self, dt):
        return .5 ** (dt / self.half_life)

    def fingerprint(self):
        return hashlib.sha1(np.round(self.matrix, 3).tobytes()).hexdigest()

    def with_name(self):
        return self if self.name else replace(self, name=species_name(self.matrix))

    def to_json(self):
        return {"name": self.name, "score": round(float(self.score), 4), "radius": self.radius,
                "beta": self.beta, "half_life": self.half_life, "force": self.force,
                "matrix": np.round(self.matrix, 4).tolist(), "notes": self.notes}

    @classmethod
    def from_json(cls, data):
        return cls(matrix=np.array(data["matrix"], float), radius=data.get("radius", .1),
                   beta=data.get("beta", .3), half_life=data.get("half_life", .04),
                   force=data.get("force", 10.0), name=data.get("name", ""),
                   score=data.get("score", 0.0), notes=data.get("notes", {}))

    def save(self, path):
        path.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def random_rules(rng, types=6):
    return Rules(rng.uniform(-1, 1, (types, types))).with_name()


def mutate(rules, rng, size=.2, rate=.35):
    """Nudge some matrix entries; occasionally swap two rows to reshuffle roles."""
    m = rules.matrix.copy()
    hit = rng.random(m.shape) < rate
    m = np.clip(m + hit * rng.normal(0, size, m.shape), -1, 1)
    if rng.random() < .1:
        a, b = rng.choice(len(m), 2, replace=False)
        m[[a, b]] = m[[b, a]]
    return replace(rules, matrix=m, name="", score=0.0, notes={}).with_name()


def crossover(a, b, rng):
    """Each row (one type's whole behaviour) comes from one parent."""
    rows = rng.random(len(a.matrix)) < .5
    m = np.where(rows[:, None], a.matrix, b.matrix)
    return replace(a, matrix=m, name="", score=0.0, notes={}).with_name()


def species_name(matrix):
    """A pronounceable, repeatable name derived from the matrix."""
    digest = hashlib.sha1(np.round(matrix, 3).tobytes()).digest()
    parts = [SYLLABLES[digest[i] % len(SYLLABLES)] for i in range(2 + digest[9] % 2)]
    return "".join(parts).capitalize()
