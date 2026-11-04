"""How "alive" does a world look? Hand-built measurements from two snapshots.

Life-like particle worlds sit between two boring extremes: a uniform gas
(nothing forms) and a frozen crystal or one giant blob (nothing happens).
Each measure below is scaled to 0..1 and the score is their product, so a
world has to do well on all of them at once.
"""
from collections import deque

import numpy as np

from .physics import wrap


def grid(pos, size, cell):
    nx, ny = max(1, int(size[0] / cell)), max(1, int(size[1] / cell))
    ix = (pos[:, 0] / size[0] * nx).astype(int) % nx
    iy = (pos[:, 1] / size[1] * ny).astype(int) % ny
    return ix * ny + iy, (nx, ny)


def clusters(dense, shape):
    """Count connected groups of dense cells on the wrapping grid."""
    nx, ny = shape
    dense = dense.reshape(nx, ny)
    seen = np.zeros_like(dense)
    count = 0
    for start in zip(*np.nonzero(dense)):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = ((x + dx) % nx, (y + dy) % ny)
                if dense[nb] and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
    return count


def measure(before, after, kinds, size, radius, types, steps):
    size = np.asarray(size, float)
    cell, shape = grid(after, size, radius / 2)
    counts = np.bincount(cell, minlength=shape[0] * shape[1]).astype(float)
    mean = counts.mean()
    clumping = (counts.std() / mean) * np.sqrt(mean)  # 1 for a random gas, larger when clumped

    dense = counts >= 3 * mean
    groups = clusters(dense, shape)
    in_dense = dense[cell]

    if in_dense.any():
        table = np.zeros((shape[0] * shape[1], types))
        np.add.at(table, (cell[in_dense], kinds[in_dense]), 1)
        rows = table[table.sum(axis=1) > 0]
        p = rows / rows.sum(axis=1, keepdims=True)
        entropy = -(p * np.log(np.where(p > 0, p, 1))).sum(axis=1) / np.log(types)
        mixing = float(np.average(entropy, weights=rows.sum(axis=1)))
    else:
        mixing = 0.0

    travel = float(np.hypot(*wrap(after - before, size).T).mean() / radius)
    return {
        "clumping": float(clumping),
        "clusters": groups,
        "captured": float(in_dense.mean()),
        "mixing": mixing,
        "travel": travel / steps * 100,  # radii per 100 steps
    }


def score(m):
    structure = np.clip((m["clumping"] - 2) / 6, 0, 1) * np.clip(m["captured"] / .4, 0, 1)
    # Ramps up to 3 organisms rather than jumping, so one merger doesn't crash the score.
    organisms = min(1.0, m["clusters"] / 3) if m["clusters"] <= 80 else .25
    motion = 1 - np.exp(-m["travel"] / 1.5)
    mixing = .3 + .7 * m["mixing"]
    return float(structure * organisms * (.3 + .7 * motion) * mixing)
