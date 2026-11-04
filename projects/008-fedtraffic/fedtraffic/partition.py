"""Split sensors into federated clients ("districts")."""
import numpy as np


def kmeans(points, k, seed=0, iterations=100):
    """Plain k-means with k-means++ initialisation."""
    rng = np.random.default_rng(seed)
    centres = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        d2 = np.min([((points - c) ** 2).sum(axis=1) for c in centres], axis=0)
        centres.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centres = np.array(centres)
    labels = np.zeros(len(points), int)
    for _ in range(iterations):
        new = np.argmin(((points[:, None] - centres[None]) ** 2).sum(axis=2), axis=1)
        if (new == labels).all() and _:
            break
        labels = new
        centres = np.array([points[labels == j].mean(axis=0) if (labels == j).any() else centres[j] for j in range(k)])
    return labels


def partition(locations, clients, method="spatial", seed=0):
    """Sorted sensor indices for each client.

    spatial: k-means on latitude/longitude, so each client is a contiguous
    district (realistic, and naturally non-IID: different freeways).
    random: sensors dealt out at random (a control that ignores geography).
    """
    if method == "spatial":
        labels = kmeans(np.asarray(locations, float), clients, seed)
    elif method == "random":
        labels = np.random.default_rng(seed).permutation(len(locations)) % clients
    else:
        raise ValueError(method)
    groups = [np.flatnonzero(labels == j) for j in range(clients)]
    return sorted(groups, key=lambda g: -len(g))


def cut_edges(adjacency, groups):
    """Share of graph edges (excluding self-loops) whose ends land in different clients."""
    owner = np.empty(len(adjacency), int)
    for k, g in enumerate(groups):
        owner[g] = k
    rows, cols = np.nonzero(adjacency)
    real = rows != cols
    return float((owner[rows[real]] != owner[cols[real]]).mean())
