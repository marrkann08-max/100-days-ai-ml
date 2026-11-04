"""Reproducible preprocessing for METR-LA.

raw files (checksummed) → speeds with missing values marked → short gaps
interpolated → chronological split → statistics from the training period only
→ normalised inputs, observed-value masks and time features → cached .npz
keyed by the config and the raw checksums.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import digest, resolve

RAW = {
    "metr-la.h5": ("https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/main/metr-la.h5",
                   "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"),
    "distances.csv": ("https://huggingface.co/datasets/witgaw/METR-LA/resolve/main/sensor_graph/distances.csv",
                      "072e7aaadd58d691f34fe528a2b10c12f5f7d616916d5092eba01aeb50971156"),
    "sensor_locations.csv": ("https://huggingface.co/datasets/witgaw/METR-LA/resolve/main/sensor_graph/sensor_locations.csv",
                             "ab2e797c2dc099c0f2a407e2600b7d6b9620d63b77369465d7415841b30c9239"),
}
STEPS_PER_DAY = 288   # 5-minute readings
STEPS_PER_WEEK = STEPS_PER_DAY * 7


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(raw_dir):
    """Download any missing raw file and refuse to continue if a checksum differs."""
    import urllib.request
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in RAW.items():
        path = raw_dir / name
        if not path.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{name}: checksum {actual} does not match the recorded {expected}")


def read_raw(raw_dir):
    """Speeds (T, N), sensor ids and timestamps, read as plain arrays.

    The HDF5 file is a pandas store, but it's read with PyTables directly so
    nothing in it is ever unpickled.
    """
    import tables
    with tables.open_file(raw_dir / "metr-la.h5", "r") as f:
        speeds = f.get_node("/df/block0_values").read().astype(np.float32)
        sensors = [s.decode() for s in f.get_node("/df/block0_items").read()]
        stamps = f.get_node("/df/axis1").read().astype("datetime64[ns]")
    return speeds, sensors, stamps


def read_locations(raw_dir, sensors):
    rows = {}
    with open(raw_dir / "sensor_locations.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            sid, lat, lon = line.strip().split(",")
            rows[sid] = (float(lat), float(lon))
    return np.array([rows[s] for s in sensors])


def read_distances(raw_dir, sensors):
    """Road-network distance (metres) between sensor pairs; inf where no path is listed."""
    index = {s: i for i, s in enumerate(sensors)}
    n = len(sensors)
    dist = np.full((n, n), np.inf)
    with open(raw_dir / "distances.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, cost = line.strip().split(",")
            if a in index and b in index:
                dist[index[a], index[b]] = float(cost)
    return dist


def kernel_adjacency(dist, threshold):
    """Thresholded Gaussian kernel W_ij = exp(-(d_ij / sigma)^2), sigma = std of finite distances."""
    finite = dist[np.isfinite(dist)]
    sigma = finite.std()
    w = np.exp(-np.square(dist / sigma))
    w[~np.isfinite(dist)] = 0
    w[w < threshold] = 0
    return w.astype(np.float32)


def fill_short_gaps(x, mask, max_gap):
    """Linear interpolation along time for missing runs of at most max_gap steps."""
    x = x.copy()
    filled = mask.copy()
    t = np.arange(len(x))
    for j in range(x.shape[1]):
        miss = ~mask[:, j]
        if not miss.any() or miss.all():
            continue
        edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        good = mask[:, j]
        for s, e in zip(starts, ends):
            if e - s <= max_gap and s > 0 and e < len(x):
                x[s:e, j] = np.interp(t[s:e], t[good], x[good, j])
                filled[s:e, j] = True
    return x, filled


def split_points(length, fractions):
    train = int(round(length * fractions[0]))
    val = int(round(length * (fractions[0] + fractions[1])))
    return train, val


def time_features(stamps):
    minutes = (stamps.astype("datetime64[m]").astype(np.int64)) % (24 * 60)
    angle = 2 * np.pi * minutes / (24 * 60)
    return np.stack((np.sin(angle), np.cos(angle)), axis=1).astype(np.float32)


def week_slot(stamps):
    """Index of each timestamp within the week, in 5-minute slots (Monday 00:00 = 0)."""
    minutes = stamps.astype("datetime64[m]").astype(np.int64)
    # 1970-01-01 was a Thursday: shift so Monday is slot 0.
    return ((minutes // 5) + 3 * STEPS_PER_DAY) % STEPS_PER_WEEK


def historical_average(speeds, mask, slots, train_end):
    """Mean observed speed per sensor and slot of the week, from the training period only."""
    n = speeds.shape[1]
    total = np.zeros((STEPS_PER_WEEK, n))
    count = np.zeros((STEPS_PER_WEEK, n))
    np.add.at(total, slots[:train_end], np.where(mask[:train_end], speeds[:train_end], 0))
    np.add.at(count, slots[:train_end], mask[:train_end])
    fallback = speeds[:train_end][mask[:train_end]].mean()
    return np.where(count > 0, total / np.maximum(count, 1), fallback).astype(np.float32)


def build(config):
    d = config["data"]
    raw_dir = resolve(config, "raw_dir")
    fetch(raw_dir)
    speeds, sensors, stamps = read_raw(raw_dir)

    observed = speeds > 0   # METR-LA records missing readings as 0 mph
    cleaned, usable = fill_short_gaps(speeds, observed, d["max_gap_fill"])
    train_end, val_end = split_points(len(speeds), d["split"])
    slots = week_slot(stamps)
    ha = historical_average(speeds, observed, slots, train_end)
    # Whatever is still missing after interpolation is filled from the historical average,
    # so the model always sees a number, plus a flag saying whether it was really observed.
    cleaned = np.where(usable, cleaned, ha[slots])

    mean = float(speeds[:train_end][observed[:train_end]].mean())
    std = float(speeds[:train_end][observed[:train_end]].std())
    dist = read_distances(raw_dir, sensors)
    return {
        "speeds": speeds,                      # raw, 0 = missing (targets come from here)
        "observed": observed,
        "inputs": ((cleaned - mean) / std).astype(np.float32),
        "interpolated": (usable & ~observed),
        "time": time_features(stamps),
        "slots": slots.astype(np.int32),
        "historical": ha,
        "stamps": stamps.astype(np.int64),
        "sensors": np.array(sensors),
        "locations": read_locations(raw_dir, sensors),
        "distances": dist,
        "adjacency": kernel_adjacency(dist, d["graph"]["kernel_threshold"]),
        "split": np.array([train_end, val_end]),
        "norm": np.array([mean, std], dtype=np.float32),
    }


def load(config):
    """Processed dataset, rebuilt only when the data config or raw files change."""
    d = config["data"]
    keyed = {k: d[k] for k in ("split", "max_gap_fill", "graph")}
    keyed["raw"] = {name: checksum for name, (_, checksum) in RAW.items()}
    key = digest(keyed)
    cache = resolve(config, "cache_dir") / f"metr-la-{key}.npz"
    if not cache.exists():
        data = build(config)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
        manifest = {"cache": cache.name, "config": keyed, "sha256": sha256(cache)}
        (cache.with_suffix(".json")).write_text(json.dumps(manifest, indent=2))
    with np.load(cache, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    data["cache_key"] = key
    return data


class Windows:
    """Sliding windows over one split for a chosen set of sensors.

    Batches are assembled on demand instead of materialising every window:
    inputs (B, steps, N, 4) = [normalised speed, observed flag, sin, cos of time of day],
    targets (B, horizon, N) in mph with mask (B, horizon, N) of truly observed values.
    """

    def __init__(self, data, part, steps, horizon, stride=1, nodes=None):
        train_end, val_end = data["split"]
        lo, hi = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, len(data["inputs"]))}[part]
        self.nodes = np.arange(data["inputs"].shape[1]) if nodes is None else np.asarray(nodes)
        self.steps, self.horizon = steps, horizon
        self.starts = np.arange(lo, hi - steps - horizon + 1, stride)
        n = self.nodes
        self.x = np.concatenate((
            data["inputs"][:, n, None],
            data["observed"][:, n, None].astype(np.float32),
            np.broadcast_to(data["time"][:, None, :], (len(data["time"]), len(n), 2)),
        ), axis=2)
        self.y = data["speeds"][:, n]
        self.m = data["observed"][:, n]
        self.slots = data["slots"]
        self.historical = data["historical"][:, n]

    def __len__(self):
        return len(self.starts)

    def batch(self, index):
        s = self.starts[index]
        past = s[:, None] + np.arange(self.steps)
        future = s[:, None] + self.steps + np.arange(self.horizon)
        return self.x[past], self.y[future], self.m[future], future

    def batches(self, size, rng=None):
        order = np.arange(len(self)) if rng is None else rng.permutation(len(self))
        for i in range(0, len(order), size):
            yield self.batch(order[i:i + size])
