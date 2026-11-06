"""Evaluate a causal seasonal-residual traffic forecaster on Day 008's splits."""
import argparse
import csv
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("cache", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
with np.load(args.cache, allow_pickle=False) as data:
    speeds, observed, slots, seasonal = (
        data[key] for key in ("speeds", "observed", "slots", "historical")
    )
    train_end, val_end = data["split"]


def windows(lo, hi):
    return np.arange(lo + 11, hi - 12, 3)


def residual(origins, lookback):
    past = origins[:, None] - np.arange(lookback)
    valid = observed[past]
    deviation = speeds[past] - seasonal[slots[past]]
    return np.where(valid, deviation, 0).sum(axis=1) / valid.sum(axis=1).clip(min=1)


def target(origins, step):
    future = origins + step
    return speeds[future], observed[future], seasonal[slots[future]]


def mae(actual, valid, prediction):
    return float(np.abs(actual[valid] - prediction[valid]).mean())


val = windows(train_end, val_end)
test = windows(val_end, len(speeds))
lookbacks = (1, 3, 6, 12)
validation_residuals = {k: residual(val, k) for k in lookbacks}
with args.output.open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(("horizon_minutes", "lookback_steps", "residual_weight",
                     "val_seasonal_mae", "val_hybrid_mae", "test_seasonal_mae",
                     "test_hybrid_mae", "test_observations"))
    for step in (3, 6, 12):
        actual, valid, baseline = target(val, step)
        score, weight, lookback = min(
            (mae(actual, valid, (baseline + weight * delta).clip(min=0)), weight, k)
            for k, delta in validation_residuals.items()
            for weight in np.linspace(0, 1, 21)
        )
        val_baseline = mae(actual, valid, baseline)
        actual, valid, baseline = target(test, step)
        prediction = (baseline + weight * residual(test, lookback)).clip(min=0)
        writer.writerow((step * 5, lookback, f"{weight:.2f}", f"{val_baseline:.6f}",
                         f"{score:.6f}", f"{mae(actual, valid, baseline):.6f}",
                         f"{mae(actual, valid, prediction):.6f}", int(valid.sum())))
