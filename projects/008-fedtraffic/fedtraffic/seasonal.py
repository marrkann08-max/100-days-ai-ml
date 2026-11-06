"""Seasonal-residual forecaster: the weekly profile, nudged by how far traffic is from it now.

forecast(t + h) = profile(t + h) + w_h * mean deviation from the profile over the last k
observed readings, clipped at 0 mph. For each horizon step h, the lookback k and the
weight w_h are chosen on the validation split only. It has no trained parameters, needs
nothing but a sensor's own history, and so is trivially federated.
"""
import numpy as np

LOOKBACKS = (1, 3, 6, 12)
WEIGHTS = np.round(np.linspace(0, 1, 21), 2)


def deviation(windows, future, lookback):
    """Mean observed deviation from the profile over the last `lookback` inputs: (B, N)."""
    past = (future[:, :1] - 1) - np.arange(lookback)            # (B, lookback), newest first
    valid = windows.m[past]
    delta = windows.y[past] - windows.historical[windows.slots[past]]
    return np.where(valid, delta, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)


class SeasonalResidual:
    name = "seasonal_residual"

    def __init__(self, lookback=None, weight=None):
        self.lookback, self.weight = lookback, weight   # one value per horizon step

    def fit(self, windows, batch_size=128):
        errors = np.zeros((len(LOOKBACKS), len(WEIGHTS), windows.horizon))
        for _, y, m, future in windows.batches(batch_size):
            base = windows.historical[windows.slots[future]]
            for i, k in enumerate(LOOKBACKS):
                dev = deviation(windows, future, k)[:, None]
                for j, w in enumerate(WEIGHTS):
                    err = np.abs(np.maximum(base + w * dev, 0) - y)
                    errors[i, j] += np.where(m, err, 0).sum(axis=(0, 2))
        flat = errors.reshape(-1, windows.horizon).argmin(axis=0)
        self.lookback = [LOOKBACKS[i] for i in flat // len(WEIGHTS)]
        self.weight = [float(WEIGHTS[j]) for j in flat % len(WEIGHTS)]
        return self

    def predict(self, windows, future):
        base = windows.historical[windows.slots[future]]
        devs = {k: deviation(windows, future, k) for k in set(self.lookback)}
        adjust = np.stack([w * devs[k] for k, w in zip(self.lookback, self.weight)], axis=1)
        return np.maximum(base + adjust, 0)

    def settings(self):
        return {"lookback": self.lookback, "weight": self.weight}
