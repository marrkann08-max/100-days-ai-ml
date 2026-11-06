"""Forecast errors computed only where a real reading exists.

Interpolated or historically filled values never count as ground truth.
"""
import numpy as np
import torch

HORIZONS = {"15 min": 2, "30 min": 5, "60 min": 11}   # 0-based index into the 12 steps


def masked_mae_loss(pred, target, mask):
    mask = mask.float()
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1)


class Accumulator:
    """Running sums per horizon step so metrics can be pooled over batches and clients."""

    def __init__(self, horizon=12):
        self.abs = np.zeros(horizon)
        self.sq = np.zeros(horizon)
        self.pct = np.zeros(horizon)
        self.count = np.zeros(horizon)
        self.days = {}   # day index -> [abs (horizon,), count (horizon,)], for paired tests over days

    def add(self, pred, target, mask, days=None):
        """pred, target, mask: (B, horizon, N) arrays in mph; days: (B,) day index of each forecast."""
        err = np.where(mask, pred - target, 0.0)
        self.abs += np.abs(err).sum(axis=(0, 2))
        self.sq += (err ** 2).sum(axis=(0, 2))
        self.pct += np.where(mask, np.abs(err) / np.where(mask, target, 1), 0).sum(axis=(0, 2))
        self.count += mask.sum(axis=(0, 2))
        if days is not None:
            per_abs, per_count = np.abs(err).sum(axis=2), mask.sum(axis=2)
            for day in np.unique(days):
                rows = days == day
                slot = self.days.setdefault(int(day), [np.zeros(len(self.abs)), np.zeros(len(self.abs))])
                slot[0] += per_abs[rows].sum(axis=0)
                slot[1] += per_count[rows].sum(axis=0)

    def merge(self, other):
        for key in ("abs", "sq", "pct", "count"):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        for day, (a, c) in other.days.items():
            slot = self.days.setdefault(day, [np.zeros(len(self.abs)), np.zeros(len(self.abs))])
            slot[0] += a
            slot[1] += c
        return self

    def daily(self):
        """Per-day absolute-error sums and counts per horizon step, JSON-ready."""
        days = sorted(self.days)
        return {"day": days, "abs": [self.days[d][0].tolist() for d in days],
                "count": [self.days[d][1].tolist() for d in days]}

    def result(self):
        c = np.maximum(self.count, 1)
        mae, rmse, mape = self.abs / c, np.sqrt(self.sq / c), 100 * self.pct / c
        out = {"mae": float(self.abs.sum() / c.sum()), "rmse": float(np.sqrt(self.sq.sum() / c.sum())),
               "mape": float(100 * self.pct.sum() / c.sum()), "count": int(self.count.sum())}
        for label, k in HORIZONS.items():
            key = label.replace(" ", "")
            out[f"mae@{key}"], out[f"rmse@{key}"], out[f"mape@{key}"] = float(mae[k]), float(rmse[k]), float(mape[k])
        return out

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("abs", "sq", "pct", "count")}

    @classmethod
    def from_dict(cls, d):
        acc = cls(len(d["abs"]))
        for k in ("abs", "sq", "pct", "count"):
            setattr(acc, k, np.array(d[k], float))
        return acc
