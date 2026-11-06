"""Paired comparisons over test days.

Forecast errors are strongly correlated within a day (one incident affects hours of
windows at many sensors), so windows are not independent samples. The unit of
resampling is the day: both methods are scored on the same resampled days, and the
confidence interval is for the difference in pooled MAE.
"""
import numpy as np


def daily_arrays(daily_list, step=None):
    """Pool one or more runs' daily records (e.g. seeds) into per-day abs and count arrays.

    step=None pools all horizon steps; otherwise one 0-based step. Seeds are pooled by
    summing, which equals averaging their MAE because every seed sees the same readings.
    """
    days = sorted(set.intersection(*(set(d["day"]) for d in daily_list)))
    abs_, count = np.zeros(len(days)), np.zeros(len(days))
    for d in daily_list:
        index = {day: i for i, day in enumerate(d["day"])}
        a = np.array([d["abs"][index[day]] for day in days])
        c = np.array([d["count"][index[day]] for day in days])
        if step is None:
            abs_ += a.sum(axis=1)
            count += c.sum(axis=1)
        else:
            abs_ += a[:, step]
            count += c[:, step]
    return np.array(days), abs_, count


def paired_bootstrap(a, b, step=None, resamples=10000, seed=0):
    """MAE(a) - MAE(b) with a 95% day-level bootstrap interval. a, b: lists of daily records.

    Negative means a is better. Also returns on how many days a had the lower MAE.
    """
    days_a, abs_a, cnt_a = daily_arrays(a, step)
    days_b, abs_b, cnt_b = daily_arrays(b, step)
    if not np.array_equal(days_a, days_b) or not np.allclose(cnt_a / len(a), cnt_b / len(b)):
        raise ValueError("the two methods were not scored on the same days and readings")
    diff = abs_a.sum() / cnt_a.sum() - abs_b.sum() / cnt_b.sum()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(days_a), size=(resamples, len(days_a)))
    boot = abs_a[pick].sum(1) / cnt_a[pick].sum(1) - abs_b[pick].sum(1) / cnt_b[pick].sum(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    wins = int((abs_a / cnt_a < abs_b / cnt_b).sum())
    return {"diff": float(diff), "low": float(low), "high": float(high), "days": len(days_a), "a_better_days": wins}
