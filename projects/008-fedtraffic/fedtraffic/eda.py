"""Exploratory analysis: what does the data look like before any model touches it?

Writes figures to results/figures/ and headline numbers to results/eda.json.
"""
import json

import numpy as np

from . import config as cfg
from . import data as ds
from .partition import partition

FIGURES = cfg.ROOT / "results" / "figures"


def stats(d):
    s, o = d["speeds"], d["observed"]
    train_end, val_end = d["split"]
    per_sensor = 1 - o.mean(axis=0)
    adj = d["adjacency"]
    return {
        "timesteps": int(s.shape[0]),
        "sensors": int(s.shape[1]),
        "start": str(d["stamps"][0].astype("datetime64[ns]").astype("datetime64[m]")),
        "end": str(d["stamps"][-1].astype("datetime64[ns]").astype("datetime64[m]")),
        "missing_pct": round(100 * float(1 - o.mean()), 2),
        "interpolated_pct": round(100 * float(d["interpolated"].mean()), 2),
        "missing_pct_by_split": [round(100 * float(1 - o[a:b].mean()), 2)
                                 for a, b in ((0, train_end), (train_end, val_end), (val_end, len(o)))],
        "worst_sensor_missing_pct": round(100 * float(per_sensor.max()), 2),
        "sensors_over_20pct_missing": int((per_sensor > .2).sum()),
        "speed_mean_mph": round(float(s[o].mean()), 2),
        "speed_std_mph": round(float(s[o].std()), 2),
        "speed_max_mph": float(s[o].max()),
        "edges_incl_self_loops": int((adj > 0).sum()),
        "mean_out_degree": round(float((adj > 0).sum(axis=1).mean() - 1), 2),
        "split_timesteps": [int(train_end), int(val_end - train_end), int(len(o) - val_end)],
    }


def correlation_by_distance(d, pairs=4000, seed=0):
    rng = np.random.default_rng(seed)
    s, o = d["speeds"], d["observed"]
    train_end = d["split"][0]
    x = np.where(o[:train_end], s[:train_end], np.nan)
    n = x.shape[1]
    a, b = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    dist = d["distances"][a, b]
    corr = []
    for i, j in zip(a, b):
        both = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
        corr.append(np.corrcoef(x[both, i], x[both, j])[0, 1])
    return dist / 1000, np.array(corr)


def figures(d, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    s, o = d["speeds"], d["observed"]

    # 1. Missingness: sensors × days.
    days = len(s) // ds.STEPS_PER_DAY
    miss = 1 - o[:days * ds.STEPS_PER_DAY].reshape(days, ds.STEPS_PER_DAY, -1).mean(axis=1)
    order = np.argsort(miss.mean(axis=0))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    im = ax.imshow(miss[:, order].T, aspect="auto", cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
    for boundary in d["split"] / ds.STEPS_PER_DAY:
        ax.axvline(boundary, color="#2a7fff", lw=1.2)
    ax.set_xlabel("day (March 1 – June 27, 2012); blue lines = train | val | test boundaries")
    ax.set_ylabel("sensor (sorted by missingness)")
    fig.colorbar(im, ax=ax, label="share of day missing")
    ax.set_title("Missing readings per sensor and day")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_missingness.png")
    plt.close(fig)

    # 2. Daily speed profile, weekday vs weekend, with spread across sensors.
    train_end = d["split"][0]
    stamps = d["stamps"][:train_end].astype("datetime64[ns]").astype("datetime64[D]")
    weekday = ((stamps.astype(np.int64) + 3) % 7) < 5
    tod = np.arange(train_end) % ds.STEPS_PER_DAY
    x = np.where(o[:train_end], s[:train_end], np.nan)
    fig, ax = plt.subplots(figsize=(7.5, 3))
    hours = np.arange(ds.STEPS_PER_DAY) / 12
    for label, sel, color in (("weekday", weekday, "#d1495b"), ("weekend", ~weekday, "#2a7fff")):
        prof = np.array([np.nanmean(x[sel & (tod == k)], axis=0) for k in range(ds.STEPS_PER_DAY)])
        mid = np.nanmedian(prof, axis=1)
        lo, hi = np.nanpercentile(prof, [10, 90], axis=1)
        ax.plot(hours, mid, color=color, label=f"{label} (median sensor)")
        ax.fill_between(hours, lo, hi, color=color, alpha=.15)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("time of day (h)")
    ax.set_ylabel("speed (mph)")
    ax.set_title("Daily speed profile (training period; band = 10th–90th percentile of sensors)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_daily_profile.png")
    plt.close(fig)

    # 3. Correlation vs road distance: the reason a graph should help.
    dist, corr = correlation_by_distance(d)
    finite = np.isfinite(dist)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.scatter(dist[finite], corr[finite], s=4, alpha=.35, color="#444")
    ax.set_xlabel("road-network distance (km)")
    ax.set_ylabel("speed correlation")
    ax.set_title(f"Nearby sensors move together\n({finite.mean():.0%} of random pairs connected by road)")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_correlation_distance.png")
    plt.close(fig)

    # 4. Sensor map with the federated districts.
    clients = config["federated"]["clients"]
    groups = partition(d["locations"], clients, "spatial", seed=config["seed"])
    lat, lon = d["locations"].T
    adj = d["adjacency"]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    rows, cols = np.nonzero(adj)
    for i, j in zip(rows, cols):
        if i != j:
            same = any(i in g and j in g for g in groups)
            ax.plot(lon[[i, j]], lat[[i, j]], color="#bbb" if same else "#e63946", lw=.5 if same else .9, zorder=1)
    colors = ["#2a9d8f", "#e9c46a", "#264653", "#f4a261", "#8ab17d", "#9b5de5", "#00bbf9", "#f15bb5"]
    for k, g in enumerate(groups):
        ax.scatter(lon[g], lat[g], s=14, color=colors[k % len(colors)], label=f"district {k + 1} ({len(g)})", zorder=2)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("METR-LA sensors, road graph and federated districts\n(red = edges cut by the partition)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_sensor_map.png")
    plt.close(fig)

    near = finite & (dist < 2)
    far = finite & (dist > 10)
    return {"corr_under_2km": round(float(np.nanmedian(corr[near])), 3),
            "corr_over_10km": round(float(np.nanmedian(corr[far])), 3),
            "corr_unconnected": round(float(np.nanmedian(corr[~finite])), 3)}


def main():
    config = cfg.load()
    d = ds.load(config)
    summary = stats(d)
    summary.update(figures(d, config))
    out = cfg.ROOT / "results" / "eda.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
