"""Round-2 tables and figures, from runs/ only (groups benchmark_v2 and federated_v2).

    python -m fedtraffic.report_v2

Every comparison between two methods is paired: both are scored on the same test
readings, and the 95% interval comes from resampling test days (stats.py).
Writes results/tables/v2_*.md|csv, results/figures/v2_*.png and results/summary_v2.json.
"""
import json

import numpy as np

from .report import FIGURES, ROOT, TABLES, family, write_table
from .stats import daily_arrays, paired_bootstrap
from .tracking import load_runs

# Reference categorical palette, light mode, fixed order (validated: adjacent CVD ΔE ≥ 9.1).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STEPS = {"15min": 2, "30min": 5, "60min": 11}

LABELS = {
    "last_value": "Last value (persistence)",
    "historical_average": "Historical average (weekly profile)",
    "seasonal_residual": "Seasonal-residual (no training)",
    "gconv_lstm_v1": "Graph LSTM, v1 budget (12 epochs)",
    "gconv_lstm_long": "Graph LSTM, 50-epoch budget",
    "lstm_seasonal": "Seasonal LSTM, no graph",
    "gconv_lstm_seasonal": "Seasonal graph LSTM",
    "centralised": "Centralised (all raw data pooled)",
    "local_only": "Local only (each district alone)",
    "fedavg": "FedAvg (one shared model)",
    "personalised": "FedAvg + local fine-tuning (personalised)",
}


class Method:
    """One row of a table: the runs behind it and where their results live."""

    def __init__(self, key, runs, part=None):
        self.key, self.runs = key, runs
        self.results = [r["results"][part] if part else r["results"] for r in runs]

    def label(self):
        return LABELS[self.key]

    def daily(self):
        return [r["test_daily"] for r in self.results]

    def stat(self, metric):
        values = np.array([r["test"][metric] for r in self.results])
        return values.mean(), values.std(ddof=0)

    def fmt(self, metric, digits=3):
        mean, sd = self.stat(metric)
        return f"{mean:.{digits}f} ± {sd:.{digits}f}" if len(self.results) > 1 else f"{mean:.{digits}f}"

    def per_step(self):
        """Test MAE at each of the 12 horizon steps, pooled over seeds."""
        out = []
        for step in range(12):
            _, a, c = daily_arrays(self.daily(), step)
            out.append(a.sum() / c.sum())
        return np.array(out)


def methods():
    bench = {}
    for r in load_runs("benchmark_v2"):
        bench.setdefault(family(r["name"]), []).append(r)
    fed = {}
    for r in load_runs("federated_v2"):
        fed.setdefault(family(r["name"]), []).append(r)
    m = {k: Method(k, v) for k, v in bench.items()}
    if "gconv_lstm_seasonal" in bench:
        m["centralised"] = Method("centralised", bench["gconv_lstm_seasonal"])
    if "local_only_seasonal" in fed:
        m["local_only"] = Method("local_only", fed["local_only_seasonal"])
    if "fedavg_seasonal" in fed:
        m["fedavg"] = Method("fedavg", fed["fedavg_seasonal"])
        m["personalised"] = Method("personalised", fed["fedavg_seasonal"], part="personalised")
    return m


def stopped_at(run):
    epochs = [row for row in run["log"] if row.get("epoch", 0) > 0]
    best = min(epochs, key=lambda row: row["val_mae"])
    return len(epochs), best["epoch"]


def benchmark_table(m, summary):
    v1 = [r for r in load_runs("benchmark") if family(r["name"]) == "gconv_lstm"]
    header = ["Model", "Seeds", "MAE 15 min", "MAE 30 min", "MAE 60 min", "MAE (all)", "Epochs run (best)",
              "Train time, median (s)"]
    rows = []
    if v1:
        vals = {h: np.mean([r["results"]["test"][f"mae@{h}"] for r in v1]) for h in STEPS}
        allm = np.mean([r["results"]["test"]["mae"] for r in v1])
        rows.append([LABELS["gconv_lstm_v1"] + " (v1, for reference)", len(v1), *(f"{vals[h]:.3f}" for h in STEPS),
                     f"{allm:.3f}", "12 (12)", f"{np.median([r['results']['train_s'] for r in v1]):.0f}"])
    for key in ("last_value", "historical_average", "seasonal_residual", "gconv_lstm_long", "lstm_seasonal",
                "gconv_lstm_seasonal"):
        if key not in m:
            continue
        meth = m[key]
        learned = "params" in meth.results[0]
        epochs = "–"
        if learned:
            st = [stopped_at(r) for r in meth.runs]
            epochs = ", ".join(f"{a} ({b})" for a, b in st)
        rows.append([meth.label(), len(meth.runs), *(meth.fmt(f"mae@{h}") for h in STEPS), meth.fmt("mae"), epochs,
                     f"{np.median([r['train_s'] for r in meth.results]):.0f}" if learned else "0"])
        summary.setdefault("benchmark_v2", {})[key] = {
            "mae": round(meth.stat("mae")[0], 4), "mae_sd": round(meth.stat("mae")[1], 4),
            **{f"mae@{h}": round(meth.stat(f"mae@{h}")[0], 4) for h in STEPS}, "seeds": len(meth.runs)}
    write_table("v2_benchmark", header, rows)


def federated_table(m, summary):
    header = ["Setting", "Seeds", "MAE 15 min", "MAE 30 min", "MAE 60 min", "MAE (all)", "Raw data leaves district?",
              "Communication (MB)"]
    rows = []
    for key in ("seasonal_residual", "centralised", "local_only", "fedavg", "personalised"):
        if key not in m:
            continue
        meth = m[key]
        leaves = "yes" if key == "centralised" else "no"
        mb = meth.runs[0]["results"].get("total_mb") if key in ("fedavg", "personalised") else 0
        rows.append([meth.label(), len(meth.runs), *(meth.fmt(f"mae@{h}") for h in STEPS), meth.fmt("mae"),
                     leaves, f"{mb:.1f}" if mb else "0"])
        summary.setdefault("federated_v2", {})[key] = {
            "mae": round(meth.stat("mae")[0], 4), "mae_sd": round(meth.stat("mae")[1], 4),
            **{f"mae@{h}": round(meth.stat(f"mae@{h}")[0], 4) for h in STEPS}, "total_mb": mb}
    write_table("v2_federated", header, rows)


COMPARISONS = [
    # (a, b, question). Negative difference = a is better.
    ("gconv_lstm_seasonal", "gconv_lstm_long", "Does the seasonal input and target help?"),
    ("gconv_lstm_seasonal", "seasonal_residual", "Does learning beat the tuned seasonal baseline?"),
    ("gconv_lstm_seasonal", "historical_average", "Does learning beat the weekly profile?"),
    ("gconv_lstm_seasonal", "lstm_seasonal", "Does the road graph still help?"),
    ("fedavg", "centralised", "Cost of federation (one shared model)"),
    ("fedavg", "local_only", "Shared model vs each district alone"),
    ("personalised", "local_only", "Personalised FL vs each district alone"),
    ("personalised", "fedavg", "Gain from local fine-tuning"),
    ("personalised", "centralised", "Personalised FL vs pooling all raw data"),
]


def comparison_table(m, summary):
    header = ["Question", "A", "B", "Horizon", "MAE(A) − MAE(B)", "95% CI (days)", "Days A better"]
    rows, forest = [], []
    for a, b, question in COMPARISONS:
        if a not in m or b not in m:
            continue
        for label, step in (("all", None), ("60 min", 11)):
            r = paired_bootstrap(m[a].daily(), m[b].daily(), step)
            rows.append([question, m[a].label(), m[b].label(), label, f"{r['diff']:+.3f}",
                         f"[{r['low']:+.3f}, {r['high']:+.3f}]", f"{r['a_better_days']}/{r['days']}"])
            summary.setdefault("comparisons", []).append({"a": a, "b": b, "horizon": label, **r})
            if step is None:
                forest.append((question, r))
    write_table("v2_comparisons", header, rows)
    return forest


def district_table(m, summary):
    if "fedavg" not in m or "local_only" not in m:
        return None
    local = np.mean([[row["test_mae"] for row in r["log"] if "client" in row] for r in m["local_only"].runs], axis=0)
    glob = np.mean([r["results"]["district_test_mae"] for r in m["fedavg"].runs], axis=0)
    pers = np.mean([[row["personal_test_mae"] for row in r["log"] if "personal_test_mae" in row]
                    for r in m["fedavg"].runs], axis=0)
    sizes = m["fedavg"].runs[0]["results"]["clients"]
    header = ["District", "Sensors", "Local only", "FedAvg", "FedAvg + fine-tuning", "Best"]
    rows = []
    for k, n in enumerate(sizes):
        values = {"Local only": local[k], "FedAvg": glob[k], "FedAvg + fine-tuning": pers[k]}
        rows.append([f"D{k + 1}", n, *(f"{v:.3f}" for v in values.values()), min(values, key=values.get)])
    write_table("v2_districts", header, rows)
    summary["districts_v2"] = {"local_only": local.round(4).tolist(), "fedavg": glob.round(4).tolist(),
                               "personalised": pers.round(4).tolist(), "sensors": sizes}
    return sizes, local, glob, pers


def style(plt):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False})


def figures(m, forest, districts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    FIGURES.mkdir(parents=True, exist_ok=True)
    minutes = np.arange(1, 13) * 5

    # Error by horizon, all 12 steps.
    keys = [k for k in ("historical_average", "seasonal_residual", "gconv_lstm_long", "lstm_seasonal",
                        "gconv_lstm_seasonal") if k in m]
    if keys:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ends = {}
        for i, key in enumerate(keys):
            y = m[key].per_step()
            ax.plot(minutes, y, color=SERIES[i], lw=2, marker="o", ms=3.5, label=m[key].label())
            ends[key] = y[-1]
        # End labels, nudged apart so close values stay readable.
        placed = []
        for key, v in sorted(ends.items(), key=lambda kv: kv[1]):
            pos = max(v, placed[-1] + 0.07) if placed else v
            placed.append(pos)
            ax.annotate(f"{v:.2f}", (minutes[-1], v), xytext=(minutes[-1] + 2.5, pos), textcoords="data",
                        va="center", fontsize=7, color=MUTED)
        ax.set_xticks([5, 15, 30, 45, 60])
        ax.set_xlim(3, 69)
        ax.set_xlabel("forecast horizon (minutes)")
        ax.set_ylabel("test MAE (mph)")
        ax.set_title("Test error by horizon (v2; seasonal graph LSTM is the mean of 3 seeds)", loc="left")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIGURES / "v2_horizons.png")
        plt.close(fig)

    # Paired differences with day-level 95% intervals.
    if forest:
        fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(forest) + 1.0))
        y = np.arange(len(forest))[::-1]
        for yi, (question, r) in zip(y, forest):
            ax.plot([r["low"], r["high"]], [yi, yi], color=SERIES[0], lw=2, solid_capstyle="round")
            ax.plot(r["diff"], yi, "o", color=SERIES[0], ms=6, markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(f"{r['diff']:+.3f}", (r["high"], yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7, color=MUTED)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(y, [q for q, _ in forest], fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("difference in test MAE, A − B (mph; left of 0 = A better)")
        ax.set_title("Paired comparisons, 95% intervals over test days", loc="left")
        fig.tight_layout()
        fig.savefig(FIGURES / "v2_comparisons.png")
        plt.close(fig)

    # Per-district: local only vs FedAvg vs personalised.
    if districts:
        sizes, local, glob, pers = districts
        fig, ax = plt.subplots(figsize=(6.4, 3.3))
        x = np.arange(len(sizes))
        width = 0.26
        for i, (label, values) in enumerate((("Local only", local), ("FedAvg", glob), ("FedAvg + fine-tuning", pers))):
            bars = ax.bar(x + (i - 1) * (width + 0.02), values, width, color=SERIES[i], label=label)
            for bar, v in zip(bars, values):
                ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
        ax.set_xticks(x, [f"D{k + 1} ({n} sensors)" for k, n in enumerate(sizes)])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("test MAE (mph)")
        ax.set_title(f"Per-district test error (mean of {len(m['fedavg'].runs)} seeds)", loc="left")
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.set_ylim(0, max(local.max(), glob.max(), pers.max()) * 1.18)
        fig.tight_layout()
        fig.savefig(FIGURES / "v2_districts.png")
        plt.close(fig)

    # Training curves, seed 0: v1's 12-epoch budget vs training to convergence.
    curves = [(k, next((r for r in m[k].runs if r["name"].endswith("_s0")), None))
              for k in ("gconv_lstm_long", "lstm_seasonal", "gconv_lstm_seasonal") if k in m]
    if curves:
        fig, ax = plt.subplots(figsize=(6.4, 3.3))
        for i, (key, run) in enumerate(curves):
            if run is None:
                continue
            rows = [row for row in run["log"] if row.get("epoch", 0) > 0]
            ax.plot([row["epoch"] for row in rows], [row["val_mae"] for row in rows], color=SERIES[i + 2], lw=2,
                    label=m[key].label())
        ax.axvline(12, color=MUTED, lw=1, ls="--")
        ax.annotate("v1 stopped here", (12, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points",
                    fontsize=7, color=MUTED)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation MAE (mph)")
        ax.set_title("Validation curves, seed 0 (up to 50 epochs, early stopping with patience 5)", loc="left")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES / "v2_training_curves.png")
        plt.close(fig)


def main():
    m = methods()
    summary = {"runs": {g: len(load_runs(g)) for g in ("benchmark_v2", "federated_v2")}}
    every = load_runs("benchmark_v2") + load_runs("federated_v2")
    summary["commits"] = sorted({r["git"]["commit"][:7] for r in every})
    summary["dirty_runs"] = sum(r["git"]["dirty"] for r in every)
    benchmark_table(m, summary)
    federated_table(m, summary)
    forest = comparison_table(m, summary)
    districts = district_table(m, summary)
    figures(m, forest, districts)
    (ROOT / "results" / "summary_v2.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
