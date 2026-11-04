"""Turn the logged runs into the paper's tables and figures.

Nothing here is typed in by hand: every number comes from runs/*/run.json, so
re-running the experiments and then this script regenerates the whole results
section. Writes results/tables/*.md|csv, results/figures/*.png, results/summary.json.
"""
import csv
import json
import re

import numpy as np

from .config import ROOT
from .tracking import load_runs

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
HORIZONS = ("15min", "30min", "60min")

LABELS = {
    "last_value": "Last value (persistence)", "historical_average": "Historical average",
    "gru": "GRU (no graph)", "lstm": "LSTM (no graph)", "gconv_mlp": "Graph conv, no recurrence",
    "gconv_gru": "Graph conv + GRU", "gconv_lstm": "Graph conv + LSTM",
    "centralised_full": "Centralised, full road graph", "centralised_cut": "Centralised, graph cut at district borders",
    "local_only": "Local only (each district alone)", "fedavg": "Federated: FedAvg", "fedprox": "Federated: FedProx (μ = 0.01)",
}


def family(name):
    return re.sub(r"_s\d+$", "", name)


def grouped(runs):
    out = {}
    for r in runs:
        out.setdefault(family(r["name"]), []).append(r)
    return out


def stat(runs, key, part="test"):
    values = np.array([r["results"][part][key] for r in runs])
    return values.mean(), values.std(ddof=0), len(values)


def fmt(mean, sd, n, digits=2):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if n > 1 else f"{mean:.{digits}f}"


def write_table(name, header, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    (TABLES / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_table(runs, summary):
    order = ("last_value", "historical_average", "gru", "lstm", "gconv_mlp", "gconv_gru", "gconv_lstm")
    groups = grouped(runs)
    header = ["Model", "Params", "Seeds"] + [f"MAE {h}" for h in HORIZONS] + ["RMSE 60min", "MAPE 60min (%)", "Train time, median (s)"]
    rows = []
    for key in order:
        if key not in groups:
            continue
        rs = groups[key]
        row = [LABELS[key], rs[0]["results"].get("params", 0), len(rs)]
        row += [fmt(*stat(rs, f"mae@{h}")) for h in HORIZONS]
        row += [fmt(*stat(rs, "rmse@60min")), fmt(*stat(rs, "mape@60min"))]
        # Median: one run was stretched by the laptop sleeping mid-training.
        row.append(f"{np.median([r['results'].get('train_s', 0) for r in rs]):.0f}")
        rows.append(row)
        summary.setdefault("benchmark", {})[key] = {
            f"mae@{h}": round(stat(rs, f"mae@{h}")[0], 3) for h in HORIZONS} | {
            "mae": round(stat(rs, "mae")[0], 3), "mae_sd": round(stat(rs, "mae")[1], 3), "seeds": len(rs)}
    write_table("benchmark", header, rows)


def federated_table(runs, summary):
    order = ("centralised_full", "centralised_cut", "local_only", "fedavg", "fedprox")
    groups = grouped(runs)
    header = ["Setting", "Seeds"] + [f"MAE {h}" for h in HORIZONS] + ["MAE (all)", "Data leaves district?", "Comm. (MB)"]
    rows = []
    for key in order:
        if key not in groups:
            continue
        rs = groups[key]
        leaves = "yes (raw data)" if key.startswith("centralised") else "no"
        mb = rs[0]["results"].get("total_mb")
        row = [LABELS[key], len(rs)] + [fmt(*stat(rs, f"mae@{h}")) for h in HORIZONS]
        row += [fmt(*stat(rs, "mae"), digits=3), leaves, f"{mb:.1f}" if mb else "–"]
        rows.append(row)
        summary.setdefault("federated", {})[key] = {
            "mae": round(stat(rs, "mae")[0], 3), "mae_sd": round(stat(rs, "mae")[1], 3),
            **{f"mae@{h}": round(stat(rs, f"mae@{h}")[0], 3) for h in HORIZONS},
            "total_mb": mb, "seeds": len(rs)}
    write_table("federated", header, rows)


TRADEOFF_LABELS = {
    "fedavg_s0": "FedAvg baseline (E = 1, all 4 districts, 32-bit, spatial)",
    "fedavg_E2": "2 local epochs per round", "fedavg_E4": "4 local epochs per round",
    "fedavg_half_clients": "2 of 4 districts per round (24 rounds)",
    "fedavg_random_partition": "Random sensor partition (76% of road edges cut)",
    "fedavg_16bit": "16-bit compressed uploads", "fedavg_8bit": "8-bit compressed uploads",
    "fedprox_s0": "FedProx μ = 0.01", "fedprox_mu0.1": "FedProx μ = 0.1",
}


def rounds_to(log, target):
    for row in log:
        if row["val_mae"] <= target:
            return row["round"], row["total_mb"]
    return None, None


def tradeoff_table(fed_runs, trade_runs, summary):
    by_name = {r["name"]: r for r in fed_runs + trade_runs}
    base = by_name.get("fedavg_s0")
    if not base:
        return
    target = round(base["log"][-1]["val_mae"] * 1.02, 3)   # within 2% of the baseline's final validation MAE
    header = ["Variant", "Test MAE", "Final val MAE", "Rounds", "Total comm. (MB)",
              f"Rounds to val MAE ≤ {target}", "MB to reach it", "Wall time (s)"]
    rows = []
    for name, label in TRADEOFF_LABELS.items():
        r = by_name.get(name)
        if not r:
            continue
        rnd, mb = rounds_to(r["log"], target)
        rows.append([label, f"{r['results']['test']['mae']:.3f}", f"{r['log'][-1]['val_mae']:.3f}", len(r["log"]),
                     f"{r['results']['total_mb']:.1f}", rnd or "not reached", f"{mb:.1f}" if mb else "–",
                     f"{r['results'].get('train_s', 0):.0f}"])
        summary.setdefault("tradeoffs", {})[name] = {
            "test_mae": round(r["results"]["test"]["mae"], 3), "total_mb": r["results"]["total_mb"],
            "rounds_to_target": rnd, "mb_to_target": mb, "train_s": r["results"].get("train_s")}
    summary["tradeoff_target_val_mae"] = target
    write_table("tradeoffs", header, rows)


def figures(bench, fed, trade):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"last_value": "#999", "historical_average": "#bbb", "gru": "#7fb3d5", "lstm": "#2e86c1",
              "gconv_mlp": "#f5b041", "gconv_gru": "#c0392b", "gconv_lstm": "#7d3c98"}

    # Forecast error by horizon.
    groups = grouped(bench)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.array([15, 30, 60])
    for key, color in colors.items():
        if key not in groups:
            continue
        means = np.array([stat(groups[key], f"mae@{h}")[0] for h in HORIZONS])
        sds = np.array([stat(groups[key], f"mae@{h}")[1] for h in HORIZONS])
        ax.errorbar(x, means, yerr=sds, marker="o", ms=4, capsize=3, color=color, label=LABELS[key])
    ax.set_xticks(x)
    ax.set_xlabel("forecast horizon (minutes)")
    ax.set_ylabel("test MAE (mph)")
    ax.set_title("RQ1–2: recurrent encoders, with and without the road graph")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_horizons.png")
    plt.close(fig)

    # Validation curves of centralised training.
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for key in ("gru", "lstm", "gconv_mlp", "gconv_gru", "gconv_lstm"):
        run = next((r for r in groups.get(key, []) if r["name"].endswith("_s0")), None)
        if run:
            ax.plot([row["epoch"] for row in run["log"]], [row["val_mae"] for row in run["log"]],
                    marker=".", color=colors[key], label=LABELS[key])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MAE (mph)")
    ax.set_title("Centralised training curves (seed 0, early stopping)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png")
    plt.close(fig)

    by_name = {r["name"]: r for r in fed + trade}
    central = next((r for r in fed if r["name"] == "centralised_full_s0"), None)
    curves = [(n, l) for n, l in TRADEOFF_LABELS.items() if n in by_name]

    # Convergence: validation MAE per round, and per megabyte communicated.
    for xkey, xlabel, fname, title in (("round", "communication round", "fl_convergence.png", "RQ4: convergence per round"),
                                       ("total_mb", "cumulative communication (MB, up + down)", "fl_communication.png",
                                        "RQ4: accuracy per megabyte communicated")):
        fig, ax = plt.subplots(figsize=(8.2, 3.6))
        for i, (name, label) in enumerate(curves):
            log = by_name[name]["log"]
            ax.plot([row[xkey] for row in log], [row["val_mae"] for row in log], marker=".", ms=3,
                    color=plt.cm.tab10(i), label=label)
        if central:
            ax.axhline(central["results"]["val"]["mae"], color="black", ls="--", lw=1, label="centralised (full graph), best val")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("validation MAE (mph)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        fig.savefig(FIGURES / fname)
        plt.close(fig)

    # Per-district error: local-only vs federated.
    local = next((r for r in fed if r["name"] == "local_only_s0"), None)
    if local:
        fig, ax = plt.subplots(figsize=(4.8, 3))
        rows = [row for row in local["log"] if "client" in row]
        ax.bar([f"D{row['client'] + 1}\n({row['sensors']} sensors)" for row in rows], [row["test_mae"] for row in rows], color="#5dade2")
        for name, color, label in (("fedavg_s0", "#c0392b", "FedAvg, all districts"),
                                   ("centralised_full_s0", "black", "centralised, all districts")):
            if name in by_name:
                ax.axhline(by_name[name]["results"]["test"]["mae"], color=color, ls="--", lw=1, label=label)
        ax.set_ylabel("test MAE (mph)")
        ax.set_title("Each district trained alone (bars)")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES / "local_only_districts.png")
        plt.close(fig)


def privacy_table(runs, fed, summary):
    if not runs:
        return
    header = ["Setting", "Noise multiplier σ", "ε at δ = 1e-5 (12 rounds)", "Test MAE", "MAE 60min", "Final val MAE"]
    rows = []
    for r in sorted(runs, key=lambda r: r["meta_privacy"]["noise_multiplier"] if "meta_privacy" in r else 0):
        p = r.get("meta_privacy", {"noise_multiplier": 0.0, "epsilon": float("inf")})
        label = "Uniform averaging, clipping off, no noise" if not p["noise_multiplier"] else f"Central DP, clip C = {p['clip']}"
        eps = "∞ (no guarantee)" if p["epsilon"] == float("inf") else f"{p['epsilon']:,.1f}"
        test = r["results"]["test"]
        rows.append([label, p["noise_multiplier"], eps, f"{test['mae']:.3f}", f"{test['mae@60min']:.2f}", f"{r['log'][-1]['val_mae']:.3f}"])
        summary.setdefault("privacy", {})[r["name"]] = {"noise": p["noise_multiplier"], "epsilon": p["epsilon"],
                                                          "test_mae": round(test["mae"], 3)}
    write_table("privacy", header, rows)


def privacy_figure(runs, bench):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    noisy = sorted((r for r in runs if r.get("meta_privacy")), key=lambda r: r["meta_privacy"]["epsilon"])
    if not noisy:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot([r["meta_privacy"]["epsilon"] for r in noisy], [r["results"]["test"]["mae"] for r in noisy], "o-", color="#7d3c98")
    for r in noisy:
        ax.annotate(f"σ = {r['meta_privacy']['noise_multiplier']}", (r["meta_privacy"]["epsilon"], r["results"]["test"]["mae"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7)
    plain = next((r for r in runs if not r.get("meta_privacy")), None)
    if plain:
        ax.axhline(plain["results"]["test"]["mae"], color="#c0392b", ls="--", lw=1, label="same setup, no noise")
    ha = next((r for r in bench if r["name"] == "historical_average"), None)
    if ha:
        ax.axhline(ha["results"]["test"]["mae"], color="#999", ls=":", lw=1, label="historical average (no learning)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("privacy budget ε (log scale; smaller = more private)")
    ax.set_ylabel("test MAE (mph, log scale)")
    ax.set_title("RQ5: the price of district-level differential privacy")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "privacy_tradeoff.png")
    plt.close(fig)


def deployment_table(deploy, fed, summary):
    """Same algorithm, two transports: Ray simulation vs real SuperLink/SuperNode processes."""
    run = next((r for r in deploy if r["name"] == "deployment_fedavg"), None)
    sim = next((r for r in fed if r["name"] == "fedavg_s0"), None)
    if not run or not sim:
        return
    header = ["Round", "Validation MAE, simulation (Ray)", "Validation MAE, deployment (SuperLink + 4 SuperNodes)", "Difference"]
    rows = []
    for a, b in zip(sim["log"], run["log"]):
        rows.append([a["round"], f"{a['val_mae']:.4f}", f"{b['val_mae']:.4f}", f"{b['val_mae'] - a['val_mae']:+.4f}"])
    write_table("deployment", header, rows)
    summary["deployment"] = {"rounds": len(run["log"]), "test_mae_after_3_rounds": round(run["results"]["test"]["mae"], 3),
                             "wall_s": run["results"]["train_s"], "transport": run["results"]["transport"],
                             "max_abs_val_diff_vs_simulation": round(max(abs(float(r[3])) for r in rows), 4)}


def main():
    bench, fed, trade = load_runs("benchmark"), load_runs("federated"), load_runs("tradeoffs")
    priv = load_runs("privacy")
    deploy = load_runs("deployment")
    for r in priv:
        meta = __import__("json").loads((r["dir"] / "run.json").read_text(encoding="utf-8"))
        if "privacy" in meta:
            r["meta_privacy"] = meta["privacy"]
    every = bench + fed + trade + priv + deploy
    summary = {"runs": {"benchmark": len(bench), "federated": len(fed), "tradeoffs": len(trade), "privacy": len(priv),
                        "deployment": len(deploy)},
               "commits": sorted({r["git"]["commit"][:7] for r in every}),
               "dirty_runs": sum(r["git"]["dirty"] for r in every)}
    benchmark_table(bench, summary)
    federated_table(fed, summary)
    tradeoff_table(fed, trade, summary)
    privacy_table(priv, fed, summary)
    deployment_table(deploy, fed, summary)
    a = next((r for r in bench if r["name"] == "gconv_gru_s0"), None)
    b = next((r for r in fed if r["name"] == "centralised_full_s0"), None)
    if a and b:   # the same configuration, run twice in different sweeps
        summary["reproducibility_check"] = {"benchmark_gconv_gru_s0": a["results"]["test"]["mae"],
                                            "federated_centralised_full_s0": b["results"]["test"]["mae"]}
    figures(bench, fed, trade)
    privacy_figure(priv, bench)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
