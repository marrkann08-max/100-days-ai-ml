"""Retrain the Day 008 graph LSTM and compare paired 60-minute forecasts."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--seasonal", type=Path, default=Path("results/tables/day008_seasonal_results.csv"),
                    help="Output of day008_seasonal_residual.py; supplies the 60-minute settings.")
args = parser.parse_args()
project = args.project.resolve()
sys.path.insert(0, str(project))
from fedtraffic import data, partition, train

seasonal_file = args.seasonal if args.seasonal.is_absolute() else project / args.seasonal
with seasonal_file.open(encoding="utf-8") as f:
    settings, = (row for row in csv.DictReader(f) if row["horizon_minutes"] == "60")
lookback, weight = int(settings["lookback_steps"]), float(settings["residual_weight"])
print(f"Seasonal-residual settings (selected on validation): lookback={lookback}, weight={weight}")

run_dir, = (project / "runs").glob("*_gconv_lstm_s0")
config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
logged_mae = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["results"]["test"]["mae@60min"]
for key in ("raw_dir", "cache_dir"):
    config["data"][key] = str(project / config["data"][key])
dataset = data.load(config)
model, _, benchmark_test = train.fit(config, dataset)
print(f"Retrained graph LSTM: {benchmark_test['mae@60min']:.6f} (logged {logged_mae:.6f})", flush=True)
if not np.isclose(benchmark_test["mae@60min"], logged_mae, rtol=0, atol=1e-3):
    raise SystemExit(f"Retrained model does not reproduce logged run {run_dir.name}; "
                     "the comparison would not use the reported model.")
nodes = np.arange(len(dataset["locations"]))
graph = train.subgraph(dataset, nodes, config["model"]["diffusion_steps"])
groups = partition.partition(dataset["locations"], 4, seed=config["federated"].get("partition_seed", 0))
windows = train.windows_for(dataset, config, "test")
mean, std = dataset["norm"]
model.eval()
graph_predictions = []
with torch.no_grad():
    for x, _, _, future in windows.batches(64):
        graph_predictions.append(train.predict(model, graph, windows, x, future, mean, std)[:, 11])
graph_prediction = np.concatenate(graph_predictions)
origin = windows.starts + 11
future = origin + 12
speeds, observed, slots, seasonal = (
    dataset[key] for key in ("speeds", "observed", "slots", "historical")
)
past = origin[:, None] - np.arange(lookback)
valid_past = observed[past]
residual = np.where(valid_past, speeds[past] - seasonal[slots[past]], 0).sum(axis=1)
residual /= valid_past.sum(axis=1).clip(min=1)
seasonal_prediction = (seasonal[slots[future]] + weight * residual).clip(min=0)
actual, observed_test = speeds[future], observed[future]
target_slots = slots[future]
hours = (target_slots % 288) // 12
rush = (target_slots // 288 < 5) & (
    ((hours >= 7) & (hours < 10)) | ((hours >= 16) & (hours < 19))
)

with args.output.open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(("district", "period", "sensors", "observations", "graph_lstm_mae_mph",
                     "seasonal_residual_mae_mph", "seasonal_minus_lstm_mph"))
    for district, sensors in [("all", nodes), *enumerate(groups, 1)]:
        for period, rows in (("all", np.ones(len(future), dtype=bool)), ("rush", rush),
                             ("other", ~rush)):
            ix = np.ix_(rows, sensors)
            valid = observed_test[ix]
            truth = actual[ix][valid]
            lstm = np.abs(graph_prediction[ix][valid] - truth).mean()
            hybrid = np.abs(seasonal_prediction[ix][valid] - truth).mean()
            writer.writerow((district, period, len(sensors), len(truth),
                             f"{lstm:.6f}", f"{hybrid:.6f}", f"{hybrid-lstm:.6f}"))
