"""Compare logged validation MAE at fixed communication budgets."""
import argparse
import csv
import json
from pathlib import Path

BUDGETS = (0.5, 1.0, 2.0, 4.0, 6.0, 7.0)
VARIANTS = ("fedavg_s0", "fedavg_s1", "fedavg_E2", "fedavg_E4",
            "fedavg_half_clients", "fedavg_16bit", "fedavg_8bit")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

with args.output.open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(("variant", "budget_mb", "round", "used_mb", "validation_mae_mph", "source_run"))
    for variant in VARIANTS:
        matches = sorted((args.project / "runs").glob(f"*_{variant}"))
        if len(matches) != 1:
            raise ValueError(f"Expected one retained run for {variant}; found {len(matches)}")
        log = matches[0] / "log.jsonl"
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        rounds = sorted((r for r in records if {"round", "total_mb", "val_mae"} <= r.keys()),
                        key=lambda r: r["round"])
        if not rounds or any(a["total_mb"] >= b["total_mb"] for a, b in zip(rounds, rounds[1:])):
            raise ValueError(f"Missing rounds or invalid cumulative communication in {log}")
        for budget in BUDGETS:
            eligible = [r for r in rounds if r["total_mb"] <= budget]
            if eligible:
                last = eligible[-1]
                writer.writerow((variant, budget, last["round"], f'{last["total_mb"]:.6f}',
                                 f'{last["val_mae"]:.6f}', matches[0].name))
            else:
                writer.writerow((variant, budget, "", "", "", matches[0].name))
