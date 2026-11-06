"""Named experiment groups. Each run is logged to runs/ and can be repeated exactly.

    python -m fedtraffic.experiments benchmark      # RQ1-2: recurrent encoders, graph or not (centralised)
    python -m fedtraffic.experiments federated      # RQ3: centralised vs local-only vs federated
    python -m fedtraffic.experiments tradeoffs      # RQ4: rounds, local epochs, participation, FedProx, compression
    python -m fedtraffic.experiments privacy        # RQ5: accuracy cost of differential privacy
    python -m fedtraffic.experiments benchmark_v2   # RQ6: seasonal-aware encoder, trained to convergence
    python -m fedtraffic.experiments federated_v2   # RQ7: personalised federated learning
    python -m fedtraffic.experiments smoke          # tiny end-to-end check
"""
import argparse
import json
import shutil
import subprocess
import sys
import time

import torch

from . import config as cfg
from . import data as ds
from . import train
from .partition import partition
from .tracking import RUNS, Run, load_runs

SEEDS = (0, 1, 2)


def keep_awake():
    """Ask Windows not to sleep while this process runs (reverts automatically on exit).

    A suspended laptop breaks Ray's connections and leaves Flower waiting forever.
    """
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def federated_job(name, overrides, group=None, timeout=1800, attempts=2):
    """Run one Flower simulation in a child process, with a time limit and one retry.

    Each attempt gets a fresh Ray cluster, so one hung simulation can't stall the sweep.
    """
    target = group or ("federated" if name.split("_s")[0] in ("fedavg", "fedprox") else "tradeoffs")
    for attempt in range(1, attempts + 1):
        if done(target, name):
            print(f"skip {name} (already in runs/)")
            return
        command = [sys.executable, "-m", "fedtraffic.federated", name, json.dumps(overrides)] + ([group] if group else [])
        try:
            out = subprocess.run(command, cwd=cfg.ROOT, capture_output=True, text=True, timeout=timeout)
            line = next((l for l in out.stdout.splitlines() if "test MAE" in l), None)
            if out.returncode == 0 and line:
                print(line, flush=True)
                return
            print(f"{name}: attempt {attempt} failed (exit {out.returncode}): {out.stderr.strip().splitlines()[-1:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"{name}: attempt {attempt} timed out after {timeout} s", flush=True)
        subprocess.run([str(cfg.ROOT / ".venv" / "Scripts" / "ray"), "stop", "--force"], capture_output=True)
        remove_unfinished(target, name)
    print(f"{name}: giving up after {attempts} attempts", flush=True)


def remove_unfinished(group, name):
    """Delete the folder of a run that never logged results, so a retry starts clean."""
    for path in RUNS.glob(f"*_{name}"):
        meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if meta.get("group") == group and "results" not in meta:
            shutil.rmtree(path)


def done(group, name):
    """Skip runs that already finished, so an interrupted sweep can be resumed."""
    return any(r["name"] == name for r in load_runs(group))


def centralised(group, name, overrides, nodes_from=None):
    if done(group, name):
        print(f"skip {name} (already in runs/)")
        return
    config = cfg.load(overrides=overrides)
    data = ds.load(config)
    run = Run(name, config, group)
    start = time.time()
    graph = nodes = None
    if nodes_from == "cut":
        groups = partition(data["locations"], config["federated"]["clients"], config["federated"]["partition"], config["federated"]["partition_seed"])
        graph = train.block_graph(data, groups, config["model"]["diffusion_steps"])
    model, val, test_acc = train.fit(config, data, run, nodes, graph, accumulators=True)
    val, test = val.result(), test_acc.result()
    extra = {}
    if hasattr(model, "parameters"):
        extra["params"] = sum(p.numel() for p in model.parameters())
        torch.save(model.state_dict(), run.dir / "checkpoint.pt")
    elif hasattr(model, "settings"):
        extra["settings"] = model.settings()
    run.finish(val=val, test=test, test_daily=test_acc.daily(), train_s=round(time.time() - start, 1), **extra)
    print(f"{name:<32} test MAE {test['mae']:.3f}  @15/30/60 {test['mae@15min']:.2f}/{test['mae@30min']:.2f}/{test['mae@60min']:.2f}")


def benchmark():
    for model in ("last_value", "historical_average"):
        centralised("benchmark", model, {"model.name": model})
    for model in ("gconv_gru", "gconv_lstm", "gru", "lstm", "gconv_mlp"):
        for seed in SEEDS:
            centralised("benchmark", f"{model}_s{seed}", {"model.name": model, "seed": seed})


def local_only(name, overrides, group="federated"):
    """Each district trains alone on its own sensors; errors pooled over all sensors."""
    if done(group, name):
        print(f"skip {name}")
        return
    from .metrics import Accumulator
    config = cfg.load(overrides=overrides)
    data = ds.load(config)
    run = Run(name, config, group)
    start = time.time()
    f = config["federated"]
    groups = partition(data["locations"], f["clients"], f["partition"], f["partition_seed"])
    norm = tuple(float(v) for v in data["norm"])
    val_acc, test_acc = Accumulator(), Accumulator()
    for k, nodes in enumerate(groups):
        model, _, _ = train.fit(config, data, None, nodes)
        graph = train.subgraph(data, nodes, config["model"]["diffusion_steps"])
        v = train.evaluate(model, graph, train.windows_for(data, config, "val", nodes), norm)
        t = train.evaluate(model, graph, train.windows_for(data, config, "test", nodes), norm)
        run.log(client=k, sensors=len(nodes), val_mae=v.result()["mae"], test_mae=t.result()["mae"])
        val_acc.merge(v)
        test_acc.merge(t)
    test = test_acc.result()
    run.finish(val=val_acc.result(), test=test, test_daily=test_acc.daily(), train_s=round(time.time() - start, 1))
    print(f"{name:<32} test MAE {test['mae']:.3f}")


def federated():
    for seed in (0, 1):
        s = {"seed": seed}
        centralised("federated", f"centralised_full_s{seed}", s)
        centralised("federated", f"centralised_cut_s{seed}", s, nodes_from="cut")
        local_only(f"local_only_s{seed}", s)
        federated_job(f"fedavg_s{seed}", s)
        federated_job(f"fedprox_s{seed}", {**s, "federated.strategy": "fedprox"})


def tradeoffs():
    for epochs in (2, 4):
        federated_job(f"fedavg_E{epochs}", {"federated.local_epochs": epochs})
    federated_job("fedavg_half_clients", {"federated.fraction": 0.5, "federated.rounds": 24})
    federated_job("fedavg_random_partition", {"federated.partition": "random"})
    for bits in (16, 8):
        federated_job(f"fedavg_{bits}bit", {"federated.update_bits": bits})
    for mu in (0.1,):
        federated_job(f"fedprox_mu{mu}", {"federated.strategy": "fedprox", "federated.proximal_mu": mu})


def privacy():
    """RQ5 (ethics): what does a formal privacy guarantee cost with only four districts?"""
    federated_job("dp_uniform_clip_only", {"federated.dp_uniform": True}, group="privacy")
    for noise in (0.01, 0.1, 1.0, 2.0):
        federated_job(f"dp_noise{noise}", {"federated.dp_noise": noise}, group="privacy")


# Round 2. The v1 models ran out of epochs while still improving (no run early-stopped), and
# a simple weekly profile beat them at 60 minutes. v2 trains until validation stops improving
# and adds a seasonal-aware variant that learns the residual on top of the weekly profile.
LONG = {"model.name": "gconv_lstm", "train.epochs": 50, "train.patience": 5}
SEASONAL = {**LONG, "model.seasonal": True}


def benchmark_v2():
    for model in ("last_value", "historical_average", "seasonal_residual"):
        centralised("benchmark_v2", model, {"model.name": model})
    for seed in SEEDS:
        centralised("benchmark_v2", f"gconv_lstm_long_s{seed}", {**LONG, "seed": seed})
        centralised("benchmark_v2", f"gconv_lstm_seasonal_s{seed}", {**SEASONAL, "seed": seed})
        centralised("benchmark_v2", f"lstm_seasonal_s{seed}", {**SEASONAL, "model.name": "lstm", "seed": seed})


def federated_v2():
    """Seasonal graph LSTM: local-only vs FedAvg vs FedAvg + per-district fine-tuning.

    The centralised reference is benchmark_v2's gconv_lstm_seasonal (same model and budget).
    40 rounds of one local epoch is close to the centralised runs' epoch count.
    """
    fed = {**SEASONAL, "federated.rounds": 40, "federated.personalise": True}
    for seed in SEEDS:
        local_only(f"local_only_seasonal_s{seed}", {**SEASONAL, "seed": seed}, group="federated_v2")
        federated_job(f"fedavg_seasonal_s{seed}", {**fed, "seed": seed}, group="federated_v2", timeout=5400)


def smoke():
    quick = {"train.epochs": 1, "data.train_stride": 60, "data.eval_stride": 60, "federated.rounds": 2}
    centralised("smoke", "gconv_gru", {**quick, "model.name": "gconv_gru"})
    federated_job("fedavg", quick, group="smoke")


def main():
    parser = argparse.ArgumentParser()
    groups = {"benchmark": benchmark, "federated": federated, "tradeoffs": tradeoffs, "privacy": privacy,
              "benchmark_v2": benchmark_v2, "federated_v2": federated_v2, "smoke": smoke}
    parser.add_argument("group", choices=tuple(groups))
    args = parser.parse_args()
    keep_awake()
    groups[args.group]()


if __name__ == "__main__":
    main()
