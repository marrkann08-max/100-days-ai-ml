"""Named experiment groups. Each run is logged to runs/ and can be repeated exactly.

    python -m fedtraffic.experiments benchmark      # RQ1-2: recurrent encoders, graph or not (centralised)
    python -m fedtraffic.experiments federated      # RQ3: centralised vs local-only vs federated
    python -m fedtraffic.experiments tradeoffs      # RQ4: rounds, local epochs, participation, FedProx, compression
    python -m fedtraffic.experiments smoke          # tiny end-to-end check
"""
import argparse
import time

from . import config as cfg
from . import data as ds
from . import train
from .partition import partition
from .tracking import Run, load_runs

SEEDS = (0, 1, 2)


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
        groups = partition(data["locations"], config["federated"]["clients"], config["federated"]["partition"], config["seed"])
        graph = train.block_graph(data, groups, config["model"]["diffusion_steps"])
    model, val, test = train.fit(config, data, run, nodes, graph)
    params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
    run.finish(val=val, test=test, params=params, train_s=round(time.time() - start, 1))
    print(f"{name:<32} test MAE {test['mae']:.3f}  @15/30/60 {test['mae@15min']:.2f}/{test['mae@30min']:.2f}/{test['mae@60min']:.2f}")


def benchmark():
    for model in ("last_value", "historical_average"):
        centralised("benchmark", model, {"model.name": model})
    for model in ("gconv_gru", "gconv_lstm", "gru", "lstm", "gconv_mlp"):
        for seed in SEEDS:
            centralised("benchmark", f"{model}_s{seed}", {"model.name": model, "seed": seed})


def local_only(name, overrides):
    """Each district trains alone on its own sensors; errors pooled over all sensors."""
    if done("federated", name):
        print(f"skip {name}")
        return
    from .metrics import Accumulator
    config = cfg.load(overrides=overrides)
    data = ds.load(config)
    run = Run(name, config, "federated")
    f = config["federated"]
    groups = partition(data["locations"], f["clients"], f["partition"], config["seed"])
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
    run.finish(val=val_acc.result(), test=test)
    print(f"{name:<32} test MAE {test['mae']:.3f}")


def federated():
    from .federated import run_federated
    for seed in (0, 1):
        s = {"seed": seed}
        centralised("federated", f"centralised_full_s{seed}", s)
        centralised("federated", f"centralised_cut_s{seed}", s, nodes_from="cut")
        local_only(f"local_only_s{seed}", s)
        run_federated(f"fedavg_s{seed}", s)
        run_federated(f"fedprox_s{seed}", {**s, "federated.strategy": "fedprox"})


def tradeoffs():
    from .federated import run_federated
    for epochs in (2, 4):
        run_federated(f"fedavg_E{epochs}", {"federated.local_epochs": epochs})
    run_federated("fedavg_half_clients", {"federated.fraction": 0.5, "federated.rounds": 24})
    run_federated("fedavg_random_partition", {"federated.partition": "random"})
    for bits in (16, 8):
        run_federated(f"fedavg_{bits}bit", {"federated.update_bits": bits})
    for mu in (0.1,):
        run_federated(f"fedprox_mu{mu}", {"federated.strategy": "fedprox", "federated.proximal_mu": mu})


def smoke():
    quick = {"train.epochs": 1, "data.train_stride": 60, "data.eval_stride": 60, "federated.rounds": 2}
    centralised("smoke", "gconv_gru", {**quick, "model.name": "gconv_gru"})
    from .federated import run_federated
    run_federated("fedavg", quick, group="smoke")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("benchmark", "federated", "tradeoffs", "smoke"))
    args = parser.parse_args()
    {"benchmark": benchmark, "federated": federated, "tradeoffs": tradeoffs, "smoke": smoke}[args.group]()


if __name__ == "__main__":
    main()
