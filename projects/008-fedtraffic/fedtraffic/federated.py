"""Federated training with Flower: each client is a district that keeps its sensor data.

Per round: the server sends the global weights to the sampled districts; each trains
on its own sensors and road subgraph for `local_epochs` and returns new weights
(optionally compressed); Flower's FedAvg/FedProx aggregates them, weighted by how
many observations each district trained on. Every district then reports validation
error for the new global model. Raw speeds never leave a client; only weights do.
"""
import os
import time

import numpy as np
import torch

from . import config as cfg
from . import data as ds
from . import models, train
from .metrics import Accumulator
from .partition import cut_edges, partition
from .tracking import Run, load_runs

_CLIENT_CACHE = {}


def client_env(config, client):
    """Data, graph and windows for one district, cached for the life of the worker process."""
    key = (cfg.digest(config["data"]), cfg.digest(config["federated"]), config["seed"], client)
    if key not in _CLIENT_CACHE:
        data = ds.load(config)
        f = config["federated"]
        nodes = partition(data["locations"], f["clients"], f["partition"], config["seed"])[client]
        _CLIENT_CACHE[key] = {
            "nodes": nodes,
            "norm": tuple(float(v) for v in data["norm"]),
            "graph": train.subgraph(data, nodes, config["model"]["diffusion_steps"]),
            "train": train.windows_for(data, config, "train", nodes),
            "val": train.windows_for(data, config, "val", nodes),
        }
    return _CLIENT_CACHE[key]


def compress(new, old, bits):
    """Simulate a compressed upload: the update (new - old) is quantised, then restored.

    Returns the restored state and the number of bytes that would be sent.
    16 bits: half-precision floats. 8 bits: per-tensor min/max plus one byte per value.
    """
    if bits == 32:
        return new, sum(v.numel() * 4 for v in new.values())
    out, size = {}, 0
    for name, value in new.items():
        delta = value - old[name]
        if bits == 16:
            restored = delta.half().float()
            size += delta.numel() * 2
        else:
            lo, hi = delta.min(), delta.max()
            scale = (hi - lo).clamp(min=1e-12) / 255
            restored = torch.round((delta - lo) / scale) * scale + lo
            size += delta.numel() + 8
        out[name] = old[name] + restored
    return out, size


def make_client_app(config):
    from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp

    app = ClientApp()

    @app.train()
    def train_round(msg, context):
        torch.set_num_threads(int(config["train"]["threads"]))
        client = int(context.node_config["partition-id"])
        env = client_env(config, client)
        model = models.build(config)
        start = msg.content["arrays"].to_torch_state_dict()
        model.load_state_dict(start)
        mu = float(msg.content["config"].get("proximal-mu", 0.0))
        reference = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
        rng = np.random.default_rng([config["seed"], client, int(msg.content["config"]["server-round"])])
        tic = time.time()
        for _ in range(config["federated"]["local_epochs"]):
            loss = train.train_epoch(model, env["graph"], env["train"], env["norm"], config, optimizer, rng,
                                     prox=(reference, mu) if reference else None)
        state, upload = compress(model.state_dict(), start, config["federated"]["update_bits"])
        observations = int(len(env["train"]) * len(env["nodes"]))
        metrics = MetricRecord({"train-loss": loss, "num-examples": observations,
                                "upload-bytes": upload, "compute-s": time.time() - tic})
        return Message(content=RecordDict({"arrays": ArrayRecord(state), "metrics": metrics}), reply_to=msg)

    @app.evaluate()
    def evaluate_round(msg, context):
        torch.set_num_threads(int(config["train"]["threads"]))
        env = client_env(config, int(context.node_config["partition-id"]))
        model = models.build(config)
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        acc = train.evaluate(model, env["graph"], env["val"], env["norm"])
        r = acc.result()
        # Weighted by observation count, these averages pool exactly into the all-sensor MAE and MSE.
        metrics = MetricRecord({"val-mae": r["mae"], "val-mse": r["rmse"] ** 2, "num-examples": r["count"]})
        return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

    return app


def make_server_app(config, out):
    from flwr.app import ArrayRecord, ConfigRecord
    from flwr.serverapp import ServerApp
    from flwr.serverapp.strategy import FedAvg, FedProx

    app = ServerApp()
    f = config["federated"]

    @app.main()
    def main(grid, context):
        train.seed_everything(config["seed"])
        model = models.build(config)
        sampled = max(1, round(f["fraction"] * f["clients"]))
        common = dict(fraction_train=f["fraction"], fraction_evaluate=1.0, min_train_nodes=sampled,
                      min_evaluate_nodes=f["clients"], min_available_nodes=f["clients"])
        strategy = (FedProx(proximal_mu=f["proximal_mu"], **common) if f["strategy"] == "fedprox"
                    else FedAvg(**common))
        # A round that hasn't answered in 10 minutes has failed (e.g. after the laptop slept);
        # better to stop than to wait Flower's default hour per round.
        out["result"] = strategy.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()),
                                       num_rounds=f["rounds"], train_config=ConfigRecord({}), timeout=600)

    return app


def run_federated(name, overrides, group=None):
    group = group or ("federated" if name.split("_s")[0] in ("fedavg", "fedprox") else "tradeoffs")
    if any(r["name"] == name for r in load_runs(group)):
        print(f"skip {name} (already in runs/)")
        return
    from flwr.simulation import run_simulation

    config = cfg.load(overrides=overrides)
    f = config["federated"]
    data = ds.load(config)
    groups = partition(data["locations"], f["clients"], f["partition"], config["seed"])
    run = Run(name, config, group)
    # Ray workers are separate processes: they inherit this, so they can import fedtraffic.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(cfg.ROOT), os.environ.get("PYTHONPATH"))))
    config["train"]["threads"] = max(1, (os.cpu_count() or 8) // f["clients"])
    out = {}
    start = time.time()
    run_simulation(
        server_app=make_server_app(config, out),
        client_app=make_client_app(config),
        num_supernodes=f["clients"],
        backend_config={
            "client_resources": {"num_cpus": config["train"]["threads"], "num_gpus": 0.0},
            "init_args": {"num_cpus": config["train"]["threads"] * f["clients"], "include_dashboard": False},
        },
        verbose_logging=bool(os.environ.get("FEDTRAFFIC_VERBOSE")),
    )
    result = out.get("result")
    if result is None or len(result.evaluate_metrics_clientapp) < f["rounds"]:
        raise RuntimeError(f"{name}: federated training did not complete all {f['rounds']} rounds")

    # Per-round log: pooled validation error and the bytes moved.
    params = models.parameters(models.build(config))
    sampled = max(1, round(f["fraction"] * f["clients"]))
    total_mb = 0.0
    for rnd in sorted(result.evaluate_metrics_clientapp):
        ev = result.evaluate_metrics_clientapp[rnd]
        tr = result.train_metrics_clientapp.get(rnd, {})
        upload = float(tr.get("upload-bytes", params * 4)) * sampled
        download = params * 4 * sampled
        total_mb += (upload + download) / 1e6
        run.log(round=rnd, val_mae=float(ev["val-mae"]), val_rmse=float(np.sqrt(ev["val-mse"])),
                train_loss=float(tr.get("train-loss", float("nan"))), round_mb=(upload + download) / 1e6,
                total_mb=total_mb)

    # Deployment-style test: every district runs the final global model on its own subgraph.
    model = models.build(config)
    model.load_state_dict(result.arrays.to_torch_state_dict())
    torch.set_num_threads(os.cpu_count() or 8)
    norm = tuple(float(v) for v in data["norm"])
    val_acc, test_acc = Accumulator(), Accumulator()
    for nodes in groups:
        graph = train.subgraph(data, nodes, config["model"]["diffusion_steps"])
        val_acc.merge(train.evaluate(model, graph, train.windows_for(data, config, "val", nodes), norm))
        test_acc.merge(train.evaluate(model, graph, train.windows_for(data, config, "test", nodes), norm))
    torch.save(model.state_dict(), run.dir / "checkpoint.pt")
    test = test_acc.result()
    run.finish(val=val_acc.result(), test=test, params=params, total_mb=round(total_mb, 3),
               cut_edges=round(cut_edges(data["adjacency"], groups), 4), clients=[len(g) for g in groups],
               train_s=round(time.time() - start, 1))
    print(f"{name:<32} test MAE {test['mae']:.3f}  ({total_mb:.1f} MB sent, {time.time() - start:.0f} s)")


if __name__ == "__main__":
    # Used by experiments.py to run each federated job in its own process.
    import json
    import sys
    run_federated(sys.argv[1], json.loads(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
