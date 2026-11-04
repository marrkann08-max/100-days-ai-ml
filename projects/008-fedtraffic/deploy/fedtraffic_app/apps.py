"""The same ClientApp and ServerApp as the simulation, served by real Flower processes.

In simulation, Ray runs every client inside one machine's process pool. Here each
district is its own SuperNode process that connects to a SuperLink over the network,
exactly as it would from a separate computer. Only the transport changes; the
training code is fedtraffic.federated, unmodified.
"""
import time

import numpy as np
import torch
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp

from fedtraffic import config as cfg
from fedtraffic import data as ds
from fedtraffic import models, train
from fedtraffic.federated import make_client_app, make_server_app
from fedtraffic.metrics import Accumulator
from fedtraffic.partition import partition
from fedtraffic.tracking import Run


def settings(run_config):
    return cfg.load(overrides={
        "federated.rounds": int(run_config["rounds"]),
        "federated.local_epochs": int(run_config["local-epochs"]),
        "federated.strategy": str(run_config["strategy"]),
        "train.threads": 2,
    })


client = ClientApp()


@client.train()
def client_train(msg, context):
    return make_client_app(settings(context.run_config))(msg, context)


@client.evaluate()
def client_evaluate(msg, context):
    return make_client_app(settings(context.run_config))(msg, context)


server = ServerApp()


@server.main()
def server_main(grid, context):
    config = settings(context.run_config)
    f = config["federated"]
    run = Run(str(context.run_config["run-name"]), config, group="deployment")
    out = {}
    start = time.time()
    make_server_app(config, out)(grid, context)
    result = out["result"]

    params = models.parameters(models.build(config))
    total_mb = 0.0
    for rnd in sorted(result.evaluate_metrics_clientapp):
        ev = result.evaluate_metrics_clientapp[rnd]
        mb = 2 * params * 4 * f["clients"] / 1e6
        total_mb += mb
        run.log(round=rnd, val_mae=float(ev["val-mae"]), val_rmse=float(np.sqrt(ev["val-mse"])),
                round_mb=mb, total_mb=total_mb)

    # Same deployment-style test as the simulation: each district, its own subgraph, pooled.
    data = ds.load(config)
    model = models.build(config)
    model.load_state_dict(result.arrays.to_torch_state_dict())
    norm = tuple(float(v) for v in data["norm"])
    test = Accumulator()
    for nodes in partition(data["locations"], f["clients"], f["partition"], f["partition_seed"]):
        graph = train.subgraph(data, nodes, config["model"]["diffusion_steps"])
        test.merge(train.evaluate(model, graph, train.windows_for(data, config, "test", nodes), norm))
    torch.save(model.state_dict(), run.dir / "checkpoint.pt")
    run.finish(test=test.result(), params=params, total_mb=round(total_mb, 3), train_s=round(time.time() - start, 1),
               transport="Flower SuperLink + 4 SuperNode processes over HTTP (TLS off, localhost)")
