"""Centralised training and evaluation, also reused by each federated client."""
import copy
import time

import numpy as np
import torch

from . import data as ds
from . import models
from .metrics import Accumulator, masked_mae_loss


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensors(x, y, m, norm):
    mean, std = norm
    return (torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(((y - mean) / std).astype(np.float32)),
            torch.from_numpy(m))


def train_epoch(model, graph, windows, norm, config, optimizer, rng, prox=None):
    """One pass over the windows. prox = (global_params, mu) adds FedProx's proximal term."""
    model.train()
    t = config["train"]
    losses = []
    for x, y, m, _ in windows.batches(t["batch_size"], rng):
        x, y, m = to_tensors(x, y, m, norm)
        loss = masked_mae_loss(model(x, graph), y, m)
        if prox is not None:
            reference, mu = prox
            loss = loss + mu / 2 * sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), reference))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, graph, windows, norm, batch_size=64):
    """Accumulated errors in mph against truly observed readings."""
    acc = Accumulator(windows.horizon)
    mean, std = norm
    if isinstance(model, torch.nn.Module):
        model.eval()
    for x, y, m, future in windows.batches(batch_size):
        acc.add(predict(model, graph, windows, x, future, mean, std), y, m)
    return acc


def predict(model, graph, windows, x, future, mean, std):
    if model == "last_value":
        last = x[:, -1, :, 0] * std + mean
        return np.repeat(last[:, None], windows.horizon, axis=1)
    if model == "historical_average":
        return windows.historical[windows.slots[future]]
    return model(torch.from_numpy(np.ascontiguousarray(x)), graph).numpy() * std + mean


def windows_for(data, config, part, nodes=None):
    d = config["data"]
    stride = d["train_stride"] if part == "train" else d["eval_stride"]
    return ds.Windows(data, part, d["input_steps"], d["horizon"], stride, nodes)


def subgraph(data, nodes, steps=2):
    nodes = np.asarray(nodes)
    return models.Graph(data["adjacency"][np.ix_(nodes, nodes)], steps)


def block_graph(data, groups, steps=2):
    """Full sensor set, but with every edge between different groups removed."""
    adj = np.zeros_like(data["adjacency"])
    for g in groups:
        adj[np.ix_(g, g)] = data["adjacency"][np.ix_(g, g)]
    return models.Graph(adj, steps)


def fit(config, data, run=None, nodes=None, graph=None):
    """Train with early stopping on validation MAE; returns (model, best validation result, test result)."""
    torch.set_num_threads(config["train"]["threads"])
    seed_everything(config["seed"])
    norm = tuple(float(v) for v in data["norm"])
    nodes = np.arange(data["inputs"].shape[1]) if nodes is None else np.asarray(nodes)
    graph = graph or subgraph(data, nodes, config["model"]["diffusion_steps"])
    splits = {p: windows_for(data, config, p, nodes) for p in ("train", "val", "test")}
    name = config["model"]["name"]

    if name in models.BASELINES:
        val, test = (evaluate(name, graph, splits[p], norm).result() for p in ("val", "test"))
        return name, val, test

    model = models.build(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    rng = np.random.default_rng(config["seed"])
    best, best_state, waited = float("inf"), None, 0
    for epoch in range(1, config["train"]["epochs"] + 1):
        start = time.time()
        loss = train_epoch(model, graph, splits["train"], norm, config, optimizer, rng)
        val = evaluate(model, graph, splits["val"], norm).result()
        if run:
            run.log(epoch=epoch, train_loss=loss, val_mae=val["mae"], val_rmse=val["rmse"],
                    epoch_s=round(time.time() - start, 1))
        if val["mae"] < best - 1e-4:
            best, best_state, waited = val["mae"], copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= config["train"]["patience"]:
                break
    model.load_state_dict(best_state)
    val = evaluate(model, graph, splits["val"], norm).result()
    test = evaluate(model, graph, splits["test"], norm).result()
    return model, val, test
