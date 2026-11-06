"""Graph and recurrent building blocks, written out rather than imported.

Every learned model maps inputs (B, steps, N, F) to forecasts (B, horizon, N)
in normalised units. Weights never depend on N, which is what lets one set of
weights be shared across federated clients that hold different sensors.
"""
import numpy as np
import torch
from torch import nn


class Graph:
    """Random-walk diffusion in both directions (Li et al., 2018), precomputed.

    hops is one dense (2K, N, N) stack: [P_f, P_f^2 .. P_f^K, P_b, P_b^2 .. P_b^K]
    with P_f = D_out^-1 W (downstream) and P_b = D_in^-1 W^T (upstream). At METR-LA's
    size, one dense multiply by the whole stack is about 5x faster on a CPU than
    repeated sparse multiplies.
    """

    def __init__(self, adjacency, steps=2):
        w = np.asarray(adjacency, np.float32)
        self.n, self.steps = len(w), steps
        hops = []
        for p in (self.random_walk(w), self.random_walk(w.T)):
            power = np.eye(len(w), dtype=np.float32)
            for _ in range(steps):
                power = power @ p
                hops.append(power)
        self.hops = torch.from_numpy(np.stack(hops)).reshape(-1, self.n)   # (2K*N, N)

    @staticmethod
    def random_walk(w):
        degree = w.sum(axis=1, keepdims=True)
        return np.divide(w, degree, out=np.zeros_like(w), where=degree > 0)


def propagate(graph, x):
    """All diffusion hops of x (B, N, F) at once: returns (B, N, 2K*F)."""
    b, n, f = x.shape
    flat = x.permute(1, 0, 2).reshape(n, b * f)
    out = (graph.hops @ flat).reshape(-1, n, b, f)
    return out.permute(2, 1, 0, 3).reshape(b, n, -1)


class DiffusionConv(nn.Module):
    """sum over directions d and hops k of P_d^k X W_{d,k}: each sensor mixes in its
    upstream and downstream neighbours up to `steps` hops away."""

    def __init__(self, inputs, outputs, steps):
        super().__init__()
        self.steps = steps
        self.linear = nn.Linear(inputs * (1 + 2 * steps), outputs)

    def forward(self, x, graph):
        assert graph.steps == self.steps, "graph was built for a different number of diffusion steps"
        return self.linear(torch.cat((x, propagate(graph, x)), dim=-1))


class GConvGRUCell(nn.Module):
    """GRU whose gates are graph convolutions instead of dense layers."""

    def __init__(self, inputs, hidden, steps):
        super().__init__()
        self.hidden = hidden
        self.gates = DiffusionConv(inputs + hidden, 2 * hidden, steps)
        self.candidate = DiffusionConv(inputs + hidden, hidden, steps)

    def forward(self, x, h, graph):
        z, r = torch.sigmoid(self.gates(torch.cat((x, h), -1), graph)).chunk(2, -1)
        c = torch.tanh(self.candidate(torch.cat((x, r * h), -1), graph))
        return (1 - z) * h + z * c


class GConvLSTMCell(nn.Module):
    """LSTM whose four gates are one graph convolution."""

    def __init__(self, inputs, hidden, steps):
        super().__init__()
        self.hidden = hidden
        self.gates = DiffusionConv(inputs + hidden, 4 * hidden, steps)

    def forward(self, x, state, graph):
        h, c = state
        i, f, o, g = self.gates(torch.cat((x, h), -1), graph).chunk(4, -1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class GraphRecurrent(nn.Module):
    """Encoder: graph-convolutional GRU or LSTM over the input hour. Head: one linear
    layer from each sensor's final state to all 12 horizons at once."""

    def __init__(self, features, hidden, horizon, steps, cell="gru"):
        super().__init__()
        self.kind = cell
        self.cell = (GConvGRUCell if cell == "gru" else GConvLSTMCell)(features, hidden, steps)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x, graph):
        b, t, n, _ = x.shape
        h = x.new_zeros(b, n, self.cell.hidden)
        state = (h, h.clone())
        for k in range(t):
            if self.kind == "gru":
                h = self.cell(x[:, k], h, graph)
            else:
                state = self.cell(x[:, k], state, graph)
                h = state[0]
        return self.head(h).transpose(1, 2)


class Recurrent(nn.Module):
    """The same encoder without the graph: one GRU/LSTM shared by every sensor independently."""

    def __init__(self, features, hidden, horizon, cell="gru"):
        super().__init__()
        self.rnn = (nn.GRU if cell == "gru" else nn.LSTM)(features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x, graph=None):
        b, t, n, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        out, _ = self.rnn(seq)
        return self.head(out[:, -1]).reshape(b, n, -1).transpose(1, 2)


class GraphMLP(nn.Module):
    """The graph without recurrence: the whole input hour flattened, then two diffusion convolutions."""

    def __init__(self, features, hidden, horizon, steps, input_steps):
        super().__init__()
        self.first = DiffusionConv(features * input_steps, hidden, steps)
        self.second = DiffusionConv(hidden, hidden, steps)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x, graph):
        b, t, n, f = x.shape
        h = x.permute(0, 2, 1, 3).reshape(b, n, t * f)
        h = torch.relu(self.first(h, graph))
        h = torch.relu(self.second(h, graph))
        return self.head(h).transpose(1, 2)


LEARNED = ("gru", "lstm", "gconv_mlp", "gconv_gru", "gconv_lstm")
BASELINES = ("last_value", "historical_average", "seasonal_residual")


def build(config, features=None):
    m, d = config["model"], config["data"]
    features = features or (5 if m.get("seasonal") else 4)
    name, hidden, horizon, steps = m["name"], m["hidden"], d["horizon"], m["diffusion_steps"]
    if name in ("gru", "lstm"):
        return Recurrent(features, hidden, horizon, name)
    if name == "gconv_mlp":
        return GraphMLP(features, hidden, horizon, steps, d["input_steps"])
    if name in ("gconv_gru", "gconv_lstm"):
        return GraphRecurrent(features, hidden, horizon, steps, name.split("_")[1])
    raise ValueError(f"{name} is not a learned model")


def parameters(model):
    return sum(p.numel() for p in model.parameters())
