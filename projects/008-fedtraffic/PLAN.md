# Project plan

## Problem

Forecast freeway speeds 15, 30 and 60 minutes ahead at every sensor of a road network, when the data is owned by several districts that can't pool raw readings. How much accuracy does federated training give up compared with centralising the data, and what does it cost in communication?

## Research questions

| # | Question | Experiment | Answered by |
|---|---|---|---|
| RQ1 | Which recurrent encoder works best: GRU or LSTM? | `benchmark`: GRU vs LSTM, with and without the graph, 3 seeds each | Table 1, Figure: horizons |
| RQ2 | Does the road graph help, and is recurrence still needed once you have it? | `benchmark`: graph+GRU, graph+LSTM, graph without recurrence, recurrence without graph, two non-learned baselines | Table 1 |
| RQ3 | How close does federated learning get to centralised training? Is it better than districts training alone? | `federated`: centralised (full graph), centralised (graph cut at district borders), local-only, FedAvg, FedProx; 2 seeds | Table 2 |
| RQ4 | What are the convergence and communication trade-offs? | `tradeoffs`: local epochs (1/2/4), client participation (all vs half), partition (spatial vs random), update compression (32/16/8-bit), FedProx strength | Table 3, Figures: convergence, communication |

The "centralised, graph cut" setting separates two costs of federation: losing the road links between districts, and the federated optimisation itself.

## Milestones

| | Milestone | Output | Status |
|---|---|---|---|
| M1 | Data acquisition with checksums; data card | `data.py`, `DATA_CARD.md` | done |
| M2 | Cleaning and exploratory analysis | `eda.py`, `results/eda.json`, 4 figures | done |
| M3 | Reproducible pipeline: config, chronological split, train-only statistics, cache | `config.py`, `data.py`, tests | done |
| M4 | Graph and recurrent components from scratch; baselines; centralised trainer | `models.py`, `train.py`, tests | done |
| M5 | Flower client and server apps, partitions, communication accounting | `federated.py`, `partition.py` | done |
| M6 | Run logging tied to git commits | `tracking.py`, `runs/` | done |
| M7 | Benchmark → federated → trade-off sweeps | `experiments.py`, `runs/` | done |
| M8 | Tables and figures generated from logs; manuscript | `report.py`, `paper/report.md` | done |

## Design decisions

- **Chronological split** (70/10/20) and normalisation from the training period only. Random splits leak the future into training.
- **Missing readings never count as ground truth.** Gaps of 15 minutes or less are interpolated for model inputs only. Longer gaps get the training-period historical average plus an "observed" flag.
- **Weights independent of sensor count.** One model can then serve districts of different sizes, which FedAvg requires.
- **Deployment-style federated evaluation.** Each district runs the final global model on its own road subgraph, and errors are pooled over all 207 sensors. Centralised results use the full graph.

## Compute budget

One laptop CPU (AMD Ryzen 7 5825U, 8 cores, no GPU). To keep the full study to a few hours:
- Models use 32 hidden units and 2 diffusion hops.
- Training uses every 6th window (30 minutes apart) for at most 12 epochs, with early stopping (patience 3).
- Validation and test use every 3rd window.

These choices are the same for every method, so comparisons are fair. Absolute errors are higher than published full-scale results (see Limitations in the report).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Flower's simulator needs Ray, which has no Windows build for Python ≥ 3.13 | Project-local Python 3.12 environment via `uv` |
| Noisy single runs | 3 seeds for the benchmark, 2 for the main federated comparison; mean ± sd reported |
| Interrupted sweeps | `experiments.py` skips runs already logged, so a sweep resumes where it stopped |
| Results drifting from the code | Each run logs its git commit and whether the working tree had uncommitted changes |
