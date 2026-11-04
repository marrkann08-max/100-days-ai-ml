# Research log

Dated working notes, kept the way they'd be brought to a weekly lab meeting: what was done, what went wrong, what was decided, and what's next. Commit hashes link each entry to the code it describes.

---

## 2026-11-04, morning: setup and data

**Done**
- Read the project brief and turned it into four research questions and a milestone plan (`PLAN.md`).
- Checked tooling. Flower 1.42's simulation engine needs Ray, and Ray only ships Windows wheels for Python ≤ 3.12. The machine has Python 3.14.
  - **Decision:** a project-local Python 3.12.16 managed by `uv`. The system Python is untouched.
- Data source: the original METR-LA download server didn't respond, so the data comes from two Hugging Face mirrors.
  - Recorded SHA-256 checksums for all three files. The pipeline refuses to run if they change.
- Inspected the HDF5 file before loading it: plain numeric arrays, no pickled objects. It's read with PyTables directly so nothing is ever unpickled.
  - Decided not to use the mirror's `adj_mx.pkl` (pickle) and to rebuild the graph from `distances.csv` instead.

**Sanity checks against the literature**
- Rebuilt road graph: **1,722 non-zero entries**, identical to the adjacency released with DCRNN (Li et al., 2018).
- Historical-average baseline: **4.20 mph MAE** on the test period vs **4.16** reported by Li et al.

**Findings from exploration** (`results/eda.json`)
- **Missing data:**
  - 8.11% of readings are missing, recorded as 0 mph.
  - Only 0.34% sit in gaps of 15 minutes or less; the rest are long outages. **Decision:** interpolate only short gaps, never score filled values.
  - Missingness rises from 7.1% (train) to 12.1% (test). Flagged as a distribution shift in the data card.
- **Spatial correlation:** sensors within 2 km by road correlate at 0.74, vs 0.19 beyond 10 km. This is the evidence that a graph should help.

**Commit:** `13b2ef9`

---

## 2026-11-04, morning: models and speed

- Implemented diffusion graph convolution, GConvGRU and GConvLSTM cells, plain GRU/LSTM and a graph-only model from scratch.
- First timing: the graph+GRU model at ~2.6 min per epoch was too slow for the planned sweeps on a laptop CPU.
  - Benchmarked sparse propagation vs a precomputed dense hop stack: **10.3 ms → 2.0 ms** per call (5×).
  - Switched to the dense stack. Epochs dropped to ~0.9 min.
- **Decision (compute budget):** train on every 6th window, at most 12 epochs, early stopping with patience 3. Identical for every method, so comparisons stay fair; documented as a limitation.
- Flower: the first simulation hung because Ray's `runtime_env` option stalls on Windows. Passing `PYTHONPATH` through the environment fixed it.

**Commit:** `4b8c671`. All benchmark runs ran on this commit with a clean working tree.

---

## 2026-11-04, midday: incident: hung run

**What happened:** one benchmark run took 2 h instead of ~6 min, and the first FedProx run sat for 3 h without completing a round. Its Ray workers were idle.

**Diagnosis:**
- FedProx alone at small scale worked (23 s).
- Two Flower simulations back-to-back in one process also worked.
- Neither is a code bug. The laptop suspended mid-run: the long benchmark run overlaps the same window, and a suspend breaks Ray's local connections, after which Flower waits forever.

**Fix:**
- The sweep process now asks Windows to stay awake while it runs (`SetThreadExecutionState`, reverts on exit; no system settings changed).
- Each Flower simulation runs in its own child process with a 30-minute limit and one retry.
- Flower's per-round timeout dropped from 1 h to 10 min.
- Unfinished run folders are removed before a retry, so logs only ever contain completed runs.

**Effect on results:** none. The slow benchmark run finished normally (its score matches the other seeds), and the hung FedProx run never logged results and was rerun. Tables report the *median* training time so the sleep doesn't distort it.

**Commit:** `1f17f22`. Federated runs from FedProx seed 0 onward ran on this commit.

---

## 2026-11-04, afternoon: early results and questions for discussion

- **Benchmark:**
  - Graph + LSTM is best at every horizon. Graph + GRU is close behind.
  - Adding the graph to a recurrent encoder helps most at 60 minutes (about −6% MAE).
  - The graph alone, without recurrence, falls in between.
- **Surprise:** each district training *alone* (3.50 MAE) beat the centralised model trained on all data (3.56), and FedAvg after 12 rounds was worse (3.66).
- **Hypothesis:** the districts' traffic differs enough (different freeways and congestion patterns) that one global model is a compromise.
- **Questions to raise at a meeting:**
  - Is 12 rounds of one local epoch simply too few? (The trade-off sweep tests 2 and 4 local epochs.)
  - Should the next step be personalised federated learning (a shared model plus per-district fine-tuning), given local-only already wins?
  - Is pooled MAE the right headline metric when districts differ in size (40 to 61 sensors)? Per-district errors are in the local-only log.

**Next:**
- Finish the trade-off sweep.
- Generate tables and figures from the logs.
- Write the report.
- Add a differential-privacy run and a real (non-simulated) Flower deployment demo.

---

## 2026-11-04, afternoon: incident: partition depended on the seed

**What happened:** checking per-district results turned up a broken seed-1 partition. The district partition (k-means on sensor locations) was seeded with the run's model seed, so:
- seed 0 gave districts of 61 / 54 / 52 / 40 sensors;
- seed 1 gave 89 / 66 / 51 / **1**, a "district" of a single sensor.

The seed-1 federated runs were therefore a different experiment, not a repeat with a different initialisation.

**Fix:**
- New setting `federated.partition_seed` (fixed at 0), used everywhere districts are computed.
- The four affected runs (centralised-cut, local-only, FedAvg and FedProx, seed 1) were **moved to `runs_discarded/`, not deleted**, and rerun on the corrected code.
- Every trade-off and privacy run used seed 0, so the correct partition, and is unaffected.

**Lesson:** a seed should control exactly one source of randomness. Data splits and partitions get their own fixed seeds.

---

## 2026-11-04, afternoon: differential privacy (RQ5)

- Added central DP with server-side fixed clipping (Flower's `DifferentialPrivacyServerSideFixedClipping`).
  - Clipping norm C = 3: measured typical late-training update norms were 2.7–3.3, and about 7 at initialisation.
  - Districts are averaged uniformly, which the DP guarantee assumes. A no-noise uniform-averaging run is the control.
- ε computed with Rényi-DP composition of the Gaussian mechanism (Mironov, 2017) at δ = 1e-5, district-level, no subsampling amplification. Unit-tested against a hand-computed case.
- **Result:**
  - σ = 0.01 costs little accuracy (3.71 vs 3.66 MAE) but gives ε ≈ 61,751, which is no meaningful protection.
  - σ = 0.1 (ε ≈ 766) is already worse than the historical-average baseline.
  - σ = 1 (ε ≈ 23) and σ = 2 (ε ≈ 10) destroy the model (MAE > 100 mph).
- **Why:** with only 4 districts, the noise added to the average (σC/4 per weight, across 17,772 weights) is far larger than a typical update. Meaningful DP needs many more participants.
- **For discussion:** report this as a negative result, and note that local DP would be even worse here.

---

## 2026-11-04, afternoon: real Flower deployment

- Added `deploy/`: a proper Flower App (`pyproject.toml`) whose ServerApp and ClientApp wrap the unmodified simulation code. `deploy/run_local.py` starts a SuperLink and four SuperNode processes, then submits the app with `flwr run`.
- Problems on the way, all fixed:
  1. Flower's processes start helpers by name, so the venv's `Scripts` folder must be on `PATH`.
  2. A leftover SQLite state file from a failed start broke the next start. Switched to in-memory state.
  3. In Flower 1.42 the SuperLink serves its Fleet, Control and Runtime APIs over HTTP on one port (8000), not the older gRPC ports 9092 and 9093.
  4. Windows' default console encoding crashed `flwr run`'s log stream. Forced UTF-8.
  5. By default the SuperLink installed the app's dependencies into a fresh 500 MB environment per run. Disabled, so it reuses the project environment.
- A 1-round trial completed with all 4 districts reporting and no failures. The full 3-round run is logged under `runs/` (group `deployment`).
