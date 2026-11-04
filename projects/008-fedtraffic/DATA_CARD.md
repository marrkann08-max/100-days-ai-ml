# Data card: METR-LA

## What it is

Average vehicle speed (mph) from **207 loop detectors on Los Angeles County freeways**:
- **Interval:** every 5 minutes.
- **Period:** **March 1 – June 27, 2012**, giving 34,272 time steps.
- **Extra files:** sensor coordinates and road-network distances between sensor pairs.

It's one of the two standard benchmarks for spatio-temporal traffic forecasting.

## Provenance

- Originally collected from Los Angeles County highway loop detectors (Jagadish et al., 2014), then preprocessed and released with DCRNN (Li et al., ICLR 2018, github.com/liyaguang/DCRNN).
- This project downloads byte-identical copies from two public Hugging Face mirrors, which state the licences below. Neither the original release nor this card makes a further licence claim.

| File | Source | Stated licence | SHA-256 |
|---|---|---|---|
| `metr-la.h5` | huggingface.co/datasets/jimmygao3218/METRLA | CC-BY-4.0 (mirror) | `64784b76…1277c05` |
| `distances.csv` | huggingface.co/datasets/witgaw/METR-LA | MIT (mirror) | `072e7aaa…0971156` |
| `sensor_locations.csv` | huggingface.co/datasets/witgaw/METR-LA | MIT (mirror) | `ab2e797c…0c9239` |

Full checksums are in `fedtraffic/data.py`. The pipeline refuses to run if any downloaded file differs.

## Stewardship

- **Raw data is not committed to git.** `data/` is ignored, and anyone can re-download and verify the files with the checksums.
- **Processed data is cached** under a key built from the preprocessing settings and the raw checksums. A JSON manifest beside each cache file records what produced it.
- **Safe loading.** The HDF5 file is a pandas store that *could* contain pickled objects. It's read with PyTables as plain numeric arrays, so nothing is ever unpickled. The mirror's `adj_mx.pkl` is deliberately not used: the graph is rebuilt from `distances.csv`.

## Known issues (measured in `results/eda.json`)

- **8.11% of readings are missing**, stored as 0 mph. Most gaps are long outages: only 0.34% of readings fall in gaps of 15 minutes or less, which is all this pipeline interpolates.
- **Missingness changes over time.** 7.1% of training readings are missing versus 12.1% in the test period, a mild distribution shift that affects every method equally.
- Speeds are capped at 70 mph in the source, so free-flow variation above that is invisible.
- The busiest sensor is missing 20% of its readings, and no sensor is entirely empty.

## Privacy and ethics

- **No personal data.** Readings are speeds averaged over all vehicles passing a fixed detector in a 5-minute window. Individual vehicles and people can't be identified.
- **Why federated learning anyway.** Real deployments often involve detectors owned by different agencies, or finer-grained data (probe vehicles, phones) that *is* sensitive. This project studies the accuracy and communication cost of not pooling raw data.
- **Federated learning is not a privacy guarantee.** Model updates can leak information about training data. This study uses no secure aggregation and no differential privacy. Its claims are about accuracy and communication, not formal privacy.
- **Intended use.** Research and education on forecasting methods. Forecasts from a 2012 dataset should not drive real traffic operations.
- **Evaluation honesty.** Interpolated and filled values are never scored as ground truth, and the test period is never used for model selection.

## References

- H. V. Jagadish et al. "Big data and its technical challenges." *Communications of the ACM*, 57(7), 2014.
- Y. Li, R. Yu, C. Shahabi, Y. Liu. "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting." *ICLR*, 2018.
