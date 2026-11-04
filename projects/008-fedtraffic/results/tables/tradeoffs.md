| Variant | Test MAE | Final val MAE | Rounds | Total comm. (MB) | Rounds to val MAE ≤ 3.366 | MB to reach it | Wall time (s) |
|---|---|---|---|---|---|---|---|
| FedAvg baseline (E = 1, all 4 districts, 32-bit, spatial) | 3.658 | 3.300 | 12 | 6.8 | 5 | 2.8 | 230 |
| 2 local epochs per round | 3.640 | 3.292 | 12 | 6.8 | 4 | 2.3 | 403 |
| 4 local epochs per round | 3.586 | 3.244 | 12 | 6.8 | 3 | 1.7 | 736 |
| 2 of 4 districts per round (24 rounds) | 3.569 | 3.223 | 24 | 6.8 | 4 | 1.1 | 320 |
| Random sensor partition (76% of road edges cut) | 3.604 | 3.256 | 12 | 6.8 | 3 | 1.7 | 233 |
| 16-bit compressed uploads | 3.667 | 3.309 | 12 | 5.1 | 5 | 2.1 | 238 |
| 8-bit compressed uploads | 3.652 | 3.297 | 12 | 4.3 | 5 | 1.8 | 230 |
| FedProx μ = 0.01 | 3.765 | 3.395 | 12 | 6.8 | not reached | – | 250 |
| FedProx μ = 0.1 | 3.934 | 3.541 | 12 | 6.8 | not reached | – | 301 |
