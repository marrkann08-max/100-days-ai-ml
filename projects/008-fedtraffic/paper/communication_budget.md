# Traffic forecasting under a communication budget

The original study measured total communication and the cost of reaching one target. This follow-up asks which configuration has the lowest validation error when the districts must stop at a fixed communication budget.

## Method

I compared seven retained FedAvg runs from the Day 008 project, covering two baseline seeds, two and four local epochs, half participation, and 16-bit and 8-bit uploads. For each run and budget, I used the last completed round that fit under the budget, with no interpolation or retrospective selection of a better earlier round. A run that finished below budget retains its final result. Communication is the cumulative MB recorded by the experiment. MAE is in mph, and lower is better.

| Budget (MB) | Lowest observed validation MAE | MAE (mph) | Round | MB used |
|---:|---|---:|---:|---:|
| 0.5 | Two of four districts per round | 3.506 | 1 | 0.284 |
| 1 | Two of four districts per round | 3.368 | 3 | 0.853 |
| 2 | Four local epochs per round | 3.334 | 3 | 1.706 |
| 4 | Two of four districts per round | 3.258 | 14 | 3.981 |
| 6 | Four local epochs per round | 3.272 | 10 | 5.687 |
| 7 | Two of four districts per round | 3.223 | 24 | 6.824 |

At 1 MB, half participation completed three rounds and reached 3.368 mph. The 8-bit full-participation run completed two rounds and reached 3.428 mph. At 2 MB, four local epochs reached 3.334 mph versus 3.360 mph for 8-bit updates. The lowest observed error can rise at a higher budget because this analysis uses the latest eligible checkpoint and validation error can fluctuate.

## Interpretation

Half participation led at four of six budgets, while extra local computation led at the other two. The comparison is descriptive. Each modified configuration had one logged seed, the budget grid was chosen after the experiments, and MAE here is from validation data. Equal communication budgets can involve different amounts of local computation. A prospective follow-up would fix the budgets in advance, rerun each configuration over matched seeds, and evaluate the selected checkpoints on held-out test data while also tracking compute time.

The companion CSV gives all 42 comparisons with source run IDs. The Python script regenerates it from the original logs.

```powershell
python analysis/day008_budget_analysis.py . results/tables/day008_budget_results.csv
```
