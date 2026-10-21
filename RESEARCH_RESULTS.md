# Initial walk-forward research results

These are development results, not a trading recommendation. No live or paper orders were placed. The 2022+ reserved period was not evaluated. Historical files had already been inspected, so it is not genuinely unseen data.

| Asset | Evaluation signals | Model CAGR | Model max drawdown | Buy/hold CAGR |
|---|---|---:|---:|---:|
| MSFT | 1992-04-06 to 2021-12-29 | 0.25% | -20.24% | 19.72% |
| AAPL | 1987-01-12 to 2021-12-29 | 1.33% | -35.39% | 22.26% |

Both model strategies triggered their drawdown stop and then remained in cash for the rest of their evaluation. Their full-period CAGR therefore describes a largely inactive strategy, not a continuously active forecasting edge. Buy/hold has no drawdown stop and much greater exposure; compare risk and exposure alongside return. The observed stop overshoot demonstrates that a threshold is not a guaranteed loss cap.

Costs assume 5 basis points per traded side. Returns use retrospective adjusted-open proxies, not executable raw-share/dividend accounting. No measured broker fills, complete exchange-calendar validation, cash yield, taxes or market impact. Existing asset selection and repeated historical research limit generalization.

Reproduce with the commands in README.md. Detailed JSON, per-fold choices, signal/position ledgers, cost sensitivity, yearly returns and equity plots are under ignored reports/msft and reports/aapl. Reports include input SHA-256 hashes. All 13 unit tests passed; identical MSFT runs reproduced the same JSON report.

Next gate: point-in-time current data and raw-share corporate-action accounting, then a frozen prospective evaluation. These results do not support moving this signal to broker execution.
