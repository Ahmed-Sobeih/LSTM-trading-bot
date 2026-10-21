# LSTM stock-price forecasting

Independent LSTM experiments for MSFT and AAPL, with chronological splits,
training-only standardization, early stopping, and a last-close persistence baseline.
This is a forecasting experiment, not a validated trading strategy.

## Run

Use Python 3.12 and install `pip install -r requirements.txt` in a virtual environment.
In the prepared cloud environment, use `source /workspace/lstm-environment/activate.sh`.
From this repository directory, run `jupyter lab` and open `LSTM_Trading_bot.ipynb`.
The CSVs `MSFT.csv`, `AAPL.csv`, and `MSFT_volume_rsi.csv` belong in this directory.
No Google Drive mount or credentials are needed.

The notebook runs the small MSFT model by default. Its `RUN` list selects independent
experiments, including RSI/volume features and a direct 360-input/30-output model.
Large windows cost more CPU time and memory. `EPOCHS` is an upper bound; early stopping
restores the best validation weights. Seeds help repeatability but do not guarantee
identical results on different hardware.

## Correctness and evaluation

Run `python -m unittest -v` for split-boundary, scaling, horizon, and recursion checks.
`forecasting.py` implements the shared data and model workflow. All target dates in a
window belong to one partition; input history may precede that partition. Multi-day
labels are future trading rows. No model or optimizer weights are reused between runs.
MAE/RMSE are reported in dollars against persistence on identical targets, including
per-horizon scores and a separate fixed-origin recursive comparison.

The full supplied history is used, so dates differ from the original notebook's 2022
MSFT cutoff. RSI and price-only files have different date coverage; align dates before
comparing feature choices. Raw Close forecasts do not account for dividends or provide
a total-return series. Overlapping windows are not independent observations.

Do not tune against test results. Further model selection should use walk-forward
validation and a new untouched final test period. Price error alone does not establish
profitability; trading evaluation also requires execution rules, costs, and drawdown.

## Walk-forward trading research

Run from this directory:

```bash
python research.py --data MSFT.csv --out reports/msft
python research.py --data AAPL.csv --out reports/aapl
python -m unittest -v
```

This separate lightweight pipeline predicts next-open to following-open returns
using regularized linear regression. At the close of session t, all features are
observable; the order executes at open t+1 and the position earns the return to
open t+2. There are no live orders. Features are lagged adjusted-price returns,
20-session volatility, the current day's high-low range, and relative volume.

Each annual evaluation fold selects ridge regularization and a long/cash threshold
on the preceding 252-session validation period, then refits on all eligible earlier
history. At least 1,260 prior training sessions are required. Labels whose exit
prices are not observable at the fit/validation time are purged. Standardization
is fitted only on the current training partition. Candidate thresholds include
an explicit cash choice. Fold selection maximizes validation ending wealth after
costs, not test performance. Overlapping training histories and a single validation
window still leave selection uncertainty; this is not a statistical edge claim.

Default assumed one-way costs are 1 bp commission + 2 bp half-spread + 2 bp
slippage. Costs apply to exposure changes and final liquidation. Positions cannot
be negative or exceed 100%; cash earns zero. The model locks into cash for the
rest of the run after a 20% marked drawdown, starting at the next execution open.
The stop cannot prevent gaps or guarantee a 20% maximum loss. Buy/hold and momentum
benchmarks do not use that stop, and this difference is recorded in the ledger.

Reports include exact configuration and input SHA-256, per-fold choices and label
cutoffs, predictions, dated position/equity ledgers, yearly returns, drawdowns,
zero-cash-rate Sharpe, exposure, turnover and cost sensitivity at 0x/1x/2x/4x costs.
The sensitivity analysis freezes selected signals; it does not reselect models.
Generated reports are ignored by Git. Run the same command again to reproduce them.
Zero-volume sessions are recorded in report data-quality metadata; samples requiring
entry or exit on those sessions are excluded rather than assuming a fill.

The default report excludes labels exiting on/after 2022-01-01. The reserved final
period is evaluated only with `--evaluate-holdout` (use a separate output directory).
Do not consume it repeatedly while tuning. These files have already been inspected
in earlier experiments: this is an enforced pipeline reservation, not genuinely
unseen research data. A future final assessment needs new data and frozen decisions.

### Interpretation and production gaps

Adjusted open is approximated as Open * Adj Close / Close. This is retrospective
adjusted-price accounting, including historical adjustment factors; it is not an
executable raw-share ledger with dividend payments, split events or point-in-time
vendor snapshots. Both strategy and benchmarks use that same return proxy. The
data ends in 2024, and exchange-calendar completeness, delistings, auction fills,
taxes, market impact and actual broker costs are not modeled. The two assets were
chosen beforehand and cannot establish cross-sectional robustness.

Before paper/live trading: obtain current point-in-time data and corporate actions,
validate exchange sessions and freshness, implement raw-share/cash accounting,
freeze walk-forward decisions and collect an untouched period. Then add broker
paper orders, idempotent client order IDs, persistent order/position reconciliation,
alerts and an operator-controlled kill switch. Live deployment additionally needs
explicit account authorization and capital/risk limits. No broker credentials or
real funds are used by this research pipeline.

## Raw data, corporate actions and forward observations

See [DATA_PIPELINE.md](DATA_PIPELINE.md) for the read-only Alpaca adapter, immutable
observations with availability checks, raw-share cash/dividend/split accounting,
and a forward signal recorder. Those components preserve the original CSVs and
require a separate verified raw data source. Current Alpaca data needs secure API
keys, allowed network access and appropriate feed/action entitlements. No orders
are placed by these tools.
