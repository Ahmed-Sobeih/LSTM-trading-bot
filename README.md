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
