# Alpaca validation — 2026-10-21

Secure credentials and data-network access were verified with authenticated
read-only requests. No broker orders were submitted.

## Accepted data

Both MSFT and AAPL snapshots contain 1,541 raw SIP daily bars from 2020-09-01
through 2026-10-20 and 24 ordinary cash-dividend records each. Session coverage,
OHLCV, availability, action fields, freshness and snapshot hashes passed validation.
Snapshots remain under ignored `data/snapshots/`; credentials are not stored there.
Alpaca action completeness has not been independently corroborated.

Earlier requested history was deliberately rejected: pre-March-2020 dividend
records lacked payment dates, and AAPL's August-2020 split included unsupported
due-bill terms. Neither dates nor event handling were invented. A common September
2020 start keeps the current two-asset comparison aligned and excludes those events.
MSFT's acquisition of Nuance is returned by Alpaca's symbol filter; as a cash
acquisition it does not exchange existing MSFT shares and is explicitly excluded.

The integration also exposed and fixed a holiday-start calendar error. Regression
coverage retains rejection of missing payment dates and due bills, distinguishes
acquirer-only cash mergers from mergers affecting held shares, and checks weekend
starts.

## Retrospective development results

Frozen configuration: 756 minimum training sessions, 126 validation sessions,
126-session folds, 5 basis points per traded side, 100% maximum long exposure,
20% next-open drawdown stop for the model, and 2026-01-01 holdout boundary.
Four evaluation folds cover signal dates 2024-04-08 through 2025-12-29.

| Asset | Model total return | Buy/hold total return | Model max drawdown | Model stop triggered |
|---|---:|---:|---:|---|
| MSFT | -3.54% | +14.36% | -15.20% | No |
| AAPL | +2.30% | +61.78% | -24.83% | Yes |

Returns are not annualized. Buy/hold has no drawdown stop; exposure differs.
These are raw-share simulations with modeled costs and payment-date cash handling,
not actual fills. The signal has not demonstrated an advantage over buy/hold.
Detailed snapshots/configuration/fills/ledgers are linked in ignored
`reports/current_raw_msft/report.json` and `reports/current_raw_aapl/report.json`.
The new 2026 period was not backtested for performance. The prospective fit below
does use already-realized 2026 observations, as a real-time model legitimately may.

## First forward research observations

One immutable-by-API research decision per asset was recorded using the completed
2026-10-20 session, before the 2026-10-21 opening session (16:30 Africa/Cairo).
Records in `data/decisions/` include observation/decision times, code/data hashes,
chosen parameters and training cutoffs. They are forecasts only: no capital was
allocated and no orders were submitted. Outcomes are not yet observable at recording
time. Do not edit these decisions after outcomes arrive.

To reproduce or continue with new accepted snapshots:

```python
from research import Config
from raw_research import run
from prospective import record_signal

config = Config(min_train=756, validation=126, fold=126,
                holdout_start='2026-01-01')
# Replace with the relevant accepted timestamped snapshot directory.
snapshot = 'data/snapshots/REPLACE_WITH_DIRECTORY'
run(snapshot, 'reports/reproduction', config=config)
record_signal(snapshot, config=config)
```

The default five-year minimum in the generic scripts is longer than this explicitly
chosen three-year research configuration. Do not silently change window lengths
or compare results from different configurations. A duplicate prospective decision
is rejected; record only once per symbol/session and before its next open.

Next: collect a new snapshot after each completed session, preserve the frozen
decisions, and evaluate subsequent realized outcomes without retuning past records.
Automated scheduling and broker execution have not been installed. Snapshot and
decision files need separate durable backup because Git intentionally excludes them.
