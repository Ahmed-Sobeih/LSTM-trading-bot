# Daily forward research

Run from the repository directory:

```bash
bash scripts/run_daily.sh
```

The wrapper activates the prepared cloud Python environment when present. On
another host, first activate a Python 3.12 environment with requirements.txt
installed and supply the two Alpaca credentials securely to the process.
No credentials belong in this script, Git, JSON records, or command arguments.

## Each cycle

1. Acquire an OS lock so two runs cannot collect/write simultaneously.
2. Check the XNYS calendar and wait until the completed close plus a 16-minute
   data-publication buffer. Holidays and early closes are handled by the calendar.
3. Reuse each asset's existing valid current snapshot, or download/validate new
   raw SIP bars and corporate actions. Data integrity failures block reuse.
4. Preserve an existing valid decision, or freeze a new decision before the
   next session open. A missed open is reported, never repaired by backdating.
5. Score older frozen decisions only when accepted daily data contains their
   entry and exit sessions. No forecasts are refitted during scoring.
6. Write a complete, timestamped run report under data/runs/.

The fixed configuration matches the first forward records: 756 training sessions,
126 validation sessions, 126-session historical folds, 2026-01-01 research boundary,
5 bps per traded side, 100% maximum exposure. The predefined fitting/selection
algorithm may refit on newly available history for a new session; earlier decisions
are never changed using their outcomes. Do not manually tune this configuration
from the forward results while describing the collection as a frozen experiment.

Nonzero exit status indicates a collection/scoring failure or a missed decision
window. An ordinary pending outcome or publication-buffer wait is not a failure.
Per-asset failures do not prevent the other asset from being processed. Error logs
record exception classes without SDK request/authentication details.

## Forecast scoring

To score without collecting or recording new forecasts:

```bash
python forward_score.py
```

The default data roots are data/decisions, data/snapshots, and data/outcomes.
The scorer verifies the original decision's data provenance and timing, checks
the outcome snapshot's integrity and source/feed, and uses the actual next-open
to following-open return. Exit-date splits and dividend entitlement are included.
The first accepted outcome is recorded with both decision and outcome hashes and
the scoring code hash. Vendor revisions do not rewrite it. Changed decisions are
detected by hash comparisons once scored; these local files are not signed or
tamper-proof, so preserve and back them up separately.

The report separates scored, pending, and errored records and groups error metrics
by asset/model-code/configuration cohort. It reports MAE/RMSE against a zero-return
forecast. It does not pool changing models into one purported fixed model result.
No sample-size threshold or significance claim is inferred from a few days.

An isolated fractional-share round-trip return proxy is also recorded, using
the frozen exposure, two sides of assumed costs and marked dividend receivables.
It is **not** live P&L or a continuous account ledger; successive decisions may
overlap and it must not be compounded as a trading equity curve. Actual share/cash
research remains in accounting.py/raw_research.py; no broker orders are submitted.

## Scheduling and persistence

The current cloud container has no cron service and is not an always-on execution
host. This change does not install an unattended scheduler. A running command or
published environment is not a guarantee that future daily collection occurs.

On a durable Linux host with its own secure credential injection, a scheduler can
invoke the wrapper every five minutes. Example crontab entry, with host paths
replaced appropriately:

```cron
*/5 * * * * /bin/bash /path/to/LSTM-trading-bot/scripts/run_daily.sh >> /path/to/private-logs/daily.log 2>&1
```

Use a writable, private log directory, an activated/configured Python runtime,
and an operator alert for nonzero exit statuses. The calendar decides the correct
market timing regardless of the scheduler host's timezone/DST. Back up data/snapshots,
data/decisions, data/outcomes and data/runs, which are deliberately ignored by Git.
Scheduling on GitHub Actions or another hosted service additionally requires its
own secret configuration and durable artifact/state handling; cloud proxy bindings
must not be copied into those systems as though they were raw credentials.

## Initial run

The first real repeat run reused both accepted snapshots and preserved both
2026-10-20 decisions. It reported 0 scored, 2 pending and 0 errors. The exit session
is 2026-10-22. With full daily bars and the publication buffer, the earliest expected
scoring time is October 22 around 23:16 Africa/Cairo, subject to provider latency.
Do not fabricate earlier results from synthetic tests or from incomplete daily bars.

## First observed outcomes — October 23, 2026 (Africa/Cairo)

The cycle downloaded completed October 22 data and scored the original October 20
decisions for the October 21 open to October 22 open horizon:

| Asset | Frozen predicted return | Observed return | Absolute error |
|---|---:|---:|---:|
| AAPL | +0.0880% | +1.6443% | 1.5563 percentage points |
| MSFT | +0.1162% | +0.6989% | 0.5826 percentage points |

Each model forecast was closer than a zero-return forecast on this single outcome,
but materially understated the move. One observation per asset does not establish
an edge. No orders were placed; these are forecast outcomes, not realized profits.
The first outcome files are preserved under data/outcomes/ with input hashes.

The same cycle froze October 22 decisions before the October 23 open. Their exit
session is October 26 (the weekend intervenes). It did not backfill a missing
October 21 decision: no unattended scheduler had run that session's collection.

The live request exposed a delayed-SIP boundary bug: requesting through the next
midnight after today's close included future/recent minutes and triggered Alpaca's
subscription restriction. Requests now cap the end at the earlier of that midnight
and current time minus 16 minutes; the complete daily bars were then accepted.
