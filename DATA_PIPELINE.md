# Raw market data and forward research

The original CSVs are retained unchanged. They are split/retrospectively adjusted
vendor histories, not attested raw-share price records. Do not rename them into
the raw schema or infer missing dividends/payment dates from Adj Close.

## Alpaca setup

Install `requirements.txt`. In the cloud, activate
`source /workspace/lstm-environment/activate.sh`. Configure `APCA_API_KEY_ID` and
`APCA_API_SECRET_KEY` securely in environment settings. The saved draft also adds
`data.alpaca.markets` to the network allowlist. Review/save the draft and publish
the environment as required by the UI; saved drafts do not apply runtime changes.
Never put credentials in source, commands, notebooks, reports or Git.

```bash
python alpaca_ingest.py --symbol MSFT --start 2020-09-01
python alpaca_ingest.py --symbol AAPL --start 2020-09-01
```

The adapter only imports Alpaca historical-data clients. It requests RAW daily
bars, follows SDK pagination for bars and corporate actions, and defaults to SIP.
Actual access depends on your data/corporate-action entitlements. `--feed iex`
is an explicit single-venue alternative; its volume differs from consolidated
SIP volume and requires separate model validation. Do not mix feeds in research.

Only completed sessions are accepted; a 16-minute delay avoids requesting recent
SIP data unavailable on some plans. A download immediately after the close may be
rejected until the latest daily bar is available. Missing sessions, zero volume,
bad OHLC, missing payment dates, due bills, special/foreign dividends and unsupported
corporate-action types fail closed. Forward splits and ordinary cash dividends
are normalized. API failure is not permission to substitute empty action history.
Corporate-action API historical coverage/completeness still needs validation
against a second source before trusting historical trading results.
Cash mergers returned solely because the requested symbol is the acquirer are
excluded: those events do not exchange existing acquirer shares. Cash mergers
affecting the held/acquired security remain blocked.

Run once after each session's complete daily bar is available, before the next
session open. Scheduling is deliberately not installed until real API access and
feed behavior have been verified. No broker trading endpoints are used.

## Snapshots and availability

Each accepted download creates a new timestamped directory under
`data/snapshots/`, containing bars.csv, actions.csv and manifest.json. The manifest
records first observation time, source, raw basis and SHA-256 hashes. The API never
overwrites an existing observation; this is not signed or tamper-proof storage.
New historical downloads are conservatively marked available at download time,
not at historical session close. Existing observations remain separate if the
vendor revises history.

Daily bars must cover every XNYS session in their date span, including holiday and
early-close rules from exchange-calendars. Freshness is checked against the latest
completed session. This is suitable for MSFT/AAPL's US session schedule, not an
arbitrary venue or an assurance of vendor publication latency.

For another explicitly attested RAW source, import canonical CSVs:

```bash
python market_data.py --bars raw_bars.csv --actions actions.csv \
  --source 'Provider/export description' --price-basis raw
```

Bars header:
`symbol,session,open,high,low,close,volume,available_at`

Actions header (also required on an empty action table):
`id,symbol,kind,effective_date,known_at,split_ratio,cash_per_share,pay_date`

Use ISO dates for sessions/effective/payment dates and timezone-aware ISO timestamps
for availability/knowledge. Split ratio means new shares per old share. `kind` is
`split` or `dividend`; leave irrelevant fields empty. Effective date is ex-date.
If split and dividend coincide, cash is per post-split share. Raw basis and complete
action coverage are explicit source attestations, not inferable from these fields.
`--allow-stale-research` permits historical imports, never fresh forward signals.

## Raw-share historical evaluation

```bash
python raw_research.py --snapshot data/snapshots/REPLACE_WITH_DIRECTORY --out reports/raw_msft
```

This reuses walk-forward ridge selection on causal realized total-return features,
then simulates an actual cash/share ledger using raw prices. Orders generated after
close execute at the following open. Splits change overnight shares before open.
Dividend entitlement accrues before ex-date trades, survives a sale, and remains a
receivable until payment-date close (or next available session close). Receivables
cannot buy shares. Opening on ex-date does not earn that dividend.

Whole-share targets obey cash/exposure limits after modeled costs; split-created
fractions are retained. Cash-in-lieu, taxes, withholding, merger/spinoff accounting,
multi-currency and real fill modeling are unsupported. Final liquidation is scheduled
at the last close; unpaid dividends remain separately marked receivables. Model
selection's total-return approximation does not simulate payment-date funding, so
inspect the executable ledger as well as validation scores.

Outputs contain snapshot hashes, fit cutoffs, predictions, fills with commission
and spread/slippage costs, daily cash/shares/receivables/equity and risk-stop state.
This is retrospective research using a recently observed history, not a historical
point-in-time backtest. Results cannot establish the history's prior availability.

## Freeze forward signals

```bash
python prospective.py --snapshot data/snapshots/REPLACE_WITH_DIRECTORY
```

This requires a snapshot observed before the current decision, containing the
latest completed session, and a decision before its next execution open. It selects
parameters using earlier eligible labels and writes one exclusive JSON record per
symbol/signal session under `data/decisions/`. Records include prediction, desired
exposure, model configuration, training cutoff, snapshot hashes, code hashes, and
execution/expiry time. Duplicate signal records are rejected rather than revised.

These are forward **research** records. They do not reserve capital, execute orders,
maintain an account-wide risk state or guarantee safe trades. Actual paper/live
orders remain out of scope until prospective results, vendor action coverage and
execution/reconciliation controls are validated. Snapshot/signal directories and
generated reports are ignored by Git; back them up separately if collecting data.

## Validation status

`python -m unittest -v` exercises holiday/freshness checks, observation-time and
hash enforcement, corporate-action normalization, split wealth invariance,
dividend entitlement/payment/cash constraints, costs/solvency, forward-signal
freezing, and a raw research run on explicitly synthetic data. Alpaca SDK requests
are checked with mocks, including RAW adjustment and unbounded action pagination.
Live SIP bars and corporate-action downloads passed on 2026-10-21; see
[ALPACA_VALIDATION.md](ALPACA_VALIDATION.md) for exact coverage, results and limits.
Synthetic fixtures are never substituted for market data or used to claim
strategy performance.
