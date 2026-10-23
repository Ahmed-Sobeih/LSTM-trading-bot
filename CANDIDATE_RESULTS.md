# Candidate research — 2026-10-24

These are exploratory historical development results. The forward experiment and 2026 performance holdout were not changed or evaluated. Source protocols, fold choices, predictions, raw-share ledgers and costs are under ignored reports/candidates-{msft,aapl}-20261024.

The shared signal evaluation interval is 2024-05-03 through 2025-12-29. It differs from earlier reports because every candidate uses the same 20-row sequence warmup. Training/validation target exits are purged, preprocessing uses training data only, and final comparisons use raw-share accounting with modeled costs.

| Model | MSFT total return | AAPL total return |
|---|---:|---:|
| ridge | 2.21% | -9.66% |
| gradient_boosting | -17.07% | 51.71% |
| lstm_seed_7 | 11.59% | -15.25% |
| lstm_seed_42 | -14.30% | -19.97% |
| lstm_seed_123 | -17.85% | 15.95% |
| buy_hold | 19.23% | 49.96% |
| momentum_20 | 14.28% | 20.98% |

LSTMs are highly seed-sensitive; none is promoted. AAPL gradient boosting narrowly exceeds buy/hold at the base cost assumption, but the frozen-signal cost replay removes that advantage when costs double. On MSFT it underperforms buy/hold even without costs. The baseline is not risk-matched: candidate drawdown stops and changing exposures differ from buy/hold.

| AAPL cost assumption | Gradient boosting | Buy/hold |
|---|---:|---:|
| 0x | 65.91% | 49.97% |
| 1x | 51.71% | 49.96% |
| 2x | 38.97% | 49.83% |
| 4x | 17.29% | 49.75% |

Sensitivity replays frozen desired signals, with risk stops responding to costs; it does not retrain or reselect thresholds. Forecast MAE for all LSTM seeds on both assets is worse than a zero-return forecast in this development run. Asset/model selection and repeated historical investigation limit inference. SIP action completeness is still awaiting independent issuer corroboration.

The registered run used a small 16-unit return LSTM, a maximum of 20 epochs with chronological validation early stopping, train-only feature/target scaling, and fresh refits using the selected epoch count. Seeds 7, 42 and 123 are all shown; none was chosen after inspecting the results. Tree settings and ridge alphas were selected using preceding validation wealth after modeled costs. No result establishes live profitability.

Reproduce using candidate_research.py and a fresh output directory. Run candidate_sensitivity.py against the same snapshot and saved candidate directory to replay costs without retraining. Both record code/data/input hashes. Further research must be preregistered and separated from the existing forward cohort.
