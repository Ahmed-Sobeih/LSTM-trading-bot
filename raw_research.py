"""Retrospective raw-share research on an observed, hash-verified snapshot."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from accounting import raw_samples, simulate
from market_data import read_snapshot
from research import Config, walk_forward


def run(snapshot, out, config=Config(), holdout=False):
    bars, actions, manifest = read_snapshot(snapshot, pd.Timestamp.now(tz='UTC'), allow_stale=True)
    samples = raw_samples(bars, actions)
    evaluated, folds = walk_forward(samples, config, holdout)
    # Include the last signal date so its next-open execution can be mapped.
    subset = bars[(bars.session >= evaluated.index[0]) & (bars.session <= evaluated.exit_date.iloc[-1])]
    events = actions[actions.effective_date.isin(subset.session)]
    strategies = {'ridge': evaluated.desired,
                  'cash': pd.Series(0., index=evaluated.index),
                  'buy_hold': pd.Series(config.max_exposure, index=evaluated.index),
                  'momentum_20': (evaluated.return_20 > 0)*config.max_exposure}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, signals in strategies.items():
        ledger, fills = simulate(subset, events, signals, config, manifest['observed_at'], stop=name=='ridge')
        ledger.to_csv(out/f'{name}_ledger.csv', index=False)
        fills.to_csv(out/f'{name}_fills.csv', index=False)
        summaries[name] = {'total_return': float(ledger.equity.iloc[-1]/10000-1),
                           'max_drawdown': float(ledger.drawdown.min()), 'cost_dollars': float(ledger.cost.sum()),
                           'final_receivable': float(ledger.receivable.iloc[-1]),
                           'risk_stop_triggered': bool(ledger.halted.any()), 'fills': len(fills)}
    evaluated.to_csv(out/'predictions.csv')
    report = {'snapshot': str(Path(snapshot).resolve()), 'observed_at': manifest['observed_at'],
              'data_sha256': manifest['sha256'], 'config': asdict(config), 'folds': folds,
              'mode': 'holdout' if holdout else 'development', 'results': summaries,
              'limitations': ['Retrospective research; downloaded histories are not historical PIT observations.',
                             'Requires actual raw prices and complete actions; source attestations are not independently verified.',
                             'Model selection uses total-return labels with immediate dividend value; executable ledger delays dividend cash until payment.',
                             'Whole-share orders, modeled costs, no actual auction fills; split-created fractional shares retained, no cash-in-lieu.',
                             'Same-day split/dividend assumes cash per post-split share; unsupported events must not be silently omitted.',
                             'Final positions liquidated at scheduled final close; unpaid dividends remain marked receivables.',
                             'No withholding/taxes, merger/spinoff support, multi-currency, interest or live orders.']}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--out', default='reports/raw')
    parser.add_argument('--evaluate-holdout', action='store_true')
    args = parser.parse_args()
    result = run(args.snapshot, args.out, holdout=args.evaluate_holdout)
    print(pd.DataFrame(result['results']).T.to_string())


if __name__ == '__main__':
    main()
