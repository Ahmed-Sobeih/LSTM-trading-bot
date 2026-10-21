"""Freeze one forward research signal per symbol/session. Does not trade."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from accounting import raw_samples
from market_data import calendar, read_snapshot, utc
from research import Config, ridge_predict, select_model


def record_signal(snapshot, root='data/decisions', config=Config(), as_of=None):
    decision = pd.Timestamp.now(tz='UTC') if as_of is None else utc(as_of)
    bars, actions, manifest = read_snapshot(snapshot, decision)
    latest = bars.session.iloc[-1]
    cal = calendar(latest, latest+pd.Timedelta(days=20))
    next_session = cal.next_session(latest)
    next_open = cal.session_open(next_session)
    if decision >= next_open:
        raise ValueError('Next execution open has passed; wait for a new completed-session snapshot')
    frame = raw_samples(bars, actions, include_unlabeled=True)
    history = frame[frame.target.notna() & (frame.exit_date <= latest)]
    alpha, threshold = select_model(history, config)
    prediction = float(ridge_predict(history, frame.loc[[latest]], alpha)[0])
    if not np.isfinite(prediction):
        raise ValueError('Nonfinite prospective prediction')
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{manifest["symbol"]}-{latest.date()}.json'
    here = Path(__file__).parent
    record = {'schema_version': 1, 'purpose': 'forward research only; no broker orders',
              'symbol': manifest['symbol'], 'signal_session': str(latest.date()), 'decision_at': decision.isoformat(),
              'execution_open': next_open.isoformat(), 'expires_at': next_open.isoformat(),
              'snapshot': str(Path(snapshot).resolve()), 'snapshot_observed_at': manifest['observed_at'],
              'snapshot_sha256': manifest['sha256'],
              'code_sha256': {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
                              for name in ['prospective.py','research.py','market_data.py','accounting.py']},
              'config': asdict(config), 'alpha': alpha,
              'threshold': None if np.isinf(threshold) else threshold,
              'prediction': prediction, 'desired_exposure': config.max_exposure if prediction > threshold else 0.,
              'training_last_label_exit': history.exit_date.max().isoformat(), 'training_rows': len(history)}
    with path.open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--root', default='data/decisions')
    args = parser.parse_args()
    print(record_signal(args.snapshot, args.root))


if __name__ == '__main__':
    main()
