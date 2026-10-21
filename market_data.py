"""Immutable raw-price observations. Historical downloads are not PIT history."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


BAR_COLUMNS = ['symbol', 'session', 'open', 'high', 'low', 'close', 'volume', 'available_at']
ACTION_COLUMNS = ['id', 'symbol', 'kind', 'effective_date', 'known_at', 'split_ratio', 'cash_per_share', 'pay_date']


def utc(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError('Timestamps must include an explicit timezone')
    return value.tz_convert('UTC')


def calendar(start, end):
    import exchange_calendars as xcals
    return xcals.get_calendar('XNYS', start=str(pd.Timestamp(start).date()), end=str(pd.Timestamp(end).date()))


def last_completed_session(as_of):
    as_of = utc(as_of)
    cal = calendar(as_of.date()-pd.Timedelta(days=20), as_of.date()+pd.Timedelta(days=1))
    completed = [s for s in cal.sessions if cal.session_close(s) <= as_of]
    if not completed:
        raise ValueError('No completed exchange session')
    return pd.Timestamp(completed[-1]).tz_localize(None).normalize()


def validate(bars, actions, observed_at, allow_stale=False):
    observed_at = utc(observed_at)
    if not set(BAR_COLUMNS).issubset(bars) or not set(ACTION_COLUMNS).issubset(actions):
        raise ValueError(f'Expected bars {BAR_COLUMNS} and actions {ACTION_COLUMNS}')
    bars, actions = bars.copy(), actions.copy()
    if bars.empty or bars.symbol.nunique(dropna=False) != 1 or bars.symbol.isna().any():
        raise ValueError('Each snapshot must contain exactly one nonmissing symbol')
    symbol = str(bars.symbol.iloc[0])
    if not symbol or not symbol.replace('.', '').replace('-', '').isalnum():
        raise ValueError('Invalid symbol')
    bars['session'] = pd.to_datetime(bars.session, errors='raise')
    if bars.session.dt.tz is not None or bars.session.isna().any() or (bars.session != bars.session.dt.normalize()).any():
        raise ValueError('Sessions must be naive exchange dates, without time components')
    if bars.session.duplicated().any() or not bars.session.is_monotonic_increasing:
        raise ValueError('Bars must be unique and ascending')
    values = bars[['open','high','low','close','volume']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] <= 0).any():
        raise ValueError('Raw OHLCV must be finite and positive; zero-volume fills are not assumed')
    if ((bars.high < bars[['open','close','low']].max(axis=1)) | (bars.low > bars[['open','close','high']].min(axis=1))).any():
        raise ValueError('Inconsistent raw OHLC')
    cal = calendar(bars.session.min(), max(bars.session.max(), pd.Timestamp(observed_at.date())))
    expected = pd.DatetimeIndex(cal.sessions_in_range(bars.session.min(), bars.session.max())).tz_localize(None)
    if not pd.DatetimeIndex(bars.session).equals(expected):
        raise ValueError('Missing or extra XNYS sessions; do not silently fill exchange gaps')
    bars['available_at'] = [utc(x) for x in bars.available_at]
    closes = pd.DatetimeIndex([cal.session_close(x) for x in bars.session])
    if (pd.DatetimeIndex(bars.available_at) < closes).any() or (bars.available_at > observed_at).any():
        raise ValueError('Daily bars must be available after session close and no later than observation')
    if not allow_stale and bars.session.iloc[-1] != last_completed_session(observed_at):
        raise ValueError('Stale daily bars: latest completed exchange session is missing')
    if not actions.empty:
        if actions.id.isna().any() or actions.id.duplicated().any() or (actions.symbol != symbol).any():
            raise ValueError('Action IDs must be unique and match the bar symbol')
        actions['known_at'] = [utc(x) for x in actions.known_at]
        if (actions.known_at > observed_at).any():
            raise ValueError('Actions cannot be known after snapshot observation')
        actions['effective_date'] = pd.to_datetime(actions.effective_date, errors='raise')
        actions['pay_date'] = pd.to_datetime(actions.pay_date, errors='raise')
        for _, action in actions.iterrows():
            if pd.isna(action.effective_date) or action.effective_date != action.effective_date.normalize():
                raise ValueError('Action effective dates must be exchange dates')
            if action.effective_date not in pd.DatetimeIndex(bars.session):
                raise ValueError('Action effective date must be one of the snapshot sessions')
            if action.kind == 'split':
                if not np.isfinite(float(action.split_ratio)) or float(action.split_ratio) <= 0:
                    raise ValueError('Split ratio must be positive new shares per old share')
            elif action.kind == 'dividend':
                if not np.isfinite(float(action.cash_per_share)) or float(action.cash_per_share) < 0:
                    raise ValueError('Dividend must be finite and nonnegative')
                if pd.isna(action.pay_date) or action.pay_date != action.pay_date.normalize() or action.pay_date < action.effective_date:
                    raise ValueError('Dividend requires a payment date on/after its ex-date')
            else:
                raise ValueError('Unsupported action type; mergers/spinoffs require explicit accounting')
    return bars, actions


def save_snapshot(bars_path, actions_path, root, source, price_basis, allow_stale=False):
    if price_basis != 'raw':
        raise ValueError('Only explicitly attested raw, unadjusted prices are supported')
    if not source.strip():
        raise ValueError('A source description is required')
    observed = pd.Timestamp.now(tz='UTC')
    bars, actions = validate(pd.read_csv(bars_path), pd.read_csv(actions_path), observed, allow_stale)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive directory creation; no observation or earlier manifest is replaced.
    destination = root / (observed.strftime('%Y%m%dT%H%M%S%fZ')+'-'+str(bars.symbol.iloc[0]))
    destination.mkdir()
    bars.to_csv(destination/'bars.csv', index=False)
    actions.to_csv(destination/'actions.csv', index=False)
    hashes = {name: hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in ['bars.csv','actions.csv']}
    manifest = {'schema_version': 1, 'observed_at': observed.isoformat(), 'source': source,
                'symbol': str(bars.symbol.iloc[0]), 'price_basis': 'raw', 'calendar': 'XNYS',
                'allow_stale_research': allow_stale, 'sha256': hashes,
                'warning': 'A new historical import does not establish historical point-in-time availability. Raw price basis and complete action history are source attestations, not independently verified facts.'}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return destination


def read_snapshot(path, as_of, allow_stale=False):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    as_of = utc(as_of)
    if manifest.get('schema_version') != 1 or manifest.get('price_basis') != 'raw':
        raise ValueError('Unsupported snapshot schema or price basis')
    if utc(manifest['observed_at']) > as_of:
        raise ValueError('Snapshot was not observed by this decision time')
    for name in ['bars.csv','actions.csv']:
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != manifest['sha256'][name]:
            raise ValueError('Snapshot integrity check failed')
    bars, actions = validate(pd.read_csv(path/'bars.csv'), pd.read_csv(path/'actions.csv'), manifest['observed_at'], True)
    if not allow_stale and bars.session.iloc[-1] != last_completed_session(as_of):
        raise ValueError('Snapshot is stale at decision time')
    return bars, actions, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars', required=True)
    parser.add_argument('--actions', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--price-basis', choices=['raw'], required=True)
    parser.add_argument('--root', default='data/snapshots')
    parser.add_argument('--allow-stale-research', action='store_true')
    args = parser.parse_args()
    print(save_snapshot(args.bars, args.actions, args.root, args.source, args.price_basis, args.allow_stale_research))


if __name__ == '__main__':
    main()
