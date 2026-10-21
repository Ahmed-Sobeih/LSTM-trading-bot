"""Score frozen forecasts after their exit session completes. Never refits."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from market_data import calendar, read_snapshot, utc
from research import Config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    """Publish a complete JSON file atomically without replacing an earlier file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def verify_decision(path):
    record = json.loads(Path(path).read_text())
    if record.get('schema_version') != 1:
        raise ValueError('Unsupported decision schema')
    decision, execution = utc(record['decision_at']), utc(record['execution_open'])
    bars, _, manifest = read_snapshot(record['snapshot'], decision)
    signal = pd.Timestamp(record['signal_session'])
    if (manifest['symbol'] != record['symbol'] or bars.session.iloc[-1] != signal or
            manifest['sha256'] != record['snapshot_sha256'] or
            utc(manifest['observed_at']) != utc(record['snapshot_observed_at'])):
        raise ValueError('Decision provenance does not match its original snapshot')
    cal = calendar(signal, signal+pd.Timedelta(days=20))
    entry_session = cal.next_session(signal)
    exit_session = cal.next_session(entry_session)
    if execution != cal.session_open(entry_session) or utc(record['expires_at']) != execution or decision >= execution:
        raise ValueError('Decision was not frozen before the correct execution open')
    config = Config(**record['config'])
    prediction, exposure = float(record['prediction']), float(record['desired_exposure'])
    if not np.isfinite([prediction, exposure]).all() or not 0 <= exposure <= config.max_exposure:
        raise ValueError('Invalid frozen forecast or exposure')
    last_label = pd.Timestamp(record['training_last_label_exit'])
    if pd.isna(last_label) or last_label > signal:
        raise ValueError('Frozen training cutoff contains future labels')
    return record, manifest, pd.Timestamp(entry_session), pd.Timestamp(exit_session)


def score_decision(path, outcome_snapshot, root='data/outcomes', as_of=None):
    as_of = pd.Timestamp.now(tz='UTC') if as_of is None else utc(as_of)
    path = Path(path)
    record, original_manifest, entry, exit_day = verify_decision(path)
    if utc(record['decision_at']) > as_of:
        raise ValueError('Decision has not yet been observed at this scoring time')
    output = Path(root)/f'{record["symbol"]}-{record["signal_session"]}.json'
    if output.exists():
        old = json.loads(output.read_text())
        if old['decision_sha256'] != digest(path):
            raise ValueError('Frozen decision changed after scoring')
        if utc(old['scored_at']) > as_of or utc(old['outcome_observed_at']) > as_of:
            return {'status':'pending', 'symbol':record['symbol'], 'signal_session':record['signal_session'],
                    'reason':'Stored outcome was observed after requested scoring time'}
        return old
    cal = calendar(entry, exit_day+pd.Timedelta(days=1))
    # Daily bars contain the open, but are only accepted once the full day closes.
    if as_of < cal.session_close(exit_day):
        return {'status':'pending', 'symbol':record['symbol'], 'signal_session':record['signal_session'],
                'reason':'Exit session has not completed', 'exit_session':str(exit_day.date()),
                'earliest_daily_data_at':(cal.session_close(exit_day)+pd.Timedelta(minutes=16)).isoformat()}
    bars, actions, manifest = read_snapshot(outcome_snapshot, as_of, allow_stale=True)
    if manifest['symbol'] != record['symbol'] or manifest['source'] != original_manifest['source']:
        raise ValueError('Outcome symbol/feed/source differs from frozen decision')
    indexed = bars.set_index('session')
    if entry not in indexed.index or exit_day not in indexed.index:
        return {'status':'pending', 'symbol':record['symbol'], 'signal_session':record['signal_session'],
                'reason':'Accepted snapshot does not yet contain entry and exit sessions', 'exit_session':str(exit_day.date())}
    events = actions[actions.effective_date == exit_day]
    ratio = float(events.loc[events.kind == 'split', 'split_ratio'].prod())
    dividend = float(events.loc[events.kind == 'dividend', 'cash_per_share'].sum())
    entry_open, exit_open = float(indexed.loc[entry,'open']), float(indexed.loc[exit_day,'open'])
    price_return = ratio*exit_open/entry_open-1
    dividend_return = ratio*dividend/entry_open
    actual = price_return+dividend_return
    config = Config(**record['config'])
    weight = float(record['desired_exposure'])
    cost = (config.fee_bps+config.half_spread_bps+config.slippage_bps)/10000
    # Fractional-share, isolated round trip. Mark ex-date entitlement as a
    # receivable; this is neither spendable cash nor continuous account P&L.
    units = weight/(entry_open*(1+weight*cost))
    cash = 1-units*entry_open*(1+cost)
    isolated_net = cash+units*ratio*(exit_open*(1-cost)+dividend)-1
    model_identity = {'config':record['config'], 'code_sha256':record['code_sha256']}
    result = {'schema_version':1, 'status':'scored', 'symbol':record['symbol'],
              'signal_session':record['signal_session'], 'entry_session':str(entry.date()),
              'exit_session':str(exit_day.date()), 'decision_sha256':digest(path),
              'decision_file':str(path.resolve()), 'scoring_code_sha256':digest(__file__),
              'decision_at':record['decision_at'], 'scored_at':as_of.isoformat(),
              'cohort':hashlib.sha256(json.dumps(model_identity,sort_keys=True).encode()).hexdigest(),
              'outcome_snapshot':str(Path(outcome_snapshot).resolve()), 'outcome_sha256':manifest['sha256'],
              'outcome_observed_at':manifest['observed_at'], 'entry_open':entry_open, 'exit_open':exit_open,
              'split_ratio':ratio, 'dividend_per_post_split_share':dividend,
              'prediction':record['prediction'], 'actual_total_return':actual,
              'absolute_error':abs(float(record['prediction'])-actual), 'zero_forecast_absolute_error':abs(actual),
              'desired_exposure':weight, 'isolated_round_trip_net_return_proxy':isolated_net,
              'warning':'Forecast evaluation only. No orders/fills. Isolated fractional-share proxy includes two cost sides and marked dividend receivables; do not compound it as continuous-account returns.'}
    try:
        write_once(output, result)
    except FileExistsError:
        old = json.loads(output.read_text())
        if old['decision_sha256'] != result['decision_sha256']:
            raise ValueError('Concurrent score used a different decision')
        return old
    return result


def score_all(decisions='data/decisions', snapshots='data/snapshots', root='data/outcomes', as_of=None):
    as_of = pd.Timestamp.now(tz='UTC') if as_of is None else utc(as_of)
    latest = {}
    for path in sorted(Path(snapshots).glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        if utc(manifest['observed_at']) <= as_of:
            symbol = manifest['symbol']
            if symbol not in latest or utc(manifest['observed_at']) > latest[symbol][0]:
                latest[symbol] = (utc(manifest['observed_at']), path.parent)
    results = []
    for path in sorted(Path(decisions).glob('*.json')):
        try:
            record = json.loads(path.read_text())
            if record['symbol'] not in latest:
                raise ValueError('No observed outcome snapshot for decision symbol')
            result = score_decision(path, latest[record['symbol']][1], root, as_of)
        except (ValueError, KeyError, OSError) as error:
            result = {'status':'error', 'decision_file':str(path), 'error_type':type(error).__name__}
        results.append(result)
    scored = [r for r in results if r['status']=='scored']
    groups = {}
    for record in scored:
        groups.setdefault((record['symbol'], record['cohort']), []).append(record)
    summaries = []
    for (symbol, cohort), rows in sorted(groups.items()):
        predicted = np.array([r['prediction'] for r in rows])
        actual = np.array([r['actual_total_return'] for r in rows])
        summaries.append({'symbol':symbol, 'cohort':cohort, 'count':len(rows),
                          'mae':float(np.abs(predicted-actual).mean()),
                          'rmse':float(np.sqrt(np.square(predicted-actual).mean())),
                          'zero_forecast_mae':float(np.abs(actual).mean())})
    return {'as_of':as_of.isoformat(), 'counts':{s:sum(r['status']==s for r in results) for s in ['scored','pending','error']},
            'cohorts':summaries, 'results':results,
            'warning':'Small forward samples do not establish trading profitability. Different model/configuration cohorts are reported separately; outcomes are never refit.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions',default='data/decisions')
    parser.add_argument('--snapshots',default='data/snapshots')
    parser.add_argument('--outcomes',default='data/outcomes')
    args = parser.parse_args()
    report = score_all(args.decisions,args.snapshots,args.outcomes)
    print(json.dumps(report,indent=2,allow_nan=False))
    if report['counts']['error']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
