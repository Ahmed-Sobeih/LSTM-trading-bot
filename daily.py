"""Run one locked, repeatable collection/forecast/scoring cycle. No orders."""
import argparse
import fcntl
import json
import uuid
from pathlib import Path

import pandas as pd

from alpaca_ingest import fetch
from forward_score import score_all, verify_decision, write_once
from market_data import calendar, last_completed_session, read_snapshot, utc
from prospective import record_signal
from research import Config


def recent_config():
    # Matches the first frozen forward decisions. Never tune from outcomes here.
    return Config(min_train=756,validation=126,fold=126,holdout_start='2026-01-01')


def existing_snapshot(root, symbol, start, feed, as_of):
    for path in sorted(Path(root).glob('*-'+symbol),reverse=True):
        manifest = json.loads((path/'manifest.json').read_text())
        if utc(manifest['observed_at']) > as_of:
            continue
        # A corrupt/new snapshot blocks reuse; never silently fall back to old data.
        bars, _, manifest = read_snapshot(path,as_of,allow_stale=True)
        if bars.session.iloc[-1] != last_completed_session(as_of):
            return None
        expected_start = calendar(start,bars.session.iloc[-1]).sessions[0]
        if bars.session.iloc[0] != expected_start or not manifest['source'].startswith(f'Alpaca {feed} daily RAW'):
            return None
        return path
    return None


def cycle(root='data', symbols=('MSFT','AAPL'), start='2020-09-01', feed='sip', as_of=None):
    now = pd.Timestamp.now(tz='UTC') if as_of is None else utc(as_of)
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    # OS lock is released even on process death. Keep the lock file's inode.
    with (root/'.daily.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another daily cycle is already running')
        updates = []
        latest = last_completed_session(now)
        ready = last_completed_session(now-pd.Timedelta(minutes=16)) == latest
        for symbol in symbols:
            try:
                if not ready:
                    updates.append({'symbol':symbol,'status':'waiting','reason':'Waiting for completed daily bar plus 16-minute publication buffer'})
                    continue
                snapshot = existing_snapshot(root/'snapshots',symbol,start,feed,now)
                reused = snapshot is not None
                if snapshot is None:
                    if as_of is not None:
                        raise ValueError('Historical cycle replay cannot fetch present-day data')
                    snapshot = fetch(symbol,start,root/'snapshots',feed)
                decision_time = pd.Timestamp.now(tz='UTC') if as_of is None else now
                bars, _, manifest = read_snapshot(snapshot,decision_time)
                session = bars.session.iloc[-1]
                decision_path = root/'decisions'/f'{symbol}-{session.date()}.json'
                if decision_path.exists():
                    record, _, _, _ = verify_decision(decision_path)
                    if record['config'] != recent_config().__dict__:
                        raise ValueError('Existing decision uses a different frozen model configuration')
                    decision_status = 'already_frozen'
                else:
                    cal = calendar(session,session+pd.Timedelta(days=20))
                    if decision_time >= cal.session_open(cal.next_session(session)):
                        decision_status = 'missed_execution_open'
                    else:
                        record_signal(snapshot,root/'decisions',recent_config(),decision_time)
                        decision_status = 'newly_frozen'
                updates.append({'symbol':symbol,'status':'missed' if decision_status=='missed_execution_open' else 'ok','snapshot':str(snapshot),
                                'reused_snapshot':reused,'session':str(session.date()),'decision':decision_status})
            except Exception as error:
                # SDK errors can contain request metadata. Persist class only.
                updates.append({'symbol':symbol,'status':'error','error_type':type(error).__name__})
        scored_at = pd.Timestamp.now(tz='UTC') if as_of is None else now
        scoring = score_all(root/'decisions',root/'snapshots',root/'outcomes',scored_at)
        report = {'run_at':scored_at.isoformat(),'collection':updates,'scoring':scoring,
                  'no_orders':True,'scheduling':'One-shot command; invoke regularly using a durable external scheduler.'}
        report_path = root/'runs'/(scored_at.strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex+'.json')
        write_once(report_path,report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data')
    args = parser.parse_args()
    try:
        report = cycle(args.root)
    except Exception as error:
        parser.exit(1,f'Daily cycle failed ({type(error).__name__}); inspect data and configuration. No orders were submitted.\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    if any(r['status'] not in ('ok','waiting') for r in report['collection']) or report['scoring']['counts']['error']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
