"""Read-only operational checks. JSON alerts and nonzero status, no messages sent."""
import argparse
import json
import tarfile
from pathlib import Path

import pandas as pd

from forward_score import digest, verify_decision
from market_data import calendar, last_completed_session, read_snapshot, utc
from state_backup import sha


def inspect(root='data', backups='backups', as_of=None, max_age_hours=30):
    now=pd.Timestamp.now(tz='UTC') if as_of is None else utc(as_of)
    root,backups=Path(root),Path(backups)
    alerts=[]
    def alert(code,severity='error',symbol=None):
        alerts.append({'code':code,'severity':severity,**({'symbol':symbol} if symbol else {})})
    expected=last_completed_session(now-pd.Timedelta(minutes=16))
    runs=sorted((root/'runs').glob('*.json'))
    if not runs:
        alert('missing_run_audit')
    else:
        try:
            report=json.loads(runs[-1].read_text());stamp=utc(report['run_at'])
            if stamp>now or now-stamp>pd.Timedelta(hours=max_age_hours):
                alert('stale_or_future_run_audit')
            for entry in report['collection']:
                if entry['status'] not in ('ok','waiting'):
                    alert('collection_'+entry['status'],symbol=entry['symbol'])
            if report['scoring']['counts']['error']:
                alert('outcome_scoring_errors')
        except (ValueError,KeyError,OSError):
            alert('invalid_run_audit')
    for symbol in ('MSFT','AAPL'):
        paths=sorted((root/'snapshots').glob('*-'+symbol))
        if not paths:
            alert('missing_snapshot',symbol=symbol)
        else:
            try:
                bars,_,_=read_snapshot(paths[-1],now,allow_stale=True)
                if bars.session.iloc[-1]!=expected:
                    alert('stale_snapshot',symbol=symbol)
            except (ValueError,KeyError,OSError):
                alert('invalid_snapshot',symbol=symbol)
        if not (root/'decisions'/f'{symbol}-{expected.date()}.json').exists():
            alert('missing_latest_decision',symbol=symbol)
    all_decisions=[]
    for path in sorted((root/'decisions').glob('*.json')):
        try:
            record,_,_,exit_day=verify_decision(path)
            if utc(record['decision_at'])>now:
                raise ValueError('Future decision')
            all_decisions.append(record)
            outcome=root/'outcomes'/path.name
            if outcome.exists():
                scored=json.loads(outcome.read_text())
                if scored.get('decision_sha256')!=digest(path) or scored.get('status')!='scored':
                    alert('invalid_frozen_outcome',symbol=record['symbol'])
            else:
                cal=calendar(exit_day,exit_day+pd.Timedelta(days=1))
                if now>cal.session_close(exit_day)+pd.Timedelta(minutes=45):
                    alert('overdue_outcome',symbol=record['symbol'])
        except (ValueError,KeyError,OSError):
            alert('invalid_decision_or_outcome')
    for symbol in ('MSFT','AAPL'):
        days=sorted(pd.Timestamp(r['signal_session']) for r in all_decisions if r['symbol']==symbol)
        if days and days[0]<expected:
            cal=calendar(days[0],expected+pd.Timedelta(days=1))
            required=set(cal.sessions_in_range(days[0],expected))
            if required-set(days):
                # Never backfill a historical decision to silence monitoring.
                alert('historical_decision_gaps','warning',symbol)
    archives=sorted(backups.glob('research-*.tar.gz'))
    if not archives:
        alert('missing_backup')
    else:
        archive=archives[-1]
        try:
            sidecar=archive.with_suffix(archive.suffix+'.sha256')
            if sidecar.read_text().strip()!=sha(archive):
                alert('backup_checksum_mismatch')
            # Archive names encode creation time; file copying must not make old
            # backups appear current by changing mtime.
            created=pd.to_datetime(archive.name[len('research-'):-len('.tar.gz')],format='%Y%m%dT%H%M%S%fZ',utc=True)
            with tarfile.open(archive,'r:gz') as tar:
                member=tar.getmember('backup_manifest.json')
                if member.size>16*1024**2:
                    raise ValueError('Oversized backup manifest')
                stored=json.load(tar.extractfile(member))
            if utc(stored['created_at'])!=created:
                alert('backup_creation_time_mismatch')
            if created>now or now-created>pd.Timedelta(hours=max_age_hours):
                alert('stale_or_future_backup')
        except (ValueError,KeyError,OSError,tarfile.TarError):
            alert('invalid_backup')
    return {'as_of':now.isoformat(),'expected_session':str(expected.date()),'alerts':alerts,
            'ok':not any(a['severity']=='error' for a in alerts),
            'notification':'Local report/exit status only; no external messages sent.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data');parser.add_argument('--backups',default='backups')
    args=parser.parse_args();report=inspect(args.root,args.backups)
    print(json.dumps(report,indent=2))
    if not report['ok']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
