"""Replay saved candidate signals under costs; no retraining or reselection."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from accounting import simulate
from market_data import read_snapshot
from research import Config


def run(snapshot,directory):
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text())
    protocol=report['protocol'];config=Config(**protocol['config'])
    bars,actions,manifest=read_snapshot(snapshot,pd.Timestamp.now(tz='UTC'),allow_stale=True)
    if manifest['sha256']!=protocol['snapshot_sha256']:
        raise ValueError('Replay snapshot differs from registered experiment')
    files=sorted(directory.glob('*_predictions.csv'))
    if not files:
        raise ValueError('No saved candidate signals')
    cases={}
    for path in files:
        frame=pd.read_csv(path,index_col=0,parse_dates=True)
        frame['exit_date']=pd.to_datetime(frame.exit_date)
        if (frame.exit_date>=pd.Timestamp(config.holdout_start)).any():
            raise ValueError('Replay would consume performance holdout')
        cases[path.name.removesuffix('_predictions.csv')]=(frame,frame.desired)
    example=next(iter(cases.values()))[0]
    cases['buy_hold']=(example,pd.Series(config.max_exposure,index=example.index))
    cases['cash']=(example,pd.Series(0.,index=example.index))
    cases['momentum_20']=(example,(example.return_20>0)*config.max_exposure)
    rows=[]
    for multiplier in (0.,1.,2.,4.):
        cfg=Config(**{**asdict(config),'fee_bps':config.fee_bps*multiplier,
                      'half_spread_bps':config.half_spread_bps*multiplier,'slippage_bps':config.slippage_bps*multiplier})
        for name,(frame,desired) in cases.items():
            subset=bars[(bars.session>=frame.index.min()) & (bars.session<=frame.exit_date.iloc[-1])]
            events=actions[actions.effective_date.isin(subset.session)]
            ledger,_=simulate(subset,events,desired,cfg,manifest['observed_at'],stop=name not in ('buy_hold','cash','momentum_20'))
            rows.append({'model':name,'cost_multiplier':multiplier,'total_return':float(ledger.equity.iloc[-1]/10000-1),
                         'max_drawdown':float(ledger.drawdown.min()),'cost_dollars':float(ledger.cost.sum()),
                         'stop_triggered':bool(ledger.halted.any())})
    provenance={'schema_version':1,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in files+[directory/'protocol.json',directory/'report.json']},
                'replay_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'warning':'Exploratory robustness replay. Desired signals frozen; risk stops respond to costs. No retraining, threshold reselection, holdout consumption or forward changes.'}
    with (directory/'cost_sensitivity.csv').open('x') as handle:pd.DataFrame(rows).to_csv(handle,index=False)
    with (directory/'cost_provenance.json').open('x') as handle:json.dump(provenance,handle,indent=2)
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True);parser.add_argument('--directory',required=True)
    args=parser.parse_args();print(run(args.snapshot,args.directory).to_string(index=False))


if __name__=='__main__':
    main()
