"""One operational run: collect/score, back up state, then evaluate health."""
import argparse
import json

from daily import cycle
from health import inspect
from state_backup import backup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data');parser.add_argument('--backups',default='backups')
    args=parser.parse_args()
    failures=[];collection=None;archive=None
    try:
        collection=cycle(args.root)
    except Exception as error:
        failures.append({'stage':'collection','error_type':type(error).__name__})
    try:
        archive=str(backup(args.root,args.backups))
    except Exception as error:
        failures.append({'stage':'backup','error_type':type(error).__name__})
    try:
        health=inspect(args.root,args.backups)
    except Exception as error:
        health={'ok':False,'alerts':[{'code':'health_check_failed','severity':'error','error_type':type(error).__name__}]}
    report={'failures':failures,'backup':archive,'health':health,
            'collection':collection['collection'] if collection else [],
            'scoring_counts':collection['scoring']['counts'] if collection else {},'no_orders':True}
    print(json.dumps(report,indent=2))
    if failures or not health['ok']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
