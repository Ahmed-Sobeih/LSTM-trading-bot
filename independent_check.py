"""Compare an independently sourced raw USD dividend reference with a snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from market_data import read_snapshot, utc


ISSUERS={'MSFT':'www.microsoft.com','AAPL':'investor.apple.com'}


def compare(snapshot,reference,provenance,as_of=None):
    now=pd.Timestamp.now(tz='UTC') if as_of is None else utc(as_of)
    bars,actions,manifest=read_snapshot(snapshot,now,allow_stale=True)
    symbol=manifest['symbol']
    metadata=json.loads(Path(provenance).read_text())
    url=urlparse(metadata['source_url'])
    if url.scheme!='https' or url.hostname!=ISSUERS.get(symbol) or url.username or url.password:
        raise ValueError('Reference must identify the independent official issuer HTTPS source')
    if metadata.get('price_basis')!='raw_usd_per_share_at_ex_date':
        raise ValueError('Reference amounts must use actual USD per share at ex-date, not retrospectively split-adjusted amounts')
    if utc(metadata['retrieved_at'])>now:
        raise ValueError('Reference retrieval time is in the future')
    reference=Path(reference)
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=metadata['reference_sha256']:
        raise ValueError('Independent reference hash mismatch')
    first,last=pd.Timestamp(metadata['covered_from']),pd.Timestamp(metadata['covered_through'])
    if first>last or first<bars.session.min() or last>bars.session.max():
        raise ValueError('Declared reference interval must fall within snapshot coverage')
    rows=pd.read_csv(reference)
    needed={'symbol','ex_date','pay_date','cash_per_share'}
    if not needed.issubset(rows) or rows.empty or (rows.symbol!=symbol).any():
        raise ValueError('Reference columns/symbol/rows invalid')
    rows['ex_date']=pd.to_datetime(rows.ex_date);rows['pay_date']=pd.to_datetime(rows.pay_date)
    if rows.ex_date.isna().any() or rows.pay_date.isna().any() or rows.ex_date.duplicated().any():
        raise ValueError('Reference dates must be nonmissing and ex-dates unique')
    if not np.isfinite(rows.cash_per_share).all() or (rows.cash_per_share<0).any() or (rows.pay_date<rows.ex_date).any():
        raise ValueError('Invalid reference dividend amounts/payment dates')
    if ((rows.ex_date<first)|(rows.ex_date>last)).any():
        raise ValueError('Reference rows fall outside declared complete coverage')
    vendor=actions[(actions.kind=='dividend') & actions.effective_date.between(first,last)].set_index('effective_date')
    reference_rows=rows.set_index('ex_date')
    discrepancies=[]
    for day in sorted(set(vendor.index)|set(reference_rows.index)):
        if day not in vendor.index:
            discrepancies.append({'ex_date':str(day.date()),'issue':'missing_from_alpaca'})
        elif day not in reference_rows.index:
            discrepancies.append({'ex_date':str(day.date()),'issue':'missing_from_independent_reference'})
        else:
            a,b=vendor.loc[day],reference_rows.loc[day]
            if isinstance(a,pd.DataFrame):
                raise ValueError('Multiple vendor dividends on one ex-date need explicit review')
            if not np.isclose(float(a.cash_per_share),float(b.cash_per_share),rtol=0,atol=1e-6):
                discrepancies.append({'ex_date':str(day.date()),'issue':'cash_amount_mismatch'})
            if pd.Timestamp(a.pay_date)!=pd.Timestamp(b.pay_date):
                discrepancies.append({'ex_date':str(day.date()),'issue':'payment_date_mismatch'})
    return {'symbol':symbol,'scope':'Cash dividends only, for explicitly declared reference interval',
            'covered_from':str(first.date()),'covered_through':str(last.date()),'source_url':metadata['source_url'],
            'reference_sha256':metadata['reference_sha256'],'snapshot_sha256':manifest['sha256'],
            'checked_events':len(vendor),'discrepancies':discrepancies,'ok':not discrepancies,
            'limitation':'Source attribution, transcription and completeness are attestations requiring human review. Does not independently verify raw bars, splits or events outside this interval.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True);parser.add_argument('--reference',required=True)
    parser.add_argument('--provenance',required=True)
    args=parser.parse_args();report=compare(args.snapshot,args.reference,args.provenance)
    print(json.dumps(report,indent=2))
    if not report['ok']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
