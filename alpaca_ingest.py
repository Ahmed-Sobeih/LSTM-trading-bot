"""Read-only Alpaca raw bars/actions ingestion; never creates broker orders."""
import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

from market_data import ACTION_COLUMNS, BAR_COLUMNS, last_completed_session, save_snapshot, utc


def delayed_sip_end(last_session, now):
    """Do not query future/recent minutes even when requesting complete daily bars."""
    following_midnight = (pd.Timestamp(last_session)+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')
    return min(following_midnight, utc(now)-pd.Timedelta(minutes=16))


def normalize_actions(payload, symbol, observed_at, first, last):
    rows = []
    for kind, events in payload.items():
        if not events:
            continue
        if kind == 'cash_mergers':
            # Alpaca's symbol filter also matches acquiring companies. Paying
            # cash for another issuer does not exchange the acquirer's shares.
            if all(e.get('acquirer_symbol') == symbol and e.get('acquiree_symbol') not in (None, symbol) for e in events):
                continue
            raise ValueError('Cash merger affects held shares; redemption accounting is required')
        if kind not in ('forward_splits', 'cash_dividends'):
            raise ValueError(f'Unsupported corporate action category: {kind}; cannot assume correct share accounting')
        for event in events:
            if event['symbol'] != symbol:
                raise ValueError('Corporate action symbol mismatch')
            day = pd.Timestamp(event['ex_date'])
            if not pd.Timestamp(first) <= day <= pd.Timestamp(last):
                continue
            if event.get('due_bill_on_date') or event.get('due_bill_off_date') or event.get('due_bill_redemption_date'):
                raise ValueError('Due-bill corporate actions require specialized accounting')
            row = dict(id=str(event['id']), symbol=symbol, effective_date=day.date().isoformat(),
                       known_at=utc(observed_at).isoformat(), split_ratio=None, cash_per_share=None, pay_date=None)
            if kind == 'forward_splits':
                old, new = float(event['old_rate']), float(event['new_rate'])
                if old <= 0 or new < old:
                    raise ValueError('Invalid forward-split ratio')
                row.update(kind='split', split_ratio=new/old)
            else:
                if event.get('foreign') or event.get('special'):
                    raise ValueError('Special/foreign dividends require currency/tax/event review')
                if not event.get('payable_date'):
                    raise ValueError('Dividend payment date missing; do not invent one')
                row.update(kind='dividend', cash_per_share=float(event['rate']), pay_date=event['payable_date'])
            rows.append(row)
    return pd.DataFrame(rows, columns=ACTION_COLUMNS)


def fetch(symbol, start, root='data/snapshots', feed='sip'):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.requests import CorporateActionsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, secret = os.environ.get('APCA_API_KEY_ID'), os.environ.get('APCA_API_SECRET_KEY')
    if not key or not secret:
        raise ValueError('Set APCA_API_KEY_ID and APCA_API_SECRET_KEY securely in environment settings')
    now = pd.Timestamp.now(tz='UTC')
    # Do not request the last 16 minutes: some SIP plans are delayed.
    last = last_completed_session(now-pd.Timedelta(minutes=16))
    start = pd.Timestamp(start)
    if start.tzinfo is not None or start != start.normalize() or start > last:
        raise ValueError('Start must be a date on/before the latest completed session')
    client = StockHistoricalDataClient(key, secret)
    result = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
        adjustment=Adjustment.RAW, feed=DataFeed(feed), start=start.tz_localize('America/New_York').to_pydatetime(),
        end=delayed_sip_end(last,now).to_pydatetime()))
    if symbol not in result.data or not result.data[symbol]:
        raise ValueError('Alpaca returned no bars; check symbol, dates and feed entitlement')
    actions_client = CorporateActionsClient(key, secret, raw_data=True)
    payload = actions_client.get_corporate_actions(CorporateActionsRequest(
        symbols=[symbol], start=start.date(), end=last.date(), limit=None))
    observed = pd.Timestamp.now(tz='UTC')
    rows = []
    for bar in result.data[symbol]:
        day = pd.Timestamp(bar.timestamp).tz_convert('America/New_York').tz_localize(None).normalize()
        if start <= day <= last:
            rows.append(dict(symbol=symbol, session=day.date().isoformat(), open=bar.open, high=bar.high,
                             low=bar.low, close=bar.close, volume=bar.volume, available_at=observed.isoformat()))
    bars = pd.DataFrame(rows, columns=BAR_COLUMNS).sort_values('session')
    if bars.empty:
        raise ValueError('No completed-session bars returned')
    if pd.Timestamp(bars.session.iloc[0]) > start:
        # A weekend/holiday start is valid; require first expected exchange session.
        from market_data import calendar
        expected = calendar(start, last).sessions[0]
        if pd.Timestamp(bars.session.iloc[0]) != pd.Timestamp(expected).tz_localize(None):
            raise ValueError('Requested starting history is incomplete')
    actions = normalize_actions(payload, symbol, observed, bars.session.iloc[0], bars.session.iloc[-1])
    with tempfile.TemporaryDirectory() as temporary:
        bars_path, actions_path = Path(temporary)/'bars.csv', Path(temporary)/'actions.csv'
        bars.to_csv(bars_path, index=False)
        actions.to_csv(actions_path, index=False)
        return save_snapshot(bars_path, actions_path, root,
                             f'Alpaca {feed} daily RAW bars and corporate-actions API; acquirer-only cash mergers excluded; first observed on download; historical action completeness unverified', 'raw')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol', choices=['MSFT','AAPL'], default='MSFT')
    parser.add_argument('--start', default='2020-09-01', help='Earlier histories can contain missing dividend dates or unsupported due-bill events')
    parser.add_argument('--feed', choices=['sip','iex'], default='sip', help='IEX is a single-venue feed, not consolidated volume')
    parser.add_argument('--root', default='data/snapshots')
    args = parser.parse_args()
    try:
        print(fetch(args.symbol, args.start, args.root, args.feed))
    except ValueError as error:
        parser.exit(1, str(error)+'\n')
    except Exception:
        # Provider exceptions may contain request details: do not dump authentication.
        parser.exit(1, 'Alpaca download failed. Check network access, credentials, feed subscription and corporate-action entitlement. No snapshot was accepted.\n')


if __name__ == '__main__':
    main()
