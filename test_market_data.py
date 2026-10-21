import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from accounting import raw_samples, simulate
from alpaca_ingest import normalize_actions
from market_data import ACTION_COLUMNS, calendar, last_completed_session, read_snapshot, save_snapshot, validate
from prospective import record_signal
from research import Config
from raw_research import run as run_raw_research


OBSERVED = '2024-03-16T00:00:00Z'
FREE = Config(min_train=20, validation=10, fold=10, fee_bps=0, half_spread_bps=0, slippage_bps=0)


def fixture():
    cal = calendar('2023-06-01', '2024-03-15')
    days = cal.sessions
    bars = pd.DataFrame({'symbol': 'TEST', 'session': days, 'open': 100., 'high': 100.,
                         'low': 100., 'close': 100., 'volume': 1000.,
                         'available_at': [cal.session_close(d)+pd.Timedelta(minutes=1) for d in days]})
    return bars, pd.DataFrame(columns=ACTION_COLUMNS)


class MarketDataTests(unittest.TestCase):
    def test_exchange_holidays_and_stale_data(self):
        self.assertEqual(last_completed_session('2024-03-30T00:00:00Z'), pd.Timestamp('2024-03-28'))
        bars, actions = fixture()
        validate(bars, actions, OBSERVED)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            validate(bars.iloc[:-1], actions, OBSERVED)
        with self.assertRaisesRegex(ValueError, 'Missing or extra'):
            validate(bars.drop(bars.index[40]), actions, OBSERVED)

    def test_bar_not_available_before_close(self):
        bars, actions = fixture()
        bars.loc[bars.index[-1], 'available_at'] = '2024-03-15T15:00:00Z'
        with self.assertRaisesRegex(ValueError, 'after session close'):
            validate(bars, actions, OBSERVED)

    def test_snapshot_hash_and_no_backdated_availability(self):
        bars, actions = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bars.to_csv(root/'bars.csv', index=False)
            actions.to_csv(root/'actions.csv', index=False)
            path = save_snapshot(root/'bars.csv', root/'actions.csv', root/'snapshots', 'synthetic test fixture', 'raw', True)
            with self.assertRaisesRegex(ValueError, 'not observed'):
                read_snapshot(path, OBSERVED, allow_stale=True)
            read_snapshot(path, pd.Timestamp.now(tz='UTC'), allow_stale=True)
            with (path/'bars.csv').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                read_snapshot(path, pd.Timestamp.now(tz='UTC'), allow_stale=True)

    def test_raw_research_end_to_end_on_explicit_synthetic_snapshot(self):
        bars, actions = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bars.to_csv(root/'bars.csv', index=False)
            actions.to_csv(root/'actions.csv', index=False)
            snapshot = save_snapshot(root/'bars.csv', root/'actions.csv', root/'snapshots', 'SYNTHETIC TEST ONLY', 'raw', True)
            cfg = Config(min_train=20, validation=10, fold=40, holdout_start='2030-01-01')
            report = run_raw_research(snapshot, root/'reports', cfg)
            self.assertEqual(report['results']['cash']['total_return'], 0.)
            self.assertEqual(report['results']['ridge']['total_return'], 0.)
            self.assertLess(report['results']['buy_hold']['total_return'], 0.)
            self.assertTrue((root/'reports'/'predictions.csv').exists())
            self.assertTrue((root/'reports'/'buy_hold_fills.csv').exists())

    def test_alpaca_fetch_uses_raw_bars_and_full_action_pagination(self):
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca_ingest import fetch
        from types import SimpleNamespace
        def bars_response(request):
            self.assertEqual(request.adjustment, Adjustment.RAW)
            self.assertEqual(request.feed, DataFeed.SIP)
            return SimpleNamespace(data={'MSFT':[SimpleNamespace(timestamp=pd.Timestamp('2024-03-15T04:00:00Z'),
                open=100.,high=101.,low=99.,close=100.,volume=1000)]})
        def actions_response(request):
            self.assertIsNone(request.limit)
            self.assertIsNone(request.types)  # Unmodeled types must be detected, not filtered away.
            return {}
        with patch.dict('os.environ',{'APCA_API_KEY_ID':'test-only-key','APCA_API_SECRET_KEY':'test-only-secret'}), \
             patch('alpaca.data.historical.StockHistoricalDataClient') as stock, \
             patch('alpaca.data.historical.corporate_actions.CorporateActionsClient') as events, \
             patch('alpaca_ingest.last_completed_session',return_value=pd.Timestamp('2024-03-15')), \
             patch('alpaca_ingest.save_snapshot',return_value=Path('synthetic-test-snapshot')) as save:
            stock.return_value.get_stock_bars.side_effect=bars_response
            events.return_value.get_corporate_actions.side_effect=actions_response
            self.assertEqual(fetch('MSFT','2024-03-15'),Path('synthetic-test-snapshot'))
            save.assert_called_once()

    def test_split_preserves_wealth_and_return_features(self):
        bars, actions = fixture()
        day = bars.session.iloc[30]
        bars.loc[30:, ['open','high','low','close']] = 50.
        bars.loc[30:, 'volume'] = 2000.
        actions = pd.DataFrame([dict(id='split', symbol='TEST', kind='split', effective_date=day,
                  known_at=OBSERVED, split_ratio=2., cash_per_share=None, pay_date=None)], columns=ACTION_COLUMNS)
        signals = pd.Series([1.], index=[bars.session.iloc[0]])
        ledger, fills = simulate(bars, actions, signals, FREE, OBSERVED, liquidate=False)
        self.assertEqual(ledger.shares.iloc[-1], 200.)
        np.testing.assert_allclose(ledger.equity, 10000.)
        samples = raw_samples(*validate(bars, actions, OBSERVED))
        self.assertAlmostEqual(samples.loc[day, 'return_1'], 0.)
        self.assertAlmostEqual(samples.loc[day, 'volume_change'], 0.)
        self.assertAlmostEqual(samples.loc[bars.session.iloc[28], 'target'], 0.)

    def test_dividend_belongs_to_overnight_holder_and_survives_sale(self):
        bars, _ = fixture()
        ex_date, pay_date = bars.session.iloc[30], bars.session.iloc[35]
        bars.loc[30:, ['open','high','low','close']] = 99.
        actions = pd.DataFrame([dict(id='div',symbol='TEST',kind='dividend',effective_date=ex_date,
                  known_at=OBSERVED,split_ratio=None,cash_per_share=1.,pay_date=pay_date)], columns=ACTION_COLUMNS)
        signals = pd.Series([1., 0.], index=[bars.session.iloc[0], bars.session.iloc[29]])
        ledger, _ = simulate(bars, actions, signals, FREE, OBSERVED, liquidate=False)
        indexed = ledger.set_index('session')
        self.assertEqual(indexed.loc[ex_date,'shares'], 0.)
        self.assertEqual(indexed.loc[ex_date,'receivable'], 100.)
        self.assertEqual(indexed.loc[ex_date,'cash'], 9900.)
        self.assertEqual(indexed.loc[pay_date,'cash'], 10000.)
        self.assertEqual(indexed.loc[pay_date,'receivable'], 0.)
        entrant = pd.Series([1.],index=[bars.session.iloc[29]])
        other, _ = simulate(bars, actions, entrant, FREE, OBSERVED, liquidate=False)
        self.assertEqual(other.dividend_accrued.sum(), 0.)

    def test_unpaid_dividend_is_not_spendable_cash(self):
        bars, _ = fixture()
        day = bars.session.iloc[30]
        bars.loc[30:, ['open','high','low','close']] = 99.
        actions = pd.DataFrame([dict(id='div',symbol='TEST',kind='dividend',effective_date=day,
                  known_at=OBSERVED,split_ratio=None,cash_per_share=1.,pay_date='2024-04-01')], columns=ACTION_COLUMNS)
        signals = pd.Series(1., index=bars.session.iloc[:40])
        ledger, _ = simulate(bars, actions, signals, FREE, OBSERVED, liquidate=False)
        indexed = ledger.set_index('session')
        self.assertEqual(indexed.loc[day,'shares'], 100.)
        self.assertEqual(indexed.loc[day,'cash'], 0.)
        self.assertEqual(ledger.receivable.iloc[-1], 100.)

    def test_cash_solvency_and_final_costs(self):
        bars, actions = fixture()
        cfg = Config(fee_bps=10,half_spread_bps=0,slippage_bps=0)
        ledger, fills = simulate(bars, actions, pd.Series([1.],index=[bars.session.iloc[0]]), cfg, OBSERVED)
        self.assertEqual(fills.quantity.tolist(), [99., -99.])
        self.assertAlmostEqual(ledger.equity.iloc[-1], 9980.2)
        self.assertTrue((ledger.cash >= 0).all())

    def test_alpaca_conversion_and_unsupported_events(self):
        payload = {'forward_splits':[dict(id='a',symbol='TEST',ex_date='2024-03-01',new_rate=4,old_rate=1)],
                   'cash_dividends':[dict(id='b',symbol='TEST',ex_date='2024-03-04',rate=.25,payable_date='2024-03-20')]}
        result = normalize_actions(payload,'TEST',OBSERVED,'2024-01-01','2024-03-15')
        self.assertEqual(result.split_ratio.iloc[0], 4.)
        self.assertEqual(result.cash_per_share.iloc[1], .25)
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            normalize_actions({'spin_offs':[{}]},'TEST',OBSERVED,'2024-01-01','2024-03-15')

    def test_freeze_prospective_signal_once_and_reject_late_execution(self):
        bars, actions = validate(*fixture(), OBSERVED)
        manifest = {'observed_at':OBSERVED,'symbol':'TEST','sha256':{'bars.csv':'synthetic','actions.csv':'synthetic'}}
        with tempfile.TemporaryDirectory() as root, patch('prospective.read_snapshot',return_value=(bars,actions,manifest)):
            p = record_signal('synthetic-test-snapshot',root,FREE,OBSERVED)
            record = json.loads(p.read_text())
            self.assertLessEqual(pd.Timestamp(record['training_last_label_exit']), bars.session.iloc[-1])
            self.assertEqual(record['desired_exposure'],0.)
            with self.assertRaises(FileExistsError):
                record_signal('synthetic-test-snapshot',root,FREE,OBSERVED)
            with self.assertRaisesRegex(ValueError, 'open has passed'):
                record_signal('synthetic-test-snapshot',root,FREE,'2024-03-18T15:00:00Z')


if __name__ == '__main__':
    unittest.main()
