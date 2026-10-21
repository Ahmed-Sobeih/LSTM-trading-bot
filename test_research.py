import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research import Config, FEATURES, backtest, make_samples, ridge_predict, walk_forward


class ResearchTests(unittest.TestCase):
    def samples(self, n=200):
        dates = pd.bdate_range('2020-01-01', periods=n)
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(0, .01, (n, len(FEATURES))), index=dates, columns=FEATURES)
        data['target'] = rng.normal(0, .01, n)
        data['execution_date'] = pd.Series(dates, index=dates).shift(-1)
        data['exit_date'] = pd.Series(dates, index=dates).shift(-2)
        return data.dropna()

    def test_execution_alignment_and_causal_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bars.csv'
            p = np.arange(100, 160, dtype=float)
            bars = pd.DataFrame({'Date': pd.bdate_range('2020-01-01', periods=60),
                                 'Open': p, 'High': p+2, 'Low': p-1, 'Close': p+1,
                                 'Adj Close': p+1, 'Volume': np.full(60, 1000)})
            bars.to_csv(path, index=False)
            original = make_samples(path)
            t = 25
            self.assertAlmostEqual(original.loc[bars.Date[t], 'target'], p[t+2]/p[t+1]-1)
            self.assertEqual(original.loc[bars.Date[t], 'execution_date'], bars.Date[t+1])
            bars.loc[40:, ['Open','High','Low','Close','Adj Close']] *= 2
            bars.to_csv(path, index=False)
            changed = make_samples(path)
            pd.testing.assert_frame_equal(original.loc[:bars.Date[39], FEATURES], changed.loc[:bars.Date[39], FEATURES])
            bars.loc[30, 'Volume'] = 0
            bars.to_csv(path, index=False)
            unavailable = make_samples(path)
            self.assertNotIn(bars.Date[28], unavailable.index)
            self.assertNotIn(bars.Date[29], unavailable.index)

    def test_bad_dates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'bad.csv'
            pd.DataFrame({'Date':['2020-01-01']*2, 'Open':[1]*2, 'High':[1]*2,
                          'Low':[1]*2, 'Close':[1]*2, 'Adj Close':[1]*2, 'Volume':[1]*2}).to_csv(p,index=False)
            with self.assertRaises(ValueError):
                make_samples(p)

    def test_entry_exit_costs_and_cash(self):
        s = self.samples(4).iloc[:2].copy()
        s.target = 0.
        cfg = Config(fee_bps=10, half_spread_bps=0, slippage_bps=0)
        ledger = backtest(s, [1, 1], cfg)
        self.assertAlmostEqual(ledger.equity.iloc[-1], .999**2)
        self.assertEqual(ledger.turnover.sum(), 2)
        self.assertEqual(backtest(s, [0, 0], cfg).equity.iloc[-1], 1)

    def test_equity_and_cost_free_hold(self):
        s = self.samples(4).iloc[:2].copy()
        s.target = [.1, -.1]
        cfg = Config(fee_bps=0, half_spread_bps=0, slippage_bps=0)
        ledger = backtest(s, [1, 1], cfg)
        self.assertAlmostEqual(ledger.equity.iloc[-1], .99)
        self.assertAlmostEqual(ledger.drawdown.iloc[-1], -.1)

    def test_risk_stop_applies_after_observed_loss(self):
        s = self.samples(6).iloc[:3].copy()
        s.target = [-.3, .5, .5]
        cfg = Config(fee_bps=0, half_spread_bps=0, slippage_bps=0)
        ledger = backtest(s, [1,1,1], cfg)
        np.testing.assert_array_equal(ledger.position, [1,0,0])
        self.assertAlmostEqual(ledger.equity.iloc[-1], .7)

    def test_exposure_limits(self):
        s = self.samples(4).iloc[:2]
        for position in ([1.1, 0], [-1, 0], [np.nan, 0]):
            with self.assertRaises(ValueError):
                backtest(s, position, Config())

    def test_test_targets_do_not_change_fitted_predictions(self):
        s = self.samples()
        pred = ridge_predict(s.iloc[:100], s.iloc[100:], 100.)
        changed = s.iloc[100:].copy()
        changed.target = 1000
        np.testing.assert_allclose(pred, ridge_predict(s.iloc[:100], changed, 100.), rtol=1e-12, atol=1e-14)

    def test_walkforward_purges_and_future_changes_do_not_affect_prior_folds(self):
        s = self.samples()
        cfg = Config(min_train=40, validation=10, fold=20, holdout_start='2030-01-01')
        first, folds = walk_forward(s, cfg)
        for fold in folds:
            self.assertLessEqual(pd.Timestamp(fold['training_last_label_exit']), pd.Timestamp(fold['first_signal']))
        changed = s.copy()
        changed.loc[first.index[40]:, 'target'] = .5
        second, _ = walk_forward(changed, cfg)
        np.testing.assert_allclose(first.prediction.iloc[:40], second.prediction.iloc[:40], rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(first.desired.iloc[:40], second.desired.iloc[:40])


if __name__ == '__main__':
    unittest.main()
