import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from daily import cycle
from forward_score import digest, score_all, score_decision
from market_data import ACTION_COLUMNS, calendar
from prospective import record_signal
from research import Config


CONFIG = Config(min_train=20, validation=10, fold=10)
ORIGIN_TIME = '2024-03-16T00:00:00Z'
OUTCOME_TIME = '2024-03-19T21:00:00Z'


def bars_through(end):
    cal = calendar('2023-06-01',end)
    return pd.DataFrame({'symbol':'TEST','session':cal.sessions,'open':100.,'high':100.,'low':100.,
                         'close':100.,'volume':1000.,
                         'available_at':[cal.session_close(d)+pd.Timedelta(minutes=1) for d in cal.sessions]})


def snapshot(root, bars, actions, observed, suffix):
    path = root/'snapshots'/f'{suffix}-TEST'
    path.mkdir(parents=True)
    bars.to_csv(path/'bars.csv',index=False)
    actions.to_csv(path/'actions.csv',index=False)
    manifest = {'schema_version':1,'observed_at':observed,'symbol':'TEST','source':'Alpaca sip daily RAW synthetic test only',
                'price_basis':'raw','sha256':{name:digest(path/name) for name in ['bars.csv','actions.csv']}}
    (path/'manifest.json').write_text(json.dumps(manifest))
    return path


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        actions = pd.DataFrame(columns=ACTION_COLUMNS)
        self.origin = snapshot(self.root,bars_through('2024-03-15'),actions,ORIGIN_TIME,'20240316')
        with patch('prospective.select_model',return_value=(100.,0.)), patch('prospective.ridge_predict',return_value=np.array([.02])):
            self.decision = record_signal(self.origin,self.root/'decisions',CONFIG,ORIGIN_TIME)
        later = bars_through('2024-03-19')
        later.loc[later.session==pd.Timestamp('2024-03-19'),['open','high','low','close']] = 102.
        self.outcome = snapshot(self.root,later,actions,OUTCOME_TIME,'20240319')

    def test_waits_for_exit_day_and_does_not_write_fake_outcome(self):
        result = score_decision(self.decision,self.outcome,self.root/'outcomes','2024-03-18T21:00:00Z')
        self.assertEqual(result['status'],'pending')
        self.assertEqual(result['exit_session'],'2024-03-19')
        self.assertFalse((self.root/'outcomes').exists())

    def test_scores_correct_future_open_and_costs_without_refitting(self):
        original = digest(self.decision)
        with patch('prospective.ridge_predict',side_effect=AssertionError('Scoring must not fit/predict')):
            result = score_decision(self.decision,self.outcome,self.root/'outcomes',OUTCOME_TIME)
        self.assertEqual(result['status'],'scored')
        self.assertAlmostEqual(result['actual_total_return'],.02)
        self.assertAlmostEqual(result['absolute_error'],0.)
        expected = (1.02*(1-.0005)/(1+.0005))-1
        self.assertAlmostEqual(result['isolated_round_trip_net_return_proxy'],expected)
        self.assertEqual(digest(self.decision),original)

    def test_first_score_is_preserved_when_vendor_history_changes(self):
        result = score_decision(self.decision,self.outcome,self.root/'outcomes',OUTCOME_TIME)
        output = self.root/'outcomes'/'TEST-2024-03-15.json'
        before = output.read_bytes()
        bars = bars_through('2024-03-19')
        bars.loc[bars.session==pd.Timestamp('2024-03-19'),['open','high','low','close']] = 105.
        revised = snapshot(self.root,bars,pd.DataFrame(columns=ACTION_COLUMNS),OUTCOME_TIME,'20240320-revision')
        again = score_decision(self.decision,revised,self.root/'outcomes',OUTCOME_TIME)
        self.assertEqual(result,again)
        self.assertEqual(output.read_bytes(),before)

    def test_changed_decision_is_detected_and_cached_future_score_is_not_leaked(self):
        score_decision(self.decision,self.outcome,self.root/'outcomes',OUTCOME_TIME)
        early = score_decision(self.decision,self.outcome,self.root/'outcomes','2024-03-18T21:00:00Z')
        self.assertEqual(early['status'],'pending')
        record=json.loads(self.decision.read_text())
        record['prediction']=.03
        self.decision.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'changed after scoring'):
            score_decision(self.decision,self.outcome,self.root/'outcomes',OUTCOME_TIME)

    def test_split_and_ex_date_dividend_are_in_target(self):
        bars = bars_through('2024-03-19')
        bars.loc[bars.session==pd.Timestamp('2024-03-19'),['open','high','low','close']] = 51.
        actions = pd.DataFrame([
            dict(id='split',symbol='TEST',kind='split',effective_date='2024-03-19',known_at=OUTCOME_TIME,split_ratio=2.,cash_per_share=None,pay_date=None),
            dict(id='div',symbol='TEST',kind='dividend',effective_date='2024-03-19',known_at=OUTCOME_TIME,split_ratio=None,cash_per_share=.5,pay_date='2024-03-25')],columns=ACTION_COLUMNS)
        path=snapshot(self.root,bars,actions,OUTCOME_TIME,'20240320-events')
        result=score_decision(self.decision,path,self.root/'outcomes',OUTCOME_TIME)
        self.assertAlmostEqual(result['actual_total_return'],.03)
        self.assertEqual(result['split_ratio'],2.)

    def test_changed_source_and_corrupt_original_are_rejected(self):
        manifest_path=self.outcome/'manifest.json'
        manifest=json.loads(manifest_path.read_text());manifest['source']='Different feed'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'source differs'):
            score_decision(self.decision,self.outcome,self.root/'outcomes',OUTCOME_TIME)
        with (self.origin/'bars.csv').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError,'integrity'):
            score_decision(self.decision,self.outcome,self.root/'outcomes',OUTCOME_TIME)

    def test_summary_reports_pending_and_scored_separately(self):
        early=score_all(self.root/'decisions',self.root/'snapshots',self.root/'outcomes',ORIGIN_TIME)
        self.assertEqual(early['counts'],{'scored':0,'pending':1,'error':0})
        late=score_all(self.root/'decisions',self.root/'snapshots',self.root/'outcomes',OUTCOME_TIME)
        self.assertEqual(late['counts'],{'scored':1,'pending':0,'error':0})
        self.assertAlmostEqual(late['cohorts'][0]['zero_forecast_mae'],.02)

    def test_repeated_daily_cycles_reuse_snapshot_and_decision(self):
        before=digest(self.decision)
        with patch('daily.fetch',side_effect=AssertionError('Must reuse accepted snapshot')), patch('daily.recent_config',return_value=CONFIG):
            first=cycle(self.root,('TEST',),'2023-06-01',as_of=ORIGIN_TIME)
            second=cycle(self.root,('TEST',),'2023-06-01',as_of=ORIGIN_TIME)
        self.assertEqual(first['collection'][0]['decision'],'already_frozen')
        self.assertEqual(second['scoring']['counts']['pending'],1)
        self.assertEqual(digest(self.decision),before)
        self.assertEqual(len(list((self.root/'runs').glob('*.json'))),2)

    def test_concurrent_daily_run_is_blocked(self):
        with (self.root/'.daily.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError,'already running'):
                cycle(self.root,('TEST',),'2023-06-01',as_of=ORIGIN_TIME)

    def test_late_first_collection_never_backdates_a_decision(self):
        self.decision.unlink()  # Only this test's temporary synthetic record.
        with patch('daily.fetch',side_effect=AssertionError('Must reuse snapshot')), patch('daily.recent_config',return_value=CONFIG):
            report=cycle(self.root,('TEST',),'2023-06-01',as_of='2024-03-18T15:00:00Z')
        self.assertEqual(report['collection'][0]['status'],'missed')
        self.assertFalse(self.decision.exists())


if __name__=='__main__':
    unittest.main()
