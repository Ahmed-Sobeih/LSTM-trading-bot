import hashlib
import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from forward_score import digest, verify_decision
from health import inspect
from independent_check import compare
from prospective import record_signal
from state_backup import backup,restore,sha
from test_forward import bars_through,snapshot,CONFIG,ORIGIN_TIME
from market_data import ACTION_COLUMNS


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'data'
        for symbol in ('MSFT','AAPL'):
            bars=bars_through('2024-03-15');bars.symbol=symbol
            source=snapshot(self.root,bars,pd.DataFrame(columns=ACTION_COLUMNS),ORIGIN_TIME,'20240316')
            with patch('prospective.select_model',return_value=(100.,0.)),patch('prospective.ridge_predict',return_value=np.array([.01])):
                record_signal(source,self.root/'decisions',CONFIG,ORIGIN_TIME)
        runs=self.root/'runs';runs.mkdir()
        (runs/'20240316.json').write_text(json.dumps({'run_at':ORIGIN_TIME,'collection':[
            {'symbol':s,'status':'ok'} for s in ('MSFT','AAPL')],'scoring':{'counts':{'error':0}}}))

    def archive(self):
        with patch('state_backup.clock',return_value=pd.Timestamp(ORIGIN_TIME)):
            return backup(self.root,self.base/'backups')

    def test_backup_restore_verifies_bytes_and_works_after_host_move(self):
        archive=self.archive()
        original=digest(self.root/'decisions'/'MSFT-2024-03-15.json')
        restored=restore(archive,self.base/'restored')
        self.assertEqual(digest(restored/'decisions'/'MSFT-2024-03-15.json'),original)
        shutil.rmtree(self.root)  # Only temporary synthetic state.
        record,_,_,_=verify_decision(restored/'decisions'/'MSFT-2024-03-15.json')
        self.assertEqual(record['symbol'],'MSFT')
        self.assertEqual(digest(restored/'decisions'/'MSFT-2024-03-15.json'),original)
        with self.assertRaisesRegex(ValueError,'empty directory'):
            restore(archive,restored)

    def test_tampered_archive_and_symlink_are_rejected(self):
        archive=self.archive()
        with archive.open('ab') as handle:handle.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):
            restore(archive,self.base/'restored')
        (self.root/'decisions'/'link.json').symlink_to(self.root/'decisions'/'MSFT-2024-03-15.json')
        with self.assertRaisesRegex(ValueError,'symbolic links'):
            self.archive()

    def test_restore_rejects_archive_path_escape(self):
        archive=self.base/'hostile.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            info=tarfile.TarInfo('../escaped');info.size=4;tar.addfile(info,io.BytesIO(b'evil'))
        archive.with_suffix('.gz.sha256').write_text(sha(archive))
        with self.assertRaisesRegex(ValueError,'Unsafe archive entry'):
            restore(archive,self.base/'restored')
        self.assertFalse((self.base/'escaped').exists())

    def test_backup_excludes_environment_files_and_rejects_unexpected_state(self):
        (self.root/'.env').write_text('SYNTHETIC_TEST_ONLY=not-a-secret')
        archive=self.archive()
        with tarfile.open(archive) as tar:
            self.assertFalse(any('.env' in x for x in tar.getnames()))
        (self.root/'decisions'/'.env').write_text('SYNTHETIC_TEST_ONLY=not-a-secret')
        with self.assertRaisesRegex(ValueError,'Unexpected state file'):
            self.archive()

    def test_backup_cannot_advance_state_beyond_default_restore_capacity(self):
        with patch('state_backup.MAX_STATE_BYTES',1):
            with self.assertRaisesRegex(ValueError,'restore limits'):
                self.archive()
        self.assertFalse(list((self.base/'backups').glob('*.tar.gz')))

    def test_health_detects_corruption_staleness_and_missing_decisions(self):
        self.archive()
        good=inspect(self.root,self.base/'backups',ORIGIN_TIME)
        self.assertTrue(good['ok'],good)
        stale=inspect(self.root,self.base/'backups','2024-03-20T21:00:00Z')
        codes={a['code'] for a in stale['alerts']}
        self.assertIn('stale_snapshot',codes);self.assertIn('overdue_outcome',codes)
        self.assertIn('stale_or_future_backup',codes)
        path=next((self.root/'snapshots').glob('*-MSFT'))/'bars.csv'
        with path.open('a') as handle:handle.write('\n')
        bad=inspect(self.root,self.base/'backups',ORIGIN_TIME)
        self.assertFalse(bad['ok']);self.assertIn('invalid_snapshot',{a['code'] for a in bad['alerts']})

    def test_independent_reference_mismatches_are_detected(self):
        path=next((self.root/'snapshots').glob('*-MSFT'))
        # Add one clearly synthetic event to both sources; this is not market evidence.
        actions=pd.DataFrame([dict(id='fixture',symbol='MSFT',kind='dividend',effective_date='2024-03-01',
            known_at=ORIGIN_TIME,split_ratio=None,cash_per_share=.5,pay_date='2024-03-08')],columns=ACTION_COLUMNS)
        actions.to_csv(path/'actions.csv',index=False)
        manifest=json.loads((path/'manifest.json').read_text());manifest['sha256']['actions.csv']=sha(path/'actions.csv')
        (path/'manifest.json').write_text(json.dumps(manifest))
        reference=self.base/'reference.csv'
        pd.DataFrame([dict(symbol='MSFT',ex_date='2024-03-01',pay_date='2024-03-08',cash_per_share=.5)]).to_csv(reference,index=False)
        metadata={'source_url':'https://www.microsoft.com/en-us/investor/dividends-and-stock-history.aspx',
            'price_basis':'raw_usd_per_share_at_ex_date','retrieved_at':ORIGIN_TIME,'covered_from':'2024-01-01',
            'covered_through':'2024-03-15','reference_sha256':sha(reference)}
        provenance=self.base/'provenance.json';provenance.write_text(json.dumps(metadata))
        self.assertTrue(compare(path,reference,provenance,ORIGIN_TIME)['ok'])
        rows=pd.read_csv(reference);rows.cash_per_share=.6;rows.to_csv(reference,index=False)
        metadata['reference_sha256']=sha(reference);provenance.write_text(json.dumps(metadata))
        result=compare(path,reference,provenance,ORIGIN_TIME)
        self.assertFalse(result['ok']);self.assertEqual(result['discrepancies'][0]['issue'],'cash_amount_mismatch')


if __name__=='__main__':
    unittest.main()
