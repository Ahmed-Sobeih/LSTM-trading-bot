import unittest
import numpy as np
import pandas as pd
from candidate_research import sequences
from research import FEATURES


class CandidateTests(unittest.TestCase):
    def test_sequences_never_use_features_after_signal_date(self):
        dates=pd.bdate_range('2020-01-01',periods=100)
        frame=pd.DataFrame(np.arange(600).reshape(100,6),index=dates,columns=FEATURES)
        rows=frame.iloc[30:40]
        original=sequences(frame,rows)
        changed=frame.copy();changed.iloc[40:]=1e9
        np.testing.assert_array_equal(original,sequences(changed,rows))
        np.testing.assert_array_equal(original[0,-1],frame.iloc[30])
        np.testing.assert_array_equal(original[0,0],frame.iloc[11])
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            sequences(frame,frame.iloc[:5])


if __name__=='__main__':
    unittest.main()
