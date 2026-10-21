import unittest
import numpy as np
import pandas as pd
from forecasting import prepare_data, recursive_forecast, metrics


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"Close": np.arange(200, dtype=float), "Volume": np.arange(200) * 1000},
                                  index=pd.bdate_range("2020-01-01", periods=200))

    def test_future_targets_and_partition_boundaries(self):
        data = prepare_data(self.frame, lookback=7, horizon=5)
        for name, lo, hi in (("train", 7, 160), ("validation", 160, 180), ("test", 180, 200)):
            split = data.splits[name]
            np.testing.assert_allclose(data.inverse_target(split.y[0]), np.arange(lo, lo+5), atol=1e-5)
            self.assertEqual(split.dates[0, 0], self.frame.index[lo])
            self.assertEqual(split.dates[-1, -1], self.frame.index[hi-1])
            self.assertEqual(split.x.shape[1:], (7, 1))
            self.assertEqual(split.baseline[0, 0], lo-1)
        self.assertLess(data.splits['train'].dates.max(), data.splits['validation'].dates.min())
        self.assertLess(data.splits['validation'].dates.max(), data.splits['test'].dates.min())

    def test_scaler_never_reads_validation_or_test(self):
        original = prepare_data(self.frame, features=("Close", "Volume"))
        changed = self.frame.copy()
        changed.iloc[160:] = 1e9
        after = prepare_data(changed, features=("Close", "Volume"))
        np.testing.assert_array_equal(original.mean, after.mean)
        np.testing.assert_array_equal(original.scale, after.scale)
        self.assertEqual(original.target_mean, after.target_mean)
        np.testing.assert_array_equal(original.splits['train'].x, after.splits['train'].x)

    def test_recursive_window_advances(self):
        seen = []
        def predict(x):
            seen.append(x.copy())
            return x[:, -1, :] + 1
        initial = np.array([[1.], [2.], [3.]])
        np.testing.assert_array_equal(recursive_forecast(predict, initial, 3), [4, 5, 6])
        np.testing.assert_array_equal(seen[1][0, :, 0], [2, 3, 4])
        np.testing.assert_array_equal(initial[:, 0], [1, 2, 3])

    def test_constant_features_are_safe(self):
        self.frame['Volume'] = 1
        data = prepare_data(self.frame, features=("Close", "Volume"))
        self.assertTrue(np.isfinite(data.splits['train'].x).all())

    def test_reject_insufficient_history_and_mismatched_metrics(self):
        with self.assertRaises(ValueError):
            prepare_data(self.frame, horizon=30)
        with self.assertRaises(ValueError):
            metrics(np.zeros((2, 1)), np.zeros(2))


if __name__ == '__main__':
    unittest.main()
