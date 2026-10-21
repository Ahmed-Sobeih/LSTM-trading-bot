"""Chronological, train-only preprocessing and independent LSTM experiments."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(path, features=("Close",)):
    frame = pd.read_csv(Path(path))
    required = {"Date", "Close", *features}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    dates = pd.to_datetime(frame.pop("Date"), errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be nonmissing, unique, and ascending")
    frame.index = dates
    values = frame[list(dict.fromkeys([*features, "Close"]))].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Price/features must be finite numeric values")
    return frame


@dataclass
class WindowSplit:
    x: np.ndarray
    y: np.ndarray
    baseline: np.ndarray
    dates: np.ndarray


@dataclass
class PreparedData:
    splits: dict
    mean: np.ndarray
    scale: np.ndarray
    target_mean: float
    target_scale: float
    features: tuple
    train_end: int
    val_end: int

    def inverse_target(self, values):
        return np.asarray(values) * self.target_scale + self.target_mean


def prepare_data(frame, features=("Close",), lookback=3, horizon=1):
    """Inputs end at t-1; labels are t..t+horizon-1, in trading rows.

    Fit scales on the first 80% of rows only. All labels in a window must
    belong to one partition. Validation/test inputs may use earlier history.
    """
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    features = tuple(features)
    raw = frame[list(features)].to_numpy(dtype=np.float32)
    close = frame.Close.to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all() or not np.isfinite(close).all():
        raise ValueError("Nonfinite inputs")
    train_end, val_end = int(len(frame) * .8), int(len(frame) * .9)
    mean, scale = raw[:train_end].mean(axis=0), raw[:train_end].std(axis=0)
    scale = np.where(scale > 0, scale, 1)
    target_mean, target_scale = float(close[:train_end].mean()), float(close[:train_end].std())
    target_scale = target_scale if target_scale > 0 else 1.
    scaled = (raw - mean) / scale
    splits = {}
    for name, lower, upper in (("train", 0, train_end), ("validation", train_end, val_end), ("test", val_end, len(frame))):
        starts = range(max(lookback, lower), upper - horizon + 1)
        starts = list(starts)
        if not starts:
            raise ValueError(f"Not enough rows for {name} with this lookback/horizon")
        splits[name] = WindowSplit(
            np.stack([scaled[t-lookback:t] for t in starts]),
            np.stack([(close[t:t+horizon] - target_mean) / target_scale for t in starts]),
            np.stack([np.repeat(close[t-1], horizon) for t in starts]),
            np.stack([frame.index.to_numpy()[t:t+horizon] for t in starts]),
        )
    return PreparedData(splits, mean, scale, target_mean, target_scale, features, train_end, val_end)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and match target shape")
    error = actual - predicted
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.square(error).mean()))}


def recursive_forecast(predict, initial_window, steps):
    """Roll a scaled close-only window forward without using future prices."""
    window = np.array(initial_window, dtype=np.float32, copy=True)
    if window.ndim != 2 or window.shape[1] != 1 or steps < 1:
        raise ValueError("Use a close-only (lookback, 1) window and positive steps")
    result = []
    for _ in range(steps):
        output = np.asarray(predict(window[None, ...]))
        if output.size != 1 or not np.isfinite(output).all():
            raise ValueError("Recursive predictor must return one finite value")
        value = float(output.item())
        result.append(value)
        window = np.concatenate([window[1:], [[value]]]).astype(np.float32)
    return np.asarray(result)


def run_experiment(data, epochs=30, seed=42, units=32, batch_size=32):
    """Create a fresh model; use validation for stopping, test only for reporting."""
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    train = data.splits["train"]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=train.x.shape[1:]),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(train.y.shape[1]),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss="mse")

    def dataset(split):
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return tf.data.Dataset.from_tensor_slices((split.x, split.y)).batch(batch_size).with_options(options)

    history = model.fit(dataset(train), validation_data=dataset(data.splits["validation"]),
                        epochs=epochs, shuffle=False, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    rows, predictions = [], {}
    for name in ("validation", "test"):
        split = data.splits[name]
        predicted = data.inverse_target(model.predict(dataset(split), verbose=0))
        actual = data.inverse_target(split.y)
        predictions[name] = predicted
        for method, values in (("LSTM", predicted), ("Persistence", split.baseline)):
            rows.append({"split": name, "method": method, **metrics(actual, values)})
    return model, history, pd.DataFrame(rows), predictions
