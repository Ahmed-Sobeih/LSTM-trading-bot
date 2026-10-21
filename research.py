"""Auditable daily long/cash research. No broker connectivity or live orders."""
import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ['return_1', 'return_5', 'return_20', 'volatility_20', 'range', 'volume_change']


@dataclass(frozen=True)
class Config:
    min_train: int = 1260
    validation: int = 252
    fold: int = 252
    fee_bps: float = 1.
    half_spread_bps: float = 2.
    slippage_bps: float = 2.
    max_exposure: float = 1.
    drawdown_stop: float = .20
    holdout_start: str = '2022-01-01'

    def __post_init__(self):
        if min(self.min_train, self.validation, self.fold) < 1:
            raise ValueError('Training, validation and fold lengths must be positive')
        costs = np.array([self.fee_bps, self.half_spread_bps, self.slippage_bps])
        if not np.isfinite(costs).all() or (costs < 0).any() or costs.sum() >= 10000:
            raise ValueError('Costs must be finite, nonnegative, and below 100% per side')
        if not 0 < self.max_exposure <= 1 or not 0 < self.drawdown_stop < 1:
            raise ValueError('Exposure must be in (0,1]; drawdown stop in (0,1)')
        if pd.isna(pd.Timestamp(self.holdout_start)):
            raise ValueError('Invalid holdout date')


def make_samples(path):
    raw = pd.read_csv(path)
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if not set(required).issubset(raw):
        raise ValueError(f'Required columns: {required}')
    dates = pd.to_datetime(raw.Date, errors='raise')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be ascending, unique and nonmissing')
    raw = raw.set_index(pd.DatetimeIndex(dates, name='signal_date'))
    numeric = raw[required[1:]].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric[:, :5] <= 0).any() or (raw.Volume < 0).any():
        raise ValueError('Prices must be finite and positive; volume finite and nonnegative')
    if ((raw.High < raw[['Open', 'Close', 'Low']].max(axis=1)) |
            (raw.Low > raw[['Open', 'Close', 'High']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC bars')
    # Retrospective adjusted-price return proxy, not an executable share ledger.
    adjusted_open = raw.Open * raw['Adj Close'] / raw.Close
    close = raw['Adj Close']
    ret = close.pct_change(fill_method=None)
    samples = pd.DataFrame(index=raw.index)
    samples['return_1'] = ret
    samples['return_5'] = close.pct_change(5, fill_method=None)
    samples['return_20'] = close.pct_change(20, fill_method=None)
    samples['volatility_20'] = ret.rolling(20).std()
    samples['range'] = (raw.High - raw.Low) / raw.Close
    samples['volume_change'] = raw.Volume / raw.Volume.rolling(20).mean() - 1
    samples['target'] = adjusted_open.shift(-2) / adjusted_open.shift(-1) - 1
    date_series = pd.Series(raw.index, index=raw.index)
    samples['execution_date'] = date_series.shift(-1)
    samples['exit_date'] = date_series.shift(-2)
    # Never assume an entry/exit fill on a vendor-reported zero-volume session.
    samples = samples[(raw.Volume.shift(-1) > 0) & (raw.Volume.shift(-2) > 0)]
    samples = samples.dropna()
    if not np.isfinite(samples[FEATURES + ['target']].to_numpy()).all():
        raise ValueError('Nonfinite features or targets')
    return samples


def ridge_predict(train, test, alpha):
    """Scale and fit solely on this training partition; unpenalized intercept."""
    x, z = train[FEATURES].to_numpy(), test[FEATURES].to_numpy()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    x = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    z = np.column_stack([np.ones(len(z)), (z-mean)/scale])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ train.target.to_numpy())
    return z @ beta


def backtest(samples, desired, config, stop=True):
    """Mark adjusted-return exposures daily; costs on entry/rebalance/exit.

    A drawdown breach at an observed equity mark locks the strategy in cash
    from the next open. It cannot prevent gaps or guarantee a maximum loss.
    """
    if len(samples) == 0:
        raise ValueError('No evaluation rows')
    desired = np.asarray(desired, dtype=float)
    if desired.shape != (len(samples),) or not np.isfinite(desired).all() or ((desired < 0) | (desired > config.max_exposure)).any():
        raise ValueError('Invalid long/cash exposure')
    cost_rate = (config.fee_bps + config.half_spread_bps + config.slippage_bps) / 10000
    equity, peak, previous, halted = 1., 1., 0., False
    rows = []
    for i, (_, sample) in enumerate(samples.iterrows()):
        position = 0. if halted else desired[i]
        turnover = abs(position - previous)
        # Liquidate at the last exit open, including its transaction costs.
        exit_turnover = position if i == len(samples)-1 else 0.
        gross = position * sample.target
        net = (1-turnover*cost_rate) * (1+gross) * (1-exit_turnover*cost_rate)-1
        equity *= 1+net
        peak = max(peak, equity)
        drawdown = equity/peak-1
        if stop and drawdown <= -config.drawdown_stop:
            halted = True
        rows.append(dict(signal_date=sample.name, execution_date=sample.execution_date,
                         exit_date=sample.exit_date, position=position, turnover=turnover+exit_turnover,
                         gross_return=gross, net_return=net, equity=equity, drawdown=drawdown, halted=halted))
        previous = position
    return pd.DataFrame(rows)


def summarize(ledger):
    ret = ledger.net_return
    years = len(ret)/252
    volatility = float(ret.std(ddof=0)*np.sqrt(252))
    return {'total_return': float(ledger.equity.iloc[-1]-1),
            'cagr': float(ledger.equity.iloc[-1]**(1/years)-1),
            'annual_volatility': volatility,
            'sharpe_zero_cash_rate': float(ret.mean()*252/volatility) if volatility else None,
            'max_drawdown': float(ledger.drawdown.min()),
            'mean_exposure': float(ledger.position.mean()),
            'one_way_turnover': float(ledger.turnover.sum()),
            'position_changes': int((ledger.turnover > 0).sum()),
            'sessions': len(ledger), 'risk_stop_triggered': bool(ledger.halted.any())}


def select_model(history, config):
    validation = history.iloc[-config.validation:]
    train = history.iloc[:-config.validation]
    # Label exits must be observable at first validation signal close.
    train = train[train.exit_date <= validation.index[0]]
    if len(train) < config.min_train:
        raise ValueError('Insufficient training history after label purge')
    choices = []
    for alpha in (1., 100., 10000.):
        prediction = ridge_predict(train, validation, alpha)
        # Include an explicit no-trade candidate. Thresholds are return units.
        for threshold in (0., .001, .003, float('inf')):
            desired = (prediction > threshold) * config.max_exposure
            ledger = backtest(validation, desired, config)
            choices.append((float(ledger.equity.iloc[-1]), alpha, threshold))
    # Conservative tie-break: choose the higher threshold when wealth ties.
    _, alpha, threshold = max(choices, key=lambda x: (x[0], x[2], x[1]))
    return alpha, threshold


def walk_forward(samples, config, holdout=False):
    cutoff = pd.Timestamp(config.holdout_start)
    evaluation = samples[samples.index >= cutoff] if holdout else samples[samples.exit_date < cutoff]
    if not holdout:
        evaluation = evaluation.iloc[config.min_train+config.validation+2:]
    if evaluation.empty:
        raise ValueError('No evaluation data for requested history/holdout')
    records, folds = [], []
    for start in range(0, len(evaluation), config.fold):
        test = evaluation.iloc[start:start+config.fold]
        history = samples[(samples.index < test.index[0]) & (samples.exit_date <= test.index[0])]
        alpha, threshold = select_model(history, config)
        prediction = ridge_predict(history, test, alpha)
        block = test.copy()
        block['prediction'] = prediction
        block['desired'] = (prediction > threshold)*config.max_exposure
        block['fold'] = len(folds)
        records.append(block)
        folds.append({'fold': len(folds), 'first_signal': str(test.index[0].date()),
                      'last_signal': str(test.index[-1].date()), 'training_last_label_exit': str(history.exit_date.max().date()),
                      'alpha': alpha, 'threshold': None if np.isinf(threshold) else threshold,
                      'no_trade_selected': bool(np.isinf(threshold)), 'train_rows': len(history)})
    return pd.concat(records), folds


def run(path, out, config, holdout=False):
    samples = make_samples(path)
    evaluated, folds = walk_forward(samples, config, holdout)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    strategies = {'ridge': evaluated.desired.to_numpy(),
                  'cash': np.zeros(len(evaluated)),
                  'buy_hold': np.full(len(evaluated), config.max_exposure),
                  'momentum_20': (evaluated.return_20.to_numpy() > 0)*config.max_exposure}
    summary, ledgers, sensitivity = {}, {}, []
    for name, desired in strategies.items():
        # Baselines have no drawdown stop; model uses stated operational risk rule.
        ledger = backtest(evaluated, desired, config, stop=name=='ridge')
        ledger.to_csv(out/f'{name}_ledger.csv', index=False)
        summary[name] = summarize(ledger)
        ledgers[name] = ledger
    for multiplier in (0., 1., 2., 4.):
        cfg = Config(**{**asdict(config), 'fee_bps': config.fee_bps*multiplier,
                        'half_spread_bps': config.half_spread_bps*multiplier,
                        'slippage_bps': config.slippage_bps*multiplier})
        # Freeze predictions/thresholds: this tests costs, not model reselection.
        sensitivity.append({'cost_multiplier': multiplier, **summarize(backtest(evaluated, evaluated.desired, cfg))})
    evaluated.to_csv(out/'predictions.csv')
    pd.DataFrame(sensitivity).to_csv(out/'cost_sensitivity.csv', index=False)
    yearly = []
    for name, ledger in ledgers.items():
        for year, group in ledger.groupby(pd.to_datetime(ledger.exit_date).dt.year):
            yearly.append({'strategy': name, 'year': int(year), 'net_return': float((1+group.net_return).prod()-1)})
    pd.DataFrame(yearly).to_csv(out/'yearly_returns.csv', index=False)
    report = {'config': asdict(config), 'data_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'input_file': Path(path).name, 'mode': 'holdout' if holdout else 'development',
              'first_signal': str(evaluated.index.min().date()), 'last_signal': str(evaluated.index.max().date()),
              'folds': folds, 'results': summary,
              'limitations': ['Retrospective adjusted-price return proxy; not raw-share/corporate-action accounting.',
                             'Data ends in 2024; no current market feed or exchange-calendar completeness verification.',
                             'Cash earns zero; no taxes, market impact, auction fills or liquidity model.',
                             'Fixed basis-point costs are assumptions, not measured broker execution.',
                             'Historical files were examined previously: reserved holdout is not genuinely unseen research data.',
                             'Drawdown stop acts at the next open and may overshoot; no live orders are implemented.',
                             'Single-asset research; choosing assets or settings after these results causes selection bias.']}
    source = pd.read_csv(path)
    report['data_quality'] = {'source_rows': len(source), 'usable_samples': len(samples),
                              'zero_volume_dates': source.loc[source.Volume == 0, 'Date'].tolist(),
                              'zero_volume_policy': 'Exclude samples with zero-volume execution or exit sessions; never impute fills.'}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, ledger in ledgers.items():
        axes[0].plot(ledger.exit_date, ledger.equity, label=name)
        axes[1].plot(ledger.exit_date, ledger.drawdown, label=name)
    axes[0].set(ylabel='Wealth from $1', title='Historical research: adjusted-price proxy, after assumed costs')
    axes[1].set(ylabel='Drawdown')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out/'equity.png')
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='MSFT.csv')
    parser.add_argument('--out', default='reports/msft')
    parser.add_argument('--evaluate-holdout', action='store_true', help='Explicitly evaluate the reserved final period')
    args = parser.parse_args()
    report = run(args.data, args.out, Config(), args.evaluate_holdout)
    print(pd.DataFrame(report['results']).T.to_string())
    print(f'Report: {Path(args.out).resolve() / "report.json"}')


if __name__ == '__main__':
    main()
