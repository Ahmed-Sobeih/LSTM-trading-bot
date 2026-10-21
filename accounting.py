"""Raw-share, long/cash accounting with explicit ex-date/pay-date events."""
import numpy as np
import pandas as pd

from market_data import validate


def raw_samples(bars, actions, include_unlabeled=False):
    """Causal realized total-return features; next-open/next-open labels.

    Uses effective historical actions only. This is retrospective research,
    not proof that a downloaded history was available at past decision times.
    """
    from research import FEATURES
    bars = bars.set_index('session')
    ratio = pd.Series(1., index=bars.index)
    dividend = pd.Series(0., index=bars.index)
    for _, action in actions.iterrows():
        if action.effective_date not in bars.index:
            continue
        if action.kind == 'split':
            ratio.loc[action.effective_date] *= float(action.split_ratio)
        else:
            dividend.loc[action.effective_date] += float(action.cash_per_share)
    # Cash dividend is per post-split share when events coincide.
    realized = ratio*(bars.close+dividend)/bars.close.shift(1)-1
    index = (1+realized.fillna(0)).cumprod()
    frame = pd.DataFrame(index=bars.index)
    frame['return_1'] = realized
    frame['return_5'] = index.pct_change(5, fill_method=None)
    frame['return_20'] = index.pct_change(20, fill_method=None)
    frame['volatility_20'] = realized.rolling(20).std()
    frame['range'] = (bars.high-bars.low)/bars.close
    # Volume is normalized to today's share basis using only effective splits.
    share_basis_volume = bars.volume / ratio.cumprod()
    frame['volume_change'] = share_basis_volume/share_basis_volume.rolling(20).mean()-1
    frame['target'] = ratio.shift(-2)*(bars.open.shift(-2)+dividend.shift(-2))/bars.open.shift(-1)-1
    dates = pd.Series(bars.index, index=bars.index)
    frame['execution_date'], frame['exit_date'] = dates.shift(-1), dates.shift(-2)
    frame = frame.dropna(subset=FEATURES if include_unlabeled else None)
    checked = FEATURES if include_unlabeled else FEATURES+['target']
    if not np.isfinite(frame[checked].to_numpy()).all():
        raise ValueError('Nonfinite raw features/targets')
    return frame


def simulate(bars, actions, signals, config, observed_at, initial_cash=10000., liquidate=True, stop=True):
    """Signals are close-date exposures; fills occur at following session open.

    Dividends accrue before ex-date trading to overnight holders and remain
    receivables until payment-date close (or next session close). Receivables
    cannot fund trades. Splits occur before ex-date open. Whole-share targets;
    split-created fractional shares are retained, without cash-in-lieu modeling.
    """
    bars, actions = validate(bars, actions, observed_at, allow_stale=True)
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError('Initial cash must be positive')
    signals = pd.Series(signals, dtype=float)
    signals.index = pd.to_datetime(signals.index)
    if signals.index.duplicated().any() or not signals.index.is_monotonic_increasing:
        raise ValueError('Signal dates must be unique and ascending')
    if not np.isfinite(signals).all() or ((signals < 0) | (signals > config.max_exposure)).any():
        raise ValueError('Signals exceed long/cash exposure limits')
    bars = bars.set_index('session')
    if not signals.index.isin(bars.index).all():
        raise ValueError('Signal dates must exist in raw bars')
    if (signals.index >= bars.index[-1]).any():
        raise ValueError('Every signal needs a following execution session')
    schedule = {bars.index[bars.index.get_loc(day)+1]: weight for day, weight in signals.items()}
    cash, shares, peak, previous_equity = float(initial_cash), 0., float(initial_cash), float(initial_cash)
    pending, rows, fills = [], [], []
    halted = False
    rate = (config.fee_bps+config.half_spread_bps+config.slippage_bps)/10000
    start = min(schedule)

    def trade(day, mark, target, reason):
        nonlocal cash, shares
        liquid = cash+shares*mark
        direction = 1 if target*liquid >= shares*mark else -1
        wanted = (target*liquid+direction*target*rate*mark*shares)/(mark*(1+direction*target*rate))
        wanted = float(max(0, np.floor(wanted+1e-10)))
        delta = wanted-shares
        cost = abs(delta)*mark*rate
        cash -= delta*mark+cost
        shares = wanted
        if cash < -1e-7 or shares < 0:
            raise ValueError('Accounting invariant failed: negative cash or shares')
        cash = max(0., cash)
        if abs(delta) > 1e-10:
            fills.append({'session': day, 'reason': reason, 'quantity': delta, 'reference_price': mark,
                          'commission': abs(delta)*mark*config.fee_bps/10000,
                          'spread_slippage': abs(delta)*mark*(config.half_spread_bps+config.slippage_bps)/10000,
                          'cash_after': cash, 'shares_after': shares})
        return cost

    for day, bar in bars.loc[start:].iterrows():
        events = actions[actions.effective_date == day]
        for _, action in events[events.kind == 'split'].iterrows():
            shares *= float(action.split_ratio)
        accrued = 0.
        for _, action in events[events.kind == 'dividend'].iterrows():
            amount = shares*float(action.cash_per_share)
            accrued += amount
            pending.append({'id': action.id, 'pay_date': action.pay_date, 'amount': amount})
        cost = 0.
        if day in schedule or halted:
            cost += trade(day, float(bar.open), 0. if halted else schedule[day], 'open')
        paid = sum(item['amount'] for item in pending if item['pay_date'] <= day)
        pending = [item for item in pending if item['pay_date'] > day]
        # Conservatively unavailable for trading until the next session open.
        cash += paid
        if liquidate and day == bars.index[-1]:
            cost += trade(day, float(bar.close), 0., 'scheduled_final_close')
        receivable = sum(item['amount'] for item in pending)
        equity = cash+shares*float(bar.close)+receivable
        peak = max(peak, equity)
        drawdown = equity/peak-1
        if stop and drawdown <= -config.drawdown_stop:
            halted = True
        rows.append({'session': day, 'cash': cash, 'shares': shares, 'close': bar.close,
                     'dividend_accrued': accrued, 'dividend_paid': paid, 'receivable': receivable,
                     'cost': cost, 'equity': equity, 'net_return': equity/previous_equity-1,
                     'drawdown': drawdown, 'halted': halted})
        previous_equity = equity
    return pd.DataFrame(rows), pd.DataFrame(fills, columns=['session','reason','quantity','reference_price','commission','spread_slippage','cash_after','shares_after'])
