"""
Regression tests for audit defect #8 (slippage units).

The volatility-slippage term is |trade| x vol x slippage_vol_multiplier with
`vol` on a DAILY scale (the 0.02 default is a 2% daily vol). Before the fix,
PortfolioEngine.mark_to_market overwrote the estimate with ANNUALIZED vol
(std x sqrt(252)), inflating the term ~16x (measured all-in cost per unit
turnover: 3.3-3.9% instead of a few tenths of a percent), and the term was
charged even with use_volatility_slippage=False.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from portfolio_engine import PortfolioEngine

TICKERS = ["A", "B", "C"]
DAILY_VOL = 0.01
FEE, SLIP, MULT = 0.001, 0.0005, 0.1


def _path(n, seed=0):
    rng = np.random.RandomState(seed)
    p = {t: 100.0 for t in TICKERS}
    out = [dict(p)]
    for _ in range(n):
        p = {t: p[t] * np.exp(rng.normal(0.0, DAILY_VOL)) for t in TICKERS}
        out.append(dict(p))
    return out


def _engine_after_history(use_vol=True, n_hist=30):
    eng = PortfolioEngine(TICKERS, transaction_fee_rate=FEE, slippage_rate=SLIP,
                          use_volatility_slippage=use_vol, slippage_vol_multiplier=MULT)
    path = _path(n_hist)
    eng.execute_trades(np.array([0.3, 0.3, 0.3]), path[0])
    eng.mark_to_market(path[1], prev_prices=path[0])
    for i in range(1, n_hist):
        eng.mark_to_market(path[i + 1], prev_prices=path[i])   # hold; returns accumulate
    return eng, path[n_hist]


def _expected_vol(eng, ticker):
    return float(np.std(eng._return_buffer[ticker]))


def test_vol_estimate_is_daily_scale():
    eng, _ = _engine_after_history()
    for t in TICKERS:
        v = eng._vol_estimates[t]
        assert abs(v - _expected_vol(eng, t)) < 1e-15
        assert 0.003 < v < 0.03, f"{t}: vol estimate {v} is not daily-scale (annualized would be ~0.16)"


def test_cost_equals_commission_plus_slippage_and_is_small():
    eng, prices = _engine_after_history()
    target = np.array([-0.2, 0.5, 0.1])
    nav = eng.compute_nav(prices)
    w0 = eng.compute_weights(prices)
    turnover = eng.compute_turnover(target, prices)
    cost, _ = eng.execute_trades(target, prices)

    expected = sum(abs(target[i] - w0[i]) * nav * (FEE + SLIP + MULT * _expected_vol(eng, t))
                   for i, t in enumerate(TICKERS))
    assert abs(cost - expected) / expected < 1e-9

    per_unit = cost / (turnover * nav)
    assert 0.0015 < per_unit < 0.005, f"cost per unit turnover {per_unit:.4%} (pre-fix bug gave ~3%+)"


def test_vol_slippage_off_when_flag_false():
    eng, prices = _engine_after_history(use_vol=False)
    target = np.array([-0.2, 0.5, 0.1])
    nav = eng.compute_nav(prices)
    turnover = eng.compute_turnover(target, prices)
    cost, _ = eng.execute_trades(target, prices)
    assert abs(cost / (turnover * nav) - (FEE + SLIP)) < 1e-12
