"""
Regression tests for the regime signal fix (A.7).

HeuristicRegimeModel.update() used to guess trend/volatility from
positional indices into the generic per-step feature vector
(observation[0] assumed a "return", an arbitrary vol_idx formula assumed a
"volatility" feature) -- fragile, and not actually reading a real SMA
crossover at all despite RegimeConfig.trend_ma_fast/trend_ma_slow existing
in config. The fix computes trend (SMA(trend_ma_fast) vs
SMA(trend_ma_slow) crossover) and volatility directly from the market
ticker's own named Close price history, the same way fit() already does,
and ignores `observation` entirely.

Most tests below pin belief_formation=SOFTMAX_TEMPERATURE explicitly: they
probe the shared trend/vol *signal* (score direction, trend_ma sensitivity)
through its continuous mapping into belief, which candidate A exposes
directly. Candidate C (hard EMA + hysteresis) is now the canonical default
(see RegimeConfig) but commits to one argmax regime, which would make e.g.
"does a bigger/smaller trend_ma window change the belief" a much noisier
thing to probe (both windows can commit to the same regime and look
identical) -- that mechanism has its own dedicated tests in
test_shield_diagnostics.py / the recalibration sweep, not here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import RegimeConfig, RegimeModelType, BeliefFormationType
from regime_model import HeuristicRegimeModel

TICKER = "TEST"


def _make_ohlcv(closes, ticker=TICKER):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    df = pd.DataFrame({(ticker, "Close"): closes}, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _price_path(n, mu, sigma, seed):
    rng = np.random.RandomState(seed)
    log_returns = rng.normal(mu, sigma, n)
    log_prices = np.cumsum(log_returns) + np.log(100.0)
    return np.exp(log_prices)


def _run_model(closes, trend_ma_fast=5, trend_ma_slow=20, n_extra_steps=30, mu=0.0, sigma=0.01, seed=1):
    """Fit on `closes`, then continue feeding `n_extra_steps` more days of
    the same regime and return the final belief. Pins candidate A
    (softmax+temperature, T=1.0) explicitly -- see module docstring."""
    config = RegimeConfig(
        model_type=RegimeModelType.HEURISTIC,
        trend_ma_fast=trend_ma_fast,
        trend_ma_slow=trend_ma_slow,
        belief_ema_alpha=0.3,
        belief_formation=BeliefFormationType.SOFTMAX_TEMPERATURE,
    )
    model = HeuristicRegimeModel(config)
    model.fit(_make_ohlcv(closes))

    rng = np.random.RandomState(seed)
    last_price = closes[-1]
    belief = model.get_belief()
    for _ in range(n_extra_steps):
        last_price = last_price * np.exp(rng.normal(mu, sigma))
        garbage_observation = np.zeros(10)  # must be ignored entirely
        belief = model.update(garbage_observation, prices={TICKER: last_price})

    return belief


def test_uptrend_scores_bull_over_bear():
    closes = _price_path(200, mu=0.0008, sigma=0.008, seed=42)
    belief = _run_model(closes, mu=0.0008, sigma=0.008, seed=43)

    assert belief[0] > belief[1], (
        f"expected Bull score > Bear score for a sustained uptrend, got "
        f"belief={belief}"
    )


def test_downtrend_scores_bear_over_bull():
    closes = _price_path(200, mu=-0.0008, sigma=0.008, seed=42)
    belief = _run_model(closes, mu=-0.0008, sigma=0.008, seed=43)

    assert belief[1] > belief[0], (
        f"expected Bear score > Bull score for a sustained downtrend, got "
        f"belief={belief}"
    )


def test_trend_ma_windows_from_config_are_actually_used():
    """RegimeConfig.trend_ma_fast/trend_ma_slow must actually affect the
    computed signal -- proves they're wired in, not dead config."""
    closes = _price_path(200, mu=0.0005, sigma=0.01, seed=7)

    belief_short_windows = _run_model(
        closes, trend_ma_fast=3, trend_ma_slow=10, mu=0.0005, sigma=0.01, seed=8
    )
    belief_long_windows = _run_model(
        closes, trend_ma_fast=20, trend_ma_slow=60, mu=0.0005, sigma=0.01, seed=8
    )

    assert not np.allclose(belief_short_windows, belief_long_windows), (
        "changing trend_ma_fast/trend_ma_slow had no effect on the belief -- "
        "they may not be wired into the trend computation"
    )


def test_update_ignores_observation_content():
    """The whole point of the fix: the generic feature vector must have no
    influence at all -- only `prices` (named Close) drives the signal."""
    closes = _price_path(150, mu=0.0006, sigma=0.01, seed=3)

    config = RegimeConfig(model_type=RegimeModelType.HEURISTIC, trend_ma_fast=5, trend_ma_slow=20)

    model_a = HeuristicRegimeModel(config)
    model_a.fit(_make_ohlcv(closes))

    model_b = HeuristicRegimeModel(config)
    model_b.fit(_make_ohlcv(closes))

    rng = np.random.RandomState(9)
    last_price = closes[-1]
    belief_a, belief_b = None, None
    for _ in range(20):
        last_price = last_price * np.exp(rng.normal(0.0006, 0.01))
        belief_a = model_a.update(np.zeros(12), prices={TICKER: last_price})
        # Wildly different, huge-magnitude "observation" -- old positional
        # code would have read garbage from this and produced a different
        # belief; the fix must be completely unaffected.
        belief_b = model_b.update(np.full(12, 1e6), prices={TICKER: last_price})

    np.testing.assert_allclose(belief_a, belief_b)


if __name__ == "__main__":
    test_uptrend_scores_bull_over_bear()
    print("PASS: uptrend scores Bull over Bear")

    test_downtrend_scores_bear_over_bull()
    print("PASS: downtrend scores Bear over Bull")

    test_trend_ma_windows_from_config_are_actually_used()
    print("PASS: trend_ma_fast/trend_ma_slow are wired in")

    test_update_ignores_observation_content()
    print("PASS: update() ignores observation content, only prices matter")
