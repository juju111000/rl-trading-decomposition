"""
Unit tests for run_baselines.py's vol-targeting and time-series-momentum
(TSMOM) logic, against hand-computed toy cases -- no environment, no data
loading, just the strategy's own signal computation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from run_baselines import PriceHistory, VolTargetStrategy, TimeSeriesMomentumStrategy
from config import Config


def _make_price_history(price_series_by_ticker, lookback_len):
    tickers = list(price_series_by_ticker.keys())
    return PriceHistory(price_series_by_ticker, lookback_len=lookback_len, tickers=tickers)


# =============================================================================
# VolTargetStrategy
# =============================================================================

def test_vol_target_matches_hand_computed_scale_for_alternating_returns():
    """20 daily log-returns alternating +r/-r have a known sample std (ddof=1):
    std = r * sqrt(n / (n-1)) for a zero-mean alternating series of even
    length n. With r=0.01, n=20: std = 0.01 * sqrt(20/19), annualized by
    sqrt(252), and the strategy's scale = target_vol / annualized_vol
    (uncapped, since it comes out well under max_leverage=2.0 here)."""
    r = 0.01
    n = 20
    rets = np.array([r if i % 2 == 0 else -r for i in range(n)])
    log_prices = np.concatenate([[0.0], np.cumsum(rets)])
    prices = np.exp(log_prices) * 100.0  # length 21: p_0..p_20

    price_history = _make_price_history({'A': prices, 'B': prices}, lookback_len=len(prices) - 1)

    config = Config()
    strategy = VolTargetStrategy(
        n_assets=2, config=config, price_history=price_history,
        target_vol=0.10, vol_window=20, rebalance_freq=1, max_leverage=2.0,
    )

    weights = strategy.get_action(obs=np.zeros(1), info=None)

    expected_std = r * np.sqrt(n / (n - 1))
    expected_annualized_vol = expected_std * np.sqrt(252)
    expected_scale = 0.10 / expected_annualized_vol
    assert expected_scale < 2.0, "test fixture should stay below max_leverage, or the cap masks the calculation"

    expected_weights = np.array([0.5, 0.5]) * expected_scale
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-9)


def test_vol_target_caps_at_max_leverage_for_near_zero_volatility():
    """A flat (zero-volatility) price series would imply infinite leverage
    to hit a nonzero vol target -- the strategy must clip to max_leverage."""
    prices = np.full(21, 100.0)  # perfectly flat: zero realized vol
    price_history = _make_price_history({'A': prices, 'B': prices}, lookback_len=len(prices) - 1)

    config = Config()
    strategy = VolTargetStrategy(
        n_assets=2, config=config, price_history=price_history,
        target_vol=0.10, vol_window=20, rebalance_freq=1, max_leverage=2.0,
    )

    weights = strategy.get_action(obs=np.zeros(1), info=None)
    np.testing.assert_allclose(weights, np.array([1.0, 1.0]), rtol=1e-9)


def test_vol_target_holds_base_weights_when_insufficient_history():
    """Fewer than vol_window+2 real prices available -- too early to compute
    a 20-day realized vol, so it must fall back to the base weights rather
    than error or fabricate a scale from a too-short window."""
    prices = np.array([100.0, 101.0, 99.0])  # only 3 prices, need vol_window+2=22
    price_history = _make_price_history({'A': prices, 'B': prices}, lookback_len=len(prices) - 1)

    config = Config()
    strategy = VolTargetStrategy(
        n_assets=2, config=config, price_history=price_history,
        target_vol=0.10, vol_window=20, rebalance_freq=1, max_leverage=2.0,
    )

    weights = strategy.get_action(obs=np.zeros(1), info=None)
    np.testing.assert_allclose(weights, np.array([0.5, 0.5]), rtol=1e-9)


# =============================================================================
# TimeSeriesMomentumStrategy
# =============================================================================

def _make_alt_prices(alt_r, drift, n_alt=60):
    """Price path = one drift day (sign controls overall momentum) followed
    by n_alt zero-sum alternating +alt_r/-alt_r log returns (controls
    realized vol exactly -- see test_tsmom_weights_winners_by_inverse_volatility
    for the closed-form std). Length n_alt + 2."""
    alt_rets = np.array([alt_r if i % 2 == 0 else -alt_r for i in range(n_alt)])
    all_rets = np.concatenate([[drift], alt_rets])
    log_prices = np.concatenate([[0.0], np.cumsum(all_rets)])
    return np.exp(log_prices) * 100.0


def test_tsmom_longs_only_the_positive_momentum_asset():
    """Asset A: positive drift (wins). Asset B: negative drift (loses), same
    realized volatility as A. A single winner is NOT handed the whole book
    (that was the 2008/CL=F bug -- see class docstring): sized against a
    2-asset reference budget, an at-reference-vol lone winner gets exactly
    1/n_assets = 0.5, the rest sits in cash. B stays flat at 0."""
    lookback = 61
    prices_a = _make_alt_prices(alt_r=0.02, drift=0.05)   # wins, vol v
    prices_b = _make_alt_prices(alt_r=0.02, drift=-0.05)  # loses, same vol v

    price_history = _make_price_history({'A': prices_a, 'B': prices_b}, lookback_len=lookback)

    config = Config()
    strategy = TimeSeriesMomentumStrategy(
        n_assets=2, config=config, price_history=price_history,
        lookback=lookback, rebalance_freq=lookback, vol_window=60,
    )

    weights = strategy.get_action(obs=np.zeros(1), info=None)
    np.testing.assert_allclose(weights, np.array([0.5, 0.0]), rtol=1e-6)


def test_tsmom_equal_splits_when_winners_have_equal_volatility():
    """Winners are weighted by inverse realized volatility, not headcount
    (see test below) -- but when volatilities are identical, that must
    reduce to a plain equal split. Same price path reused for both assets
    guarantees identical volatility."""
    lookback = 20
    prices_a = np.linspace(100.0, 110.0, lookback + 1)
    prices_b = prices_a.copy()

    price_history = _make_price_history({'A': prices_a, 'B': prices_b}, lookback_len=lookback)

    config = Config()
    strategy = TimeSeriesMomentumStrategy(
        n_assets=2, config=config, price_history=price_history,
        lookback=lookback, rebalance_freq=lookback,
    )

    weights = strategy.get_action(obs=np.zeros(1), info=None)
    np.testing.assert_allclose(weights, np.array([0.5, 0.5]), rtol=1e-9)


def test_tsmom_weights_two_winners_by_inverse_volatility():
    """When enough winners stack up to exceed full investment, they delever
    pro rata -- which reduces to a pure inverse-vol split, same as if they'd
    been renormalized directly. Asset A: alternating +-0.02 (high vol).
    Asset B: alternating +-0.01 (exactly half A's vol). Sample std (ddof=1)
    of a zero-mean n=60 alternating series is r*sqrt(60/59); that factor
    cancels in the A:B ratio, leaving vol_A:vol_B = 2:1 exactly, so the
    delevered split must be exactly [1/3, 2/3] (A more volatile -> smaller
    share)."""
    prices_a = _make_alt_prices(alt_r=0.02, drift=0.05)  # high vol, wins
    prices_b = _make_alt_prices(alt_r=0.01, drift=0.05)  # low vol (half A's), wins
    lookback = len(prices_a) - 1  # momentum measured over the whole array

    price_history = _make_price_history({'A': prices_a, 'B': prices_b}, lookback_len=lookback)

    config = Config()
    strategy = TimeSeriesMomentumStrategy(
        n_assets=2, config=config, price_history=price_history,
        lookback=lookback, rebalance_freq=lookback, vol_window=60,
    )

    weights = strategy.get_action(obs=np.zeros(1), info=None)
    np.testing.assert_allclose(weights, np.array([1.0 / 3.0, 2.0 / 3.0]), rtol=1e-6)


def test_tsmom_lone_volatile_winner_gets_a_modest_slice_not_the_whole_book():
    """The actual 2008/CL=F bug case: ONE winner, more volatile than the
    universe reference, among several losers. It must NOT get the whole
    book (that's what let CL=F take 100% right before its H2 2008 crash) --
    it gets (median_vol / its_vol) / n_assets, the rest stays in cash.

    3 assets: A wins with vol=2v, B and C lose with vol=v each. Median of
    [2v, v, v] is v, so A's weight = (v / 2v) / 3 = 1/6 -- a modest slice,
    not 1/3 (equal-share) and nowhere near 1.0 (whole book)."""
    prices_a = _make_alt_prices(alt_r=0.02, drift=0.05)    # wins, vol 2v
    prices_b = _make_alt_prices(alt_r=0.01, drift=-0.05)   # loses, vol v
    prices_c = _make_alt_prices(alt_r=0.01, drift=-0.03)   # loses, vol v
    lookback = len(prices_a) - 1

    price_history = _make_price_history(
        {'A': prices_a, 'B': prices_b, 'C': prices_c}, lookback_len=lookback
    )

    config = Config()
    strategy = TimeSeriesMomentumStrategy(
        n_assets=3, config=config, price_history=price_history,
        lookback=lookback, rebalance_freq=lookback, vol_window=60,
    )

    weights = strategy.get_action(obs=np.zeros(1), info=None)
    np.testing.assert_allclose(weights, np.array([1.0 / 6.0, 0.0, 0.0]), rtol=1e-6)


def test_tsmom_goes_all_cash_when_nothing_has_positive_momentum():
    lookback = 20
    prices_a = np.linspace(100.0, 90.0, lookback + 1)
    prices_b = np.linspace(100.0, 95.0, lookback + 1)

    price_history = _make_price_history({'A': prices_a, 'B': prices_b}, lookback_len=lookback)

    config = Config()
    strategy = TimeSeriesMomentumStrategy(
        n_assets=2, config=config, price_history=price_history,
        lookback=lookback, rebalance_freq=lookback,
    )

    weights = strategy.get_action(obs=np.zeros(1), info=None)
    np.testing.assert_allclose(weights, np.array([0.0, 0.0]), rtol=1e-9)


def test_tsmom_holds_flat_when_insufficient_history():
    lookback = 20
    prices_a = np.array([100.0, 101.0])  # only 2 prices, need lookback+1=21
    price_history = _make_price_history({'A': prices_a, 'B': prices_a}, lookback_len=len(prices_a) - 1)

    config = Config()
    strategy = TimeSeriesMomentumStrategy(
        n_assets=2, config=config, price_history=price_history,
        lookback=lookback, rebalance_freq=lookback,
    )

    weights = strategy.get_action(obs=np.zeros(1), info=None)
    np.testing.assert_allclose(weights, np.array([0.0, 0.0]), rtol=1e-9)


if __name__ == "__main__":
    test_vol_target_matches_hand_computed_scale_for_alternating_returns()
    print("PASS: vol target matches hand-computed scale")

    test_vol_target_caps_at_max_leverage_for_near_zero_volatility()
    print("PASS: vol target caps at max leverage for near-zero volatility")

    test_vol_target_holds_base_weights_when_insufficient_history()
    print("PASS: vol target falls back to base weights with insufficient history")

    test_tsmom_longs_only_the_positive_momentum_asset()
    print("PASS: TSMOM longs only the positive-momentum asset")

    test_tsmom_equal_splits_when_winners_have_equal_volatility()
    print("PASS: TSMOM equal-splits when winners have equal volatility")

    test_tsmom_weights_two_winners_by_inverse_volatility()
    print("PASS: TSMOM weights two winners by inverse volatility (2:1 vol -> 1:2 weight)")

    test_tsmom_lone_volatile_winner_gets_a_modest_slice_not_the_whole_book()
    print("PASS: TSMOM lone volatile winner gets a modest slice, not the whole book")

    test_tsmom_goes_all_cash_when_nothing_has_positive_momentum()
    print("PASS: TSMOM goes all-cash when nothing has positive momentum")

    test_tsmom_holds_flat_when_insufficient_history()
    print("PASS: TSMOM holds flat with insufficient history")
