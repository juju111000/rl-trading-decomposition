"""
Regression tests for the PortfolioEngine NAV double-append fix.

Prior to the fix, execute_trades() and update_prices() each appended a
separate entry to _nav_history per environment step, so the reward computed
as log(nav_history[-1] / nav_history[-2]) only ever captured the price-return
leg and silently dropped the fee/slippage leg. These tests pin down the two
invariants that fix must satisfy:

1. Telescoping: summing every step's mark_to_market() log-return must exactly
   reproduce log(NAV_final / NAV_initial) -- only possible if nav_history has
   exactly one entry per step.
2. Cost sensitivity: with nonzero turnover, charging fees must strictly lower
   cumulative log-return relative to an identical trade/price sequence with
   zero fees -- only possible if the fee leg is actually included somewhere
   in the reported per-step returns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from portfolio_engine import PortfolioEngine

TICKERS = ["A", "B", "C"]


def _price_path(n_steps: int, seed: int = 0):
    """Deterministic multi-asset price path, starting at 100 per asset."""
    rng = np.random.RandomState(seed)
    prices = {t: 100.0 for t in TICKERS}
    path = [dict(prices)]
    for _ in range(n_steps):
        prices = {
            t: prices[t] * np.exp(rng.normal(0.0002, 0.01))
            for t in TICKERS
        }
        path.append(dict(prices))
    return path


def _weight_targets(n_steps: int, seed: int = 1):
    """Deterministic sequence of rebalance targets that actually vary,
    so turnover is nonzero at (most) every step."""
    rng = np.random.RandomState(seed)
    targets = []
    for _ in range(n_steps):
        w = rng.uniform(-0.3, 0.6, size=len(TICKERS))
        targets.append(w)
    return targets


def _run(engine: PortfolioEngine, prices_path, weight_targets):
    """Drive an engine through execute_trades -> mark_to_market once per
    step, exactly as trading_env.TradingCMDPEnv.step() now does."""
    log_returns = []
    for i in range(len(weight_targets)):
        current_prices = prices_path[i]
        next_prices = prices_path[i + 1]

        engine.execute_trades(weight_targets[i], current_prices)
        log_return = engine.mark_to_market(next_prices, prev_prices=current_prices)
        log_returns.append(log_return)

    return log_returns


def test_sum_of_step_log_returns_matches_total_nav_change():
    n_steps = 50
    prices_path = _price_path(n_steps, seed=0)
    weight_targets = _weight_targets(n_steps, seed=1)

    engine = PortfolioEngine(
        TICKERS,
        initial_capital=1_000_000.0,
        transaction_fee_rate=0.001,
        slippage_rate=0.0005,
    )

    log_returns = _run(engine, prices_path, weight_targets)

    nav_history = engine.get_nav_history()
    # Exactly one entry per step, plus the initial value.
    assert len(nav_history) == n_steps + 1

    total_log_return = np.log(nav_history[-1] / nav_history[0])
    summed_log_return = np.sum(log_returns)

    assert abs(summed_log_return - total_log_return) < 1e-8, (
        f"sum(step log-returns)={summed_log_return!r} does not match "
        f"log(NAV_final/NAV_initial)={total_log_return!r}"
    )


def test_fees_strictly_reduce_cumulative_reward_vs_zero_fee_case():
    n_steps = 50
    prices_path = _price_path(n_steps, seed=0)
    weight_targets = _weight_targets(n_steps, seed=1)

    # Sanity: turnover is actually nonzero at every step (weights vary and
    # start from an all-zero position), otherwise this test would be vacuous.
    turnovers = [np.sum(np.abs(w)) for w in weight_targets]
    assert all(t > 0.05 for t in turnovers)

    engine_with_fees = PortfolioEngine(
        TICKERS,
        initial_capital=1_000_000.0,
        transaction_fee_rate=0.001,
        slippage_rate=0.0005,
    )
    engine_zero_fees = PortfolioEngine(
        TICKERS,
        initial_capital=1_000_000.0,
        transaction_fee_rate=0.0,
        slippage_rate=0.0,
    )

    returns_with_fees = _run(engine_with_fees, prices_path, weight_targets)
    returns_zero_fees = _run(engine_zero_fees, prices_path, weight_targets)

    cumulative_with_fees = np.sum(returns_with_fees)
    cumulative_zero_fees = np.sum(returns_zero_fees)

    assert cumulative_with_fees < cumulative_zero_fees, (
        f"cumulative reward with fees ({cumulative_with_fees!r}) should be "
        f"strictly less than the zero-fee case ({cumulative_zero_fees!r})"
    )

    # And the gap should be commensurate with fees actually being charged
    # (not some unrelated numerical drift).
    assert cumulative_zero_fees - cumulative_with_fees > 1e-6


if __name__ == "__main__":
    test_sum_of_step_log_returns_matches_total_nav_change()
    print("PASS: sum(step log-returns) matches log(NAV_final/NAV_initial)")

    test_fees_strictly_reduce_cumulative_reward_vs_zero_fee_case()
    print("PASS: fees strictly reduce cumulative reward vs. zero-fee case")
