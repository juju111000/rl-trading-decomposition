#!/usr/bin/env python3
"""
External (non-RL) baseline strategies for the paper grid: no training, no
shield, evaluated on the SAME calendar folds, transaction costs, and
environment (main.create_trading_env) as main.py's RL grid, using
evaluation.BaselineEvaluator to run each strategy and evaluation.Evaluator's
episode-metric machinery to score it.

Strategies:
  - equal_weight_monthly   equal-weight buy-and-hold, monthly rebalance
  - markowitz_252d_monthly rolling long-only max-Sharpe mean-variance,
                           252-trading-day window, monthly rebalance
  - vol_target_10pct       volatility targeting: equal-weight base scaled to
                           a 10% annualized target using 20-day realized vol
  - tsmom_12m              time-series momentum, 12-month (252d) lookback,
                           long-or-cash per asset, monthly rebalance
  - tsmom_20d              time-series momentum, 20-day lookback,
                           long-or-cash per asset, rebalanced every 20 days
  - hmm_regime             2-3 state Gaussian HMM (regime_model.HMMRegimeModel),
                           fit on the fold's TRAIN period only, state-ranked
                           exposure applied to equal weights, monthly rebalance
  - equal_weight_k5_band010  equal-weight target, decided every 5 trading
                           days with a 0.10 per-asset no-trade band on each
                           decision -- the SAME execution-model mechanics as
                           the RL grid's frozen canonical config (see
                           PROJECT_STATE.md), applied to the simplest
                           possible target, so the RL policy's improvement
                           can be benchmarked against how much is just the
                           execution model itself vs. genuine signal.

All strategies interpret an env step's `info['portfolio_state'].weights` as
the actual (price-drifted) current allocation: on non-rebalance days they
return that unchanged, so "rebalance every N days" produces zero turnover
in between rather than the environment re-trading back to a stale target
every single day.

Usage:
    python run_baselines.py [--output-dir ./results/baselines]
"""
import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import main as main_module
from config import RegimeConfig, RegimeModelType, ShieldType
from evaluation import BaselineStrategy, BaselineEvaluator, WalkForwardEvaluator
from regime_model import HMMRegimeModel

TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Causal price-history accessor
# =============================================================================

class PriceHistory:
    """Causal price-history accessor shared by every strategy in a fold.

    `close_prices` spans [lookback window][test period] as one array per
    ticker; `lookback_len` is how many of those leading rows belong to the
    lookback (real history before the fold's test period starts, from the
    SAME combined series main.load_and_prepare_data() built -- never from
    another fold's test period). `history_up_to(t)` returns everything
    available at test-day t (0-indexed), i.e. rows [0 : lookback_len + t + 1]
    -- strictly causal, no peeking past "today"."""

    def __init__(self, close_prices: Dict[str, np.ndarray], lookback_len: int, tickers: List[str]):
        self.close_prices = close_prices
        self.lookback_len = lookback_len
        self.tickers = tickers

    def history_up_to(self, t: int) -> np.ndarray:
        """(n_days, n_assets) array of Close prices through test-day t."""
        end = self.lookback_len + t + 1
        return np.column_stack([self.close_prices[tk][:end] for tk in self.tickers])

    def n_available(self, t: int) -> int:
        return self.lookback_len + t + 1


# =============================================================================
# Shared rebalance-cadence base class
# =============================================================================

class PeriodicRebalanceStrategy(BaselineStrategy):
    """On a rebalance day, computes a fresh target via `_compute_target_weights`.
    On every other day, returns the portfolio's actual current weights (from
    `info['portfolio_state'].weights`) unchanged, so the environment sees zero
    turnover -- "rebalance every N days" means literally that, not "retarget
    a stale allocation every day." Day 0 (right after reset(), no
    'portfolio_state' in info yet) is always a rebalance day."""

    def __init__(self, n_assets: int, config, rebalance_freq: int):
        super().__init__(n_assets, config)
        self.rebalance_freq = rebalance_freq
        self._step = -1
        self._last_target = None

    def _compute_target_weights(self) -> np.ndarray:
        raise NotImplementedError

    def get_action(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        self._step += 1
        is_rebalance_day = self._last_target is None or (self._step % self.rebalance_freq == 0)

        if is_rebalance_day:
            target = np.asarray(self._compute_target_weights(), dtype=np.float64)
            self._last_target = target
            return target

        if info is not None and 'portfolio_state' in info:
            return np.asarray(info['portfolio_state'].weights, dtype=np.float64)
        return self._last_target

    def reset(self):
        self._step = -1
        self._last_target = None


# =============================================================================
# Strategies
# =============================================================================

class EqualWeightMonthlyStrategy(PeriodicRebalanceStrategy):
    """Equal-weight buy-and-hold, rebalanced back to 1/N every `rebalance_freq`
    trading days (drifts with prices in between, like a real B&H portfolio)."""

    def __init__(self, n_assets: int, config, rebalance_freq: int = 21):
        super().__init__(n_assets, config, rebalance_freq)

    def _compute_target_weights(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets


class EqualWeightBandedStrategy(PeriodicRebalanceStrategy):
    """Equal-weight target (1/N), decided every `rebalance_freq` trading
    days (k-day decision frequency) with a per-asset no-trade band applied
    on each decision day: an asset only trades if its actual (price-drifted)
    current weight has moved more than `no_trade_band` away from 1/N,
    otherwise it's left alone -- mirrors trading_env.py's
    environment.decision_frequency_days / environment.no_trade_band
    execution-model mechanics (the RL grid's frozen canonical config, see
    PROJECT_STATE.md) applied to the simplest possible target, giving a
    matched passive comparator: any return/turnover improvement the RL
    grid shows over plain equal_weight_monthly could just be the execution
    model: this strategy isolates that by giving the SAME mechanics to a
    target with no signal at all.

    Overrides get_action() (not just _compute_target_weights, like every
    other strategy here) because banding needs the actual current weights
    from `info`, not just a rebalance-day/hold-day split."""

    def __init__(self, n_assets: int, config, rebalance_freq: int = 5, no_trade_band: float = 0.10):
        super().__init__(n_assets, config, rebalance_freq)
        self.no_trade_band = no_trade_band
        self.target_weights = np.ones(n_assets) / n_assets

    def get_action(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        self._step += 1
        is_decision_day = self._last_target is None or (self._step % self.rebalance_freq == 0)

        if not is_decision_day:
            if info is not None and 'portfolio_state' in info:
                return np.asarray(info['portfolio_state'].weights, dtype=np.float64)
            return self._last_target

        current = (
            np.asarray(info['portfolio_state'].weights, dtype=np.float64)
            if info is not None and 'portfolio_state' in info
            else self.target_weights
        )
        delta = self.target_weights - current
        banded = np.where(np.abs(delta) > self.no_trade_band, self.target_weights, current)
        self._last_target = banded
        return banded


def _max_sharpe_long_only(mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0) -> np.ndarray:
    """Long-only, fully-invested (sum=1) max-Sharpe (tangency) portfolio via
    SLSQP. Falls back to equal-weight if optimization fails or every asset
    has non-positive expected excess return (max-Sharpe is degenerate then)."""
    n = len(mu)
    excess = mu - risk_free

    if np.all(excess <= 0):
        return np.ones(n) / n

    def neg_sharpe(w):
        port_ret = w @ excess
        port_vol = np.sqrt(max(w @ cov @ w, 1e-12))
        return -port_ret / port_vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return np.ones(n) / n
    w = np.clip(result.x, 0.0, None)
    total = w.sum()
    return w / total if total > 1e-8 else np.ones(n) / n


class RollingMarkowitzStrategy(PeriodicRebalanceStrategy):
    """Long-only max-Sharpe mean-variance over a rolling `window`-day lookback
    of daily log returns, rebalanced every `rebalance_freq` days."""

    def __init__(self, n_assets: int, config, price_history: PriceHistory,
                 window: int = 252, rebalance_freq: int = 21, risk_free: float = 0.0):
        super().__init__(n_assets, config, rebalance_freq)
        self.price_history = price_history
        self.window = window
        self.risk_free = risk_free

    def _compute_target_weights(self) -> np.ndarray:
        prices = self.price_history.history_up_to(self._step)
        if len(prices) < self.window + 1:
            window_prices = prices
        else:
            window_prices = prices[-(self.window + 1):]

        if len(window_prices) < 3:
            return np.ones(self.n_assets) / self.n_assets

        log_rets = np.diff(np.log(window_prices), axis=0)
        mu = log_rets.mean(axis=0) * TRADING_DAYS_PER_YEAR
        cov = np.cov(log_rets.T) * TRADING_DAYS_PER_YEAR
        cov = np.atleast_2d(cov) + np.eye(self.n_assets) * 1e-6
        return _max_sharpe_long_only(mu, cov, self.risk_free)


class VolTargetStrategy(PeriodicRebalanceStrategy):
    """Equal-weight base portfolio, uniformly scaled so its trailing
    `vol_window`-day realized volatility matches `target_vol` annualized
    (leverage capped at `max_leverage`).

    rebalance_freq defaults to monthly (21), not daily: a daily cadence
    retrades the FULL current target every single day even when the
    underlying vol estimate has barely moved, and small-but-constant trades
    still pay the full proportional fee+slippage rate each time. On fold 2
    (2015-2017) this measured ~943%/year turnover and ~$423k in costs on a
    $1M book (42% of capital) -- daily rebalancing was the bug, not the
    vol-scaling math itself (realized vol still landed at 10.7% vs a 10%
    target). Monthly, like every other baseline here, is standard practice
    for a vol-targeting overlay and keeps turnover in a sane range."""

    def __init__(self, n_assets: int, config, price_history: PriceHistory,
                 target_vol: float = 0.10, vol_window: int = 20,
                 rebalance_freq: int = 21, max_leverage: float = 2.0):
        super().__init__(n_assets, config, rebalance_freq)
        self.price_history = price_history
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.base_weights = np.ones(n_assets) / n_assets

    def _compute_target_weights(self) -> np.ndarray:
        prices = self.price_history.history_up_to(self._step)
        if len(prices) < self.vol_window + 1:
            return self.base_weights

        window_prices = prices[-(self.vol_window + 1):]
        asset_log_rets = np.diff(np.log(window_prices), axis=0)
        port_rets = asset_log_rets @ self.base_weights
        realized_vol = float(np.std(port_rets, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        realized_vol = max(realized_vol, 1e-4)

        scale = min(self.target_vol / realized_vol, self.max_leverage)
        return self.base_weights * scale


class TimeSeriesMomentumStrategy(PeriodicRebalanceStrategy):
    """Per-asset long-or-cash: long any asset whose price is higher than
    `lookback` trading days ago, flat (0 weight) otherwise. All-cash if
    nothing has positive momentum.

    Each winner i is sized at (median universe vol / vol_i) / n_assets --
    NOT renormalized to sum to 1 across however many assets currently
    qualify. That distinction is the whole point: renormalizing-to-1 still
    hands a SINGLE winner the entire book regardless of its own volatility
    (equal-weight and "inverse-vol-then-renormalize" are identical when
    there's only one winner -- 1/1 either way). CL=F was the only asset
    with positive 12-month momentum from 2008-03 through 2008-09 (riding
    its H1 2008 spike); either of those schemes goes 100% into crude right
    before its H2 2008 crash, losing far more than buy-and-hold instead of
    protecting capital the way TSMOM is supposed to. Sizing against a fixed
    per-asset reference budget instead means a lone volatile winner gets a
    modest, risk-appropriate slice and the rest sits in cash -- gross
    exposure is capped at 1.0 (delevered pro rata if multiple winners'
    shares would sum past full investment) but is NOT forced up to 1.0
    when there are few winners or they're unusually volatile."""

    def __init__(self, n_assets: int, config, price_history: PriceHistory,
                 lookback: int, rebalance_freq: int, vol_window: int = 60):
        super().__init__(n_assets, config, rebalance_freq)
        self.price_history = price_history
        self.lookback = lookback
        self.vol_window = vol_window

    def _compute_target_weights(self) -> np.ndarray:
        prices = self.price_history.history_up_to(self._step)
        if len(prices) < self.lookback + 1:
            return np.zeros(self.n_assets)

        past = prices[-(self.lookback + 1)]
        current = prices[-1]
        is_positive = current > past
        if not is_positive.any():
            return np.zeros(self.n_assets)

        vol_lookback = min(self.vol_window + 1, len(prices))
        window_prices = prices[-vol_lookback:]
        log_rets = np.diff(np.log(window_prices), axis=0)
        vol = np.std(log_rets, axis=0, ddof=1)
        vol = np.maximum(vol, 1e-6)

        ref_vol = float(np.median(vol))  # cross-sectional reference, whole universe
        raw_weight = is_positive * (ref_vol / vol) / self.n_assets
        total = raw_weight.sum()
        if total > 1.0:
            raw_weight = raw_weight / total  # delever pro rata, never force UP to full investment
        return raw_weight


class HMMRegimeAllocationStrategy(PeriodicRebalanceStrategy):
    """2-3 state Gaussian HMM (regime_model.HMMRegimeModel) fit ONCE on the
    fold's train-period OHLCV. States are ranked by fitted mean return and
    mapped to a fixed exposure ladder (lowest-mean state = most defensive);
    the belief-weighted exposure scales an equal-weight base allocation.
    The HMM's own filtered belief is updated once per rebalance using the
    market ticker's (log-return, 20d realized vol) pair as of that day."""

    def __init__(self, n_assets: int, config, price_history: PriceHistory,
                 train_aligned_data: pd.DataFrame, n_states: int = 3,
                 rebalance_freq: int = 21):
        super().__init__(n_assets, config, rebalance_freq)
        self.price_history = price_history
        self.market_ticker = price_history.tickers[0]
        self.base_weights = np.ones(n_assets) / n_assets

        regime_config = RegimeConfig(model_type=RegimeModelType.HMM, n_regimes=n_states)
        self.hmm_model = HMMRegimeModel(regime_config)
        try:
            self.hmm_model.fit(train_aligned_data)
            self._fitted = True
        except Exception as e:
            print(f"WARNING: HMM fit failed ({e}) -- falling back to plain equal-weight for this fold.")
            self._fitted = False

        if self._fitted:
            means = self.hmm_model._hmm.means_[:, 0]  # mean return per state
            order = np.argsort(means)  # worst (lowest mean) -> best
            levels = np.linspace(0.3, 1.0, n_states)
            self.state_exposure = np.zeros(n_states)
            self.state_exposure[order] = levels

    def _compute_target_weights(self) -> np.ndarray:
        if not self._fitted:
            return self.base_weights  # HMM fit failed for this fold -- fall back to plain equal-weight

        market_prices = self.price_history.close_prices[self.market_ticker][:self.price_history.n_available(self._step)]
        if len(market_prices) < 22:
            belief = np.ones(self.hmm_model.n_regimes) / self.hmm_model.n_regimes
        else:
            log_ret = np.log(market_prices[-1] / market_prices[-2])
            recent_rets = np.diff(np.log(market_prices[-21:]))
            vol = float(np.std(recent_rets, ddof=1))
            belief = self.hmm_model.update(np.array([log_ret, vol]), prices=None)

        exposure = float(np.dot(belief, self.state_exposure))
        return self.base_weights * exposure

    def reset(self):
        super().reset()
        self.hmm_model.reset()


# =============================================================================
# Metrics
# =============================================================================

def compute_return_metrics(nav_history: List[float], risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Return/Sharpe/Sortino/Calmar/max_dd/CVaR5, all computed directly from
    nav_history so every metric shares the same return series and
    annualization convention (evaluation.BaselineEvaluator's own episode
    loop currently stubs sortino_ratio to 0.0 for baselines -- this avoids
    depending on that)."""
    nav = np.asarray(nav_history, dtype=float)
    log_rets = np.diff(np.log(nav))

    total_return = float(nav[-1] / nav[0] - 1)
    n_years = len(log_rets) / TRADING_DAYS_PER_YEAR
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = log_rets - rf_daily
    sharpe = float(np.mean(excess) / (np.std(excess, ddof=0) + 1e-8) * np.sqrt(TRADING_DAYS_PER_YEAR))

    downside = log_rets[log_rets < 0]
    downside_std = float(np.std(downside, ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 0 else 1e-8
    sortino = float(ann_return / downside_std)

    running_max = np.maximum.accumulate(nav)
    drawdown = (running_max - nav) / running_max
    max_dd = float(drawdown.max())
    calmar = float(ann_return / max(max_dd, 0.01))

    if len(log_rets) > 0:
        var5 = np.percentile(log_rets, 5)
        tail = log_rets[log_rets <= var5]
        cvar5 = float(tail.mean()) if len(tail) > 0 else float(var5)
    else:
        cvar5 = 0.0

    return {
        'return': total_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd': max_dd,
        'cvar5': cvar5,
    }


def _window_label(test_dates) -> str:
    start, end = test_dates
    return f"{start[:4]}-{end[:4]}"


# =============================================================================
# Strategy factory + driver
# =============================================================================

def build_strategies(n_assets: int, config, price_history: PriceHistory,
                      train_aligned: pd.DataFrame) -> Dict[str, BaselineStrategy]:
    return {
        'equal_weight_monthly': EqualWeightMonthlyStrategy(n_assets, config, rebalance_freq=21),
        'markowitz_252d_monthly': RollingMarkowitzStrategy(
            n_assets, config, price_history, window=252, rebalance_freq=21),
        'vol_target_10pct': VolTargetStrategy(
            n_assets, config, price_history, target_vol=0.10, vol_window=20, rebalance_freq=21),
        'tsmom_12m': TimeSeriesMomentumStrategy(
            n_assets, config, price_history, lookback=252, rebalance_freq=21),
        'tsmom_20d': TimeSeriesMomentumStrategy(
            n_assets, config, price_history, lookback=20, rebalance_freq=20),
        'hmm_regime': HMMRegimeAllocationStrategy(
            n_assets, config, price_history, train_aligned, n_states=3, rebalance_freq=21),
        'equal_weight_k5_band010': EqualWeightBandedStrategy(
            n_assets, config, rebalance_freq=5, no_trade_band=0.10),
    }


def run_baselines(tickers: Optional[List[str]] = None, output_dir: str = "./results/baselines",
                   lookback_days: int = 260) -> pd.DataFrame:
    if tickers is None:
        tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]

    windows = main_module.PAPER_WALK_FORWARD_WINDOWS
    all_dates = [d for w in windows for d in (w['train'][0], w['train'][1], w['test'][0], w['test'][1])]
    nominal_start, overall_end = min(all_dates), max(all_dates)
    overall_start = main_module._buffered_start_date(nominal_start)

    base_config = main_module.create_experiment_config(
        tickers=tickers, start_date=overall_start, end_date=overall_end,
        total_timesteps=1, use_synthetic=False, seed=1,
    )
    base_config.shield.shield_type = ShieldType.NONE  # baselines are unshielded, like the "none" RL arm

    features, prices_df, splits, returns, aligned_data = main_module.load_and_prepare_data(
        base_config, use_synthetic=False
    )
    prices = {t: prices_df[t].values for t in tickers}

    wf_evaluator = WalkForwardEvaluator(base_config)
    calendar_splits = wf_evaluator.generate_calendar_splits(aligned_data.index, windows)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for fold_idx, split in enumerate(calendar_splits):
        train_start_idx, train_end_idx = split['train']
        test_start_idx, test_end_idx = split['test']
        window_label = _window_label(split['test_dates'])

        test_features = features.iloc[test_start_idx:test_end_idx]
        test_prices = {t: p[test_start_idx:test_end_idx] for t, p in prices.items()}
        test_aligned = aligned_data.iloc[test_start_idx:test_end_idx]
        train_aligned = aligned_data.iloc[train_start_idx:train_end_idx]

        lookback_start_idx = max(0, test_start_idx - lookback_days)
        actual_lookback_len = test_start_idx - lookback_start_idx
        combined_prices = {t: p[lookback_start_idx:test_end_idx] for t, p in prices.items()}
        price_history = PriceHistory(combined_prices, lookback_len=actual_lookback_len, tickers=tickers)

        print(f"\n{'#' * 60}\n# FOLD {fold_idx + 1}: {window_label} "
              f"(lookback={actual_lookback_len}d, test={test_end_idx - test_start_idx}d)\n{'#' * 60}")

        strategies = build_strategies(len(tickers), base_config, price_history, train_aligned)

        for name, strategy in strategies.items():
            arm_config = copy.deepcopy(base_config)
            test_env = main_module.create_trading_env(
                test_features, test_prices, arm_config, aligned_data=test_aligned, mode="test"
            )
            evaluator = BaselineEvaluator(test_env, arm_config)
            episode_metrics, aggregate = evaluator.evaluate_baseline(strategy, n_episodes=1)
            ep = episode_metrics[0]

            row = compute_return_metrics(ep.nav_history, risk_free_rate=arm_config.evaluation.risk_free_rate)
            row.update({
                'strategy': name,
                'window': window_label,
                'violation_rate': ep.dd_violation_rate,
                'turnover': ep.avg_turnover,
            })
            all_rows.append(row)
            print(f"  {name:24s} return={row['return']:+.4f}  sharpe={row['sharpe']:+.3f}  "
                  f"max_dd={row['max_dd']:.3f}  turnover={row['turnover']:.4f}")

    df = pd.DataFrame(all_rows)
    df = df[['strategy', 'window', 'return', 'sharpe', 'sortino', 'calmar',
             'max_dd', 'violation_rate', 'cvar5', 'turnover']]

    csv_path = output_path / "baselines_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(df)} rows to {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate external (non-RL) baseline strategies")
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="./results/baselines")
    args = parser.parse_args()

    run_baselines(tickers=args.tickers, output_dir=args.output_dir)
