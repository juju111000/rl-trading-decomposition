"""
Evaluation Module for CMDP Trading System

Implements:
- Performance metrics (returns, Sharpe, drawdown, etc.)
- Safety metrics (violation rates, shield interventions)
- Baseline strategies (buy-and-hold, equal weight)
- Walk-forward evaluation
- Regime-aware performance analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
from collections import defaultdict

from config import Config, EvaluationConfig
from trading_env import TradingCMDPEnv
from networks import ActorCritic, SharedActorCritic
from portfolio_engine import PortfolioEngine, PortfolioState


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluation episode."""
    # Performance
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trading activity
    total_turnover: float
    avg_turnover: float
    n_trades: int
    avg_transaction_cost: float

    # Safety metrics
    dd_violation_rate: float
    dd_violation_magnitude: float
    turnover_violation_rate: float
    leverage_violation_rate: float
    time_in_unsafe_region: float

    # Shield metrics
    shield_intervention_rate: float
    avg_shield_magnitude: float
    shield_interventions_by_type: Dict[str, int] = field(default_factory=dict)

    # Regime metrics
    performance_by_regime: Dict[int, float] = field(default_factory=dict)
    performance_during_transitions: float = 0.0
    avg_entropy: float = 0.0

    # Time series
    nav_history: List[float] = field(default_factory=list)
    drawdown_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'total_turnover': self.total_turnover,
            'avg_turnover': self.avg_turnover,
            'n_trades': self.n_trades,
            'avg_transaction_cost': self.avg_transaction_cost,
            'dd_violation_rate': self.dd_violation_rate,
            'dd_violation_magnitude': self.dd_violation_magnitude,
            'turnover_violation_rate': self.turnover_violation_rate,
            'leverage_violation_rate': self.leverage_violation_rate,
            'time_in_unsafe_region': self.time_in_unsafe_region,
            'shield_intervention_rate': self.shield_intervention_rate,
            'avg_shield_magnitude': self.avg_shield_magnitude,
            'performance_by_regime': self.performance_by_regime,
            'performance_during_transitions': self.performance_during_transitions,
            'avg_entropy': self.avg_entropy,
        }


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple episodes."""
    n_episodes: int
    metrics: Dict[str, Tuple[float, float]]  # (mean, std) for each metric

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for metric, (mean, std) in self.metrics.items():
            data.append({
                'metric': metric,
                'mean': mean,
                'std': std,
                'ci_low': mean - 1.96 * std / np.sqrt(self.n_episodes),
                'ci_high': mean + 1.96 * std / np.sqrt(self.n_episodes),
            })
        return pd.DataFrame(data)


class Evaluator:
    """
    Evaluator for trained policies.

    Computes comprehensive metrics including:
    - Standard financial metrics
    - Safety/constraint metrics
    - Shield intervention statistics
    - Regime-aware performance
    """

    def __init__(
            self,
            env: TradingCMDPEnv,
            config: Config,
            device: str = 'cpu'
    ):
        self.env = env
        self.config = config
        self.eval_config = config.evaluation
        self.device = device

        # Constraint thresholds
        self.dd_limit = config.constraints.dd_limit
        self.turnover_limit = config.constraints.turnover_limit
        self.leverage_limit = config.constraints.leverage_limit

    def evaluate_policy(
            self,
            policy: Union[ActorCritic, SharedActorCritic],
            n_episodes: int = 1,
            deterministic: bool = True,
            store_trajectories: bool = True
    ) -> Tuple[List[EpisodeMetrics], AggregateMetrics, Optional[Dict]]:
        """
        Evaluate a trained policy over multiple episodes.

        Args:
            policy: Trained actor-critic network
            n_episodes: Number of evaluation episodes
            deterministic: If True, use mean action (no sampling)
            store_trajectories: If True, store trajectories for visualization

        Returns:
            Tuple of (episode_metrics_list, aggregate_metrics, trajectories)
            trajectories contains nav_history, returns, drawdown_history for best episode
        """
        policy.eval()
        episode_metrics = []

        # Store best trajectory for visualization
        best_trajectory = None
        best_return = float('-inf')

        for ep in range(n_episodes):
            metrics, trajectory = self._run_episode(policy, deterministic, store_trajectories)
            episode_metrics.append(metrics)

            # Keep best episode trajectory
            if store_trajectories and metrics.total_return > best_return:
                best_return = metrics.total_return
                best_trajectory = trajectory

        # Aggregate metrics
        aggregate = self._aggregate_metrics(episode_metrics)

        return episode_metrics, aggregate, best_trajectory

    def _run_episode(
            self,
            policy: Union[ActorCritic, SharedActorCritic],
            deterministic: bool,
            store_trajectories: bool = True
    ) -> Tuple[EpisodeMetrics, Optional[Dict]]:
        """Run a single evaluation episode."""
        obs, info = self.env.reset()

        # Tracking variables - use initial capital since portfolio was just reset
        nav_history = [self.env.portfolio.initial_capital]
        drawdown_history = [0.0]
        returns = []
        turnovers = []
        transaction_costs = []

        # Violation tracking
        dd_violations = []
        turnover_violations = []
        leverage_violations = []

        # Shield tracking
        shield_interventions = []
        shield_magnitudes = []
        shield_types = defaultdict(int)

        # Regime tracking
        regime_returns = defaultdict(list)
        entropies = []
        transition_returns = []

        done = False
        step = 0

        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Use get_action with deterministic flag
                action, _, _ = policy.get_action(obs_tensor, deterministic=deterministic)

                action_np = action.cpu().numpy().squeeze()

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Track metrics
            nav_history.append(info.get('nav', nav_history[-1]))
            drawdown_history.append(info.get('drawdown', 0.0))
            returns.append(info.get('log_return', 0.0))
            turnovers.append(info.get('turnover', 0.0))
            transaction_costs.append(info.get('transaction_cost', 0.0))

            # Track violations
            cost_vec = info.get('cost_vec', np.zeros(3))
            dd_violations.append(cost_vec[0] > 0)
            turnover_violations.append(cost_vec[1] > 0)
            leverage_violations.append(cost_vec[2] > 0)

            # Track shield. trading_env.py nests shield diagnostics under
            # info['shield'] (ShieldDiagnostics.to_dict()), not flat
            # top-level 'shield_modified'/'shield_magnitude'/'shield_reasons'
            # keys -- same bug class as trainer.py's collect_rollouts() (see
            # A.6): reading the wrong keys here silently zeroed out
            # shield_intervention_rate/avg_shield_magnitude in every
            # evaluation result, even when the shield was actively
            # intervening. info has no 'shield' key at all when no shield is
            # attached (ShieldType.NONE), hence the empty-dict default.
            shield_diag = info.get('shield', {})
            shield_interventions.append(shield_diag.get('was_modified', False))
            shield_magnitudes.append(shield_diag.get('modification_magnitude', 0.0))

            for reason in shield_diag.get('reasons', []):
                shield_types[reason] += 1

            # Track regime
            regime_belief = info.get('regime_belief', None)
            entropy = info.get('regime_entropy', 0.0)
            entropies.append(entropy)

            if regime_belief is not None:
                dominant_regime = np.argmax(regime_belief)
                regime_returns[dominant_regime].append(returns[-1])

                # High entropy = transition period
                if entropy > 0.5:
                    transition_returns.append(returns[-1])

            obs = next_obs
            step += 1

        # Compute metrics
        nav_array = np.array(nav_history)
        returns_array = np.array(returns)

        # Performance metrics
        total_return = (nav_array[-1] / nav_array[0]) - 1
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        volatility = np.std(returns_array) * np.sqrt(252)

        risk_free_rate = self.eval_config.risk_free_rate / 252
        excess_returns = returns_array - risk_free_rate
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)

        negative_returns = returns_array[returns_array < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 1e-8
        sortino = annualized_return / downside_std

        max_drawdown = max(drawdown_history)
        calmar = annualized_return / max(max_drawdown, 0.01)

        # Trading metrics
        total_turnover = sum(turnovers)
        avg_turnover = np.mean(turnovers)
        n_trades = sum(1 for t in turnovers if t > 0.01)
        avg_tc = np.mean(transaction_costs)

        # Violation metrics
        dd_violation_rate = np.mean(dd_violations)
        dd_violation_mag = np.mean([d for d, v in zip(drawdown_history, dd_violations) if v])
        if np.isnan(dd_violation_mag):
            dd_violation_mag = 0.0

        turnover_violation_rate = np.mean(turnover_violations)
        leverage_violation_rate = np.mean(leverage_violations)

        unsafe_steps = sum(1 for d in drawdown_history if d > self.dd_limit)
        time_unsafe = unsafe_steps / max(len(drawdown_history), 1)

        # Shield metrics
        shield_rate = np.mean(shield_interventions)
        avg_shield_mag = np.mean(shield_magnitudes) if shield_magnitudes else 0.0

        # Regime metrics
        perf_by_regime = {}
        for regime, rets in regime_returns.items():
            if len(rets) > 0:
                perf_by_regime[regime] = float(np.sum(rets))

        perf_transitions = np.sum(transition_returns) if transition_returns else 0.0
        avg_entropy = np.mean(entropies) if entropies else 0.0

        episode_metrics = EpisodeMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            total_turnover=total_turnover,
            avg_turnover=avg_turnover,
            n_trades=n_trades,
            avg_transaction_cost=avg_tc,
            dd_violation_rate=dd_violation_rate,
            dd_violation_magnitude=dd_violation_mag,
            turnover_violation_rate=turnover_violation_rate,
            leverage_violation_rate=leverage_violation_rate,
            time_in_unsafe_region=time_unsafe,
            shield_intervention_rate=shield_rate,
            avg_shield_magnitude=avg_shield_mag,
            shield_interventions_by_type=dict(shield_types),
            performance_by_regime=perf_by_regime,
            performance_during_transitions=perf_transitions,
            avg_entropy=avg_entropy,
            nav_history=nav_history,
            drawdown_history=drawdown_history,
        )

        # Build trajectory dictionary for visualization
        trajectory = None
        if store_trajectories:
            trajectory = {
                'nav_history': nav_history,
                'returns': returns,
                'drawdown_history': drawdown_history,
                'dd_violations': dd_violations,
                'turnover_violations': turnover_violations,
                'leverage_violations': leverage_violations,
            }

        return episode_metrics, trajectory

    def _aggregate_metrics(self, episode_metrics: List[EpisodeMetrics]) -> AggregateMetrics:
        """Aggregate metrics across episodes."""
        metrics_dict = defaultdict(list)

        for ep_metrics in episode_metrics:
            ep_dict = ep_metrics.to_dict()
            for key, value in ep_dict.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metrics_dict[key].append(value)

        aggregated = {}
        for key, values in metrics_dict.items():
            if len(values) > 0:
                aggregated[key] = (float(np.mean(values)), float(np.std(values)))

        return AggregateMetrics(
            n_episodes=len(episode_metrics),
            metrics=aggregated
        )


class BaselineStrategy:
    """Base class for baseline strategies."""

    def __init__(self, n_assets: int, config: Config):
        self.n_assets = n_assets
        self.config = config

    def get_action(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        """Return target weights."""
        raise NotImplementedError

    def reset(self):
        """Reset strategy state."""
        pass


class BuyAndHoldStrategy(BaselineStrategy):
    """Buy and hold strategy for a single asset or equal-weight portfolio."""

    def __init__(
            self,
            n_assets: int,
            config: Config,
            target_asset: Optional[int] = None,
            equal_weight: bool = True
    ):
        super().__init__(n_assets, config)
        self.target_asset = target_asset
        self.equal_weight = equal_weight
        self.initial_weights = None

    def get_action(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        if self.initial_weights is None:
            if self.target_asset is not None:
                # Single asset
                self.initial_weights = np.zeros(self.n_assets)
                self.initial_weights[self.target_asset] = 1.0
            elif self.equal_weight:
                # Equal weight
                self.initial_weights = np.ones(self.n_assets) / self.n_assets
            else:
                raise ValueError("Must specify target_asset or equal_weight")

        return self.initial_weights

    def reset(self):
        self.initial_weights = None


class MomentumStrategy(BaselineStrategy):
    """Simple momentum strategy based on recent returns."""

    def __init__(
            self,
            n_assets: int,
            config: Config,
            lookback: int = 20,
            rebalance_freq: int = 5
    ):
        super().__init__(n_assets, config)
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.step_count = 0
        self.current_weights = np.ones(n_assets) / n_assets

    def get_action(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        self.step_count += 1

        # Only rebalance periodically
        if self.step_count % self.rebalance_freq != 0:
            return self.current_weights

        # Extract momentum signals from observation
        # Assuming returns are in first n_assets features
        momentum = obs[:self.n_assets]

        # Long assets with positive momentum, short negative
        weights = np.tanh(momentum * 10)  # Scale and bound

        # Normalize to sum to 1 with some cash
        weights = weights / (np.sum(np.abs(weights)) + 1e-8) * 0.8

        self.current_weights = weights
        return weights

    def reset(self):
        self.step_count = 0
        self.current_weights = np.ones(self.n_assets) / self.n_assets


class MinVarianceStrategy(BaselineStrategy):
    """Minimum variance portfolio (rebalanced periodically)."""

    def __init__(
            self,
            n_assets: int,
            config: Config,
            lookback: int = 60,
            rebalance_freq: int = 20
    ):
        super().__init__(n_assets, config)
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.step_count = 0
        self.current_weights = np.ones(n_assets) / n_assets
        self.returns_buffer = []

    def get_action(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        self.step_count += 1

        # Track returns (assuming they're in observation)
        returns = obs[:self.n_assets]
        self.returns_buffer.append(returns)

        if len(self.returns_buffer) > self.lookback:
            self.returns_buffer.pop(0)

        # Only rebalance periodically
        if self.step_count % self.rebalance_freq != 0:
            return self.current_weights

        if len(self.returns_buffer) < self.lookback // 2:
            return self.current_weights

        # Compute covariance matrix
        returns_array = np.array(self.returns_buffer)
        cov = np.cov(returns_array.T)

        # Minimum variance weights (inverse volatility as approximation)
        try:
            inv_vol = 1.0 / (np.sqrt(np.diag(cov)) + 1e-8)
            weights = inv_vol / inv_vol.sum()
        except Exception:
            weights = np.ones(self.n_assets) / self.n_assets

        self.current_weights = weights
        return weights

    def reset(self):
        self.step_count = 0
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.returns_buffer = []


class BaselineEvaluator:
    """Evaluator for baseline strategies."""

    def __init__(self, env: TradingCMDPEnv, config: Config):
        self.env = env
        self.config = config
        self.n_assets = len(config.data.tickers)

    def evaluate_baseline(
            self,
            strategy: BaselineStrategy,
            n_episodes: int = 1
    ) -> Tuple[List[EpisodeMetrics], AggregateMetrics]:
        """Evaluate a baseline strategy."""
        episode_metrics = []

        for ep in range(n_episodes):
            metrics = self._run_baseline_episode(strategy)
            episode_metrics.append(metrics)
            strategy.reset()

        # Aggregate
        aggregator = Evaluator(self.env, self.config)
        aggregate = aggregator._aggregate_metrics(episode_metrics)

        return episode_metrics, aggregate

    def _run_baseline_episode(self, strategy: BaselineStrategy) -> EpisodeMetrics:
        """Run single episode with baseline strategy."""
        obs, info = self.env.reset()

        nav_history = [self.env.portfolio.initial_capital]
        drawdown_history = [0.0]
        returns = []
        turnovers = []
        transaction_costs = []
        dd_violations = []

        done = False

        while not done:
            action = strategy.get_action(obs, info)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            nav_history.append(info.get('nav', nav_history[-1]))
            drawdown_history.append(info.get('drawdown', 0.0))
            returns.append(info.get('log_return', 0.0))
            turnovers.append(info.get('turnover', 0.0))
            transaction_costs.append(info.get('transaction_cost', 0.0))

            cost_vec = info.get('cost_vec', np.zeros(3))
            dd_violations.append(cost_vec[0] > 0)

            obs = next_obs

        # Compute metrics (simplified)
        nav_array = np.array(nav_history)
        returns_array = np.array(returns)

        total_return = (nav_array[-1] / nav_array[0]) - 1
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        volatility = np.std(returns_array) * np.sqrt(252)
        sharpe = ann_return / (volatility + 1e-8)
        max_dd = max(drawdown_history)

        return EpisodeMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=0.0,
            max_drawdown=max_dd,
            calmar_ratio=ann_return / max(max_dd, 0.01),
            total_turnover=sum(turnovers),
            avg_turnover=np.mean(turnovers),
            n_trades=sum(1 for t in turnovers if t > 0.01),
            avg_transaction_cost=np.mean(transaction_costs),
            dd_violation_rate=np.mean(dd_violations),
            dd_violation_magnitude=0.0,
            turnover_violation_rate=0.0,
            leverage_violation_rate=0.0,
            time_in_unsafe_region=sum(1 for d in drawdown_history if d > self.config.constraints.dd_limit) / len(
                drawdown_history),
            shield_intervention_rate=0.0,
            avg_shield_magnitude=0.0,
            nav_history=nav_history,
            drawdown_history=drawdown_history,
        )

    def evaluate_all_baselines(
            self,
            n_episodes: int = 1
    ) -> Dict[str, Tuple[List[EpisodeMetrics], AggregateMetrics]]:
        """Evaluate all baseline strategies."""
        results = {}

        # Buy and hold per asset
        for i in range(self.n_assets):
            strategy = BuyAndHoldStrategy(self.n_assets, self.config, target_asset=i)
            name = f"buy_hold_{self.config.data.tickers[i]}"
            results[name] = self.evaluate_baseline(strategy, n_episodes)

        # Equal weight
        strategy = BuyAndHoldStrategy(self.n_assets, self.config, equal_weight=True)
        results['equal_weight'] = self.evaluate_baseline(strategy, n_episodes)

        # Momentum
        strategy = MomentumStrategy(self.n_assets, self.config)
        results['momentum'] = self.evaluate_baseline(strategy, n_episodes)

        # Min variance
        strategy = MinVarianceStrategy(self.n_assets, self.config)
        results['min_variance'] = self.evaluate_baseline(strategy, n_episodes)

        return results


def _resolve_date_range(dates: pd.DatetimeIndex, start: str, end: str) -> Tuple[int, int]:
    """
    Resolve an inclusive calendar date range to a half-open [start_idx, end_idx)
    integer index range against a sorted DatetimeIndex.

    Uses searchsorted rather than exact-date lookup so that boundaries
    falling on a weekend/holiday resolve to the nearest actual trading day
    inside the requested range, instead of raising a KeyError.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_idx = dates.searchsorted(start_ts, side='left')
    end_idx = dates.searchsorted(end_ts, side='right')
    return int(start_idx), int(end_idx)


class WalkForwardEvaluator:
    """
    Walk-forward evaluation for proper out-of-sample testing. Supports two
    ways to generate folds:

    1. Fixed-size rolling windows (generate_splits): splits data into
       - Train on window [t, t+train_size]
       - Validate on [t+train_size, t+train_size+val_size]
       - Test on [t+train_size+val_size, t+train_size+val_size+test_size]
       - Roll forward by step_size
       Used when run_walk_forward() is called with data_length.

    2. Explicit fixed calendar windows (generate_calendar_splits): each fold
       is a {'train': (start_date, end_date), 'test': (start_date, end_date)}
       pair resolved to integer indices against a real DatetimeIndex --
       e.g. main.py's PAPER_WALK_FORWARD_WINDOWS. Used when run_walk_forward()
       is called with an explicit `splits` list. There is no separate
       validation window in this mode.
    """

    def __init__(
            self,
            config: Config,
            train_window: int = 504,  # ~2 years
            val_window: int = 63,  # ~3 months
            test_window: int = 63,  # ~3 months
            step_size: int = 63  # Roll forward 3 months
    ):
        self.config = config
        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.step_size = step_size

    def generate_splits(
            self,
            total_length: int
    ) -> List[Dict[str, Tuple[int, int]]]:
        """
        Generate train/val/test split indices.

        Returns:
            List of dicts with 'train', 'val', 'test' keys
            Each value is (start_idx, end_idx) tuple
        """
        splits = []
        total_window = self.train_window + self.val_window + self.test_window

        start = 0
        while start + total_window <= total_length:
            train_start = start
            train_end = start + self.train_window
            val_start = train_end
            val_end = val_start + self.val_window
            test_start = val_end
            test_end = test_start + self.test_window

            splits.append({
                'train': (train_start, train_end),
                'val': (val_start, val_end),
                'test': (test_start, test_end),
            })

            start += self.step_size

        return splits

    def generate_calendar_splits(
            self,
            dates,
            windows: List[Dict[str, Tuple[str, str]]]
    ) -> List[Dict[str, Any]]:
        """
        Convert explicit calendar-date fold definitions into integer index
        ranges against `dates` (the full aligned dataset's DatetimeIndex),
        for fixed (non-rolling) walk-forward windows -- e.g. the paper's
        three train/test periods, as opposed to generate_splits()'s
        fixed-size rolling windows.

        Args:
            dates: DatetimeIndex (or array-like of dates) of the full
                aligned dataset, sorted ascending.
            windows: List of {'train': (start_date, end_date),
                'test': (start_date, end_date)} date strings, both ends
                inclusive.

        Returns:
            List of dicts: {'train': (start_idx, end_idx),
            'test': (start_idx, end_idx), 'train_dates': (start, end),
            'test_dates': (start, end)}. end_idx is exclusive (usable
            directly in .iloc[start:end] slicing).

        Raises:
            ValueError: if any fold's train range overlaps its test range,
                or either range contains no data.
        """
        dates = pd.DatetimeIndex(dates)
        splits = []

        for window in windows:
            train_start, train_end = window['train']
            test_start, test_end = window['test']

            train_start_idx, train_end_idx = _resolve_date_range(dates, train_start, train_end)
            test_start_idx, test_end_idx = _resolve_date_range(dates, test_start, test_end)

            if train_end_idx <= train_start_idx:
                raise ValueError(
                    f"Empty train range for window {window}: no data in "
                    f"{train_start}..{train_end}"
                )
            if test_end_idx <= test_start_idx:
                raise ValueError(
                    f"Empty test range for window {window}: no data in "
                    f"{test_start}..{test_end}"
                )
            if train_end_idx > test_start_idx:
                raise ValueError(
                    f"Walk-forward fold overlaps: train range {train_start}..{train_end} "
                    f"(rows {train_start_idx}:{train_end_idx}) overlaps test range "
                    f"{test_start}..{test_end} (rows {test_start_idx}:{test_end_idx})"
                )

            splits.append({
                'train': (train_start_idx, train_end_idx),
                'test': (test_start_idx, test_end_idx),
                'train_dates': (train_start, train_end),
                'test_dates': (test_start, test_end),
            })

        return splits

    def run_walk_forward(
            self,
            env_factory,
            trainer_factory,
            evaluator_factory,
            data_length: Optional[int] = None,
            splits: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward evaluation.

        Args:
            env_factory: Callable(start_idx, end_idx) -> env
            trainer_factory: Callable(env) -> trainer
            evaluator_factory: Callable(env) -> evaluator
            data_length: Total data length; used to generate fixed-size
                rolling-window splits via generate_splits() if `splits` is
                not given.
            splits: Explicit fold definitions (e.g. from
                generate_calendar_splits()) to use instead of rolling-window
                splits. Only 'train'/'test' keys are used.

        Returns:
            Dict with aggregate results
        """
        if splits is None:
            if data_length is None:
                raise ValueError(
                    "run_walk_forward requires either data_length (to generate "
                    "rolling-window splits) or an explicit splits list."
                )
            splits = self.generate_splits(data_length)

        all_test_metrics = []

        for i, split in enumerate(splits):
            print(f"\n=== Walk-forward fold {i + 1}/{len(splits)} ===")
            if 'val' in split:
                print(f"Train: {split['train']}, Val: {split['val']}, Test: {split['test']}")
            else:
                print(f"Train: {split.get('train_dates', split['train'])}, "
                      f"Test: {split.get('test_dates', split['test'])}")

            # Create train environment
            train_env = env_factory(split['train'][0], split['train'][1])

            # Train
            trainer = trainer_factory(train_env)
            trainer.train()

            # Get trained policy
            policy = trainer.get_policy()

            # Evaluate on test set
            test_env = env_factory(split['test'][0], split['test'][1])
            evaluator = evaluator_factory(test_env)

            test_metrics, _, _ = evaluator.evaluate_policy(policy, n_episodes=1)
            all_test_metrics.extend(test_metrics)

            print(f"Test return: {test_metrics[0].total_return:.4f}")
            print(f"Test Sharpe: {test_metrics[0].sharpe_ratio:.4f}")

        # Aggregate across all test periods
        aggregator = Evaluator(test_env, self.config)
        overall_aggregate = aggregator._aggregate_metrics(all_test_metrics)

        return {
            'n_folds': len(splits),
            'splits': splits,
            'test_metrics': [m.to_dict() for m in all_test_metrics],
            'aggregate': overall_aggregate.to_dataframe().to_dict(),
        }


def compare_methods(
        results: Dict[str, Tuple[List[EpisodeMetrics], AggregateMetrics]],
        metric_keys: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create comparison table across methods.

    Args:
        results: Dict mapping method_name -> (episode_metrics, aggregate)
        metric_keys: Metrics to compare (default: key performance metrics)

    Returns:
        DataFrame with comparison
    """
    if metric_keys is None:
        metric_keys = [
            'total_return', 'annualized_return', 'sharpe_ratio',
            'max_drawdown', 'dd_violation_rate', 'shield_intervention_rate'
        ]

    rows = []
    for method_name, (episode_metrics, aggregate) in results.items():
        row = {'method': method_name}
        for key in metric_keys:
            if key in aggregate.metrics:
                mean, std = aggregate.metrics[key]
                row[f'{key}_mean'] = mean
                row[f'{key}_std'] = std
            else:
                row[f'{key}_mean'] = np.nan
                row[f'{key}_std'] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index('method')

    return df


def export_results(
        results: Dict[str, Any],
        output_dir: Path,
        prefix: str = 'eval'
):
    """Export evaluation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export aggregate metrics
    if 'aggregate' in results:
        if isinstance(results['aggregate'], AggregateMetrics):
            df = results['aggregate'].to_dataframe()
        else:
            df = pd.DataFrame(results['aggregate'])
        df.to_csv(output_dir / f'{prefix}_aggregate.csv', index=False)

    # Export episode metrics
    if 'episode_metrics' in results:
        metrics_list = [
            m.to_dict() if isinstance(m, EpisodeMetrics) else m
            for m in results['episode_metrics']
        ]
        with open(output_dir / f'{prefix}_episodes.json', 'w') as f:
            json.dump(metrics_list, f, indent=2)

    # Export comparison table
    if 'comparison' in results:
        results['comparison'].to_csv(output_dir / f'{prefix}_comparison.csv')

    print(f"Results exported to {output_dir}")


# =============================================================================
# Visualization Functions
# =============================================================================

def generate_visualizations(
        results: Dict[str, Any],
        output_dir: Path,
        prefix: str = "eval"
):
    """
    Generate visualization plots for evaluation results.

    Args:
        results: Dictionary containing evaluation results with trajectories
        output_dir: Directory to save plots
        prefix: Prefix for output files
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Skipping visualizations.")
        return

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Color scheme - handles partial name matching
    color_map = {
        'ppo_lagrangian': '#2ecc71',
        'vanilla_ppo': '#3498db',
        'reward_shaping': '#e74c3c',
        'buy_hold': '#9b59b6',
        'equal_weight': '#f39c12',
        'momentum': '#1abc9c',
        'min_variance': '#34495e',
    }

    label_map = {
        'ppo_lagrangian': 'PPO-Lagrangian (Ours)',
        'vanilla_ppo': 'Vanilla PPO',
        'reward_shaping': 'Reward Shaping',
        'buy_hold': 'Buy & Hold',
        'equal_weight': 'Equal Weight',
        'momentum': 'Momentum',
        'min_variance': 'Min Variance',
    }

    def get_color(method):
        """Get color for method, handling partial matches."""
        for key, color in color_map.items():
            if key in method.lower():
                return color
        return 'gray'

    def get_label(method):
        """Get label for method, handling partial matches."""
        for key, label in label_map.items():
            if key in method.lower():
                return label
        return method

    # Extract trajectories if available
    trajectories = results.get('trajectories', {})

    if not trajectories:
        print("No trajectory data available for visualization.")
        return

    print(f"\nGenerating visualizations for {len(trajectories)} methods:")
    for method, traj in trajectories.items():
        nav_len = len(traj.get('nav_history', [])) if traj.get('nav_history') else 0
        print(f"  {method}: {nav_len} data points")

    # 1. Cumulative Returns Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for method, traj in trajectories.items():
        if 'nav_history' in traj and len(traj['nav_history']) > 0:
            nav = np.array(traj['nav_history'])
            cum_return = (nav / nav[0] - 1) * 100
            color = get_color(method)
            label = get_label(method)
            ax.plot(cum_return, label=label, color=color, linewidth=2)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / f'{prefix}_cumulative_returns.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {viz_dir / f'{prefix}_cumulative_returns.png'}")
    plt.close()

    # 2. Drawdown Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for method, traj in trajectories.items():
        if 'drawdown_history' in traj and len(traj['drawdown_history']) > 0:
            dd = np.array(traj['drawdown_history']) * 100
            color = get_color(method)
            label = get_label(method)
            ax.fill_between(range(len(dd)), 0, -dd, alpha=0.3, color=color)
            ax.plot(-dd, label=label, color=color, linewidth=1.5)

    ax.axhline(y=-10, color='red', linestyle='--', alpha=0.7, label='DD Limit (10%)')
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / f'{prefix}_drawdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {viz_dir / f'{prefix}_drawdown.png'}")
    plt.close()

    # 3. Metrics Bar Chart
    if 'comparison' in results and results['comparison'] is not None:
        comparison_df = results['comparison']

        fig, ax = plt.subplots(figsize=(14, 6))

        metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

        if available_metrics:
            x = np.arange(len(available_metrics))
            width = 0.15

            methods = comparison_df.index.tolist() if hasattr(comparison_df, 'index') else []

            for i, method in enumerate(methods):
                values = []
                for m in available_metrics:
                    val = comparison_df.loc[method, m] if method in comparison_df.index else 0
                    if 'drawdown' in m:
                        val = -val  # Flip for visualization
                    values.append(val)

                color = get_color(method)
                label = get_label(method)
                ax.bar(x + i * width, values, width, label=label, color=color)

            ax.set_xticks(x + width * len(methods) / 2)
            ax.set_xticklabels(['Total Return', 'Sharpe Ratio', 'Max DD (neg)'])
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(viz_dir / f'{prefix}_metrics.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {viz_dir / f'{prefix}_metrics.png'}")
            plt.close()

    # 4. Risk-Return Scatter
    if 'comparison' in results and results['comparison'] is not None:
        comparison_df = results['comparison']

        if 'volatility' in comparison_df.columns and 'annualized_return' in comparison_df.columns:
            fig, ax = plt.subplots(figsize=(10, 8))

            for method in comparison_df.index:
                vol = comparison_df.loc[method, 'volatility'] * 100
                ret = comparison_df.loc[method, 'annualized_return'] * 100
                sharpe = comparison_df.loc[method, 'sharpe_ratio'] if 'sharpe_ratio' in comparison_df.columns else 0

                color = get_color(method)
                label = get_label(method)

                ax.scatter(vol, ret, s=200, c=color, label=label, edgecolors='black', linewidths=1.5)
                ax.annotate(f'SR={sharpe:.2f}', (vol, ret), xytext=(10, 10),
                            textcoords='offset points', fontsize=9)

            ax.set_xlabel('Annual Volatility (%)', fontsize=12)
            ax.set_ylabel('Annual Return (%)', fontsize=12)
            ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            plt.tight_layout()
            plt.savefig(viz_dir / f'{prefix}_risk_return.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {viz_dir / f'{prefix}_risk_return.png'}")
            plt.close()

    print(f"\nAll visualizations saved to: {viz_dir}")