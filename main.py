#!/usr/bin/env python3
"""
Main Experiment Runner for Regime-Conditioned CMDP Trading System

Two entry points:
- run_full_experiment(): single ratio-based train/test split (60/20/20 of
  one continuous date range).
- run_calendar_walk_forward_experiment(): fixed calendar train/test windows
  per fold (e.g. PAPER_WALK_FORWARD_WINDOWS), one combined data load.

Both run the shield x trainer grid (run_method_grid): shield in
SHIELD_ARMS x trainer in TRAINER_ARMS, each arm trained/evaluated in its
own freshly-built environment from its own config, when run_ablations=True
(run_full_experiment) or by default (walk-forward). Pipeline stages:
1. Data loading and preprocessing
2. Feature engineering (computed once over the full series, then sliced --
   see load_and_prepare_data)
3. Model training (per grid arm)
4. Evaluation and comparison (vs. baseline strategies)
5. Report generation

Assets tested:
- ^GSPC (S&P 500)
- ^DJI (Dow Jones)
- 000001.SS (China A50 proxy - SSE Composite)
- ^GDAXI (Germany DAX -- Europe proxy; replaced ^STOXX50E, whose real Yahoo
  history only starts 2007-03-30 and truncated fold 1's train window to
  ~158 rows. See DATA_NOTES.md.)
- CL=F (WTI Crude Oil)
"""

import os
import sys
import json
import copy
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    Config, DataConfig, FeatureConfig, RegimeConfig, ConstraintConfig,
    ShieldConfig, ActionSpaceConfig, PPOConfig, TrainingConfig,
    EnvironmentConfig, EvaluationConfig, ActionSpaceType, RegimeModelType,
    BeliefFormationType, ShieldType, get_default_config, get_ablation_configs
)
from data_loader import DataLoader, SyntheticDataGenerator
from features import FeatureBuilder, PrecomputedFeatureExtractor
from regime_model import create_regime_model, RegimeAnalyzer
from portfolio_engine import PortfolioEngine, RiskManager
from policy_shield import create_shield
from trading_env import TradingCMDPEnv, make_env
from seeding import seed_everything

# Optional imports
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not installed. Training will not be available.")

if HAS_TORCH:
    from networks import create_actor_critic
    from trainer import create_trainer, PPOLagrangianTrainer
    from evaluation import (
        Evaluator, BaselineEvaluator, WalkForwardEvaluator,
        EpisodeMetrics, compare_methods, export_results
    )
    from logging_utils import (
        ExperimentLogger, plot_training_curves, plot_evaluation_results,
        plot_comparison_table, create_results_dashboard, generate_summary_report
    )


# =============================================================================
# Experiment Configuration
# =============================================================================

# Calendar-day buffer requested before a fold's nominal train start, so
# rolling-window features (normalization's min_periods=20, and the longer
# SMA/vol/correlation lookbacks) have real prior history to warm up on
# instead of being backfilled (audit #14: bfill is forward-looking leakage).
# Whatever warmup this buffer doesn't fully cover (e.g. the 200-day MA
# lookback, or a ticker's real listing date sitting later than the buffered
# start -- see DATA_NOTES.md) is handled by load_and_prepare_data's leading-
# NaN trim, not by extending this number indefinitely.
FEATURE_WARMUP_BUFFER_DAYS = 40


def _buffered_start_date(nominal_start: str, buffer_days: int = FEATURE_WARMUP_BUFFER_DAYS) -> str:
    """nominal_start minus buffer_days, for requesting warmup history ahead
    of a fold's actual train start (see FEATURE_WARMUP_BUFFER_DAYS)."""
    return (pd.Timestamp(nominal_start) - pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')


def create_experiment_config(
        tickers: List[str] = None,
        start_date: str = "2010-01-01",
        end_date: str = "2024-12-31",
        total_timesteps: int = 100000,
        use_synthetic: bool = False,
        seed: int = 42
) -> Config:
    """
    Create experiment configuration.

    Args:
        tickers: List of ticker symbols
        start_date: Training data start date
        end_date: Training data end date
        total_timesteps: Total training timesteps
        use_synthetic: Use synthetic data instead of real data
        seed: Random seed (drives config.training.seed, applied globally via
            seed_everything() before this config is used)

    Returns:
        Config object
    """
    if tickers is None:
        tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]

    config = Config(
        data=DataConfig(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            cache_dir="./data_cache"
        ),
        features=FeatureConfig(
            return_windows=[1, 5, 20],
            volatility_windows=[20, 60],
            ma_windows=[20, 50, 200],
            roc_windows=[20],
            use_atr=True,
            atr_window=14,
            use_cross_asset_features=True,
            correlation_window=60,
            normalization_window=252,
            clip_zscore=5.0
        ),
        regime=RegimeConfig(
            model_type=RegimeModelType.HEURISTIC,
            n_regimes=3,
            trend_ma_fast=20,
            trend_ma_slow=60,
            belief_ema_alpha=0.1,
            # Canonical belief-formation candidate (C: hard EMA + hysteresis),
            # locked in from the recalibration sweep -- see RegimeConfig's
            # own docstring/comment in config.py for the rationale.
            belief_formation=BeliefFormationType.HARD_EMA_HYSTERESIS,
            hysteresis_days=3,
        ),
        constraints=ConstraintConfig(
            dd_limit=0.20,  # RELAXED from 0.10
            dd_stop=0.30,  # RELAXED from 0.15
            turnover_limit=1.0,  # RELAXED from 0.5
            turnover_cap=2.0,  # RELAXED from 1.0
            leverage_limit=1.5,
            leverage_cap=2.0
        ),
        shield=ShieldConfig(
            shield_type=ShieldType.UNCERTAINTY_AWARE,
            entropy_tightening_kappa=0.3,  # RELAXED from 0.5
            high_vol_position_cap=0.5,  # RELAXED from 0.3
            high_vol_leverage_cap=1.5  # RELAXED from 1.0
        ),
        action_space=ActionSpaceConfig(
            action_type=ActionSpaceType.CONTINUOUS,
            discrete_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],
            allow_short=True,
            max_position_size=0.8  # RELAXED from 0.5
        ),
        ppo=PPOConfig(
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.05,  # Increased for more exploration
            policy_lr=1e-4,  # Reduced for stability
            value_lr=5e-4,
            dual_lr=0.005,  # Reduced for stability
            n_epochs=10
        ),
        training=TrainingConfig(
            total_timesteps=total_timesteps,
            rollout_length=2048,
            episode_length=252,
            log_frequency=1000,
            eval_frequency=10000,
            save_frequency=50000,
            seed=seed
        ),
        environment=EnvironmentConfig(
            initial_capital=1_000_000,
            transaction_fee_rate=0.001,
            slippage_rate=0.0005,
            use_volatility_slippage=True,
            slippage_vol_multiplier=0.1,
            reward_scale=100.0,  # Scale up rewards for better learning
            inactivity_penalty=0.02,  # Penalize for not being invested
            include_portfolio_state=True,
            include_regime_belief=True,
            include_entropy=True
        ),
        evaluation=EvaluationConfig(
            risk_free_rate=0.02,
            sharpe_window=252,
            benchmark_ticker="^GSPC",
            results_dir="./results",
            save_plots=True
        )
    )

    return config


def convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON export."""
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'to_dict'):
        return convert_to_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj


# =============================================================================
# Data Pipeline
# =============================================================================

def load_and_prepare_data(
        config: Config,
        use_synthetic: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load and prepare data for training.

    Returns:
        Tuple of (features, prices, splits, returns)
    """
    print("=" * 60)
    print("STEP 1: Data Loading")
    print("=" * 60)

    if use_synthetic:
        print("Using synthetic data...")
        # Span exactly the requested config date range so calendar-window
        # experiments (e.g. PAPER_WALK_FORWARD_WINDOWS) resolve against
        # synthetic data the same way they would against real data --
        # a fixed 2000-day/2010-start series wouldn't cover folds outside
        # that range (e.g. window 1's 2006-2007 train period).
        n_days = max(len(pd.bdate_range(config.data.start_date, config.data.end_date)), 1)
        gen = SyntheticDataGenerator(
            n_assets=len(config.data.tickers),
            n_days=n_days,
            start_date=config.data.start_date,
            seed=42
        )
        aligned_data, _ = gen.generate_regime_switching_data()

        # SyntheticDataGenerator creates ASSET_0, ASSET_1, etc.
        # Map them to our tickers
        generated_tickers = [f"ASSET_{i}" for i in range(len(config.data.tickers))]

        # Create returns from Close prices
        returns = pd.DataFrame()
        prices = {}
        for i, ticker in enumerate(config.data.tickers):
            gen_ticker = generated_tickers[i % len(generated_tickers)]
            close_prices = aligned_data[(gen_ticker, 'Close')]
            returns[ticker] = np.log(close_prices / close_prices.shift(1))
            prices[ticker] = close_prices.values
        returns = returns.dropna()

        # Rename columns in aligned_data to match our tickers
        new_columns = []
        for col in aligned_data.columns:
            gen_ticker, ohlcv = col
            idx = generated_tickers.index(gen_ticker) if gen_ticker in generated_tickers else 0
            new_ticker = config.data.tickers[idx % len(config.data.tickers)]
            new_columns.append((new_ticker, ohlcv))
        aligned_data.columns = pd.MultiIndex.from_tuples(new_columns)

        # Trim aligned_data to match returns (dropped first row due to NaN)
        aligned_data = aligned_data.iloc[1:]

        # Create splits
        n = len(returns)
        train_end = int(n * config.data.train_ratio)
        val_end = int(n * (config.data.train_ratio + config.data.val_ratio))

        splits = {
            'train': returns.iloc[:train_end],
            'val': returns.iloc[train_end:val_end],
            'test': returns.iloc[val_end:]
        }

    else:
        print(f"Loading real data for: {config.data.tickers}")
        print(f"Period: {config.data.start_date} to {config.data.end_date}")

        loader = DataLoader(config.data)

        try:
            aligned_data, returns, splits = loader.prepare_dataset()

            # Get price data from aligned data
            prices = {}
            for ticker in config.data.tickers:
                if (ticker, 'Close') in aligned_data.columns:
                    prices[ticker] = aligned_data[(ticker, 'Close')].values
                elif ticker in aligned_data.columns:
                    prices[ticker] = aligned_data[ticker].values
                else:
                    # Find any close price column for this ticker
                    close_cols = [c for c in aligned_data.columns if ticker in str(c) and 'Close' in str(c)]
                    if close_cols:
                        prices[ticker] = aligned_data[close_cols[0]].values
                    else:
                        print(f"Warning: Could not find price data for {ticker}")
                        prices[ticker] = None

        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Falling back to synthetic data...")
            return load_and_prepare_data(config, use_synthetic=True)

    print(f"\nData loaded successfully!")
    print(f"  Total samples: {len(returns)}")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    print(f"  Assets: {list(returns.columns)}")

    # Build features
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)

    feature_builder = FeatureBuilder(config.features)

    # Compute features ONCE over the full aligned OHLCV series using only
    # causal (backward-looking) rolling/expanding windows, then slice into
    # train/val/test. Computing each split separately (as before) "cold
    # starts" the rolling windows at the beginning of val/test with no real
    # prior history, which would be its own lookahead-leak class -- a single
    # pass over the full series means every row's features are computed from
    # actual preceding history.
    all_features = feature_builder.compute_features(aligned_data, fit_normalization=True)

    print(f"\nFeatures computed!")
    print(f"  Feature dimension: {all_features.shape[1]}")
    print(f"  Shape before warmup trim: {all_features.shape}")

    # Rolling windows (up to the 200-day MA) and normalization's
    # min_periods=20 leave a leading block of NaN rows at the very start of
    # the series -- FEATURE_WARMUP_BUFFER_DAYS requests real history ahead
    # of the nominal start to cover most of this, but whatever's left (e.g.
    # a ticker's real listing date sitting later than the buffer, or a
    # lookback longer than the buffer) is dropped here, never backfilled
    # (audit #14: bfill is forward-looking leakage). Verify the NaN is
    # actually a leading block before trimming -- NaN anywhere else would
    # mean a real bug, not a warmup gap.
    nan_row_positions = np.where(all_features.isna().any(axis=1).to_numpy())[0]
    if len(nan_row_positions) > 0:
        trim_start = int(nan_row_positions.max()) + 1
        if not np.array_equal(nan_row_positions, np.arange(trim_start)):
            raise ValueError(
                "Non-leading NaN found in computed features (expected only a "
                "leading warmup gap) -- investigate before trimming."
            )
        print(f"  Trimming {trim_start} leading warmup-NaN row(s) "
              f"({all_features.index[0].date()} to {all_features.index[trim_start - 1].date()})")
        all_features = all_features.iloc[trim_start:]
        aligned_data = aligned_data.iloc[trim_start:]
        for ticker in prices:
            if prices[ticker] is not None:
                prices[ticker] = prices[ticker][trim_start:]

    assert not np.isnan(all_features.values).any(), "NaN survived warmup trimming"
    print(f"  Shape after warmup trim: {all_features.shape}")

    # Create prices DataFrame from the (possibly trimmed) prices dict
    prices_df = pd.DataFrame(prices)

    return all_features, prices_df, splits, returns, aligned_data


# =============================================================================
# Environment Creation
# =============================================================================

def calibrate_regime(config: Config, train_aligned: pd.DataFrame) -> Config:
    """Calibrate the regime model's volatility thresholds on TRAINING data and
    record them in config.regime.calibrated_vol_thresholds (audit defect #9).
    For paths that build only a test env (untrained arm, re-evaluating a saved
    checkpoint whose config predates the fix). Consumes no random numbers."""
    model = create_regime_model(config.regime, input_dim=1)
    model.fit(train_aligned, mode="train")
    return config


def create_trading_env(
        features: pd.DataFrame,
        prices: Dict[str, np.ndarray],
        config: Config,
        aligned_data: pd.DataFrame = None,
        mode: str = "train"
) -> TradingCMDPEnv:
    """Create trading environment with regime model and shield.

    Args:
        features: Computed features DataFrame
        prices: Dict of ticker -> price array
        config: Configuration object. config.shield.shield_type determines
            the shield -- ShieldType.NONE means no shield is attached at all
            (create_shield returns None; the raw action is executed as-is).
        aligned_data: Multi-indexed OHLCV data for regime model fitting
        mode: 'train' or 'test'
    """

    print(f"\nCreating {mode} environment...")

    # For test mode, use full test period length
    if mode == "test":
        # Make a copy of config to modify episode length
        config = copy.deepcopy(config)
        config.training.episode_length = len(features)  # Use full test period
        print(f"  Using full test period: {len(features)} steps")

    # Create regime model
    regime_model = create_regime_model(
        config.regime,
        input_dim=features.shape[1]
    )

    # Fit regime model on OHLCV data if provided
    if aligned_data is not None:
        regime_model.fit(aligned_data, mode=mode)

    # Create shield
    shield = create_shield(
        config.shield,
        config.constraints,
        n_assets=len(config.data.tickers)
    )

    # Create environment
    env = TradingCMDPEnv(
        features=features.values,
        prices=prices,
        tickers=config.data.tickers,
        config=config,
        regime_model=regime_model,
        shield=shield,
        mode=mode
    )

    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")

    return env


# =============================================================================
# Training Pipeline
# =============================================================================

def train_model(
        env: TradingCMDPEnv,
        config: Config,
        trainer_type: str = "ppo_lagrangian",
        experiment_name: str = None,
        log_dir: str = "./results"
) -> Tuple[Any, List[Dict]]:
    """
    Train a model using specified trainer type.

    Args:
        env: Trading environment
        config: Configuration
        trainer_type: One of 'ppo_lagrangian', 'vanilla_ppo', 'reward_shaping'
        experiment_name: Name for logging
        log_dir: Directory for logs

    Returns:
        Tuple of (trained_policy, training_metrics)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for training")

    print(f"\n{'=' * 60}")
    print(f"Training: {trainer_type}")
    print(f"{'=' * 60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create trainer
    trainer = create_trainer(
        env=env,
        config=config,
        trainer_type=trainer_type,
        device=device
    )

    # Set up logging
    if experiment_name is None:
        experiment_name = f"{trainer_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    trainer.log_dir = Path(log_dir) / experiment_name
    trainer.log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = trainer.log_dir / "config.json"
    config.save_json(str(config_path))

    # Train
    print(f"\nStarting training for {config.training.total_timesteps} timesteps...")
    metrics_history = trainer.train()

    # Get trained policy
    policy = trainer.get_policy()

    print(f"\nTraining complete!")
    print(f"  Final policy loss: {metrics_history[-1].policy_loss:.4f}")
    print(f"  Final value loss: {metrics_history[-1].value_loss:.4f}")

    # Save training log to CSV
    try:
        training_log_path = trainer.log_dir / "training_log.csv"
        log_data = [m.to_dict() for m in metrics_history]
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(training_log_path, index=False)
        print(f"  Training log saved to: {training_log_path}")
    except Exception as e:
        print(f"  Could not save training log: {e}")

    return policy, [m.to_dict() for m in metrics_history]


# =============================================================================
# Evaluation Pipeline
# =============================================================================

def evaluate_model(
        env: TradingCMDPEnv,
        policy: Any,
        config: Config,
        n_episodes: int = 5
) -> Tuple[List[Dict], Dict, Optional[Dict]]:
    """
    Evaluate a trained policy.

    Returns:
        Tuple of (episode_metrics, aggregate_metrics, trajectory)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for evaluation")

    print(f"\nEvaluating policy over {n_episodes} episodes...")

    evaluator = Evaluator(env, config)
    episode_metrics, aggregate, trajectory = evaluator.evaluate_policy(
        policy,
        n_episodes=n_episodes,
        deterministic=True,
        store_trajectories=True
    )

    print(f"\nEvaluation Results:")
    print(f"  Total Return: {aggregate.metrics['total_return'][0]:.4f} ± {aggregate.metrics['total_return'][1]:.4f}")
    print(f"  Sharpe Ratio: {aggregate.metrics['sharpe_ratio'][0]:.4f} ± {aggregate.metrics['sharpe_ratio'][1]:.4f}")
    print(f"  Max Drawdown: {aggregate.metrics['max_drawdown'][0]:.4f} ± {aggregate.metrics['max_drawdown'][1]:.4f}")
    print(f"  DD Violation Rate: {aggregate.metrics['dd_violation_rate'][0]:.4f}")
    print(f"  Shield Intervention Rate: {aggregate.metrics['shield_intervention_rate'][0]:.4f}")

    return [m.to_dict() for m in episode_metrics], aggregate.metrics, trajectory


def evaluate_baselines(
        env: TradingCMDPEnv,
        config: Config,
        n_episodes: int = 5
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Evaluate all baseline strategies.

    Returns:
        Tuple of (metrics_dict, trajectories_dict)
    """

    if not HAS_TORCH:
        raise ImportError("PyTorch required for evaluation")

    print(f"\n{'=' * 60}")
    print("Evaluating Baselines")
    print(f"{'=' * 60}")

    baseline_evaluator = BaselineEvaluator(env, config)
    results = baseline_evaluator.evaluate_all_baselines(n_episodes=n_episodes)

    metrics_dict = {}
    trajectories_dict = {}

    print("\nBaseline Results:")
    for name, (episode_metrics, aggregate) in results.items():
        print(f"\n  {name}:")
        print(f"    Return: {aggregate.metrics['total_return'][0]:.4f}")
        print(f"    Sharpe: {aggregate.metrics['sharpe_ratio'][0]:.4f}")
        print(f"    Max DD: {aggregate.metrics['max_drawdown'][0]:.4f}")

        metrics_dict[name] = aggregate.metrics

        # Extract trajectory from first episode for visualization
        if episode_metrics and hasattr(episode_metrics[0], 'nav_history'):
            nav_hist = episode_metrics[0].nav_history
            dd_hist = episode_metrics[0].drawdown_history
            trajectories_dict[name] = {
                'nav_history': nav_hist if isinstance(nav_hist, list) else list(nav_hist),
                'drawdown_history': dd_hist if isinstance(dd_hist, list) else list(dd_hist),
                'returns': [],
            }
            print(f"    Trajectory length: {len(nav_hist)}")
        else:
            print(f"    WARNING: No trajectory data for {name}")

    return metrics_dict, trajectories_dict


# =============================================================================
# Shield x Trainer Grid
# =============================================================================

# The full ablation grid: shield in {none, uncertainty_aware} x trainer in
# {vanilla_ppo, reward_shaping, ppo_lagrangian}. Each arm gets its own
# freshly-built train_env/test_env from its own config (see run_method_grid)
# so shield behavior is never silently shared across methods that are
# supposed to differ -- e.g. "vanilla_ppo" with shield_arm="none" genuinely
# runs unshielded, rather than inheriting whatever shield a single env
# created earlier happened to have attached.
SHIELD_ARMS = ["none", "uncertainty_aware"]
TRAINER_ARMS = ["vanilla_ppo", "reward_shaping", "ppo_lagrangian"]


def _make_arm_config(base_config: Config, shield_arm: str) -> Config:
    """Deep copy of base_config with shield_type set for the requested arm."""
    config = copy.deepcopy(base_config)
    config.shield.shield_type = (
        ShieldType.NONE if shield_arm == "none" else ShieldType.UNCERTAINTY_AWARE
    )
    return config


def _load_completed_arm(arm_log_dir: Path) -> Optional[Dict[str, Any]]:
    """If arm_log_dir holds a fully-completed run (eval_result.json present,
    written only after both training and evaluation succeed -- see
    _save_arm_result), reload it from disk: training metrics from
    training_log.csv, evaluation from eval_result.json, trajectory from
    trajectory.json (empty if that file wasn't kept). Returns None if the
    arm isn't fully completed (missing eval_result.json), meaning it must
    be (re-)trained from scratch -- there is no mid-training checkpoint to
    resume from."""
    eval_result_path = arm_log_dir / "eval_result.json"
    if not eval_result_path.exists():
        return None

    with open(eval_result_path) as f:
        evaluation = json.load(f)

    training_log_path = arm_log_dir / "training_log.csv"
    train_metrics = pd.read_csv(training_log_path).to_dict('records') if training_log_path.exists() else []

    trajectory_path = arm_log_dir / "trajectory.json"
    trajectory = json.load(open(trajectory_path)) if trajectory_path.exists() else {}

    return {'training': train_metrics, 'evaluation': evaluation, 'trajectory': trajectory}


def _save_arm_result(arm_log_dir: Path, evaluation: Dict[str, Any], trajectory: Any) -> None:
    """Persist an arm's evaluation (+ trajectory) once training+eval both
    succeed. eval_result.json's presence is the sole "this arm is fully
    done" marker _load_completed_arm checks for on resume."""
    with open(arm_log_dir / "eval_result.json", 'w') as f:
        json.dump(convert_to_serializable(evaluation), f, indent=2, default=str)
    with open(arm_log_dir / "trajectory.json", 'w') as f:
        json.dump(convert_to_serializable(trajectory), f, indent=2, default=str)


def run_method_grid(
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_prices: Dict[str, np.ndarray],
        test_prices: Dict[str, np.ndarray],
        train_aligned: pd.DataFrame,
        test_aligned: pd.DataFrame,
        base_config: Config,
        log_dir: Path,
        shield_arms: List[str] = None,
        trainer_arms: List[str] = None,
        n_eval_episodes: int = 5,
        resume: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Train + evaluate every (shield, trainer) arm, each with its own config
    and its own freshly-built train_env/test_env.

    Arm names are "{shield_arm}/{trainer_arm}" (e.g. "none/vanilla_ppo"),
    used both as the dict key here and as train_model()'s experiment_name --
    train_model() saves each arm's own config.json under
    log_dir/{arm_name}/config.json, so it reflects that arm's actual
    shield_type rather than some other arm's.

    Args:
        train_features, test_features: Precomputed feature DataFrames.
        train_prices, test_prices: Dict of ticker -> price array.
        train_aligned, test_aligned: Multi-indexed OHLCV, for regime fitting.
        base_config: Config to derive each arm's config from (only
            shield.shield_type differs per arm).
        log_dir: Root directory; each arm logs to log_dir/{arm_name}/.
        shield_arms: Defaults to SHIELD_ARMS.
        trainer_arms: Defaults to TRAINER_ARMS.
        n_eval_episodes: Evaluation episodes per arm.
        resume: If True, an arm whose log_dir/{arm_name}/eval_result.json
            already exists (a prior fully-completed run) is loaded from disk
            and skipped rather than retrained -- for restarting a
            multi-hour grid without redoing finished work. An arm that only
            got partway (e.g. training_log.csv present but no
            eval_result.json, because the process died between training
            and evaluation) is NOT considered complete and is retrained
            from scratch, since no mid-training checkpoint exists.

    Returns:
        Dict mapping arm_name -> {'shield', 'trainer', 'training',
        'evaluation', 'trajectory'}.
    """
    if shield_arms is None:
        shield_arms = SHIELD_ARMS
    if trainer_arms is None:
        trainer_arms = TRAINER_ARMS

    grid_results = {}

    for shield_arm in shield_arms:
        for trainer_arm in trainer_arms:
            arm_name = f"{shield_arm}/{trainer_arm}"
            arm_config = _make_arm_config(base_config, shield_arm)
            arm_log_dir = Path(log_dir) / arm_name

            if resume:
                completed = _load_completed_arm(arm_log_dir)
                if completed is not None:
                    print(f"\n# ARM: shield={shield_arm}, trainer={trainer_arm} -- "
                          f"SKIP (already completed, loaded from {arm_log_dir})")
                    grid_results[arm_name] = {
                        'shield': shield_arm, 'trainer': trainer_arm, **completed,
                    }
                    continue

            print(f"\n{'#' * 60}")
            print(f"# ARM: shield={shield_arm}, trainer={trainer_arm}")
            print(f"{'#' * 60}")

            try:
                train_env = create_trading_env(
                    train_features, train_prices, arm_config,
                    aligned_data=train_aligned, mode="train"
                )
                test_env = create_trading_env(
                    test_features, test_prices, arm_config,
                    aligned_data=test_aligned, mode="test"
                )

                policy, train_metrics = train_model(
                    env=train_env,
                    config=arm_config,
                    trainer_type=trainer_arm,
                    experiment_name=arm_name,
                    log_dir=str(log_dir)
                )

                episode_metrics, aggregate, trajectory = evaluate_model(
                    env=test_env,
                    policy=policy,
                    config=arm_config,
                    n_episodes=n_eval_episodes
                )

                evaluation = {'episodes': episode_metrics, 'aggregate': aggregate}
                _save_arm_result(arm_log_dir, evaluation, trajectory)

                grid_results[arm_name] = {
                    'shield': shield_arm,
                    'trainer': trainer_arm,
                    'training': train_metrics,
                    'evaluation': evaluation,
                    'trajectory': trajectory,
                }

            except Exception as e:
                print(f"Error running arm {arm_name}: {e}")
                import traceback
                traceback.print_exc()

    return grid_results


# =============================================================================
# Shield Component Ablation
# =============================================================================

# Ablates individual mechanisms inside UncertaintyAwareShield, holding the
# trainer fixed (ppo_lagrangian by default). Planned scope (see
# run_paper_experiments.py): window 1 only, 5 seeds -- NOT part of the main
# shield x trainer grid above, and not run automatically by anything in this
# module; wired here so the grid script can pick it up when asked to.
SHIELD_COMPONENT_VARIANTS = {
    "full": dict(use_entropy_tightening=True, use_bear_cap=True, use_dd_scaling=True),
    "no_entropy_term": dict(use_entropy_tightening=False, use_bear_cap=True, use_dd_scaling=True),
    "no_bear_cap": dict(use_entropy_tightening=True, use_bear_cap=False, use_dd_scaling=True),
    "no_dd_scaling": dict(use_entropy_tightening=True, use_bear_cap=True, use_dd_scaling=False),
}


def _make_shield_component_config(base_config: Config, variant_name: str) -> Config:
    """Deep copy of base_config with shield_type=UNCERTAINTY_AWARE and the
    named component toggle applied (see SHIELD_COMPONENT_VARIANTS)."""
    config = copy.deepcopy(base_config)
    config.shield.shield_type = ShieldType.UNCERTAINTY_AWARE
    for key, value in SHIELD_COMPONENT_VARIANTS[variant_name].items():
        setattr(config.shield, key, value)
    return config


def run_shield_component_grid(
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_prices: Dict[str, np.ndarray],
        test_prices: Dict[str, np.ndarray],
        train_aligned: pd.DataFrame,
        test_aligned: pd.DataFrame,
        base_config: Config,
        log_dir: Path,
        shield_variants: List[str] = None,
        trainer_arm: str = "ppo_lagrangian",
        n_eval_episodes: int = 5,
        resume: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Ablate UncertaintyAwareShield's components on a single trainer type,
    each variant getting its own freshly-built train_env/test_env -- mirrors
    run_method_grid()'s per-arm isolation, but varies shield component
    toggles instead of (shield_arm, trainer_arm).

    Arm names are the variant name itself (e.g. "no_bear_cap"), used as
    train_model()'s experiment_name so each variant's config.json lands
    under log_dir/{variant_name}/config.json with its own toggle values.

    resume: see run_method_grid's resume -- same eval_result.json-based
    skip/reload behavior, one level down (variant_name instead of arm_name).

    Returns:
        Dict mapping variant_name -> {'shield_variant', 'trainer', 'training',
        'evaluation', 'trajectory'}.
    """
    if shield_variants is None:
        shield_variants = list(SHIELD_COMPONENT_VARIANTS.keys())

    grid_results = {}

    for variant_name in shield_variants:
        arm_config = _make_shield_component_config(base_config, variant_name)
        arm_log_dir = Path(log_dir) / variant_name

        if resume:
            completed = _load_completed_arm(arm_log_dir)
            if completed is not None:
                print(f"\n# SHIELD COMPONENT VARIANT: {variant_name} -- "
                      f"SKIP (already completed, loaded from {arm_log_dir})")
                grid_results[variant_name] = {
                    'shield_variant': variant_name, 'trainer': trainer_arm, **completed,
                }
                continue

        print(f"\n{'#' * 60}")
        print(f"# SHIELD COMPONENT VARIANT: {variant_name} (trainer={trainer_arm})")
        print(f"{'#' * 60}")

        try:
            train_env = create_trading_env(
                train_features, train_prices, arm_config,
                aligned_data=train_aligned, mode="train"
            )
            test_env = create_trading_env(
                test_features, test_prices, arm_config,
                aligned_data=test_aligned, mode="test"
            )

            policy, train_metrics = train_model(
                env=train_env,
                config=arm_config,
                trainer_type=trainer_arm,
                experiment_name=variant_name,
                log_dir=str(log_dir)
            )

            episode_metrics, aggregate, trajectory = evaluate_model(
                env=test_env,
                policy=policy,
                config=arm_config,
                n_episodes=n_eval_episodes
            )

            evaluation = {'episodes': episode_metrics, 'aggregate': aggregate}
            _save_arm_result(arm_log_dir, evaluation, trajectory)

            grid_results[variant_name] = {
                'shield_variant': variant_name,
                'trainer': trainer_arm,
                'training': train_metrics,
                'evaluation': evaluation,
                'trajectory': trajectory,
            }

        except Exception as e:
            print(f"Error running shield variant {variant_name}: {e}")
            import traceback
            traceback.print_exc()

    return grid_results


def run_shield_ablation_experiment(
        tickers: List[str] = None,
        window: Optional[Dict[str, Tuple[str, str]]] = None,
        shield_variants: List[str] = None,
        trainer_arm: str = "ppo_lagrangian",
        total_timesteps: int = 50000,
        use_synthetic: bool = False,
        output_dir: str = "./results",
        seed: int = 42,
        n_eval_episodes: int = 5,
        config_overrides: Optional[Dict[str, Any]] = None,
        resume: bool = False
) -> Dict[str, Any]:
    """
    Run the shield-component ablation for a single calendar window (window 1
    by default -- PAPER_WALK_FORWARD_WINDOWS[0]) and a single trainer type.
    Mirrors run_calendar_walk_forward_experiment()'s data-loading pattern
    (one combined load, causal features, calendar-resolved train/test split)
    but for exactly one window and one run_shield_component_grid() call
    instead of the full shield x trainer grid.

    config_overrides: see run_calendar_walk_forward_experiment's -- applied
        uniformly to every shield-component variant.
    resume: if True, output_dir is used as-is (no fresh datetime.now()
        subdirectory) and any variant with a completed eval_result.json
        under it is skipped -- see run_method_grid's resume for the same
        contract one level up. Callers doing a resumable multi-run job must
        pass a stable, already-timestamped output_dir themselves.

    Returns:
        Dict with 'window' and 'variants' (per-variant test metrics, keyed
        by variant name).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for training")

    if tickers is None:
        tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]
    if window is None:
        window = PAPER_WALK_FORWARD_WINDOWS[0]
    if shield_variants is None:
        shield_variants = list(SHIELD_COMPONENT_VARIANTS.keys())

    print("\n" + "=" * 70)
    print("  SHIELD COMPONENT ABLATION")
    print("=" * 70)
    print(f"Assets: {tickers}")
    print(f"Window: {window}")
    print(f"Trainer: {trainer_arm}")
    print(f"Shield variants: {shield_variants}")

    seed_everything(seed)

    output_path = Path(output_dir) if resume else Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path.mkdir(parents=True, exist_ok=True)

    train_start, train_end = window['train']
    test_start, test_end = window['test']
    buffered_start = _buffered_start_date(train_start)
    print(f"Requesting data from {buffered_start} (nominal train start {train_start} minus "
          f"{FEATURE_WARMUP_BUFFER_DAYS}-day feature-warmup buffer)")

    base_config = create_experiment_config(
        tickers=tickers, start_date=buffered_start, end_date=test_end,
        total_timesteps=total_timesteps, use_synthetic=use_synthetic, seed=seed
    )

    if config_overrides:
        print(f"Applying config_overrides uniformly to every shield variant: {config_overrides}")
        for dotted_path, value in config_overrides.items():
            section, field_name = dotted_path.split('.', 1)
            setattr(getattr(base_config, section), field_name, value)

    base_config.save_json(str(output_path / "config.json"))

    features, prices_df, splits, returns, aligned_data = load_and_prepare_data(base_config, use_synthetic)

    prices = {}
    for ticker in tickers:
        if ticker in prices_df.columns:
            prices[ticker] = prices_df[ticker].values
        else:
            prices[ticker] = prices_df.iloc[:, tickers.index(ticker)].values

    wf_evaluator = WalkForwardEvaluator(base_config)
    split = wf_evaluator.generate_calendar_splits(aligned_data.index, [window])[0]
    train_start_idx, train_end_idx = split['train']
    test_start_idx, test_end_idx = split['test']

    train_features = features.iloc[train_start_idx:train_end_idx]
    test_features = features.iloc[test_start_idx:test_end_idx]
    train_prices = {t: p[train_start_idx:train_end_idx] for t, p in prices.items()}
    test_prices = {t: p[test_start_idx:test_end_idx] for t, p in prices.items()}
    train_aligned = aligned_data.iloc[train_start_idx:train_end_idx]
    test_aligned = aligned_data.iloc[test_start_idx:test_end_idx]

    grid_results = run_shield_component_grid(
        train_features, test_features, train_prices, test_prices,
        train_aligned, test_aligned,
        base_config=base_config,
        log_dir=output_path,
        shield_variants=shield_variants,
        trainer_arm=trainer_arm,
        n_eval_episodes=n_eval_episodes,
        resume=resume
    )

    results = {'window': window, 'variants': grid_results}

    serializable_results = convert_to_serializable(
        {'window': window, 'variants': {k: v['evaluation'] for k, v in grid_results.items()}}
    )
    with open(output_path / "shield_ablation_results.json", 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nShield ablation complete! Results saved to: {output_path}")

    return results


# =============================================================================
# Inactivity Penalty Ablation
# =============================================================================

# Ablates the inactivity-penalty reward-shaping term (trading_env.py step(),
# gated behind environment.use_inactivity_penalty -- see
# tests/test_inactivity_penalty_flag.py) on shield=none/ppo_lagrangian, the
# arm this whole investigation centered on, now that the Phase 2-4
# execution-model fixes are in place. Window 1 only, same scope as
# SHIELD_COMPONENT_VARIANTS; wired into run_paper_experiments.py the same
# way (its own grid, own seed list, included by default alongside the
# shield-component ablation).
INACTIVITY_PENALTY_VARIANTS = {
    "with_penalty": dict(use_inactivity_penalty=True),
    "without_penalty": dict(use_inactivity_penalty=False),
}


def _make_inactivity_penalty_config(
        base_config: Config, variant_name: str,
        shield_arm: str = "none", trainer_arm: str = "ppo_lagrangian"
) -> Config:
    """Deep copy of base_config with shield_arm's shield_type (via
    _make_arm_config) and the named inactivity-penalty toggle applied (see
    INACTIVITY_PENALTY_VARIANTS). trainer_arm isn't set on the config itself
    (it's selected at train_model() call time, same as every other grid
    here) -- accepted only so callers have one place to see both axes."""
    config = _make_arm_config(base_config, shield_arm)
    for key, value in INACTIVITY_PENALTY_VARIANTS[variant_name].items():
        setattr(config.environment, key, value)
    return config


def run_inactivity_penalty_grid(
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_prices: Dict[str, np.ndarray],
        test_prices: Dict[str, np.ndarray],
        train_aligned: pd.DataFrame,
        test_aligned: pd.DataFrame,
        base_config: Config,
        log_dir: Path,
        variants: List[str] = None,
        shield_arm: str = "none",
        trainer_arm: str = "ppo_lagrangian",
        n_eval_episodes: int = 5,
        resume: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Ablate the inactivity-penalty reward-shaping term, holding shield_arm
    and trainer_arm fixed -- mirrors run_shield_component_grid's structure
    exactly, varying environment.use_inactivity_penalty instead of shield
    component toggles.

    Arm names are the variant name itself (e.g. "without_penalty"), used as
    train_model()'s experiment_name so each variant's config.json lands
    under log_dir/{variant_name}/config.json.

    resume: see run_method_grid's resume -- same eval_result.json-based
    skip/reload behavior.

    Returns:
        Dict mapping variant_name -> {'inactivity_variant', 'shield',
        'trainer', 'training', 'evaluation', 'trajectory'}.
    """
    if variants is None:
        variants = list(INACTIVITY_PENALTY_VARIANTS.keys())

    grid_results = {}

    for variant_name in variants:
        arm_config = _make_inactivity_penalty_config(base_config, variant_name, shield_arm, trainer_arm)
        arm_log_dir = Path(log_dir) / variant_name

        if resume:
            completed = _load_completed_arm(arm_log_dir)
            if completed is not None:
                print(f"\n# INACTIVITY PENALTY VARIANT: {variant_name} -- "
                      f"SKIP (already completed, loaded from {arm_log_dir})")
                grid_results[variant_name] = {
                    'inactivity_variant': variant_name, 'shield': shield_arm,
                    'trainer': trainer_arm, **completed,
                }
                continue

        print(f"\n{'#' * 60}")
        print(f"# INACTIVITY PENALTY VARIANT: {variant_name} (shield={shield_arm}, trainer={trainer_arm})")
        print(f"{'#' * 60}")

        try:
            train_env = create_trading_env(
                train_features, train_prices, arm_config,
                aligned_data=train_aligned, mode="train"
            )
            test_env = create_trading_env(
                test_features, test_prices, arm_config,
                aligned_data=test_aligned, mode="test"
            )

            policy, train_metrics = train_model(
                env=train_env,
                config=arm_config,
                trainer_type=trainer_arm,
                experiment_name=variant_name,
                log_dir=str(log_dir)
            )

            episode_metrics, aggregate, trajectory = evaluate_model(
                env=test_env,
                policy=policy,
                config=arm_config,
                n_episodes=n_eval_episodes
            )

            evaluation = {'episodes': episode_metrics, 'aggregate': aggregate}
            _save_arm_result(arm_log_dir, evaluation, trajectory)

            grid_results[variant_name] = {
                'inactivity_variant': variant_name,
                'shield': shield_arm,
                'trainer': trainer_arm,
                'training': train_metrics,
                'evaluation': evaluation,
                'trajectory': trajectory,
            }

        except Exception as e:
            print(f"Error running inactivity penalty variant {variant_name}: {e}")
            import traceback
            traceback.print_exc()

    return grid_results


def run_inactivity_penalty_ablation_experiment(
        tickers: List[str] = None,
        window: Optional[Dict[str, Tuple[str, str]]] = None,
        variants: List[str] = None,
        shield_arm: str = "none",
        trainer_arm: str = "ppo_lagrangian",
        total_timesteps: int = 50000,
        use_synthetic: bool = False,
        output_dir: str = "./results",
        seed: int = 42,
        n_eval_episodes: int = 5,
        config_overrides: Optional[Dict[str, Any]] = None,
        resume: bool = False
) -> Dict[str, Any]:
    """
    Run the inactivity-penalty ablation for a single calendar window (window
    1 by default) and a single (shield_arm, trainer_arm) pair (default
    "none"/"ppo_lagrangian" -- the arm this investigation centered on).
    Mirrors run_shield_ablation_experiment's data-loading pattern exactly.

    config_overrides: see run_calendar_walk_forward_experiment's -- applied
        uniformly to both variants.
    resume: if True, output_dir is used as-is (no fresh datetime.now()
        subdirectory) and any variant with a completed eval_result.json
        under it is skipped -- see run_method_grid's resume for the same
        contract one level up.

    Returns:
        Dict with 'window' and 'variants' (per-variant test metrics, keyed
        by variant name).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for training")

    if tickers is None:
        tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]
    if window is None:
        window = PAPER_WALK_FORWARD_WINDOWS[0]
    if variants is None:
        variants = list(INACTIVITY_PENALTY_VARIANTS.keys())

    print("\n" + "=" * 70)
    print("  INACTIVITY PENALTY ABLATION")
    print("=" * 70)
    print(f"Assets: {tickers}")
    print(f"Window: {window}")
    print(f"Shield/Trainer: {shield_arm}/{trainer_arm}")
    print(f"Variants: {variants}")

    seed_everything(seed)

    output_path = Path(output_dir) if resume else Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path.mkdir(parents=True, exist_ok=True)

    train_start, train_end = window['train']
    test_start, test_end = window['test']
    buffered_start = _buffered_start_date(train_start)
    print(f"Requesting data from {buffered_start} (nominal train start {train_start} minus "
          f"{FEATURE_WARMUP_BUFFER_DAYS}-day feature-warmup buffer)")

    base_config = create_experiment_config(
        tickers=tickers, start_date=buffered_start, end_date=test_end,
        total_timesteps=total_timesteps, use_synthetic=use_synthetic, seed=seed
    )

    if config_overrides:
        print(f"Applying config_overrides uniformly to every inactivity-penalty variant: {config_overrides}")
        for dotted_path, value in config_overrides.items():
            section, field_name = dotted_path.split('.', 1)
            setattr(getattr(base_config, section), field_name, value)

    base_config.save_json(str(output_path / "config.json"))

    features, prices_df, splits, returns, aligned_data = load_and_prepare_data(base_config, use_synthetic)

    prices = {}
    for ticker in tickers:
        if ticker in prices_df.columns:
            prices[ticker] = prices_df[ticker].values
        else:
            prices[ticker] = prices_df.iloc[:, tickers.index(ticker)].values

    wf_evaluator = WalkForwardEvaluator(base_config)
    split = wf_evaluator.generate_calendar_splits(aligned_data.index, [window])[0]
    train_start_idx, train_end_idx = split['train']
    test_start_idx, test_end_idx = split['test']

    train_features = features.iloc[train_start_idx:train_end_idx]
    test_features = features.iloc[test_start_idx:test_end_idx]
    train_prices = {t: p[train_start_idx:train_end_idx] for t, p in prices.items()}
    test_prices = {t: p[test_start_idx:test_end_idx] for t, p in prices.items()}
    train_aligned = aligned_data.iloc[train_start_idx:train_end_idx]
    test_aligned = aligned_data.iloc[test_start_idx:test_end_idx]

    grid_results = run_inactivity_penalty_grid(
        train_features, test_features, train_prices, test_prices,
        train_aligned, test_aligned,
        base_config=base_config,
        log_dir=output_path,
        variants=variants,
        shield_arm=shield_arm,
        trainer_arm=trainer_arm,
        n_eval_episodes=n_eval_episodes,
        resume=resume
    )

    results = {'window': window, 'variants': grid_results}

    serializable_results = convert_to_serializable(
        {'window': window, 'variants': {k: v['evaluation'] for k, v in grid_results.items()}}
    )
    with open(output_path / "inactivity_penalty_ablation_results.json", 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nInactivity penalty ablation complete! Results saved to: {output_path}")

    return results


# =============================================================================
# Full Experiment Pipeline
# =============================================================================

def run_full_experiment(
        tickers: List[str] = None,
        start_date: str = "2010-01-01",
        end_date: str = "2024-12-31",
        total_timesteps: int = 50000,
        use_synthetic: bool = False,
        output_dir: str = "./results",
        run_ablations: bool = True,
        seed: int = 42
) -> Dict[str, Any]:
    """
    Run the complete experiment pipeline: single ratio-based train/test
    split (60/20/20 of one continuous [start_date, end_date] range), then
    the shield x trainer grid via run_method_grid().

    Args:
        tickers: List of assets to trade
        start_date: Data start date
        end_date: Data end date
        total_timesteps: Training timesteps per grid arm
        use_synthetic: Use synthetic data
        output_dir: Output directory for results
        run_ablations: If True, run the full SHIELD_ARMS x TRAINER_ARMS grid
            (6 arms by default); if False, run just the single
            "uncertainty_aware/ppo_lagrangian" arm.
        seed: Random seed for the whole run (config.training.seed;
            applied via seed_everything() before any data/model creation)

    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 70)
    print("  REGIME-CONDITIONED CMDP TRADING EXPERIMENT")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Default tickers
    if tickers is None:
        tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]

    print(f"Assets: {tickers}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Training timesteps: {total_timesteps}")
    print(f"Using synthetic data: {use_synthetic}")

    # Create output directory
    output_path = Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Results container
    results = {
        'config': None,
        'training': {},
        'evaluation': {},
        'baselines': {},
        'comparison': None
    }

    # =========================================================================
    # Step 1: Create configuration
    # =========================================================================
    config = create_experiment_config(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        total_timesteps=total_timesteps,
        use_synthetic=use_synthetic,
        seed=seed
    )
    results['config'] = config.to_dict()

    # Seed every global RNG (python/numpy/torch) before any data loading or
    # model construction, so the whole run is reproducible for this seed.
    seed_everything(config.training.seed)

    # Save config
    config.save_json(str(output_path / "config.json"))

    # =========================================================================
    # Step 2: Load and prepare data
    # =========================================================================
    features, prices_df, splits, returns, aligned_data = load_and_prepare_data(config, use_synthetic)

    # Convert prices to dict of arrays
    prices = {}
    for ticker in config.data.tickers:
        if ticker in prices_df.columns:
            prices[ticker] = prices_df[ticker].values
        else:
            # Handle case where ticker might have different column name
            prices[ticker] = prices_df.iloc[:, config.data.tickers.index(ticker)].values

    # Get split indices
    train_end = len(splits['train'])
    val_end = train_end + len(splits['val'])

    # Create feature/price subsets
    train_features = features.iloc[:train_end]
    test_features = features.iloc[val_end:]

    train_prices = {t: p[:train_end] for t, p in prices.items()}
    test_prices = {t: p[val_end:] for t, p in prices.items()}

    # Create aligned_data subsets for regime model fitting
    train_aligned = aligned_data.iloc[:train_end]
    test_aligned = aligned_data.iloc[val_end:]

    if not HAS_TORCH:
        print("\n" + "=" * 60)
        print("WARNING: PyTorch not installed. Skipping training and evaluation.")
        print("Install with: pip install torch")
        print("=" * 60)

        # Save what we have
        with open(output_path / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    # =========================================================================
    # Steps 3-5: Train + evaluate the shield x trainer grid
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3-5: Shield x Trainer Grid (Environments, Training, Evaluation)")
    print("=" * 60)

    if run_ablations:
        shield_arms, trainer_arms = SHIELD_ARMS, TRAINER_ARMS
    else:
        # Single arm: the full method only, no ablations.
        shield_arms, trainer_arms = ["uncertainty_aware"], ["ppo_lagrangian"]

    grid_results = run_method_grid(
        train_features, test_features, train_prices, test_prices,
        train_aligned, test_aligned,
        base_config=config,
        log_dir=output_path,
        shield_arms=shield_arms,
        trainer_arms=trainer_arms,
        n_eval_episodes=5
    )

    all_trajectories = {}  # Store trajectories for visualization

    for arm_name, arm_result in grid_results.items():
        results['training'][arm_name] = arm_result['training']
        results['evaluation'][arm_name] = arm_result['evaluation']
        if arm_result['trajectory'] is not None:
            all_trajectories[arm_name] = arm_result['trajectory']

    # =========================================================================
    # Step 6: Evaluate baselines
    # =========================================================================
    # Baselines get their own env, built from the base (unmodified) config --
    # they're simple reference strategies, not part of the shield x trainer
    # ablation grid.
    baseline_trajectories = {}
    try:
        baseline_test_env = create_trading_env(
            test_features, test_prices, config, aligned_data=test_aligned, mode="test"
        )
        baseline_results, baseline_trajectories = evaluate_baselines(
            env=baseline_test_env,
            config=config,
            n_episodes=5  # Default eval episodes
        )
        results['baselines'] = baseline_results

        # Add baseline trajectories to all_trajectories
        all_trajectories.update(baseline_trajectories)

    except Exception as e:
        print(f"Error evaluating baselines: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Step 7: Generate comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Results Comparison")
    print("=" * 60)

    comparison_data = []

    # Add trained models
    for method, eval_result in results['evaluation'].items():
        if 'aggregate' in eval_result:
            row = {'method': method}
            for metric, (mean, std) in eval_result['aggregate'].items():
                row[f'{metric}_mean'] = mean
                row[f'{metric}_std'] = std
            comparison_data.append(row)

    # Add baselines
    for baseline, metrics in results['baselines'].items():
        row = {'method': baseline}
        for metric, (mean, std) in metrics.items():
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
        comparison_data.append(row)

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('method')
        results['comparison'] = comparison_df.to_dict()

        # Print comparison table
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)

        display_cols = ['total_return_mean', 'sharpe_ratio_mean', 'max_drawdown_mean',
                        'dd_violation_rate_mean', 'shield_intervention_rate_mean']
        available_cols = [c for c in display_cols if c in comparison_df.columns]

        if available_cols:
            print(comparison_df[available_cols].round(4).to_string())

        # Save comparison
        comparison_df.to_csv(output_path / "comparison.csv")

    # =========================================================================
    # Step 8: Save results and generate report
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Saving Results")
    print("=" * 60)

    # Convert results to serializable format
    serializable_results = convert_to_serializable(results)

    # Save full results
    with open(output_path / "results.json", 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    # Save trajectories for visualization
    if all_trajectories:
        serializable_trajectories = convert_to_serializable(all_trajectories)
        with open(output_path / "trajectories.json", 'w') as f:
            json.dump(serializable_trajectories, f, indent=2, default=str)
        print(f"Saved trajectories to: {output_path / 'trajectories.json'}")

    # Generate summary report
    try:
        generate_summary_report(results, str(output_path / "report.md"))
    except Exception as e:
        print(f"Could not generate report: {e}")

    # Create visualizations with trajectories
    try:
        from evaluation import generate_visualizations

        viz_results = {
            'trajectories': all_trajectories,
            'comparison': comparison_df if comparison_data else None,
        }
        generate_visualizations(viz_results, output_path, prefix="eval")

    except Exception as e:
        print(f"Could not generate trajectory visualizations: {e}")

    # Plot training curves for each trained method
    try:
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        for arm_name in grid_results.keys():
            log_path = output_path / arm_name / "training_log.csv"
            if log_path.exists():
                # arm_name contains "/" (e.g. "none/vanilla_ppo") for the log
                # subdirectory above, but flatten it for the plot filename.
                flat_name = arm_name.replace("/", "__")
                plot_training_curves(
                    str(log_path),
                    save_path=str(plots_dir / f'{flat_name}_training.png')
                )
                print(f"Training curves saved for {arm_name}")
    except Exception as e:
        print(f"Could not plot training curves: {e}")

    print(f"\nExperiment complete!")
    print(f"Results saved to: {output_path}")

    return results


# =============================================================================
# Calendar Walk-Forward Experiment
# =============================================================================

# Fixed calendar walk-forward windows: each fold's train and test periods are
# disjoint, non-overlapping calendar ranges (both ends inclusive).
PAPER_WALK_FORWARD_WINDOWS = [
    {'train': ('2006-01-01', '2007-12-31'), 'test': ('2008-01-01', '2010-12-31')},
    {'train': ('2010-01-01', '2014-12-31'), 'test': ('2015-01-01', '2017-12-31')},
    {'train': ('2017-01-01', '2019-12-31'), 'test': ('2020-01-01', '2023-12-31')},
]


def run_calendar_walk_forward_experiment(
        tickers: List[str] = None,
        windows: List[Dict[str, Tuple[str, str]]] = None,
        shield_arms: List[str] = None,
        trainer_arms: List[str] = None,
        total_timesteps: int = 50000,
        use_synthetic: bool = False,
        output_dir: str = "./results",
        seed: int = 42,
        n_eval_episodes: int = 5,
        config_overrides: Optional[Dict[str, Any]] = None,
        resume: bool = False
) -> Dict[str, Any]:
    """
    Run the fixed-calendar-window walk-forward evaluation.

    Downloads/aligns ONE combined dataset spanning the union of all window
    boundaries and computes features once over the full aligned series
    (causal rolling windows only -- see load_and_prepare_data), then for
    each fold slices train/test by calendar date (not by ratio) and runs the
    full shield x trainer grid via run_method_grid() -- every (shield,
    trainer) arm gets its own freshly-built train_env/test_env from its own
    config, so e.g. "vanilla_ppo" with shield_arm="none" genuinely runs
    unshielded within each fold. WalkForwardEvaluator.generate_calendar_splits
    resolves the calendar boundaries to integer indices and raises if any
    fold's train range overlaps its test range.

    Args:
        tickers: Assets to trade.
        windows: List of {'train': (start, end), 'test': (start, end)} date
            strings (inclusive). Defaults to PAPER_WALK_FORWARD_WINDOWS.
        shield_arms: Which shield arms to run per fold (default: SHIELD_ARMS,
            i.e. ["none", "uncertainty_aware"]).
        trainer_arms: Which trainer types to run per fold (default:
            TRAINER_ARMS, i.e. ["vanilla_ppo", "reward_shaping", "ppo_lagrangian"]).
        total_timesteps: Training timesteps per (fold, arm).
        use_synthetic: Use synthetic data instead of real market data.
        output_dir: Root output directory.
        seed: Random seed.
        n_eval_episodes: Evaluation episodes per (fold, arm).
        config_overrides: Optional {"section.field": value} dict applied to
            base_config right after it's built, BEFORE any arm derives its
            own config from it (_make_arm_config deep-copies base_config) --
            so every arm sees the same override uniformly. E.g. the fold-2
            tuning ladder's canonical execution-model config (see
            PROJECT_STATE.md): {"environment.action_smoothing_beta": 0.1,
            "environment.no_trade_band": 0.10, "environment.decision_frequency_days": 5,
            "action_space.gross_leverage_cap": 2.0, "ppo.lambda_init": [0.4, 0.4, 0.5],
            "ppo.entropy_decay": 0.98435, "ppo.log_std_decay_floor": -1.897}.
        resume: If True, output_dir is used as-is (no fresh datetime.now()
            subdirectory nesting) and each fold's run_method_grid() call
            skips/reloads any arm with a completed eval_result.json under
            its log_dir instead of retraining -- see run_method_grid's
            resume for the exact contract. Callers doing a resumable
            multi-run job must pass a stable, already-timestamped
            output_dir themselves (the caller decides once, up front, and
            reuses it across restarts).

    Returns:
        Dict with 'windows' and 'folds' (list of per-fold results, each with
        resolved calendar boundaries and per-arm test metrics keyed by
        "{shield_arm}/{trainer_arm}").
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for training")

    if tickers is None:
        tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]
    if windows is None:
        windows = PAPER_WALK_FORWARD_WINDOWS
    if shield_arms is None:
        shield_arms = SHIELD_ARMS
    if trainer_arms is None:
        trainer_arms = TRAINER_ARMS

    print("\n" + "=" * 70)
    print("  CALENDAR WALK-FORWARD EXPERIMENT")
    print("=" * 70)
    print(f"Assets: {tickers}")
    print(f"Windows: {windows}")
    print(f"Shield arms: {shield_arms}")
    print(f"Trainer arms: {trainer_arms}")

    seed_everything(seed)

    output_path = Path(output_dir) if resume else Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path.mkdir(parents=True, exist_ok=True)

    # Union date range spanning every fold, so data is downloaded/aligned/
    # feature-ized once rather than once per fold.
    all_dates = [
        d for w in windows
        for d in (w['train'][0], w['train'][1], w['test'][0], w['test'][1])
    ]
    nominal_start, overall_end = min(all_dates), max(all_dates)
    overall_start = _buffered_start_date(nominal_start)
    print(f"Requesting data from {overall_start} (nominal start {nominal_start} minus "
          f"{FEATURE_WARMUP_BUFFER_DAYS}-day feature-warmup buffer)")

    base_config = create_experiment_config(
        tickers=tickers,
        start_date=overall_start,
        end_date=overall_end,
        total_timesteps=total_timesteps,
        use_synthetic=use_synthetic,
        seed=seed
    )

    if config_overrides:
        print(f"Applying config_overrides uniformly to every arm: {config_overrides}")
        for dotted_path, value in config_overrides.items():
            section, field_name = dotted_path.split('.', 1)
            setattr(getattr(base_config, section), field_name, value)

    base_config.save_json(str(output_path / "config.json"))

    features, prices_df, splits, returns, aligned_data = load_and_prepare_data(
        base_config, use_synthetic
    )

    prices = {}
    for ticker in tickers:
        if ticker in prices_df.columns:
            prices[ticker] = prices_df[ticker].values
        else:
            prices[ticker] = prices_df.iloc[:, tickers.index(ticker)].values

    # Resolve the fixed calendar windows to integer index ranges against the
    # single combined aligned series -- raises ValueError on any overlap.
    wf_evaluator = WalkForwardEvaluator(base_config)
    calendar_splits = wf_evaluator.generate_calendar_splits(aligned_data.index, windows)

    fold_results = []

    for fold_idx, split in enumerate(calendar_splits):
        train_start_idx, train_end_idx = split['train']
        test_start_idx, test_end_idx = split['test']

        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{len(calendar_splits)}")
        print(f"  Train: {split['train_dates']} (rows {train_start_idx}:{train_end_idx}, "
              f"{aligned_data.index[train_start_idx].date()} to "
              f"{aligned_data.index[train_end_idx - 1].date()})")
        print(f"  Test:  {split['test_dates']} (rows {test_start_idx}:{test_end_idx}, "
              f"{aligned_data.index[test_start_idx].date()} to "
              f"{aligned_data.index[test_end_idx - 1].date()})")
        print(f"{'=' * 60}")

        fold_train_features = features.iloc[train_start_idx:train_end_idx]
        fold_test_features = features.iloc[test_start_idx:test_end_idx]

        fold_train_prices = {t: p[train_start_idx:train_end_idx] for t, p in prices.items()}
        fold_test_prices = {t: p[test_start_idx:test_end_idx] for t, p in prices.items()}

        fold_train_aligned = aligned_data.iloc[train_start_idx:train_end_idx]
        fold_test_aligned = aligned_data.iloc[test_start_idx:test_end_idx]

        fold_dir = output_path / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        grid_results = run_method_grid(
            fold_train_features, fold_test_features,
            fold_train_prices, fold_test_prices,
            fold_train_aligned, fold_test_aligned,
            base_config=base_config,
            log_dir=fold_dir,
            shield_arms=shield_arms,
            trainer_arms=trainer_arms,
            n_eval_episodes=n_eval_episodes,
            resume=resume
        )

        method_results = {
            arm_name: arm_result['evaluation']
            for arm_name, arm_result in grid_results.items()
        }

        fold_results.append({
            'fold': fold_idx + 1,
            'train_dates': split['train_dates'],
            'test_dates': split['test_dates'],
            'results': method_results,
        })

    results = {'windows': windows, 'folds': fold_results}

    serializable_results = convert_to_serializable(results)
    with open(output_path / "walk_forward_results.json", 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nWalk-forward experiment complete!")
    print(f"Results saved to: {output_path}")

    return results


# =============================================================================
# Quick Test Function
# =============================================================================

def run_quick_test(use_synthetic: bool = True, seed: int = 42):
    """
    Run a quick test with synthetic data and minimal training.
    Good for verifying the pipeline works.
    """
    print("\n" + "=" * 70)
    print("  QUICK TEST MODE")
    print("=" * 70)

    return run_full_experiment(
        tickers=["Asset1", "Asset2", "Asset3", "Asset4"],
        total_timesteps=5000,  # Very short training
        use_synthetic=True,
        output_dir="./test_results",
        run_ablations=False,
        seed=seed
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CMDP Trading Experiment")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "quick", "test", "walk_forward"],
                        help="Experiment mode: full, quick, test, or walk_forward "
                             "(fixed calendar train/test windows)")
    parser.add_argument("--tickers", type=str, nargs="+",
                        default=["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"],
                        help="List of ticker symbols")
    parser.add_argument("--start-date", type=str, default="2010-01-01",
                        help="Start date for data")
    parser.add_argument("--end-date", type=str, default="2024-12-31",
                        help="End date for data")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--no-ablations", action="store_true",
                        help="Skip ablation studies")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    if args.mode == "test":
        # Just run import tests
        print("Testing imports...")
        print("✓ config")
        print("✓ data_loader")
        print("✓ features")
        print("✓ regime_model")
        print("✓ portfolio_engine")
        print("✓ policy_shield")
        print("✓ trading_env")
        if HAS_TORCH:
            print("✓ networks")
            print("✓ trainer")
            print("✓ evaluation")
            print("✓ logging_utils")
        print("\nAll imports successful!")

    elif args.mode == "quick":
        results = run_quick_test(use_synthetic=True, seed=args.seed)

    elif args.mode == "walk_forward":
        results = run_calendar_walk_forward_experiment(
            tickers=args.tickers,
            total_timesteps=args.timesteps,
            use_synthetic=args.synthetic,
            output_dir=args.output_dir,
            seed=args.seed
        )

    else:  # full
        results = run_full_experiment(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            total_timesteps=args.timesteps,
            use_synthetic=args.synthetic,
            output_dir=args.output_dir,
            run_ablations=not args.no_ablations,
            seed=args.seed
        )

    print("\nDone!")