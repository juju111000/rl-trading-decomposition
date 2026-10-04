#!/usr/bin/env python3
"""
Standalone Evaluation Script

Run evaluation on previously trained models without retraining. Discovers
saved models under both the old flat layout (results_dir/{trainer_type}/)
and the shield x trainer grid layout
(results_dir/{shield_arm}/{trainer_arm}/), and re-evaluates each arm using
its OWN saved config.json (shield_type in particular can differ per arm) --
not the top-level results_dir/config.json.
Usage: python evaluate_saved.py --results-dir results/20251231_013357
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import DataLoader
from regime_model import create_regime_model
from policy_shield import create_shield
from trading_env import TradingCMDPEnv
from networks import create_actor_critic, ActorCritic
from config import ActionSpaceType


def load_config(results_dir: Path) -> Config:
    """Load config from results directory."""
    config_path = results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return Config.from_json(str(config_path))


def load_model(model_path: Path, obs_dim: int, action_dim: int, config: Config, device: str = 'cpu') -> ActorCritic:
    """Load a saved model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create model architecture
    model = create_actor_critic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_cost_constraints=3,
        config=config.ppo,
        action_type=config.action_space.action_type,
        n_discrete_levels=len(config.action_space.discrete_levels),
        shared=False
    )

    # Load weights (weights_only=False needed for PyTorch 2.6+)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'actor_critic_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['actor_critic_state_dict'])
        else:
            # Try loading directly (maybe it's just the state dict)
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def compute_simple_features(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple features from returns DataFrame."""
    features = {}

    for ticker in returns_df.columns:
        ret = returns_df[ticker]

        features[f'{ticker}_ret_1d'] = ret
        features[f'{ticker}_ret_5d'] = ret.rolling(5).sum()
        features[f'{ticker}_ret_20d'] = ret.rolling(20).sum()
        features[f'{ticker}_vol_20d'] = ret.rolling(20).std()
        features[f'{ticker}_vol_60d'] = ret.rolling(60).std()
        cum_ret = ret.cumsum()
        features[f'{ticker}_ma_20'] = cum_ret.rolling(20).mean()
        features[f'{ticker}_ma_50'] = cum_ret.rolling(50).mean()
        features[f'{ticker}_mom_20d'] = ret.rolling(20).mean()

    feature_df = pd.DataFrame(features)
    feature_df = feature_df.ffill().bfill()
    feature_df = (feature_df - feature_df.mean()) / (feature_df.std() + 1e-8)
    feature_df = feature_df.clip(-5, 5)

    return feature_df


def prepare_test_data(config: Config) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """Prepare test data for evaluation."""
    print("Loading data...")
    loader = DataLoader(config.data)
    aligned_data, returns_all, splits_all = loader.prepare_dataset()

    # Filter tickers
    tickers = config.data.tickers
    available_tickers = [t for t in tickers if t in returns_all.columns]
    returns = returns_all[available_tickers]

    # Get test split
    test_returns = splits_all['test'][available_tickers]

    # Compute features
    print("Computing features...")
    test_features = compute_simple_features(test_returns)

    # Get prices
    prices = {}
    for ticker in available_tickers:
        try:
            prices[ticker] = loader.get_price_data('test')[ticker].values
        except:
            prices[ticker] = aligned_data[(ticker, 'Close')].values[-len(test_returns):]

    # Get OHLCV for regime model
    train_end = len(splits_all['train'])
    val_end = train_end + len(splits_all['val'])
    test_ohlcv = aligned_data.iloc[val_end:]

    return test_features, prices, test_ohlcv


def create_test_env(features: pd.DataFrame, prices: Dict[str, np.ndarray],
                    config: Config, ohlcv_data: pd.DataFrame) -> TradingCMDPEnv:
    """Create test environment."""
    print("Creating test environment...")

    regime_model = create_regime_model(config.regime, input_dim=features.shape[1])
    try:
        regime_model.fit(ohlcv_data)
    except:
        if hasattr(regime_model, '_vol_thresholds'):
            regime_model._vol_thresholds = (0.15, 0.25)

    shield = create_shield(config.shield, config.constraints, n_assets=len(config.data.tickers))

    env = TradingCMDPEnv(
        features=features.values,
        prices=prices,
        tickers=config.data.tickers,
        config=config,
        regime_model=regime_model,
        shield=shield,
        mode="test"
    )

    return env


def evaluate_policy(env: TradingCMDPEnv, policy: ActorCritic, n_episodes: int = 5,
                    deterministic: bool = True, device: str = 'cpu') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a policy on the environment.

    Returns:
        metrics: Aggregated metrics (mean, std)
        trajectories: Full trajectories for visualization
    """

    all_returns = []
    all_sharpes = []
    all_max_dds = []
    all_violations = {'dd': [], 'turnover': [], 'leverage': []}

    # Store trajectories for visualization
    trajectories = {
        'nav_history': [],
        'returns': [],
        'drawdown_history': [],
        'dd_violations': [],
        'turnover_violations': [],
        'leverage_violations': [],
    }

    for ep in range(n_episodes):
        obs, info = env.reset()

        nav_history = [env.portfolio.initial_capital]
        returns = []
        drawdown_history = [0.0]
        dd_violations = []
        turnover_violations = []
        leverage_violations = []
        steps = 0

        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                # Use get_action with deterministic flag
                action, _, _ = policy.get_action(obs_tensor, deterministic=deterministic)

                action_np = action.cpu().numpy().squeeze()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            nav_history.append(info.get('nav', nav_history[-1]))
            returns.append(info.get('log_return', 0.0))
            drawdown_history.append(info.get('drawdown', 0.0))

            cost_vec = info.get('cost_vec', np.zeros(3))
            dd_violations.append(1 if cost_vec[0] > 0 else 0)
            turnover_violations.append(1 if cost_vec[1] > 0 else 0)
            leverage_violations.append(1 if cost_vec[2] > 0 else 0)

            obs = next_obs
            steps += 1

        # Compute episode metrics
        nav_history = np.array(nav_history)
        returns = np.array(returns)

        total_return = (nav_history[-1] / nav_history[0]) - 1

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        peak = np.maximum.accumulate(nav_history)
        drawdowns = (peak - nav_history) / (peak + 1e-8)
        max_dd = np.max(drawdowns)

        all_returns.append(total_return)
        all_sharpes.append(sharpe)
        all_max_dds.append(max_dd)
        all_violations['dd'].append(np.mean(dd_violations))
        all_violations['turnover'].append(np.mean(turnover_violations))
        all_violations['leverage'].append(np.mean(leverage_violations))

        # Store best episode trajectory for visualization
        if ep == 0 or total_return > max(all_returns[:-1]):
            trajectories['nav_history'] = nav_history.tolist()
            trajectories['returns'] = returns.tolist()
            trajectories['drawdown_history'] = drawdown_history
            trajectories['dd_violations'] = dd_violations
            trajectories['turnover_violations'] = turnover_violations
            trajectories['leverage_violations'] = leverage_violations

        print(f"  Episode {ep + 1}: Return={total_return:.4f}, Sharpe={sharpe:.4f}, MaxDD={max_dd:.4f}")

    metrics = {
        'total_return': (np.mean(all_returns), np.std(all_returns)),
        'sharpe_ratio': (np.mean(all_sharpes), np.std(all_sharpes)),
        'max_drawdown': (np.mean(all_max_dds), np.std(all_max_dds)),
        'dd_violation_rate': (np.mean(all_violations['dd']), np.std(all_violations['dd'])),
        'turnover_violation_rate': (np.mean(all_violations['turnover']), np.std(all_violations['turnover'])),
        'leverage_violation_rate': (np.mean(all_violations['leverage']), np.std(all_violations['leverage'])),
    }

    return metrics, trajectories


def evaluate_buy_hold(env: TradingCMDPEnv, n_episodes: int = 5) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate Buy & Hold strategy as benchmark.

    This invests 100% in the asset and holds throughout.
    """

    all_returns = []
    all_sharpes = []
    all_max_dds = []

    trajectories = {
        'nav_history': [],
        'returns': [],
        'drawdown_history': [],
    }

    for ep in range(n_episodes):
        obs, info = env.reset()

        nav_history = [env.portfolio.initial_capital]
        returns = []
        drawdown_history = [0.0]

        done = False
        while not done:
            # Buy & Hold: always fully invested (weight = 1.0)
            action = np.array([1.0])  # 100% invested

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            nav_history.append(info.get('nav', nav_history[-1]))
            returns.append(info.get('log_return', 0.0))
            drawdown_history.append(info.get('drawdown', 0.0))

            obs = next_obs

        # Compute episode metrics
        nav_history = np.array(nav_history)
        returns = np.array(returns)

        total_return = (nav_history[-1] / nav_history[0]) - 1

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        peak = np.maximum.accumulate(nav_history)
        drawdowns = (peak - nav_history) / (peak + 1e-8)
        max_dd = np.max(drawdowns)

        all_returns.append(total_return)
        all_sharpes.append(sharpe)
        all_max_dds.append(max_dd)

        # Store best episode trajectory
        if ep == 0 or total_return > max(all_returns[:-1]):
            trajectories['nav_history'] = nav_history.tolist()
            trajectories['returns'] = returns.tolist()
            trajectories['drawdown_history'] = drawdown_history

        print(f"  Episode {ep + 1}: Return={total_return:.4f}, Sharpe={sharpe:.4f}, MaxDD={max_dd:.4f}")

    metrics = {
        'total_return': (np.mean(all_returns), np.std(all_returns)),
        'sharpe_ratio': (np.mean(all_sharpes), np.std(all_sharpes)),
        'max_drawdown': (np.mean(all_max_dds), np.std(all_max_dds)),
        'dd_violation_rate': (0.0, 0.0),  # No constraint tracking for benchmark
        'turnover_violation_rate': (0.0, 0.0),
        'leverage_violation_rate': (0.0, 0.0),
    }

    return metrics, trajectories


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved models")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("=" * 60)
    print("  EVALUATING SAVED MODELS")
    print("=" * 60)
    print(f"\nResults directory: {results_dir}")

    # Load config
    print("\nLoading configuration...")
    config = load_config(results_dir)

    # Seed with the run's own recorded seed for reproducible re-evaluation.
    from seeding import seed_everything
    seed_everything(config.training.seed)

    # Prepare test data
    test_features, test_prices, test_ohlcv = prepare_test_data(config)
    action_dim = len(config.data.tickers)

    # Discover saved models: supports both the old flat layout
    # (results_dir/{trainer_type}/final_model.pt) and the shield x trainer
    # grid layout (results_dir/{shield_arm}/{trainer_arm}/final_model.pt).
    model_paths = sorted(results_dir.glob("*/final_model.pt")) + sorted(
        results_dir.glob("*/*/final_model.pt")
    )
    methods = [
        str(p.parent.relative_to(results_dir)).replace(os.sep, "/")
        for p in model_paths
    ]

    results = {}
    all_trajectories = {}

    for method in methods:
        model_path = results_dir / method / "final_model.pt"
        method_config_path = results_dir / method / "config.json"

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {method}")
        print(f"{'=' * 60}")

        try:
            # Each arm gets its own env, built from ITS OWN saved config
            # (shield_type in particular can differ per arm) rather than the
            # top-level results_dir/config.json -- otherwise every arm would
            # silently be re-evaluated under whichever shield the top-level
            # config happens to have, not the one it was actually trained
            # and is comparable under.
            if method_config_path.exists():
                method_config = Config.from_json(str(method_config_path))
            else:
                method_config = config

            env = create_test_env(test_features, test_prices, method_config, test_ohlcv)
            obs_dim = env.observation_space.shape[0]

            # Load model
            policy = load_model(model_path, obs_dim, action_dim, method_config, args.device)

            # Evaluate
            metrics, trajectories = evaluate_policy(env, policy, args.n_episodes, deterministic=True,
                                                    device=args.device)
            results[method] = metrics
            all_trajectories[method] = trajectories

            print(f"\n{method} Results:")
            print(f"  Total Return: {metrics['total_return'][0]:.4f} ± {metrics['total_return'][1]:.4f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio'][0]:.4f} ± {metrics['sharpe_ratio'][1]:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown'][0]:.4f} ± {metrics['max_drawdown'][1]:.4f}")
            print(f"  DD Violation Rate: {metrics['dd_violation_rate'][0]:.4f}")

        except Exception as e:
            print(f"Error evaluating {method}: {e}")
            import traceback
            traceback.print_exc()

    # Buy & Hold benchmark gets its own env from the base (top-level) config
    # -- it's a simple reference strategy, not part of the shield x trainer
    # grid, so it isn't tied to any single arm's shield.
    env = create_test_env(test_features, test_prices, config, test_ohlcv)

    # Evaluate Buy & Hold benchmark
    print(f"\n{'=' * 60}")
    print(f"Evaluating: Buy & Hold (Benchmark)")
    print(f"{'=' * 60}")

    try:
        bh_metrics, bh_trajectories = evaluate_buy_hold(env, args.n_episodes)
        results['buy_hold'] = bh_metrics
        all_trajectories['buy_hold'] = bh_trajectories

        print(f"\nBuy & Hold Results:")
        print(f"  Total Return: {bh_metrics['total_return'][0]:.4f} ± {bh_metrics['total_return'][1]:.4f}")
        print(f"  Sharpe Ratio: {bh_metrics['sharpe_ratio'][0]:.4f} ± {bh_metrics['sharpe_ratio'][1]:.4f}")
        print(f"  Max Drawdown: {bh_metrics['max_drawdown'][0]:.4f} ± {bh_metrics['max_drawdown'][1]:.4f}")
    except Exception as e:
        print(f"Error evaluating Buy & Hold: {e}")

    # Print comparison table
    if results:
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)

        comparison_data = []
        for method, metrics in results.items():
            row = {'method': method}
            for metric, values in metrics.items():
                row[f'{metric}_mean'] = values[0]
                row[f'{metric}_std'] = values[1]
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('method')

        display_cols = ['total_return_mean', 'sharpe_ratio_mean', 'max_drawdown_mean', 'dd_violation_rate_mean']
        available_cols = [c for c in display_cols if c in comparison_df.columns]

        if available_cols:
            print(comparison_df[available_cols].round(4).to_string())

        # Save results
        output_path = results_dir / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump({k: {m: list(v) for m, v in metrics.items()} for k, metrics in results.items()}, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        comparison_df.to_csv(results_dir / "evaluation_comparison.csv")
        print(f"Comparison saved to: {results_dir / 'evaluation_comparison.csv'}")

        # Save trajectories for visualization
        traj_path = results_dir / "trajectories.json"
        with open(traj_path, 'w') as f:
            json.dump(all_trajectories, f, indent=2)
        print(f"Trajectories saved to: {traj_path}")

        # Generate visualizations
        if args.plot:
            print("\n" + "=" * 80)
            print("GENERATING VISUALIZATIONS")
            print("=" * 80)

            try:
                import matplotlib.pyplot as plt

                viz_dir = results_dir / "visualizations"
                viz_dir.mkdir(exist_ok=True)

                # 1. Cumulative Returns Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                # method names are now "{shield_arm}/{trainer_arm}" (e.g.
                # "none/vanilla_ppo") -- match by substring so the trainer
                # type still resolves to a color/label regardless of prefix.
                color_map = {'ppo_lagrangian': '#2ecc71', 'vanilla_ppo': '#3498db',
                             'reward_shaping': '#e74c3c', 'buy_hold': '#9b59b6'}
                label_map = {'ppo_lagrangian': 'PPO-Lagrangian (Ours)', 'vanilla_ppo': 'Vanilla PPO',
                             'reward_shaping': 'Reward Shaping', 'buy_hold': 'Buy & Hold (Benchmark)'}

                def get_color(method):
                    for key, color in color_map.items():
                        if key in method.lower():
                            return color
                    return 'gray'

                def get_label(method):
                    for key, label in label_map.items():
                        if key in method.lower():
                            return label
                    return method

                for method, traj in all_trajectories.items():
                    if 'nav_history' in traj:
                        nav = np.array(traj['nav_history'])
                        cum_return = (nav / nav[0] - 1) * 100
                        ax.plot(cum_return, label=get_label(method),
                                color=get_color(method), linewidth=2)

                ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                ax.set_xlabel('Trading Days', fontsize=12)
                ax.set_ylabel('Cumulative Return (%)', fontsize=12)
                ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / 'cumulative_returns.png', dpi=300, bbox_inches='tight')
                print(f"Saved: {viz_dir / 'cumulative_returns.png'}")
                plt.close()

                # 2. Drawdown Plot
                fig, ax = plt.subplots(figsize=(12, 6))

                for method, traj in all_trajectories.items():
                    if 'drawdown_history' in traj:
                        dd = np.array(traj['drawdown_history']) * 100
                        ax.fill_between(range(len(dd)), 0, -dd, alpha=0.3, color=get_color(method))
                        ax.plot(-dd, label=get_label(method),
                                color=get_color(method), linewidth=1.5)

                ax.axhline(y=-10, color='red', linestyle='--', alpha=0.7, label='DD Limit (10%)')
                ax.set_xlabel('Trading Days', fontsize=12)
                ax.set_ylabel('Drawdown (%)', fontsize=12)
                ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / 'drawdown_comparison.png', dpi=300, bbox_inches='tight')
                print(f"Saved: {viz_dir / 'drawdown_comparison.png'}")
                plt.close()

                # 3. Metrics Bar Chart
                fig, ax = plt.subplots(figsize=(12, 6))

                metrics_to_plot = ['total_return_mean', 'sharpe_ratio_mean', 'max_drawdown_mean']
                x = np.arange(len(metrics_to_plot))
                width = 0.25

                for i, method in enumerate(results.keys()):
                    values = []
                    for m in metrics_to_plot:
                        if m in comparison_df.columns:
                            val = comparison_df.loc[method, m]
                            if 'drawdown' in m:
                                val = -val  # Flip for visualization
                            values.append(val)
                        else:
                            values.append(0)

                    ax.bar(x + i * width, values, width, label=get_label(method),
                           color=get_color(method))

                ax.set_xticks(x + width)
                ax.set_xticklabels(['Total Return', 'Sharpe Ratio', 'Max DD (neg)'])
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(viz_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
                print(f"Saved: {viz_dir / 'metrics_comparison.png'}")
                plt.close()

                # 4. Constraint Violations Bar Chart
                fig, ax = plt.subplots(figsize=(10, 6))

                violation_metrics = ['dd_violation_rate_mean', 'turnover_violation_rate_mean',
                                     'leverage_violation_rate_mean']
                x = np.arange(len(results))
                width = 0.25

                for i, v_metric in enumerate(violation_metrics):
                    if v_metric in comparison_df.columns:
                        values = [comparison_df.loc[m, v_metric] * 100 for m in results.keys()]
                        ax.bar(x + i * width, values, width,
                               label=v_metric.replace('_mean', '').replace('_', ' ').title())

                ax.set_xticks(x + width)
                ax.set_xticklabels([get_label(m) for m in results.keys()], rotation=15)
                ax.set_ylabel('Violation Rate (%)')
                ax.set_title('Constraint Violation Rates', fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(viz_dir / 'constraint_violations.png', dpi=300, bbox_inches='tight')
                print(f"Saved: {viz_dir / 'constraint_violations.png'}")
                plt.close()

                print(f"\nAll visualizations saved to: {viz_dir}")

            except ImportError as e:
                print(f"Matplotlib not available: {e}")
                print("Install with: pip install matplotlib")

    print("\nDone!")


if __name__ == "__main__":
    main()