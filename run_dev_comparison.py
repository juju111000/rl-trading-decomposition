#!/usr/bin/env python3
"""
strategy-v2-overlay item 4: dev-fold retrofit comparison.

Folds 1-3 (test windows 2008-2010 / 2015-2017 / 2020-2023) are DEVELOPMENT
data here -- fold 4 (2024-25) is reserved for the final frozen evaluation
and is NEVER loaded by this script (main.PAPER_WALK_FORWARD_WINDOWS only
has 3 entries; there is nothing to accidentally touch yet).

Compares, per fold, on IDENTICAL data/features/execution-model plumbing:
  - the 7 existing baselines (run_baselines.py's own strategy classes),
    re-run fresh here (cheap, deterministic) so nav_history is available for
    trailing_63d_max_dd -- not saved in results/baselines/baselines_results.csv
  - RC-Overlay and its 10%-vol-targeted variant (overlay_strategies.py),
    under the trailing shield (30% cap, 63d anchor) + k=5/band=0.10 execution
  - Gated-RL (overlay_strategies.GatedRLStrategy): gates RC-Overlay against
    this fold's best (highest-return of 5 seeds) canonical ppo_lagrangian
    shielded checkpoint, same trailing shield + execution
  - that same best shielded checkpoint, re-evaluated standalone with ONLY
    its dd_anchor flipped to "trailing" (no retraining, shield otherwise
    identical to how it was trained) -- the direct before/after for the
    absorbing-trap fix. The pre-fix ("peak") numbers for this exact
    checkpoint are already in results/full_grid/*/results.csv.
  - the best unshielded (shield=none) ppo_lagrangian checkpoint, re-run
    completely unmodified, as a reference point

All rows share ONE metric-computation path (return/sharpe/calmar/max_dd/
trailing_63d_max_dd/violation_rate/turnover, all derived from each row's own
nav_history + per-step turnover/violation series) so the comparison is
apples-to-apples, not a mix of precomputed CSV summaries and fresh numbers.

Output: results/v2_dev/dev_comparison.csv

Usage:
    python run_dev_comparison.py [--output-dir ./results/v2_dev]
"""
import argparse
import copy
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import main as main_module
from config import Config, ShieldType
from evaluation import BaselineEvaluator, WalkForwardEvaluator
from evaluate_saved import load_model as load_saved_policy
from run_baselines import PriceHistory, build_strategies, compute_return_metrics, _window_label
from overlay_strategies import (
    RCOverlayStrategy, RCOverlayVolTargetStrategy, GatedRLStrategy,
    RCOverlayConfig, make_trailing_shield_config,
)

GRID_RESULTS_DIR = Path("./results/full_grid/20260803_174817")
GRID_RESULTS_CSV = GRID_RESULTS_DIR / "results.csv"
DEV_FOLDS = main_module.PAPER_WALK_FORWARD_WINDOWS  # folds 1-3 only -- fold 4 (2024-25) does not exist yet


def trailing_max_drawdown(nav_history, window: int = 63) -> float:
    """Worst drawdown a 63-day trailing observer would ever have seen:
    max over t of 1 - nav[t] / max(nav[t-window+1 : t+1]) -- the same anchor
    logic as ShieldConfig.dd_anchor="trailing", applied here purely as a
    reporting metric (not fed back into any shield)."""
    nav = np.asarray(nav_history, dtype=float)
    if len(nav) == 0:
        return 0.0
    trailing_peak = pd.Series(nav).rolling(window, min_periods=1).max().values
    dd = 1 - nav / trailing_peak
    return float(np.max(np.clip(dd, 0.0, None)))


def _select_best_seed_checkpoints(fold_window_label: str) -> Dict[str, int]:
    """Highest-return (of the 5 seeds) ppo_lagrangian seed for shield in
    {uncertainty_aware, none}, on this fold's test window, from the full
    canonical grid's results.csv."""
    df = pd.read_csv(GRID_RESULTS_CSV)
    df = df[(df['window'] == fold_window_label) & (df['method'] == 'ppo_lagrangian')]
    best = {}
    for shield in ('uncertainty_aware', 'none'):
        sub = df[df['shield'] == shield]
        if len(sub) == 0:
            continue
        best[shield] = int(sub.loc[sub['return'].idxmax(), 'seed'])
    return best


def _load_arm_config_and_model_path(fold_idx: int, shield: str, seed: int):
    arm_dir = GRID_RESULTS_DIR / f"seed_{seed}" / f"fold_{fold_idx + 1}" / shield / "ppo_lagrangian"
    config = Config.from_json(str(arm_dir / "config.json"))
    return config, arm_dir / "final_model.pt"


def _run_policy_episode(env, policy, device: str = "cpu") -> Dict:
    """One deterministic episode for a loaded torch policy -- mirrors
    BaselineEvaluator._run_baseline_episode's collection, for a policy
    instead of a BaselineStrategy."""
    obs, info = env.reset()
    nav_history = [env.portfolio.initial_capital]
    turnovers = []
    dd_violations = []
    done = False
    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = policy.get_action(obs_t, deterministic=True)
        action_np = action.cpu().numpy().squeeze()
        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        nav_history.append(info.get('nav', nav_history[-1]))
        turnovers.append(info.get('turnover', 0.0))
        cost_vec = info.get('cost_vec', np.zeros(3))
        dd_violations.append(cost_vec[0] > 0)
    return {
        'nav_history': nav_history,
        'avg_turnover': float(np.mean(turnovers)) if turnovers else 0.0,
        'dd_violation_rate': float(np.mean(dd_violations)) if dd_violations else 0.0,
    }


def _run_strategy_episode(env, strategy, config: Config) -> Dict:
    evaluator = BaselineEvaluator(env, config)
    episode_metrics, _ = evaluator.evaluate_baseline(strategy, n_episodes=1)
    ep = episode_metrics[0]
    return {
        'nav_history': ep.nav_history,
        'avg_turnover': ep.avg_turnover,
        'dd_violation_rate': ep.dd_violation_rate,
    }


def _finalize_row(variant: str, window_label: str, run_result: Dict, risk_free_rate: float) -> Dict:
    row = compute_return_metrics(run_result['nav_history'], risk_free_rate=risk_free_rate)
    row.update({
        'variant': variant,
        'window': window_label,
        'violation_rate': run_result['dd_violation_rate'],
        'turnover': run_result['avg_turnover'],
        'trailing_63d_max_dd': trailing_max_drawdown(run_result['nav_history'], window=63),
    })
    return row


def _print_row(row: Dict):
    print(f"  {row['variant']:34s} return={row['return']:+.4f}  sharpe={row['sharpe']:+.3f}  "
          f"calmar={row['calmar']:+.3f}  max_dd={row['max_dd']:.3f}  "
          f"trail63d_dd={row['trailing_63d_max_dd']:.3f}  turnover={row['turnover']:.4f}")


def run_dev_comparison(tickers=None, output_dir: str = "./results/v2_dev", lookback_days: int = 260) -> pd.DataFrame:
    if tickers is None:
        tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]

    windows = DEV_FOLDS
    all_dates = [d for w in windows for d in (w['train'][0], w['train'][1], w['test'][0], w['test'][1])]
    nominal_start, overall_end = min(all_dates), max(all_dates)
    overall_start = main_module._buffered_start_date(nominal_start)

    base_config = main_module.create_experiment_config(
        tickers=tickers, start_date=overall_start, end_date=overall_end,
        total_timesteps=1, use_synthetic=False, seed=1,
    )

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

        print(f"\n{'#' * 60}\n# DEV FOLD {fold_idx + 1}: {window_label} (development data)\n{'#' * 60}")

        # ---- 7 baselines: fresh run on identical data, for nav_history access ----
        baseline_config = copy.deepcopy(base_config)
        baseline_config.shield.shield_type = ShieldType.NONE  # unshielded, matching run_baselines.py
        strategies = build_strategies(len(tickers), baseline_config, price_history, train_aligned)
        for name, strategy in strategies.items():
            arm_config = copy.deepcopy(baseline_config)
            env = main_module.create_trading_env(test_features, test_prices, arm_config,
                                                  aligned_data=test_aligned, mode="test")
            result = _run_strategy_episode(env, strategy, arm_config)
            row = _finalize_row(name, window_label, result, arm_config.evaluation.risk_free_rate)
            all_rows.append(row)
            _print_row(row)

        # ---- RC-Overlay + vol-targeted RC-Overlay: trailing shield (30% cap, 63d anchor), k=5/band=0.10 ----
        overlay_env_config = copy.deepcopy(base_config)
        overlay_env_config.shield = make_trailing_shield_config()
        overlay_env_config.environment.decision_frequency_days = 5
        overlay_env_config.environment.no_trade_band = 0.10
        overlay_env_config.environment.action_smoothing_beta = None

        overlay_variants = [
            ('rc_overlay', RCOverlayStrategy),
            ('rc_overlay_vol_target_10pct', RCOverlayVolTargetStrategy),
        ]
        for name, strategy_cls in overlay_variants:
            arm_config = copy.deepcopy(overlay_env_config)
            env = main_module.create_trading_env(test_features, test_prices, arm_config,
                                                  aligned_data=test_aligned, mode="test")
            strategy = strategy_cls(len(tickers), arm_config, price_history, RCOverlayConfig())
            result = _run_strategy_episode(env, strategy, arm_config)
            row = _finalize_row(name, window_label, result, arm_config.evaluation.risk_free_rate)
            all_rows.append(row)
            _print_row(row)

        # ---- best shielded / unshielded RL checkpoints for this fold ----
        best_seeds = _select_best_seed_checkpoints(window_label)
        shielded_policy = None

        if 'uncertainty_aware' in best_seeds:
            seed = best_seeds['uncertainty_aware']
            shielded_config, shielded_model_path = _load_arm_config_and_model_path(
                fold_idx, 'uncertainty_aware', seed
            )

            # Retrofit: SAME shield the checkpoint was trained/evaluated
            # with, only dd_anchor flipped to trailing -- no retraining.
            retrofit_config = copy.deepcopy(shielded_config)
            retrofit_config.shield.dd_anchor = "trailing"
            retrofit_config.shield.dd_anchor_window = 63
            env = main_module.create_trading_env(test_features, test_prices, retrofit_config,
                                                  aligned_data=test_aligned, mode="test")
            action_dim = len(tickers)
            shielded_policy = load_saved_policy(
                shielded_model_path, env.observation_space.shape[0], action_dim, retrofit_config
            )
            result = _run_policy_episode(env, shielded_policy)
            row = _finalize_row(f"best_shielded_rl_trailing_dd(seed{seed})", window_label, result,
                                 retrofit_config.evaluation.risk_free_rate)
            all_rows.append(row)
            _print_row(row)

        if 'none' in best_seeds:
            seed = best_seeds['none']
            unshielded_config, unshielded_model_path = _load_arm_config_and_model_path(fold_idx, 'none', seed)
            env = main_module.create_trading_env(test_features, test_prices, unshielded_config,
                                                  aligned_data=test_aligned, mode="test")
            action_dim = len(tickers)
            unshielded_policy = load_saved_policy(
                unshielded_model_path, env.observation_space.shape[0], action_dim, unshielded_config
            )
            result = _run_policy_episode(env, unshielded_policy)
            row = _finalize_row(f"best_unshielded_rl(seed{seed})", window_label, result,
                                 unshielded_config.evaluation.risk_free_rate)
            all_rows.append(row)
            _print_row(row)

        # ---- Gated-RL: gates the best shielded checkpoint against RC-Overlay, trailing shield + k=5/band=0.10 ----
        if shielded_policy is not None:
            arm_config = copy.deepcopy(overlay_env_config)
            env = main_module.create_trading_env(test_features, test_prices, arm_config,
                                                  aligned_data=test_aligned, mode="test")
            strategy = GatedRLStrategy(len(tickers), arm_config, price_history, shielded_policy,
                                        overlay_config=RCOverlayConfig())
            result = _run_strategy_episode(env, strategy, arm_config)
            row = _finalize_row(f"gated_rl(seed{best_seeds['uncertainty_aware']})", window_label, result,
                                 arm_config.evaluation.risk_free_rate)
            all_rows.append(row)
            _print_row(row)

    df = pd.DataFrame(all_rows)
    df = df[['variant', 'window', 'return', 'sharpe', 'calmar', 'max_dd',
             'trailing_63d_max_dd', 'violation_rate', 'turnover']]

    csv_path = output_path / "dev_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(df)} rows to {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="strategy-v2-overlay dev-fold retrofit comparison (folds 1-3 only)")
    parser.add_argument("--output-dir", type=str, default="./results/v2_dev")
    args = parser.parse_args()

    run_dev_comparison(output_dir=args.output_dir)
