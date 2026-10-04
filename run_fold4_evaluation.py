#!/usr/bin/env python3
"""
FINAL EVIDENCE SESSION, Part A step 2-3 (evaluation half). ONE look at the
fold-4 (2024-2025) test window -- no re-runs, no threshold adjustments,
regardless of outcome.

Runs, all on the SAME fold-4 test slice:
  - the 7 existing baselines (run_baselines.py's own strategies), fresh,
    for nav_history access.
  - both FROZEN v2 finalists (see PROJECT_STATE.md): shield-regime_gated_
    bear0.6_band0.15_k5, and equal_weight_k5_band010 + regime-gated shield.
  - canonical ppo_lagrangian shielded (uncertainty_aware) and unshielded
    (none), ALL 5 seeds each, trained on 2021-2023 (results/fold4/grid/).
  - the shielded arm, all 5 seeds, re-evaluated with ONLY dd_anchor flipped
    to "trailing" (no retraining) -- the fold-4 trailing-anchor retrofit.

All rows share ONE metric-computation path (return/sharpe/calmar/max_dd/
trailing_63d_max_dd/violation_rate/turnover), matching every other table in
this project. Output: results/fold4/fold4_results.csv.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import main as main_module
from config import Config, ShieldType
from evaluation import BaselineEvaluator, WalkForwardEvaluator
from evaluate_saved import load_model as load_saved_policy
from run_baselines import PriceHistory, build_strategies, compute_return_metrics, EqualWeightBandedStrategy
from run_dev_comparison import trailing_max_drawdown
from overlay_strategies import (
    RCOverlayStrategy, RCOverlayConfig, make_shield_for_overlay, build_env_with_shield,
)

FOLD_4_WINDOW = {'train': ('2021-01-01', '2023-12-31'), 'test': ('2024-01-01', '2025-12-31')}
FOLD_4_LABEL = "2024-2025"
GRID_DIR = Path("./results/fold4/grid/seed_{seed}/fold_1/{shield}/ppo_lagrangian")
OUTPUT_DIR = Path("./results/fold4")
TICKERS = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]
SEEDS = [1, 2, 3, 4, 5]


def _finalize_row(variant: str, run_result: dict, risk_free_rate: float) -> dict:
    row = compute_return_metrics(run_result['nav_history'], risk_free_rate=risk_free_rate)
    row.update({
        'variant': variant,
        'window': FOLD_4_LABEL,
        'violation_rate': run_result['dd_violation_rate'],
        'turnover': run_result['avg_turnover'],
        'trailing_63d_max_dd': trailing_max_drawdown(run_result['nav_history'], window=63),
    })
    return row


def _run_strategy_episode(env, strategy, config) -> dict:
    evaluator = BaselineEvaluator(env, config)
    episode_metrics, _ = evaluator.evaluate_baseline(strategy, n_episodes=1)
    ep = episode_metrics[0]
    return {'nav_history': ep.nav_history, 'avg_turnover': ep.avg_turnover, 'dd_violation_rate': ep.dd_violation_rate}


def _run_policy_episode(env, policy, device: str = "cpu") -> dict:
    obs, info = env.reset()
    nav_history = [env.portfolio.initial_capital]
    turnovers, dd_violations = [], []
    done = False
    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = policy.get_action(obs_t, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze())
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


def _print_row(row: dict):
    print(f"  {row['variant']:42s} return={row['return']:+.4f}  sharpe={row['sharpe']:+.3f}  "
          f"calmar={row['calmar']:+.3f}  max_dd={row['max_dd']:.3f}  "
          f"trail63d_dd={row['trailing_63d_max_dd']:.3f}  turnover={row['turnover']:.4f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dates = [FOLD_4_WINDOW['train'][0], FOLD_4_WINDOW['train'][1],
                 FOLD_4_WINDOW['test'][0], FOLD_4_WINDOW['test'][1]]
    overall_start = main_module._buffered_start_date(min(all_dates))
    overall_end = max(all_dates)

    base_config = main_module.create_experiment_config(
        tickers=TICKERS, start_date=overall_start, end_date=overall_end,
        total_timesteps=1, use_synthetic=False, seed=1,
    )
    risk_free_rate = base_config.evaluation.risk_free_rate

    features, prices_df, splits, returns, aligned_data = main_module.load_and_prepare_data(
        base_config, use_synthetic=False
    )
    prices = {t: prices_df[t].values for t in TICKERS}

    wf_evaluator = WalkForwardEvaluator(base_config)
    split = wf_evaluator.generate_calendar_splits(aligned_data.index, [FOLD_4_WINDOW])[0]
    train_start_idx, train_end_idx = split['train']
    test_start_idx, test_end_idx = split['test']

    test_features = features.iloc[test_start_idx:test_end_idx]
    test_prices = {t: p[test_start_idx:test_end_idx] for t, p in prices.items()}
    test_aligned = aligned_data.iloc[test_start_idx:test_end_idx]
    train_aligned = aligned_data.iloc[train_start_idx:train_end_idx]

    lookback_days = 260
    lookback_start_idx = max(0, test_start_idx - lookback_days)
    actual_lookback_len = test_start_idx - lookback_start_idx
    combined_prices = {t: p[lookback_start_idx:test_end_idx] for t, p in prices.items()}
    price_history = PriceHistory(combined_prices, lookback_len=actual_lookback_len, tickers=TICKERS)

    print(f"\n{'#' * 70}\n# FOLD 4: {FOLD_4_LABEL} -- ONE LOOK, no re-runs\n{'#' * 70}")

    all_rows = []

    # ---- 7 baselines ----
    print("\n-- 7 baselines --")
    baseline_config = copy.deepcopy(base_config)
    baseline_config.shield.shield_type = ShieldType.NONE
    strategies = build_strategies(len(TICKERS), baseline_config, price_history, train_aligned)
    for name, strategy in strategies.items():
        arm_config = copy.deepcopy(baseline_config)
        env = main_module.create_trading_env(test_features, test_prices, arm_config,
                                              aligned_data=test_aligned, mode="test")
        result = _run_strategy_episode(env, strategy, arm_config)
        row = _finalize_row(name, result, risk_free_rate)
        all_rows.append(row)
        _print_row(row)

    # ---- frozen finalist 1: shield-regime_gated_bear0.6_band0.15_k5 ----
    print("\n-- frozen finalist 1: shield-regime_gated_bear0.6_band0.15_k5 --")
    overlay_config = RCOverlayConfig(
        entropy_scale_coef=0.0, switch_hysteresis_days=5,
        bear_entry_threshold=0.6, shield_mode="regime_gated",
        bear_shield_threshold=0.6, entropy_gate_threshold=0.5,
    )
    arm_config = copy.deepcopy(base_config)
    arm_config.environment.decision_frequency_days = 5
    arm_config.environment.no_trade_band = 0.15
    arm_config.environment.action_smoothing_beta = None
    shield = make_shield_for_overlay(overlay_config, arm_config.constraints, len(TICKERS),
                                      n_regimes=arm_config.regime.n_regimes)
    env = build_env_with_shield(test_features, test_prices, arm_config, test_aligned, mode="test", shield=shield)
    strategy = RCOverlayStrategy(len(TICKERS), arm_config, price_history, overlay_config)
    result = _run_strategy_episode(env, strategy, arm_config)
    row = _finalize_row("finalist_shield-regime_gated_bear0.6_band0.15_k5", result, risk_free_rate)
    all_rows.append(row)
    _print_row(row)

    # ---- frozen finalist 2: equal_weight_k5_band010 + regime-gated shield ----
    print("\n-- frozen finalist 2: equal_weight_k5_band010 + regime-gated shield --")
    shield_cfg_holder = RCOverlayConfig(shield_mode="regime_gated", bear_shield_threshold=0.6,
                                         entropy_gate_threshold=0.5)
    arm_config = copy.deepcopy(base_config)
    arm_config.environment.decision_frequency_days = 1
    arm_config.environment.no_trade_band = None
    arm_config.environment.action_smoothing_beta = None
    shield = make_shield_for_overlay(shield_cfg_holder, arm_config.constraints, len(TICKERS),
                                      n_regimes=arm_config.regime.n_regimes)
    env = build_env_with_shield(test_features, test_prices, arm_config, test_aligned, mode="test", shield=shield)
    strategy = EqualWeightBandedStrategy(len(TICKERS), arm_config, rebalance_freq=5, no_trade_band=0.10)
    result = _run_strategy_episode(env, strategy, arm_config)
    row = _finalize_row("finalist_gated_passive_ew_k5_band010", result, risk_free_rate)
    all_rows.append(row)
    _print_row(row)

    # ---- canonical shielded/unshielded RL, 5 seeds each + trailing-anchor retrofit ----
    print("\n-- canonical ppo_lagrangian shielded/unshielded, 5 seeds --")
    for shield_name in ("none", "uncertainty_aware"):
        for seed in SEEDS:
            arm_dir = Path(str(GRID_DIR).format(seed=seed, shield=shield_name))
            arm_config = Config.from_json(str(arm_dir / "config.json"))
            env = main_module.create_trading_env(test_features, test_prices, arm_config,
                                                  aligned_data=test_aligned, mode="test")
            policy = load_saved_policy(arm_dir / "final_model.pt", env.observation_space.shape[0],
                                        len(TICKERS), arm_config)
            result = _run_policy_episode(env, policy)
            row = _finalize_row(f"canonical_{shield_name}_ppo_lagrangian_seed{seed}", result, risk_free_rate)
            all_rows.append(row)
            _print_row(row)

            if shield_name == "uncertainty_aware":
                retrofit_config = copy.deepcopy(arm_config)
                retrofit_config.shield.dd_anchor = "trailing"
                retrofit_config.shield.dd_anchor_window = 63
                retrofit_env = main_module.create_trading_env(test_features, test_prices, retrofit_config,
                                                                aligned_data=test_aligned, mode="test")
                retrofit_policy = load_saved_policy(arm_dir / "final_model.pt",
                                                      retrofit_env.observation_space.shape[0],
                                                      len(TICKERS), retrofit_config)
                retrofit_result = _run_policy_episode(retrofit_env, retrofit_policy)
                retrofit_row = _finalize_row(f"canonical_uncertainty_aware_trailing_retrofit_seed{seed}",
                                              retrofit_result, risk_free_rate)
                all_rows.append(retrofit_row)
                _print_row(retrofit_row)

    df = pd.DataFrame(all_rows)
    df = df[['variant', 'window', 'return', 'sharpe', 'calmar', 'max_dd',
             'trailing_63d_max_dd', 'violation_rate', 'turnover']]
    csv_path = OUTPUT_DIR / "fold4_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(df)} rows to {csv_path}")
    return df


if __name__ == "__main__":
    main()
