#!/usr/bin/env python3
"""
Phase 2 diagnostic driver: train ONE arm (fold 2, shield=none, ppo_lagrangian
by default) with specific config overrides, evaluate on the test period, and
report the same NAV/turnover/leverage/lambda diagnostics as the Phase 1
autopsy -- for the sequential 3-step test protocol (smoothing+cap ->
+dual_lr -> +PID), not a grid.

Usage:
    python run_single_experiment.py --label step1 --timesteps 200000 \
        --action-smoothing-beta 0.2 --gross-leverage-cap 2.0
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import main as main_module
from config import ShieldType
from evaluation import WalkForwardEvaluator


VALIDATION_CUTOVER_DATE = "2013-01-01"  # validation = 2013-2014 (extended from 2014-only)


def build_fold2_data(tickers, eval_mode="test"):
    """eval_mode="test" (default): original behavior -- train on fold 2's
    full train window (2010-2014), evaluate on its real test period
    (2015-2017).

    eval_mode="validation": fold 2 is now the DEVELOPMENT fold for the
    tuning ladder -- carves 2013-2014 (extended from the original 2014-only
    slice, for a longer/more representative validation window) off the
    2010-2014 train window as a held-out validation set. Training uses ONLY
    2010-2012 (never sees 2013-2014); "test_features"/"test_prices"/etc.
    below are actually the 2013-2014 validation slice. This keeps every
    remaining tuning run off of both the data it trained on AND fold 2's
    real 2015-2017 test period, so that period stays available for one
    final, honest confirmation run. Folds 1 and 3 are untouched by this --
    see DATA_NOTES.md / PROJECT_STATE.md."""
    windows = main_module.PAPER_WALK_FORWARD_WINDOWS
    all_dates = [d for w in windows for d in (w['train'][0], w['train'][1], w['test'][0], w['test'][1])]
    nominal_start, overall_end = min(all_dates), max(all_dates)
    overall_start = main_module._buffered_start_date(nominal_start)

    base_config = main_module.create_experiment_config(
        tickers=tickers, start_date=overall_start, end_date=overall_end,
        total_timesteps=1, use_synthetic=False, seed=1,
    )
    features, prices_df, splits, returns, aligned_data = main_module.load_and_prepare_data(base_config, False)
    prices = {t: prices_df[t].values for t in tickers}

    wf = WalkForwardEvaluator(base_config)
    calendar_splits = wf.generate_calendar_splits(aligned_data.index, windows)
    split = calendar_splits[1]  # fold 2: 2015-2017
    train_start_idx, train_end_idx = split['train']
    test_start_idx, test_end_idx = split['test']

    if eval_mode == "validation":
        train_dates = aligned_data.index[train_start_idx:train_end_idx]
        cutover_pos = int(train_dates.searchsorted(pd.Timestamp(VALIDATION_CUTOVER_DATE)))
        inner_train_end_idx = train_start_idx + cutover_pos
        actual_train_start_idx, actual_train_end_idx = train_start_idx, inner_train_end_idx
        eval_start_idx, eval_end_idx = inner_train_end_idx, train_end_idx
        pct_of_train = (eval_end_idx - eval_start_idx) / (train_end_idx - train_start_idx) * 100
        window_label = "2013-2014-val"
        print(f"[validation mode] inner-train {aligned_data.index[actual_train_start_idx].date()} to "
              f"{aligned_data.index[actual_train_end_idx - 1].date()} ({actual_train_end_idx - actual_train_start_idx} rows), "
              f"validation {aligned_data.index[eval_start_idx].date()} to {aligned_data.index[eval_end_idx - 1].date()} "
              f"({eval_end_idx - eval_start_idx} rows, {pct_of_train:.1f}% of original train window). "
              f"Real 2015-2017 test period untouched.")
    else:
        actual_train_start_idx, actual_train_end_idx = train_start_idx, train_end_idx
        eval_start_idx, eval_end_idx = test_start_idx, test_end_idx
        window_label = f"{split['test_dates'][0][:4]}-{split['test_dates'][1][:4]}"

    data = {
        'base_config': base_config,
        'train_features': features.iloc[actual_train_start_idx:actual_train_end_idx],
        'test_features': features.iloc[eval_start_idx:eval_end_idx],
        'train_prices': {t: p[actual_train_start_idx:actual_train_end_idx] for t, p in prices.items()},
        'test_prices': {t: p[eval_start_idx:eval_end_idx] for t, p in prices.items()},
        'train_aligned': aligned_data.iloc[actual_train_start_idx:actual_train_end_idx],
        'test_aligned': aligned_data.iloc[eval_start_idx:eval_end_idx],
        'window_label': window_label,
        'eval_mode': eval_mode,
    }
    return data


def instrumented_eval(env, policy):
    obs, info = env.reset()
    initial_capital = env.portfolio.initial_capital
    log = []
    done = False
    while not done:
        current_prices = env._get_prices(env._current_step)
        nav_before_trade = env.portfolio.compute_nav(current_prices)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _ = policy.get_action(obs_tensor, deterministic=True)
            action_np = action.cpu().numpy().squeeze()
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        ps = info['portfolio_state']
        tc = info['transaction_cost']
        price_pnl = info['nav'] - (nav_before_trade - tc)
        log.append({
            'weights': np.asarray(ps.weights).copy(),
            'gross_leverage': ps.gross_leverage,
            'turnover': info['turnover'],
            'transaction_cost': tc,
            'price_pnl': price_pnl,
            'nav': info['nav'],
        })
        obs = next_obs
    df = pd.DataFrame(log)
    final_nav = df['nav'].iloc[-1]
    total_price_pnl = df['price_pnl'].sum()
    total_cost = df['transaction_cost'].sum()
    turnovers = df['turnover'].values
    gross_lev = df['gross_leverage'].values
    n_days = len(df)
    total_return = final_nav / initial_capital - 1
    # Annualized net return: standard CAGR-style annualization over the
    # ACTUAL evaluated window length (259 days for the old 2014-only
    # validation slice, ~500 for 2013-2014, ~780 for the real test period --
    # annualizing makes runs of different length comparable, which raw
    # cumulative % is not).
    annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    # Annualized turnover: mean daily turnover * 252 trading days/year --
    # "~250%/yr, i.e. ~1%/day" per the gate.
    annualized_turnover = float(turnovers.mean()) * 252
    return {
        'initial_capital': initial_capital,
        'final_nav': float(final_nav),
        'n_days': n_days,
        'total_return_pct': float(total_return * 100),
        'annualized_return_pct': float(annualized_return * 100),
        'price_pnl_pct': float(total_price_pnl / initial_capital * 100),
        'txn_cost_pct': float(total_cost / initial_capital * 100),
        'turnover_mean': float(turnovers.mean()),
        'turnover_median': float(np.median(turnovers)),
        'turnover_max': float(turnovers.max()),
        'annualized_turnover_pct': float(annualized_turnover * 100),
        'gross_leverage_mean': float(gross_lev.mean()),
        'gross_leverage_median': float(np.median(gross_lev)),
        'gross_leverage_max': float(gross_lev.max()),
        'pct_days_over_1_5x': float((gross_lev > 1.5).mean() * 100),
    }


def main(args):
    tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]
    data = build_fold2_data(tickers, eval_mode=args.eval_mode)

    arm_config = copy.deepcopy(data['base_config'])
    arm_config.shield.shield_type = ShieldType.NONE
    arm_config.training.total_timesteps = args.timesteps

    if args.action_smoothing_beta is not None:
        arm_config.environment.action_smoothing_beta = args.action_smoothing_beta
    if args.no_trade_band is not None:
        arm_config.environment.no_trade_band = args.no_trade_band
    if args.decision_frequency_days is not None:
        arm_config.environment.decision_frequency_days = args.decision_frequency_days
    if args.gross_leverage_cap is not None:
        arm_config.action_space.gross_leverage_cap = args.gross_leverage_cap
    if args.dual_lr is not None:
        arm_config.ppo.dual_lr = args.dual_lr
    if args.dual_update_method is not None:
        arm_config.ppo.dual_update_method = args.dual_update_method
    if args.pid_kp is not None:
        arm_config.ppo.pid_kp = args.pid_kp
    if args.pid_ki is not None:
        arm_config.ppo.pid_ki = args.pid_ki
    if args.pid_kd is not None:
        arm_config.ppo.pid_kd = args.pid_kd
    if args.lambda_init is not None:
        arm_config.ppo.lambda_init = args.lambda_init
    if args.lambda_init_dd is not None or args.lambda_init_turnover is not None or args.lambda_init_leverage is not None:
        # order [dd, turnover, leverage] -- see RiskManager.get_cost_vector
        arm_config.ppo.lambda_init = [
            args.lambda_init_dd if args.lambda_init_dd is not None else 0.0,
            args.lambda_init_turnover if args.lambda_init_turnover is not None else 0.0,
            args.lambda_init_leverage if args.lambda_init_leverage is not None else 0.0,
        ]
    if args.entropy_coef is not None:
        arm_config.ppo.entropy_coef = args.entropy_coef
    if args.entropy_decay is not None:
        arm_config.ppo.entropy_decay = args.entropy_decay
    if args.min_entropy_coef is not None:
        arm_config.ppo.min_entropy_coef = args.min_entropy_coef
    if args.log_std_init is not None:
        arm_config.ppo.log_std_init = args.log_std_init
    if args.log_std_decay_floor is not None:
        arm_config.ppo.log_std_decay_floor = args.log_std_decay_floor

    output_dir = Path(args.output_dir) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}\n# {args.label}: timesteps={args.timesteps}  eval_mode={args.eval_mode}  "
          f"smoothing_beta={arm_config.environment.action_smoothing_beta}  "
          f"no_trade_band={arm_config.environment.no_trade_band}  "
          f"decision_freq={arm_config.environment.decision_frequency_days}  "
          f"gross_cap={arm_config.action_space.gross_leverage_cap}  "
          f"dual_lr={arm_config.ppo.dual_lr}  method={arm_config.ppo.dual_update_method}  "
          f"pid=({arm_config.ppo.pid_kp},{arm_config.ppo.pid_ki},{arm_config.ppo.pid_kd})  "
          f"lambda_init={arm_config.ppo.lambda_init}  "
          f"entropy=({arm_config.ppo.entropy_coef}->{arm_config.ppo.min_entropy_coef} "
          f"decay={arm_config.ppo.entropy_decay})  "
          f"log_std=(init={arm_config.ppo.log_std_init}, floor={arm_config.ppo.log_std_decay_floor})"
          f"\n{'#' * 70}")

    train_env = main_module.create_trading_env(
        data['train_features'], data['train_prices'], arm_config,
        aligned_data=data['train_aligned'], mode="train"
    )
    policy, train_metrics = main_module.train_model(
        env=train_env, config=arm_config, trainer_type="ppo_lagrangian",
        experiment_name=args.label, log_dir=str(output_dir)
    )

    test_env = main_module.create_trading_env(
        data['test_features'], data['test_prices'], arm_config,
        aligned_data=data['test_aligned'], mode="test"
    )
    eval_result = instrumented_eval(test_env, policy)

    training_log = pd.DataFrame(train_metrics)
    lambdas_series = training_log['lambdas'].apply(lambda x: x if isinstance(x, list) else eval(x))
    lambda_first = lambdas_series.iloc[0]
    lambda_last = lambdas_series.iloc[-1]
    mean_costs_series = training_log['mean_costs'].apply(lambda x: x if isinstance(x, list) else eval(x))
    action_std_trajectory = training_log['mean_action_std'].tolist()
    entropy_coef_trajectory = training_log['entropy_coef'].tolist()

    report = {
        'label': args.label,
        'eval_mode': args.eval_mode,
        'window_label': data['window_label'],
        'config': {
            'timesteps': args.timesteps,
            'action_smoothing_beta': arm_config.environment.action_smoothing_beta,
            'no_trade_band': arm_config.environment.no_trade_band,
            'decision_frequency_days': arm_config.environment.decision_frequency_days,
            'gross_leverage_cap': arm_config.action_space.gross_leverage_cap,
            'dual_lr': arm_config.ppo.dual_lr,
            'dual_update_method': arm_config.ppo.dual_update_method,
            'pid_kp': arm_config.ppo.pid_kp,
            'pid_ki': arm_config.ppo.pid_ki,
            'pid_kd': arm_config.ppo.pid_kd,
            'lambda_init': arm_config.ppo.lambda_init,
            'entropy_coef': arm_config.ppo.entropy_coef,
            'entropy_decay': arm_config.ppo.entropy_decay,
            'min_entropy_coef': arm_config.ppo.min_entropy_coef,
            'log_std_init': arm_config.ppo.log_std_init,
            'log_std_decay_floor': arm_config.ppo.log_std_decay_floor,
        },
        'eval': eval_result,
        'lambda_first_iter': lambda_first,
        'lambda_last_iter': lambda_last,
        'lambda_trajectory': lambdas_series.tolist(),
        'mean_costs_trajectory': mean_costs_series.tolist(),
        'action_std_trajectory': action_std_trajectory,
        'entropy_coef_trajectory': entropy_coef_trajectory,
        'n_iterations': len(training_log),
    }

    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 70}\nRESULTS: {args.label}\n{'=' * 70}")
    print(f"  Total return: {eval_result['total_return_pct']:.2f}%  "
          f"(annualized: {eval_result['annualized_return_pct']:+.2f}%, over {eval_result['n_days']} days)  "
          f"(price P&L: {eval_result['price_pnl_pct']:+.2f}%, txn cost: {eval_result['txn_cost_pct']:.2f}%)")
    print(f"  Turnover: mean={eval_result['turnover_mean']:.4f}  median={eval_result['turnover_median']:.4f}  "
          f"max={eval_result['turnover_max']:.4f}  (annualized: {eval_result['annualized_turnover_pct']:.1f}%/yr)")
    print(f"  Gross leverage: mean={eval_result['gross_leverage_mean']:.3f}  "
          f"median={eval_result['gross_leverage_median']:.3f}  max={eval_result['gross_leverage_max']:.3f}  "
          f"days>1.5x={eval_result['pct_days_over_1_5x']:.1f}%")
    print(f"  Lambdas: iter0={lambda_first} -> iter{len(training_log)-1}={lambda_last}")
    n = len(action_std_trajectory)
    sample_idx = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    std_samples = [f"iter{i}={action_std_trajectory[i]:.4f}" for i in sample_idx]
    ent_samples = [f"iter{i}={entropy_coef_trajectory[i]:.5f}" for i in sample_idx]
    print(f"  mean_action_std trajectory: {', '.join(std_samples)}")
    print(f"  entropy_coef trajectory:    {', '.join(ent_samples)}")
    print(f"  Report saved to {output_dir / 'report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--output-dir", type=str, default="./results/phase2")
    parser.add_argument("--eval-mode", type=str, default="test", choices=["test", "validation"])
    parser.add_argument("--action-smoothing-beta", type=float, default=None)
    parser.add_argument("--no-trade-band", type=float, default=None)
    parser.add_argument("--decision-frequency-days", type=int, default=None)
    parser.add_argument("--gross-leverage-cap", type=float, default=None)
    parser.add_argument("--dual-lr", type=float, default=None)
    parser.add_argument("--dual-update-method", type=str, default=None, choices=["additive", "pid"])
    parser.add_argument("--pid-kp", type=float, default=None)
    parser.add_argument("--pid-ki", type=float, default=None)
    parser.add_argument("--pid-kd", type=float, default=None)
    parser.add_argument("--lambda-init", type=float, default=None)
    parser.add_argument("--lambda-init-dd", type=float, default=None)
    parser.add_argument("--lambda-init-turnover", type=float, default=None)
    parser.add_argument("--lambda-init-leverage", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--entropy-decay", type=float, default=None)
    parser.add_argument("--min-entropy-coef", type=float, default=None)
    parser.add_argument("--log-std-init", type=float, default=None)
    parser.add_argument("--log-std-decay-floor", type=float, default=None)
    main(parser.parse_args())
