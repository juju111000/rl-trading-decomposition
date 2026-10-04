#!/usr/bin/env python3
"""
Resumable driver for the hardening session's seed-expansion validation
workstream. Fold 2 (development) ONLY -- fold 4 is never imported or
referenced anywhere in this script. No canonical config changes: the
canonical/oracle arms use exactly _base_config()'s existing construction
(shield=none, CANONICAL_FOLD2_CONFIG_OVERRIDES applied, 300k timesteps,
unchanged from run_validation_workstream.py's V1/V3).

Meant to be launched detached via launch_hardening.py and polled with
check_hardening_progress.py -- same pattern as run_full_grid.py /
launch_full_grid.py / check_full_grid_progress.py.

Four sub-workstreams, seeds 1-20 each, each independently resumable via a
per-(arm, seed) eval_result.json marker under
results/hardening/{arm}/seed_{n}/eval_result.json (same skip-if-exists
convention as the main grid):
  - untrained: 20 random initializations, no training (fast).
  - canonical: unshielded ppo_lagrangian. Seeds 1-5 REUSE the existing
    results/full_grid/*/seed_{n}/fold_2/none/ppo_lagrangian/ checkpoints
    (re-evaluated fresh here for the extended metric set -- gross P&L,
    txn cost -- not retrained); seeds 6-20 trained fresh.
  - oracle60: V1's oracle construction (60% accuracy), unchanged. Seeds
    1-3 REUSE the existing
    results/v2_dev/validation/v1_training/v1_oracle_seed{n}/ checkpoints;
    seeds 4-20 trained fresh.
  - oracle75: same construction, 75% accuracy (sign flipped w.p. 25%
    instead of 40%). All 20 seeds trained fresh (nothing to reuse).

Writes one row per (arm, seed) to results/hardening/hardening_results.csv,
REWRITTEN after every single completion (so a partial/interrupted run
always leaves a valid, complete-so-far CSV on disk).

If this process dies outright, heartbeat.json is written with
status="failed"; recovery is a manual relaunch (same launch_hardening.py
invocation), which resumes cheaply from the last completed (arm, seed) via
the eval_result.json markers -- no work already on disk is repeated.

Usage:
    python run_hardening.py --output-dir ./results/hardening
"""
import sys
import os
import json
import argparse
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import main as main_module
from config import Config
from evaluate_saved import load_model as load_saved_policy
from networks import create_actor_critic
from seeding import seed_everything
from run_dev_comparison import trailing_max_drawdown
from run_baselines import compute_return_metrics
from run_validation_workstream import (
    _base_config, _load_fold2_slices, build_env_with_features, _oracle_signal,
)

HEARTBEAT_INTERVAL_SECONDS = 30
SEEDS = list(range(1, 21))

CANONICAL_EXISTING_DIR = "./results/full_grid/20260803_174817/seed_{seed}/fold_2/none/ppo_lagrangian"
ORACLE60_EXISTING_DIR = "./results/v2_dev/validation/v1_training/v1_oracle_seed{seed}"
CANONICAL_REUSE_UPTO = 5
ORACLE60_REUSE_UPTO = 3


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Metric computation (shared across all 4 arms)
# =============================================================================

def _gross_price_pnl(weights_seq, test_prices, tickers) -> float:
    """Frictionless replay of the policy's own chosen weights against
    realized asset returns, zero costs -- isolates directional skill from
    cost effects (same technique as run_validation_workstream.run_v2)."""
    close = np.column_stack([test_prices[t] for t in tickers])
    simple_rets = close[1:] / close[:-1] - 1.0
    daily = np.zeros(len(weights_seq))
    for i, w in enumerate(weights_seq):
        if i < len(simple_rets):
            daily[i] = float(np.asarray(w) @ simple_rets[i])
    nav = np.cumprod(1 + daily)
    return float(nav[-1] - 1) if len(nav) else 0.0


def _run_deterministic_episode(env, policy, tickers, test_prices, risk_free_rate=0.02) -> dict:
    obs, info = env.reset()
    nav_history = [env.portfolio.initial_capital]
    turnovers, tx_costs, dd_violations, weights_seq = [], [], [], []
    done = False
    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _ = policy.get_action(obs_t, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze())
        done = terminated or truncated
        nav_history.append(info.get('nav', nav_history[-1]))
        turnovers.append(info.get('turnover', 0.0))
        tx_costs.append(info.get('transaction_cost', 0.0))
        cost_vec = info.get('cost_vec', np.zeros(3))
        dd_violations.append(cost_vec[0] > 0)
        weights_seq.append(info['portfolio_state'].weights.copy())

    row = compute_return_metrics(nav_history, risk_free_rate=risk_free_rate)
    nav = np.asarray(nav_history, dtype=float)
    row['txn_cost'] = float(np.sum(np.array(tx_costs) / nav[:-1])) if len(tx_costs) else 0.0
    row['turnover'] = float(np.mean(turnovers)) if turnovers else 0.0
    row['dd_violation_rate'] = float(np.mean(dd_violations)) if dd_violations else 0.0
    row['trailing_63d_max_dd'] = trailing_max_drawdown(nav_history, window=63)
    row['gross_price_pnl'] = _gross_price_pnl(weights_seq, test_prices, tickers)
    return row


# =============================================================================
# Resumability markers
# =============================================================================

def _marker_path(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / arm / f"seed_{seed}" / "eval_result.json"


def _load_marker(output_root: Path, arm: str, seed: int):
    p = _marker_path(output_root, arm, seed)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save_marker(output_root: Path, arm: str, seed: int, row: dict):
    p = _marker_path(output_root, arm, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(row, f, indent=2, default=str)


# =============================================================================
# Per-arm processing
# =============================================================================

def _build_oracle_slices(config, accuracy: float):
    slices = _load_fold2_slices(config)
    n_train, n_test = len(slices['train_features']), len(slices['test_features'])
    oracle_train = _oracle_signal(slices['train_prices'], slices['tickers'], n_train, accuracy=accuracy)
    oracle_test = _oracle_signal(slices['test_prices'], slices['tickers'], n_test, accuracy=accuracy)
    slices['train_features'] = np.hstack([slices['train_features'], oracle_train])
    slices['test_features'] = np.hstack([slices['test_features'], oracle_test])
    return slices


def _eval_existing(arm_config, model_path, slices) -> dict:
    test_env = build_env_with_features(slices['test_features'], slices['test_prices'], arm_config,
                                        slices['test_aligned'], mode="test")
    policy = load_saved_policy(model_path, test_env.observation_space.shape[0], len(slices['tickers']), arm_config)
    return _run_deterministic_episode(test_env, policy, slices['tickers'], slices['test_prices'])


def _train_and_eval(seed, config, slices, experiment_name, log_dir) -> dict:
    seed_everything(seed)
    train_env = build_env_with_features(slices['train_features'], slices['train_prices'], config,
                                         slices['train_aligned'], mode="train")
    policy, _ = main_module.train_model(train_env, config, trainer_type="ppo_lagrangian",
                                         experiment_name=experiment_name, log_dir=str(log_dir))
    test_env = build_env_with_features(slices['test_features'], slices['test_prices'], config,
                                        slices['test_aligned'], mode="test")
    return _run_deterministic_episode(test_env, policy, slices['tickers'], slices['test_prices'])


def process_untrained(output_root: Path, seed: int) -> dict:
    existing = _load_marker(output_root, 'untrained', seed)
    if existing is not None:
        return existing

    config = _base_config(seed)
    slices = _load_fold2_slices(config)
    seed_everything(seed)
    test_env = build_env_with_features(slices['test_features'], slices['test_prices'], config,
                                        slices['test_aligned'], mode="test")
    action_dim = len(slices['tickers'])
    policy = create_actor_critic(
        obs_dim=test_env.observation_space.shape[0], action_dim=action_dim, n_cost_constraints=3,
        config=config.ppo, action_type=config.action_space.action_type,
        n_discrete_levels=len(config.action_space.discrete_levels), shared=False,
    )
    policy.eval()
    row = _run_deterministic_episode(test_env, policy, slices['tickers'], slices['test_prices'])
    row.update({'arm': 'untrained', 'seed': seed})
    _save_marker(output_root, 'untrained', seed, row)
    return row


def process_canonical(output_root: Path, seed: int) -> dict:
    existing = _load_marker(output_root, 'canonical', seed)
    if existing is not None:
        return existing

    if seed <= CANONICAL_REUSE_UPTO:
        arm_dir = Path(CANONICAL_EXISTING_DIR.format(seed=seed))
        arm_config = Config.from_json(str(arm_dir / "config.json"))
        slices = _load_fold2_slices(arm_config)
        row = _eval_existing(arm_config, arm_dir / "final_model.pt", slices)
    else:
        config = _base_config(seed)
        slices = _load_fold2_slices(config)
        row = _train_and_eval(seed, config, slices, f"canonical_seed{seed}",
                               output_root / "canonical_training")

    row.update({'arm': 'canonical', 'seed': seed})
    _save_marker(output_root, 'canonical', seed, row)
    return row


def process_oracle(output_root: Path, arm_name: str, accuracy: float, seed: int,
                    reuse_upto: int, existing_dir_template: str) -> dict:
    existing = _load_marker(output_root, arm_name, seed)
    if existing is not None:
        return existing

    if seed <= reuse_upto:
        arm_dir = Path(existing_dir_template.format(seed=seed))
        arm_config = Config.from_json(str(arm_dir / "config.json"))
        slices = _build_oracle_slices(arm_config, accuracy)
        row = _eval_existing(arm_config, arm_dir / "final_model.pt", slices)
    else:
        config = _base_config(seed)
        slices = _build_oracle_slices(config, accuracy)
        row = _train_and_eval(seed, config, slices, f"{arm_name}_seed{seed}",
                               output_root / f"{arm_name}_training")

    row.update({'arm': arm_name, 'seed': seed})
    _save_marker(output_root, arm_name, seed, row)
    return row


# =============================================================================
# Driver
# =============================================================================

def _count_completed(output_root: Path) -> int:
    return sum(1 for _ in output_root.rglob("eval_result.json"))


def _rewrite_csv(output_root: Path, rows: list):
    df = pd.DataFrame(rows)
    preferred = ['arm', 'seed', 'return', 'gross_price_pnl', 'txn_cost', 'max_dd',
                 'turnover', 'sharpe', 'sortino', 'calmar', 'cvar5',
                 'trailing_63d_max_dd', 'dd_violation_rate']
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df[cols].to_csv(output_root / "hardening_results.csv", index=False)


def main(args):
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    log_file = open(output_root / "run.log", "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    heartbeat_path = output_root / "heartbeat.json"
    (output_root / "hardening.pid").write_text(str(os.getpid()))

    expected_total = len(SEEDS) * 4  # untrained + canonical + oracle60 + oracle75

    state = {
        'status': 'running', 'started_at': _now_iso(), 'updated_at': _now_iso(), 'finished_at': None,
        'pid': os.getpid(), 'output_dir': str(output_root),
        'expected_total_combos': expected_total, 'completed_combos': 0, 'pct_complete': 0.0,
        'current_arm': None,
    }

    def write_heartbeat():
        tmp = heartbeat_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, heartbeat_path)

    stop_scanning = threading.Event()

    def scan_loop():
        while not stop_scanning.is_set():
            completed = _count_completed(output_root)
            state['completed_combos'] = completed
            state['pct_complete'] = round(100.0 * completed / expected_total, 1) if expected_total else 0.0
            state['updated_at'] = _now_iso()
            write_heartbeat()
            stop_scanning.wait(HEARTBEAT_INTERVAL_SECONDS)

    scanner = threading.Thread(target=scan_loop, daemon=True)
    write_heartbeat()
    scanner.start()

    all_rows = []
    try:
        print(f"=== UNTRAINED ({len(SEEDS)} seeds, no training) ===", flush=True)
        state['current_arm'] = 'untrained'
        for seed in SEEDS:
            print(f"  untrained seed {seed}...", flush=True)
            all_rows.append(process_untrained(output_root, seed))
            _rewrite_csv(output_root, all_rows)

        print(f"=== CANONICAL ({len(SEEDS)} seeds, 1-{CANONICAL_REUSE_UPTO} reused) ===", flush=True)
        state['current_arm'] = 'canonical'
        for seed in SEEDS:
            print(f"  canonical seed {seed}...", flush=True)
            all_rows.append(process_canonical(output_root, seed))
            _rewrite_csv(output_root, all_rows)

        print(f"=== ORACLE60 ({len(SEEDS)} seeds, 1-{ORACLE60_REUSE_UPTO} reused) ===", flush=True)
        state['current_arm'] = 'oracle60'
        for seed in SEEDS:
            print(f"  oracle60 seed {seed}...", flush=True)
            all_rows.append(process_oracle(output_root, 'oracle60', 0.6, seed,
                                            ORACLE60_REUSE_UPTO, ORACLE60_EXISTING_DIR))
            _rewrite_csv(output_root, all_rows)

        print(f"=== ORACLE75 ({len(SEEDS)} seeds, all fresh) ===", flush=True)
        state['current_arm'] = 'oracle75'
        for seed in SEEDS:
            print(f"  oracle75 seed {seed}...", flush=True)
            all_rows.append(process_oracle(output_root, 'oracle75', 0.75, seed, 0, ""))
            _rewrite_csv(output_root, all_rows)

        state['status'] = 'done'
        print("HARDENING_DONE", flush=True)

    except Exception as e:
        state['status'] = 'failed'
        state['error'] = str(e)
        state['traceback'] = traceback.format_exc()
        print(f"HARDENING FAILED: {e}")
        traceback.print_exc()
        raise

    finally:
        state['finished_at'] = _now_iso()
        state['completed_combos'] = _count_completed(output_root)
        state['pct_complete'] = (
            round(100.0 * state['completed_combos'] / expected_total, 1) if expected_total else 0.0
        )
        stop_scanning.set()
        write_heartbeat()
        log_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resumable hardening seed-expansion driver (see launch_hardening.py)")
    parser.add_argument("--output-dir", type=str, default="./results/hardening")
    main(parser.parse_args())
