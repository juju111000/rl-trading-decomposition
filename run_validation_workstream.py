#!/usr/bin/env python3
"""
FINAL EVIDENCE SESSION, Part B: validation workstream (development folds
only -- fold 2, train 2010-2014 / test 2015-2017 -- fold 4 is never loaded
here). No strategy changes: this only builds NEW validation harnesses
around the existing, unmodified env/trainer/network code.

Scoping assumption (stated explicitly, since the task didn't pin shield_type
for V1/V2/V3): all three use shield=NONE. Rationale: the point of this
workstream is to sanity-check the underlying policy/reward/PPO-Lagrangian
machinery itself (does it learn real edge when edge exists? did training
move the policy at all? does a lower-friction environment reveal positive
net edge?) -- the shield's own action-modification would confound "did the
POLICY learn" with "did the SHIELD intervene", and V3 explicitly names the
unshielded arm, so V1/V2 use the same arm for a consistent, comparable
workstream baseline.

V1 (positive control): shield=none/ppo_lagrangian/fold2, canonical
execution config, 3 seeds, with a NON-CAUSAL oracle feature appended per
ticker -- sign(next-day log return), corrupted to exactly 60% accuracy
(P(oracle==true sign)=0.6, fixed oracle-quality seed 20260805 shared across
all 3 training seeds so signal quality is identical, only policy
initialization/training stochasticity differs). This feature is
deliberately look-ahead and used ONLY for this validation control -- it is
never part of the causal FeatureBuilder pipeline and never used by any
trading strategy.

V2 (learning verification): no new training -- reuses the EXISTING 5
canonical shield=none/ppo_lagrangian/fold2 checkpoints
(results/full_grid/*/seed_{s}/fold_2/none/ppo_lagrangian/), compared
against a same-seed UNTRAINED network (fresh init via seed_everything(seed)
then create_actor_critic(), no checkpoint loaded). Reports frictionless
price P&L (each policy's own chosen weights replayed against realized
asset returns, zero costs) and action-feature correlation.

V3 (sensitivity): shield=none/ppo_lagrangian/fold2, canonical execution
config MINUS costs -- transaction_fee_rate=0.0010 (10bps), slippage_rate=0,
use_volatility_slippage=False (a clean flat 10bps per trade, no
volatility-scaled add-on) -- 2 seeds.

Output: results/v2_dev/validation/{v1_results.csv, v3_results.csv,
v2_results.csv}.
"""
import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import main as main_module
from config import ShieldType, Config
from evaluation import Evaluator, WalkForwardEvaluator
from evaluate_saved import load_model as load_saved_policy
from networks import create_actor_critic
from regime_model import create_regime_model
from policy_shield import create_shield
from trading_env import TradingCMDPEnv
from seeding import seed_everything
from run_paper_experiments import CANONICAL_FOLD2_CONFIG_OVERRIDES
from run_dev_comparison import trailing_max_drawdown

FOLD_2_WINDOW = main_module.PAPER_WALK_FORWARD_WINDOWS[1]  # train 2010-2014, test 2015-2017
GRID_RESULTS_DIR = Path("./results/full_grid/20260803_174817")
OUTPUT_DIR = Path("./results/v2_dev/validation")
ORACLE_ACCURACY = 0.6
ORACLE_SEED = 20260805  # fixed, independent of training seed


def _apply_overrides(config: Config, overrides: Dict) -> Config:
    for dotted_path, value in overrides.items():
        section, field_name = dotted_path.split('.', 1)
        setattr(getattr(config, section), field_name, value)
    return config


def _base_config(seed: int, cost_overrides: Optional[Dict] = None) -> Config:
    tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]
    all_dates = [FOLD_2_WINDOW['train'][0], FOLD_2_WINDOW['train'][1],
                 FOLD_2_WINDOW['test'][0], FOLD_2_WINDOW['test'][1]]
    overall_start = main_module._buffered_start_date(min(all_dates))
    overall_end = max(all_dates)

    config = main_module.create_experiment_config(
        tickers=tickers, start_date=overall_start, end_date=overall_end,
        total_timesteps=300000, use_synthetic=False, seed=seed,
    )
    config.shield.shield_type = ShieldType.NONE
    _apply_overrides(config, CANONICAL_FOLD2_CONFIG_OVERRIDES)
    if cost_overrides:
        _apply_overrides(config, cost_overrides)
    return config


def _load_fold2_slices(config: Config, extra_features: Optional[np.ndarray] = None):
    """Loads combined data once and slices fold 2's train/test by calendar
    date, mirroring main.run_calendar_walk_forward_experiment's own slicing
    exactly. `extra_features`, if given, is an (T, k) array ALREADY aligned
    1:1 with the returned features' rows (same length, same row order) --
    hstacked onto the normal causal features before slicing."""
    features, prices_df, splits, returns, aligned_data = main_module.load_and_prepare_data(
        config, use_synthetic=False
    )
    feature_matrix = features.values
    if extra_features is not None:
        assert len(extra_features) == len(feature_matrix), (
            f"extra_features length {len(extra_features)} != features length {len(feature_matrix)}"
        )
        feature_matrix = np.hstack([feature_matrix, extra_features])

    tickers = config.data.tickers
    prices = {t: prices_df[t].values for t in tickers}

    wf_evaluator = WalkForwardEvaluator(config)
    split = wf_evaluator.generate_calendar_splits(aligned_data.index, [FOLD_2_WINDOW])[0]
    train_start, train_end = split['train']
    test_start, test_end = split['test']

    return {
        'train_features': feature_matrix[train_start:train_end],
        'test_features': feature_matrix[test_start:test_end],
        'train_prices': {t: p[train_start:train_end] for t, p in prices.items()},
        'test_prices': {t: p[test_start:test_end] for t, p in prices.items()},
        'train_aligned': aligned_data.iloc[train_start:train_end],
        'test_aligned': aligned_data.iloc[test_start:test_end],
        'tickers': tickers,
    }


def build_env_with_features(features_array: np.ndarray, prices: Dict[str, np.ndarray], config: Config,
                             aligned_data, mode: str) -> TradingCMDPEnv:
    """Mirrors main.create_trading_env's construction exactly, but takes a
    raw features ndarray directly instead of requiring a DataFrame -- needed
    for V1's oracle-augmented feature matrix, which isn't part of the causal
    FeatureBuilder pipeline main.create_trading_env expects."""
    if mode == "test":
        config = copy.deepcopy(config)
        config.training.episode_length = len(features_array)

    regime_model = create_regime_model(config.regime, input_dim=features_array.shape[1])
    if aligned_data is not None:
        regime_model.fit(aligned_data)

    shield = create_shield(config.shield, config.constraints, n_assets=len(config.data.tickers))

    return TradingCMDPEnv(
        features=features_array, prices=prices, tickers=config.data.tickers,
        config=config, regime_model=regime_model, shield=shield, mode=mode,
    )


def _oracle_signal(prices_df_slice: Dict[str, np.ndarray], tickers: List[str], n_rows: int,
                    accuracy: float = ORACLE_ACCURACY) -> np.ndarray:
    """sign(next-day log return) per ticker, corrupted to exactly
    `accuracy`: with that probability the true sign is kept, otherwise
    it's flipped. Last row has no next-day return -> 0 (neutral).
    Deliberately non-causal (uses tomorrow's price); validation-only.
    Uses the SAME fixed ORACLE_SEED regardless of `accuracy` (a stricter
    accuracy is then a strict superset of a looser one's "correct" days --
    the same underlying noise draw, just a different threshold), so
    run_hardening.py's oracle60/oracle75 arms share one consistent
    underlying random process."""
    close = np.column_stack([prices_df_slice[t] for t in tickers])
    next_ret = np.zeros_like(close)
    next_ret[:-1] = np.log(close[1:] / close[:-1])
    true_sign = np.sign(next_ret)

    rng = np.random.RandomState(ORACLE_SEED)
    correct = rng.random(true_sign.shape) < accuracy
    oracle = np.where(correct, true_sign, -true_sign)
    oracle[-1] = 0.0
    return oracle[:n_rows]


def _run_episode_metrics(env, policy, device: str = "cpu") -> Dict:
    evaluator = Evaluator(env, env.config)
    episode_metrics, aggregate, _ = evaluator.evaluate_policy(
        policy, n_episodes=1, deterministic=True, store_trajectories=True
    )
    ep = episode_metrics[0]
    return {
        'nav_history': ep.nav_history,
        'avg_turnover': ep.avg_turnover,
        'dd_violation_rate': ep.dd_violation_rate,
    }


def _metrics_from_run(run_result: Dict, risk_free_rate: float = 0.02) -> Dict:
    from run_baselines import compute_return_metrics
    row = compute_return_metrics(run_result['nav_history'], risk_free_rate=risk_free_rate)
    row.update({
        'violation_rate': run_result['dd_violation_rate'],
        'turnover': run_result['avg_turnover'],
        'trailing_63d_max_dd': trailing_max_drawdown(run_result['nav_history'], window=63),
    })
    return row


def run_v1(seeds: List[int] = (1, 2, 3)) -> pd.DataFrame:
    print(f"\n{'#' * 70}\n# V1 POSITIVE CONTROL: {int(ORACLE_ACCURACY * 100)}% oracle, fold2, "
          f"seeds {list(seeds)}\n{'#' * 70}")
    rows = []
    for seed in seeds:
        config = _base_config(seed)
        seed_everything(seed)

        slices = _load_fold2_slices(config)
        n_train, n_test = len(slices['train_features']), len(slices['test_features'])
        oracle_train = _oracle_signal(slices['train_prices'], slices['tickers'], n_train)
        oracle_test = _oracle_signal(slices['test_prices'], slices['tickers'], n_test)

        train_features_aug = np.hstack([slices['train_features'], oracle_train])
        test_features_aug = np.hstack([slices['test_features'], oracle_test])

        train_env = build_env_with_features(train_features_aug, slices['train_prices'], config,
                                             slices['train_aligned'], mode="train")
        log_dir = OUTPUT_DIR / "v1_training"
        policy, _ = main_module.train_model(
            train_env, config, trainer_type="ppo_lagrangian",
            experiment_name=f"v1_oracle_seed{seed}", log_dir=str(log_dir)
        )

        test_env = build_env_with_features(test_features_aug, slices['test_prices'], config,
                                            slices['test_aligned'], mode="test")
        run_result = _run_episode_metrics(test_env, policy)
        row = _metrics_from_run(run_result, config.evaluation.risk_free_rate)
        row.update({'variant': f'v1_oracle60pct_seed{seed}', 'seed': seed})
        rows.append(row)
        print(f"  seed {seed}: return={row['return']:+.4f}  sharpe={row['sharpe']:+.3f}  "
              f"calmar={row['calmar']:+.3f}  max_dd={row['max_dd']:.3f}")

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "v1_results.csv", index=False)
    print(f"Wrote {OUTPUT_DIR / 'v1_results.csv'}")
    return df


def run_v3(seeds: List[int] = (1, 2)) -> pd.DataFrame:
    cost_overrides = {
        "environment.transaction_fee_rate": 0.0010,
        "environment.slippage_rate": 0.0,
        "environment.use_volatility_slippage": False,
    }
    print(f"\n{'#' * 70}\n# V3 SENSITIVITY: unshielded ppo_lagrangian @ 10bps costs, fold2, "
          f"seeds {list(seeds)}\n{'#' * 70}")
    rows = []
    for seed in seeds:
        config = _base_config(seed, cost_overrides=cost_overrides)
        seed_everything(seed)

        slices = _load_fold2_slices(config)
        train_env = build_env_with_features(slices['train_features'], slices['train_prices'], config,
                                             slices['train_aligned'], mode="train")
        log_dir = OUTPUT_DIR / "v3_training"
        policy, _ = main_module.train_model(
            train_env, config, trainer_type="ppo_lagrangian",
            experiment_name=f"v3_10bps_seed{seed}", log_dir=str(log_dir)
        )

        test_env = build_env_with_features(slices['test_features'], slices['test_prices'], config,
                                            slices['test_aligned'], mode="test")
        run_result = _run_episode_metrics(test_env, policy)
        row = _metrics_from_run(run_result, config.evaluation.risk_free_rate)
        row.update({'variant': f'v3_10bps_seed{seed}', 'seed': seed})
        rows.append(row)
        print(f"  seed {seed}: return={row['return']:+.4f}  sharpe={row['sharpe']:+.3f}  "
              f"calmar={row['calmar']:+.3f}  max_dd={row['max_dd']:.3f}")

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "v3_results.csv", index=False)
    print(f"Wrote {OUTPUT_DIR / 'v3_results.csv'}")
    return df


def _frictionless_price_pnl(nav_history_actions: List[np.ndarray], test_prices: Dict, tickers: List[str]) -> float:
    """Replays a policy's own chosen (post-execution, actual) weight
    trajectory against realized asset returns with zero transaction costs --
    isolates whether the policy found genuine directional edge, separate
    from cost effects."""
    close = np.column_stack([test_prices[t] for t in tickers])
    simple_rets = close[1:] / close[:-1] - 1.0
    daily = np.zeros(len(nav_history_actions))
    for i, w in enumerate(nav_history_actions):
        if i < len(simple_rets):
            daily[i] = float(np.asarray(w) @ simple_rets[i])
    nav = np.cumprod(1 + daily)
    return float(nav[-1] - 1) if len(nav) else 0.0


def run_v2(seeds: List[int] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    print(f"\n{'#' * 70}\n# V2 LEARNING VERIFICATION: trained vs untrained init, fold2, "
          f"seeds {list(seeds)}\n{'#' * 70}")

    config0 = _base_config(seeds[0])
    slices = _load_fold2_slices(config0)
    tickers = slices['tickers']
    action_dim = len(tickers)

    rows = []
    for seed in seeds:
        arm_dir = GRID_RESULTS_DIR / f"seed_{seed}" / "fold_2" / "none" / "ppo_lagrangian"
        arm_config = Config.from_json(str(arm_dir / "config.json"))

        test_env = build_env_with_features(slices['test_features'], slices['test_prices'], arm_config,
                                            slices['test_aligned'], mode="test")
        obs_dim = test_env.observation_space.shape[0]

        trained_policy = load_saved_policy(arm_dir / "final_model.pt", obs_dim, action_dim, arm_config)

        seed_everything(seed)
        untrained_policy = create_actor_critic(
            obs_dim=obs_dim, action_dim=action_dim, n_cost_constraints=3,
            config=arm_config.ppo, action_type=arm_config.action_space.action_type,
            n_discrete_levels=len(arm_config.action_space.discrete_levels), shared=False,
        )
        untrained_policy.eval()

        for label, policy in [('trained', trained_policy), ('untrained', untrained_policy)]:
            test_env_run = build_env_with_features(slices['test_features'], slices['test_prices'], arm_config,
                                                     slices['test_aligned'], mode="test")
            obs, info = test_env_run.reset()
            weights_seq, feature_seq, action_seq = [], [], []
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action, _, _ = policy.get_action(obs_t, deterministic=True)
                action_np = action.cpu().numpy().squeeze()
                feature_seq.append(obs[:test_env_run.feature_dim].copy())
                action_seq.append(np.atleast_1d(action_np).copy())
                obs, reward, terminated, truncated, info = test_env_run.step(action_np)
                done = terminated or truncated
                weights_seq.append(info['portfolio_state'].weights.copy())

            price_pnl = _frictionless_price_pnl(weights_seq, slices['test_prices'], tickers)

            feature_arr = np.array(feature_seq)
            action_arr = np.array(action_seq)
            per_asset_corr = []
            for i in range(action_dim):
                if feature_arr.shape[1] > i and np.std(action_arr[:, i]) > 1e-9 and np.std(feature_arr[:, i]) > 1e-9:
                    per_asset_corr.append(float(np.corrcoef(action_arr[:, i], feature_arr[:, i])[0, 1]))
            mean_abs_corr = float(np.mean(np.abs(per_asset_corr))) if per_asset_corr else 0.0

            rows.append({
                'seed': seed, 'arm': label,
                'price_pnl_frictionless': price_pnl,
                'mean_abs_action_feature_corr': mean_abs_corr,
            })
            print(f"  seed {seed} [{label:9s}]  price_pnl(frictionless)={price_pnl:+.4f}  "
                  f"mean|corr(action,feature)|={mean_abs_corr:.4f}")

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "v2_results.csv", index=False)
    print(f"Wrote {OUTPUT_DIR / 'v2_results.csv'}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part B validation workstream (fold 2 only)")
    parser.add_argument("--part", choices=["v1", "v2", "v3", "all"], default="all")
    args = parser.parse_args()

    if args.part in ("v1", "all"):
        run_v1()
    if args.part in ("v3", "all"):
        run_v3()
    if args.part in ("v2", "all"):
        run_v2()

    print("\nVALIDATION_WORKSTREAM_DONE")
