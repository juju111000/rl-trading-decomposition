#!/usr/bin/env python3
"""
strategy-v2-overlay redesign sweep, built from run_rc_overlay_forensics.py's
findings. Folds 1-3 (development) only -- fold 4 (2024-25) is NOT loaded
anywhere in this script; that's the next step after this report, once a
config is frozen.

Fixed (per the redesign decision, not swept):
  - kappa (entropy_scale_coef) = 0.0 -- forensics showed it's immaterial.
  - switch_hysteresis_days = 5 -- the ablation grid's best crisis-fold lever.
  - bear branch entry no longer requires the regime model's own
    `resolved` flag (RCOverlayConfig.bear_entry_threshold) -- lateness,
    not wrongness, was the dominant timing-miss mode.

Swept (16 = 2 x 2 x 4):
  - shield_mode: "always" | "regime_gated" (policy_shield.RegimeGatedShield)
  - bear threshold in {0.5, 0.6} -- drives BOTH the branch-entry rule
    (bear_entry_threshold) and, when shield_mode="regime_gated", the
    shield's own activation gate (bear_shield_threshold) -- tied together
    since the sweep only has one "bear threshold" dimension.
  - execution (no_trade_band, decision_frequency_days) in
    {0.10,0.15} x {5,10}.

Also runs equal_weight_k5_band010 (run_baselines.py's own strategy,
unmodified) wrapped in the regime_gated shield, at both bear thresholds --
the "simplicity" competitor.

Selection rule (decided in advance, applied mechanically, no post-hoc
adjustment): rank the 16 sweep variants by mean Calmar across the 3 folds;
eligible only if 2008-2010 return >= tsmom_12m's actual -13.03% AND every
fold's return >= -15%. The top eligible variant is reported as the frozen
config for fold-4 evaluation.

Output: results/v2_dev/v2_sweep/{sweep_grid.csv, gated_passive_competitor.csv,
selection_ranking.csv}
"""
import copy
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import main as main_module
from run_baselines import EqualWeightBandedStrategy
from overlay_strategies import RCOverlayStrategy, RCOverlayConfig, make_shield_for_overlay, build_env_with_shield
from run_rc_overlay_forensics import _load_dev_data, _fold_slices, _finalize_row

OUTPUT_DIR = Path("./results/v2_dev/v2_sweep")

# tsmom_12m's actual 2008-2010 return, from results/baselines/baselines_results.csv
# (row: strategy=tsmom_12m, window=2008-2010) -- the selection rule's floor.
TSMOM_2008_2010_RETURN = -0.1302963018642156
FOLD_FLOOR = -0.15

SHIELD_MODES = ["always", "regime_gated"]
BEAR_THRESHOLDS = [0.5, 0.6]
EXECUTION_COMBOS = [(0.10, 5), (0.10, 10), (0.15, 5), (0.15, 10)]  # (no_trade_band, decision_frequency_days)

SWEEP_GRID = [
    {'shield_mode': sm, 'bear_threshold': bt, 'band': band, 'k': k}
    for sm in SHIELD_MODES
    for bt in BEAR_THRESHOLDS
    for band, k in EXECUTION_COMBOS
]
assert len(SWEEP_GRID) == 16


def _variant_name(combo: Dict) -> str:
    return f"shield-{combo['shield_mode']}_bear{combo['bear_threshold']}_band{combo['band']}_k{combo['k']}"


def _run_episode(env, strategy) -> Dict:
    obs, info = env.reset()
    nav_history = [env.portfolio.initial_capital]
    turnovers, dd_violations = [], []
    done = False
    while not done:
        action = strategy.get_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
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


def run_sweep_variant(base_config, test_features, test_prices, test_aligned, price_history, tickers,
                       combo: Dict) -> Dict:
    overlay_config = RCOverlayConfig(
        entropy_scale_coef=0.0,  # item 4: fixed
        switch_hysteresis_days=5,  # item 3: fixed
        bear_entry_threshold=combo['bear_threshold'],  # item 2: resolved requirement dropped
        shield_mode=combo['shield_mode'],  # item 1
        bear_shield_threshold=combo['bear_threshold'],
        entropy_gate_threshold=0.5,
    )

    arm_config = copy.deepcopy(base_config)
    arm_config.environment.decision_frequency_days = combo['k']
    arm_config.environment.no_trade_band = combo['band']
    arm_config.environment.action_smoothing_beta = None

    shield = make_shield_for_overlay(overlay_config, arm_config.constraints, len(tickers),
                                      n_regimes=arm_config.regime.n_regimes)
    env = build_env_with_shield(test_features, test_prices, arm_config, test_aligned, mode="test", shield=shield)
    strategy = RCOverlayStrategy(len(tickers), arm_config, price_history, overlay_config)

    return _run_episode(env, strategy)


def run_gated_passive_competitor(base_config, test_features, test_prices, test_aligned, price_history, tickers,
                                  bear_threshold: float) -> Dict:
    """equal_weight_k5_band010 (unmodified, self-contained k=5/band=0.10
    cadence like run_baselines.py) wrapped in the regime-gated shield --
    env-level decision_frequency_days/no_trade_band left at pass-through
    (1 / None) so the strategy's own cadence isn't double-applied."""
    shield_cfg_holder = RCOverlayConfig(shield_mode="regime_gated", bear_shield_threshold=bear_threshold,
                                         entropy_gate_threshold=0.5)

    arm_config = copy.deepcopy(base_config)
    arm_config.environment.decision_frequency_days = 1
    arm_config.environment.no_trade_band = None
    arm_config.environment.action_smoothing_beta = None

    shield = make_shield_for_overlay(shield_cfg_holder, arm_config.constraints, len(tickers),
                                      n_regimes=arm_config.regime.n_regimes)
    env = build_env_with_shield(test_features, test_prices, arm_config, test_aligned, mode="test", shield=shield)
    strategy = EqualWeightBandedStrategy(len(tickers), arm_config, rebalance_freq=5, no_trade_band=0.10)

    return _run_episode(env, strategy)


def rank_by_selection_rule(df: pd.DataFrame) -> pd.DataFrame:
    """One row per variant: mean Calmar across folds + the two gate checks,
    sorted by mean Calmar descending. The selection rule itself (decided in
    advance): eligible only if 2008-2010 return >= TSMOM_2008_2010_RETURN
    AND every fold's return >= FOLD_FLOOR."""
    rows = []
    for variant, g in df.groupby('variant', sort=False):
        g = g.set_index('window')
        returns = g['return'].to_dict()
        fold1_return = returns.get('2008-2010', float('-inf'))
        floor_2008_ok = fold1_return >= TSMOM_2008_2010_RETURN
        all_folds_ok = all(r >= FOLD_FLOOR for r in returns.values())
        rows.append({
            'variant': variant,
            'mean_calmar': g['calmar'].mean(),
            'return_2008_2010': returns.get('2008-2010'),
            'return_2015_2017': returns.get('2015-2017'),
            'return_2020_2023': returns.get('2020-2023'),
            'floor_2008_ok': floor_2008_ok,
            'all_folds_above_-15pct': all_folds_ok,
            'eligible': floor_2008_ok and all_folds_ok,
        })
    ranking = pd.DataFrame(rows).sort_values('mean_calmar', ascending=False).reset_index(drop=True)
    return ranking


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers, base_config, features, prices, aligned_data, calendar_splits = _load_dev_data()
    risk_free_rate = base_config.evaluation.risk_free_rate

    sweep_rows = []
    competitor_rows = []

    for fold_idx, split in enumerate(calendar_splits):
        window_label, test_features, test_prices, test_aligned, test_dates, price_history = _fold_slices(
            split, features, prices, aligned_data, tickers
        )
        print(f"\n{'#' * 70}\n# DEV FOLD {fold_idx + 1}: {window_label} (development data)\n{'#' * 70}")

        print("\n-- 16-variant sweep --")
        for combo in SWEEP_GRID:
            result = run_sweep_variant(base_config, test_features, test_prices, test_aligned, price_history,
                                        tickers, combo)
            row = _finalize_row(_variant_name(combo), window_label, result, risk_free_rate)
            sweep_rows.append(row)
            print(f"  {row['variant']:40s} return={row['return']:+.4f}  calmar={row['calmar']:+.3f}  "
                  f"max_dd={row['max_dd']:.3f}  trail63d_dd={row['trailing_63d_max_dd']:.3f}")

        print("\n-- gated-passive competitor (equal_weight_k5_band010 + regime-gated shield) --")
        for bt in BEAR_THRESHOLDS:
            result = run_gated_passive_competitor(base_config, test_features, test_prices, test_aligned,
                                                    price_history, tickers, bear_threshold=bt)
            row = _finalize_row(f"gated_passive_bear{bt}", window_label, result, risk_free_rate)
            competitor_rows.append(row)
            print(f"  {row['variant']:40s} return={row['return']:+.4f}  calmar={row['calmar']:+.3f}  "
                  f"max_dd={row['max_dd']:.3f}  trail63d_dd={row['trailing_63d_max_dd']:.3f}")

    cols = ['variant', 'window', 'return', 'sharpe', 'calmar', 'max_dd',
            'trailing_63d_max_dd', 'violation_rate', 'turnover']
    sweep_df = pd.DataFrame(sweep_rows)[cols]
    competitor_df = pd.DataFrame(competitor_rows)[cols]

    sweep_df.to_csv(OUTPUT_DIR / "sweep_grid.csv", index=False)
    competitor_df.to_csv(OUTPUT_DIR / "gated_passive_competitor.csv", index=False)

    sweep_ranking = rank_by_selection_rule(sweep_df)
    competitor_ranking = rank_by_selection_rule(competitor_df)
    sweep_ranking.to_csv(OUTPUT_DIR / "selection_ranking.csv", index=False)
    competitor_ranking.to_csv(OUTPUT_DIR / "competitor_ranking.csv", index=False)

    print(f"\n{'=' * 70}\nSELECTION RANKING (sweep, mean Calmar desc, eligible = passes both floors)\n{'=' * 70}")
    print(sweep_ranking.round(4).to_string(index=False))

    print(f"\n{'=' * 70}\nGATED-PASSIVE COMPETITOR ranking (same rule)\n{'=' * 70}")
    print(competitor_ranking.round(4).to_string(index=False))

    eligible = sweep_ranking[sweep_ranking['eligible']]
    if len(eligible):
        winner = eligible.iloc[0]
        print(f"\nTOP ELIGIBLE CONFIG: {winner['variant']} (mean Calmar={winner['mean_calmar']:.4f})")
    else:
        print("\nNO sweep variant passes both floors -- see selection_ranking.csv.")

    print(f"\nWrote sweep_grid.csv ({len(sweep_df)} rows), gated_passive_competitor.csv ({len(competitor_df)} rows), "
          f"selection_ranking.csv, competitor_ranking.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
