#!/usr/bin/env python3
"""
Full paper experiment grid: shield x trainer x calendar window x seed.

Runs main.run_calendar_walk_forward_experiment() once per seed -- each call
already covers every fold in `windows` (default: main.PAPER_WALK_FORWARD_WINDOWS,
the paper's three fixed train/test periods) and the full shield x trainer
grid within each fold (default: main.SHIELD_ARMS x main.TRAINER_ARMS, i.e.
2 x 3 = 6 arms) -- collects every (fold, arm) result across all seeds, and
writes:

  - results.csv: one tidy row per (method, shield, window, seed) with
    return, sharpe, max_dd, violation_rate, turnover, and
    shield_intervention_rate.
  - summary.csv: mean +/- std across seeds, grouped by (method, shield, window).

Data is downloaded/aligned/feature-computed once per seed (not shared across
seeds), since main.run_calendar_walk_forward_experiment() doesn't currently
expose a way to reuse a pre-loaded dataset across calls -- each call is
otherwise independent and reproducible for its own seed.

Usage:
    python run_paper_experiments.py
    python run_paper_experiments.py --seeds 1 2 3 --timesteps 5000 --synthetic
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import main as main_module

DEFAULT_SEEDS = [1, 2, 3, 4, 5]

METRIC_COLUMNS = [
    'return', 'sharpe', 'max_dd', 'violation_rate', 'turnover',
    'shield_intervention_rate',
]

# Frozen canonical execution-model config (band010_k5 confirmation run,
# fold 2's real 2015-2017 test period -- see PROJECT_STATE.md). Applied
# uniformly to every arm (main grid + both ablations) via config_overrides
# when --canonical-fold2-config is passed; module-level so run_full_grid.py
# can import it directly instead of duplicating the dict.
CANONICAL_FOLD2_CONFIG_OVERRIDES = {
    "environment.action_smoothing_beta": 0.1,
    "environment.no_trade_band": 0.10,
    "environment.decision_frequency_days": 5,
    "action_space.gross_leverage_cap": 2.0,
    "ppo.lambda_init": [0.4, 0.4, 0.5],
    "ppo.entropy_decay": 0.98435,
    "ppo.log_std_decay_floor": -1.897,
    "ppo.dual_lr": 0.005,
}


def _window_label(test_dates: Tuple[str, str]) -> str:
    """Compact, paper-table-friendly label for a fold's test period, e.g.
    ('2008-01-01', '2010-12-31') -> '2008-2010'."""
    start, end = test_dates
    return f"{start[:4]}-{end[:4]}"


def _extract_rows(results: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    """Flatten one seed's run_calendar_walk_forward_experiment() result into
    tidy (method, shield, window, seed, ...) rows -- one per (fold, arm)."""
    rows = []
    for fold in results['folds']:
        window = _window_label(fold['test_dates'])

        for arm_name, arm_result in fold['results'].items():
            shield_arm, trainer_arm = arm_name.split('/', 1)
            aggregate = arm_result.get('aggregate', {})

            def _mean(key: str) -> float:
                value = aggregate.get(key)
                return float(value[0]) if value is not None else float('nan')

            rows.append({
                'method': trainer_arm,
                'shield': shield_arm,
                'window': window,
                'seed': seed,
                'return': _mean('total_return'),
                'sharpe': _mean('sharpe_ratio'),
                'max_dd': _mean('max_drawdown'),
                'violation_rate': _mean('dd_violation_rate'),
                'turnover': _mean('avg_turnover'),
                'shield_intervention_rate': _mean('shield_intervention_rate'),
            })

    return rows


def _extract_shield_ablation_rows(results: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    """Flatten one seed's run_shield_ablation_experiment() result into tidy
    (method, shield_variant, window, seed, ...) rows -- one per variant."""
    rows = []
    window_label = _window_label(results['window']['test'])

    for variant_name, variant_result in results['variants'].items():
        aggregate = variant_result.get('evaluation', {}).get('aggregate', {})

        def _mean(key: str) -> float:
            value = aggregate.get(key)
            return float(value[0]) if value is not None else float('nan')

        rows.append({
            'method': 'ppo_lagrangian',
            'shield_variant': variant_name,
            'window': window_label,
            'seed': seed,
            'return': _mean('total_return'),
            'sharpe': _mean('sharpe_ratio'),
            'max_dd': _mean('max_drawdown'),
            'violation_rate': _mean('dd_violation_rate'),
            'turnover': _mean('avg_turnover'),
            'shield_intervention_rate': _mean('shield_intervention_rate'),
        })

    return rows


def _extract_inactivity_ablation_rows(results: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    """Flatten one seed's run_inactivity_penalty_ablation_experiment() result
    into tidy (method, shield, inactivity_variant, window, seed, ...) rows --
    one per variant."""
    rows = []
    window_label = _window_label(results['window']['test'])

    for variant_name, variant_result in results['variants'].items():
        aggregate = variant_result.get('evaluation', {}).get('aggregate', {})

        def _mean(key: str) -> float:
            value = aggregate.get(key)
            return float(value[0]) if value is not None else float('nan')

        rows.append({
            'method': variant_result.get('trainer', 'ppo_lagrangian'),
            'shield': variant_result.get('shield', 'none'),
            'inactivity_variant': variant_name,
            'window': window_label,
            'seed': seed,
            'return': _mean('total_return'),
            'sharpe': _mean('sharpe_ratio'),
            'max_dd': _mean('max_drawdown'),
            'violation_rate': _mean('dd_violation_rate'),
            'turnover': _mean('avg_turnover'),
            'shield_intervention_rate': _mean('shield_intervention_rate'),
        })

    return rows


def run_shield_ablation_grid(
        tickers: Optional[List[str]] = None,
        window: Optional[Dict[str, Tuple[str, str]]] = None,
        shield_variants: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        total_timesteps: int = 50000,
        use_synthetic: bool = False,
        output_dir: str = "./results/paper_experiments",
        n_eval_episodes: int = 5,
        config_overrides: Optional[Dict[str, Any]] = None,
        resume: bool = False,
) -> Dict[str, Any]:
    """
    Shield-component ablation: ppo_lagrangian only, window 1 only (default),
    across main.SHIELD_COMPONENT_VARIANTS x seeds. Writes
    shield_ablation_results.csv + shield_ablation_summary.csv into
    `output_dir` (same layout convention as run_paper_experiments()).

    config_overrides: forwarded to main.run_shield_ablation_experiment(), so
        the canonical execution-model config (if passed) is applied
        uniformly here too -- the reproducibility claim requires every arm,
        including ablations, to reflect the same overrides in its
        config.json.
    resume: forwarded to main.run_shield_ablation_experiment() per seed --
        each seed's fixed seed_output_dir (not re-timestamped) lets a
        variant with a completed eval_result.json be skipped/reloaded on
        restart instead of retrained.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if window is None:
        window = main_module.PAPER_WALK_FORWARD_WINDOWS[0]
    if shield_variants is None:
        shield_variants = list(main_module.SHIELD_COMPONENT_VARIANTS.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  SHIELD COMPONENT ABLATION GRID")
    print("=" * 70)
    print(f"Window: {window}")
    print(f"Shield variants: {shield_variants}")
    print(f"Seeds: {seeds}")

    all_rows = []
    for seed in seeds:
        print(f"\n{'#' * 70}")
        print(f"# SHIELD ABLATION SEED {seed}")
        print(f"{'#' * 70}")

        seed_output_dir = output_path / f"shield_ablation_seed_{seed}"
        results = main_module.run_shield_ablation_experiment(
            tickers=tickers,
            window=window,
            shield_variants=shield_variants,
            total_timesteps=total_timesteps,
            use_synthetic=use_synthetic,
            output_dir=str(seed_output_dir),
            seed=seed,
            n_eval_episodes=n_eval_episodes,
            config_overrides=config_overrides,
            resume=resume,
        )
        all_rows.extend(_extract_shield_ablation_rows(results, seed))

    results_df = pd.DataFrame(all_rows)
    results_csv_path = output_path / "shield_ablation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nWrote tidy shield ablation results to {results_csv_path} ({len(results_df)} rows)")

    if results_df.empty:
        return {'results_df': results_df, 'summary_df': pd.DataFrame(), 'output_dir': str(output_path)}

    summary = (
        results_df
        .groupby(['shield_variant'])[METRIC_COLUMNS]
        .agg(['mean', 'std'])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    summary_csv_path = output_path / "shield_ablation_summary.csv"
    summary.to_csv(summary_csv_path, index=False)
    print(f"Wrote shield ablation summary to {summary_csv_path}")
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(summary.round(3).to_string(index=False))

    return {'results_df': results_df, 'summary_df': summary, 'output_dir': str(output_path)}


def run_inactivity_penalty_ablation_grid(
        tickers: Optional[List[str]] = None,
        window: Optional[Dict[str, Tuple[str, str]]] = None,
        variants: Optional[List[str]] = None,
        shield_arm: str = "none",
        trainer_arm: str = "ppo_lagrangian",
        seeds: Optional[List[int]] = None,
        total_timesteps: int = 50000,
        use_synthetic: bool = False,
        output_dir: str = "./results/paper_experiments",
        n_eval_episodes: int = 5,
        config_overrides: Optional[Dict[str, Any]] = None,
        resume: bool = False,
) -> Dict[str, Any]:
    """
    Inactivity-penalty ablation: shield=none/ppo_lagrangian only (default --
    the arm this investigation centered on), window 1 only (default), across
    main.INACTIVITY_PENALTY_VARIANTS x seeds. Writes
    inactivity_ablation_results.csv + inactivity_ablation_summary.csv into
    `output_dir` (same layout convention as run_shield_ablation_grid()).

    config_overrides/resume: same contract as run_shield_ablation_grid's.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if window is None:
        window = main_module.PAPER_WALK_FORWARD_WINDOWS[0]
    if variants is None:
        variants = list(main_module.INACTIVITY_PENALTY_VARIANTS.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  INACTIVITY PENALTY ABLATION GRID")
    print("=" * 70)
    print(f"Window: {window}")
    print(f"Shield/Trainer: {shield_arm}/{trainer_arm}")
    print(f"Variants: {variants}")
    print(f"Seeds: {seeds}")

    all_rows = []
    for seed in seeds:
        print(f"\n{'#' * 70}")
        print(f"# INACTIVITY ABLATION SEED {seed}")
        print(f"{'#' * 70}")

        seed_output_dir = output_path / f"inactivity_ablation_seed_{seed}"
        results = main_module.run_inactivity_penalty_ablation_experiment(
            tickers=tickers,
            window=window,
            variants=variants,
            shield_arm=shield_arm,
            trainer_arm=trainer_arm,
            total_timesteps=total_timesteps,
            use_synthetic=use_synthetic,
            output_dir=str(seed_output_dir),
            seed=seed,
            n_eval_episodes=n_eval_episodes,
            config_overrides=config_overrides,
            resume=resume,
        )
        all_rows.extend(_extract_inactivity_ablation_rows(results, seed))

    results_df = pd.DataFrame(all_rows)
    results_csv_path = output_path / "inactivity_ablation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nWrote tidy inactivity ablation results to {results_csv_path} ({len(results_df)} rows)")

    if results_df.empty:
        return {'results_df': results_df, 'summary_df': pd.DataFrame(), 'output_dir': str(output_path)}

    summary = (
        results_df
        .groupby(['inactivity_variant'])[METRIC_COLUMNS]
        .agg(['mean', 'std'])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    summary_csv_path = output_path / "inactivity_ablation_summary.csv"
    summary.to_csv(summary_csv_path, index=False)
    print(f"Wrote inactivity ablation summary to {summary_csv_path}")
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(summary.round(3).to_string(index=False))

    return {'results_df': results_df, 'summary_df': summary, 'output_dir': str(output_path)}


def run_paper_experiments(
        tickers: Optional[List[str]] = None,
        windows: Optional[List[Dict[str, Tuple[str, str]]]] = None,
        shield_arms: Optional[List[str]] = None,
        trainer_arms: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        total_timesteps: int = 50000,
        use_synthetic: bool = False,
        output_dir: str = "./results/paper_experiments",
        n_eval_episodes: int = 5,
        include_shield_ablation: bool = True,
        shield_ablation_seeds: Optional[List[int]] = None,
        include_inactivity_ablation: bool = True,
        inactivity_ablation_seeds: Optional[List[int]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        resume: bool = False,
) -> Dict[str, Any]:
    """
    Run the full paper grid and write results.csv + summary.csv, plus (by
    default) the shield-component ablation and the inactivity-penalty
    ablation (each ppo_lagrangian-scoped, window 1 only, its own seed list)
    alongside it.

    Args:
        tickers: Assets to trade (default: main.py's 5-asset default).
        windows: Calendar train/test windows (default: PAPER_WALK_FORWARD_WINDOWS).
        shield_arms: Shield arms to run (default: SHIELD_ARMS, both arms).
        trainer_arms: Trainer arms to run (default: TRAINER_ARMS, all three).
        seeds: Random seeds (default: [1, 2, 3, 4, 5]).
        total_timesteps: Training timesteps per (fold, arm, seed).
        use_synthetic: Use synthetic data instead of real market data.
        output_dir: Root output directory; a timestamped subdirectory holds
            this run's per-seed outputs plus results.csv/summary.csv (unless
            resume=True -- see below).
        n_eval_episodes: Evaluation episodes per (fold, arm, seed).
        include_shield_ablation: If True (default), also run
            run_shield_ablation_grid() into the same timestamped directory.
            Set False to run only the main shield x trainer grid (e.g. for a
            quick dry run scoped to just that grid).
        shield_ablation_seeds: Seeds for the shield ablation specifically
            (default: same as `seeds`).
        include_inactivity_ablation: If True (default), also run
            run_inactivity_penalty_ablation_grid() (shield=none/ppo_lagrangian,
            window 1, main.INACTIVITY_PENALTY_VARIANTS) into the same
            timestamped directory.
        inactivity_ablation_seeds: Seeds for the inactivity ablation
            specifically (default: same as `seeds`).
        config_overrides: Optional {"section.field": value} dict forwarded
            to main.run_calendar_walk_forward_experiment() AND to both
            ablation grids, applied uniformly to every arm (main grid and
            both ablations alike) -- carries a frozen execution-model
            config (e.g. canonical no_trade_band/decision_frequency_days/
            action_smoothing_beta) into production runs, recorded in every
            arm's own config.json for reproducibility.
        resume: If True, `output_dir` is used AS-IS (no fresh
            datetime.now() timestamp subdirectory) and every (fold, shield,
            trainer, seed) combo -- plus every ablation variant/seed -- that
            already has a completed eval_result.json is skipped and reloaded
            from disk instead of retrained. Pass the SAME output_dir across
            restarts of a long grid to resume it; a fresh (non-resumed) call
            always gets its own new timestamped directory, so resuming
            requires the caller to have captured that directory from the
            first call (e.g. from a detached launcher that picks the
            timestamp once up front -- see run_full_grid.py).

    Returns:
        Dict with 'results_df', 'summary_df', 'output_dir', and (if
        include_shield_ablation / include_inactivity_ablation)
        'shield_ablation_results_df'/'shield_ablation_summary_df' and/or
        'inactivity_ablation_results_df'/'inactivity_ablation_summary_df'.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if windows is None:
        windows = main_module.PAPER_WALK_FORWARD_WINDOWS

    output_path = Path(output_dir) if resume else Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  PAPER EXPERIMENT GRID")
    print("=" * 70)
    print(f"Windows: {len(windows)}")
    print(f"Shield arms: {shield_arms or main_module.SHIELD_ARMS}")
    print(f"Trainer arms: {trainer_arms or main_module.TRAINER_ARMS}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {output_path}")
    if resume:
        print("Resume mode: skipping any (fold, shield, trainer, seed) combo "
              "with a completed eval_result.json already on disk.")

    all_rows = []
    for seed in seeds:
        print(f"\n{'#' * 70}")
        print(f"# SEED {seed}")
        print(f"{'#' * 70}")

        seed_output_dir = output_path / f"seed_{seed}"
        results = main_module.run_calendar_walk_forward_experiment(
            tickers=tickers,
            windows=windows,
            shield_arms=shield_arms,
            trainer_arms=trainer_arms,
            total_timesteps=total_timesteps,
            use_synthetic=use_synthetic,
            output_dir=str(seed_output_dir),
            seed=seed,
            n_eval_episodes=n_eval_episodes,
            config_overrides=config_overrides,
            resume=resume,
        )
        all_rows.extend(_extract_rows(results, seed))

    results_df = pd.DataFrame(all_rows)

    results_csv_path = output_path / "results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nWrote tidy results to {results_csv_path} ({len(results_df)} rows)")

    result = {'results_df': results_df, 'summary_df': pd.DataFrame(), 'output_dir': str(output_path)}

    if not results_df.empty:
        summary = (
            results_df
            .groupby(['method', 'shield', 'window'])[METRIC_COLUMNS]
            .agg(['mean', 'std'])
        )
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()

        summary_csv_path = output_path / "summary.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"Wrote summary table to {summary_csv_path}")

        print("\n" + "=" * 100)
        print("SUMMARY (mean +/- std across seeds)")
        print("=" * 100)
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            print(summary.to_string(index=False))

        result['summary_df'] = summary
    else:
        print("No results collected -- skipping summary (every arm/fold/seed failed?).")

    if include_shield_ablation:
        ablation_result = run_shield_ablation_grid(
            tickers=tickers,
            seeds=shield_ablation_seeds if shield_ablation_seeds is not None else seeds,
            total_timesteps=total_timesteps,
            use_synthetic=use_synthetic,
            output_dir=str(output_path),
            n_eval_episodes=n_eval_episodes,
            config_overrides=config_overrides,
            resume=resume,
        )
        result['shield_ablation_results_df'] = ablation_result['results_df']
        result['shield_ablation_summary_df'] = ablation_result['summary_df']

    if include_inactivity_ablation:
        inactivity_result = run_inactivity_penalty_ablation_grid(
            tickers=tickers,
            seeds=inactivity_ablation_seeds if inactivity_ablation_seeds is not None else seeds,
            total_timesteps=total_timesteps,
            use_synthetic=use_synthetic,
            output_dir=str(output_path),
            n_eval_episodes=n_eval_episodes,
            config_overrides=config_overrides,
            resume=resume,
        )
        result['inactivity_ablation_results_df'] = inactivity_result['results_df']
        result['inactivity_ablation_summary_df'] = inactivity_result['summary_df']

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full paper experiment grid")
    parser.add_argument("--tickers", type=str, nargs="+", default=None,
                        help="List of ticker symbols (default: main.py's 5-asset default)")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Training timesteps per (fold, arm, seed)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds to run (default: 1 2 3 4 5)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--output-dir", type=str, default="./results/paper_experiments",
                        help="Root output directory")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                        help="Evaluation episodes per (fold, arm, seed)")
    parser.add_argument("--no-shield-ablation", action="store_true",
                        help="Skip the shield-component ablation (runs by default alongside the main grid)")
    parser.add_argument("--no-inactivity-ablation", action="store_true",
                        help="Skip the inactivity-penalty ablation (runs by default alongside the main grid)")
    parser.add_argument("--canonical-fold2-config", action="store_true",
                        help="Apply the frozen canonical execution-model config (band010_k5, see "
                             "PROJECT_STATE.md) uniformly to every arm via config_overrides.")
    parser.add_argument("--resume", action="store_true",
                        help="Use --output-dir as-is (no fresh timestamp subdirectory) and skip any "
                             "(fold, shield, trainer, seed) combo or ablation variant/seed that already "
                             "has a completed eval_result.json on disk, instead of retraining it.")
    args = parser.parse_args()

    run_paper_experiments(
        tickers=args.tickers,
        seeds=args.seeds,
        total_timesteps=args.timesteps,
        use_synthetic=args.synthetic,
        output_dir=args.output_dir,
        n_eval_episodes=args.n_eval_episodes,
        include_shield_ablation=not args.no_shield_ablation,
        include_inactivity_ablation=not args.no_inactivity_ablation,
        config_overrides=CANONICAL_FOLD2_CONFIG_OVERRIDES if args.canonical_fold2_config else None,
        resume=args.resume,
    )

    print("\nDone!")
