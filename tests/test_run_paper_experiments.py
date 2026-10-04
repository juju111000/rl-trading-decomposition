"""
Regression tests for run_paper_experiments.py (Phase C).

Covers the tidy-CSV flattening logic in isolation (fast, no training), and
an end-to-end smoke run of the real entry point against cached data,
restricted to a small slice of the full grid for speed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import run_paper_experiments as rpe


def _fake_arm_result(total_return, sharpe_ratio, max_drawdown, dd_violation_rate,
                      avg_turnover, shield_intervention_rate):
    return {
        'aggregate': {
            'total_return': (total_return, 0.0),
            'sharpe_ratio': (sharpe_ratio, 0.0),
            'max_drawdown': (max_drawdown, 0.0),
            'dd_violation_rate': (dd_violation_rate, 0.0),
            'avg_turnover': (avg_turnover, 0.0),
            'shield_intervention_rate': (shield_intervention_rate, 0.0),
        },
    }


def test_window_label_uses_test_period_years():
    assert rpe._window_label(('2008-01-01', '2010-12-31')) == '2008-2010'
    assert rpe._window_label(('2015-01-01', '2017-12-31')) == '2015-2017'


def test_extract_rows_flattens_fold_and_arm_results_correctly():
    fake_results = {
        'windows': [{'train': ('2006-01-01', '2007-12-31'), 'test': ('2008-01-01', '2010-12-31')}],
        'folds': [
            {
                'fold': 1,
                'train_dates': ('2006-01-01', '2007-12-31'),
                'test_dates': ('2008-01-01', '2010-12-31'),
                'results': {
                    'none/vanilla_ppo': _fake_arm_result(-0.30, -2.0, 0.35, 0.80, 0.5, 0.0),
                    'uncertainty_aware/ppo_lagrangian': _fake_arm_result(0.02, 0.3, 0.06, 0.0, 0.3, 0.85),
                },
            },
        ],
    }

    rows = rpe._extract_rows(fake_results, seed=7)

    assert len(rows) == 2
    by_arm = {(r['shield'], r['method']): r for r in rows}

    row = by_arm[('none', 'vanilla_ppo')]
    assert row['window'] == '2008-2010'
    assert row['seed'] == 7
    assert row['return'] == pytest.approx(-0.30)
    assert row['sharpe'] == pytest.approx(-2.0)
    assert row['max_dd'] == pytest.approx(0.35)
    assert row['violation_rate'] == pytest.approx(0.80)
    assert row['turnover'] == pytest.approx(0.5)
    assert row['shield_intervention_rate'] == pytest.approx(0.0)

    row = by_arm[('uncertainty_aware', 'ppo_lagrangian')]
    assert row['return'] == pytest.approx(0.02)
    assert row['shield_intervention_rate'] == pytest.approx(0.85)


def test_extract_rows_handles_missing_aggregate_gracefully():
    fake_results = {
        'folds': [
            {
                'test_dates': ('2020-01-01', '2023-12-31'),
                'results': {'none/vanilla_ppo': {}},  # arm ran but has no 'aggregate' key
            },
        ],
    }
    rows = rpe._extract_rows(fake_results, seed=1)
    assert len(rows) == 1
    assert np.isnan(rows[0]['return'])


def test_run_paper_experiments_end_to_end_smoke(tmp_path):
    """Real entry point, restricted to one fold / one arm / two seeds for
    speed, against cached ^GSPC data (fold 1's combined range matches an
    existing data_cache file, so no network access is needed)."""
    fold_1 = [{'train': ('2006-01-01', '2007-12-31'), 'test': ('2008-01-01', '2010-12-31')}]

    result = rpe.run_paper_experiments(
        tickers=["^GSPC"],
        windows=fold_1,
        shield_arms=["uncertainty_aware"],
        trainer_arms=["ppo_lagrangian"],
        seeds=[1, 2],
        total_timesteps=2048,  # one rollout iteration
        use_synthetic=False,
        output_dir=str(tmp_path),
        n_eval_episodes=1,
    )

    results_df = result['results_df']
    summary_df = result['summary_df']

    # 1 window x 1 arm x 2 seeds = 2 rows.
    assert len(results_df) == 2
    assert set(results_df['seed']) == {1, 2}
    assert (results_df['method'] == 'ppo_lagrangian').all()
    assert (results_df['shield'] == 'uncertainty_aware').all()
    assert (results_df['window'] == '2008-2010').all()
    for col in rpe.METRIC_COLUMNS:
        assert col in results_df.columns

    # 1 window x 1 arm combo -> 1 summary row.
    assert len(summary_df) == 1
    assert summary_df.loc[0, 'method'] == 'ppo_lagrangian'
    assert summary_df.loc[0, 'shield'] == 'uncertainty_aware'
    assert summary_df.loc[0, 'window'] == '2008-2010'

    # Summary mean/std must match manual computation from the raw rows.
    expected_mean = results_df['return'].mean()
    expected_std = results_df['return'].std()
    assert summary_df.loc[0, 'return_mean'] == pytest.approx(expected_mean)
    assert summary_df.loc[0, 'return_std'] == pytest.approx(expected_std)

    # Files actually written to disk.
    output_dir = Path(result['output_dir'])
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "summary.csv").exists()

    reloaded = pd.read_csv(output_dir / "results.csv")
    assert len(reloaded) == 2


if __name__ == "__main__":
    test_window_label_uses_test_period_years()
    print("PASS: window label uses test period years")

    test_extract_rows_flattens_fold_and_arm_results_correctly()
    print("PASS: _extract_rows flattens fold/arm results correctly")

    test_extract_rows_handles_missing_aggregate_gracefully()
    print("PASS: _extract_rows handles missing aggregate gracefully")
