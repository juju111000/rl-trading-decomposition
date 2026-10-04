"""
Regression tests for the calendar walk-forward wiring (A.4).

Covers:
- WalkForwardEvaluator.generate_calendar_splits() resolves the paper's three
  fixed calendar windows to non-overlapping integer index ranges.
- Deliberately overlapping windows are rejected (proves the guard works).
- An end-to-end smoke run of main.run_calendar_walk_forward_experiment()
  against real cached data for fold 1 (whose combined date range matches an
  existing data_cache file, so no network access is needed), asserting no
  train/test date overlap in the actual saved results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from evaluation import WalkForwardEvaluator
from config import Config

PAPER_WINDOWS = [
    {'train': ('2006-01-01', '2007-12-31'), 'test': ('2008-01-01', '2010-12-31')},
    {'train': ('2010-01-01', '2014-12-31'), 'test': ('2015-01-01', '2017-12-31')},
    {'train': ('2017-01-01', '2019-12-31'), 'test': ('2020-01-01', '2023-12-31')},
]


def _business_day_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq='B')


def test_paper_windows_resolve_to_non_overlapping_folds():
    dates = _business_day_index("2005-06-01", "2024-06-30")
    evaluator = WalkForwardEvaluator(Config())

    splits = evaluator.generate_calendar_splits(dates, PAPER_WINDOWS)

    assert len(splits) == 3

    for i, (window, split) in enumerate(zip(PAPER_WINDOWS, splits)):
        train_start_idx, train_end_idx = split['train']
        test_start_idx, test_end_idx = split['test']

        # Non-overlapping: train ends strictly before test begins.
        assert train_end_idx <= test_start_idx, f"fold {i}: train/test overlap"

        # Non-empty on both sides.
        assert train_end_idx > train_start_idx
        assert test_end_idx > test_start_idx

        # Resolved dates land within a few calendar days of the request
        # (weekends/holidays), never outside it.
        resolved_train_start = dates[train_start_idx]
        resolved_train_end = dates[train_end_idx - 1]
        resolved_test_start = dates[test_start_idx]
        resolved_test_end = dates[test_end_idx - 1]

        assert resolved_train_start >= pd.Timestamp(window['train'][0])
        assert resolved_train_end <= pd.Timestamp(window['train'][1])
        assert resolved_test_start >= pd.Timestamp(window['test'][0])
        assert resolved_test_end <= pd.Timestamp(window['test'][1])

        # And strictly ordered in time.
        assert resolved_train_end < resolved_test_start


def test_overlapping_windows_are_rejected():
    dates = _business_day_index("2005-06-01", "2024-06-30")
    evaluator = WalkForwardEvaluator(Config())

    overlapping_windows = [
        {'train': ('2006-01-01', '2008-06-30'), 'test': ('2008-01-01', '2010-12-31')},
    ]

    with pytest.raises(ValueError, match="overlaps"):
        evaluator.generate_calendar_splits(dates, overlapping_windows)


def test_empty_range_is_rejected():
    dates = _business_day_index("2010-01-01", "2020-12-31")
    evaluator = WalkForwardEvaluator(Config())

    # Entirely before the available data -- empty train range.
    windows = [{'train': ('1990-01-01', '1990-06-30'), 'test': ('2015-01-01', '2015-12-31')}]

    with pytest.raises(ValueError, match="Empty"):
        evaluator.generate_calendar_splits(dates, windows)


def test_calendar_walk_forward_end_to_end_smoke(tmp_path):
    """Runs main.run_calendar_walk_forward_experiment against real cached
    data (fold 1's combined range 2006-01-01..2010-12-31 matches an existing
    data_cache/raw_data_2006-01-01_2010-12-31.pkl, so this needs no network
    access), restricted to one fold and one (shield, trainer) arm for speed."""
    import main

    fold_1 = [PAPER_WINDOWS[0]]

    results = main.run_calendar_walk_forward_experiment(
        tickers=["^GSPC"],
        windows=fold_1,
        shield_arms=["uncertainty_aware"],
        trainer_arms=["ppo_lagrangian"],
        total_timesteps=2048,  # one rollout iteration
        use_synthetic=False,
        output_dir=str(tmp_path),
        seed=1,
        n_eval_episodes=1,
    )

    assert results['windows'] == fold_1
    assert len(results['folds']) == 1

    fold = results['folds'][0]
    assert fold['train_dates'] == fold_1[0]['train']
    assert fold['test_dates'] == fold_1[0]['test']

    arm_name = "uncertainty_aware/ppo_lagrangian"
    assert arm_name in fold['results']
    assert 'aggregate' in fold['results'][arm_name]

    # The saved per-arm config must reflect THIS arm's own shield_type, not
    # some other arm's or the top-level base config's.
    from config import Config
    arm_config_path = list(tmp_path.rglob(f"fold_1/{arm_name}/config.json"))
    assert len(arm_config_path) == 1
    arm_config = Config.from_json(str(arm_config_path[0]))
    assert arm_config.shield.shield_type.value == "uncertainty_aware"

    # No overlap between this fold's actual train/test date ranges.
    assert pd.Timestamp(fold['train_dates'][1]) < pd.Timestamp(fold['test_dates'][0])

    # Results were actually written to disk.
    saved = list(tmp_path.rglob("walk_forward_results.json"))
    assert len(saved) == 1


if __name__ == "__main__":
    test_paper_windows_resolve_to_non_overlapping_folds()
    print("PASS: paper windows resolve to non-overlapping folds")

    test_overlapping_windows_are_rejected()
    print("PASS: overlapping windows rejected")

    test_empty_range_is_rejected()
    print("PASS: empty range rejected")
