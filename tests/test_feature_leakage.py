"""
Regression test for feature leakage (A.3).

main.py::load_and_prepare_data() used to call FeatureBuilder.compute_features()
separately per split (train_ohlcv, val_ohlcv, test_ohlcv). Because rolling
windows (returns, volatility, MA ratios, RSI, MACD, cross-asset correlation)
only look backward, restarting them at the start of each split "cold starts"
with no real prior history, and compute_features()'s NaN-fill
(ffill().bfill()) would then backfill those leading NaNs using LATER rows
within that same split -- a real, if localized, lookahead leak at the start
of val/test.

The fix computes features once over the full aligned OHLCV series and then
slices into train/val/test. This test pins down the invariant that fix
depends on: a given row's features must be identical whether or not rows
strictly after it exist in the DataFrame passed to compute_features().
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import FeatureConfig
from data_loader import SyntheticDataGenerator
from features import FeatureBuilder


def test_first_test_set_row_unaffected_by_later_test_set_data():
    gen = SyntheticDataGenerator(n_assets=2, n_days=400, seed=11)
    aligned_data, _ = gen.generate_regime_switching_data()

    train_end = 200
    val_end = 250  # test split starts at row index 250

    feature_config = FeatureConfig()

    # What main.py now does: one pass over the whole series, then slice.
    full_builder = FeatureBuilder(feature_config)
    all_features = full_builder.compute_features(aligned_data, fit_normalization=True)
    first_test_row = all_features.iloc[val_end]

    # Standalone computation using only data up to and including the first
    # test row -- i.e. a world where later test-set rows don't exist yet.
    truncated_builder = FeatureBuilder(feature_config)
    features_no_future = truncated_builder.compute_features(
        aligned_data.iloc[:val_end + 1], fit_normalization=True
    )
    first_test_row_no_future = features_no_future.iloc[-1]

    pd.testing.assert_series_equal(
        first_test_row, first_test_row_no_future, check_names=False
    )


def test_old_per_split_computation_would_have_differed(monkeypatch=None):
    """Documents *why* the fix matters: recomputing the test split in
    isolation (the old behavior) gives a different first row than computing
    causally over the full series, because rolling windows cold-start and
    bfill borrows from within-split future rows instead of real history."""
    gen = SyntheticDataGenerator(n_assets=2, n_days=400, seed=11)
    aligned_data, _ = gen.generate_regime_switching_data()

    val_end = 250

    feature_config = FeatureConfig()

    full_builder = FeatureBuilder(feature_config)
    all_features = full_builder.compute_features(aligned_data, fit_normalization=True)
    first_test_row_correct = all_features.iloc[val_end]

    # Old behavior: compute features on the test slice ALONE (no preceding
    # history at all), exactly as the pre-fix main.py did.
    test_ohlcv = aligned_data.iloc[val_end:]
    isolated_builder = FeatureBuilder(feature_config)
    isolated_features = isolated_builder.compute_features(test_ohlcv, fit_normalization=False)
    first_test_row_isolated = isolated_features.iloc[0]

    # These should NOT match -- confirming the old approach was indeed wrong,
    # not merely differently-but-equivalently computed.
    assert not first_test_row_correct.equals(first_test_row_isolated)


if __name__ == "__main__":
    test_first_test_set_row_unaffected_by_later_test_set_data()
    print("PASS: first test-set row identical whether or not later test-set data exists")

    test_old_per_split_computation_would_have_differed()
    print("PASS: confirmed the old per-split computation actually differed (regression guard)")
