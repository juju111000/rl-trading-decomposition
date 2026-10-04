"""
Regression tests for feature-warmup handling (audit #14).

bfill()-based warmup healing backfills leading NaN rows from LATER rows --
forward-looking leakage, the same class flagged by audit #14. The fix is to
request real prior history ahead of a fold's nominal start
(main.FEATURE_WARMUP_BUFFER_DAYS) and trim whatever leading NaN survives
that buffer (main.load_and_prepare_data), never to backfill it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import main as main_module


def test_no_bfill_call_anywhere_in_feature_pipeline():
    source = (Path(__file__).parent.parent / "features.py").read_text()
    assert "bfill(" not in source, (
        "features.py must not backfill NaN -- that is forward-looking "
        "leakage (audit #14). Request more warmup history and trim leading "
        "NaN instead (see main.load_and_prepare_data / "
        "main.FEATURE_WARMUP_BUFFER_DAYS)."
    )


def test_load_and_prepare_data_trims_leading_warmup_nan_without_bfill():
    """The 200-day MA lookback (config.py's default ma_windows) needs far
    more real history than FEATURE_WARMUP_BUFFER_DAYS (~28 trading days)
    provides, so this must exercise real trimming, not merely find nothing
    to trim -- otherwise the test would pass vacuously."""
    nominal_start = "2015-01-01"
    end_date = "2016-06-30"
    buffered_start = main_module._buffered_start_date(nominal_start)

    config = main_module.create_experiment_config(
        tickers=["ASSET_0", "ASSET_1"],
        start_date=buffered_start,
        end_date=end_date,
        total_timesteps=1000,
        use_synthetic=True,
        seed=1,
    )

    features, prices_df, splits, returns, aligned_data = main_module.load_and_prepare_data(
        config, use_synthetic=True
    )

    assert not np.isnan(features.values).any(), "NaN survived warmup trimming"
    assert len(features) == len(aligned_data), "features/aligned_data must stay positionally aligned"

    total_bdays_requested = len(_bdate_range(buffered_start, end_date))
    assert len(features) < total_bdays_requested, (
        "expected leading rows to actually be trimmed (200-day MA lookback "
        "exceeds the warmup buffer) -- if this fails, the test fixture no "
        "longer exercises real trimming"
    )


def _bdate_range(start, end):
    import pandas as pd
    return pd.bdate_range(start, end)


if __name__ == "__main__":
    test_no_bfill_call_anywhere_in_feature_pipeline()
    print("PASS: no bfill() call anywhere in features.py")

    test_load_and_prepare_data_trims_leading_warmup_nan_without_bfill()
    print("PASS: load_and_prepare_data trims leading warmup NaN without bfill")
