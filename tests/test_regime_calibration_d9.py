"""
Regression tests for audit defect #9 (regime look-ahead).

HeuristicRegimeModel's volatility-percentile thresholds must be calibrated on
TRAINING data only and frozen at test time: a non-training fit must never
compute anything from (or be altered by) test-window data, and must fail
closed if no training calibration exists.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from config import RegimeConfig
from regime_model import HeuristicRegimeModel, RegimeCalibrationError

TICKERS = ["MKT", "B"]


def _aligned(n, daily_vol, seed, start="2010-01-01"):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range(start, periods=n)
    cols = {}
    for t in TICKERS:
        close = 100 * np.exp(np.cumsum(rng.normal(0, daily_vol, n)))
        for f in ("Open", "High", "Low", "Close"):
            cols[(t, f)] = close
        cols[(t, "Volume")] = np.full(n, 1e6)
    return pd.DataFrame(cols, index=idx)


def test_test_window_cannot_alter_thresholds():
    cfg = RegimeConfig()
    train = _aligned(500, 0.01, seed=0)
    test_calm = _aligned(300, 0.003, seed=1, start="2015-01-01")
    test_wild = _aligned(300, 0.05, seed=2, start="2015-01-01")

    m_train = HeuristicRegimeModel(cfg)
    m_train.fit(train, mode="train")
    frozen = tuple(m_train._vol_thresholds)
    assert cfg.calibrated_vol_thresholds == list(frozen)

    for test in (test_calm, test_wild):
        m_test = HeuristicRegimeModel(copy.deepcopy(cfg))
        m_test.fit(test, mode="test")
        assert tuple(m_test._vol_thresholds) == frozen        # unchanged by test data
        assert m_test._close_history == []                    # no test values read
    assert cfg.calibrated_vol_thresholds == list(frozen)

    # a train-mode fit on the wild window WOULD give different thresholds (sanity: the test is not vacuous)
    m_wild = HeuristicRegimeModel(RegimeConfig())
    m_wild.fit(test_wild, mode="train")
    assert tuple(m_wild._vol_thresholds) != frozen


def test_non_training_fit_fails_closed_without_calibration():
    m = HeuristicRegimeModel(RegimeConfig())
    with pytest.raises(RegimeCalibrationError):
        m.fit(_aligned(300, 0.01, seed=3), mode="test")
