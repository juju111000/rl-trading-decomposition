"""
Regression tests for gating the inactivity penalty behind a config flag
(B.10).

The inactivity penalty (trading_env.py) is reward shaping bolted onto the
raw log-return reward -- not part of the CMDP cost/constraint machinery --
and was previously unconditional (always active for every trainer type, with
no way to run a clean ablation arm without it). EnvironmentConfig now has
use_inactivity_penalty (default True, preserving prior behavior), and
trading_env.py gates the penalty behind it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from config import (
    Config, DataConfig, FeatureConfig, RegimeConfig, ConstraintConfig,
    ShieldConfig, ActionSpaceConfig, TrainingConfig, EnvironmentConfig,
    ActionSpaceType, RegimeModelType, ShieldType,
)
from data_loader import SyntheticDataGenerator
from features import FeatureBuilder
from regime_model import create_regime_model
from policy_shield import create_shield
from trading_env import TradingCMDPEnv

TICKER = "ASSET_0"


def _build_env(use_inactivity_penalty):
    config = Config()
    config.data = DataConfig(tickers=[TICKER])
    config.features = FeatureConfig()
    config.regime = RegimeConfig(model_type=RegimeModelType.HEURISTIC, n_regimes=3)
    config.constraints = ConstraintConfig()
    config.shield = ShieldConfig(shield_type=ShieldType.NONE)  # isolate the penalty itself
    config.action_space = ActionSpaceConfig(action_type=ActionSpaceType.CONTINUOUS)
    config.environment = EnvironmentConfig(use_inactivity_penalty=use_inactivity_penalty)
    config.training = TrainingConfig(episode_length=50, use_rolling_episodes=False, seed=1)

    gen = SyntheticDataGenerator(n_assets=1, n_days=100, seed=3)
    data, _ = gen.generate_regime_switching_data()

    builder = FeatureBuilder(config.features)
    features_df = builder.compute_features(data)
    prices = {TICKER: data[(TICKER, "Close")].values}

    regime_model = create_regime_model(config.regime, input_dim=features_df.shape[1])
    regime_model.fit(data)
    shield = create_shield(config.shield, config.constraints, n_assets=1)

    env = TradingCMDPEnv(
        features=features_df.values,
        prices=prices,
        tickers=[TICKER],
        config=config,
        regime_model=regime_model,
        shield=shield,
        mode="train",
    )
    return env


def test_inactivity_penalty_applied_by_default_when_underinvested():
    """Default behavior (use_inactivity_penalty=True) must be unchanged."""
    env = _build_env(use_inactivity_penalty=True)
    env.reset(seed=1)

    # All-cash action: zero exposure, and since the portfolio starts fully
    # in cash, log_return is exactly 0 -- any reward != 0 is purely the
    # inactivity penalty.
    _, reward, _, _, _ = env.step(np.array([0.0]))

    # inactivity_penalty_rate(0.02) * (min_exposure(0.2) - 0) * reward_scale(100) = 0.4
    assert reward == pytest.approx(-0.4)


def test_inactivity_penalty_disabled_when_flag_is_off():
    """Core acceptance check: the flag must actually turn the penalty off."""
    env = _build_env(use_inactivity_penalty=False)
    env.reset(seed=1)

    _, reward, _, _, _ = env.step(np.array([0.0]))

    assert reward == pytest.approx(0.0, abs=1e-9)


if __name__ == "__main__":
    test_inactivity_penalty_applied_by_default_when_underinvested()
    print("PASS: inactivity penalty applied by default when underinvested")

    test_inactivity_penalty_disabled_when_flag_is_off()
    print("PASS: inactivity penalty disabled when flag is off")
