"""
Regression tests for wiring ConstraintConfig.constraint_budgets into the
trainer (B.9).

constraint_budgets (d_k in the CMDP formulation) was defined in config.py
and saved into every config.json, but never read anywhere in the codebase --
PPOLagrangianTrainer.__init__ hardcoded self.constraint_limits =
[0.0, 0.0, 0.0] directly instead of reading it from config, making the field
pure dead weight (and, without the docstring added alongside this fix,
misleadingly suggesting budgets were "unset" when the real per-constraint
budgets are dd_limit/turnover_limit/leverage_limit, already folded into the
cost computation upstream in RiskManager.compute_costs()).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from config import (
    Config, DataConfig, FeatureConfig, RegimeConfig, ConstraintConfig,
    ShieldConfig, ActionSpaceConfig, PPOConfig, TrainingConfig,
    EnvironmentConfig, ActionSpaceType, RegimeModelType, ShieldType,
)
from data_loader import SyntheticDataGenerator
from features import FeatureBuilder
from regime_model import create_regime_model
from policy_shield import create_shield
from trading_env import TradingCMDPEnv
from trainer import PPOLagrangianTrainer


def _build_env_and_config(constraint_budgets):
    config = Config()
    config.data = DataConfig(tickers=["ASSET_0", "ASSET_1"])
    config.features = FeatureConfig()
    config.regime = RegimeConfig(model_type=RegimeModelType.HEURISTIC, n_regimes=3)
    config.constraints = ConstraintConfig(constraint_budgets=constraint_budgets)
    config.shield = ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE)
    config.action_space = ActionSpaceConfig(action_type=ActionSpaceType.CONTINUOUS)
    config.ppo = PPOConfig(hidden_sizes=[16, 16], batch_size=16, n_epochs=1)
    config.training = TrainingConfig(
        total_timesteps=64, rollout_length=64, episode_length=64,
        use_rolling_episodes=True, seed=1,
    )
    config.environment = EnvironmentConfig()

    gen = SyntheticDataGenerator(n_assets=2, n_days=150, seed=5)
    data, _ = gen.generate_regime_switching_data()
    builder = FeatureBuilder(config.features)
    features_df = builder.compute_features(data)
    prices = {t: data[(t, "Close")].values for t in config.data.tickers}

    regime_model = create_regime_model(config.regime, input_dim=features_df.shape[1])
    regime_model.fit(data)
    shield = create_shield(config.shield, config.constraints, n_assets=len(config.data.tickers))

    env = TradingCMDPEnv(
        features=features_df.values,
        prices=prices,
        tickers=config.data.tickers,
        config=config,
        regime_model=regime_model,
        shield=shield,
        mode="train",
    )
    return env, config


def test_default_constraint_limits_are_still_zero():
    """No behavior change for the default config."""
    env, config = _build_env_and_config(constraint_budgets=[0.0, 0.0, 0.0])
    trainer = PPOLagrangianTrainer(env, config, device="cpu")
    np.testing.assert_array_equal(trainer.constraint_limits, [0.0, 0.0, 0.0])


def test_nonzero_constraint_budgets_are_actually_wired_in():
    """Core acceptance check: constraint_budgets must actually drive
    trainer.constraint_limits -- previously this config field was defined
    but never read anywhere in the codebase."""
    env, config = _build_env_and_config(constraint_budgets=[0.05, 0.2, 0.3])
    trainer = PPOLagrangianTrainer(env, config, device="cpu")
    np.testing.assert_allclose(trainer.constraint_limits, [0.05, 0.2, 0.3])


def test_update_dual_variables_uses_the_configured_budgets():
    """End-to-end: the dual update's violation calc must actually subtract
    the configured budget, not a hardcoded zero."""
    env, config = _build_env_and_config(constraint_budgets=[0.05, 0.0, 0.0])
    trainer = PPOLagrangianTrainer(env, config, device="cpu")

    # Fabricate buffer costs directly rather than running a full rollout.
    trainer.buffer.ptr = 4
    trainer.buffer.costs[:4] = np.array([
        [0.10, 0.0, 0.0],
        [0.10, 0.0, 0.0],
        [0.10, 0.0, 0.0],
        [0.10, 0.0, 0.0],
    ])

    _, violations = trainer.update_dual_variables()

    # mean_cost_dd = 0.10, budget = 0.05 -> violation should be 0.05, not the
    # full 0.10 a hardcoded d_k=0 would have given.
    assert violations[0] == pytest.approx(0.05)
    assert violations[1] == pytest.approx(0.0)
    assert violations[2] == pytest.approx(0.0)


if __name__ == "__main__":
    test_default_constraint_limits_are_still_zero()
    print("PASS: default constraint_limits are still [0, 0, 0]")

    test_nonzero_constraint_budgets_are_actually_wired_in()
    print("PASS: nonzero constraint_budgets are wired into trainer.constraint_limits")

    test_update_dual_variables_uses_the_configured_budgets()
    print("PASS: update_dual_variables() uses the configured budgets")
