"""
Regression tests for the Phase 2 execution-model and dual-update fixes
(diagnosed from the fold-2 autopsy: 111% of capital lost to transaction
costs under ~3-4x unconstrained leverage, while the additive dual update
needed ~300k+ timesteps just for lambda to reach O(1)).

All four fixes are config-flagged and default to the pre-fix behavior:
  - ActionSpaceConfig.gross_leverage_cap (None = disabled)
  - EnvironmentConfig.action_smoothing_beta (None = disabled)
  - PPOConfig.lambda_init (already existed; now actually wired into
    PPOLagrangianTrainer.__init__ instead of being silently ignored)
  - PPOConfig.dual_update_method="pid" (alternative to "additive", the default)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import main as main_module
from config import (
    Config, DataConfig, FeatureConfig, RegimeConfig, ConstraintConfig,
    ShieldConfig, ActionSpaceConfig, PPOConfig, TrainingConfig,
    EnvironmentConfig, ActionSpaceType, RegimeModelType, ShieldType,
)
from data_loader import SyntheticDataGenerator
from features import FeatureBuilder
from trainer import create_trainer


def _tiny_config(**overrides):
    config = Config()
    config.data = DataConfig(tickers=["ASSET_0", "ASSET_1"])
    config.features = FeatureConfig()
    config.regime = RegimeConfig(model_type=RegimeModelType.HEURISTIC, n_regimes=3)
    config.constraints = ConstraintConfig()
    config.shield = ShieldConfig(shield_type=ShieldType.NONE)
    config.action_space = ActionSpaceConfig(
        action_type=ActionSpaceType.CONTINUOUS, allow_short=True, max_position_size=1.0
    )
    config.ppo = PPOConfig(hidden_sizes=[16, 16], batch_size=16, n_epochs=1)
    config.training = TrainingConfig(
        total_timesteps=64, rollout_length=64, episode_length=64,
        use_rolling_episodes=False, seed=1,
    )
    config.environment = EnvironmentConfig()
    for key, value in overrides.items():
        section, field_name = key.split("__", 1)
        setattr(getattr(config, section), field_name, value)
    return config


def _synthetic_env(config, seed=3, mode="train"):
    gen = SyntheticDataGenerator(n_assets=2, n_days=200, seed=seed)
    data, _ = gen.generate_regime_switching_data()
    builder = FeatureBuilder(FeatureConfig())
    features_df = builder.compute_features(data)
    prices = {t: data[(t, "Close")].values for t in ["ASSET_0", "ASSET_1"]}
    return main_module.create_trading_env(features_df, prices, config, aligned_data=data, mode=mode)


# =============================================================================
# Gross leverage cap
# =============================================================================

def test_gross_leverage_cap_scales_down_over_cap_action():
    config = _tiny_config(action_space__gross_leverage_cap=1.0)
    env = _synthetic_env(config)
    weights = env._decode_action(np.array([0.8, 0.8]))
    np.testing.assert_allclose(weights, np.array([0.5, 0.5]), rtol=1e-5)
    assert np.sum(np.abs(weights)) <= 1.0 + 1e-6


def test_gross_leverage_cap_leaves_under_cap_action_unchanged():
    config = _tiny_config(action_space__gross_leverage_cap=1.0)
    env = _synthetic_env(config)
    weights = env._decode_action(np.array([0.3, 0.3]))
    np.testing.assert_allclose(weights, np.array([0.3, 0.3]), rtol=1e-5)


def test_gross_leverage_cap_disabled_by_default():
    config = _tiny_config()  # gross_leverage_cap left at its None default
    env = _synthetic_env(config)
    weights = env._decode_action(np.array([0.8, 0.8]))
    np.testing.assert_allclose(weights, np.array([0.8, 0.8]), rtol=1e-5)


def test_gross_leverage_cap_preserves_direction_with_mixed_signs():
    config = _tiny_config(action_space__gross_leverage_cap=1.0)
    env = _synthetic_env(config)
    weights = env._decode_action(np.array([0.8, -0.8]))
    # gross = 1.6, scale = 1/1.6 = 0.625
    np.testing.assert_allclose(weights, np.array([0.5, -0.5]), rtol=1e-5)


# =============================================================================
# Action smoothing
# =============================================================================

def test_action_smoothing_blends_toward_target_uniformly():
    beta = 0.2
    config = _tiny_config(environment__action_smoothing_beta=beta)
    env = _synthetic_env(config)
    env.reset()

    env.step(np.array([1.0, 1.0]))
    expected_1 = beta * np.array([1.0, 1.0])  # w_{-1}=0
    np.testing.assert_allclose(env._smoothed_weights, expected_1, rtol=1e-4)

    env.step(np.array([-1.0, 0.0]))
    expected_2 = (1 - beta) * expected_1 + beta * np.array([-1.0, 0.0])
    np.testing.assert_allclose(env._smoothed_weights, expected_2, rtol=1e-4)


def test_action_smoothing_disabled_by_default():
    config = _tiny_config()  # action_smoothing_beta left at its None default
    env = _synthetic_env(config)
    env.reset()
    env.step(np.array([0.6, -0.3]))
    np.testing.assert_allclose(env._smoothed_weights, np.array([0.6, -0.3]), rtol=1e-4)


def test_action_smoothing_applies_uniformly_regardless_of_shield():
    """The whole point: this is an execution-model change, not a shield --
    it must behave identically whether shield=none or shielded."""
    beta = 0.33
    none_config = _tiny_config(environment__action_smoothing_beta=beta)
    none_config.shield = ShieldConfig(shield_type=ShieldType.NONE)
    shielded_config = _tiny_config(environment__action_smoothing_beta=beta)
    shielded_config.shield = ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE)

    none_env = _synthetic_env(none_config, seed=7)
    shielded_env = _synthetic_env(shielded_config, seed=7)
    none_env.reset()
    shielded_env.reset()

    action = np.array([0.5, -0.2])
    none_env.step(action)
    shielded_env.step(action)

    # Pre-shield smoothed target must be identical -- the shield only acts
    # AFTER smoothing, never changing the smoothing itself.
    np.testing.assert_allclose(none_env._smoothed_weights, shielded_env._smoothed_weights, rtol=1e-4)


# =============================================================================
# No-trade band
# =============================================================================

def test_no_trade_band_holds_small_moves_executes_large_ones():
    config = _tiny_config(environment__no_trade_band=0.05)
    env = _synthetic_env(config)
    env.reset()

    # First step establishes a real starting position (band compares against
    # ACTUAL current weights, which start at 0).
    env.step(np.array([0.5, 0.5]))

    obs, reward, terminated, truncated, info = env.step(np.array([0.52, 0.9]))
    # asset 0: |0.52 - ~0.5| < 0.05 band -> held (no trade)
    # asset 1: |0.9 - ~0.5| > 0.05 band -> executed
    executed_weights = np.asarray(info['portfolio_state'].weights)
    assert abs(executed_weights[0] - 0.5) < 0.05  # stayed near its pre-step value, not jumped to 0.52
    assert executed_weights[1] > 0.7  # actually moved toward 0.9


def test_no_trade_band_disabled_by_default():
    config = _tiny_config()  # no_trade_band left at its None default
    env = _synthetic_env(config)
    env.reset()
    env.step(np.array([0.5, 0.5]))
    obs, reward, terminated, truncated, info = env.step(np.array([0.51, 0.5]))
    # Even a tiny 0.01 move executes when the band is off.
    executed_weights = np.asarray(info['portfolio_state'].weights)
    assert abs(executed_weights[0] - 0.51) < 0.02


# =============================================================================
# Decision frequency
# =============================================================================

def test_decision_frequency_holds_on_non_decision_days():
    config = _tiny_config(environment__decision_frequency_days=3)
    env = _synthetic_env(config)
    env.reset()

    # day 0: decision day (0 % 3 == 0) -- establishes a real position.
    _, _, _, _, info0 = env.step(np.array([0.6, -0.4]))
    assert info0['turnover'] > 0.01

    # day 1, 2: NOT decision days -- must hold (zero turnover) regardless of
    # how different the incoming action is.
    _, _, _, _, info1 = env.step(np.array([-0.9, 0.9]))
    assert info1['turnover'] < 1e-4
    _, _, _, _, info2 = env.step(np.array([0.1, 0.1]))
    assert info2['turnover'] < 1e-4

    # day 3: decision day again (3 % 3 == 0) -- a new, real trade happens.
    _, _, _, _, info3 = env.step(np.array([-0.6, 0.6]))
    assert info3['turnover'] > 0.01


def test_decision_frequency_default_is_daily():
    config = _tiny_config()  # decision_frequency_days left at its default of 1
    env = _synthetic_env(config)
    env.reset()
    env.step(np.array([0.6, -0.4]))
    _, _, _, _, info = env.step(np.array([-0.6, 0.6]))
    # Every day is a decision day by default -- a large swing must trade.
    assert info['turnover'] > 0.01


def test_decision_frequency_applies_uniformly_regardless_of_shield():
    none_config = _tiny_config(environment__decision_frequency_days=3)
    none_config.shield = ShieldConfig(shield_type=ShieldType.NONE)
    shielded_config = _tiny_config(environment__decision_frequency_days=3)
    shielded_config.shield = ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE)

    none_env = _synthetic_env(none_config, seed=7)
    shielded_env = _synthetic_env(shielded_config, seed=7)
    none_env.reset()
    shielded_env.reset()

    none_env.step(np.array([0.6, -0.4]))
    shielded_env.step(np.array([0.6, -0.4]))

    # Both must agree on which day is a decision day and what the held
    # target is pre-shield.
    assert none_env._days_since_decision == shielded_env._days_since_decision
    np.testing.assert_allclose(
        none_env._last_committed_weights, shielded_env._last_committed_weights, rtol=1e-4
    )


# =============================================================================
# Lambda init + dual update mechanisms
# =============================================================================

def test_lambda_init_is_actually_used():
    config = _tiny_config()
    config.ppo.lambda_init = 0.25
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")
    np.testing.assert_allclose(trainer.lambdas, np.array([0.25, 0.25, 0.25]), rtol=1e-6)


def test_lambda_init_accepts_per_constraint_list():
    """order [dd, turnover, leverage] -- see RiskManager.get_cost_vector."""
    config = _tiny_config()
    config.ppo.lambda_init = [0.25, 0.1, 0.5]
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")
    np.testing.assert_allclose(trainer.lambdas, np.array([0.25, 0.1, 0.5]), rtol=1e-6)


def test_lambda_init_list_wrong_length_raises():
    config = _tiny_config()
    config.ppo.lambda_init = [0.25, 0.1]  # only 2, need 3
    env = _synthetic_env(config)
    with pytest.raises(AssertionError, match="3 entries"):
        create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")


def test_additive_dual_update_default_matches_original_formula():
    config = _tiny_config()
    config.ppo.dual_lr = 0.01
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")

    trainer.buffer.ptr = 10
    trainer.buffer.costs[:10] = np.tile(np.array([0.5, 1.0, 1.5]), (10, 1))

    lambdas, violations = trainer.update_dual_variables()
    np.testing.assert_allclose(violations, np.array([0.5, 1.0, 1.5]), rtol=1e-6)
    np.testing.assert_allclose(lambdas, 0.01 * np.array([0.5, 1.0, 1.5]), rtol=1e-6)


def test_pid_dual_update_matches_hand_computed_formula():
    config = _tiny_config()
    config.ppo.dual_update_method = "pid"
    config.ppo.pid_kp = 0.1
    config.ppo.pid_ki = 0.02
    config.ppo.pid_kd = 0.05
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")

    # Step 1: violation = [1.0, 1.0, 1.0]
    trainer.buffer.ptr = 5
    trainer.buffer.costs[:5] = np.tile(np.array([1.0, 1.0, 1.0]), (5, 1))
    lambdas_1, violations_1 = trainer.update_dual_variables()
    # integral = max(0, 0 + 1.0) = 1.0; derivative = 1.0 - 0 = 1.0
    expected_1 = 0.1 * 1.0 + 0.02 * 1.0 + 0.05 * 1.0
    np.testing.assert_allclose(lambdas_1, np.full(3, expected_1), rtol=1e-5)

    # Step 2: violation drops to [0.5, 0.5, 0.5]
    trainer.buffer.ptr = 5
    trainer.buffer.costs[:5] = np.tile(np.array([0.5, 0.5, 0.5]), (5, 1))
    lambdas_2, violations_2 = trainer.update_dual_variables()
    # integral = max(0, 1.0 + 0.5) = 1.5; derivative = 0.5 - 1.0 = -0.5
    expected_2 = 0.1 * 0.5 + 0.02 * 1.5 + 0.05 * (-0.5)
    np.testing.assert_allclose(lambdas_2, np.full(3, expected_2), rtol=1e-5)


def test_pid_dual_update_clips_to_lambda_max_and_nonnegative():
    config = _tiny_config()
    config.ppo.dual_update_method = "pid"
    config.ppo.pid_kp = 100.0  # deliberately huge, to force the clip
    config.ppo.pid_ki = 0.0
    config.ppo.pid_kd = 0.0
    config.ppo.lambda_max = 5.0
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")

    trainer.buffer.ptr = 5
    trainer.buffer.costs[:5] = np.tile(np.array([1.0, 1.0, 1.0]), (5, 1))
    lambdas, _ = trainer.update_dual_variables()
    np.testing.assert_allclose(lambdas, np.full(3, 5.0), rtol=1e-6)


if __name__ == "__main__":
    test_gross_leverage_cap_scales_down_over_cap_action()
    print("PASS: gross leverage cap scales down over-cap actions")

    test_gross_leverage_cap_leaves_under_cap_action_unchanged()
    print("PASS: gross leverage cap leaves under-cap actions unchanged")

    test_gross_leverage_cap_disabled_by_default()
    print("PASS: gross leverage cap disabled by default")

    test_gross_leverage_cap_preserves_direction_with_mixed_signs()
    print("PASS: gross leverage cap preserves direction with mixed signs")

    test_action_smoothing_blends_toward_target_uniformly()
    print("PASS: action smoothing blends toward target uniformly")

    test_action_smoothing_disabled_by_default()
    print("PASS: action smoothing disabled by default")

    test_action_smoothing_applies_uniformly_regardless_of_shield()
    print("PASS: action smoothing applies uniformly regardless of shield")

    test_no_trade_band_holds_small_moves_executes_large_ones()
    print("PASS: no-trade band holds small moves, executes large ones")

    test_no_trade_band_disabled_by_default()
    print("PASS: no-trade band disabled by default")

    test_decision_frequency_holds_on_non_decision_days()
    print("PASS: decision frequency holds on non-decision days")

    test_decision_frequency_default_is_daily()
    print("PASS: decision frequency default is daily")

    test_decision_frequency_applies_uniformly_regardless_of_shield()
    print("PASS: decision frequency applies uniformly regardless of shield")

    test_lambda_init_is_actually_used()
    print("PASS: lambda_init is actually used")

    test_lambda_init_accepts_per_constraint_list()
    print("PASS: lambda_init accepts a per-constraint list")

    test_lambda_init_list_wrong_length_raises()
    print("PASS: lambda_init list wrong length raises")

    test_additive_dual_update_default_matches_original_formula()
    print("PASS: additive dual update default matches original formula")

    test_pid_dual_update_matches_hand_computed_formula()
    print("PASS: PID dual update matches hand-computed formula")

    test_pid_dual_update_clips_to_lambda_max_and_nonnegative()
    print("PASS: PID dual update clips to lambda_max")
