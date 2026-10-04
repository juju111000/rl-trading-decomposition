"""
Regression tests for Phase 3's entropy/log_std annealing fixes.

The fold-2 Phase 2 experiments found mean_action_std GROWING (0.62 -> ~1.35)
across every dual-mechanism variant tried, because entropy_decay/min_entropy_coef
were defined in PPOConfig but never read anywhere in trainer.py -- entropy_coef
stayed fixed at its initial value for the whole run, a constant, undiminishing
incentive toward higher action variance. This file pins down the fix: the
schedule itself (pure function of iteration), that update_policy() actually
uses it, the optional log_std_init/log_std_decay_floor levers, and the
mean_return -> rollout_return_sum rename.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import main as main_module
from config import (
    Config, DataConfig, FeatureConfig, RegimeConfig, ConstraintConfig,
    ShieldConfig, ActionSpaceConfig, PPOConfig, TrainingConfig,
    EnvironmentConfig, ActionSpaceType, RegimeModelType, ShieldType,
)
from data_loader import SyntheticDataGenerator
from features import FeatureBuilder
from trainer import create_trainer, TrainingMetrics


def _tiny_config(**ppo_overrides):
    config = Config()
    config.data = DataConfig(tickers=["ASSET_0", "ASSET_1"])
    config.features = FeatureConfig()
    config.regime = RegimeConfig(model_type=RegimeModelType.HEURISTIC, n_regimes=3)
    config.constraints = ConstraintConfig()
    config.shield = ShieldConfig(shield_type=ShieldType.NONE)
    config.action_space = ActionSpaceConfig(action_type=ActionSpaceType.CONTINUOUS, max_position_size=1.0)
    config.ppo = PPOConfig(hidden_sizes=[16, 16], batch_size=16, n_epochs=1, **ppo_overrides)
    config.training = TrainingConfig(
        total_timesteps=64, rollout_length=64, episode_length=64,
        use_rolling_episodes=False, seed=1,
    )
    config.environment = EnvironmentConfig()
    return config


def _synthetic_env(config, seed=3):
    gen = SyntheticDataGenerator(n_assets=2, n_days=200, seed=seed)
    data, _ = gen.generate_regime_switching_data()
    builder = FeatureBuilder(FeatureConfig())
    features_df = builder.compute_features(data)
    prices = {t: data[(t, "Close")].values for t in ["ASSET_0", "ASSET_1"]}
    return main_module.create_trading_env(features_df, prices, config, aligned_data=data, mode="train")


# =============================================================================
# Entropy coefficient schedule
# =============================================================================

def test_entropy_coef_schedule_at_iter_zero_matches_initial_value():
    config = _tiny_config(entropy_coef=0.05, entropy_decay=0.998, min_entropy_coef=0.005)
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")
    assert trainer._entropy_coef_for_iteration(0) == 0.05


def test_entropy_coef_schedule_decays_at_mid_iteration():
    config = _tiny_config(entropy_coef=0.05, entropy_decay=0.998, min_entropy_coef=0.005)
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")
    expected = 0.05 * (0.998 ** 50)
    assert abs(trainer._entropy_coef_for_iteration(50) - expected) < 1e-9
    assert trainer._entropy_coef_for_iteration(50) < 0.05


def test_entropy_coef_schedule_clips_to_min_at_late_iteration():
    config = _tiny_config(entropy_coef=0.05, entropy_decay=0.998, min_entropy_coef=0.005)
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")
    # 0.05 * 0.998^5000 is far below 0.005 -- must clip, not go to ~0
    assert trainer._entropy_coef_for_iteration(5000) == 0.005


def test_entropy_coef_schedule_is_monotonically_nonincreasing():
    config = _tiny_config(entropy_coef=0.05, entropy_decay=0.998, min_entropy_coef=0.005)
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")
    values = [trainer._entropy_coef_for_iteration(i) for i in range(0, 200, 10)]
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def test_update_policy_actually_uses_the_annealed_entropy_coef():
    config = _tiny_config(entropy_coef=0.05, entropy_decay=0.5, min_entropy_coef=0.005)
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")

    trainer.iteration = 3
    trainer.collect_rollouts()
    trainer.update_policy()

    expected = max(0.005, 0.05 * (0.5 ** 3))
    assert trainer.current_entropy_coef == expected
    assert trainer.current_entropy_coef != config.ppo.entropy_coef  # actually changed, not the dead-field bug


# =============================================================================
# log_std_init / log_std_decay_floor
# =============================================================================

def test_log_std_init_reaches_the_network():
    config = _tiny_config(log_std_init=-1.75)
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")
    np.testing.assert_allclose(
        trainer.actor_critic.policy.log_std.detach().numpy(),
        np.full(2, -1.75), rtol=1e-6
    )


def test_log_std_decay_floor_disabled_by_default_returns_none():
    config = _tiny_config()  # log_std_decay_floor left at its None default
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")
    assert trainer._log_std_ceiling_for_iteration(0) is None
    assert trainer._log_std_ceiling_for_iteration(500) is None


def test_log_std_decay_floor_schedule_matches_entropy_progress():
    config = _tiny_config(
        entropy_coef=0.05, entropy_decay=0.998, min_entropy_coef=0.005,
        log_std_init=-0.5, log_std_decay_floor=-2.0,
    )
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")

    # iter 0: entropy progress = 1.0 -> floor_progress = 0 -> ceiling = log_std_init
    assert abs(trainer._log_std_ceiling_for_iteration(0) - (-0.5)) < 1e-9

    # very late: entropy bottomed out at min_entropy_coef -> progress = 0.005/0.05 = 0.1
    # -> floor_progress = 0.9 -> ceiling = -0.5 + (-2.0 - (-0.5)) * 0.9 = -1.85
    late_ceiling = trainer._log_std_ceiling_for_iteration(5000)
    assert abs(late_ceiling - (-1.85)) < 1e-6


def test_log_std_ceiling_actually_clamps_the_parameter_after_update():
    config = _tiny_config(
        entropy_coef=0.05, entropy_decay=0.998, min_entropy_coef=0.005,
        log_std_init=-0.5, log_std_decay_floor=-5.0,
    )
    env = _synthetic_env(config)
    trainer = create_trainer(env=env, config=config, trainer_type="ppo_lagrangian", device="cpu")

    # Force log_std above what the iter-0 ceiling (-0.5) allows.
    with torch.no_grad():
        trainer.actor_critic.policy.log_std.fill_(2.0)

    trainer.iteration = 0
    trainer.collect_rollouts()
    trainer.update_policy()

    assert torch.all(trainer.actor_critic.policy.log_std <= -0.5 + 1e-4)


# =============================================================================
# mean_return -> rollout_return_sum rename
# =============================================================================

def test_training_metrics_uses_rollout_return_sum_not_mean_return():
    metrics = TrainingMetrics(
        iteration=0, total_timesteps=100, policy_loss=0.0, value_loss=0.0,
        cost_value_losses=[0.0, 0.0, 0.0], entropy=0.0, approx_kl=0.0,
        clip_fraction=0.0, lambdas=[0.0, 0.0, 0.0], mean_costs=[0.0, 0.0, 0.0],
        constraint_violations=[0.0, 0.0, 0.0], mean_reward=-1.0,
        rollout_return_sum=-500.0, shield_intervention_rate=0.0,
        mean_shield_magnitude=0.0, action_std_per_asset=[0.1, 0.1],
        mean_action_std=0.1, entropy_coef=0.05, rollout_time=0.0, update_time=0.0,
    )
    d = metrics.to_dict()
    assert 'rollout_return_sum' in d
    assert d['rollout_return_sum'] == -500.0
    assert 'mean_return' not in d


if __name__ == "__main__":
    test_entropy_coef_schedule_at_iter_zero_matches_initial_value()
    print("PASS: entropy coef schedule at iter 0")

    test_entropy_coef_schedule_decays_at_mid_iteration()
    print("PASS: entropy coef schedule decays at mid iteration")

    test_entropy_coef_schedule_clips_to_min_at_late_iteration()
    print("PASS: entropy coef schedule clips to min at late iteration")

    test_entropy_coef_schedule_is_monotonically_nonincreasing()
    print("PASS: entropy coef schedule is monotonically nonincreasing")

    test_update_policy_actually_uses_the_annealed_entropy_coef()
    print("PASS: update_policy actually uses the annealed entropy coef")

    test_log_std_init_reaches_the_network()
    print("PASS: log_std_init reaches the network")

    test_log_std_decay_floor_disabled_by_default_returns_none()
    print("PASS: log_std decay floor disabled by default")

    test_log_std_decay_floor_schedule_matches_entropy_progress()
    print("PASS: log_std decay floor schedule matches entropy progress")

    test_log_std_ceiling_actually_clamps_the_parameter_after_update()
    print("PASS: log_std ceiling actually clamps the parameter")

    test_training_metrics_uses_rollout_return_sum_not_mean_return()
    print("PASS: TrainingMetrics uses rollout_return_sum not mean_return")
