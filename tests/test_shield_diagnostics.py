"""
Regression tests for the shield diagnostics fix (A.6, extended in Phase C).

trading_env.py nests shield diagnostics under info['shield']
(ShieldDiagnostics.to_dict(), keys 'was_modified'/'modification_magnitude'/
'reasons'), but two independent call sites used to read flat top-level keys
that never existed, so every read silently fell back to its default:
- trainer.py's collect_rollouts() (info.get('shield_modified')/
  info.get('shield_magnitude')) -- fixed in A.6.
- evaluation.py's Evaluator._run_episode() (the same two keys, plus
  info.get('shield_reasons')) -- an independent instance of the same bug,
  found while building run_paper_experiments.py's shield_intervention_rate
  column, which was silently always 0.0 as a result.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def test_old_info_keys_would_have_read_as_zero():
    """Fast, deterministic proof of the bug/fix logic, independent of any
    stochastic RL behavior: given an info dict exactly as trading_env.py
    actually produces it, the old top-level lookup silently returns the
    default even when the shield clearly did intervene."""
    info = {'shield': {'was_modified': True, 'modification_magnitude': 0.5}}

    old_shield_modified = info.get('shield_modified', False)
    old_shield_magnitude = info.get('shield_magnitude', 0.0)
    assert old_shield_modified is False
    assert old_shield_magnitude == 0.0

    new_shield_modified = info.get('shield', {}).get('was_modified', False)
    new_shield_magnitude = info.get('shield', {}).get('modification_magnitude', 0.0)
    assert new_shield_modified is True
    assert new_shield_magnitude == 0.5


def test_no_shield_attached_defaults_safely():
    """When no shield is attached (ShieldType.NONE), info has no 'shield'
    key at all -- the fix's .get('shield', {}) must not raise."""
    info = {}
    shield_modified = info.get('shield', {}).get('was_modified', False)
    shield_magnitude = info.get('shield', {}).get('modification_magnitude', 0.0)
    assert shield_modified is False
    assert shield_magnitude == 0.0


def _build_tight_shield_setup(seed=42):
    config = Config()
    config.data = DataConfig(tickers=["ASSET_0", "ASSET_1"])
    config.features = FeatureConfig()
    config.regime = RegimeConfig(model_type=RegimeModelType.HEURISTIC, n_regimes=3)
    # Deliberately tight caps: a freshly-initialized ContinuousPolicy has a
    # long-bias mean (init_mean_bias=0.5) and std ~exp(-0.5)=0.6, so sampled
    # actions routinely land well above these caps -> the shield must
    # intervene almost every step, regardless of the exact random draw.
    config.constraints = ConstraintConfig(leverage_cap=0.1, turnover_cap=0.1, dd_stop=0.3)
    config.shield = ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE)
    config.action_space = ActionSpaceConfig(action_type=ActionSpaceType.CONTINUOUS)
    config.ppo = PPOConfig(hidden_sizes=[16, 16], batch_size=16, n_epochs=1)
    config.training = TrainingConfig(
        total_timesteps=64, rollout_length=64, episode_length=64,
        use_rolling_episodes=True, seed=seed,
    )
    config.environment = EnvironmentConfig()

    gen = SyntheticDataGenerator(n_assets=2, n_days=200, seed=9)
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


def test_shield_intervention_rate_is_nonzero_in_short_smoke_run():
    """Core A.6 acceptance check: with the shield on and constraints tight
    enough to bind, a short rollout must report a nonzero intervention rate
    -- not the always-zero default the old (wrong) info keys produced."""
    env, config = _build_tight_shield_setup(seed=42)

    trainer = PPOLagrangianTrainer(env, config, device="cpu")
    stats = trainer.collect_rollouts()

    assert stats['shield_intervention_rate'] > 0.0, (
        "shield_intervention_rate was 0.0 even with tight constraints -- "
        "trainer.py is likely reading the wrong info keys again"
    )

    ptr = trainer.buffer.ptr
    assert trainer.buffer.shield_modified[:ptr].sum() > 0
    assert trainer.buffer.shield_magnitudes[:ptr].sum() > 0.0


def test_evaluator_shield_intervention_rate_is_nonzero_in_short_smoke_run():
    """The same bug class independently present in evaluation.py:
    Evaluator._run_episode() also read top-level info['shield_modified'] /
    info['shield_magnitude'] / info['shield_reasons'] instead of the nested
    info['shield'] dict, so every evaluate_policy() result reported
    shield_intervention_rate == 0.0 regardless of actual shield activity
    (discovered while building run_paper_experiments.py's
    shield_intervention_rate column)."""
    from networks import create_actor_critic
    from evaluation import Evaluator

    env, config = _build_tight_shield_setup(seed=43)
    policy = create_actor_critic(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        n_cost_constraints=3,
        config=config.ppo,
        action_type=config.action_space.action_type,
    )

    evaluator = Evaluator(env, config)
    _, aggregate, _ = evaluator.evaluate_policy(
        policy, n_episodes=1, deterministic=True, store_trajectories=False
    )

    assert aggregate.metrics['shield_intervention_rate'][0] > 0.0, (
        "shield_intervention_rate was 0.0 even with tight constraints -- "
        "evaluation.py is likely reading the wrong info keys again"
    )


if __name__ == "__main__":
    test_old_info_keys_would_have_read_as_zero()
    print("PASS: old top-level info keys would have silently read as zero")

    test_no_shield_attached_defaults_safely()
    print("PASS: no-shield case defaults safely")

    test_shield_intervention_rate_is_nonzero_in_short_smoke_run()
    print("PASS: trainer.py shield intervention rate is nonzero in a short smoke run")

    test_evaluator_shield_intervention_rate_is_nonzero_in_short_smoke_run()
    print("PASS: evaluation.py shield intervention rate is nonzero in a short smoke run")
