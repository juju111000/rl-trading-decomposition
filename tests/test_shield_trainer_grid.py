"""
Regression tests for the shield x trainer grid (A.5).

Before this fix, every method in an experiment run shared ONE train_env /
test_env built from a single shared config, so all methods -- including
"vanilla_ppo", supposed to be an unshielded baseline -- silently ran under
whatever shield that one shared env happened to have attached (confirmed
empirically during the original audit: a saved
results/*/vanilla_ppo/config.json recorded shield_type="uncertainty_aware",
identical to the full method's own config).

The fix builds a fresh train_env/test_env per (shield, trainer) arm from a
per-arm config. config.py/policy_shield.py add ShieldType.NONE so "no
shield" is itself an explicit config value, not an implicit/accidental
absence. These tests confirm: ShieldType.NONE really means "no shield" at
the environment level, the grid is 2 shields x 3 trainers, and each arm's
SAVED config.json on disk reflects its own shield_type.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    Config, DataConfig, FeatureConfig, RegimeConfig, ConstraintConfig,
    ShieldConfig, ActionSpaceConfig, PPOConfig, TrainingConfig,
    EnvironmentConfig, ActionSpaceType, RegimeModelType, ShieldType,
)
from policy_shield import create_shield, UncertaintyAwareShield
from data_loader import SyntheticDataGenerator
from features import FeatureBuilder


def test_shield_type_none_returns_no_shield():
    shield_config = ShieldConfig(shield_type=ShieldType.NONE)
    constraint_config = ConstraintConfig()

    shield = create_shield(shield_config, constraint_config, n_assets=3)

    assert shield is None


def test_grid_is_two_shields_by_three_trainers():
    import main

    assert main.SHIELD_ARMS == ["none", "uncertainty_aware"]
    assert main.TRAINER_ARMS == ["vanilla_ppo", "reward_shaping", "ppo_lagrangian"]
    assert len(main.SHIELD_ARMS) * len(main.TRAINER_ARMS) == 6


def _tiny_config(shield_type):
    config = Config()
    config.data = DataConfig(tickers=["ASSET_0", "ASSET_1"])
    config.features = FeatureConfig()
    config.regime = RegimeConfig(model_type=RegimeModelType.HEURISTIC, n_regimes=3)
    config.constraints = ConstraintConfig()
    config.shield = ShieldConfig(shield_type=shield_type)
    config.action_space = ActionSpaceConfig(action_type=ActionSpaceType.CONTINUOUS)
    config.ppo = PPOConfig(hidden_sizes=[16, 16], batch_size=16, n_epochs=1)
    config.training = TrainingConfig(
        total_timesteps=64, rollout_length=64, episode_length=64,
        use_rolling_episodes=True, seed=1,
    )
    config.environment = EnvironmentConfig()
    return config


def _synthetic_features_and_prices(seed):
    gen = SyntheticDataGenerator(n_assets=2, n_days=200, seed=seed)
    data, _ = gen.generate_regime_switching_data()

    builder = FeatureBuilder(FeatureConfig())
    features_df = builder.compute_features(data)
    prices = {t: data[(t, "Close")].values for t in ["ASSET_0", "ASSET_1"]}
    return data, features_df, prices


def test_create_trading_env_has_no_shield_for_none_arm():
    import main

    data, features_df, prices = _synthetic_features_and_prices(seed=3)

    none_config = _tiny_config(ShieldType.NONE)
    shielded_config = _tiny_config(ShieldType.UNCERTAINTY_AWARE)

    none_env = main.create_trading_env(
        features_df, prices, none_config, aligned_data=data, mode="train"
    )
    shielded_env = main.create_trading_env(
        features_df, prices, shielded_config, aligned_data=data, mode="train"
    )

    assert none_env.shield is None
    assert isinstance(shielded_env.shield, UncertaintyAwareShield)


def test_each_arm_saved_config_reflects_its_own_shield_type(tmp_path):
    """Core A.5 acceptance criterion: run a reduced grid (both shield arms,
    one trainer arm) and confirm each arm's saved config.json on disk
    reflects ITS OWN shield_type, not another arm's or a shared default."""
    import main

    data, features_df, prices = _synthetic_features_and_prices(seed=4)

    train_features = features_df.iloc[:150]
    test_features = features_df.iloc[150:]
    train_prices = {t: p[:150] for t, p in prices.items()}
    test_prices = {t: p[150:] for t, p in prices.items()}
    train_aligned = data.iloc[:150]
    test_aligned = data.iloc[150:]

    # base_config's own shield_type deliberately doesn't matter -- each arm
    # overrides it via _make_arm_config().
    base_config = _tiny_config(ShieldType.UNCERTAINTY_AWARE)

    grid_results = main.run_method_grid(
        train_features, test_features, train_prices, test_prices,
        train_aligned, test_aligned,
        base_config=base_config,
        log_dir=tmp_path,
        shield_arms=["none", "uncertainty_aware"],
        trainer_arms=["ppo_lagrangian"],
        n_eval_episodes=1,
    )

    assert set(grid_results.keys()) == {
        "none/ppo_lagrangian", "uncertainty_aware/ppo_lagrangian"
    }

    for shield_arm in ["none", "uncertainty_aware"]:
        arm_name = f"{shield_arm}/ppo_lagrangian"
        config_path = tmp_path / arm_name / "config.json"
        assert config_path.exists(), f"missing config.json for {arm_name}"

        saved_config = Config.from_json(str(config_path))
        assert saved_config.shield.shield_type.value == shield_arm, (
            f"{arm_name}'s saved config.json has shield_type="
            f"{saved_config.shield.shield_type.value!r}, expected {shield_arm!r}"
        )


if __name__ == "__main__":
    test_shield_type_none_returns_no_shield()
    print("PASS: ShieldType.NONE -> create_shield returns None")

    test_grid_is_two_shields_by_three_trainers()
    print("PASS: grid is 2 shields x 3 trainers = 6 arms")
