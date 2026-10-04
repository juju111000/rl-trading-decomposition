"""
Regression tests for the shield-component ablation wiring.

UncertaintyAwareShield gained three independent toggles
(use_entropy_tightening / use_bear_cap / use_dd_scaling) so each mechanism
can be ablated on its own. main.SHIELD_COMPONENT_VARIANTS names the four
planned variants (full / no_entropy_term / no_bear_cap / no_dd_scaling);
main.run_shield_component_grid() trains/evaluates ppo_lagrangian once per
variant, each with its own env built from its own config, and saves each
variant's config.json under its own subdirectory -- mirroring the
shield x trainer grid's per-arm isolation (A.5).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import (
    Config, DataConfig, FeatureConfig, RegimeConfig, ConstraintConfig,
    ShieldConfig, ActionSpaceConfig, PPOConfig, TrainingConfig,
    EnvironmentConfig, ActionSpaceType, RegimeModelType, ShieldType,
)
from policy_shield import create_shield
from portfolio_engine import PortfolioState
import main as main_module


def test_four_variants_are_defined():
    assert set(main_module.SHIELD_COMPONENT_VARIANTS.keys()) == {
        "full", "no_entropy_term", "no_bear_cap", "no_dd_scaling"
    }


def test_each_variant_toggles_exactly_one_component_off():
    full = main_module.SHIELD_COMPONENT_VARIANTS["full"]
    assert all(full.values()), "full variant should have every component on"

    for variant_name in ["no_entropy_term", "no_bear_cap", "no_dd_scaling"]:
        variant = main_module.SHIELD_COMPONENT_VARIANTS[variant_name]
        off_keys = [k for k, v in variant.items() if not v]
        assert len(off_keys) == 1, f"{variant_name} should disable exactly one component"


def _tiny_base_config():
    config = Config()
    config.data = DataConfig(tickers=["ASSET_0"])
    config.features = FeatureConfig()
    config.regime = RegimeConfig(model_type=RegimeModelType.HEURISTIC, n_regimes=3)
    config.constraints = ConstraintConfig()
    config.shield = ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE)
    config.action_space = ActionSpaceConfig(action_type=ActionSpaceType.CONTINUOUS)
    config.ppo = PPOConfig(hidden_sizes=[16, 16], batch_size=16, n_epochs=1)
    config.training = TrainingConfig(
        total_timesteps=64, rollout_length=64, episode_length=64, seed=1,
    )
    config.environment = EnvironmentConfig()
    return config


def test_make_shield_component_config_applies_toggles_and_shield_type():
    base_config = _tiny_base_config()
    base_config.shield.shield_type = ShieldType.NONE  # deliberately wrong, must be overridden

    for variant_name, expected_toggles in main_module.SHIELD_COMPONENT_VARIANTS.items():
        arm_config = main_module._make_shield_component_config(base_config, variant_name)
        assert arm_config.shield.shield_type == ShieldType.UNCERTAINTY_AWARE
        for key, expected_value in expected_toggles.items():
            assert getattr(arm_config.shield, key) == expected_value, (
                f"{variant_name}: {key} should be {expected_value}"
            )


def test_no_entropy_term_variant_ignores_entropy_in_shield_output():
    """End-to-end (shield-level, not full training): the no_entropy_term
    variant's position cap must NOT shrink with entropy."""
    base_config = _tiny_base_config()
    variant_config = main_module._make_shield_component_config(base_config, "no_entropy_term")

    shield = create_shield(variant_config.shield, variant_config.constraints, n_assets=1)
    ps = PortfolioState(nav=1.0, cash=0.0, weights=np.zeros(1), positions={}, peak_nav=1.0,
                         current_drawdown=0.0, gross_leverage=0.0, net_exposure=0.0)

    safe_low_entropy, _ = shield.apply(np.array([10.0]), ps, np.array([0.4, 0.3, 0.3]), entropy=0.0)
    safe_high_entropy, _ = shield.apply(np.array([10.0]), ps, np.array([0.4, 0.3, 0.3]), entropy=1.0)

    np.testing.assert_allclose(safe_low_entropy, safe_high_entropy)


def test_no_bear_cap_variant_does_not_apply_conservative_cap():
    base_config = _tiny_base_config()
    variant_config = main_module._make_shield_component_config(base_config, "no_bear_cap")

    shield = create_shield(variant_config.shield, variant_config.constraints, n_assets=1)
    ps = PortfolioState(nav=1.0, cash=0.0, weights=np.zeros(1), positions={}, peak_nav=1.0,
                         current_drawdown=0.0, gross_leverage=0.0, net_exposure=0.0)

    safe_action, diag = shield.apply(
        np.array([10.0]), ps, regime_belief=np.array([0.1, 0.85, 0.05]), entropy=0.05
    )
    assert not any("BEAR_REGIME" in r for r in diag.reasons)


def test_no_dd_scaling_variant_ignores_drawdown():
    base_config = _tiny_base_config()
    variant_config = main_module._make_shield_component_config(base_config, "no_dd_scaling")

    shield = create_shield(variant_config.shield, variant_config.constraints, n_assets=1)
    ps_calm = PortfolioState(nav=1.0, cash=0.0, weights=np.zeros(1), positions={}, peak_nav=1.0,
                              current_drawdown=0.0, gross_leverage=0.0, net_exposure=0.0)
    ps_deep_dd = PortfolioState(nav=1.0, cash=0.0, weights=np.zeros(1), positions={}, peak_nav=1.0,
                                 current_drawdown=0.29, gross_leverage=0.0, net_exposure=0.0)

    safe_calm, _ = shield.apply(np.array([10.0]), ps_calm, np.array([0.4, 0.3, 0.3]), entropy=0.5)
    safe_deep_dd, diag = shield.apply(np.array([10.0]), ps_deep_dd, np.array([0.4, 0.3, 0.3]), entropy=0.5)

    np.testing.assert_allclose(safe_calm, safe_deep_dd)
    assert not any("DD_SCALING" in r or "DD_HARD_ZERO" in r for r in diag.reasons)


if __name__ == "__main__":
    test_four_variants_are_defined()
    print("PASS: four variants defined")

    test_each_variant_toggles_exactly_one_component_off()
    print("PASS: each variant toggles exactly one component off")

    test_make_shield_component_config_applies_toggles_and_shield_type()
    print("PASS: _make_shield_component_config applies toggles and shield_type")

    test_no_entropy_term_variant_ignores_entropy_in_shield_output()
    print("PASS: no_entropy_term ignores entropy")

    test_no_bear_cap_variant_does_not_apply_conservative_cap()
    print("PASS: no_bear_cap never applies the conservative cap")

    test_no_dd_scaling_variant_ignores_drawdown()
    print("PASS: no_dd_scaling ignores drawdown")
