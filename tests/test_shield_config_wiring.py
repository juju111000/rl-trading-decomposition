"""
Regression tests for wiring the shield's position cap to config (B.8).

UncertaintyAwareShield._adjust_limits() computed 'position_cap' as
max(0.1, 0.5 * (1 - tighten)) -- a hardcoded 0.5 base value, not
shield_config.high_vol_position_cap (which happens to default to 0.5 too,
masking the bug: changing high_vol_position_cap in any experiment config had
zero effect on this shield's actual position cap).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from config import ShieldConfig, ConstraintConfig, ShieldType
from policy_shield import UncertaintyAwareShield
from portfolio_engine import PortfolioState


def _make_shield(high_vol_position_cap, n_assets=3):
    shield_config = ShieldConfig(
        shield_type=ShieldType.UNCERTAINTY_AWARE,
        high_vol_position_cap=high_vol_position_cap,
    )
    constraint_config = ConstraintConfig()
    return UncertaintyAwareShield(shield_config, constraint_config, n_assets=n_assets)


def test_position_cap_at_zero_entropy_equals_configured_value():
    """At entropy=0 (no tightening), position_cap should equal exactly the
    configured high_vol_position_cap, not a hardcoded 0.5."""
    shield_default = _make_shield(0.5)
    shield_custom = _make_shield(0.3)

    assert shield_default._adjust_limits(0.0)['position_cap'] == pytest.approx(0.5)
    assert shield_custom._adjust_limits(0.0)['position_cap'] == pytest.approx(0.3)


def test_changing_high_vol_position_cap_changes_the_shield_output():
    """Core acceptance check: two shields differing only in
    high_vol_position_cap must produce different adjusted position caps."""
    shield_default = _make_shield(0.5)
    shield_custom = _make_shield(0.3)

    entropy = 0.5
    limits_default = shield_default._adjust_limits(entropy)
    limits_custom = shield_custom._adjust_limits(entropy)

    assert limits_default['position_cap'] != limits_custom['position_cap'], (
        "high_vol_position_cap had no effect on the shield's position_cap -- "
        "it's likely still hardcoded"
    )


def test_apply_clips_actions_to_the_configured_position_cap():
    """End-to-end: the configured cap must actually constrain the executed
    action, not just an internal dict value nothing reads."""
    shield = _make_shield(high_vol_position_cap=0.2)

    portfolio_state = PortfolioState(
        nav=1_000_000,
        cash=1_000_000,
        weights=np.zeros(3),
        positions={},
        peak_nav=1_000_000,
        current_drawdown=0.0,
        gross_leverage=0.0,
        net_exposure=0.0,
    )
    proposed_action = np.array([0.9, 0.9, 0.9])
    regime_belief = np.array([0.6, 0.2, 0.2])

    safe_action, diag = shield.apply(
        proposed_action, portfolio_state, regime_belief, entropy=0.0
    )

    assert np.max(np.abs(safe_action)) <= 0.2 + 1e-9, (
        f"safe_action {safe_action} exceeds the configured position cap 0.2"
    )
    assert diag.was_modified


if __name__ == "__main__":
    test_position_cap_at_zero_entropy_equals_configured_value()
    print("PASS: position_cap at zero entropy equals the configured value")

    test_changing_high_vol_position_cap_changes_the_shield_output()
    print("PASS: changing high_vol_position_cap changes the shield's output")

    test_apply_clips_actions_to_the_configured_position_cap()
    print("PASS: apply() clips actions to the configured position cap")
