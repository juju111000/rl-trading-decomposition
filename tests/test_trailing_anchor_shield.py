"""
Regression tests for ShieldConfig.dd_anchor="trailing" (strategy-v2-overlay,
item 1).

Motivation: with dd_anchor="peak" (the original/default behavior),
UncertaintyAwareShield's continuous drawdown scaling (Rule 4) is anchored to
the all-time NAV peak, so a single deep drawdown depresses the shield's
position cap indefinitely -- even long after NAV has recovered and
stabilized well above the crash, because the OLD peak is still the
reference point (the "absorbing trap"; see results/full_grid's
uncertainty_aware arm, whose max_dd clusters right at dd_stop across nearly
every fold/seed). dd_anchor="trailing" anchors drawdown to the max NAV over
only the last dd_anchor_window trading days instead, so once a crash is
further in the past than the window, the anchor forgets it and the position
cap releases.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from config import ShieldConfig, ConstraintConfig, ShieldType
from policy_shield import RuleBasedShield, UncertaintyAwareShield
from portfolio_engine import PortfolioState


def _ps(nav: float, peak_nav: float, weights=None, n_assets: int = 1) -> PortfolioState:
    weights = np.zeros(n_assets) if weights is None else weights
    return PortfolioState(
        nav=nav, cash=0.0, weights=weights, positions={},
        peak_nav=peak_nav, current_drawdown=max(0.0, 1 - nav / peak_nav),
        gross_leverage=0.0, net_exposure=0.0,
    )


def test_anchor_rolls_forward_correctly():
    """Trailing anchor must equal max(nav_history[-window:]) at every point
    in time, not the global max and not stuck at a stale value."""
    shield = RuleBasedShield(
        ShieldConfig(shield_type=ShieldType.RULE_BASED, dd_anchor="trailing", dd_anchor_window=5),
        ConstraintConfig(), n_assets=1,
    )
    nav_history = np.array([100.0, 110.0, 90.0, 95.0, 130.0, 120.0, 115.0, 105.0, 100.0, 98.0])

    # "Today" = index 4 (nav=130, a fresh high) -- window [0..4], anchor=130.
    ps_at_4 = _ps(nav=130.0, peak_nav=130.0)
    dd_at_4 = shield._effective_drawdown(ps_at_4, nav_history[:5])
    assert dd_at_4 == pytest.approx(0.0)

    # "Today" = index 9 (nav=98) -- window is indices [5..9] = [120,115,105,100,98],
    # anchor=120 (the old peak=130 at index 4 has rolled OUT of the window).
    ps_at_9 = _ps(nav=98.0, peak_nav=130.0)
    dd_at_9 = shield._effective_drawdown(ps_at_9, nav_history[:10])
    expected = 1 - 98.0 / 120.0
    assert dd_at_9 == pytest.approx(expected)
    # Confirms the anchor forgot the global max (130): if it hadn't,
    # dd would be 1 - 98/130, a different (larger) number.
    assert dd_at_9 != pytest.approx(1 - 98.0 / 130.0)


def test_peak_mode_ignores_nav_history_unchanged_behavior():
    """dd_anchor="peak" (default) must reproduce the original behavior
    exactly -- current_drawdown as-is, regardless of nav_history."""
    shield = RuleBasedShield(
        ShieldConfig(shield_type=ShieldType.RULE_BASED, dd_anchor="peak"),
        ConstraintConfig(), n_assets=1,
    )
    ps = _ps(nav=70.0, peak_nav=100.0)
    nav_history = np.array([100.0, 90.0, 80.0, 75.0, 70.0])

    assert shield._effective_drawdown(ps, nav_history) == pytest.approx(ps.current_drawdown)
    assert shield._effective_drawdown(ps, None) == pytest.approx(ps.current_drawdown)


def test_crash_then_recovery_releases_trailing_dd_but_not_peak_dd():
    """The absorbing-trap regression test: after a crash far enough in the
    past, plus a long flat/recovered plateau, dd_anchor="trailing" must
    release back toward zero while dd_anchor="peak" stays elevated (still
    referencing the pre-crash all-time high)."""
    window = 63

    # 20 days at the all-time peak, a 1-day 35% crash, then 70 flat days at
    # the recovered level -- long enough (70 > window) that the trailing
    # window no longer contains either the crash day or the old peak.
    pre_crash = np.full(20, 1_000_000.0)
    crash = np.array([650_000.0])
    recovered = np.full(70, 720_000.0)
    nav_history = np.concatenate([pre_crash, crash, recovered])
    assert len(nav_history) == 91

    current_nav = float(nav_history[-1])  # 720,000
    all_time_peak = float(nav_history.max())  # 1,000,000
    ps = _ps(nav=current_nav, peak_nav=all_time_peak)

    constraint_config = ConstraintConfig(dd_limit=0.20, dd_stop=0.30, leverage_cap=2.0, turnover_cap=2.0)

    peak_shield = UncertaintyAwareShield(
        ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE, dd_anchor="peak",
                     use_entropy_tightening=False, use_bear_cap=False),
        constraint_config, n_assets=1,
    )
    trailing_shield = UncertaintyAwareShield(
        ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE, dd_anchor="trailing", dd_anchor_window=window,
                     use_entropy_tightening=False, use_bear_cap=False),
        constraint_config, n_assets=1,
    )

    # Peak-anchored dd = 1 - 720k/1,000k = 0.28 -- still inside the
    # dd_limit..dd_stop scaling band 27 days after the crash trough.
    peak_dd = peak_shield._effective_drawdown(ps, nav_history)
    assert peak_dd == pytest.approx(0.28)
    assert constraint_config.dd_limit < peak_dd < constraint_config.dd_stop

    # Trailing-anchored dd = 1 - 720k/720k = 0.0 -- the crash and the old
    # peak have both rolled out of the 63-day window; released.
    trailing_dd = trailing_shield._effective_drawdown(ps, nav_history)
    assert trailing_dd == pytest.approx(0.0)

    proposed_action = np.array([0.5])
    regime_belief = np.array([0.5, 0.2, 0.3])

    peak_safe, peak_diag = peak_shield.apply(proposed_action, ps, regime_belief, entropy=0.5, nav_history=nav_history)
    trailing_safe, trailing_diag = trailing_shield.apply(
        proposed_action, ps, regime_belief, entropy=0.5, nav_history=nav_history
    )

    # Peak mode: still meaningfully suppressed (dd_scale = 1-(0.28-0.20)/(0.30-0.20) = 0.2).
    assert peak_safe[0] == pytest.approx(0.5 * 0.2, abs=1e-6)
    assert any("DD_SCALING" in r for r in peak_diag.reasons)

    # Trailing mode: the limit has reopened -- action passes through unscaled.
    assert trailing_safe[0] == pytest.approx(0.5, abs=1e-6)
    assert not any("DD_SCALING" in r or "DD_HARD_ZERO" in r for r in trailing_diag.reasons)


def test_composite_shield_forwards_nav_history_to_sub_shields():
    """CompositeShield.apply must thread nav_history through to every
    sub-shield, not just the top-level call."""
    from policy_shield import CompositeShield

    constraint_config = ConstraintConfig(dd_limit=0.20, dd_stop=0.30, leverage_cap=2.0, turnover_cap=2.0)
    shield_config = ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE, dd_anchor="trailing", dd_anchor_window=5,
                                  use_entropy_tightening=False, use_bear_cap=False)
    inner = UncertaintyAwareShield(shield_config, constraint_config, n_assets=1)
    composite = CompositeShield([inner], shield_config, constraint_config, n_assets=1)

    nav_history = np.concatenate([np.full(10, 100.0), np.array([70.0, 71.0, 72.0, 73.0, 74.0])])
    ps = _ps(nav=74.0, peak_nav=100.0)

    safe, diag = composite.apply(np.array([0.5]), ps, np.array([0.5, 0.2, 0.3]), entropy=0.5, nav_history=nav_history)

    # Trailing window [70..74] -> anchor=74=current nav -> dd=0 -> unscaled,
    # even though peak-anchored dd (1-74/100=0.26) would have suppressed it.
    assert safe[0] == pytest.approx(0.5, abs=1e-6)


if __name__ == "__main__":
    test_anchor_rolls_forward_correctly()
    print("PASS: anchor rolls forward correctly")

    test_peak_mode_ignores_nav_history_unchanged_behavior()
    print("PASS: peak mode unchanged / ignores nav_history")

    test_crash_then_recovery_releases_trailing_dd_but_not_peak_dd()
    print("PASS: crash+recovery releases trailing dd, not peak dd (absorbing-trap regression)")

    test_composite_shield_forwards_nav_history_to_sub_shields()
    print("PASS: CompositeShield forwards nav_history to sub-shields")
