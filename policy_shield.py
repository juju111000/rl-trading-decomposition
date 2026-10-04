"""
Policy Shield module.
Enforces hard instantaneous constraints on actions before execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.optimize import minimize

from config import ShieldConfig, ShieldType, ConstraintConfig
from portfolio_engine import PortfolioState


@dataclass
class ShieldDiagnostics:
    """Diagnostics from shield intervention."""
    was_modified: bool
    modification_magnitude: float
    original_action: np.ndarray
    safe_action: np.ndarray
    reasons: List[str]
    constraint_violations: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            'was_modified': self.was_modified,
            'modification_magnitude': self.modification_magnitude,
            'reasons': self.reasons,
            'constraint_violations': self.constraint_violations
        }


class PolicyShield(ABC):
    """
    Abstract base class for policy shields.
    Shields modify unsafe actions to ensure hard constraint satisfaction.
    """

    def __init__(
            self,
            shield_config: ShieldConfig,
            constraint_config: ConstraintConfig,
            n_assets: int
    ):
        self.shield_config = shield_config
        self.constraint_config = constraint_config
        self.n_assets = n_assets

        # Track statistics
        self.intervention_count = 0
        self.total_calls = 0
        self.modification_magnitudes: List[float] = []

    def _effective_drawdown(
            self,
            portfolio_state: PortfolioState,
            nav_history: Optional[np.ndarray]
    ) -> float:
        """Drawdown the shield's own rules should react to, per
        shield_config.dd_anchor. "peak" (default) is exactly
        portfolio_state.current_drawdown (anchored to the all-time NAV peak,
        existing behavior). "trailing" recomputes drawdown against the max
        NAV over the last dd_anchor_window entries of nav_history instead --
        see ShieldConfig.dd_anchor's docstring for why (the absorbing-trap
        fix). Falls back to "peak" if nav_history isn't supplied (e.g. older
        callers that don't pass it), so this is backward compatible."""
        if self.shield_config.dd_anchor != "trailing" or nav_history is None or len(nav_history) == 0:
            return portfolio_state.current_drawdown

        window = max(1, self.shield_config.dd_anchor_window)
        anchor = float(np.max(nav_history[-window:]))
        if anchor <= 0:
            return 0.0
        return max(0.0, 1.0 - portfolio_state.nav / anchor)

    @abstractmethod
    def apply(
            self,
            proposed_action: np.ndarray,
            portfolio_state: PortfolioState,
            regime_belief: np.ndarray,
            entropy: float,
            nav_history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        """
        Apply shield to proposed action.

        Args:
            proposed_action: Action proposed by policy.
            portfolio_state: Current portfolio state.
            regime_belief: Current regime belief vector.
            entropy: Current belief entropy.
            nav_history: NAV history up to (and including) the current step,
                used only when shield_config.dd_anchor == "trailing" (see
                _effective_drawdown). Optional/defaulted for backward
                compatibility with existing callers.

        Returns:
            Tuple of (safe_action, diagnostics)
        """
        pass

    def get_statistics(self) -> Dict[str, float]:
        """Return shield intervention statistics."""
        if self.total_calls == 0:
            return {
                'intervention_rate': 0.0,
                'avg_modification': 0.0,
                'total_interventions': 0
            }

        return {
            'intervention_rate': self.intervention_count / self.total_calls,
            'avg_modification': np.mean(self.modification_magnitudes) if self.modification_magnitudes else 0.0,
            'total_interventions': self.intervention_count
        }

    def reset_statistics(self):
        """Reset intervention statistics."""
        self.intervention_count = 0
        self.total_calls = 0
        self.modification_magnitudes = []


class RuleBasedShield(PolicyShield):
    """
    Rule-based shield with simple gating logic.
    Fast and interpretable, good baseline.
    """

    def apply(
            self,
            proposed_action: np.ndarray,
            portfolio_state: PortfolioState,
            regime_belief: np.ndarray,
            entropy: float,
            nav_history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        self.total_calls += 1

        safe_action = proposed_action.copy()
        reasons = []
        violations = {}

        # Rule 1: Hard drawdown stop
        dd = self._effective_drawdown(portfolio_state, nav_history)
        if dd >= self.constraint_config.dd_stop:
            # Force to cash (zero weights)
            safe_action = np.zeros(self.n_assets)
            reasons.append(
                f"DD_STOP: drawdown {dd:.2%} >= {self.constraint_config.dd_stop:.2%}")
            violations['dd_stop'] = dd - self.constraint_config.dd_stop

        # Rule 2: Leverage cap
        gross_leverage = np.sum(np.abs(safe_action))
        if gross_leverage > self.constraint_config.leverage_cap:
            # Scale down to cap
            scale = self.constraint_config.leverage_cap / gross_leverage
            safe_action = safe_action * scale
            reasons.append(f"LEV_CAP: leverage {gross_leverage:.2f} > {self.constraint_config.leverage_cap:.2f}")
            violations['leverage_cap'] = gross_leverage - self.constraint_config.leverage_cap

        # Rule 3: Turnover cap
        turnover = np.sum(np.abs(safe_action - portfolio_state.weights))
        if turnover > self.constraint_config.turnover_cap:
            # Interpolate towards target to stay within cap
            alpha = self.constraint_config.turnover_cap / turnover
            safe_action = portfolio_state.weights + alpha * (safe_action - portfolio_state.weights)
            reasons.append(f"TURN_CAP: turnover {turnover:.2f} > {self.constraint_config.turnover_cap:.2f}")
            violations['turnover_cap'] = turnover - self.constraint_config.turnover_cap

        # Rule 4: High volatility regime -> reduce positions
        bear_regime_prob = regime_belief[1] if len(regime_belief) > 1 else 0
        if bear_regime_prob > 0.6:
            # Cap maximum absolute weight in high-vol regime
            max_weight = self.shield_config.high_vol_position_cap
            safe_action = np.clip(safe_action, -max_weight, max_weight)
            reasons.append(f"HIGH_VOL_REGIME: bear prob {bear_regime_prob:.2f} > 0.6")

        # Compute diagnostics
        modification = np.linalg.norm(safe_action - proposed_action)
        was_modified = modification > 1e-6

        if was_modified:
            self.intervention_count += 1
            self.modification_magnitudes.append(modification)

        diagnostics = ShieldDiagnostics(
            was_modified=was_modified,
            modification_magnitude=modification,
            original_action=proposed_action.copy(),
            safe_action=safe_action.copy(),
            reasons=reasons,
            constraint_violations=violations
        )

        return safe_action, diagnostics


class ProjectionShield(PolicyShield):
    """
    Projection-based shield using quadratic programming.
    Projects actions onto the feasible constraint set.
    """

    def __init__(
            self,
            shield_config: ShieldConfig,
            constraint_config: ConstraintConfig,
            n_assets: int
    ):
        super().__init__(shield_config, constraint_config, n_assets)

    def apply(
            self,
            proposed_action: np.ndarray,
            portfolio_state: PortfolioState,
            regime_belief: np.ndarray,
            entropy: float,
            nav_history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        self.total_calls += 1

        violations = {}
        reasons = []

        # Check if hard stop triggered
        dd = self._effective_drawdown(portfolio_state, nav_history)
        if dd >= self.constraint_config.dd_stop:
            safe_action = np.zeros(self.n_assets)
            reasons.append("DD_STOP triggered")
            violations['dd_stop'] = dd - self.constraint_config.dd_stop

            diagnostics = ShieldDiagnostics(
                was_modified=True,
                modification_magnitude=np.linalg.norm(proposed_action),
                original_action=proposed_action.copy(),
                safe_action=safe_action,
                reasons=reasons,
                constraint_violations=violations
            )
            self.intervention_count += 1
            self.modification_magnitudes.append(np.linalg.norm(proposed_action))
            return safe_action, diagnostics

        # Project to feasible set
        def objective(w):
            """Minimize distance to proposed action."""
            return np.sum((w - proposed_action) ** 2)

        def grad_objective(w):
            return 2 * (w - proposed_action)

        # Constraints
        constraints = []

        # Leverage constraint: sum(|w|) <= cap
        # Approximated as: each |w_i| <= cap / n (can be refined)
        leverage_cap = self.constraint_config.leverage_cap

        # Turnover constraint: sum(|w - w_prev|) <= cap
        turnover_cap = self.constraint_config.turnover_cap
        w_prev = portfolio_state.weights

        # For simplicity, use box constraints and iterate
        bounds = [(-1.0, 1.0) for _ in range(self.n_assets)]

        # Constraint: gross leverage
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: leverage_cap - np.sum(np.abs(w))
        })

        # Run optimization
        result = minimize(
            objective,
            proposed_action,
            method='SLSQP',
            jac=grad_objective,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        if result.success:
            safe_action = result.x
        else:
            # Fallback to simple clipping
            safe_action = proposed_action.copy()
            gross = np.sum(np.abs(safe_action))
            if gross > leverage_cap:
                safe_action = safe_action * (leverage_cap / gross)

        # Additional turnover limiting (post-projection)
        turnover = np.sum(np.abs(safe_action - w_prev))
        if turnover > turnover_cap:
            alpha = turnover_cap / turnover
            safe_action = w_prev + alpha * (safe_action - w_prev)
            reasons.append(f"TURN_CAP after projection")
            violations['turnover_cap'] = turnover - turnover_cap

        modification = np.linalg.norm(safe_action - proposed_action)
        was_modified = modification > 1e-6

        if was_modified:
            self.intervention_count += 1
            self.modification_magnitudes.append(modification)
            reasons.append("Projection applied")

        diagnostics = ShieldDiagnostics(
            was_modified=was_modified,
            modification_magnitude=modification,
            original_action=proposed_action.copy(),
            safe_action=safe_action.copy(),
            reasons=reasons,
            constraint_violations=violations
        )

        return safe_action, diagnostics


class UncertaintyAwareShield(PolicyShield):
    """
    Shield that tightens constraints based on regime uncertainty (entropy).
    Novel contribution: connects regime belief to risk control.

    For leverage/turnover/position caps, tightening is multiplicative on
    normalized entropy, each with its own safety floor:
        cap_t = max(floor, cap_base * (1 - kappa * H(b_t) / H_max))
    Separately, when the Bear regime is dominant with high confidence
    (P(Bear) > 0.7 and low entropy), a fixed conservative position cap of
    0.2 is applied on top of the above -- there is no "Crisis" regime in
    this codebase's Bull/Bear/Sideways taxonomy (see regime_model.py).

    Drawdown is handled last, as a CONTINUOUS scaling factor applied on top
    of the (already entropy-tightened, already bear-capped) position, not a
    hard threshold:
        w_limit *= max(0, 1 - (DD - dd_soft) / (dd_hard - dd_soft))
    This naturally reaches exactly 1.0 at/below dd_soft (constraint_config.dd_limit)
    and exactly 0.0 at/beyond dd_hard (constraint_config.dd_stop), ramping
    linearly between -- so the position degrades smoothly through the
    dd_soft..dd_hard band instead of freezing to zero the instant dd_soft
    (or some entropy-adjusted version of it) is crossed. This also means the
    bear-regime cap above can actually bind (previously, a hard DD cutoff
    forced the action to exactly zero before the bear-cap check ever ran,
    so the bear cap could only matter when DD was NOT already breached).
    """

    def __init__(
            self,
            shield_config: ShieldConfig,
            constraint_config: ConstraintConfig,
            n_assets: int,
            n_regimes: int = 3
    ):
        super().__init__(shield_config, constraint_config, n_assets)
        self.n_regimes = n_regimes
        self.max_entropy = np.log(n_regimes)
        self.kappa = shield_config.entropy_tightening_kappa

    def _adjust_limits(self, entropy: float) -> Dict[str, float]:
        """Compute uncertainty-adjusted leverage/turnover/position limits.
        Drawdown is handled separately in apply() as a continuous post-hoc
        scaling factor, not an entropy-tightened threshold."""
        if self.shield_config.use_entropy_tightening:
            # Normalize entropy to [0, 1]
            normalized_entropy = entropy / self.max_entropy
            # Tightening factor
            tighten = self.kappa * normalized_entropy
        else:
            tighten = 0.0

        return {
            'leverage_cap': max(0.5, self.constraint_config.leverage_cap * (1 - tighten)),
            'turnover_cap': max(0.2, self.constraint_config.turnover_cap * (1 - tighten)),
            'position_cap': max(0.1, self.shield_config.high_vol_position_cap * (1 - tighten)),
        }

    def apply(
            self,
            proposed_action: np.ndarray,
            portfolio_state: PortfolioState,
            regime_belief: np.ndarray,
            entropy: float,
            nav_history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        self.total_calls += 1

        safe_action = proposed_action.copy()
        reasons = []
        violations = {}

        # Get uncertainty-adjusted limits
        adjusted_limits = self._adjust_limits(entropy)

        # Rule 1: Adjusted leverage cap
        gross_leverage = np.sum(np.abs(safe_action))
        if gross_leverage > adjusted_limits['leverage_cap']:
            scale = adjusted_limits['leverage_cap'] / gross_leverage
            safe_action = safe_action * scale
            reasons.append(
                f"ADJ_LEV_CAP: leverage {gross_leverage:.2f} > "
                f"adjusted cap {adjusted_limits['leverage_cap']:.2f}"
            )
            violations['leverage_cap'] = gross_leverage - adjusted_limits['leverage_cap']

        # Rule 2: Adjusted position cap (per asset)
        if np.max(np.abs(safe_action)) > adjusted_limits['position_cap']:
            safe_action = np.clip(
                safe_action,
                -adjusted_limits['position_cap'],
                adjusted_limits['position_cap']
            )
            reasons.append(f"ADJ_POS_CAP: position > {adjusted_limits['position_cap']:.2f}")

        # Rule 3: Adjusted turnover cap
        turnover = np.sum(np.abs(safe_action - portfolio_state.weights))
        if turnover > adjusted_limits['turnover_cap']:
            alpha = adjusted_limits['turnover_cap'] / turnover
            safe_action = portfolio_state.weights + alpha * (safe_action - portfolio_state.weights)
            reasons.append(
                f"ADJ_TURN_CAP: turnover {turnover:.2f} > "
                f"adjusted cap {adjusted_limits['turnover_cap']:.2f}"
            )
            violations['turnover_cap'] = turnover - adjusted_limits['turnover_cap']

        # Regime-specific rules
        # If bear regime is dominant and high confidence, reduce further
        if self.shield_config.use_bear_cap and len(regime_belief) > 1:
            bear_prob = regime_belief[1]
            if bear_prob > 0.7 and entropy < 0.5 * self.max_entropy:
                # High confidence bear regime -> very conservative
                conservative_cap = 0.2
                if np.max(np.abs(safe_action)) > conservative_cap:
                    safe_action = np.clip(safe_action, -conservative_cap, conservative_cap)
                    reasons.append(f"BEAR_REGIME: high confidence ({bear_prob:.2f})")

        # Rule 4: Continuous drawdown scaling, applied LAST (after entropy
        # tightening and the regime cap above), so it degrades whatever
        # position survived those rules rather than pre-empting them.
        if self.shield_config.use_dd_scaling:
            dd_soft = self.constraint_config.dd_limit
            dd_hard = self.constraint_config.dd_stop
            dd = self._effective_drawdown(portfolio_state, nav_history)
            if dd_hard > dd_soft:
                dd_scale = max(0.0, min(1.0, 1 - (dd - dd_soft) / (dd_hard - dd_soft)))
            else:
                dd_scale = 0.0 if dd >= dd_hard else 1.0
        else:
            dd_soft = dd_hard = dd = None
            dd_scale = 1.0

        if dd_scale < 1.0:
            safe_action = safe_action * dd_scale
            if dd_scale <= 0.0:
                reasons.append(f"DD_HARD_ZERO: dd {dd:.2%} >= dd_hard {dd_hard:.2%}")
                violations['dd_stop'] = dd - dd_hard
            else:
                reasons.append(
                    f"DD_SCALING: dd {dd:.2%} in [{dd_soft:.2%}, {dd_hard:.2%}], scale={dd_scale:.2f}"
                )
                violations['dd_scaling'] = dd - dd_soft

        # Compute diagnostics
        modification = np.linalg.norm(safe_action - proposed_action)
        was_modified = modification > 1e-6

        if was_modified:
            self.intervention_count += 1
            self.modification_magnitudes.append(modification)

        diagnostics = ShieldDiagnostics(
            was_modified=was_modified,
            modification_magnitude=modification,
            original_action=proposed_action.copy(),
            safe_action=safe_action.copy(),
            reasons=reasons,
            constraint_violations=violations
        )

        return safe_action, diagnostics


class RegimeGatedShield(PolicyShield):
    """
    Wraps an inner shield and only invokes it when
    P(Bear) > bear_shield_threshold OR normalized regime-belief entropy >
    entropy_gate_threshold -- inactive (passes the action through
    completely unmodified) in a confidently-resolved Bull/Sideways regime.

    Built for strategy-v2-overlay's RCOverlayConfig.shield_mode=
    "regime_gated": the hypothesis being tested is that clamping a passive/
    regime-timed book even while it's confidently NOT in a bear regime is
    pure drag with no protective value, so the shield should only turn on
    when there's actually something to protect against (a real bear signal
    or genuine regime uncertainty).
    """

    def __init__(
            self,
            inner_shield: PolicyShield,
            shield_config: ShieldConfig,
            constraint_config: ConstraintConfig,
            n_assets: int,
            n_regimes: int = 3,
            bear_shield_threshold: float = 0.5,
            entropy_gate_threshold: float = 0.5
    ):
        super().__init__(shield_config, constraint_config, n_assets)
        self.inner_shield = inner_shield
        self.n_regimes = n_regimes
        self.max_entropy = np.log(n_regimes) if n_regimes > 1 else 1.0
        self.bear_shield_threshold = bear_shield_threshold
        self.entropy_gate_threshold = entropy_gate_threshold

    def apply(
            self,
            proposed_action: np.ndarray,
            portfolio_state: PortfolioState,
            regime_belief: np.ndarray,
            entropy: float,
            nav_history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        self.total_calls += 1

        p_bear = regime_belief[1] if len(regime_belief) > 1 else 0.0
        h_norm = (entropy / self.max_entropy) if self.max_entropy > 0 else 0.0
        gate_active = (p_bear > self.bear_shield_threshold) or (h_norm > self.entropy_gate_threshold)

        if not gate_active:
            return proposed_action.copy(), ShieldDiagnostics(
                was_modified=False,
                modification_magnitude=0.0,
                original_action=proposed_action.copy(),
                safe_action=proposed_action.copy(),
                reasons=["REGIME_GATE_INACTIVE: resolved Bull/Sideways -- shield off"],
                constraint_violations={},
            )

        safe_action, diag = self.inner_shield.apply(
            proposed_action, portfolio_state, regime_belief, entropy, nav_history
        )
        if diag.was_modified:
            self.intervention_count += 1
            self.modification_magnitudes.append(diag.modification_magnitude)
        return safe_action, diag


class CompositeShield(PolicyShield):
    """
    Combines multiple shields in sequence.
    Useful for ablation studies.
    """

    def __init__(
            self,
            shields: List[PolicyShield],
            shield_config: ShieldConfig,
            constraint_config: ConstraintConfig,
            n_assets: int
    ):
        super().__init__(shield_config, constraint_config, n_assets)
        self.shields = shields

    def apply(
            self,
            proposed_action: np.ndarray,
            portfolio_state: PortfolioState,
            regime_belief: np.ndarray,
            entropy: float,
            nav_history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        self.total_calls += 1

        current_action = proposed_action.copy()
        all_reasons = []
        all_violations = {}
        total_modification = 0.0

        for shield in self.shields:
            safe_action, diag = shield.apply(
                current_action, portfolio_state, regime_belief, entropy, nav_history
            )

            if diag.was_modified:
                all_reasons.extend(diag.reasons)
                all_violations.update(diag.constraint_violations)
                total_modification += diag.modification_magnitude

            current_action = safe_action

        was_modified = total_modification > 1e-6
        if was_modified:
            self.intervention_count += 1
            self.modification_magnitudes.append(total_modification)

        diagnostics = ShieldDiagnostics(
            was_modified=was_modified,
            modification_magnitude=total_modification,
            original_action=proposed_action.copy(),
            safe_action=current_action.copy(),
            reasons=all_reasons,
            constraint_violations=all_violations
        )

        return current_action, diagnostics


def create_shield(
        shield_config: ShieldConfig,
        constraint_config: ConstraintConfig,
        n_assets: int,
        n_regimes: int = 3
) -> Optional[PolicyShield]:
    """Factory function to create appropriate shield. Returns None for
    ShieldType.NONE -- TradingCMDPEnv already treats shield=None as
    "no shield applied", so ShieldType.NONE is a genuinely unconstrained arm,
    not just a differently-configured shield."""
    if shield_config.shield_type == ShieldType.NONE:
        return None
    elif shield_config.shield_type == ShieldType.RULE_BASED:
        return RuleBasedShield(shield_config, constraint_config, n_assets)
    elif shield_config.shield_type == ShieldType.PROJECTION:
        return ProjectionShield(shield_config, constraint_config, n_assets)
    elif shield_config.shield_type == ShieldType.UNCERTAINTY_AWARE:
        return UncertaintyAwareShield(
            shield_config, constraint_config, n_assets, n_regimes
        )
    else:
        raise ValueError(f"Unknown shield type: {shield_config.shield_type}")


if __name__ == "__main__":
    # Test shields
    from config import ShieldConfig, ConstraintConfig, ShieldType

    print("Testing Policy Shields...")

    shield_config = ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE)
    constraint_config = ConstraintConfig(
        dd_stop=0.15,
        leverage_cap=2.0,
        turnover_cap=1.0
    )
    n_assets = 4

    shield = create_shield(shield_config, constraint_config, n_assets)

    # Create mock portfolio state
    portfolio_state = PortfolioState(
        nav=1_000_000,
        cash=200_000,
        weights=np.array([0.2, 0.2, 0.2, 0.2]),
        positions={},
        peak_nav=1_100_000,
        current_drawdown=0.09,  # 9% drawdown
        gross_leverage=0.8,
        net_exposure=0.8
    )

    # Test with different scenarios
    print("\n--- Scenario 1: Normal conditions ---")
    proposed = np.array([0.3, 0.3, 0.2, 0.2])
    regime_belief = np.array([0.6, 0.2, 0.2])  # Bull dominant
    entropy = 0.5

    safe, diag = shield.apply(proposed, portfolio_state, regime_belief, entropy)
    print(f"Proposed: {proposed}")
    print(f"Safe: {safe}")
    print(f"Modified: {diag.was_modified}")
    print(f"Reasons: {diag.reasons}")

    print("\n--- Scenario 2: High leverage action ---")
    proposed = np.array([0.8, 0.7, 0.6, 0.5])  # High leverage

    safe, diag = shield.apply(proposed, portfolio_state, regime_belief, entropy)
    print(f"Proposed: {proposed}")
    print(f"Safe: {safe}")
    print(f"Modified: {diag.was_modified}")
    print(f"Reasons: {diag.reasons}")

    print("\n--- Scenario 3: High entropy (regime transition) ---")
    entropy = 1.0  # High uncertainty

    safe, diag = shield.apply(proposed, portfolio_state, regime_belief, entropy)
    print(f"Proposed: {proposed}")
    print(f"Safe: {safe}")
    print(f"Modified: {diag.was_modified}")
    print(f"Reasons: {diag.reasons}")

    print(f"\n--- Shield Statistics ---")
    print(shield.get_statistics())