#!/usr/bin/env python3
"""
strategy-v2-overlay: RC-Overlay and Gated-RL, the two new deterministic-
overlay strategies (no training), plus the shared "trailing shield" config
they and the dev-fold retrofit comparison (run_dev_comparison.py) all use.

Both strategies return PRE-shield target weights every environment step.
The caller is expected to attach make_trailing_shield_config()'s shield and
set environment.decision_frequency_days=5 / environment.no_trade_band=0.10
on the env's config -- exactly the k=5/band=0.10 execution model used
throughout this project (see PROJECT_STATE.md) -- rather than either
strategy re-implementing that cadence itself. This mirrors how
run_baselines.py's own strategies are driven by main.create_trading_env.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from config import ShieldConfig, ShieldType
from evaluation import BaselineStrategy
from run_baselines import PriceHistory

TRADING_DAYS_PER_YEAR = 252


def make_trailing_shield_config(position_cap: float = 0.30, dd_anchor_window: int = 63) -> ShieldConfig:
    """The "trailing shield (30% cap, 63d anchor)" referenced throughout
    strategy-v2-overlay: a flat position cap (entropy-tightening and the
    bear-regime cap both disabled -- RC-Overlay/Gated-RL already apply
    their own regime/entropy response upstream, so this shield is meant as
    a hard rail, not a second copy of the same logic) plus trailing-
    anchored continuous drawdown scaling (the absorbing-trap fix). Leverage/
    turnover caps (UncertaintyAwareShield's Rules 1/3) still apply at
    whatever constraint_config the caller uses -- generic safety rails, not
    part of this spec, left at their defaults."""
    return ShieldConfig(
        shield_type=ShieldType.UNCERTAINTY_AWARE,
        high_vol_position_cap=position_cap,
        use_entropy_tightening=False,
        use_bear_cap=False,
        use_dd_scaling=True,
        dd_anchor="trailing",
        dd_anchor_window=dd_anchor_window,
    )


def make_shield_for_overlay(overlay_config: "RCOverlayConfig", constraint_config, n_assets: int,
                             n_regimes: int = 3):
    """Builds whichever shield RCOverlayConfig.shield_mode calls for.
    "always" (default) -> exactly make_trailing_shield_config()'s shield,
    unconditional, as before. "regime_gated" -> that same shield wrapped in
    policy_shield.RegimeGatedShield, active only when P(Bear) >
    bear_shield_threshold or normalized entropy > entropy_gate_threshold."""
    from policy_shield import create_shield, RegimeGatedShield

    inner_shield_config = make_trailing_shield_config()
    inner = create_shield(inner_shield_config, constraint_config, n_assets, n_regimes)

    if overlay_config.shield_mode == "always":
        return inner
    if overlay_config.shield_mode == "regime_gated":
        return RegimeGatedShield(
            inner, inner_shield_config, constraint_config, n_assets, n_regimes,
            bear_shield_threshold=overlay_config.bear_shield_threshold,
            entropy_gate_threshold=overlay_config.entropy_gate_threshold,
        )
    raise ValueError(f"Unknown shield_mode: {overlay_config.shield_mode!r}")


def build_env_with_shield(features_df, prices, config, aligned_data, mode: str, shield):
    """Mirrors main.create_trading_env's construction exactly, but takes an
    already-built shield object directly instead of constructing one from
    config.shield -- needed for RegimeGatedShield, which config.shield alone
    can't express (it isn't a ShieldType the central create_shield()
    factory knows about, by design -- see make_shield_for_overlay)."""
    import copy as _copy
    from regime_model import create_regime_model
    from trading_env import TradingCMDPEnv

    if mode == "test":
        config = _copy.deepcopy(config)
        config.training.episode_length = len(features_df)

    regime_model = create_regime_model(config.regime, input_dim=features_df.shape[1])
    if aligned_data is not None:
        regime_model.fit(aligned_data, mode=mode)

    return TradingCMDPEnv(
        features=features_df.values, prices=prices, tickers=config.data.tickers,
        config=config, regime_model=regime_model, shield=shield, mode=mode,
    )


def tsmom_inverse_vol_weights(price_window: np.ndarray, n_assets: int, vol_window: int = 60) -> np.ndarray:
    """Per-asset long-or-cash momentum sizing, replicated from
    run_baselines.TimeSeriesMomentumStrategy._compute_target_weights (see
    its docstring for the rationale for inverse-vol-vs-fixed-reference
    sizing rather than renormalizing to 1). `price_window` must be exactly
    the (lookback+1)-row window the caller wants momentum measured over --
    this function only looks at its first/last rows for the momentum sign
    and its trailing `vol_window` rows for sizing."""
    if len(price_window) < 2:
        return np.zeros(n_assets)

    past = price_window[0]
    current = price_window[-1]
    is_positive = current > past
    if not is_positive.any():
        return np.zeros(n_assets)

    vol_lookback = min(vol_window + 1, len(price_window))
    window_prices = price_window[-vol_lookback:]
    log_rets = np.diff(np.log(window_prices), axis=0)
    vol = np.std(log_rets, axis=0, ddof=1)
    vol = np.maximum(vol, 1e-6)

    ref_vol = float(np.median(vol))
    raw_weight = is_positive * (ref_vol / vol) / n_assets
    total = raw_weight.sum()
    if total > 1.0:
        raw_weight = raw_weight / total
    return raw_weight


@dataclass
class RCOverlayConfig:
    """Config-exposed thresholds/coefficients for RCOverlayStrategy.
    Defaults are strategy-v2-overlay's stated defaults; switch_hysteresis_days
    and partial_switch_blend are forensics-ablation additions (see
    run_rc_overlay_forensics.py) whose defaults exactly reproduce the
    original (immediate, full-swap) behavior -- setting them is opt-in."""
    bull_side_confidence: float = 0.7  # P(Bull)+P(Side) >= this -> equal-weight branch
    bear_confidence: float = 0.7  # P(Bear) > this (AND regime resolved) -> tsmom branch
    entropy_scale_coef: float = 0.3  # weights *= (1 - entropy_scale_coef * H_norm)
    tsmom_lookback: int = 252  # 12-month momentum lookback (tsmom_12m)
    tsmom_vol_window: int = 60  # inverse-vol sizing window

    # Branch-switch hysteresis: 1 (default) = commit to whatever the raw
    # rule says immediately, every call -- the original behavior. N>1 =
    # only commit to a NEW branch once the raw rule has indicated it for N
    # consecutive RESOLVED days (any unresolved day, or a day the raw rule
    # disagrees with the pending branch, breaks the streak) -- a second,
    # overlay-level hysteresis layer on top of the regime model's own.
    switch_hysteresis_days: int = 1

    # Partial switch: None (default) = jump straight to the new branch's
    # full weights the moment it commits (a full book swap). A float alpha
    # in (0, 1] = on the switching day ONLY, return
    # (1-alpha)*old_branch_weights + alpha*new_branch_weights instead --
    # e.g. 0.5 softens the immediate transition to a 50/50 blend.
    partial_switch_blend: Optional[float] = None

    # --- v2 redesign additions (run_v2_sweep.py); all default to exactly
    # reproducing the pre-redesign behavior -- opt-in only. ---

    # Bear-entry rule override: None (default) = the original
    # "P(Bear) > bear_confidence AND resolved" rule. A float = REPLACES that
    # rule with "P(Bear) > bear_entry_threshold" (resolved requirement
    # dropped entirely) -- the forensics found lateness, not wrongness, was
    # the dominant timing-miss mode, traced to the regime model's own
    # resolved-gate stacking on top of this overlay's switch hysteresis.
    bear_entry_threshold: Optional[float] = None

    # shield_mode="always" (default): the attached shield (see
    # make_trailing_shield_config) applies on every call, exactly as
    # before. "regime_gated": the shield only activates when
    # P(Bear) > bear_shield_threshold OR normalized entropy >
    # entropy_gate_threshold -- see policy_shield.RegimeGatedShield, which
    # is what actually implements the gating (this field only selects
    # which shield make_shield_for_overlay() builds).
    shield_mode: str = "always"
    bear_shield_threshold: float = 0.5
    entropy_gate_threshold: float = 0.5


class RCOverlayStrategy(BaselineStrategy):
    """Deterministic regime-conditioned overlay (no training). Inputs are
    the env's own regime beliefs (from the canonical hysteresis-EMA
    HeuristicRegimeModel attached to the env -- this strategy reads
    info['regime_belief']/['regime_entropy']/['regime_resolved'] rather than
    owning a duplicate model instance).

    Rules (regime_belief = [P(Bull), P(Bear), P(Side)], see regime_model.py):
      - P(Bull)+P(Side) >= bull_side_confidence -> equal-weight (1/N) target,
        the same target as baselines.equal_weight_k5_band010.
      - P(Bear) > bear_confidence AND the regime call is resolved (not
        mid hysteresis-transition, see RegimeModel.is_resolved) -> tsmom_12m
        -style inverse-vol weights.
      - Otherwise (the ambiguous/transitioning middle the two thresholds
        don't cover) -> falls back to the equal-weight branch, the more
        neutral of the two signals.
    All three branches are then scaled by (1 - entropy_scale_coef * H_norm),
    H_norm = normalized regime-belief entropy: the less confident the
    belief, the smaller the position, regardless of which branch fired.

    Returns PRE-shield weights -- see module docstring for the execution
    model this is meant to run inside."""

    def __init__(self, n_assets: int, config, price_history: PriceHistory,
                 overlay_config: Optional[RCOverlayConfig] = None,
                 trace: Optional[List[Dict]] = None):
        super().__init__(n_assets, config)
        self.price_history = price_history
        self.overlay = overlay_config or RCOverlayConfig()
        self.n_regimes = config.regime.n_regimes
        self._max_entropy = np.log(self.n_regimes) if self.n_regimes > 1 else 1.0
        self._step = -1
        # Branch-switch hysteresis state (only used when switch_hysteresis_days > 1).
        self._committed_branch: Optional[str] = None
        self._pending_branch: Optional[str] = None
        self._pending_count = 0
        # Optional forensics hook: if given a list, one dict is appended per
        # get_action call (see run_rc_overlay_forensics.py). None (default)
        # adds no overhead and changes no behavior.
        self.trace = trace

    def _parse_info(self, info: Optional[Dict]):
        if info is not None and 'regime_belief' in info:
            belief = np.asarray(info['regime_belief'], dtype=float)
        else:
            belief = np.ones(self.n_regimes) / self.n_regimes
        entropy = float(info.get('regime_entropy', 0.0)) if info is not None else 0.0
        resolved = bool(info.get('regime_resolved', True)) if info is not None else True
        return belief, entropy, resolved

    def _raw_branch(self, belief: np.ndarray, resolved: bool) -> str:
        """Which branch today's raw rule indicates, ignoring any switch
        hysteresis -- exactly the original (pre-ablation) selection logic."""
        p_bull = belief[0]
        p_bear = belief[1] if len(belief) > 1 else 0.0
        p_side = belief[2] if len(belief) > 2 else 0.0

        if p_bull + p_side >= self.overlay.bull_side_confidence:
            return 'bull_side'

        if self.overlay.bear_entry_threshold is not None:
            if p_bear > self.overlay.bear_entry_threshold:
                return 'bear'
        elif p_bear > self.overlay.bear_confidence and resolved:
            return 'bear'

        return 'bull_side'  # ambiguous middle -> the more neutral branch

    def _weights_for_branch(self, branch: str) -> np.ndarray:
        equal_weight = np.ones(self.n_assets) / self.n_assets
        if branch != 'bear':
            return equal_weight

        lookback = self.overlay.tsmom_lookback
        prices = self.price_history.history_up_to(self._step)
        if len(prices) < lookback + 1:
            return np.zeros(self.n_assets)
        window = prices[-(lookback + 1):]
        return tsmom_inverse_vol_weights(window, self.n_assets, self.overlay.tsmom_vol_window)

    def _branch_and_weights(self, belief: np.ndarray, resolved: bool):
        """Resolve today's committed branch (applying switch hysteresis if
        configured) and its weights (applying the partial-switch blend on
        the switching day if configured). Returns (raw_branch,
        committed_branch, weights, switched)."""
        raw_branch = self._raw_branch(belief, resolved)
        prev_committed = self._committed_branch

        if self.overlay.switch_hysteresis_days <= 1 or prev_committed is None:
            committed_branch = raw_branch
            self._pending_branch = None
            self._pending_count = 0
        elif raw_branch == prev_committed:
            committed_branch = prev_committed
            self._pending_branch = None
            self._pending_count = 0
        else:
            if raw_branch == self._pending_branch and resolved:
                self._pending_count += 1
            else:
                self._pending_branch = raw_branch
                self._pending_count = 1 if resolved else 0

            if self._pending_count >= self.overlay.switch_hysteresis_days:
                committed_branch = raw_branch
                self._pending_branch = None
                self._pending_count = 0
            else:
                committed_branch = prev_committed  # not yet confirmed -- stay put

        switched = prev_committed is not None and committed_branch != prev_committed
        weights = self._weights_for_branch(committed_branch)

        if switched and self.overlay.partial_switch_blend is not None:
            old_weights = self._weights_for_branch(prev_committed)
            alpha = self.overlay.partial_switch_blend
            weights = (1 - alpha) * old_weights + alpha * weights

        self._committed_branch = committed_branch
        return raw_branch, committed_branch, weights, switched

    def _pre_entropy_weights(self, info: Optional[Dict]):
        """Base regime-conditioned weights, before entropy scaling --
        factored out so RCOverlayVolTargetStrategy can wrap it. Advances
        the price-history step counter exactly once per call."""
        self._step += 1
        belief, entropy, resolved = self._parse_info(info)
        raw_branch, committed_branch, base, switched = self._branch_and_weights(belief, resolved)

        if self.trace is not None:
            self.trace.append({
                'step': self._step, 'raw_branch': raw_branch, 'branch': committed_branch,
                'switched': switched, 'entropy': entropy, 'resolved': resolved,
                'base_weights': base.copy(),
            })

        return base, entropy

    def get_action(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        base, entropy = self._pre_entropy_weights(info)
        h_norm = float(np.clip(entropy / self._max_entropy, 0.0, 1.0)) if self._max_entropy > 0 else 0.0
        return base * (1 - self.overlay.entropy_scale_coef * h_norm)

    def reset(self):
        self._step = -1
        self._committed_branch = None
        self._pending_branch = None
        self._pending_count = 0


class RCOverlayVolTargetStrategy(RCOverlayStrategy):
    """RCOverlayStrategy wrapped with realized-vol targeting (matching
    run_baselines.VolTargetStrategy's scaling formula), applied to the
    regime-conditioned base weights BEFORE the entropy scale, so the two
    shrinkage factors compose multiplicatively and independently."""

    def __init__(self, n_assets: int, config, price_history: PriceHistory,
                 overlay_config: Optional[RCOverlayConfig] = None,
                 target_vol: float = 0.10, vol_window: int = 20, max_leverage: float = 2.0,
                 trace: Optional[List[Dict]] = None):
        super().__init__(n_assets, config, price_history, overlay_config, trace)
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage

    def _vol_target_scale(self, base: np.ndarray) -> float:
        prices = self.price_history.history_up_to(self._step)
        if len(prices) < self.vol_window + 1 or not np.any(base):
            return 1.0

        window_prices = prices[-(self.vol_window + 1):]
        asset_log_rets = np.diff(np.log(window_prices), axis=0)
        port_rets = asset_log_rets @ base
        realized_vol = float(np.std(port_rets, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        realized_vol = max(realized_vol, 1e-4)
        return min(self.target_vol / realized_vol, self.max_leverage)

    def _pre_entropy_weights(self, info: Optional[Dict]):
        base, entropy = super()._pre_entropy_weights(info)
        scale = self._vol_target_scale(base)
        return base * scale, entropy


class GatedRLStrategy(BaselineStrategy):
    """Gates between a loaded canonical ppo_lagrangian shielded checkpoint's
    deterministic action (when P(Bull) > bull_gate_threshold and the regime
    call is resolved) and RCOverlayStrategy's weights otherwise. Both
    branches flow through the SAME trailing shield + k=5/band=0.10
    execution at the env level -- this class only picks which raw
    (pre-shield) action to hand the env each step; it is not a second
    shield."""

    def __init__(self, n_assets: int, config, price_history: PriceHistory, policy,
                 device: str = "cpu", overlay_config: Optional[RCOverlayConfig] = None,
                 bull_gate_threshold: float = 0.7):
        super().__init__(n_assets, config)
        self.rc_overlay = RCOverlayStrategy(n_assets, config, price_history, overlay_config)
        self.policy = policy
        self.device = device
        self.bull_gate_threshold = bull_gate_threshold

    def get_action(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        # Always compute the RC-Overlay fallback: keeps its price-history
        # step counter in sync regardless of which branch fires, and is
        # exactly what's returned on any non-RL-gated day.
        rc_weights = self.rc_overlay.get_action(obs, info)

        belief = info.get('regime_belief') if info is not None else None
        resolved = bool(info.get('regime_resolved', True)) if info is not None else True

        if belief is not None and belief[0] > self.bull_gate_threshold and resolved:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action, _, _ = self.policy.get_action(obs_t, deterministic=True)
            return action.cpu().numpy().squeeze()

        return rc_weights

    def reset(self):
        self.rc_overlay.reset()
