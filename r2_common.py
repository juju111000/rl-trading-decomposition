#!/usr/bin/env python3
"""
Shared helpers for the reviewer-response round-2 experiments (E1-E6,
outputs under results/r2/).

FOLD 4 (2024-2025) IS NEVER LOADED HERE. The only data loaders used are
run_validation_workstream._load_fold2_slices (fold 2, 2009-11-22..2017-12-31)
and run_rc_overlay_forensics._load_dev_data (folds 1-3, 2005-11-22..2023-12-31);
assert_no_fold4() checks every loaded index anyway.

Environment construction mirrors main.create_trading_env /
run_validation_workstream.build_env_with_features exactly (regime model
created and fit() on the same aligned slice the pipeline passes, then
TradingCMDPEnv) -- the only difference is that the shield OBJECT is passed in,
so the regime-gated wrapper (policy_shield.RegimeGatedShield) can be used.
"""
import copy
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from config import Config, ShieldType
from evaluate_saved import load_model as load_saved_policy
from policy_shield import create_shield, RegimeGatedShield
from regime_model import create_regime_model
from trading_env import TradingCMDPEnv

# R2_DIR=./results/r2/corrected selects the corrected-accounting master batch
# (audit defects #8/#9 fixed): all outputs and all checkpoint lookups then live
# under that tree; nothing under the original results/ paths is touched.
import os
R2_DIR = Path(os.environ.get("R2_DIR", "./results/r2"))
CORRECTED = "corrected" in R2_DIR.parts
FOLD4_START = pd.Timestamp("2024-01-01")


# =============================================================================
# Fail-closed fold-4 guard -- installed at import time, before any loader runs
# =============================================================================

class Fold4AccessError(RuntimeError):
    pass


def _check_end(end, what):
    if end is not None and pd.Timestamp(end) >= FOLD4_START:
        raise Fold4AccessError(f"{what} requests data through {end} -- fold 4 (>= {FOLD4_START.date()}) "
                               f"is prohibited in the R2 batch")


def install_fold4_guard():
    """Every real-data path in this codebase goes through data_loader.DataLoader
    (constructed by main.load_and_prepare_data) and every calendar split through
    evaluation.WalkForwardEvaluator.generate_calendar_splits. Both are wrapped so
    that a request reaching 2024-01-01 or later raises BEFORE any download,
    cache read, or slicing happens."""
    import data_loader
    import evaluation
    if getattr(data_loader.DataLoader, "_r2_guarded", False):
        return
    orig_init = data_loader.DataLoader.__init__

    def guarded_init(self, config, *a, **k):
        _check_end(config.end_date, f"DataLoader(start={config.start_date})")
        return orig_init(self, config, *a, **k)

    data_loader.DataLoader.__init__ = guarded_init
    data_loader.DataLoader._r2_guarded = True

    orig_splits = evaluation.WalkForwardEvaluator.generate_calendar_splits

    def guarded_splits(self, index, windows, *a, **k):
        for w in windows:
            _check_end(w['test'][1], f"calendar window {w}")
            _check_end(w['train'][1], f"calendar window {w}")
        return orig_splits(self, index, windows, *a, **k)

    evaluation.WalkForwardEvaluator.generate_calendar_splits = guarded_splits


install_fold4_guard()
GATE_BEAR_THRESHOLD = 0.7
GATE_ENTROPY_THRESHOLD = 0.5
N_REGIMES = 3
REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Sideways"}

GRID_DIR = (R2_DIR / "grid") if CORRECTED else Path("./results/full_grid/20260803_174817")
HARDENING_DIR = (R2_DIR / "hardening") if CORRECTED else Path("./results/hardening")


def assert_no_fold4(index) -> None:
    idx = pd.DatetimeIndex(index)
    assert idx.max() < FOLD4_START, f"fold-4 data loaded (max date {idx.max()}) -- forbidden"


# =============================================================================
# Checkpoint locations
# =============================================================================

def fold2_canonical_checkpoint(seed: int) -> Path:
    """The n=20 hardening canonical arm: seeds 1-5 reuse the grid checkpoint,
    seeds 6-20 were trained fresh by run_hardening (see run_hardening.py)."""
    if CORRECTED:   # P1: all 20 seeds trained fresh under corrected accounting
        return HARDENING_DIR / "canonical_training" / f"canonical_seed{seed}"
    if seed <= 5:
        return GRID_DIR / f"seed_{seed}" / "fold_2" / "none" / "ppo_lagrangian"
    return HARDENING_DIR / "canonical_training" / f"canonical_seed{seed}"


def grid_checkpoint(seed: int, fold: int) -> Path:
    return GRID_DIR / f"seed_{seed}" / f"fold_{fold}" / "none" / "ppo_lagrangian"


# =============================================================================
# Shields and environments
# =============================================================================

def make_shield(mode: str, config: Config):
    """off | always_on | gated. always_on = the grid's shielded arm
    (UNCERTAINTY_AWARE, every other ShieldConfig field as in the checkpoint's
    config -- all-time-peak drawdown anchor). gated = that identical shield
    wrapped in RegimeGatedShield, active only when P(Bear) > 0.7 OR
    normalized entropy > 0.5."""
    if mode == "off":
        return None
    shield_config = copy.deepcopy(config.shield)
    shield_config.shield_type = ShieldType.UNCERTAINTY_AWARE
    assert shield_config.dd_anchor == "peak", "E2 spec: all-time-peak anchor"
    inner = create_shield(shield_config, config.constraints, n_assets=len(config.data.tickers),
                          n_regimes=config.regime.n_regimes)
    if mode == "always_on":
        return inner
    if mode == "gated":
        return RegimeGatedShield(inner, shield_config, config.constraints, len(config.data.tickers),
                                 config.regime.n_regimes,
                                 bear_shield_threshold=GATE_BEAR_THRESHOLD,
                                 entropy_gate_threshold=GATE_ENTROPY_THRESHOLD)
    if mode == "hybrid":
        return TrailingDDPlusGatedCapsShield(shield_config, config.constraints, len(config.data.tickers),
                                             config.regime.n_regimes)
    raise ValueError(mode)


def build_env(features_array: np.ndarray, prices: Dict[str, np.ndarray], config: Config,
              aligned_data, mode: str, shield) -> TradingCMDPEnv:
    if mode == "test":
        config = copy.deepcopy(config)
        config.training.episode_length = len(features_array)
    regime_model = create_regime_model(config.regime, input_dim=features_array.shape[1])
    if aligned_data is not None:
        regime_model.fit(aligned_data, mode=mode)
    return TradingCMDPEnv(features=features_array, prices=prices, tickers=config.data.tickers,
                          config=config, regime_model=regime_model, shield=shield, mode=mode)


def gate_condition(belief, entropy, max_entropy=np.log(N_REGIMES)) -> bool:
    return (belief[1] > GATE_BEAR_THRESHOLD) or (entropy / max_entropy > GATE_ENTROPY_THRESHOLD)


def run_episode(env: TradingCMDPEnv, policy) -> pd.DataFrame:
    """One deterministic test episode; one row per env step (= test day)."""
    obs, _ = env.reset()
    rows = []
    nav0 = env.portfolio.initial_capital
    done = False
    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _ = policy.get_action(obs_t, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action.cpu().numpy().squeeze())
        done = terminated or truncated
        cv = np.asarray(info['cost_vec'], dtype=float)
        belief = np.asarray(info.get('regime_belief', [np.nan] * N_REGIMES), dtype=float)
        ent = float(info.get('regime_entropy', np.nan))
        rows.append({
            'nav': info['nav'], 'turnover': info.get('turnover', 0.0),
            'txn_cost': info.get('transaction_cost', 0.0), 'drawdown': info.get('drawdown', np.nan),
            'leverage': info.get('leverage', np.nan),
            'cost_dd': cv[0], 'cost_turnover': cv[1], 'cost_leverage': cv[2],
            'p_bull': belief[0], 'p_bear': belief[1], 'p_sideways': belief[2],
            'entropy_norm': ent / np.log(N_REGIMES),
            'gate_condition': gate_condition(belief, ent),
            'shield_modified': bool(info.get('shield', {}).get('was_modified', False)),
        })
    df = pd.DataFrame(rows)
    df.attrs['nav0'] = nav0
    return df


def episode_summary(df: pd.DataFrame) -> Dict:
    nav = np.concatenate([[df.attrs.get('nav0', 1_000_000.0)], df['nav'].to_numpy()])
    peak = np.maximum.accumulate(nav)
    return {
        'return': float(nav[-1] / nav[0] - 1),
        'max_dd': float(np.max(1 - nav / peak)),
        # drawdown > 0.20 (peak-anchored), computed from the drawdown series itself so it stays
        # comparable when E2's sensitivity add-on moves the configured dd_limit; identical to the
        # grid/hardening dd_violation_rate at the canonical 0.20 setting.
        'violation_rate': float(np.mean(df['drawdown'] > 0.20)),
        'any_violation_rate': float(np.mean((df[['cost_dd', 'cost_turnover', 'cost_leverage']] > 0).any(axis=1))),
        'gate_active_share': float(np.mean(df['gate_condition'])),
        'shield_intervention_rate': float(np.mean(df['shield_modified'])),
        'mean_turnover': float(df['turnover'].mean()),
        'mean_gross_exposure': float(df['leverage'].mean()),
        'n_days': int(len(df)),
    }


def load_policy_for(ckpt_dir: Path, obs_dim: int, config: Config):
    policy = load_saved_policy(ckpt_dir / "final_model.pt", obs_dim, len(config.data.tickers), config)
    policy.eval()
    return policy


# =============================================================================
# Statistics
# =============================================================================

def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1)) if n > 1 else float('nan')
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float('nan')
    return m, sd, (m - h, m + h)


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def paired_stats(a, b) -> Dict:
    """Paired t AND Wilcoxon on d = a - b, plus the MDE for a paired t-test
    at 80% power given the observed SD of the differences."""
    from statsmodels.stats.power import TTestPower
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    m, sd, (lo, hi) = ci95(d)
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(d)
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except ValueError:  # all differences zero
        w_stat, w_p = float('nan'), float('nan')
    mde = float(TTestPower().solve_power(effect_size=None, nobs=len(d), alpha=0.05, power=0.8,
                                         alternative='two-sided') * sd) if sd > 0 else 0.0
    return {'n': len(d), 'mean_d': m, 'sd_d': sd, 'ci_lo': lo, 'ci_hi': hi,
            't': float(t.statistic), 't_p': float(t.pvalue), 'w': w_stat, 'w_p': w_p,
            'n_pos': int((d > 0).sum()), 'mde': mde}


def independent_stats(a, b) -> Dict:
    from statsmodels.stats.power import TTestIndPower
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    na, nb = len(a), len(b)
    pooled = float(np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)))
    mde = float(TTestIndPower().solve_power(effect_size=None, nobs1=na, alpha=0.05, power=0.8,
                                            ratio=nb / na, alternative='two-sided') * pooled)
    return {'diff': float(a.mean() - b.mean()), 'welch_t': float(w.statistic), 'welch_p': float(w.pvalue),
            'mw_u': float(u.statistic), 'mw_p': float(u.pvalue), 'pooled_sd': pooled, 'mde': mde}


# =============================================================================
# E3: shadow belief operator for the gate only (policy inputs unchanged)
# =============================================================================

class ShadowBeliefGatedShield(RegimeGatedShield):
    """RegimeGatedShield whose GATE reads beliefs from a separate 'shadow'
    regime model (e.g. softmax operator) updated in lockstep with the env's own
    canonical model. Everything else -- the policy's observations, and the
    regime_belief/entropy passed to the inner shield's own rules -- stays on the
    env's canonical hysteresis beliefs, so the ONLY difference between
    softmax-gated and hysteresis-gated runs is the operator that drives the gate."""

    def __init__(self, inner, shield_config, constraint_config, n_assets, n_regimes, shadow_model):
        super().__init__(inner, shield_config, constraint_config, n_assets, n_regimes,
                         bear_shield_threshold=GATE_BEAR_THRESHOLD,
                         entropy_gate_threshold=GATE_ENTROPY_THRESHOLD)
        self.shadow = shadow_model
        self.gate_log = []

    def apply(self, proposed_action, portfolio_state, regime_belief, entropy, nav_history=None):
        sb, se = self.shadow.get_belief(), self.shadow.get_entropy()
        active = gate_condition(sb, se, self.max_entropy)
        self.gate_log.append(active)
        self.total_calls += 1
        if not active:
            return RegimeGatedShield.apply(self, proposed_action, portfolio_state,
                                           np.array([1.0, 0.0, 0.0]), 0.0, nav_history)  # forces gate-off path
        safe, diag = self.inner_shield.apply(proposed_action, portfolio_state, regime_belief, entropy, nav_history)
        if diag.was_modified:
            self.intervention_count += 1
            self.modification_magnitudes.append(diag.modification_magnitude)
        return safe, diag


def attach_shadow_model(env, shadow_model, aligned_data):
    """Fit the shadow model on the same slice as env.regime_model, and make the
    env's reset()/update() calls drive both models identically."""
    shadow_model.config.calibrated_vol_thresholds = env.regime_model.config.calibrated_vol_thresholds
    shadow_model.fit(aligned_data, mode="test")
    main_model = env.regime_model
    orig_update, orig_reset = main_model.update, main_model.reset

    def update(obs, prices=None):
        shadow_model.update(obs, prices=prices)
        return orig_update(obs, prices=prices)

    def reset():
        shadow_model.reset()
        return orig_reset()

    main_model.update, main_model.reset = update, reset


# =============================================================================
# Manifests
# =============================================================================

def file_sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(name: str, payload: dict):
    import json
    from datetime import datetime, timezone
    payload = dict(payload)
    payload.setdefault('generated_utc', datetime.now(timezone.utc).isoformat())
    out = R2_DIR / f"{name}_manifest.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    return out


# =============================================================================
# P4 fourth shield mode: trailing-anchor DD scaling always on + gated caps
# =============================================================================

from policy_shield import PolicyShield, ShieldDiagnostics, UncertaintyAwareShield  # noqa: E402


class TrailingDDPlusGatedCapsShield(PolicyShield):
    """Every day: continuous drawdown scaling, exposure x max(0, 1 - (DD - dd_limit)/(dd_stop - dd_limit)),
    with DD anchored to the max NAV over the trailing 63 days (not the all-time peak, so the 30% level
    is not absorbing). Gate-active days only (P(Bear) > 0.7 OR normalized entropy > 0.5): the
    always-on shield's caps (entropy-tightened leverage / position / turnover caps, bear cap), applied
    BEFORE the drawdown scaling -- i.e. UncertaintyAwareShield with its own DD scaling switched off."""

    def __init__(self, shield_config, constraint_config, n_assets, n_regimes=3, dd_anchor_window=63):
        trailing = copy.deepcopy(shield_config)
        trailing.dd_anchor, trailing.dd_anchor_window = "trailing", dd_anchor_window
        super().__init__(trailing, constraint_config, n_assets)
        caps_cfg = copy.deepcopy(shield_config)
        caps_cfg.use_dd_scaling = False
        self.caps = UncertaintyAwareShield(caps_cfg, constraint_config, n_assets, n_regimes)
        self.max_entropy = np.log(n_regimes)

    def apply(self, proposed_action, portfolio_state, regime_belief, entropy, nav_history=None):
        self.total_calls += 1
        action = np.asarray(proposed_action, dtype=float).copy()
        reasons = []
        if gate_condition(regime_belief, entropy, self.max_entropy):
            action, d = self.caps.apply(action, portfolio_state, regime_belief, entropy, nav_history)
            action = np.asarray(action, dtype=float)
            reasons += d.reasons
        soft, hard = self.constraint_config.dd_limit, self.constraint_config.dd_stop
        dd = self._effective_drawdown(portfolio_state, nav_history)
        scale = max(0.0, min(1.0, 1 - (dd - soft) / (hard - soft))) if hard > soft else (0.0 if dd >= hard else 1.0)
        if scale < 1.0:
            action = action * scale
            reasons.append(f"TRAILING_DD_SCALING: dd {dd:.2%} scale={scale:.2f}")
        mod = float(np.linalg.norm(action - np.asarray(proposed_action, dtype=float)))
        if mod > 1e-6:
            self.intervention_count += 1
            self.modification_magnitudes.append(mod)
        return action.astype(np.float32), ShieldDiagnostics(
            was_modified=mod > 1e-6, modification_magnitude=mod,
            original_action=np.asarray(proposed_action).copy(), safe_action=action.copy(),
            reasons=reasons, constraint_violations={})


def ensure_calibrated(config, train_aligned):
    """Audit defect #9: a test env needs the regime model's TRAINING-calibrated
    thresholds. Checkpoints trained after the fix carry them in config.json;
    otherwise calibrate on the fold's training slice (consumes no RNG)."""
    if getattr(config.regime, "calibrated_vol_thresholds", None) is None:
        import main as _main
        _main.calibrate_regime(config, train_aligned)
    return config
