#!/usr/bin/env python3
"""
P6 (approved 2026-10-16, strictly scoped): ACCOUNTING CORRECTION of the
previously frozen fold-4 (2024-2025) evaluation -- not a second look.

Re-runs exactly the frozen artifacts of `run_fold4_evaluation.py` (the
2026-08-06 one-look) under the corrected accounting:
  * audit defect #8 (slippage units) -- fixed in portfolio_engine, applies
    automatically;
  * audit defect #9 (regime look-ahead) -- every config's regime thresholds
    calibrated on the fold-4 TRAINING window (2021-2023) via
    main.calibrate_regime and frozen for the test window (previously fit on
    the test window itself).
Artifacts: the 7 deterministic baselines, both frozen v2 finalists, the
existing canonical ppo_lagrangian fold-4 checkpoints (none / uncertainty_aware,
seeds 1-5, trained 2021-2023 in August; EVAL ONLY, no retraining) and their
trailing-anchor retrofit (part of the frozen protocol), plus
ew_matched_execution (parameter-free: constant 1/N through the canonical
execution layer), explicitly requested. No new designs, no thresholds chosen,
same data cache as the one-look (asserted unchanged), same metric path.

ONE PASS: refuses to run if the output CSV already exists.
This script deliberately does not import r2_common (whose import-time guard
blocks fold-4 data); fold-4 access is the purpose of this approved run.
"""
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import main as main_module
from config import Config, ShieldType
from evaluation import WalkForwardEvaluator
from evaluate_saved import load_model as load_saved_policy
from run_baselines import PriceHistory, build_strategies, EqualWeightBandedStrategy
from overlay_strategies import RCOverlayStrategy, RCOverlayConfig, make_shield_for_overlay, build_env_with_shield
from run_paper_experiments import CANONICAL_FOLD2_CONFIG_OVERRIDES
from run_validation_workstream import _apply_overrides
import run_fold4_evaluation as F   # the frozen protocol: window, paths, episode runners, metric path

OUT_DIR = Path("./results/r2/corrected/fold4")
OUT_CSV = OUT_DIR / "fold4_corrected_results.csv"
PROV = OUT_DIR / "fold4_corrected_provenance.json"
CACHE = Path("./data_cache/raw_data_2020-11-22_2025-12-31_000001.SS-CLF-DJI-GDAXI-GSPC.pkl")
EXEC_OVERRIDES = {k: v for k, v in CANONICAL_FOLD2_CONFIG_OVERRIDES.items()
                  if k.startswith(("environment.", "action_space."))}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class ConstantEW:
    def __init__(self, n):
        self.a = np.full(n, 1.0 / n, dtype=np.float32)

    def get_action(self, obs, info=None):
        return self.a

    def reset(self):
        pass


def calibrated(cfg, train_aligned):
    cfg = copy.deepcopy(cfg)
    assert getattr(cfg.regime, "calibrated_vol_thresholds", None) is None, "unexpected pre-set thresholds"
    main_module.calibrate_regime(cfg, train_aligned)        # audit defect #9: train-window fit, frozen at test
    assert cfg.regime.calibrated_vol_thresholds is not None
    return cfg


def main():
    if OUT_CSV.exists():
        sys.exit(f"{OUT_CSV} exists -- P6 is one pass; refusing to re-run.")
    assert CACHE.exists(), "one-look data cache missing -- refusing to download fresh fold-4 data"
    cache_sha, cache_mtime = sha256(CACHE), CACHE.stat().st_mtime
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    W = F.FOLD_4_WINDOW
    dates = [W['train'][0], W['train'][1], W['test'][0], W['test'][1]]
    base = main_module.create_experiment_config(
        tickers=F.TICKERS, start_date=main_module._buffered_start_date(min(dates)), end_date=max(dates),
        total_timesteps=1, use_synthetic=False, seed=1)
    rf = base.evaluation.risk_free_rate
    features, prices_df, _, _, aligned = main_module.load_and_prepare_data(base, use_synthetic=False)
    assert CACHE.stat().st_mtime == cache_mtime and sha256(CACHE) == cache_sha, "data cache was rewritten"
    prices = {t: prices_df[t].values for t in F.TICKERS}
    sp = WalkForwardEvaluator(base).generate_calendar_splits(aligned.index, [W])[0]
    (tr0, tr1), (te0, te1) = sp['train'], sp['test']
    tf, ta, tra = features.iloc[te0:te1], aligned.iloc[te0:te1], aligned.iloc[tr0:tr1]
    tp = {t: p[te0:te1] for t, p in prices.items()}
    lb0 = max(0, te0 - 260)
    ph = PriceHistory({t: p[lb0:te1] for t, p in prices.items()}, lookback_len=te0 - lb0, tickers=F.TICKERS)
    base = calibrated(base, tra)
    rows, ckpts = [], {}

    def add(name, result):
        r = F._finalize_row(name, result, rf)
        rows.append(r)
        F._print_row(r)

    # ---- 7 baselines (frozen) ----
    bcfg = copy.deepcopy(base)
    bcfg.shield.shield_type = ShieldType.NONE
    for name, strat in build_strategies(len(F.TICKERS), bcfg, ph, tra).items():
        env = main_module.create_trading_env(tf, tp, copy.deepcopy(bcfg), aligned_data=ta, mode="test")
        add(name, F._run_strategy_episode(env, strat, bcfg))

    # ---- ew_matched_execution (requested; parameter-free) ----
    ecfg = _apply_overrides(copy.deepcopy(bcfg), EXEC_OVERRIDES)
    env = main_module.create_trading_env(tf, tp, ecfg, aligned_data=ta, mode="test")
    add("ew_matched_execution", F._run_strategy_episode(env, ConstantEW(len(F.TICKERS)), ecfg))

    # ---- frozen finalist 1 (verbatim settings from run_fold4_evaluation) ----
    oc = RCOverlayConfig(entropy_scale_coef=0.0, switch_hysteresis_days=5, bear_entry_threshold=0.6,
                         shield_mode="regime_gated", bear_shield_threshold=0.6, entropy_gate_threshold=0.5)
    c = copy.deepcopy(base)
    c.environment.decision_frequency_days, c.environment.no_trade_band = 5, 0.15
    c.environment.action_smoothing_beta = None
    sh = make_shield_for_overlay(oc, c.constraints, len(F.TICKERS), n_regimes=c.regime.n_regimes)
    env = build_env_with_shield(tf, tp, c, ta, mode="test", shield=sh)
    add("finalist_shield-regime_gated_bear0.6_band0.15_k5",
        F._run_strategy_episode(env, RCOverlayStrategy(len(F.TICKERS), c, ph, oc), c))

    # ---- frozen finalist 2 ----
    oc2 = RCOverlayConfig(shield_mode="regime_gated", bear_shield_threshold=0.6, entropy_gate_threshold=0.5)
    c = copy.deepcopy(base)
    c.environment.decision_frequency_days, c.environment.no_trade_band = 1, None
    c.environment.action_smoothing_beta = None
    sh = make_shield_for_overlay(oc2, c.constraints, len(F.TICKERS), n_regimes=c.regime.n_regimes)
    env = build_env_with_shield(tf, tp, c, ta, mode="test", shield=sh)
    add("finalist_gated_passive_ew_k5_band010",
        F._run_strategy_episode(env, EqualWeightBandedStrategy(len(F.TICKERS), c, rebalance_freq=5,
                                                               no_trade_band=0.10), c))

    # ---- existing canonical checkpoints, eval only (+ frozen trailing retrofit) ----
    for shield_name in ("none", "uncertainty_aware"):
        for seed in F.SEEDS:
            d = Path(str(F.GRID_DIR).format(seed=seed, shield=shield_name))
            ckpts[str(d / "final_model.pt")] = sha256(d / "final_model.pt")
            cfg = calibrated(Config.from_json(str(d / "config.json")), tra)
            env = main_module.create_trading_env(tf, tp, cfg, aligned_data=ta, mode="test")
            pol = load_saved_policy(d / "final_model.pt", env.observation_space.shape[0], len(F.TICKERS), cfg)
            add(f"canonical_{shield_name}_ppo_lagrangian_seed{seed}", F._run_policy_episode(env, pol))
            if shield_name == "uncertainty_aware":
                rc_ = copy.deepcopy(cfg)
                rc_.shield.dd_anchor, rc_.shield.dd_anchor_window = "trailing", 63
                env = main_module.create_trading_env(tf, tp, rc_, aligned_data=ta, mode="test")
                pol = load_saved_policy(d / "final_model.pt", env.observation_space.shape[0], len(F.TICKERS), rc_)
                add(f"canonical_uncertainty_aware_trailing_retrofit_seed{seed}", F._run_policy_episode(env, pol))

    df = pd.DataFrame(rows)[['variant', 'window', 'return', 'sharpe', 'calmar', 'max_dd',
                             'trailing_63d_max_dd', 'violation_rate', 'turnover']]
    df.to_csv(OUT_CSV, index=False)
    PROV.write_text(json.dumps({
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        'data_cache': str(CACHE), 'data_cache_sha256': cache_sha,
        'test_window': [str(ta.index.min().date()), str(ta.index.max().date()), len(ta)],
        'train_window_for_regime_calibration': [str(tra.index.min().date()), str(tra.index.max().date())],
        'calibrated_vol_thresholds': list(map(float, base.regime.calibrated_vol_thresholds)),
        'checkpoint_sha256': ckpts, 'retraining': False, 'one_pass': True}, indent=2), encoding="utf-8")
    print(f"wrote {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
