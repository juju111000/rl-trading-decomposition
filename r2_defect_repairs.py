#!/usr/bin/env python3
"""
Audit defects #8 (slippage units) and #9 (regime look-ahead): measured effect
of the repairs. Writes results/r2/corrected/defect_repairs.md.

D8: realized cost per unit of turnover in the real fold-2 test env (canonical
    costs, a deterministic every-5-days rebalance between two fixed books with
    smoothing/band off so every decision trades), per asset and all-in, with the
    corrected code; the pre-fix figure is recomputed analytically from the same
    20-day return windows with the annualized vol the old code used.
D9: for folds 1-3 test windows, hysteresis and softmax beliefs with thresholds
    calibrated on the fold's TRAINING slice (fixed) vs on the test window itself
    (the pre-fix behaviour, reproduced by a train-mode fit on the test slice).

Fold 4 never loaded (fail-closed guard).
"""
import copy
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("R2_DIR", "./results/r2/corrected")
sys.path.insert(0, str(Path(__file__).parent))

import r2_common as rc  # guard first

import numpy as np
import pandas as pd

import main as main_module
from config import BeliefFormationType
from run_rc_overlay_forensics import _load_dev_data, _fold_slices
from run_validation_workstream import _base_config, _load_fold2_slices, build_env_with_features

OUT = rc.R2_DIR / "defect_repairs.md"


def d8(L):
    cfg = _base_config(1)
    sl = _load_fold2_slices(cfg)
    rc.ensure_calibrated(cfg, sl['train_aligned'])
    cfg.environment.action_smoothing_beta = None
    cfg.environment.no_trade_band = None
    env = build_env_with_features(sl['test_features'], sl['test_prices'], cfg, sl['test_aligned'], "test")
    env.reset()
    pe, tk = env.portfolio, sl['tickers']
    books = [np.array([0.3, -0.2, 0.2, -0.3, 0.2]), np.array([-0.2, 0.3, -0.3, 0.2, -0.2])]
    per_unit, per_unit_old, asset_rows, done, step = [], [], [], False, 0
    fee, slip, mult = pe.transaction_fee_rate, pe.slippage_rate, pe.slippage_vol_multiplier
    while not done:
        t = env._current_step
        prices = env._get_prices(t)
        nav = pe.compute_nav(prices)
        w0 = pe.compute_weights(prices)
        vols = {k: pe._vol_estimates[k] for k in tk}
        target = books[(step // 5) % 2]
        _, _, term, trunc, info = env.step(target.astype(np.float32))
        done = term or trunc
        if info['turnover'] > 1e-6 and step >= 25:   # vol estimate warmed up
            per_unit.append(info['transaction_cost'] / (info['turnover'] * nav))
            dw = np.abs(env.portfolio.compute_weights(prices) * 0 + target - w0)
            old = sum(dw[i] * (fee + slip + mult * vols[k] * np.sqrt(252)) for i, k in enumerate(tk)) / dw.sum()
            per_unit_old.append(old)
            for i, k in enumerate(tk):
                asset_rows.append({'asset': k, 'daily_vol': vols[k], 'new': fee + slip + mult * vols[k],
                                   'old': fee + slip + mult * vols[k] * np.sqrt(252)})
        step += 1
    a = pd.DataFrame(asset_rows).groupby('asset').mean()
    L += ["## D8 -- Slippage units (`portfolio_engine.py`)", "",
          "**Defect.** Slippage = |trade| x (slippage_rate + vol x slippage_vol_multiplier). `vol` defaulted to 0.02 "
          "(daily scale) but `mark_to_market` overwrote it with *annualized* volatility (std x sqrt(252)), inflating "
          "the volatility term ~15.9x; the term was also charged when `use_volatility_slippage=False` (affecting the "
          "validation workstream's 'flat 10 bps' V3 variant). **Fix:** daily-scale estimate (std of the last 20 daily "
          "log returns) and the term applied only when the flag is set. Parameters unchanged: fee 10 bps, base "
          "slippage 5 bps, multiplier 0.1. Regression test: `tests/test_cost_units_d8.py`.", "",
          f"**Effective all-in cost per unit of turnover** (turnover = sum |dw|, two-sided; fold-2 test env, "
          f"{len(per_unit)} rebalances):", "",
          "| | mean | min | max |", "|---|---|---|---|",
          f"| corrected | {np.mean(per_unit):.4%} | {np.min(per_unit):.4%} | {np.max(per_unit):.4%} |",
          f"| pre-fix (same trades, annualized vol) | {np.mean(per_unit_old):.4%} | {np.min(per_unit_old):.4%} | {np.max(per_unit_old):.4%} |",
          "", "| asset | mean daily vol | corrected cost / unit | pre-fix cost / unit |", "|---|---|---|---|"]
    for k, r in a.iterrows():
        L.append(f"| {k} | {r.daily_vol:.4f} | {r.new:.4%} | {r.old:.4%} |")
    L += ["", f"Per unit of two-sided turnover the corrected all-in cost is ~{np.mean(per_unit):.2%}; a round trip "
          f"(buy then later sell the same notional = 2 units) costs ~{2 * np.mean(per_unit):.2%}. "
          "**Note:** this is below the ~0.5-0.7% per unit you anticipated. With the configured parameters "
          "(10 + 5 bps + 0.1 x daily vol), 0.5-0.7% per unit would require daily vols of 3.5-5.5% on every asset. "
          "The parameters were not changed; if a higher cost level is intended, it is a parameter decision, "
          "separate from this units defect.", ""]


def _sim(cfg, tf, tp, ta, op):
    c = copy.deepcopy(cfg)
    c.regime.belief_formation = op
    env = main_module.create_trading_env(tf, tp, c, aligned_data=ta, mode="test")
    flat = np.zeros(env.action_space.shape, dtype=np.float32)
    env.reset()
    rows, done = [], False
    while not done:
        com = getattr(env.regime_model, "_committed_regime", None)
        _, _, a, b, info = env.step(flat)
        done = a or b
        bel = np.asarray(info['regime_belief'])
        rows.append({'p_bear': bel[1], 'argmax': int(np.argmax(bel)), 'committed': com,
                     'gate': rc.gate_condition(bel, info['regime_entropy'])})
    return pd.DataFrame(rows), tuple(env.regime_model._vol_thresholds)


def d9(L):
    tickers, base, features, prices, aligned, splits = _load_dev_data()
    rc.assert_no_fold4(aligned.index)
    L += ["## D9 -- Regime look-ahead (`regime_model.HeuristicRegimeModel.fit`)", "",
          "**Defect.** Test envs called `fit(test_aligned)`, which set the volatility-percentile thresholds from the "
          "whole test window. **Fix:** only a training-mode fit computes thresholds; it records them in "
          "`config.regime.calibrated_vol_thresholds` (saved with every checkpoint). Any other fit uses them frozen, "
          "reads no test values, and raises `RegimeCalibrationError` if none exist (fail-closed). Regression test: "
          "`tests/test_regime_calibration_d9.py`.", "",
          "**Effect on fold 1-3 test-window beliefs** (flat action; old = thresholds fit on the test window, "
          "new = thresholds fit on the fold's training window):", "",
          "| fold | operator | thresholds old (test-fit) | thresholds new (train-fit) | regime call agrees old vs new | mean abs change in P(Bear) | gate active share old -> new |",
          "|---|---|---|---|---|---|---|"]
    for fold in (1, 2, 3):
        sp = splits[fold - 1]
        tr_al = aligned.iloc[sp['train'][0]:sp['train'][1]]
        _, tf, tp, ta, _, _ = _fold_slices(sp, features, prices, aligned, tickers)
        for name, op in (("hysteresis", BeliefFormationType.HARD_EMA_HYSTERESIS),
                         ("softmax", BeliefFormationType.SOFTMAX_TEMPERATURE)):
            new_cfg = copy.deepcopy(base)
            new_cfg.regime.calibrated_vol_thresholds = None
            main_module.calibrate_regime(new_cfg, tr_al)
            old_cfg = copy.deepcopy(base)
            old_cfg.regime.calibrated_vol_thresholds = None
            main_module.calibrate_regime(old_cfg, ta)          # the pre-fix behaviour: calibrate on the test window
            new, th_new = _sim(new_cfg, tf, tp, ta, op)
            old, th_old = _sim(old_cfg, tf, tp, ta, op)
            key = 'committed' if name == "hysteresis" else 'argmax'
            agree = float((new[key].astype(float) == old[key].astype(float)).mean())
            L.append(f"| {fold} | {name} | ({th_old[0]:.3f}, {th_old[1]:.3f}) | ({th_new[0]:.3f}, {th_new[1]:.3f}) | "
                     f"{agree:.3f} | {float((new.p_bear - old.p_bear).abs().mean()):.4f} | "
                     f"{old.gate.mean():.3f} -> {new.gate.mean():.3f} |")
    L.append("")


def main():
    rc.R2_DIR.mkdir(parents=True, exist_ok=True)
    L = ["# Audit defects #8 and #9 -- repairs and measured effect", "",
         f"_Generated {date.today().isoformat()} by `r2_defect_repairs.py`. Fold 4 never loaded._", ""]
    d8(L)
    d9(L)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
