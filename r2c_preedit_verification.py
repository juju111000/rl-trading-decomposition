#!/usr/bin/env python3
"""
Pre-edit verification pass (eval/analysis only; no training). Fold 4 never
loaded (r2_common guard). Writes results/r2/corrected/preedit_verification.md
and preedit_*.csv.

V-A  Post-shock regime-confidence recovery for the 2008, 2011 and 2020 shocks,
     canonical hysteresis-EMA model (softmax T=1.0 alongside), under
       new = D9: volatility thresholds calibrated on the fold's TRAINING window
       old = pre-D9: thresholds fit on the evaluated slice itself
     Shock dates fixed a priori: 2008-09-15 (Lehman), 2011-08-08 (first session
     after the US downgrade), 2020-02-19 (S&P 500 peak; as in the COVID figure).
     Definition A: trading days from the first committed-regime flip at/after the
       shock (hysteresis: committed regime; softmax: belief argmax) to the first
       day at/after that flip with normalized entropy < h (h = 0.7 and 0.5).
     Definition B: within [shock, shock + 60 trading days], length of the first
       contiguous run with normalized entropy > 0.7 (0 if none), and total days
       > 0.7 in that window.
     2008 and 2020 lie in the fold-1 / fold-3 TEST windows. 2011 lies in fold 2's
     TRAINING window (2010-2014): there old and new thresholds are both fit on
     2010-2014, so D9 cannot change it (computed for completeness).
V-C  Overshoot-cause probe: the 15 defective-era shielded grid checkpoints on
     fold 3 (2020-2023; 15/15 overshot 0.300 in the submitted grid) re-evaluated
     under corrected accounting. Eval path first checked to reproduce the
     corrected grid's stored fold-3 shielded results.
"""
import copy
import json
import os
import sys
from pathlib import Path

os.environ["R2_DIR"] = "./results/r2/corrected"   # corrected tree (as run_r2c), before r2_common reads it
sys.path.insert(0, str(Path(__file__).parent))

import r2_common as rc  # fold-4 guard first
assert Path(rc.R2_DIR).as_posix().endswith("results/r2/corrected"), rc.R2_DIR

import numpy as np
import pandas as pd
import torch

import main as main_module
from config import Config, BeliefFormationType, ShieldType
from evaluation import WalkForwardEvaluator
from evaluate_saved import load_model as load_saved_policy
from run_hardening import _run_deterministic_episode

OUT = rc.R2_DIR
TICKERS = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]
SHOCKS = [("2008", pd.Timestamp("2008-09-15"), 1, "test"), ("2011", pd.Timestamp("2011-08-08"), 2, "train"),
          ("2020", pd.Timestamp("2020-02-19"), 3, "test")]
OPS = {"hysteresis": BeliefFormationType.HARD_EMA_HYSTERESIS, "softmax": BeliefFormationType.SOFTMAX_TEMPERATURE}
OLD_GRID = Path("./results/full_grid/20260803_174817")
NEW_GRID = rc.R2_DIR / "grid"
TRAINERS = ["vanilla_ppo", "reward_shaping", "ppo_lagrangian"]


def load():
    windows = main_module.PAPER_WALK_FORWARD_WINDOWS
    dates = [d for w in windows for d in (*w['train'], *w['test'])]
    base = main_module.create_experiment_config(tickers=TICKERS, start_date=main_module._buffered_start_date(min(dates)),
                                                end_date=max(dates), total_timesteps=1, use_synthetic=False, seed=1)
    features, prices_df, _, _, aligned = main_module.load_and_prepare_data(base, use_synthetic=False)
    rc.assert_no_fold4(aligned.index)
    prices = {t: prices_df[t].values for t in TICKERS}
    splits = WalkForwardEvaluator(base).generate_calendar_splits(aligned.index, windows)
    return base, features, prices, aligned, splits


def calibrated(cfg, calib_aligned):
    cfg = copy.deepcopy(cfg)
    cfg.regime.calibrated_vol_thresholds = None
    main_module.calibrate_regime(cfg, calib_aligned)
    return cfg


def slice_(features, prices, aligned, a, b):
    return features.iloc[a:b], {t: p[a:b] for t, p in prices.items()}, aligned.iloc[a:b]


def trace(cfg, op, f, p, al):
    c = copy.deepcopy(cfg)
    c.regime.belief_formation = OPS[op]
    c.shield.shield_type = ShieldType.NONE
    env = main_module.create_trading_env(f, p, c, aligned_data=al, mode="test")
    hmax = env.regime_model.get_max_entropy()
    env.reset()
    ent, reg, done, flat = [], [], False, np.zeros(env.action_space.shape, dtype=np.float32)
    while not done:
        committed = env.regime_model._committed_regime if op == "hysteresis" else None
        _, _, term, trunc, info = env.step(flat)
        done = term or trunc
        ent.append(info['regime_entropy'] / hmax)
        reg.append(committed if op == "hysteresis" else int(np.argmax(info['regime_belief'])))
    idx = al.index[:len(ent)]
    return pd.DataFrame({'entropy': ent, 'regime': reg}, index=idx)


def stats_for(df, shock):
    pre = df[df.index < shock]
    base = pre.regime.iloc[-1] if len(pre) else df.regime.iloc[0]
    post = df[df.index >= shock]
    flips = post[post.regime != base]
    out = {'regime_before_shock': base}
    if len(flips):
        t_flip = flips.index[0]
        out['first_flip'] = t_flip.date()
        out['shock_to_flip_days'] = int((post.index < t_flip).sum())
        after = df[df.index >= t_flip]
        for h in (0.7, 0.5):
            ok = after[after.entropy < h]
            out[f'A_flip_to_H<{h}_days'] = int((after.index < ok.index[0]).sum()) if len(ok) else np.nan
    else:
        out.update({'first_flip': None, 'shock_to_flip_days': np.nan, 'A_flip_to_H<0.7_days': np.nan,
                    'A_flip_to_H<0.5_days': np.nan})
    w = post.iloc[:61]
    m = (w.entropy > 0.7).to_numpy()
    first_run, i = 0, 0
    while i < len(m) and not m[i]:
        i += 1
    start = w.index[i].date() if i < len(m) else None
    while i < len(m) and m[i]:
        first_run += 1
        i += 1
    out.update({'B_first_run_H>0.7_days': first_run, 'B_first_run_start': start,
                'B_days_H>0.7_in_60d': int(m.sum()), 'max_entropy_60d': float(w.entropy.max())})
    return out


def v_a(base, features, prices, aligned, splits):
    rows, traces = [], {}
    for name, shock, fold, part in SHOCKS:
        sp = splits[fold - 1]
        (tr0, tr1), (te0, te1) = sp['train'], sp['test']
        a, b = (te0, te1) if part == "test" else (tr0, tr1)
        f, p, al = slice_(features, prices, aligned, a, b)
        assert al.index.min() <= shock <= al.index.max()
        for tag, calib in (("new (D9: train-window thresholds)", aligned.iloc[tr0:tr1]),
                           ("old (pre-D9: fit on evaluated slice)", al)):
            cfg = calibrated(base, calib)
            for op in OPS:
                df = trace(cfg, op, f, p, al)
                traces[(name, tag[:3], op)] = df
                rows.append({'shock': name, 'shock_date': shock.date(), 'slice': f"fold {fold} {part}",
                             'thresholds': tag, 'operator': op, **stats_for(df, shock)})
    R = pd.DataFrame(rows)
    # cross-check new test-window traces against the committed E3 daily file
    daily = pd.read_csv(OUT / "belief_validation_daily.csv.gz", parse_dates=['date'])
    for name, shock, fold, part in SHOCKS:
        if part != "test":
            continue
        for op in OPS:
            d = daily[(daily.fold == fold) & (daily.operator == op)].set_index('date').entropy_norm
            t = traces[(name, "new", op)].entropy
            common = d.index.intersection(t.index)
            assert len(common) > 500 and np.allclose(d.loc[common], t.loc[common], atol=1e-9), (name, op)
    R.to_csv(OUT / "preedit_shock_recovery.csv", index=False)
    return R


def v_c(base, features, prices, aligned, splits):
    sp = splits[2]
    (tr0, tr1), (te0, te1) = sp['train'], sp['test']
    f, p, al = slice_(features, prices, aligned, te0, te1)
    tra = aligned.iloc[tr0:tr1]
    rf = base.evaluation.risk_free_rate
    old_csv = pd.read_csv(OLD_GRID / "results.csv")
    new_csv = pd.read_csv(NEW_GRID / "results.csv")

    def evaluate(d, need_calib):
        cfg = Config.from_json(str(d / "config.json"))
        if need_calib or cfg.regime.calibrated_vol_thresholds is None:
            cfg = calibrated(cfg, tra)
        env = main_module.create_trading_env(f, p, cfg, aligned_data=al, mode="test")
        pol = load_saved_policy(d / "final_model.pt", env.observation_space.shape[0], len(TICKERS), cfg)
        return _run_deterministic_episode(env, pol, TICKERS, p, rf)

    # path check: corrected-era checkpoints must reproduce their stored corrected results
    chk = []
    for s in range(1, 6):
        for t in TRAINERS:
            r = evaluate(NEW_GRID / f"seed_{s}" / "fold_3" / "uncertainty_aware" / t, need_calib=False)
            st = new_csv[(new_csv.window == "2020-2023") & (new_csv.shield == "uncertainty_aware") &
                         (new_csv.method == t) & (new_csv.seed == s)].iloc[0]
            chk.append(max(abs(r['return'] - st['return']), abs(r['max_dd'] - st['max_dd'])))
    rows = []
    for s in range(1, 6):
        for t in TRAINERS:
            d = OLD_GRID / f"seed_{s}" / "fold_3" / "uncertainty_aware" / t
            r = evaluate(d, need_calib=True)
            st = old_csv[(old_csv.window == "2020-2023") & (old_csv.shield == "uncertainty_aware") &
                         (old_csv.method == t) & (old_csv.seed == s)].iloc[0]
            rows.append({'trainer': t, 'seed': s, 'max_dd_submitted_eval': st['max_dd'],
                         'max_dd_corrected_eval': r['max_dd'], 'return_submitted_eval': st['return'],
                         'return_corrected_eval': r['return']})
    V = pd.DataFrame(rows)
    V.to_csv(OUT / "preedit_overshoot_probe.csv", index=False)
    return V, max(chk)


def main():
    base, features, prices, aligned, splits = load()
    R = v_a(base, features, prices, aligned, splits)
    V, chk = v_c(base, features, prices, aligned, splits)
    L = ["# Pre-edit verification (V-A, V-C) -- corrected accounting", "",
         "> Generated by `r2c_preedit_verification.py`. Eval/analysis only; fold 4 never loaded.", "",
         "## V-A. Post-shock regime-confidence recovery", "",
         "**Provenance of the manuscript's '0-4 trading days' figure:** no producing script exists in the repo or "
         "its history (see `plot_entropy_shock_2020._verify_resolution_stat_definition`); its definition and "
         "shock dates cannot be recovered, so it cannot be recomputed as-is. Below, two explicit definitions, "
         "computed with identical code under the old (pre-D9) and new (D9) threshold calibration.", "",
         "Shock dates (fixed a priori): 2008-09-15, 2011-08-08, 2020-02-19. **A** = trading days from the first "
         "committed-regime flip at/after the shock to the first day with normalized entropy < h. **B** = length of "
         "the first contiguous run of entropy > 0.7 within 60 trading days of the shock (0 = none), plus total "
         "such days. Softmax uses its belief argmax as the 'regime'. 2011 lies in fold 2's training window, where "
         "D9 leaves thresholds unchanged. The D9 traces for 2008/2020 match the committed E3 daily beliefs "
         "(`belief_validation_daily.csv.gz`) exactly (asserted).", "",
         "| shock | operator | thresholds | regime before | first flip | shock→flip (d) | A: flip→H<0.7 (d) | "
         "A: flip→H<0.5 (d) | B: first H>0.7 run (d) | B: run start | B: days H>0.7 in 60d | max H (60d) |",
         "|---|---|---|---|---|---:|---:|---:|---:|---|---:|---:|"]
    names = {0: "Bull", 1: "Bear", 2: "Sideways"}
    def g(r, k):
        v = r[k]
        return "none" if v is None or (isinstance(v, float) and np.isnan(v)) else (int(v) if isinstance(v, float) else v)

    for _, r in R.iterrows():
        L.append(f"| {r['shock']} | {r['operator']} | {r['thresholds']} | "
                 f"{names.get(r['regime_before_shock'], r['regime_before_shock'])} | {r['first_flip'] or 'none'} | "
                 f"{g(r, 'shock_to_flip_days')} | {g(r, 'A_flip_to_H<0.7_days')} | {g(r, 'A_flip_to_H<0.5_days')} | "
                 f"{r['B_first_run_H>0.7_days']} | {r['B_first_run_start'] or '—'} | {r['B_days_H>0.7_in_60d']} | "
                 f"{r['max_entropy_60d']:.3f} |")
    hy = R[R.operator == "hysteresis"]
    parts = []
    for s in ("2008", "2011", "2020"):
        o = hy[(hy.shock == s) & hy.thresholds.str.startswith('old')].iloc[0]
        n = hy[(hy.shock == s) & hy.thresholds.str.startswith('new')].iloc[0]
        parts.append(f"{s}: A(H<0.7) {g(o, 'A_flip_to_H<0.7_days')} -> {g(n, 'A_flip_to_H<0.7_days')} d, "
                     f"B {o['B_first_run_H>0.7_days']} -> {n['B_first_run_H>0.7_days']} d")
    L += ["", "Old -> new, hysteresis (canonical): " + "; ".join(parts) + ".", ""]
    over_old = int((V.max_dd_submitted_eval > 0.300 + 1e-9).sum())
    over_new = int((V.max_dd_corrected_eval > 0.300 + 1e-9).sum())
    L += ["## V-C. Overshoot-cause probe (fold 3, 2020-2023)", "",
          f"The 15 defective-era shielded checkpoints (`{OLD_GRID.as_posix()}`, trained under the defective "
          "accounting) re-evaluated under corrected accounting (D8 costs, D9 train-window thresholds); nothing "
          f"retrained. Eval-path check: the same code reproduces all 15 corrected-grid fold-3 shielded runs "
          f"(max |Δ| in return/max DD = {chk:.1e}).", "",
          "| trainer | seed | max DD, submitted eval | max DD, corrected eval | return, submitted | return, corrected |",
          "|---|---|---:|---:|---:|---:|"]
    for r in V.itertuples():
        L.append(f"| {r.trainer} | {r.seed} | {r.max_dd_submitted_eval:.6f} | {r.max_dd_corrected_eval:.6f} | "
                 f"{r.return_submitted_eval:+.4f} | {r.return_corrected_eval:+.4f} |")
    L += ["", f"Overshoot (> 0.300): submitted evaluation {over_old}/15; corrected evaluation of the SAME "
          f"checkpoints {over_new}/15 (max {V.max_dd_corrected_eval.max():.6f}).", ""]
    (OUT / "preedit_verification.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
