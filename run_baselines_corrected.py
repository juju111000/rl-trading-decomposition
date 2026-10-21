#!/usr/bin/env python3
"""
Re-run the 7 deterministic baselines (run_baselines.build_strategies, unchanged)
on folds 1-3 under corrected accounting (audit defects #8 slippage units, #9
regime look-ahead), plus EW-matched-execution. Eval-only, deterministic; no
training. Fold 4 never loaded (r2_common guard + windows end 2023-12-31).

Differences from run_baselines.run_baselines (which produced the old,
defective-cost results/baselines/baselines_results.csv, frozen 2026-08-05):
  * the env's regime thresholds are calibrated on each fold's TRAINING slice
    (main.calibrate_regime) -- required by the #9 fix, the test env no longer
    calibrates itself;
  * extra columns: txn cost / NAV (sum of daily cost / NAV) and gross P&L
    (frictionless replay of held weights), as in the hardening reports.
Everything else -- strategies, parameters, env config (unshielded, no
smoothing/band/cadence: each baseline runs its own rebalance rule), lookback,
metric code -- is identical.

EW-matched-execution: constant equal-weight target (0.2 per asset, gross 1.0)
pushed through the RL agents' canonical execution layer
(CANONICAL_FOLD2_CONFIG_OVERRIDES: smoothing beta 0.1, band 0.10, 5-day
decisions, gross cap 2.0) -- the passive benchmark under exactly the execution
constraints the learned policies face.

Writes results/r2/corrected/baselines_corrected.{csv,md}.
"""
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("R2_DIR", "./results/r2/corrected")
sys.path.insert(0, str(Path(__file__).parent))

import r2_common as rc  # fold-4 guard first

import numpy as np
import pandas as pd

import main as main_module
from config import ShieldType
from evaluation import WalkForwardEvaluator
from run_baselines import PriceHistory, build_strategies, compute_return_metrics, _window_label
from run_paper_experiments import CANONICAL_FOLD2_CONFIG_OVERRIDES
from run_validation_workstream import _apply_overrides

TICKERS = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]
OLD_CSV = Path("./results/baselines/baselines_results.csv")
OUT_CSV = rc.R2_DIR / "baselines_corrected.csv"
OUT_MD = rc.R2_DIR / "baselines_corrected.md"
LOOKBACK_DAYS = 260
EW_MATCHED = "ew_matched_execution"
EXEC_OVERRIDES = {k: v for k, v in CANONICAL_FOLD2_CONFIG_OVERRIDES.items()
                  if k.startswith(("environment.", "action_space."))}


class ConstantEW:
    def __init__(self, n):
        self.a = np.full(n, 1.0 / n, dtype=np.float32)

    def get_action(self, obs, info):
        return self.a


def episode(env, strategy, close, rf):
    obs, info = env.reset()
    navs, costs, turns, weights, dd, done = [env.portfolio.initial_capital], [], [], [], [], False
    while not done:
        action = strategy.get_action(obs, info)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        navs.append(info.get('nav', navs[-1]))
        costs.append(info.get('transaction_cost', 0.0))
        turns.append(info.get('turnover', 0.0))
        dd.append(info.get('cost_vec', np.zeros(3))[0] > 0)
        weights.append(info['portfolio_state'].weights.copy())
    row = compute_return_metrics(navs, risk_free_rate=rf)
    nav = np.asarray(navs)
    rets = close[1:] / close[:-1] - 1.0
    row.update({'violation_rate': float(np.mean(dd)), 'turnover': float(np.mean(turns)),
                'txn_cost_nav': float(np.sum(np.array(costs) / nav[:-1])),
                'gross_pnl': float(np.prod([1 + float(w @ rets[i]) for i, w in enumerate(weights) if i < len(rets)]) - 1)})
    return row


def main():
    windows = main_module.PAPER_WALK_FORWARD_WINDOWS
    all_dates = [d for w in windows for d in (*w['train'], *w['test'])]
    start, end = main_module._buffered_start_date(min(all_dates)), max(all_dates)
    base = main_module.create_experiment_config(tickers=TICKERS, start_date=start, end_date=end,
                                                total_timesteps=1, use_synthetic=False, seed=1)
    base.shield.shield_type = ShieldType.NONE
    features, prices_df, _, _, aligned = main_module.load_and_prepare_data(base, use_synthetic=False)
    rc.assert_no_fold4(aligned.index)
    prices = {t: prices_df[t].values for t in TICKERS}
    splits = WalkForwardEvaluator(base).generate_calendar_splits(aligned.index, windows)

    rows = []
    for fold, sp in enumerate(splits, 1):
        (tr0, tr1), (te0, te1) = sp['train'], sp['test']
        label = _window_label(sp['test_dates'])
        fold_cfg = copy.deepcopy(base)
        fold_cfg.regime.calibrated_vol_thresholds = None
        main_module.calibrate_regime(fold_cfg, aligned.iloc[tr0:tr1])     # audit defect #9
        tf, ta = features.iloc[te0:te1], aligned.iloc[te0:te1]
        tp = {t: p[te0:te1] for t, p in prices.items()}
        close = np.column_stack([tp[t] for t in TICKERS])
        lb0 = max(0, te0 - LOOKBACK_DAYS)
        ph = PriceHistory({t: p[lb0:te1] for t, p in prices.items()}, lookback_len=te0 - lb0, tickers=TICKERS)
        strategies = build_strategies(len(TICKERS), fold_cfg, ph, aligned.iloc[tr0:tr1])
        for name, strat in strategies.items():
            cfg = copy.deepcopy(fold_cfg)
            env = main_module.create_trading_env(tf, tp, cfg, aligned_data=ta, mode="test")
            r = episode(env, strat, close, cfg.evaluation.risk_free_rate)
            r.update({'strategy': name, 'window': label, 'fold': fold,
                      'note': ('HMM fit failed -> equal-weight fallback' if name == 'hmm_regime'
                               and not getattr(strat, '_fitted', True) else '')})
            rows.append(r)
            print(f"fold {fold} {name:26s} ret {r['return']:+.4f} cost/NAV {r['txn_cost_nav']:.4f}", flush=True)
        cfg = _apply_overrides(copy.deepcopy(fold_cfg), EXEC_OVERRIDES)
        env = main_module.create_trading_env(tf, tp, cfg, aligned_data=ta, mode="test")
        r = episode(env, ConstantEW(len(TICKERS)), close, cfg.evaluation.risk_free_rate)
        r.update({'strategy': EW_MATCHED, 'window': label, 'fold': fold, 'note': ''})
        rows.append(r)
        print(f"fold {fold} {EW_MATCHED:26s} ret {r['return']:+.4f} cost/NAV {r['txn_cost_nav']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    old = pd.read_csv(OLD_CSV).set_index(['strategy', 'window'])

    L = ["# Deterministic baselines -- corrected costs (folds 1-3)", "",
         "> Corrected accounting (audit defects #8 slippage units, #9 regime look-ahead). Eval-only, deterministic, "
         "single value per cell (no seeds). Fold 4 never loaded. Generated by `run_baselines_corrected.py`; raw rows in "
         f"`{OUT_CSV.as_posix()}`. Old values: `{OLD_CSV.as_posix()}` (frozen 2026-08-05, **defective costs**).", "",
         "- The 7 baselines are `run_baselines.build_strategies` unchanged (unshielded; each runs its own rebalance "
         "rule, no smoothing / band / cadence from the RL execution layer). Only change in procedure: env regime "
         "thresholds calibrated on each fold's training slice, as the #9 fix requires.",
         f"- **{EW_MATCHED}** (new, no old value): constant 0.2-per-asset target through the RL agents' canonical "
         "execution layer (smoothing β 0.1, band 0.10, 5-day decisions, gross cap 2.0). Starting flat, smoothing "
         "needs ~7 decisions (~35 trading days) before the target clears the band.",
         "- cost/NAV = sum of daily transaction cost / NAV; gross P&L = frictionless replay of held weights. The "
         "old file has neither column.", ""]
    for fold in (1, 2, 3):
        sub = df[df.fold == fold]
        label = sub.window.iloc[0]
        L += [f"## Fold {fold} ({label})", "",
              "| strategy | net return | old net | Δ | Sharpe | old Sharpe | max DD | old max DD | turnover/day | old turnover | cost/NAV | gross P&L | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            if (r.strategy, label) in old.index:
                o = old.loc[(r.strategy, label)]
                oc = (f"{o['return']:+.4f} | {r['return'] - o['return']:+.4f} | {r.sharpe:+.3f} | {o.sharpe:+.3f} | "
                      f"{r.max_dd:.3f} | {o.max_dd:.3f} | {r.turnover:.4f} | {o.turnover:.4f}")
            else:
                oc = f"n/a | n/a | {r.sharpe:+.3f} | n/a | {r.max_dd:.3f} | n/a | {r.turnover:.4f} | n/a"
            L.append(f"| {r.strategy} | {r['return']:+.4f} | {oc} | {r.txn_cost_nav:.4f} | {r.gross_pnl:+.4f} | {r.note} |")
        L.append("")
    L += ["## Mean over folds 1-3", "",
          "| strategy | mean net return | old mean net | mean Sharpe | old mean Sharpe | mean cost/NAV |",
          "|---|---|---|---|---|---|"]
    old_r = old.reset_index()
    for name, g in df.groupby('strategy', sort=False):
        og = old_r[old_r.strategy == name]
        on = f"{og['return'].mean():+.4f}" if len(og) == 3 else "n/a"
        osr = f"{og.sharpe.mean():+.3f}" if len(og) == 3 else "n/a"
        L.append(f"| {name} | {g['return'].mean():+.4f} | {on} | {g.sharpe.mean():+.3f} | {osr} | {g.txn_cost_nav.mean():.4f} |")
    L.append("")

    p1 = pd.read_csv(rc.R2_DIR / "hardening" / "hardening_results.csv")
    f2 = df[df.fold == 2].set_index('strategy')
    L += ["## Fold-2 reference: P1 arms on the same test window (2015-2017, 780 steps)", "",
          "P1 means over 20 seeds (corrected costs, `p1_stats_interim.md`) beside the two equal-weight benchmarks. "
          "Oracle arms are NON-DEPLOYABLE.", "",
          "| arm | net return | gross P&L | cost/NAV |", "|---|---|---|---|"]
    for arm in ("untrained", "canonical", "oracle60", "oracle75"):
        a = p1[p1.arm == arm]
        L.append(f"| P1 {arm} (mean of {len(a)}) | {a['return'].mean():+.4f} | {a.gross_price_pnl.mean():+.4f} | "
                 f"{a.txn_cost.mean():.4f} |")
    for name in (EW_MATCHED, 'equal_weight_k5_band010'):
        L.append(f"| {name} | {f2.loc[name, 'return']:+.4f} | {f2.loc[name, 'gross_pnl']:+.4f} | "
                 f"{f2.loc[name, 'txn_cost_nav']:.4f} |")
    L += ["", "## Notes", "",
          "- `hmm_regime` fold 1: the HMM fit fails ('covars' must be symmetric, positive-definite), so the strategy "
          "falls back to equal-weight monthly -- identical to `equal_weight_monthly` in both old and new runs. "
          "hmmlearn 'Model is not converging' warnings on stderr during the other folds' fits are benign.",
          "- `equal_weight_k5_band010` fold 2 is bit-identical old vs new: in that window the band never triggers "
          "after the day-0 purchase (cost/NAV = 0.0035 = the day-0 rate, which the #8 defect did not affect), so "
          "the cost defect never bit.",
          "- Turnover is essentially unchanged old vs new for every strategy (rules are price-driven, not "
          "cost-driven); the return differences are the cost correction (plus, negligibly, the #9 env change, "
          "which no baseline reads).", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
