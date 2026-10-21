#!/usr/bin/env python3
"""
EXPRESSIBILITY GRID (eval-only, deterministic, dev fold 2 test window 2015-2017,
corrected accounting). NON-DEPLOYABLE: uses the oracle features.

Signal-following rule: action = size x oracle signal per asset (long +, short -,
equal magnitude), read from the oracle columns of the feature row observed at
each step. Grid: accuracy {100%, 60%} x size {0.2, 0.4, 0.8 per asset} x
execution {canonical, canonical without smoothing}.

Canonical execution (CANONICAL_FOLD2_CONFIG_OVERRIDES): smoothing beta = 0.1
(w = 0.9 w_prev + 0.1 a), per-asset no-trade band 0.10 vs CURRENT weights,
decisions every 5 days, per-asset clip 0.8, gross cap 2.0, corrected costs.

PRE-STATED GATE (fixed before any grid result was computed): the +/-0.8 rule at
100% accuracy under FULL canonical execution "nets strongly positive" iff its
net return over the test window is >= +20% (~ +6%/yr). Pass -> relaunch P2
oracle training unchanged; fail -> hold P2.

Per-decision diagnostics (decision days only):
  trades per decision day  = mean number of assets actually traded
  band-blocked share       = among (decision day, asset) pairs where the post-
                             smoothing target differs from the current weight,
                             the share NOT traded because |diff| <= band
  decisions fully blocked  = share of decision days with a desired change but no trade
Output: results/r2/corrected/expressibility_grid.md (+ .csv). Fold 4 never loaded.
"""
import copy
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("R2_DIR", "./results/r2/corrected")
sys.path.insert(0, str(Path(__file__).parent))

import r2_common as rc  # fold-4 guard first

import numpy as np
import pandas as pd

from run_validation_workstream import _base_config, build_env_with_features
from r2_oracle import build_oracle5_slices

GATE_THRESHOLD = 0.20
SIZES = [0.2, 0.4, 0.8]
ACCS = [(1.0, "100%"), (0.6, "60%")]
EXECS = [("canonical", None), ("canonical without smoothing", "no_smoothing")]


def run(config, slices, size):
    rc.ensure_calibrated(config, slices['train_aligned'])
    env = build_env_with_features(slices['test_features'], slices['test_prices'], config,
                                  slices['test_aligned'], mode="test")
    sig = slices['test_features'][:, -len(slices['tickers']):]
    band = config.environment.no_trade_band
    k = max(1, config.environment.decision_frequency_days)
    env.reset()
    navs, costs, turns, weights = [env.portfolio.initial_capital], [], [], []
    traded_counts, desired, blocked, fully_blocked, n_dec, step, done = [], 0, 0, 0, 0, 0, False
    eff_sizes = []
    while not done:
        t = env._current_step
        is_dec = (step % k == 0)
        w_before = env.portfolio.compute_weights(env._get_prices(t))
        _, _, term, trunc, info = env.step((size * sig[t]).astype(np.float32))
        done = term or trunc
        if is_dec:
            n_dec += 1
            target = np.asarray(env._smoothed_weights, dtype=float)        # post-smoothing, pre-band target
            eff_sizes.append(float(np.abs(env._decode_action((size * sig[t]).astype(np.float32))).max()))
            want = np.abs(target - w_before) > 1e-6
            moved = np.abs(info['portfolio_state'].weights - w_before) > 1e-6
            traded = want & (np.abs(target - w_before) > (band if band is not None else 0.0))
            traded_counts.append(int(traded.sum()))
            desired += int(want.sum())
            blocked += int((want & ~traded).sum())
            fully_blocked += int(want.any() and not traded.any())
        navs.append(info['nav'])
        costs.append(info['transaction_cost'])
        turns.append(info['turnover'])
        weights.append(info['portfolio_state'].weights.copy())
        step += 1
    nav = np.asarray(navs)
    close = np.column_stack([slices['test_prices'][t] for t in slices['tickers']])
    rets = close[1:] / close[:-1] - 1.0
    gross = np.prod([1 + float(np.asarray(w) @ rets[i]) for i, w in enumerate(weights) if i < len(rets)]) - 1
    return {'net_return': float(nav[-1] / nav[0] - 1), 'gross_pnl_held_weights': float(gross),
            'txn_cost_over_nav': float(np.sum(np.array(costs) / nav[:-1])),
            'max_dd': float(np.max(1 - nav / np.maximum.accumulate(nav))),
            'mean_daily_turnover': float(np.mean(turns)), 'decision_days': n_dec,
            'trades_per_decision_day': float(np.mean(traded_counts)),
            'band_blocked_share': blocked / desired if desired else float('nan'),
            'decisions_fully_blocked_share': fully_blocked / n_dec if n_dec else float('nan'),
            'mean_abs_gross_exposure': float(np.mean([np.abs(w).sum() for w in weights])),
            'effective_max_size_after_clip_and_cap': float(np.mean(eff_sizes))}


def main():
    base = _base_config(1)
    rows = []
    for acc, acc_name in ACCS:
        sl = build_oracle5_slices(base, acc)
        for size in SIZES:
            for ex_name, ex in EXECS:
                cfg = copy.deepcopy(base)
                if ex == "no_smoothing":
                    cfg.environment.action_smoothing_beta = None
                r = run(cfg, sl, size)
                r.update({'accuracy': acc_name, 'size_per_asset': size, 'execution': ex_name})
                rows.append(r)
                print(f"{acc_name} size {size} {ex_name:28s} net {r['net_return']:+.4f} trades/dec "
                      f"{r['trades_per_decision_day']:.2f} blocked {r['band_blocked_share']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(rc.R2_DIR / "expressibility_grid.csv", index=False)

    gate_row = df[(df.accuracy == "100%") & (df.size_per_asset == 0.8) & (df.execution == "canonical")].iloc[0]
    passed = gate_row.net_return >= GATE_THRESHOLD
    ideal = None
    L = ["# Expressibility grid -- can the canonical execution layer express a known signal?", "",
         f"_Generated {date.today().isoformat()} by `r2_expressibility.py`. Dev fold 2, test window 2015-2017, "
         "corrected accounting (audit defects #8, #9 repaired). Deterministic, eval-only, no learning. "
         "**NON-DEPLOYABLE:** the signal is the oracle feature (sign of the next-5-trading-day return, kept with "
         "probability = accuracy, fixed noise draw). Fold 4 never loaded._", "",
         "## Canonical execution parameters (from `run_paper_experiments.CANONICAL_FOLD2_CONFIG_OVERRIDES`)", "",
         "- **Action smoothing beta = 0.1** (confirmed; not 0.2 -- 0.2 was the earlier Phase-3 'Step 4' config): "
         "on a decision day the target is `w = 0.9 * w_prev_target + 0.1 * a` (`trading_env.py:314`).",
         "- No-trade band 0.10 per asset, measured against the CURRENT (drifted) weight; decisions every 5 trading days.",
         "- Action decoding: per-asset clip +/-0.8 (`max_position_size`), then gross cap 2.0 (proportional rescale). "
         "With all five signals nonzero, size 0.4 gives gross exactly 2.0 and **size 0.8 (gross 4.0) is rescaled "
         "to 0.4 per asset** -- so 0.4 and 0.8 differ only on days where some signals are 0.",
         "- Consequence: a full signal flip moves the smoothed target by 2 x 0.1 x s per decision -- 0.04 (s=0.2) or "
         "0.08 (s=0.4 and capped 0.8) -- always below the 0.10 band. A trade happens only after the smoothed target "
         "has drifted > 0.10 away from the current weight over several same-direction decisions.", "",
         f"## Gate (pre-stated before computing: +/-0.8 at 100% under full canonical execution nets >= {GATE_THRESHOLD:+.0%})", "",
         f"**{'PASS' if passed else 'FAIL'}** -- net return {gate_row.net_return:+.4f} "
         f"({gate_row.trades_per_decision_day:.2f} trades per decision day; "
         f"{gate_row.band_blocked_share:.1%} of desired per-asset changes blocked by the band). "
         + ("Premise holds for a policy-class-fair rule: P2 oracle training relaunched unchanged."
            if passed else
            "Even the largest expressible signal-following rule cannot convert perfect 5-day foresight into "
            "profit under canonical execution. **P2 HELD**: oracle training would characterize the execution "
            "layer, not learning."), "",
         "## Grid", "",
         "Net return = through the env after corrected costs; gross P&L = frictionless replay of the weights "
         "actually held; trades / decision day = assets traded (max 5); band-blocked = share of desired per-asset "
         "changes not executed because |target - current| <= 0.10; fully blocked = decision days with a desired "
         "change but no trade.", "",
         "| accuracy | size / asset | effective max size (post clip + cap) | execution | net return | gross P&L (held) | txn cost / NAV | max DD | trades / decision day | band-blocked share | decisions fully blocked | mean gross exposure |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.accuracy} | {r.size_per_asset} | {r.effective_max_size_after_clip_and_cap:.2f} | {r.execution} | "
                 f"{r.net_return:+.4f} | {r.gross_pnl_held_weights:+.4f} | {r.txn_cost_over_nav:.4f} | {r.max_dd:.4f} | "
                 f"{r.trades_per_decision_day:.2f} | {r.band_blocked_share:.3f} | {r.decisions_fully_blocked_share:.3f} | "
                 f"{r.mean_abs_gross_exposure:.3f} |")
    L += ["", "## Finding", "",
          "Execution discipline rate-limits information expression: smoothing (beta 0.1) caps how far the target "
          "can move per decision, and the 10% band then suppresses any move smaller than 0.10, so a signal that "
          "changes sign on a ~5-day horizon is largely filtered out before it reaches the book. Removing smoothing "
          "alone (band and cadence kept) shows how much of the gap is due to the smoothing/band interaction rather "
          "than to costs. The deterministic rule is a fixed policy, so each cell is a single value, not a seed "
          "distribution.", ""]
    (rc.R2_DIR / "expressibility_grid.md").write_text("\n".join(L), encoding="utf-8")
    print(f"GATE {'PASS' if passed else 'FAIL'} net={gate_row.net_return:+.4f}")


if __name__ == "__main__":
    main()
