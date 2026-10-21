#!/usr/bin/env python3
"""
R2 / E5 -- horizon-aligned oracle features (NON-DEPLOYABLE, DIAGNOSTIC ONLY).

oracle{60,100}_5d: 5 extra observation features = sign of each instrument's
NEXT-5-TRADING-DAY cumulative log return, sign(log(p[t+5]/p[t])), each entry
kept with probability `accuracy` and flipped otherwise (fixed noise draw,
run_validation_workstream.ORACLE_SEED -- the same draw the 1-day oracle60 arm
uses; accuracy 1.0 = no flips). Computed SEPARATELY within the train slice
and within the test slice (last 5 rows of each = 0), so no label ever uses a
price from across the train/test boundary.

Run directly to write the pre-launch verification + the deterministic
signal-following benchmark:  python r2_oracle.py
-> results/r2/e5/oracle_verification.md, results/r2/e5/signal_following_benchmark.csv
"""
import copy
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import r2_common as rc  # installs the fold-4 guard first

import numpy as np
import pandas as pd

from run_validation_workstream import (_base_config, _load_fold2_slices, build_env_with_features, ORACLE_SEED)

HORIZON = 5
E5_DIR = rc.R2_DIR / "e5"
NON_DEPLOYABLE = ("NON-DEPLOYABLE / DIAGNOSTIC ONLY: oracle features use future prices by construction "
                  "(sign of the next-5-day return).")


COST_NOTE = ("> **Cost-model finding (affects every arm and every existing result; NOT changed here).** "
             "`Portfolio.execute_trades` charges slippage = |trade| x vol x 0.1 (`slippage_vol_multiplier`). `vol` "
             "starts at a 0.02 default (daily scale) but `Portfolio.mark_to_market` overwrites it with ANNUALIZED "
             "volatility (std x sqrt(252), `portfolio_engine.py:323`): 0.17 (S&P) to 0.58 (WTI) on this window. "
             "Measured all-in cost per unit of turnover: 0.35% on day 0, then 3.3-3.9% thereafter -- roughly 10x the "
             "day-0 level. This units inconsistency plausibly explains why execution costs dominate results "
             "throughout the project, and why this benchmark nets a loss despite a large frictionless signal value.")


def true_forward_sign(prices_slice, tickers, horizon=HORIZON):
    close = np.column_stack([prices_slice[t] for t in tickers])
    fwd = np.zeros_like(close)
    fwd[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    return np.sign(fwd)


def oracle_signal_5d(prices_slice, tickers, n_rows, accuracy, horizon=HORIZON):
    true_sign = true_forward_sign(prices_slice, tickers, horizon)
    rng = np.random.RandomState(ORACLE_SEED)
    correct = rng.random(true_sign.shape) < accuracy
    oracle = np.where(correct, true_sign, -true_sign)
    oracle[-horizon:] = 0.0
    return oracle[:n_rows]


def build_oracle5_slices(config, accuracy):
    s = _load_fold2_slices(config)
    rc.assert_no_fold4(s['test_aligned'].index)
    tk = s['tickers']
    s['train_features'] = np.hstack([s['train_features'],
                                     oracle_signal_5d(s['train_prices'], tk, len(s['train_features']), accuracy)])
    s['test_features'] = np.hstack([s['test_features'],
                                    oracle_signal_5d(s['test_prices'], tk, len(s['test_features']), accuracy)])
    return s


# =============================================================================
# Deterministic signal-following benchmark (no learning)
# =============================================================================

def signal_following_episode(config, slices, per_asset=0.2):
    """At every step, action = per_asset * oracle signal (long +, short -, equal
    magnitude; 5 assets x 0.2 = gross 1.0 when all signals are nonzero), read
    from the oracle columns of the feature row the policy would observe at that
    step. Canonical execution semantics (smoothing, band, 5-day cadence, gross
    cap, costs) are applied by the env exactly as for the learned policy; on
    non-decision days the env ignores the action."""
    rc.ensure_calibrated(config, slices['train_aligned'])
    env = build_env_with_features(slices['test_features'], slices['test_prices'], config,
                                  slices['test_aligned'], mode="test")
    sig = slices['test_features'][:, -len(slices['tickers']):]
    obs, _ = env.reset()
    navs, turns, costs, weights, done = [env.portfolio.initial_capital], [], [], [], False
    while not done:
        t = env._current_step
        obs, _, term, trunc, info = env.step((per_asset * sig[t]).astype(np.float32))
        done = term or trunc
        navs.append(info['nav'])
        turns.append(info['turnover'])
        costs.append(info['transaction_cost'])
        weights.append(info['portfolio_state'].weights.copy())
    nav = np.asarray(navs)
    close = np.column_stack([slices['test_prices'][t] for t in slices['tickers']])
    rets = close[1:] / close[:-1] - 1.0
    gross = np.prod([1 + float(np.asarray(w) @ rets[i]) for i, w in enumerate(weights) if i < len(rets)]) - 1
    dec = np.arange(0, len(close) - 5, 5)
    ideal = np.prod(1 + (per_asset * sig[dec] * (close[dec + 5] / close[dec] - 1)).sum(axis=1)) - 1
    return {'return': float(nav[-1] / nav[0] - 1), 'max_dd': float(np.max(1 - nav / np.maximum.accumulate(nav))),
            'mean_turnover': float(np.mean(turns)), 'gross_price_pnl': float(gross),
            'txn_cost_over_nav': float(np.sum(np.array(costs) / nav[:-1])),
            'ideal_frictionless_signal_pnl': float(ideal)}


def main():
    E5_DIR.mkdir(parents=True, exist_ok=True)
    config = _base_config(1)
    base = _load_fold2_slices(config)
    rc.assert_no_fold4(base['test_aligned'].index)
    tk = base['tickers']
    L = ["# R2 / E5 -- Oracle feature verification (pre-launch)", "",
         f"_Generated {date.today().isoformat()} by `r2_oracle.py`. **{NON_DEPLOYABLE}** Fold 4 never loaded._", ""]

    # 1. alignment at decision times
    rc.ensure_calibrated(config, base['train_aligned'])
    cad_cfg = copy.deepcopy(config)   # smoothing/band off ONLY for this check, so every decision trades
    cad_cfg.environment.action_smoothing_beta = None
    cad_cfg.environment.no_trade_band = None
    env = build_env_with_features(base['test_features'], base['test_prices'], cad_cfg, base['test_aligned'], "test")
    obs, _ = env.reset()
    a = np.full(env.action_space.shape, 0.3, dtype=np.float32)
    trade_days, done, step = [], False, 0
    while not done:
        if step % 5 == 0:   # alternate the target on decision days so every decision actually trades
            a = -a
        _, _, term, trunc, info = env.step(a)
        done = term or trunc
        if info['turnover'] > 1e-9:
            trade_days.append(step)
        step += 1
    off_cadence = [d for d in trade_days if d % 5 != 0]
    L += ["## 1. Forward-label window alignment at decision times", "",
          "- Env timing (`trading_env.step`): at step t the policy observes feature row t, the decision executes at "
          "close p[t], and the position is marked over p[t]->p[t+1]->...; with `decision_frequency_days=5` the next "
          "decision is at step t+5 (step 0 of an episode is a decision day).",
          f"- Empirical cadence check (test env, k=5, smoothing/band off for this check only, target flipped each decision): trades on {len(trade_days)} steps; "
          f"steps not divisible by 5 with a trade: {len(off_cadence)} -> decisions only at t = 0, 5, 10, ...",
          "- Oracle feature at row t = sign(log(p[t+5]/p[t])) = sign of the return over exactly the holding window "
          "[t, t+5] of the decision taken at t. The next decision (t+5) observes sign(log(p[t+10]/p[t+5])): the two "
          "label windows share only the boundary price p[t+5], which is the next decision's own execution price "
          "(known at that decision), so there is no overlap leakage into the next decision's label window.",
          "- Training uses rolling episodes starting at arbitrary rows; the feature exists on every row, so whichever "
          "row is a decision day, its oracle value describes exactly that decision's 5-day holding window.", ""]

    # 2. isolation
    L += ["## 2. Isolation of future information", ""]
    for acc in (0.6, 1.0):
        o = build_oracle5_slices(config, acc)
        nb = base['train_features'].shape[1]
        same_tr = np.array_equal(o['train_features'][:, :nb], base['train_features'])
        same_te = np.array_equal(o['test_features'][:, :nb], base['test_features'])
        L.append(f"- accuracy {acc:.0%}: base feature columns identical to the canonical features (train: {same_tr}, "
                 f"test: {same_te}); oracle columns appended = {o['train_features'].shape[1] - nb}; "
                 f"label computed within each slice (last {HORIZON} rows of train and of test set to 0).")
    L += ["- Only `build_oracle5_slices` creates oracle columns, and only the E5 oracle arms and the "
          "signal-following benchmark call it. The canonical, untrained, E4 and E6 arms build features via "
          "`_load_fold2_slices` alone (see `run_r2.py`). No other feature, and no non-oracle arm, receives future "
          "information.", ""]

    # 3. realized accuracy
    L += ["## 3. Realized post-flip accuracy (rows with a defined, nonzero 5-day label)", "",
          "| arm | slice | rows | accuracy |", "|---|---|---|---|"]
    for acc, name in ((0.6, "oracle60_5d"), (1.0, "oracle100_5d")):
        for part in ("train", "test"):
            ts = true_forward_sign(base[f'{part}_prices'], tk)[:len(base[f'{part}_features'])]
            os_ = oracle_signal_5d(base[f'{part}_prices'], tk, len(base[f'{part}_features']), acc)
            m = ts != 0
            m[-HORIZON:] = False
            L.append(f"| {name} | {part} | {int(m.sum())} | {float((os_[m] == ts[m]).mean()):.4f} |")
    L.append("")

    # 4. signal-following benchmark
    rows = []
    for acc, name in ((0.6, "oracle60_5d"), (1.0, "oracle100_5d")):
        o = build_oracle5_slices(config, acc)
        r = signal_following_episode(config, o)
        r.update({'signal': name, 'execution': 'canonical (smoothing 0.1, band 0.10, k=5, cap 2.0, costs)'})
        rows.append(r)
        cfg2 = copy.deepcopy(config)
        cfg2.environment.action_smoothing_beta = None
        r2 = signal_following_episode(cfg2, o)
        r2.update({'signal': name, 'execution': 'DIAGNOSTIC: canonical but no smoothing'})
        rows.append(r2)
    bench = pd.DataFrame(rows)
    bench.to_csv(E5_DIR / "signal_following_benchmark.csv", index=False)
    L += ["## 4. Deterministic signal-following benchmark (no learning) -- NON-DEPLOYABLE", "",
          "Rule: target weight = 0.2 x signal per asset (long +, short -, equal magnitude; gross 1.0 when all five "
          "signals are nonzero), through the canonical env (smoothing, band, cadence, cap, costs). Deterministic: "
          "the oracle noise draw is fixed, so each row is a single value, not a seed distribution.", "",
          "Columns: *net return* (through the env, after costs); *gross P&L* = frictionless replay of the weights "
          "the env actually held; *txn cost* = sum of daily cost / NAV; *ideal* = frictionless P&L of holding "
          "0.2 x signal over every 5-day decision window with no execution model at all (the signal's raw value).", "",
          "| signal | execution | net return | max DD | mean daily turnover | gross P&L (held weights) | txn cost / NAV | ideal frictionless signal P&L |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in bench.iterrows():
        L.append(f"| {r.signal} | {r.execution} | {r['return']:+.4f} | {r.max_dd:.4f} | {r.mean_turnover:.4f} | "
                 f"{r.gross_price_pnl:+.4f} | {r.txn_cost_over_nav:.4f} | {r.ideal_frictionless_signal_pnl:+.4f} |")
    L += ["", COST_NOTE, ""]
    (E5_DIR / "oracle_verification.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
