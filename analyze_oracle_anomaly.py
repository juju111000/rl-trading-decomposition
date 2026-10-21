#!/usr/bin/env python3
"""
Oracle75 < oracle60 investigation (P1, corrected costs). Analysis + deterministic
eval-only replays of existing P1 checkpoints; no training, R2C batch untouched,
fold 4 never loaded (r2_common guard). NON-DEPLOYABLE: oracle arms see future
prices by construction. Writes results/r2/corrected/oracle_anomaly.md.

1. Holm correction across the 8 interim tests (4 comparisons x {net, gross}),
   applied separately to the Welch and the Mann-Whitney p-values; a rejection
   survives only if BOTH adjusted p < 0.05 (same both-tests rule as the interim
   report).
2. Per-seed paired oracle75 - oracle60 deltas on matched seeds 1-20.
3. Mechanism probe: replay each oracle checkpoint deterministically on the fold-2
   test window with per-decision instrumentation. Primary probe = 3 matched seeds
   chosen by a fixed rule (seeds at the min, median and max of the paired net
   delta); the same metrics on all 20 seeds are reported as a robustness check.
   Every replay must reproduce the stored eval_result net return, else abort.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("R2_DIR", "./results/r2/corrected")
sys.path.insert(0, str(Path(__file__).parent))

import r2_common as rc  # fold-4 guard first

import json
import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests

import run_hardening as rh
from config import Config
from evaluate_saved import load_model as load_saved_policy

HARD = rc.R2_DIR / "hardening"
CSV = HARD / "hardening_results.csv"
OUT = rc.R2_DIR / "oracle_anomaly.md"
OUT_CSV = rc.R2_DIR / "oracle_anomaly_probe.csv"
ARMS = {"oracle60": 0.6, "oracle75": 0.75}
SEEDS = list(range(1, 21))
COMPARISONS = [("canonical", "untrained"), ("canonical", "oracle60"),
               ("canonical", "oracle75"), ("oracle75", "oracle60")]
METRICS = [("return", "net"), ("gross_price_pnl", "gross")]
EPS = 1e-6


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def vals(df, arm, metric):
    return df[df.arm == arm].set_index("seed").loc[SEEDS, metric].to_numpy(dtype=float)


# ----------------------------------------------------------------------------- 3. instrumented replay

def replay(arm, seed, slices):
    ckpt_dir = HARD / f"{arm}_training" / f"{arm}_seed{seed}"
    cfg = Config.from_json(str(ckpt_dir / "config.json"))
    assert cfg.regime.calibrated_vol_thresholds is not None
    env = rh.build_env_with_features(slices['test_features'], slices['test_prices'], cfg,
                                     slices['test_aligned'], mode="test")
    n = len(slices['tickers'])
    policy = load_saved_policy(ckpt_dir / "final_model.pt", env.observation_space.shape[0], n, cfg)
    band = cfg.environment.no_trade_band
    k = max(1, cfg.environment.decision_frequency_days)
    oracle = slices['test_features'][:, -n:]
    close = np.column_stack([slices['test_prices'][t] for t in slices['tickers']])

    obs, _ = env.reset()
    navs, step, done = [env.portfolio.initial_capital], 0, False
    dec = []          # per decision day: t, intended a, smoothed target, w_before, w_after
    held = []
    while not done:
        t = env._current_step
        is_dec = step % k == 0
        w_before = env.portfolio.compute_weights(env._get_prices(t)).copy()
        with torch.no_grad():
            action, _, _ = policy.get_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True)
        a_raw = action.cpu().numpy().squeeze()
        a = env._decode_action(a_raw).astype(float)          # intended target: post clip/cap, PRE-smoothing
        obs, _, term, trunc, info = env.step(a_raw)
        done = term or trunc
        w_after = info['portfolio_state'].weights.copy()
        if is_dec:
            dec.append({'t': t, 'a': a, 'target': np.asarray(env._smoothed_weights, float).copy(),
                        'w_before': w_before, 'w_after': w_after})
        held.append(w_after)
        navs.append(info['nav'])
        step += 1

    nav = np.asarray(navs)
    net = float(nav[-1] / nav[0] - 1)
    stored = json.loads((HARD / arm / f"seed_{seed}" / "eval_result.json").read_text())['return']
    assert abs(net - stored) < 1e-9, f"{arm} seed {seed}: replay {net} != stored {stored}"

    rets1 = close[1:] / close[:-1] - 1.0
    A = np.array([d['a'] for d in dec])
    TGT = np.array([d['target'] for d in dec])
    WB = np.array([d['w_before'] for d in dec])
    WA = np.array([d['w_after'] for d in dec])
    T = np.array([d['t'] for d in dec])
    O = oracle[T]

    # action vs oracle feature (decision days, nonzero oracle)
    m = O != 0
    corr_ao = float(np.corrcoef(A[m], O[m])[0, 1])
    agree = float(np.mean(np.sign(A[m]) == np.sign(O[m])))
    # pre-smoothing action change and sign flips between consecutive decisions
    dA = np.abs(np.diff(A, axis=0))
    nz = (np.abs(A[1:]) > EPS) & (np.abs(A[:-1]) > EPS)
    flip = float(np.mean(np.sign(A[1:])[nz] != np.sign(A[:-1])[nz]))
    tgt_nz = (np.abs(TGT[1:]) > EPS) & (np.abs(TGT[:-1]) > EPS)
    tgt_flip = float(np.mean(np.sign(TGT[1:])[tgt_nz] != np.sign(TGT[:-1])[tgt_nz]))
    # band truncation: post-smoothing target vs current weight
    gap = np.abs(TGT - WB)
    want = gap > EPS
    traded = want & (gap > band)
    blocked = float((want & ~traded).sum() / want.sum())
    # what share of the INTENDED move (a - w_before) was executed
    intended_move = np.abs(A - WB).sum()
    executed_move = np.abs(WA - WB).sum()
    exec_ratio = float(executed_move / intended_move)
    # timing: frictionless P&L of intended targets held over each k-day window vs held weights
    fwd_k = np.array([close[min(t + k, len(close) - 1)] / close[t] - 1 for t in T])
    intended_pnl = float(np.prod(1 + (A * fwd_k).sum(axis=1)) - 1)
    held_gross = float(np.prod([1 + float(w @ rets1[i]) for i, w in enumerate(held) if i < len(rets1)]) - 1)
    fwd_1 = np.array([rets1[t] if t < len(rets1) else np.zeros(len(slices['tickers'])) for t in T])
    corr_a_fwd1 = float(np.corrcoef(A.ravel(), fwd_1.ravel())[0, 1])
    dW = (WA - WB)
    tr = np.abs(dW) > EPS
    corr_trade_fwdk = float(np.corrcoef(dW[tr], fwd_k[tr])[0, 1]) if tr.sum() > 2 else float('nan')
    return {'arm': arm, 'seed': seed, 'net_return': net, 'gross_held': held_gross,
            'corr_action_oracle': corr_ao, 'sign_agree_action_oracle': agree,
            'mean_abs_action_change': float(dA.mean()), 'action_sign_flip_rate': flip,
            'smoothed_target_sign_flip_rate': tgt_flip, 'band_blocked_share': blocked,
            'executed_over_intended_move': exec_ratio, 'intended_frictionless_pnl': intended_pnl,
            'expression_gap': intended_pnl - held_gross, 'corr_action_fwd1d': corr_a_fwd1,
            'corr_executed_trade_fwd5d': corr_trade_fwdk,
            'mean_net_exposure': float(np.mean([w.sum() for w in held])),
            'mean_gross_exposure': float(np.mean([np.abs(w).sum() for w in held])),
            'n_decisions': len(dec)}


PROBE_COLS = [
    ('corr_action_oracle', 'corr(intended action, oracle feature)', '+.3f'),
    ('sign_agree_action_oracle', 'sign agreement action vs oracle', '.3f'),
    ('mean_abs_action_change', 'mean |Δ intended action| per asset per decision (pre-smoothing)', '.4f'),
    ('action_sign_flip_rate', 'intended-action sign-flip rate', '.3f'),
    ('smoothed_target_sign_flip_rate', 'smoothed-target sign-flip rate', '.3f'),
    ('band_blocked_share', 'share of desired changes blocked by band', '.3f'),
    ('executed_over_intended_move', 'executed / intended move (L1)', '.3f'),
    ('intended_frictionless_pnl', 'intended-target frictionless P&L (5d holds)', '+.4f'),
    ('gross_held', 'realized gross P&L (held weights)', '+.4f'),
    ('expression_gap', 'expression gap (intended − realized gross)', '+.4f'),
    ('corr_action_fwd1d', 'corr(intended action, next-day return)', '+.4f'),
    ('corr_executed_trade_fwd5d', 'corr(executed trade, next-5d return)', '+.4f'),
    ('mean_net_exposure', 'mean net exposure Σw (held)', '+.3f'),
    ('mean_gross_exposure', 'mean gross exposure Σ|w| (held)', '.3f'),
    ('net_return', 'net return (replay = stored)', '+.4f'),
]


FINDINGS = """## Findings (written against the numbers above, run of 2026-10-14)

1. **Multiplicity.** No interim rejection survives Holm across the 8-test family: canonical<untrained net
   (Welch 0.064 / MW 0.124 adjusted), oracle75<oracle60 net (0.075 / 0.145) and gross (0.120 / 0.202) all
   fall back to non-significant under the independent-samples tests.
2. **The oracle75 deficit is broad, not a 2-3 seed artifact.** On matched seeds it is negative for 15/20
   seeds (net and gross), the paired tests give p ~ 0.002 (net) / 0.005 (gross), and it remains significant
   after dropping the 3 most negative seeds (mean -0.059, Wilcoxon p 0.017 net). The paired analysis is a
   separate, more powerful design than the Holm family in (1) and was not pre-specified; it should be
   reported as such.
3. **The hypothesized mechanism is NOT supported.** Neither arm meaningfully uses the oracle feature:
   corr(intended action, oracle) ~ +0.01 to +0.03 and sign agreement ~ 0.51 in both arms (a 1-day signal
   observed once per 5-day decision). Intended-action change magnitude, sign-flip rates and the band-blocked
   share (~0.89 in both) do not differ between arms (all Wilcoxon p > 0.16) and do not track the per-seed
   net delta. Only ~6% of the intended move is executed in either arm. Stronger signal -> larger reversals
   -> more truncation -> worse timing fails at its first link: the stronger signal is not reaching the
   action.
4. **What does move with the delta.** The realized gross delta tracks the net delta almost one-for-one
   (rho 0.98) -- the gap is a gross-P&L (position) effect, not costs. Per seed, net held exposure partially
   explains which arm wins (rho +0.48, p 0.03), but there is no systematic exposure difference between arms
   (mean delta +0.004). The systematic oracle75 < oracle60 gap therefore remains unexplained by the metrics
   probed here; with the oracle barely used, the most likely route is that the different oracle columns
   alter the training trajectory (what the policy learns from the other features / per-asset allocation),
   not signal expression. Per-asset allocation is the natural next probe.
"""


def main():
    df = pd.read_csv(CSV)
    L = ["# Oracle75 < oracle60 investigation -- corrected costs (P1, fold 2)", "",
         "> **NON-DEPLOYABLE / DIAGNOSTIC.** Oracle arms observe sign(next-day return) per asset, kept with "
         "probability 0.60 / 0.75 (same fixed noise draw, so oracle75's correct entries are a superset of "
         "oracle60's). Analysis + deterministic eval-only replays of the existing P1 checkpoints; no training, "
         "R2C batch untouched, fold 4 never loaded. Generated by `analyze_oracle_anomaly.py`.", ""]

    # ---- 1. Holm
    rows = []
    for metric, mlab in METRICS:
        for a, b in COMPARISONS:
            x, y = vals(df, a, metric), vals(df, b, metric)
            rows.append({'test': f"{a} vs {b}", 'metric': mlab, 'diff': x.mean() - y.mean(),
                         'welch_p': stats.ttest_ind(x, y, equal_var=False).pvalue,
                         'mw_p': stats.mannwhitneyu(x, y, alternative="two-sided").pvalue})
    H = pd.DataFrame(rows)
    H['welch_holm'] = multipletests(H.welch_p, alpha=0.05, method='holm')[1]
    H['mw_holm'] = multipletests(H.mw_p, alpha=0.05, method='holm')[1]
    L += ["## 1. Holm correction across the 8 interim tests", "",
          "Holm-Bonferroni (family = 8 tests: 4 comparisons x {net, gross}), applied separately to the Welch and "
          "the Mann-Whitney p-values. Decision rule unchanged from the interim report: reject only if BOTH tests "
          "reject.", "",
          "| comparison | metric | mean diff | Welch p | Welch p (Holm) | MW p | MW p (Holm) | unadjusted | Holm |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in H.iterrows():
        raw = "reject" if (r.welch_p < .05 and r.mw_p < .05) else ("mixed" if min(r.welch_p, r.mw_p) < .05 else "—")
        adj = "**survives**" if (r.welch_holm < .05 and r.mw_holm < .05) else (
            "mixed" if min(r.welch_holm, r.mw_holm) < .05 else "—")
        L.append(f"| {r.test} | {r.metric} | {r['diff']:+.4f} | {fmt_p(r.welch_p)} | {fmt_p(r.welch_holm)} | "
                 f"{fmt_p(r.mw_p)} | {fmt_p(r.mw_holm)} | {raw} | {adj} |")
    L.append("")

    # ---- 2. paired oracle75 - oracle60
    o60 = df[df.arm == "oracle60"].set_index("seed").loc[SEEDS]
    o75 = df[df.arm == "oracle75"].set_index("seed").loc[SEEDS]
    L += ["## 2. Per-seed paired deltas oracle75 − oracle60 (matched seeds 1-20)", "",
          "Same seed, same `_base_config`, same code path (`run_hardening.process_oracle`) and the same oracle noise "
          "draw; the arms differ only in the flip threshold. Initial-weight identity across the two arms was not "
          "separately tensor-checked (both have the same obs_dim and the same seed_everything -> network path).", "",
          "| metric | mean Δ | median Δ | std Δ | 95% CI | paired t p | Wilcoxon p | Δ < 0 | min / max Δ | mean Δ excl. 3 most negative | Wilcoxon p excl. 3 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    deltas = {}
    for metric, mlab in METRICS:
        d = (o75[metric] - o60[metric]).to_numpy()
        deltas[mlab] = d
        m, sd = d.mean(), d.std(ddof=1)
        h = stats.t.ppf(.975, len(d) - 1) * sd / np.sqrt(len(d))
        trim = np.sort(d)[3:]
        L.append(f"| {mlab} | {m:+.4f} | {np.median(d):+.4f} | {sd:.4f} | [{m - h:+.4f}, {m + h:+.4f}] | "
                 f"{fmt_p(stats.ttest_1samp(d, 0).pvalue)} | {fmt_p(stats.wilcoxon(d).pvalue)} | {int((d < 0).sum())} / 20 | "
                 f"{d.min():+.4f} / {d.max():+.4f} | {trim.mean():+.4f} | {fmt_p(stats.wilcoxon(trim).pvalue)} |")
    L += ["", "| seed | oracle60 net | oracle75 net | Δ net | oracle60 gross | oracle75 gross | Δ gross |",
          "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(SEEDS):
        L.append(f"| {s} | {o60.loc[s, 'return']:+.4f} | {o75.loc[s, 'return']:+.4f} | {deltas['net'][i]:+.4f} | "
                 f"{o60.loc[s, 'gross_price_pnl']:+.4f} | {o75.loc[s, 'gross_price_pnl']:+.4f} | {deltas['gross'][i]:+.4f} |")
    L.append("")

    # ---- 3. mechanism probe
    order = np.argsort(deltas['net'])
    probe_seeds = [SEEDS[order[0]], SEEDS[order[len(order) // 2]], SEEDS[order[-1]]]
    slices = {arm: rh._build_oracle_slices(rh._base_config(1), acc) for arm, acc in ARMS.items()}
    for arm in ARMS:
        rc.assert_no_fold4(slices[arm]['test_aligned'].index)
    probe = pd.DataFrame([replay(arm, s, slices[arm]) for s in SEEDS for arm in ARMS])
    probe.to_csv(OUT_CSV, index=False)
    p60 = probe[probe.arm == "oracle60"].set_index("seed").loc[SEEDS]
    p75 = probe[probe.arm == "oracle75"].set_index("seed").loc[SEEDS]

    L += ["## 3. Mechanism probe", "",
          "Hypothesis: stronger signal → larger intended reversals → more band truncation → worse realized timing.",
          "",
          f"Primary probe seeds chosen by a fixed rule before looking at any probe metric: the seeds at the min, "
          f"median and max of the paired net Δ → **seeds {probe_seeds[0]}, {probe_seeds[1]}, {probe_seeds[2]}** "
          f"(Δ net {deltas['net'][order[0]]:+.4f}, {deltas['net'][order[len(order) // 2]]:+.4f}, "
          f"{deltas['net'][order[-1]]:+.4f}). Each checkpoint was replayed deterministically on the fold-2 test window and "
          "reproduces its stored net return exactly (asserted). Decision days only (every 5th step).", "",
          "Definitions: *intended action* = policy output after per-asset clip 0.8 and gross cap 2.0, BEFORE smoothing; "
          "*smoothed target* = 0.9·prev + 0.1·intended; *blocked* = desired change (|target − current| > 0) not "
          "executed because ≤ 0.10 band; *executed / intended move* = Σ|w_after − w_before| / Σ|intended − w_before|; "
          "*intended-target frictionless P&L* = intended action held over each 5-day window with no execution layer "
          "and no costs.", ""]
    hdr = "| metric | " + " | ".join(f"s{s} o60 | s{s} o75" for s in probe_seeds) + " |"
    L += [hdr, "|---|" + "---|" * (2 * len(probe_seeds))]
    for col, lab, f in PROBE_COLS:
        cells = " | ".join(f"{p60.loc[s, col]:{f}} | {p75.loc[s, col]:{f}}" for s in probe_seeds)
        L.append(f"| {lab} | {cells} |")
    L.append("")

    L += ["### Robustness: same metrics on all 20 matched seeds", "",
          "| metric | oracle60 mean | oracle75 mean | mean Δ (75−60) | Δ > 0 seeds | Wilcoxon p | Spearman ρ(Δ metric, Δ net) | ρ p |",
          "|---|---|---|---|---|---|---|---|"]
    for col, lab, f in PROBE_COLS:
        if col == 'net_return':
            continue
        dd = (p75[col] - p60[col]).to_numpy()
        ok = ~np.isnan(dd)
        wp = stats.wilcoxon(dd[ok]).pvalue if np.any(dd[ok] != 0) else float('nan')
        rho = stats.spearmanr(dd[ok], deltas['net'][ok])
        L.append(f"| {lab} | {p60[col].mean():{f}} | {p75[col].mean():{f}} | {dd[ok].mean():+.4f} | "
                 f"{int((dd[ok] > 0).sum())} / {int(ok.sum())} | {fmt_p(wp)} | {rho.correlation:+.3f} | {fmt_p(rho.pvalue)} |")
    L += ["", f"Per-seed probe values for both arms: `{OUT_CSV.as_posix()}`.", ""]
    L += [FINDINGS, ""]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
