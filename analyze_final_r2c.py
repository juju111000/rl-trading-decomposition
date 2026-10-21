#!/usr/bin/env python3
"""
FINAL R2C statistics pass (corrected accounting, audit defects #8/#9 repaired).
Analysis only: reads the completed batch outputs under results/r2/corrected/
plus the frozen pre-correction artifacts, and runs deterministic eval-only
replays of the P2' checkpoints for the mechanism probe (S1d; each replay is
asserted to reproduce its stored net return). No training. Fold 4 never loaded
(r2_common guard); P6 not run. Writes results/r2/corrected/FINAL_REPORT.md
(+ final_p2prime_probe.csv).

Statistics discipline: Holm within each named test family; for unpaired
comparisons a test "rejects" only if BOTH Welch and Mann-Whitney reject after
Holm (each list adjusted separately), for paired ones both paired t and
Wilcoxon -- except where n=5 makes Wilcoxon unable to reach the threshold
(stated per family). Paired designs are primary only where init identity was
verified. MDE at 80% power (two-sided, alpha 0.05) is reported for every test.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_r2c as C   # sets R2_DIR to the corrected tree; r2_common fold-4 guard installed first

import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower, TTestPower

import run_hardening as rh
from config import Config
from evaluate_saved import load_model as load_saved_policy
from r2_oracle import build_oracle5_slices, oracle_signal_5d

ROOT = C.ROOT
OUT = ROOT / "FINAL_REPORT.md"
PROBE_CSV = ROOT / "final_p2prime_probe.csv"
OLD = Path("./results")
ALPHA, POWER, EPS = 0.05, 0.80, 1e-6
CEILING_NET, CEILING_GROSS = 223.0598, 444.5217   # expressibility_grid.md: 100%, size 0.4, canonical w/o smoothing
SEEDS20 = list(range(1, 21))
TRAINERS = ["vanilla_ppo", "reward_shaping", "ppo_lagrangian"]
FOLD_OF = {"2008-2010": 1, "2015-2017": 2, "2020-2023": 3}


# ----------------------------------------------------------------------------- helpers

def f4(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.4f}"


def fp(p):
    return "n/a" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.4f}")


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def desc(x):
    lo, hi = ci(x)
    return f"{np.mean(x):+.4f} ± {np.std(x, ddof=1):.4f}", f"[{lo:+.4f}, {hi:+.4f}]"


def mde_ind(x, y):
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    return TTestIndPower().solve_power(nobs1=n1, ratio=n2 / n1, alpha=ALPHA, power=POWER) * sp


def mde_paired(d):
    return TTestPower().solve_power(nobs=len(d), alpha=ALPHA, power=POWER) * np.std(d, ddof=1)


def unpaired(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return {'diff': x.mean() - y.mean(), 'p1': stats.ttest_ind(x, y, equal_var=False).pvalue,
            'p2': stats.mannwhitneyu(x, y, alternative="two-sided").pvalue, 'mde': mde_ind(x, y),
            'n': f"{len(x)} vs {len(y)}", 'design': 'Welch / MWU'}


def paired(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    lo, hi = ci(d)
    return {'diff': d.mean(), 'sd': d.std(ddof=1), 'lo': lo, 'hi': hi, 'p1': stats.ttest_1samp(d, 0).pvalue,
            'p2': stats.wilcoxon(d).pvalue, 'mde': mde_paired(d), 'n': f"{len(d)} pairs",
            'pos': f"{int((d > 0).sum())}/{len(d)}", 'design': 'paired t / Wilcoxon'}


def holm(rows, rank_floor_ok=True):
    """Adds Holm-adjusted p's and the decision. rank_floor_ok=False: the rank test cannot reach
    the threshold (n=5), so the decision uses the parametric test alone (stated in the report)."""
    a1 = multipletests([r['p1'] for r in rows], method='holm')[1]
    a2 = multipletests([r['p2'] for r in rows], method='holm')[1]
    for r, h1, h2 in zip(rows, a1, a2):
        r['h1'], r['h2'] = h1, h2
        r['verdict'] = ("**reject**" if (h1 < ALPHA and (h2 < ALPHA or not rank_floor_ok))
                        else "mixed" if min(h1, h2) < ALPHA else "null")
    return rows


def test_table(rows, label="comparison"):
    L = [f"| {label} | metric | design | n | mean diff | p (t-type) | Holm | p (rank) | Holm | MDE@80% | verdict |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['metric']} | {r['design']} | {r['n']} | {r['diff']:+.4f} | {fp(r['p1'])} | "
                 f"{fp(r['h1'])} | {fp(r['p2'])} | {fp(r['h2'])} | {r['mde']:.4f} | {r['verdict']} |")
    return L


# ----------------------------------------------------------------------------- S1d replay probe

def replay_p2p(arm, seed, slices, true5):
    ckpt_dir = ROOT / "p2prime" / f"{arm}_training" / f"{arm}_seed{seed}"
    cfg = Config.from_json(str(ckpt_dir / "config.json"))
    assert cfg.regime.calibrated_vol_thresholds is not None
    assert cfg.environment.action_smoothing_beta is None
    env = rh.build_env_with_features(slices['test_features'], slices['test_prices'], cfg,
                                     slices['test_aligned'], mode="test")
    n = len(slices['tickers'])
    policy = load_saved_policy(ckpt_dir / "final_model.pt", env.observation_space.shape[0], n, cfg)
    band, k = cfg.environment.no_trade_band, max(1, cfg.environment.decision_frequency_days)
    feat = slices['test_features'][:, -n:] if C.P2P_ARMS[arm][0] is not None else None
    close = np.column_stack([slices['test_prices'][t] for t in slices['tickers']])
    obs, _ = env.reset()
    navs, step, done, dec, held = [env.portfolio.initial_capital], 0, False, [], []
    while not done:
        t = env._current_step
        w_before = env.portfolio.compute_weights(env._get_prices(t)).copy()
        with torch.no_grad():
            act, _, _ = policy.get_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True)
        a_raw = act.cpu().numpy().squeeze()
        a = env._decode_action(a_raw).astype(float)
        obs, _, term, trunc, info = env.step(a_raw)
        done = term or trunc
        if step % k == 0:
            dec.append((t, a, np.asarray(env._smoothed_weights, float).copy(), w_before,
                        info['portfolio_state'].weights.copy()))
        held.append(info['portfolio_state'].weights.copy())
        navs.append(info['nav'])
        step += 1
    net = navs[-1] / navs[0] - 1
    stored = json.loads((ROOT / "p2prime" / arm / f"seed_{seed}" / "eval_result.json").read_text())['return']
    assert abs(net - stored) < 1e-9 * max(1.0, abs(stored)), f"{arm} s{seed}: replay {net} != stored {stored}"
    T = np.array([d[0] for d in dec]); A = np.array([d[1] for d in dec]); TGT = np.array([d[2] for d in dec])
    WB = np.array([d[3] for d in dec]); WA = np.array([d[4] for d in dec])

    def corr_agree(sig):
        S = sig[T]
        m = S != 0
        return float(np.corrcoef(A[m], S[m])[0, 1]), float(np.mean(np.sign(A[m]) == np.sign(S[m])))

    c_feat, a_feat = corr_agree(feat) if feat is not None else (np.nan, np.nan)
    c_true, a_true = corr_agree(true5)
    gap = np.abs(TGT - WB)
    want = gap > EPS
    fwd_k = np.array([close[min(t + k, len(close) - 1)] / close[t] - 1 for t in T])
    dW = WA - WB
    tr = np.abs(dW) > EPS
    return {'arm': arm, 'seed': seed, 'net_return': net,
            'corr_action_oracle_feature': c_feat, 'sign_agree_oracle_feature': a_feat,
            'corr_action_true_5d_sign': c_true, 'sign_agree_true_5d_sign': a_true,
            'band_blocked_share': float((want & (gap <= band)).sum() / want.sum()),
            'executed_over_intended_move': float(np.abs(dW).sum() / np.abs(A - WB).sum()),
            'corr_executed_trade_fwd5d': float(np.corrcoef(dW[tr], fwd_k[tr])[0, 1]),
            'mean_gross_exposure': float(np.mean([np.abs(w).sum() for w in held]))}


def run_probe():
    rows = []
    for arm, (acc, seeds) in C.P2P_ARMS.items():
        cfg = C.nosmooth_config(1)
        sl = build_oracle5_slices(cfg, acc) if acc is not None else rh._load_fold2_slices(cfg)
        C.rc.assert_no_fold4(sl['test_aligned'].index)
        true5 = oracle_signal_5d(sl['test_prices'], sl['tickers'], len(sl['test_features']), 1.0)
        for s in seeds:
            rows.append(replay_p2p(arm, s, sl, true5))
            print(f"probe {arm} s{s} ok", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROBE_CSV, index=False)
    return df


# ----------------------------------------------------------------------------- report

def main():
    L = []
    P = pd.read_csv(ROOT / "p2prime" / "p2prime_results.csv")
    H = pd.read_csv(ROOT / "hardening" / "hardening_results.csv")
    assert P.groupby('arm').size().to_dict() == {'canonical_nosmooth': 20, 'oracle100_5d_nosmooth': 20,
                                                  'oracle60_5d_nosmooth': 5}
    assert (P[P.arm == 'canonical_nosmooth'].init_equal_to_p1_untrained == True).all()
    assert (H[H.arm == 'canonical'].init_equal == True).all() and len(H[H.arm == 'canonical']) == 20
    # RNG state at network creation identical across P2' arms for every seed
    assert (P.pivot(index='seed', columns='arm', values='rng_sha_at_init').nunique(axis=1) == 1).all()

    def arm(df, a, m, seeds=SEEDS20):
        s = df[df.arm == a].set_index('seed')
        return s.loc[[x for x in seeds if x in s.index], m].to_numpy(float)

    cn = lambda m, seeds=SEEDS20: arm(P, 'canonical_nosmooth', m, seeds)
    o100 = lambda m: arm(P, 'oracle100_5d_nosmooth', m)
    o60 = lambda m: arm(P, 'oracle60_5d_nosmooth', m, [1, 2, 3, 4, 5])
    can = lambda m: arm(H, 'canonical', m)
    unt = lambda m: arm(H, 'untrained', m)

    L += ["# R2C FINAL REPORT -- corrected accounting (audit defects #8 slippage units, #9 regime look-ahead)", "",
          "> Generated by `analyze_final_r2c.py` from the completed corrected batch (172/172 combos, no failures, "
          "ended 2026-10-16 19:19 UTC). Analysis only, plus deterministic eval-only replays for S1d. **Fold 4 never "
          "loaded; P6 (fold-4 re-evaluation) still held.** Oracle arms are NON-DEPLOYABLE diagnostics.", "",
          "> **Statistics.** Holm within each named family. Unpaired: reject only if Welch AND Mann-Whitney both "
          "reject after Holm. Paired: paired t AND Wilcoxon, used as primary only where initial-network identity "
          "was verified. MDE = minimum detectable effect at 80% power, two-sided alpha 0.05, from each comparison's "
          "own observed SD. A null with |diff| < MDE is uninformative, not evidence of no effect.", "",
          "> **'Submitted' values** (S4) come from the frozen pre-correction artifacts in this repo "
          "(`results/hardening/`, `results/full_grid/20260803_174817/`, `results/baselines/`, `PROJECT_STATE.md`, "
          "pre-correction `results/r2/*.md`). The manuscript text is not in the repo; check the wording against it.", ""]

    # ================================================================== S1
    L += ["## S1 -- P2': can the policy exploit a signal it can express? (no smoothing; band 0.10, 5-day "
          "cadence, cap 2.0, corrected costs; fold 2 test 2015-2017)", "",
          "Arms: `canonical_nosmooth` (n=20, no oracle), `oracle100_5d_nosmooth` (n=20, oracle = sign of next-5-day "
          "return, 100% accurate), `oracle60_5d_nosmooth` (n=5, 60%). Only `environment.action_smoothing_beta` "
          "differs from canonical (asserted in `run_r2c.nosmooth_config`).", "",
          "**Pairing validity.** `canonical_nosmooth` init is tensor-equal to P1's untrained init on 20/20 seeds, "
          "and P1 canonical's init is tensor-equal to the same untrained init on 20/20 -> canonical_nosmooth, "
          "canonical and untrained are **paired** (identical initial network per seed). Oracle arms observe 5 extra "
          "features (obs_dim 103 vs 98), so their networks differ from step 0: they are **seed-matched only** (the "
          "torch RNG state at network creation is identical across all P2' arms for every seed, verified by "
          "fingerprint), and are compared to non-oracle arms by unpaired tests.", ""]

    fam1 = []
    for m, ml in (("return", "net"), ("gross_price_pnl", "gross")):
        fam1.append({**unpaired(o100(m), cn(m)), 'name': 'oracle100_nosmooth vs canonical_nosmooth', 'metric': ml})
    for m, ml in (("return", "net"), ("gross_price_pnl", "gross")):
        fam1.append({**unpaired(o60(m), cn(m)), 'name': 'oracle60_nosmooth vs canonical_nosmooth', 'metric': ml})
    for m, ml in (("return", "net"), ("gross_price_pnl", "gross")):
        fam1.append({**paired(cn(m), unt(m)), 'name': 'canonical_nosmooth − untrained', 'metric': ml})
    for m, ml in (("return", "net"), ("gross_price_pnl", "gross")):
        fam1.append({**paired(cn(m), can(m)), 'name': 'canonical_nosmooth − canonical', 'metric': ml})
    holm(fam1)

    L += ["### S1a. oracle100_nosmooth vs canonical_nosmooth", "",
          "| arm | n | net return mean ± sd | 95% CI | gross P&L mean ± sd | 95% CI |", "|---|---|---|---|---|---|"]
    for name, f, n in (("oracle100_5d_nosmooth", o100, 20), ("oracle60_5d_nosmooth", o60, 5),
                       ("canonical_nosmooth", cn, 20), ("canonical (smoothed, P1)", can, 20),
                       ("untrained (P1)", unt, 20)):
        a, b = desc(f("return")); c, d = desc(f("gross_price_pnl"))
        L.append(f"| {name} | {n} | {a} | {b} | {c} | {d} |")
    L += ["", "Test family S1 (8 tests, Holm):", ""] + test_table(fam1) + [""]
    L += ["Paired rows report seeds with diff > 0: " + "; ".join(
        f"{r['name']} {r['metric']} {r['pos']} (95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}])"
        for r in fam1 if 'pos' in r) + ".", ""]

    mo, mg = o100("return").mean(), o100("gross_price_pnl").mean()
    g_lo, g_hi = ci(np.log1p(o100("return")))
    L += ["### S1b. Exploitation efficiency vs the deterministic no-smoothing ceiling", "",
          f"Ceiling = deterministic 100%-accurate signal-following rule (size 0.4/asset) through the SAME execution "
          f"(no smoothing, band 0.10, 5-day, cap 2.0, corrected costs): net **{CEILING_NET:+.2f}** "
          f"(x{1 + CEILING_NET:.1f} NAV), gross {CEILING_GROSS:+.2f} (`expressibility_grid.md`).", "",
          "| measure | oracle100_nosmooth (n=20) | ceiling | ratio |", "|---|---:|---:|---:|",
          f"| mean net return | {mo:+.3f} | {CEILING_NET:+.2f} | {mo / CEILING_NET:.4f} |",
          f"| mean gross P&L | {mg:+.3f} | {CEILING_GROSS:+.2f} | {mg / CEILING_GROSS:.4f} |",
          f"| mean log-growth log(1+r) | {np.log1p(o100('return')).mean():.3f} (95% CI [{g_lo:.3f}, {g_hi:.3f}]) | "
          f"{np.log1p(CEILING_NET):.3f} | {np.log1p(o100('return')).mean() / np.log1p(CEILING_NET):.3f} |",
          f"| per-seed net range | {o100('return').min():+.2f} to {o100('return').max():+.2f} | | |", "",
          "Compounded over 780 days, the return ratio understates partial exploitation; the log-growth ratio "
          "(share of the ceiling's per-day compounding rate captured) is the more interpretable efficiency.", ""]

    r60 = [r for r in fam1 if r['name'].startswith('oracle60')]
    L += ["### S1c. oracle60_nosmooth (n=5) vs canonical_nosmooth (n=20) -- directional read only", "",
          f"Net diff {r60[0]['diff']:+.4f} (Welch p {fp(r60[0]['p1'])}, MWU p {fp(r60[0]['p2'])}; MDE "
          f"{r60[0]['mde']:.4f}); gross diff {r60[1]['diff']:+.4f} (Welch p {fp(r60[1]['p1'])}; MDE "
          f"{r60[1]['mde']:.4f}). With n=5 the design can only detect effects of about the MDE; the deterministic "
          "60% no-smoothing rule nets -0.020 (gross +1.08), so even a perfect 60% follower is not expected to beat "
          "a +0.19 policy on net. Direction only; no claim.", ""]

    # S1d
    probe = pd.read_csv(PROBE_CSV) if '--reuse-probe' in sys.argv and PROBE_CSV.exists() else run_probe()
    old = pd.read_csv(ROOT / "oracle_anomaly_probe.csv")
    L += ["### S1d. Mechanism: does the policy use the signal when it is expressible?", "",
          "Deterministic replays of all 45 P2' checkpoints (each reproduces its stored net return, asserted); "
          "decision days only. *Oracle feature* = the column the arm observes; *true 5d sign* = sign of the actual "
          "next-5-day return (= the oracle100 feature), computed also for canonical_nosmooth as a no-access "
          "control. Canonical-execution oracle arms from `oracle_anomaly_probe.csv` (P1, smoothing 0.1).", "",
          "| arm | execution | signal observed | n | corr(action, oracle feature) | sign agreement | corr(action, true 5d sign) | "
          "band-blocked share | executed / intended move | corr(executed trade, next-5d return) | mean gross exposure |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]

    def pm(df, a, c):
        x = df[df.arm == a][c].to_numpy(float)
        return "n/a" if np.all(np.isnan(x)) else f"{np.nanmean(x):+.3f} ± {np.nanstd(x, ddof=1):.3f}"

    for a, ex, sig in (("oracle100_5d_nosmooth", "no smoothing", "5d sign, 100%"),
                       ("oracle60_5d_nosmooth", "no smoothing", "5d sign, 60%"),
                       ("canonical_nosmooth", "no smoothing", "none (control)")):
        L.append(f"| {a} | {ex} | {sig} | {(probe.arm == a).sum()} | {pm(probe, a, 'corr_action_oracle_feature')} | "
                 f"{pm(probe, a, 'sign_agree_oracle_feature')} | {pm(probe, a, 'corr_action_true_5d_sign')} | "
                 f"{pm(probe, a, 'band_blocked_share')} | {pm(probe, a, 'executed_over_intended_move')} | "
                 f"{pm(probe, a, 'corr_executed_trade_fwd5d')} | {pm(probe, a, 'mean_gross_exposure')} |")
    for a in ("oracle60", "oracle75"):
        L.append(f"| {a} (P1) | canonical (smoothing 0.1) | 1d sign, {a[-2:]}% | {(old.arm == a).sum()} | "
                 f"{pm(old, a, 'corr_action_oracle')} | {pm(old, a, 'sign_agree_action_oracle')} | n/a | "
                 f"{pm(old, a, 'band_blocked_share')} | {pm(old, a, 'executed_over_intended_move')} | "
                 f"{pm(old, a, 'corr_executed_trade_fwd5d')} | {pm(old, a, 'mean_gross_exposure')} |")
    c100 = probe[probe.arm == 'oracle100_5d_nosmooth'].corr_action_oracle_feature
    ctrl = probe[probe.arm == 'canonical_nosmooth'].corr_action_true_5d_sign
    u = stats.mannwhitneyu(c100, ctrl, alternative="two-sided").pvalue
    L += ["", f"oracle100_nosmooth corr(action, feature) vs the no-access control's corr(action, true 5d sign): "
          f"MWU p {fp(u)} (n=20 vs 20). Per-seed values: `{PROBE_CSV.as_posix()}`.", "",
          "**Confound (must accompany any reading of this table):** the P1 canonical-execution oracle arms observe "
          "a **1-day** sign; the P2' arms observe a **5-day** sign. The side-by-side therefore changes smoothing "
          "AND signal horizon together. No trained arm isolates smoothing at a fixed horizon (P2, the 5-day oracle "
          "under canonical execution, was held after its gate failed), so the hypothesis 'signal ignored when "
          "inexpressible, used when expressible' is consistent with these numbers but not isolated by them.", ""]

    # S1e
    L += ["### S1e. Risk profile (all arms unshielded, mode off)", "",
          "None of these arms runs a shield: the 20%/30% drawdown levels only enter training as the drawdown cost "
          "stream. Violation rate = share of test days with drawdown > 20% (drawdown cost active). Oracle arms are "
          "non-deployable; their low drawdown reflects foresight, not risk control.", "",
          "| arm | n | max DD | trailing-63d max DD | turnover/day | cost/NAV | violation rate | net return |",
          "|---|---|---|---|---|---|---|---|"]
    for name, df, a in (("oracle100_5d_nosmooth", P, "oracle100_5d_nosmooth"), ("oracle60_5d_nosmooth", P, "oracle60_5d_nosmooth"),
                        ("canonical_nosmooth", P, "canonical_nosmooth"), ("canonical (smoothed)", H, "canonical"),
                        ("untrained", H, "untrained")):
        s = df[df.arm == a]
        cell = lambda c: f"{s[c].mean():.4f} ± {s[c].std(ddof=1):.4f}"
        L.append(f"| {name} | {len(s)} | {cell('max_dd')} | {cell('trailing_63d_max_dd')} | {cell('turnover')} | "
                 f"{cell('txn_cost')} | {cell('dd_violation_rate')} | {s['return'].mean():+.4f} |")

    # S1f
    L += ["", "### S1f. Does smoothing help or hurt the trained policy itself? (canonical_nosmooth − canonical, "
          "paired, identical init, n=20)", "",
          "| metric | mean diff | SD | 95% CI | paired t p | Wilcoxon p | seeds > 0 | MDE@80% |",
          "|---|---:|---:|---|---:|---:|---|---:|"]
    for m, ml in (("return", "net return"), ("gross_price_pnl", "gross P&L"), ("txn_cost", "cost/NAV"),
                  ("turnover", "turnover/day"), ("max_dd", "max DD"), ("dd_violation_rate", "violation rate")):
        r = paired(cn(m), can(m))
        L.append(f"| {ml} | {r['diff']:+.4f} | {r['sd']:.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {fp(r['p1'])} | "
                 f"{fp(r['p2'])} | {r['pos']} | {r['mde']:.4f} |")
    L += ["", "Net and gross rows are Holm-adjusted in family S1 above; cost/turnover/DD rows are descriptive.", ""]

    # ================================================================== S2
    G = pd.read_csv(ROOT / "grid" / "results.csv")
    O = pd.read_csv(OLD / "full_grid" / "20260803_174817" / "results.csv")
    for d in (G, O):
        d['fold'] = d.window.map(FOLD_OF)
        assert len(d) == 90 and d.groupby(['fold', 'shield', 'method']).size().eq(5).all()
    L += ["## S2 -- P5 main grid under corrected costs (3 trainers x {none, uncertainty_aware} x folds 1-3 x "
          "seeds 1-5, 300k steps, canonical config)", "",
          "Grid arms within a seed share one `seed_everything` and run sequentially, so arms are **not** "
          "init-matched: all grid tests are unpaired. Seed 3 was interrupted mid-arm (2026-10-15 Windows restart "
          "attempt) and resumed; its later arms saw a different RNG stream than an uninterrupted run.", ""]
    sh, osh = G[G.shield == 'uncertainty_aware'], O[O.shield == 'uncertainty_aware']

    def bcount(d, lo, hi):
        return int(((d.max_dd >= lo) & (d.max_dd <= hi)).sum())

    L += ["### S2a. The boundary: max DD of all 45 shielded runs", "",
          "| grid | min | median | max | in [0.30, 0.311] | in [0.299, 0.300) | > 0.300 | < 0.299 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, d in (("corrected", sh), ("submitted (defective costs)", osh)):
        L.append(f"| {name} | {d.max_dd.min():.6f} | {d.max_dd.median():.6f} | {d.max_dd.max():.6f} | "
                 f"{bcount(d, 0.30, 0.311)}/45 | {int(((d.max_dd >= 0.299) & (d.max_dd < 0.30)).sum())}/45 | "
                 f"{int((d.max_dd > 0.30).sum())}/45 | {int((d.max_dd < 0.299).sum())}/45 |")
    low = sh[sh.max_dd < 0.299]
    L += ["", "Submitted claim (`PROJECT_STATE.md:294-295`): shielded max_dd 'clusters right at dd_stop (0.30) across "
          "nearly every fold/seed'; submitted grid 38/45 in [0.30, 0.311], max 0.3111 (overshoot up to 1.1 pp). "
          f"Corrected: the cluster at 0.30 holds and **the overshoot is gone** -- no shielded run exceeds 0.300; "
          f"{45 - len(low)}/45 lie within 0.001 below it. Below 0.299: " +
          (", ".join(f"fold {r.fold} {r.method} s{r.seed} ({r.max_dd:.4f})" for r in low.itertuples()) or "none") +
          ". The bound is now hard rather than approximate; it is still binding in essentially every run.", ""]

    fam_trap = []
    L += ["### S2b. The trap: shielded vs unshielded returns per fold (pooled over trainers, n=15 vs 15)", "",
          "| fold | unshielded mean ± sd | shielded mean ± sd | shielded runs < 0 | unshielded runs < 0 | "
          "submitted unshielded | submitted shielded |", "|---|---|---|---|---|---|---|"]
    for f in (1, 2, 3):
        u_, s_ = G[(G.fold == f) & (G.shield == 'none')]['return'], G[(G.fold == f) & (G.shield != 'none')]['return']
        ou, os_ = O[(O.fold == f) & (O.shield == 'none')]['return'], O[(O.fold == f) & (O.shield != 'none')]['return']
        L.append(f"| {f} | {desc(u_)[0]} | {desc(s_)[0]} | {int((s_ < 0).sum())}/15 | {int((u_ < 0).sum())}/15 | "
                 f"{ou.mean():+.4f} | {os_.mean():+.4f} |")
        fam_trap.append({**unpaired(s_, u_), 'name': f'fold {f}: shielded − unshielded', 'metric': 'net'})
    holm(fam_trap)
    L += ["", "Test family S2b (3 tests, Holm):", ""] + test_table(fam_trap) + [""]
    L += ["Mechanism (E2, `gated_same_policy_summary.md`): with the all-time-peak drawdown anchor the 30% "
          "zero-exposure state is absorbing -- once hit, the always-on book is flat on 35-58% of later days in the "
          "corrected run (98-100% submitted).", ""]

    # S2c trainer invariance
    om, pw = [], []
    for sname in ("none", "uncertainty_aware"):
        for f in (1, 2, 3):
            cell = G[(G.fold == f) & (G.shield == sname)]
            groups = [cell[cell.method == t]['return'].to_numpy() for t in TRAINERS]
            om.append({'name': f"fold {f}, shield={sname}", 'metric': 'net', 'design': 'ANOVA / Kruskal-Wallis',
                       'n': '3 x 5', 'diff': max(g.mean() for g in groups) - min(g.mean() for g in groups),
                       'p1': stats.f_oneway(*groups).pvalue, 'p2': stats.kruskal(*groups).pvalue,
                       'mde': max(mde_ind(groups[i], groups[j]) for i in range(3) for j in range(i + 1, 3))})
            for i in range(3):
                for j in range(i + 1, 3):
                    pw.append({**unpaired(groups[i], groups[j]), 'metric': 'net',
                               'name': f"f{f} {sname[:4]}: {TRAINERS[i]} − {TRAINERS[j]}"})
    holm(om)
    holm(pw)
    L += ["### S2c. Trainer invariance (per fold at matched shield status)", "",
          "Primary family: 6 omnibus tests (one per fold x shield cell; 'mean diff' = max − min trainer mean; MDE = "
          "largest pairwise MDE in the cell), Holm over 6.", ""] + test_table(om, "cell") + [""]
    L += ["Secondary family: 18 pairwise trainer comparisons (n=5 vs 5), Holm over 18. With 5 vs 5 the smallest "
          "attainable two-sided MWU p is 0.0079 > the first Holm threshold 0.05/18 = 0.0028, so the conjunction "
          "rule **cannot** reject in this family whatever the effect size; Welch-Holm is shown for information.", ""]
    L += test_table(pw) + [""]

    L += ["Per-cell trainer means (net return):", "",
          "| fold | shield | " + " | ".join(TRAINERS) + " |", "|---|---|---:|---:|---:|"]
    for f in (1, 2, 3):
        for sname in ("none", "uncertainty_aware"):
            c = G[(G.fold == f) & (G.shield == sname)].groupby('method')['return']
            L.append(f"| {f} | {sname} | " + " | ".join(f"{c.mean()[t]:+.4f} ± {c.std()[t]:.4f}" for t in TRAINERS) + " |")

    L += ["", "### S2d. Corrected vs submitted (defective-cost) grid, per cell means", "",
          "| fold | shield | trainer | net (corr.) | net (subm.) | Δ | max DD (corr.) | max DD (subm.) | "
          "turnover (corr.) | turnover (subm.) | violation (corr.) | violation (subm.) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    gm = G.groupby(['fold', 'shield', 'method'])[['return', 'max_dd', 'turnover', 'violation_rate']].mean()
    omn = O.groupby(['fold', 'shield', 'method'])[['return', 'max_dd', 'turnover', 'violation_rate']].mean()
    for key in gm.index:
        a, b = gm.loc[key], omn.loc[key]
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {a['return']:+.4f} | {b['return']:+.4f} | "
                 f"{a['return'] - b['return']:+.4f} | {a.max_dd:.4f} | {b.max_dd:.4f} | {a.turnover:.4f} | "
                 f"{b.turnover:.4f} | {a.violation_rate:.4f} | {b.violation_rate:.4f} |")

    # ================================================================== S3
    E = pd.read_csv(ROOT / "gated_same_policy.csv")
    MODES = ["off", "always_on", "gated", "hybrid"]
    L += ["", "## S3 -- P4b: E2 shield modes and E3 beliefs on folds 1-3 (corrected; folds 1 and 3 POST-HOC "
          "SUPPLEMENTARY)", "",
          "Same trained policies evaluated under four shield modes (fold 2: the 20 P1 canonical checkpoints; folds "
          "1/3: the 5 grid `none/ppo_lagrangian` checkpoints). Mode `off` reproduces the stored returns exactly "
          "(max |diff| <= 2.2e-16). *hybrid* (fourth mode) = continuous 20->30% drawdown scaling every day "
          "anchored to the trailing-63-day NAV max (not absorbing) + the always-on caps only on gate-active days.", "",
          "| fold | mode | n | net return mean ± sd | 95% CI | mean max DD | violation rate | mean gross exposure |",
          "|---|---|---|---|---|---:|---:|---:|"]
    for f in (2, 1, 3):
        for m in MODES:
            s = E[(E.fold == f) & (E['mode'] == m)]
            a, b = desc(s['return'])
            L.append(f"| {f}{' (post-hoc)' if f != 2 else ''} | {m} | {len(s)} | {a} | {b} | {s.max_dd.mean():.4f} | "
                     f"{s.violation_rate.mean():.4f} | {s.mean_gross_exposure.mean():.3f} |")
    for f in (2, 1, 3):
        fam = []
        piv = E[E.fold == f].pivot(index='seed', columns='mode', values='return')
        for a, b in (("gated", "always_on"), ("hybrid", "always_on"), ("hybrid", "gated"),
                     ("gated", "off"), ("hybrid", "off"), ("always_on", "off")):
            fam.append({**paired(piv[a], piv[b]), 'name': f"{a} − {b}", 'metric': 'net'})
        n5 = f != 2
        holm(fam, rank_floor_ok=not n5)
        L += ["", f"Family S3-fold{f} (6 paired same-policy contrasts, Holm over 6)" +
              (": n=5 -> smallest attainable Wilcoxon p is 0.0625, so the decision uses Holm-adjusted paired t "
               "alone (Wilcoxon reported)." if n5 else ":"), ""] + test_table(fam)
    L += ["", "Submitted / pre-correction E2 (fold 2, n=20, `results/r2/gated_same_policy_summary.md`): off "
          "+0.0701, always-on -0.2323, gated -0.3118; gated − always-on **-0.0795** (4/20 seeds > 0). Corrected: "
          "gated − always-on **reverses sign** (see family S3-fold2).", ""]

    # E3 belief deltas
    def belief_rows(path):
        rows = {}
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            mt = re.match(r"^\| ([123]) \| (hysteresis|softmax) \| (\d+) \| [\d/]+ \| ([\d.]+) \| ([\d.]+) \| "
                          r"([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
            if mt:
                rows[(int(mt[1]), mt[2])] = [float(mt[i]) for i in range(4, 9)]
        assert len(rows) == 6, path
        return rows

    nb, ob = belief_rows(ROOT / "belief_validation.md"), belief_rows(OLD / "r2" / "belief_validation.md")
    L += ["### S3b. E3 belief validation: D9-corrected thresholds (train-window calibration) vs pre-correction "
          "(test-window fit, a look-ahead)", "",
          "Proxy labels are prespecified historical proxies, not ground truth. Values parsed from "
          "`results/r2/corrected/belief_validation.md` and `results/r2/belief_validation.md`.", "",
          "| fold | operator | agreement (new / old / Δ) | Brier (new / old / Δ) | Bear gate precision (new / old / Δ) | "
          "Bear gate recall (new / old / Δ) | gate active share (new / old / Δ) |", "|---|---|---|---|---|---|---|"]
    for key in sorted(nb):
        cells = [f"{n:.3f} / {o:.3f} / {n - o:+.3f}" for n, o in zip(nb[key], ob[key])]
        L.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    L += ["", "Fold-2 Part B (softmax-gated vs hysteresis-gated, same 20 policies) remains degenerate under correction: "
          "softmax entropy never falls below 0.706, so its gate is on 100% of days and softmax-gated = always-on. "
          "Corrected: softmax − hysteresis net **-0.2890** (0/20 seeds > 0, p<0.001); submitted pre-correction: "
          "**+0.0795** (16/20). The sign flip is the E2 reversal seen from the other side; it does not separate "
          "belief quality from shield engagement.", ""]

    # ================================================================== S4
    lad = pd.read_csv(ROOT / "ladder_seeded.csv").pivot(index='seed', columns='config', values='return')
    lad_mde = {k: mde_paired(lad[a] - lad[b]) for k, (a, b) in
               {'b−a': ('b', 'a'), 'c−a': ('c', 'a'), 'd−a': ('d', 'a'), 'e−d': ('e', 'd'), 'f−e': ('f', 'e')}.items()}
    pen_net = paired(can("return"), unt("return"))
    pen_gross = paired(can("gross_price_pnl"), unt("gross_price_pnl"))
    pen_h = multipletests([pen_net['p1'], pen_gross['p1']], method='holm')[1]
    pen_hw = multipletests([pen_net['p2'], pen_gross['p2']], method='holm')[1]
    r100, sm = fam1[0], [r for r in fam1 if r['name'] == 'canonical_nosmooth − canonical'][0]
    lo_om = min(r['mde'] for r in om if 'uncertainty' in r['name'])
    hi_om = max(r['mde'] for r in om if 'none' in r['name'])
    L += ["## S4 -- Consolidated claims", "",
          "Verdicts: **holds** (direction and significance survive correction), **revised** (survives in changed "
          "form or size, or only part of it survives), **retired** (no longer supported). Every null is given with "
          "its MDE.", "",
          "| # | claim | submitted value (pre-correction) | corrected value | verdict |", "|---|---|---|---|---|",
          f"| 1 | **Training penalty**: trained canonical < untrained network (fold 2) | net +0.0701 vs +0.2902, "
          f"diff -0.2202 (Welch p<0.001, n=20 unpaired); paired seeds 6-20: -0.2209 (Wilcoxon p<0.001); gross "
          f"-0.1008 (p 0.0012) (`results/hardening/stats_report.md`) | paired n=20, init verified (primary), Holm "
          f"over net/gross: net **{pen_net['diff']:+.4f}** [{pen_net['lo']:+.4f}, {pen_net['hi']:+.4f}], t-Holm "
          f"{fp(pen_h[0])}, W-Holm {fp(pen_hw[0])}, {pen_net['pos']} seeds trained > untrained; gross "
          f"{pen_gross['diff']:+.4f}, t-Holm {fp(pen_h[1])} (null, MDE {pen_gross['mde']:.3f}). Unpaired 8-test "
          f"family: net no longer survives Holm (Welch 0.064) | **revised**: net penalty holds at less than half "
          f"the size; the gross-P&L penalty is retired (null, inconclusive below MDE) |",
          f"| 2 | **Oracle interpretation**: the policy cannot exploit an explicit signal | oracle60 +0.1039 / oracle75 "
          f"+0.0855 vs canonical +0.0701, all diffs < MDE (0.069-0.125); 'did not reliably learn to exploit it' | "
          f"canonical execution (1d oracle): no difference survives Holm; the actions ignore the feature (corr "
          f"+0.014 / +0.028). No smoothing (5d oracle): oracle100 **+{o100('return').mean():.2f}** vs "
          f"canonical_nosmooth +{cn('return').mean():.2f} (Welch/MWU-Holm <0.001), corr(action, feature) "
          f"**+0.420**; captures {mo / CEILING_NET:.1%} of the ceiling's return, "
          f"{np.log1p(o100('return')).mean() / np.log1p(CEILING_NET):.0%} of its log-growth. oracle60_nosmooth "
          f"vs canonical_nosmooth: {r60[0]['diff']:+.3f}, null (MDE {r60[0]['mde']:.3f}, n=5) | **revised**: the "
          "learning-failure reading is not supported. The policy exploits a signal it can express; under canonical "
          "execution the signal is not expressible. (Confound: the oracle horizon also differs, 1d vs 5d) |",
          f"| 3 | **Execution effects**: smoothing/cap/band/cadence are what make the policy viable | band 0.10 x k=5 "
          f"'breakthrough': +16.55%, turnover 234.5%/yr; smoothing+cap: turnover 49%/day -> 4.65%/day "
          f"(`PROJECT_STATE.md`) | costs: ladder cost/NAV a 0.79 -> f 0.03 (n=5, every config); smoothing on the "
          f"trained policy (paired n=20): costs -0.210, turnover -0.097/day (p<0.001), but net "
          f"{-sm['diff']:+.4f} for smoothing, Holm null (MDE {sm['mde']:.3f}); ladder net steps all null, paired "
          f"MDEs " + ", ".join(f"{k} {v:.2f}" for k, v in lad_mde.items()) + "; deterministic 100% rule: +0.07 "
          "with smoothing vs +223 without | **revised**: the cost and turnover reductions hold; a net-return "
          f"benefit is not shown (n=5 ladder MDEs {min(lad_mde.values()):.2f}-{max(lad_mde.values()):.2f}; n=20 "
          f"smoothing MDE {sm['mde']:.2f}); smoothing suppresses signal "
          "expression. The +16.55% figure is superseded (canonical now +0.296, n=20) |",
          f"| 4 | **Boundary**: shielded max DD clusters at the 0.30 hard stop | 38/45 in [0.30, 0.311], max 0.3111; "
          f"'clusters right at dd_stop (0.30) across nearly every fold/seed' | 0/45 in [0.30, 0.311]; "
          f"{45 - len(low)}/45 in [0.299, 0.300); max {sh.max_dd.max():.6f}; 0/45 > 0.300 | **revised**: the "
          "cluster at 0.30 holds; the overshoot to 0.311 is retired: re-evaluating the 15 defective-era fold-3 "
          "shielded checkpoints under corrected accounting removes it (15/15 -> 0/15 above 0.300; "
          "`preedit_verification.md` V-C), so it was an accounting artifact, not a property of the trained "
          "policies (D8 and D9 not separated). The "
          "bound is now exact |",
          f"| 5 | **Trap**: the always-on shield turns profitable folds into losses | grid fold 2 shielded -0.207 vs "
          f"unshielded +0.133, fold 3 -0.302 vs +0.466; fold 1 shield better (-0.289 vs -0.382); absorbing share "
          f"98-100% | fold 2 {fam_trap[1]['diff']:+.3f}, fold 3 {fam_trap[2]['diff']:+.3f} (both Holm reject); "
          f"fold 1 {fam_trap[0]['diff']:+.3f} (Holm reject: the shield now hurts in 2008-10 too); shielded runs < 0: "
          "15/15, 15/15, 13/15; absorbing share 35-58% | **holds**, and is larger: the losses grow and the fold-1 "
          "exception disappears |",
          "| 6 | **Gating**: the regime-gated shield fixes the trap | v2 sweep: gated +1.0..+6.6% vs always -31..-42% "
          "on 2015-17; pre-correction same-policy E2 contradicted it: gated − always-on **-0.0795** (4/20) | "
          "same-policy E2: gated − always-on **+0.289** (20/20; fold 2 Holm reject), folds 1/3 +0.216 / +0.408 "
          "(paired-t Holm reject, n=5, post-hoc); gated − off -0.188 (fold 2) / -0.743 (fold 3) but +0.187 "
          "(fold 1); hybrid − gated +0.021 (fold 2, null, MDE 0.068) | **revised**: gated > always-on now holds "
          "on every fold (the pre-correction E2 reversal is itself retired). In profitable folds gating still "
          "costs return vs no shield. The fourth mode is not distinguishable from gated on fold 2 |",
          "| 7 | **Beliefs**: the hysteresis belief operator gives better-timed regime calls | pre-correction E3 fold 2: "
          "agreement 0.772, Brier 0.433, Bear precision/recall 0.812/0.422; softmax-gated − hysteresis-gated "
          "+0.0795 (16/20) | fold 2 0.746 / 0.479 / 0.811 / 0.324; fold 1 agreement +0.220, fold 3 -0.116 "
          "(precision 0.169); Part B still degenerate (softmax gate on 100% of days) and its sign flips to "
          "-0.2890 (0/20) | **revised**: removing the look-ahead changes belief quality differently per fold "
          "(better fold 1, worse folds 2-3); no belief-quality advantage is supported; Part B only restates E2 |",
          "| 8 | **Baselines**: the policy adds no value over a passive book with matched execution | k5_band010 "
          "+23.30% beats RL +16.55% with ~7x less trading; 'RL hasn't learned to add value over the simplest "
          "target' | fold 2: canonical +0.296 (n=20) vs ew_matched_execution +0.193: t p 0.004; vs k5_band010 "
          "+0.233: t p 0.057, Wilcoxon 0.036, sign 0.012 (`ew_decomposition.md`, outside Holm, seed variance "
          "only); untrained +0.392 beats all; folds 1/3 unshielded grid -0.153 / +0.920 vs ew_matched +0.019 / "
          "+0.492 | **revised**: RL now beats the passive matched benchmark on fold 2, but an untrained network "
          "does better still, so the margin is not attributable to learning; fold-dependent across 1-3 |",
          f"| 9 | **Trainer invariance**: trainer choice does not matter at matched shield status | implied only "
          f"(no explicit claim found); shielded cells near-identical, unshielded spread up to 0.21 | 6 omnibus "
          f"tests, all Holm null; MDE shielded folds 1/2 {lo_om:.3f}-0.095, fold 3 0.489; unshielded up to "
          f"{hi_om:.3f} | **revised**: no detectable difference anywhere. Invariance is only informative for "
          "shielded folds 1-2 (and there the shield flattens everyone); unshielded and fold-3 nulls are "
          "uninformative |", ""]

    # ================================================================== S5
    tracked = set(Path(p).as_posix() for p in __import__('subprocess').run(
        ["git", "ls-files", ROOT.as_posix()], capture_output=True, text=True).stdout.split())
    commit = sorted({p.as_posix() for p in ROOT.glob("*") if p.is_file() and p.suffix in ('.md', '.csv', '.json', '.gz')}
                    | {(ROOT / x).as_posix() for x in ("grid/results.csv", "p2prime/p2prime_results.csv",
                                                       "hardening/hardening_results.csv", "e5/oracle_verification.md",
                                                       "e5/signal_following_benchmark.csv")})
    rest = [p for p in ROOT.rglob("*") if p.is_file() and p.as_posix() not in commit and p.as_posix() not in tracked]
    by = {}
    for p in rest:
        k = p.relative_to(ROOT).parts[0] + ("/" if len(p.relative_to(ROOT).parts) > 1 else "")
        c = by.setdefault(k, [0, 0, set()])
        c[0] += 1; c[1] += p.stat().st_size; c[2].add(p.suffix or p.name)
    L += ["## S5 -- Repo hygiene", "",
          "Remotes `origin` and `backup` both point to the same **private** repository (anonymous GitHub API "
          "returns 404); the public-release repository is separate and was not touched. `results/` is "
          "git-ignored, so corrected artifacts are committed by explicit force-add.", "",
          f"**Committed with this report** ({len(commit)} files: every report, summary CSV, manifest and "
          "per-step archive):", ""]
    L += [f"- `{Path(p).relative_to(ROOT).as_posix()}`" for p in commit]
    L += ["", "**Not committed** (per-run artifacts; reproducible from code + seeds, or archived separately):", "",
          "| path | files | size | types |", "|---|---:|---:|---|"]
    for k, (n, b, t) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        L.append(f"| `{k}` | {n} | {b / 2**20:.1f} MB | {', '.join(sorted(t))} |")
    L += ["", "These per-run trees are archived as release assets (tag `r2c-final`: grid.zip, hardening.zip, "
          "p2prime.zip, e6.zip). `checkpoint_archive_manifest.json` records the SHA-256 of every archived file. "
          "Of the 270 checkpoints, 100 (all P2' and ladder checkpoints, the 20 P1 canonical) also carry a hash in "
          "a manifest committed before archiving; the other 170 (90 grid, 40 P1 oracle, 40 P1 init snapshots) "
          "were first hashed at archiving time (2026-10-16), so their integrity is attested from then on, not "
          "from training time.", ""]

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
