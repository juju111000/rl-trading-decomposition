#!/usr/bin/env python3
"""
R2 / E3 -- belief validation against prespecified historical regime proxies,
plus the fixed-policy gating comparison (eval only).

Part A (folds 1-3 test windows): beliefs from the canonical hard-EMA-
hysteresis operator and the softmax (T=1.0) operator, each driven exactly as
the pipeline drives them (main.create_trading_env, fit() on the test slice,
flat action -- the regime model never sees the action). Compared against
PRESPECIFIED PROXY labels (not ground truth), defined once, no tuning:
  universe = equal-weight daily-rebalanced index of the 5 assets
  Bear     = trailing-60d universe return < -10%  OR  universe drawdown > 15%
  Bull     = trailing-60d universe return > +10%  AND universe drawdown < 5%
  Sideways = otherwise
Drawdown is measured from the running peak since the start of the loaded data
(2005-11-22). Labels at date t use prices through t only.

Part B (fold 2, n=20 canonical checkpoints): regime-gated shield, identical
except the GATE's belief operator -- hysteresis-gated vs softmax-gated
(r2_common.ShadowBeliefGatedShield). Policy observations and the inner
shield's own inputs stay on the canonical hysteresis beliefs in both.

Fold 4 never loaded (fail-closed guard). Output: results/r2/belief_validation.md
(+ belief_validation_daily.csv.gz, belief_validation_fixed_policy.csv, manifest).
"""
import copy
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import r2_common as rc  # installs the fold-4 guard first

import numpy as np
import pandas as pd

import main as main_module
from config import BeliefFormationType, Config
from regime_model import create_regime_model
from run_rc_overlay_forensics import _load_dev_data, _fold_slices
from run_validation_workstream import _load_fold2_slices

OPERATORS = {"hysteresis": BeliefFormationType.HARD_EMA_HYSTERESIS,
             "softmax": BeliefFormationType.SOFTMAX_TEMPERATURE}
FOLD_WINDOWS = {1: "2008-2010", 2: "2015-2017", 3: "2020-2023"}
LABELS = ["Bull", "Bear", "Sideways"]   # index order matches the regime model (0 Bull, 1 Bear, 2 Sideways)


def proxy_labels(prices: dict, tickers, index) -> pd.DataFrame:
    close = np.column_stack([prices[t] for t in tickers])
    simple = np.vstack([np.zeros(len(tickers)), close[1:] / close[:-1] - 1.0])
    idx = np.cumprod(1.0 + simple.mean(axis=1))
    s = pd.Series(idx, index=index)
    trail60 = s / s.shift(60) - 1.0
    dd = 1.0 - s / s.cummax()
    lab = np.where((trail60 < -0.10) | (dd > 0.15), 1, np.where((trail60 > 0.10) & (dd < 0.05), 0, 2))
    lab = pd.Series(lab, index=index).where(trail60.notna())
    return pd.DataFrame({'label': lab, 'trail60': trail60, 'univ_dd': dd})


def simulate_beliefs(base_config, test_features, test_prices, test_aligned, operator):
    cfg = copy.deepcopy(base_config)
    cfg.regime.belief_formation = OPERATORS[operator]
    env = main_module.create_trading_env(test_features, test_prices, cfg, aligned_data=test_aligned, mode="test")
    flat = np.zeros(env.action_space.shape, dtype=np.float32)
    env.reset()
    rows, done = [], False
    while not done:
        committed = getattr(env.regime_model, "_committed_regime", None)
        _, _, term, trunc, info = env.step(flat)
        done = term or trunc
        b = np.asarray(info['regime_belief'], float)
        rows.append({'p_bull': b[0], 'p_bear': b[1], 'p_sideways': b[2],
                     'entropy_norm': info['regime_entropy'] / np.log(3),
                     'argmax': int(np.argmax(b)), 'committed': committed,
                     'gate': rc.gate_condition(b, info['regime_entropy'])})
    return pd.DataFrame(rows)


def confusion(lab, pred):
    m = np.zeros((3, 3), int)
    for a, b in zip(lab, pred):
        m[int(a), int(b)] += 1
    return m


def part_a(L, daily_out):
    tickers, base_config, features, prices, aligned_data, splits = _load_dev_data()
    rc.assert_no_fold4(aligned_data.index)
    labels_all = proxy_labels(prices, tickers, aligned_data.index)
    summary_rows = []
    for fold in (1, 2, 3):
        sp = splits[fold - 1]
        _, tf, tp, ta, tdates, _ = _fold_slices(sp, features, prices, aligned_data, tickers)
        lab = labels_all.loc[tdates]
        fold_cfg = copy.deepcopy(base_config)
        if rc.CORRECTED:   # audit defect #9: thresholds from this fold's TRAINING window, frozen for the test window
            fold_cfg.regime.calibrated_vol_thresholds = None
            main_module.calibrate_regime(fold_cfg, aligned_data.iloc[sp['train'][0]:sp['train'][1]])
        else:              # reproduce the published (pre-fix) behaviour: thresholds fit on the test window
            fold_cfg.regime.calibrated_vol_thresholds = None
            main_module.calibrate_regime(fold_cfg, ta)
        for op in OPERATORS:
            d = simulate_beliefs(fold_cfg, tf, tp, ta, op)
            d.index = tdates[:len(d)]
            d = d.join(lab)
            d['fold'], d['operator'] = fold, op
            daily_out.append(d.reset_index(names='date'))
            v = d.dropna(subset=['label'])
            y = v['label'].astype(int).to_numpy()
            pred = (v['committed'].astype(int) if op == "hysteresis" else v['argmax']).to_numpy()
            onehot = np.eye(3)[y]
            B = v[['p_bull', 'p_bear', 'p_sideways']].to_numpy()
            gate, bear = v['gate'].to_numpy(bool), (y == 1)
            tp_ = int((gate & bear).sum())
            summary_rows.append({
                'fold': fold, 'operator': op, 'days': len(v),
                'agreement': float((pred == y).mean()),
                'brier': float(((B - onehot) ** 2).sum(axis=1).mean()),
                'gate_precision': tp_ / gate.sum() if gate.sum() else float('nan'),
                'gate_recall': tp_ / bear.sum() if bear.sum() else float('nan'),
                'gate_active_share': float(gate.mean()),
                'entropy_bear': float(v.loc[bear, 'entropy_norm'].mean()) if bear.any() else float('nan'),
                'entropy_bull': float(v.loc[y == 0, 'entropy_norm'].mean()) if (y == 0).any() else float('nan'),
                'confusion': confusion(y, pred), 'label_counts': np.bincount(y, minlength=3)})
    S = pd.DataFrame(summary_rows)

    L += ["## Part A -- Belief validation against prespecified historical proxies (folds 1-3 test windows)", "",
          "> **The proxy labels are PRESPECIFIED HISTORICAL PROXIES, not ground truth.** They were fixed before "
          "looking at results (definition in the header of `r2_belief_validation.py`, no tuning): universe = "
          "equal-weight index of the 5 assets; Bear = trailing-60d return < -10% OR drawdown > 15%; Bull = "
          "trailing-60d return > +10% AND drawdown < 5%; Sideways otherwise. The regime model scores trend and "
          "volatility on the S&P 500 alone, so some disagreement is definitional.", "",
          "> Hysteresis is evaluated by its **committed** regime, softmax by its belief **argmax**; the Brier score "
          "uses each operator's full belief vector vs the one-hot proxy (lower = better calibrated to the proxy; "
          "0 = perfect, 2 = maximally wrong). Gate = P(Bear) > 0.7 OR normalized entropy > 0.5 computed from that "
          "operator's beliefs. **Lower entropy alone is not evidence of better prediction** -- a confidently wrong "
          "operator also has low entropy; read entropy together with agreement and Brier.", "",
          "| fold | operator | days | proxy Bull/Bear/Sideways days | agreement | Brier | gate precision (Bear) | gate recall (Bear) | gate active share | mean entropy, proxy-Bear days | mean entropy, proxy-Bull days |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        c = r.label_counts
        L.append(f"| {r.fold} | {r.operator} | {r.days} | {c[0]}/{c[1]}/{c[2]} | {r.agreement:.3f} | {r.brier:.3f} | "
                 f"{r.gate_precision:.3f} | {r.gate_recall:.3f} | {r.gate_active_share:.3f} | {r.entropy_bear:.3f} | "
                 f"{'n/a (no proxy-Bull days)' if np.isnan(r.entropy_bull) else f'{r.entropy_bull:.3f}'} |")
    L += ["", "Confusion matrices (rows = proxy label, columns = operator's regime call; order Bull, Bear, Sideways):", ""]
    for _, r in S.iterrows():
        m = r.confusion
        L += [f"**Fold {r.fold}, {r.operator}**", "", "| proxy \\ predicted | Bull | Bear | Sideways |", "|---|---|---|---|"]
        for i, name in enumerate(LABELS):
            L.append(f"| {name} | {m[i, 0]} | {m[i, 1]} | {m[i, 2]} |")
        L.append("")
    return labels_all


def part_b(L, labels_all, manifest):
    rows, steps_all = [], []
    cache = {}
    for seed in range(1, 21):
        ckpt = rc.fold2_canonical_checkpoint(seed)
        config = Config.from_json(str(ckpt / "config.json"))
        key = (config.data.start_date, config.data.end_date)
        if key not in cache:
            cache[key] = _load_fold2_slices(config)
            rc.assert_no_fold4(cache[key]['test_aligned'].index)
        sl = cache[key]
        rc.ensure_calibrated(config, sl['train_aligned'])   # audit defect #9
        dates = pd.DatetimeIndex(sl['test_aligned'].index)
        policy = None
        for op in OPERATORS:
            shadow_cfg = copy.deepcopy(config.regime)
            shadow_cfg.belief_formation = OPERATORS[op]
            shadow = create_regime_model(shadow_cfg, input_dim=sl['test_features'].shape[1])
            shield_cfg = copy.deepcopy(config.shield)
            shield_cfg.shield_type = rc.ShieldType.UNCERTAINTY_AWARE
            inner = rc.create_shield(shield_cfg, config.constraints, len(config.data.tickers), config.regime.n_regimes)
            shield = rc.ShadowBeliefGatedShield(inner, shield_cfg, config.constraints, len(config.data.tickers),
                                                config.regime.n_regimes, shadow)
            env = rc.build_env(sl['test_features'], sl['test_prices'], config, sl['test_aligned'], "test", shield)
            rc.attach_shadow_model(env, shadow, sl['test_aligned'])
            if policy is None:
                policy = rc.load_policy_for(ckpt, env.observation_space.shape[0], config)
            st = rc.run_episode(env, policy)
            st['gate_used'] = np.array(shield.gate_log[:len(st)], bool)
            st['date'] = dates[:len(st)]
            st['label'] = labels_all.reindex(st['date'])['label'].to_numpy()
            st['log_ret'] = np.log(st['nav'] / np.concatenate([[1_000_000.0], st['nav'].to_numpy()[:-1]]))
            s = rc.episode_summary(st)
            bear = st['label'] == 1
            missed = bear & ~st['gate_used']
            s.update({'seed': seed, 'gate_operator': op, 'gate_active_share': float(st['gate_used'].mean()),
                      'bear_days': int(bear.sum()), 'missed_stress_days': int(missed.sum()),
                      'missed_stress_share_of_bear': float(missed.sum() / bear.sum()) if bear.sum() else float('nan'),
                      'missed_stress_cum_log_return': float(st.loc[missed, 'log_ret'].sum()),
                      'bear_days_cum_log_return': float(st.loc[bear, 'log_ret'].sum())})
            rows.append(s)
            steps_all.append(st.assign(seed=seed, gate_operator=op))
            print(f"  seed {seed:2d} {op:10s} return {s['return']:+.4f} max_dd {s['max_dd']:.3f} gate "
                  f"{s['gate_active_share']:.3f} missed {s['missed_stress_days']}", flush=True)
        manifest.append({'seed': seed, 'checkpoint': str(ckpt / "final_model.pt"),
                         'sha256': rc.file_sha256(ckpt / "final_model.pt")})
    df = pd.DataFrame(rows)
    df.to_csv(rc.R2_DIR / "belief_validation_fixed_policy.csv", index=False)

    # consistency: hysteresis-gated here must equal E2's gated mode
    e2 = pd.read_csv(rc.R2_DIR / "gated_same_policy.csv")
    e2g = e2[(e2.fold == 2) & (e2['mode'] == "gated")].set_index('seed')['return']
    hy = df[df.gate_operator == "hysteresis"].set_index('seed')['return']
    err = float((hy - e2g.loc[hy.index]).abs().max())
    e2a = e2[(e2.fold == 2) & (e2['mode'] == "always_on")].set_index('seed')['return']
    sm = df[df.gate_operator == "softmax"].set_index('seed')
    err_ao = float((sm['return'] - e2a.loc[sm.index]).abs().max())
    sm_share = float(sm['gate_active_share'].mean())
    daily = pd.read_csv(rc.R2_DIR / "belief_validation_daily.csv.gz")
    min_ent = float(daily[(daily.fold == 2) & (daily.operator == "softmax")].entropy_norm.min())

    metrics = [("return", "net return"), ("max_dd", "max drawdown"), ("mean_gross_exposure", "avg gross exposure"),
               ("gate_active_share", "shield activation (gate-on) share"),
               ("shield_intervention_rate", "days shield modified weights"),
               ("missed_stress_days", "missed-stress days (proxy-Bear, gate off)"),
               ("missed_stress_cum_log_return", "cum. log return on missed-stress days")]
    L += ["## Part B -- Fixed-policy gating comparison: softmax-gated vs hysteresis-gated (fold 2, 20 seeds)", "",
          "Same 20 canonical checkpoints, same regime-gated shield (inner = always-on `uncertainty_aware`, gate "
          "P(Bear) > 0.7 OR entropy > 0.5). **Only the belief operator driving the gate differs.** The policy's "
          "observations and the inner shield's regime inputs stay on the canonical hysteresis beliefs in both arms, "
          "so differences are attributable to gate timing alone.", "",
          f"Consistency check: hysteresis-gated returns vs E2 `gated` mode: max |difference| = {err:.2e} "
          f"({'identical' if err < 1e-9 else 'MISMATCH'}).", "",
          f"Proxy-Bear days in the fold-2 test window: {int(df.bear_days.iloc[0])} of {int(df.n_days.iloc[0])}.", "",
          f"> **This comparison is degenerate on fold 2.** The softmax operator's normalized entropy never falls below "
          f"{min_ent:.3f} in this window, so its gate (entropy > 0.5) is ON on {sm_share:.0%} of days. Softmax-gated is "
          f"therefore identical to the always-on shield (max |return difference| vs E2 always-on = {err_ao:.1e}), and "
          "the table below is, in effect, always-on vs hysteresis-gated. It does not separate 'better belief "
          "timing' from 'shield always engaged'. Of the two gates, only hysteresis ever switches off.", "",
          "| metric | hysteresis-gated mean +/- std | softmax-gated mean +/- std | paired delta (softmax - hysteresis) | 95% CI | paired t p | Wilcoxon p | seeds delta > 0 | MDE (paired, 80%) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for col, name in metrics:
        h = df[df.gate_operator == "hysteresis"].set_index('seed').sort_index()[col]
        s_ = df[df.gate_operator == "softmax"].set_index('seed').sort_index()[col]
        d = (s_ - h).to_numpy()
        if np.allclose(d, d[0]):
            L.append(f"| {name} | {h.mean():+.4f} +/- {h.std(ddof=1):.4f} | {s_.mean():+.4f} +/- {s_.std(ddof=1):.4f} | "
                     f"{d[0]:+.4f} (identical every seed) | -- | deterministic, no test | -- | -- | -- |")
            continue
        st = rc.paired_stats(s_, h)
        L.append(f"| {name} | {h.mean():+.4f} +/- {h.std(ddof=1):.4f} | {s_.mean():+.4f} +/- {s_.std(ddof=1):.4f} | "
                 f"{st['mean_d']:+.4f} | [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}] | {rc.fmt_p(st['t_p'])} | "
                 f"{rc.fmt_p(st['w_p'])} | {st['n_pos']} / {st['n']} | {st['mde']:.4f} |")
    L += ["", "Gate activation and missed-stress days depend on beliefs alone, so they are identical across seeds "
          "within an operator; no test is reported for them. The return, drawdown, exposure and "
          "modification rows carry the seed-level evidence.", ""]
    pd.concat(steps_all, ignore_index=True).to_csv(rc.R2_DIR / "belief_validation_fixed_policy_steps.csv.gz", index=False)


def main():
    t0 = time.time()
    L = ["# R2 / E3 -- Belief validation and fixed-policy gating comparison", "",
         f"_R2 revision, generated {date.today().isoformat()} by `r2_belief_validation.py`. Evaluation only; fold 4 "
         "never loaded (fail-closed guard). Manifest: `belief_validation_manifest.json`._", "",
         ("> **CORRECTED ACCOUNTING (audit defects #8, #9 repaired):** regime thresholds calibrated on each fold's "
          "training window and frozen at test time; corrected-batch checkpoints." if rc.CORRECTED else
          "> **Pipeline caveat:** as in every existing test evaluation, the regime model's volatility-percentile "
          "thresholds are fit on the whole test window (`HeuristicRegimeModel.fit(test_aligned)`), a look-ahead in "
          "calibration. Both operators share it; it is reproduced, not corrected."), ""]
    daily = []
    labels_all = part_a(L, daily)
    pd.concat(daily, ignore_index=True).to_csv(rc.R2_DIR / "belief_validation_daily.csv.gz", index=False)
    manifest = []
    part_b(L, labels_all, manifest)
    (rc.R2_DIR / "belief_validation.md").write_text("\n".join(L), encoding="utf-8")
    rc.write_manifest("belief_validation", {
        'experiment': 'E3 belief validation + fixed-policy gating comparison', 'script': 'r2_belief_validation.py',
        'proxy_definition': 'EW 5-asset index; Bear: trail60<-10% or DD>15%; Bull: trail60>+10% and DD<5%; else Sideways',
        'operators': list(OPERATORS), 'checkpoints_part_b': manifest,
        'runtime_seconds': round(time.time() - t0, 1), 'fold4_loaded': False})
    print(f"Wrote {rc.R2_DIR / 'belief_validation.md'}")


if __name__ == "__main__":
    main()
