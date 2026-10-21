#!/usr/bin/env python3
"""
R2 / E2 -- same-policy shield modes (eval only, no retraining).

Each existing checkpoint is evaluated under off | always_on | gated
(r2_common.make_shield). Execution semantics are identical across modes (each
checkpoint's own config: k=5 cadence, 10% band, smoothing 0.1, 2.0x cap,
canonical hard-EMA-hysteresis beliefs).

Policies:
  * fold 2 (dev): the n=20 hardening canonical arm (seeds 1-5 grid
    checkpoints, 6-20 hardening checkpoints), data loaded exactly as
    run_hardening.py did (_load_fold2_slices(checkpoint config)) so mode=off
    reproduces hardening_results.csv -- checked in the output.
  * folds 1 and 3: grid none/ppo_lagrangian, seeds 1-5 -- POST-HOC
    SUPPLEMENTARY (grid results already observed).
Exploratory add-on: always_on / gated at (dd_limit, dd_stop) in
{(0.15,0.25), (0.20,0.30), (0.25,0.35)} on fold-2 canonical seeds.

Fold 4 is never loaded (r2_common installs a fail-closed guard).
Outputs: results/r2/gated_same_policy.csv, gated_same_policy_steps.csv.gz,
gated_same_policy_sensitivity.csv, gated_same_policy_summary.md,
gated_same_policy_manifest.json
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

from config import Config
from run_validation_workstream import _load_fold2_slices
from run_rc_overlay_forensics import _load_dev_data, _fold_slices

MODES = ["off", "always_on", "gated", "hybrid"]   # hybrid = trailing-anchor DD scaling always on + gated caps (P4)
SENSITIVITY = [(0.15, 0.25), (0.20, 0.30), (0.25, 0.35)]
_slice_cache = {}


def fold2_slices(config):
    key = (config.data.start_date, config.data.end_date, tuple(config.data.tickers))
    if key not in _slice_cache:
        s = _load_fold2_slices(config)
        rc.assert_no_fold4(s['test_aligned'].index)
        _slice_cache[key] = s
    return _slice_cache[key]


def fold2_policies():
    for seed in range(1, 21):
        ckpt = rc.fold2_canonical_checkpoint(seed)
        yield seed, ckpt, Config.from_json(str(ckpt / "config.json"))


def _eval(policy_dir, config, features, prices, aligned, mode, policy=None):
    env = rc.build_env(features, prices, config, aligned, "test", rc.make_shield(mode, config))
    if policy is None:
        policy = rc.load_policy_for(policy_dir, env.observation_space.shape[0], config)
    steps = rc.run_episode(env, policy)
    steps = steps.assign(day=np.arange(len(steps)), date=pd.DatetimeIndex(aligned.index[:len(steps)]))
    return steps, rc.episode_summary(steps), policy


def main():
    t0 = time.time()
    rc.R2_DIR.mkdir(parents=True, exist_ok=True)
    rows, steps_all, sens_rows, manifest_ckpts = [], [], [], []

    def record(fold, seed, mode, steps, s):
        s.update({'fold': fold, 'seed': seed, 'mode': mode})
        rows.append(s)
        steps_all.append(steps.assign(fold=fold, seed=seed, mode=mode))
        print(f"  fold {fold} seed {seed:2d} {mode:9s} return {s['return']:+.4f} max_dd {s['max_dd']:.3f} "
              f"gate {s['gate_active_share']:.3f} modified {s['shield_intervention_rate']:.3f}", flush=True)

    for seed, ckpt, config in fold2_policies():
        sl = fold2_slices(config)
        rc.ensure_calibrated(config, sl['train_aligned'])   # audit defect #9
        manifest_ckpts.append({'fold': 2, 'seed': seed, 'checkpoint': str(ckpt / "final_model.pt"),
                               'sha256': rc.file_sha256(ckpt / "final_model.pt"),
                               'data_range': [config.data.start_date, config.data.end_date]})
        policy = None
        for mode in MODES:
            st, s, policy = _eval(ckpt, config, sl['test_features'], sl['test_prices'], sl['test_aligned'], mode, policy)
            record(2, seed, mode, st, s)
        # exploratory threshold sensitivity (same policy object, eval-only)
        for dd_soft, dd_hard in SENSITIVITY:
            cfg = copy.deepcopy(config)
            cfg.constraints.dd_limit, cfg.constraints.dd_stop = dd_soft, dd_hard
            for mode in ("always_on", "gated"):
                _, s, _ = _eval(ckpt, cfg, sl['test_features'], sl['test_prices'], sl['test_aligned'], mode, policy)
                s.update({'seed': seed, 'mode': mode, 'dd_soft': dd_soft, 'dd_hard': dd_hard})
                sens_rows.append(s)

    tickers, _, features, prices, aligned_data, calendar_splits = _load_dev_data()
    rc.assert_no_fold4(aligned_data.index)
    folds_13 = [f for f in (1, 3) if all((rc.grid_checkpoint(s, f) / "final_model.pt").exists() for s in range(1, 6))]
    if len(folds_13) < 2:
        print(f"  folds 1/3: grid checkpoints not all present yet -- evaluating folds {folds_13} only", flush=True)
    for fold in folds_13:
        _, test_features, test_prices, test_aligned, _, _ = _fold_slices(
            calendar_splits[fold - 1], features, prices, aligned_data, tickers)
        for seed in range(1, 6):
            ckpt = rc.grid_checkpoint(seed, fold)
            config = Config.from_json(str(ckpt / "config.json"))
            sp = calendar_splits[fold - 1]
            rc.ensure_calibrated(config, aligned_data.iloc[sp['train'][0]:sp['train'][1]])   # audit defect #9
            manifest_ckpts.append({'fold': fold, 'seed': seed, 'checkpoint': str(ckpt / "final_model.pt"),
                                   'sha256': rc.file_sha256(ckpt / "final_model.pt"), 'post_hoc': True})
            policy = None
            for mode in MODES:
                st, s, policy = _eval(ckpt, config, test_features.values, test_prices, test_aligned, mode, policy)
                record(fold, seed, mode, st, s)

    cols = ['fold', 'seed', 'mode', 'return', 'max_dd', 'violation_rate', 'any_violation_rate', 'gate_active_share',
            'shield_intervention_rate', 'mean_gross_exposure', 'mean_turnover', 'n_days']
    df = pd.DataFrame(rows)[cols]
    df.to_csv(rc.R2_DIR / "gated_same_policy.csv", index=False)
    pd.concat(steps_all, ignore_index=True).to_csv(rc.R2_DIR / "gated_same_policy_steps.csv.gz", index=False)
    sens = pd.DataFrame(sens_rows)[['dd_soft', 'dd_hard', 'seed', 'mode', 'return', 'max_dd', 'violation_rate',
                                     'shield_intervention_rate', 'mean_gross_exposure']]
    sens.to_csv(rc.R2_DIR / "gated_same_policy_sensitivity.csv", index=False)

    checks = reproduction_checks(df)
    write_summary(df, sens, checks)
    rc.write_manifest("gated_same_policy", {
        'experiment': 'E2 same-policy shield modes (eval only)', 'script': 'r2_gated_same_policy.py',
        'modes': MODES, 'gate': {'p_bear_gt': rc.GATE_BEAR_THRESHOLD, 'entropy_norm_gt': rc.GATE_ENTROPY_THRESHOLD},
        'sensitivity_settings': SENSITIVITY, 'checkpoints': manifest_ckpts,
        'runtime_seconds': round(time.time() - t0, 1), 'fold4_loaded': False})


def reproduction_checks(df):
    checks = []
    hard = pd.read_csv(rc.HARDENING_DIR / "hardening_results.csv")
    hard = hard[hard.arm == "canonical"].set_index("seed")['return']
    off2 = df[(df.fold == 2) & (df['mode'] == "off")].set_index("seed")['return']
    err = float((off2 - hard.loc[off2.index]).abs().max())
    checks.append(f"- fold 2, mode=off vs `hardening_results.csv` canonical return (20 seeds): max |difference| = "
                  f"{err:.2e} -> {'REPRODUCED' if err < 1e-9 else 'MISMATCH'}")
    if not (rc.GRID_DIR / "results.csv").exists():
        return checks + ["- folds 1/3: grid results.csv not yet available"]
    grid = pd.read_csv(rc.GRID_DIR / "results.csv")
    for fold, window in ((1, "2008-2010"), (3, "2020-2023")):
        if not (df.fold == fold).any():
            checks.append(f"- fold {fold}: not evaluated (grid checkpoints not yet available)")
            continue
        ref = grid[(grid.method == "ppo_lagrangian") & (grid.window == window) & (grid.shield == "none")].set_index("seed")['return']
        mine = df[(df.fold == fold) & (df['mode'] == "off")].set_index("seed")['return']
        err = float((mine - ref.loc[mine.index]).abs().max())
        checks.append(f"- fold {fold}, mode=off vs grid `none/ppo_lagrangian` return (5 seeds): max |difference| = "
                      f"{err:.2e} -> {'REPRODUCED' if err < 1e-9 else 'MISMATCH'}")
    return checks


CAVEAT_MD = ("> **CORRECTED ACCOUNTING (audit defects #8, #9 repaired):** slippage uses daily-scale volatility; regime "
             "thresholds are calibrated on each fold's training window and frozen at test time. Checkpoints are the "
             "corrected-batch retrains (`results/r2/corrected/`)." if rc.CORRECTED else
             "> **Pipeline caveat (all modes, all existing test evaluations):** test envs call "
             "`HeuristicRegimeModel.fit(test_aligned)`, which sets the regime model's volatility-percentile thresholds "
             "from the whole test window -- a look-ahead in calibration. (The price buffer `fit()` also seeds is "
             "cleared by `env.reset()`, so no future prices enter the moving averages.) Reproduced as published.")

GATE_CODE_MD = """## What the regime gate controls (verified in code before evaluating)

- `policy_shield.RegimeGatedShield.apply` computes `gate_active = P(Bear) > 0.7 or H/ln(3) > 0.5` from the
  regime belief and entropy the env passes in (the canonical hysteresis beliefs), and **when the gate is inactive
  returns the proposed weights completely unmodified -- the ENTIRE inner shield is bypassed**, not only its
  regime-dependent adjustment. When active, the full inner shield runs unchanged.
- The inner (= always-on) shield is `UncertaintyAwareShield` with the grid's settings. Its rules, in order:
  (1) gross-leverage cap 2.0 x (1 - 0.3 H/ln3), floor 0.5; (2) per-asset position cap 0.5 x (1 - 0.3 H/ln3),
  floor 0.1; (3) turnover cap 2.0 x (1 - 0.3 H/ln3), floor 0.2; bear cap 0.2 per asset when P(Bear) > 0.7 and
  H < 0.5 ln3; (4) continuous drawdown scaling: exposure x max(0, 1 - (DD - 0.20)/(0.30 - 0.20)) -- i.e. full
  size at DD <= 20% (soft), linear ramp, **zero exposure at DD >= 30% (hard)**. DD is anchored to the
  all-time NAV peak (`dd_anchor="peak"`).
- **Consequence for the 30% hard mechanism under gating: it does NOT remain as in always-on.** It acts only on
  gate-active days. On gate-inactive days the book can hold any exposure regardless of drawdown, so a gated
  run can (and does) exceed 30% drawdown. Under always-on, the peak anchor makes the 30% zero-exposure state
  absorbing: once flat, NAV stops moving, DD stays >= 30%, and exposure stays zero (quantified below).
- **hybrid (fourth mode, corrected batch):** continuous drawdown scaling (20% -> 30% ramp to zero exposure)
  applied EVERY day with the drawdown anchored to the trailing-63-day NAV max (so 30% is not absorbing), plus
  the always-on shield's caps (rules 1-3 and the bear cap) only on gate-active days.
- The 30% level exists only inside the shields; unshielded runs (mode off) are never stopped by it.
  The 20% level is also the drawdown cost-stream threshold used in training (`RiskManager`).
"""


def write_summary(df, sens, checks):
    L = ["# R2 / E2 -- Same-policy shield modes (off | always-on | regime-gated)", "",
         f"_R2 revision, generated {date.today().isoformat()} by `r2_gated_same_policy.py`. Evaluation only, no "
         "retraining. Fold 4 never loaded (fail-closed guard). Manifest: `gated_same_policy_manifest.json`._", "",
         "> **Folds 1 and 3 are POST-HOC SUPPLEMENTARY EVALUATIONS.** Their grid results were observed before this "
         "analysis was designed. Fold 2 (development fold, n=20) is the primary comparison.", "",
         CAVEAT_MD, "",
         GATE_CODE_MD, "## Reproduction checks", ""] + checks + [
         "", "## Per fold x mode", "",
         "Violation rate = share of test days with peak-anchored drawdown > 20%. *Shield modifies weights* = share "
         "of days the shield actually changed the executed target weights.", "",
         "| fold | mode | n | return mean +/- std | 95% CI | mean max DD | violation rate | gate active-day share | shield modifies weights | mean gross exposure |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for fold in (2, 1, 3):
        for mode in MODES:
            g = df[(df.fold == fold) & (df['mode'] == mode)]
            if g.empty:
                continue
            m, sd, (lo, hi) = rc.ci95(g['return'])
            tag = "" if fold == 2 else " (post-hoc)"
            L.append(f"| {fold}{tag} | {mode} | {len(g)} | {m:+.4f} +/- {sd:.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{g['max_dd'].mean():.4f} | {g['violation_rate'].mean():.4f} | {g['gate_active_share'].mean():.4f} | "
                     f"{g['shield_intervention_rate'].mean():.4f} | {g['mean_gross_exposure'].mean():.3f} |")
    L += ["", "Gate active-day share depends on beliefs only (the regime model never sees the action), so it is "
          "identical across modes.", "",
          "## Paired per-seed effects (same policy, same seed; paired t + Wilcoxon)", "",
          "| fold | comparison | metric | n | mean delta | std delta | 95% CI | paired t p | Wilcoxon p | seeds delta > 0 | MDE (paired, 80% power) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fold in (2, 1, 3):
        piv = {m: df[(df.fold == fold) & (df['mode'] == m)].set_index('seed').sort_index() for m in MODES}
        if not piv["off"].shape[0]:
            continue
        for (a, b) in (("gated", "always_on"), ("gated", "off"), ("always_on", "off"),
                       ("hybrid", "always_on"), ("hybrid", "gated"), ("hybrid", "off")):
            for metric in ("return", "max_dd"):
                st = rc.paired_stats(piv[a][metric], piv[b][metric])
                tag = "" if fold == 2 else " (post-hoc)"
                L.append(f"| {fold}{tag} | {a} - {b} | {metric} | {st['n']} | {st['mean_d']:+.4f} | {st['sd_d']:.4f} | "
                         f"[{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}] | {rc.fmt_p(st['t_p'])} | {rc.fmt_p(st['w_p'])} | "
                         f"{st['n_pos']} / {st['n']} | {st['mde']:.4f} |")
    L += ["", "With n=5 (folds 1, 3) the smallest attainable two-sided Wilcoxon p is 0.0625: no fold-1/3 comparison "
          "can reach p<0.05 on Wilcoxon regardless of effect size; their CIs are wide.", ""]
    L += mechanism_section()
    L += sensitivity_section(sens)
    (rc.R2_DIR / "gated_same_policy_summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {rc.R2_DIR / 'gated_same_policy_summary.md'}")


def mechanism_section():
    s = pd.read_csv(rc.R2_DIR / "gated_same_policy_steps.csv.gz")
    L = ["## Mechanism: with the all-time-peak anchor the 30% zero-exposure state is absorbing", "",
         "| fold | mode | seeds reaching >=29.5% DD | mean first day at >=29.5% DD | share of later days with zero gross exposure |",
         "|---|---|---|---|---|"]
    for fold in (2, 1, 3):
        for mode in MODES:
            hits, firsts, flats = 0, [], []
            groups = s[(s.fold == fold) & (s['mode'] == mode)].groupby('seed')
            if groups.ngroups == 0:
                continue
            for _, g in groups:
                g = g.reset_index(drop=True)
                idx = g.index[g.drawdown >= 0.295]  # zero-exposure level is reached at ~0.2999 on some seeds
                if len(idx):
                    hits += 1
                    firsts.append(int(idx[0]))
                    after = g.iloc[idx[0] + 1:]
                    if len(after):
                        flats.append(float((after.leverage < 1e-6).mean()))
            L.append(f"| {fold} | {mode} | {hits} / {groups.ngroups} | {np.mean(firsts) if firsts else float('nan'):.0f} | "
                     f"{np.mean(flats) if flats else float('nan'):.3f} |")
    L += ["", "Under always-on, once drawdown reaches the hard level the book is flat on essentially every remaining "
          "day, so always-on returns mostly reflect WHEN the hard level was hit. Under gated, the shield switches "
          "off once beliefs resolve and the policy re-enters, so drawdowns exceed 30%.", ""]
    return L


def sensitivity_section(sens):
    L = ["## EXPLORATORY ADD-ON -- drawdown threshold sensitivity (eval only, fold 2, 20 seeds)", "",
         "> **Not like-for-like safety comparisons.** The hard boundary itself moves across settings (25% / 30% / "
         "35%), so a lower max DD at a tighter setting is partly definitional. Violation rate here is always "
         "measured at the fixed 20% level for comparability. Exploratory; no hypothesis test was pre-registered.", "",
         "| soft / hard | mode | return mean +/- std | 95% CI | mean max DD | violation rate (DD>20%) | shield modifies weights | mean gross exposure |",
         "|---|---|---|---|---|---|---|---|"]
    for (soft, hard) in SENSITIVITY:
        for mode in ("always_on", "gated"):
            g = sens[(sens.dd_soft == soft) & (sens.dd_hard == hard) & (sens['mode'] == mode)]
            m, sd, (lo, hi) = rc.ci95(g['return'])
            L.append(f"| {soft:.0%} / {hard:.0%} | {mode} | {m:+.4f} +/- {sd:.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{g['max_dd'].mean():.4f} | {g['violation_rate'].mean():.4f} | "
                     f"{g['shield_intervention_rate'].mean():.4f} | {g['mean_gross_exposure'].mean():.3f} |")
    base = sens[(sens.dd_soft == 0.20)]
    L += ["", f"Consistency: the 20%/30% rows reproduce the main table's always-on / gated means "
          f"({base[base['mode'] == 'always_on']['return'].mean():+.4f} / {base[base['mode'] == 'gated']['return'].mean():+.4f}).", ""]
    return L


if __name__ == "__main__":
    main()
