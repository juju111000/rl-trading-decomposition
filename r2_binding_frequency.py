#!/usr/bin/env python3
"""
R2 / E1 -- units, binding frequency, thresholds (analysis only).

Training side: each canonical seed's training_log.jsonl (seeds 1-5 grid
fold-2 none/ppo_lagrangian; 6-20 hardening canonical_training). The logs keep
per-iteration MEAN excess per stream (2048 steps) + lambdas, not per-step
values; exact per-step training counts are recorded by the E4 retrain of
seeds 1-5 (run_r2.py) and are appended when E4 completes.

Evaluation side: exact, from E2's per-day records (mode=off == canonical
deployment; reproduces hardening_results.csv to 1e-16).

Unbanded daily reference: checkpoint_50k fold-2 none/ppo_lagrangian
(k=1, no band/smoothing/cap), re-evaluated.

Fold 4 never loaded (fail-closed guard in r2_common).
Output: results/r2/binding_frequency.md (+ per-seed CSV, manifest).
"""
import json
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

STREAMS = [("drawdown", "dd", "cost_dd", "drawdown > 0.20"),
           ("turnover", "turnover", "cost_turnover", "turnover > 1.00 (sum |dw|, per step)"),
           ("leverage", "leverage", "cost_leverage", "gross leverage > 1.5x")]
UNBANDED_CKPT = Path("./results/checkpoint_50k/20260731_144255/fold_2/none/ppo_lagrangian")

CODE_TRACE_MD = """## Code trace: what the turnover constraint measures

1. **Weights.** The policy action is decoded to target weights `w_target` (`trading_env._decode_action`: per-asset
   clip to +/-0.8, optional gross cap). On a decision day the execution model then applies smoothing
   (`w = 0.9 w_prev_target + 0.1 a`), the per-asset 10% no-trade band, and the shield (if any). On non-decision
   days (k=5 cadence) the target is the current drifted weights, so nothing trades.
2. **Executed trades.** `portfolio_engine.Portfolio.compute_turnover(w_safe, prices)` =
   **sum_i |w_safe,i - w_current,i|** -- the L1 change in portfolio weights, i.e. **two-sided** turnover as a
   fraction of NAV (selling 50% of NAV and buying 50% elsewhere = 1.00). It is computed once per env step, and one
   env step is one trading day, so on non-decision days it is ~0.
3. **Cost stream c2.** `RiskManager.compute_costs`: `cost_turnover = max(0, turnover - turnover_limit)` with
   `turnover_limit = 1.0`. **So the threshold is 100% of NAV of two-sided weight change *within a single
   step (day)*, i.e. 50% of NAV one-way, per decision -- not a daily average and not an annualized figure.**
4. **Dual update.** Each PPO iteration (2048 steps), `PPOLagrangianTrainer.update_dual_variables`:
   `lambda_k <- clip(lambda_k + dual_lr * (mean_k(c) - d_k), 0, lambda_max)` with `d_k = 0`, `dual_lr = 0.005`.
5. **How lambda enters the policy.** `update_policy`: `A = A_r - sum_k lambda_k A_c,k`, where `A_r` and each
   `A_c,k` are **separately normalized to zero mean / unit std** within the rollout (`RolloutBuffer.get`,
   normalization applied to a cost stream whenever its advantage std > 1e-8). The penalty's weight in the
   gradient is therefore ~lambda_k relative to the reward advantage, *independent of the raw cost magnitude*.

Other two streams: `cost_dd = max(0, DD - 0.20)` with DD anchored to the all-time NAV peak;
`cost_leverage = max(0, gross leverage - 1.5)`.

## Soft (20%) vs hard (30%) drawdown thresholds, in code

- **20% (`ConstraintConfig.dd_limit`)** -- (a) the drawdown cost-stream threshold the dual trains against
  (`RiskManager`, all trainer arms, shielded or not); (b) inside `UncertaintyAwareShield`, the point where its
  continuous drawdown scaling starts (exposure x max(0, 1 - (DD - 0.20)/0.10)).
- **30% (`ConstraintConfig.dd_stop`)** -- used **only by shields**: `UncertaintyAwareShield` reaches zero exposure
  at DD >= 30%; `RuleBasedShield`/`ProjectionShield` zero the action at DD >= 30%. **The environment never
  terminates or acts on 30%, and the canonical (unshielded) arm is never affected by it.**
- Operational effect with the all-time-peak anchor (E2): once a shielded book hits 30% it goes flat and stays
  flat (NAV stops moving, DD never falls back) -- the hard level is absorbing. No thresholds changed here.
"""


def training_side():
    per_seed = []
    for seed in range(1, 21):
        ckpt = rc.fold2_canonical_checkpoint(seed)
        recs = [json.loads(l) for l in open(ckpt / "training_log.jsonl")]
        mc = np.array([r['mean_costs'] for r in recs], dtype=float)
        lam = np.array([r['lambdas'] for r in recs], dtype=float)
        row = {'seed': seed, 'iterations': len(recs), 'timesteps': recs[-1]['total_timesteps'],
               'checkpoint': str(ckpt / "final_model.pt")}
        for k, (_, short, _, _) in enumerate(STREAMS):
            nz = mc[:, k] > 0
            row[f'{short}_iter_share_nonzero'] = float(nz.mean())
            row[f'{short}_mean_excess_all_steps'] = float(mc[:, k].mean())
            row[f'{short}_mean_iter_excess_when_nonzero'] = float(mc[nz, k].mean()) if nz.any() else 0.0
            row[f'{short}_initial_lambda'] = float(lam[0, k])
            row[f'{short}_final_lambda'] = float(lam[-1, k])
            row[f'{short}_max_lambda'] = float(lam[:, k].max())
        per_seed.append(row)
    return pd.DataFrame(per_seed)


def unbanded_turnover():
    cfg = Config.from_json(str(UNBANDED_CKPT / "config.json"))
    sl = _load_fold2_slices(cfg)
    rc.assert_no_fold4(sl['test_aligned'].index)
    env = rc.build_env(sl['test_features'], sl['test_prices'], cfg, sl['test_aligned'], "test", None)
    policy = rc.load_policy_for(UNBANDED_CKPT, env.observation_space.shape[0], cfg)
    steps = rc.run_episode(env, policy)
    return cfg, steps, rc.episode_summary(steps)


def _dist_row(label, x, thr=1.0):
    x = np.asarray(x, float)
    return (f"| {label} | {len(x)} | {x.mean():.4f} | {np.median(x):.4f} | {np.quantile(x, 0.95):.4f} | "
            f"{x.max():.4f} | {(x > thr).mean():.4f} |")


def main():
    t0 = time.time()
    tr = training_side()
    tr.to_csv(rc.R2_DIR / "binding_frequency_training_per_seed.csv", index=False)
    s = pd.read_csv(rc.R2_DIR / "gated_same_policy_steps.csv.gz")
    ev = s[(s.fold == 2) & (s['mode'] == "off")].copy()
    ev['decision_day'] = ev['day'] % 5 == 0   # canonical k=5; step 0 of the test episode is a decision day
    ub_cfg, ub, ub_sum = unbanded_turnover()

    L = ["# R2 / E1 -- Units, constraint binding, thresholds (canonical, fold 2, n=20)", "",
         f"_R2 revision, generated {date.today().isoformat()} by `r2_binding_frequency.py`. Analysis only; no "
         "thresholds changed; fold 4 never loaded (fail-closed guard). Manifest: `binding_frequency_manifest.json`._", "",
         CODE_TRACE_MD,
         "## Training: binding frequency and duals (`training_log.jsonl`, 20 seeds x 146 iterations x 2048 steps)", "",
         "> **Limitation (step-level frequency).** The logs keep each iteration's mean excess, not per-step values, so "
         "the fraction of individual training *steps* with nonzero excess is not recoverable for these 20 runs. Exact "
         "here: share of iterations whose rollout had any nonzero excess (iteration-scale binding) and the mean "
         "per-step excess over all training steps. **Exact per-step fractions are being logged for the E4 retrain of "
         "canonical seeds 1-5 (instrumented `env.step`, RNG-neutral) and will be appended when E4 completes.**", "",
         "| stream | iterations with nonzero excess (mean +/- std) | mean per-step excess (all steps) | mean iteration-excess when nonzero | lambda initial | lambda final (mean +/- std) | max lambda change over training |",
         "|---|---|---|---|---|---|---|"]
    for _, short, _, desc in STREAMS:
        a = tr[f'{short}_iter_share_nonzero']
        lf = tr[f'{short}_final_lambda']
        L.append(f"| {desc} | {a.mean():.3f} +/- {a.std(ddof=1):.3f} | {tr[f'{short}_mean_excess_all_steps'].mean():.2e} | "
                 f"{tr[f'{short}_mean_iter_excess_when_nonzero'].mean():.2e} | {tr[f'{short}_initial_lambda'].iloc[0]:.3f} | "
                 f"{lf.mean():.4f} +/- {lf.std(ddof=1):.4f} | {(tr[f'{short}_max_lambda'] - tr[f'{short}_initial_lambda']).max():+.4f} |")
    L += ["", "### Are the duals operationally active?", "",
          "- **Drawdown and leverage: the penalty acts, the dual barely adapts.** Each stream has nonzero excess in "
          "most rollouts, so its normalized cost advantage enters the policy gradient with weight ~lambda "
          f"(~0.40 and ~0.50-0.52 relative to the unit-variance reward advantage). But lambda moves very little: "
          f"drawdown {tr.dd_initial_lambda.iloc[0]:.3f} -> {tr.dd_final_lambda.mean():.4f}, leverage "
          f"{tr.leverage_initial_lambda.iloc[0]:.3f} -> {tr.leverage_final_lambda.mean():.4f} (dual_lr 0.005 x mean "
          "excess ~1e-3 to 3e-2 per iteration). Functionally these are near-fixed penalties at their warm-start "
          "values, not adaptively tuned multipliers.",
          "- **Turnover: the dual is inactive.** Raw turnover excess is exactly zero in every iteration of every seed, so "
          "lambda_turnover never moves from 0.40. Its cost advantage is then built only from the cost critic's value "
          "estimates (no cost signal), and `RolloutBuffer.get` still rescales it to unit variance whenever its std "
          "exceeds 1e-8, so a lambda-weighted signal unrelated to turnover can enter the gradient. The E4 retrain "
          "logs the raw pre-normalization advantage std per stream to size this.", "",
          "Final lambda per seed (seeds 1-5: grid checkpoints; 6-20: hardening):", "",
          "| seed | lambda_dd | lambda_turnover | lambda_leverage |", "|---|---|---|---|"]
    for _, r in tr.iterrows():
        L.append(f"| {int(r.seed)} | {r.dd_final_lambda:.4f} | {r.turnover_final_lambda:.4f} | {r.leverage_final_lambda:.4f} |")

    L += ["", "## Evaluation days (fold-2 test 2015-2017, deterministic, no shield; 20 seeds x 779 days)", "",
          "| stream | share of test days with nonzero excess (pooled) | per-seed mean +/- std | mean excess when nonzero (pooled) |",
          "|---|---|---|---|"]
    for _, short, col, desc in STREAMS:
        c = ev[col]
        per = ev.groupby('seed')[col].apply(lambda x: (x > 0).mean())
        L.append(f"| {desc} | {(c > 0).mean():.4f} | {per.mean():.4f} +/- {per.std(ddof=1):.4f} | "
                 f"{c[c > 0].mean() if (c > 0).any() else 0.0:.4f} |")

    dec = ev[ev.decision_day]
    traded = ev[ev.turnover > 0]
    L += ["", "## Realized turnover vs the actual threshold (1.00 = two-sided sum |dw| within one step)", "",
          "Per-step values; for canonical, a step is one trading day and trades happen only on decision days "
          "(every 5th day) -- daily averages over all days dilute the per-decision figure by ~5x.", "",
          "| config | steps | mean | median | p95 | max | share of steps > 1.00 |", "|---|---|---|---|---|---|---|",
          _dist_row("canonical, all test days (daily), 20 seeds pooled", ev.turnover),
          _dist_row("canonical, decision days only (per decision step)", dec.turnover),
          _dist_row("canonical, steps with any trade", traded.turnover),
          _dist_row("unbanded daily config (k=1: every day is a decision step), checkpoint_50k, 1 seed", ub.turnover),
          "", f"Per-seed mean daily turnover, canonical: {ev.groupby('seed').turnover.mean().mean():.4f} +/- "
          f"{ev.groupby('seed').turnover.mean().std(ddof=1):.4f}. Unbanded run's test return {ub_sum['return']:+.4f} "
          f"(max DD {ub_sum['max_dd']:.4f}) -- a catastrophic loss, recorded as a result.", "",
          "**Location relative to the threshold.** Canonical's largest single-step turnover is well below 1.00, so "
          "the turnover constraint never binds; turnover is controlled by the execution model (band + 5-day "
          "cadence), not by the dual. The unbanded daily config averages ~0.49 per day -- about half the threshold "
          "-- and exceeds 1.00 on ~7% of days (its p95 is just above 1.00).", "",
          "**Caveats on the unbanded reference.** Single seed (1), 50k timesteps (not 300k), and it predates the "
          "^STOXX50E -> ^GDAXI universe change (its saved config loads the old universe), re-evaluated on current "
          "environment code. Its measured mean matches the ~49%/day figure in the project notes. E6 config (a) "
          "retrains this execution model from scratch on the current universe (5 seeds) for a like-for-like number.", "",
          "> **Pipeline caveat:** test envs fit the regime model's volatility-percentile thresholds on the whole test "
          "window (`HeuristicRegimeModel.fit(test_aligned)`) -- a look-ahead in calibration that reaches the policy "
          "through its regime-belief inputs. Reproduced as published. See `belief_validation.md`.", ""]
    (rc.R2_DIR / "binding_frequency.md").write_text("\n".join(L), encoding="utf-8")
    rc.write_manifest("binding_frequency", {
        'experiment': 'E1 units/binding/thresholds (analysis only)', 'script': 'r2_binding_frequency.py',
        'training_logs': tr[['seed', 'checkpoint']].to_dict('records'),
        'eval_source': 'results/r2/gated_same_policy_steps.csv.gz (fold 2, mode off)',
        'unbanded_checkpoint': str(UNBANDED_CKPT / "final_model.pt"),
        'unbanded_sha256': rc.file_sha256(UNBANDED_CKPT / "final_model.pt"),
        'runtime_seconds': round(time.time() - t0, 1), 'fold4_loaded': False})
    print(f"Wrote {rc.R2_DIR / 'binding_frequency.md'}")


if __name__ == "__main__":
    main()
