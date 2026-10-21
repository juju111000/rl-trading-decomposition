#!/usr/bin/env python3
"""
INTERIM statistics for the corrected-cost P1 arms (R2C batch), computed while
the batch continues. Analysis only -- no training. Writes
results/r2/corrected/p1_stats_interim.md.

Same methodology as analyze_hardening.py + paired_test_analysis.py, applied to
results/r2/corrected/hardening/hardening_results.csv (n=20/arm):
  * per-arm mean/std/95% t-CI of net return, gross price P&L, turnover, cost/NAV
  * Welch + Mann-Whitney U for the four standard comparisons, MDE @ 80% power
  * paired analysis canonical-vs-untrained on all 20 seeds (every canonical
    run carries init_equal=True from run_r2c's pre-training tensor check),
    net AND gross, with paired MDE @ 80% power
Old (defective-cost, pre-fix #8/#9) values from results/hardening/ are shown
side by side. Old paired analysis was restricted to the 15 verified seeds 6-20.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower

NEW_CSV = Path("./results/r2/corrected/hardening/hardening_results.csv")
OLD_CSV = Path("./results/hardening/hardening_results.csv")
OUT_PATH = Path("./results/r2/corrected/p1_stats_interim.md")
ALPHA = 0.05
POWER = 0.80
ARMS = ["untrained", "canonical", "oracle60", "oracle75"]
METRICS = [("return", "net return"), ("gross_price_pnl", "gross P&L"),
           ("turnover", "turnover (mean daily)"), ("txn_cost", "cost/NAV (cumulative)")]
TEST_METRICS = [("return", "net return"), ("gross_price_pnl", "gross P&L")]
COMPARISONS = [("canonical", "untrained"), ("canonical", "oracle60"),
               ("canonical", "oracle75"), ("oracle75", "oracle60")]
OLD_PAIRED_SEEDS = list(range(6, 21))


def ci95(x):
    n = len(x)
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, (m - h, m + h)


def pooled_sd(a, b):
    na, nb = len(a), len(b)
    return float(np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)))


def mde_ind(sd, n):
    return float(TTestIndPower().solve_power(effect_size=None, nobs1=n, alpha=ALPHA, power=POWER,
                                               ratio=1.0, alternative='two-sided') * sd)


def mde_paired(sd_d, n):
    return float(TTestPower().solve_power(effect_size=None, nobs=n, alpha=ALPHA, power=POWER,
                                            alternative='two-sided') * sd_d)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def verdict(p1, p2):
    if p1 < ALPHA and p2 < ALPHA:
        return "**reject H0** (both)"
    if p1 < ALPHA or p2 < ALPHA:
        return "mixed -- inconclusive"
    return "fail to reject"


def arm_vals(df, arm, metric, seeds=None):
    sub = df[df.arm == arm]
    if seeds is not None:
        sub = sub[sub.seed.isin(seeds)]
    return sub.sort_values("seed")[metric].to_numpy(dtype=float)


def independent(df, a_arm, b_arm, metric):
    a, b = arm_vals(df, a_arm, metric), arm_vals(df, b_arm, metric)
    w = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    sd = pooled_sd(a, b)
    return dict(diff=a.mean() - b.mean(), t=w.statistic, tp=w.pvalue, u=u.statistic, up=u.pvalue,
                sd=sd, mde=mde_ind(sd, min(len(a), len(b))))


def paired(df, metric, seeds):
    can = df[df.arm == "canonical"].set_index("seed").loc[seeds, metric].to_numpy(dtype=float)
    unt = df[df.arm == "untrained"].set_index("seed").loc[seeds, metric].to_numpy(dtype=float)
    d = can - unt
    m, sd, (lo, hi) = ci95(d)
    t = stats.ttest_rel(can, unt)
    w = stats.wilcoxon(d)
    return dict(n=len(d), m=m, sd=sd, lo=lo, hi=hi, t=t.statistic, tp=t.pvalue, w=w.statistic,
                wp=w.pvalue, pos=int((d > 0).sum()), mde=mde_paired(sd, len(d)))


def main():
    new = pd.read_csv(NEW_CSV)
    old = pd.read_csv(OLD_CSV)
    for arm in ARMS:
        assert (new.arm == arm).sum() == 20, f"corrected {arm}: expected 20 rows"
        assert (old.arm == arm).sum() == 20, f"old {arm}: expected 20 rows"
    can = new[new.arm == "canonical"]
    assert can.init_equal.astype(str).str.lower().eq("true").all(), "not all canonical seeds init-verified"
    assert sorted(can.seed) == list(range(1, 21))

    L = ["# P1 statistics -- INTERIM, CORRECTED COSTS", "",
         "> **Interim — corrected costs.** Computed from the completed P1 arms of the R2C batch while "
         "P3/P5/P4/P2' are still running. P1 is complete (20/20 seeds x 4 arms), so these numbers "
         "will not change unless P1 is re-run; \"interim\" refers to the overall R2C results package.",
         "",
         f"- Corrected data: `{NEW_CSV.as_posix()}` (audit fixes #8 slippage units, #9 regime look-ahead applied).",
         f"- Old data (**defective costs**, pre-fix): `{OLD_CSV.as_posix()}`, from `results/hardening/stats_report.md` "
         "and `paired_test_report.md`.",
         f"- Fold 2 test, n=20 seeds/arm. alpha={ALPHA}; MDE at {int(POWER*100)}% power (two-sided, exact "
         "noncentral t via statsmodels) using each comparison's own observed SD.",
         "- `turnover` = mean daily turnover; `cost/NAV` = sum over days of transaction cost / NAV.",
         "- Generated by `analyze_p1_interim.py`. No training was run. Fold 4 not touched.", ""]

    # --- per-arm descriptive ---
    L += ["## 1. Per-arm descriptive statistics (corrected) with old means alongside", "",
          "| arm | metric | mean | std | 95% CI | old mean (defective) | Δ mean (new − old) |",
          "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for metric, label in METRICS:
            m, sd, (lo, hi) = ci95(arm_vals(new, arm, metric))
            om = float(arm_vals(old, arm, metric).mean())
            L.append(f"| {arm} | {label} | {m:+.4f} | {sd:.4f} | [{lo:+.4f}, {hi:+.4f}] | {om:+.4f} | {m - om:+.4f} |")
    L.append("")

    # --- independent tests ---
    L += ["## 2. Independent-samples tests (Welch + Mann-Whitney U), corrected", ""]
    all_mde = []
    for metric, label in TEST_METRICS:
        L += [f"### {label}", "",
              "| comparison | mean diff | Welch t | Welch p | MW U | MW p | conclusion | pooled SD | MDE@80% | "
              "old diff | old Welch p | old MW p | old conclusion |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, b in COMPARISONS:
            r, o = independent(new, a, b, metric), independent(old, a, b, metric)
            all_mde.append(r["mde"])
            L.append(f"| {a} vs {b} | {r['diff']:+.4f} | {r['t']:+.3f} | {fmt_p(r['tp'])} | {r['u']:.1f} | "
                     f"{fmt_p(r['up'])} | {verdict(r['tp'], r['up'])} | {r['sd']:.4f} | {r['mde']:.4f} | "
                     f"{o['diff']:+.4f} | {fmt_p(o['tp'])} | {fmt_p(o['up'])} | {verdict(o['tp'], o['up'])} |")
        L.append("")

    # --- paired ---
    L += ["## 3. Paired analysis: canonical vs untrained", "",
          "Corrected batch: **all 20 seeds** have `init_equal=True` (31/31 tensors bit-identical to the "
          "untrained network before training, checked by `run_r2c`), so d_i = trained_i − untrained_i is a "
          "within-initialization training effect for n=20, df=19. The old analysis could only verify "
          "seeds 6-20 (n=15); old values are shown on those 15. The corrected-data 6-20 subset is included "
          "for a like-for-like comparison.", "",
          "| data | metric | n | mean d | std d | 95% CI | paired t | t p | Wilcoxon W | W p | trained > untrained | paired MDE@80% |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    paired_mde = []
    for metric, label in TEST_METRICS:
        for tag, df, seeds in [("**corrected, seeds 1-20**", new, list(range(1, 21))),
                               ("corrected, seeds 6-20", new, OLD_PAIRED_SEEDS),
                               ("old (defective), seeds 6-20", old, OLD_PAIRED_SEEDS)]:
            p = paired(df, metric, seeds)
            if tag.startswith("**"):
                paired_mde.append(p["mde"])
            L.append(f"| {tag} | {label} | {p['n']} | {p['m']:+.4f} | {p['sd']:.4f} | [{p['lo']:+.4f}, {p['hi']:+.4f}] | "
                     f"{p['t']:+.3f} | {fmt_p(p['tp'])} | {p['w']:.1f} | {fmt_p(p['wp'])} | {p['pos']} / {p['n']} | "
                     f"{p['mde']:.4f} |")
    L.append("")

    # --- power ---
    L += ["## 4. Power statement", "",
          f"Independent comparisons (n=20/group): MDE at {int(POWER*100)}% power ranges "
          f"**{min(all_mde):.3f} to {max(all_mde):.3f}** across the 8 comparison x metric cells. "
          f"Paired canonical-vs-untrained (n=20 pairs): MDE **{min(paired_mde):.3f} to {max(paired_mde):.3f}**. "
          "An observed difference smaller than its own MDE that is non-significant is uninformative about "
          "a smaller real effect -- not evidence of zero effect.", "",
          "## 5. Comparability with old values", "",
          "- Old net-return and cost/NAV figures were computed with the defective slippage units (#8) and "
          "regime features carrying look-ahead (#9); they are **not** valid results, shown only to "
          "quantify the correction's effect.",
          "- Trained arms were re-trained under corrected costs, so new−old differences in trained arms mix "
          "the accounting fix with a different learned policy. The untrained arm (no training) is the closest "
          "to a pure accounting/observation comparison, but old-vs-new untrained init identity was not "
          "verified here.",
          "- Old paired results cover seeds 6-20 only; the new n=20 paired result has no old counterpart.", ""]

    OUT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT_PATH}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
