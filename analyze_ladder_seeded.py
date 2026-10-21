#!/usr/bin/env python3
"""
Full analysis of the seeded execution ladder (P3 / E6, configs a-h x seeds 1-5)
under corrected costs. Analysis only -- no training. Reads
results/r2/corrected/ladder_seeded.csv, writes
results/r2/corrected/ladder_seeded_analysis.md.

  * per-config mean +- std (and 95% t-CI) of net return, plus per-seed values
  * per-step paired deltas b-a, c-a, d-a, e-d (same seed), mean, SD, 95% t-CI, paired t p
  * 2x2 band x cadence factorial on e/f/g/h:
        band off/daily = e, band on/daily = g, band off/5d = h, band on/5d = f
    per-seed contrasts: band main effect ((f+g)-(e+h))/2, cadence main effect
    ((f+h)-(e+g))/2, interaction (f-h)-(g-e) (band effect at 5d minus band effect daily);
    each with mean, SD, 95% paired t-CI (df=4), paired t p, and a 10k bootstrap CI
Pairing is by seed index (same seed_everything seed per config). Configs differ
in environment/exploration settings, so trajectories diverge from step 0 and
no init-tensor equality across configs is claimed. f is reused from P1
canonical (identical config/seed/code path; see run_r2c.p3).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats

CSV = Path("./results/r2/corrected/ladder_seeded.csv")
OUT_PATH = Path("./results/r2/corrected/ladder_seeded_analysis.md")
CONFIGS = list("abcdefgh")
STEPS = [("b", "a"), ("c", "a"), ("d", "a"), ("e", "d")]
METRICS = [("return", "net return"), ("gross_price_pnl", "gross P&L"),
           ("txn_cost", "cost/NAV"), ("turnover", "turnover")]
N_BOOT = 10_000


def tci(x):
    x = np.asarray(x, float)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return m, s, m - h, m + h


def paired_p(x):
    return stats.ttest_1samp(x, 0.0).pvalue


def boot_ci(x, rng):
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def fmt(v, d=3):
    return f"{v:+.{d}f}"


def main():
    df = pd.read_csv(CSV)
    counts = df.groupby("config").seed.nunique()
    assert all(counts.get(c, 0) == 5 for c in CONFIGS), f"ladder incomplete: {counts.to_dict()}"
    wide = {m: df.pivot(index="seed", columns="config", values=m)[CONFIGS] for m, _ in METRICS}
    R = wide["return"]
    seeds = list(R.index)
    desc = df.drop_duplicates("config").set_index("config").description
    rng = np.random.default_rng(0)
    L = []

    L += ["# Seeded execution ladder (P3 / E6) -- corrected costs, full analysis", "",
          f"Source: `{CSV.as_posix()}` (8 configs x seeds {seeds[0]}-{seeds[-1]}, all complete). "
          "Fold 2 test window. Net return = cumulative net-of-cost return over the test window. "
          "Config f is reused from P1 canonical (identical config, seed, code path). "
          "Pairing is by seed index; configs differ in env/exploration settings, so no init-tensor "
          "equality across configs is claimed. n = 5 per config: CIs are wide and p-values are "
          "descriptive, uncorrected for multiplicity.", ""]

    L += ["## 1. Per-config net return", "",
          "| cfg | description | mean | std | 95% t-CI | " + " | ".join(f"s{s}" for s in seeds) + " |",
          "|---|---|---:|---:|---|" + "---:|" * len(seeds)]
    for c in CONFIGS:
        m, s, lo, hi = tci(R[c])
        L.append(f"| {c} | {desc[c]} | {fmt(m)} | {s:.3f} | [{fmt(lo)}, {fmt(hi)}] | "
                 + " | ".join(fmt(v) for v in R[c]) + " |")
    L += ["", "Secondary metrics (mean +- std):", "",
          "| cfg | gross P&L | cost/NAV | turnover |", "|---|---:|---:|---:|"]
    for c in CONFIGS:
        L.append(f"| {c} | " + " | ".join(f"{wide[m][c].mean():+.3f} +- {wide[m][c].std(ddof=1):.3f}"
                                          for m in ("gross_price_pnl", "txn_cost", "turnover")) + " |")

    L += ["", "## 2. Per-step paired deltas (net return, same seed)", "",
          "| step | mean | SD | 95% t-CI | paired t p | " + " | ".join(f"s{s}" for s in seeds) + " |",
          "|---|---:|---:|---|---:|" + "---:|" * len(seeds)]
    for hi_c, lo_c in STEPS:
        d = R[hi_c] - R[lo_c]
        m, s, lo, hi = tci(d)
        L.append(f"| {hi_c}-{lo_c} | {fmt(m)} | {s:.3f} | [{fmt(lo)}, {fmt(hi)}] | {paired_p(d):.3f} | "
                 + " | ".join(fmt(v) for v in d) + " |")
    L += ["", "Same deltas decomposed (mean, paired): gross P&L / cost/NAV", "",
          "| step | d gross P&L | d cost/NAV | d turnover |", "|---|---:|---:|---:|"]
    for hi_c, lo_c in STEPS:
        L.append(f"| {hi_c}-{lo_c} | " + " | ".join(
            f"{(wide[m][hi_c] - wide[m][lo_c]).mean():+.3f} +- {(wide[m][hi_c] - wide[m][lo_c]).std(ddof=1):.3f}"
            for m in ("gross_price_pnl", "txn_cost", "turnover")) + " |")

    e, f, g, h = (R[k] for k in "efgh")
    L += ["", "## 3. 2x2 factorial: band x cadence (e, f, g, h)", "",
          "Cell means (net return, mean +- std, n=5):", "",
          "| | daily | 5-day cadence | cadence effect within row |", "|---|---:|---:|---:|",
          f"| no band | e: {fmt(e.mean())} +- {e.std(ddof=1):.3f} | h: {fmt(h.mean())} +- {h.std(ddof=1):.3f} | {fmt((h - e).mean())} |",
          f"| band 10% | g: {fmt(g.mean())} +- {g.std(ddof=1):.3f} | f: {fmt(f.mean())} +- {f.std(ddof=1):.3f} | {fmt((f - g).mean())} |",
          f"| band effect within column | {fmt((g - e).mean())} | {fmt((f - h).mean())} | |", "",
          "Per-seed contrasts (paired by seed; df = 4):", "",
          "| effect | definition | mean | SD | 95% t-CI | paired t p | bootstrap 95% CI | "
          + " | ".join(f"s{s}" for s in seeds) + " |",
          "|---|---|---:|---:|---|---:|---|" + "---:|" * len(seeds)]
    contrasts = [
        ("band main", "((f+g)-(e+h))/2", ((f + g) - (e + h)) / 2),
        ("cadence main", "((f+h)-(e+g))/2", ((f + h) - (e + g)) / 2),
        ("interaction", "(f-h)-(g-e)", (f - h) - (g - e)),
        ("band @ daily", "g-e", g - e),
        ("band @ 5d", "f-h", f - h),
        ("cadence @ no band", "h-e", h - e),
        ("cadence @ band", "f-g", f - g),
        ("both vs neither", "f-e", f - e),
    ]
    for name, defn, x in contrasts:
        m, s, lo, hi = tci(x)
        blo, bhi = boot_ci(x, rng)
        L.append(f"| {name} | {defn} | {fmt(m)} | {s:.3f} | [{fmt(lo)}, {fmt(hi)}] | {paired_p(x):.3f} | "
                 f"[{fmt(blo)}, {fmt(bhi)}] | " + " | ".join(fmt(v) for v in x) + " |")
    L += ["", "Note: with n=5 the percentile bootstrap under-covers; the t-CI is the primary interval. "
          "Interaction = (f-h)-(g-e) = f-g-h+e, the difference of the two simple band effects "
          "(unhalved); positive means band and cadence are super-additive.", ""]

    OUT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
