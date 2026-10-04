#!/usr/bin/env python3
"""
Statistics for the hardening session's seed-expansion results
(results/hardening/hardening_results.csv, n=20/arm). Writes
results/hardening/stats_report.md.

Per arm: mean, std, 95% CI (t-based) of net return and gross (frictionless)
price P&L. Welch's t-test AND Mann-Whitney U, on both metrics, for:
canonical-vs-untrained, canonical-vs-oracle60, canonical-vs-oracle75,
oracle75-vs-oracle60. Conclusion at alpha=0.05 for each, plus the minimum
detectable effect (MDE) at n=20/group given each comparison's own observed
(pooled) variance, alpha=0.05, power=0.80 (stated explicitly -- the task
didn't pin a power target, 0.80 is the standard default).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path("./results/hardening/hardening_results.csv")
OUT_PATH = Path("./results/hardening/stats_report.md")
FIGURE_PATH = Path("./results/hardening/hardening_distributions.png")
ALPHA = 0.05
POWER_TARGET = 0.80
METRICS = ["return", "gross_price_pnl"]
ARMS = ["canonical", "untrained", "oracle60", "oracle75"]
ARM_ORDER = ["untrained", "canonical", "oracle60", "oracle75"]  # figure x-axis order, per spec
ARM_COLORS = {"untrained": "#4C72B0", "canonical": "#DD8452", "oracle60": "#55A868", "oracle75": "#C44E52"}

COMPARISONS = [
    ("canonical", "untrained"),
    ("canonical", "oracle60"),
    ("canonical", "oracle75"),
    ("oracle75", "oracle60"),
]


def ci95_t(x: np.ndarray):
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, sd, (mean - t_crit * se, mean + t_crit * se)


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    return float(np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)))


def mde_at_n(sd_pooled: float, n_per_group: int, alpha: float = ALPHA, power: float = POWER_TARGET) -> float:
    """Minimum detectable difference in raw units, given the observed pooled
    SD, for a two-sided two-sample t-test at the given alpha/power/n."""
    analysis = TTestIndPower()
    d = analysis.solve_power(effect_size=None, nobs1=n_per_group, alpha=alpha,
                              power=power, ratio=1.0, alternative='two-sided')
    return float(d * sd_pooled)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def plot_return_distributions(df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df.loc[df['arm'] == arm, 'return'].to_numpy() for arm in ARM_ORDER]
    positions = np.arange(1, len(ARM_ORDER) + 1)

    bp = ax.boxplot(data, positions=positions, widths=0.5, showfliers=False, patch_artist=True,
                     medianprops=dict(color='#333333', linewidth=1.5),
                     whiskerprops=dict(color='#333333'), capprops=dict(color='#333333'))
    for patch, arm in zip(bp['boxes'], ARM_ORDER):
        patch.set_facecolor(ARM_COLORS[arm])
        patch.set_alpha(0.35)
        patch.set_edgecolor(ARM_COLORS[arm])

    rng = np.random.RandomState(0)
    for pos, arm, values in zip(positions, ARM_ORDER, data):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(pos + jitter, values, color=ARM_COLORS[arm], s=24, alpha=0.85,
                   edgecolor='white', linewidth=0.5, zorder=3, label=f"{arm} (n={len(values)})")

    ax.axhline(0.0, color='#888888', linestyle='--', linewidth=1, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(ARM_ORDER)
    ax.set_ylabel("Net return (fold 2 test)")
    ax.set_title("Net return distribution by arm (n=20 seeds each)", fontsize=12, pad=10)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    df = pd.read_csv(CSV_PATH)
    assert set(df['arm'].unique()) == set(ARMS), f"unexpected arms: {df['arm'].unique()}"
    for arm in ARMS:
        n = (df['arm'] == arm).sum()
        assert n == 20, f"{arm} has {n} rows, expected 20"

    lines = []
    lines.append("# Hardening session: statistics report\n")
    lines.append(f"n=20 seeds/arm, from `results/hardening/hardening_results.csv`. "
                  f"alpha={ALPHA}, MDE computed at power={POWER_TARGET} (not specified in the "
                  f"original spec -- 0.80 is the standard default, stated explicitly here since it "
                  f"changes the MDE numbers).\n")

    # --- Part 1: per-arm descriptive stats ---
    lines.append("## Per-arm mean / std / 95% CI (t-based)\n")
    lines.append("| arm | metric | n | mean | std | 95% CI |")
    lines.append("|---|---|---|---|---|---|")
    per_arm_stats = {}
    for arm in ARMS:
        per_arm_stats[arm] = {}
        for metric in METRICS:
            x = df.loc[df['arm'] == arm, metric].to_numpy()
            mean, sd, (lo, hi) = ci95_t(x)
            per_arm_stats[arm][metric] = {'mean': mean, 'sd': sd, 'ci': (lo, hi), 'x': x}
            lines.append(f"| {arm} | {metric} | 20 | {mean:+.4f} | {sd:.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines.append("")

    # --- Part 1b: mean turnover / txn cost per arm (descriptive only) ---
    lines.append("## Mean turnover and transaction cost per arm\n")
    lines.append("| arm | n | mean turnover | std turnover | mean txn_cost | std txn_cost |")
    lines.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        turnover = df.loc[df['arm'] == arm, 'turnover'].to_numpy()
        txn_cost = df.loc[df['arm'] == arm, 'txn_cost'].to_numpy()
        lines.append(
            f"| {arm} | 20 | {np.mean(turnover):.4f} | {np.std(turnover, ddof=1):.4f} | "
            f"{np.mean(txn_cost):.4f} | {np.std(txn_cost, ddof=1):.4f} |"
        )
    lines.append("")

    # --- Part 2: hypothesis tests ---
    lines.append("## Hypothesis tests\n")
    all_mdes = []
    for metric in METRICS:
        lines.append(f"### Metric: `{metric}`\n")
        lines.append("| comparison | mean diff (A-B) | Welch t | Welch p | Mann-Whitney U | MW p | "
                      "conclusion (alpha=0.05) | pooled SD | MDE @ n=20 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for arm_a, arm_b in COMPARISONS:
            a = per_arm_stats[arm_a][metric]['x']
            b = per_arm_stats[arm_b][metric]['x']
            mean_diff = float(np.mean(a) - np.mean(b))

            t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative='two-sided')

            significant = (t_p < ALPHA) and (u_p < ALPHA)
            either_significant = (t_p < ALPHA) or (u_p < ALPHA)
            if significant:
                conclusion = f"**reject H0** (both tests p<{ALPHA})"
            elif either_significant:
                conclusion = f"mixed (one test p<{ALPHA}, one not) -- treat as inconclusive"
            else:
                conclusion = f"fail to reject H0 (both p>={ALPHA})"

            sd_pool = pooled_sd(a, b)
            mde = mde_at_n(sd_pool, n_per_group=20)
            all_mdes.append(mde)

            lines.append(
                f"| {arm_a} vs {arm_b} | {mean_diff:+.4f} | {t_stat:+.3f} | {fmt_p(t_p)} | "
                f"{u_stat:.1f} | {fmt_p(u_p)} | {conclusion} | {sd_pool:.4f} | {mde:.4f} |"
            )
        lines.append("")

    lines.append("## Power statement\n")
    lines.append(
        f"At n=20/group, alpha={ALPHA}, {int(POWER_TARGET*100)}% power, and GIVEN each comparison's "
        f"own observed pooled standard deviation (exact noncentral-t solution via "
        f"`statsmodels.stats.power.TTestIndPower`, not the normal approximation), the minimum true "
        f"difference in means this design could reliably detect ranges **{min(all_mdes):.3f} to "
        f"{max(all_mdes):.3f}** across the 8 comparison/metric combinations below (see the `MDE @ "
        f"n=20` column for the exact figure per comparison). A comparison's observed `mean diff` "
        f"smaller than its own MDE means: even if a real effect of exactly that size existed, this "
        f"sample size had at best even-odds of detecting it -- the observed non-significance there "
        f"is uninformative about whether a real, smaller effect exists, not evidence the effect is "
        f"zero.\n"
    )

    lines.append("## Distribution figure\n")
    lines.append(f"![Net return distributions by arm]({FIGURE_PATH.name})\n")
    plot_return_distributions(df, FIGURE_PATH)
    print(f"Wrote {FIGURE_PATH}")

    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
