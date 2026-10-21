#!/usr/bin/env python3
"""
Paper exhibits from the corrected-accounting results (analysis/plots only, no
runs). Reads ONLY results/r2/corrected/* (incl. fold4/) and writes
results/paper_exhibits/: one CSV per table, figures as 600-dpi PNG + vector
PDF without embedded titles (captions carry them), and exhibits_manifest.md
listing sources and the at-target-size figure QA. Style, semantic palette and
sizes come from scripts/paper_style.py. Returns in tables are in percent.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

C = Path("./results/r2/corrected")
OUT = Path("./results/paper_exhibits")
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from paper_style import apply, PALETTE as P, SIZES, FS, SECONDARY_LS, SECONDARY_HATCH, save, qa  # noqa: E402
apply()
FOLD_OF = {"2008-2010": 1, "2015-2017": 2, "2020-2023": 3}
FOLD_LAB = {1: "Fold 1\n2008–10", 2: "Fold 2\n2015–17", 3: "Fold 3\n2020–23"}
MANIFEST, QA = [], []


def save_table(df, name, source, note=""):
    df.to_csv(OUT / f"{name}.csv", index=False)
    MANIFEST.append((f"{name}.csv", source, note))
    print(f"\n=== {name} ===  ({source})")
    print(df.to_string(index=False))


def save_fig(fig, name, source, note=""):
    ok, rep = qa(fig, name)
    QA.extend(rep)
    save(fig, str(OUT / name))
    plt.close(fig)
    MANIFEST.append((f"{name}.png / .pdf", source, note))


def pct(x, d=1):
    return f"{100 * x:+.{d}f}"


def msd(x, d=1):
    return f"{100 * np.mean(x):+.{d}f} ± {100 * np.std(x, ddof=1):.{d}f}"


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ============================================================================ tables

def t4():
    rep = (C / "FINAL_REPORT.md").read_text(encoding="utf-8")
    sec = rep.split("## S4")[1].split("## S5")[0]
    rows = []
    for line in sec.splitlines():
        if line.startswith("| ") and line[2].isdigit():
            cells = [c.strip() for c in line.strip("|").split("|")]
            verdict = cells[4].split(":")[0].replace("*", "").strip()
            rows.append({'#': int(cells[0]), 'component / claim': cells[1].split(":")[0].replace("*", ""),
                         'claim': cells[1].split(":", 1)[1].strip() if ":" in cells[1] else "",
                         'submitted (pre-correction)': cells[2], 'corrected': cells[3], 'verdict': verdict,
                         'verdict detail': cells[4].split(":", 1)[1].strip() if ":" in cells[4] else ""})
    df = pd.DataFrame(rows)
    assert len(df) == 9
    save_table(df, "T4_component_summary", "FINAL_REPORT.md S4 (verbatim cells)")


def t5():
    h = pd.read_csv(C / "hardening" / "hardening_results.csv")
    can = h[h.arm == "canonical"].set_index("seed").sort_index()
    unt = h[h.arm == "untrained"].set_index("seed").sort_index()
    assert list(can.index) == list(unt.index) == list(range(1, 21)) and can.init_equal.all()
    rows, ps = [], []
    for m, lab, scale in (("return", "net return (%)", 100), ("gross_price_pnl", "gross P&L (%)", 100),
                          ("turnover", "turnover (% NAV / day)", 100), ("txn_cost", "cost / NAV (%)", 100)):
        d = can[m] - unt[m]
        lo, hi = tci(d)
        p_t, p_w = stats.ttest_1samp(d, 0).pvalue, stats.wilcoxon(d).pvalue
        ps.append((p_t, p_w))
        rows.append({'metric': lab, 'trained mean ± sd': f"{scale * can[m].mean():.2f} ± {scale * can[m].std():.2f}",
                     'untrained mean ± sd': f"{scale * unt[m].mean():.2f} ± {scale * unt[m].std():.2f}",
                     'paired diff (trained − untrained)': f"{scale * d.mean():+.2f}",
                     '95% CI': f"[{scale * lo:+.2f}, {scale * hi:+.2f}]",
                     'paired t p': fp(p_t), 'Wilcoxon p': fp(p_w),
                     'seeds trained > untrained': f"{int((d > 0).sum())}/20"})
    ht = multipletests([p[0] for p in ps], method="holm")[1]
    hw = multipletests([p[1] for p in ps], method="holm")[1]
    for r, a, b in zip(rows, ht, hw):
        r['Holm p (t / Wilcoxon)'] = f"{fp(a)} / {fp(b)}"
    save_table(pd.DataFrame(rows), "T5_trained_vs_untrained",
               "hardening/hardening_results.csv (P1, fold 2, n=20 paired, init tensor-equal 20/20)",
               "Holm family = the 4 rows")


def t6():
    h = pd.read_csv(C / "hardening" / "hardening_results.csv")
    p = pd.read_csv(C / "p2prime" / "p2prime_results.csv")
    old = pd.read_csv(C / "oracle_anomaly_probe.csv")
    new = pd.read_csv(C / "final_p2prime_probe.csv")
    rep = (C / "FINAL_REPORT.md").read_text(encoding="utf-8")
    anom = (C / "oracle_anomaly.md").read_text(encoding="utf-8")

    def holm_from(text, comp, metric="net"):
        for line in text.splitlines():
            c = [x.strip() for x in line.strip("|").split("|")]
            if len(c) > 6 and c[0] == comp and c[1] == metric:
                return c
        raise KeyError(comp)

    o60 = holm_from(anom, "canonical vs oracle60")
    o75 = holm_from(anom, "canonical vs oracle75")
    n100 = holm_from(rep, "oracle100_nosmooth vs canonical_nosmooth")
    n60 = holm_from(rep, "oracle60_nosmooth vs canonical_nosmooth")
    ceil = np.log1p(223.0598)

    def row(execution, signal, arm, df, armkey, probe, pkey, corrcol, agcol, holm, ref):
        x = df[df.arm == armkey]
        pr = probe[probe.arm == pkey] if probe is not None else None
        lg = np.log1p(x['return']).mean()
        return {'execution': execution, 'signal observed': signal, 'arm': arm, 'n': len(x),
                'net return % (mean ± sd)': msd(x['return']), 'gross P&L % (mean ± sd)': msd(x['gross_price_pnl']),
                'vs no-signal arm (net, pp)': ref,
                'Holm p (Welch / MWU)': holm,
                'corr(action, oracle feature)': f"{pr[corrcol].mean():+.3f}" if pr is not None else "n/a",
                'sign agreement': f"{pr[agcol].mean():.3f}" if pr is not None else "n/a",
                'share of ceiling log-growth': f"{lg / ceil:.0%}" if execution.startswith("no smoothing") and
                armkey == "oracle100_5d_nosmooth" else "n/a"}

    can = h[h.arm == "canonical"]['return'].mean()
    cn = p[p.arm == "canonical_nosmooth"]['return'].mean()
    rows = [
        row("canonical (smoothing 0.1)", "none", "canonical", h, "canonical", None, None, None, None, "—", "—"),
        row("canonical (smoothing 0.1)", "1-day sign, 60%", "oracle60", h, "oracle60", old, "oracle60",
            'corr_action_oracle', 'sign_agree_action_oracle', f"{o60[4]} / {o60[6]}",
            f"{100 * (h[h.arm == 'oracle60']['return'].mean() - can):+.1f}"),
        row("canonical (smoothing 0.1)", "1-day sign, 75%", "oracle75", h, "oracle75", old, "oracle75",
            'corr_action_oracle', 'sign_agree_action_oracle', f"{o75[4]} / {o75[6]}",
            f"{100 * (h[h.arm == 'oracle75']['return'].mean() - can):+.1f}"),
        row("no smoothing", "none", "canonical_nosmooth", p, "canonical_nosmooth", None, None, None, None, "—", "—"),
        row("no smoothing", "5-day sign, 100%", "oracle100_nosmooth", p, "oracle100_5d_nosmooth", new,
            "oracle100_5d_nosmooth", 'corr_action_oracle_feature', 'sign_agree_oracle_feature',
            f"{n100[6]} / {n100[8]}", f"{100 * (p[p.arm == 'oracle100_5d_nosmooth']['return'].mean() - cn):+.1f}"),
        row("no smoothing", "5-day sign, 60%", "oracle60_nosmooth", p, "oracle60_5d_nosmooth", new,
            "oracle60_5d_nosmooth", 'corr_action_oracle_feature', 'sign_agree_oracle_feature',
            f"{n60[6]} / {n60[8]}", f"{100 * (p[p.arm == 'oracle60_5d_nosmooth']['return'].mean() - cn):+.1f}"),
    ]
    save_table(pd.DataFrame(rows), "T6_signal_expressibility_exploitation",
               "hardening_results.csv, p2prime_results.csv, oracle_anomaly(.md/_probe.csv), FINAL_REPORT.md S1, "
               "final_p2prime_probe.csv",
               "Holm families: canonical rows = oracle_anomaly 8-test family; no-smoothing rows = FINAL_REPORT S1 "
               "8-test family. Oracle horizons differ (1d vs 5d). Ceiling = deterministic 100% rule, +22306%. "
               "NON-DEPLOYABLE oracle arms.")


def tnew():
    e = pd.read_csv(C / "expressibility_grid.csv")
    e = e[e.accuracy.astype(str).str.strip() == "100%"].copy()
    assert len(e) == 6
    e['execution'] = e.execution.str.replace("canonical without smoothing", "no smoothing")
    df = pd.DataFrame({
        'size / asset': e.size_per_asset, 'effective size (post clip + cap)': e.effective_max_size_after_clip_and_cap.round(2),
        'execution': e.execution, 'net return %': e.net_return.map(lambda v: f"{100 * v:+,.1f}"),
        'gross P&L % (held)': e.gross_pnl_held_weights.map(lambda v: f"{100 * v:+,.1f}"),
        'cost / NAV %': e.txn_cost_over_nav.map(lambda v: f"{100 * v:.1f}"),
        'max DD %': e.max_dd.map(lambda v: f"{100 * v:.1f}"),
        'trades / decision day': e.trades_per_decision_day.round(2),
        'band-blocked share': e.band_blocked_share.round(3),
        'mean gross exposure': e.mean_abs_gross_exposure.round(2)})
    save_table(df, "TNEW_expressibility_grid", "expressibility_grid.csv (100% accurate 5-day oracle, deterministic)",
               "6 rows = 3 sizes x 2 executions; size 0.8 is clipped/capped to 0.4 effective, so its rows "
               "nearly duplicate size 0.4")
    # paper render: merge the 0.4 / 0.8 near-duplicates within each execution mode
    rows = []
    for ex in ("canonical", "no smoothing"):
        s = df[df.execution == ex].set_index('size / asset')
        rows.append({'size / asset': "0.2", **s.loc[0.2].drop('effective size (post clip + cap)').to_dict()})
        m = {}
        for col in s.columns.drop('effective size (post clip + cap)'):
            a, b = s.loc[0.4, col], s.loc[0.8, col]
            m[col] = a if str(a) == str(b) else f"{a} / {b}"
        rows.append({'size / asset': "0.4 / 0.8 (gross cap binds at 0.4)", **m})
    paper = pd.DataFrame(rows)
    save_table(paper, "TNEW_expressibility_grid_paper",
               "TNEW_expressibility_grid.csv, 0.4 and 0.8 rows merged per execution mode",
               "paper render; where 0.4 and 0.8 differ the cell shows '0.4 value / 0.8 value'")


# ============================================================================ figures

def _panel_tag(ax, t):
    ax.text(0.0, 1.07, f"({t})", transform=ax.transAxes, fontsize=FS["label"], fontweight="bold", va="bottom",
            ha="left")


def f2():
    d = pd.read_csv(C / "fold4" / "fold4_corrected_results.csv").set_index("variant")
    seeds = np.array([d.loc[f"canonical_none_ppo_lagrangian_seed{s}", "return"] for s in range(1, 6)])
    passive = d.loc["finalist_gated_passive_ew_k5_band010", "return"]
    assert passive == d.loc["equal_weight_k5_band010", "return"]
    mean, sd, best = seeds.mean(), seeds.std(ddof=1), seeds.max()
    fig, ax = plt.subplots(figsize=SIZES["single"])
    x = np.arange(1, 6)
    ax.bar(x, 100 * seeds, width=0.6, color=P["rl"], zorder=2)
    lines_y = (100 * mean, 100 * passive)
    for xi, v in zip(x, seeds):
        y = 100 * v + 0.8
        crosses = any(y - 0.5 <= ly <= y + 4.0 for ly in lines_y)     # label box ~ 3.5 pp tall at this size
        ax.text(xi, y, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=FS["annot"], zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8) if crosses else None)
    x0, x1 = 0.55, 5.45
    ax.fill_between([x0, x1], 100 * (mean - sd), 100 * (mean + sd), color=P["neutral"], alpha=0.12, lw=0, zorder=1)
    ax.hlines(100 * mean, x0, x1, color=P["ink"], lw=1.0, zorder=3)
    ax.hlines(100 * passive, x0, x1, color=P["passive"], lw=1.3, ls=SECONDARY_LS, zorder=3)
    ax.text(x1 + 0.1, 100 * mean + 0.4, f"seed mean {100 * mean:.1f}", ha="left", va="bottom", fontsize=FS["annot"],
            clip_on=False)
    ax.text(x1 + 0.1, 100 * passive - 0.4, f"passive {100 * passive:.1f}", ha="left", va="top", fontsize=FS["annot"],
            color=P["passive"], clip_on=False)
    ax.text(x1 + 0.1, 100 * (mean + sd), "±1 SD", ha="left", va="center", fontsize=FS["small"], color=P["neutral"],
            clip_on=False)
    ax.text(1.6, 100 * best + 1.0, f"single seed: {100 * (best - passive):+.1f} pp edge\n"
            f"mean: {100 * (mean - passive):+.1f} pp, within ±{100 * sd:.1f} seed spread",
            ha="left", va="bottom", fontsize=FS["annot"])
    ax.set_xticks(x, [f"seed {s}" for s in x])
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(0, 100 * best + 12)
    ax.set_ylabel("Net return, fold 4 (%)")
    save_fig(fig, "F2_seed_illusion_fold4", "fold4/fold4_corrected_results.csv",
             "canonical ppo_lagrangian unshielded, 5 seeds; passive = equal_weight_k5_band010 (= gated passive "
             "finalist); band = mean ± 1 sd")
    pd.DataFrame({'seed': x, 'net_return': seeds}).assign(passive=passive, mean=mean, sd=sd).to_csv(
        OUT / "F2_data.csv", index=False)


def f3():
    g = pd.read_csv(C / "grid" / "results.csv")
    g['fold'] = g.window.map(FOLD_OF)
    assert len(g) == 90 and (g[g.shield == "uncertainty_aware"].max_dd <= 0.300 + 1e-9).all()
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=SIZES["double"])
    arms = (("none", P["unshielded"], "^", "white", -0.17), ("uncertainty_aware", P["shielded"], "o", None, 0.17))
    for ax, metric, ylab in ((axes[0], "max_dd", "Max drawdown (%)"), (axes[1], "return", "Net return (%)")):
        for sh, col, mk, face, off in arms:
            for f in (1, 2, 3):
                v = 100 * g[(g.fold == f) & (g.shield == sh)][metric].to_numpy()
                xs = f + off + rng.uniform(-0.07, 0.07, len(v))
                ax.scatter(xs, v, s=14, marker=mk, facecolor=face or col, edgecolor=col, linewidth=0.8, zorder=3)
                ax.hlines(np.median(v), f + off - 0.12, f + off + 0.12, color=P["ink"], lw=1.4, zorder=4)
        ax.set_xticks([1, 2, 3], [FOLD_LAB[f] for f in (1, 2, 3)])
        ax.set_xlim(0.45, 3.55)
        ax.set_ylabel(ylab)
    a = axes[0]
    a.axhline(30, color=P["reference"], lw=0.8, ls=SECONDARY_LS, zorder=2)
    a.text(0.5, 30.7, "30% hard stop", ha="left", va="bottom", fontsize=FS["annot"], color=P["reference"])
    f1u = 100 * g[(g.fold == 1) & (g.shield == "none")].max_dd
    a.text(1 - 0.17, f1u.max() + 2.0, "unshielded", ha="center", va="bottom", fontsize=FS["annot"])
    a.text(1 + 0.17, 27.5, "shielded", ha="center", va="top", fontsize=FS["annot"])
    a.set_ylim(15, f1u.max() + 9)
    axes[1].axhline(0, color=P["reference"], lw=0.5, zorder=1)
    for ax, t in zip(axes, "ab"):
        _panel_tag(ax, t)
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "F3_boundary_and_trap", "grid/results.csv (P5, 3 trainers x 5 seeds per fold x shield)",
             "points = runs (jittered; open triangles unshielded, filled circles shielded), black bars = medians; "
             "all 45 shielded max DD <= 0.300")
    g[['fold', 'shield', 'method', 'seed', 'max_dd', 'return']].to_csv(OUT / "F3_data.csv", index=False)


def f4():
    b = pd.read_csv(C / "belief_validation_daily.csv.gz")
    df = (b.groupby(['fold', 'operator']).entropy_norm.agg(days='size', share=lambda s: (s > 0.7).mean())
          .reset_index().rename(columns={'share': 'share entropy_norm > 0.7'}))
    save_table(df, "F4_data_entropy_shares", "belief_validation_daily.csv.gz (D9-corrected)",
               "entropy normalized by ln 3; threshold 0.7")
    fig, ax = plt.subplots(figsize=SIZES["single"])
    w = 0.34
    pos = {}
    for k, (op, col, hatch) in enumerate((("hysteresis", P["hysteresis"], None),
                                          ("softmax", P["softmax"], SECONDARY_HATCH))):
        v = df[df.operator == op].sort_values('fold')['share entropy_norm > 0.7'].to_numpy()
        xs = np.arange(1, 4) + (k - 0.5) * (w + 0.03)
        ax.bar(xs, 100 * v, width=w, color=col, hatch=hatch, edgecolor="white", linewidth=0.6, zorder=2)
        for xi, vi in zip(xs, v):
            ax.text(xi, 100 * vi + 1.5, f"{100 * vi:.1f}", ha="center", va="bottom", fontsize=FS["annot"])
        pos[op] = (xs[0], 100 * v[0])
    ax.text(pos["hysteresis"][0] + w / 2, pos["hysteresis"][1] + 11, "hysteresis", ha="right", va="bottom",
            fontsize=FS["annot"])
    ax.text(pos["softmax"][0], pos["softmax"][1] + 9, "softmax", ha="center", va="bottom", fontsize=FS["annot"])
    ax.set_xticks([1, 2, 3], [FOLD_LAB[f] for f in (1, 2, 3)])
    ax.set_xlim(0.12, 3.55)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Days with entropy > 0.7 (%)")
    save_fig(fig, "F4_belief_entropy", "belief_validation_daily.csv.gz (D9-corrected)",
             "caption carries 'no accuracy advantage (see text)'")


def f5():
    lad = pd.read_csv(C / "ladder_seeded.csv")
    cfgs = list("abcdefgh")
    agg = []
    for c in cfgs:
        s = lad[lad.config == c]
        assert len(s) == 5
        lo, hi = tci(s['return'])
        agg.append({'config': c, 'net_mean': s['return'].mean(), 'net_sd': s['return'].std(),
                    'ci_lo': lo, 'ci_hi': hi, 'cost_nav': s.txn_cost.mean(), 'turnover': s.turnover.mean()})
    a = pd.DataFrame(agg)
    a.to_csv(OUT / "F5_data.csv", index=False)
    x = np.arange(len(cfgs))
    fig, axes = plt.subplots(1, 2, figsize=SIZES["double"])
    ax = axes[0]
    ax.axhline(0, color=P["reference"], lw=0.5, zorder=1)
    ax.vlines(x, 100 * a.ci_lo, 100 * a.ci_hi, color=P["primary"], lw=3.0, zorder=2)
    ax.errorbar(x + 0.2, 100 * a.net_mean, yerr=100 * a.net_sd, fmt="none", ecolor=P["ink"], elinewidth=0.7,
                capsize=2, capthick=0.7, zorder=3)
    ax.scatter(x, 100 * a.net_mean, s=24, color=P["primary"], edgecolor="white", linewidth=0.7, zorder=4)
    ax.set_xticks(x, cfgs)
    ax.set_xlabel("Ladder configuration")
    ax.set_ylabel("Net return, fold 2 (%)")
    ax = axes[1]
    ax.plot(x, 100 * a.cost_nav, color=P["primary"], marker="o", ms=4, mec="white", mew=0.6)
    ax.plot(x, 100 * a.turnover, color=P["secondary"], marker="s", ms=4, mec="white", mew=0.6, ls=SECONDARY_LS)
    ax.set_yscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(1, 100)
    ax.set_xlim(-0.4, len(cfgs) - 0.6)
    ax.text(x[-1] + 0.25, 100 * a.cost_nav.iloc[-1], "cost / NAV", ha="left", va="center", fontsize=FS["annot"],
            clip_on=False)
    ax.text(x[-1] + 0.25, 100 * a.turnover.iloc[-1], "turnover / day", ha="left", va="center", fontsize=FS["annot"],
            clip_on=False)
    ax.set_xticks(x, cfgs)
    ax.set_xlabel("Ladder configuration")
    ax.set_ylabel("Percent of NAV (log scale)")
    for ax_, t in zip(axes, "ab"):
        _panel_tag(ax_, t)
    fig.tight_layout(w_pad=3.0)
    save_fig(fig, "F5_ladder", "ladder_seeded.csv (P3, configs a-h x seeds 1-5)",
             "(a) dot = mean, blue bar = 95% t-CI, black whisker = ±1 SD (n=5); (b) both series are fractions of "
             "NAV, shared log axis. CAPTION (b): Trading costs fall by roughly an order of magnitude from the "
             f"unconstrained configuration (cost/NAV {a.cost_nav.iloc[0]:.2f} -> {a.cost_nav.iloc[5]:.2f} with band "
             "and 5-day cadence); the small rise at the annealing step (d->e) is a training-side change, not an "
             "execution change.")
    print("\n=== F5_data ===")
    print(a.round(4).to_string(index=False))


def f_entropy():
    import matplotlib.dates as mdates
    b = pd.read_csv(C / "belief_validation_daily.csv.gz", parse_dates=['date'])
    b = b[(b.fold == 3) & (b.date >= "2020-01-01") & (b.date <= "2020-06-30")]
    hy = b[b.operator == "hysteresis"].set_index('date')
    sm = b[b.operator == "softmax"].set_index('date')
    peak = pd.Timestamp("2020-02-19")
    base = hy.committed[hy.index < peak].iloc[-1]
    flip = hy[(hy.index >= peak) & (hy.committed != base)].index[0]
    early = hy.entropy_norm[hy.entropy_norm > 0.7]
    assert early.index.max() <= pd.Timestamp("2020-01-13")     # the only >0.7 hysteresis episode is the start
    fig, (ax, axr) = plt.subplots(2, 1, figsize=SIZES["double"], sharex=True,
                                  gridspec_kw={"height_ratios": [6, 0.55], "hspace": 0.08})
    ax.plot(hy.index, hy.entropy_norm, color=P["hysteresis"], lw=1.4, zorder=3)
    ax.plot(sm.index, sm.entropy_norm, color=P["softmax"], lw=1.4, ls=SECONDARY_LS, zorder=3)
    end = hy.index[-1] + pd.Timedelta(days=2)
    ax.text(end, hy.entropy_norm.iloc[-1] + 0.02, "hysteresis", ha="left", va="bottom", fontsize=FS["annot"],
            clip_on=False)
    ax.text(end, sm.entropy_norm.iloc[-1], "softmax", ha="left", va="center", fontsize=FS["annot"], clip_on=False)
    ax.axhline(0.7, color=P["reference"], lw=0.8, ls=(0, (1, 1.5)), zorder=2)
    ax.text(hy.index[0] + pd.Timedelta(days=1), 0.68, "0.7", ha="left", va="top", fontsize=FS["annot"],
            color=P["reference"], zorder=5)   # under the line, left of where the curve crosses it (Jan 13)
    for d, lab, ha in ((peak, f"S&P 500 peak ({peak:%b %d})", "right"),
                       (flip, f"hysteresis commits to Bear ({flip:%b %d})", "left")):
        ax.axvline(d, color=P["reference"], lw=0.7, zorder=2)
        ax.annotate(lab, xy=(d, 1.02), xycoords=("data", "axes fraction"), ha=ha, va="bottom",
                    fontsize=FS["annot"], annotation_clip=False,
                    xytext=(-2 if ha == "right" else 2, 0), textcoords="offset points")
    ax.annotate("initialization from\nuniform prior", xy=(pd.Timestamp("2020-01-09"), hy.entropy_norm.loc["2020-01-09"]),
                xytext=(pd.Timestamp("2020-01-20"), 0.78), fontsize=FS["small"], color="#777777", ha="left",
                va="center", arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=1, shrinkB=1))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Normalized entropy H / ln 3")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    # committed-regime ribbon (hysteresis), greys so it survives grayscale
    names, shade, ink = {0: "Bull", 1: "Bear", 2: "Sideways"}, {0: "#DDDDDD", 1: "#555555", 2: "#BBBBBB"}, \
        {0: "#000000", 1: "#FFFFFF", 2: "#000000"}
    reg = hy.committed.astype(int)
    full = pd.date_range(hy.index[0], hy.index[-1], freq="D")
    rd = reg.reindex(full).ffill()
    segs, s0 = [], full[0]
    for i in range(1, len(full)):
        if rd.iloc[i] != rd.iloc[i - 1]:
            segs.append((s0, full[i - 1], int(rd.iloc[i - 1])))
            s0 = full[i]
    segs.append((s0, full[-1], int(rd.iloc[-1])))
    for a0, a1, v in segs:
        axr.axvspan(a0, a1 + pd.Timedelta(days=1), color=shade[v], lw=0)
        axr.text(a0 + (a1 - a0) / 2, 0.5, names[v], ha="center", va="center", fontsize=FS["annot"], color=ink[v])
    axr.set_ylim(0, 1)
    axr.set_yticks([])
    axr.set_ylabel("Committed\nregime", rotation=0, ha="right", va="center", fontsize=FS["annot"])
    for sp_ in axr.spines.values():
        sp_.set_visible(False)
    axr.xaxis.set_major_locator(mdates.MonthLocator())
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(hy.index[0], hy.index[-1])
    save_fig(fig, "entropy_trace_2020", "belief_validation_daily.csv.gz (D9-corrected, fold 3)",
             f"hysteresis vs softmax (T=1.0), Jan-Jun 2020; flip {flip.date()}; >0.7 hysteresis only "
             f"{early.index.min().date()}..{early.index.max().date()} (uniform-prior initialization)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t4(); t5(); t6(); tnew()
    f2(); f3(); f4(); f5(); f_entropy()
    L = ["# Paper exhibits (corrected accounting)", "",
         "Generated by `make_paper_exhibits.py` (style: `scripts/paper_style.py`) from `results/r2/corrected/` "
         "only (analysis only). Figures: 600-dpi PNG + vector PDF, no embedded titles; captions carry them. "
         "Returns in percent.", "",
         "| file | source | note |", "|---|---|---|"]
    L += [f"| `{f}` | {s} | {n} |" for f, s, n in MANIFEST]
    L += ["", "Retired: F5 waterfall (replaced by `F5_ladder`).", "", "## Figure QA (at target size)", ""]
    L += [f"    {q}" for q in QA]
    (OUT / "exhibits_manifest.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n=== FIGURE QA ===")
    print("\n".join(QA))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    main()
