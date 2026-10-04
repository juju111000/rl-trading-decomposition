#!/usr/bin/env python3
"""
Paper figure: daily normalized belief entropy, canonical hard-EMA-hysteresis
regime model vs. the softmax-temperature variant (T=1.0 -- the canonical
temperature, see RegimeConfig.softmax_temperature), fold-3 test window
(2020-01-01..2023-12-31), Jan-Jun 2020 slice. A committed-regime ribbon for
the hysteresis model is drawn under the x-axis, and both series are shaded
wherever they exceed the 0.7 "high-uncertainty" threshold.

Both variants are driven by an actual TradingCMDPEnv (mode="test") built via
main.create_trading_env exactly as the real pipeline builds fold 3's test
env (fit() on test_aligned, per-step update()) -- only entropy and the
hysteresis model's committed regime are captured, using a flat (zero)
action every step, since the regime model's belief update depends only on
prices/features, never on the action taken.

Output: results/figures/entropy_shock_2020.png
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import to_rgb

import main as main_module
from config import BeliefFormationType
from run_rc_overlay_forensics import _load_dev_data, _fold_slices

OUTPUT_PATH = Path("./results/figures/entropy_shock_2020.png")
FOLD_IDX = 2  # fold 3 (0-indexed): test window 2020-01-01..2023-12-31
PRICE_PEAK = pd.Timestamp("2020-02-19")
PLOT_START, PLOT_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30")
RESOLUTION_THRESHOLD = 0.1
HIGH_UNCERTAINTY_THRESHOLD = 0.7

HYST_LABEL = "Hysteresis-EMA (canonical)"
SOFTMAX_LABEL = "Softmax (T=1.0, canonical temperature)"
VARIANTS = [
    (HYST_LABEL, BeliefFormationType.HARD_EMA_HYSTERESIS, "#4C72B0"),
    (SOFTMAX_LABEL, BeliefFormationType.SOFTMAX_TEMPERATURE, "#DD8452"),
]

# 0: Bull, 1: Bear, 2: Sideways -- see HeuristicRegimeModel docstring/_compute_scores
REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Sideways"}
REGIME_COLORS = {0: "#2ca02c", 1: "#d62728", 2: "#999999"}


def _simulate(base_config, test_features, test_prices, test_aligned,
              belief_formation: BeliefFormationType, track_regime: bool):
    """Step a fresh fold-3 test env end-to-end with a flat action, capturing
    (per day) info['regime_entropy'] normalized by log(n_regimes), and --
    when track_regime -- the committed regime entering that day's decision
    (env.regime_model._committed_regime, read BEFORE env.step(), matching
    the point at which trading_env.py itself reads get_entropy() for info).
    """
    arm_config = copy.deepcopy(base_config)
    arm_config.regime.belief_formation = belief_formation

    env = main_module.create_trading_env(
        test_features, test_prices, arm_config, aligned_data=test_aligned, mode="test"
    )
    max_entropy = env.regime_model.get_max_entropy()
    flat_action = np.zeros(env.action_space.shape, dtype=np.float32)

    env.reset()
    entropies, regimes = [], []
    done = False
    while not done:
        if track_regime:
            regimes.append(env.regime_model._committed_regime)
        _, _, terminated, truncated, info = env.step(flat_action)
        done = terminated or truncated
        entropies.append(info['regime_entropy'] / max_entropy)

    entropy_arr = np.array(entropies)
    regime_arr = np.array(regimes) if track_regime else None
    return entropy_arr, regime_arr


def _high_entropy_episodes(s: pd.Series, threshold: float):
    """Contiguous runs (in trading days) where s > threshold ->
    list of dicts {start, end, n_days, left_censored, right_censored}."""
    mask = s.values > threshold
    idx = s.index
    episodes = []
    run_start = None
    for i, m in enumerate(mask):
        if m and run_start is None:
            run_start = i
        if run_start is not None and (not m or i == len(mask) - 1):
            run_end = i - 1 if not m else i
            episodes.append({
                'start': idx[run_start], 'end': idx[run_end],
                'n_days': run_end - run_start + 1,
                'left_censored': run_start == 0,
                'right_censored': run_end == len(mask) - 1,
            })
            run_start = None
    return episodes


def _first_flip_after(regime_series: pd.Series, after_date: pd.Timestamp):
    """First date strictly after `after_date` where the committed regime
    differs from its value as of (on or before) `after_date`."""
    baseline = regime_series.loc[:after_date]
    if baseline.empty:
        return None
    baseline_regime = baseline.iloc[-1]
    later = regime_series[regime_series.index > after_date]
    changed = later[later != baseline_regime]
    return changed.index[0] if len(changed) else None


def _all_flip_dates(regime_series: pd.Series):
    changes = regime_series[regime_series.diff().fillna(0) != 0]
    return list(changes.index)


def _softmax_dip_date(s: pd.Series):
    april = s[(s.index.year == 2020) & (s.index.month == 4)]
    return april.idxmin() if len(april) else None


def _verify_resolution_stat_definition():
    print("=" * 78)
    print("VERIFICATION: 'post-shock resolution 0-4 days' statistic")
    print("=" * 78)
    print(
        "Searched: regime_model.py, config.py, policy_shield.py, overlay_strategies.py,\n"
        "run_diagnostics.py, PROJECT_STATE.md, DATA_NOTES.md, results/phase4_stage2.log,\n"
        "and the full git history (`git log -S\"days_to_resolution\"` / `-S\"resolution\"`,\n"
        "across all commits and all branches) for any script that computes this stat."
    )
    print(
        "\nRESULT: no such script exists anywhere in this repository or its git history.\n"
        "Only two OUTPUT artifacts from that (never-committed) analysis survive:\n"
        "  results/stage2_diagnostics/recalibration_ranking.csv\n"
        "  results/stage2_diagnostics/recalibration_sweep.csv\n"
        "with columns days_to_resolution / resolution_pass / sustained_high_pass /\n"
        "calm_p_gt_0.7 / pre_onset_mean_entropy -- but the code that produced those\n"
        "numbers (the entropy threshold used, and whether 'episode' means first-crossing,\n"
        "sustained-run, or something else) was never committed. The repo's initial\n"
        "commit (3b6c307) added the two CSVs directly as data; there is no producing\n"
        "script in that commit or any other."
    )
    print(
        "\n=> There is no verifiable threshold/episode definition to cite for a\n"
        "'post-shock resolution 0-4 days' claim. Do not state it in the paper as-is --\n"
        "either recover/reconstruct the original script, or define the statistic fresh\n"
        "(e.g. via _high_entropy_episodes()/RESOLUTION_THRESHOLD in this script) and\n"
        "document that definition explicitly."
    )
    print("=" * 78)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tickers, base_config, features, prices, aligned_data, calendar_splits = _load_dev_data()
    split = calendar_splits[FOLD_IDX]
    window_label, test_features, test_prices, test_aligned, test_dates, _ = _fold_slices(
        split, features, prices, aligned_data, tickers
    )
    print(f"Fold 3 test window: {window_label}")

    series = {}
    regime_series = None
    for label, belief_formation, color in VARIANTS:
        track = (belief_formation == BeliefFormationType.HARD_EMA_HYSTERESIS)
        entropy, regimes = _simulate(base_config, test_features, test_prices, test_aligned,
                                      belief_formation, track_regime=track)
        dates = test_dates[:len(entropy)]
        s = pd.Series(entropy, index=dates)
        s_window = s[(s.index >= PLOT_START) & (s.index <= PLOT_END)]
        series[label] = (s_window, color)
        if track:
            r = pd.Series(regimes, index=dates)
            regime_series = r[(r.index >= PLOT_START) & (r.index <= PLOT_END)]

    hyst_s, hyst_color = series[HYST_LABEL]
    softmax_s, softmax_color = series[SOFTMAX_LABEL]

    flip_date = _first_flip_after(regime_series, PRICE_PEAK)
    dip_date = _softmax_dip_date(softmax_s)

    # =========================================================================
    # Figure: entropy panel + committed-regime ribbon, sharing the date axis
    # =========================================================================
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9.5, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [5, 0.45], "hspace": 0.06},
        constrained_layout=True,
    )

    for label, (s, color) in series.items():
        ax.plot(s.index, s.values, label=label, color=color, linewidth=1.8, zorder=3)
        mask = s.values > HIGH_UNCERTAINTY_THRESHOLD
        ax.fill_between(s.index, 0.0, 1.0, where=mask, facecolor=color, alpha=0.15,
                         linewidth=0, zorder=1)

    ax.axhline(HIGH_UNCERTAINTY_THRESHOLD, color="#333333", linestyle=(0, (4, 2)),
               linewidth=1.1, zorder=2)
    ax.text(PLOT_START + pd.Timedelta(days=5), 0.72, "high-uncertainty threshold (H > 0.7)",
            ha="left", va="bottom", fontsize=8, color="#333333")

    # Marker 1: price peak (light) -- single line, above the axes AND above the
    # legend (which occupies roughly the [1.02, 1.08] axes-fraction band)
    ax.axvline(PRICE_PEAK, color="#aaaaaa", linestyle=":", linewidth=1.1, zorder=2)
    ax.annotate("price peak (2020-02-19)", xy=(PRICE_PEAK, 1.14), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=8, color="#999999", annotation_clip=False)

    # Marker 2: crash registers in signals (first hysteresis commitment flip after the peak)
    # -- placed on its own row, above marker 1, so the two labels never collide
    if flip_date is not None:
        ax.axvline(flip_date, color="#333333", linestyle="--", linewidth=1.3, zorder=3)
        ax.annotate(f"crash registers in signals ({flip_date.date()})",
                    xy=(flip_date, 1.22), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=8, color="#222222", annotation_clip=False)

    # Marker 3: April softmax dip
    if dip_date is not None:
        dip_val = softmax_s.loc[dip_date]
        ax.annotate("softmax sharpens only\nat score extremes",
                    xy=(dip_date, dip_val), xytext=(0, -34), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=softmax_color,
                    arrowprops=dict(arrowstyle="->", color=softmax_color, lw=0.9))

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized belief entropy $H(b_t) / \\log K$")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # ---- committed-regime ribbon (hysteresis model only), forward-filled to
    # every calendar day so it renders as a continuous strip (weekends included) ----
    full_idx = pd.date_range(PLOT_START, PLOT_END, freq="D")
    regime_daily = regime_series.reindex(full_idx).ffill().bfill().astype(int)
    rgb_row = np.array([to_rgb(REGIME_COLORS[c]) for c in regime_daily.values])
    img = rgb_row.reshape(1, -1, 3)
    x0, x1 = mdates.date2num(full_idx[0]), mdates.date2num(full_idx[-1]) + 1
    axr.imshow(img, aspect="auto", extent=[x0, x1, 0, 1], origin="lower")
    axr.set_ylim(0, 1)
    axr.set_yticks([])
    axr.set_ylabel("Committed regime", rotation=0, ha="right", va="center", fontsize=8.5)
    for spine in axr.spines.values():
        spine.set_visible(False)

    # Label segments directly (no legend): text inside each segment wide enough
    # to hold it, "white" on Bull/Bear, dark on Sideways for contrast; segments
    # too narrow get no text (reported to the caption instead -- see below).
    REGIME_TEXT_COLOR = {0: "white", 1: "white", 2: "#222222"}
    segments = []
    seg_start = regime_daily.index[0]
    seg_val = regime_daily.iloc[0]
    for i in range(1, len(regime_daily)):
        if regime_daily.iloc[i] != seg_val:
            segments.append((seg_start, regime_daily.index[i - 1], seg_val))
            seg_start, seg_val = regime_daily.index[i], regime_daily.iloc[i]
    segments.append((seg_start, regime_daily.index[-1], seg_val))

    ax.set_xlim(PLOT_START, PLOT_END)
    axr.xaxis.set_major_locator(mdates.MonthLocator())
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axr.tick_params(axis="x", rotation=0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    unlabeled_segments = []
    for seg_start, seg_end, seg_val in segments:
        center = seg_start + (seg_end - seg_start) / 2
        txt = axr.text(center, 0.5, REGIME_NAMES[seg_val], ha="center", va="center",
                        fontsize=8, color=REGIME_TEXT_COLOR[seg_val], zorder=4)
        seg_px0 = axr.transData.transform((mdates.date2num(seg_start), 0.5))[0]
        seg_px1 = axr.transData.transform((mdates.date2num(seg_end) + 1, 0.5))[0]
        seg_width_px = seg_px1 - seg_px0
        txt_width_px = txt.get_window_extent(renderer=renderer).width
        if txt_width_px > 0.85 * seg_width_px:
            txt.remove()
            unlabeled_segments.append((seg_start, seg_end, seg_val))

    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

    if unlabeled_segments:
        print("\nCAPTION NOTE (segments too narrow to label directly in the ribbon):")
        for seg_start, seg_end, seg_val in unlabeled_segments:
            print(f"  {REGIME_NAMES[seg_val]}: {seg_start.date()} to {seg_end.date()}")

    # =========================================================================
    # Item 5: verification + stats printout
    # =========================================================================
    _verify_resolution_stat_definition()

    print("\nHigh-uncertainty (H > 0.7) episodes within the plotted window (2020 H1):")
    for label, (s, _) in series.items():
        print(f"\n  {label}:")
        episodes = _high_entropy_episodes(s, HIGH_UNCERTAINTY_THRESHOLD)
        if not episodes:
            print("    (none)")
        for ep in episodes:
            flags = []
            if ep['left_censored']:
                flags.append("left-censored: already >0.7 at window start")
            if ep['right_censored']:
                flags.append("right-censored: still >0.7 at window end")
            flag_str = f"  [{'; '.join(flags)}]" if flags else ""
            print(f"    {ep['start'].date()} -> {ep['end'].date()}  "
                  f"({ep['n_days']} trading days){flag_str}")

    print(f"\nHysteresis commitment flip immediately after the {PRICE_PEAK.date()} price peak: "
          f"{flip_date.date() if flip_date is not None else 'none within window'}")
    all_flips = _all_flip_dates(regime_series)
    print(f"All hysteresis commitment flips within the plotted window ({len(all_flips)}): "
          + ", ".join(str(d.date()) for d in all_flips))

    post_shock = hyst_s[hyst_s.index >= PRICE_PEAK]
    resolved = post_shock[post_shock < RESOLUTION_THRESHOLD]
    if len(resolved):
        print(f"\n(supplementary) Hysteresis-EMA first day with entropy < {RESOLUTION_THRESHOLD} "
              f"at/after the price peak: {resolved.index[0].date()}")


if __name__ == "__main__":
    main()
