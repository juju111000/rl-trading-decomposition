#!/usr/bin/env python3
"""
strategy-v2-overlay forensics: decompose RC-Overlay's dev-fold performance
BEFORE any redesign. Folds 1-3 only (test windows 2008-2010 / 2015-2017 /
2020-2023) -- fold 4 (2024-25) is not loaded anywhere in this script.

1. Attribution (per fold), all relative to the SAME actual RC-Overlay run
   (kappa=0.3, trailing shield 30%cap/63d, k=5/band=0.10, no hysteresis/
   blend -- i.e. exactly the RC-Overlay reported previously):
     (a) regime_branch_pnl -- frictionless price P&L of whichever branch
         was actually active each day (the active branch's own weights,
         held against realized asset returns, zero costs) -- the pure
         informational value of the regime call, decoupled from every
         friction below.
     (b) switching_cost -- real transaction cost actually incurred on
         days that were BOTH a real decision day (step % 5 == 0) AND a
         branch switch, summed as tx_cost/nav_prev.
     (c) entropy_tightening_drag = total_return(kappa=0) - total_return(actual).
     (d) trailing_shield_drag = total_return(shield off) - total_return(actual).
   (a)/(b) are measured directly from the actual run's own trace; (c)/(d)
   are full counterfactual re-runs. They are NOT the same accounting
   framework and won't sum exactly to the actual return -- see 'residual'.

2. Regime timing: a monthly oracle -- for each calendar month in the test
   window, which branch (equal-weight vs tsmom inverse-vol) would have had
   the higher frictionless P&L, using the SAME weight formulas RC-Overlay
   itself uses -- compared against RC-Overlay's actual (majority-vote)
   branch that month. Each month is classified MATCH / LATE (one month
   behind the oracle) / WRONG (neither). Saves a per-fold timeline PNG.

3. Cheap ablation grid: 2(kappa) x 2(shield) x 2(switch hysteresis) x
   2(partial-switch blend) = 16 RC-Overlay variants x 3 folds, same metric
   table as run_dev_comparison.py's dev_comparison.csv. No other changes.

Output: results/v2_dev/forensics/{attribution.csv, timing_accuracy.csv,
regime_timing_fold{1,2,3}.csv, regime_timeline_fold{1,2,3}.png,
ablation_grid.csv}
"""
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main as main_module
from config import ShieldConfig, ShieldType
from evaluation import WalkForwardEvaluator
from run_baselines import PriceHistory, compute_return_metrics, _window_label
from overlay_strategies import RCOverlayStrategy, RCOverlayConfig, make_trailing_shield_config

DEV_FOLDS = main_module.PAPER_WALK_FORWARD_WINDOWS  # folds 1-3 only -- fold 4 (2024-25) does not exist here
OUTPUT_DIR = Path("./results/v2_dev/forensics")

# Two-category categorical palette (bull/side = cool blue, bear = warm
# orange): far apart in both hue and lightness, the standard safe pairing
# for a 2-way categorical distinction under all common CVD types.
BRANCH_COLORS = {'bull_side': '#4C72B0', 'bear': '#DD8452'}
BRANCH_LABELS = {'bull_side': 'Equal-Weight (Bull/Side branch)', 'bear': 'TSMOM inv-vol (Bear branch)'}


# =============================================================================
# Shared helpers (duplicated from run_dev_comparison.py to keep this script
# standalone; NOT a modification of that already-shipped script)
# =============================================================================

def trailing_max_drawdown(nav_history, window: int = 63) -> float:
    nav = np.asarray(nav_history, dtype=float)
    if len(nav) == 0:
        return 0.0
    trailing_peak = pd.Series(nav).rolling(window, min_periods=1).max().values
    dd = 1 - nav / trailing_peak
    return float(np.max(np.clip(dd, 0.0, None)))


def _finalize_row(variant: str, window_label: str, run_result: Dict, risk_free_rate: float) -> Dict:
    row = compute_return_metrics(run_result['nav_history'], risk_free_rate=risk_free_rate)
    row.update({
        'variant': variant,
        'window': window_label,
        'violation_rate': run_result['dd_violation_rate'],
        'turnover': run_result['avg_turnover'],
        'trailing_63d_max_dd': trailing_max_drawdown(run_result['nav_history'], window=63),
    })
    return row


def _load_dev_data(tickers=None):
    if tickers is None:
        tickers = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]

    windows = DEV_FOLDS
    all_dates = [d for w in windows for d in (w['train'][0], w['train'][1], w['test'][0], w['test'][1])]
    nominal_start, overall_end = min(all_dates), max(all_dates)
    overall_start = main_module._buffered_start_date(nominal_start)

    base_config = main_module.create_experiment_config(
        tickers=tickers, start_date=overall_start, end_date=overall_end,
        total_timesteps=1, use_synthetic=False, seed=1,
    )
    features, prices_df, splits, returns, aligned_data = main_module.load_and_prepare_data(
        base_config, use_synthetic=False
    )
    prices = {t: prices_df[t].values for t in tickers}
    wf_evaluator = WalkForwardEvaluator(base_config)
    calendar_splits = wf_evaluator.generate_calendar_splits(aligned_data.index, windows)
    return tickers, base_config, features, prices, aligned_data, calendar_splits


def _fold_slices(split, features, prices, aligned_data, tickers, lookback_days: int = 260):
    train_start_idx, train_end_idx = split['train']
    test_start_idx, test_end_idx = split['test']
    window_label = _window_label(split['test_dates'])

    test_features = features.iloc[test_start_idx:test_end_idx]
    test_prices = {t: p[test_start_idx:test_end_idx] for t, p in prices.items()}
    test_aligned = aligned_data.iloc[test_start_idx:test_end_idx]
    test_dates = aligned_data.index[test_start_idx:test_end_idx]

    lookback_start_idx = max(0, test_start_idx - lookback_days)
    actual_lookback_len = test_start_idx - lookback_start_idx
    combined_prices = {t: p[lookback_start_idx:test_end_idx] for t, p in prices.items()}
    price_history = PriceHistory(combined_prices, lookback_len=actual_lookback_len, tickers=tickers)

    return window_label, test_features, test_prices, test_aligned, test_dates, price_history


def _asset_simple_returns(test_prices: Dict[str, np.ndarray], tickers: List[str]) -> np.ndarray:
    """simple_ret[s, i] = price_i[s+1]/price_i[s] - 1 -- the return realized
    AFTER acting at env step s (s aligned with RCOverlayStrategy._step)."""
    arr = np.column_stack([test_prices[t] for t in tickers])
    return arr[1:] / arr[:-1] - 1.0


def _run_rc_overlay(base_config, test_features, test_prices, test_aligned, price_history, tickers,
                     overlay_config: RCOverlayConfig, shield_on: bool = True,
                     trace: Optional[List[Dict]] = None) -> Dict:
    """One RC-Overlay episode under the trailing shield (30% cap, 63d
    anchor) + k=5/band=0.10 execution, or with the shield removed entirely
    if shield_on=False. Manual step loop (not BaselineEvaluator) so we keep
    the raw per-step transaction_cost series the attribution needs."""
    arm_config = copy.deepcopy(base_config)
    arm_config.shield = make_trailing_shield_config() if shield_on else ShieldConfig(shield_type=ShieldType.NONE)
    arm_config.environment.decision_frequency_days = 5
    arm_config.environment.no_trade_band = 0.10
    arm_config.environment.action_smoothing_beta = None

    env = main_module.create_trading_env(test_features, test_prices, arm_config,
                                          aligned_data=test_aligned, mode="test")
    strategy = RCOverlayStrategy(len(tickers), arm_config, price_history, overlay_config, trace=trace)

    obs, info = env.reset()
    nav_history = [env.portfolio.initial_capital]
    turnovers, tx_costs, dd_violations = [], [], []
    done = False
    while not done:
        action = strategy.get_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        nav_history.append(info.get('nav', nav_history[-1]))
        turnovers.append(info.get('turnover', 0.0))
        tx_costs.append(info.get('transaction_cost', 0.0))
        cost_vec = info.get('cost_vec', np.zeros(3))
        dd_violations.append(cost_vec[0] > 0)

    return {
        'nav_history': nav_history,
        'turnovers': turnovers,
        'tx_costs': tx_costs,
        'dd_violations': dd_violations,
        'avg_turnover': float(np.mean(turnovers)) if turnovers else 0.0,
        'dd_violation_rate': float(np.mean(dd_violations)) if dd_violations else 0.0,
    }


# =============================================================================
# 1. Attribution
# =============================================================================

def _frictionless_branch_pnl(trace: List[Dict], test_prices: Dict, tickers: List[str]) -> float:
    """(a) regime_branch_pnl: compound daily return of holding each day's
    ACTUAL committed branch weights (pre-entropy, pre-shield) against
    realized asset returns, with zero transaction costs."""
    simple_rets = _asset_simple_returns(test_prices, tickers)
    daily = np.zeros(len(trace))
    for i, entry in enumerate(trace):
        s = entry['step']
        if s < len(simple_rets):
            daily[i] = float(entry['base_weights'] @ simple_rets[s])
    nav = np.cumprod(1 + daily)
    return float(nav[-1] - 1) if len(nav) else 0.0


def _switching_cost(trace: List[Dict], actual_run: Dict, decision_frequency_days: int = 5) -> float:
    """(b) switching_cost: real transaction cost incurred on days that were
    BOTH a real decision day AND a branch switch (per the actual run's own
    trace/tx_cost series), summed as sum(tx_cost / nav_before_that_step)."""
    nav = actual_run['nav_history']
    tx_costs = actual_run['tx_costs']
    total = 0.0
    for entry in trace:
        s = entry['step']
        is_decision_day = (s % decision_frequency_days == 0)
        if entry['switched'] and is_decision_day and s < len(tx_costs):
            nav_prev = nav[s]
            if nav_prev > 0:
                total += tx_costs[s] / nav_prev
    return float(total)


def compute_attribution(window_label: str, base_config, test_features, test_prices, test_aligned,
                         price_history, tickers) -> Dict:
    trace_actual: List[Dict] = []
    actual = _run_rc_overlay(base_config, test_features, test_prices, test_aligned, price_history, tickers,
                              RCOverlayConfig(), shield_on=True, trace=trace_actual)
    actual_return = compute_return_metrics(actual['nav_history'])['return']

    kappa0 = _run_rc_overlay(base_config, test_features, test_prices, test_aligned, price_history, tickers,
                              RCOverlayConfig(entropy_scale_coef=0.0), shield_on=True)
    kappa0_return = compute_return_metrics(kappa0['nav_history'])['return']

    noshield = _run_rc_overlay(base_config, test_features, test_prices, test_aligned, price_history, tickers,
                                RCOverlayConfig(), shield_on=False)
    noshield_return = compute_return_metrics(noshield['nav_history'])['return']

    branch_pnl = _frictionless_branch_pnl(trace_actual, test_prices, tickers)
    switching_cost = _switching_cost(trace_actual, actual)
    entropy_drag = kappa0_return - actual_return
    shield_drag = noshield_return - actual_return

    # branch_pnl is frictionless; switching_cost/entropy_drag/shield_drag
    # are measured by three DIFFERENT methods (a live-run cost trace vs two
    # one-at-a-time counterfactual diffs) that don't compose into an exact
    # waterfall -- what's left over is baseline (non-switch) k=5/band=0.10
    # transaction cost plus entropy/shield interaction effects not captured
    # by either counterfactual in isolation. Reported explicitly rather
    # than silently dropped, so the table doesn't look like a complete
    # accounting when it isn't.
    unattributed_drag = branch_pnl - switching_cost - entropy_drag - shield_drag - actual_return

    n_switch_days = sum(1 for e in trace_actual if e['switched'])
    n_decision_switch_days = sum(1 for e in trace_actual if e['switched'] and e['step'] % 5 == 0)

    return {
        'window': window_label,
        'actual_return': actual_return,
        'regime_branch_pnl': branch_pnl,
        'switching_cost': switching_cost,
        'entropy_tightening_drag': entropy_drag,
        'trailing_shield_drag': shield_drag,
        'unattributed_drag': unattributed_drag,
        'n_branch_flips': n_switch_days,
        'n_acted_on_flips': n_decision_switch_days,
    }, trace_actual


# =============================================================================
# 2. Regime timing oracle
# =============================================================================

def _both_branch_weights_series(base_config, price_history, n_assets: int, n_steps: int,
                                 overlay_config: RCOverlayConfig) -> (np.ndarray, np.ndarray):
    """EW and TSMOM weights for EVERY day 0..n_steps-1, regardless of which
    was actually active -- for the oracle comparison below."""
    strat = RCOverlayStrategy(n_assets, base_config, price_history, overlay_config)
    ew_series, tsmom_series = [], []
    for s in range(n_steps):
        strat._step = s
        ew_series.append(strat._weights_for_branch('bull_side'))
        tsmom_series.append(strat._weights_for_branch('bear'))
    return np.array(ew_series), np.array(tsmom_series)


def regime_timing_oracle(trace: List[Dict], test_dates, test_prices, tickers, base_config, price_history,
                          overlay_config: RCOverlayConfig):
    n_assets = len(tickers)
    n_steps = len(trace)
    ew_w, tsmom_w = _both_branch_weights_series(base_config, price_history, n_assets, n_steps, overlay_config)
    simple_rets = _asset_simple_returns(test_prices, tickers)

    daily_ew = np.zeros(n_steps)
    daily_tsmom = np.zeros(n_steps)
    for s in range(n_steps):
        if s < len(simple_rets):
            daily_ew[s] = float(ew_w[s] @ simple_rets[s])
            daily_tsmom[s] = float(tsmom_w[s] @ simple_rets[s])

    months = pd.to_datetime(test_dates[:n_steps]).strftime('%Y-%m')
    actual_branch = [e['branch'] for e in trace]

    daily_df = pd.DataFrame({
        'step': range(n_steps), 'month': months, 'actual_branch': actual_branch,
        'ew_ret': daily_ew, 'tsmom_ret': daily_tsmom,
    })

    def _majority(s):
        return s.value_counts().idxmax()

    monthly = daily_df.groupby('month', sort=True).agg(
        ew_month_ret=('ew_ret', lambda x: float(np.prod(1 + x) - 1)),
        tsmom_month_ret=('tsmom_ret', lambda x: float(np.prod(1 + x) - 1)),
        actual_branch=('actual_branch', _majority),
    ).reset_index()
    monthly['ideal_branch'] = np.where(monthly['ew_month_ret'] >= monthly['tsmom_month_ret'], 'bull_side', 'bear')

    classifications = []
    prev_ideal = None
    for _, row in monthly.iterrows():
        if row['actual_branch'] == row['ideal_branch']:
            classifications.append('MATCH')
        elif prev_ideal is not None and row['actual_branch'] == prev_ideal:
            classifications.append('LATE')
        else:
            classifications.append('WRONG')
        prev_ideal = row['ideal_branch']
    monthly['classification'] = classifications

    return daily_df, monthly


def plot_regime_timeline(monthly: pd.DataFrame, window_label: str, output_path: Path):
    n = len(monthly)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), 2.8))

    for row_y, col in [(0, 'ideal_branch'), (1, 'actual_branch')]:
        colors = [BRANCH_COLORS[b] for b in monthly[col]]
        ax.bar(x, height=0.8, bottom=row_y, color=colors, width=1.0, align='edge',
               edgecolor='white', linewidth=0.6)

    mismatch_x = x[monthly['classification'] != 'MATCH'] + 0.5
    if len(mismatch_x):
        ax.plot(mismatch_x, np.full(len(mismatch_x), 2.05), marker='v', linestyle='none',
                color='#333333', markersize=6, clip_on=False, label='mismatch month')

    tick_step = max(1, n // 14)
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(['Oracle\n(best that month)', 'RC-Overlay\n(actual)'], fontsize=9)
    ax.set_xticks(x[::tick_step] + 0.5)
    ax.set_xticklabels(monthly['month'].iloc[::tick_step], rotation=45, ha='right', fontsize=8)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 2.3)
    ax.set_title(f"Regime timing: RC-Overlay actual vs oracle-best branch -- {window_label}",
                 fontsize=11, pad=10)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BRANCH_COLORS[b]) for b in ('bull_side', 'bear')]
    handles.append(plt.Line2D([0], [0], marker='v', linestyle='none', color='#333333', markersize=6))
    labels = [BRANCH_LABELS['bull_side'], BRANCH_LABELS['bear'], 'mismatch month']
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.55), ncol=3,
              frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 3. Cheap ablation grid
# =============================================================================

ABLATION_GRID = [
    {'entropy_scale_coef': k, 'shield_on': s, 'switch_hysteresis_days': h, 'partial_switch_blend': b}
    for k in (0.3, 0.0)
    for s in (True, False)
    for h in (1, 5)
    for b in (None, 0.5)
]


def _variant_name(combo: Dict) -> str:
    blend = 'full' if combo['partial_switch_blend'] is None else f"{combo['partial_switch_blend']:.0%}blend"
    hyst = 'immediate' if combo['switch_hysteresis_days'] <= 1 else f"{combo['switch_hysteresis_days']}d-hyst"
    shield = 'shieldOn' if combo['shield_on'] else 'shieldOff'
    return f"kappa{combo['entropy_scale_coef']}_{shield}_{hyst}_{blend}"


def run_ablation_grid(base_config, test_features, test_prices, test_aligned, price_history, tickers,
                       window_label: str, risk_free_rate: float) -> List[Dict]:
    rows = []
    for combo in ABLATION_GRID:
        overlay_config = RCOverlayConfig(
            entropy_scale_coef=combo['entropy_scale_coef'],
            switch_hysteresis_days=combo['switch_hysteresis_days'],
            partial_switch_blend=combo['partial_switch_blend'],
        )
        result = _run_rc_overlay(base_config, test_features, test_prices, test_aligned, price_history, tickers,
                                  overlay_config, shield_on=combo['shield_on'])
        row = _finalize_row(_variant_name(combo), window_label, result, risk_free_rate)
        rows.append(row)
        print(f"  {row['variant']:34s} return={row['return']:+.4f}  max_dd={row['max_dd']:.3f}  "
              f"trail63d_dd={row['trailing_63d_max_dd']:.3f}  turnover={row['turnover']:.4f}")
    return rows


# =============================================================================
# Driver
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers, base_config, features, prices, aligned_data, calendar_splits = _load_dev_data()
    risk_free_rate = base_config.evaluation.risk_free_rate

    attribution_rows = []
    timing_rows = []
    ablation_rows = []

    for fold_idx, split in enumerate(calendar_splits):
        window_label, test_features, test_prices, test_aligned, test_dates, price_history = _fold_slices(
            split, features, prices, aligned_data, tickers
        )
        print(f"\n{'#' * 70}\n# DEV FOLD {fold_idx + 1}: {window_label} (development data)\n{'#' * 70}")

        print("\n-- 1. attribution --")
        attribution_row, trace_actual = compute_attribution(
            window_label, base_config, test_features, test_prices, test_aligned, price_history, tickers
        )
        attribution_rows.append(attribution_row)
        for k, v in attribution_row.items():
            print(f"  {k:24s} {v}")

        print("\n-- 2. regime timing oracle --")
        daily_df, monthly_df = regime_timing_oracle(
            trace_actual, test_dates, test_prices, tickers, base_config, price_history, RCOverlayConfig()
        )
        monthly_df.to_csv(OUTPUT_DIR / f"regime_timing_fold{fold_idx + 1}.csv", index=False)
        plot_path = OUTPUT_DIR / f"regime_timeline_fold{fold_idx + 1}.png"
        plot_regime_timeline(monthly_df, window_label, plot_path)
        rates = monthly_df['classification'].value_counts(normalize=True).to_dict()
        timing_row = {
            'window': window_label, 'n_months': len(monthly_df),
            'match_rate': rates.get('MATCH', 0.0), 'late_rate': rates.get('LATE', 0.0),
            'wrong_rate': rates.get('WRONG', 0.0),
        }
        timing_rows.append(timing_row)
        print(f"  n_months={timing_row['n_months']}  match={timing_row['match_rate']:.2%}  "
              f"late={timing_row['late_rate']:.2%}  wrong={timing_row['wrong_rate']:.2%}")
        print(f"  saved: {plot_path}")

        print("\n-- 3. cheap ablation grid --")
        ablation_rows.extend(
            run_ablation_grid(base_config, test_features, test_prices, test_aligned, price_history, tickers,
                               window_label, risk_free_rate)
        )

    pd.DataFrame(attribution_rows).to_csv(OUTPUT_DIR / "attribution.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(OUTPUT_DIR / "timing_accuracy.csv", index=False)
    ablation_df = pd.DataFrame(ablation_rows)[
        ['variant', 'window', 'return', 'sharpe', 'calmar', 'max_dd',
         'trailing_63d_max_dd', 'violation_rate', 'turnover']
    ]
    ablation_df.to_csv(OUTPUT_DIR / "ablation_grid.csv", index=False)

    print(f"\nWrote attribution.csv ({len(attribution_rows)} rows), timing_accuracy.csv "
          f"({len(timing_rows)} rows), ablation_grid.csv ({len(ablation_df)} rows) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
