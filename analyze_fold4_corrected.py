#!/usr/bin/env python3
"""
Writes results/r2/corrected/fold4_corrected.md: the P6 accounting-corrected
fold-4 rows (run_fold4_corrected.py, one pass) side by side with the frozen
2026-08-06 one-look (results/fold4/fold4_results.csv). Descriptive only -- no
new tests, no selection.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NEW = Path("./results/r2/corrected/fold4/fold4_corrected_results.csv")
OLD = Path("./results/fold4/fold4_results.csv")
PROV = Path("./results/r2/corrected/fold4/fold4_corrected_provenance.json")
OUT = Path("./results/r2/corrected/fold4_corrected.md")
METRICS = [("return", "net return", "+.4f"), ("sharpe", "Sharpe", "+.3f"), ("max_dd", "max DD", ".4f"),
           ("violation_rate", "violation rate", ".4f"), ("turnover", "turnover/day", ".4f")]
GROUPS = [("canonical_none_ppo_lagrangian_seed", "canonical ppo_lagrangian, unshielded"),
          ("canonical_uncertainty_aware_ppo_lagrangian_seed", "canonical ppo_lagrangian, shielded (peak anchor)"),
          ("canonical_uncertainty_aware_trailing_retrofit_seed", "shielded, trailing-63d anchor retrofit")]


def main():
    new, old = pd.read_csv(NEW).set_index("variant"), pd.read_csv(OLD).set_index("variant")
    prov = json.loads(PROV.read_text())
    assert set(old.index) <= set(new.index) and set(new.index) - set(old.index) == {"ew_matched_execution"}
    L = ["# Fold 4 (2024-2025) -- accounting-corrected re-evaluation of the frozen one-look", "",
         "> **This is an ACCOUNTING CORRECTION of previously frozen evaluations, not a second look.** The "
         "artifacts, settings and metric path are those of the 2026-08-06 one-look (`run_fold4_evaluation.py`, "
         "`results/fold4/fold4_results.csv`), all frozen before fold-4 data was first fetched. Only the two audited "
         "accounting defects change: **#8** slippage units (daily-scale volatility, fixed in `portfolio_engine`) and "
         "**#9** regime look-ahead (volatility thresholds now calibrated on the fold-4 training window 2021-2023 "
         "and frozen; previously fit on the 2024-2025 test window itself). No retraining, no new designs, no "
         "threshold or parameter choices, no selection among results. Approved as P6, run once "
         f"({prov['run_utc'][:19]} UTC, `run_fold4_corrected.py`, git {prov['git_head'][:7]}); results are recorded "
         "as they came out.", "",
         "> **Scope notes.** (1) The canonical checkpoints were trained in August under the defective accounting "
         "and were NOT retrained (P6 is eval-only): their rows are corrected-evaluation of defective-training "
         "policies. (2) `ew_matched_execution` (constant 1/N through the canonical execution layer) was defined "
         "2026-10-14, after fold 4 was fetched; it is parameter-free (1/N + the frozen canonical execution "
         "settings) and is included because it was explicitly requested -- it has no old value. (3) The trailing-"
         "anchor retrofit rows are part of the frozen one-look protocol (same checkpoints, dd_anchor flipped). "
         "(4) Oracle arms are not part of fold 4.", "",
         f"Provenance: same data cache as the one-look (`{prov['data_cache']}`, sha256 "
         f"`{prov['data_cache_sha256'][:16]}...`, asserted not rewritten); test window {prov['test_window'][0]} to "
         f"{prov['test_window'][1]} ({prov['test_window'][2]} rows); calibrated vol thresholds "
         f"{[round(x, 4) for x in prov['calibrated_vol_thresholds']]}; checkpoint SHA-256s in "
         "`fold4/fold4_corrected_provenance.json`. Fold-4 test data used only by this run.", "",
         "## 1. Every frozen row: corrected vs one-look", "",
         "| variant | " + " | ".join(f"{lab} (corr. / old / Δ)" for _, lab, _ in METRICS) + " |",
         "|---|" + "---|" * len(METRICS)]
    for v in new.index:
        cells = []
        for m, _, f in METRICS:
            n = new.loc[v, m]
            if v in old.index:
                o = old.loc[v, m]
                cells.append(f"{n:{f}} / {o:{f}} / {n - o:+{f.lstrip('+')}}")
            else:
                cells.append(f"{n:{f}} / n/a / n/a")
        L.append(f"| {v} | " + " | ".join(cells) + " |")

    L += ["", "## 2. Canonical checkpoints: 5-seed summaries (descriptive; n=5, no tests)", "",
          "| arm | net return corr. (mean ± sd) | old | Δ mean | max DD corr. | old | violation corr. | old | "
          "seeds < 0 corr. / old |", "|---|---|---|---:|---|---|---|---|---|"]
    for pre, lab in GROUPS:
        ids = [f"{pre}{s}" for s in range(1, 6)]
        n, o = new.loc[ids], old.loc[ids]
        L.append(f"| {lab} | {n['return'].mean():+.4f} ± {n['return'].std():.4f} | {o['return'].mean():+.4f} ± "
                 f"{o['return'].std():.4f} | {n['return'].mean() - o['return'].mean():+.4f} | "
                 f"{n.max_dd.mean():.4f} | {o.max_dd.mean():.4f} | {n.violation_rate.mean():.4f} | "
                 f"{o.violation_rate.mean():.4f} | {int((n['return'] < 0).sum())} / {int((o['return'] < 0).sum())} |")

    L += ["", "## 3. Ranking by net return (corrected vs one-look)", "",
          "| rank | corrected | net | one-look | net |", "|---|---|---:|---|---:|"]
    keep = [v for v in new.index if not v.startswith("canonical_")]
    nr = new.loc[keep, 'return'].sort_values(ascending=False)
    orr = old.loc[[v for v in keep if v in old.index], 'return'].sort_values(ascending=False)
    for i in range(len(nr)):
        o = (orr.index[i], f"{orr.iloc[i]:+.4f}") if i < len(orr) else ("", "")
        L.append(f"| {i + 1} | {nr.index[i]} | {nr.iloc[i]:+.4f} | {o[0]} | {o[1]} |")
    for pre, lab in GROUPS:
        ids = [f"{pre}{s}" for s in range(1, 6)]
        L.append(f"| -- | {lab} (5-seed mean) | {new.loc[ids, 'return'].mean():+.4f} | {lab} (5-seed mean) | "
                 f"{old.loc[ids, 'return'].mean():+.4f} |")
    same = [v for v in old.index if new.loc[v, 'return'] == old.loc[v, 'return']]
    dup = [(a, b) for i, a in enumerate(new.index) for b in new.index[i + 1:]
           if not a.startswith("canonical_") and new.loc[a, 'return'] == new.loc[b, 'return']]
    L += ["", "Notes:", "",
          f"- Bit-identical old vs corrected (neither defect bites): {', '.join(f'`{v}`' for v in same) or 'none'}. "
          "For the k=5/band-0.10 EW book this matches fold 2: after the day-0 purchase the band never triggers, "
          "so only the day-0 cost (unaffected by #8) is paid.",
          "- Identical rows within the corrected run: " + "; ".join(f"`{a}` = `{b}`" for a, b in dup) +
          " (hmm_regime falls back to equal-weight monthly, as in the one-look; the gated finalist's shield never "
          "modifies the passive book).",
          "- Retrofit rows equal their peak-anchor rows for every seed under corrected accounting (the trailing "
          "and all-time-peak anchors never diverged while the drawdown ramp was active; mechanism not probed "
          "further); in the one-look they differed only for seed 3.", ""]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
