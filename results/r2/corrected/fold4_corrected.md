# Fold 4 (2024-2025) -- accounting-corrected re-evaluation of the frozen one-look

> **This is an ACCOUNTING CORRECTION of previously frozen evaluations, not a second look.** The artifacts, settings and metric path are those of the 2026-08-06 one-look (`run_fold4_evaluation.py`, `results/fold4/fold4_results.csv`), all frozen before fold-4 data was first fetched. Only the two audited accounting defects change: **#8** slippage units (daily-scale volatility, fixed in `portfolio_engine`) and **#9** regime look-ahead (volatility thresholds now calibrated on the fold-4 training window 2021-2023 and frozen; previously fit on the 2024-2025 test window itself). No retraining, no new designs, no threshold or parameter choices, no selection among results. Approved as P6, run once (2026-10-16T22:39:34 UTC, `run_fold4_corrected.py`, git d5c13b9); results are recorded as they came out.

> **Scope notes.** (1) The canonical checkpoints were trained in August under the defective accounting and were NOT retrained (P6 is eval-only): their rows are corrected-evaluation of defective-training policies. (2) `ew_matched_execution` (constant 1/N through the canonical execution layer) was defined 2026-10-14, after fold 4 was fetched; it is parameter-free (1/N + the frozen canonical execution settings) and is included because it was explicitly requested -- it has no old value. (3) The trailing-anchor retrofit rows are part of the frozen one-look protocol (same checkpoints, dd_anchor flipped). (4) Oracle arms are not part of fold 4.

Provenance: same data cache as the one-look (`data_cache\raw_data_2020-11-22_2025-12-31_000001.SS-CLF-DJI-GDAXI-GSPC.pkl`, sha256 `9c5c003576bbf3b4...`, asserted not rewritten); test window 2024-01-02 to 2025-12-30 (520 rows); calibrated vol thresholds [0.1202, 0.1809]; checkpoint SHA-256s in `fold4/fold4_corrected_provenance.json`. Fold-4 test data used only by this run.

## 1. Every frozen row: corrected vs one-look

| variant | net return (corr. / old / Δ) | Sharpe (corr. / old / Δ) | max DD (corr. / old / Δ) | violation rate (corr. / old / Δ) | turnover/day (corr. / old / Δ) |
|---|---|---|---|---|---|
| equal_weight_monthly | +0.2589 / +0.2433 / +0.0156 | +0.855 / +0.799 / +0.057 | 0.1362 / 0.1375 / -0.0013 | 0.0000 / 0.0000 / +0.0000 | 0.0032 / 0.0032 / -0.0000 |
| markowitz_252d_monthly | +0.3845 / +0.2864 / +0.0981 | +1.120 / +0.818 / +0.301 | 0.1448 / 0.1545 / -0.0097 | 0.0000 / 0.0000 / +0.0000 | 0.0129 / 0.0130 / -0.0000 |
| vol_target_10pct | +0.2071 / +0.0771 / +0.1300 | +0.619 / +0.136 / +0.483 | 0.1317 / 0.1478 / -0.0161 | 0.0000 / 0.0000 / +0.0000 | 0.0162 / 0.0162 / -0.0000 |
| tsmom_12m | +0.2831 / +0.2411 / +0.0420 | +1.231 / +1.027 / +0.203 | 0.1092 / 0.1135 / -0.0042 | 0.0000 / 0.0000 / +0.0000 | 0.0051 / 0.0051 / -0.0000 |
| tsmom_20d | +0.0664 / -0.0840 / +0.1505 | +0.187 / -0.949 / +1.136 | 0.0808 / 0.1831 / -0.1023 | 0.0000 / 0.0000 / +0.0000 | 0.0217 / 0.0217 / -0.0001 |
| hmm_regime | +0.2589 / +0.2433 / +0.0156 | +0.855 / +0.799 / +0.057 | 0.1362 / 0.1375 / -0.0013 | 0.0000 / 0.0000 / +0.0000 | 0.0032 / 0.0032 / -0.0000 |
| equal_weight_k5_band010 | +0.2679 / +0.2679 / +0.0000 | +0.904 / +0.904 / +0.000 | 0.1371 / 0.1371 / +0.0000 | 0.0000 / 0.0000 / +0.0000 | 0.0019 / 0.0019 / +0.0000 |
| ew_matched_execution | +0.1086 / n/a / n/a | +0.432 / n/a / n/a | 0.0920 / n/a / n/a | 0.0000 / n/a / n/a | 0.0012 / n/a / n/a |
| finalist_shield-regime_gated_bear0.6_band0.15_k5 | +0.2528 / +0.2512 / +0.0016 | +0.836 / +0.828 / +0.008 | 0.1371 / 0.1371 / +0.0000 | 0.0000 / 0.0000 / +0.0000 | 0.0026 / 0.0026 / -0.0000 |
| finalist_gated_passive_ew_k5_band010 | +0.2679 / +0.2679 / +0.0000 | +0.904 / +0.904 / +0.000 | 0.1371 / 0.1371 / +0.0000 | 0.0000 / 0.0000 / +0.0000 | 0.0019 / 0.0019 / +0.0000 |
| canonical_none_ppo_lagrangian_seed1 | +0.4597 / +0.3090 / +0.1508 | +1.085 / +0.733 / +0.352 | 0.1964 / 0.1945 / +0.0019 | 0.0000 / 0.0000 / +0.0000 | 0.0121 / 0.0121 / +0.0000 |
| canonical_none_ppo_lagrangian_seed2 | +0.3360 / +0.2347 / +0.1012 | +0.725 / +0.503 / +0.223 | 0.2130 / 0.2041 / +0.0089 | 0.0039 / 0.0019 / +0.0019 | 0.0106 / 0.0111 / -0.0005 |
| canonical_none_ppo_lagrangian_seed3 | +0.2332 / +0.0999 / +0.1333 | +0.551 / +0.172 / +0.380 | 0.1934 / 0.2069 / -0.0136 | 0.0000 / 0.0039 / -0.0039 | 0.0142 / 0.0138 / +0.0003 |
| canonical_none_ppo_lagrangian_seed4 | +0.3224 / +0.2093 / +0.1131 | +0.915 / +0.562 / +0.353 | 0.1692 / 0.1740 / -0.0049 | 0.0000 / 0.0000 / +0.0000 | 0.0108 / 0.0113 / -0.0005 |
| canonical_none_ppo_lagrangian_seed5 | +0.1840 / +0.0533 / +0.1307 | +0.410 / +0.033 / +0.377 | 0.1372 / 0.1690 / -0.0318 | 0.0000 / 0.0000 / +0.0000 | 0.0145 / 0.0141 / +0.0004 |
| canonical_uncertainty_aware_ppo_lagrangian_seed1 | +0.2272 / +0.0683 / +0.1589 | +0.582 / +0.089 / +0.493 | 0.1710 / 0.1784 / -0.0074 | 0.0000 / 0.0000 / +0.0000 | 0.0168 / 0.0163 / +0.0005 |
| canonical_uncertainty_aware_trailing_retrofit_seed1 | +0.2272 / +0.0683 / +0.1589 | +0.582 / +0.089 / +0.493 | 0.1710 / 0.1784 / -0.0074 | 0.0000 / 0.0000 / +0.0000 | 0.0168 / 0.0163 / +0.0005 |
| canonical_uncertainty_aware_ppo_lagrangian_seed2 | +0.1844 / +0.0039 / +0.1804 | +0.407 / -0.118 / +0.526 | 0.1913 / 0.1858 / +0.0055 | 0.0000 / 0.0000 / +0.0000 | 0.0155 / 0.0162 / -0.0007 |
| canonical_uncertainty_aware_trailing_retrofit_seed2 | +0.1844 / +0.0039 / +0.1804 | +0.407 / -0.118 / +0.526 | 0.1913 / 0.1858 / +0.0055 | 0.0000 / 0.0000 / +0.0000 | 0.0155 / 0.0162 / -0.0007 |
| canonical_uncertainty_aware_ppo_lagrangian_seed3 | +0.0331 / -0.2197 / +0.2528 | -0.026 / -0.969 / +0.943 | 0.2307 / 0.3000 / -0.0693 | 0.0424 / 0.3680 / -0.3256 | 0.0225 / 0.0165 / +0.0060 |
| canonical_uncertainty_aware_trailing_retrofit_seed3 | +0.0331 / -0.2677 / +0.3008 | -0.026 / -1.091 / +1.065 | 0.2307 / 0.3746 / -0.1438 | 0.0424 / 0.3680 / -0.3256 | 0.0225 / 0.0274 / -0.0049 |
| canonical_uncertainty_aware_ppo_lagrangian_seed4 | +0.2493 / +0.1247 / +0.1246 | +0.559 / +0.239 / +0.320 | 0.2152 / 0.1935 / +0.0217 | 0.0039 / 0.0000 / +0.0039 | 0.0163 / 0.0155 / +0.0008 |
| canonical_uncertainty_aware_trailing_retrofit_seed4 | +0.2493 / +0.1247 / +0.1246 | +0.559 / +0.239 / +0.320 | 0.2152 / 0.1935 / +0.0217 | 0.0039 / 0.0000 / +0.0039 | 0.0163 / 0.0155 / +0.0008 |
| canonical_uncertainty_aware_ppo_lagrangian_seed5 | +0.2117 / +0.0523 / +0.1595 | +0.508 / +0.033 / +0.475 | 0.1846 / 0.1863 / -0.0017 | 0.0000 / 0.0000 / +0.0000 | 0.0163 / 0.0160 / +0.0003 |
| canonical_uncertainty_aware_trailing_retrofit_seed5 | +0.2117 / +0.0523 / +0.1595 | +0.508 / +0.033 / +0.475 | 0.1846 / 0.1863 / -0.0017 | 0.0000 / 0.0000 / +0.0000 | 0.0163 / 0.0160 / +0.0003 |

## 2. Canonical checkpoints: 5-seed summaries (descriptive; n=5, no tests)

| arm | net return corr. (mean ± sd) | old | Δ mean | max DD corr. | old | violation corr. | old | seeds < 0 corr. / old |
|---|---|---|---:|---|---|---|---|---|
| canonical ppo_lagrangian, unshielded | +0.3070 ± 0.1060 | +0.1812 ± 0.1036 | +0.1258 | 0.1818 | 0.1897 | 0.0008 | 0.0012 | 0 / 0 |
| canonical ppo_lagrangian, shielded (peak anchor) | +0.1811 ± 0.0861 | +0.0059 ± 0.1333 | +0.1752 | 0.1986 | 0.2088 | 0.0092 | 0.0736 | 0 / 1 |
| shielded, trailing-63d anchor retrofit | +0.1811 ± 0.0861 | -0.0037 ± 0.1537 | +0.1848 | 0.1986 | 0.2237 | 0.0092 | 0.0736 | 0 / 1 |

## 3. Ranking by net return (corrected vs one-look)

| rank | corrected | net | one-look | net |
|---|---|---:|---|---:|
| 1 | markowitz_252d_monthly | +0.3845 | markowitz_252d_monthly | +0.2864 |
| 2 | tsmom_12m | +0.2831 | finalist_gated_passive_ew_k5_band010 | +0.2679 |
| 3 | finalist_gated_passive_ew_k5_band010 | +0.2679 | equal_weight_k5_band010 | +0.2679 |
| 4 | equal_weight_k5_band010 | +0.2679 | finalist_shield-regime_gated_bear0.6_band0.15_k5 | +0.2512 |
| 5 | equal_weight_monthly | +0.2589 | equal_weight_monthly | +0.2433 |
| 6 | hmm_regime | +0.2589 | hmm_regime | +0.2433 |
| 7 | finalist_shield-regime_gated_bear0.6_band0.15_k5 | +0.2528 | tsmom_12m | +0.2411 |
| 8 | vol_target_10pct | +0.2071 | vol_target_10pct | +0.0771 |
| 9 | ew_matched_execution | +0.1086 | tsmom_20d | -0.0840 |
| 10 | tsmom_20d | +0.0664 |  |  |
| -- | canonical ppo_lagrangian, unshielded (5-seed mean) | +0.3070 | canonical ppo_lagrangian, unshielded (5-seed mean) | +0.1812 |
| -- | canonical ppo_lagrangian, shielded (peak anchor) (5-seed mean) | +0.1811 | canonical ppo_lagrangian, shielded (peak anchor) (5-seed mean) | +0.0059 |
| -- | shielded, trailing-63d anchor retrofit (5-seed mean) | +0.1811 | shielded, trailing-63d anchor retrofit (5-seed mean) | -0.0037 |

Notes:

- Bit-identical old vs corrected (neither defect bites): `equal_weight_k5_band010`, `finalist_gated_passive_ew_k5_band010`. For the k=5/band-0.10 EW book this matches fold 2: after the day-0 purchase the band never triggers, so only the day-0 cost (unaffected by #8) is paid.
- Identical rows within the corrected run: `equal_weight_monthly` = `hmm_regime`; `equal_weight_k5_band010` = `finalist_gated_passive_ew_k5_band010` (hmm_regime falls back to equal-weight monthly, as in the one-look; the gated finalist's shield never modifies the passive book).
- Retrofit rows equal their peak-anchor rows for every seed under corrected accounting (the trailing and all-time-peak anchors never diverged while the drawdown ramp was active; mechanism not probed further); in the one-look they differed only for seed 3.

