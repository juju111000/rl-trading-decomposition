# R2 / E2 -- Same-policy shield modes (off | always-on | regime-gated)

_R2 revision, generated 2026-10-16 by `r2_gated_same_policy.py`. Evaluation only, no retraining. Fold 4 never loaded (fail-closed guard). Manifest: `gated_same_policy_manifest.json`._

> **Folds 1 and 3 are POST-HOC SUPPLEMENTARY EVALUATIONS.** Their grid results were observed before this analysis was designed. Fold 2 (development fold, n=20) is the primary comparison.

> **CORRECTED ACCOUNTING (audit defects #8, #9 repaired):** slippage uses daily-scale volatility; regime thresholds are calibrated on each fold's training window and frozen at test time. Checkpoints are the corrected-batch retrains (`results/r2/corrected/`).

## What the regime gate controls (verified in code before evaluating)

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

## Reproduction checks

- fold 2, mode=off vs `hardening_results.csv` canonical return (20 seeds): max |difference| = 5.55e-17 -> REPRODUCED
- fold 1, mode=off vs grid `none/ppo_lagrangian` return (5 seeds): max |difference| = 5.55e-17 -> REPRODUCED
- fold 3, mode=off vs grid `none/ppo_lagrangian` return (5 seeds): max |difference| = 2.22e-16 -> REPRODUCED

## Per fold x mode

Violation rate = share of test days with peak-anchored drawdown > 20%. *Shield modifies weights* = share of days the shield actually changed the executed target weights.

| fold | mode | n | return mean +/- std | 95% CI | mean max DD | violation rate | gate active-day share | shield modifies weights | mean gross exposure |
|---|---|---|---|---|---|---|---|---|---|
| 2 | off | 20 | +0.2960 +/- 0.1389 | [+0.2310, +0.3610] | 0.4043 | 0.2662 | 0.0950 | 0.0000 | 1.332 |
| 2 | always_on | 20 | -0.1808 +/- 0.0515 | [-0.2049, -0.1566] | 0.2998 | 0.7815 | 0.0950 | 0.5969 | 0.248 |
| 2 | gated | 20 | +0.1082 +/- 0.1286 | [+0.0481, +0.1684] | 0.4117 | 0.5183 | 0.0950 | 0.0438 | 1.244 |
| 2 | hybrid | 20 | +0.1289 +/- 0.1483 | [+0.0595, +0.1983] | 0.4029 | 0.5143 | 0.0950 | 0.1017 | 1.235 |
| 1 (post-hoc) | off | 5 | -0.2364 +/- 0.1017 | [-0.3627, -0.1100] | 0.6500 | 0.7292 | 0.4487 | 0.0000 | 1.351 |
| 1 (post-hoc) | always_on | 5 | -0.2649 +/- 0.0220 | [-0.2923, -0.2376] | 0.3000 | 0.7744 | 0.4487 | 0.5431 | 0.231 |
| 1 (post-hoc) | gated | 5 | -0.0494 +/- 0.0398 | [-0.0989, +0.0000] | 0.3372 | 0.4444 | 0.4487 | 0.3113 | 0.867 |
| 1 (post-hoc) | hybrid | 5 | -0.0833 +/- 0.0671 | [-0.1666, +0.0001] | 0.4651 | 0.5113 | 0.4487 | 0.3256 | 1.085 |
| 3 (post-hoc) | off | 5 | +0.9754 +/- 0.1007 | [+0.8503, +1.1004] | 0.3285 | 0.0542 | 0.3404 | 0.0000 | 1.359 |
| 3 (post-hoc) | always_on | 5 | -0.1764 +/- 0.2688 | [-0.5101, +0.1574] | 0.2987 | 0.8288 | 0.3404 | 0.5792 | 0.210 |
| 3 (post-hoc) | gated | 5 | +0.2320 +/- 0.2293 | [-0.0527, +0.5168] | 0.3395 | 0.3073 | 0.3404 | 0.2512 | 1.136 |
| 3 (post-hoc) | hybrid | 5 | +0.3373 +/- 0.1981 | [+0.0913, +0.5833] | 0.3084 | 0.2277 | 0.3404 | 0.2223 | 1.220 |

Gate active-day share depends on beliefs only (the regime model never sees the action), so it is identical across modes.

## Paired per-seed effects (same policy, same seed; paired t + Wilcoxon)

| fold | comparison | metric | n | mean delta | std delta | 95% CI | paired t p | Wilcoxon p | seeds delta > 0 | MDE (paired, 80% power) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | gated - always_on | return | 20 | +0.2890 | 0.1236 | [+0.2312, +0.3468] | <0.001 | <0.001 | 20 / 20 | 0.0816 |
| 2 | gated - always_on | max_dd | 20 | +0.1118 | 0.0694 | [+0.0794, +0.1443] | <0.001 | <0.001 | 19 / 20 | 0.0458 |
| 2 | gated - off | return | 20 | -0.1877 | 0.0795 | [-0.2249, -0.1505] | <0.001 | <0.001 | 0 / 20 | 0.0525 |
| 2 | gated - off | max_dd | 20 | +0.0074 | 0.0339 | [-0.0085, +0.0232] | 0.3433 | 0.3488 | 12 / 20 | 0.0224 |
| 2 | always_on - off | return | 20 | -0.4767 | 0.1237 | [-0.5346, -0.4188] | <0.001 | <0.001 | 0 / 20 | 0.0817 |
| 2 | always_on - off | max_dd | 20 | -0.1045 | 0.0540 | [-0.1297, -0.0792] | <0.001 | <0.001 | 1 / 20 | 0.0356 |
| 2 | hybrid - always_on | return | 20 | +0.3096 | 0.1223 | [+0.2524, +0.3669] | <0.001 | <0.001 | 20 / 20 | 0.0808 |
| 2 | hybrid - always_on | max_dd | 20 | +0.1031 | 0.0478 | [+0.0807, +0.1255] | <0.001 | <0.001 | 19 / 20 | 0.0316 |
| 2 | hybrid - gated | return | 20 | +0.0206 | 0.1031 | [-0.0277, +0.0689] | 0.3825 | 0.2943 | 12 / 20 | 0.0681 |
| 2 | hybrid - gated | max_dd | 20 | -0.0087 | 0.0530 | [-0.0335, +0.0161] | 0.4695 | 0.3118 | 8 / 20 | 0.0350 |
| 2 | hybrid - off | return | 20 | -0.1671 | 0.0930 | [-0.2106, -0.1236] | <0.001 | <0.001 | 0 / 20 | 0.0614 |
| 2 | hybrid - off | max_dd | 20 | -0.0014 | 0.0366 | [-0.0185, +0.0157] | 0.8673 | 0.7562 | 9 / 20 | 0.0242 |
| 1 (post-hoc) | gated - always_on | return | 5 | +0.2155 | 0.0533 | [+0.1493, +0.2817] | <0.001 | 0.0625 | 5 / 5 | 0.0897 |
| 1 (post-hoc) | gated - always_on | max_dd | 5 | +0.0372 | 0.0088 | [+0.0263, +0.0482] | <0.001 | 0.0625 | 5 / 5 | 0.0149 |
| 1 (post-hoc) | gated - off | return | 5 | +0.1870 | 0.0872 | [+0.0787, +0.2952] | 0.0087 | 0.0625 | 5 / 5 | 0.1467 |
| 1 (post-hoc) | gated - off | max_dd | 5 | -0.3128 | 0.0311 | [-0.3514, -0.2742] | <0.001 | 0.0625 | 0 / 5 | 0.0523 |
| 1 (post-hoc) | always_on - off | return | 5 | -0.0286 | 0.1147 | [-0.1709, +0.1138] | 0.6071 | 0.6250 | 1 / 5 | 0.1929 |
| 1 (post-hoc) | always_on - off | max_dd | 5 | -0.3500 | 0.0284 | [-0.3853, -0.3148] | <0.001 | 0.0625 | 0 / 5 | 0.0478 |
| 1 (post-hoc) | hybrid - always_on | return | 5 | +0.1817 | 0.0857 | [+0.0752, +0.2881] | 0.0091 | 0.0625 | 5 / 5 | 0.1442 |
| 1 (post-hoc) | hybrid - always_on | max_dd | 5 | +0.1651 | 0.0344 | [+0.1223, +0.2078] | <0.001 | 0.0625 | 5 / 5 | 0.0579 |
| 1 (post-hoc) | hybrid - gated | return | 5 | -0.0339 | 0.0437 | [-0.0881, +0.0204] | 0.1581 | 0.1875 | 1 / 5 | 0.0735 |
| 1 (post-hoc) | hybrid - gated | max_dd | 5 | +0.1279 | 0.0347 | [+0.0848, +0.1709] | 0.0012 | 0.0625 | 5 / 5 | 0.0584 |
| 1 (post-hoc) | hybrid - off | return | 5 | +0.1531 | 0.0579 | [+0.0812, +0.2250] | 0.0041 | 0.0625 | 5 / 5 | 0.0974 |
| 1 (post-hoc) | hybrid - off | max_dd | 5 | -0.1849 | 0.0187 | [-0.2082, -0.1617] | <0.001 | 0.0625 | 0 / 5 | 0.0315 |
| 3 (post-hoc) | gated - always_on | return | 5 | +0.4084 | 0.2095 | [+0.1483, +0.6685] | 0.0121 | 0.0625 | 5 / 5 | 0.3523 |
| 3 (post-hoc) | gated - always_on | max_dd | 5 | +0.0407 | 0.0295 | [+0.0042, +0.0773] | 0.0365 | 0.1250 | 4 / 5 | 0.0495 |
| 3 (post-hoc) | gated - off | return | 5 | -0.7433 | 0.2426 | [-1.0445, -0.4421] | 0.0024 | 0.0625 | 0 / 5 | 0.4080 |
| 3 (post-hoc) | gated - off | max_dd | 5 | +0.0110 | 0.0282 | [-0.0240, +0.0460] | 0.4337 | 0.4375 | 3 / 5 | 0.0474 |
| 3 (post-hoc) | always_on - off | return | 5 | -1.1517 | 0.3107 | [-1.5375, -0.7659] | 0.0012 | 0.0625 | 0 / 5 | 0.5226 |
| 3 (post-hoc) | always_on - off | max_dd | 5 | -0.0298 | 0.0246 | [-0.0603, +0.0008] | 0.0537 | 0.0625 | 0 / 5 | 0.0414 |
| 3 (post-hoc) | hybrid - always_on | return | 5 | +0.5136 | 0.2131 | [+0.2491, +0.7782] | 0.0057 | 0.0625 | 5 / 5 | 0.3584 |
| 3 (post-hoc) | hybrid - always_on | max_dd | 5 | +0.0096 | 0.0140 | [-0.0077, +0.0270] | 0.1977 | 0.3125 | 3 / 5 | 0.0235 |
| 3 (post-hoc) | hybrid - gated | return | 5 | +0.1052 | 0.0576 | [+0.0337, +0.1768] | 0.0150 | 0.0625 | 5 / 5 | 0.0969 |
| 3 (post-hoc) | hybrid - gated | max_dd | 5 | -0.0311 | 0.0330 | [-0.0720, +0.0098] | 0.1026 | 0.1875 | 1 / 5 | 0.0555 |
| 3 (post-hoc) | hybrid - off | return | 5 | -0.6381 | 0.2068 | [-0.8949, -0.3813] | 0.0023 | 0.0625 | 0 / 5 | 0.3479 |
| 3 (post-hoc) | hybrid - off | max_dd | 5 | -0.0201 | 0.0308 | [-0.0584, +0.0181] | 0.2177 | 0.1875 | 1 / 5 | 0.0518 |

With n=5 (folds 1, 3) the smallest attainable two-sided Wilcoxon p is 0.0625: no fold-1/3 comparison can reach p<0.05 on Wilcoxon regardless of effect size; their CIs are wide.

## Mechanism: with the all-time-peak anchor the 30% zero-exposure state is absorbing

| fold | mode | seeds reaching >=29.5% DD | mean first day at >=29.5% DD | share of later days with zero gross exposure |
|---|---|---|---|---|
| 2 | off | 19 / 20 | 207 | 0.000 |
| 2 | always_on | 20 / 20 | 312 | 0.348 |
| 2 | gated | 19 / 20 | 205 | 0.054 |
| 2 | hybrid | 19 / 20 | 225 | 0.000 |
| 1 | off | 5 / 5 | 186 | 0.000 |
| 1 | always_on | 5 / 5 | 252 | 0.464 |
| 1 | gated | 5 / 5 | 231 | 0.102 |
| 1 | hybrid | 5 / 5 | 188 | 0.000 |
| 3 | off | 5 / 5 | 189 | 0.000 |
| 3 | always_on | 4 / 5 | 216 | 0.575 |
| 3 | gated | 4 / 5 | 203 | 0.023 |
| 3 | hybrid | 4 / 5 | 313 | 0.000 |

Under always-on, once drawdown reaches the hard level the book is flat on essentially every remaining day, so always-on returns mostly reflect WHEN the hard level was hit. Under gated, the shield switches off once beliefs resolve and the policy re-enters, so drawdowns exceed 30%.

## EXPLORATORY ADD-ON -- drawdown threshold sensitivity (eval only, fold 2, 20 seeds)

> **Not like-for-like safety comparisons.** The hard boundary itself moves across settings (25% / 30% / 35%), so a lower max DD at a tighter setting is partly definitional. Violation rate here is always measured at the fixed 20% level for comparability. Exploratory; no hypothesis test was pre-registered.

| soft / hard | mode | return mean +/- std | 95% CI | mean max DD | violation rate (DD>20%) | shield modifies weights | mean gross exposure |
|---|---|---|---|---|---|---|---|
| 15% / 25% | always_on | -0.1224 +/- 0.0552 | [-0.1482, -0.0965] | 0.2500 | 0.7858 | 0.5623 | 0.215 |
| 15% / 25% | gated | +0.0944 +/- 0.1316 | [+0.0328, +0.1560] | 0.4218 | 0.5601 | 0.0352 | 1.230 |
| 20% / 30% | always_on | -0.1808 +/- 0.0515 | [-0.2049, -0.1566] | 0.2998 | 0.7815 | 0.5969 | 0.248 |
| 20% / 30% | gated | +0.1082 +/- 0.1286 | [+0.0481, +0.1684] | 0.4117 | 0.5183 | 0.0438 | 1.244 |
| 25% / 35% | always_on | -0.2037 +/- 0.1595 | [-0.2784, -0.1291] | 0.3457 | 0.7266 | 0.6048 | 0.358 |
| 25% / 35% | gated | +0.1381 +/- 0.1452 | [+0.0701, +0.2060] | 0.3971 | 0.4632 | 0.0497 | 1.259 |

Consistency: the 20%/30% rows reproduce the main table's always-on / gated means (-0.1808 / +0.1082).
