# R2 / E3 -- Belief validation and fixed-policy gating comparison

_R2 revision, generated 2026-10-16 by `r2_belief_validation.py`. Evaluation only; fold 4 never loaded (fail-closed guard). Manifest: `belief_validation_manifest.json`._

> **CORRECTED ACCOUNTING (audit defects #8, #9 repaired):** regime thresholds calibrated on each fold's training window and frozen at test time; corrected-batch checkpoints.

## Part A -- Belief validation against prespecified historical proxies (folds 1-3 test windows)

> **The proxy labels are PRESPECIFIED HISTORICAL PROXIES, not ground truth.** They were fixed before looking at results (definition in the header of `r2_belief_validation.py`, no tuning): universe = equal-weight index of the 5 assets; Bear = trailing-60d return < -10% OR drawdown > 15%; Bull = trailing-60d return > +10% AND drawdown < 5%; Sideways otherwise. The regime model scores trend and volatility on the S&P 500 alone, so some disagreement is definitional.

> Hysteresis is evaluated by its **committed** regime, softmax by its belief **argmax**; the Brier score uses each operator's full belief vector vs the one-hot proxy (lower = better calibrated to the proxy; 0 = perfect, 2 = maximally wrong). Gate = P(Bear) > 0.7 OR normalized entropy > 0.5 computed from that operator's beliefs. **Lower entropy alone is not evidence of better prediction** -- a confidently wrong operator also has low entropy; read entropy together with agreement and Brier.

| fold | operator | days | proxy Bull/Bear/Sideways days | agreement | Brier | gate precision (Bear) | gate recall (Bear) | gate active share | mean entropy, proxy-Bear days | mean entropy, proxy-Bull days |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | hysteresis | 780 | 0/672/108 | 0.496 | 0.943 | 0.917 | 0.478 | 0.449 | 0.135 | n/a (no proxy-Bull days) |
| 1 | softmax | 780 | 0/672/108 | 0.481 | 0.581 | 0.862 | 1.000 | 1.000 | 0.830 | n/a (no proxy-Bull days) |
| 2 | hysteresis | 779 | 54/185/540 | 0.746 | 0.479 | 0.811 | 0.324 | 0.095 | 0.227 | 0.001 |
| 2 | softmax | 779 | 54/185/540 | 0.738 | 0.497 | 0.237 | 1.000 | 1.000 | 0.917 | 0.855 |
| 3 | hysteresis | 1040 | 117/60/863 | 0.688 | 0.545 | 0.169 | 1.000 | 0.340 | 0.279 | 0.153 |
| 3 | softmax | 1040 | 117/60/863 | 0.700 | 0.488 | 0.058 | 1.000 | 1.000 | 0.527 | 0.936 |

Confusion matrices (rows = proxy label, columns = operator's regime call; order Bull, Bear, Sideways):

**Fold 1, hysteresis**

| proxy \ predicted | Bull | Bear | Sideways |
|---|---|---|---|
| Bull | 0 | 0 | 0 |
| Bear | 0 | 285 | 387 |
| Sideways | 0 | 6 | 102 |

**Fold 1, softmax**

| proxy \ predicted | Bull | Bear | Sideways |
|---|---|---|---|
| Bull | 0 | 0 | 0 |
| Bear | 0 | 276 | 396 |
| Sideways | 0 | 9 | 99 |

**Fold 2, hysteresis**

| proxy \ predicted | Bull | Bear | Sideways |
|---|---|---|---|
| Bull | 0 | 0 | 54 |
| Bear | 0 | 41 | 144 |
| Sideways | 0 | 0 | 540 |

**Fold 2, softmax**

| proxy \ predicted | Bull | Bear | Sideways |
|---|---|---|---|
| Bull | 0 | 0 | 54 |
| Bear | 0 | 35 | 150 |
| Sideways | 0 | 0 | 540 |

**Fold 3, hysteresis**

| proxy \ predicted | Bull | Bear | Sideways |
|---|---|---|---|
| Bull | 0 | 14 | 103 |
| Bear | 0 | 60 | 0 |
| Sideways | 0 | 208 | 655 |

**Fold 3, softmax**

| proxy \ predicted | Bull | Bear | Sideways |
|---|---|---|---|
| Bull | 0 | 24 | 93 |
| Bear | 0 | 60 | 0 |
| Sideways | 0 | 195 | 668 |

## Part B -- Fixed-policy gating comparison: softmax-gated vs hysteresis-gated (fold 2, 20 seeds)

Same 20 canonical checkpoints, same regime-gated shield (inner = always-on `uncertainty_aware`, gate P(Bear) > 0.7 OR entropy > 0.5). **Only the belief operator driving the gate differs.** The policy's observations and the inner shield's regime inputs stay on the canonical hysteresis beliefs in both arms, so differences are attributable to gate timing alone.

Consistency check: hysteresis-gated returns vs E2 `gated` mode: max |difference| = 7.65e-17 (identical).

Proxy-Bear days in the fold-2 test window: 185 of 779.

> **This comparison is degenerate on fold 2.** The softmax operator's normalized entropy never falls below 0.706 in this window, so its gate (entropy > 0.5) is ON on 100% of days. Softmax-gated is therefore identical to the always-on shield (max |return difference| vs E2 always-on = 8.3e-17), and the table below is, in effect, always-on vs hysteresis-gated. It does not separate 'better belief timing' from 'shield always engaged'. Of the two gates, only hysteresis ever switches off.

| metric | hysteresis-gated mean +/- std | softmax-gated mean +/- std | paired delta (softmax - hysteresis) | 95% CI | paired t p | Wilcoxon p | seeds delta > 0 | MDE (paired, 80%) |
|---|---|---|---|---|---|---|---|---|
| net return | +0.1082 +/- 0.1286 | -0.1808 +/- 0.0515 | -0.2890 | [-0.3468, -0.2312] | <0.001 | <0.001 | 0 / 20 | 0.0816 |
| max drawdown | +0.4117 +/- 0.0697 | +0.2998 +/- 0.0004 | -0.1118 | [-0.1443, -0.0794] | <0.001 | <0.001 | 1 / 20 | 0.0458 |
| avg gross exposure | +1.2436 +/- 0.0929 | +0.2478 +/- 0.0581 | -0.9958 | [-1.0447, -0.9469] | <0.001 | <0.001 | 0 / 20 | 0.0690 |
| shield activation (gate-on) share | +0.0950 +/- 0.0000 | +1.0000 +/- 0.0000 | +0.9050 (identical every seed) | -- | deterministic, no test | -- | -- | -- |
| days shield modified weights | +0.0438 +/- 0.0138 | +0.5969 +/- 0.0590 | +0.5531 | [+0.5260, +0.5802] | <0.001 | <0.001 | 20 / 20 | 0.0383 |
| missed-stress days (proxy-Bear, gate off) | +125.0000 +/- 0.0000 | +0.0000 +/- 0.0000 | -125.0000 (identical every seed) | -- | deterministic, no test | -- | -- | -- |
| cum. log return on missed-stress days | -0.0778 +/- 0.0742 | +0.0000 +/- 0.0000 | +0.0778 | [+0.0431, +0.1126] | <0.001 | <0.001 | 17 / 20 | 0.0490 |

Gate activation and missed-stress days depend on beliefs alone, so they are identical across seeds within an operator; no test is reported for them. The return, drawdown, exposure and modification rows carry the seed-level evidence.
