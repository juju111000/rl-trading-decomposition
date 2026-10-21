# Audit defects #8 and #9 -- repairs and measured effect

_Generated 2026-10-13 by `r2_defect_repairs.py`. Fold 4 never loaded._

## D8 -- Slippage units (`portfolio_engine.py`)

**Defect.** Slippage = |trade| x (slippage_rate + vol x slippage_vol_multiplier). `vol` defaulted to 0.02 (daily scale) but `mark_to_market` overwrote it with *annualized* volatility (std x sqrt(252)), inflating the volatility term ~15.9x; the term was also charged when `use_volatility_slippage=False` (affecting the validation workstream's 'flat 10 bps' V3 variant). **Fix:** daily-scale estimate (std of the last 20 daily log returns) and the term applied only when the flag is set. Parameters unchanged: fee 10 bps, base slippage 5 bps, multiplier 0.1. Regression test: `tests/test_cost_units_d8.py`.

**Effective all-in cost per unit of turnover** (turnover = sum |dw|, two-sided; fold-2 test env, 151 rebalances):

| | mean | min | max |
|---|---|---|---|
| corrected | 0.2645% | 0.1937% | 0.4265% |
| pre-fix (same trades, annualized vol) | 1.9676% | 0.8439% | 4.5395% |

| asset | mean daily vol | corrected cost / unit | pre-fix cost / unit |
|---|---|---|---|
| 000001.SS | 0.0126 | 0.2763% | 2.1551% |
| CL=F | 0.0230 | 0.3800% | 3.8017% |
| ^DJI | 0.0066 | 0.2158% | 1.1950% |
| ^GDAXI | 0.0106 | 0.2564% | 1.8387% |
| ^GSPC | 0.0067 | 0.2170% | 1.2142% |

Per unit of two-sided turnover the corrected all-in cost is ~0.26%; a round trip (buy then later sell the same notional = 2 units) costs ~0.53%. **Note:** this is below the ~0.5-0.7% per unit you anticipated. With the configured parameters (10 + 5 bps + 0.1 x daily vol), 0.5-0.7% per unit would require daily vols of 3.5-5.5% on every asset. The parameters were not changed; if a higher cost level is intended, it is a parameter decision, separate from this units defect.

## D9 -- Regime look-ahead (`regime_model.HeuristicRegimeModel.fit`)

**Defect.** Test envs called `fit(test_aligned)`, which set the volatility-percentile thresholds from the whole test window. **Fix:** only a training-mode fit computes thresholds; it records them in `config.regime.calibrated_vol_thresholds` (saved with every checkpoint). Any other fit uses them frozen, reads no test values, and raises `RegimeCalibrationError` if none exist (fail-closed). Regression test: `tests/test_regime_calibration_d9.py`.

**Effect on fold 1-3 test-window beliefs** (flat action; old = thresholds fit on the test window, new = thresholds fit on the fold's training window):

| fold | operator | thresholds old (test-fit) | thresholds new (train-fit) | regime call agrees old vs new | mean abs change in P(Bear) | gate active share old -> new |
|---|---|---|---|---|---|---|
| 1 | hysteresis | (0.171, 0.247) | (0.083, 0.137) | 0.728 | 0.2359 | 0.274 -> 0.449 |
| 1 | softmax | (0.171, 0.247) | (0.083, 0.137) | 0.760 | 0.1303 | 1.000 -> 1.000 |
| 2 | hysteresis | (0.075, 0.117) | (0.100, 0.147) | 0.974 | 0.0257 | 0.123 -> 0.095 |
| 2 | softmax | (0.075, 0.117) | (0.100, 0.147) | 0.961 | 0.0367 | 1.000 -> 1.000 |
| 3 | hysteresis | (0.128, 0.196) | (0.071, 0.116) | 0.816 | 0.1731 | 0.178 -> 0.340 |
| 3 | softmax | (0.128, 0.196) | (0.071, 0.116) | 0.831 | 0.1083 | 1.000 -> 1.000 |
