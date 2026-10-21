# R2 / E5 -- Oracle feature verification (pre-launch)

_Generated 2026-10-13 by `r2_oracle.py`. **NON-DEPLOYABLE / DIAGNOSTIC ONLY: oracle features use future prices by construction (sign of the next-5-day return).** Fold 4 never loaded._

## 1. Forward-label window alignment at decision times

- Env timing (`trading_env.step`): at step t the policy observes feature row t, the decision executes at close p[t], and the position is marked over p[t]->p[t+1]->...; with `decision_frequency_days=5` the next decision is at step t+5 (step 0 of an episode is a decision day).
- Empirical cadence check (test env, k=5, smoothing/band off for this check only, target flipped each decision): trades on 156 steps; steps not divisible by 5 with a trade: 0 -> decisions only at t = 0, 5, 10, ...
- Oracle feature at row t = sign(log(p[t+5]/p[t])) = sign of the return over exactly the holding window [t, t+5] of the decision taken at t. The next decision (t+5) observes sign(log(p[t+10]/p[t+5])): the two label windows share only the boundary price p[t+5], which is the next decision's own execution price (known at that decision), so there is no overlap leakage into the next decision's label window.
- Training uses rolling episodes starting at arbitrary rows; the feature exists on every row, so whichever row is a decision day, its oracle value describes exactly that decision's 5-day holding window.

## 2. Isolation of future information

- accuracy 60%: base feature columns identical to the canonical features (train: True, test: True); oracle columns appended = 5; label computed within each slice (last 5 rows of train and of test set to 0).
- accuracy 100%: base feature columns identical to the canonical features (train: True, test: True); oracle columns appended = 5; label computed within each slice (last 5 rows of train and of test set to 0).
- Only `build_oracle5_slices` creates oracle columns, and only the E5 oracle arms and the signal-following benchmark call it. The canonical, untrained, E4 and E6 arms build features via `_load_fold2_slices` alone (see `run_r2.py`). No other feature, and no non-oracle arm, receives future information.

## 3. Realized post-flip accuracy (rows with a defined, nonzero 5-day label)

| arm | slice | rows | accuracy |
|---|---|---|---|
| oracle60_5d | train | 6414 | 0.5915 |
| oracle60_5d | test | 3866 | 0.5859 |
| oracle100_5d | train | 6414 | 1.0000 |
| oracle100_5d | test | 3866 | 1.0000 |

## 4. Deterministic signal-following benchmark (no learning) -- NON-DEPLOYABLE

Rule: target weight = 0.2 x signal per asset (long +, short -, equal magnitude; gross 1.0 when all five signals are nonzero), through the canonical env (smoothing, band, cadence, cap, costs). Deterministic: the oracle noise draw is fixed, so each row is a single value, not a seed distribution.

Columns: *net return* (through the env, after costs); *gross P&L* = frictionless replay of the weights the env actually held; *txn cost* = sum of daily cost / NAV; *ideal* = frictionless P&L of holding 0.2 x signal over every 5-day decision window with no execution model at all (the signal's raw value).

| signal | execution | net return | max DD | mean daily turnover | gross P&L (held weights) | txn cost / NAV | ideal frictionless signal P&L |
|---|---|---|---|---|---|---|---|
| oracle60_5d | canonical (smoothing 0.1, band 0.10, k=5, cap 2.0, costs) | -0.0324 | 0.0384 | 0.0020 | -0.0254 | 0.0039 | +0.5293 |
| oracle60_5d | DIAGNOSTIC: canonical but no smoothing | +0.0060 | 0.2136 | 0.1980 | +0.5104 | 0.4202 | +0.5293 |
| oracle100_5d | canonical (smoothing 0.1, band 0.10, k=5, cap 2.0, costs) | -0.0201 | 0.0717 | 0.0018 | -0.0101 | 0.0035 | +24.2187 |
| oracle100_5d | DIAGNOSTIC: canonical but no smoothing | +14.8034 | 0.0272 | 0.2029 | +22.4447 | 0.4286 | +24.2187 |

> **Cost-model finding (affects every arm and every existing result; NOT changed here).** `Portfolio.execute_trades` charges slippage = |trade| x vol x 0.1 (`slippage_vol_multiplier`). `vol` starts at a 0.02 default (daily scale) but `Portfolio.mark_to_market` overwrites it with ANNUALIZED volatility (std x sqrt(252), `portfolio_engine.py:323`): 0.17 (S&P) to 0.58 (WTI) on this window. Measured all-in cost per unit of turnover: 0.35% on day 0, then 3.3-3.9% thereafter -- roughly 10x the day-0 level. This units inconsistency plausibly explains why execution costs dominate results throughout the project, and why this benchmark nets a loss despite a large frictionless signal value.
