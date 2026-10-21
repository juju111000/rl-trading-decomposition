# Expressibility grid -- can the canonical execution layer express a known signal?

_Generated 2026-10-13 by `r2_expressibility.py`. Dev fold 2, test window 2015-2017, corrected accounting (audit defects #8, #9 repaired). Deterministic, eval-only, no learning. **NON-DEPLOYABLE:** the signal is the oracle feature (sign of the next-5-trading-day return, kept with probability = accuracy, fixed noise draw). Fold 4 never loaded._

## Canonical execution parameters (from `run_paper_experiments.CANONICAL_FOLD2_CONFIG_OVERRIDES`)

- **Action smoothing beta = 0.1** (confirmed; not 0.2 -- 0.2 was the earlier Phase-3 'Step 4' config): on a decision day the target is `w = 0.9 * w_prev_target + 0.1 * a` (`trading_env.py:314`).
- No-trade band 0.10 per asset, measured against the CURRENT (drifted) weight; decisions every 5 trading days.
- Action decoding: per-asset clip +/-0.8 (`max_position_size`), then gross cap 2.0 (proportional rescale). With all five signals nonzero, size 0.4 gives gross exactly 2.0 and **size 0.8 (gross 4.0) is rescaled to 0.4 per asset** -- so 0.4 and 0.8 differ only on days where some signals are 0.
- Consequence: a full signal flip moves the smoothed target by 2 x 0.1 x s per decision -- 0.04 (s=0.2) or 0.08 (s=0.4 and capped 0.8) -- always below the 0.10 band. A trade happens only after the smoothed target has drifted > 0.10 away from the current weight over several same-direction decisions.

## Gate (pre-stated before computing: +/-0.8 at 100% under full canonical execution nets >= +20%)

**FAIL** -- net return +0.0726 (0.44 trades per decision day; 91.3% of desired per-asset changes blocked by the band). Even the largest expressible signal-following rule cannot convert perfect 5-day foresight into profit under canonical execution. **P2 HELD**: oracle training would characterize the execution layer, not learning.

## Grid

Net return = through the env after corrected costs; gross P&L = frictionless replay of the weights actually held; trades / decision day = assets traded (max 5); band-blocked = share of desired per-asset changes not executed because |target - current| <= 0.10; fully blocked = decision days with a desired change but no trade.

| accuracy | size / asset | effective max size (post clip + cap) | execution | net return | gross P&L (held) | txn cost / NAV | max DD | trades / decision day | band-blocked share | decisions fully blocked | mean gross exposure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100% | 0.2 | 0.20 | canonical | -0.0201 | -0.0101 | 0.0035 | 0.0717 | 0.08 | 0.983 | 0.917 | 0.144 |
| 100% | 0.2 | 0.20 | canonical without smoothing | +14.8034 | +22.4447 | 0.4286 | 0.0272 | 2.61 | 0.478 | 0.071 | 0.976 |
| 100% | 0.4 | 0.40 | canonical | +0.0722 | +0.0967 | 0.0239 | 0.0922 | 0.46 | 0.909 | 0.628 | 0.338 |
| 100% | 0.4 | 0.40 | canonical without smoothing | +223.0598 | +444.5217 | 0.8504 | 0.0528 | 2.66 | 0.468 | 0.064 | 1.909 |
| 100% | 0.8 | 0.40 | canonical | +0.0726 | +0.0973 | 0.0230 | 0.0922 | 0.44 | 0.913 | 0.635 | 0.343 |
| 100% | 0.8 | 0.40 | canonical without smoothing | +223.3258 | +445.8528 | 0.8527 | 0.0528 | 2.69 | 0.463 | 0.064 | 1.914 |
| 60% | 0.2 | 0.20 | canonical | -0.0324 | -0.0254 | 0.0039 | 0.0384 | 0.09 | 0.982 | 0.917 | 0.071 |
| 60% | 0.2 | 0.20 | canonical without smoothing | +0.0060 | +0.5104 | 0.4202 | 0.2136 | 2.51 | 0.499 | 0.026 | 0.997 |
| 60% | 0.4 | 0.40 | canonical | -0.0783 | -0.0393 | 0.0270 | 0.1076 | 0.56 | 0.887 | 0.564 | 0.387 |
| 60% | 0.4 | 0.40 | canonical without smoothing | -0.0202 | +1.0799 | 0.8408 | 0.3884 | 2.51 | 0.499 | 0.026 | 1.998 |
| 60% | 0.8 | 0.40 | canonical | -0.0752 | -0.0361 | 0.0273 | 0.1049 | 0.57 | 0.886 | 0.551 | 0.386 |
| 60% | 0.8 | 0.40 | canonical without smoothing | -0.0216 | +1.0796 | 0.8423 | 0.3884 | 2.52 | 0.496 | 0.026 | 2.000 |

## Finding

Execution discipline rate-limits information expression: smoothing (beta 0.1) caps how far the target can move per decision, and the 10% band then suppresses any move smaller than 0.10, so a signal that changes sign on a ~5-day horizon is largely filtered out before it reaches the book. Removing smoothing alone (band and cadence kept) shows how much of the gap is due to the smoothing/band interaction rather than to costs. The deterministic rule is a fixed policy, so each cell is a single value, not a seed distribution.
