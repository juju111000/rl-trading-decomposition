# rl-trading-decomposition (v1.1)

Code and evidence for the paper **"A Decomposition Study of Constrained Reinforcement Learning for Risk-Managed Trading."**

The study takes a constrained-RL trading pipeline apart and tests each component separately. The pipeline combines PPO, PPO-Lagrangian, and reward-shaping agents with a regime model and a risk shield, applied to a five-asset daily universe in a calendar walk-forward design. That design has three development folds (2006–2023) and one held-out fold, fold 4 (2024–2025), which was evaluated once and later re-scored under the corrected accounting described below (an accounting correction of the frozen evaluations, not a second look).

The components tested are the learned policy, the regime signal, the shield, and the overlay strategies, compared against rule-based baselines and oracle-signal arms. The central finding is the seed-expanded test on fold 2 (n = 20 seeds per arm, paired by verified identical initial networks). Under the corrected accounting the trained canonical agent earns a mean net return of **+29.6%**, while an **untrained** (random-initialization) network under the same pipeline earns **+39.2%**: a paired difference of **−9.6 pp** (95% CI [−16.4, −2.7], Holm-adjusted p = 0.018; the trained network beats its own initialization in 5 of 20 seeds). The return attributed to "the RL agent" is therefore not produced by learning.

This repository holds the final pipeline code, the regression tests, and every result file and figure the paper reports.

## Corrected evidence (R2)

Two accounting defects were found during review and repaired; **every result in the paper was regenerated under the corrected accounting**:

- **#8 Slippage units** (`portfolio_engine.py`): the volatility-slippage term used an annualized volatility estimate against a daily-scale parameter, inflating transaction costs roughly 16x (and was charged even when volatility slippage was disabled). It now uses daily-scale volatility and respects the flag.
- **#9 Regime look-ahead** (`regime_model.py`, `main.calibrate_regime`): the regime model's volatility thresholds were fit on each test window; they are now calibrated on the fold's training window and frozen for testing (a test environment without calibrated thresholds fails closed).

Regression tests: `tests/test_cost_units_d8.py` and `tests/test_regime_calibration_d9.py`.

**All paper numbers trace to the corrected results under [`results/r2/corrected/`](results/r2/corrected/)**; [`FINAL_REPORT.md`](results/r2/corrected/FINAL_REPORT.md) consolidates them, including a claim-by-claim table of submitted vs corrected values. The paper's tables and figures are in [`results/paper_exhibits/`](results/paper_exhibits/) (manifest: `exhibits_manifest.md`). Files elsewhere under `results/` (`hardening/`, `full_grid/`, `baselines/`, `fold4/`, `v2_dev/`, `figures/entropy_shock_2020.png`) are the **pre-correction v1.0 evidence**, kept for the record and superseded.

**Transaction costs.** Per unit traded: commission 10 bps + slippage 5 bps + 0.1 × daily volatility. Measured all-in cost on the fold-2 test window: **≈ 26 bps per unit of two-sided turnover** (≈ 53 bps per round trip); see `results/r2/corrected/defect_repairs.md`.

## Install

Developed and tested on Python 3.13.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repository root. Every script uses relative paths such as `./results/...` and `./data_cache/...`.

## Reproducing the tables and figures

### A. From the committed corrected CSVs (no data download, no training, about a minute)

| Paper output | Command | Source → output |
|---|---|---|
| Trained vs untrained, paired n=20: **+29.6% vs +39.2%, −9.6 pp**; per-arm stats, Welch/MWU, MDE | `python analyze_p1_interim.py` | `results/r2/corrected/hardening/hardening_results.csv` → `results/r2/corrected/p1_stats_interim.md` |
| All paper tables (T4–T6, expressibility grid) and figures (F2–F5, 2020 entropy trace), PNG + PDF | `python make_paper_exhibits.py` | `results/r2/corrected/**` → `results/paper_exhibits/` |
| Execution ladder analysis (configs a–h, band × cadence factorial) | `python analyze_ladder_seeded.py` | `results/r2/corrected/ladder_seeded.csv` → `ladder_seeded_analysis.md` |
| Fold 4, corrected vs frozen one-look | `python analyze_fold4_corrected.py` | `results/r2/corrected/fold4/` → `results/r2/corrected/fold4_corrected.md` |

To check the regenerated output against the committed version, run `git diff --stat results/`. The committed reports, tables and PNG figures should be unchanged; the PDF figures differ only in their embedded creation timestamp.

`results/r2/corrected/FINAL_REPORT.md` (`analyze_final_r2c.py`) and the oracle-mechanism probes (`analyze_oracle_anomaly.py`, `r2c_preedit_verification.py`) replay trained checkpoints, which are not distributed, and `analyze_final_r2c.py` also compares against pre-correction reports that are not part of this repository; their outputs are committed.

### B. Figures and evaluations that need price data (download only, no training)

```bash
python scripts/download_data.py                   # fetch raw prices into ./data_cache/
R2_DIR=./results/r2/corrected python r2_expressibility.py   # -> results/r2/corrected/expressibility_grid.*
#   (Windows PowerShell: $env:R2_DIR="./results/r2/corrected"; python r2_expressibility.py)
python run_baselines_corrected.py                 # -> results/r2/corrected/baselines_corrected.*
python analyze_ew_decomposition.py                # -> results/r2/corrected/ew_decomposition.md
python plot_entropy_shock_2020.py                 # -> results/figures/corrected/entropy_shock_2020.png
```

### C. Full corrected re-run from scratch (training; GPU optional, many CPU-hours)

Model checkpoints (`*.pt`) are not distributed, so every stage has to be retrained. The corrected master batch (P1 seed expansion, P3 execution ladder, P5 main grid, P2′ no-smoothing oracle arms, and the eval-only E2/E3 stages) is resumable per combination:

```bash
python scripts/download_data.py
python run_r2c.py --include-p2prime     # or detached: python launch_r2c.py --include-p2prime; poll: python check_r2c_progress.py
python analyze_p1_interim.py && python analyze_ladder_seeded.py
python make_paper_exhibits.py
```

The held-out fold-4 re-scoring (`run_fold4_corrected.py`) evaluates the frozen fold-4 artifacts once under the corrected accounting and refuses to run a second time.

### Tests

```bash
python -m pytest tests/ -q
```

Some tests load real price data. Run `python scripts/download_data.py` before them.

## Repository map

```
config.py                 canonical configuration (dataclasses, shield/regime/trainer settings)
data_loader.py            yfinance download, caching, calendar alignment, data cleaning
features.py               causal feature engineering (rolling windows, warmup handling)
regime_model.py           heuristic / learned regime models, belief formation, train-window calibration (#9)
policy_shield.py          rule-based, projection, uncertainty-aware, trailing-anchor shields
portfolio_engine.py       NAV / position accounting, transaction costs (#8)
trading_env.py            constrained trading MDP (gymnasium env)
networks.py, trainer.py   actor-critic networks; PPO, PPO-Lagrangian, reward-shaping trainers
evaluation.py             metrics and walk-forward evaluation
overlay_strategies.py     RC-Overlay / Gated-RL overlay strategies
main.py                   pipeline entry points, walk-forward fold definitions, calibrate_regime()
seeding.py                seed_everything()
logging_utils.py, evaluate_saved.py
r2_common.py              corrected-tree paths and a fail-closed guard against loading fold-4 data
run_r2c.py, launch_r2c.py, check_r2c_progress.py, run_r2.py
                          corrected master batch (P1-P5, P2') and its helpers
r2_oracle.py, r2_expressibility.py, r2_gated_same_policy.py, r2_belief_validation.py,
r2_binding_frequency.py, r2_defect_repairs.py
                          oracle arms, expressibility grid, same-policy shield modes, belief validation
run_baselines_corrected.py, run_fold4_corrected.py
                          rule-based baselines and fold-4 re-scoring under corrected accounting
analyze_*.py, r2c_preedit_verification.py
                          statistics and reports for results/r2/corrected/
make_paper_exhibits.py, scripts/paper_style.py
                          paper tables and figures (shared style and palette)
run_paper_experiments.py, run_full_grid.py, launch_full_grid.py, check_full_grid_progress.py,
run_single_experiment.py, run_baselines.py, run_dev_comparison.py, run_rc_overlay_forensics.py,
run_v2_sweep.py, run_validation_workstream.py, run_hardening.py, launch_hardening.py,
check_hardening_progress.py, analyze_hardening.py, run_fold4_grid.py, run_fold4_evaluation.py
                          v1.0 pipeline stages (reused by the corrected batch)
plot_entropy_shock_2020.py  regime-entropy diagnostic (COVID-19 shock, 2020)
scripts/download_data.py  re-fetch raw price data
tests/                    regression tests (incl. test_cost_units_d8.py, test_regime_calibration_d9.py)
results/r2/corrected/     corrected evidence (reports, summary CSVs, manifests)
results/paper_exhibits/   paper tables (CSV) and figures (PNG + PDF)
results/ (other)          pre-correction v1.0 evidence, superseded
```

## Data notes

- **Source and re-download.** Daily OHLCV data comes from Yahoo Finance through `yfinance`. Raw price data is not redistributed here. `python scripts/download_data.py` re-fetches the exact date ranges the pipeline requests and caches them under `./data_cache/`, which is git-ignored. Each fold's range starts 40 calendar days before its nominal train start, to warm up the features. Yahoo occasionally revises historical data, so a fresh download can differ slightly from the data behind the committed CSVs.
- **Universe.** `^GSPC` (S&P 500), `^DJI` (Dow Jones), `000001.SS` (SSE Composite), `^GDAXI` (DAX), and `CL=F` (WTI crude futures).
- **DAX substitution.** The European slot was originally `^STOXX50E` (Euro Stoxx 50). Its Yahoo history starts only on 2007-03-30, which left fold 1's 2006–2007 training window with 158 trading days instead of about 510. It also reports zero volume before mid-2013. `^GDAXI`, whose history goes back to 2000, replaced it everywhere. Results computed on the Stoxx universe are not comparable and are not included. The cache filename includes the ticker set, so cached data from different universes can never be mixed.
- **WTI negative-price masking.** `CL=F` printed non-positive prices on 2020-04-20 (Close = −37.63) and 2020-04-21 (negative Open and Low). That was a real front-month delivery event, but a continuous, roll-free series cannot represent it, and the pipeline takes logarithms of prices. `DataLoader.align_data()` masks any non-positive OHLC value to NaN and forward-fills it from the last valid price, and prints a warning naming the dates. For those two sessions, results involving `CL=F` use the prior day's price.
- **Rolling-window warmup.** Rolling features use `min_periods = min(window, 20)`. Combined with the 40-day pre-fold buffer and a leading-NaN trim (never a backfill), this keeps warmup short without introducing lookahead.
- **Checkpoints.** Trained model checkpoints are not distributed; the committed manifests record their SHA-256 hashes.

## Seeds and determinism

Every entry point calls `seeding.seed_everything(seed)`, which seeds Python's `random`, NumPy, and PyTorch (CPU and CUDA). It also sets deterministic cuDNN flags and `PYTHONHASHSEED` for subprocesses. Seeds are fixed in the code:

- Main grid and execution ladder: seeds 1–5.
- Seed-expansion study and oracle arms: seeds 1–20 per arm. The trained and untrained arms are paired: each trained run's initial network is checked tensor-equal to the untrained arm's before any gradient step.

`tests/test_determinism.py` checks that two runs with the same seed produce identical training metrics, and that different seeds do not. Bit-for-bit reproduction of the committed numbers is guaranteed only for the CSV-to-table step in section A. Retraining on different hardware, different PyTorch/BLAS versions, a GPU instead of a CPU, or re-downloaded price data can change individual runs numerically, although the paper's conclusions are reported at the level of seed distributions.

## License

MIT; see [LICENSE](LICENSE).
