# rl-trading-decomposition

Code and evidence for the paper **"A Decomposition Study of Constrained Reinforcement Learning for Risk-Managed Trading."**

The study takes a constrained-RL trading pipeline apart and tests each component separately. The pipeline combines PPO, PPO-Lagrangian, and reward-shaping agents with a regime model and a risk shield, applied to a five-asset daily universe in a calendar walk-forward design. That design has three development folds (2006–2023) and one held-out fold, fold 4 (2024–2025), which was evaluated once.

The components tested are the learned policy, the regime signal, the shield, and the overlay strategies, compared against rule-based baselines and oracle-signal arms. The central finding is the seed-expanded test on fold 2 (n = 20 seeds per arm). The trained canonical agent earns a mean net return of **+7.0%**, while an **untrained** (random-initialization) network under the same pipeline earns **+29.0%** (Welch p < 0.001). The return attributed to "the RL agent" is therefore not produced by learning.

This repository holds the final pipeline code, the regression tests, and every result CSV and figure the paper reports.

## Install

Developed and tested on Python 3.13.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repository root. Every script uses relative paths such as `./results/...` and `./data_cache/...`.

## Reproducing the tables and figures

### A. From the committed CSVs (no data download, no training, under a minute)

| Paper output | Command | Source CSV → output |
|---|---|---|
| Seed-expansion table: per-arm mean/std/95% CI, Welch + Mann–Whitney tests, MDE; trained vs. untrained **+7.0% vs. +29.0%** | `python analyze_hardening.py` | `results/hardening/hardening_results.csv` → `results/hardening/stats_report.md` |
| Net-return distribution figure | `python analyze_hardening.py` (same run) | → `results/hardening/hardening_distributions.png` |

To check the regenerated output against the committed version, run `git diff --stat results/hardening/`. It should show no changes to `stats_report.md`.

Several tables are read directly from committed CSVs, with no script step needed:

| Paper table | Committed CSV |
|---|---|
| Main walk-forward grid (3 trainers × 4 shield arms × 3 folds × 5 seeds) | `results/full_grid/20260803_174817/results.csv` |
| Shield-component ablation | `results/full_grid/20260803_174817/shield_ablation_results.csv` |
| Inactivity-penalty ablation | `results/full_grid/20260803_174817/inactivity_ablation_results.csv` |
| Rule-based baselines | `results/baselines/baselines_results.csv` |
| Development-fold strategy comparison | `results/v2_dev/dev_comparison.csv` |
| Return attribution / forensics decomposition | `results/v2_dev/forensics/attribution.csv`, `timing_accuracy.csv`, `ablation_grid.csv`, `regime_timing_fold{1,2,3}.csv` |
| v2 redesign sweep and selection | `results/v2_dev/v2_sweep/*.csv` |
| Validation workstream (oracle signal V1, learning check V2, V3) | `results/v2_dev/validation/v{1,2,3}_results.csv` |
| Fold 4 (2024–2025) evaluation | `results/fold4/fold4_results.csv`, `results/fold4/grid/{results,summary}.csv` |

### B. Figures that need price data (download only, no training)

```bash
python scripts/download_data.py                   # fetch raw prices into ./data_cache/
python plot_entropy_shock_2020.py                 # -> results/figures/entropy_shock_2020.png
python run_rc_overlay_forensics.py                # -> results/v2_dev/forensics/*.csv + regime_timeline_fold{1,2,3}.png
python run_v2_sweep.py                            # -> results/v2_dev/v2_sweep/*.csv
python run_baselines.py --output-dir ./results/baselines
```

### C. Full re-run from scratch (training; GPU optional, many CPU-hours)

Model checkpoints (`*.pt`) are not distributed, so every downstream stage has to be retrained in this order. The later scripts look for the grid's checkpoints at the hard-coded path `results/full_grid/20260803_174817/`, so pass that exact output directory:

```bash
python scripts/download_data.py
python run_full_grid.py --output-dir ./results/full_grid/20260803_174817 --timesteps 300000   # ~26 h; resumable
#   (or detached: python launch_full_grid.py --output-dir ./results/full_grid/20260803_174817 --timesteps 300000
#    and poll with: python check_full_grid_progress.py --dir ./results/full_grid/20260803_174817)
python run_dev_comparison.py
python run_validation_workstream.py
python run_hardening.py --output-dir ./results/hardening       # or launch_hardening.py + check_hardening_progress.py
python analyze_hardening.py
python run_fold4_grid.py && python run_fold4_evaluation.py     # held-out fold 4
```

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
regime_model.py           heuristic / learned regime models and belief formation
policy_shield.py          rule-based, projection, uncertainty-aware, trailing-anchor shields
portfolio_engine.py       NAV / position accounting, transaction costs
trading_env.py            constrained trading MDP (gymnasium env)
networks.py, trainer.py   actor-critic networks; PPO, PPO-Lagrangian, reward-shaping trainers
evaluation.py             metrics and walk-forward evaluation
overlay_strategies.py     RC-Overlay / Gated-RL overlay strategies
main.py                   pipeline entry points, walk-forward fold definitions
seeding.py                seed_everything()
logging_utils.py, evaluate_saved.py
run_paper_experiments.py, run_full_grid.py, launch_full_grid.py, check_full_grid_progress.py
                          main grid (+ shield and inactivity ablations)
run_single_experiment.py  single-config runner
run_baselines.py          rule-based baselines
run_dev_comparison.py, run_rc_overlay_forensics.py, run_v2_sweep.py
                          development-fold comparison, attribution forensics, v2 sweep
run_validation_workstream.py  oracle-signal and learning-verification checks
run_hardening.py, launch_hardening.py, check_hardening_progress.py, analyze_hardening.py
                          n=20 seed-expansion study and its statistics
run_fold4_grid.py, run_fold4_evaluation.py   held-out fold 4
plot_entropy_shock_2020.py  regime-entropy figure (COVID-19 shock, 2020)
scripts/download_data.py  re-fetch raw price data
tests/                    regression tests
results/                  committed evidence (CSVs, reports, figures)
```

## Data notes

- **Source and re-download.** Daily OHLCV data comes from Yahoo Finance through `yfinance`. Raw price data is not redistributed here. `python scripts/download_data.py` re-fetches the exact date ranges the pipeline requests and caches them under `./data_cache/`, which is git-ignored. Each fold's range starts 40 calendar days before its nominal train start, to warm up the features. Yahoo occasionally revises historical data, so a fresh download can differ slightly from the data behind the committed CSVs.
- **Universe.** `^GSPC` (S&P 500), `^DJI` (Dow Jones), `000001.SS` (SSE Composite), `^GDAXI` (DAX), and `CL=F` (WTI crude futures).
- **DAX substitution.** The European slot was originally `^STOXX50E` (Euro Stoxx 50). Its Yahoo history starts only on 2007-03-30, which left fold 1's 2006–2007 training window with 158 trading days instead of about 510. It also reports zero volume before mid-2013. `^GDAXI`, whose history goes back to 2000, replaced it everywhere. Results computed on the Stoxx universe are not comparable and are not included. The cache filename includes the ticker set, so cached data from different universes can never be mixed.
- **WTI negative-price masking.** `CL=F` printed non-positive prices on 2020-04-20 (Close = −37.63) and 2020-04-21 (negative Open and Low). That was a real front-month delivery event, but a continuous, roll-free series cannot represent it, and the pipeline takes logarithms of prices. `DataLoader.align_data()` masks any non-positive OHLC value to NaN and forward-fills it from the last valid price, and prints a warning naming the dates. For those two sessions, results involving `CL=F` use the prior day's price.
- **Rolling-window warmup.** Rolling features use `min_periods = min(window, 20)`. Combined with the 40-day pre-fold buffer and a leading-NaN trim (never a backfill), this keeps warmup short without introducing lookahead.

## Seeds and determinism

Every entry point calls `seeding.seed_everything(seed)`, which seeds Python's `random`, NumPy, and PyTorch (CPU and CUDA). It also sets deterministic cuDNN flags and `PYTHONHASHSEED` for subprocesses. Seeds are fixed in the code:

- Main grid: seeds 1–5.
- Seed-expansion study: seeds 1–20 per arm.

`tests/test_determinism.py` checks that two runs with the same seed produce identical training metrics, and that different seeds do not. Bit-for-bit reproduction of the committed numbers is guaranteed only for the CSV-to-table step in section A. Retraining on different hardware, different PyTorch/BLAS versions, a GPU instead of a CPU, or re-downloaded price data can change individual runs numerically, although the paper's conclusions are reported at the level of seed distributions.

## License

MIT; see [LICENSE](LICENSE).
