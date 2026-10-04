#!/usr/bin/env python3
"""
Resumable driver for the full paper grid (run_paper_experiments.py with
--canonical-fold2-config), meant to be launched detached via
launch_full_grid.py and polled with check_full_grid_progress.py.

Always calls run_paper_experiments(resume=True) against a single, stable
output directory (the caller picks it once, up front, and reuses it across
restarts -- see launch_full_grid.py, which stamps one timestamp and passes
it through). Restarting this script with the SAME --output-dir picks up
exactly where a prior run left off: every (fold, shield, trainer, seed)
combo and every ablation variant/seed that already has a completed
eval_result.json on disk (see main.py's _load_completed_arm/_save_arm_result)
is skipped and reloaded rather than retrained. A combo that only got
partway (process died between training finishing and evaluation finishing)
is NOT considered complete and is retrained from scratch on restart --
there is no mid-training checkpoint to resume from, only whole-arm skip.

A background thread scans the output directory for eval_result.json files
every HEARTBEAT_INTERVAL_SECONDS and writes heartbeat.json with a coarse
completed/expected-total count. This is independent of exactly which arm
run_paper_experiments() is mid-training on, so it stays accurate across
process restarts and doesn't require any monkeypatching of main.py.

If this process dies outright (e.g. a transient data-download failure
during one seed's data load, which is NOT caught per-arm the way individual
arm training/eval failures are -- see run_paper_experiments.py's seed
loop), heartbeat.json is written with status="failed" and the traceback,
but recovery is a MANUAL relaunch with the same --output-dir, not automatic
self-restart. Thanks to eval_result.json markers, that relaunch resumes
cheaply from wherever it stopped.

Usage:
    python run_full_grid.py --output-dir ./results/full_grid/20260802_120000 --timesteps 300000
"""
import sys
import os
import json
import argparse
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

import run_paper_experiments as rpe
import main as main_module

HEARTBEAT_INTERVAL_SECONDS = 30


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _count_completed(output_dir: Path) -> int:
    return sum(1 for _ in output_dir.rglob("eval_result.json"))


def _expected_total(seeds, windows, shield_arms, trainer_arms,
                     include_shield_ablation, include_inactivity_ablation) -> int:
    total = len(seeds) * len(windows) * len(shield_arms) * len(trainer_arms)
    if include_shield_ablation:
        total += len(seeds) * len(main_module.SHIELD_COMPONENT_VARIANTS)
    if include_inactivity_ablation:
        total += len(seeds) * len(main_module.INACTIVITY_PENALTY_VARIANTS)
    return total


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Redirect this process's own stdout/stderr to a single log file,
    # independent of however the launcher started us (mirrors
    # run_dryrun.py's pattern).
    log_file = open(output_dir / "run.log", "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    heartbeat_path = output_dir / "heartbeat.json"
    (output_dir / "full_grid.pid").write_text(str(os.getpid()))

    windows = main_module.PAPER_WALK_FORWARD_WINDOWS
    shield_arms = main_module.SHIELD_ARMS
    trainer_arms = main_module.TRAINER_ARMS
    seeds = args.seeds
    include_shield_ablation = not args.no_shield_ablation
    include_inactivity_ablation = not args.no_inactivity_ablation

    expected_total = _expected_total(
        seeds, windows, shield_arms, trainer_arms,
        include_shield_ablation, include_inactivity_ablation
    )

    state = {
        'status': 'running',
        'started_at': _now_iso(),
        'updated_at': _now_iso(),
        'finished_at': None,
        'pid': os.getpid(),
        'output_dir': str(output_dir),
        'timesteps_per_run': args.timesteps,
        'seeds': seeds,
        'canonical_config': not args.no_canonical_config,
        'include_shield_ablation': include_shield_ablation,
        'include_inactivity_ablation': include_inactivity_ablation,
        'expected_total_combos': expected_total,
        'completed_combos': 0,
        'pct_complete': 0.0,
    }

    def write_heartbeat():
        tmp = heartbeat_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, heartbeat_path)

    stop_scanning = threading.Event()

    def scan_loop():
        while not stop_scanning.is_set():
            completed = _count_completed(output_dir)
            state['completed_combos'] = completed
            state['pct_complete'] = round(100.0 * completed / expected_total, 1) if expected_total else 0.0
            state['updated_at'] = _now_iso()
            write_heartbeat()
            stop_scanning.wait(HEARTBEAT_INTERVAL_SECONDS)

    scanner = threading.Thread(target=scan_loop, daemon=True)
    write_heartbeat()
    scanner.start()

    try:
        rpe.run_paper_experiments(
            tickers=args.tickers,
            seeds=seeds,
            total_timesteps=args.timesteps,
            use_synthetic=args.synthetic,
            output_dir=str(output_dir),
            n_eval_episodes=args.n_eval_episodes,
            include_shield_ablation=include_shield_ablation,
            include_inactivity_ablation=include_inactivity_ablation,
            config_overrides=None if args.no_canonical_config else rpe.CANONICAL_FOLD2_CONFIG_OVERRIDES,
            resume=True,
        )
        state['status'] = 'done'

    except Exception as e:
        state['status'] = 'failed'
        state['error'] = str(e)
        state['traceback'] = traceback.format_exc()
        print(f"FULL GRID FAILED: {e}")
        traceback.print_exc()
        raise

    finally:
        state['finished_at'] = _now_iso()
        state['completed_combos'] = _count_completed(output_dir)
        state['pct_complete'] = (
            round(100.0 * state['completed_combos'] / expected_total, 1) if expected_total else 0.0
        )
        stop_scanning.set()
        write_heartbeat()
        log_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resumable full paper grid driver (see launch_full_grid.py)")
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--seeds", type=int, nargs="+", default=rpe.DEFAULT_SEEDS)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no-shield-ablation", action="store_true")
    parser.add_argument("--no-inactivity-ablation", action="store_true")
    parser.add_argument("--no-canonical-config", action="store_true",
                         help="Do not apply the frozen canonical execution-model config (applied by default).")
    main(parser.parse_args())
