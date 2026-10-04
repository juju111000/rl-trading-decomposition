#!/usr/bin/env python3
"""
Poll run_hardening.py's progress -- same pattern as
check_full_grid_progress.py. Scans results/hardening/ directly for
eval_result.json completion markers (accurate regardless of process
restarts), and per-combo timing is derived from their mtimes.

Usage:
    python check_hardening_progress.py
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = "./results/hardening"
ARMS = ["untrained", "canonical", "oracle60", "oracle75"]


def _fmt_seconds(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _pid_alive(pid: int):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                              capture_output=True, text=True, timeout=5)
        return str(pid) in out.stdout
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Check run_hardening.py progress")
    parser.add_argument("--dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--log-lines", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.dir)
    heartbeat_path = run_dir / "heartbeat.json"

    if not heartbeat_path.exists():
        print(f"No heartbeat.json found at {heartbeat_path} -- has the run started?")
        return

    with open(heartbeat_path) as f:
        state = json.load(f)

    print("=" * 70)
    print(f"  HARDENING PROGRESS -- {run_dir}")
    print("=" * 70)
    print(f"Status (last heartbeat): {state.get('status')}")
    print(f"Current arm: {state.get('current_arm')}")
    print(f"Started:  {state.get('started_at')}")
    print(f"Updated:  {state.get('updated_at')} (heartbeat writes every ~30s)")
    if state.get('finished_at'):
        print(f"Finished: {state.get('finished_at')}")

    pid = state.get('pid')
    if pid is not None and state.get('status') == 'running':
        alive = _pid_alive(pid)
        if alive is None:
            print(f"PID {pid}: liveness check unavailable on this platform")
        else:
            print(f"PID {pid}: {'ALIVE' if alive else 'NOT RUNNING'}")

    eval_results = sorted(run_dir.rglob("eval_result.json"), key=lambda p: p.stat().st_mtime)
    completed = len(eval_results)
    expected = state.get('expected_total_combos', 0)
    pct = round(100.0 * completed / expected, 1) if expected else 0.0
    print(f"\nProgress (live scan): {completed}/{expected} combos ({pct}%)")

    print("\n--- Per-arm breakdown ---")
    for arm in ARMS:
        arm_dir = run_dir / arm
        n = len(list(arm_dir.rglob("eval_result.json"))) if arm_dir.exists() else 0
        print(f"  {arm:10s} {n}/20")

    if state.get('status') == 'failed':
        print("\n--- FAILURE (per last heartbeat) ---")
        print(state.get('error', '(no error message recorded)'))
        print("Relaunch to resume from the last completed (arm, seed):")
        print("  python launch_hardening.py")

    if completed:
        mtimes = [p.stat().st_mtime for p in eval_results]
        deltas = [b - a for a, b in zip(mtimes, mtimes[1:])]
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            print("\n--- Timing (derived from eval_result.json mtimes) ---")
            print(f"  Per-combo elapsed: min={_fmt_seconds(min(deltas))}  "
                  f"mean={_fmt_seconds(mean_delta)}  max={_fmt_seconds(max(deltas))}")
            remaining = expected - completed
            if remaining > 0:
                print(f"  Est. remaining ({remaining} combos left): ~{_fmt_seconds(remaining * mean_delta)}")
        most_recent = datetime.fromtimestamp(mtimes[-1])
        print(f"  Most recent combo finished at: {most_recent.isoformat(sep=' ', timespec='seconds')} "
              f"(local time where the file lives)")
    else:
        print("\n--- Timing ---")
        print("  No combos completed yet.")

    log_path = run_dir / "run.log"
    if log_path.exists():
        print(f"\n--- Tail of {log_path} (last {args.log_lines} lines) ---")
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        stdout_encoding = sys.stdout.encoding or "utf-8"
        for line in lines[-args.log_lines:]:
            safe_line = line.rstrip().encode(stdout_encoding, errors="replace").decode(stdout_encoding)
            print(f"  {safe_line}")

    results_csv = run_dir / "hardening_results.csv"
    if results_csv.exists():
        print(f"\nhardening_results.csv is being written incrementally at {results_csv} "
              f"({completed} rows so far).")


if __name__ == "__main__":
    main()
