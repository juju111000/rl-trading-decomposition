#!/usr/bin/env python3
"""
Poll a run_full_grid.py run's progress. Unlike check_progress.py (which
relies on run_dryrun.py's in-process monkeypatch instrumentation), this
scans the output directory directly for eval_result.json completion
markers -- accurate regardless of process restarts, and works whether the
heartbeat thread's last periodic write is slightly stale or not. Per-combo
timing is derived from eval_result.json mtimes (time between consecutive
completions), not from any live instrumentation.

Usage:
    python check_full_grid_progress.py --dir ./results/full_grid/20260802_120000
"""
import argparse
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _parse_iso(ts):
    return datetime.fromisoformat(ts)


def _fmt_seconds(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _pid_alive(pid: int) -> "bool | None":
    """Best-effort liveness check via `tasklist` (Windows). Returns None if
    the check itself couldn't be run (e.g. non-Windows), not just "unknown
    alive" -- callers should treat None as "couldn't determine", not False."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=5
        )
        return str(pid) in out.stdout
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Check run_full_grid.py progress")
    parser.add_argument("--dir", type=str, required=True, help="Output directory passed to run_full_grid.py")
    parser.add_argument("--log-lines", type=int, default=20, help="Lines of run.log to tail")
    args = parser.parse_args()

    run_dir = Path(args.dir)
    heartbeat_path = run_dir / "heartbeat.json"

    if not heartbeat_path.exists():
        print(f"No heartbeat.json found at {heartbeat_path} -- has the run started?")
        return

    with open(heartbeat_path) as f:
        state = json.load(f)

    print("=" * 70)
    print(f"  FULL GRID PROGRESS -- {run_dir}")
    print("=" * 70)
    print(f"Status (last heartbeat): {state.get('status')}")
    print(f"Started:  {state.get('started_at')}")
    print(f"Updated:  {state.get('updated_at')} (heartbeat writes every ~30s)")
    if state.get('finished_at'):
        print(f"Finished: {state.get('finished_at')}")

    pid = state.get('pid')
    if pid is not None and state.get('status') == 'running':
        # Only meaningful while the last heartbeat claims "running" -- once
        # the process has exited (done/failed), the PID may have been
        # reused by something unrelated, making an "ALIVE" result here
        # misleading rather than informative.
        alive = _pid_alive(pid)
        if alive is None:
            print(f"PID {pid}: liveness check unavailable on this platform")
        else:
            print(f"PID {pid}: {'ALIVE' if alive else 'NOT RUNNING'}")

    print(f"Timesteps per run: {state.get('timesteps_per_run')}")
    print(f"Canonical config applied: {state.get('canonical_config')}")
    print(f"Shield ablation included: {state.get('include_shield_ablation')}  "
          f"Inactivity ablation included: {state.get('include_inactivity_ablation')}")

    # Live filesystem scan -- authoritative, independent of heartbeat staleness.
    eval_results = sorted(run_dir.rglob("eval_result.json"), key=lambda p: p.stat().st_mtime)
    completed = len(eval_results)
    expected = state.get('expected_total_combos', 0)
    pct = round(100.0 * completed / expected, 1) if expected else 0.0

    print(f"\nProgress (live scan): {completed}/{expected} combos ({pct}%)")

    if state.get('status') == 'failed':
        print("\n--- FAILURE (per last heartbeat) ---")
        print(state.get('error', '(no error message recorded)'))
        print("Relaunch with the SAME --output-dir to resume from the last completed combo:")
        print(f"  python launch_full_grid.py --output-dir {run_dir}")

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
        # NOTE: printed as an absolute local timestamp, not "N ago" -- this
        # script may run in a different environment/clock than the detached
        # run_full_grid.py process (e.g. a sandboxed shell vs. the real
        # Windows host), so comparing datetime.now() here against the
        # file's mtime there is not reliable.
        most_recent = datetime.fromtimestamp(mtimes[-1])
        print(f"  Most recent combo finished at: {most_recent.isoformat(sep=' ', timespec='seconds')} (local time where the file lives)")
    else:
        print("\n--- Timing ---")
        print("  No combos completed yet -- no timing data (still on the first arm, "
              "or still loading/aligning data).")

    log_path = run_dir / "run.log"
    if log_path.exists():
        print(f"\n--- Tail of {log_path} (last {args.log_lines} lines) ---")
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # The log may contain characters this console's codepage can't
        # display (e.g. Windows cp1252) -- re-encode defensively so a
        # display-only issue can't crash the progress check itself.
        stdout_encoding = sys.stdout.encoding or "utf-8"
        for line in lines[-args.log_lines:]:
            safe_line = line.rstrip().encode(stdout_encoding, errors="replace").decode(stdout_encoding)
            print(f"  {safe_line}")

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        print(f"\nresults.csv already written ({results_csv}) -- the main grid's seed loop has fully finished "
              f"(ablations may still be running).")


if __name__ == "__main__":
    main()
