#!/usr/bin/env python3
"""
Progress for the corrected-accounting master batch (run_r2c.py): liveness,
heartbeat age, completed combos per phase/arm, failures, measured minutes per
training run, ETA, log tail.  Usage: python check_r2c_progress.py [--log-lines 15]
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
BATCH = Path("./results/r2/corrected/batch")


def pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-lines", type=int, default=15)
    args = ap.parse_args()
    hb_path = BATCH / "heartbeat.json"
    hb = json.loads(hb_path.read_text()) if hb_path.exists() else {}
    import run_r2c as C
    combos = C.combo_list(hb.get('include_p2', False), hb.get('include_p2prime', False))
    q = BATCH / "queued.json"
    if q.exists():
        qq = json.loads(q.read_text())
        print(f"QUEUED follow-on batch: PID {qq['pid']} ({'ALIVE' if pid_alive(qq['pid']) else 'NOT RUNNING'}) "
              f"waiting for PID {qq['waiting_for_pid']}; P2' included: {qq.get('include_p2prime')}")
        if qq.get('include_p2prime'):
            combos = C.combo_list(hb.get('include_p2', False), True)
    pid = int((BATCH / "r2c.pid").read_text()) if (BATCH / "r2c.pid").exists() else hb.get('pid')
    alive = pid_alive(pid) if pid else None
    age = ((datetime.now(timezone.utc) - datetime.fromisoformat(hb['last_update'])).total_seconds()
           if hb.get('last_update') else None)
    print(f"PID {pid}: {'ALIVE' if alive else 'NOT RUNNING' if alive is False else 'unknown'} | status: "
          f"{hb.get('status', 'no heartbeat yet')} | heartbeat age: {f'{age:.0f}s' if age is not None else 'n/a'}"
          f" | P2 oracles included: {hb.get('include_p2', False)}")
    if hb.get('current'):
        print(f"current: {hb['current']['combo']} since {hb['current']['started']}")
    groups = {}
    for c in combos:
        g = groups.setdefault((c[0], c[1]), [0, 0])
        g[1] += 1
        g[0] += C.marker(*c).exists()
    print(f"\ncompleted combos: {sum(g[0] for g in groups.values())} / {len(combos)}")
    for (exp, arm), (d, n) in groups.items():
        print(f"  {exp:4s} {arm:13s} {d:3d} / {n}")
    rt = [json.loads(l) for l in (BATCH / "runtimes.jsonl").read_text().splitlines()] \
        if (BATCH / "runtimes.jsonl").exists() else []
    tr = [(r['seconds'], r['training_runs']) for r in rt if r['training_runs']]
    per = sum(s for s, _ in tr) / sum(n for _, n in tr) if tr else 13.3 * 60
    remaining = sum(C.n_training(c) for c in combos if not C.marker(*c).exists())
    print(f"\ntraining runs remaining: {remaining}; {per / 60:.1f} min/run "
          f"({'measured' if tr else 'assumed'}); ETA ~{remaining * per / 3600:.1f} h (+ eval-only steps)")
    for f in hb.get('failed', []):
        print(f"FAILED: {f['combo']} -- {f['error']} ({f['at']})")
    log = BATCH / "run.log"
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\n--- last {args.log_lines} log lines ---\n" + "\n".join(lines[-args.log_lines:]))


if __name__ == "__main__":
    main()
