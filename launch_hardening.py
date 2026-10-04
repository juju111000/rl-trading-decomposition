#!/usr/bin/env python3
"""
Detached launcher for run_hardening.py -- same pattern as
launch_full_grid.py. Starts the hardening seed-expansion workstream as a
fully independent process (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, no
console window via pythonw if available) that keeps running after this
launcher process -- and the shell/session that invoked it -- exits.

Unlike launch_full_grid.py, there's no per-run timestamp: the hardening
workstream always resumes against the SAME ./results/hardening directory
(there's only ever one hardening run in flight), so relaunching this
script after a stop/crash just picks up where it left off via
run_hardening.py's per-(arm,seed) eval_result.json markers -- no
--output-dir juggling needed.

Usage:
    python launch_hardening.py
"""
import sys
import subprocess
from pathlib import Path

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

OUTPUT_DIR = "./results/hardening"


def main():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).parent / "run_hardening.py"

    python_dir = Path(sys.executable).parent
    pythonw = python_dir / "pythonw.exe"
    interpreter = str(pythonw) if pythonw.exists() else sys.executable

    proc = subprocess.Popen(
        [interpreter, str(script_path), "--output-dir", str(output_path)],
        cwd=str(Path(__file__).parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )

    pid_path = output_path / "hardening.pid"
    pid_path.write_text(str(proc.pid))

    print(f"Launched detached hardening run: PID {proc.pid}")
    print(f"  Output dir: {output_path}")
    print(f"  Log:        {output_path / 'run.log'}")
    print(f"  Heartbeat:  {output_path / 'heartbeat.json'}")
    print(f"  PID file:   {pid_path}")
    print(f"\nTo check progress:")
    print(f"  python check_hardening_progress.py")
    print(f"\nTo resume after a stop/crash, just relaunch:")
    print(f"  python launch_hardening.py")


if __name__ == "__main__":
    main()
