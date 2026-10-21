#!/usr/bin/env python3
"""
Launch run_r2c.py as a fully detached background process (survives this
console closing), same pattern as launch_hardening.py. Relaunching after a
stop/crash resumes from the per-combo eval_result.json markers.

Usage:  python launch_r2c.py           Progress:  python check_r2c_progress.py
"""
import subprocess
import sys
from pathlib import Path

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
BATCH_DIR = Path("./results/r2/corrected/batch")


def main():
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent / "run_r2c.py"
    pythonw = Path(sys.executable).parent / "pythonw.exe"
    interpreter = str(pythonw) if pythonw.exists() else sys.executable
    proc = subprocess.Popen(
        [interpreter, str(script)] + sys.argv[1:],
        cwd=str(Path(__file__).parent),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True,
    )
    if "--wait-for-pid" in sys.argv:
        (BATCH_DIR / "r2c_queued.pid").write_text(str(proc.pid))   # takes over r2c.pid once it starts
    else:
        (BATCH_DIR / "r2c.pid").write_text(str(proc.pid))
    print(f"Launched detached CORRECTED master batch: PID {proc.pid}")
    print(f"  Log:       {BATCH_DIR / 'run.log'}")
    print(f"  Heartbeat: {BATCH_DIR / 'heartbeat.json'}")
    print("Progress:  python check_r2c_progress.py")
    print(f"Stop:      taskkill /PID {proc.pid} /F    (resume later with: python launch_r2c.py)")


if __name__ == "__main__":
    main()
