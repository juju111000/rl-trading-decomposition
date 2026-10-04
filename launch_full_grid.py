#!/usr/bin/env python3
"""
Detached launcher for run_full_grid.py -- the Windows equivalent of
`nohup python run_full_grid.py & `. Starts the full paper grid as a fully
independent process (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, no
console window via pythonw if available) that keeps running after this
launcher process -- and the shell/session that invoked it -- exits, then
writes the child's PID to <output_dir>/full_grid.pid and returns
immediately without waiting on it.

Unless --output-dir is passed explicitly, this launcher stamps ONE
timestamp under --output-root and passes that resolved, concrete directory
to run_full_grid.py -- so if you need to relaunch after a crash or an
intentional stop, pass that SAME resolved --output-dir back in (printed
below and recorded in the directory's own name) rather than re-running this
script with --output-root again, which would stamp a NEW timestamp and
start over from scratch instead of resuming.

run_full_grid.py handles its own stdout/stderr redirection to
<output_dir>/run.log and its own heartbeat.json, so this launcher does no
redirection itself -- it only has to detach the process.

Usage:
    python launch_full_grid.py --timesteps 300000
    python launch_full_grid.py --output-dir ./results/full_grid/20260802_120000   # resume
    (any extra args are forwarded verbatim to run_full_grid.py)
"""
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

DEFAULT_OUTPUT_ROOT = "./results/full_grid"


def main():
    forwarded_args = sys.argv[1:]

    if "--output-dir" in forwarded_args:
        output_dir = forwarded_args[forwarded_args.index("--output-dir") + 1]
    else:
        output_root = DEFAULT_OUTPUT_ROOT
        if "--output-root" in forwarded_args:
            idx = forwarded_args.index("--output-root")
            output_root = forwarded_args[idx + 1]
            forwarded_args = forwarded_args[:idx] + forwarded_args[idx + 2:]
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        output_dir = str(Path(output_root) / stamp)
        forwarded_args = forwarded_args + ["--output-dir", output_dir]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).parent / "run_full_grid.py"

    python_dir = Path(sys.executable).parent
    pythonw = python_dir / "pythonw.exe"
    interpreter = str(pythonw) if pythonw.exists() else sys.executable

    proc = subprocess.Popen(
        [interpreter, str(script_path), *forwarded_args],
        cwd=str(Path(__file__).parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )

    pid_path = output_path / "full_grid.pid"
    pid_path.write_text(str(proc.pid))

    print(f"Launched detached full grid: PID {proc.pid}")
    print(f"  Output dir: {output_path}")
    print(f"  Log:        {output_path / 'run.log'}")
    print(f"  Heartbeat:  {output_path / 'heartbeat.json'}")
    print(f"  PID file:   {pid_path}")
    print(f"\nTo check progress:")
    print(f"  python check_full_grid_progress.py --dir {output_path}")
    print(f"\nTo resume after a stop/crash, relaunch with the SAME directory:")
    print(f"  python launch_full_grid.py --output-dir {output_path} [other args]")


if __name__ == "__main__":
    main()
