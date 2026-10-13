"""Run a local E2E, preserving its status and cleaning up only processes we start.

The shell compatibility entry point is runs/e2e/cycle.sh. Python owns the watchdogs
and process groups so reporting and server cleanup cannot erase the child status.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = re.compile(r"^(tools offered|tool |sources|agent|calls under|folder check|/|local|@path|quit|web searches)")


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def ready():
    try:
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2):
            return True
    except OSError:
        return False


def stop_group(proc, seconds):
    """The group belongs to this invocation, even if its launcher already exited."""
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        pass
    # A launcher may have exited while a descendant stayed alive. Kill only its
    # known process group; never discover or stop an unrelated port owner.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=max(seconds, 1))


def run_cycle(root, label, script, *, python=None, server_command=None, ready=ready,
              startup_seconds=300, run_seconds=2700, stop_seconds=5, poll_seconds=0.2):
    root = Path(root)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError("label must contain only letters, numbers, dots, underscores or hyphens")
    logs = root / "runs/e2e"
    logs.mkdir(parents=True, exist_ok=True)
    python = python or str(root / ".pixi/envs/default/bin/python")
    server_command = server_command or ["pixi", "run", "-e", "serve", "python", "deploy/local_vllm.py"]
    server = child = None
    server_log = None
    try:
        if not ready():
            server_log = (logs / f"e2e-{label}-server.log").open("w")
            server = subprocess.Popen(server_command, cwd=root, stdout=server_log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            deadline = time.monotonic() + startup_seconds
            while not ready():
                code = server.poll()
                if code is not None:
                    print(f"server exited before readiness: {code}", flush=True)
                    return code if code > 0 else 1
                if time.monotonic() >= deadline:
                    print(f"server not ready after {startup_seconds:g} s", flush=True)
                    return 124
                time.sleep(poll_seconds)
            print("server ready", flush=True)
        run_log = logs / f"e2e-{label}-run.log"
        with run_log.open("w") as output:
            child = subprocess.Popen([python, *script], cwd=root, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = child.wait(timeout=run_seconds)
                code = code if code >= 0 else 128 - code
            except subprocess.TimeoutExpired:
                output.write(f"\nE2E deadline exceeded ({run_seconds:g} seconds).\n")
                code = 124
                stop_group(child, stop_seconds)
        print(f"run exit {code}", flush=True)
        with run_log.open(errors="replace") as output:
            for line in output:
                if SUMMARY.match(line):
                    print(line.rstrip()[:330], flush=True)
        return code
    except Interrupted as error:
        return 128 + error.signum
    except OSError as error:
        print(f"E2E could not start: {error}", file=sys.stderr, flush=True)
        return 127
    finally:
        stop_group(child, stop_seconds)
        stop_group(server, stop_seconds)
        if server_log is not None:
            server_log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("script", nargs="*")
    args = parser.parse_args(argv)
    script = args.script or ["runs/e2e/e2e-2026-10-05.py", "run"]
    # Preserve the old single-string 'script.py run' form, while also accepting
    # an actual filename containing spaces and properly separated arguments.
    if len(script) == 1 and not Path(script[0]).is_file():
        script = shlex.split(script[0])
    previous = {}
    def interrupted(signum, frame):
        raise Interrupted(signum)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, interrupted)
    try:
        return run_cycle(ROOT, args.label, script)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
