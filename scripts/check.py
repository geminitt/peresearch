"""Run the default, ci and clean Docker suites with deadlines and durable, resumable gates.

    pixi run gate
    pixi run gate --passes 2
    pixi run gate --passes 2 --resume runs/checks/<directory printed by the interrupted run>
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_files():
    raw = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT)
    return sorted({n for n in raw.decode().split("\0") if n and (ROOT / n).is_file()})


def source_signature():
    digest = hashlib.sha256()
    for name in project_files():
        digest.update(name.encode() + b"\0")
        with (ROOT / name).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _append(path, record):
    with path.open("ab") as f:
        torn = f.tell() and path.read_bytes()[-1:] != b"\n"
        f.write((b"\n" if torn else b"") + json.dumps(record).encode() + b"\n")
        f.flush()
        os.fsync(f.fileno())


def _stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)


def run_steps(commands, log_dir, identity, timeout=1800, verify=None):
    """Exit on the first failure; reuse only successful units with identical sources and commands."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = log_dir / "checkpoint.jsonl"
    signature = hashlib.sha256(json.dumps([identity, commands], sort_keys=True).encode()).hexdigest()
    done = set()
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec["signature"] == signature and rec["exit_code"] == 0:
                    done.add(rec["step"])
            except (ValueError, KeyError, TypeError):
                pass
    for name, command in commands:
        if verify is not None and not verify():
            print("Sources changed during the gate; start a new gate.", flush=True)
            return 2
        if name in done:
            print(f"{name}: resumed completed unit", flush=True)
            continue
        print(f"{name}: running; log {log_dir / (name + '.log')}", flush=True)
        started, proc = time.monotonic(), None
        with (log_dir / (name + ".log")).open("w") as log:
            try:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.write(f"\nGate deadline exceeded ({timeout} seconds).\n")
                _stop(proc)
                code = 124
            except OSError as e:
                log.write(f"\nCould not execute the check: {e}\n")
                code = 127
            except BaseException:
                if proc is not None:
                    _stop(proc)
                raise
            if verify is not None and not verify():
                log.write("\nSources changed during the check; result not accepted.\n")
                code = 2
            log.flush()
            os.fsync(log.fileno())
        _append(checkpoint, {"signature": signature, "source": identity, "step": name,
                             "exit_code": code, "seconds": round(time.monotonic() - started, 3)})
        print(f"{name}: exit {code}", flush=True)
        if code:
            return code
    return 0


def docker_suite():
    """Fresh install, unprivileged user, offline execution; remove only this invocation's temporary resources."""
    tag = "peresearch-gate-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="peresearch-gate-") as tmp:
        context = Path(tmp)
        with tarfile.open(context / "source.tar", "w") as archive:
            for name in project_files():
                archive.add(ROOT / name, arcname=name)
        (context / "Dockerfile").write_text('''FROM ghcr.io/prefix-dev/pixi:0.79.0
RUN useradd -m tester
WORKDIR /home/tester/peresearch
ADD source.tar .
RUN chown -R tester:tester /home/tester/peresearch
USER tester
RUN pixi install -e ci --locked
ENV LANG=C LC_ALL=C
CMD ["pixi", "run", "-e", "ci", "pytest", "-q", "-p", "no:cacheprovider", "-o", "faulthandler_timeout=60"]
''')
        try:
            build = subprocess.run(["docker", "build", "--no-cache", "-t", tag, str(context)], timeout=900)
            if build.returncode:
                return build.returncode
            return subprocess.run(["docker", "run", "--name", tag, "--rm", "--init", "--network", "none",
                                   "--memory", "4g", "--cpus", "4", tag], timeout=600).returncode
        except subprocess.TimeoutExpired:
            return 124
        finally:
            for command in (["docker", "rm", "-f", tag], ["docker", "image", "rm", tag]):
                try:
                    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=int, default=1, choices=(1, 2))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--docker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.docker:
        return docker_suite()
    identity = source_signature()
    log_dir = args.resume or ROOT / "runs/checks" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    commands = []
    for number in range(1, args.passes + 1):
        for env in ("default", "ci"):
            python = ROOT / ".pixi/envs" / env / "bin/python"
            commands.append((f"pass-{number}-{env}", ["timeout", "-k", "5", "600", str(python), "-m", "pytest",
                                                      "-q", "-p", "no:cacheprovider", "-o", "faulthandler_timeout=60"]))
        commands.append((f"pass-{number}-docker", [sys.executable, str(Path(__file__).resolve()), "--docker"]))
    print(f"Gate source: {identity}\nResume: pixi run gate --passes {args.passes} --resume {log_dir}", flush=True)
    return run_steps(commands, log_dir, identity, verify=lambda: source_signature() == identity)


if __name__ == "__main__":
    raise SystemExit(main())
