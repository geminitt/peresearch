"""E2E failures and owned-process cleanup must survive summary reporting."""
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_shell_entry_point_is_available_without_ignored_run_files():
    assert (ROOT / "scripts/e2e_cycle.sh").is_file()


@pytest.fixture
def cycle():
    spec = importlib.util.spec_from_file_location("e2e_cycle", ROOT / "scripts/e2e_cycle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrapper_tree(tmp_path):
    (tmp_path / "runs/e2e").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / ".pixi/envs/default/bin").mkdir(parents=True)
    (tmp_path / ".pixi/envs/default/bin/python").symlink_to(sys.executable)
    shutil.copyfile(ROOT / "scripts/e2e_cycle.sh", tmp_path / "scripts/e2e_cycle.sh")
    if (ROOT / "scripts/e2e_cycle.py").exists():
        shutil.copyfile(ROOT / "scripts/e2e_cycle.py", tmp_path / "scripts/e2e_cycle.py")
    # Both the legacy shell and repaired Python readiness probe are controlled; no GPU starts.
    (tmp_path / "bash-fixture.sh").write_text("curl() { return 0; }\n")
    (tmp_path / "sitecustomize.py").write_text(
        "import urllib.request\nfrom io import BytesIO\n"
        "urllib.request.urlopen = lambda *a, **k: BytesIO(b'{}')\n")
    env = {**os.environ, "BASH_ENV": str(tmp_path / "bash-fixture.sh"), "PYTHONPATH": str(tmp_path)}
    return env


@pytest.mark.parametrize("code,summary", [(7, True), (0, False), (0, True)])
def test_actual_shell_wrapper_preserves_child_status_with_or_without_summary(tmp_path, code, summary):
    env = wrapper_tree(tmp_path)
    script = tmp_path / "child script.py"
    script.write_text(("print('tool audit: summary')\n" if summary else "print('unmatched output')\n")
                      + f"raise SystemExit({code})\n")
    result = subprocess.run(["bash", str(tmp_path / "scripts/e2e_cycle.sh"), "status-regression", str(script)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == code, result.stdout + result.stderr
    assert f"run exit {code}" in result.stdout


def test_existing_server_is_never_started_or_stopped(cycle, tmp_path):
    result = cycle.run_cycle(tmp_path, "existing-server", ["-c", "print('unmatched output')"],
        python=sys.executable, ready=lambda: True, server_command=["no-such-server"], stop_seconds=0.1)
    assert result == 0


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def wait_dead(pid):
    deadline = time.monotonic() + 3
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not alive(pid), f"owned process {pid} remained running"


def test_startup_failure_stops_before_child(cycle, tmp_path):
    marker = tmp_path / "child-ran"
    result = cycle.run_cycle(tmp_path, "startup-failure", ["-c", f"open({str(marker)!r}, 'w').close()"],
        python=sys.executable, ready=lambda: False, server_command=[sys.executable, "-c", "raise SystemExit(9)"],
        startup_seconds=1, poll_seconds=0.01, stop_seconds=0.1)
    assert result == 9 and not marker.exists()


def test_startup_timeout_cleans_owned_server_without_killing_unrelated_process(cycle, tmp_path):
    pidfile = tmp_path / "owned.pid"
    command = [sys.executable, "-c", f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(30)"]
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = cycle.run_cycle(tmp_path, "startup-timeout", ["-c", "raise SystemExit(0)"], python=sys.executable,
            ready=lambda: False, server_command=command, startup_seconds=0.2, poll_seconds=0.01, stop_seconds=0.1)
        assert result == 124
        wait_dead(int(pidfile.read_text()))
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_run_deadline_kills_child_and_owned_server(cycle, tmp_path):
    server_pid, child_pid = tmp_path / "server.pid", tmp_path / "child.pid"
    def sleep_command(path):
        return [sys.executable, "-c", f"import os,time; open({str(path)!r},'w').write(str(os.getpid())); time.sleep(30)"]
    result = cycle.run_cycle(tmp_path, "run-timeout", sleep_command(child_pid)[1:], python=sys.executable,
        ready=lambda: server_pid.exists(), server_command=sleep_command(server_pid),
        startup_seconds=1, run_seconds=0.2, poll_seconds=0.01, stop_seconds=0.1)
    assert result == 124
    wait_dead(int(server_pid.read_text()))
    wait_dead(int(child_pid.read_text()))


def test_sigterm_preserves_interruption_status_and_stops_child(tmp_path):
    env = wrapper_tree(tmp_path)
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "waiting.py"
    script.write_text(f"import os,time\nopen({str(pidfile)!r},'w').write(str(os.getpid()))\ntime.sleep(30)\n")
    proc = subprocess.Popen(["bash", str(tmp_path / "scripts/e2e_cycle.sh"), "interruption", str(script)],
                            cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pidfile.exists(), proc.communicate(timeout=5)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=10)
        assert proc.returncode == 128 + signal.SIGTERM, (out, err)
        wait_dead(int(pidfile.read_text()))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if pidfile.exists() and alive(int(pidfile.read_text())):
            os.kill(int(pidfile.read_text()), signal.SIGTERM)
            wait_dead(int(pidfile.read_text()))
