"""The unattended gate must stop on failure, enforce deadlines and resume only the same inputs."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checks():
    spec = importlib.util.spec_from_file_location("checks", ROOT / "scripts/check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gate_stops_before_later_commands_after_failure(checks, tmp_path):
    later = tmp_path / "later"
    commands = [("fails", [sys.executable, "-c", "raise SystemExit(7)"]),
                ("later", [sys.executable, "-c", f"from pathlib import Path; Path({str(later)!r}).touch()"])]
    assert checks.run_steps(commands, tmp_path, "same-source", timeout=5) == 7
    assert not later.exists()
    checkpoint = [json.loads(line) for line in (tmp_path / "checkpoint.jsonl").read_text().splitlines()]
    assert checkpoint[-1]["exit_code"] == 7


def test_gate_deadline_terminates_the_command(checks, tmp_path):
    assert checks.run_steps([("hang", [sys.executable, "-c", "import time; time.sleep(30)"])],
                            tmp_path, "same-source", timeout=0.1) == 124
    assert "deadline" in (tmp_path / "hang.log").read_text()


def test_gate_resumes_completed_units_but_changed_inputs_start_again(checks, tmp_path):
    counter, fail = tmp_path / "count", tmp_path / "fail"
    fail.touch()
    commands = [("first", [sys.executable, "-c", f"from pathlib import Path; p=Path({str(counter)!r}); "
                          "p.write_text(p.read_text()+'x' if p.exists() else 'x')"]),
                ("second", [sys.executable, "-c", f"from pathlib import Path; raise SystemExit(int(Path({str(fail)!r}).exists()))"])]
    assert checks.run_steps(commands, tmp_path, "same-source", timeout=5) == 1
    # Simulate an interrupted final append; successful units remain readable.
    with (tmp_path / "checkpoint.jsonl").open("ab") as f:
        f.write(b'{"torn":')
    fail.unlink()
    assert checks.run_steps(commands, tmp_path, "same-source", timeout=5) == 0
    assert counter.read_text() == "x"
    assert checks.run_steps(commands, tmp_path, "changed-source", timeout=5) == 0
    assert counter.read_text() == "xx"


def test_each_test_has_a_watchdog_that_actually_exits(tmp_path):
    (tmp_path / "conftest.py").write_text((ROOT / "tests/conftest.py").read_text())
    (tmp_path / "test_hang.py").write_text("def test_hang():\n    import time\n    time.sleep(30)\n")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", str(tmp_path), "--test-timeout=0.2"],
                            capture_output=True, text=True, timeout=15, cwd=tmp_path)
    assert result.returncode == 124 and "TEST TIMEOUT" in result.stderr
