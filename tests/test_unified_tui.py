"""One TUI entry, first-run setup, and deliberate service checks without live requests."""

import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

from textual.widgets import Input, Static

from peresearch import cli, settings, tui
from tests.helpers import settle, texts, tui_run


def test_bare_command_opens_tui_and_preserves_project_selection(home, monkeypatch):
    opened = []
    monkeypatch.setattr(cli, "cmd_chat", lambda args: opened.append(args.project))
    assert cli.main([]) == 0
    assert cli.main(["--project", "research"]) == 0
    assert opened == ["default", "research"]


def test_first_launch_opens_setup_without_constructing_agent(home, monkeypatch):
    monkeypatch.delenv("PERESEARCH_LLM_URL", raising=False)

    async def steps(app, pilot):
        await pilot.pause()
        assert app.screen.id == "model-window"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen.id != "model-window"
        assert app.is_running

    tui_run(lambda event: (_ for _ in ()).throw(AssertionError("unexpected agent construction")),
            steps, setup_on_start=True)


def test_installed_console_opens_first_run_setup_and_quits_in_a_terminal(home):
    master, slave = pty.openpty()
    process = None
    output = bytearray()
    try:
        process = subprocess.Popen([str(Path(sys.executable).with_name("peresearch"))], stdin=slave,
                                   stdout=slave, stderr=slave, env={**os.environ, "TERM": "xterm-256color"},
                                   start_new_session=True)
        os.close(slave)
        slave = None
        deadline = time.monotonic() + 20
        while not (b"Models and APIs" in output and b"connect a model" in output) and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                output.extend(os.read(master, 65536))
            assert process.poll() is None, output.decode(errors="replace")
        assert b"Models and APIs" in output and b"connect a model" in output
        assert not (home / "settings.env").exists()
        os.write(master, b"\x1b")
        time.sleep(0.1)
        os.write(master, b"\x11")
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.05)[0]:
                try:
                    output.extend(os.read(master, 65536))
                except OSError:
                    break
        assert process.wait(timeout=1) == 0
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if slave is not None:
            os.close(slave)
        os.close(master)


def test_saving_service_changes_does_not_start_connection_requests(home):
    calls = []

    async def steps(app, pilot):
        app.setup_done({"PERESEARCH_LLM_URL": "https://model.test/v1",
                        "PERESEARCH_EMBED_URL": "https://embed.test/v1"})
        await settle(app, pilot, lambda: not any(t.is_alive() for t in app.background))
        assert calls == []
        assert settings.saved()["PERESEARCH_EMBED_URL"] == "https://embed.test/v1"

    tui_run(lambda event: None, steps,
            model_check=lambda: calls.append("model") or (True, []),
            endpoint_check=lambda: calls.append("search") or [])


def test_connection_check_requires_confirmation_and_does_not_overlap_question(home):
    checks = []

    async def steps(app, pilot):
        app.action_command("/check")
        await pilot.pause()
        assert checks == []
        assert "charged" in " ".join(str(w.render()) for w in app.screen.query(Static))
        await pilot.press("escape")
        await pilot.pause()
        assert checks == []
        app.busy = True
        app.action_command("/check")
        await pilot.pause()
        assert not isinstance(app.screen, tui.ConnectionCheck)
        app.busy = False
        app.action_command("/check")
        await pilot.pause()
        assert "r run checks" in " ".join(str(w.render()) for w in app.screen.query(Static))
        await pilot.press("r")
        await settle(app, pilot, lambda: checks == ["model", "search"] and not app._checking_connections)
        assert checks == ["model", "search"]

    tui_run(lambda event: None, steps,
            model_check=lambda: checks.append("model") or (True, ["synthetic model check"]),
            endpoint_check=lambda: checks.append("search") or ["synthetic search check"])
