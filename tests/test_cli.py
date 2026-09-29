"""The command line: setup, and asking on the terminal before reading outside the declared folders."""

from peresearch import guard
from peresearch.zetokrag.index import Index
from tests.conftest import FakeEmbedder
from tests.helpers import notes, user_home


def test_setup_declares_folders_indexes_and_saves_the_keys_privately(home, notes, monkeypatch):
    import stat

    from peresearch import cli
    from peresearch.workspace import Workspace
    monkeypatch.setattr(cli, "Workspace", lambda: Workspace(Index(home, embedder=FakeEmbedder())))
    answers = iter([str(notes), "", "http://localhost:8000/v1"])
    secrets = iter(["", "tvly-" + "x" * 24, ""])
    cli.cmd_setup(None, ask=lambda _: next(answers), secret=lambda _: next(secrets))
    env = (home / "settings.env").read_text()
    assert "PERESEARCH_LLM_URL=http://localhost:8000/v1" in env and "TAVILY_API_KEY=tvly-" in env
    assert stat.S_IMODE((home / "settings.env").stat().st_mode) == 0o600
    assert guard.roots() == [notes.resolve()] and Index(home, embedder=FakeEmbedder()).rows()


def test_the_command_line_asks_too(home, notes, user_home, monkeypatch):
    from peresearch.cli import ask_in_terminal

    replies = iter(["s", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))
    assert ask_in_terminal("read", user_home / "Downloads" / "paper.md") == "session"
    assert ask_in_terminal("read", user_home / "Downloads" / "paper.md") == "no"      # Enter means no


def test_the_heading_counts_folders_inside_another_declared_one_once(home, tmp_path):
    from peresearch import guard
    from peresearch.cli import heading

    (tmp_path / "projects" / "a").mkdir(parents=True)
    (tmp_path / "class").mkdir()
    guard.set_roots([tmp_path / "projects", tmp_path / "projects" / "a", tmp_path / "class"])
    assert "· 2 folders ·" in heading("default")


def test_ask_explains_a_model_that_does_not_answer(home, monkeypatch, capsys):
    from types import SimpleNamespace

    from peresearch import cli

    class APIConnectionError(Exception):
        pass

    def broken(**kw):
        raise APIConnectionError("Connection error.")
    monkeypatch.setattr(cli, "make_agent", lambda project, on_event: SimpleNamespace(
        toolbox=SimpleNamespace(ask=None), ask=lambda q: broken()))
    code = cli.cmd_ask(SimpleNamespace(project="default", question="hello"))
    err = capsys.readouterr().err
    assert code == 1 and "does not answer" in err and "Traceback" not in err


def test_the_heading_shows_todays_model_cost_for_a_paid_endpoint(home, monkeypatch):
    import time as _t

    from peresearch import guard
    from peresearch.cli import heading
    from peresearch.llm import Budget

    monkeypatch.setenv("PERESEARCH_LLM_URL", "https://ws--peresearch-llm-server.modal.run/v1")
    guard.set_roots([])
    b = Budget.from_settings("https://ws--peresearch-llm-server.modal.run/v1")
    b.record(_t.time() - 60, _t.time() - 30)                      # 30 s + 300 s idle at $1.95/h ≈ $0.18
    assert "today ≤ $0.18 of $1.00" in heading("default")
    monkeypatch.setenv("PERESEARCH_LLM_URL", "http://localhost:8000/v1")
    assert "today" not in heading("default")                      # a local server costs nothing
