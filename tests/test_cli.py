"""Contracts of the command line (peresearch.cli): every command's output and exit status, terminal-safe output,
asking on the terminal before reading outside the declared folders, the heading, and setup."""

import os
import stat
import sys
from types import SimpleNamespace as NS

import pytest
from hypothesis import given
from hypothesis import strategies as st

from peresearch import cli, guard
from peresearch.agent import Answer, Check
from peresearch.zetokrag.index import Index
from tests.conftest import FakeEmbedder
from tests.helpers import notes, user_home


@pytest.fixture
def fake_index(home, monkeypatch):
    """Every command's Index is one with the fake embedder, in the test's home."""
    from peresearch.zetokrag import index as index_module
    real = index_module.Index
    monkeypatch.setattr(index_module, "Index", lambda home=None, embedder=None: real(home, embedder=FakeEmbedder()))


def run(argv, capsys):
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_folders_are_added_listed_indexed_and_removed(home, notes, fake_index, capsys, tmp_path):
    assert run(["folders"], capsys)[1] == "(no folders declared; use `peresearch add`)\n"
    with pytest.raises(SystemExit, match="no folders declared; use `peresearch add`"):
        cli.main(["index"])
    (tmp_path / "other").mkdir()
    code, out, _ = run(["add", str(notes), str(tmp_path / "other")], capsys)
    both = sorted([notes.resolve(), (tmp_path / "other").resolve()], key=str)
    assert code == 0 and out == f"folders: {both[0]}, {both[1]}\nrun `peresearch index` (or open `peresearch chat`)\n"
    assert run(["folders"], capsys)[1] == f"{both[0]}\n{both[1]}\n"
    run(["remove", str(tmp_path / "other")], capsys)
    with pytest.raises(SystemExit, match="not a folder"):
        cli.main(["add", str(tmp_path / "nope")])
    code, out, _ = run(["index"], capsys)
    assert code == 0 and out.startswith("3 added · 0 changed · 0 unchanged · 0 removed · ")
    assert run(["folders"], capsys)[1] == f"{notes.resolve()}\n"
    code, out, _ = run(["remove", str(notes)], capsys)
    assert out == "folders: (none)\nrun `peresearch index` to drop their chunks\n"


def test_find_prints_the_verdict_and_each_excerpt_where_it_is(home, monkeypatch, capsys):
    from peresearch.zetokrag import search
    hits = [NS(where="a.md:L1-2", score=0.71, section="S", file_changed=False, text="line one\nline two"),
            NS(where="b.md:L9", score=0.3, section="", file_changed=True, text="kept \x1b[31mred\x1b[0m")]
    asked = []
    monkeypatch.setattr(search, "Searcher", lambda index: NS(find=lambda q, k: (asked.append((index, q, k)), (hits[:k], "partial", 2))[1]))
    monkeypatch.setattr("peresearch.zetokrag.index.Index", lambda: "the index")
    run(["find", "q0"], capsys)
    assert asked == [("the index", "q0", 5)]
    code, out, _ = run(["find", "query", "-k", "2"], capsys)
    assert asked[-1] == ("the index", "query", 2)
    assert code == 0 and out == ("Your files touch on this.\n\na.md:L1-2  [0.71]  §S\n  line one\n  line two\n\n"
                                 "b.md:L9  [0.30]  (file changed since indexing; excerpt still present)\n  kept red\n"
                                 "\n2 excerpts dropped: their files changed; run `peresearch index`\n")
    for verdict, line in (("enough", "Your files cover this."), ("none", "Nothing in your files seems to answer this.")):
        monkeypatch.setattr(search, "Searcher", lambda index, v=verdict: NS(find=lambda q, k: ([], v, 0)))
        assert run(["find", "q"], capsys)[1] == line + "\n"


def test_ask_prints_events_on_stderr_and_the_answer_on_stdout(home, monkeypatch, capsys):
    made = {}

    def make(project, on_event):
        made["project"] = project
        on_event("tool", "grep(\x1b[2Jx)")
        on_event("answer", "")
        return NS(toolbox=NS(ask="unset"), ask=lambda q: Answer(f"answer to {q}", {}, Check(), 1))
    monkeypatch.setattr(cli, "make_agent", make)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    run(["ask", "why?"], capsys)
    assert made["project"] == "default"
    code, out, err = run(["ask", "why?", "--project", "p1"], capsys)
    assert code == 0 and made["project"] == "p1" and out.startswith("answer to why?")
    assert err == "· tool: grep(x)\n· answer\n"                                 # no terminal control sequence


@pytest.mark.parametrize("tty", [True, False])
def test_ask_reads_outside_the_declared_folders_only_with_someone_to_ask(home, monkeypatch, tty):
    agent = NS(toolbox=NS(ask="unset"), ask=lambda q: Answer("ok", {}, Check(), 1))
    monkeypatch.setattr(cli, "make_agent", lambda project, on_event: agent)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty)
    cli.cmd_ask(NS(project="default", question="q"))
    assert agent.toolbox.ask is (cli.ask_in_terminal if tty else None)


def test_ask_explains_a_failure_without_a_traceback(home, monkeypatch, capsys):
    def broken(**kw):
        raise type("APIConnectionError", (Exception,), {})("Connection error.")
    monkeypatch.setattr(cli, "make_agent", lambda project, on_event: NS(toolbox=NS(ask=None), ask=lambda q: broken()))
    code, out, err = run(["ask", "hello"], capsys)
    assert code == 1 and out == "" and "does not answer" in err and "Traceback" not in err
    assert "**" not in err and "`" not in err and "<sub>" not in err and "/retry" not in err and not err.startswith(">")


@given(st.sampled_from(["o", "O", " once ", "s", "Session", "", "n", "no", "x", "yes"]))
def test_the_terminal_answer_maps_to_once_session_or_no(reply):
    import builtins
    asked = []
    real = builtins.input
    builtins.input = lambda prompt: (asked.append(prompt), reply)[1]
    try:
        got = cli.ask_in_terminal("read", guard.Path("/home/u/Downloads/pa\x1b[2Jper.md"))
    finally:
        builtins.input = real
    want = {"o": "once", "once": "once", "s": "session", "session": "session"}.get(reply.strip().lower(), "no")
    assert got == want and "\x1b" not in asked[0]
    assert asked[0] == ("peresearch wants to read /home/u/Downloads/paper.md, outside your declared folders. "
                        "Allow [o]nce, [s]ession (/home/u/Downloads), or [N]o? ")


def test_a_folder_is_offered_whole(user_home, monkeypatch):
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: (prompts.append(prompt), "s")[1])
    assert cli.ask_in_terminal("tree", user_home / "Downloads") == "session"
    assert f"[s]ession ({user_home / 'Downloads'})" in prompts[0]


@pytest.mark.parametrize("url, keys, model, web", [
    ("", {}, "no model endpoint set", "off (no search key)"),
    ("http://localhost:8000/v1", {"TAVILY_API_KEY": "t"}, "localhost:8000", "Tavily"),
    ("https://ws--x.modal.run/v1", {"TAVILY_API_KEY": "t", "EXA_API_KEY": "e"}, "ws--x.modal.run", "Tavily → Exa"),
    ("http://h/v1", {"EXA_API_KEY": "e"}, "h", "Exa")])
def test_the_heading_names_the_project_folders_model_cost_and_web(home, monkeypatch, tmp_path, url, keys, model, web):
    for k in ("PERESEARCH_LLM_URL", "TAVILY_API_KEY", "EXA_API_KEY", "PERESEARCH_GPU_USD_PER_HOUR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PERESEARCH_LLM_URL", url)
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "projects" / "a").mkdir(parents=True)
    (tmp_path / "class").mkdir()
    guard.set_roots([tmp_path / "projects", tmp_path / "projects" / "a", tmp_path / "class"])
    cost = " · today ≤ $0.00 of $1.00" if "modal." in url else ""
    assert cli.heading("p") == f"peresearch · project p · 2 folders · model {model}{cost} · web {web}"


def test_the_heading_shows_todays_model_cost_for_a_paid_endpoint(home, monkeypatch):
    import time as _t
    from peresearch.llm import Budget

    monkeypatch.setenv("PERESEARCH_LLM_URL", "https://ws--peresearch-llm-server.modal.run/v1")
    guard.set_roots([])
    Budget.from_settings("https://ws--peresearch-llm-server.modal.run/v1").record(_t.time() - 60, _t.time() - 30)
    assert "today ≤ $0.18 of $1.00" in cli.heading("default")             # 30 s + 300 s idle at $1.95/h


def test_setup_declares_folders_indexes_and_saves_the_keys_privately(home, notes, fake_index, tmp_path, capsys,
                                                                     monkeypatch):
    modes = []                                    # the file's mode whenever keys are about to be written into it
    real_fdopen = cli.os.fdopen
    monkeypatch.setattr(cli.os, "fdopen", lambda fd, *a, **kw: (modes.append(stat.S_IMODE(os.fstat(fd).st_mode)),
                                                                real_fdopen(fd, *a, **kw))[1])
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.env").write_text("EXA_API_KEY=old-exa\nOTHER=kept\n")
    (home / "settings.env").chmod(0o644)
    answers = iter([str(tmp_path / "nope"), str(notes), "", "http://localhost:8000/v1"])
    secrets, prompts = iter(["", "tvly-" + "x" * 24, ""]), []
    cli.cmd_setup(None, ask=lambda p: (prompts.append(p), next(answers))[1],
                  secret=lambda p: (prompts.append(p), next(secrets))[1])
    out = capsys.readouterr().out
    assert "not a folder" in out and f"added {notes.resolve()}" in out and "3 added" in out
    env = (home / "settings.env").read_text().splitlines()
    assert sorted(env) == sorted(["EXA_API_KEY=old-exa", "OTHER=kept", "PERESEARCH_LLM_URL=http://localhost:8000/v1",
                                  "TAVILY_API_KEY=tvly-" + "x" * 24])
    assert stat.S_IMODE((home / "settings.env").stat().st_mode) == 0o600
    assert prompts == ["Add a folder (empty to go on): "] * 3 + [
        "model endpoint (…/v1), e.g. http://localhost:8000/v1 or the URL `modal deploy` prints []: ",
        "model key (a Modal proxy token wk-….ws-…; empty for a local server) []: ",
        "Tavily key (free; empty to skip) []: ", "Exa key (free credits; empty to skip) [(set)]: "]
    assert out.startswith("Folders peresearch may read: none yet\n") and out.endswith("Saved. Start with `peresearch chat`.\n")
    assert guard.roots() == [notes.resolve()]
    shown = []
    cli.cmd_setup(None, ask=lambda p: shown.append(p) or "", secret=lambda p: shown.append(p) or "")
    assert shown[1].endswith("[http://localhost:8000/v1]: ") and shown[3].endswith("[(set)]: ")   # a URL is shown, a key not
    assert Index(home, embedder=FakeEmbedder()).rows()
    (home / "settings.env").unlink()                                      # a first setup: the file is new
    cli.cmd_setup(None, ask=lambda p: "", secret=lambda p: "")
    # never readable by others while the keys go in: the file found at 0o644, the one kept, a new one
    assert modes == [0o600] * 3


def test_setup_without_folders_indexes_nothing(tmp_path, monkeypatch, capsys):
    home = tmp_path / "not" / "made"
    monkeypatch.setenv("PERESEARCH_HOME", str(home))
    monkeypatch.setattr(cli, "Workspace", lambda: NS(folders=lambda: [], update=lambda: pytest.fail("indexed")))
    cli.cmd_setup(None, ask=lambda p: "", secret=lambda p: "")
    assert "Indexing" not in capsys.readouterr().out and (home / "settings.env").read_text() == ""


def test_the_agent_and_the_chat_are_built_from_their_real_parts(home, monkeypatch):
    from peresearch import agent as agent_module, llm as llm_module, web as web_module
    made = {}
    monkeypatch.setattr(llm_module, "LLM", lambda on_wait: made.update(on_wait=on_wait, llm=NS()) or made["llm"])
    monkeypatch.setattr(web_module, "Web", lambda: "web")
    monkeypatch.setattr("peresearch.zetokrag.search.Searcher", lambda index: ("searcher", index))
    events = []
    on_event = lambda k, d: events.append((k, d))
    a = cli.make_agent("proj", on_event, index="idx")
    assert isinstance(a, agent_module.Agent) and a.project == "proj" and a.llm is made["llm"] and a.on_event is on_event
    monkeypatch.setattr("peresearch.zetokrag.index.Index", lambda: "own index")
    quiet = cli.make_agent()
    assert quiet.toolbox.searcher == ("searcher", "own index") and quiet.project == "default"
    made["on_wait"]("ignored")                                    # no listener: nothing breaks
    assert a.toolbox.searcher == ("searcher", "idx") and a.toolbox.web == "web"
    a2 = cli.make_agent("proj", on_event, index="idx")
    made["on_wait"]("waking up")
    assert events == [("wait", "waking up")] and a2.on_event is on_event
    shown = {}
    monkeypatch.setattr("peresearch.tui.Chat", lambda make, head, workspace, project: NS(
        run=lambda: shown.update(make=make, head=head(), project=project, ws=workspace)))
    monkeypatch.setattr(cli, "heading", lambda project: f"heading {project}")
    cli.main(["chat", "--project", "p2"])
    assert shown["head"] == "heading p2" and shown["project"] == "p2"
    monkeypatch.setattr(cli, "make_agent", lambda project, on_event, index: (project, on_event, index))
    assert shown["make"](on_event) == ("p2", on_event, shown["ws"].index)
    cli.main(["chat"])
    assert shown["project"] == "default"


def test_an_unknown_command_or_missing_argument_is_a_usage_error(capsys):
    for argv in ([], ["fly"], ["find"], ["add"]):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2 and capsys.readouterr().err.startswith("usage: peresearch ")
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "peresearch setup" in capsys.readouterr().out                 # the commands are described
