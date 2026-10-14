"""Contracts of the command line (peresearch.cli): every command's output and exit status, terminal-safe output,
asking on the terminal before reading outside the declared folders, a bad setting stopping a command before it
starts, the heading, and setup (on a machine without a GPU: the rerank trade-off and depth)."""

import io
import os
import signal
import stat
import sys
import threading
import time
from types import SimpleNamespace as NS

import pytest
from hypothesis import given
from hypothesis import strategies as st

from peresearch import cli, guard, workspace
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
    import os
    assert not home.exists()
    assert run(["folders"], capsys)[1] == "(no folders declared; use `peresearch add`)\n"
    # peresearch's own folder holds the index (every note's text), sessions, keys: the user's alone, made so or closed
    # again on any command (it was 0755 with its files 0644 on the owner's machine, 2026-10-07)
    assert oct(os.stat(home).st_mode & 0o777) == "0o700"
    os.chmod(home, 0o755)
    run(["folders"], capsys)
    assert oct(os.stat(home).st_mode & 0o777) == "0o700"
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


def test_find_prints_a_page_of_ranked_excerpts_where_they_are(home, monkeypatch, capsys):
    from peresearch.zetokrag import search
    hits = [NS(where="a.md:L1-2", score=0.71, section="S", file_changed=False, text="line one\nline two",
               copies=["a.template.md:L1-2", "a.copy.md:L1-2"]),
            NS(where="b.md:L9", score=0.03, section="", file_changed=True, text="kept \x1b[31mred\x1b[0m", copies=[])]
    asked = []
    monkeypatch.setattr(search, "Searcher", lambda index: NS(find=lambda q, k, page: (
        asked.append((index, q, k, page)), (hits[(page - 1) * k:page * k], 2, len(hits)))[1]))
    monkeypatch.setattr("peresearch.zetokrag.index.Index", lambda: "the index")
    run(["find", "q0"], capsys)
    assert asked == [("the index", "q0", 10, 1)]
    code, out, _ = run(["find", "query", "-k", "1", "--page", "2"], capsys)
    assert asked[-1] == ("the index", "query", 1, 2)
    assert code == 0 and out == ("results 2–2 of 2\n\nb.md:L9  [0.03]  (file changed since indexing; excerpt still "
                                 "present)\n  kept red\n\n2 excerpts dropped: their files changed; run `peresearch index`\n")
    assert run(["find", "query", "-k", "1"], capsys)[1] == ("results 1–1 of 2\n\na.md:L1-2  [0.71]  §S\n  line one\n"
                                                        "  line two\n  (also in: a.template.md:L1-2, a.copy.md:L1-2)\n\n"
                                                        "2 excerpts dropped: their files changed; run "
                                                        "`peresearch index`\nmore: --page 2\n")
    assert run(["find", "query", "--page", "5"], capsys)[1].startswith("no results on page 5 of 2\n")
    monkeypatch.setattr(search, "Searcher", lambda index: NS(find=lambda q, k, page: ([], 0, 0)))
    assert run(["find", "q"], capsys)[1] == "Nothing in your files matched.\n"
    tail = [NS(where="c.md:L3", score=None, section="", file_changed=False, text="c", copies=[])]
    monkeypatch.setattr(search, "Searcher", lambda index: NS(find=lambda q, k, page: (tail, 0, 1)))
    assert run(["find", "q"], capsys)[1] == "results 1–1 of 1\n\nc.md:L3  [not reranked]\n  c\n"


@pytest.mark.parametrize("command", [["find", "q"], ["ask", "q"], ["chat"]])
def test_a_bad_rerank_depth_stops_a_command_before_it_starts(home, monkeypatch, command):
    monkeypatch.setenv("PERESEARCH_RERANK_DEPTH", "50")
    monkeypatch.setattr(cli, "make_agent", lambda *a, **kw: pytest.fail("started"))
    monkeypatch.setattr("peresearch.zetokrag.index.Index", lambda: pytest.fail("started"))
    with pytest.raises(SystemExit, match=r"PERESEARCH_RERANK_DEPTH must be a whole number from 0 to 30 \(got '50'\)") as e:
        cli.main(command)
    assert str(e.value).endswith(f"; it is set in the environment or in {home / 'settings.env'}")


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
    assert err == ("Error: The model server does not answer: check that it runs and that PERESEARCH_LLM_URL points to "
                   "it (/check tests it); a server that is starting may need minutes. "
                   "(APIConnectionError)\n")                                             # the TUI's /retry left out


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


@pytest.mark.parametrize("url, model", [("", "not set: /model chooses it"),
                                        ("localhost:8000/v1", "localhost:8000"),           # set without a scheme
                                        ("http://localhost:8000/v1", "localhost:8000"),
                                        ("https://ws--x.modal.run/v1", "ws--x.modal.run")])
@pytest.mark.parametrize("folders", [0, 2, 3, 4, 5])
def test_the_welcome_names_the_model_the_folders_and_where_to_go_next(home, user_home, monkeypatch, url, model,
                                                                      folders):
    """What one needs at a glance, as Claude Code opens: the model, the folders (a few, then how many more), and the
    two commands that lead on. The search services and the costs are details: /status."""
    monkeypatch.setenv("PERESEARCH_LLM_URL", url)
    made = [user_home / n for n in ("projects", "class", "papers", "notes", "books")[:folders]]
    for d in made:
        d.mkdir()
    (user_home / "projects" / "a").mkdir(exist_ok=True) if folders else None
    guard.set_roots(made + ([user_home / "projects" / "a"] if folders else []))      # a folder inside: counted once
    shown = {0: "none yet: /folders, then a, lets peresearch read one", 2: "~/class, ~/projects",
             3: "~/class, ~/papers, ~/projects", 4: "~/class, ~/notes, ~/papers and 1 more",
             5: "~/books, ~/class, ~/notes and 2 more"}[folders]
    lines = ["✻ peresearch · research over your files and the web", "", f"  model    {model}", f"  folders  {shown}"]
    assert cli.welcome("default") == "\n".join(lines + ["", "/help for the commands · /status for the setup and the index"])
    assert cli.welcome("thesis").splitlines()[4] == "  project  thesis"                  # only when not the default


@pytest.mark.parametrize("keys, search", [
    ({}, "web: none (add a search MCP server in /settings) · papers: Semantic Scholar, arXiv · Wikipedia"),
    ({"PERESEARCH_SEARCH_MCP": "https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-k", "OPENALEX_API_KEY": "o"},
     "web: https://mcp.tavily.com/mcp/ · papers: OpenAlex, Semantic Scholar, arXiv · Wikipedia")])   # never its key
def test_status_shows_the_setup_the_index_and_what_was_left_out(home, user_home, monkeypatch, keys, search):
    """/status: the model, the search services, the folders, what the index holds, and what it left out counted by
    reason (each a command that lists those files); `/status <reason>` lists them, a long list cut with its count."""
    from peresearch.zetokrag.index import Index

    for k in ("PERESEARCH_SEARCH_MCP", "OPENALEX_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PERESEARCH_LLM_URL", "http://localhost:8000/v1")
    root, papers = user_home / "notes", user_home / "papers"
    root.mkdir()
    papers.mkdir()
    (root / "bpe.md").write_text("# BPE\n\nMerges the most frequent pair.\n")
    for i in range(53):
        (root / f"t{i:02}.csv").write_text("a,b\n1,2\n")
    (root / ".draft.md").write_text("hidden\n")
    (papers / "urls.txt").write_text("".join(f"http://x.org/{i}\n" for i in range(60)))     # judged data
    guard.set_roots([root, papers])
    index = Index(embedder=FakeEmbedder())
    index.update([root, papers])
    real = index.coverage
    index.coverage = lambda: {**real(), "some_new_reason": 2}             # a reason without a label: shown as named
    text = cli.status("default", index)
    assert text.splitlines()[:5] == ["- **Model** localhost:8000", f"- **Search** {search}",
                                     "- **Project** default (its conversations: /resume)",
                                     "- **Folders** ~/notes, ~/papers", "- **Index** 1 file · 1 chunk"]
    assert text.splitlines()[5:] == [                       # most first, ties by name, each with its command
        "", "**Left out of the search by meaning** (grep and read still reach them, protected paths excepted):",
        f"- 53 {workspace.PROBLEMS['data_type']} · `/status data_type`",
        "- 2 some_new_reason · `/status some_new_reason`",
        f"- 1 {workspace.PROBLEMS['data_file']} · `/status data`",
        f"- 1 {workspace.PROBLEMS['hidden']} · `/status hidden`"]
    listed = cli.status("default", index, "data_type").splitlines()
    assert listed[0] == f"**53 {workspace.PROBLEMS['data_type']}**"
    assert listed[1:51] == [f"- ~/notes/t{i:02}.csv" for i in range(50)] and listed[51:] == ["- … and 3 more"]
    assert cli.status("default", index, "hidden").splitlines() == [f"**1 {workspace.PROBLEMS['hidden']}**",
                                                                   "- ~/notes/.draft.md"]       # all shown: no "more"
    assert cli.status("default", index, "data").splitlines()[0] == f"**1 {workspace.PROBLEMS['data_file']}**"
    assert cli.status("default", index, "nope") == ("No files left out for `nope`. The reasons: `data`, `data_type`, "
                                                    "`hidden`, `some_new_reason`.")
    guard.set_roots([])
    assert cli.status("default", index).splitlines()[3] == "- **Folders** none"


def test_setup_declares_folders_indexes_and_saves_the_keys_privately(home, notes, fake_index, tmp_path, capsys,
                                                                     monkeypatch):
    modes = []                                    # the file's mode whenever keys are about to be written into it
    real_fdopen = cli.os.fdopen
    monkeypatch.setattr(cli.os, "fdopen", lambda fd, *a, **kw: (
        modes.append(stat.S_IMODE(os.fstat(fd).st_mode)) if os.readlink(f"/proc/self/fd/{fd}").endswith("settings.env")
        else None, real_fdopen(fd, *a, **kw))[1])                      # the settings file's opens (the audit log has its own)
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.env").write_text('PERESEARCH_SEARCH_MCP_KEY="EXA_API_KEY=old-exaX"\nOTHER=kept\n')   # quotes go
    (home / "settings.env").chmod(0o644)
    checks = []
    monkeypatch.setattr(cli, "print_check", lambda: checks.append(1) or 0)    # a model URL: checked after saving
    search_checks = []
    # Wizard wiring is tested here; MCP exchanges have their own local-server contracts.
    monkeypatch.setattr("peresearch.web.check", lambda searcher: (
        search_checks.append(searcher), ["web search: checked synthetic configuration"])[1])
    answers = iter([str(tmp_path / "nope"), str(notes), "", "http://localhost:8000/v1", "", "", "", "", "",
                    "npx -y tavily-mcp"])
    secrets, prompts = iter(["", "", "", "TAVILY_API_KEY=tvly-" + "x" * 24, "oa-key", ""]), []   # the model's,
    #                                                       embeddings', rerank's keys, the search server's, then papers'

    cli.cmd_setup(None, ask=lambda p: (prompts.append(p), next(answers))[1],
                  secret=lambda p: (prompts.append(p), next(secrets))[1])
    out = capsys.readouterr().out
    assert "not a folder" in out and f"added {notes.resolve()}" in out and "3 added" in out
    env = (home / "settings.env").read_text().splitlines()
    assert f"Settings go to {home / 'settings.env'}. Enter keeps the current value." in out
    assert sorted(env) == sorted(["OTHER=kept", "PERESEARCH_LLM_URL=http://localhost:8000/v1",
                                  "PERESEARCH_SEARCH_MCP=npx -y tavily-mcp",
                                  "PERESEARCH_SEARCH_MCP_KEY=TAVILY_API_KEY=tvly-" + "x" * 24, "OPENALEX_API_KEY=oa-key"])
    assert stat.S_IMODE((home / "settings.env").stat().st_mode) == 0o600
    assert prompts == ["Add a folder (empty to go on): "] * 3 + [
        "model endpoint (an OpenAI-compatible …/v1), e.g. http://localhost:11434/v1 or your provider's []: ",
        "model name, as your server or provider calls it (e.g. qwen3:8b; empty: the server's only model) []: ",
        "model key (your provider's API key; empty for a server on this machine) []: ",
        "embeddings endpoint (an OpenAI-compatible …/v1 serving /embeddings; empty: the search goes by words alone) []: ",
        "embedding model, as that server calls it (e.g. bge-m3; empty: the server's only model) []: ",
        "embeddings key (empty for a server on this machine) []: ",
        "rerank endpoint (a …/v1 serving /rerank; optional, empty to skip) []: ",
        "rerank model, as that server calls it (e.g. bge-reranker-v2-m3) []: ",
        "rerank key (empty for a server on this machine) []: ",
        "web search: an MCP server, as a command (e.g. npx -y tavily-mcp) or a URL (e.g. https://mcp.exa.ai/mcp); "
        "empty: no web search []: ",
        "its key, as NAME=value (the variable the server reads, e.g. TAVILY_API_KEY=tvly-…; empty if none) [(set)]: ",
        "OpenAlex key (free; adds a third scholarly source; empty to skip) []: ",
        "Semantic Scholar key (free for a school or work email; without one it shares a public pool that often "
        "refuses) []: "]
    assert out.startswith("Folders peresearch may read: none yet\n") and out.endswith("Saved. Start with `peresearch chat`.\n")
    assert checks == [1] and "\nChecking the model…\n" in out
    assert "\nChecking the search's endpoints…\n  embeddings: none set — the search goes by words alone" in out
    assert guard.roots() == [notes.resolve()]
    shown = []
    cli.cmd_setup(None, ask=lambda p: shown.append(p) or "", secret=lambda p: shown.append(p) or "")
    assert shown[1].endswith("[http://localhost:8000/v1]: ") and shown[10].endswith("[npx -y tavily-mcp]: ")
    assert shown[11].endswith("[(set)]: ")                                       # a URL or command is shown, a key not
    assert Index(home, embedder=FakeEmbedder()).rows()
    (home / "settings.env").unlink()                                      # a first setup: the file is new
    cli.cmd_setup(None, ask=lambda p: "", secret=lambda p: "")
    # never readable by others while the keys go in: the file found at 0o644, the one kept, a new one
    assert modes == [0o600] * 3
    assert [s.server.command if s else None for s in search_checks] == [
        ["npx", "-y", "tavily-mcp"], ["npx", "-y", "tavily-mcp"], None]
    assert all(s.server.proc is None for s in search_checks if s is not None)
    assert not hasattr(cli, "CPU_TRADE_OFF")            # no model runs in peresearch: no CPU trade-off to ask about


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
    monkeypatch.setenv("PERESEARCH_STEPS", "20")                       # /config's limits reach the agent
    a = cli.make_agent("proj", on_event, index="idx")
    assert isinstance(a, agent_module.Agent) and a.project == "proj" and a.llm is made["llm"] and a.on_event is on_event
    assert a.limits == agent_module.Limits(steps=20)
    from peresearch.papers import Papers
    from peresearch.wiki import Wikipedia
    assert isinstance(a.toolbox.papers, Papers) and isinstance(a.toolbox.wiki, Wikipedia)   # the scholarly sources
    monkeypatch.delenv("PERESEARCH_STEPS")
    monkeypatch.setattr("peresearch.zetokrag.index.Index", lambda: "own index")
    quiet = cli.make_agent()
    assert quiet.toolbox.searcher == ("searcher", "own index") and quiet.project == "default"
    made["on_wait"]("ignored")                                    # no listener: nothing breaks
    assert a.toolbox.searcher == ("searcher", "idx") and a.toolbox.web == "web"
    a2 = cli.make_agent("proj", on_event, index="idx")
    made["on_wait"]("waking up")
    assert events == [("wait", "waking up")] and a2.on_event is on_event
    shown = {}
    monkeypatch.setattr("peresearch.tui.Chat", lambda make, welcome, status, workspace, project: NS(
        run=lambda: (shown.update(ws=workspace), shown.update(make=make, welcome=welcome(), status=status("data"),
                                                               project=project))))
    monkeypatch.setattr(cli, "welcome", lambda project: f"welcome {project}")
    monkeypatch.setattr(cli, "status", lambda project, index, reason=None: f"status {project} {reason}"
                        + ("" if index is shown["ws"].index else " of another index"))
    cli.main(["chat", "--project", "p2"])
    assert (shown["welcome"], shown["status"], shown["project"]) == ("welcome p2", "status p2 data", "p2")
    monkeypatch.setattr(cli, "make_agent", lambda project, on_event, index: (project, on_event, index))
    assert shown["make"](on_event) == ("p2", on_event, shown["ws"].index)
    cli.main(["chat"])
    assert shown["project"] == "default"


def _ctrl_c(*args, **kwargs):
    """A real Ctrl+C under `pixi run`: SIGINT now and again a moment later (the terminal signals the whole process
    group and pixi forwards it to its child once more, measured on the real command)."""
    threading.Timer(0.05, os.kill, (os.getpid(), signal.SIGINT)).start()
    os.kill(os.getpid(), signal.SIGINT)           # raises here: the timer is set first
    end = time.monotonic() + 5
    while time.monotonic() < end:
        pass
    pytest.fail("Ctrl+C did not stop the command")


UPDATE_STOPPED = "stopped: what was saved is kept (an update saves its work every minute); `peresearch index` goes on from there\n"


@pytest.mark.parametrize("command, stdin, where, said", [
    (["index"], "", "update", UPDATE_STOPPED),
    (["setup"], "\n", "update", UPDATE_STOPPED),
    (["setup"], None, "prompt", "stopped\n"),
    (["find", "q"], "", "search", "stopped\n"),
    (["ask", "q"], "", "model", "stopped\n"),
], ids=["index updating", "setup updating", "setup at a prompt", "find searching", "ask waiting on the model"])
def test_ctrl_c_ends_any_command_with_the_shells_status_and_no_traceback(home, notes, fake_index, monkeypatch, capsys,
                                                                         request, command, stdin, where, said):
    from peresearch.workspace import Workspace
    from peresearch.zetokrag import search

    original = signal.getsignal(signal.SIGINT)
    request.addfinalizer(lambda: signal.signal(signal.SIGINT, original))                # pytest's own, whatever happens
    handler = signal.default_int_handler          # a known one: an earlier test (or a broken main) may have left another
    signal.signal(signal.SIGINT, handler)
    assert run(["add", str(notes)], capsys)[0] == 0 and signal.getsignal(signal.SIGINT) is handler  # restored: no Ctrl+C
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin) if stdin is not None else NS(readline=_ctrl_c, isatty=lambda: False))
    monkeypatch.setattr(Workspace, "update", _ctrl_c if where == "update" else lambda self, stop=None: pytest.fail("indexed"))
    monkeypatch.setattr(search.Searcher, "find", _ctrl_c)
    monkeypatch.setattr(cli, "make_agent", lambda project, on_event: NS(toolbox=NS(ask=None), ask=_ctrl_c))
    code, out, err = run(command, capsys)
    end = time.monotonic() + 0.3                  # the forwarded signal arrives in here if not already handled
    while time.monotonic() < end:
        pass
    assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN                           # the command is ending
    assert code == 130 and "Traceback" not in out + err and err == said


def test_an_unknown_command_or_missing_argument_is_a_usage_error(capsys, home, monkeypatch):
    opened = []
    monkeypatch.setattr(cli, "cmd_chat", lambda args: opened.append(args.project))
    assert cli.main([]) == 0 and opened == ["default"]
    for argv in (["fly"], ["find"], ["add"]):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2 and capsys.readouterr().err.startswith("usage: peresearch ")
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "peresearch setup" in capsys.readouterr().out                 # the commands are described


@pytest.mark.parametrize("text, said", [
    ('{"roots": ["/tmp"],}\n', "is not valid JSON (line 1, column 20): Expecting property name enclosed in double quotes"),
    ('["a list"]\n', "must hold an object ({…}), not a list"),
    ('{"exclude": "drafts/"}\n', "`exclude` must be a list of patterns (strings)")])
@pytest.mark.parametrize("command", [["folders"], ["index"], ["find", "x"], ["chat"]])
def test_a_config_file_the_user_broke_is_said_never_a_traceback(home, capsys, monkeypatch, text, said, command):
    """config.json is the user's to edit (files.py: exclude, include): a typo stops every command before it starts,
    the interface too, with where and what — not a traceback."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text(text)
    monkeypatch.setattr(cli, "cmd_chat", lambda args: pytest.fail("the interface started on a broken config"))
    assert cli.main(command) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"config.json ({home / 'config.json'}) ") and said in err and "Traceback" not in err
    assert err.rstrip().endswith("Fix it, or remove it to start again (it lists your folders).")


def test_a_valid_config_with_patterns_lets_every_command_run(home, capsys):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text('{"roots": [], "exclude": ["drafts/", "Xold/"], "include": ["!keep.md"]}')
    assert cli.main(["folders"]) == 0 and "config.json" not in capsys.readouterr().err


def test_check_tells_the_model_and_the_search_endpoints(home, monkeypatch, capsys):
    """`peresearch check`: the model check's code is the command's; the search's endpoints follow it, said whether
    set or not."""
    monkeypatch.setattr(cli, "print_check", lambda: 1)
    for name in ("PERESEARCH_EMBED_URL", "PERESEARCH_RERANK_URL", "PERESEARCH_SEARCH_MCP", "TAVILY_API_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert cli.cmd_check(None) == 1
    out = capsys.readouterr().out
    assert "Checking the search's endpoints…\n  embeddings: none set" in out and "\n  rerank: none set" in out
    assert "\n  web search: none set — what ships: arXiv" in out
    closed = []                                               # a server started to be checked is stopped after
    from peresearch import web
    monkeypatch.setattr(web.McpSearch, "from_settings", classmethod(lambda cls: type("S", (), {
        "search": lambda self, q, n: (_ for _ in ()).throw(web.Unavailable("down")),
        "close": lambda self: closed.append(1)})()))
    cli.cmd_check(None)
    assert "\n  web search: FAILED: down" in capsys.readouterr().out and closed == [1]
