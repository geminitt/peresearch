"""Contracts of what the interface writes (peresearch.tui, no app running): an answer as Markdown, an error as the
reader can act on it, the command menu's and /resume's rows, code colors, a source's place, and what a clicked link
may open."""

import re
from types import SimpleNamespace as NS

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from rich.cells import cell_len
from rich.console import Console
from textual.highlight import highlight
from textual.widgets._markdown import MarkdownFence

from peresearch import tui
from peresearch.agent import Answer, Check, Conversation
from peresearch.tools import Source
from peresearch.tui import (COMMANDS, REASONING_SHOWN, WHY, CodeColors, Fence, command_row, conversation_rows, explain,
                            location, render)

CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
ids = st.sampled_from(["N1", "N2", "W1", "W3"])
cite = st.lists(ids, min_size=1, max_size=3).map(lambda xs: "[" + ", ".join(xs) + "]")
words = st.text(alphabet="BPE merges pairs. \x1b[31m\x07é\n", max_size=15)
answer_text = st.lists(st.one_of(words, cite), max_size=8).map("".join)
sources = st.dictionaries(ids, st.tuples(st.text(alphabet="/a.md:L1-2 \x1b", min_size=1, max_size=12),
                                         st.text(alphabet="title\x1b", max_size=6)), max_size=4)


@given(answer_text, sources, st.lists(ids, max_size=2), st.lists(st.text(alphabet="quoted \x1b", min_size=1, max_size=90),
       max_size=2), st.booleans(), st.sampled_from(["", "steps", "time", "cancelled", "length", "refused", "odd"]),
       st.integers(0, 12), st.integers(0, 2_000_000))
@settings(max_examples=200)
@example("BPE merges pairs [N1, W1]. More [W1].", {"N1": ("/n/bpe.md:L3-5", "bpe.md"), "W1": ("https://e/bpe", "BPE")},
         [], [], False, "", 1, 10)
@example("SuperBPE+ appeared in 2026.", {}, [], [], False, "", 1, 0)                 # the model's memory only
@example("_(interrupted)_", {}, [], [], False, "cancelled", 1, 0)                   # not "without any source"
def test_an_answer_is_rendered_with_links_sources_checks_and_why_it_stopped(text, srcs, unknown, quotes, uncited,
                                                                           stopped, steps, tokens):
    sources = {k: Source(k, "web" if k[0] == "W" else "file", where, title, "body") for k, (where, title) in srcs.items()}
    answer = Answer(text, sources, Check(unknown, quotes, uncited), steps, tokens // 2, tokens - tokens // 2, stopped)
    out = render(answer)
    assert not CONTROL.search(out.replace("\n", "")), "terminal control characters must never be drawn"
    assert not re.search(r"\[[NW]\d+\](?!\()", out.split("\n**Sources read**")[0])    # every citation is a link
    assert out == "\n".join(expected_render(text, srcs, unknown, quotes, uncited, stopped, steps, tokens))


def expected_render(text, srcs, unknown, quotes, uncited, stopped, steps, tokens):
    """The layout, line by line: the answer with its citations as links, a blank line, each source once in order
    (with its title when it has one), what the check found, the no-source remark, why it stopped, the counts."""
    clean = tui.guard.sanitize
    lines = [tui._CITE.sub(lambda m: ", ".join(f"[{i.strip()}](source:{i.strip()})" for i in m.group(1).split(",")),
                           clean(text or "_(no answer)_")), ""]
    if srcs:
        lines += ["**Sources read**"] + [f"- [{k}](source:{k}) {clean(tui.home_path(where))}"
                                         + (f" — {clean(title)}" if title else "")
                                    for k, (where, title) in srcs.items()]
    if unknown:
        lines.append("\n> **Check:** citation evidence unavailable; read again: " + ", ".join(unknown))
    if quotes:
        lines.append("\n> **Check:** quotes not found in the cited sources: "
                     + "; ".join(f"“{clean(q)[:80]}”" for q in quotes))
    if uncited:
        lines.append("\n> **Check:** the answer cites none of the sources read")
    if not srcs and stopped != "cancelled":
        lines.append("\n_Answered without any source: from the model's own knowledge._")
    why = {"steps": "the step limit was reached", "time": "the time limit was reached", "cancelled": "interrupted",
           "length": "the answer was cut at the model's token limit",
           "refused": "the model kept making calls that were refused"}
    if stopped:
        lines.append("\n> stopped: " + why.get(stopped, stopped))
    checked = "✓ citations checked · " if srcs and not (unknown or quotes or uncited) else ""   # shown when it passed
    return lines + [f"\n<sub>{checked}{steps} model calls · {tokens:,} tokens</sub>"]


@given(st.text(alphabet="thought.\n\x1b", max_size=3 * REASONING_SHOWN), st.booleans())
@settings(max_examples=40)
@example("early thought. " * 2000 + "the last step before answering", True)
@example("x" * REASONING_SHOWN, True)                     # exactly the limit: all of it, nothing said to be cut
@example("y" + "x" * REASONING_SHOWN, True)               # one over: the first character goes
def test_the_reasoning_is_shown_on_request_from_its_end(thought, shown):
    out = render(Answer("ok", {}, Check(), 1, reasoning=thought), reasoning=shown)
    clean = tui.guard.sanitize(thought)
    if not shown or not thought:
        assert "**Reasoning**" not in out
        # an answer kept from an earlier session has no reasoning at all
        bare = NS(text="ok", sources={}, check=Check(), stopped="", steps=1, prompt_tokens=0, completion_tokens=0)
        assert render(bare, reasoning=True) == out
        return
    cut = len(clean) - REASONING_SHOWN
    head = out.split("\n\nok\n")[0]
    assert head.startswith("> **Reasoning**\n>\n> ")
    if cut > 0:
        assert f"_({cut:,} earlier characters not shown)_" in head and head.endswith(clean[cut:].replace("\n", "\n> "))
    else:
        assert head.endswith(clean.replace("\n", "\n> ")) and "earlier characters" not in head


@pytest.mark.parametrize("name", sorted(WHY))
def test_every_known_error_is_explained_in_plain_words(name):
    e = type(name, (Exception,), {})("detail \x1b[2J" + "x" * 400)
    assert explain(e) == f"> **Error:** {WHY[name]} `/retry` asks again. <sub>({name})</sub>"


def test_what_the_user_reads_never_names_the_owners_test_tools():
    """The model is the user's (any OpenAI-compatible API): no text a user reads points to the owner's test setup."""
    from peresearch import llm, settings, tui
    texts = list(tui.WHY.values()) + [label for _, label, _ in settings.FIELDS]
    texts += [s for s in vars(llm).values() if isinstance(s, str)]
    import inspect
    texts += [inspect.getsource(llm.LLM._create)]                      # the waiting messages are said from here
    for text in texts:
        assert not any(name in text for name in ("chat-local", "Modal container", "Modal endpoint", "modal_vllm",
                                                 "modal deploy")), text


def test_other_errors_keep_their_own_text_safe_and_short():
    assert explain(RuntimeError("no model endpoint: set PERESEARCH_LLM_URL")) == (
        "> **Error:** No model endpoint is set: /model chooses it (or set "
        "`PERESEARCH_LLM_URL`). `/retry` asks again. <sub>(RuntimeError: no model endpoint: set PERESEARCH_LLM_URL)</sub>")
    assert explain(RuntimeError("disk full")) == ("> **Error:** disk full `/retry` asks again. "
                                                  "<sub>(RuntimeError: disk full)</sub>")
    odd = explain(ValueError("weird \x1b[2J" + "y" * 400))
    assert "\x1b" not in odd and odd == f"> **Error:** weird {'y' * 294} `/retry` asks again. <sub>(ValueError: weird {'y' * 294})</sub>"


@given(st.lists(st.tuples(st.floats(0, 2e9), st.integers(1, 1200), st.text(alphabet="ab \n\x1bé机", min_size=1, max_size=200)),
                max_size=6), st.integers(20, 200))
@example([(0, 1, "x"), (0, 12, "y")], 60)
def test_resume_rows_line_up_fit_the_width_and_keep_one_line(convs, width):
    past = [Conversation(f"c{i}", t, t, first, n) for i, (t, n, first) in enumerate(convs)]
    rows = conversation_rows(past, width)
    assert len(rows) == len(past)
    counts = [f"{c.questions} question{'' if c.questions == 1 else 's'}" for c in past]
    starts = set()
    for row, c, count in zip(rows, past, counts):
        plain = row.plain
        assert "\n" not in plain and not CONTROL.search(plain)
        head = f"{tui.time.strftime('%Y-%m-%d %H:%M', tui.time.localtime(c.last))}  {count:<{max(map(len, counts))}}  "
        assert plain.startswith(head)
        starts.add(cell_len(head))                                       # the questions start in one column
        first = " ".join(tui.guard.sanitize(c.first).split())
        room = max(8, width - cell_len(head))
        if cell_len(first) <= room:
            assert plain[len(head):] == first
        else:
            assert plain.endswith("…") and cell_len(plain) - cell_len(head) <= room
        drawn = Console(width=width, color_system=None).render_lines(row, pad=False)
        assert len(drawn) == 1                                           # drawn on one line, even when too wide
        assert all(seg.style and seg.style.dim for seg in drawn[0][1:] if seg.text.strip())   # the question dimmed
    assert len(starts) <= 1


@given(st.integers(26, 140))                              # 24 before the pointer's two-cell gutter
def test_a_command_row_keeps_its_description_in_one_column(width):
    """The descriptions start two columns after the longest command; a wrapped one goes on under itself, with no
    blank line; they are dimmed, the commands not."""
    column = max(map(len, COMMANDS)) + 2 + 2          # after the pointer's two-cell gutter
    console = Console(width=width, color_system=None)
    for command, description in COMMANDS.items():
        lines = console.render_lines(command_row(command, description), pad=False)
        plain = ["".join(seg.text for seg in line).rstrip() for line in lines]
        assert plain[0].startswith("  " + command.ljust(column - 2)) and plain[0][column] != " "
        assert all(p.startswith(" " * column) and p[column] != " " for p in plain[1:]), plain
        assert " ".join(p[column:] for p in plain).split() == description.split()
        for line in lines:
            at = 0
            for seg in line:                                # by column: the command undimmed, the description dimmed
                if seg.text.strip():
                    assert bool(seg.style and seg.style.dim) == (at >= column), (seg, plain)
                at += len(seg.text)


@pytest.mark.parametrize("language", ["python", "bash", ""])
@pytest.mark.parametrize("ansi, dark", [(True, True), (True, False), (False, True), (False, False)])
def test_code_is_colored_by_the_terminal_palette_when_the_theme_is(language, ansi, dark):
    """With the terminal's palette (ANSI), code takes CodeColors and the fence's language (none: plain); otherwise
    Textual's own colors."""
    code = "def f(x):\n    return `ls` + 1  # c\n"
    got = Fence.highlight(code, language, ansi=ansi, dark=dark)
    want = (highlight(code, language=language or None, theme=CodeColors) if ansi
            else MarkdownFence.highlight(code, language, ansi, dark))
    assert [(sp.start, sp.end, str(sp.style)) for sp in got.spans] == [(sp.start, sp.end, str(sp.style)) for sp in want.spans]
    assert str(got) == code.rstrip("\n")


@pytest.mark.parametrize("where, want", [
    ("/n/bpe.md:L3-5", ("/n/bpe.md", 3)), ("/n/bpe.md:L7", ("/n/bpe.md", 7)), ("/n/a.pdf:p.3", ("/n/a.pdf", None)),
    ("/n/a.pdf:p.2-3", ("/n/a.pdf", None)), ("/n/b.ipynb:cell 4", ("/n/b.ipynb", None)),
    ("/n/b.ipynb:cell 6-10", ("/n/b.ipynb", None)), ("/n, /m (listing)", ("/n", None)),
    ("/home/user/notes, /m (listing)", ("/home/user/notes", None)),
    ("https://example.org/a:L3", ("https://example.org/a", 3)), ("/weird:name.md", ("/weird:name.md", None)),
    ("plain", ("plain", None))])
def test_a_sources_place_is_read_back_as_a_path_and_a_line(where, want):
    assert location(where) == want


@pytest.mark.parametrize("wsl, code, platform", [(True, True, "linux"), (True, False, "linux"), (False, True, "linux"),
                                                 (False, False, "linux"), (False, False, "darwin")])
def test_a_link_opens_a_page_or_shows_a_file_and_never_runs_one(monkeypatch, tmp_path, wsl, code, platform):
    runs = []
    monkeypatch.setattr(tui.subprocess, "Popen", lambda cmd, **kw: runs.append((cmd, kw)))
    monkeypatch.setattr(tui, "on_wsl", lambda: wsl)
    monkeypatch.setattr(tui.sys, "platform", platform)
    monkeypatch.setattr(tui.shutil, "which", lambda name: "/usr/bin/code" if code and name == "code" else None)
    monkeypatch.setattr(tui, "windows_path", lambda p: "C:\\x" + p.replace("/", "\\"))
    evil = tmp_path / "run me.bat"
    evil.write_text("echo hi")
    assert tui.open_outside(str(evil), 3) is None
    assert tui.open_outside(str(evil)) is None
    assert tui.open_outside(str(tmp_path)) is None
    assert tui.open_outside("HTTPS://example.org/a?b=1&c=2") is None
    files = (["code", "-g", f"{evil}:3"], ["code", "-g", f"{evil}:1"]) if code else \
        ([["explorer.exe", "/select,C:\\x" + str(evil).replace("/", "\\")]] * 2 if wsl else [["xdg-open", str(tmp_path)]] * 2)
    folder = ["explorer.exe", "C:\\x" + str(tmp_path).replace("/", "\\")] if wsl else ["xdg-open", str(tmp_path)]
    page = ["explorer.exe" if wsl else "open" if platform == "darwin" else "xdg-open", "HTTPS://example.org/a?b=1&c=2"]
    assert [cmd for cmd, _ in runs] == [*files, folder, page]
    assert all(kw == {"stdin": tui.subprocess.DEVNULL, "stdout": tui.subprocess.DEVNULL,
                      "stderr": tui.subprocess.DEVNULL, "start_new_session": True} for _, kw in runs)
    for bad, why in (("javascript:alert(1)", "only web links (http, https) and your own files open"),
                     ("file:///etc/passwd", "only web links (http, https) and your own files open"),
                     (str(tmp_path / "missing.md"), f"{tmp_path / 'missing.md'}: not found")):
        assert tui.open_outside(bad) == why
    assert len(runs) == 4                                                # refused: nothing run
    for name in ("talk.pptx", "paper.pdf", "notes.docx"):              # converted: an editor cannot show it
        doc = tmp_path / name
        doc.write_bytes(b"PK")
        assert tui.open_outside(str(doc), 2) is None
        want = ["explorer.exe", "/select,C:\\x" + str(doc).replace("/", "\\")] if wsl else ["xdg-open", str(tmp_path)]
        assert runs[-1][0] == want, name                                 # selected for its own program to open


def test_wsl_is_recognized_and_paths_are_converted_by_wslpath(monkeypatch):
    monkeypatch.setattr(tui.platform, "release", lambda: "6.6.87.2-microsoft-standard-WSL2")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    assert tui.on_wsl()
    monkeypatch.setattr(tui.platform, "release", lambda: "6.8.0-generic")
    assert not tui.on_wsl()
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert tui.on_wsl()
    seen = []
    monkeypatch.setattr(tui.subprocess, "run", lambda cmd, **kw: (seen.append((cmd, kw)), NS(stdout="C:\\x\\y\n"))[1])
    assert tui.windows_path("/x/y") == "C:\\x\\y" and seen == [
        (["wslpath", "-w", "/x/y"], {"capture_output": True, "text": True, "check": True})]


def test_a_failed_internal_check_says_it_is_a_bug_and_where(tmp_path, monkeypatch):
    from peresearch.zetokrag import models
    with pytest.raises(AssertionError) as caught:
        models.windows(5, 1, 1)
    text = explain(caught.value)
    assert text.startswith("> **Error:** an internal check of peresearch failed at `peresearch/zetokrag/models.py:")
    assert "This is a bug in peresearch" in text and text.endswith("<sub>(AssertionError)</sub>")
    # Installed, peresearch sits in site-packages next to every library: the place is peresearch's own innermost
    # frame, never a library's (an assert inside textual was reported as peresearch's, at textual/widget.py)
    site = tmp_path / "site-packages"
    monkeypatch.setattr(tui, "__file__", str(site / "peresearch" / "tui.py"))
    code = {}
    exec(compile("def check():\n    assert False, 'the library checks itself'\n", str(site / "textual" / "widget.py"),
                 "exec"), code)
    exec(compile("def call(check):\n    check()\n", str(site / "peresearch" / "agent.py"), "exec"), code)
    for run, where in ((lambda: code["call"](code["check"]), "peresearch/agent.py:2"), (code["check"], "")):
        with pytest.raises(AssertionError) as caught:
            run()
        assert tui.place(caught.value) == where
    assert explain(caught.value).startswith(          # a library's own: no place of ours to name
        "> **Error:** an internal check of peresearch failed: the library checks itself. This is a bug")
    assert explain(AssertionError()).startswith("> **Error:** an internal check of peresearch failed: (no message). ")
