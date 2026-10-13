"""Regressions for state changes between otherwise valid operations."""

from pathlib import Path

import pytest

from peresearch import guard
from peresearch.tools import Toolbox
from peresearch.workspace import Workspace
from peresearch.zetokrag.index import Index
from peresearch.zetokrag.search import Searcher
from tests.conftest import FakeReranker


@pytest.fixture
def live_workspace(home, tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("HOME", str(user))
    root = user / "notes"
    root.mkdir()
    path = root / "research.md"
    path.write_text("# Research\n\nRevocable evidence about gradient descent.\n")
    guard.set_roots([root])
    index = Index(home, embedder=None)
    index.update()
    return Workspace(index), root, path


@pytest.mark.parametrize("operation", ["read", "tree", "glob", "grep", "search_notes", "overview", "catalog"])
def test_retained_tools_revoke_removed_folder_before_index_update(live_workspace, operation):
    ws, root, path = live_workspace
    tool = Toolbox(Searcher(ws.index, reranker=None))
    assert "Revocable evidence" in tool.read(str(path))
    assert ws.remove(str(root))
    # Deliberately retain the index: its asynchronous cleanup has not run yet.
    calls = {"read": lambda: tool.call("read", {"path": str(path)}),
             "tree": tool.tree, "glob": lambda: tool.glob("**/*"), "grep": lambda: tool.grep("Revocable"),
             "search_notes": lambda: tool.search_notes("Revocable evidence"),
             "overview": lambda: tool.overview("Revocable evidence"), "catalog": tool.overview}
    result = calls[operation]()
    assert "Revocable evidence" not in result
    if operation != "read":
        assert str(path) not in result
    assert tool.roots == []
    ws.add(str(root))
    assert "Revocable evidence" in tool.read(str(path))


@pytest.mark.parametrize("operation", ["search_notes", "overview"])
def test_revoked_index_text_never_reaches_reranker(live_workspace, operation):
    ws, root, _ = live_workspace
    class Spy(FakeReranker):
        sent = []
        def scores(self, query, texts):
            self.sent.extend(texts)
            return super().scores(query, texts)
    spy = Spy()
    tool = Toolbox(Searcher(ws.index, reranker=spy))
    ws.remove(str(root))
    getattr(tool, operation)("Revocable evidence")
    assert spy.sent == []


def test_removed_folder_requires_a_new_explicit_read_grant(live_workspace):
    ws, root, path = live_workspace
    asked = []
    tool = Toolbox(ask=lambda name, p: asked.append((name, p)) or "once")
    ws.remove(str(root))
    assert "Revocable evidence" in tool.read(str(path))
    assert asked == [("read", path)]


@pytest.mark.parametrize("heading", ["# Research\n\n", ""])
def test_shifted_passage_uses_current_lines_without_old_neighbors(live_workspace, heading):
    ws, root, path = live_workspace
    original = heading + "\n\n".join(f"paragraph {i} " + chr(65+i) * 900 for i in range(4)) + "\n"
    path.write_text(original)
    ws.index.update()
    search = Searcher(ws.index, reranker=None)
    path.write_text("\n" * 10 + original.replace("paragraph 0 " + "A" * 900, "replacement neighbor"))
    hits, _, _ = search.find("paragraph 1")
    hit = next(h for h in hits if "B" * 900 in h.text)
    current = path.read_text().splitlines()
    assert hit.start == next(i for i, line in enumerate(current, 1) if "B" * 900 in line)
    assert hit.before is None and hit.after is None
    assert hit.file_changed
    assert not hit.section
    shown = Toolbox(search).search_notes("paragraph 1")
    assert "A" * 900 not in shown
    assert "location verified in the current file" in shown


def test_changed_passage_cannot_join_paragraphs_in_reversed_order(live_workspace):
    ws, _, path = live_workspace
    path.write_text("# Research\n\nalpha beta gamma\n\ndelta epsilon zeta\n")
    ws.index.update()
    path.write_text("# Research\n\ndelta epsilon zeta\n\nalpha beta gamma\n")
    hits, stale, _ = Searcher(ws.index, reranker=None).find("alpha beta delta epsilon")
    assert not hits and stale


@pytest.mark.parametrize("operation", ["search_notes", "overview"])
def test_revoked_candidates_do_not_hide_allowed_candidates(live_workspace, monkeypatch, operation):
    from peresearch.zetokrag import search
    ws, root, _ = live_workspace
    other = root.parent / "other"
    other.mkdir()
    path = other / "current.md"
    path.write_text("# Current\n\nRevocable evidence is available here. " + "unrelated " * 100 + "\n")
    ws.add(str(other))
    ws.update()
    monkeypatch.setattr(search, "N_FIRST", 1)
    monkeypatch.setattr(search, "N_RERANK", 1)
    tool = Toolbox(Searcher(ws.index, reranker=None))
    ws.remove(str(root))
    assert str(path) in getattr(tool, operation)("Revocable evidence")


def test_a_file_changed_during_validation_does_not_return_old_text(live_workspace, monkeypatch):
    from peresearch.zetokrag import search
    ws, _, path = live_workspace
    real = search.sha256
    def race(p):
        digest = real(p)
        p.write_text("# Research\n\nEntirely replaced evidence.\n")
        return digest
    monkeypatch.setattr(search, "sha256", race)
    hits, stale, _ = Searcher(ws.index, reranker=None).find("Revocable evidence")
    assert hits == [] and stale == 1


def test_changed_copy_locations_are_verified_too(live_workspace):
    ws, root, original = live_workspace
    original.write_text("# Research\n\nGradient descent optimizes weights. " + " ".join(f"word{i}" for i in range(60)) + "\n")
    copy = root / "copy.md"
    copy.write_text(original.read_text().replace("word59", "word58"))
    ws.update()
    search = Searcher(ws.index, reranker=None)
    rows, _, copies = search.candidates("Gradient descent")
    assert copies
    copied = Path(copies[0][0][1])
    copied.write_text("\n" * 10 + copied.read_text())
    hits, _, _ = search.find("Gradient descent")
    assert hits[0].copies
    assert hits[0].copies[0].endswith(":L11-13")


@pytest.mark.parametrize("change", ["model", "url", "key", "disabled"])
def test_workspace_rebuilds_configured_index_for_effective_embedding_changes(live_workspace, monkeypatch, change):
    from peresearch import settings
    from peresearch.zetokrag import embed
    from tests.conftest import FakeEmbedder
    _, root, _ = live_workspace
    for suffix in ("URL", "MODEL", "KEY"):
        monkeypatch.delenv("PERESEARCH_EMBED_" + suffix, raising=False)
    settings.save({"PERESEARCH_EMBED_URL": "http://first/v1", "PERESEARCH_EMBED_MODEL": "first",
                   "PERESEARCH_EMBED_KEY": "first-credential"})
    class Configured(FakeEmbedder):
        def __init__(self):
            self.signature = settings.get("PERESEARCH_EMBED_MODEL") + " at " + settings.get("PERESEARCH_EMBED_URL")
            self.key = settings.get("PERESEARCH_EMBED_KEY")
    monkeypatch.setattr(embed, "from_settings", lambda: (Configured() if settings.get("PERESEARCH_EMBED_URL") else None, None))
    ws = Workspace()
    first = ws.index
    ws.update()
    assert first.snapshot()[2] is not None
    updated = dict(settings.saved())
    setting, value = {"model": ("MODEL", "second"), "url": ("URL", "http://second/v1"),
                      "key": ("KEY", "second-credential"), "disabled": ("URL", "")}[change]
    updated["PERESEARCH_EMBED_" + setting] = value
    settings.save(updated)
    current = ws.index
    assert current is not first and current is ws.index
    assert current.rows() and current.home == first.home
    if change == "disabled":
        assert current.embedder is None and current.snapshot()[2] is None
    elif change == "key":
        assert current.embedder.key == value and current.snapshot()[2] is not None
    else:
        assert value in current.signature() and current.snapshot()[2] is None
        ws.update()
        assert current.snapshot()[2] is not None


@pytest.mark.parametrize("existing_agent", [False, True])
@pytest.mark.parametrize("usage,expected_usage", [(None, (0, 0, 0)), ({"requests": 3}, (3, 0, 0)),
    ({"requests": 3, "prompt_tokens": 101, "completion_tokens": 17}, (3, 101, 17))])
@pytest.mark.parametrize("body", ["selected answer [N1]", "selected answer [N1]\n<sub>quoted note</sub>\nend of answer"])
def test_resume_rebinds_export_retry_and_sources(home, tmp_path, existing_agent, usage, expected_usage, body):
    import json
    from peresearch.agent import Agent, Answer, Check, history_file
    from peresearch.llm import Reply
    from peresearch.tools import Source
    from tests.helpers import ScriptedLLM
    from peresearch.tui import Chat
    from types import SimpleNamespace
    history = history_file("default")
    history.parent.mkdir(parents=True, exist_ok=True)
    turn = {"conversation": "selected", "question": "selected question", "answer": body,
            "sources": {"N1": "/selected.md:L2"}}
    if usage is not None:
        turn["usage"] = usage
    history.write_text(json.dumps(turn) + "\n")
    left = Answer("left behind", {"N9": Source("N9", "file", "/left.md:L1", "", "left")}, Check(), 1)
    app = Chat(lambda e: Agent(ScriptedLLM(Reply("next")), Toolbox(roots=[])), index_on_start=False)
    app.hide_menu = lambda: None
    app.query_one = lambda *args: SimpleNamespace(remove_children=lambda: None)
    displayed = []
    app.add = displayed.append
    app.note = lambda text: None
    app.last, app.last_question, app.exchanges = left, ("left question", "left question"), [("left question", left)]
    if existing_agent:
        app.agent = app.make_agent(None)
    app.resume("selected")
    assert [(q, a.text) for q, a in app.exchanges] == [("selected question", body)]
    assert app.last_question == ("selected question", "selected question")
    assert set(app.last.sources) == {"N1"}
    assert (app.last.steps, app.last.prompt_tokens, app.last.completion_tokens) == expected_usage
    assert displayed and app.sources_of[displayed[-1]] is app.last
    assert "selected answer" in displayed[-1]._initial_markdown
    assert "model calls" not in displayed[-1]._initial_markdown
    if "quoted note" in body:
        assert "quoted note" in displayed[-1]._initial_markdown and "end of answer" in displayed[-1]._initial_markdown
    target = tmp_path / "selected.md"
    app.export(str(target))
    assert "left" not in target.read_text() and "selected answer" in target.read_text()
    app.resume("empty")
    assert app.exchanges == [] and app.last is None and app.last_question is None


@pytest.mark.parametrize("command", ["/new", "/clear", "/reset"])
def test_new_conversation_clears_retry_and_last_sources(home, command):
    from peresearch.agent import Answer, Check
    from peresearch.tui import Chat
    from types import SimpleNamespace
    app = Chat(lambda e: None, index_on_start=False)
    app.query_one = lambda *args: SimpleNamespace(remove_children=lambda: None)
    app.note = lambda text: None
    app.last = Answer("old", {}, Check(), 0)
    app.last_question = ("old", "old")
    app.command(command)
    assert app.last is None and app.last_question is None and app.exchanges == []


def test_workspace_notices_settings_changed_before_it_received_the_index(home, monkeypatch):
    from peresearch import settings
    from peresearch.zetokrag import embed
    monkeypatch.setattr(embed, "from_settings", lambda: (None, None))
    monkeypatch.delenv("PERESEARCH_EMBED_MODEL", raising=False)
    before = Index()
    settings.save({"PERESEARCH_EMBED_MODEL": "new-model"})
    assert Workspace(before).index is not before


def test_old_and_new_embedding_workers_cannot_interleave_vector_batches(live_workspace, monkeypatch):
    import threading
    from peresearch import settings
    from peresearch.zetokrag import embed
    from tests.conftest import FakeEmbedder
    old_entered, new_entered, old_release, new_release = [threading.Event() for _ in range(4)]
    class Blocking(FakeEmbedder):
        def __init__(self, name):
            self.signature, self.name = name, name
        def documents(self, texts):
            (old_entered if self.name == "old" else new_entered).set()
            assert (old_release if self.name == "old" else new_release).wait(5)
            return super().documents(texts)
    monkeypatch.setattr(embed, "from_settings", lambda: (Blocking(settings.get("PERESEARCH_EMBED_MODEL")), None))
    settings.save({"PERESEARCH_EMBED_MODEL": "old"})
    ws = Workspace()
    old = ws.index
    errors = []
    def update(index):
        try:
            index.update()
        except BaseException as error:
            errors.append(error)
    first = threading.Thread(target=update, args=(old,))
    first.start()
    second = None
    try:
        assert old_entered.wait(3)
        settings.save({"PERESEARCH_EMBED_MODEL": "new"})
        second = threading.Thread(target=update, args=(ws.index,))
        second.start()
        interleaved = new_entered.wait(0.2)
    finally:
        old_release.set()
        first.join(5)
        new_release.set()
        if second is not None:
            second.join(5)
    assert not first.is_alive() and second is not None and not second.is_alive() and not errors
    assert not interleaved
    assert ws.index.read("SELECT value FROM meta WHERE key='embedder'") == [("new",)]


def test_independent_index_writers_discard_batches_from_a_replaced_embedder(live_workspace):
    """Independent Index objects also represent separate processes; their Python locks cannot protect SQLite."""
    import threading
    import numpy as np
    from tests.conftest import FakeEmbedder
    ws, root, _ = live_workspace
    entered, release = [threading.Event() for _ in range(4)], [threading.Event() for _ in range(2)]
    class Blocking(FakeEmbedder):
        def __init__(self, number):
            self.number, self.signature = number, f"model-{number}"
        def documents(self, texts):
            entered[self.number].set()
            assert release[self.number].wait(5)
            return np.roll(super().documents(texts), self.number, axis=1)
    older, newer = Index(ws.index.home, embedder=Blocking(0)), Index(ws.index.home, embedder=Blocking(1))
    reports, errors = {}, []
    def update(index, name):
        try:
            reports[name] = index.update([root])
        except BaseException as error:
            errors.append(error)
    first = threading.Thread(target=update, args=(older, "older"))
    second = threading.Thread(target=update, args=(newer, "newer"))
    first.start()
    try:
        assert entered[0].wait(3)
        second.start()
        assert entered[1].wait(3)
        release[0].set()
        first.join(5)             # its delayed batch arrives after the other writer changed meta
    finally:
        for event in release:
            event.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive() and not errors
    expected = newer.embedder.documents([newer.ranking_text(r) for r in newer.rows()])
    assert np.array_equal(newer.dense()[1], expected)
    assert "configuration changed" in reports["older"].embed_error


def test_old_dimension_probe_cannot_delete_a_new_embedders_vectors(live_workspace):
    import threading
    from peresearch.zetokrag.embed import EndpointError
    from tests.conftest import FakeEmbedder
    ws, root, _ = live_workspace
    class Original(FakeEmbedder):
        signature = "original"
    Index(ws.index.home, embedder=Original()).update([root])
    entered, release = threading.Event(), threading.Event()
    class OldProbe(Original):
        dim = 512
        def queries(self, texts):
            entered.set()
            assert release.wait(5)
            return super().queries(texts)
    class New(FakeEmbedder):
        signature = "new"
    older, newer = Index(ws.index.home, embedder=OldProbe()), Index(ws.index.home, embedder=New())
    errors = []
    def probe():
        try:
            older.embed()
        except EndpointError as error:
            errors.append(error)
    thread = threading.Thread(target=probe)
    thread.start()
    try:
        assert entered.wait(3)
        newer.update([root])
        expected = newer.dense()[1].copy()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert newer.dense()[1].tolist() == expected.tolist()
    assert errors and "configuration changed" in str(errors[0])


def test_explicit_root_assignment_changes_retained_tools_scope(live_workspace):
    ws, root, path = live_workspace
    other = root.parent / "other"
    other.mkdir()
    second = other / "other.md"
    second.write_text("# Other\n\nDifferent evidence about tokenization.\n")
    ws.add(str(other))
    ws.update()
    tool = Toolbox(Searcher(ws.index, reranker=None))
    tool.roots = [str(other), str(other / "nested")]
    assert tool.roots == [other]
    assert "Different evidence" in tool.read(str(second))
    assert tool.call("read", {"path": str(path)}).startswith("refused")
    assert tool.searcher.allowed(str(second)) and not tool.searcher.allowed(str(path))


def test_live_permission_filter_keeps_embeddings_for_allowed_documents(live_workspace):
    from tests.conftest import FakeEmbedder
    ws, root, _ = live_workspace
    class Spy(FakeEmbedder):
        calls = []
        def queries(self, texts):
            self.calls.extend(texts)
            return super().queries(texts)
    embedder = Spy()
    index = Index(ws.index.home, embedder=embedder)
    index.update([root])
    search = Searcher(index, reranker=None)
    Toolbox(search)
    assert search.find("gradient descent")[0]
    assert embedder.calls == ["gradient descent"]
    ws.remove(str(root))
    rows, scores = search.rank("gradient descent")
    assert rows == [] and scores.size == 0


def test_shifted_crlf_passage_has_current_lines_and_normalized_text(live_workspace):
    ws, _, path = live_workspace
    original = "# Heading\r\n\r\nDistinctive evidence on gradient descent.\r\n"
    path.write_bytes(original.encode())
    ws.update()
    path.write_bytes(("\r\n" * 3 + original).encode())
    hits, stale, _ = Searcher(ws.index, reranker=None).find("gradient descent")
    assert hits and not stale
    assert hits[0].start == 4 and hits[0].end == 6
    assert "\r" not in hits[0].text and not hits[0].section


@pytest.mark.parametrize("remove_first", [False, True])
def test_one_revoked_or_missing_copy_does_not_hide_a_later_allowed_copy(live_workspace, monkeypatch, remove_first):
    import numpy as np
    from peresearch.zetokrag.index import sha256
    ws, root, main = live_workspace
    copies = [root / (name + ".md") for name in ("first", "second")]
    for path in copies:
        path.write_text(main.read_text())
    row = (1, str(main), "Research", "line", 1, 3, main.read_text().strip(), sha256(main), 0)
    twins = [(i+2, str(path), "Research", "line", 1, 3, row[6], sha256(path), 0)
             for i, path in enumerate(copies)]
    search = Searcher(ws.index, reranker=None)
    monkeypatch.setattr(search, "candidates", lambda query: ([row], np.array([np.nan]), {0: twins}))
    if remove_first:
        copies[0].unlink()
    else:
        search.allowed = lambda path: path != str(copies[0])
    hits, _, _ = search.find("gradient descent")
    assert hits[0].copies == [str(copies[1]) + ":L1-3"]


@pytest.mark.parametrize("change", ["URL", "MODEL", "KEY"])
def test_injected_configured_index_stays_stable_until_its_own_settings_change(home, monkeypatch, change):
    from peresearch import settings
    from peresearch.zetokrag import embed
    monkeypatch.setattr(embed, "from_settings", lambda: (None, None))
    settings.save({"PERESEARCH_EMBED_URL": "http://initial/v1", "PERESEARCH_EMBED_MODEL": "initial",
                   "PERESEARCH_EMBED_KEY": "initial-key"})
    custom = Index(home / "custom")
    ws = Workspace(custom)
    assert ws.index is custom and ws.index is custom
    values = settings.saved()
    values["PERESEARCH_EMBED_" + change] = "changed"
    settings.save(values)
    assert ws.index is not custom and ws.index.home == custom.home


def test_explicit_index_without_configuration_metadata_is_not_replaced(home):
    from peresearch import settings
    index = object()
    ws = Workspace(index)
    settings.save({"PERESEARCH_EMBED_MODEL": "changed"})
    assert ws.index is index


@pytest.mark.parametrize("operation", ["rank", "documents"])
def test_scope_revoked_after_candidate_selection_never_leaves_for_reranking(live_workspace, monkeypatch, operation):
    ws, root, path = live_workspace
    other = root.parent / "other"
    other.mkdir()
    kept = other / "allowed.md"
    kept.write_text("# Allowed\n\nDistinct current evidence about gradient descent.\n")
    ws.add(str(other))
    ws.update()
    spy = FakeReranker()
    sent = []
    original_scores = spy.scores
    spy.scores = lambda q, texts: (sent.extend(texts), original_scores(q, texts))[1]
    search = Searcher(ws.index, reranker=spy)
    allowed = {str(path), str(kept)}
    search.allowed = allowed.__contains__
    original_rows = ws.index.rows
    def rows(ids=None):
        result = original_rows(ids)
        # The file disappears from the live scope between candidate selection and external scoring.
        allowed.discard(str(path))
        return sorted(result, key=lambda r: r[1] != str(path))
    monkeypatch.setattr(ws.index, "rows", rows)
    result = getattr(search, operation)("gradient descent")
    rows = result[0] if operation == "rank" else result
    assert rows
    assert not any("Revocable evidence" in text for text in sent)
    assert any("Distinct current evidence" in text for text in sent)


@pytest.mark.parametrize("unit", ["line", "page"])
@pytest.mark.parametrize("changed", [False, True])
def test_unchanged_non_line_copies_survive_but_changed_ones_are_omitted(live_workspace, monkeypatch, unit, changed):
    import numpy as np
    from peresearch.zetokrag.index import sha256
    ws, root, main = live_workspace
    copy = root / "copy.pdf"
    # No parsing is under test: these are the already-indexed row contracts for each location unit.
    text = main.read_text().strip()
    from peresearch.zetokrag import parse
    monkeypatch.setattr(parse, "MAX_BYTES", 256)
    copy.write_bytes((text + "\n\n" + text).encode() if unit == "line" else b"a packed document with a text layer " * 20)
    row = (1, str(main), "Research", "line", 1, 3, text, sha256(main), 0)
    twin = (2, str(copy), "Research", unit, 1 if unit == "line" else 2, 3 if unit == "line" else 4, text, sha256(copy), 0)
    search = Searcher(ws.index, reranker=None)
    monkeypatch.setattr(search, "candidates", lambda query: ([row], np.array([np.nan]), {0: [twin]}))
    if changed:
        copy.write_bytes(b"entirely different evidence")
    hits, _, _ = search.find("gradient descent")
    assert hits
    expected = [] if changed else [str(copy) + (":L1-3" if unit == "line" else ":p.2-4")]
    assert hits[0].copies == expected



def test_changed_file_beyond_parser_bound_is_not_materialized_as_evidence(live_workspace, monkeypatch):
    from peresearch.zetokrag import parse
    ws, root, path = live_workspace
    original = path.read_text()
    monkeypatch.setattr(parse, "MAX_BYTES", 256)
    path.write_text(original + "\n" + "large added content " * 100)
    hits, stale, count = Searcher(ws.index, reranker=None).find("gradient descent")
    assert count and stale
    assert not hits



def test_file_growing_during_validation_still_obeys_read_bound(live_workspace, monkeypatch):
    from peresearch.zetokrag import parse, search
    ws, root, path = live_workspace
    original = path.read_text()
    monkeypatch.setattr(parse, "MAX_BYTES", 256)
    actual_hash = search.sha256
    def grow_after_hash(p):
        digest = actual_hash(p)
        p.write_text(original + "\n" + "large added content " * 100)
        return digest
    monkeypatch.setattr(search, "sha256", grow_after_hash)
    hits, stale, count = Searcher(ws.index, reranker=None).find("gradient descent")
    assert count and stale
    assert not hits



def test_text_at_exact_parser_size_bound_remains_searchable(live_workspace, monkeypatch):
    from peresearch.zetokrag import parse
    ws, root, path = live_workspace
    original = path.read_text()
    monkeypatch.setattr(parse, "MAX_BYTES", 256)
    path.write_text(original + "\n" * (256 - len(original.encode())))
    hits, stale, count = Searcher(ws.index, reranker=None).find("gradient descent")
    assert count and hits and not stale
    assert "Revocable evidence" in hits[0].text


@pytest.mark.parametrize("grows_after_hash", [False, True])
def test_all_oversized_file_candidates_are_dropped_without_unbounded_io(live_workspace, monkeypatch, grows_after_hash):
    from contextlib import contextmanager
    import numpy as np
    from peresearch.zetokrag import parse, search
    ws, root, path = live_workspace
    original = path.read_text()
    row = ws.index.rows()[0]
    finder = Searcher(ws.index, reranker=None)
    monkeypatch.setattr(finder, "candidates", lambda query: ([row, row], np.array([np.nan, np.nan]), {}))
    monkeypatch.setattr(parse, "MAX_BYTES", 256)
    def grow():
        path.write_text(original + "\n" + "large added content " * 100)
    actual_hash, actual_open = search.sha256, Path.open
    enlarged = False
    def validate_hash(p):
        nonlocal enlarged
        assert grows_after_hash, "An already oversized file must be rejected without hashing all its bytes."
        digest = actual_hash(p)
        grow()
        enlarged = True
        return digest
    @contextmanager
    def bounded_open(p, *args, **kwargs):
        with actual_open(p, *args, **kwargs) as stream:
            if p == path and enlarged and args and args[0] == "rb":
                class Reader:
                    def read(self, size=-1):
                        assert isinstance(size, int) and 0 < size <= 2 * parse.MAX_BYTES, "Unbounded evidence allocation"
                        return stream.read(size)
                yield Reader()
            else:
                yield stream
    if not grows_after_hash:
        grow()
    monkeypatch.setattr(search, "sha256", validate_hash)
    monkeypatch.setattr(Path, "open", bounded_open)
    hits, stale, count = finder.find("gradient descent")
    assert count == stale == 2 and not hits


@pytest.mark.parametrize("change", ["removed", "protected_symlink"])
def test_cli_find_applies_current_scope_before_external_reranking(live_workspace, monkeypatch, capsys, change):
    from types import SimpleNamespace
    from peresearch import cli
    from peresearch.zetokrag import embed
    ws, root, path = live_workspace
    sent = []
    spy = FakeReranker()
    score = spy.scores
    spy.scores = lambda q, texts: (sent.extend(texts), score(q, texts))[1]
    monkeypatch.setattr(embed, "from_settings", lambda: (None, spy))
    args = SimpleNamespace(query="gradient descent", k=10, page=1)
    if change == "removed":
        ws.remove(str(root))
    else:
        protected = root.parent/".env"
        protected.write_text(path.read_text() + "\nSYNTHETIC_PRIVATE_CONFIGURATION\n")
        path.unlink()
        path.symlink_to(protected)
    cli.cmd_find(args)
    assert "Revocable evidence" not in capsys.readouterr().out
    assert sent == []
    if change == "removed":
        ws.add(str(root))
        cli.cmd_find(args)
        assert "Revocable evidence" in capsys.readouterr().out and sent
