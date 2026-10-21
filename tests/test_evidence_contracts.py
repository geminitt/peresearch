"""Evidence identity and content freshness across turns, resume and retained retrieval."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from peresearch import guard
from peresearch.agent import Agent, read_turns
from peresearch.llm import Reply, ToolCall
from peresearch.tools import Toolbox
from peresearch.zetokrag.index import Index, sha256
from peresearch.zetokrag.search import Searcher
from tests.test_access_isolation import Scripted
from tests.conftest import FakeEmbedder


def reading(path, cid="read", **kwargs):
    return Reply("", [ToolCall(cid, "read", {"path": str(path), **kwargs})])


def test_followup_citations_keep_original_sources_after_other_reads_and_resume(home, tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text("Original alpha evidence."); b.write_text("Different beta evidence.")
    agent = Agent(Scripted(reading(a), Reply("Alpha [N1]."), reading(b), Reply("Beta [N2].")),
                  Toolbox(roots=[tmp_path]), project="source-identity")
    first = agent.ask("Read alpha."); second = agent.ask("Read beta.")
    assert first.sources["N1"].text == a.read_text()
    assert second.sources["N2"].text == b.read_text() and "N1" not in second.sources
    resumed = Agent(Scripted(Reply('"Original alpha evidence." [N1].')), Toolbox(roots=[tmp_path]), project=agent.project)
    resumed.resume(agent.conversation)
    answer = resumed.ask("Quote the original alpha.")
    assert answer.sources["N1"] == first.sources["N1"] and answer.check.ok
    assert answer.sources["N1"].text != second.sources["N2"].text


def test_shorter_fresh_read_gets_a_new_immutable_version(home, tmp_path):
    path = tmp_path / "edited.md";path.write_text("Old evidence is much longer than the new version.")
    model = Scripted(reading(path), Reply("Old [N1]."), reading(path), Reply('"New." [N2].'))
    agent = Agent(model, Toolbox(roots=[tmp_path]), project="shorter-evidence")
    before = agent.ask("Read it.")
    path.write_text("New.")
    after = agent.ask("Read it again.")
    assert after.sources["N2"].text == "New." and after.check.ok
    assert before.sources["N1"].text == "Old evidence is much longer than the new version."
    turns=read_turns(agent.history_path)
    assert turns[0]["source_versions"]["N1"] != turns[1]["source_versions"]["N2"]


def test_retained_source_links_do_not_duplicate_large_result_text(home, tmp_path):
    path=tmp_path/"large.md";text="unique evidence fragment "*4000;path.write_text(text)
    agent=Agent(Scripted(reading(path),Reply("Read [N1].")),Toolbox(roots=[tmp_path]),project="linked-evidence")
    before=agent.ask("Read the whole document.")
    records=[json.loads(s) for s in agent.results().path.read_text().splitlines()]
    assert sum(text in r.get("text","") for r in records)==1
    metadata=[r for r in records if "source_versions" in r]
    assert metadata and all("unique evidence fragment" not in json.dumps(r) for r in metadata)
    resumed=Agent(Scripted(Reply("Use the original [N1].")),Toolbox(roots=[tmp_path]),project=agent.project)
    resumed.resume(agent.conversation);path.write_text("Replacement document.")
    assert resumed.ask("Continue.").sources["N1"]==before.sources["N1"]


def test_legacy_colliding_ids_are_unknown_and_reserved_for_new_reads(home,tmp_path):
    path=tmp_path/"new.md";path.write_text("Current evidence.")
    agent=Agent(Scripted(reading(path),Reply("Current [N2]; old [N1].")),Toolbox(roots=[tmp_path]),project="legacy-ids")
    agent.history_path.parent.mkdir(parents=True)
    turns=[{"conversation":agent.conversation,"question":"Old question","answer":"Old [N1].",
            "sources":{"N1":str(tmp_path/f"old-{i}.md:L1")},"scope_version":2,"scope":[]} for i in range(2)]
    agent.history_path.write_text("".join(json.dumps(t)+"\n" for t in turns))
    answer=agent.ask("Read the current file.")
    assert answer.sources["N2"].text==path.read_text()
    assert answer.sources["N1"].known is False and "N1" in answer.check.unknown_ids
    assert "read" in answer.sources["N1"].text.lower()
    assert "unknown" in answer.sources["N1"].title.lower()
    assert read_turns(agent.history_path)[-1]["source_versions"]["N1"] is None


def test_legacy_answer_ids_without_locations_cannot_alias_new_evidence(home, tmp_path):
    path = tmp_path / "current.md"
    path.write_text("Current evidence.")
    agent = Agent(Scripted(reading(path), Reply("Current [N8]; old [N7, W4].")),
                  Toolbox(roots=[tmp_path]), project="unmapped-legacy-ids")
    agent.history_path.parent.mkdir(parents=True)
    turn = {"conversation": agent.conversation, "question": "Old question",
            "answer": "Old [N7, W4].", "scope_version": 2, "scope": []}
    agent.history_path.write_text(json.dumps(turn) + "\n")
    answer = agent.ask("Read the current file.")
    assert answer.sources["N8"].text == path.read_text()
    assert set(answer.check.unknown_ids) == {"N7", "W4"}
    assert all(answer.sources[i].known is False for i in ("N7", "W4"))
    preview = agent.source_registry().for_turn(turn)
    assert set(preview) == {"N7", "W4"} and all(not s.known for s in preview.values())
    assert all(s.where == "" and s.id == identifier for identifier, s in preview.items())
    assert answer.sources["N7"].kind == "file" and answer.sources["W4"].kind == "web"
    assert answer.sources["N7"].where == answer.sources["W4"].where == ""


def test_large_unverified_citation_ids_do_not_break_resume_or_alias_a_new_read(home, tmp_path):
    import sys
    identifier = "N" + "9" * 5000
    successor = "N1" + "0" * 5000
    limit = sys.get_int_max_str_digits()
    original = Agent(Scripted(Reply(f"Unverified [{identifier}].")), Toolbox(roots=[tmp_path]))
    original.ask("Name the source.")
    path = tmp_path / "current.md"
    path.write_text("Current evidence.")
    resumed = Agent(Scripted(reading(path), Reply(f"Fresh [{successor}]; old [{identifier}].")),
                    Toolbox(roots=[tmp_path]))
    resumed.resume(original.conversation)
    answer = resumed.ask("Read the current file.")
    assert answer.sources[successor].text == path.read_text()
    assert answer.check.unknown_ids == [identifier] and not answer.sources[identifier].known
    assert sys.get_int_max_str_digits() == limit


@pytest.mark.parametrize("record", [False, True])
def test_unverified_live_citation_ids_cannot_alias_later_reads(home, tmp_path, record):
    path = tmp_path / "current.md"
    path.write_text("Current evidence.")
    model = Scripted(Reply("Unverified [N7]."), reading(path), Reply("Fresh [N8]; old [N7]."))
    agent = Agent(model, Toolbox(roots=[tmp_path]), record=record)
    first = agent.ask("Name the source.")
    answer = agent.ask("Read the current file.")
    assert first.check.unknown_ids == ["N7"]
    assert answer.sources["N8"].text == path.read_text()
    assert answer.check.unknown_ids == ["N7"] and not answer.sources["N7"].known


def test_source_counter_boundary_works_with_the_minimum_python_digit_guard():
    import subprocess
    import sys
    script = '''
import sys
from peresearch.agent import MemoryStore
from peresearch.evidence import Sources
identifier = "N" + "9" * 640
registry = Sources.load([{"answer": "Unverified [" + identifier + "]."}], MemoryStore())
source = registry.add("file", "current.md:L1", "current.md", "Current evidence.")
assert source.id == "N1" + "0" * 640
assert not registry.items[identifier].known
assert sys.get_int_max_str_digits() == 640
'''
    result = subprocess.run([sys.executable, "-c", script],
                            env={**os.environ, "PYTHONINTMAXSTRDIGITS": "640"},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("existing_agent", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_resume_preview_restores_exact_versions_or_reports_unknown(home, tmp_path, existing_agent, legacy):
    from types import SimpleNamespace
    from peresearch.tui import Chat
    path = tmp_path / "evidence.md"
    path.write_text("Original evidence.")
    guard.set_roots([tmp_path])
    agent = Agent(Scripted(reading(path), Reply('"Original evidence." [N1].')), Toolbox(roots=[tmp_path]))
    original = agent.ask("Read it.")
    if legacy:
        turn = read_turns(agent.history_path)[0]
        turn.pop("source_versions")
        agent.history_path.write_text(json.dumps(turn) + "\n")
    path.write_text("Replaced evidence.")
    app = Chat(lambda e: Agent(Scripted(), Toolbox(roots=[tmp_path])), index_on_start=False)
    app.hide_menu = lambda: None
    app.query_one = lambda *args: SimpleNamespace(remove_children=lambda: None)
    app.add = lambda widget: None
    app.note = lambda text: None
    if existing_agent:
        app.agent = agent
    app.resume(agent.conversation)
    if legacy:
        assert app.last.sources["N1"].known is False
        assert app.last.check.unknown_ids == ["N1"]
    else:
        assert app.last.sources["N1"] == original.sources["N1"] and app.last.check.ok


@pytest.mark.parametrize("damage", ["digest", "range", "negative-range", "body", "missing-result", "collision", "missing-version",
                                   "unmapped-legacy-collision"])
def test_damaged_retained_evidence_requires_a_read_and_reserves_its_id(home, tmp_path, damage):
    path = tmp_path / "evidence.md"
    path.write_text("Original evidence.")
    agent = Agent(Scripted(reading(path), Reply("Original [N1].")), Toolbox(roots=[tmp_path]))
    agent.ask("Read it.")
    store = agent.results()
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    record = next(row["source_versions"][0] for row in rows if "source_versions" in row)
    if damage == "digest":
        record["version"] = "bad digest"
    elif damage == "range":
        record["parts"][0][1] = 10**9
    elif damage == "negative-range":
        result = next(row for row in rows if row.get("ref") == record["ref"])
        record["parts"][0][0] -= len(result["text"])
    elif damage == "body":
        result = next(row for row in rows if row.get("ref") == record["ref"])
        result["text"] = result["text"].replace("Original evidence.", "Tampered evidence.")
    elif damage == "missing-result":
        rows = [row for row in rows if row.get("ref") != record["ref"]]
    elif damage == "collision":
        rows.append({"source_versions": [{**record, "where": "other.md:L1"}]})
    elif damage == "missing-version":
        turn = read_turns(agent.history_path)[0]
        turn.pop("source_versions")
        agent.history_path.write_text(json.dumps(turn) + "\n")
    else:
        legacy = {"conversation": agent.conversation, "question": "Unrelated legacy question",
                  "answer": "Unrelated original [N1].", "scope_version": 2, "scope": []}
        agent.history_path.write_text(json.dumps(legacy) + "\n" + agent.history_path.read_text())
    store.path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "{torn")
    resumed = Agent(Scripted(reading(path), Reply("Fresh [N2]; old [N1].")), Toolbox(roots=[tmp_path]))
    resumed.resume(agent.conversation)
    answer = resumed.ask("Read again.")
    assert answer.sources["N2"].text == path.read_text()
    assert not answer.sources["N1"].known and answer.check.unknown_ids == ["N1"]
    assert answer.sources["N1"].id == "N1" and answer.sources["N1"].kind == "file"


def test_retained_evidence_preview_withholds_revoked_content(home, tmp_path):
    path = tmp_path / "evidence.md"
    path.write_text("Private original evidence.")
    guard.set_roots([tmp_path])
    agent = Agent(Scripted(reading(path), Reply("Original [N1]."), Reply("Old [N1].")), Toolbox())
    original = agent.ask("Read it.")
    guard.set_roots([])
    answer = agent.ask("Continue.")
    assert not answer.sources["N1"].known and answer.check.unknown_ids == ["N1"]
    assert "Private original evidence" not in answer.sources["N1"].text
    assert answer.sources["N1"].id == "N1" and answer.sources["N1"].where == original.sources["N1"].where
    assert original.sources["N1"].text == path.read_text()


def test_revocation_after_resume_keeps_the_source_link_policy_active(home, tmp_path):
    path = tmp_path / "evidence.md"
    path.write_text("Private original evidence.")
    guard.set_roots([tmp_path])
    agent = Agent(Scripted(reading(path), Reply("Original [N1].")), Toolbox())
    original = agent.ask("Read it.")
    resumed = Agent(Scripted(Reply("Original [N1]."), Reply("Old [N1].")), Toolbox())
    resumed.resume(agent.conversation)
    assert resumed.ask("Continue.").sources["N1"] == original.sources["N1"]
    guard.set_roots([])
    answer = resumed.ask("Continue again.")
    assert not answer.sources["N1"].known and answer.check.unknown_ids == ["N1"]
    assert answer.sources["N1"].id == "N1" and answer.sources["N1"].where == original.sources["N1"].where
    assert "Private original evidence" not in answer.sources["N1"].text


@pytest.mark.parametrize("existing_agent", [False, True])
def test_resume_withholds_revoked_excerpt_even_in_the_active_conversation(home, tmp_path, existing_agent):
    from types import SimpleNamespace
    from peresearch.tui import Chat
    path = tmp_path / "evidence.md"
    path.write_text("Private original evidence.")
    guard.set_roots([tmp_path])
    agent = Agent(Scripted(reading(path), Reply("Original [N1].")), Toolbox())
    original = agent.ask("Read it.")
    guard.set_roots([])
    app = Chat(lambda e: Agent(Scripted(), Toolbox()), index_on_start=False)
    app.hide_menu = lambda: None
    app.query_one = lambda *args: SimpleNamespace(remove_children=lambda: None)
    app.add = lambda widget: None
    app.note = lambda text: None
    if existing_agent:
        app.agent = agent
    app.resume(agent.conversation)
    assert not app.last.sources["N1"].known and app.last.check.unknown_ids == ["N1"]
    assert "Private original evidence" not in app.last.sources["N1"].text
    assert original.sources["N1"].text == path.read_text()


def test_resume_preserves_the_existing_agents_session_granted_evidence(home, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from peresearch.tui import Chat
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "outside.md"
    path.write_text("Session-granted evidence.")
    agent = Agent(Scripted(reading(path), Reply("Original [N1].")),
                  Toolbox(roots=[], ask=lambda tool, path: "session"))
    original = agent.ask("Read it.")
    app = Chat(lambda e: agent, index_on_start=False)
    app.agent = agent
    app.hide_menu = lambda: None
    app.query_one = lambda *args: SimpleNamespace(remove_children=lambda: None)
    app.add = lambda widget: None
    app.note = lambda text: None
    app.resume(agent.conversation)
    assert app.last.sources["N1"] == original.sources["N1"] and app.last.check.ok


def test_new_ephemeral_conversation_starts_with_fresh_evidence_and_results(home, tmp_path):
    path = tmp_path / "evidence.md"
    path.write_text("Original evidence.")
    agent = Agent(Scripted(reading(path), Reply("Original [N1]."), reading(path), Reply("New [N1].")),
                  Toolbox(roots=[tmp_path]), record=False)
    original = agent.ask("Read it.")
    agent.new_session()
    path.write_text("New evidence.")
    answer = agent.ask("Read again.")
    assert answer.sources["N1"].text == path.read_text()
    assert original.sources["N1"].text == "Original evidence."
    assert len(agent.results()._lines()) == 1
    assert not agent.history_path.exists()


def test_whitespace_transformed_results_restore_the_exact_immutable_excerpt():
    from dataclasses import FrozenInstanceError
    from peresearch.agent import MemoryStore
    from peresearch.evidence import Sources
    registry, store = Sources(), MemoryStore()
    source = registry.add("web", "https://example.org/evidence", "Research", "Alpha\n\n  beta\tgamma.")
    ref = store.put("web_search", {}, "[W1] Alpha beta gamma.")
    registry.retain(ref, store.get(ref)["text"], store)
    assert registry.add("web", source.where, source.title, source.text) is source
    with pytest.raises(FrozenInstanceError):
        source.text = "Replacement."
    restored = Sources.load([], store)
    assert restored.items["W1"] == source
    changed = restored.add("web", source.where, "Other title", source.text)
    assert changed.id == "W2" and changed.version != source.version
    restored.begin_question()
    assert not restored.urls and not restored.current
    assert restored.items["W1"] == source


@pytest.mark.parametrize("text,result", [("Alpha\n beta", "Alpha beta"),
                                         ("Alpha\n beta Alpha\n gamma.", "Alpha beta Alpha gamma."), ("", "")])
def test_result_links_preserve_excerpts_at_the_start_and_repeated_words(text, result):
    from peresearch.agent import MemoryStore
    from peresearch.evidence import Sources
    registry, store = Sources(), MemoryStore()
    source = registry.add("web", "https://example.org/evidence", "Evidence", text)
    ref = store.put("web_search", {}, result)
    registry.retain(ref, result, store)
    assert Sources.load([], store).items["W1"] == source


def test_misordered_or_missing_result_text_cannot_restore_an_excerpt():
    from peresearch.agent import MemoryStore
    from peresearch.evidence import Sources
    for result in ("beta Alpha", "Alpha only"):
        registry, store = Sources(), MemoryStore()
        source = registry.add("web", "https://example.org/evidence", "Evidence", "Alpha\n beta")
        ref = store.put("web_search", {}, result)
        registry.retain(ref, result, store)
        turn = {"answer": "Earlier [W1].", "sources": {"W1": source.where},
                "source_versions": {"W1": source.version}}
        assert not Sources.load([turn], store).items["W1"].known


def test_torn_utf8_and_malformed_records_do_not_hide_later_version_metadata(home):
    from peresearch.agent import ResultStore
    from peresearch.evidence import Sources
    store, registry = ResultStore("version-lines", "conversation"), Sources()
    source = registry.add("web", "https://example.org/evidence", "Evidence", "Original evidence.")
    ref = store.put("web_search", {}, source.text)
    with store.path.open("ab") as stream:
        stream.write(b'\xff\n{broken\n[]\n{"source_versions":null}\n')
    registry.retain(ref, source.text, store)
    assert Sources.load([], store).items["W1"] == source


def test_legacy_location_only_ids_are_reserved_even_without_citations():
    from peresearch.agent import MemoryStore
    from peresearch.evidence import Sources
    turn = {"answer": "An uncited answer.", "sources": {"N7": "original.md:L1"}}
    restored = Sources.load([turn], MemoryStore())
    assert restored.add("file", "fresh.md:L1", "Fresh", "Fresh evidence.").id == "N8"
    unknown = restored.for_turn(turn)["N7"]
    assert unknown.id == "N7" and unknown.where == "original.md:L1" and not unknown.known
    selected = restored.selected("Earlier [N7].", MemoryStore())
    assert selected["N7"].where == "original.md:L1" and not selected["N7"].known


def test_unregistered_turn_evidence_still_has_an_explicit_unknown_preview():
    from peresearch.evidence import Sources
    preview = Sources().for_turn({"answer": "Earlier [N7].", "sources": {"N7": "original.md:L1"}})
    assert preview["N7"].id == "N7" and preview["N7"].where == "original.md:L1"
    assert not preview["N7"].known


def test_retained_web_versions_survive_resume_and_new_search_requirements(home, tmp_path):
    from tests.helpers import agent as make_agent
    root = tmp_path / "notes"
    root.mkdir()
    (root / "notes.md").write_text("BPE merges pairs.")
    calls = [Reply("", [ToolCall("s", "web_search", {"query": "BPE"})]),
             Reply("", [ToolCall("f", "fetch", {"url": "https://example.org/bpe"})]), Reply("Page [W2].")]
    agent, _ = make_agent(home, root, *calls)
    original = agent.ask("Find its history.")
    assert original.sources["W2"].title == original.sources["W1"].title
    resumed, model = make_agent(home, root, Reply('"neural machine translation in 2016" [W2].'))
    resumed.resume(agent.conversation)
    answer = resumed.ask("Quote the fetched page.")
    assert answer.sources["W2"] == original.sources["W2"] and answer.check.ok
    assert resumed.toolbox.call("fetch", {"url": "https://example.org/bpe"}).startswith("refused:")


def test_recorded_unicode_excerpt_versions_remain_readable():
    from peresearch.agent import MemoryStore
    from peresearch.evidence import Source, Sources
    fixture = json.loads((Path(__file__).parent / "data/evidence/retained-unicode.json").read_text())
    store = MemoryStore()
    store.items = fixture["results"]
    store.set_sources(fixture["source_versions"])
    restored = Sources.load([], store, lambda path: False)
    assert restored.items["W1"] == Source(**fixture["expected"])
    turn = {"answer": "Earlier [W1].", "sources": {"W1": fixture["expected"]["where"]}}
    preview = restored.for_turn(turn)
    assert not preview["W1"].known and preview["W1"].id == "W1"
    assert preview["W1"].where == turn["sources"]["W1"]
    turn["source_versions"] = {"W1": restored.items["W1"].version}
    assert restored.for_turn(turn)["W1"] == restored.items["W1"]
    turn["sources"]["W1"] = "https://example.org/other"
    assert not restored.for_turn(turn)["W1"].known


@pytest.mark.parametrize("metadata", [None, [], "invalid", {"id": 1}, {"id": "N7", "kind": "web"}])
def test_bad_version_metadata_does_not_hide_later_valid_results(metadata):
    from peresearch.agent import MemoryStore
    from peresearch.evidence import Sources
    registry, store = Sources(), MemoryStore()
    source = registry.add("web", "https://example.org/evidence", "Evidence", "Original evidence.")
    ref = store.put("web_search", {}, source.text)
    store.set_sources([metadata])
    registry.retain(ref, source.text, store)
    restored = Sources.load([], store)
    assert restored.items["W1"] == source


@pytest.mark.parametrize("conflict", [False, True])
def test_duplicate_version_records_only_conflict_when_their_identities_differ(conflict):
    from peresearch.agent import MemoryStore
    from peresearch.evidence import Source, Sources, link
    registry, store = Sources(), MemoryStore()
    source = registry.add("web", "https://example.org/evidence", "Evidence", "Original evidence.")
    ref = store.put("web_search", {}, "Original evidence. Changed evidence.")
    registry.retain(ref, store.get(ref)["text"], store)
    duplicate = Source("W1", "web", source.where, source.title, "Changed evidence.") if conflict else source
    store.set_sources([link(duplicate, ref, store.get(ref)["text"])])
    restored = Sources.load([], store)
    assert restored.items["W1"].known is not conflict
    if not conflict:
        assert restored.items["W1"] == source
    assert restored.add("web", source.where, source.title, "Fresh evidence.").id == "W2"


def test_unknown_citations_accumulate_across_separate_positions():
    from peresearch.agent import check
    from peresearch.evidence import unknown
    sources = {identifier: unknown(identifier) for identifier in ("N1", "W2")}
    assert check("First [N1]. Later [W2]. Repeat [N1].", sources).unknown_ids == ["N1", "W2"]


@pytest.mark.parametrize("damage", ["title", "where", "kind", "boolean-range", "extra-range", "scope"])
def test_matching_hash_cannot_make_malformed_retained_metadata_valid(damage):
    from peresearch.agent import MemoryStore
    from peresearch.evidence import Source, Sources
    store, registry = MemoryStore(), Sources()
    source = registry.add("file", "evidence.md:L1", "Evidence", "Original evidence.")
    ref = store.put("read", {}, "x" + source.text)
    store.set_scope(ref, ["evidence.md"])
    registry.retain(ref, store.get(ref)["text"], store)
    record = store.source_versions()[0]
    if damage in ("where", "title", "kind"):
        record[damage] = "web" if damage == "kind" else None
        record["version"] = Source(source.id, record["kind"], record["where"], record["title"], source.text).version
    elif damage == "boolean-range":
        record["parts"][0][0] = True  # True slices like 1 but is invalid persisted range metadata
    elif damage == "extra-range":
        record["parts"][0].append(3)
    else:
        store.get(ref)["scope"] = [7]
    restored = Sources.load([], store, lambda p: p == "evidence.md")
    assert not restored.items["N1"].known and restored.items["N1"].where == ""


def test_injected_file_evidence_without_a_policy_can_restore_without_scope_metadata():
    from peresearch.agent import MemoryStore
    from peresearch.evidence import Sources
    store, registry = MemoryStore(), Sources()
    source = registry.add("file", "evidence.md:L1", "Evidence", "Original evidence.")
    ref = store.put("read", {}, source.text)
    registry.retain(ref, source.text, store)
    assert Sources.load([], store).items["N1"] == source
    assert not Sources.load([], store, lambda p: True).items["N1"].known


def test_nonfinite_query_similarity_uses_lexical_evidence_with_a_reason(home, tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Evidence\n\nGradient descent adjusts weights.\n")
    embedder = FakeEmbedder()
    index = Index(home, embedder=embedder)
    index.update([tmp_path])
    embedder.queries = lambda texts: np.full((len(texts), embedder.dim), np.nan)
    search = Searcher(index, reranker=None)
    hits, _, _ = search.find("gradient descent", k=None)
    assert hits and hits[0].path == str(path) and search.words_only


def test_equal_size_and_mtime_notebook_edit_refreshes_grep_and_index(home,tmp_path):
    notes=tmp_path/"notes";notes.mkdir();path=notes/"notes.ipynb"
    def notebook(word):
        return json.dumps({"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[
            {"id":"evidence","cell_type":"markdown","metadata":{},"source":"# Evidence\n\n"+word+" evidence."}]})
    path.write_text(notebook("older"));guard.set_roots([notes])
    index=Index(home,embedder=None);index.update([notes])
    stamp=path.stat();old_digest=sha256(path)
    path.write_text(notebook("fresh"));os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert path.stat().st_size==stamp.st_size and sha256(path)!=old_digest
    tool=Toolbox(Searcher(index,reranker=None),roots=[notes])
    assert "fresh evidence" in tool.grep("fresh") and tool.grep("older")=="no match"
    report=index.update([notes])
    assert report.changed==1 and report.unchanged==0
    assert index.read("SELECT sha FROM files WHERE path=?",(str(path),))[0][0]==sha256(path)
    assert "fresh evidence" in tool.search_notes("fresh evidence")


def test_permitted_subset_without_any_vectors_uses_lexical_evidence(home,tmp_path):
    allowed,removed=tmp_path/"allowed",tmp_path/"removed";allowed.mkdir();removed.mkdir()
    for root,word in ((allowed,"permitted"),(removed,"revoked")):
        (root/"notes.md").write_text(f"# Research\n\n{word} gradient descent evidence.\n")
    index=Index(home,embedder=FakeEmbedder());index.update([allowed,removed])
    index.db.execute("DELETE FROM vectors WHERE id IN (SELECT id FROM chunks WHERE path=?)",(str(allowed/"notes.md"),))
    search=Searcher(index,reranker=None);search.allowed=lambda p:Path(p).is_relative_to(allowed)
    hits,_,_=search.find("gradient descent",k=None)
    assert hits and all(h.path==str(allowed/"notes.md") for h in hits)
    assert "vector" in search.words_only and "permitted" in search.words_only
