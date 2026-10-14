"""Replay keeps valid functionality as well as withholding unauthorized evidence."""
import json
from pathlib import Path

import pytest

from peresearch import guard
from peresearch.agent import Agent, ResultStore, REVOKED_RESULT, EARLIER, SUMMARIZE
from peresearch.llm import Interrupted, Reply, ToolCall
from peresearch.tools import Toolbox
from tests.test_access_isolation import Scripted
from peresearch.messages import Message
from tests.helpers import request


def test_scope_metadata_ignores_incomplete_updates_without_losing_valid_result(home, tmp_path):
    store = ResultStore("metadata-contract", "conversation")
    ref = store.put("read", {"path": str(tmp_path / "note.md")}, "kept text")
    store.set_scope(ref, [str(tmp_path / "note.md")])
    with store.path.open("a") as stream:
        for bad in ({"scope_ref": ref}, {"paths": []}, {"scope_ref": 7, "paths": []},
                    {"scope_ref": ref, "paths": None}, ["not a record"]):
            stream.write(json.dumps(bad) + "\n")
    assert store.get(ref)["text"] == "kept text"
    assert store.get(ref)["scope"] == [str(tmp_path / "note.md")]


@pytest.mark.parametrize("location", ["plain.md:L1", "notes:2024.md:L1-3", "notes:2024.md:p2", "notes:2024.md:c4",
                                     "deck.pdf:p.2-3", "notebook.ipynb:cell 4-5"])
def test_legacy_file_location_replays_with_a_file_only_grant(home, tmp_path, location):
    name = location.rsplit(":", 1)[0]
    path = tmp_path / name
    path.write_text("file evidence")
    toolbox = Toolbox(roots=[])
    toolbox.granted = [path]
    model = Scripted(Reply("New answer."))
    agent = Agent(model, toolbox, project="legacy-file-grants")
    agent.history_path.parent.mkdir(parents=True)
    agent.history_path.write_text(json.dumps({"conversation": agent.conversation, "question": "Read it",
        "answer": "PERMITTED-LEGACY-FILE [N1]", "sources": {"N1": str(tmp_path / location)}}) + "\n")
    agent.ask("Continue.")
    assert "PERMITTED-LEGACY-FILE" in json.dumps(model.seen[-1])


@pytest.mark.parametrize("sources", [{"N1": "notes (listing)"}, {"N1": "note.md"},
                                     {"N1": None}, ["broken metadata"]])
def test_legacy_unknown_scope_withholds_history_and_dependent_turns(home, tmp_path, sources):
    if isinstance(sources, dict):
        sources = {key: str(tmp_path / value) if isinstance(value, str) else value
                   for key, value in sources.items()}
    model = Scripted(Reply("New answer."))
    agent = Agent(model, Toolbox(roots=[tmp_path]), project="unknown-history-scope")
    agent.history_path.parent.mkdir(parents=True)
    records = [{"conversation": agent.conversation, "question": "Read permitted file",
                "answer": "PERMITTED-INDEPENDENT-EVIDENCE", "sources": {"N2": str(tmp_path / "valid.md:L1")}},
               {"conversation": agent.conversation, "question": "Inspect files",
                "answer": "UNKNOWN-LOCAL-EVIDENCE", "sources": sources},
               {"conversation": agent.conversation, "question": "Repeat that evidence",
                "answer": "DEPENDENT-UNKNOWN-EVIDENCE", "sources": {}}]
    original = "".join(json.dumps(turn) + "\n" for turn in records)
    agent.history_path.write_text(original)
    agent.ask("Continue.")
    sent = json.dumps(model.seen[-1])
    assert "UNKNOWN-LOCAL-EVIDENCE" not in sent
    assert "DEPENDENT-UNKNOWN-EVIDENCE" not in sent
    assert "PERMITTED-INDEPENDENT-EVIDENCE" in sent
    assert sent.count(REVOKED_RESULT) == 2
    assert agent.history_path.read_text().startswith(original)


def test_legacy_invalid_explicit_scope_is_withheld_without_breaking_followup(home, tmp_path):
    agent = Agent(Scripted(Reply("New answer.")), Toolbox(roots=[tmp_path]), project="invalid-explicit-scope")
    agent.history_path.parent.mkdir(parents=True)
    agent.history_path.write_text(json.dumps({"conversation": agent.conversation, "question": "Inspect",
        "answer": "INVALID-SCOPE-EVIDENCE", "scope": [{}]}) + "\n")
    assert agent.history()[-1].text == REVOKED_RESULT
    agent.ask("Continue.")
    assert "INVALID-SCOPE-EVIDENCE" not in json.dumps(agent.llm.seen[-1])


def test_legacy_web_sources_do_not_require_local_file_permissions(home):
    model = Scripted(Reply("New answer."))
    agent = Agent(model, Toolbox(), project="legacy-web")
    agent.history_path.parent.mkdir(parents=True)
    agent.history_path.write_text(json.dumps({"conversation": agent.conversation, "question": "Look it up",
        "answer": "PERMITTED-WEB-EVIDENCE [W1]", "sources": {"W1": "https://example.org/evidence"}}) + "\n")
    agent.ask("Continue.")
    assert "PERMITTED-WEB-EVIDENCE" in json.dumps(model.seen[-1])


@pytest.mark.parametrize("cancelled,external", [(False, True), (True, False)])
def test_replay_stop_wrapper_preserves_both_cancellation_channels(home, cancelled, external):
    class Respecting(Scripted):
        def answer(self, messages, tools=None, **kwargs):
            if kwargs["stop"]():
                raise Interrupted("stopped")
            return Reply("Should not complete.")
    agent = Agent(Respecting(), Toolbox(), record=False)
    agent.cancelled, agent._prompt = cancelled, "rules"
    with pytest.raises(Interrupted):
        agent._chat([Message.user("question")], None, stop=lambda: external)


def test_repeated_revoked_recalls_stop_after_three_refused_rounds(home, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "note.md"
    path.write_text("private text")
    guard.set_roots([notes])
    class Repeating(Scripted):
        def answer(self, messages, tools=None, **kwargs):
            self.seen.append(json.loads(json.dumps(messages)))
            if tools is None:
                return Reply("Cannot use revoked evidence.")
            return Reply("", [ToolCall(f"recall-{len(self.seen)}", "recall", {"ref": "R1", "from": len(self.seen)})])
    model = Repeating()
    agent = Agent(model, Toolbox(), record=True)
    ref = agent.results().put("read", {"path": str(path)}, "private text long enough")
    agent.results().set_scope(ref, [str(path)])
    guard.set_roots([])
    answer = agent.ask("Recall it.")
    assert len(model.seen) == 4 and answer.steps == 4
    assert answer.stopped == "refused"
    contents = [message["content"] for request in model.seen for message in request if message["role"] == "tool"]
    assert contents and all(text.startswith('<tool_output tool="recall" trust="untrusted data">\nrefused:')
                            and text.endswith("\n</tool_output>") for text in contents)
    assert "private text" not in json.dumps(model.seen)


def retained_turns(agent, paths, answers):
    agent.history_path.parent.mkdir(parents=True, exist_ok=True)
    agent.history_path.write_text("".join(json.dumps({"conversation": agent.conversation,
        "question": "Read the evidence.", "answer": answer, "sources": {}, "scope_version": 2,
        "scope": [str(path)]}) + "\n"
        for path, answer in zip(paths, answers)))


def fold_messages(agent):
    """The messages of the next question; the system prompt, sent before them, is agent._prompt."""
    agent._base = agent._prompt = "base rules"
    past = agent.history()
    agent._past = len(past) // 2
    return [*past, Message.user("Continue.")]


def test_direct_history_retains_permitted_turns_and_withholds_revoked_answers(home, tmp_path):
    paths = [tmp_path / "allowed.md", tmp_path / "revoked.md"]
    agent = Agent(Scripted(), Toolbox(roots=paths))
    retained_turns(agent, paths, ["PERMITTED [N1]", "PRIVATE [N2]"])
    agent.toolbox.roots = [paths[0]]
    assert [m.text for m in agent.history() if m.role == "assistant"] == ["PERMITTED [N1]", REVOKED_RESULT]
    assert "PRIVATE" in agent.history_path.read_text()


@pytest.mark.parametrize("retained", ["summary", "turn", "tool"])
def test_access_filter_reports_changes_and_preserves_useful_context(home, tmp_path, retained):
    path = tmp_path / "note.md"
    toolbox = Toolbox(roots=[path])
    agent = Agent(Scripted(), toolbox)
    retained_turns(agent, [path], ["PRIVATE"])
    messages = fold_messages(agent)
    if retained == "summary":
        agent._summary, agent._done, agent._past = "PRIVATE SUMMARY", 1, 0
        messages = [messages[-1]]
        agent._prompt += EARLIER.format(summary=agent._summary)
    elif retained == "tool":
        agent._past = 0
        agent._kept["read"] = ("R1", "read", {}, 7)
        agent._scopes["R1"] = ("read", [str(path)])
        messages = [messages[-1], Message.tool("read", "read", "PRIVATE")]
    assert agent._filter_access(messages) is False
    toolbox.roots = []
    assert agent._filter_access(messages) is True
    assert "PRIVATE" not in agent._prompt + "".join(m.text for m in messages)     # nothing sent holds it
    assert agent._prompt == "base rules"
    if retained == "summary":
        assert agent._summary == ""
        assert agent._fold(messages, 100) is False
    else:
        assert REVOKED_RESULT in [m.text for m in messages]
    assert agent._filter_access(messages) is False


def test_stop_scope_ignores_denied_results_and_keeps_allowed_history(home, tmp_path):
    allowed, denied = tmp_path / "allowed.md", tmp_path / "denied.md"
    agent = Agent(Scripted(), Toolbox(roots=[allowed]))
    retained_turns(agent, [allowed], ["PERMITTED"])
    agent._scopes = {"R1": ("read", [str(denied)])}
    stop = agent._stop_with_access(None)
    assert stop.access_paths == {str(allowed)} and not stop()
    agent.toolbox.roots = []
    assert stop()


@pytest.mark.parametrize("explicit", [False, True])
def test_search_scope_uses_hit_path_or_legacy_location(home, tmp_path, explicit):
    from tests.test_tools import FakeSearcher, hit
    path = tmp_path / "notes:2024 with spaces.md"
    passage = hit(str(path) + ":L1-3", "PERMITTED LEGACY SEARCH")
    if explicit:
        passage.path = str(tmp_path / "authoritative.md")
    tb = Toolbox(FakeSearcher([passage]), roots=[tmp_path])
    assert "PERMITTED LEGACY SEARCH" in tb.search_notes("research")
    assert tb.last_scope == {str(Path(passage.path).resolve() if explicit else path.resolve())}


def test_scope_stop_supports_a_toolbox_without_a_permission_policy(home):
    from types import SimpleNamespace
    agent = Agent(Scripted(), SimpleNamespace(), record=False)
    agent._scopes = {"R1": ("read", ["unscoped-injected-tool.md"])}
    stop = agent._stop_with_access(None)
    assert not stop() and stop.access_paths == set()


def test_injected_toolbox_without_policy_keeps_history_and_results_usable(home, tmp_path):
    from tests.test_agent import FakeToolbox
    agent = Agent(Scripted(Reply("", [ToolCall("tree", "tree", {})]), Reply("Allowed answer")), FakeToolbox())
    retained_turns(agent, [tmp_path / "injected.md"], ["PERMITTED HISTORY"])
    assert agent.history()[1].text == "PERMITTED HISTORY"
    assert not agent._stop_with_access(None)()
    agent.ask("Continue.")
    assert agent._recall("R1", "recall")[0] == "R1"


@pytest.mark.parametrize("scoped", [False, True])
def test_gateway_accepts_plain_stop_and_explains_scoped_preflight(home, scoped):
    from peresearch.llm import LLM
    from tests.test_llm import client_of
    from tests.helpers import streamed
    sent = []
    model = LLM(client=client_of(lambda **kw: sent.append(kw) or streamed(content="Permitted")), model="test")
    stop = lambda: scoped
    if scoped:
        stop.access_paths = {"revoked.md"}
        with pytest.raises(Interrupted, match="^stopped before the model request$"):
            model.chat(request([{"role": "user", "content": "Question"}]), stop=stop)
        assert not sent
    else:
        assert model.chat(request([{"role": "user", "content": "Question"}]), stop=stop).text == "Permitted"


def test_revoked_summary_blocks_further_folds_but_preserves_remaining_allowed_turn(home, tmp_path):
    class Folding(Scripted):
        def context_length(self):
            return 1024
        def answer(self, *args, **kwargs):
            pytest.fail("a revoked summary cannot be extended")
    denied, permitted = tmp_path / "denied.md", tmp_path / "permitted.md"
    agent = Agent(Folding(), Toolbox(roots=[permitted]))
    retained_turns(agent, [denied, permitted], ["PRIVATE", "PERMITTED " * 1000])
    agent._base, agent._summary, agent._done, agent._past = "base rules", "PRIVATE SUMMARY", 1, 1
    agent._prompt = "PRIVATE SUMMARY"
    messages = [Message.user("Read"), Message.assistant("PERMITTED " * 1000), Message.user("Continue")]
    assert agent._fold(messages, 100) is False
    assert agent._prompt == "base rules" and messages[1].text.startswith("PERMITTED")


def test_history_that_fits_keeps_verbatim_answers_without_summary_requests(home):
    class Roomy(Scripted):
        def context_length(self):
            return 16384
        def answer(self, messages, tools=None, **kwargs):
            assert messages[0]["content"] != SUMMARIZE
            return super().answer(messages, tools, **kwargs)
    model = Roomy(Reply("PERMITTED VERBATIM ANSWER"), Reply("Follow-up answer"))
    agent = Agent(model, Toolbox())
    agent.ask("First question")
    agent.ask("Follow-up question")
    assert len(model.seen) == 2
    assert any(m["content"] == "PERMITTED VERBATIM ANSWER" for m in model.seen[-1])
    assert '"summary"' not in agent.history_path.read_text()


def test_refused_recall_has_no_retained_ref_or_continuation_offset(home):
    agent = Agent(Scripted(), Toolbox())
    ref = agent.results().put("read", {}, "UNKNOWN-PRIVATE-EVIDENCE")
    assert agent._recall(ref, "recall", 3) == (None, REVOKED_RESULT, 0)


@pytest.mark.parametrize("summary,resume", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("legacy", [False, True])
def test_follow_up_answers_inherit_replayed_file_scope(home, tmp_path, summary, resume, legacy):
    path = tmp_path / "note.md"
    path.write_text("TRANSITIVE-PRIVATE-EVIDENCE")
    guard.set_roots([tmp_path])
    model = Scripted(Reply("", [ToolCall("read", "read", {"path": str(path)})]),
                     Reply("TRANSITIVE-PRIVATE-EVIDENCE [N1]"),
                     Reply("Follow-up: TRANSITIVE-PRIVATE-EVIDENCE"), Reply("No evidence"))
    agent = Agent(model, Toolbox(), project="transitive-replay")
    agent.ask("Read the evidence.")
    if summary:
        with agent.history_path.open("a") as stream:
            stream.write(json.dumps({"conversation": agent.conversation, "summary": "TRANSITIVE-PRIVATE-EVIDENCE",
                                     "turns": 1}) + "\n")
    agent.ask("Repeat that information.")
    if legacy:
        lines = [json.loads(line) for line in agent.history_path.read_text().splitlines()]
        for line in lines:
            line.pop("scope_version", None)
            if line.get("question") == "Repeat that information.":
                line["scope"] = []
        agent.history_path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert agent._turns()[1]["scope"] == [str(path)]
    guard.set_roots([])
    if resume:
        old = agent.conversation
        model = Scripted(Reply("No evidence"))
        agent = Agent(model, Toolbox(), project="transitive-replay")
        agent.resume(old)
    offset = len(model.seen)
    agent.ask("Continue.")
    assert "TRANSITIVE-PRIVATE-EVIDENCE" not in json.dumps(model.seen[offset:])
    assert "Follow-up: TRANSITIVE-PRIVATE-EVIDENCE" in agent.history_path.read_text()


def test_versioned_turn_with_missing_scope_requires_a_new_read(home):
    agent = Agent(Scripted(), Toolbox())
    agent.history_path.parent.mkdir(parents=True)
    agent.history_path.write_text(json.dumps({"conversation": agent.conversation, "question": "Read",
        "answer": "UNKNOWN-PRIVATE-EVIDENCE", "sources": {}, "scope_version": 2, "scope": None}) + "\n")
    assert agent.history()[1].text == REVOKED_RESULT


def test_fresh_allowed_history_does_not_inherit_already_withheld_dependencies(home, tmp_path):
    old, fresh = tmp_path / "old.md", tmp_path / "fresh.md"
    old.write_text("OLD-PRIVATE-EVIDENCE")
    fresh.write_text("FRESH-ALLOWED-EVIDENCE")
    guard.set_roots([old])
    model = Scripted(Reply("", [ToolCall("old", "read", {"path": str(old)})]), Reply("OLD-PRIVATE-EVIDENCE [N1]"),
                     Reply("", [ToolCall("fresh", "read", {"path": str(fresh)})]), Reply("FRESH-ALLOWED-EVIDENCE [N1]"),
                     Reply("Continue with allowed evidence"))
    agent = Agent(model, Toolbox())
    agent.ask("Read the old file.")
    guard.set_roots([fresh])
    agent.ask("Read the fresh file.")
    assert agent._turns()[1]["scope"] == [str(fresh)]
    agent.ask("Continue.")
    assert "FRESH-ALLOWED-EVIDENCE" in json.dumps(model.seen[-1])
    assert "OLD-PRIVATE-EVIDENCE" not in json.dumps(model.seen[-1])


def test_turn_provenance_keeps_accessed_files_when_their_result_is_withheld(home, tmp_path):
    path = tmp_path / "note.md"
    path.write_text("REVOKED-BEFORE-REQUEST")
    guard.set_roots([path])
    model = Scripted(Reply("", [ToolCall("read", "read", {"path": str(path)})]), Reply("No accessible evidence"))
    agent = Agent(model, Toolbox(), on_event=lambda kind, detail: guard.set_roots([]) if kind == "result" else None)
    agent.ask("Read the evidence.")
    assert "REVOKED-BEFORE-REQUEST" not in json.dumps(model.seen[-1])
    assert agent._turns()[0]["scope"] == [str(path)]
    assert agent.results().get("R1")["scope"] == [str(path)]


def test_empty_legacy_summary_does_not_discard_its_recorded_count(home, tmp_path):
    agent = Agent(Scripted(), Toolbox(roots=[]))
    retained_turns(agent, [tmp_path / "denied.md"], ["PRIVATE"])
    with agent.history_path.open("a") as stream:
        stream.write(json.dumps({"conversation": agent.conversation, "summary": "", "turns": 1}) + "\n")
    assert agent.earlier() == ("", 1)


def test_unrecorded_agent_does_not_read_colliding_disk_summary(home):
    agent = Agent(Scripted(), Toolbox(), record=False)
    agent.history_path.parent.mkdir(parents=True)
    agent.history_path.write_text(json.dumps({"conversation": agent.conversation, "summary": "OTHER-SESSION", "turns": 1}) + "\n")
    assert agent.earlier() == ("", 0)
    assert agent.history() == []
    assert all("summary" not in name for name, _ in agent.context_report()["parts"])


@pytest.mark.parametrize("channel", ["agent", "caller"])
def test_summary_requests_preserve_cancellation_channels(home, tmp_path, channel):
    class Folding(Scripted):
        def context_length(self):
            return 1024
        def answer(self, messages, tools, **kwargs):
            assert messages[0]["content"] == SUMMARIZE and tools is None
            if channel == "agent":
                agent.cancel()
            else:
                stopped[0] = True
            if kwargs["stop"]():
                raise Interrupted("stopped")
            return Reply("Unwanted summary")
    stopped = [False]
    agent = Agent(Folding(), Toolbox(roots=[tmp_path]))
    retained_turns(agent, [tmp_path / "note.md"], ["Evidence " * 3000])
    messages = fold_messages(agent)
    with pytest.raises(Interrupted):
        agent._chat(messages, None, stop=lambda: stopped[0])
    assert '"summary"' not in agent.history_path.read_text()


@pytest.mark.parametrize("retry", [False, True])
def test_multi_part_fold_abandons_serialized_evidence_after_revocation(home, tmp_path, retry):
    class Folding(Scripted):
        def context_length(self):
            return 1024
        def answer(self, messages, tools, **kwargs):
            assert tools is None and kwargs["stop"] is not None and kwargs["thinking"] is False
            self.seen.append(json.loads(json.dumps(messages)))
            guard.set_roots([])
            if retry:
                import httpx
                from openai import BadRequestError
                raise BadRequestError("maximum context length is 1024 tokens", body=None,
                    response=httpx.Response(400, request=httpx.Request("POST", "https://example.org/v1/chat/completions")))
            return Reply("PRIVATE SUMMARY")
    guard.set_roots([tmp_path])
    model = Folding()
    agent = Agent(model, Toolbox())
    retained_turns(agent, [tmp_path / "note.md"], ["PRIVATE " * 3000])
    messages = fold_messages(agent)
    assert agent._fold(messages, 100) is False
    assert len(model.seen) == 1
    assert REVOKED_RESULT in [m.text for m in messages]
    assert '"summary"' not in agent.history_path.read_text()


def test_a_later_earlier_turn_revoked_mid_question_is_withheld_at_its_own_place(home, tmp_path):
    """Earlier turns sit at the start of the messages, question then answer: a turn whose evidence is revoked after
    the messages were built has its own answer withheld, at 1 + 2·i, and no other message changes."""
    paths = [tmp_path / "first.md", tmp_path / "second.md", tmp_path / "third.md"]
    agent = Agent(Scripted(), Toolbox(roots=paths))
    retained_turns(agent, paths, ["FIRST [N1]", "SECOND [N2]", "THIRD [N3]"])
    messages = fold_messages(agent)
    before = list(messages)
    agent.toolbox.roots = [paths[0], paths[2]]                  # the second turn's evidence is revoked
    assert agent._filter_access(messages) is True
    assert [m.text for m in messages] == ["Read the evidence.", "FIRST [N1]", "Read the evidence.", REVOKED_RESULT,
                                          "Read the evidence.", "THIRD [N3]", "Continue."]
    assert [i for i, (a, b) in enumerate(zip(before, messages)) if a != b] == [3]
